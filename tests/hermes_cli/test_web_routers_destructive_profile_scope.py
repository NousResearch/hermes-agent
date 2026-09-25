"""One backend serves every profile, so a REST handler that reads ``get_hermes_home()``
directly mutates the LAUNCH profile's data no matter which profile the request named.

These pin both halves of the contract:

* a named profile is the one that gets wiped / written / spawned against, and the launch
  profile survives untouched;
* an UNNAMED profile is refused (400) on a destructive or privileged route while several
  profiles are served, and still means the launch profile on a genuinely single-profile host.

Multi-profile hosting is NOT faked here. ``destructive_profile`` asks
``agent.secret_scope.is_multiplex_active()``, and that flag is only ever set by the boot
wiring (``activate_multi_profile_hosting_eagerly``) or by the lazy backstop inside
``_config_profile_scope``. Patching the predicate would leave exactly that wiring — the
thing the whole 400 branch rests on — untested, so the tests drive the real path and the
autouse fixture below puts the process-global flag back (activation is one-way per process).
"""
import json
import zipfile

import pytest
import hermes_yaml as yaml


@pytest.fixture(autouse=True)
def _multiplex_state_is_per_test():
    """Multi-profile activation is a one-way PROCESS-global flip; contain it to one test.

    Entered single-profile (so a leak from an earlier module cannot make an unnamed
    request 400 by accident) and restored exactly as found, including the frozen launch
    env snapshot activation captures.
    """
    import agent.secret_scope as secret_scope
    from tui_gateway import launch_profile_policy

    was_active = secret_scope.is_multiplex_active()
    snapshot = launch_profile_policy._snapshot
    secret_scope.set_multiplex_active(False)
    launch_profile_policy._snapshot = None
    try:
        yield
    finally:
        secret_scope.set_multiplex_active(was_active)
        launch_profile_policy._snapshot = snapshot


@pytest.fixture
def homes(tmp_path, monkeypatch, _isolate_hermes_home):
    """Isolated launch home + one named profile, both seeded with real files."""
    from hermes_constants import get_hermes_home
    from hermes_cli import profiles

    launch_home = get_hermes_home()
    profiles_root = launch_home / "profiles"
    beta = profiles_root / "worker_beta"
    for home in (launch_home, beta):
        (home / "memories").mkdir(parents=True, exist_ok=True)
        (home / "memories" / "MEMORY.md").write_text(f"memory of {home.name}\n", encoding="utf-8")
        (home / "memories" / "USER.md").write_text(f"user of {home.name}\n", encoding="utf-8")
        (home / "webhook_subscriptions.json").write_text(
            json.dumps({"alerts": {"secret": "s", "events": []}}), encoding="utf-8")
        (home / "config.yaml").write_text(yaml.safe_dump({
            "hooks": {"pre_tool_call": [{"command": "/bin/true"}]},
            # A per-home marker every scoped READ path can be identified by.
            "proxy": {"label": home.name},
        }), encoding="utf-8")
        (home / ".update_check").write_text("cached\n", encoding="utf-8")
    # Non-empty: an empty ``.env`` is the crashed-``profile create`` shell that
    # ``named_profile_has_servable_identity`` deliberately refuses to count as a tenant.
    (beta / ".env").write_text("BETA=1\n", encoding="utf-8")

    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: launch_home)
    monkeypatch.setattr(profiles, "_get_profiles_root", lambda: profiles_root)
    return {"launch": launch_home, "worker_beta": beta}


@pytest.fixture
def client(monkeypatch, homes):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    import hermes_state
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", homes["launch"] / "state.db")
    c = TestClient(app)
    c.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return c


class _StubDB:
    """Just enough session store for the delete/prune bodies; records nothing itself."""

    def delete_sessions(self, ids):
        return len(ids)

    def delete_empty_sessions(self):
        return 1

    def count_open_prune_matches(self, **_filters):
        return 0

    def list_prune_candidates(self, **_filters):
        return []

    def prune_sessions(self, **_kwargs):
        return 1

    def close(self):
        pass


@pytest.fixture
def seams(monkeypatch):
    """Record the three off-process effects these routes have, and neutralise them.

    ``spawn`` = the argv of every backgrounded ``hermes`` action, ``db`` = the profile every
    session-store open names, ``pool_home`` = the home the credential-pool body resolves.
    A route that reaches any of them after a 400 shows up as a non-empty list.
    """
    import agent.credential_pool as credential_pool
    import agent.credential_sources as credential_sources
    from hermes_cli import web_server_gateway, web_server_sessions
    from hermes_cli.config import get_hermes_home

    record = {"spawn": [], "db": [], "pool_home": []}

    class _Proc:
        pid = 4242

    def _spawn(argv, name):
        record["spawn"].append((name, list(argv)))
        return _Proc()

    def _open_db(profile, *, read_only):
        record["db"].append(profile)
        return _StubDB()

    class _Pool:
        def remove_index(self, _index):
            record["pool_home"].append(str(get_hermes_home()))
            return type("_Entry", (), {"source": ""})()

        def entries(self):
            return []

    monkeypatch.setattr(web_server_gateway, "_spawn_hermes_action", _spawn)
    monkeypatch.setattr(web_server_sessions, "_open_session_db_for_profile", _open_db)
    monkeypatch.setattr(credential_pool, "load_pool", lambda _provider: _Pool())
    monkeypatch.setattr(credential_sources, "find_removal_step", lambda *_a: None)
    return record


@pytest.fixture
def multiplexed(homes):
    """Arm the guard the way a real two-profile host does: the BOOT probe.

    ``activate_multi_profile_hosting_eagerly`` enumerates this host's servable profile
    homes and flips ``set_multiplex_active`` itself; asserting it returned True is what
    proves the wiring — not just the flag — is what refuses the unnamed requests below.
    """
    from agent.secret_scope import is_multiplex_active
    from tui_gateway.launch_profile_policy import activate_multi_profile_hosting_eagerly

    assert activate_multi_profile_hosting_eagerly() is True, "boot probe did not see two homes"
    assert is_multiplex_active()


def _cfg(home):
    return yaml.safe_load((home / "config.yaml").read_text()) or {}


def _hooks(home):
    return _cfg(home).get("hooks") or {}


def _zip(tmp_path):
    path = tmp_path / "backup.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("manifest.json", "{}")
    return path


def _q(profile):
    """``?profile=`` fragment, or nothing at all when the request names no profile."""
    return f"?profile={profile}" if profile else ""


def _body_profile(profile):
    """The session family takes its profile in the BODY, so an unnamed request omits the key."""
    return {"profile": profile} if profile else {}


# Every destructive/privileged route, as a call taking (client, profile-name-or-"", tmp_path).
DESTRUCTIVE = {
    "memory-reset": lambda c, p, t: c.post(f"/api/memory/reset{_q(p)}", json={"target": "all"}),
    "webhook-delete": lambda c, p, t: c.delete(f"/api/webhooks/alerts{_q(p)}"),
    "hook-delete": lambda c, p, t: c.request(
        "DELETE", f"/api/ops/hooks{_q(p)}", json={"event": "pre_tool_call", "command": "/bin/true"}),
    "hook-create": lambda c, p, t: c.post(
        f"/api/ops/hooks{_q(p)}", json={"event": "pre_tool_call", "command": "/bin/armed"}),
    "checkpoints-prune": lambda c, p, t: c.post(f"/api/ops/checkpoints/prune{_q(p)}"),
    "import": lambda c, p, t: c.post(f"/api/ops/import{_q(p)}", json={"archive": str(_zip(t))}),
    "import-upload": lambda c, p, t: c.post(
        f"/api/ops/import-upload{_q(p)}",
        files={"file": ("backup.zip", _zip(t).read_bytes(), "application/zip")}),
    "credentials-pool-delete": lambda c, p, t: c.delete(f"/api/credentials/pool/anthropic/0{_q(p)}"),
    "curator-run": lambda c, p, t: c.post(f"/api/curator/run{_q(p)}"),
    "sessions-prune": lambda c, p, t: c.post(
        "/api/sessions/prune", json={"dry_run": False, **_body_profile(p)}),
    "sessions-empty": lambda c, p, t: c.delete(f"/api/sessions/empty{_q(p)}"),
    "sessions-bulk-delete": lambda c, p, t: c.post(
        "/api/sessions/bulk-delete", json={"ids": ["abc"], **_body_profile(p)}),
}


@pytest.mark.parametrize("route", sorted(DESTRUCTIVE))
def test_destructive_route_refuses_an_unnamed_profile_while_multiplexing(
    client, homes, seams, multiplexed, tmp_path, route
):
    """No profile named + several served = refused, and provably nothing happened."""
    resp = DESTRUCTIVE[route](client, "", tmp_path)

    assert resp.status_code == 400, resp.text
    assert seams == {"spawn": [], "db": [], "pool_home": []}, f"{route} reached a backend anyway"
    for home in homes.values():
        assert (home / "memories" / "MEMORY.md").exists()
        assert "alerts" in json.loads((home / "webhook_subscriptions.json").read_text())
        assert _hooks(home) == {"pre_tool_call": [{"command": "/bin/true"}]}


# Routes whose effect is a file inside the target home: (call, "did it happen here?").
FILE_EFFECTS = {
    "memory-reset": (DESTRUCTIVE["memory-reset"],
                     lambda home: not (home / "memories" / "MEMORY.md").exists()),
    "webhook-delete": (DESTRUCTIVE["webhook-delete"],
                       lambda home: "alerts" not in json.loads((home / "webhook_subscriptions.json").read_text())),
    "hook-delete": (DESTRUCTIVE["hook-delete"], lambda home: not _hooks(home)),
    "hook-create": (DESTRUCTIVE["hook-create"],
                    lambda home: any(e.get("command") == "/bin/armed"
                                     for e in _hooks(home).get("pre_tool_call", []))),
}


@pytest.mark.parametrize("route", sorted(FILE_EFFECTS))
def test_named_profile_is_the_only_one_touched(client, homes, seams, tmp_path, route):
    call, happened = FILE_EFFECTS[route]
    resp = call(client, "worker_beta", tmp_path)

    assert resp.status_code == 200, resp.text
    assert happened(homes["worker_beta"]), f"{route} did not act on worker_beta"
    assert not happened(homes["launch"]), f"{route} also hit the launch profile"


# Routes whose effect is a backgrounded ``hermes`` subprocess: the profile must reach its argv,
# because nothing else in that process knows which home the request meant.
SPAWNING = {
    "checkpoints-prune": DESTRUCTIVE["checkpoints-prune"],
    "curator-run": DESTRUCTIVE["curator-run"],
    "import": DESTRUCTIVE["import"],
}


@pytest.mark.parametrize("route", sorted(SPAWNING))
def test_spawned_action_carries_the_named_profile(client, homes, seams, tmp_path, route):
    resp = SPAWNING[route](client, "worker_beta", tmp_path)

    assert resp.status_code == 200, resp.text
    assert [argv[:2] for _name, argv in seams["spawn"]] == [["-p", "worker_beta"]]


@pytest.mark.parametrize("route", ["sessions-prune", "sessions-empty", "sessions-bulk-delete"])
def test_session_route_opens_the_named_profiles_store(client, homes, seams, tmp_path, route):
    resp = DESTRUCTIVE[route](client, "worker_beta", tmp_path)

    assert resp.status_code == 200, resp.text
    assert seams["db"] == ["worker_beta"]


def test_session_prune_dry_run_still_works_unnamed_while_multiplexing(client, seams, multiplexed):
    """A preview deletes nothing, so the confirm dialog must keep working without a profile."""
    resp = client.post("/api/sessions/prune", json={"dry_run": True})

    assert resp.status_code == 200, resp.text
    assert resp.json()["removed"] == 0
    assert seams["db"] == [None]


def test_credential_pool_delete_runs_in_the_named_profiles_home(client, homes, seams):
    resp = client.delete("/api/credentials/pool/anthropic/0?profile=worker_beta")

    assert resp.status_code == 200, resp.text
    assert seams["pool_home"] == [str(homes["worker_beta"])]


# --- activation: the wiring the 400 branch depends on -------------------------------


def test_a_request_for_another_profile_arms_the_guard(client, homes):
    """The lazy backstop: the first cross-profile request is itself the activation."""
    from agent.secret_scope import is_multiplex_active

    assert not is_multiplex_active()
    assert client.get("/api/memory?profile=worker_beta").status_code == 200
    assert is_multiplex_active()

    assert client.post("/api/memory/reset", json={"target": "all"}).status_code == 400
    assert (homes["launch"] / "memories" / "MEMORY.md").exists()


def test_unnamed_profile_still_means_the_launch_profile_on_a_single_profile_host(
    client, homes, monkeypatch, tmp_path
):
    """A plain ``hermes serve`` has nothing to confuse: `curl` with no profile is unchanged."""
    from agent.secret_scope import is_multiplex_active
    from hermes_cli import profiles
    from tui_gateway.launch_profile_policy import activate_multi_profile_hosting_eagerly

    empty_root = tmp_path / "no-named-profiles"
    empty_root.mkdir()
    monkeypatch.setattr(profiles, "_get_profiles_root", lambda: empty_root)
    assert activate_multi_profile_hosting_eagerly() is False
    assert not is_multiplex_active()

    resp = client.post("/api/memory/reset", json={"target": "all"})

    assert resp.status_code == 200, resp.text
    assert not (homes["launch"] / "memories" / "MEMORY.md").exists()
    assert (homes["worker_beta"] / "memories" / "MEMORY.md").exists()


# --- config writes that are scoped but not destructive ------------------------------


def test_plugin_providers_writes_the_named_profiles_config(client, homes):
    resp = client.put("/api/dashboard/plugin-providers?profile=worker_beta",
                      json={"context_engine": "compaction-v2"})

    assert resp.status_code == 200, resp.text
    assert _cfg(homes["worker_beta"]).get("context") == {"engine": "compaction-v2"}
    assert "context" not in _cfg(homes["launch"])


@pytest.fixture
def readiness_only_in_beta(homes, monkeypatch):
    """A memory provider that is ready in worker_beta and nowhere else.

    Readiness resolves through ``load_config()``, so a readiness check that runs OUTSIDE the
    request's scope answers about the launch profile: it refuses a provider the target has
    configured, and — the dangerous direction — accepts one only the launch profile has and
    writes it into the target as a broken setting.
    """
    from hermes_cli import web_server_memory
    from hermes_cli.config import get_hermes_home

    def _statuses():
        ready = get_hermes_home().resolve() == homes["worker_beta"].resolve()
        return [{"name": "mem0", "status": "ready" if ready else "not_configured"}]

    monkeypatch.setattr(web_server_memory, "_discover_memory_provider_statuses", _statuses)


@pytest.mark.parametrize("call", [
    pytest.param(lambda c: c.put("/api/memory/provider?profile=worker_beta", json={"provider": "mem0"}),
                 id="memory-provider"),
    pytest.param(lambda c: c.put("/api/dashboard/plugin-providers?profile=worker_beta",
                                 json={"memory_provider": "mem0"}),
                 id="plugin-providers"),
])
def test_memory_provider_readiness_is_judged_in_the_profile_being_written(
    client, homes, readiness_only_in_beta, call
):
    resp = call(client)

    assert resp.status_code == 200, resp.text
    assert _cfg(homes["worker_beta"])["memory"]["provider"] == "mem0"
    assert "memory" not in _cfg(homes["launch"])


def test_local_models_quickstart_activates_into_the_named_profile(client, homes, monkeypatch):
    """Quickstart is ``activate`` plus a download; its config writes must follow ``?profile=``."""
    from hermes_cli.config import get_hermes_home
    from hermes_cli.web_routers import local_models as lm

    entry = type("_Entry", (), {"id": "m1", "display_name": "M One", "min_engine": None})()
    variant = type("_Variant", (), {"model_id": "m1-q4"})()
    seen = []

    monkeypatch.setattr(lm.hardware, "probe_budget", lambda **_kw: None)
    monkeypatch.setattr(lm, "_quickstart_target", lambda _body, _budget: (entry, variant))
    monkeypatch.setattr(lm, "_runtime_target", lambda *_a: ("b1", "cpu"))
    monkeypatch.setattr(lm, "_download_plan", lambda *_a: [])
    monkeypatch.setattr(lm.binaries, "installed_engine", lambda _backend: "installed")
    monkeypatch.setattr(lm.bootstrap, "staged_model_ids", lambda: {"m1-q4"})
    monkeypatch.setattr(lm, "_set_runtime_enabled", lambda _on: (lambda: None))
    monkeypatch.setattr(lm, "_ensure_server", lambda *_a, **_k: None)
    monkeypatch.setattr(lm, "_assign_default", lambda *_a: seen.append(str(get_hermes_home())))
    # Run the job body inline; the real spawner's ``on_exit`` is what frees the quickstart lock.
    monkeypatch.setattr(lm, "_spawn_job", lambda job, name, body, **kw: (body(), kw["on_exit"]()))

    resp = client.post("/api/local-models/quickstart?profile=worker_beta", json={})

    assert resp.status_code == 200, resp.text
    assert seen == [str(homes["worker_beta"])]


# --- the read/side-effect routes scoped in the same sweep ---------------------------


def test_curator_pause_writes_the_named_profiles_state(client, homes):
    resp = client.put("/api/curator/paused?profile=worker_beta", json={"paused": True})

    assert resp.status_code == 200, resp.text
    assert json.loads((homes["worker_beta"] / "skills" / ".curator_state").read_text())["paused"] is True
    assert not (homes["launch"] / "skills" / ".curator_state").exists()


def test_forced_update_check_runs_in_the_named_profiles_scope(client, homes, monkeypatch):
    from hermes_cli import source_check
    from hermes_cli.config import get_hermes_home
    from hermes_cli.web_routers import actions

    seen = []

    def check(*, force):
        seen.append((get_hermes_home(), force))
        return {"behind": 0, "commits": []}

    monkeypatch.setattr(source_check, "check_for_updates", check)
    monkeypatch.setattr(actions, "_dashboard_local_update_managed_externally", lambda: False)
    monkeypatch.setattr(actions, "detect_install_method", lambda _root: "git")

    resp = client.get("/api/hermes/update/check?force=true&profile=worker_beta")

    assert resp.status_code == 200, resp.text
    assert resp.json()["behind"] == 0
    assert seen == [(homes["worker_beta"], True)]


def test_egress_status_reads_the_named_profiles_config(client, homes, monkeypatch):
    from hermes_cli import proxy_cli
    from hermes_cli.config import load_config

    monkeypatch.setattr(proxy_cli, "format_status_text",
                        lambda **_kw: str((load_config().get("proxy") or {}).get("label")))

    resp = client.get("/api/egress/status?profile=worker_beta")

    assert resp.status_code == 200, resp.text
    assert resp.json()["text"] == "worker_beta"


def test_memory_provider_setup_runs_in_the_named_profiles_home(client, homes, monkeypatch):
    from hermes_cli.config import get_hermes_home
    from hermes_cli.web_routers import memory_providers as mp

    monkeypatch.setattr(mp, "_memory_provider_manifest", lambda _name: {"name": "mem0"})
    monkeypatch.setattr(mp, "_load_memory_provider", lambda _name: None)
    monkeypatch.setattr(mp, "_install_memory_provider_setup",
                        lambda name: {"ok": True, "provider": name, "home": str(get_hermes_home())})

    resp = client.post("/api/memory/providers/mem0/setup?profile=worker_beta", json={})

    assert resp.status_code == 200, resp.text
    assert resp.json()["home"] == str(homes["worker_beta"])

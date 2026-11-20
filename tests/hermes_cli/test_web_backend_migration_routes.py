"""The dashboard's backend-migration routes (``/api/gateway/migrate/backends/*``) through the real app.

The engine's host is a fake for the service manager, the process table and the network (as in the
engine's own tests); the plan, the manifest reader and ``status_payload`` are the real engine code, the
auth gate is the app's real middleware, and only the action spawner is recorded instead of run (it
would start ``systemd-run --user`` on the test machine).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from hermes_cli import web_server
from hermes_cli import gateway_migrate_backends_apply as engine
from hermes_cli import gateway_migrate_backends_plan
from hermes_cli.gateway_migrate_backends import BackendItem
from hermes_cli.gateway_migrate_backends_host import SystemHost
from hermes_cli.web_routers import backend_migration as routes
from tests.waiting import wait_until

_PASSWORD = "pw-route-never-returned-5e2a"
_PID = 4242
_ROUTES = [("GET", "/api/gateway/migrate/backends/plan"), ("POST", "/api/gateway/migrate/backends/apply"),
           ("POST", "/api/gateway/migrate/backends/restore"), ("GET", "/api/gateway/migrate/backends/status")]


class FakeHost(SystemHost):
    """One old dashboard as a systemd user unit on 0.0.0.0:9119 whose ExecStart carries the password."""

    def __init__(self, home, *, running_turns=0):
        super().__init__(home)
        self.turns = running_turns

    def backends(self):
        return [BackendItem(supervisor="systemd", running=True, pid=_PID,
                            argv=["/usr/bin/env", f"HERMES_DASHBOARD_BASIC_AUTH_PASSWORD={_PASSWORD}",
                                  "/venv/bin/hermes", "dashboard", "--no-open", "--host", "0.0.0.0", "--port", "9119"],
                            unit="hermes-dashboard.service", scope="user",
                            unit_path="/home/u/.config/systemd/user/hermes-dashboard.service", state="enabled",
                            hermes_home=str(self.home), this_home=True)]

    def is_root(self):
        return False

    def os_family(self):
        return "linux"

    def unit_props(self, unit, scope):
        return {"UnitFileState": "enabled", "ActiveState": "active", "ControlGroup": f"/fake/{unit}"}

    def process_env(self, pid):
        return {"HERMES_DASHBOARD_BASIC_AUTH_USERNAME": "spike", "HERMES_DASHBOARD_BASIC_AUTH_PASSWORD": _PASSWORD}

    def process_cwd(self, pid):
        return f"/home/u/{_PASSWORD}"

    def start_time(self, pid):
        return 1000 + pid

    def same_process(self, pid, start_time):
        return False

    def own_unit(self):  # never the test runner's own cgroup
        return None

    def running_turns(self, pid, start_time):
        return self.turns

    def bundle_missing(self):
        return None

    def gateway_kind(self):
        return "systemd"

    def gateway_state(self):
        return {"kind": "systemd", "unit": "hermes-gateway", "installed": False, "enabled": False, "active": False,
                "running_pid": None}

    def gateway_preflight(self):
        return None

    def linger(self):
        return False

    def gateway_owner(self):  # every test profile has its own gateway service
        return "own", None

    def http_status(self, host, port):  # shaped like SystemHost's: GET /api/health (+ providers)
        return {"ok": True, "auth_required": True, "auth_providers": ["basic"]}

    def same_network_namespace(self, pid, *, container):  # the fake pid is never looked up in /proc
        return True

    def health_status(self, host, port):  # the by-hand plan's health-only read, from the same fake answer
        return self.http_status(host, port)


_IS_STANDARD_ROOT = gateway_migrate_backends_plan._is_standard_root


@pytest.fixture(autouse=True)
def _standard_root(monkeypatch):
    """The homes here stand for the user's standard Hermes root, so commands read ``hermes -p NAME`` (as in
    the engine's tests); the root pin for a temp root is tested on its own below."""
    monkeypatch.setattr(gateway_migrate_backends_plan, "_is_standard_root", lambda root: True)


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "hermes-home"
    (h / "logs").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(h))
    (h / "config.yaml").write_text("model:\n  default: fake-model\n", encoding="utf-8")
    (h / ".env").write_text("OPENAI_API_KEY=***\n", encoding="utf-8")
    os.chmod(h / ".env", 0o600)
    return h


@pytest.fixture
def spawned(home, tmp_path, monkeypatch):
    """Records each launcher instead of starting it; the launcher answers as the real one does when its
    background run holds the lock: exit 0 and its ``--json`` line (``confirmed: true``) in its own output."""
    calls: list[dict] = []
    monkeypatch.setattr("hermes_cli.web_server_gateway._ACTION_LOG_DIR", tmp_path / "action-logs")

    def record(subcommand, name, *, env_overrides=None, output=None):
        calls.append({"argv": list(subcommand), "name": name, "env": dict(env_overrides or {})})
        launcher_says(Path((env_overrides or {})["HERMES_HOME"]), output, confirmed=True)
        return SimpleNamespace(pid=777, poll=lambda: 0, wait=lambda timeout=None: 0)

    monkeypatch.setattr("hermes_cli.web_server_gateway._spawn_hermes_action", record)
    return calls


def launcher_says(home: Path, output, *, confirmed: bool) -> None:
    line = {"detached": True, "mode": "apply", "unit": "x.service", "confirmed": confirmed,
            "log": str(home / "logs" / "backend-migration.log")}
    output.write(f"=== gateway-migrate-backends started ===\n{json.dumps(line)}\n".encode())


def _client(home, monkeypatch, **host_kw):
    monkeypatch.setattr(routes, "_host", lambda target_home: FakeHost(target_home, **host_kw))
    client = TestClient(web_server.app)
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    return client


def test_routes_need_the_dashboard_session_and_restore_runs_detached_on_the_gateway(home, spawned, monkeypatch):
    """Same gate as every action route: no session token, no answer; and a restore asked of the gateway's
    own listener (which restore stops) starts nothing."""
    authed = _client(home, monkeypatch)
    anon = TestClient(web_server.app)
    for method, path in _ROUTES:
        assert anon.request(method, path).status_code == 401, path

    # A signed-in-only bind (the old dashboard on 0.0.0.0 with a password): the cookie gate, even with the
    # loopback session token, which that gate does not accept.
    from hermes_cli.dashboard_auth import clear_providers, register_provider
    from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider
    state = web_server.app.state
    prev_gate = (getattr(state, "bound_host", None), getattr(state, "bound_port", None),
                 getattr(state, "auth_required", None))
    clear_providers()
    register_provider(StubAuthProvider())
    state.bound_host, state.bound_port, state.auth_required = "hermes.example", 443, True
    try:
        gated = TestClient(web_server.app, base_url="https://hermes.example")
        gated.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
        for method, path in _ROUTES:
            r = gated.request(method, path)
            assert r.status_code == 401 and r.json()["error"] == "unauthenticated", path
    finally:
        clear_providers()
        state.bound_host, state.bound_port, state.auth_required = prev_gate
    assert spawned == []

    # On the gateway's own listener the restore runs detached (it outlives the gateway it stops).
    engine.write_manifest(home, _manifest("done", started_at=1.0))
    prev = getattr(web_server.app.state, "ui_surface", None)
    web_server.app.state.ui_surface = "webapp"
    try:
        r = authed.post("/api/gateway/migrate/backends/restore")
    finally:
        web_server.app.state.ui_surface = prev
    assert r.status_code == 200, r.text
    assert [s["argv"] for s in spawned] == [["-p", "default", "gateway", "migrate", "--backends", "--restore",
                                             "--yes", "--detach", "--json"]]


def test_apply_maps_the_contract_flags_and_refuses_unknown_or_blocked(home, spawned, monkeypatch):
    client = _client(home, monkeypatch)

    r = client.post("/api/gateway/migrate/backends/apply",
                    json={"address": "0.0.0.0:9119", "force_interrupt": True, "keep_webapp_address": False})
    assert r.status_code == 200, r.text
    assert r.json()["ok"] is True and r.json()["old_address"] == "0.0.0.0:9119"
    # ``-p``: a sticky ``hermes profile use`` choice must not re-home the run away from this home.
    assert spawned[-1]["argv"] == ["-p", "default", "gateway", "migrate", "--backends", "--apply", "--yes", "--detach",
                                   "--json", "--address", "0.0.0.0:9119", "--force-interrupt"]
    assert spawned[-1]["name"] == "gateway-migrate-backends"
    assert spawned[-1]["env"] == {"HERMES_HOME": str(home)}

    n = len(spawned)
    assert client.post("/api/gateway/migrate/backends/apply", json={"force": True}).status_code == 422
    blocked = _client(home, monkeypatch, running_turns=1).post("/api/gateway/migrate/backends/apply", json={})
    assert blocked.status_code == 409 and blocked.json()["error"] == "blocked"
    assert any("--force-interrupt" in b for b in blocked.json()["blockers"])
    assert len(spawned) == n


def test_no_secret_value_in_any_response(home, spawned, monkeypatch):
    client = _client(home, monkeypatch)
    # Before the move the password is only in the old unit's ExecStart, cwd and environment.
    bodies = [client.get("/api/gateway/migrate/backends/plan"), client.post("/api/gateway/migrate/backends/apply")]
    # After it the value lives in .env; a log line that ever echoed it must not reach the page.
    (home / ".env").write_text(f"HERMES_DASHBOARD_BASIC_AUTH_PASSWORD={_PASSWORD}\n", encoding="utf-8")
    (home / "logs" / "backend-migration.log").write_text(f"oops {_PASSWORD}\n✓ Migrated\n", encoding="utf-8")
    bodies.append(client.get("/api/gateway/migrate/backends/status"))

    assert [b.status_code for b in bodies] == [200, 200, 200]
    plan = bodies[0].json()["plan"]
    assert "HERMES_DASHBOARD_BASIC_AUTH_PASSWORD" in plan["copy_env_keys"]  # the name is shown...
    for b in bodies:
        assert _PASSWORD not in b.text  # ...the value never is (not via argv, cwd, env or the log)


def test_status_passes_the_engine_status_through(home, spawned, monkeypatch):
    client = _client(home, monkeypatch)
    assert client.get("/api/gateway/migrate/backends/status").json()["state"] == "none"

    manifest = {"kind": "apply", "state": "rolled_back", "step": "verify", "message": "verify failed; rolled back",
                "old_address": "0.0.0.0:9119", "new_address": "0.0.0.0:9119", "started_at": 10.0,
                "finished_at": 20.0, "steps": [{"name": "verify", "state": "undone", "detail": "x"}],
                "old_backend": {"supervisor": "systemd", "unit": "hermes-dashboard.service", "scope": "user"},
                "backups": {"config": str(home / "backend_migration" / "config.yaml.bak"), "env": None},
                "gateway": {"installed": False, "active": False}, "dial": _manifest("done")["dial"]}
    engine.write_manifest(home, manifest)
    body = client.get("/api/gateway/migrate/backends/status").json()
    expected = engine.status_payload(FakeHost(home))
    assert {k: body[k] for k in expected} == json.loads(json.dumps(expected))
    assert body["served_by"] == "dashboard" and body["manual_steps"] == []

    engine.write_manifest(home, {**manifest, "state": "failed"})
    assert client.get("/api/gateway/migrate/backends/status").json()["manual_steps"]


# --------------------------------------------------------------------------- ?profile=


@pytest.fixture
def work(home, monkeypatch):
    """A second profile, ``work``, beside the process's own (default) home."""
    from hermes_cli import profiles
    w = home / "profiles" / "work"
    (w / "logs").mkdir(parents=True)
    (w / "config.yaml").write_text("model:\n  default: fake-model\n", encoding="utf-8")
    (w / ".env").write_text("", encoding="utf-8")
    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: home)
    monkeypatch.setattr(profiles, "_get_profiles_root", lambda: home / "profiles")
    return w


def _manifest(state: str, **extra) -> dict:
    """A migration record with every field the engine needs to trust it (``unreadable_manifest``)."""
    return {"kind": "apply", "state": state, "step": "verify", "message": "", "old_address": "0.0.0.0:9119",
            "new_address": "0.0.0.0:9119", "started_at": 10.0, "finished_at": None, "steps": [],
            "old_backend": {"supervisor": "systemd", "unit": "hermes-dashboard.service", "scope": "user"},
            "gateway": {"installed": False, "active": False}, "backups": {"config": None, "env": None},
            "dial": {"host": "127.0.0.1", "old_host": "127.0.0.1", "verify_hosts": ["127.0.0.1"],
                     "old_port": 9119, "new_port": 9119}, **extra}


def test_a_named_profile_is_planned_moved_and_followed_in_its_own_home(home, work, spawned, monkeypatch):
    """Desktop sends ``?profile=``: every route acts on that profile's home, never the process's own, and the
    detached run is started for it with ``-p``."""
    homes: list = []
    monkeypatch.setattr(routes, "_host", lambda target_home: homes.append(Path(target_home)) or FakeHost(target_home))
    client = TestClient(web_server.app)
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    engine.write_manifest(home, _manifest("done"))  # the default home's own record must not leak into work's

    plan = client.get("/api/gateway/migrate/backends/plan", params={"profile": "work"})
    assert plan.status_code == 200, plan.text
    assert plan.json()["profile"] == "work" and plan.json()["run"]["state"] == "none"

    started = client.post("/api/gateway/migrate/backends/apply", params={"profile": "work"}, json={})
    assert started.status_code == 200, started.text
    assert spawned[-1]["argv"][:2] == ["-p", "work"] and "--apply" in spawned[-1]["argv"]
    assert spawned[-1]["env"] == {"HERMES_HOME": str(work)}

    engine.write_manifest(work, _manifest("rolled_back", message="verify failed: no answer; rolled back, old "
                                                                 "backend verified on 0.0.0.0:9119"))
    status = client.get("/api/gateway/migrate/backends/status", params={"profile": "work"}).json()
    assert (status["state"], status["reason"], status["profile"]) == ("rolled_back", "verify failed: no answer", "work")
    assert {h.resolve() for h in homes} == {work.resolve()}

    # Without ?profile= the same routes keep acting on the process's own home, as before.
    assert client.get("/api/gateway/migrate/backends/status").json()["state"] == "done"


@pytest.mark.parametrize("name", ["ghost", "../work", "Bad Name"])
def test_a_profile_this_server_does_not_have_is_404_and_never_another_home(home, work, spawned, monkeypatch, name):
    homes: list = []
    monkeypatch.setattr(routes, "_host", lambda target_home: homes.append(target_home) or FakeHost(target_home))
    client = TestClient(web_server.app)
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    engine.write_manifest(home, _manifest("done"))

    for method, path in _ROUTES:
        r = client.request(method, path, params={"profile": name})
        assert r.status_code == 404, (path, r.text)
        assert r.json()["ok"] is False and r.json()["error"] == "unknown_profile" and r.json()["message"], path
    assert homes == [] and spawned == []


def test_status_says_whether_a_running_records_process_is_alive(home, spawned, monkeypatch):
    client = _client(home, monkeypatch)
    engine.write_manifest(home, _manifest("running", step="start_gateway", pid=_PID, pid_start_time=1))
    status = client.get("/api/gateway/migrate/backends/status").json()
    assert (status["state"], status["alive"], status["reason"]) == ("running", False, None)

    monkeypatch.setattr(FakeHost, "same_process", lambda self, pid, start_time: pid == _PID and start_time == 1)
    status = client.get("/api/gateway/migrate/backends/status").json()
    assert (status["state"], status["alive"]) == ("running", True)
    # A finished record carries no liveness: only a running one has a process to ask about.
    engine.write_manifest(home, _manifest("done"))
    assert "alive" not in client.get("/api/gateway/migrate/backends/status").json()


def test_refusals_name_the_profile_in_the_commands_they_offer(home, work, spawned, monkeypatch):
    client = _client(home, monkeypatch)
    engine.write_manifest(work, _manifest("failed", step="stop_old_backend"))
    r = client.post("/api/gateway/migrate/backends/apply", params={"profile": "work"}, json={})
    assert r.status_code == 409 and r.json()["error"] == "unfinished_run"
    assert r.json()["command"] == "hermes -p work gateway migrate --backends --apply"
    assert "hermes -p work gateway migrate --backends --apply" in r.json()["message"]

    monkeypatch.setattr(FakeHost, "same_process", lambda self, pid, start_time: True)
    engine.write_manifest(work, _manifest("running", pid=_PID, pid_start_time=1))
    r = client.post("/api/gateway/migrate/backends/restore", params={"profile": "work"})
    assert r.status_code == 409 and r.json()["error"] == "already_running"
    assert r.json()["status_command"] == "hermes -p work gateway migrate --backends --status"
    assert spawned == []


# --------------------------------------------------------------------------- auth and the gateway's listener


@pytest.fixture
def gated(monkeypatch):
    """The old dashboard on a public bind behind the sign-in gate (stub provider), signed in."""
    from hermes_cli.dashboard_auth import clear_providers, register_provider
    from tests.hermes_cli.conftest_dashboard_auth import StubAuthProvider
    for key, value in (("bound_host", "hermes.example"), ("bound_port", 443), ("auth_required", True)):
        monkeypatch.setattr(web_server.app.state, key, value, raising=False)
    clear_providers()
    register_provider(StubAuthProvider())
    client = TestClient(web_server.app, base_url="https://hermes.example")
    login = client.get("/auth/login?provider=stub", follow_redirects=False)
    state = login.headers["location"].split("state=")[1]
    assert client.get(f"/auth/callback?code=stub_code&state={state}", follow_redirects=False).status_code == 302
    yield client
    clear_providers()


@pytest.mark.parametrize(("origin", "fetch_site"), [("https://evil.example", "cross-site"),
                                                    ("http://127.0.0.1:8080", "same-site")])
def test_a_signed_in_move_from_another_pages_origin_is_refused(home, spawned, monkeypatch, gated, origin, fetch_site):
    """apply and restore are writes: with the sign-in cookie they must come from this server's own page."""
    monkeypatch.setattr(routes, "_host", lambda target_home: FakeHost(target_home))
    engine.write_manifest(home, _manifest("done"))
    page = {"Origin": origin, "Sec-Fetch-Site": fetch_site}
    for path in ("/api/gateway/migrate/backends/apply", "/api/gateway/migrate/backends/restore"):
        assert gated.post(path, headers=page, json={}).status_code == 403, path
    assert spawned == []

    own = gated.post("/api/gateway/migrate/backends/restore",
                     headers={"Origin": "https://hermes.example", "Sec-Fetch-Site": "same-origin"})
    assert own.status_code == 200, own.text
    assert len(spawned) == 1


def test_on_the_gateways_listener_desktop_moves_and_restores_the_profile_it_names(home, work, spawned, monkeypatch):
    """Native Desktop reaches the gateway's own listener with a per-request grant; a restore asked there (the
    gateway it stops answers it) starts the detached run for that profile."""
    from tests.hermes_cli.native_gateway_client import native_desktop_client
    monkeypatch.setattr(routes, "_host", lambda target_home: FakeHost(target_home))
    desktop = native_desktop_client(monkeypatch, profiles={"work": work})

    status = desktop.get("/api/gateway/migrate/backends/status", params={"profile": "work"})
    assert status.status_code == 200, status.text
    assert status.json()["served_by"] == "gateway" and status.json()["profile"] == "work"

    engine.write_manifest(work, _manifest("done"))
    restore = desktop.post("/api/gateway/migrate/backends/restore", params={"profile": "work"})
    assert restore.status_code == 200, restore.text
    assert spawned[-1]["argv"] == ["-p", "work", *routes.RESTORE_ARGV]
    assert spawned[-1]["env"] == {"HERMES_HOME": str(work)}

    # A grant is for one profile's home: a request naming another never reaches the routes (main's
    # native-grant rule; the gateway mints no grant at all for a profile it does not serve).
    mismatch = desktop.post("/api/gateway/migrate/backends/restore", params={"profile": "ghost"})
    assert (mismatch.status_code, mismatch.json()["detail"]) == (403, "profile_scope_mismatch")
    # The same listener without a grant (a browser tab, or a request without one) gets nothing.
    assert TestClient(web_server.app).post("/api/gateway/migrate/backends/restore").status_code == 401
    assert len(spawned) == 1


# --------------------------------------------------------------------------- the System page's plan read


def test_page_loads_share_one_scan_and_a_move_always_scans_again(home, spawned, monkeypatch):
    """The System page asks for the plan on every load: one scan answers the loads of five minutes, and the
    plan route scans once per request (not once for the plan and again for the report). A started move
    forgets the scan, and the apply route never uses it."""
    scans: list[int] = []
    monkeypatch.setattr(FakeHost, "backends", lambda self: scans.append(1) or [])  # nothing to move
    client = _client(home, monkeypatch)

    for _ in range(3):
        body = client.get("/api/gateway/migrate/backends/plan").json()
        assert body["plan"]["nothing_to_do"] is True and body["report"] == []
    assert len(scans) == 1

    assert client.post("/api/gateway/migrate/backends/apply", json={}).json()["error"] == "nothing_to_do"
    assert len(scans) == 2  # the apply's own scan

    monkeypatch.setattr(routes, "_SCAN_TTL_SECONDS", 0.0)
    client.get("/api/gateway/migrate/backends/plan")
    assert len(scans) == 3


# --------------------------------------------------------------------------- review round


class MovedHost(FakeHost):
    """The old dashboard was replaced meanwhile by another one on port 9200."""

    def backends(self):
        [item] = super().backends()
        item.pid = _PID + 1
        item.argv = [a if a != "9119" else "9200" for a in item.argv]
        return [item]


def test_a_confirmed_move_never_moves_another_backend_than_the_one_shown(home, spawned, monkeypatch):
    client = _client(home, monkeypatch)
    shown = client.get("/api/gateway/migrate/backends/plan").json()["plan"]
    token, address = shown["expect_token"], shown["old_address"]
    assert address == "0.0.0.0:9119"

    # The card's own payload (address + token) after the backend moved to 9200: the fresh plan to confirm
    # again is unfiltered, so it shows the backend on 9200 with nothing blocking.
    monkeypatch.setattr(routes, "_host", lambda target_home: MovedHost(target_home))
    changed = client.post("/api/gateway/migrate/backends/apply", json={"address": address, "expect": token})
    assert changed.status_code == 409 and changed.json()["error"] == "plan_changed", changed.text
    fresh = changed.json()["plan"]
    assert (fresh["old_address"], fresh["blockers"], fresh["nothing_to_do"]) == ("0.0.0.0:9200", [], False)
    assert fresh["expect_token"] != token
    assert spawned == []

    monkeypatch.setattr(routes, "_host", lambda target_home: FakeHost(target_home))
    same = client.post("/api/gateway/migrate/backends/apply", json={"address": address, "expect": token})
    assert same.status_code == 200, same.text
    # The background run checks the same token again under the run lock.
    assert spawned[-1]["argv"][-4:] == ["--address", "0.0.0.0:9119", "--expect", token]


def test_a_redacted_name_never_turns_a_confirmation_into_plan_changed(home, spawned, monkeypatch):
    """A sign-in username that also spells part of the unit name is redacted for display only; the
    token compares what would really move."""
    (home / ".env").write_text("HERMES_DASHBOARD_BASIC_AUTH_USERNAME=hermes\n", encoding="utf-8")
    client = _client(home, monkeypatch)
    shown = client.get("/api/gateway/migrate/backends/plan").json()["plan"]
    assert shown["old_backend"]["unit"] == "***-dashboard.service"  # display redaction
    r = client.post("/api/gateway/migrate/backends/apply",
                    json={"address": shown["old_address"], "expect": shown["expect_token"]})
    assert r.status_code == 200, r.text


@pytest.mark.parametrize(("text", "shown"), [
    ("https://admin:pw@hermes.example/x?y=1#z", "https://***@hermes.example/x?y=1#z"),
    ("https://a:p'w@q@hermes.example/x", "https://***@hermes.example/x"),
    ("see https://hermes.example/?a=b@c and more", "see https://hermes.example/?a=b@c and more"),
    ("https://hermes.example/path@x#frag@y", "https://hermes.example/path@x#frag@y"),
])
def test_only_a_urls_userinfo_is_stripped(text, shown):
    assert routes._scrub(text, []) == shown


class HeldLauncher:
    """A launcher still waiting for its background run's answer, until released."""

    def __init__(self):
        import threading
        self.pid, self.released = 4321, threading.Event()

    def poll(self):
        return 0 if self.released.is_set() else None

    def wait(self, timeout=None):
        assert self.released.wait(30)
        return 0


def test_a_launch_for_one_profile_never_blocks_another_but_blocks_its_own(home, work, spawned, monkeypatch):
    import threading
    held = HeldLauncher()

    def spawn(subcommand, name, *, env_overrides=None, output=None):
        target_home = Path((env_overrides or {})["HERMES_HOME"])
        spawned.append({"argv": list(subcommand), "env": dict(env_overrides or {})})
        launcher_says(target_home, output, confirmed=True)
        if target_home == work:
            return held
        return SimpleNamespace(pid=778, poll=lambda: 0, wait=lambda timeout=None: 0)

    monkeypatch.setattr("hermes_cli.web_server_gateway._spawn_hermes_action", spawn)
    client = _client(home, monkeypatch)
    answers: dict = {}
    first = threading.Thread(target=lambda: answers.setdefault(
        "work", client.post("/api/gateway/migrate/backends/apply", params={"profile": "work"}, json={})))
    first.start()
    try:
        wait_until(lambda: len(spawned) == 1, timeout=30, what="work's launcher")
        again = client.post("/api/gateway/migrate/backends/apply", params={"profile": "work"}, json={})
        assert (again.status_code, again.json()["error"]) == (409, "already_running")
        assert "hermes -p work" in again.json()["status_command"]
        other = client.post("/api/gateway/migrate/backends/apply", params={"profile": "default"}, json={})
        assert other.status_code == 200, other.text  # another home: no false "already starting"
    finally:
        held.released.set()
        first.join(timeout=30)
    assert answers["work"].status_code == 200, answers["work"].text
    assert [c["argv"][:2] for c in spawned] == [["-p", "work"], ["-p", "default"]]


@pytest.mark.parametrize("launcher", ["timeout", "unconfirmed", "silent"])
def test_a_launch_nobody_confirmed_is_reported_as_unconfirmed_never_as_started(home, spawned, monkeypatch, launcher):
    import subprocess

    def spawn(subcommand, name, *, env_overrides=None, output=None):
        spawned.append(list(subcommand))
        if launcher == "unconfirmed":
            launcher_says(Path((env_overrides or {})["HERMES_HOME"]), output, confirmed=False)

        def wait(timeout=None):
            if launcher == "timeout":
                raise subprocess.TimeoutExpired("hermes", timeout)
            return 0
        return SimpleNamespace(pid=779, poll=lambda: None if launcher == "timeout" else 0, wait=wait)

    monkeypatch.setattr("hermes_cli.web_server_gateway._spawn_hermes_action", spawn)
    client = _client(home, monkeypatch)
    r = client.post("/api/gateway/migrate/backends/apply", json={})
    assert r.status_code == 202, r.text
    body = r.json()
    assert (body["ok"], body["confirmed"], body["outcome"]) == (True, False, "unconfirmed")
    assert body["status_command"] == "hermes -p default gateway migrate --backends --status"
    assert isinstance(body["requested_at"], float)


def test_no_url_credential_reaches_any_response(home, spawned, monkeypatch):
    """A configured ``dashboard.public_url`` may carry a user:password; it is no HERMES_DASHBOARD_* value.
    The engine refuses such a URL (a blocker that never prints it), and no response carries it, the run
    log's tail included."""
    secret = "pw-in-url-9f3c"
    (home / "config.yaml").write_text("model:\n  default: fake-model\ndashboard:\n"
                                      f"  public_url: https://admin:{secret}@hermes.example/\n", encoding="utf-8")
    (home / "logs" / "backend-migration.log").write_text(f"public URL https://admin:{secret}@hermes.example/\n",
                                                          encoding="utf-8")
    client = _client(home, monkeypatch)
    plan = client.get("/api/gateway/migrate/backends/plan")
    assert plan.json()["plan"]["public_url"] is None and plan.json()["plan"]["blockers"]
    bodies = [plan, client.post("/api/gateway/migrate/backends/apply", json={}),
              client.get("/api/gateway/migrate/backends/status")]
    engine.write_manifest(home, _manifest("done"))
    bodies.append(client.post("/api/gateway/migrate/backends/restore"))
    assert [b.status_code for b in bodies] == [200, 409, 200, 200]
    assert "https://***@hermes.example/" in bodies[2].text  # the log line, its credential redacted
    for b in bodies:
        assert secret not in b.text and "admin:" not in b.text, b.request.url


def test_manual_steps_name_the_profile(home, work, spawned, monkeypatch):
    client = _client(home, monkeypatch)
    engine.write_manifest(work, _manifest(
        "failed", home=str(work), message="verify failed; rollback incomplete: x",
        old_backend={"supervisor": "systemd", "unit": "hermes-dashboard.service", "scope": "user"},
        backups={"config": str(work / "backend_migration" / "config.yaml.bak"), "env": None},
        gateway={"installed": True, "active": True}))
    steps = client.get("/api/gateway/migrate/backends/status", params={"profile": "work"}).json()["manual_steps"]
    commands = [s for s in steps if "hermes " in s]
    assert commands and all("hermes -p work " in s for s in commands), steps
    assert any(str(work / "config.yaml") in s for s in steps)  # paths are left alone


def test_qualify_adds_the_profile_once_and_leaves_paths_and_units_alone():
    q = routes.qualify
    verb = "gateway " + "start"
    assert q(f"hermes {verb}", "work") == f"hermes -p work {verb}"
    assert q(f"hermes -p work {verb}", "work") == f"hermes -p work {verb}"
    assert q(f"hermes gateway install && hermes {verb}", "w") == f"hermes -p w gateway install && hermes -p w {verb}"
    assert q("cp /x/.hermes/config.yaml.bak /home/u/.hermes/config.yaml", "w") == \
        "cp /x/.hermes/config.yaml.bak /home/u/.hermes/config.yaml"
    assert q("systemctl --user enable --now hermes-dashboard.service", "w") == \
        "systemctl --user enable --now hermes-dashboard.service"
    assert q([f"hermes {verb}"], None) == [f"hermes {verb}"]


# --------------------------------------------------------------------------- final PR5/PR6 contract


def test_commands_carry_the_engines_root_pin_for_a_profile_outside_the_standard_root(home, work, spawned,
                                                                                    monkeypatch):
    """A home outside ``~/.hermes`` needs ``HERMES_HOME=ROOT hermes -p NAME``: the routes take the engine's
    spelling (``hermes_command``) for every command they offer, and hand the card the ones it shows."""
    monkeypatch.setattr(gateway_migrate_backends_plan, "_is_standard_root", _IS_STANDARD_ROOT)  # a temp root
    pinned = gateway_migrate_backends_plan.hermes_command(str(work))
    assert pinned.startswith("HERMES_HOME=") and pinned.endswith(" hermes -p work"), pinned
    client = _client(home, monkeypatch)
    plan = client.get("/api/gateway/migrate/backends/plan?profile=work").json()
    status = client.get("/api/gateway/migrate/backends/status?profile=work").json()
    want = {"status": f"{pinned} gateway migrate --backends --status",
            "restore": f"{pinned} gateway migrate --backends --restore"}
    assert plan["commands"] == want and status["commands"] == want
    r = client.post("/api/gateway/migrate/backends/apply?profile=work", json={})
    assert r.json()["status_command"] == want["status"]


class HealthMissingHost(FakeHost):
    def http_status(self, host, port):  # a Hermes before v2026.7.30: no GET /api/health
        return {"health": "missing"}


def test_an_old_backend_without_health_is_found_and_blocked(home, spawned, monkeypatch):
    monkeypatch.setattr(routes, "_host", lambda target_home: HealthMissingHost(target_home))
    client = TestClient(web_server.app)
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    plan = client.get("/api/gateway/migrate/backends/plan").json()["plan"]
    assert plan["found"] is True and plan["steps"] == []
    assert any("/api/health" in b for b in plan["blockers"]), plan["blockers"]
    r = client.post("/api/gateway/migrate/backends/apply", json={})
    assert (r.status_code, r.json()["error"]) == (409, "blocked") and spawned == []

# --------------------------------------------------------------------------- the engine's final launcher outcomes


@pytest.mark.parametrize(("code", "line", "status", "error"), [
    (7, {"detached": True, "mode": "apply", "confirmed": False, "outcome": "unconfirmed"}, 202, None),
    (3, {"detached": False, "mode": "apply", "outcome": "refused", "message": "cannot start the run outside x"},
     409, "not_started"),
    (3, {"applied": False, "mode": "apply", "error": "already_running", "message": "busy"}, 409, "already_running"),
])
def test_the_launchers_exit_codes_map_to_answers(home, spawned, monkeypatch, code, line, status, error):
    def spawn(subcommand, name, *, env_overrides=None, output=None):
        spawned.append(list(subcommand))
        target_home = Path((env_overrides or {})["HERMES_HOME"])
        output.write((json.dumps({**line, **({"log": str(target_home / "logs" / "backend-migration.log")}
                                             if line.get("detached") else {})}) + "\n").encode())
        return SimpleNamespace(pid=780, poll=lambda: code, wait=lambda timeout=None: code)

    monkeypatch.setattr("hermes_cli.web_server_gateway._spawn_hermes_action", spawn)
    r = _client(home, monkeypatch).post("/api/gateway/migrate/backends/apply", json={})
    assert r.status_code == status, r.text
    if error:
        assert r.json()["error"] == error
        if error == "not_started":
            assert "cannot start the run outside x" in r.json()["message"]
    else:
        assert (r.json()["confirmed"], r.json()["launcher_exit_code"]) == (False, 7)


def test_an_already_migrated_home_is_never_offered_the_move_again(home, spawned, monkeypatch):
    client = _client(home, monkeypatch)  # the retired unit is still listed by the scan
    engine.write_manifest(home, _manifest("done"))
    plan = client.get("/api/gateway/migrate/backends/plan").json()
    assert plan["plan"]["nothing_to_do"] is True and plan["plan"]["expect_token"] is None
    assert plan["run"]["state"] == "done"
    r = client.post("/api/gateway/migrate/backends/apply", json={})
    assert (r.status_code, r.json()["error"]) == (409, "nothing_to_do")
    assert spawned == []


class InsideGatewayHost(FakeHost):
    """The answering process is the gateway itself (its Webapp's listener)."""

    def gateway_state(self):
        return {**super().gateway_state(), "unit": "hermes-gateway", "installed": True}

    def own_unit(self):
        return "hermes-gateway.service"


def test_the_gateway_answering_the_button_does_not_block_the_detached_run(home, spawned, monkeypatch):
    monkeypatch.setattr(routes, "_host", lambda target_home: InsideGatewayHost(target_home))
    client = TestClient(web_server.app)
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    plan = client.get("/api/gateway/migrate/backends/plan").json()["plan"]
    assert not any("runs inside" in b for b in plan["blockers"]), plan["blockers"]
    assert client.post("/api/gateway/migrate/backends/apply", json={}).status_code == 200


def test_a_start_line_is_this_launchs_even_when_a_busy_launcher_wrote_after_it(home, spawned):
    """A start line naming this home's run log still counts as this launch's answer even if a busy line
    appears in the same output."""
    import tempfile
    with tempfile.TemporaryFile() as output:
        launcher_says(home, output, confirmed=True)
        output.write((json.dumps({"applied": False, "mode": "apply", "error": "already_running",
                                  "message": "busy"}) + "\n").encode())
        assert routes._launcher_report(home, output)["confirmed"] is True


# --------------------------------------------------------------------------- final check


@pytest.mark.parametrize("password", ["pa'ss-7c1e", "pa:ss-7c1e", "pa%40ss-7c1e", 'pa"ss-7c1e', "pa;ss-7c1e"])
def test_the_status_log_tail_never_carries_a_url_password(home, spawned, monkeypatch, password):
    """URL userinfo may hold ', :, %40, " or ; : everything up to the last @ of the authority is redacted."""
    (home / "logs" / "backend-migration.log").write_text(
        f"public URL https://admin:{password}@hermes.example/x?y=1 checked\n", encoding="utf-8")
    body = _client(home, monkeypatch).get("/api/gateway/migrate/backends/status").json()
    tail = "\n".join(body["log_tail"])
    assert "ss-7c1e" not in tail and "admin" not in tail, tail
    assert "https://***@hermes.example/x?y=1 checked" in tail


class NothingHere(FakeHost):
    def backends(self):
        return []


def test_a_blocker_without_any_old_backend_is_not_a_finding(home, spawned, monkeypatch):
    """No backend found, and config.yaml cannot be parsed: the plan has a blocker (not "nothing to do"),
    but nothing was found, so the System page must not say an older server runs."""
    (home / "config.yaml").write_text("model: [unclosed\n", encoding="utf-8")
    monkeypatch.setattr(routes, "_host", lambda target_home: NothingHere(target_home))
    client = TestClient(web_server.app)
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    plan = client.get("/api/gateway/migrate/backends/plan").json()["plan"]
    assert plan["blockers"] and plan["nothing_to_do"] is False and plan["old_backend"] is None
    assert plan["found"] is False


def test_an_unreadable_manifest_refusal_from_the_launcher_is_a_refusal(home, spawned):
    """The engine's ``manifest_unreadable`` JSON carries no ``mode``; it is still the launcher's answer."""
    import tempfile
    with tempfile.TemporaryFile() as output:
        output.write((json.dumps({"applied": False, "error": "manifest_unreadable",
                                  "message": "cannot trust it"}) + "\n").encode())
        assert routes._launcher_report(home, output)["error"] == "manifest_unreadable"


# --------------------------------------------------------------------------- launcher exit 6 (manual-only host)


class BecomesSystemUnitHost(FakeHost):
    """Supported when the route checks; a system unit (manual steps only) by the time the launcher plans."""

    switched = False

    def backends(self):
        [item] = super().backends()
        if BecomesSystemUnitHost.switched:
            item.scope, item.unit_path = "system", "/etc/systemd/system/hermes-dashboard.service"
        return [item]


def test_a_launcher_exit_6_is_the_unsupported_refusal_with_the_steps(home, spawned, monkeypatch):
    """``--apply --detach`` exits 6 with the manual steps on a manual-only host (it starts nothing): the
    route answers what its own pre-check answers for an unsupported plan, never 202 unconfirmed."""
    BecomesSystemUnitHost.switched = False
    monkeypatch.setattr(routes, "_host", lambda target_home: BecomesSystemUnitHost(target_home))

    def spawn(subcommand, name, *, env_overrides=None, output=None):
        spawned.append(list(subcommand))
        BecomesSystemUnitHost.switched = True
        output.write((json.dumps({"applied": False, "dry_run": False,
                                  "plan": {"unsupported": [{"unit": "x"}]}}) + "\n").encode())
        return SimpleNamespace(pid=781, poll=lambda: 6, wait=lambda timeout=None: 6)

    monkeypatch.setattr("hermes_cli.web_server_gateway._spawn_hermes_action", spawn)
    client = TestClient(web_server.app)
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    r = client.post("/api/gateway/migrate/backends/apply?profile=default", json={})
    assert r.status_code == 409, r.text
    body = r.json()
    assert body["error"] == "unsupported" and body["launcher_exit_code"] == 6
    [u] = body["unsupported"]
    assert u["unit"] == "hermes-dashboard.service" and u["steps"], u
    assert all(s.startswith("hermes -p default ") for s in u["steps"] if s.startswith("hermes ")), u["steps"]
    assert len(spawned) == 1
    # The same answer as the route's own pre-check once the plan is unsupported (nothing spawned then).
    again = client.post("/api/gateway/migrate/backends/apply?profile=default", json={})
    assert (again.status_code, again.json()["error"]) == (409, "unsupported")
    assert again.json()["unsupported"] == body["unsupported"] and len(spawned) == 1


class ManualUnitHost(FakeHost):
    """A system-scope unit (manual steps only), with optional foreign-home or legacy-unit gateway conflicts."""

    def __init__(self, home, *, foreign_home=None, legacy=False):
        super().__init__(home)
        self._foreign_home = foreign_home
        self._legacy = legacy

    def backends(self):
        [item] = super().backends()
        item.scope, item.unit_path = "system", "/etc/systemd/system/hermes-dashboard.service"
        return [item]

    def foreign_gateway_home(self):
        return self._foreign_home

    def legacy_gateway_units(self):
        return self._legacy


@pytest.mark.parametrize(("kw", "expected_reason"), [
    ({"foreign_home": "/srv/other-hermes"}, "HERMES_HOME=/srv/other-hermes"),
    ({"legacy": True}, "hermes.service"),
])
def test_manual_plan_runs_gateway_safety_refusals_and_withholds_steps(home, spawned, monkeypatch, kw, expected_reason):
    monkeypatch.setattr(routes, "_host", lambda target_home: ManualUnitHost(target_home, **kw))
    client = TestClient(web_server.app)
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    data = client.get("/api/gateway/migrate/backends/plan").json()
    assert data["exit_code"] == engine.EXIT_BLOCKED
    plan = data["plan"]
    assert any(expected_reason in b for b in plan["blockers"]), plan["blockers"]
    assert plan["unsupported"] and all(u["steps"] == [] for u in plan["unsupported"]), plan["unsupported"]


def test_unknown_turns_blocks_inside_old_cgroup_and_respects_force_interrupt(home, spawned, monkeypatch):
    """Stripping the cgroup-only blocker before ``_owner_blockers`` keeps the unknown-turns warning when
    the answering server runs inside the old unit, and ``force_interrupt=True`` clears that blocker."""
    monkeypatch.setattr(gateway_migrate_backends_plan, "_own_cgroup_inside", lambda plan: True)
    client = _client(home, monkeypatch, running_turns=None)
    plan = client.get("/api/gateway/migrate/backends/plan").json()["plan"]
    assert any("turn activity could not be read" in b for b in plan["blockers"]), plan["blockers"]
    assert not any("runs inside" in b for b in plan["blockers"]), plan["blockers"]

    blocked = client.post("/api/gateway/migrate/backends/apply", json={})
    assert (blocked.status_code, blocked.json()["error"]) == (409, "blocked")

    forced = client.post("/api/gateway/migrate/backends/apply", json={"force_interrupt": True})
    assert forced.status_code == 200, forced.text
    assert "--force-interrupt" in spawned[-1]["argv"]


# --------------------------------------------------------------------------- launcher output stays per launch


def test_the_generic_action_status_route_never_serves_the_launchers_output(home, spawned, monkeypatch):
    """The launcher's output (which can hold the whole plan) is not an entry of the shared action-log table,
    so the unscoped, unscrubbed ``/api/actions/{name}/status`` answers 404 for it, before and after a launch."""
    client = _client(home, monkeypatch)
    assert client.get("/api/actions/gateway-migrate-backends/status").status_code == 404
    assert client.post("/api/gateway/migrate/backends/apply", json={}).status_code == 200
    assert client.get("/api/actions/gateway-migrate-backends/status").status_code == 404


def test_another_profiles_launcher_line_never_answers_this_launch(home, spawned, monkeypatch):
    """A refusal another profile's launcher writes while this home's launcher runs (exit 3: another run holds
    this home's lock) is not this launch's answer: it stays "already running", never "not started" with the
    other profile's message. This launch's own output file is removed once read."""
    import hermes_cli.web_server_gateway as gw

    def spawn(subcommand, name, **kw):
        spawned.append(list(subcommand))
        shared = Path(gw._ACTION_LOG_DIR) / "gateway-migrate-backends.log"
        shared.parent.mkdir(parents=True, exist_ok=True)
        with shared.open("a", encoding="utf-8") as fh:  # what a launcher in the shared action log would add
            fh.write(json.dumps({"detached": False, "mode": "apply", "outcome": "refused",
                                 "message": "work profile refusal"}) + "\n")
        return SimpleNamespace(pid=790, poll=lambda: 3, wait=lambda timeout=None: 3)

    monkeypatch.setattr("hermes_cli.web_server_gateway._spawn_hermes_action", spawn)
    r = _client(home, monkeypatch).post("/api/gateway/migrate/backends/apply", json={})
    assert (r.status_code, r.json()["error"]) == (409, "already_running"), r.text
    assert "work profile refusal" not in r.text
    assert len(spawned) == 1 and list((home / "logs").glob("backend-migration-launch-*")) == []


def _home_files(home: Path) -> set[Path]:
    return {f for f in home.rglob("*") if f.is_file()}


@pytest.mark.parametrize(("outcome", "status"), [("started", 200), ("refused", 409), ("timeout", 202)])
def test_a_launchs_output_never_has_a_file_in_the_home(home, spawned, monkeypatch, outcome, status):
    """The launcher writes into an anonymous file the route holds open: nothing appears in the home (where
    the Files browser could list and read it) while the launch runs, nor after it, and its answer is read."""
    import subprocess
    before = _home_files(home)
    during: list[set[Path]] = []

    def spawn(subcommand, name, **kw):
        during.append(_home_files(home) - before)
        assert during[-1] == set(), during  # the launcher's output has no file in the home
        if outcome == "started":
            line = {"detached": True, "mode": "apply", "confirmed": True, "log": str(home / "logs" / "backend-migration.log")}
        else:
            line = {"detached": False, "mode": "apply", "outcome": "refused", "message": "could not start"}
        kw["output"].write((json.dumps(line) + "\n").encode())
        code = 0 if outcome == "started" else 3

        def wait(timeout=None):
            if outcome == "timeout":
                raise subprocess.TimeoutExpired("hermes", timeout)
            return code
        return SimpleNamespace(pid=793, poll=lambda: None if outcome == "timeout" else code, wait=wait)

    monkeypatch.setattr("hermes_cli.web_server_gateway._spawn_hermes_action", spawn)
    r = _client(home, monkeypatch).post("/api/gateway/migrate/backends/apply", json={})
    assert during == [set()], during
    assert _home_files(home) - before == set()
    assert r.status_code == status, r.text
    if outcome == "refused":
        assert (r.json()["error"], r.json()["message"]) == ("not_started", "could not start")


def test_two_profiles_launching_at_once_each_read_only_their_own_answer(home, work, spawned, monkeypatch):
    """Both launchers are in flight together and both refuse: each launch answers with its own reason."""
    import threading
    both = threading.Barrier(2, timeout=10)

    def spawn(subcommand, name, **kw):
        who = "work" if Path(kw["env_overrides"]["HERMES_HOME"]) == work else "default"
        both.wait()
        kw["output"].write((json.dumps({"detached": False, "mode": "apply", "outcome": "refused",
                                        "message": f"{who} reason"}) + "\n").encode())
        both.wait()
        return SimpleNamespace(pid=794, poll=lambda: 3, wait=lambda timeout=None: 3)

    monkeypatch.setattr("hermes_cli.web_server_gateway._spawn_hermes_action", spawn)
    client = _client(home, monkeypatch)
    answers: dict = {}
    threads = [threading.Thread(target=lambda p=p: answers.__setitem__(p, client.post(
        "/api/gateway/migrate/backends/apply", params={"profile": p}, json={}))) for p in ("default", "work")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    for p in ("default", "work"):
        assert (answers[p].status_code, answers[p].json()["message"]) == (409, f"{p} reason"), answers[p].text


def test_a_restore_of_a_hand_started_backend_answers_with_the_engines_refusal(home, spawned, monkeypatch):
    """``--restore --detach`` of a hand-started old backend exits 3 in the launcher itself with a refusal line:
    the route answers 409 ``not_started`` with the engine's own message (which names the terminal command)."""
    message = ("Nothing changed: the old backend was started by hand, and a background run is a transient systemd "
               "unit that stops what it starts when the run ends. Restore from a terminal, without --detach: "
               "hermes -p default gateway migrate --backends --restore.")

    def spawn(subcommand, name, *, env_overrides=None, output=None):
        spawned.append(list(subcommand))
        output.write((json.dumps({"detached": False, "mode": "restore", "outcome": "refused",
                                  "message": message}) + "\n").encode())
        return SimpleNamespace(pid=795, poll=lambda: 3, wait=lambda timeout=None: 3)

    monkeypatch.setattr("hermes_cli.web_server_gateway._spawn_hermes_action", spawn)
    engine.write_manifest(home, _manifest("done", started_at=1.0))
    r = _client(home, monkeypatch).post("/api/gateway/migrate/backends/restore")
    assert (r.status_code, r.json()["error"], r.json()["message"]) == (409, "not_started", message), r.text
    assert "--restore" in spawned[-1]


@pytest.mark.parametrize("turns", [1, None])
def test_a_by_hand_plan_held_by_chat_activity_lists_its_steps_when_forced(home, spawned, monkeypatch, turns):
    """The card's "Show the steps anyway": the read-only plan with force_interrupt lists a by-hand plan's steps
    (exit 6) where the plain one only blocks, and starts nothing."""
    host = ManualUnitHost(home)
    host.turns = turns
    monkeypatch.setattr(routes, "_host", lambda target_home: host)
    client = TestClient(web_server.app)
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    held = client.get("/api/gateway/migrate/backends/plan").json()
    assert held["exit_code"] == engine.EXIT_BLOCKED and held["plan"]["blockers"], held["plan"]
    assert all(u["steps"] == [] for u in held["plan"]["unsupported"])
    forced = client.get("/api/gateway/migrate/backends/plan", params={"force_interrupt": "true"}).json()
    assert forced["exit_code"] == engine.EXIT_UNSUPPORTED and forced["plan"]["blockers"] == [], forced["plan"]
    assert forced["plan"]["unsupported"][0]["steps"]
    # A system unit shares the planner's network namespace: no "check inside the container" note.
    assert "outside its container" not in forced["plan"]["unsupported"][0]["instructions"]
    assert spawned == []

"""Two lifecycle invariants, using real SQLite and a local GitHub HTTP contract."""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_connect import connect


@pytest.fixture
def github(tmp_path, monkeypatch):
    state = {"conclusion": "success", "head": "a" * 40, "reads": 0, "requests": []}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state["requests"].append(self.path)
            sha = state["head"]
            if self.path == "/graphql":
                value = {"data": {"repository": {"pullRequest": {
                    "headRefOid": sha, "baseRefName": "main", "state": "OPEN",
                    "baseRef": {"branchProtectionRule": {"requiredStatusChecks": [
                        {"context": "required", "app": {"databaseId": 1}}]}}}}}}
            elif "/rules/branches/" in self.path:
                value = [[]]
            elif "/check-runs" in self.path:
                run = {"id": 42, "name": "required", "head_sha": sha,
                       "app": {"id": 1}, "status": "in_progress" if state["conclusion"] == "pending" else "completed", "conclusion": state["conclusion"],
                       "html_url": "https://github.com/acme/repo/actions/runs/42"}
                if state.get("stale"):
                    run["head_sha"] = "b" * 40
                runs = [] if state.get("missing") else [run]
                value = [{"total_count": 100 + len(runs), "check_runs": [
                    {**run, "id": 1000 + i, "name": "optional", "conclusion": "skipped"}
                    for i in range(100)]}, {"total_count": 100 + len(runs), "check_runs": runs}]
                if state.get("race"):
                    state["race"]()
                if state.get("head_change"):
                    state["head"] = "b" * 40
            elif "/statuses" in self.path:
                value = [[]]
            elif "/pulls/" in self.path:
                value = {"head": {"sha": sha}, "base": {"ref": "main"}, "state": "open"}
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(value).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    shim = tmp_path / "bin"
    shim.mkdir()
    gh = shim / "gh"
    gh.write_text(f"#!{sys.executable}\nimport sys,urllib.request\n"
                  f"u='http://127.0.0.1:{server.server_port}/'+sys.argv[2]\n"
                  "print(urllib.request.urlopen(u).read().decode())\n")
    gh.chmod(0o755)
    monkeypatch.setenv("PATH", str(shim) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    kb.init_db()
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.platforms("linux")
def test_pr_completion_requires_current_required_evidence(github):
    with connect() as conn:
        for conclusion in ("failure", "pending", "cancelled", "timed_out", "action_required", "neutral", "skipped", None, "success"):
            github.update(conclusion=conclusion, head="a" * 40)
            tid = kb.create_task(conn, title="Publish", completion_contract="acme/repo")
            ok = kb.complete_task(conn, tid, result="done", metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
            assert ok is (conclusion == "success")
            task = kb.get_task(conn, tid)
            assert (task.status == "done") is ok
            receipts = [json.loads(r[0]) for r in conn.execute(
                "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,))]
            assert receipts and receipts[-1]["head_sha"] == "a" * 40
            if not ok:
                assert task.status in {"running", "ready", "blocked", "review"}
                assert "retry" in receipts[-1]["recovery"]
                assert receipts[-1]["checks"][0]["id"] == 42
        for fault in ("missing", "stale", "head_change"):
            github.update(conclusion="success", head="a" * 40)
            github[fault] = True
            tid = kb.create_task(conn, title=fault, completion_contract="acme/repo")
            assert not kb.complete_task(conn, tid, result="done", metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
            assert kb.get_task(conn, tid).status != "done"
            github.pop(fault)
        # Omission and a sibling repository cannot downgrade the stored declaration.
        tid = kb.create_task(conn, title="publish", completion_contract="acme/repo")
        assert not kb.complete_task(conn, tid, summary="local green")
        assert not kb.complete_task(conn, tid, result="done", metadata={"published_pr": "https://github.com/other/repo/pull/7"})
        before = len(github["requests"])
        local = kb.create_task(conn, title="local", completion_contract="local-only")
        assert kb.complete_task(conn, local, summary="https://github.com/acme/repo/pull/7 is background context")
        assert len(github["requests"]) == before


@pytest.mark.platforms("linux")
def test_acceptance_receipts_and_terminal_write_share_run_ownership(github):
    with connect() as conn:
        for conclusion in ("success", "failure"):
            tid = kb.create_task(conn, title="race", completion_contract="acme/repo")
            owner = kb.claim_task(conn, tid)
            run_id = owner.current_run_id
            def reclaim():
                with connect() as rival:
                    assert kb.block_task(rival, tid, reason="Reassigned during acceptance")
                    assert kb.unblock_task(rival, tid)
                    github["replacement"] = kb.claim_task(rival, tid).current_run_id
            github.update(conclusion=conclusion, race=reclaim)
            assert not kb.complete_task(conn, tid, result="done", expected_run_id=run_id,
                metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
            assert kb.get_task(conn, tid).current_run_id == github["replacement"]
            assert github["replacement"] != run_id
            assert kb.get_task(conn, tid).status != "done"
            assert conn.execute("SELECT count(*) FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,)).fetchone()[0] == 0
            github.pop("race")


@pytest.fixture
def cold_secret_profiles(tmp_path, monkeypatch):
    from pathlib import Path

    from hermes_cli import env_loader, plugins

    launch = tmp_path / "launch"
    target = launch / "profiles" / "worker"
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setenv("GH_TOKEN", "launch-token")
    monkeypatch.setenv("GITHUB_TOKEN", "launch-github-token")
    monkeypatch.setenv("GH_CONFIG_DIR", str(launch / "gh"))
    for home in (launch, target):
        (home / ".env").parent.mkdir(parents=True, exist_ok=True)
        (home / ".env").write_text(f"FIXTURE_BOOTSTRAP={home.name}-bootstrap\n", encoding="utf-8")
        plugin = home / "plugins" / "fixture-vault"
        plugin.mkdir(parents=True)
        (plugin / "plugin.yaml").write_text("name: fixture-vault\n", encoding="utf-8")
        (plugin / "__init__.py").write_text('''
from agent.secret_sources.base import FetchResult, SecretSource
from agent.secret_scope import get_secret, is_multiplex_active
from hermes_constants import get_hermes_home

class FixtureVault(SecretSource):
    name = "fixturevault"
    shape = "bulk"

    def fetch(self, cfg, home_path):
        with (home_path / "fetch-count").open("a", encoding="utf-8") as stream:
            stream.write("fetch\\n")
        (home_path / "fetched").write_text("yes", encoding="utf-8")
        if (home_path / "fail").exists():
            raise RuntimeError("offline fixture fetch failure")
        if (home_path / "protected").exists():
            return FetchResult(secrets={"FIXTURE_PLANE_TOKEN": "must-not-publish"})
        return FetchResult(secrets={"GH_TOKEN": home_path.name + "-vault-token"})

def register(ctx):
    home = get_hermes_home()
    assert get_secret("FIXTURE_BOOTSTRAP") == home.name + "-bootstrap"
    (home / "multiplex").write_text(str(is_multiplex_active()), encoding="utf-8")
    (home / "registered").write_text(str(home), encoding="utf-8")
    ctx.register_secret_source(FixtureVault())
''', encoding="utf-8")
        (home / "config.yaml").write_text(
            "plugins:\n  enabled: [fixture-vault]\n"
            "secrets:\n  sources: [fixturevault]\n  fixturevault:\n    enabled: true\n",
            encoding="utf-8",
        )
    shim = tmp_path / "cold-bin"
    shim.mkdir()
    gh = shim / "gh"
    gh.write_text(f"#!{sys.executable}\nimport json, os, sys\n"
                  "if not os.environ.get('GH_TOKEN'):\n    sys.exit(4)\n"
                  "print(json.dumps({key: os.environ.get(key) for key in "
                  "('GH_TOKEN', 'GITHUB_TOKEN', 'GH_CONFIG_DIR', 'HERMES_HOME')}))\n",
                  encoding="utf-8")
    gh.chmod(0o755)
    monkeypatch.setenv("PATH", str(shim) + os.pathsep + os.environ["PATH"])
    env_loader.reset_secret_source_cache()
    try:
        yield launch, target
    finally:
        plugins._reset_plugin_managers_for_tests()
        env_loader.reset_secret_source_cache()


@pytest.mark.platforms("posix")
def test_cold_acceptance_loads_only_the_assignee_plugin_secrets(cold_secret_profiles):
    from hermes_cli import env_loader
    from hermes_cli.kanban_pr_acceptance import _api, _gh_env
    from hermes_constants import get_hermes_home, get_hermes_home_override
    from agent.secret_scope import is_multiplex_active

    launch, target = cold_secret_profiles
    original_home = get_hermes_home_override()
    original_multiplex = is_multiplex_active()
    assert env_loader.get_secret_source_values(target) == {}
    captured = _api("user", env=_gh_env(str(target)))
    assert captured["GH_TOKEN"] == "worker-vault-token"
    assert captured["GITHUB_TOKEN"] is None
    assert captured["GH_CONFIG_DIR"] != str(launch / "gh")
    assert captured["HERMES_HOME"] == str(target)
    assert (target / "registered").read_text(encoding="utf-8") == str(target)
    assert (target / "fetched").exists()
    assert not (launch / "registered").exists()
    assert os.environ["GH_TOKEN"] == "launch-token"
    assert get_hermes_home() == launch
    assert get_hermes_home_override() == original_home
    assert is_multiplex_active() == original_multiplex
    assert _gh_env(None) is None
    assert _api("user")["GH_TOKEN"] == "launch-token"


@pytest.mark.platforms("posix")
def test_cold_acceptance_fetch_failure_never_uses_launch_login(cold_secret_profiles):
    from agent.secret_scope import is_multiplex_active
    from hermes_cli.kanban_pr_acceptance import _api, _GateAuthError, _gh_env
    from hermes_constants import get_hermes_home, get_hermes_home_override

    launch, target = cold_secret_profiles
    (target / "fail").touch()
    original_home = get_hermes_home_override()
    original_multiplex = is_multiplex_active()
    with pytest.raises(_GateAuthError, match="no login"):
        _api("user", env=_gh_env(str(target)))
    assert (target / "fetched").exists()
    env = _gh_env(str(target))
    assert env is not None
    assert "GH_TOKEN" not in env and "GITHUB_TOKEN" not in env
    assert env["GH_CONFIG_DIR"] == str(target / "gh")
    assert os.environ["GH_TOKEN"] == "launch-token"
    assert get_hermes_home() == launch
    assert get_hermes_home_override() == original_home
    assert is_multiplex_active() == original_multiplex


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("routed", [False, True])
@pytest.mark.parametrize("multiplex", [False, True])
def test_discovery_binds_owner_secrets_and_restores_the_caller(cold_secret_profiles, routed, multiplex):
    from agent.secret_scope import (
        current_secret_scope, current_secret_scope_home, is_multiplex_active,
        reset_multiplex_context, reset_secret_scope, set_multiplex_context, set_secret_scope,
    )
    from hermes_cli.kanban_pr_acceptance import _gh_env
    from hermes_constants import get_hermes_home_override

    launch, target = cold_secret_profiles
    home = target if routed else launch
    caller_scope = {"FIXTURE_BOOTSTRAP": "caller-only"}
    secret_token = set_secret_scope(caller_scope, profile_home="caller")
    multiplex_token = set_multiplex_context(multiplex)
    original_home = get_hermes_home_override()
    try:
        env = _gh_env(str(home))
        assert env is not None
        assert (home / "registered").read_text() == str(home)
        assert (home / "multiplex").read_text() == str(routed or multiplex)
        assert env["GH_TOKEN"] == ("worker-vault-token" if routed else "launch-token")
        # A second discovery must reuse a correctly loaded manager, not a cached failed plugin.
        assert _gh_env(str(home))["GH_TOKEN"] == env["GH_TOKEN"]
        assert current_secret_scope() is caller_scope
        assert current_secret_scope_home() == "caller"
        assert is_multiplex_active() is multiplex
        assert get_hermes_home_override() == original_home
        assert os.environ["GH_TOKEN"] == "launch-token"
    finally:
        reset_multiplex_context(multiplex_token)
        reset_secret_scope(secret_token)


def test_config_backend_refusal_is_a_redacted_auth_receipt(cold_secret_profiles, monkeypatch):
    from agent.secret_scope import current_secret_scope, is_multiplex_active
    from hermes_cli import config_backend
    from hermes_cli.kanban_pr_acceptance import collect_acceptance
    from hermes_constants import get_hermes_home_override

    _launch, target = cold_secret_profiles
    (target / "protected").touch()

    class ProtectedBackend(config_backend.FileBackend):
        name = "fixture-remote"

        def protected_env_names(self):
            return frozenset({"FIXTURE_PLANE_TOKEN"})

    monkeypatch.setattr(config_backend, "get_config_backend", lambda: ProtectedBackend())
    original = get_hermes_home_override(), current_secret_scope(), is_multiplex_active()
    receipt = collect_acceptance("acme/repo", "https://github.com/acme/repo/pull/7", assignee="worker")
    assert (target / "fetched").exists()
    assert receipt["classification"] == "auth" and not receipt["ok"]
    assert "must-not-publish" not in json.dumps(receipt)
    assert "FIXTURE_PLANE_TOKEN" not in os.environ
    assert os.environ["GH_TOKEN"] == "launch-token"
    assert (get_hermes_home_override(), current_secret_scope(), is_multiplex_active()) == original


@pytest.mark.platforms("posix")
def test_acceptance_freezes_one_env_even_when_a_source_fails(github, cold_secret_profiles, tmp_path, monkeypatch):
    from hermes_cli import kanban_pr_acceptance as acceptance

    _launch, target = cold_secret_profiles
    (target / "fail").touch()
    monkeypatch.setenv("PATH", str(tmp_path / "bin") + os.pathsep + os.environ["PATH"])
    monkeypatch.setattr(acceptance, "_assignee_profile_home", lambda assignee: str(target))
    original_env = acceptance._gh_env
    original_run = acceptance.subprocess.run
    builds, used = [], []

    def build(home):
        env = original_env(home)
        builds.append(env)
        return env

    def capture(*args, **kwargs):
        if args[0][:2] == ["gh", "api"]:
            used.append(kwargs["env"])
        return original_run(*args, **kwargs)

    monkeypatch.setattr(acceptance, "_gh_env", build)
    invalid = acceptance.collect_acceptance("acme/repo", "https://github.com/other/repo/pull/7", assignee="worker")
    assert not invalid["ok"] and builds == []
    monkeypatch.setattr(acceptance.subprocess, "run", capture)
    receipt = acceptance.collect_acceptance("acme/repo", "https://github.com/acme/repo/pull/7", assignee="worker")
    assert receipt["ok"]
    assert len(github["requests"]) == 5
    assert len(builds) == 1
    assert len(used) == 5 and all(env is builds[0] for env in used)
    # Discovery refresh + explicit hydration can each try once; API reads never re-pull.
    assert len((target / "fetch-count").read_text().splitlines()) == 2
    assert "GH_TOKEN" not in builds[0] and "GITHUB_TOKEN" not in builds[0]
    assert os.environ["GH_TOKEN"] == "launch-token"


# --- #122689: acceptance must read the repo as the ASSIGNEE profile's gh login ---

@pytest.mark.platforms("posix")
def test_acceptance_runs_gh_as_the_assignee_profile(tmp_path, monkeypatch):
    """The gh child env carries the assignee's own GH credentials (its .env),
    never the ambient/launch residue, and an invisible repo is classified
    `auth` naming the repository — not a retryable `infra` failure."""
    from pathlib import Path

    launch_home = tmp_path / "home"
    launch_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    assignee_home = launch_home / "profiles" / "b"
    assignee_home.mkdir(parents=True)
    (assignee_home / ".env").write_text("GH_TOKEN=b-token\n", encoding="utf-8")
    # Ambient residue that must NOT decide the login.
    monkeypatch.setenv("GH_TOKEN", "launch-token")
    monkeypatch.setenv("GH_CONFIG_DIR", "/nonexistent/launch/gh")

    env_dump = tmp_path / "gh_env.json"
    shim = tmp_path / "bin"
    shim.mkdir()
    gh = shim / "gh"
    gh.write_text(f"#!{sys.executable}\nimport json, os\n"
                  f"json.dump(dict(os.environ), open({str(env_dump)!r}, 'w'))\n"
                  "print(json.dumps({'data': {'repository': None}}))\n")
    gh.chmod(0o755)
    monkeypatch.setenv("PATH", str(shim) + os.pathsep + os.environ["PATH"])
    kb.init_db()
    with connect() as conn:
        tid = kb.create_task(conn, title="as-b", completion_contract="acme/repo", assignee="b")
        assert not kb.complete_task(conn, tid, result="done",
                                    metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
        assert kb.get_task(conn, tid).status != "done"
        receipts = [json.loads(r[0]) for r in conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,))]
        assert receipts[-1]["classification"] == "auth"
        assert "acme/repo" in receipts[-1]["detail"]
    captured = json.loads(env_dump.read_text())
    assert captured["GH_TOKEN"] == "b-token"
    assert captured.get("GH_CONFIG_DIR") != "/nonexistent/launch/gh"
    assert "credentials" in (kb.get_task(conn, tid).last_failure_error or "")


@pytest.mark.platforms("posix")
def test_assignee_without_own_gh_login_never_falls_through_to_ambient_login(tmp_path, monkeypatch):
    """An assignee profile with no GH_TOKEN/GH_CONFIG_DIR of its own must not inherit the
    launch user's ~/.config/gh (HOME/XDG_CONFIG_HOME stay the launch process's): gh is pinned
    to a profile-owned config dir, its 'not logged in' exit is classified `auth` naming the profile."""
    launch_home = tmp_path / "home"
    launch_home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    assignee_home = launch_home / "profiles" / "b"
    assignee_home.mkdir(parents=True)
    (assignee_home / ".env").write_text("", encoding="utf-8")
    monkeypatch.setenv("GH_TOKEN", "launch-token")
    monkeypatch.setenv("GH_CONFIG_DIR", "/nonexistent/launch/gh")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "launch-xdg"))

    env_dump = tmp_path / "gh_env.json"
    shim = tmp_path / "bin"
    shim.mkdir()
    gh = shim / "gh"
    # Real gh: GH_CONFIG_DIR wins; a config dir without hosts.yml means "not logged in" (exit 4).
    gh.write_text(f"#!{sys.executable}\nimport json, os, pathlib, sys\n"
                  f"pathlib.Path({str(env_dump)!r}).write_text(json.dumps(dict(os.environ)), encoding='utf-8')\n"
                  "if 'GH_CONFIG_DIR' in os.environ and not os.path.exists(os.environ['GH_CONFIG_DIR']):\n"
                  "    sys.exit(4)\n"
                  "print(json.dumps({'data': {'repository': None}}))\n")
    gh.chmod(0o755)
    monkeypatch.setenv("PATH", str(shim) + os.pathsep + os.environ["PATH"])
    kb.init_db()
    with connect() as conn:
        tid = kb.create_task(conn, title="as-b", completion_contract="acme/repo", assignee="b")
        assert not kb.complete_task(conn, tid, result="done",
                                    metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
        receipt = json.loads(conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,)).fetchone()[0])
    assert receipt["classification"] == "auth"
    assert "'b'" in receipt["detail"] and "no login" in receipt["detail"]
    captured = json.loads(env_dump.read_text(encoding="utf-8-sig"))
    assert captured["GH_CONFIG_DIR"] == str(assignee_home / "gh")
    assert "GH_TOKEN" not in captured and "GITHUB_TOKEN" not in captured


def test_assigned_card_with_unresolvable_profile_is_auth_not_ambient(tmp_path, monkeypatch):
    """A card assigned to a profile that no longer exists must not run gh as the completing
    process's ambient login: classification `auth` naming the profile, gh never invoked."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))  # any gh spawn would fail as infra
    kb.init_db()
    with connect() as conn:
        tid = kb.create_task(conn, title="as-ghost", completion_contract="acme/repo", assignee="ghost")
        assert not kb.complete_task(conn, tid, result="done",
                                    metadata={"published_pr": "https://github.com/acme/repo/pull/7"})
        receipt = json.loads(conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='pr_acceptance'", (tid,)).fetchone()[0])
    assert receipt["classification"] == "auth"
    assert "'ghost'" in receipt["detail"] and "cannot be resolved" in receipt["detail"]

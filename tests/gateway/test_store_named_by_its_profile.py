"""Callers name a store by its profile folder, never by a ``<home>/state.db`` path they build themselves.

The fixture keeps the SQLite store under another file name in the profile folder, so a caller that builds
``<home>/state.db`` by hand names a file that does not exist, while a caller that names the profile reaches the store.
"""
import pytest

import persistence_assembly


@pytest.fixture
def store_elsewhere(monkeypatch):
    monkeypatch.setattr(persistence_assembly, "_state_db", lambda scope: scope.profile_dir / "store-elsewhere.db")


def test_the_session_store_keeps_one_handle_per_profile_however_its_folder_is_spelled(tmp_path, monkeypatch,
                                                                                       store_elsewhere):
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    real = tmp_path / "home"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    monkeypatch.setenv("HERMES_HOME", str(link))
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    try:
        token = set_hermes_home_override(real.resolve())  # a profile turn's scope names the canonical folder
        try:
            assert store._scoped_persistence() is store._routing_persistence
        finally:
            reset_hermes_home_override(token)
        assert list(store._db_handles) == [real.resolve()]
    finally:
        store.close_all_db_handles()


def _home():
    from hermes_constants import get_hermes_home
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    return home


def _seed(home, session_id, source, **fields):
    from persistence_assembly import open_persistence, profile_scope
    from tests.fixtures.storage_seed import seed_messages, seed_session
    with open_persistence(profile_scope(home=home)) as db:
        seed_session(db, session_id, source, **fields)
        seed_messages(db, session_id, {"role": "user", "content": "hello"})


@pytest.mark.asyncio
async def test_the_editor_lists_and_attaches_through_the_profile(monkeypatch, store_elsewhere):
    from acp_adapter import gateway_session_meta
    from acp_adapter.gateway_server import GatewayACPAgent
    home = _home()
    _seed(home, "s", "acp", cwd=str(home))

    async def no_options(*args):
        return []
    monkeypatch.setattr(gateway_session_meta, "model_state", lambda info: None)
    monkeypatch.setattr(gateway_session_meta, "session_config_options", no_options)
    agent = GatewayACPAgent()
    listed = await agent.list_sessions(cwd=str(home))
    assert [row.session_id for row in listed.sessions] == ["s"]
    fields = await agent._session_fields("s", {}, "editor")
    assert fields["field_meta"]["hermes"]["sessionProvenance"]


def test_a_wake_finds_the_session_in_its_profiles_store(tmp_path, store_elsewhere):
    from types import SimpleNamespace
    from gateway.wake import session_owned_by_profile
    home = tmp_path / "profiles" / "worker"
    home.mkdir(parents=True)
    _seed(home, "s", "api_server")
    assert session_owned_by_profile(SimpleNamespace(_runtime_profile_homes=[("worker", home)]), "worker", "s")


def test_approvals_suggest_reads_the_current_profiles_store(store_elsewhere, capsys):
    import json
    from argparse import Namespace
    from hermes_cli.approvals_suggest import suggest_command
    home = _home()
    _seed(home, "s", "cli")
    assert suggest_command(Namespace(db=None, days=0, min_count=1, limit=20, apply_indices=None, json=True)) == 0
    assert json.loads(capsys.readouterr().out)["proposals"] == []


def test_a_recreated_worktree_is_locked_for_the_profiles_store(tmp_path, monkeypatch, store_elsewhere):
    import subprocess
    from hermes_cli.tui_worktree_resume import record_tui_worktree, recreate_tui_worktree
    from hermes_cli.worktree_ops import _retained_store
    _seed(_home(), "s", "tui")
    repo = tmp_path / "repo"
    git = ["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "-C", str(repo)]
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    subprocess.run([*git, "commit", "-q", "--allow-empty", "-m", "fixture"], check=True)
    tree = repo / ".worktrees" / "kept"
    record_tui_worktree({"path": str(tree), "branch": "hermes/kept", "repo_root": str(repo)})
    assert recreate_tui_worktree(str(tree), "s") == str(tree)
    listing = subprocess.run([*git, "worktree", "list", "--porcelain"], capture_output=True, text=True,
                             check=True).stdout
    reason = next(line for line in listing.splitlines() if line.startswith("locked "))
    assert _retained_store(reason.split(" db=", 1)[1]) is not None, reason


def test_a_manual_cron_runs_usage_line_reads_the_profiles_store(store_elsewhere):
    from persistence.conversation.sessions import NewSession
    from persistence.conversation.usage import UsageDelta
    from persistence_assembly import open_persistence, profile_scope
    from tools.cronjob_tools import _run_session_usage
    with open_persistence(profile_scope(home=_home()), eager=True) as store:
        epoch = store.runtime.begin_epoch(instance_id="test")
        store.sessions.create(NewSession(session_id="run-1", source="cron", model="model-a"))
        store.admissions.admit(epoch=epoch, principal_id="cron-owner", session_id="run-1",
                               request_id="cron:job-1:exec-1", payload={"text": "go"})
        store.usage.queue_usage("run-1", UsageDelta(api_call_count=2))
        assert store.usage.flush_usage()
    assert _run_session_usage("job-1", "exec-1") == {"api_calls": 2, "model": "model-a"}


def test_the_tui_reads_another_profiles_store_through_its_folder(tmp_path, monkeypatch, store_elsewhere):
    from tui_gateway import server
    foreign = tmp_path / "profiles" / "code"
    foreign.mkdir(parents=True)
    _seed(foreign, "seed", "cli", model="m")
    monkeypatch.setattr(server, "_profile_home", lambda name: foreign if name == "code" else None)
    with server._profile_db({"profile": "code"}) as db:
        assert db is not None and db.sessions.get("seed") is not None


def test_an_unopenable_profile_store_is_named_by_the_file_that_holds_it(tmp_path, store_elsewhere):
    from tui_gateway import server
    profile = tmp_path / "profiles" / "code"
    (profile / "store-elsewhere.db").mkdir(parents=True)  # sqlite cannot open a directory
    with pytest.raises(RuntimeError, match="store-elsewhere.db"):
        server._open_profile_session_db(str(profile))


def test_the_clis_session_lookups_read_the_current_profiles_store(store_elsewhere):
    from hermes_cli import main
    _seed(_home(), "s", "cli")
    assert main._resolve_last_session(source="cli") == "s"

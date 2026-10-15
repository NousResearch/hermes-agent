"""Durability of a Telegram project session (#138777): rows, lineage, git metadata, transitions.

The workspace lives on the session ROW, so these drive the real SessionStore + SessionDB through
compression rotation, cold resume, rollback and concurrent route transitions.
"""

import asyncio
import subprocess
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.project_session import (
    PROJECT_MARKER_KEY,
    ProjectWorkspaceError,
    project_cwd_for_turn,
)
from gateway.session import SessionSource, SessionStore
from hermes_state import SessionDB


@pytest.fixture
def env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="c", user_id="u", chat_type="dm")
    store = SessionStore(home / "sessions", GatewayConfig())
    db = SessionDB(home / "state.db")
    store._db = db
    yield SimpleNamespace(home=home, source=source, store=store, db=db, tmp=tmp_path)
    db.close()


def _git(path, *args):
    subprocess.run(["git", "-C", str(path), *args], check=True, capture_output=True)


def _start_project_session(env, workspace, *, project_id="p_1", **kwargs):
    old = env.store.get_or_create_session(env.source)
    env.db.create_session(old.session_id, "telegram", session_key=old.session_key)
    new = env.store.reset_session_for_project(
        old.session_key, env.source, expected_session_id=old.session_id,
        cwd=str(workspace), project_id=project_id, **kwargs,
    )
    assert new is not None
    return old, new


async def _turn_cwd(env, entry):
    runner = SimpleNamespace(session_store=env.store)
    return await project_cwd_for_turn(runner, entry, env.source)


def test_project_cwd_survives_compression_and_cold_resume(env):
    workspace = env.tmp / "ws"
    workspace.mkdir()
    _, new = _start_project_session(env, workspace)
    child_id = "compressed-child"
    env.db.publish_compression_child(
        parent_session_id=new.session_id, child_session_id=child_id, source="telegram",
        messages=[{"role": "user", "content": "summary"}], model_config={"max_iterations": 3},
        require_compression_lease=False,
    )
    advanced = env.store.advance_compression_session(new.session_key, new.session_id, child_id)
    assert advanced is not None and advanced.session_id == child_id
    # The routing metadata is not the source of truth: the row marker rides the rotation.
    assert env.db.get_session_model_config_value(child_id, PROJECT_MARKER_KEY)["cwd"] == str(workspace)
    assert asyncio.run(_turn_cwd(env, advanced)) == str(workspace)

    cold = SessionStore(env.home / "sessions", GatewayConfig())
    cold._db = env.db
    resumed = cold.lookup_by_session_key(new.session_key)
    assert resumed.session_id == child_id
    assert asyncio.run(project_cwd_for_turn(SimpleNamespace(session_store=cold), resumed, env.source)) == str(workspace)


def test_resume_switch_session_does_not_lose_the_project_workspace(env):
    workspace = env.tmp / "ws"
    workspace.mkdir()
    _, new = _start_project_session(env, workspace)
    other = env.store.reset_session(new.session_key)  # plain /new afterwards
    assert asyncio.run(_turn_cwd(env, other)) == ""
    back = env.store.switch_session(new.session_key, new.session_id)  # /resume the project session
    assert back.metadata == {}  # switch_session drops routing metadata ...
    assert asyncio.run(_turn_cwd(env, back)) == str(workspace)  # ... the row marker still anchors it


def test_selected_non_git_folder_does_not_inherit_the_old_sessions_git_repo(env):
    repo = env.tmp / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "feature-x")
    plain = env.tmp / "plain-notes"
    plain.mkdir()
    old = env.store.get_or_create_session(env.source)
    env.db.create_session(old.session_id, "telegram", session_key=old.session_key, cwd=str(repo))
    env.db.update_session_cwd(old.session_id, str(repo), "feature-x", str(repo), replace_git_meta=True)
    new = env.store.reset_session_for_project(
        old.session_key, env.source, expected_session_id=old.session_id,
        cwd=str(plain), project_id="p_plain",
    )
    row = env.db.get_session(new.session_id)
    assert row["cwd"] == str(plain)
    assert not row["git_repo_root"] and not row["git_branch"]

    # The production Desktop resolver files it under the selected project, not the old repo.
    from tui_gateway import git_probe
    from tui_gateway.project_tree import build_tree

    project = {
        "id": "p_plain", "name": "Plain", "primary_path": str(plain),
        "folders": [{"path": str(plain), "label": "Plain", "is_primary": True}],
    }
    tree = build_tree([project], [row], [], resolve=git_probe.resolve)
    owner = next(p for p in tree["projects"] if p["id"] == "p_plain")
    assert new.session_id in owner["sessionIds"]


def test_selected_git_folder_records_its_own_branch_and_root(env):
    repo = env.tmp / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "trunk")
    _, new = _start_project_session(env, repo, git_branch="trunk", git_repo_root=str(repo))
    row = env.db.get_session(new.session_id)
    assert (row["git_branch"], row["git_repo_root"]) == ("trunk", str(repo))


def test_deleted_workspace_fails_closed_each_turn(env):
    workspace = env.tmp / "ws"
    workspace.mkdir()
    _, new = _start_project_session(env, workspace)
    assert asyncio.run(_turn_cwd(env, new)) == str(workspace)
    workspace.rmdir()
    with pytest.raises(ProjectWorkspaceError, match="no longer available"):
        asyncio.run(_turn_cwd(env, new))


def test_non_local_terminal_backend_is_rejected_not_remapped(env, monkeypatch):
    workspace = env.tmp / "ws"
    workspace.mkdir()
    _, new = _start_project_session(env, workspace)
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    with pytest.raises(ProjectWorkspaceError, match="local terminal backend"):
        asyncio.run(_turn_cwd(env, new))


def test_ordinary_session_has_no_project_cwd(env):
    entry = env.store.get_or_create_session(env.source)
    env.db.create_session(entry.session_id, "telegram", session_key=entry.session_key)
    assert asyncio.run(_turn_cwd(env, entry)) == ""


def test_missing_session_database_refuses_and_leaves_the_route(env):
    workspace = env.tmp / "ws"
    workspace.mkdir()
    old = env.store.get_or_create_session(env.source)
    env.store._db = None
    with pytest.raises(RuntimeError, match="no session database"):
        env.store.reset_session_for_project(
            old.session_key, env.source, expected_session_id=old.session_id,
            cwd=str(workspace), project_id="p",
        )
    assert env.store.peek_session_id(old.session_key) == old.session_id


@pytest.mark.parametrize("site", ["git_metadata", "routing_save"])
def test_failure_after_row_creation_removes_the_unpublished_row(env, monkeypatch, site):
    workspace = env.tmp / "ws"
    workspace.mkdir()
    old = env.store.get_or_create_session(env.source)
    env.db.create_session(old.session_id, "telegram", session_key=old.session_key)
    if site == "git_metadata":
        monkeypatch.setattr(env.db, "update_session_cwd", lambda *a, **k: (_ for _ in ()).throw(OSError("x")))
    else:
        monkeypatch.setattr(env.store, "_persist_routing_data", lambda *a: (_ for _ in ()).throw(OSError("x")))
    with pytest.raises(OSError):
        env.store.reset_session_for_project(
            old.session_key, env.source, expected_session_id=old.session_id,
            cwd=str(workspace), project_id="p",
        )
    assert env.store.peek_session_id(old.session_key) == old.session_id
    assert env.db.session_count() == 1  # the half-created row did not survive
    assert env.db.get_session(old.session_id)["ended_at"] is None


def test_unpublished_row_cleanup_failure_is_reported_not_hidden(env, monkeypatch, caplog):
    workspace = env.tmp / "ws"
    workspace.mkdir()
    old = env.store.get_or_create_session(env.source)
    env.db.create_session(old.session_id, "telegram", session_key=old.session_key)
    monkeypatch.setattr(env.store, "_persist_routing_data", lambda *a: (_ for _ in ()).throw(OSError("save")))
    monkeypatch.setattr(env.db, "delete_session", lambda *a, **k: (_ for _ in ()).throw(OSError("rm")))
    with caplog.at_level("ERROR"), pytest.raises(OSError, match="save"):
        env.store.reset_session_for_project(
            old.session_key, env.source, expected_session_id=old.session_id,
            cwd=str(workspace), project_id="p",
        )
    assert "could not be removed" in caplog.text
    assert env.store.peek_session_id(old.session_key) == old.session_id


def test_concurrent_route_transitions_publish_exactly_one_winner(env):
    import threading

    workspace = env.tmp / "ws"
    workspace.mkdir()
    old = env.store.get_or_create_session(env.source)
    env.db.create_session(old.session_id, "telegram", session_key=old.session_key)
    results = []
    barrier = threading.Barrier(4)

    def project():
        barrier.wait()
        results.append(("project", env.store.reset_session_for_project(
            old.session_key, env.source, expected_session_id=old.session_id,
            cwd=str(workspace), project_id="p")))

    threads = [threading.Thread(target=project) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    winners = [entry for _, entry in results if entry is not None]
    assert len(winners) == 1
    assert env.store.peek_session_id(old.session_key) == winners[0].session_id
    assert env.db.session_count() == 2  # old + the single winner; losers wrote nothing


def test_plain_reset_after_a_project_session_clears_the_pin(env):
    from tools.terminal_tool import get_session_cwd, resolve_task_overrides

    from gateway.project_session import bind_workspace, release_workspace

    workspace = env.tmp / "ws"
    workspace.mkdir()
    _, new = _start_project_session(env, workspace)
    bind_workspace(new.session_key, new.session_id, str(workspace))
    assert get_session_cwd(new.session_key) == str(workspace)
    release_workspace(new.session_key, new.session_id)
    assert get_session_cwd(new.session_key) is None
    assert resolve_task_overrides(new.session_key) == {}
    assert resolve_task_overrides(new.session_id) == {}

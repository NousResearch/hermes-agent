from __future__ import annotations

import pytest

import hermes_state
from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from tui_gateway import project_tree


def _source(platform: Platform = Platform.TELEGRAM) -> SessionSource:
    return SessionSource(
        platform=platform,
        chat_id=f"chat-{platform.value}",
        chat_type="dm",
        user_id="user-1",
        scope_id="workspace-1" if platform == Platform.SLACK else None,
    )


def _runner(
    tmp_path,
    monkeypatch,
    *,
    terminal_cwd: str | None,
    multiplex_profiles: bool = False,
) -> GatewayRunner:
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    if terminal_cwd is None:
        monkeypatch.delenv("TERMINAL_CWD", raising=False)
    else:
        monkeypatch.setenv("TERMINAL_CWD", terminal_cwd)
    return GatewayRunner(
        GatewayConfig(
            sessions_dir=tmp_path / "sessions",
            multiplex_profiles=multiplex_profiles,
        )
    )


def _project_session(row):
    return {
        "id": row["id"],
        "cwd": row.get("cwd"),
        "git_branch": row.get("git_branch") or "",
        "git_repo_root": row.get("git_repo_root") or "",
        "started_at": row.get("started_at") or 0,
        "last_active": row.get("last_active") or 0,
        "title": row.get("title"),
        "preview": None,
        "source": row.get("source") or "telegram",
    }


@pytest.mark.parametrize(
    "platform",
    [Platform.TELEGRAM, Platform.SLACK, Platform.DISCORD],
)
def test_runner_persists_configured_cwd_and_project_tree_uses_it(
    tmp_path, monkeypatch, platform
):
    workspace = tmp_path / "Assistant"
    workspace.mkdir()
    runner = _runner(
        tmp_path,
        monkeypatch,
        terminal_cwd=f"  {workspace}  ",
    )
    try:
        entry = runner.session_store.get_or_create_session(_source(platform))
        row = runner.session_store._db.get_session(entry.session_id)
        assert row["cwd"] == str(workspace)

        project = {
            "id": "personal-assistant",
            "name": "Personal Assistant",
            "primary_path": str(workspace),
            "archived": False,
            "folders": [{"path": str(workspace), "is_primary": True}],
        }
        tree = project_tree.build_tree(
            [project],
            [_project_session(row)],
            [],
            resolve=lambda _cwd: None,
            hydrate=True,
        )
        personal = next(
            node for node in tree["projects"] if node["id"] == "personal-assistant"
        )
        grouped_ids = {
            session["id"]
            for repo in personal["repos"]
            for group in repo["groups"]
            for session in group["sessions"]
        }
        assert grouped_ids == {entry.session_id}
        assert not any(
            node["id"] == project_tree.NO_PROJECT_ID and node["sessionCount"]
            for node in tree["projects"]
        )
    finally:
        runner.session_store._db.close()


def test_reset_and_force_new_sessions_persist_configured_cwd(tmp_path, monkeypatch):
    workspace = tmp_path / "Assistant"
    workspace.mkdir()
    runner = _runner(tmp_path, monkeypatch, terminal_cwd=str(workspace))
    source = _source()
    try:
        first = runner.session_store.get_or_create_session(source)
        reset = runner.session_store.reset_session(first.session_key)
        forced = runner.session_store.get_or_create_session(source, force_new=True)

        assert reset is not None
        assert runner.session_store._db.get_session(reset.session_id)["cwd"] == str(
            workspace
        )
        assert runner.session_store._db.get_session(forced.session_id)["cwd"] == str(
            workspace
        )
    finally:
        runner.session_store._db.close()


def test_missing_configured_cwd_stays_unbound(tmp_path, monkeypatch):
    runner = _runner(tmp_path, monkeypatch, terminal_cwd=None)
    try:
        entry = runner.session_store.get_or_create_session(_source())
        assert runner.session_store._db.get_session(entry.session_id)["cwd"] is None
    finally:
        runner.session_store._db.close()


def test_existing_unbound_session_is_not_backfilled(tmp_path, monkeypatch):
    source = _source()
    first_runner = _runner(tmp_path, monkeypatch, terminal_cwd=None)
    first = first_runner.session_store.get_or_create_session(source)
    assert first_runner.session_store._db.get_session(first.session_id)["cwd"] is None
    first_runner.session_store._db.close()

    workspace = tmp_path / "Assistant"
    workspace.mkdir()
    restarted = _runner(tmp_path, monkeypatch, terminal_cwd=str(workspace))
    try:
        recovered = restarted.session_store.get_or_create_session(source)
        assert recovered.session_id == first.session_id
        assert restarted.session_store._db.get_session(first.session_id)["cwd"] is None
    finally:
        restarted.session_store._db.close()


def test_multiplex_runner_does_not_persist_launch_profile_cwd(
    tmp_path, monkeypatch
):
    primary_workspace = tmp_path / "primary"
    primary_workspace.mkdir()
    runner = _runner(
        tmp_path,
        monkeypatch,
        terminal_cwd=str(primary_workspace),
        multiplex_profiles=True,
    )
    source = _source()
    source.profile = "secondary"
    try:
        entry = runner.session_store.get_or_create_session(source)
        assert runner.session_store._db.get_session(entry.session_id)["cwd"] is None
    finally:
        runner.session_store._db.close()

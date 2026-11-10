"""Opt-in review rework ceiling uses the native blocked -> promote lifecycle."""
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.kanban_review_rework_budget import max_review_rejections


@pytest.fixture
def guard_settings(monkeypatch):
    from hermes_cli import config

    settings = {"kanban": {
        "review_rework_boards": ["pilot-board"],
        "max_review_rejections": 2,
    }}
    monkeypatch.setattr(config, "load_config_readonly", lambda: settings)
    return settings["kanban"]


@pytest.fixture
def board_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    with kbc.connect() as conn:
        yield conn


def _reject_once(conn, task_id: str, n: int):
    impl = kb.claim_task(conn, task_id, claimer=f"builder:{n}")
    assert impl is not None
    assert kb.request_review(
        conn, task_id, summary=f"implementation {n}", reviewer="reviewer",
        expected_run_id=impl.current_run_id,
    )
    rev = kb.claim_review_task(conn, task_id, claimer=f"reviewer:{n}")
    assert rev is not None
    assert kb.request_changes(
        conn, task_id, reason=f"finding {n}",
        expected_run_id=rev.current_run_id,
    ) == (True, "builder")
    return kb.get_task(conn, task_id)


def test_review_limit_off_by_default(monkeypatch, guard_settings):
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "pilot-board")
    guard_settings["max_review_rejections"] = 0
    assert max_review_rejections() is None


def test_board_allowlist_scopes_guard(monkeypatch, guard_settings):
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "different-board")
    assert max_review_rejections() is None
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "PILOT-BOARD")
    assert max_review_rejections() == 2


@pytest.mark.parametrize("value", ["-1", "21", "junk"])
def test_bad_selected_board_limit_fails_closed(monkeypatch, guard_settings, value):
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "pilot-board")
    guard_settings["max_review_rejections"] = value
    with pytest.raises(ValueError, match="max_review_rejections"):
        max_review_rejections()


def test_repeated_review_rejections_park_in_native_blocked_state(
    monkeypatch, guard_settings, board_db,
):
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "pilot-board")
    task_id = kb.create_task(board_db, title="Review budget test", assignee="builder")
    first = _reject_once(board_db, task_id, 1)
    assert first.status == "ready"
    assert first.assignee == "builder"
    last = _reject_once(board_db, task_id, 2)
    assert last.status == "blocked"
    assert last.block_kind == "needs_input"
    assert last.assignee == "builder"
    assert kb.claim_task(board_db, task_id, claimer="builder:not-allowed") is None
    events = kb.list_events(board_db, task_id)
    assert len([e for e in events if e.kind == "changes_requested"]) == 2
    blocked = [e for e in events if e.kind == "blocked"][-1]
    assert blocked.payload["kind"] == "needs_input"
    assert "2/2" in blocked.payload["reason"]
    assert kb.promote_task(board_db, task_id, actor="operator", reason="approved") == (True, None)
    assert kb.get_task(board_db, task_id).status == "ready"


def test_unselected_board_preserves_normal_review_flow(
    monkeypatch, guard_settings, board_db,
):
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "other-board")
    guard_settings["max_review_rejections"] = 1
    task_id = kb.create_task(board_db, title="normal", assignee="builder")
    for i in range(3):
        task = _reject_once(board_db, task_id, i)
        assert task.status == "ready"
        assert task.block_kind != "needs_input"

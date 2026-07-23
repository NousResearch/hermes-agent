from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _contract(**overrides):
    value = {
        "category": "decision_required",
        "action": "Choose the rollout window.",
        "recommendation": "Use the next low-traffic window.",
        "why_now": "The tested candidate is ready.",
        "urgency": "Before the next release cut.",
        "consequence": "The release remains paused.",
        "review_url": "https://example.com/reviews/123",
        "reply_format": "Reply with: approve or reject.",
    }
    value.update(overrides)
    return value


def _blocked_with_action(conn, **overrides):
    tid = kb.create_task(conn, title="Owner decision", assignee="worker")
    assert kb.block_task(
        conn,
        tid,
        reason="A decision is required.",
        kind="needs_input",
        owner_action=_contract(**overrides),
        actor="orchestrator",
    )
    return tid


@pytest.mark.parametrize("category", sorted(kb.OWNER_ACTION_CATEGORIES))
def test_all_owner_action_categories_are_valid(category):
    assert kb.validate_owner_action(_contract(category=category))["category"] == category


@pytest.mark.parametrize(
    "mutator,error",
    [
        (lambda c: {**c, "category": "unknown"}, "category"),
        (lambda c: {**c, "action": ""}, "action"),
        (lambda c: {**c, "recommendation": 42}, "recommendation"),
        (lambda c: {**c, "review_url": "file:///tmp/review"}, r"http\(s\)"),
        (lambda c: {**c, "extra": "no"}, "unknown fields"),
        (lambda c: {**c, "urgency": "x" * 201}, "at most 200"),
    ],
)
def test_owner_action_validation_rejects_invalid_fields(mutator, error):
    with pytest.raises(ValueError, match=error):
        kb.validate_owner_action(mutator(_contract()))


def test_owner_action_validation_enforces_aggregate_notification_bound():
    contract = _contract(
        action="a" * 500,
        recommendation="r" * 500,
        why_now="w" * 500,
        consequence="c" * 500,
        review_url="https://example.com/" + "x" * 1500,
    )
    with pytest.raises(ValueError, match="total at most"):
        kb.validate_owner_action(contract)


def test_owner_action_open_repeat_and_material_change_events(kanban_home):
    with kb.connect_closing() as conn:
        tid = _blocked_with_action(conn)
        assert kb.block_task(
            conn,
            tid,
            reason="Repeated explanation.",
            kind="needs_input",
            owner_action=_contract(),
            actor="orchestrator",
        )
        assert kb.get_task(conn, tid).block_recurrences == 1
        assert kb.set_owner_action(
            conn,
            tid,
            _contract(urgency="Reply today."),
            actor="orchestrator",
        ) == "changed"

        events = kb.list_events(conn, tid)
        assert [event.kind for event in events].count("owner_action_opened") == 1
        assert [event.kind for event in events].count("owner_action_changed") == 1
        blocked = [event for event in events if event.kind == "blocked"]
        assert len(blocked) == 1
        assert blocked[0].payload["represented_by_owner_action"] is True
        assert kb._has_sticky_block(conn, tid) is True


def test_owner_action_requires_a_human_action_block(kanban_home):
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="Legacy blocker", assignee="worker")
        assert kb.block_task(conn, tid, reason="ordinary")
        with pytest.raises(ValueError, match="block kind"):
            kb.set_owner_action(conn, tid, _contract())
        with pytest.raises(ValueError, match="requires block kind"):
            kb.block_task(
                conn,
                tid,
                reason="dependency",
                kind="dependency",
                owner_action=_contract(),
            )


@pytest.mark.parametrize(
    "transition",
    [
        lambda conn, tid: kb.unblock_task(conn, tid),
        lambda conn, tid: kb.complete_task(conn, tid, summary="done"),
        lambda conn, tid: kb.archive_task(conn, tid),
        lambda conn, tid: kb.schedule_task(conn, tid, reason="wait until tomorrow"),
    ],
    ids=["unblock", "complete", "archive", "schedule"],
)
def test_lifecycle_transitions_clear_active_owner_action(kanban_home, transition):
    with kb.connect_closing() as conn:
        tid = _blocked_with_action(conn)
        assert transition(conn, tid)
        action = kb.get_task(conn, tid).owner_action
        assert action["active"] is False
        assert action["resolution"] == "cleared"
        assert [event.kind for event in kb.list_events(conn, tid)].count(
            "owner_action_cleared"
        ) == 1


@pytest.mark.parametrize("outcome", sorted(kb.OWNER_ACTION_RESOLUTIONS))
def test_explicit_owner_action_resolution_is_audited_without_routing_task(
    kanban_home, outcome
):
    with kb.connect_closing() as conn:
        tid = _blocked_with_action(conn)
        assert kb.resolve_owner_action(
            conn,
            tid,
            outcome=outcome,
            actor="owner",
            note="Recorded response.",
        )
        task = kb.get_task(conn, tid)
        assert task.status == "blocked"
        assert task.owner_action["active"] is False
        assert task.owner_action["resolution"] == outcome
        event = [
            event for event in kb.list_events(conn, tid)
            if event.kind == "owner_action_resolved"
        ][-1]
        assert event.payload["outcome"] == outcome
        assert kb.list_active_owner_actions(conn) == []


def test_resolution_note_rejects_non_strings(kanban_home):
    with kb.connect_closing() as conn:
        tid = _blocked_with_action(conn)
        with pytest.raises(ValueError, match="note must be a string"):
            kb.resolve_owner_action(
                conn, tid, outcome="accepted", note=cast(Any, 42)
            )
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.owner_action is not None
        assert task.owner_action["active"] is True


def test_legacy_rows_and_malformed_projection_remain_readable(kanban_home):
    db_path = kb.kanban_db_path()
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="Legacy task", assignee="worker")
        conn.execute("UPDATE tasks SET owner_action = 'not-json' WHERE id = ?", (tid,))
        conn.commit()
        assert kb.get_task(conn, tid).owner_action is None

    # Simulate a board created before the additive owner_action column.
    with kb.connect_closing(db_path) as conn:
        conn.execute("ALTER TABLE tasks DROP COLUMN owner_action")
        conn.commit()
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    with kb.connect_closing(db_path) as migrated:
        columns = {row["name"] for row in migrated.execute("PRAGMA table_info(tasks)")}
        assert "owner_action" in columns
        assert kb.get_task(migrated, tid).owner_action is None


def test_stale_run_guard_rolls_back_owner_action(kanban_home):
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="Guarded block", assignee="worker")
        assert not kb.block_task(
            conn,
            tid,
            reason="decision",
            kind="needs_input",
            owner_action=_contract(),
            expected_run_id=999,
        )
        task = kb.get_task(conn, tid)
        assert task.status == "ready"
        assert task.owner_action is None
        assert not any(
            event.kind.startswith("owner_action_") for event in kb.list_events(conn, tid)
        )

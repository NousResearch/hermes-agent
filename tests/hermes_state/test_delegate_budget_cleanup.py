"""Session deletion must retire nested delegation budgets (#52484)."""

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    with SessionDB(tmp_path / "state.db") as database:
        yield database


class TestBulkDeleteSessions:
    def test_empty_session_cleanup_preserves_budget_until_deletion(self, db):
        from tools.delegate_tool_config import (
            _session_children_counts,
            _session_children_lock,
            cleanup_session_budget,
        )

        session_id = "empty-budget-session"
        db.create_session(session_id, "cli")
        assert db.set_session_title(session_id, "Keep this session") is True
        with _session_children_lock:
            _session_children_counts[session_id] = 2
        try:
            assert db.delete_session_if_empty(session_id) is False
            with _session_children_lock:
                assert _session_children_counts[session_id] == 2
            assert db.delete_session(session_id) is True
            with _session_children_lock:
                assert session_id not in _session_children_counts
        finally:
            cleanup_session_budget(session_id)

        empty_id = "untitled-budget-session"
        db.create_session(empty_id, "cli")
        with _session_children_lock:
            _session_children_counts[empty_id] = 1
        try:
            assert db.delete_session_if_empty(empty_id) is True
            with _session_children_lock:
                assert empty_id not in _session_children_counts
        finally:
            cleanup_session_budget(empty_id)

    def test_cleans_all_cascade_delegate_budgets(self, db):
        from tools.delegate_tool import (
            _session_children_counts,
            _session_children_lock,
            cleanup_session_budget,
        )

        db.create_session("bulk-budget-parent", "cli")
        db.create_session(
            "bulk-budget-orchestrator",
            "cli",
            parent_session_id="bulk-budget-parent",
            model_config={"_delegate_from": "bulk-budget-parent"},
        )
        db.create_session(
            "bulk-budget-worker",
            "cli",
            parent_session_id="bulk-budget-orchestrator",
            model_config={"_delegate_from": "bulk-budget-orchestrator"},
        )
        budget_ids = {
            "bulk-budget-parent",
            "bulk-budget-orchestrator",
            "bulk-budget-worker",
        }
        with _session_children_lock:
            _session_children_counts.update(dict.fromkeys(budget_ids, 1))

        try:
            assert db.delete_sessions(["bulk-budget-parent"]) == 1
            with _session_children_lock:
                assert budget_ids.isdisjoint(_session_children_counts)
        finally:
            for session_id in budget_ids:
                cleanup_session_budget(session_id)


class TestListSessionsRich:
    def test_delete_parent_cleans_all_cascade_delegate_budgets(self, db):
        from tools.delegate_tool import (
            _session_children_counts,
            _session_children_lock,
            cleanup_session_budget,
        )

        db.create_session("budget-parent", "cli")
        db.create_session(
            "budget-orchestrator",
            "cli",
            parent_session_id="budget-parent",
            model_config={"_delegate_from": "budget-parent"},
        )
        db.create_session(
            "budget-worker",
            "cli",
            parent_session_id="budget-orchestrator",
            model_config={"_delegate_from": "budget-orchestrator"},
        )
        budget_ids = {
            "budget-parent",
            "budget-orchestrator",
            "budget-worker",
        }
        with _session_children_lock:
            _session_children_counts.update(dict.fromkeys(budget_ids, 1))

        try:
            assert db.delete_session("budget-parent") is True
            with _session_children_lock:
                assert budget_ids.isdisjoint(_session_children_counts)
        finally:
            for session_id in budget_ids:
                cleanup_session_budget(session_id)

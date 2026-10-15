"""Session deletion must retire nested delegation budgets (#52484)."""

import pytest

from persistence_assembly import open_persistence, profile_scope
from session_retention import after_sessions_removed
from tests.fixtures.storage_seed import seed_session


@pytest.fixture
def db(tmp_path):
    with open_persistence(profile_scope(home=tmp_path)) as database:
        yield database


class TestBulkDeleteSessions:
    def test_empty_session_cleanup_preserves_budget_until_deletion(self, db, tmp_path):
        from tools.delegate_tool_config import (
            _session_children_counts,
            _session_children_lock,
            cleanup_session_budget,
        )

        session_id = "empty-budget-session"
        seed_session(db, session_id, "cli")
        assert db.sessions.set_title(session_id, "Keep this session") is True
        with _session_children_lock:
            _session_children_counts[session_id] = 2
        try:
            removed = db.sessions.delete_if_empty(session_id, media_root=str(tmp_path))
            assert removed == ()
            after_sessions_removed(db, tmp_path, removed, reason="empty_sweep")
            with _session_children_lock:
                assert _session_children_counts[session_id] == 2
            removed = db.sessions.delete(session_id, media_root=str(tmp_path))
            assert removed == (session_id,)
            after_sessions_removed(db, tmp_path, removed, reason="delete")
            with _session_children_lock:
                assert session_id not in _session_children_counts
        finally:
            cleanup_session_budget(session_id)

        empty_id = "untitled-budget-session"
        seed_session(db, empty_id, "cli")
        with _session_children_lock:
            _session_children_counts[empty_id] = 1
        try:
            removed = db.sessions.delete_if_empty(empty_id, media_root=str(tmp_path))
            assert removed == (empty_id,)
            after_sessions_removed(db, tmp_path, removed, reason="empty_sweep")
            with _session_children_lock:
                assert empty_id not in _session_children_counts
        finally:
            cleanup_session_budget(empty_id)

    def test_cleans_all_cascade_delegate_budgets(self, db, tmp_path):
        from tools.delegate_tool import (
            _session_children_counts,
            _session_children_lock,
            cleanup_session_budget,
        )

        seed_session(db, "bulk-budget-parent", "cli")
        seed_session(
            db,
            "bulk-budget-orchestrator",
            "cli",
            parent_session_id="bulk-budget-parent",
            model_config={"_delegate_from": "bulk-budget-parent"},
        )
        seed_session(
            db,
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
            outcome = db.sessions.delete_many(["bulk-budget-parent"], media_root=str(tmp_path))
            assert outcome.deleted == 1
            assert set(outcome.removed_ids) == budget_ids
            after_sessions_removed(db, tmp_path, outcome.removed_ids, reason="delete")
            with _session_children_lock:
                assert budget_ids.isdisjoint(_session_children_counts)
        finally:
            for session_id in budget_ids:
                cleanup_session_budget(session_id)


class TestListSessionsRich:
    def test_delete_parent_cleans_all_cascade_delegate_budgets(self, db, tmp_path):
        from tools.delegate_tool import (
            _session_children_counts,
            _session_children_lock,
            cleanup_session_budget,
        )

        seed_session(db, "budget-parent", "cli")
        seed_session(
            db,
            "budget-orchestrator",
            "cli",
            parent_session_id="budget-parent",
            model_config={"_delegate_from": "budget-parent"},
        )
        seed_session(
            db,
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
            removed = db.sessions.delete("budget-parent", media_root=str(tmp_path))
            assert set(removed) == budget_ids
            after_sessions_removed(db, tmp_path, removed, reason="delete")
            with _session_children_lock:
                assert budget_ids.isdisjoint(_session_children_counts)
        finally:
            for session_id in budget_ids:
                cleanup_session_budget(session_id)

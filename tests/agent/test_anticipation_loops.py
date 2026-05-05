"""Tests for stale-task anticipation candidate generation."""

from datetime import datetime, timedelta, timezone

from agent.anticipation import AnticipationPermission
from agent.anticipation_loops import build_stale_task_candidates

NOW = datetime(2026, 5, 5, 18, 0, tzinfo=timezone.utc)


class FakeSessionDB:
    def __init__(self, sessions, messages_by_session):
        self.sessions = sessions
        self.messages_by_session = messages_by_session

    def list_sessions_rich(self, **kwargs):
        self.list_kwargs = kwargs
        return self.sessions

    def get_messages_as_conversation(self, session_id):
        return self.messages_by_session.get(session_id, [])


def ts(days_ago):
    return (NOW - timedelta(days=days_ago)).timestamp()


def test_stale_task_resurfacer_builds_candidate_from_unanswered_offer():
    db = FakeSessionDB(
        sessions=[
            {
                "id": "sess-1",
                "source": "telegram",
                "title": "Hermes anticipation planning",
                "last_active": ts(1),
                "message_count": 2,
                "preview": "Can we turn this into a plan?",
            }
        ],
        messages_by_session={
            "sess-1": [
                {"role": "user", "content": "Can we turn this into a plan?"},
                {"role": "assistant", "content": "If you want, I can turn this into a concrete implementation plan next."},
            ]
        },
    )

    candidates = build_stale_task_candidates(db, now=NOW, lookback_days=14, limit=5)

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.loop_id == "stale_task_resurfacer"
    assert candidate.proposed_permission is AnticipationPermission.SUGGEST
    assert candidate.confidence >= 0.7
    assert "Hermes anticipation planning" in candidate.title
    assert "If you want" in candidate.body
    assert candidate.dedupe_key.startswith("stale_task_resurfacer:sess-1:")


def test_stale_task_resurfacer_skips_when_later_user_response_completes_thread():
    db = FakeSessionDB(
        sessions=[{"id": "sess-1", "source": "telegram", "last_active": ts(1), "message_count": 3, "preview": ""}],
        messages_by_session={
            "sess-1": [
                {"role": "assistant", "content": "Next I can wire the dry-run command."},
                {"role": "user", "content": "Done, that part is complete."},
            ]
        },
    )

    assert build_stale_task_candidates(db, now=NOW, lookback_days=14, limit=5) == []


def test_stale_task_resurfacer_ignores_tool_output_with_unresolved_language():
    db = FakeSessionDB(
        sessions=[{"id": "sess-1", "source": "telegram", "last_active": ts(1), "message_count": 2, "preview": ""}],
        messages_by_session={
            "sess-1": [
                {"role": "user", "content": "Check the repo."},
                {"role": "tool", "content": "TODO: internal fixture output with next step secrets."},
            ]
        },
    )

    assert build_stale_task_candidates(db, now=NOW, lookback_days=14, limit=5) == []


def test_stale_task_resurfacer_skips_sessions_outside_lookback_and_current_session():
    db = FakeSessionDB(
        sessions=[
            {"id": "old", "source": "telegram", "last_active": ts(30), "message_count": 2, "preview": ""},
            {"id": "current", "source": "telegram", "last_active": ts(1), "message_count": 2, "preview": ""},
        ],
        messages_by_session={
            "old": [{"role": "assistant", "content": "Next step: build this."}],
            "current": [{"role": "assistant", "content": "Next step: build this."}],
        },
    )

    candidates = build_stale_task_candidates(
        db,
        now=NOW,
        lookback_days=14,
        limit=5,
        current_session_id="current",
    )

    assert candidates == []


def test_stale_task_resurfacer_orders_by_confidence_then_recency_and_limits():
    db = FakeSessionDB(
        sessions=[
            {"id": "low", "source": "telegram", "title": "Low", "last_active": ts(1), "message_count": 2, "preview": ""},
            {"id": "high", "source": "telegram", "title": "High", "last_active": ts(3), "message_count": 2, "preview": ""},
        ],
        messages_by_session={
            "low": [{"role": "assistant", "content": "We should revisit this later."}],
            "high": [{"role": "assistant", "content": "TODO: next step is to add the dry-run command."}],
        },
    )

    candidates = build_stale_task_candidates(db, now=NOW, lookback_days=14, limit=1)

    assert len(candidates) == 1
    assert "High" in candidates[0].title

"""Regression coverage for coalesced deferred reviews and durable cadence cleanup."""
from types import SimpleNamespace

import pytest

from agent import background_review
from agent.review_cadence import schedule_turn_review
from agent.review_idle_queue import ReviewIdleQueue
from hermes_state import SessionDB


def _plain_agent():
    return SimpleNamespace(_background_review_closing=False)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ({"review_memory": False, "review_skills": True}, {"review_memory": True, "review_skills": False}),
        ({"review_memory": True, "review_skills": False}, {"review_memory": False, "review_skills": True}),
    ],
)
def test_deferred_queue_coalesces_review_kinds_but_keeps_newest_snapshot(monkeypatch, first, second):
    queue = ReviewIdleQueue()
    monkeypatch.setattr(queue, "_ensure_thread", lambda: None)
    agent = _plain_agent()
    queue._now = lambda: 10.0

    assert queue.enqueue(
        agent, "profile::session", {**first, "messages_snapshot": ["old"], "task_cfg": {}},
    )
    queue._now = lambda: 20.0
    assert queue.enqueue(
        agent, "profile::session", {**second, "messages_snapshot": ["new"], "task_cfg": {}},
    )

    item = queue._pending["profile::session"]
    assert item.kwargs["messages_snapshot"] == ["new"]
    assert item.kwargs["review_memory"] is True
    assert item.kwargs["review_skills"] is True
    assert item.enqueued_at == 10.0


@pytest.mark.parametrize("failure", ["rejected", "raised"])
def test_failed_deferred_dispatch_refunds_every_merged_skill_claim(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(background_review, "load_background_review_settings", lambda: (True, {}))
    queue = ReviewIdleQueue()
    monkeypatch.setattr(queue, "_ensure_thread", lambda: None)
    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("session", "desktop")
        agent = SimpleNamespace(
            session_id="session",
            _session_db=db,
            _skill_nudge_interval=2,
            _iters_since_skill=2,
            valid_tool_names={"skill_manage"},
            _background_review_closing=False,
        )

        def defer(**kwargs):
            return queue.enqueue(
                agent, "profile::session", {**kwargs, "task_cfg": {"defer_max_age_s": 1}},
            )

        agent._spawn_background_review = defer
        schedule_turn_review(agent, [{"content": "first"}], final_response="done", interrupted=False)
        agent._iters_since_skill = 3
        schedule_turn_review(
            agent, [{"content": "newest"}], final_response="done", interrupted=False, review_memory=True,
        )

        item = queue._pending["profile::session"]
        assert item.kwargs["messages_snapshot"] == [{"content": "newest"}]
        assert item.kwargs["review_memory"] is True
        assert item.kwargs["review_skills"] is True
        assert len(item.claim_refunds) == 2
        assert db.get_meta("skill-review-cadence:session:session") == "0"

        def fail_spawn(**_kwargs):
            if failure == "raised":
                raise RuntimeError("spawn failed")
            return False

        agent._spawn_background_review_now = fail_spawn
        queued_at = item.enqueued_at
        queue._now = lambda: queued_at + 2.0
        assert queue._dispatch_one() is True

        # Both transactional claims (2 + 3) are returned, not merely the last one.
        assert db.get_meta("skill-review-cadence:session:session") == "5"
        assert queue.pending_count() == 0
        assert queue._dispatching == {}
    finally:
        db.close()


def test_cadence_maintenance_prunes_deleted_compression_root(tmp_path, monkeypatch):
    monkeypatch.setattr(background_review, "load_background_review_settings", lambda: (True, {}))
    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("root", "desktop")
        db.end_session("root", "compression")
        db.create_session("tip", "desktop", parent_session_id="root")
        agent = SimpleNamespace(
            session_id="tip",
            _session_db=db,
            _skill_nudge_interval=10,
            _iters_since_skill=2,
            valid_tool_names={"skill_manage"},
            _spawn_background_review=lambda **_kwargs: True,
        )

        schedule_turn_review(agent, [], final_response="done", interrupted=False)
        root_key = "skill-review-cadence:session:root"
        tip_key = "skill-review-cadence:session:tip"
        assert db.get_meta(root_key) == "2"

        assert db.delete_session("root") is True
        assert db.get_session("tip")["parent_session_id"] is None
        assert db.get_meta(root_key) == "2", "state_meta is intentionally not a session FK cascade"

        resumed = SimpleNamespace(
            session_id="tip",
            _session_db=db,
            _skill_nudge_interval=10,
            _iters_since_skill=1,
            valid_tool_names={"skill_manage"},
            _spawn_background_review=lambda **_kwargs: True,
        )
        schedule_turn_review(resumed, [], final_response="done", interrupted=False)

        assert db.get_meta(root_key) is None
        assert db.get_meta(tip_key) == "1"
    finally:
        db.close()

"""Tests for finalizer.check_session_before_exit (Slice 5)."""

from __future__ import annotations

import uuid

import pytest


@pytest.fixture
def clean_bus(tmp_path, monkeypatch):
    db_path = tmp_path / f"agent_bus_{uuid.uuid4().hex[:8]}.db"
    monkeypatch.setenv("AGENT_BUS_DB_PATH", str(db_path))
    from agent_bus import storage as _storage
    _storage._DB_CONN = None
    from agent_bus import core as _core
    monkeypatch.setattr(_core, "_slack_post_assignment", lambda *a, **k: (None, None))
    monkeypatch.setattr(_core, "_slack_reply", lambda *a, **k: True)
    monkeypatch.setattr(_core, "_notify_agent", lambda *a, **k: None)
    monkeypatch.setattr(_core, "_notify_openclaw", lambda *a, **k: None)
    monkeypatch.setattr(_core, "_notify_hermes_via_slack", lambda *a, **k: None)
    monkeypatch.setattr(_core, "_notify_user_of_outcome", lambda *a, **k: False)
    yield _core
    _storage._DB_CONN = None


class TestCheckSessionBeforeExit:
    def test_empty_when_no_tasks(self, clean_bus):
        from agent_bus.finalizer import check_session_before_exit
        assert check_session_before_exit("openclaw") == []

    def test_returns_open_tasks_for_agent(self, clean_bus):
        from agent_bus.finalizer import check_session_before_exit
        core = clean_bus
        t1 = core.assign_task(
            from_agent="hermes", to_agent="openclaw",
            goal="do work 1", skip_prior_learnings=True,
        )
        t2 = core.assign_task(
            from_agent="hermes", to_agent="openclaw",
            goal="do work 2", skip_prior_learnings=True,
        )
        result = check_session_before_exit("openclaw")
        task_ids = {r["task_id"] for r in result}
        assert task_ids == {t1["task_id"], t2["task_id"]}

    def test_excludes_terminal_tasks(self, clean_bus):
        from agent_bus.finalizer import check_session_before_exit
        core = clean_bus
        t1 = core.assign_task(
            from_agent="hermes", to_agent="openclaw",
            goal="will complete", skip_prior_learnings=True,
        )
        core.ack_task(task_id=t1["task_id"], agent="openclaw")
        core.complete_task(task_id=t1["task_id"], agent="openclaw", result="done")

        t2 = core.assign_task(
            from_agent="hermes", to_agent="openclaw",
            goal="still pending", skip_prior_learnings=True,
        )
        result = check_session_before_exit("openclaw")
        assert [r["task_id"] for r in result] == [t2["task_id"]]

    def test_different_agent_not_included(self, clean_bus):
        from agent_bus.finalizer import check_session_before_exit
        core = clean_bus
        core.assign_task(
            from_agent="hermes", to_agent="openclaw",
            goal="openclaw only", skip_prior_learnings=True,
        )
        assert check_session_before_exit("hermes") == []

    def test_keep_alive_included(self, clean_bus):
        from agent_bus.finalizer import check_session_before_exit
        core = clean_bus
        t = core.assign_task(
            from_agent="hermes", to_agent="openclaw",
            goal="running long", skip_prior_learnings=True,
        )
        core.ack_task(task_id=t["task_id"], agent="openclaw")
        core.keep_alive_task(task_id=t["task_id"], agent="openclaw")
        result = check_session_before_exit("openclaw")
        assert len(result) == 1
        assert result[0]["status"] == "keep-alive"

    def test_result_shape(self, clean_bus):
        from agent_bus.finalizer import check_session_before_exit
        core = clean_bus
        core.assign_task(
            from_agent="hermes", to_agent="openclaw",
            goal="check shape", skip_prior_learnings=True,
        )
        result = check_session_before_exit("openclaw")
        assert len(result) == 1
        r = result[0]
        for key in ("task_id", "status", "goal", "age_sec", "from_agent"):
            assert key in r

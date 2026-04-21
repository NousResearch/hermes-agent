"""Integration tests: finalizer gate plumbed into core.complete_task / fail_task /
keep_alive_task / amend_learning.

Uses a tmp SQLite DB via AGENT_BUS_DB_PATH env var. Slack broadcasts and
wiki writes are noops because we stub out those side-effect hooks.
"""

import os
import uuid
from pathlib import Path
from unittest import mock

import pytest

from agent_bus import finalizer


@pytest.fixture
def core_module(tmp_path, monkeypatch):
    """Reset agent_bus.core + storage with a tmp DB, no slack / no wiki writes."""
    db_path = tmp_path / f"agent_bus_{uuid.uuid4().hex[:8]}.db"
    monkeypatch.setenv("AGENT_BUS_DB_PATH", str(db_path))

    # Drop any cached connection from earlier tests
    from agent_bus import storage as _storage
    _storage._DB_CONN = None

    from agent_bus import core as _core
    # Stub Slack
    monkeypatch.setattr(_core, "_slack_post_assignment", lambda *a, **k: (None, None))
    monkeypatch.setattr(_core, "_slack_reply", lambda *a, **k: True)
    # Stub notify + user ping
    monkeypatch.setattr(_core, "_notify_agent", lambda *a, **k: None)
    monkeypatch.setattr(_core, "_notify_openclaw", lambda *a, **k: None)
    monkeypatch.setattr(_core, "_notify_hermes_via_slack", lambda *a, **k: None)
    monkeypatch.setattr(_core, "_notify_user_of_outcome", lambda *a, **k: False)
    # Stub wiki learning writes — point to tmp
    wiki_dir = tmp_path / "wiki_memory"
    wiki_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(_core, "_wiki_memory_dir", lambda: wiki_dir)
    monkeypatch.setattr(_core, "_append_wiki_log", lambda *a, **k: None)
    monkeypatch.setattr(_core, "_rebuild_agent_bus_moc", lambda: None)

    yield _core

    _storage._DB_CONN = None


def _assign(core, **over):
    defaults = dict(
        from_agent="hermes",
        to_agent="openclaw",
        goal="do the thing",
        priority="P2",
    )
    defaults.update(over)
    return core.assign_task(**defaults)


class TestCompleteTaskIdempotent:
    def test_simple_done(self, core_module):
        core = core_module
        t = _assign(core)
        core.ack_task(task_id=t["task_id"], agent="openclaw")
        done = core.complete_task(task_id=t["task_id"], agent="openclaw", result="ok")
        assert done["status"] == "done"

    def test_same_outcome_reclose_is_idempotent(self, core_module):
        core = core_module
        t = _assign(core)
        core.ack_task(task_id=t["task_id"], agent="openclaw")
        core.complete_task(task_id=t["task_id"], agent="openclaw", result="first close")
        # second close with same outcome must NOT raise, must return existing row
        again = core.complete_task(task_id=t["task_id"], agent="openclaw", result="second close")
        assert again["status"] == "done"
        # result should NOT be overwritten on idempotent close
        assert again["result"] == "first close"

    def test_flip_done_to_fail_rejected(self, core_module):
        core = core_module
        t = _assign(core)
        core.ack_task(task_id=t["task_id"], agent="openclaw")
        core.complete_task(task_id=t["task_id"], agent="openclaw", result="ok")
        with pytest.raises(ValueError) as ei:
            core.fail_task(task_id=t["task_id"], agent="openclaw", reason="changed mind")
        assert finalizer.ERR_INVALID_TERMINAL_FLIP in str(ei.value)

    def test_pending_to_done_rejected(self, core_module):
        core = core_module
        t = _assign(core)
        # no ack → status remains pending
        with pytest.raises(ValueError) as ei:
            core.complete_task(task_id=t["task_id"], agent="openclaw", result="skipping ack")
        assert finalizer.ERR_INVALID_TRANSITION in str(ei.value)

    def test_close_with_wrong_recipient_still_rejected(self, core_module):
        core = core_module
        t = _assign(core)
        core.ack_task(task_id=t["task_id"], agent="openclaw")
        with pytest.raises(ValueError):
            core.complete_task(task_id=t["task_id"], agent="hermes", result="wrong caller")


class TestFailTaskIdempotent:
    def test_simple_fail(self, core_module):
        core = core_module
        t = _assign(core)
        core.ack_task(task_id=t["task_id"], agent="openclaw")
        failed = core.fail_task(task_id=t["task_id"], agent="openclaw", reason="blocked")
        assert failed["status"] == "fail"

    def test_same_outcome_reclose_is_idempotent(self, core_module):
        core = core_module
        t = _assign(core)
        core.ack_task(task_id=t["task_id"], agent="openclaw")
        core.fail_task(task_id=t["task_id"], agent="openclaw", reason="first fail")
        again = core.fail_task(task_id=t["task_id"], agent="openclaw", reason="second fail")
        assert again["status"] == "fail"
        assert again["result"] == "first fail"


class TestKeepAliveTask:
    def test_keep_alive_from_ack(self, core_module):
        core = core_module
        t = _assign(core)
        core.ack_task(task_id=t["task_id"], agent="openclaw")
        out = core.keep_alive_task(task_id=t["task_id"], agent="openclaw", note="still going")
        assert out["status"] == "keep-alive"
        assert out["deadline"] is not None

    def test_keep_alive_from_progress(self, core_module):
        core = core_module
        t = _assign(core)
        core.ack_task(task_id=t["task_id"], agent="openclaw")
        core.progress_task(task_id=t["task_id"], agent="openclaw", note="working")
        out = core.keep_alive_task(task_id=t["task_id"], agent="openclaw")
        assert out["status"] == "keep-alive"

    def test_keep_alive_then_done(self, core_module):
        core = core_module
        t = _assign(core)
        core.ack_task(task_id=t["task_id"], agent="openclaw")
        core.keep_alive_task(task_id=t["task_id"], agent="openclaw")
        done = core.complete_task(task_id=t["task_id"], agent="openclaw", result="finished")
        assert done["status"] == "done"

    def test_keep_alive_from_pending_rejected(self, core_module):
        core = core_module
        t = _assign(core)
        with pytest.raises(ValueError) as ei:
            core.keep_alive_task(task_id=t["task_id"], agent="openclaw")
        assert finalizer.ERR_INVALID_TRANSITION in str(ei.value)

    def test_keep_alive_from_terminal_rejected(self, core_module):
        core = core_module
        t = _assign(core)
        core.ack_task(task_id=t["task_id"], agent="openclaw")
        core.complete_task(task_id=t["task_id"], agent="openclaw", result="ok")
        with pytest.raises(ValueError) as ei:
            core.keep_alive_task(task_id=t["task_id"], agent="openclaw")
        assert finalizer.ERR_INVALID_TRANSITION in str(ei.value)


class TestAmendLearning:
    def test_amend_after_done(self, core_module):
        core = core_module
        t = _assign(core)
        core.ack_task(task_id=t["task_id"], agent="openclaw")
        core.complete_task(task_id=t["task_id"], agent="openclaw", result="ok")
        out = core.amend_learning(
            task_id=t["task_id"], agent="openclaw", learning="learned something"
        )
        assert out["learning_wiki_path"]

    def test_amend_before_terminal_rejected(self, core_module):
        core = core_module
        t = _assign(core)
        core.ack_task(task_id=t["task_id"], agent="openclaw")
        with pytest.raises(ValueError):
            core.amend_learning(
                task_id=t["task_id"], agent="openclaw", learning="too early"
            )

    def test_amend_by_wrong_agent_rejected(self, core_module):
        core = core_module
        t = _assign(core)
        core.ack_task(task_id=t["task_id"], agent="openclaw")
        core.complete_task(task_id=t["task_id"], agent="openclaw", result="ok")
        with pytest.raises(ValueError):
            core.amend_learning(
                task_id=t["task_id"], agent="hermes", learning="not my task"
            )


class TestAdvisoryMode:
    def test_advisory_mode_does_not_raise_on_pending_close(self, core_module, monkeypatch):
        monkeypatch.setenv("HERMES_FINALIZER_GATE", "advisory")
        core = core_module
        t = _assign(core)
        # pending → done should NOT raise in advisory mode
        out = core.complete_task(task_id=t["task_id"], agent="openclaw", result="advisory close")
        assert out["status"] == "done"

    def test_advisory_mode_still_allows_idempotent(self, core_module, monkeypatch):
        monkeypatch.setenv("HERMES_FINALIZER_GATE", "advisory")
        core = core_module
        t = _assign(core)
        core.ack_task(task_id=t["task_id"], agent="openclaw")
        core.complete_task(task_id=t["task_id"], agent="openclaw", result="first")
        # idempotent should still work
        again = core.complete_task(task_id=t["task_id"], agent="openclaw", result="second")
        assert again["status"] == "done"
        assert again["result"] == "first"

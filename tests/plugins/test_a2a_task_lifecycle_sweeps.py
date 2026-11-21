"""Orphan sweeps, the clarification barrier, and the park handoff on the A2A adapter.

These live in a topical sibling of ``test_a2a_plugin.py`` on purpose: that file is already past the
per-file line cap, and new behaviour goes into a new ``test_<stem>_<topic>.py`` rather than growing
the facade (root AGENTS.md).
"""
from __future__ import annotations

import time

from plugins.platforms.a2a import adapter as adapter_mod
from plugins.platforms.a2a import protocol
from tests.plugins.test_a2a_plugin import _make_live_adapter


class TestTaskLifecycleSweeps:
    def test_clarifications_survive_the_soft_orphan_sweep(self, monkeypatch):
        """A task awaiting clarification waits on the user, not on a worker.

        STATE_INPUT_REQUIRED is non-terminal by spec, so the 300s orphan grace would otherwise fail a
        question that was never answered and destroy the clarification mid-composition. Only the hard
        job budget may end it.
        """
        adapter, _base = _make_live_adapter(monkeypatch, reply_fn=lambda e: None)
        adapter.tasks.create("t-ask", "c1", "peer")
        adapter.tasks.set_state("t-ask", protocol.STATE_INPUT_REQUIRED)
        adapter.tasks.note("t-ask", "Which repository?")
        adapter.tasks._tasks["t-ask"]["created_at"] = time.time() - (adapter_mod._orphan_timeout() + 60)

        assert adapter.tasks.fail_orphans(adapter_mod._orphan_timeout()) == []
        assert adapter._fail_orphans_once() == []
        assert adapter.tasks.get("t-ask")["state"] == protocol.STATE_INPUT_REQUIRED

    def test_clarifications_are_still_bounded_by_the_job_budget(self, monkeypatch):
        """The clarification exemption is not an exemption from the hard bound.

        A question nobody answers must still end, or the store accumulates it forever. The job budget
        ends it through the force_terminal path — complete() refuses to re-settle a clarification
        otherwise, which is the whole point of the barrier.
        """
        adapter, _base = _make_live_adapter(monkeypatch, reply_fn=lambda e: None)
        adapter.tasks.create("t-ask-old", "c1", "peer")
        adapter.tasks.set_state("t-ask-old", protocol.STATE_INPUT_REQUIRED)
        adapter.tasks.note("t-ask-old", "Which repository?")
        adapter.tasks._tasks["t-ask-old"]["created_at"] = time.time() - (adapter_mod._job_timeout() + 60)

        assert adapter._fail_orphans_once() == ["t-ask-old"]
        stored = adapter.tasks.get("t-ask-old")
        assert stored["state"] == protocol.STATE_FAILED
        assert "A2A_JOB_TIMEOUT" in stored["reply"]

    def test_complete_refuses_to_overwrite_a_clarification_unless_forced(self):
        """A late worker answer must not overwrite a question still waiting on the user.

        Without the barrier, TaskStore.complete() re-settles INPUT_REQUIRED (it is non-terminal) and
        the clarification's question is replaced by an answer to a question nobody asked.
        """
        store = protocol.TaskStore()
        store.create("t-ask", "c1", "peer")
        store.set_state("t-ask", protocol.STATE_INPUT_REQUIRED)
        store.note("t-ask", "Which repository?")

        assert store.complete("t-ask", protocol.STATE_COMPLETED, "here is the review") is None
        assert store.get("t-ask")["state"] == protocol.STATE_INPUT_REQUIRED
        assert store.get("t-ask")["reply"] == "Which repository?"

        forced = store.complete("t-ask", protocol.STATE_FAILED, "[timed out]", force_terminal=True)
        assert forced is not None and forced["state"] == protocol.STATE_FAILED
        # A terminal record stays terminal — the barrier does not reopen it.
        assert store.complete("t-ask", protocol.STATE_COMPLETED, "too late", force_terminal=True) is None

    def test_park_handoff_failure_falls_back_to_ordinary_cleanup(self, monkeypatch):
        """A registration failure must not strand the task in _pending with nothing able to end it.

        The park IS the handoff: `_park_task` registers the late finalizer that ends the task once the
        agent answers. If that registration raises, the task has no watcher, so it takes the ordinary
        cleanup path — recorded terminal and popped — instead of being left active forever.
        """
        adapter, _base = _make_live_adapter(monkeypatch, reply_fn=lambda e: None)
        rec = adapter.tasks.create("t-handoff", "c1", "peer")
        pending = {"task_id": "t-handoff", "context_id": "c1", "peer": "peer",
                   "future": adapter._add_pending("t-handoff", "c1"), "created_iso": rec["created_iso"],
                   "started": time.time()}

        def _registration_fails(_pending, _note):
            raise RuntimeError("registration failed")

        def _client_gone():
            raise BrokenPipeError("client disconnected")

        monkeypatch.setattr(adapter, "_park_task", _registration_fails)
        state, _reply = adapter._finish(pending, keepalive=_client_gone)

        assert state == protocol.STATE_FAILED
        assert adapter.tasks.get("t-handoff")["state"] == protocol.STATE_FAILED
        assert "t-handoff" not in adapter._pending
        assert "t-handoff" not in adapter._parked
        assert "t-handoff" not in adapter._active_tasks

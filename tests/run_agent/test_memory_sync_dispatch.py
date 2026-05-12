"""Regression guard for #24453 — external memory sync at end-of-turn must
NOT block ``run_conversation`` from returning.

Before this fix, ``run_conversation`` called
``self._sync_external_memory_for_turn(...)`` inline.  With the Hindsight
external memory provider, that call does a daemon connect plus LLM-based
entity resolution, which routinely takes 30-50s of wall time per turn.
The SSE writer was therefore parked between ``response.output_text.delta``
(which streamed the assistant reply quickly) and ``response.completed``
(which couldn't fire until ``run_conversation`` returned).  The user saw
their reply text, then a 30+ second loading spinner.

The fix introduces ``_dispatch_memory_sync_for_turn``: a thin dispatcher
that spawns a daemon thread targeting the existing
``_sync_external_memory_for_turn`` worker.  The dispatcher returns
near-instantly, ``run_conversation`` continues, and the SSE writer can
emit ``response.completed`` immediately.  The worker is unchanged and
its existing behaviour (interrupt guard, no-manager no-op, exception
swallowing) is still exercised by ``test_memory_sync_interrupted.py``.

These tests cover the dispatcher contract directly.
"""
import threading
import time
from unittest.mock import MagicMock, patch


def _bare_agent():
    """Build a bare ``AIAgent`` with only the attributes the dispatcher
    and worker touch — same pattern as ``test_memory_sync_interrupted``.
    """
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    agent._memory_manager = MagicMock()
    agent.session_id = "test_session_dispatch"
    return agent


class TestDispatchMemorySyncForTurn:
    # --- The promptness guarantee (the #24453 fix) ----------------------

    def test_dispatcher_returns_immediately_when_worker_blocks(self):
        """Even if the underlying ``sync_all`` would take seconds, the
        dispatcher must hand control back to ``run_conversation`` in
        well under a single SSE poll interval (0.5s) — otherwise the
        client still sees the original delayed ``response.completed``.
        """
        agent = _bare_agent()
        sync_started = threading.Event()
        sync_finished = threading.Event()

        def _slow_sync(*args, **kwargs):
            sync_started.set()
            # Simulate the 30-50s Hindsight retain — short enough to keep
            # the test fast but long enough that an inline call would
            # blow the deadline by 10x.
            time.sleep(0.5)
            sync_finished.set()

        agent._memory_manager.sync_all.side_effect = _slow_sync

        start = time.monotonic()
        agent._dispatch_memory_sync_for_turn(
            original_user_message="hi",
            final_response="hey",
            interrupted=False,
        )
        elapsed = time.monotonic() - start

        # Headroom on the 0.5s SSE poll: 0.1s is generous and still
        # 5x faster than the synchronous baseline this PR replaces.
        assert elapsed < 0.1, (
            f"dispatcher blocked for {elapsed:.3f}s; must return "
            "near-instantly so the SSE writer can emit response.completed"
        )

        # The worker really did start on the thread (proof the dispatcher
        # didn't silently swallow the call entirely).
        assert sync_started.wait(timeout=2.0), "worker never ran"
        assert sync_finished.wait(timeout=2.0), "worker never finished"

    def test_dispatcher_uses_daemon_thread(self):
        """Memory sync is best-effort and we don't want a stuck Hindsight
        retain to keep the process alive past ``sys.exit`` / interpreter
        shutdown.  Daemon-flag the thread so the runtime can reap it.
        Matches ``_spawn_background_review`` (issue #15216 lineage).
        """
        agent = _bare_agent()
        captured = {}

        class _ImmediateThread:
            def __init__(self, *args, **kwargs):
                captured["kwargs"] = kwargs
                self._target = kwargs.get("target")
                self._target_kwargs = kwargs.get("kwargs") or {}

            def start(self):
                # Run synchronously so we can assert the target+kwargs
                # were wired up correctly without flaky timing.
                self._target(**self._target_kwargs)

        with patch("threading.Thread", _ImmediateThread):
            agent._dispatch_memory_sync_for_turn(
                original_user_message="hi",
                final_response="hey",
                interrupted=False,
            )

        assert captured["kwargs"].get("daemon") is True, (
            "dispatcher must use daemon=True so a stuck sync does not "
            "keep the interpreter alive past shutdown"
        )
        # Target is the existing worker — proves the dispatcher delegates
        # rather than re-implementing the sync logic and drifting away
        # from the interrupted-turn guard.
        assert captured["kwargs"].get("target") == agent._sync_external_memory_for_turn
        # Kwargs are forwarded unchanged so the worker sees the same
        # state run_conversation observed at end-of-turn.
        assert captured["kwargs"]["kwargs"] == {
            "original_user_message": "hi",
            "final_response": "hey",
            "interrupted": False,
        }

    # --- Cheap early-skips (don't spin up a no-op thread) ---------------

    def test_interrupted_turn_does_not_spawn_thread(self):
        """An interrupted turn would be skipped by the worker anyway; we
        also skip at the dispatcher so we don't pay thread-creation cost
        on every partial turn.  Matches the #15218 contract.
        """
        agent = _bare_agent()
        with patch("threading.Thread") as mock_thread:
            agent._dispatch_memory_sync_for_turn(
                original_user_message="hi",
                final_response="hey",
                interrupted=True,
            )
        mock_thread.assert_not_called()
        agent._memory_manager.sync_all.assert_not_called()

    def test_no_memory_manager_does_not_spawn_thread(self):
        """Sessions without a memory provider must not spawn idle
        threads on every turn."""
        from run_agent import AIAgent

        agent = AIAgent.__new__(AIAgent)
        agent._memory_manager = None
        agent.session_id = "test_session_no_mgr"

        with patch("threading.Thread") as mock_thread:
            agent._dispatch_memory_sync_for_turn(
                original_user_message="hi",
                final_response="hey",
                interrupted=False,
            )
        mock_thread.assert_not_called()

    def test_missing_final_response_does_not_spawn_thread(self):
        agent = _bare_agent()
        with patch("threading.Thread") as mock_thread:
            agent._dispatch_memory_sync_for_turn(
                original_user_message="hi",
                final_response=None,
                interrupted=False,
            )
        mock_thread.assert_not_called()

    def test_missing_original_user_message_does_not_spawn_thread(self):
        agent = _bare_agent()
        with patch("threading.Thread") as mock_thread:
            agent._dispatch_memory_sync_for_turn(
                original_user_message=None,
                final_response="hey",
                interrupted=False,
            )
        mock_thread.assert_not_called()

    # --- Old behaviour regression: dispatcher does sync end-to-end ------

    def test_normal_turn_invokes_sync_all_and_prefetch(self):
        """End-to-end smoke: dispatcher → daemon thread → worker calls
        both ``sync_all`` and ``queue_prefetch_all`` exactly once with
        the kwargs the caller provided.  Without joining the thread the
        assertions would race, so wait for it before checking.
        """
        agent = _bare_agent()

        agent._dispatch_memory_sync_for_turn(
            original_user_message="What's the weather in Paris?",
            final_response="Sunny and 22C.",
            interrupted=False,
        )

        # Find and join the worker thread so the assertions don't race.
        for t in threading.enumerate():
            if t.name.startswith("hermes-memory-sync-"):
                t.join(timeout=2.0)

        agent._memory_manager.sync_all.assert_called_once_with(
            "What's the weather in Paris?", "Sunny and 22C.",
            session_id="test_session_dispatch",
        )
        agent._memory_manager.queue_prefetch_all.assert_called_once_with(
            "What's the weather in Paris?",
            session_id="test_session_dispatch",
        )

    def test_worker_exception_does_not_crash_dispatcher(self):
        """A backend error inside ``sync_all`` must not propagate up out
        of the worker thread — same best-effort contract that
        ``_sync_external_memory_for_turn`` enforces inline.  We test by
        joining the thread; a bubbled-up exception would surface in the
        thread's logger but never reach the dispatcher caller.
        """
        agent = _bare_agent()
        agent._memory_manager.sync_all.side_effect = RuntimeError(
            "backend unreachable"
        )

        # Must not raise.
        agent._dispatch_memory_sync_for_turn(
            original_user_message="hi",
            final_response="hey",
            interrupted=False,
        )

        for t in threading.enumerate():
            if t.name.startswith("hermes-memory-sync-"):
                t.join(timeout=2.0)

        agent._memory_manager.sync_all.assert_called_once()

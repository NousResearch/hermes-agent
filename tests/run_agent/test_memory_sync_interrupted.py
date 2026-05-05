"""Regression guard for #15218 — external memory sync must skip interrupted turns.

Before this fix, ``run_conversation`` called
``memory_manager.sync_all(original_user_message, final_response)`` at the
end of every turn where both args were present.  That gate didn't check
the ``interrupted`` flag, so an external memory backend received partial
assistant output, aborted tool chains, or mid-stream resets as durable
conversational truth.  Downstream recall then treated that not-yet-real
state as if the user had seen it complete.

The fix is ``AIAgent._sync_external_memory_for_turn`` — a small helper
that replaces the inline block and returns early when ``interrupted``
is True (regardless of whether ``final_response`` and
``original_user_message`` happen to be populated).

These tests exercise the helper directly on a bare ``AIAgent`` built
via ``__new__`` so the full ``run_conversation`` machinery isn't needed
— the method is pure logic and three state arguments.
"""
from unittest.mock import MagicMock

import pytest


def _bare_agent():
    """Build an ``AIAgent`` with only the attributes
    ``_sync_external_memory_for_turn`` touches — matches the bare-agent
    pattern used across ``tests/run_agent/test_interrupt_propagation.py``.
    """
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    agent._memory_manager = MagicMock()
    # session_id is now propagated into sync_all / queue_prefetch_all so
    # providers that cache per-session state can update it mid-process
    # (see #6672).
    agent.session_id = "test_session_001"
    return agent


class TestSyncExternalMemoryForTurn:
    # --- Interrupt guard (the #15218 fix) -------------------------------

    def test_interrupted_turn_does_not_sync(self):
        """The whole point of #15218: even with a final_response and a
        user message, an interrupted turn must NOT reach the memory
        backend."""
        agent = _bare_agent()
        agent._sync_external_memory_for_turn(
            original_user_message="What time is it?",
            final_response="It is 3pm.",  # looks complete — but partial
            interrupted=True,
        )
        agent._memory_manager.sync_all.assert_not_called()
        agent._memory_manager.queue_prefetch_all.assert_not_called()

    def test_interrupted_turn_skips_even_when_response_is_full(self):
        """A long, seemingly-complete assistant response is still
        partial if ``interrupted`` is True — an interrupt may have
        landed between the streamed reply and the next tool call.  The
        memory backend has no way to distinguish on its own, so we must
        gate at the source."""
        agent = _bare_agent()
        agent._sync_external_memory_for_turn(
            original_user_message="Plan a trip to Lisbon",
            final_response="Here's a detailed 7-day itinerary: [...]",
            interrupted=True,
        )
        agent._memory_manager.sync_all.assert_not_called()

    # --- Normal completed turn still syncs ------------------------------

    def test_completed_turn_syncs_and_queues_prefetch(self):
        """Regression guard for the positive path: a normal completed
        turn must still trigger both ``sync_all`` AND
        ``queue_prefetch_all`` — otherwise the external memory backend
        never learns about anything and every user complains.
        """
        agent = _bare_agent()
        agent._sync_external_memory_for_turn(
            original_user_message="What's the weather in Paris?",
            final_response="It's sunny and 22°C.",
            interrupted=False,
        )
        agent._memory_manager.sync_all.assert_called_once_with(
            "What's the weather in Paris?", "It's sunny and 22°C.",
            session_id="test_session_001",
        )
        agent._memory_manager.queue_prefetch_all.assert_called_once_with(
            "What's the weather in Paris?",
            session_id="test_session_001",
        )

    # --- Edge cases (pre-existing behaviour preserved) ------------------

    def test_no_final_response_skips(self):
        """If the model produced no final_response (e.g. tool-only turn
        that never resolved), we must not fabricate an empty sync."""
        agent = _bare_agent()
        agent._sync_external_memory_for_turn(
            original_user_message="Hello",
            final_response=None,
            interrupted=False,
        )
        agent._memory_manager.sync_all.assert_not_called()

    def test_no_original_user_message_skips(self):
        """No user-origin message means this wasn't a user turn (e.g.
        a system-initiated refresh).  Don't sync an assistant-only
        exchange as if a user said something."""
        agent = _bare_agent()
        agent._sync_external_memory_for_turn(
            original_user_message=None,
            final_response="Proactive notification text",
            interrupted=False,
        )
        agent._memory_manager.sync_all.assert_not_called()

    def test_no_memory_manager_is_a_no_op(self):
        """Sessions without an external memory manager must not crash
        or try to call .sync_all on None."""
        from run_agent import AIAgent

        agent = AIAgent.__new__(AIAgent)
        agent._memory_manager = None

        # Must not raise.
        agent._sync_external_memory_for_turn(
            original_user_message="hi",
            final_response="hey",
            interrupted=False,
        )

    # --- Exception safety ----------------------------------------------

    def test_sync_exception_is_swallowed(self):
        """External memory providers are best-effort; a misconfigured
        or offline backend must not block the user from seeing their
        response by propagating the exception up."""
        agent = _bare_agent()
        agent._memory_manager.sync_all.side_effect = RuntimeError(
            "backend unreachable"
        )

        # Must not raise.
        agent._sync_external_memory_for_turn(
            original_user_message="hi",
            final_response="hey",
            interrupted=False,
        )
        # sync_all was attempted.
        agent._memory_manager.sync_all.assert_called_once()

    def test_prefetch_exception_is_swallowed(self):
        """Same best-effort contract applies to the prefetch step — a
        failure in queue_prefetch_all must not bubble out."""
        agent = _bare_agent()
        agent._memory_manager.queue_prefetch_all.side_effect = RuntimeError(
            "prefetch worker dead"
        )

        # Must not raise.
        agent._sync_external_memory_for_turn(
            original_user_message="hi",
            final_response="hey",
            interrupted=False,
        )
        # sync_all still happened before the prefetch blew up.
        agent._memory_manager.sync_all.assert_called_once()

    # --- The specific matrix the reporter asked about ------------------

    @pytest.mark.parametrize("interrupted,final,user,expect_sync", [
        (False, "resp", "user",  True),   # normal completed → sync
        (True,  "resp", "user",  False),  # interrupted → skip (the fix)
        (False, None,   "user",  False),  # no response → skip
        (False, "resp", None,    False),  # no user msg → skip
        (True,  None,   "user",  False),  # interrupted + no response → skip
        (True,  "resp", None,    False),  # interrupted + no user → skip
        (False, None,   None,    False),  # nothing → skip
        (True,  None,   None,    False),  # interrupted + nothing → skip
    ])
    def test_sync_matrix(self, interrupted, final, user, expect_sync):
        agent = _bare_agent()
        agent._sync_external_memory_for_turn(
            original_user_message=user,
            final_response=final,
            interrupted=interrupted,
        )
        if expect_sync:
            agent._memory_manager.sync_all.assert_called_once()
            agent._memory_manager.queue_prefetch_all.assert_called_once()
        else:
            agent._memory_manager.sync_all.assert_not_called()
            agent._memory_manager.queue_prefetch_all.assert_not_called()

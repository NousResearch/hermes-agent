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


    # --- Normal completed turn still syncs ------------------------------




    # --- Edge cases (pre-existing behaviour preserved) ------------------




    # --- Exception safety ----------------------------------------------



    # --- Multimodal content flattening ----------------------------------




    # --- The specific matrix the reporter asked about ------------------


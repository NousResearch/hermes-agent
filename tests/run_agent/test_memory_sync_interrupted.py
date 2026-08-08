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

These tests exercise the helper directly on a bare ``AIAgent`` built via
``__new__`` so interruption safety, provider-only transcript continuity,
authoritative-history replacement, and session isolation are independently
observable.
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

    def test_historyless_completed_turns_accumulate_for_provider_reconciliation(self):
        agent = _bare_agent()
        first = [
            {"role": "user", "content": "first user"},
            {"role": "assistant", "content": "first answer"},
        ]
        second = [
            {"role": "user", "content": "second user"},
            {"role": "assistant", "content": "second answer"},
        ]

        agent._sync_external_memory_for_turn(
            original_user_message="first user",
            final_response="first answer",
            interrupted=False,
            messages=first,
        )
        agent._sync_external_memory_for_turn(
            original_user_message="second user",
            final_response="second answer",
            interrupted=False,
            messages=second,
        )

        delivered = agent._memory_manager.sync_all.call_args.kwargs["messages"]
        assert [message["content"] for message in delivered] == [
            "first user",
            "first answer",
            "second user",
            "second answer",
        ]

    def test_accumulated_history_replaces_the_provider_snapshot(self):
        agent = _bare_agent()
        agent._sync_external_memory_for_turn(
            original_user_message="old user",
            final_response="old answer",
            interrupted=False,
            messages=[
                {"role": "user", "content": "old user"},
                {"role": "assistant", "content": "old answer"},
            ],
        )
        authoritative = [
            {"role": "user", "content": "compressed summary"},
            {"role": "assistant", "content": "new answer"},
        ]

        agent._sync_external_memory_for_turn(
            original_user_message="compressed summary",
            final_response="new answer",
            interrupted=False,
            messages=authoritative,
            messages_are_authoritative=True,
        )

        delivered = agent._memory_manager.sync_all.call_args.kwargs["messages"]
        assert delivered == authoritative

    def test_provider_snapshot_resets_when_the_session_changes(self):
        agent = _bare_agent()
        agent._sync_external_memory_for_turn(
            original_user_message="old user",
            final_response="old answer",
            interrupted=False,
            messages=[
                {"role": "user", "content": "old user"},
                {"role": "assistant", "content": "old answer"},
            ],
        )
        agent.session_id = "test_session_002"

        agent._sync_external_memory_for_turn(
            original_user_message="new user",
            final_response="new answer",
            interrupted=False,
            messages=[
                {"role": "user", "content": "new user"},
                {"role": "assistant", "content": "new answer"},
            ],
        )

        delivered = agent._memory_manager.sync_all.call_args.kwargs["messages"]
        assert all(message["content"] != "old user" for message in delivered)




    # --- Edge cases (pre-existing behaviour preserved) ------------------




    # --- Exception safety ----------------------------------------------



    # --- Multimodal content flattening ----------------------------------




    # --- The specific matrix the reporter asked about ------------------

"""System-injected notification turns must never reach external memory providers."""
from unittest.mock import MagicMock

import pytest

from agent.memory_provider import is_synthetic_notification_turn
from tools.process_registry_notifications import TimelineNotification


@pytest.mark.parametrize("text", [
    "[IMPORTANT: Background process proc_45108bc56f5b exited (exit code -15, SIGTERM).\nCommand: bash x.sh",
    "[IMPORTANT: Background process proc_eed9f05bf8d6 completed normally (exit code 0).",
    '[IMPORTANT: Background process proc_1 matched watch pattern "ready".',
    "[Background process proc_1 heartbeat #3 — still running after 5m]",
    "[ASYNC DELEGATION COMPLETE] child finished",
    "[ASYNC BATCH COMPLETE] 3 children",
    "A background subagent you dispatched earlier has finished.",
    "A background fan-out of 4 subagent(s) you dispatched earlier has finished.",
])
def test_notification_texts_are_synthetic(text):
    assert is_synthetic_notification_turn(text)


@pytest.mark.parametrize("text", [
    "proceed",
    "fix all the outstanding issues mentioned in Loose ends",
    "Why did the background process fail?",
    "[OUT-OF-BAND USER MESSAGE — a direct message from the user]\nstop",
    "",
])
def test_user_texts_are_not_synthetic(text):
    assert not is_synthetic_notification_turn(text)


def test_timeline_notification_object_is_synthetic_even_without_marker_text():
    note = TimelineNotification("anything at all", display_text="x", display_kind="process_notification")
    assert is_synthetic_notification_turn(note)


def test_multimodal_first_text_part_is_judged():
    assert is_synthetic_notification_turn([{"type": "text", "text": "[IMPORTANT: Background process proc_1 completed"}])
    assert not is_synthetic_notification_turn([{"type": "text", "text": "hello"}, {"type": "image_url", "image_url": {}}])


def _bare_agent():
    """Bare AIAgent with only what _sync_external_memory_for_turn touches (as in test_memory_sync_interrupted)."""
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    agent._memory_manager = MagicMock()
    agent.session_id = "test_session_001"
    return agent


def test_sync_skips_notification_turn():
    agent = _bare_agent()
    agent._sync_external_memory_for_turn(
        original_user_message="[IMPORTANT: Background process proc_1 completed normally (exit code 0).",
        final_response="Nothing to act on.", interrupted=False)
    agent._memory_manager.sync_all.assert_not_called()
    agent._memory_manager.queue_prefetch_all.assert_not_called()


def test_sync_keeps_human_turn():
    agent = _bare_agent()
    agent._sync_external_memory_for_turn(
        original_user_message="I prefer Open Brain for infrastructure knowledge",
        final_response="Noted.", interrupted=False)
    agent._memory_manager.sync_all.assert_called_once()

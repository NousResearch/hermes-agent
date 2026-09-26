"""Runtime anti-loop guard (paraphrase loop & cross-turn escalation).

Tests for:
1. `paraphrase_loop_detected` detector in `agent/agent_runtime_helpers.py`
2. `_segment_continuation_intent_segments` segment counter
3. `finish_text_response` paraphrase loop handling and cross-turn escalation logic
4. `run_tool_round` and genuine turn-end counter resets
"""

import os
from pathlib import Path
from unittest.mock import MagicMock
import pytest

# Prevent hermes_bootstrap dependency activation from reading real home during test imports
import pm.environments
pm.environments.payload_venv = MagicMock()

from agent.agent_runtime_helpers import (
    _segment_continuation_intent_segments,
    paraphrase_loop_detected,
    trailing_continue_intent,
)


@pytest.fixture(autouse=True)
def isolate_hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def test_paraphrase_loop_detected_positive():
    sample_loop = (
        "Let me read the JSON file to get the full extracted data, then write a clean discovery report.\n"
        "Now I have the data. Let me write a clean discovery report.\n"
        "Let me write the discovery report now.\n"
        "I have enough from the raw output. Let me write the report directly and stop the session."
    )
    assert paraphrase_loop_detected(sample_loop) is True
    assert _segment_continuation_intent_segments(sample_loop) >= 2


def test_paraphrase_loop_detected_negative_normal_reply():
    normal_text = (
        "I have analyzed the codebase. The main entry point is `cli.py` which initializes `HermesCLI`.\n"
        "The configuration files are loaded from `~/.hermes/config.yaml`.\n"
        "All features are tested and working."
    )
    assert paraphrase_loop_detected(normal_text) is False
    assert _segment_continuation_intent_segments(normal_text) == 0


def test_paraphrase_loop_detected_single_continuation_intent():
    single_intent = (
        "I have scanned the directory and found 3 test files.\n"
        "Let me check the logs."
    )
    assert paraphrase_loop_detected(single_intent) is False


def test_paraphrase_loop_detected_empty_and_none():
    assert paraphrase_loop_detected("") is False
    assert paraphrase_loop_detected(None) is False
    assert _segment_continuation_intent_segments("") == 0
    assert _segment_continuation_intent_segments(None) == 0


def test_finish_text_response_paraphrase_loop_reprompts_first_time():
    from agent.turn_final_response import finish_text_response

    agent = MagicMock()
    agent._strip_think_blocks = lambda x: x or ""
    agent.valid_tool_names = {"read_file", "write_file"}
    agent._stall_guards = True
    agent.api_mode = "chat_completions"
    agent._consecutive_stall_turns = 0
    agent._extract_reasoning = lambda msg: None
    agent._has_content_after_think_block = lambda x: True
    agent._emit_pending_fallback_notice = lambda: None
    agent._clear_status_buffer = lambda: None
    agent._build_assistant_message = lambda msg, reason: {"role": "assistant", "content": msg.content}
    agent._emit_interim_assistant_message = lambda msg: None

    assistant_msg = MagicMock()
    assistant_msg.content = (
        "Let me read the JSON file now.\n"
        "Let me write the report now."
    )
    assistant_msg.tool_calls = None

    messages = [{"role": "user", "content": "Generate report"}]

    verdict = finish_text_response(
        agent,
        assistant_message=assistant_msg,
        response=MagicMock(),
        finish_reason="stop",
        messages=messages,
        api_messages=[],
        conversation_history=[],
        api_call_count=1,
        effective_task_id="task1",
        user_message="Generate report",
        active_system_prompt="system",
        final_response=assistant_msg.content,
        _turn_exit_reason="unknown",
        _preflight_compression_blocked=False,
        codex_ack_continuations=0,
        truncated_response_parts=[],
        length_continue_retries=0,
        _pending_verification_response=None,
        _pending_verification_response_previewed=False,
    )

    assert verdict.action == "continue"
    assert verdict.final_response is None
    assert messages[-1]["role"] == "user"
    assert "re-announced the same next step" in messages[-1]["content"]


def test_finish_text_response_paraphrase_loop_hard_stops_second_time():
    from agent.turn_final_response import finish_text_response

    agent = MagicMock()
    agent._strip_think_blocks = lambda x: x or ""
    agent.valid_tool_names = {"read_file", "write_file"}
    agent._stall_guards = True
    agent.api_mode = "chat_completions"
    agent._consecutive_stall_turns = 1  # Already stalled once!
    agent._extract_reasoning = lambda msg: None
    agent._has_content_after_think_block = lambda x: True
    agent._emit_pending_fallback_notice = lambda: None
    agent._clear_status_buffer = lambda: None
    agent._build_assistant_message = lambda msg, reason: {"role": "assistant", "content": msg.content}
    agent._emit_interim_assistant_message = lambda msg: None
    agent._flush_messages_to_session_db = lambda msgs, history: None

    assistant_msg = MagicMock()
    assistant_msg.content = (
        "Let me read the JSON file now.\n"
        "Let me write the report now."
    )
    assistant_msg.tool_calls = None

    messages = [{"role": "user", "content": "Generate report"}]

    verdict = finish_text_response(
        agent,
        assistant_message=assistant_msg,
        response=MagicMock(),
        finish_reason="stop",
        messages=messages,
        api_messages=[],
        conversation_history=[],
        api_call_count=2,
        effective_task_id="task1",
        user_message="Generate report",
        active_system_prompt="system",
        final_response=assistant_msg.content,
        _turn_exit_reason="unknown",
        _preflight_compression_blocked=False,
        codex_ack_continuations=0,
        truncated_response_parts=[],
        length_continue_retries=0,
        _pending_verification_response=None,
        _pending_verification_response_previewed=False,
    )

    assert verdict.action == "break"
    assert "re-announcing the same plan" in verdict.final_response
    assert agent._consecutive_stall_turns >= 2


def test_run_tool_round_resets_consecutive_stall_turns():
    from agent.turn_tool_round import run_tool_round

    class FakeAgent:
        def __init__(self):
            self._consecutive_stall_turns = 2
            self.quiet_mode = True
            self.verbose_logging = False
            self.log_prefix = ""

        def _vprint(self, *args, **kwargs):
            pass

    agent = FakeAgent()

    assistant_msg = MagicMock()
    tc = MagicMock()
    tc.function.name = "read_file"
    tc.function.arguments = "{}"
    tc.id = "tc1"
    assistant_msg.tool_calls = [tc]

    try:
        run_tool_round(
            agent,
            assistant_message=assistant_msg,
            finish_reason="tool_calls",
            messages=[],
            conversation_history=[],
            api_call_count=1,
            effective_task_id="t1",
            user_message="u1",
            system_message="s1",
            active_system_prompt="asp1",
            compression_attempts=0,
            max_compression_attempts=3,
            final_response=None,
            failed=False,
            _turn_exit_reason="unknown",
            truncated_tool_call_retries=0,
            current_turn_user_idx=0,
        )
    except Exception:
        pass

    assert agent._consecutive_stall_turns == 0

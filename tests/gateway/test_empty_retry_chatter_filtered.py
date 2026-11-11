"""Empty-response retries reach a chat as ONE reply, the plain explanation (#101138).

The retry counter (``retrying (1/3) in 8s``) and the terminal "no content after all retries"
status used to slip past the chat noise filter and arrive beside the turn's own explanation.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent import turn_empty_response as ter
from gateway.run import _prepare_gateway_status_message


@pytest.mark.parametrize("message", [
    "⚠️ Empty response from model — retrying (1/3) in 8s",
    "⚠️ Empty response from model — retrying (1/1) in 6s — high-cost request, reduced retry budget",
    "⏳ Retrying in 4.2s (attempt 1/3)...",
])
def test_counted_retry_chatter_stays_out_of_chats_but_not_the_cli(message):
    assert _prepare_gateway_status_message("telegram", "status", message) is None
    assert _prepare_gateway_status_message("local", "status", message) == message


@pytest.mark.parametrize("reasoning", [None, "thought about it"])
def test_terminal_empty_status_is_left_to_the_final_reply(reasoning):
    agent = MagicMock()
    agent._empty_streak_cost_usd = None  # no cost estimate: the streak-cost line is skipped
    agent._extract_reasoning.return_value = reasoning
    agent._build_assistant_message.return_value = {"role": "assistant"}
    agent._fallback_chain = []
    emitted = []
    agent._emit_diagnostic_status.side_effect = emitted.append
    assert ter._terminal_empty(agent, SimpleNamespace(), "stop", []) == "(empty)"
    (status,) = emitted
    # The chat's copy is dropped (the final reply explains the empty turn); the CLI keeps it.
    assert _prepare_gateway_status_message("telegram", "status", status) is None
    assert _prepare_gateway_status_message("local", "status", status) == status


def test_other_warnings_still_reach_the_chat():
    text = "⚠️ Tool output was truncated to 50,000 characters."
    assert _prepare_gateway_status_message("telegram", "status", text) == text


@pytest.mark.parametrize("deterministic", [False, True])
def test_failed_fallback_ends_on_the_plain_explanation(deterministic):
    """Fallback configured but it cannot activate: the buffered "switching to fallback..." (and the
    deterministic-empty "skipping further retries") flush at the end and must not be the chat's last
    status, since nothing after them says the switch failed. The CLI keeps the whole trace."""
    from unittest.mock import patch

    from agent.status_output import StatusOutputMixin

    agent = MagicMock()
    for name in ("_buffer_retry_message", "_buffer_status", "_buffer_diagnostic_status",
                 "_flush_status_buffer", "_emit_diagnostic_status"):
        setattr(agent, name, getattr(StatusOutputMixin, name).__get__(agent))
    agent._retry_status_buffer = []
    agent._empty_streak_cost_usd = None
    agent._current_streamed_assistant_text = ""
    agent._has_content_after_think_block.return_value = False
    agent._strip_think_blocks.side_effect = lambda t: t or ""
    agent._last_content_with_tools = None
    agent._thinking_prefill_retries = 2
    agent._extract_reasoning.return_value = None
    agent._build_assistant_message.return_value = {"role": "assistant"}
    agent._fallback_chain = ["openrouter/nope"]
    agent._try_activate_fallback.return_value = False
    emitted = []
    agent._emit_status.side_effect = emitted.append
    msg = SimpleNamespace(reasoning=None, reasoning_content=None, reasoning_details=None)
    with patch.object(ter, "_retry_empty", return_value=(None, None, deterministic)):
        verdict = ter.recover_empty_response(
            agent, msg, None, "stop", final_response="", messages=[{"role": "user", "content": "hi"}],
            api_messages=[], conversation_history=[], active_system_prompt="sys", api_call_count=4,
            turn_exit_reason=None, preflight_compression_blocked=False,
        )
    assert (verdict.action, verdict.final_response) == ("break", "(empty)")
    assert any("switching to fallback" in m for m in emitted)
    assert any("skipping further retries" in m for m in emitted) is deterministic
    assert [m for m in emitted if _prepare_gateway_status_message("telegram", "status", m)] == []
    assert [_prepare_gateway_status_message("local", "status", m) for m in emitted] == emitted

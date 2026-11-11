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

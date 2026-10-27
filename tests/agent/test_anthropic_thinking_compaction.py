"""Preserved thinking across compaction: a signed block is bound to the prefix it was produced
under (system prompt, tools, every earlier message). Compaction replaces that prefix, so every
block the verbatim tail still carries would 400 on accounts under the history-editing check.
"""

import copy
from unittest.mock import patch

import pytest

from agent.anthropic_thinking_replay import apply_rejected_thinking_suppression
from agent.context_compressor import ContextCompressor
from agent.conversation_compression import compress_context

ANTHROPIC = "https://api.anthropic.com"
OPENROUTER = "https://openrouter.ai/api/v1"
MODEL = "claude-sonnet-5-5"


class _TodoStore:
    def format_for_injection(self):
        return ""


class _Agent:
    def __init__(self, base_url):
        with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
            self.context_compressor = ContextCompressor(
                model=MODEL, provider="anthropic", base_url=base_url, api_mode="anthropic_messages",
                threshold_percent=0.50, quiet_mode=True, config_context_length=100_000,
            )
        self.context_compressor.tail_token_budget = 2_000
        self.api_mode, self.base_url, self.model, self.provider = "anthropic_messages", base_url, MODEL, "anthropic"
        self.session_id, self.platform, self.tools = "s-compaction", "cli", []
        self._compression_feasibility_checked = True
        self.compression_in_place = False
        self._memory_manager = None
        self._session_db = None
        self._persist_disabled = False
        self._todo_store = _TodoStore()
        self._cached_system_prompt = None

    def _emit_status(self, _message):
        pass

    def _emit_warning(self, _message):
        pass

    def _invalidate_system_prompt(self):
        self._cached_system_prompt = None

    def _build_system_prompt(self, system_message):
        return system_message

    def commit_memory_session(self, _messages):
        pass


def _signed_turn(idx, sig):
    filler = " ".join(["context"] * 200)
    return [
        {"role": "user", "content": f"user message {idx} {filler}"},
        {
            "role": "assistant",
            "content": f"assistant reply {idx}",
            "reasoning_details": [{"type": "thinking", "thinking": f"thought {idx}", "signature": sig}],
        },
    ]


def _history():
    messages = [{"role": "system", "content": "system prompt"}]
    for idx in range(10):
        messages += _signed_turn(idx, f"sig-before-{idx}")
    return messages


def _replayed_signatures(agent, messages):
    request = copy.deepcopy(messages)
    apply_rejected_thinking_suppression(agent, request)
    return {
        block["signature"]
        for message in request
        for block in message.get("reasoning_details") or ()
        if block.get("type") == "thinking"
    }


def _compact(agent):
    with patch.object(agent.context_compressor, "_generate_summary", return_value="SUMMARY"):
        compressed, _ = compress_context(agent, _history(), "system prompt", approx_tokens=80_000)
    kept = {
        block["signature"] for message in compressed for block in message.get("reasoning_details") or ()
    }
    assert kept, "the verbatim tail must still carry pre-compaction thinking for this test to mean anything"
    return compressed


def test_no_pre_compaction_block_reaches_the_next_request_but_new_blocks_still_replay():
    agent = _Agent(ANTHROPIC)
    compressed = _compact(agent)

    after = compressed + _signed_turn(99, "sig-after-compaction")

    assert _replayed_signatures(agent, after) == {"sig-after-compaction"}


@pytest.mark.parametrize("base_url", [OPENROUTER])
def test_routes_without_signed_replay_keep_their_tail_thinking(base_url):
    agent = _Agent(base_url)
    compressed = _compact(agent)

    assert _replayed_signatures(agent, compressed) == {
        block["signature"] for message in compressed for block in message.get("reasoning_details") or ()
    }

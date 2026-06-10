"""Regression tests for tool_call envelope accounting in the compression
tail-protection budget walks (issue #28053).

The budget walks used to estimate an assistant message's tokens from
content + ``function.arguments`` only, dropping each ``tool_call``'s ``id``,
``type`` and ``function.name`` (plus JSON structure). For assistant turns
that fan out into parallel tool calls this undercounted by 2-15x, so the
protected tail overshot ``tail_token_budget`` and compression became
ineffective. The fix routes all three walks through
``_estimate_msg_budget_tokens``, which counts the full envelope.
"""

import pytest
from unittest.mock import patch

from agent.context_compressor import (
    ContextCompressor,
    _CHARS_PER_TOKEN,
    _estimate_msg_budget_tokens,
)


def _assistant_with_tool_calls(n_calls: int, *, args: str = '{"path":"a"}') -> dict:
    """An assistant turn fanning into ``n_calls`` parallel tool calls with
    realistic id/name overhead but a small arguments string."""
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": f"call_{i:02d}_{'a' * 24}",  # ~32 chars, UUID-ish id
                "type": "function",
                "function": {"name": "read_file", "arguments": args},
            }
            for i in range(n_calls)
        ],
    }


def _args_only_estimate(msg: dict) -> int:
    """Reproduce the OLD (buggy) arguments-only walk for comparison."""
    content = msg.get("content") or ""
    tokens = len(content) // _CHARS_PER_TOKEN + 10
    for tc in msg.get("tool_calls") or []:
        if isinstance(tc, dict):
            tokens += len(tc.get("function", {}).get("arguments", "")) // _CHARS_PER_TOKEN
    return tokens


class TestToolCallEnvelopeEstimate:
    def test_envelope_counted_not_just_arguments(self):
        msg = _assistant_with_tool_calls(4)
        new = _estimate_msg_budget_tokens(msg)
        old = _args_only_estimate(msg)
        # id/type/name + JSON structure dwarf the tiny arguments string.
        assert new > old * 3, (new, old)
        # The estimate covers the full serialized tool_call envelope.
        envelope = sum(len(str(tc)) for tc in msg["tool_calls"]) // _CHARS_PER_TOKEN
        assert new >= envelope

    def test_scales_with_number_of_parallel_calls(self):
        one = _estimate_msg_budget_tokens(_assistant_with_tool_calls(1))
        five = _estimate_msg_budget_tokens(_assistant_with_tool_calls(5))
        assert five > one * 3

    def test_no_tool_calls_matches_content_estimate(self):
        msg = {"role": "user", "content": "x" * 400}
        # Plain message: content//4 + 10 overhead, behavior unchanged.
        assert _estimate_msg_budget_tokens(msg) == 400 // _CHARS_PER_TOKEN + 10

    def test_non_dict_tool_calls_do_not_crash(self):
        msg = {"role": "assistant", "content": "hi", "tool_calls": ["weird", None]}
        # Non-dict entries are ignored (as before) without raising.
        assert _estimate_msg_budget_tokens(msg) == len("hi") // _CHARS_PER_TOKEN + 10


@pytest.fixture()
def compressor():
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        return ContextCompressor(
            model="test/model",
            threshold_percent=0.85,
            protect_first_n=2,
            protect_last_n=2,
            quiet_mode=True,
        )


class TestTailCutAccountsForToolCalls:
    def test_tail_cut_stops_on_tool_call_heavy_tail(self, compressor):
        # 20 assistant turns, each fanning into 5 short-arg tool calls.
        heavy = [_assistant_with_tool_calls(5) for _ in range(20)]
        messages = [{"role": "user", "content": "start"}] + heavy

        per_msg = _estimate_msg_budget_tokens(messages[-1])
        assert per_msg > 30  # sanity: a heavy turn is non-trivial once the envelope counts

        # Budget sized so ~6 heavy turns fit under the 1.5x soft ceiling.
        token_budget = int(per_msg * 6 / 1.5)
        cut = compressor._find_tail_cut_by_tokens(messages, head_end=1, token_budget=token_budget)
        protected = len(messages) - cut

        # With the envelope counted, the walk stops well short of protecting all
        # 20 turns. The old arguments-only estimate (~25 tokens/turn) never
        # reaches the ceiling and would protect the entire transcript.
        assert protected < len(heavy)
        assert 3 <= protected <= 12

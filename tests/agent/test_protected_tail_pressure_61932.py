"""Algorithmic reproduction and regression for issue #61932.

After several in-place compactions a tool-heavy session can be short enough
that nearly every remaining message sits inside the protected recent tail,
yet those messages are huge completed ``read_file`` / tool outputs.  The
middle compress window is then empty or tiny, preflight makes no material
token progress, and the turn dies with::

    Context length exceeded (174,833 tokens). Cannot compress further.

This is the core compressor contract — not Desktop/Windows-specific.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agent.context_compressor import (
    ContextCompressor,
    _MAX_TAIL_MESSAGE_FLOOR,
    _PRESSURE_KEEP_RECENT_MESSAGES,
)
from agent.model_metadata import estimate_messages_tokens_rough
from agent.turn_context import _compression_made_progress


def _unique_tool_pair(i: int, chars: int) -> list[dict]:
    """Assistant tool_call + unique tool result (no dedupe shortcut)."""
    body = f"FILE_{i}_START\n" + (f"line {i} unique payload " * (chars // 22))[:chars]
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": f'{{"path":"f{i}.py"}}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": f"call_{i}",
            "content": body,
        },
    ]


def _already_compacted_session(
    *,
    n_pairs: int,
    tool_chars: int,
    user_chars: int,
) -> list[dict]:
    """Shape after multiple in-place compactions: head + handoff + heavy tail."""
    msgs: list[dict] = [
        {"role": "system", "content": "You are Hermes."},
        {"role": "user", "content": "Investigate thoroughly"},
        {"role": "assistant", "content": "OK"},
        {
            "role": "user",
            "content": (
                "[CONTEXT COMPACTION — REFERENCE ONLY]\n"
                + ("Prior findings. " * 200)
            ),
        },
        {"role": "assistant", "content": "Continuing from compacted context."},
    ]
    for i in range(n_pairs):
        msgs.extend(_unique_tool_pair(i, tool_chars))
    msgs.append(
        {
            "role": "user",
            "content": "Full structured report:\n" + ("U" * user_chars),
        }
    )
    return msgs


@pytest.fixture()
def compressor_128k():
    with patch(
        "agent.context_compressor.get_model_context_length",
        return_value=128_000,
    ):
        c = ContextCompressor(
            model="openai-codex/gpt-test",
            threshold_percent=0.50,
            summary_target_ratio=0.20,
            protect_first_n=3,
            protect_last_n=20,
            quiet_mode=True,
            config_context_length=128_000,
        )
    c._generate_summary = lambda *a, **k: "compact summary of earlier investigation"
    return c


class TestProtectedTailPressure61932:
    def test_pressure_constants_aligned_with_tail_floor(self):
        assert _MAX_TAIL_MESSAGE_FLOOR == 8
        assert _PRESSURE_KEEP_RECENT_MESSAGES == 3
        assert _PRESSURE_KEEP_RECENT_MESSAGES <= _MAX_TAIL_MESSAGE_FLOOR

    def test_prune_demotes_protected_tail_when_tool_bodies_dominate(
        self, compressor_128k
    ):
        """Protected-region tool bodies that blow the soft budget must demote.

        With protect_last_n=20 and only ~14 messages, the old floor treated
        every remaining tool result as sacred and prune was a pure no-op.
        """
        c = compressor_128k
        msgs = _already_compacted_session(
            n_pairs=4, tool_chars=200_000, user_chars=80_000
        )
        before = estimate_messages_tokens_rough(msgs)
        assert before > c.context_length

        pruned, n = c._prune_old_tool_results(
            msgs,
            protect_tail_count=c.protect_last_n,
            protect_tail_tokens=c.tail_token_budget,
        )
        after = estimate_messages_tokens_rough(pruned)

        assert n >= 1, "pressure demotion should touch at least one tool body"
        assert after < before * 0.5, (
            f"expected large reclaim from protected-tail demotion: "
            f"{before:,} → {after:,}"
        )
        # Active user ask must remain verbatim.
        assert pruned[-1]["role"] == "user"
        assert pruned[-1]["content"].startswith("Full structured report:")
        assert "U" * 100 in pruned[-1]["content"]

    def test_compress_escapes_cannot_compress_further_dead_end(
        self, compressor_128k
    ):
        """Full compress path must materially reduce an over-context tail.

        Reproduces the #61932 failure class: multipass compression previously
        dropped a couple of message rows while leaving ~170k tokens intact,
        then reported no further progress.
        """
        c = compressor_128k
        msgs = _already_compacted_session(
            n_pairs=4, tool_chars=200_000, user_chars=80_000
        )
        rough0 = estimate_messages_tokens_rough(msgs)
        assert rough0 > c.context_length

        cur = msgs
        tok = rough0
        last_progress = False
        for _pass in range(3):
            o_len, o_tok = len(cur), tok
            out = c.compress(list(cur), current_tokens=tok)
            n_tok = estimate_messages_tokens_rough(out)
            last_progress = _compression_made_progress(
                o_len, len(out), o_tok, n_tok
            )
            cur, tok = out, n_tok
            if n_tok < c.threshold_tokens and n_tok < c.context_length:
                break

        assert tok < c.context_length, (
            f"still over context after compression: {tok:,} >= {c.context_length:,}"
        )
        assert tok < rough0 * 0.5, (
            f"compression did not reclaim enough headroom: {rough0:,} → {tok:,}"
        )
        # Either we recovered under threshold, or the last pass still made
        # progress (never a pure no-op dead-end above the window).
        assert tok < c.threshold_tokens or last_progress

    def test_light_tail_still_keeps_recent_tool_bodies(self, compressor_128k):
        """Pressure demotion must not fire on a normal-sized protected tail."""
        c = compressor_128k
        msgs = _already_compacted_session(
            n_pairs=2, tool_chars=800, user_chars=200
        )
        before_tools = [
            m["content"]
            for m in msgs
            if m.get("role") == "tool" and isinstance(m.get("content"), str)
        ]
        pruned, n = c._prune_old_tool_results(
            msgs,
            protect_tail_count=c.protect_last_n,
            protect_tail_tokens=c.tail_token_budget,
        )
        after_tools = [
            m["content"]
            for m in pruned
            if m.get("role") == "tool" and isinstance(m.get("content"), str)
        ]
        assert after_tools, "expected tool messages to remain"
        # Light unique bodies fit inside the soft budget — none should demote.
        assert n == 0
        assert after_tools == before_tools

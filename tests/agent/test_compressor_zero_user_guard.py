"""Regression coverage for #58753 — compression could drop the only
user-role message, leaving a transcript with ZERO user turns.

The compressor already pins the handoff summary to ``role="user"`` when
the only protected head message is the system prompt (#52160). But that
guard keys off ``last_head_role == "system"``, which is only true when
the system prompt actually sits inside ``messages`` — the gateway
``/compress`` path. The main auto-compression path passes the transcript
WITHOUT the system prompt (it is prepended at request-build time, see
``conversation_loop`` — ``api_messages = [{"role": "system", ...}] +
api_messages``). There ``last_head_role`` defaults to ``"user"`` and the
summary is emitted as ``role="assistant"``.

On a session whose only genuine user turn falls into the compressed
middle — the canonical shape being a ``hermes kanban`` worker seeded with
a single short ``"work kanban task <id>"`` prompt followed by nothing but
assistant/tool turns — the compressed output then contains no user-role
message at all. OpenAI-compatible backends (vLLM/Qwen) reject such a
request with a non-retryable ``400 No user query found in messages``,
crashing the worker with no possible recovery (every resume replays the
same poisoned history).

The fix generalises the #52160 guard: when NO user-role message survives
in the protected head or preserved tail, the summary MUST carry
``role="user"``.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.fixture()
def compressor():
    from agent.context_compressor import ContextCompressor

    with patch(
        "agent.context_compressor.get_model_context_length",
        return_value=100_000,
    ):
        c = ContextCompressor(
            model="test/model",
            threshold_percent=0.50,
            protect_first_n=3,
            protect_last_n=20,
            quiet_mode=True,
        )
        c.tail_token_budget = 40
        return c


def _tool_turns(start: int, n: int) -> list[dict]:
    out: list[dict] = []
    for i in range(start, start + n):
        out.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"c{i}",
                        "function": {"name": "read_task", "arguments": "{}"},
                    }
                ],
            }
        )
        out.append({"role": "tool", "content": "x" * 300, "tool_call_id": f"c{i}"})
    return out


def _role_hist(messages: list[dict]) -> dict[str, int]:
    hist: dict[str, int] = {}
    for m in messages:
        hist[m.get("role")] = hist.get(m.get("role"), 0) + 1
    return hist


class TestCompressAlwaysKeepsAUserTurn:
    def test_kanban_worker_recompaction_keeps_user_turn(self, compressor):
        """The exact #58753 shape: no system prompt in the list, a
        re-compaction (``protect_first_n`` decayed to 0), and the only
        user turn old enough to fall into the compressed middle. Before
        the fix, the summary was emitted as ``assistant`` and the output
        had zero user-role messages."""
        from agent.context_compressor import SUMMARY_PREFIX

        c = compressor
        # A prior compaction has already happened → protect_first_n decays
        # to 0 so compress_start lands at 0 (no protected head).
        c.compression_count = 1
        # No system message: the main loop prepends it separately.
        messages = [{"role": "user", "content": "work kanban task 42"}]
        messages += _tool_turns(0, 12)

        mocked = f"{SUMMARY_PREFIX}\nrolled-up summary of the tool work"
        with patch.object(c, "_generate_summary", return_value=mocked):
            out = c.compress(messages, current_tokens=90_000)

        hist = _role_hist(out)
        assert hist.get("user", 0) >= 1, (
            "REGRESSION (#58753): compression produced a transcript with "
            f"zero user-role messages, which vLLM/Qwen reject with a "
            f"non-retryable 400. Role histogram: {hist}"
        )

    def test_summary_pinned_to_user_when_no_user_survives(self, compressor):
        """When the whole compressible region is assistant/tool and no
        user message survives in head or tail, the inserted summary
        itself must be the user turn."""
        from agent.context_compressor import (
            SUMMARY_PREFIX,
            COMPRESSED_SUMMARY_METADATA_KEY,
        )

        c = compressor
        c.compression_count = 1
        messages = [{"role": "user", "content": "work kanban task 7"}]
        messages += _tool_turns(0, 12)

        mocked = f"{SUMMARY_PREFIX}\nsummary body"
        with patch.object(c, "_generate_summary", return_value=mocked):
            out = c.compress(messages, current_tokens=90_000)

        summary_rows = [m for m in out if m.get(COMPRESSED_SUMMARY_METADATA_KEY)]
        assert len(summary_rows) == 1
        assert summary_rows[0].get("role") == "user", (
            "The handoff summary must carry role=user when it is the only "
            "possible user turn in the compressed transcript (#58753)."
        )

    def test_no_consecutive_user_roles_introduced(self, compressor):
        """Forcing the summary to role=user must not create two
        consecutive user-role messages (strict alternation invariant).
        When a user survives in the tail we do NOT force, so the pinned
        summary can never collide with a user-role neighbour."""
        from agent.context_compressor import SUMMARY_PREFIX

        c = compressor
        c.compression_count = 1
        messages = [{"role": "user", "content": "work kanban task 9"}]
        messages += _tool_turns(0, 12)

        mocked = f"{SUMMARY_PREFIX}\nsummary body"
        with patch.object(c, "_generate_summary", return_value=mocked):
            out = c.compress(messages, current_tokens=90_000)

        for prev, cur in zip(out, out[1:]):
            assert not (
                prev.get("role") == "user" and cur.get("role") == "user"
            ), "compression introduced consecutive user-role messages"

    def test_preserved_tail_user_is_not_overridden(self, compressor):
        """When a genuine user message survives in the tail, the guard
        must NOT fire (the summary keeps its alternation-driven role) —
        the request already has a user turn."""
        from agent.context_compressor import SUMMARY_PREFIX

        c = compressor
        c.compression_count = 1
        c.tail_token_budget = 10  # tight tail so most turns compress
        messages = [{"role": "user", "content": "old task"}]
        messages += _tool_turns(0, 10)
        # A recent, genuine user turn that will be preserved in the tail.
        messages += [
            {"role": "user", "content": "the latest live user question"},
            {"role": "assistant", "content": "on it"},
        ]

        mocked = f"{SUMMARY_PREFIX}\nsummary body"
        with patch.object(c, "_generate_summary", return_value=mocked):
            out = c.compress(messages, current_tokens=90_000)

        hist = _role_hist(out)
        assert hist.get("user", 0) >= 1
        joined = "\n".join(
            m.get("content") for m in out if isinstance(m.get("content"), str)
        )
        assert "the latest live user question" in joined

"""Coverage for the pathological-tail token cap in ``_find_tail_cut_by_tokens``.

The protected recent tail is bounded by a message-count floor (``protect_last_n``,
capped at ``_MAX_TAIL_MESSAGE_FLOOR``). That floor keeps a short run of recent
turns verbatim so the active task survives compaction. But when the tail holds a
few *enormous* messages — e.g. giant tool results from a file read — the count
floor forces all of them to be kept regardless of their token mass, leaving an
incompressible tail that can pin the request over the model's context window
forever (compression exhaustion; see issue #21916).

Fix: a ``hard_ceiling = token_budget * 3`` lets the token budget override the
count floor once the tail is already viable — it holds the absolute minimum of 3
messages and the most recent user message is captured — but only when the next
message is individually oversized (the genuine few-but-huge pathology). A long
run of small turns under a tiny budget must still honour the count floor
(#9413), and the last user/assistant turn is never dropped.

Fixtures here are estimator-agnostic: message sizes are measured with the
compressor's own ``_estimate_msg_budget_tokens`` and the tail budget is derived
from those measurements, so the tests hold regardless of the token-estimator's
exact chars-per-token calibration.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agent.context_compressor import (
    _MAX_TAIL_MESSAGE_FLOOR,
    _estimate_msg_budget_tokens,
)


@pytest.fixture()
def compressor():
    """ContextCompressor whose count floor wants 8 tail messages, so the
    hard-ceiling override (which drops the floor to 3) is observable."""
    from agent.context_compressor import ContextCompressor

    with patch(
        "agent.context_compressor.get_model_context_length",
        return_value=100_000,
    ):
        return ContextCompressor(
            model="test/model",
            threshold_percent=0.85,
            protect_first_n=0,
            protect_last_n=8,  # count floor wants 8 — the fix must override it
            quiet_mode=True,
        )


def _budget_for(huge_msg: dict, small_msgs: list[dict]) -> int:
    """Pick a ``tail_token_budget`` (estimator-agnostic) such that:

    * every small message is comfortably below the soft ceiling (1.5x budget),
      so the oversized-message gate never trips on them; and
    * the huge message alone exceeds the hard ceiling (3x budget), so the
      override engages exactly on it.
    """
    huge_tok = _estimate_msg_budget_tokens(huge_msg)
    small_max = max(_estimate_msg_budget_tokens(m) for m in small_msgs)
    budget = max(small_max + 10, huge_tok // 6)
    # Precondition: the pathology is only meaningful if the huge message dwarfs
    # both ceilings derived from this budget.
    assert huge_tok > budget * 3 > (small_max + 10) * 2
    return budget


class TestHardCeilingTailShrink:
    def _pathological_messages(self):
        """system + an OLD tool group + a huge OLD non-tool message + a recent
        small tail carrying the most recent user message.

        The count floor (``protect_last_n`` == 8) would otherwise drag the huge
        OLD message (index 5) into the protected tail. The hard-ceiling override
        breaks on that giant message once the recent tail already holds >= 3
        messages and the last user message is captured. The older tool group
        (indices 2/3) stays whole in the compressed region — the cut never
        splits it.
        """
        huge = "x" * 40_000  # dwarfs any reasonable tail budget
        return [
            {"role": "system", "content": "sys"},                       # 0 head
            {"role": "user", "content": "old q1"},                      # 1
            {"role": "assistant", "content": None,
             "tool_calls": [{"id": "c1", "function": {"name": "t", "arguments": "{}"}}]},  # 2
            {"role": "tool", "tool_call_id": "c1", "content": "small"},  # 3 (old group)
            {"role": "user", "content": "old q2"},                      # 4
            {"role": "assistant", "content": huge},                     # 5 HUGE (non-tool)
            {"role": "user", "content": "recent q"},                    # 6 recent window
            {"role": "assistant", "content": "small reply"},            # 7
            {"role": "user", "content": "LATEST TASK"},                 # 8 last user
            {"role": "assistant", "content": "small reply 2"},          # 9
        ]

    def _tune(self, compressor, messages):
        huge_msg = messages[5]
        small = [m for i, m in enumerate(messages) if i != 5]
        compressor.tail_token_budget = _budget_for(huge_msg, small)

    def test_tail_shrinks_below_count_floor_but_not_below_three(self, compressor):
        messages = self._pathological_messages()
        self._tune(compressor, messages)
        cut = compressor._find_tail_cut_by_tokens(messages, head_end=1)

        tail = messages[cut:]
        # The count floor would normally protect 8 messages …
        min_tail_floor = max(3, min(compressor.protect_last_n, _MAX_TAIL_MESSAGE_FLOOR))
        assert min_tail_floor == 8
        assert len(tail) < min_tail_floor, (
            f"hard ceiling should have shrunk the tail below {min_tail_floor}; "
            f"got {len(tail)} messages"
        )
        # … but never below the absolute minimum of 3.
        assert len(tail) >= 3

    def test_pathology_regression_giant_excluded_where_floor_would_keep_it(
        self, compressor
    ):
        """Behavioural delta vs the unfixed count floor.

        Without the fix the tail floor is ``n - min_tail`` messages, which spans
        the giant OLD message and keeps it verbatim forever. With the fix the
        cut lands *after* the giant, excluding it — the whole point of the
        auto-forget escape hatch.
        """
        messages = self._pathological_messages()
        self._tune(compressor, messages)
        n = len(messages)
        giant_idx = 5

        # Pre-fix behaviour: the count floor alone would start the tail at
        # ``n - min_tail`` (== 2 here), which is at or before the giant — i.e.
        # the giant would have been retained in the protected tail.
        min_tail_floor = max(3, min(compressor.protect_last_n, _MAX_TAIL_MESSAGE_FLOOR))
        unfixed_floor_cut = n - min_tail_floor
        assert unfixed_floor_cut <= giant_idx, (
            "fixture invariant: the count floor must otherwise reach the giant"
        )

        cut = compressor._find_tail_cut_by_tokens(messages, head_end=1)
        assert cut > giant_idx, (
            f"the giant message (idx {giant_idx}) must be excluded from the "
            f"tail; got cut={cut}"
        )
        huge_bodies = [
            m for m in messages[cut:]
            if isinstance(m.get("content"), str) and len(m["content"]) >= 40_000
        ]
        assert huge_bodies == [], (
            "the huge old message must not be dragged into the tail by the "
            "message-count floor"
        )

    def test_last_user_message_kept_in_tail(self, compressor):
        messages = self._pathological_messages()
        self._tune(compressor, messages)
        cut = compressor._find_tail_cut_by_tokens(messages, head_end=1)
        tail_contents = [
            m.get("content") for m in messages[cut:]
            if isinstance(m.get("content"), str)
        ]
        assert any("LATEST TASK" in (t or "") for t in tail_contents)

    def test_no_tool_group_split(self, compressor):
        """The old tool group (assistant tool_call at idx 2 + its result at
        idx 3) must land wholly on one side of the cut — never split."""
        messages = self._pathological_messages()
        self._tune(compressor, messages)
        cut = compressor._find_tail_cut_by_tokens(messages, head_end=1)
        assistant_side = 2 < cut
        result_side = 3 < cut
        assert assistant_side == result_side, (
            f"cut={cut} split the tool group (assistant idx2 / result idx3)"
        )

    def test_normal_tail_unaffected_when_no_pathology(self, compressor):
        """Small, uniform messages must still honour the count floor — the hard
        ceiling only engages on a few-but-huge tail (regression guard for
        #9413)."""
        messages = [{"role": "system", "content": "sys"}]
        for i in range(14):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": f"m{i}"})
        # A budget whose soft ceiling sits above any single small message, so no
        # message is ever "individually oversized" and the override cannot fire.
        small_max = max(_estimate_msg_budget_tokens(m) for m in messages)
        compressor.tail_token_budget = small_max * 4
        cut = compressor._find_tail_cut_by_tokens(messages, head_end=1)
        # Nothing huge → hard ceiling never fires → count floor preserved.
        assert len(messages) - cut >= 8

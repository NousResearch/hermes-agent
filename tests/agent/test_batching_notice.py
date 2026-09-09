"""Batching-score notice: cross-turn scoreboard on the tool-guardrail controller.

Emitted ONLY when single-call tool turns dominate (>60% after >=6 turns), at most once per
10-turn window, via the stall-notice channel (appended after the last tool result of a turn —
never the system prompt, which would break the provider prefix cache when the score moved).

Regression guard for the 2026-09 speed work (measured 84-87% single-call turns on real
heavy sessions; each single-call turn costs a full context re-send round-trip).
"""

from agent.tool_guardrails import ToolCallGuardrailController


def _feed(controller, turn_counts):
    """Feed one tool turn per entry; return the notices that fired."""
    return [controller.batching_notice(n) for n in turn_counts]


class TestBatchingNotice:
    def test_silent_below_min_sample(self):
        c = ToolCallGuardrailController()
        assert _feed(c, [1, 1, 1, 1, 1]) == [None] * 5

    def test_silent_when_batching_is_healthy(self):
        c = ToolCallGuardrailController()
        # 6 turns, 3 batched -> 50% single share <= 0.6 threshold: no notice
        notices = _feed(c, [1, 3, 1, 2, 1, 2])
        assert notices == [None] * 6

    def test_notice_when_single_call_turns_dominate(self):
        c = ToolCallGuardrailController()
        notices = _feed(c, [1] * 7)
        # Turn 6 crosses the min sample; single share 100% -> notice on turn 6
        assert notices[5] is not None
        assert "batching score" in notices[5]
        assert "100%" in notices[5]
        # Window: turn 7 is inside the 10-turn cooldown -> silent
        assert notices[6] is None

    def test_notice_rate_limited_to_one_per_window(self):
        c = ToolCallGuardrailController()
        notices = _feed(c, [1] * 17)
        fired = [i for i, n in enumerate(notices) if n is not None]
        # With a 100% single share: fire at turn 6 and again at 16 (>=10 later), not in between.
        assert fired[0] == 5
        assert all(f - fired[0] >= 10 for f in fired[1:]), fired

    def test_batched_turns_move_the_score_down(self):
        c = ToolCallGuardrailController()
        # 5 single turns then batched turns: the turn-6 check sees 5/6 single (83% > 60%, one
        # notice fires), but afterwards the batched turns pull the share down and the next
        # window checks stay silent.
        notices = _feed(c, [1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4])
        assert notices[5] is not None  # turn 6: 83% single — still above threshold
        assert all(n is None for i, n in enumerate(notices) if i > 5)

    def test_score_text_carries_real_numbers(self):
        c = ToolCallGuardrailController()
        notices = _feed(c, [1, 1, 1, 1, 1, 3, 1])  # first check at turn 6: 5 of 6 single (83%)
        assert notices[5] is not None
        assert "5 of 6" in notices[5]

    def test_counters_survive_reset_for_turn(self):
        # reset_for_turn clears per-turn streak state only; the batching scoreboard
        # spans the conversation and must NOT be reset — otherwise every turn looks
        # like turn 1 and the notice never fires.
        c = ToolCallGuardrailController()
        _feed(c, [1, 1, 1])
        c.reset_for_turn()
        notices = _feed(c, [1, 1, 1])
        assert notices[2] is not None  # turn 6 overall, not turn 3


# ponytail: thresholds (6/10/0.6) are module constants; tune via config only if real
# sessions show the notice is too quiet or too noisy.

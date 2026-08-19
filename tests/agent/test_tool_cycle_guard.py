"""Alternating tool-call cycle detection (loop_tool_cycle).

Covers the gap left by the four existing guards: cycles of SUCCESSFUL calls
across >= 2 distinct signatures (A,B,A,B,...), including mutating tools and
reads whose results vary between iterations — none of which the exact-failure,
same-tool-failure, idempotent-no-progress, or per-turn cap guards can see.
"""

from agent.tool_guardrails import (
    ToolCallGuardrailConfig,
    ToolCallGuardrailController,
    ToolCallSignature,
    detect_signature_cycle,
)


def _sig(name, **args):
    return ToolCallSignature.from_call(name, args)


def _run_alternation(controller, spec, results=None):
    """Feed (tool, args) pairs through after_call; return the last decision."""
    last = None
    for i, (tool, args) in enumerate(spec):
        result = (results or {}).get(i, '{"ok": %d}' % i)
        last = controller.after_call(tool, args, result, failed=False)
    return last


class TestDetectSignatureCycle:
    def test_two_signature_alternation_detected(self):
        a, b = _sig("terminal", cmd="x"), _sig("read_file", path="y")
        assert detect_signature_cycle([a, b, a, b], min_repeats=2) == (2, 2)

    def test_mono_signature_sequence_excluded(self):
        a = _sig("terminal", cmd="x")
        assert detect_signature_cycle([a] * 12, min_repeats=2) is None

    def test_progressing_sequence_not_flagged(self):
        seq = [_sig("read_file", path=str(i)) for i in range(10)]
        assert detect_signature_cycle(seq, min_repeats=2) is None

    def test_three_call_cycle_detected(self):
        a, b, c = _sig("t", x=1), _sig("u", x=2), _sig("v", x=3)
        assert detect_signature_cycle([a, b, c] * 3, min_repeats=3) == (3, 3)

    def test_noise_breaks_the_tail(self):
        a, b, x = _sig("t", x=1), _sig("u", x=2), _sig("w", x=9)
        # ...A,B,A,B,X,A,B — tail block [A,B] only repeats once before X.
        assert detect_signature_cycle([a, b, a, b, x, a, b], min_repeats=2) is None

    def test_smallest_period_wins(self):
        a, b = _sig("t", x=1), _sig("u", x=2)
        # A 2-cycle repeated 4 times is also a 4-cycle repeated twice;
        # report the tight one.
        assert detect_signature_cycle([a, b] * 4, min_repeats=2) == (2, 4)


class TestControllerCycleWarn:
    def test_warns_after_two_full_repetitions(self):
        c = ToolCallGuardrailController(ToolCallGuardrailConfig())
        spec = [("terminal", {"cmd": "poll"}), ("read_file", {"path": "log"})] * 2
        decision = _run_alternation(c, spec)
        assert decision.action == "warn"
        assert decision.code == "loop_tool_cycle_warning"
        assert decision.count == 2

    def test_varying_results_do_not_hide_the_cycle(self):
        # Distinct results defeat idempotent no-progress; the cycle guard
        # keys on call signatures and still fires.
        c = ToolCallGuardrailController(ToolCallGuardrailConfig())
        spec = [("read_file", {"path": "a"}), ("read_file", {"path": "b"})] * 2
        results = {i: '{"content": "%d"}' % i for i in range(4)}
        decision = _run_alternation(c, spec, results)
        assert decision.action == "warn"
        assert decision.code == "loop_tool_cycle_warning"

    def test_no_warning_when_warnings_disabled(self):
        c = ToolCallGuardrailController(ToolCallGuardrailConfig(warnings_enabled=False))
        spec = [("terminal", {"cmd": "poll"}), ("read_file", {"path": "log"})] * 3
        decision = _run_alternation(c, spec)
        assert decision.action == "allow"

    def test_specific_guards_speak_first(self):
        # A failing alternation must keep surfacing the exact-failure warning,
        # not stack a second cycle warning on the same call.
        c = ToolCallGuardrailController(ToolCallGuardrailConfig())
        d = None
        for _ in range(3):
            c.after_call("terminal", {"cmd": "x"}, '{"exit_code": 1}', failed=True)
            d = c.after_call("patch", {"file": "y"}, '{"exit_code": 1}', failed=True)
        assert d.action == "warn"
        assert d.code in {"repeated_exact_failure_warning", "same_tool_failure_warning"}

    def test_reset_for_turn_clears_history(self):
        c = ToolCallGuardrailController(ToolCallGuardrailConfig())
        spec = [("terminal", {"cmd": "poll"}), ("read_file", {"path": "log"})] * 2
        assert _run_alternation(c, spec).action == "warn"
        c.reset_for_turn()
        assert _run_alternation(c, spec[:2]).action == "allow"


class TestControllerCycleBlock:
    def _looped_controller(self, repeats, hard_stop=True):
        c = ToolCallGuardrailController(
            ToolCallGuardrailConfig(hard_stop_enabled=hard_stop)
        )
        spec = [("terminal", {"cmd": "poll"}), ("read_file", {"path": "log"})] * repeats
        _run_alternation(c, spec)
        return c

    def test_blocks_the_call_extending_the_fourth_repetition(self):
        # 3.5 repetitions recorded; the next "read_file" would complete the
        # 4th block-repeat, so before_call refuses it.
        c = self._looped_controller(3)
        c.after_call("terminal", {"cmd": "poll"}, '{"ok": 1}', failed=False)
        decision = c.before_call("read_file", {"path": "log"})
        assert decision.action == "block"
        assert decision.code == "loop_tool_cycle"
        assert decision.count == 4
        assert c.halt_decision is decision

    def test_no_block_without_hard_stop_opt_in(self):
        c = self._looped_controller(5, hard_stop=False)
        decision = c.before_call("terminal", {"cmd": "poll"})
        assert decision.action == "allow"

    def test_breaking_the_pattern_is_allowed(self):
        c = self._looped_controller(3)
        decision = c.before_call("web_search", {"query": "different approach"})
        assert decision.action == "allow"

"""Tests for `hermes -z --done-when` (deterministic completion gate).

The gate loop is tested with a stub agent: no model calls, real subprocess gates.
"""

import pytest

from hermes_cli.oneshot import _run_done_when_gate, _run_done_when_loop


class _StubAgent:
    """Records repair prompts; returns canned run_conversation results."""

    def __init__(self, results):
        self._results = list(results)
        self.prompts = []

    def run_conversation(self, message):
        self.prompts.append(message)
        return self._results.pop(0) if self._results else {"final_response": ""}


class TestRunDoneWhenGate:
    def test_passing_command(self):
        passed, code, tail = _run_done_when_gate("echo ok")
        assert passed is True
        assert code == 0
        assert "ok" in tail

    def test_failing_command_reports_exit_code_and_output(self):
        passed, code, tail = _run_done_when_gate("echo broken >&2; exit 3")
        assert passed is False
        assert code == 3
        assert "broken" in tail

    def test_timeout_counts_as_failure(self, monkeypatch):
        import hermes_cli.oneshot as mod

        monkeypatch.setattr(mod, "_DONE_WHEN_TIMEOUT_SECONDS", 1)
        passed, code, tail = _run_done_when_gate("sleep 5")
        assert passed is False
        assert code == -1
        assert "timed out" in tail

    def test_output_tail_is_bounded(self):
        passed, code, tail = _run_done_when_gate("python3 -c 'print(\"x\" * 100000)'")
        assert passed is True
        assert len(tail) <= 3000 + 10  # bounded tail + small slack


class TestRunDoneWhenLoop:
    def test_gate_passes_first_try_no_repair_turn(self):
        agent = _StubAgent([{"final_response": "done"}])
        result = {"final_response": "done"}
        _run_done_when_loop(agent, result, "true", retries=3)
        assert result["done_when_passed"] is True
        assert result["done_when_exit_code"] == 0
        assert agent.prompts == []  # no repair turn burned

    def test_gate_fails_then_repair_turn_fixes_it(self, tmp_path, monkeypatch):
        # Gate flips from fail to pass after the agent's first repair turn
        # (the repair creates the file the gate checks).
        marker = tmp_path / "fixed.txt"

        def gate_spy(command):
            return (marker.exists(), 0 if marker.exists() else 1, "missing marker")

        monkeypatch.setattr("hermes_cli.oneshot._run_done_when_gate", gate_spy)

        def repair(message):
            marker.write_text("fixed", encoding="utf-8")
            return {"final_response": "fixed it"}

        agent = _StubAgent([])
        agent.run_conversation = repair
        result = {"final_response": "not yet"}
        _run_done_when_loop(agent, result, "test -f marker", retries=3)
        assert result["done_when_passed"] is True
        # The repair turn's result replaced the original one.
        assert result["final_response"] == "fixed it"

    def test_gate_exhausts_retries_and_reports_failure(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.oneshot._run_done_when_gate",
            lambda command: (False, 2, "still red"),
        )
        agent = _StubAgent([{"final_response": "attempt 1"}, {"final_response": "attempt 2"}])
        result = {"final_response": "original"}
        _run_done_when_loop(agent, result, "pytest -q", retries=2)
        assert result["done_when_passed"] is False
        assert result["done_when_exit_code"] == 2
        # retries=2 → exactly 2 repair turns, then the gate result stands.
        assert len(agent.prompts) == 2
        assert "pytest -q" in agent.prompts[0]
        assert "still red" in agent.prompts[0]
        # The final repair turn's response wins.
        assert result["final_response"] == "attempt 2"

    def test_zero_retries_single_gate_run_no_repair(self, monkeypatch):
        calls = []

        def gate_spy(command):
            calls.append(command)
            return (False, 1, "red")

        monkeypatch.setattr("hermes_cli.oneshot._run_done_when_gate", gate_spy)
        agent = _StubAgent([])
        result = {"final_response": "claimed done"}
        _run_done_when_loop(agent, result, "pytest -q", retries=0)
        assert result["done_when_passed"] is False
        assert len(calls) == 1
        assert agent.prompts == []

    def test_repair_turn_budget_label_counts_down(self, monkeypatch):
        """Regression: the `repair turn(s) left` label must report turns REMAINING.

        A review caught that the message was built from `retries - attempts_left`
        (turns USED, counting up 1,2,3) after the decrement — telling the repair
        agent "3 left" on its final turn, exactly backwards as a budget signal.
        Under the old code the first prompt said "1 repair turn(s) left", so this
        test asserting "2 left" could not have passed before the fix.
        """
        monkeypatch.setattr(
            "hermes_cli.oneshot._run_done_when_gate",
            lambda command: (False, 1, "still red"),
        )
        agent = _StubAgent(
            [
                {"final_response": "a1"},
                {"final_response": "a2"},
                {"final_response": "a3"},
            ]
        )
        result = {"final_response": "original"}
        _run_done_when_loop(agent, result, "pytest -q", retries=3)
        assert len(agent.prompts) == 3
        budgets = [p.split("repair turn(s) left")[0].rsplit(", ", 1)[1].strip() for p in agent.prompts]
        assert budgets == ["2", "1", "0"], f"expected countdown 2,1,0 — got {budgets}"
        for prompt in agent.prompts:
            assert "still red" in prompt and "pytest -q" in prompt

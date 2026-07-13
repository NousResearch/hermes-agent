"""Tests for the completion gate system.

Covers:
  - GoalRegistry CRUD
  - Trace compression (build_verify_trace)
  - JSON extraction (fail-open, markdown-wrapped)
  - Gate 1: goal contract rejection
  - Gate 2: judge verdict parsing
  - Fail-open: verifyRounds cap
  - Tool handler: mark_goal_met, cancel_goal
  - Agent-loop integration: declare_complete via invoke_tool
  - AIAgent: enable_completion_gate flag
"""

import json
import sys
import os
import pytest

# Ensure we can import hermes-agent modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.completion_gate import (
    CompletionGate,
    GoalRegistry,
    Goal,
    build_verify_trace,
    _extract_json,
    VERIFY_MAX_ROUNDS,
    MAX_TRACE_CHARS,
)


# ─── GoalRegistry Tests ─────────────────────────────────────────────────────

class TestGoalRegistry:
    def test_add_goal(self):
        reg = GoalRegistry()
        g = reg.add("Fix all bugs")
        assert g.id == "goal-1"
        assert g.status == "active"
        assert g.condition == "Fix all bugs"

    def test_mark_met(self):
        reg = GoalRegistry()
        reg.add("Fix bugs")
        ok = reg.mark_met("goal-1", "npm test exit 0, 41/41 passed")
        assert ok
        assert reg.goals["goal-1"].status == "met"
        assert "41/41" in reg.goals["goal-1"].evidence

    def test_mark_met_nonexistent(self):
        reg = GoalRegistry()
        ok = reg.mark_met("goal-99", "evidence")
        assert not ok

    def test_mark_met_already_resolved(self):
        reg = GoalRegistry()
        reg.add("Task")
        reg.mark_met("goal-1", "done")
        ok = reg.mark_met("goal-1", "done again")
        assert not ok  # Already met

    def test_cancel(self):
        reg = GoalRegistry()
        reg.add("Impossible task")
        ok = reg.cancel("goal-1", "Blocked by external dependency")
        assert ok
        assert reg.goals["goal-1"].status == "cancelled"

    def test_active_goals(self):
        reg = GoalRegistry()
        reg.add("Task A")
        reg.add("Task B")
        assert len(reg.active_goals) == 2
        reg.mark_met("goal-1", "done")
        assert len(reg.active_goals) == 1
        assert reg.active_goals[0].id == "goal-2"

    def test_all_resolved(self):
        reg = GoalRegistry()
        assert reg.all_resolved  # No goals = all resolved
        reg.add("Task")
        assert not reg.all_resolved
        reg.mark_met("goal-1", "done")
        assert reg.all_resolved

    def test_mixed_resolution(self):
        reg = GoalRegistry()
        reg.add("Task A")
        reg.add("Task B")
        reg.add("Task C")
        reg.mark_met("goal-1", "done")
        reg.cancel("goal-3", "N/A")
        assert not reg.all_resolved  # goal-2 still active
        reg.mark_met("goal-2", "done too")
        assert reg.all_resolved


# ─── Trace Compression Tests ─────────────────────────────────────────────────

class TestBuildVerifyTrace:
    def test_basic_compression(self):
        msgs = [
            {"role": "user", "content": "Fix the parser bugs"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "read_file"}},
                {"function": {"name": "edit_file"}},
                {"function": {"name": "run_shell"}},
            ]},
            {"role": "tool", "content": "npm test -> 41/41 passed, exit 0", "tool_name": "run_shell"},
        ]
        reg = GoalRegistry()
        reg.add("Fix parser bugs")
        reg.mark_met("goal-1", "npm test exit 0, 41/41 passed")

        trace = build_verify_trace(msgs, reg, {"status": "complete", "summary": "All fixed"})

        assert "Messages: 3" in trace
        assert "Tool results: 1" in trace
        assert "[1] read_file" in trace
        assert "[2] edit_file" in trace
        assert "[3] run_shell" in trace
        assert "goal-1" in trace
        assert "(met)" in trace
        assert "41/41" in trace

    def test_compression_ratio(self):
        """Trace should be significantly smaller than the full transcript."""
        msgs = [
            {"role": "user", "content": "x" * 100},
            {"role": "assistant", "content": "y" * 200, "tool_calls": [
                {"function": {"name": "terminal"}},
            ]},
            {"role": "tool", "content": "z" * 5000, "tool_name": "terminal"},
        ]
        reg = GoalRegistry()
        reg.add("Task")
        reg.mark_met("goal-1", "done")

        trace = build_verify_trace(msgs, reg, {"status": "complete", "summary": "Done"})
        transcript_chars = sum(len(str(m)) for m in msgs)

        assert len(trace) < transcript_chars
        # Large tool outputs should be excluded (names only in trace)
        assert "z" * 100 not in trace

    def test_empty_messages(self):
        trace = build_verify_trace([], GoalRegistry(), {"status": "complete", "summary": ""})
        assert "Messages: 0" in trace
        assert "Tool call sequence: (none)" in trace

    def test_max_chars_truncation(self):
        """Trace should not exceed MAX_TRACE_CHARS."""
        msgs = [{"role": "user", "content": "x" * 100}] * 100
        reg = GoalRegistry()
        for i in range(20):
            reg.add(f"Goal {i}: " + "do something " * 10)
            reg.mark_met(f"goal-{i+1}", "evidence " * 50)

        trace = build_verify_trace(msgs, reg, {"status": "complete", "summary": "y" * 2000})
        assert len(trace) <= MAX_TRACE_CHARS

    def test_no_tool_calls(self):
        """Trace handles conversations with no tool calls gracefully."""
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        reg = GoalRegistry()
        trace = build_verify_trace(msgs, reg, {"status": "complete", "summary": "Greeted"})
        assert "Tool call sequence: (none)" in trace
        assert "Hi there!" in trace


# ─── JSON Extraction Tests ──────────────────────────────────────────────────

class TestExtractJson:
    def test_plain_json(self):
        result = _extract_json('{"pass": true, "reason": "ok", "feedback": "", "unmet_goal_ids": []}')
        assert result["pass"] is True
        assert result["reason"] == "ok"

    def test_markdown_wrapped(self):
        result = _extract_json(
            '```json\n{"pass": false, "reason": "no", "feedback": "try harder", "unmet_goal_ids": ["goal-1"]}\n```'
        )
        assert result["pass"] is False
        assert result["unmet_goal_ids"] == ["goal-1"]

    def test_prose_wrapped(self):
        result = _extract_json(
            'I think the task was completed. Here is my verdict: {"pass": true, "reason": "good", "feedback": "", "unmet_goal_ids": []}'
        )
        assert result["pass"] is True

    def test_fail_open_unparseable(self):
        """Malformed output should fail-open (pass the task)."""
        result = _extract_json("I'm sorry, I cannot produce valid JSON today")
        assert result["pass"] is True
        assert "fail-open" in result["reason"]

    def test_fail_open_empty(self):
        result = _extract_json("")
        assert result["pass"] is True

    def test_fail_open_none(self):
        result = _extract_json(None)
        assert result["pass"] is True


# ─── CompletionGate Tests ────────────────────────────────────────────────────

class TestCompletionGate:
    def test_gate_1_rejects_active_goals(self):
        """Gate 1: if goals are still active, declare_complete fails."""
        gate = CompletionGate()
        gate.goals.add("Fix all 3 bugs")
        result = gate.declare_complete(None, [], status="complete", summary="Done")
        assert "error" in result
        assert "still active" in result["error"]
        assert "goal-1" in result["error"]

    def test_gate_1_passes_when_all_resolved(self):
        """Gate 1 passes when all goals are met or cancelled."""
        gate = CompletionGate()
        gate.goals.add("Fix bugs")
        gate.goals.mark_met("goal-1", "npm test exit 0")
        # With no agent, judge call fails → fail-open
        result = gate.declare_complete(None, [], status="complete", summary="Fixed")
        assert "output" in result
        assert "fail-open" in result["output"]

    def test_fail_open_after_verify_rounds(self):
        """After VERIFY_MAX_ROUNDS rejections, gate fail-opens."""
        gate = CompletionGate()
        gate.goals.add("Task")
        gate.goals.mark_met("goal-1", "evidence")

        # Simulate a stubborn judge
        stubborn = lambda trace: {"pass": False, "reason": "Nope", "feedback": "Try again", "unmet_goal_ids": []}
        gate._judge_fn = stubborn

        for i in range(VERIFY_MAX_ROUNDS):
            result = gate.declare_complete(None, [], status="complete", summary="x")
            assert "error" in result, f"Round {i+1} should reject"

        # Next call should pass (fail-open)
        result = gate.declare_complete(None, [], status="complete", summary="x")
        assert "output" in result
        assert "fail-open" in result["output"]

    def test_verify_rounds_reset_on_pass(self):
        """verify_rounds counter resets after a successful pass."""
        gate = CompletionGate()
        gate.verify_rounds = 2
        gate.goals.add("Task")
        gate.goals.mark_met("goal-1", "done")
        result = gate.declare_complete(None, [], status="complete", summary="x")
        # Fail-open since no agent
        assert gate.verify_rounds == 0

    def test_reset(self):
        """reset() clears all state."""
        gate = CompletionGate()
        gate.goals.add("Task")
        gate.verify_rounds = 2
        gate.reset()
        assert gate.verify_rounds == 0
        assert len(gate.goals.goals) == 0

    def test_judge_unmet_goals_reopen(self):
        """When judge flags unmet_goal_ids, those goals are re-opened."""
        gate = CompletionGate()
        gate.goals.add("Task A")
        gate.goals.mark_met("goal-1", "vague evidence")
        gate.goals.add("Task B")
        gate.goals.mark_met("goal-2", "good evidence")

        def judge_that_hates_vague(trace):
            return {"pass": False, "reason": "Vague", "feedback": "Be specific",
                    "unmet_goal_ids": ["goal-1"]}

        gate._judge_fn = judge_that_hates_vague
        gate.declare_complete(None, [], status="complete", summary="x")

        # goal-1 should be re-opened, goal-2 should stay met
        assert gate.goals.goals["goal-1"].status == "active"
        assert gate.goals.goals["goal-2"].status == "met"

    def test_partial_status(self):
        """'partial' status works (some cancelled)."""
        gate = CompletionGate()
        gate.goals.add("Task A")
        gate.goals.mark_met("goal-1", "done")
        gate.goals.add("Task B")
        gate.goals.cancel("goal-2", "Blocked")
        # With no agent → fail-open
        result = gate.declare_complete(None, [], status="partial", summary="A done, B cancelled")
        assert "output" in result


# ─── Integration Tests (no API calls) ────────────────────────────────────────

class TestCompletionGateIntegration:
    """Tests that exercise the full flow without real API calls."""

    def test_full_happy_path(self):
        """Simulate a successful autonomous task with proper evidence."""
        gate = CompletionGate()

        # Agent establishes goals
        gate.goals.add("Fix all 3 parser bugs")
        gate.goals.add("Run regression tests")

        # Agent does the work...
        msgs = [
            {"role": "user", "content": "Fix parser bugs and run tests"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "read_file"}},
            ]},
            {"role": "tool", "content": "parser.ts: bugs at lines 88, 132, 201", "tool_name": "read_file"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "edit_file"}},
                {"function": {"name": "edit_file"}},
                {"function": {"name": "edit_file"}},
            ]},
            {"role": "tool", "content": "Fixed line 88", "tool_name": "edit_file"},
            {"role": "tool", "content": "Fixed line 132", "tool_name": "edit_file"},
            {"role": "tool", "content": "Fixed line 201", "tool_name": "edit_file"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "run_shell"}},
            ]},
            {"role": "tool", "content": "npm test -> 41/41 passed, exit 0", "tool_name": "run_shell"},
        ]

        # Agent provides evidence
        gate.goals.mark_met("goal-1", "npm test exit 0, 41/41 passed; fixes at parser.ts:88,132,201")
        gate.goals.mark_met("goal-2", "npm test exit 0, 41/41 passed, all regression tests green")

        # With a judge that checks for verification actions
        def fair_judge(trace):
            if "run_shell" in trace and "41/41" in trace:
                return {"pass": True, "reason": "Verification present with concrete results",
                        "feedback": "", "unmet_goal_ids": []}
            return {"pass": False, "reason": "Missing verification", "feedback": "Run tests",
                    "unmet_goal_ids": []}

        gate._judge_fn = fair_judge
        result = gate.declare_complete(None, msgs, status="complete",
                                       summary="All 3 parser bugs fixed, 41/41 tests passing")

        assert "output" in result
        assert "complete" in result["output"]

    def test_vague_evidence_rejected(self):
        """Self-assessments without verification should be rejected."""
        gate = CompletionGate()
        gate.goals.add("Fix all bugs")
        gate.goals.mark_met("goal-1", "I checked and it looks fine")  # Vague!

        msgs = [
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "edit_file"}},
            ]},
            {"role": "tool", "content": "edited something", "tool_name": "edit_file"},
        ]

        # No verification action (no run_shell/terminal)
        def strict_judge(trace):
            has_verify = "run_shell" in trace or "terminal" in trace or "execute_code" in trace
            has_concrete = "exit 0" in trace or "passed" in trace
            if not has_verify:
                return {"pass": False, "reason": "No verification action in trace",
                        "feedback": "Run tests and include output in evidence",
                        "unmet_goal_ids": ["goal-1"]}
            if not has_concrete:
                return {"pass": False, "reason": "Evidence is vague self-assessment",
                        "feedback": "Include exit codes or pass counts",
                        "unmet_goal_ids": ["goal-1"]}
            return {"pass": True, "reason": "ok", "feedback": "", "unmet_goal_ids": []}

        gate._judge_fn = strict_judge
        result = gate.declare_complete(None, msgs, status="complete", summary="Fixed")

        assert "error" in result
        assert "verification" in result["error"].lower()


# ─── Tool Handler Tests ─────────────────────────────────────────────────────

class TestToolHandlers:
    """Test the tool handler functions directly."""

    def test_mark_goal_met_no_gate(self):
        """When gate is not enabled, mark_goal_met returns error."""
        from tools.completion_gate import _tool_mark_goal_met
        result = _tool_mark_goal_met(goal_id="goal-1", evidence="done")
        data = json.loads(result)
        assert "error" in data
        assert "not enabled" in data["error"]

    def test_cancel_goal_no_gate(self):
        """When gate is not enabled, cancel_goal returns error."""
        from tools.completion_gate import _tool_cancel_goal
        result = _tool_cancel_goal(goal_id="goal-1", reason="N/A")
        data = json.loads(result)
        assert "error" in data
        assert "not enabled" in data["error"]

    def test_declare_complete_no_gate(self):
        """When gate is not enabled, declare_complete returns error."""
        from tools.completion_gate import invoke_declare_complete

        class FakeAgent:
            _completion_gate = None

        result = invoke_declare_complete(FakeAgent(), status="complete", summary="Done")
        data = json.loads(result)
        assert "error" in data
        assert "not enabled" in data["error"]

    def test_mark_goal_met_with_gate(self):
        """With a gate, mark_goal_met works."""
        from tools.completion_gate import _tool_mark_goal_met
        from agent.completion_gate import CompletionGate

        gate = CompletionGate()
        gate.goals.add("Test task")

        result = _tool_mark_goal_met(
            goal_id="goal-1",
            evidence="npm test exit 0, 41/41 passed",
            _agent=FakeAgentWithGate(gate),
        )
        data = json.loads(result)
        assert data["success"] is True
        assert gate.goals.goals["goal-1"].status == "met"


class FakeAgentWithGate:
    def __init__(self, gate):
        self._completion_gate = gate


# ─── AIAgent Integration Tests ──────────────────────────────────────────────

class TestAIAgentIntegration:
    """Test that enable_completion_gate flag works on AIAgent."""

    def test_flag_off_by_default(self):
        """Without the flag, no gate is created."""
        # Import carefully to avoid triggering side effects
        # We just check the __init__ signature/attribute pattern
        from run_agent import AIAgent

        # Can't easily instantiate without API keys, but we can check
        # the default value in the signature
        import inspect
        sig = inspect.signature(AIAgent.__init__)
        params = sig.parameters
        assert "enable_completion_gate" in params
        assert params["enable_completion_gate"].default is False

    def test_agent_has_gate_attribute(self):
        """AIAgent always has _completion_gate = None by default."""
        # Mock enough of AIAgent to test attribute initialization
        # This validates the attribute exists even without full init
        from run_agent import AIAgent
        # The _completion_gate attr is set in __init__ after the forwarder
        # We'll verify by checking the code path exists
        import inspect
        source = inspect.getsource(AIAgent.__init__)
        assert "_completion_gate" in source
        assert "enable_completion_gate" in source
        assert "CompletionGate" in source


# ─── Edge Cases ──────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_goal_id_counter_reset_on_new_gate(self):
        """Each gate starts with goal-1."""
        g1 = CompletionGate()
        g1.goals.add("Task")
        assert g1.goals.goals["goal-1"].condition == "Task"

        g2 = CompletionGate()
        g2.goals.add("Other task")
        assert g2.goals.goals["goal-1"].condition == "Other task"

    def test_mark_met_idempotency(self):
        """Marking an already-met goal is a no-op."""
        reg = GoalRegistry()
        reg.add("Task")
        assert reg.mark_met("goal-1", "evidence v1")
        assert not reg.mark_met("goal-1", "evidence v2")  # Already met

    def test_cancel_idempotency(self):
        """Cancelling an already-cancelled goal is a no-op."""
        reg = GoalRegistry()
        reg.add("Task")
        assert reg.cancel("goal-1", "reason 1")
        assert not reg.cancel("goal-1", "reason 2")  # Already cancelled

    def test_large_summary_truncation(self):
        """Very long summaries are truncated in the trace."""
        trace = build_verify_trace(
            [], GoalRegistry(),
            {"status": "complete", "summary": "x" * 3000}
        )
        assert len(trace) <= MAX_TRACE_CHARS

    def test_unicode_in_trace(self):
        """Unicode content doesn't break trace building."""
        msgs = [
            {"role": "user", "content": "修复解析器错误 🐛"},
            {"role": "assistant", "content": "已修复 ✓"},
        ]
        reg = GoalRegistry()
        reg.add("修复所有错误")
        reg.mark_met("goal-1", "测试通过 ✓")
        trace = build_verify_trace(msgs, reg, {"status": "complete", "summary": "完成"})
        assert "🐛" in trace or "修复" in trace

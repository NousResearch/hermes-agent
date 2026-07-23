"""Tests for the post-tool completion-report gate policy."""

from agent.completion_report_gate import (
    CompletionReportGate,
    completion_report_gate_enabled,
)


def test_gate_is_disabled_by_default():
    assert completion_report_gate_enabled({"agent": {}}) is False


def test_gate_reads_persisted_config():
    assert completion_report_gate_enabled(
        {"agent": {"completion_report_gate": True}}
    ) is True
    assert completion_report_gate_enabled(
        {"agent": {"completion_report_gate": "yes"}}
    ) is True
    assert completion_report_gate_enabled(
        {"agent": {"completion_report_gate": False}}
    ) is False


def test_reportable_batch_arms_gate_but_housekeeping_does_not():
    gate = CompletionReportGate(enabled=True)

    assert gate.arm(["todo", "skill_view"]) is False
    assert gate.pending is False

    assert gate.arm(["web_search", "todo"]) is True
    assert gate.pending is True
    assert gate.pending_tools == ("web_search",)


def test_visible_report_allows_next_tool_batch_and_clears_gate():
    gate = CompletionReportGate(enabled=True)
    gate.arm(["terminal"])

    decision = gate.before_tool_batch(
        "Update finished successfully. Cleanup is the next phase."
    )

    assert decision.action == "allow"
    assert gate.pending is False


def test_structured_visible_report_allows_next_tool_batch():
    gate = CompletionReportGate(enabled=True)
    gate.arm(["terminal"])

    decision = gate.before_tool_batch(
        [{"type": "output_text", "text": "Update finished successfully."}]
    )

    assert decision.action == "allow"
    assert gate.pending is False


def test_silent_consecutive_batch_requires_safe_deterministic_report():
    gate = CompletionReportGate(enabled=True)
    gate.arm(["terminal", "web_search", "terminal"])

    decision = gate.before_tool_batch("<think>silent</think>")

    assert decision.action == "report"
    assert decision.message == (
        "Previous tool phase ended (terminal, web search). "
        "Continuing with the next step."
    )
    assert gate.pending is False


def test_disabled_gate_never_reports():
    gate = CompletionReportGate(enabled=False)
    assert gate.arm(["terminal"]) is False
    assert gate.before_tool_batch("").action == "allow"

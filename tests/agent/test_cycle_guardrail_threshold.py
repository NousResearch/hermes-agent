"""Configured cycle hard stops remain reachable beyond the default history window."""

import pytest

from agent.tool_guardrails import ToolCallGuardrailConfig, ToolCallGuardrailController


@pytest.mark.parametrize("period,laps", [(4, 5), (2, 40), (3, 24), (4, 20)])
def test_cycle_halts_at_configured_lap_count(period, laps):
    controller = ToolCallGuardrailController(ToolCallGuardrailConfig.from_mapping({
        "hard_stop_enabled": True,
        "hard_stop_after": {"idempotent_no_progress": laps},
    }))
    for index in range(period * laps):
        args = {"command": f"probe-{index % period}"}
        result = f"unchanged-{index % period}"
        assert controller.before_call("terminal", args).action == "allow"
        controller.after_call("terminal", args, result, failed=False)
        controller.observe_call("terminal", args, result)
        if index < period * laps - 1:
            assert controller.halt_decision is None

    halt = controller.halt_decision
    assert halt is not None, "the configured lap count must be reachable"
    assert halt.code == "identical_cycle_halt"
    assert halt.count == laps


@pytest.mark.parametrize("kind", ["progress", "poll", "warn_only"])
def test_large_cycle_threshold_preserves_progress_and_exemptions(kind):
    controller = ToolCallGuardrailController(ToolCallGuardrailConfig(
        hard_stop_enabled=kind != "warn_only", no_progress_block_after=20,
    ))
    tool = "mcp_probe_job_poll" if kind == "poll" else "terminal"
    notices = []
    for index in range(100):
        phase = index % 4
        args = {"command": f"probe-{phase}"}
        result = f"output-{index}" if kind == "progress" else f"unchanged-{phase}"
        assert controller.before_call(tool, args).action == "allow"
        controller.after_call(tool, args, result, failed=False)
        notices.append(controller.observe_call(tool, args, result).notice)
        assert controller.halt_decision is None

    assert any(notices) == (kind == "warn_only")

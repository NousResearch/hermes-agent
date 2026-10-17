"""Runtime disposition of tool guardrails; skill preparation blocks stay local."""

import logging

from agent.skill_resolution import LOOKUP_TOOLS, handle_lookup_halt, resolution_for

logger = logging.getLogger(__name__)


def set_tool_guardrail_halt(agent, decision):
    from run_agent import record_guardrail_decision

    if not decision.should_halt:
        return
    record_guardrail_decision(agent, decision.action, decision.code)
    if handle_lookup_halt(agent, decision):
        return
    if agent._tool_guardrail_halt_decision is None:
        agent._tool_guardrail_halt_decision = decision


def guardrail_block_result(agent, decision):
    from agent.tool_guardrails import toolguard_synthetic_result

    agent._set_tool_guardrail_halt(decision)
    if handle_lookup_halt(agent, decision):
        return resolution_for(agent).blocked_result(decision)
    return toolguard_synthetic_result(decision)


def append_guardrail_observation(self, tool_name: str, function_args: dict, function_result: str, *,
                                  failed: bool, tool_call_id: str = "") -> str:
    from run_agent import append_toolguard_guidance, record_guardrail_warnings

    decision = self._tool_guardrails.after_call(tool_name, function_args, function_result, failed=failed)
    # Identical-call stall guards observe the RAW result (before the per-call loop suffix) and are applied
    # at result construction so tool results stay append-only / cache-safe.
    stall_notice = result_stub = stall_kind = None
    if self._stall_guards_enabled():
        try:
            observation = self._tool_guardrails.observe_call(
                tool_name, function_args, function_result if isinstance(function_result, str) else None,
                tool_call_id=tool_call_id, failed=failed,
            )
            stall_notice, result_stub, stall_kind = observation.notice, observation.stub, observation.kind
        except Exception:
            logger.exception("stall-guard identical-call observation failed")
    # A retained skill may be restoring instructions lost in compaction. A reference
    # stub could point straight back to that pruned body; keep the restored content.
    if result_stub and isinstance(function_result, str) and tool_name not in LOOKUP_TOOLS:
        function_result = result_stub
    if decision.action in {"warn", "halt"}:
        function_result = append_toolguard_guidance(function_result, decision)
    record_guardrail_warnings(self, decision, stall_kind if stall_notice else None)
    if decision.should_halt:
        self._set_tool_guardrail_halt(decision)
    else:
        # observe_call may have raised the identical-call streak or batch-cycle halt (hard_stop_enabled, tool-agnostic).
        streak_halt = self._tool_guardrails.halt_decision
        if streak_halt is not None and streak_halt.code in ("identical_call_streak_halt", "identical_cycle_halt"):
            function_result = append_toolguard_guidance(function_result, streak_halt)
            self._set_tool_guardrail_halt(streak_halt)
    if stall_notice:
        function_result = (function_result or "") + "\n\n" + stall_notice
    return function_result

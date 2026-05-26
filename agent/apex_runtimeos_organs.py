"""APEX RuntimeOS organ hooks for Hermes runtime checkpoints.

These hooks record Router, Planner, and GeneSelector checkpoints with
deterministic recommendations. Default mode is observational. Automatic control
is deliberately opt-in and narrow: when enabled in enforce mode, recommendation
severity can mark the checkpoint as blocking, but the hook still does not mutate
messages, tools, model selection, memory/skills, or final response text.
"""

from __future__ import annotations

import os
import time
from types import MappingProxyType
from typing import Any, Dict, Mapping

_ALLOWED_POINTS = {
    "router.pre_api_request",
    "planner.pre_api_request",
    "gene_selector.pre_completion",
}
_ALLOWED_SEVERITIES = {"info", "warn", "error"}


def _mode() -> str:
    value = os.environ.get("APEX_RUNTIMEOS_GATE_MODE", "warn").strip().lower() or "warn"
    return value if value in {"dry_run", "warn", "enforce"} else "warn"


def _enabled() -> bool:
    return os.environ.get("APEX_RUNTIMEOS_GATE_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off", ""}


def _auto_control_enabled() -> bool:
    return os.environ.get("APEX_RUNTIMEOS_AUTO_CONTROL_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"}


def _auto_control_min_severity() -> str:
    value = os.environ.get("APEX_RUNTIMEOS_AUTO_CONTROL_MIN_SEVERITY", "error").strip().lower() or "error"
    return value if value in _ALLOWED_SEVERITIES else "error"


def _severity_rank(value: Any) -> int:
    return {"info": 0, "warn": 1, "error": 2}.get(str(value or "info").lower(), 0)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _recommendation(code: str, severity: str = "info", reason: str = "", *, actions: list[str] | None = None) -> Dict[str, Any]:
    severity = severity if severity in _ALLOWED_SEVERITIES else "info"
    return {
        "code": code,
        "severity": severity,
        "reason": reason,
        "actions": actions or [],
        "applied": False,
        "mutates_runtime": False,
    }


def _router_recommendation(context: Mapping[str, Any]) -> Dict[str, Any]:
    tool_count = _as_int(context.get("tool_count"))
    message_count = _as_int(context.get("message_count"))
    provider = str(context.get("provider") or "")
    if not provider:
        return _recommendation("router_provider_missing", "warn", "provider 未记录，建议复核路由上下文", actions=["review_provider_resolution"])
    if tool_count > 80 or message_count > 120:
        return _recommendation("router_high_context_pressure", "warn", "工具或消息数量偏高，建议进入压缩/降载候选", actions=["consider_context_compression", "review_toolset_scope"])
    return _recommendation("router_route_ok", "info", "当前路由观测未发现异常")


def _planner_recommendation(context: Mapping[str, Any]) -> Dict[str, Any]:
    chars = _as_int(context.get("request_char_count"))
    approx_tokens = _as_int(context.get("approx_input_tokens"))
    api_calls = _as_int(context.get("api_call_count"))
    if approx_tokens > 180_000 or chars > 700_000:
        return _recommendation("planner_context_heavy", "warn", "输入规模偏高，建议压缩、分段或复核预算", actions=["compress_context", "split_request", "review_budget"])
    if api_calls > 40:
        return _recommendation("planner_many_api_calls", "warn", "本轮 API 调用次数偏高，建议复核循环退出条件", actions=["review_loop_exit", "summarize_progress"])
    return _recommendation("planner_budget_ok", "info", "当前预算观测未发现异常")


def _gene_recommendation(context: Mapping[str, Any]) -> Dict[str, Any]:
    completed = bool(context.get("completed"))
    interrupted = bool(context.get("interrupted"))
    final_present = bool(context.get("final_response_present"))
    if interrupted:
        return _recommendation("gene_interrupted", "warn", "会话被中断，建议不沉淀进化结论", actions=["mark_incomplete", "avoid_gene_write"])
    if not completed or not final_present:
        return _recommendation("gene_completion_incomplete", "warn", "完成态或最终回复缺失，建议只记录过程不入库", actions=["hold_gene_entry", "review_completion_state"])
    return _recommendation("gene_completion_ok", "info", "完成态正常，可作为后续人工复核线索")


def evaluate_recommendation_control(recommendation: Mapping[str, Any], *, mode: str | None = None) -> Dict[str, Any]:
    """Return the opt-in automatic-control decision for a recommendation.

    This function is intentionally deterministic and side-effect free. It never
    applies the recommendation payload; it only decides whether an enforce-mode
    checkpoint should block on the recommendation severity.
    """
    current_mode = mode or _mode()
    auto_enabled = _auto_control_enabled()
    min_severity = _auto_control_min_severity()
    severity = str(recommendation.get("severity") or "info").lower()
    if severity not in _ALLOWED_SEVERITIES:
        severity = "info"
    should_block = bool(
        _enabled()
        and auto_enabled
        and current_mode == "enforce"
        and _severity_rank(severity) >= _severity_rank(min_severity)
    )
    return {
        "auto_control_enabled": auto_enabled,
        "mode": current_mode,
        "min_severity": min_severity,
        "severity": severity,
        "action": "block" if should_block else "allow",
        "blocking": should_block,
        "applied": should_block,
        "mutates_runtime": False,
        "reason": "severity_gate" if should_block else "not_applicable",
    }


def _checkpoint(point: str, context: Mapping[str, Any], output: Mapping[str, Any]) -> Dict[str, Any]:
    start = time.perf_counter()
    if point not in _ALLOWED_POINTS:
        raise ValueError(f"unsupported APEX RuntimeOS organ point: {point}")
    mode = _mode()
    recommendation = output.get("recommendation") if isinstance(output, Mapping) else None
    control = evaluate_recommendation_control(recommendation, mode=mode) if isinstance(recommendation, Mapping) else {
        "auto_control_enabled": _auto_control_enabled(),
        "mode": mode,
        "min_severity": _auto_control_min_severity(),
        "severity": "info",
        "action": "allow",
        "blocking": False,
        "applied": False,
        "mutates_runtime": False,
        "reason": "no_recommendation",
    }
    result = {
        "enabled": _enabled(),
        "mode": mode,
        "point": point,
        "status": "BLOCKED" if control.get("blocking") else "PASS",
        "blocking": bool(control.get("blocking")),
        "context_unchanged": True,
        "output": dict(output),
        "control": control,
        "elapsed_ms": 0.0,
    }
    if isinstance(recommendation, Mapping) and control.get("applied"):
        result["output"]["recommendation"] = {**dict(recommendation), "applied": True}
    if not result["enabled"]:
        result["status"] = "SKIPPED"
        result["blocking"] = False
        result["control"] = {**control, "action": "allow", "blocking": False, "applied": False, "reason": "gate_disabled"}
    result["elapsed_ms"] = round((time.perf_counter() - start) * 1000, 3)
    return result


def run_router_checkpoint(context: Mapping[str, Any]) -> Dict[str, Any]:
    readonly = MappingProxyType(dict(context))
    output = {
        "model": readonly.get("model"),
        "provider": readonly.get("provider"),
        "api_mode": readonly.get("api_mode"),
        "tool_count": readonly.get("tool_count", 0),
        "message_count": readonly.get("message_count", 0),
        "decision": "observe_current_route",
        "recommendation": _router_recommendation(readonly),
        "mutates_route": False,
    }
    return _checkpoint("router.pre_api_request", readonly, output)


def run_planner_checkpoint(context: Mapping[str, Any]) -> Dict[str, Any]:
    readonly = MappingProxyType(dict(context))
    output = {
        "api_call_count": readonly.get("api_call_count"),
        "approx_input_tokens": readonly.get("approx_input_tokens"),
        "request_char_count": readonly.get("request_char_count"),
        "max_tokens": readonly.get("max_tokens"),
        "decision": "observe_current_plan_budget",
        "recommendation": _planner_recommendation(readonly),
        "mutates_plan": False,
    }
    return _checkpoint("planner.pre_api_request", readonly, output)


def run_gene_selector_checkpoint(context: Mapping[str, Any]) -> Dict[str, Any]:
    readonly = MappingProxyType(dict(context))
    output = {
        "completed": readonly.get("completed"),
        "interrupted": readonly.get("interrupted"),
        "turn_exit_reason": readonly.get("turn_exit_reason"),
        "message_count": readonly.get("message_count"),
        "final_response_present": readonly.get("final_response_present"),
        "decision": "observe_completion_gene_signal",
        "recommendation": _gene_recommendation(readonly),
        "mutates_memory_or_skills": False,
    }
    return _checkpoint("gene_selector.pre_completion", readonly, output)


def _aggregate_recommendations(results: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    items = []
    has_error = False
    has_warn = False
    applied = False
    for organ, checkpoint in results.items():
        output = checkpoint.get("output") if isinstance(checkpoint, Mapping) else {}
        rec = output.get("recommendation") if isinstance(output, Mapping) else None
        control = checkpoint.get("control") if isinstance(checkpoint, Mapping) else None
        if isinstance(rec, Mapping):
            item = {"organ": organ, **dict(rec)}
            if isinstance(control, Mapping):
                item["control"] = dict(control)
                if control.get("applied"):
                    item["applied"] = True
                    applied = True
            items.append(item)
            has_error = has_error or item.get("severity") == "error"
            has_warn = has_warn or item.get("severity") in {"warn", "error"}
    return {
        "status": "REVIEW" if has_error else ("WATCH" if has_warn else "OK"),
        "mutates_runtime": False,
        "applied": applied,
        "items": items,
    }


def run_pre_api_organs(context: Mapping[str, Any]) -> Dict[str, Any]:
    before = dict(context)
    results = {
        "router": run_router_checkpoint(context),
        "planner": run_planner_checkpoint(context),
    }
    blocking = any(item.get("blocking") for item in results.values())
    return {
        "enabled": _enabled(),
        "mode": _mode(),
        "stage": "pre_api_request",
        "results": results,
        "recommendations": _aggregate_recommendations(results),
        "blocking": blocking,
        "context_unchanged": dict(context) == before,
    }


def run_pre_completion_organs(context: Mapping[str, Any]) -> Dict[str, Any]:
    before = dict(context)
    results = {"gene_selector": run_gene_selector_checkpoint(context)}
    blocking = any(item.get("blocking") for item in results.values())
    return {
        "enabled": _enabled(),
        "mode": _mode(),
        "stage": "pre_completion",
        "results": results,
        "recommendations": _aggregate_recommendations(results),
        "blocking": blocking,
        "context_unchanged": dict(context) == before,
    }


__all__ = [
    "evaluate_recommendation_control",
    "run_router_checkpoint",
    "run_planner_checkpoint",
    "run_gene_selector_checkpoint",
    "run_pre_api_organs",
    "run_pre_completion_organs",
]

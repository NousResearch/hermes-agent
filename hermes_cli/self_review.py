"""Hermes self-review and self-improvement loop.

Architecture notes
------------------
This module inspects Hermes's local telemetry and converts repeated runtime
patterns into conservative configuration recommendations:

- Loop diagnostics influence `agent.max_turns` and
  `observability.loop_window_size`, because repeated loop detections usually
  mean either the task budget is too generous for a bad retry path or the loop
  detector needs a wider sequence window.
- Slow tool analysis influences `observability.slow_tool_threshold_ms`, because
  an overly sensitive threshold can create noisy `slow_tool` events that bury
  the real regressions.
- Task completion health influences `agent.max_turns` and
  `compression.threshold`, because near-budget completions and compression
  failures reflect whether Hermes is spending enough turns and compressing
  early enough.
- Model cost efficiency influences `model_roles.classifier`, because leaving
  every role empty can route lightweight classification work through the most
  expensive model.
- Scheduler health produces `schedule.yaml` recommendations only. It does not
  touch `config.yaml`, because scheduled task enablement is operational state,
  not agent runtime policy.

The module never writes `config.yaml` unless a caller explicitly requests
application after user confirmation. Missing files are treated as empty data.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sqlite3
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from utils import atomic_json_write, atomic_text_write

try:
    import yaml as _yaml
except ModuleNotFoundError:
    _yaml = None

_HERMES_HOME = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))
HERMES_ROOT = _HERMES_HOME
AGENT_ROOT = HERMES_ROOT / "hermes-agent"
CONFIG_PATH = HERMES_ROOT / "config.yaml"
STATE_DB_PATH = HERMES_ROOT / "state.db"
STRUCTURED_LOG_PATH = HERMES_ROOT / "logs" / "structured.jsonl"
SCHEDULER_LOG_PATH = HERMES_ROOT / "logs" / "scheduler.jsonl"
SCHEDULE_PATH = HERMES_ROOT / "schedule.yaml"
REPORTS_DIR = HERMES_ROOT / "reports"
PROPOSALS_DIR = HERMES_ROOT / "proposals"
CONFIG_AUDIT_LOG_PATH = HERMES_ROOT / "logs" / "config_changes.jsonl"
DEFAULT_SLOW_TOOL_THRESHOLD_MS = 3000
DEFAULT_LOOP_WINDOW_SIZE = 8
DEFAULT_MAX_TURNS = 120
DEFAULT_COMPRESSION_THRESHOLD = 0.72
DEFAULT_APPROVAL_MODE = "yolo"
KNOWN_COSTS = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5-20251001": (0.25, 1.25),
}
DEFAULT_CONFIG = {
    "observability": {
        "slow_tool_threshold_ms": DEFAULT_SLOW_TOOL_THRESHOLD_MS,
        "loop_window_size": DEFAULT_LOOP_WINDOW_SIZE,
    },
    "model_roles": {
        "planner": "",
        "executor": "",
        "reviewer": "",
        "classifier": "",
    },
    "agent": {"max_turns": DEFAULT_MAX_TURNS},
    "compression": {"threshold": DEFAULT_COMPRESSION_THRESHOLD},
    "approvals": {"mode": DEFAULT_APPROVAL_MODE},
}
COMPLETED_STATUSES = {"completed", "complete", "success", "succeeded", "done"}
RUNNING_STATUSES = {"running", "queued", "pending", "started", "in_progress"}
FAILURE_STATUSES = {"failed", "error", "errored", "timeout", "timed_out"}

if AGENT_ROOT.exists():
    agent_root_str = str(AGENT_ROOT)
    if agent_root_str not in sys.path:
        sys.path.insert(0, agent_root_str)


def analyze(days: int = 7) -> dict:
    """Run all 5 analyses. Returns structured findings dict."""

    window_days = max(int(days or 1), 1)
    cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
    config = _load_config()
    structured_events = _read_jsonl(STRUCTURED_LOG_PATH, cutoff=cutoff)
    scheduler_events = _read_jsonl(SCHEDULER_LOG_PATH, cutoff=cutoff)
    db_tasks = _load_db_tasks(cutoff=cutoff)
    schedule_tasks = _load_schedule_tasks()

    findings = {
        "generated_at": _now_iso(),
        "window_days": window_days,
        "paths": {
            "hermes_root": str(HERMES_ROOT),
            "config": str(CONFIG_PATH),
            "state_db": str(STATE_DB_PATH),
            "structured_log": str(STRUCTURED_LOG_PATH),
            "scheduler_log": str(SCHEDULER_LOG_PATH),
            "schedule": str(SCHEDULE_PATH),
            "reports": str(REPORTS_DIR),
            "proposals": str(PROPOSALS_DIR),
        },
        "current_config": config,
        "sources": {
            "structured_events": len(structured_events),
            "scheduler_events": len(scheduler_events),
            "db_tasks": len(db_tasks),
            "schedule_tasks": len(schedule_tasks),
        },
    }
    findings["loop_patterns"] = _analyze_loop_patterns(structured_events, window_days)
    findings["slow_tools"] = _analyze_slow_tools(
        structured_events,
        config.get("observability", {}).get(
            "slow_tool_threshold_ms", DEFAULT_SLOW_TOOL_THRESHOLD_MS
        ),
    )
    findings["task_health"] = _analyze_task_health(
        structured_events,
        db_tasks,
        config.get("agent", {}).get("max_turns", DEFAULT_MAX_TURNS),
    )
    findings["model_efficiency"] = _analyze_model_efficiency(
        structured_events,
        config.get("model_roles", {}),
        window_days,
    )
    findings["scheduler_health"] = _analyze_scheduler_health(
        scheduler_events,
        schedule_tasks,
    )
    return findings


def propose(findings: dict) -> dict:
    """Convert findings into specific config change proposals."""

    config = findings.get("current_config", {})
    window_days = max(int(findings.get("window_days") or 1), 1)
    loop_patterns = findings.get("loop_patterns", {})
    slow_tools = findings.get("slow_tools", {})
    task_health = findings.get("task_health", {})
    model_efficiency = findings.get("model_efficiency", {})
    scheduler_health = findings.get("scheduler_health", {})

    proposals_by_key: dict[str, dict[str, Any]] = {}
    advisories: list[dict[str, str]] = []
    scheduler_proposals: list[dict[str, Any]] = []

    def add_proposal(
        key_path: str,
        proposed_value: Any,
        rationale: str,
        confidence: str,
    ) -> None:
        if key_path == "approvals.mode":
            return
        current_value = _get_nested(config, key_path)
        if current_value == proposed_value:
            return
        existing = proposals_by_key.get(key_path)
        candidate = {
            "key_path": key_path,
            "current_value": current_value,
            "proposed_value": proposed_value,
            "rationale": rationale,
            "confidence": confidence,
        }
        if not existing:
            proposals_by_key[key_path] = candidate
            return
        if _confidence_rank(confidence) > _confidence_rank(existing.get("confidence", "low")):
            proposals_by_key[key_path] = candidate

    def add_advisory(topic: str, finding: str, recommendation: str) -> None:
        advisories.append(
            {
                "topic": topic,
                "finding": finding,
                "recommendation": recommendation,
            }
        )

    current_max_turns = _safe_int(_get_nested(config, "agent.max_turns"), DEFAULT_MAX_TURNS)
    current_loop_window = _safe_int(
        _get_nested(config, "observability.loop_window_size"),
        DEFAULT_LOOP_WINDOW_SIZE,
    )
    current_threshold = _safe_int(
        _get_nested(config, "observability.slow_tool_threshold_ms"),
        DEFAULT_SLOW_TOOL_THRESHOLD_MS,
    )
    current_compression = _safe_float(
        _get_nested(config, "compression.threshold"),
        DEFAULT_COMPRESSION_THRESHOLD,
    )

    web_search_stats = loop_patterns.get("tools", {}).get("web_search", {})
    terminal_stats = loop_patterns.get("tools", {}).get("terminal", {})
    web_search_weekly = _safe_float(web_search_stats.get("weekly_rate"), 0.0)
    terminal_weekly = _safe_float(terminal_stats.get("weekly_rate"), 0.0)
    total_loop_weekly = _safe_float(loop_patterns.get("weekly_rate"), 0.0)

    reduce_max_turns_reason = None
    if web_search_weekly > 3.0:
        overlap = _safe_int(web_search_stats.get("tasks_with_failures"), 0)
        proposed_max_turns = max(60, current_max_turns - 20)
        reduce_max_turns_reason = (
            f"web_search looped {web_search_stats.get('total_loops', 0)} times in the last "
            f"{window_days} days ({web_search_weekly:.1f}/week), with {overlap} overlapping "
            f"failed tasks. Reducing max_turns from {current_max_turns} to "
            f"{proposed_max_turns} should cut off bad retry spirals sooner."
        )

    increase_max_turns_reason = None
    near_max_turn_rate = _safe_float(task_health.get("near_max_turn_rate"), 0.0)
    if near_max_turn_rate > 0.10:
        near_count = _safe_int(task_health.get("near_max_turn_tasks"), 0)
        total_tasks = max(_safe_int(task_health.get("total_tasks"), 0), 1)
        proposed_max_turns = int(math.ceil((current_max_turns + 20) / 10.0) * 10)
        increase_max_turns_reason = (
            f"{near_count} of {total_tasks} finished tasks ({near_max_turn_rate:.1%}) landed "
            f"within 5 API calls of max_turns={current_max_turns}. Increasing max_turns to "
            f"{proposed_max_turns} gives complex tasks more room to finish."
        )

    if reduce_max_turns_reason and increase_max_turns_reason:
        add_advisory(
            "agent.max_turns",
            f"Conflicting signals: {reduce_max_turns_reason}",
            increase_max_turns_reason + " Review task traces before changing the turn budget.",
        )
    elif reduce_max_turns_reason:
        add_proposal(
            "agent.max_turns",
            max(60, current_max_turns - 20),
            reduce_max_turns_reason,
            "medium",
        )
    elif increase_max_turns_reason:
        add_proposal(
            "agent.max_turns",
            int(math.ceil((current_max_turns + 20) / 10.0) * 10),
            increase_max_turns_reason,
            "medium",
        )

    if current_loop_window == 8 and total_loop_weekly >= 6.0:
        add_proposal(
            "observability.loop_window_size",
            12,
            f"{loop_patterns.get('total_loop_events', 0)} loop_detected events landed in the "
            f"last {window_days} days ({total_loop_weekly:.1f}/week) while loop_window_size is "
            f"still 8. Increasing to 12 should improve sequence-cycle detection before the agent "
            f"burns more turns.",
            "medium",
        )

    if terminal_weekly > 2.0:
        add_advisory(
            "terminal loops",
            f"terminal looped {terminal_stats.get('total_loops', 0)} times in the last "
            f"{window_days} days ({terminal_weekly:.1f}/week).",
            "Review terminal retry logic and tool arguments. No automatic config change is safe here.",
        )

    threshold_candidates = []
    for tool_name, stats in slow_tools.get("tools", {}).items():
        p95 = stats.get("p95_ms")
        calls = _safe_int(stats.get("calls"), 0)
        error_rate = _safe_float(stats.get("error_rate"), 0.0)
        if p95 is None or calls < 3:
            continue
        if p95 > current_threshold * 3 and error_rate <= 0.20:
            proposed_threshold = min(
                15000,
                max(current_threshold + 1000, int(math.ceil(p95 / 1000.0) * 1000)),
            )
            threshold_candidates.append((proposed_threshold, tool_name, p95, calls))
        if error_rate > 0.20:
            add_advisory(
                f"{tool_name} errors",
                f"{tool_name} failed {stats.get('failures', 0)} of {calls} calls "
                f"({error_rate:.1%}) in the last {window_days} days.",
                "Investigate tool implementation and caller inputs before changing config.",
            )

    overall_p95 = slow_tools.get("overall", {}).get("p95_ms")
    slow_event_count = _safe_int(slow_tools.get("slow_event_count"), 0)
    if threshold_candidates:
        proposed_threshold, tool_name, p95, calls = max(threshold_candidates, key=lambda item: item[0])
        add_proposal(
            "observability.slow_tool_threshold_ms",
            proposed_threshold,
            f"{tool_name} reached p95={p95:.0f}ms across {calls} calls, which is more than 3x the "
            f"current threshold ({current_threshold}ms). Raising the threshold to "
            f"{proposed_threshold}ms will reduce repetitive slow_tool noise while keeping large "
            f"outliers visible.",
            "medium",
        )
    elif overall_p95 is not None and overall_p95 < 500 and slow_event_count > 0:
        proposed_threshold = max(current_threshold, 5000)
        add_proposal(
            "observability.slow_tool_threshold_ms",
            proposed_threshold,
            f"Overall tool p95 is only {overall_p95:.0f}ms, but Hermes still emitted "
            f"{slow_event_count} slow_tool events in the last {window_days} days. Raising the "
            f"threshold from {current_threshold}ms to {proposed_threshold}ms should reduce alert noise.",
            "low",
        )

    completion_rate = _safe_float(task_health.get("completion_rate"), 0.0)
    if completion_rate < 0.70 and _safe_int(task_health.get("total_tasks"), 0) > 0:
        add_advisory(
            "task completion health",
            f"Completion rate is {completion_rate:.1%} over {task_health.get('total_tasks', 0)} "
            f"finished tasks.",
            "Review failing task traces before changing agent policy. This is a reliability issue, not a safe auto-fix.",
        )

    compression_count = 0
    for item in task_health.get("error_slugs", []):
        if item.get("slug") == "context_length_max_compression":
            compression_count = _safe_int(item.get("count"), 0)
            break
    if compression_count > 2:
        proposed_compression = _recommended_compression_threshold(compression_count)
        if proposed_compression < current_compression:
            add_proposal(
                "compression.threshold",
                proposed_compression,
                f"context_length_max_compression appeared {compression_count} times in the last "
                f"{window_days} days. Lowering compression.threshold from {current_compression} to "
                f"{proposed_compression} triggers compression earlier and should reduce hard context limits.",
                "high",
            )

    roles = model_efficiency.get("model_roles", {})
    roles_empty = model_efficiency.get("roles_empty", False)
    if model_efficiency.get("single_model_only") and roles_empty and not roles.get("classifier"):
        candidate = model_efficiency.get("classifier_candidate") or {}
        recommended_model = candidate.get("recommended_model")
        estimated_savings = _safe_float(candidate.get("estimated_monthly_savings_usd"), 0.0)
        observed_calls = _safe_int(candidate.get("observed_calls"), 0)
        if recommended_model and estimated_savings > 0:
            add_proposal(
                "model_roles.classifier",
                recommended_model,
                f"All {observed_calls} observed model calls used {candidate.get('source_model')}, and "
                f"model_roles is still empty. Assigning classifier to {recommended_model} keeps the "
                f"approval-classification path cheap; if even 10% of current token volume shifts there, "
                f"the observed workload projects about ${estimated_savings:.2f}/month in savings.",
                "low",
            )

    verbose_models = [
        f"{model} ({stats.get('avg_output_tokens', 0):.0f} avg output tokens)"
        for model, stats in model_efficiency.get("models", {}).items()
        if _safe_float(stats.get("avg_output_tokens"), 0.0) > 2000
        and _safe_int(stats.get("calls"), 0) >= 3
    ]
    if verbose_models:
        add_advisory(
            "verbose model output",
            "High average model output tokens detected: " + ", ".join(verbose_models) + ".",
            "Review prompting and response-shaping settings before changing models.",
        )

    for task_id, task_stats in scheduler_health.get("tasks", {}).items():
        failures = _safe_int(task_stats.get("failures"), 0)
        timeouts = _safe_int(task_stats.get("timeouts"), 0)
        enabled = bool(task_stats.get("enabled"))
        if enabled and (failures + timeouts) >= 3:
            scheduler_proposals.append(
                {
                    "task_id": task_id,
                    "name": task_stats.get("name") or task_id,
                    "current_enabled": True,
                    "proposed_enabled": False,
                    "rationale": f"{task_stats.get('name') or task_id} recorded {failures} failures and "
                    f"{timeouts} timeouts in the last {window_days} days. Disable it until the task prompt or "
                    f"runtime is repaired.",
                    "confidence": "medium",
                }
            )

    never_run_disabled = scheduler_health.get("never_run_disabled_tasks", [])
    if never_run_disabled:
        idle_summary = ", ".join(
            f"{item.get('name') or item.get('task_id')}: {item.get('prompt_preview')}"
            for item in never_run_disabled[:5]
        )
        add_advisory(
            "idle scheduled tasks",
            f"{len(never_run_disabled)} disabled scheduled tasks have never run.",
            f"Review whether these should stay dormant or be deleted. Examples: {idle_summary}",
        )

    timed_out_tasks = scheduler_health.get("timed_out_tasks", [])
    if timed_out_tasks:
        timeout_summary = ", ".join(
            f"{item.get('task_id')} ({item.get('count')} timeouts)"
            for item in timed_out_tasks[:5]
        )
        add_advisory(
            "scheduler timeouts",
            f"Timed out scheduled tasks detected: {timeout_summary}.",
            "Inspect the affected prompts or runtime limits before re-enabling them.",
        )

    proposal_list = sorted(proposals_by_key.values(), key=lambda item: item["key_path"])
    scheduler_proposals.sort(key=lambda item: item["task_id"])
    summary_parts = []
    if proposal_list:
        summary_parts.append(f"{len(proposal_list)} config changes")
    if scheduler_proposals:
        summary_parts.append(f"{len(scheduler_proposals)} scheduler changes")
    if advisories:
        summary_parts.append(f"{len(advisories)} advisories")
    rationale = (
        "Self-review derived from Hermes runtime telemetry over the last "
        f"{window_days} days: " + ", ".join(summary_parts or ["no changes"])
    )
    return {
        "proposed_at": _now_iso(),
        "rationale": rationale,
        "proposals": proposal_list,
        "scheduler_proposals": scheduler_proposals,
        "advisories": advisories,
        "summary": {
            "proposal_count": len(proposal_list),
            "scheduler_proposal_count": len(scheduler_proposals),
            "advisory_count": len(advisories),
        },
    }


def render_report(findings: dict, proposals: dict) -> str:
    """Render a human-readable markdown report of findings + proposals."""

    lines = [
        "# Hermes Self-Improvement Review",
        "",
        f"- Generated: {findings.get('generated_at', 'unknown')}",
        f"- Window: last {findings.get('window_days', '?')} days",
        f"- Structured events: {findings.get('sources', {}).get('structured_events', 0)}",
        f"- Scheduler events: {findings.get('sources', {}).get('scheduler_events', 0)}",
        f"- DB tasks: {findings.get('sources', {}).get('db_tasks', 0)}",
        "",
        "## Findings",
        "",
    ]

    loop_patterns = findings.get("loop_patterns", {})
    lines.extend(
        [
            "### Loop Pattern Diagnosis",
            f"- Total loop events: {loop_patterns.get('total_loop_events', 0)}",
            f"- Hard vs soft loops: {loop_patterns.get('hard_loop_events', 0)} / {loop_patterns.get('soft_loop_events', 0)}",
            f"- Tasks with both loops and failures: {loop_patterns.get('tasks_with_loops_and_failures', 0)}",
        ]
    )
    for tool_name, stats in list(loop_patterns.get("tools", {}).items())[:5]:
        lines.append(
            f"- `{tool_name}`: {stats.get('total_loops', 0)} loops, "
            f"{stats.get('hard_loops', 0)} hard, {stats.get('soft_loops', 0)} soft, "
            f"{stats.get('weekly_rate', 0):.1f}/week"
        )
    lines.append("")

    slow_tools = findings.get("slow_tools", {})
    overall_latency = slow_tools.get("overall", {})
    lines.extend(
        [
            "### Slow Tool Identification",
            f"- Current threshold: {slow_tools.get('current_threshold_ms', 0)}ms",
            f"- Overall p50/p95/p99: { _fmt_ms(overall_latency.get('p50_ms')) } / "
            f"{ _fmt_ms(overall_latency.get('p95_ms')) } / { _fmt_ms(overall_latency.get('p99_ms')) }",
            f"- `slow_tool` events: {slow_tools.get('slow_event_count', 0)}",
        ]
    )
    for tool_name, stats in list(slow_tools.get("tools", {}).items())[:5]:
        lines.append(
            f"- `{tool_name}`: p95 { _fmt_ms(stats.get('p95_ms')) }, "
            f"error rate {stats.get('error_rate', 0):.1%}, "
            f"{stats.get('calls', 0)} calls"
        )
    lines.append("")

    task_health = findings.get("task_health", {})
    lines.extend(
        [
            "### Task Completion Health",
            f"- Completion rate: {task_health.get('completion_rate', 0):.1%}",
            f"- Avg API calls (completed / failed): "
            f"{_fmt_float(task_health.get('avg_api_calls_completed'))} / "
            f"{_fmt_float(task_health.get('avg_api_calls_failed'))}",
            f"- Near max-turn tasks: {task_health.get('near_max_turn_tasks', 0)} "
            f"({task_health.get('near_max_turn_rate', 0):.1%})",
        ]
    )
    if task_health.get("error_slugs"):
        lines.append(
            "- Common errors: "
            + ", ".join(
                f"{item.get('slug')} ({item.get('count')})"
                for item in task_health.get("error_slugs", [])[:5]
            )
        )
    lines.append("")

    model_efficiency = findings.get("model_efficiency", {})
    lines.extend(
        [
            "### Model Cost Efficiency",
            f"- Roles empty: {model_efficiency.get('roles_empty', False)}",
            f"- Single-model traffic: {model_efficiency.get('single_model_only', False)}",
            f"- Estimated observed cost: ${model_efficiency.get('total_estimated_cost_usd', 0):.4f}",
        ]
    )
    for model_name, stats in model_efficiency.get("models", {}).items():
        lines.append(
            f"- `{model_name}`: {stats.get('calls', 0)} calls, "
            f"{_fmt_float(stats.get('avg_input_tokens'))} avg input tokens, "
            f"{_fmt_float(stats.get('avg_output_tokens'))} avg output tokens, "
            f"${stats.get('estimated_cost_usd', 0):.4f}"
        )
    lines.append("")

    scheduler_health = findings.get("scheduler_health", {})
    lines.extend(
        [
            "### Scheduler Health",
            f"- Enabled / disabled tasks: {scheduler_health.get('enabled_tasks', 0)} / {scheduler_health.get('disabled_tasks', 0)}",
            f"- Dispatch success rate: {_fmt_percent(scheduler_health.get('dispatch_success_rate'))}",
            f"- Never-run disabled tasks: {len(scheduler_health.get('never_run_disabled_tasks', []))}",
        ]
    )
    if scheduler_health.get("timed_out_tasks"):
        lines.append(
            "- Timed out tasks: "
            + ", ".join(
                f"{item.get('task_id')} ({item.get('count')})"
                for item in scheduler_health.get("timed_out_tasks", [])[:5]
            )
        )
    lines.extend(["", "## Proposed Changes", ""])

    if proposals.get("proposals"):
        for change in proposals["proposals"]:
            lines.append(
                f"- `{change['key_path']}`: {change['current_value']!r} -> "
                f"{change['proposed_value']!r} ({change['confidence']})"
            )
            lines.append(f"  - {change['rationale']}")
    else:
        lines.append("- No config changes proposed.")

    lines.extend(["", "## Scheduler Proposals", ""])
    if proposals.get("scheduler_proposals"):
        for change in proposals["scheduler_proposals"]:
            lines.append(
                f"- `{change['task_id']}`: enabled {change['current_enabled']} -> "
                f"{change['proposed_enabled']} ({change['confidence']})"
            )
            lines.append(f"  - {change['rationale']}")
    else:
        lines.append("- No scheduler changes proposed.")

    lines.extend(["", "## Advisories", ""])
    if proposals.get("advisories"):
        for advisory in proposals["advisories"]:
            lines.append(f"- **{advisory['topic']}**: {advisory['finding']}")
            lines.append(f"  - {advisory['recommendation']}")
    else:
        lines.append("- No advisories.")

    return "\n".join(lines)


def save_proposal(proposals: dict) -> Path:
    """Write proposals to HERMES_HOME/proposals/YYYY-MM-DD-HH-self-review.yaml."""

    PROPOSALS_DIR.mkdir(parents=True, exist_ok=True)
    proposal_path = PROPOSALS_DIR / f"{datetime.now().strftime('%Y-%m-%d-%H')}-self-review.yaml"
    payload = {
        "proposed_at": proposals.get("proposed_at", _now_iso()),
        "rationale": proposals.get("rationale", ""),
        "changes": [
            {
                "key_path": item.get("key_path"),
                "current": item.get("current_value"),
                "proposed": item.get("proposed_value"),
                "reason": item.get("rationale"),
            }
            for item in proposals.get("proposals", [])
        ],
        "scheduler_changes": [
            {
                "task_id": item.get("task_id"),
                "name": item.get("name"),
                "current_enabled": item.get("current_enabled"),
                "proposed_enabled": item.get("proposed_enabled"),
                "reason": item.get("rationale"),
            }
            for item in proposals.get("scheduler_proposals", [])
        ],
        "advisories": proposals.get("advisories", []),
    }
    atomic_text_write(proposal_path, _yaml_dump(payload))

    print(_render_stdout_summary(proposals, proposal_path))
    return proposal_path


def _write_review_artifacts(findings: dict, proposals: dict, report: str, proposal_path: Path | None = None) -> dict:
    """Persist the latest self-review as markdown + JSON dashboard artifacts."""

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    markdown_path = REPORTS_DIR / "self-review-latest.md"
    json_path = REPORTS_DIR / "self-review-latest.json"

    atomic_text_write(markdown_path, report)

    proposal_summary = {
        "config_change_count": len(proposals.get("proposals", [])),
        "scheduler_change_count": len(proposals.get("scheduler_proposals", [])),
        "advisory_count": len(proposals.get("advisories", [])),
    }
    snapshot = {
        "generated_at": findings.get("generated_at", _now_iso()),
        "window_days": findings.get("window_days"),
        "artifacts": {
            "markdown": str(markdown_path),
            "json": str(json_path),
            "proposal": str(proposal_path) if proposal_path else None,
        },
        "summary": {
            "loop_events": findings.get("loop_patterns", {}).get("total_loop_events", 0),
            "slow_tool_events": findings.get("slow_tools", {}).get("slow_event_count", 0),
            "completion_rate": findings.get("task_health", {}).get("completion_rate"),
            **proposal_summary,
        },
        "findings": findings,
        "proposals": proposals,
    }
    atomic_json_write(json_path, snapshot)

    return {"markdown_path": markdown_path, "json_path": json_path, "snapshot": snapshot}


def apply_proposals(proposals: dict, dry_run: bool = True) -> dict:
    """Apply config changes. dry_run=True (default) only shows what would change."""

    config_changes = list(proposals.get("proposals", []))
    scheduler_changes = list(proposals.get("scheduler_proposals", []))
    result = {
        "applied": [],
        "skipped": [],
        "backup_path": "",
        "schedule_backup_path": "",
    }

    if dry_run:
        for change in config_changes:
            result["skipped"].append(
                {
                    "target": "config",
                    "key_path": change.get("key_path"),
                    "current_value": change.get("current_value"),
                    "proposed_value": change.get("proposed_value"),
                    "reason": "dry_run",
                }
            )
        for change in scheduler_changes:
            result["skipped"].append(
                {
                    "target": "schedule",
                    "task_id": change.get("task_id"),
                    "current_enabled": change.get("current_enabled"),
                    "proposed_enabled": change.get("proposed_enabled"),
                    "reason": "dry_run",
                }
            )
        print(_render_apply_preview(config_changes, scheduler_changes))
        return result

    if config_changes:
        current_config = _read_yaml(CONFIG_PATH, default={})
        if not isinstance(current_config, dict):
            current_config = {}
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        backup_path = CONFIG_PATH.with_name("config.yaml.bak")
        if CONFIG_PATH.exists():
            shutil.copy2(CONFIG_PATH, backup_path)
        else:
            with backup_path.open("w", encoding="utf-8") as handle:
                handle.write(_yaml_dump(current_config))
        result["backup_path"] = str(backup_path)

        for change in config_changes:
            key_path = change.get("key_path")
            if not key_path:
                result["skipped"].append({"target": "config", "reason": "missing key_path"})
                continue
            _set_nested(current_config, key_path, change.get("proposed_value"))
            result["applied"].append(
                {
                    "target": "config",
                    "key_path": key_path,
                    "value": change.get("proposed_value"),
                }
            )

        with CONFIG_PATH.open("w", encoding="utf-8") as handle:
            handle.write(_yaml_dump(current_config))

    if scheduler_changes:
        raw_schedule = _read_yaml(SCHEDULE_PATH, default=[])
        schedule_root, schedule_tasks = _coerce_schedule_root(raw_schedule)
        schedule_backup_path = SCHEDULE_PATH.with_name("schedule.yaml.bak")
        SCHEDULE_PATH.parent.mkdir(parents=True, exist_ok=True)
        if SCHEDULE_PATH.exists():
            shutil.copy2(SCHEDULE_PATH, schedule_backup_path)
        else:
            with schedule_backup_path.open("w", encoding="utf-8") as handle:
                handle.write(_yaml_dump(schedule_root))
        result["schedule_backup_path"] = str(schedule_backup_path)

        tasks_by_id = {str(task.get("id")): task for task in schedule_tasks if task.get("id") is not None}
        for change in scheduler_changes:
            task_id = str(change.get("task_id"))
            task = tasks_by_id.get(task_id)
            if not task:
                result["skipped"].append(
                    {"target": "schedule", "task_id": task_id, "reason": "task not found"}
                )
                continue
            task["enabled"] = bool(change.get("proposed_enabled"))
            result["applied"].append(
                {
                    "target": "schedule",
                    "task_id": task_id,
                    "enabled": bool(change.get("proposed_enabled")),
                }
            )
        _write_schedule_root(SCHEDULE_PATH, schedule_root, schedule_tasks)

    if config_changes or scheduler_changes:
        _append_jsonl(
            CONFIG_AUDIT_LOG_PATH,
            {
                "ts": _now_iso(),
                "source": "self_review",
                "changes": config_changes,
                "scheduler_changes": scheduler_changes,
                "backup_path": result["backup_path"],
                "schedule_backup_path": result["schedule_backup_path"],
            },
        )
    print(_render_apply_result(result))
    return result


def run_full_review(days: int = 7, apply: bool = False, dry_run: bool = True) -> str:
    """End-to-end: analyze -> propose -> render -> save -> optionally apply."""

    findings = analyze(days=days)
    proposals = propose(findings)
    report = render_report(findings, proposals)
    proposal_path = save_proposal(proposals)
    _write_review_artifacts(findings, proposals, report, proposal_path=proposal_path)
    if apply:
        apply_proposals(proposals, dry_run=dry_run)
    return report


def _load_config() -> dict:
    raw = _read_yaml(CONFIG_PATH, default={})
    merged = deepcopy(DEFAULT_CONFIG)
    if isinstance(raw, dict):
        _deep_merge(merged, raw)
    return merged


def _analyze_loop_patterns(events: list[dict], window_days: int) -> dict:
    loop_events = [event for event in events if event.get("event") == "loop_detected"]
    failed_task_ids = {
        str(event.get("task_id"))
        for event in events
        if event.get("event") == "task_end" and _is_failed_status(event.get("status"))
    }

    tools: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "total_loops": 0,
            "hard_loops": 0,
            "soft_loops": 0,
            "loop_types": Counter(),
            "tasks": set(),
            "tasks_with_failures": set(),
        }
    )
    hard_loops = 0
    soft_loops = 0
    loop_failure_overlap: set[str] = set()
    for event in loop_events:
        tool_name = str(event.get("tool_name") or "unknown")
        severity = str(event.get("severity") or "soft").strip().lower()
        loop_type = str(event.get("loop_type") or "unknown")
        task_id = str(event.get("task_id") or "")
        stats = tools[tool_name]
        stats["total_loops"] += 1
        stats["loop_types"][loop_type] += 1
        if severity == "hard":
            stats["hard_loops"] += 1
            hard_loops += 1
        else:
            stats["soft_loops"] += 1
            soft_loops += 1
        if task_id:
            stats["tasks"].add(task_id)
            if task_id in failed_task_ids:
                stats["tasks_with_failures"].add(task_id)
                loop_failure_overlap.add(task_id)

    serialized_tools = {}
    for tool_name, stats in sorted(
        tools.items(),
        key=lambda item: item[1]["total_loops"],
        reverse=True,
    ):
        serialized_tools[tool_name] = {
            "total_loops": stats["total_loops"],
            "hard_loops": stats["hard_loops"],
            "soft_loops": stats["soft_loops"],
            "tasks_with_failures": len(stats["tasks_with_failures"]),
            "unique_tasks": len(stats["tasks"]),
            "weekly_rate": round((stats["total_loops"] / max(window_days, 1)) * 7, 2),
            "loop_types": dict(stats["loop_types"]),
        }

    top_tools = [
        {"tool_name": tool_name, **stats}
        for tool_name, stats in list(serialized_tools.items())[:5]
    ]
    return {
        "total_loop_events": len(loop_events),
        "hard_loop_events": hard_loops,
        "soft_loop_events": soft_loops,
        "weekly_rate": round((len(loop_events) / max(window_days, 1)) * 7, 2),
        "tasks_with_loops_and_failures": len(loop_failure_overlap),
        "top_tools": top_tools,
        "tools": serialized_tools,
    }


def _analyze_slow_tools(events: list[dict], current_threshold_ms: int) -> dict:
    tool_result_events = [event for event in events if event.get("event") == "tool_result"]
    slow_tool_events = [event for event in events if event.get("event") == "slow_tool"]

    per_tool: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "durations": [],
            "failures": 0,
            "calls": 0,
            "tasks": set(),
        }
    )
    all_durations = []
    for event in tool_result_events:
        tool_name = str(event.get("tool_name") or "unknown")
        duration_ms = _safe_float(event.get("duration_ms"))
        task_id = str(event.get("task_id") or "")
        stats = per_tool[tool_name]
        stats["calls"] += 1
        if task_id:
            stats["tasks"].add(task_id)
        if duration_ms is not None:
            stats["durations"].append(duration_ms)
            all_durations.append(duration_ms)
        if event.get("success") is False:
            stats["failures"] += 1

    tools = {}
    for tool_name, stats in sorted(per_tool.items()):
        durations = stats["durations"]
        calls = stats["calls"]
        unique_tasks = len(stats["tasks"])
        tools[tool_name] = {
            "calls": calls,
            "unique_tasks": unique_tasks,
            "calls_per_task": round(calls / max(unique_tasks, 1), 2),
            "failures": stats["failures"],
            "error_rate": round(stats["failures"] / max(calls, 1), 4),
            "p50_ms": _percentile(durations, 50),
            "p95_ms": _percentile(durations, 95),
            "p99_ms": _percentile(durations, 99),
        }

    return {
        "current_threshold_ms": current_threshold_ms,
        "slow_event_count": len(slow_tool_events),
        "overall": {
            "sample_count": len(all_durations),
            "p50_ms": _percentile(all_durations, 50),
            "p95_ms": _percentile(all_durations, 95),
            "p99_ms": _percentile(all_durations, 99),
        },
        "tools": dict(
            sorted(
                tools.items(),
                key=lambda item: (item[1].get("p95_ms") or 0, item[1].get("calls") or 0),
                reverse=True,
            )
        ),
    }


def _analyze_task_health(events: list[dict], db_tasks: list[dict], max_turns: int) -> dict:
    task_end_events = [event for event in events if event.get("event") == "task_end"]
    merged_tasks: dict[str, dict[str, Any]] = {}

    for task in db_tasks:
        task_id = str(task.get("task_id") or task.get("id") or "")
        if not task_id:
            continue
        merged_tasks[task_id] = {
            "task_id": task_id,
            "status": task.get("status"),
            "api_calls": task.get("api_calls"),
            "error_info": task.get("error_info"),
            "ts": task.get("ts"),
        }

    for event in task_end_events:
        task_id = str(event.get("task_id") or "")
        if not task_id:
            continue
        event_ts = _parse_ts(event.get("ts"))
        existing_ts = _parse_ts(merged_tasks.get(task_id, {}).get("ts"))
        if existing_ts and event_ts and event_ts < existing_ts:
            continue
        merged_tasks[task_id] = {
            "task_id": task_id,
            "status": event.get("status"),
            "api_calls": event.get("api_calls"),
            "error_info": event.get("error_info"),
            "ts": event.get("ts"),
        }

    finished_tasks = [
        task
        for task in merged_tasks.values()
        if _normalize_status(task.get("status")) not in RUNNING_STATUSES
    ]
    completed_tasks = [task for task in finished_tasks if _is_completed_status(task.get("status"))]
    failed_tasks = [task for task in finished_tasks if _is_failed_status(task.get("status"))]

    error_counter = Counter()
    for task in failed_tasks:
        slug = _extract_error_slug(task.get("error_info"))
        if slug:
            error_counter[slug] += 1

    completed_api_calls = [
        _safe_float(task.get("api_calls"))
        for task in completed_tasks
        if _safe_float(task.get("api_calls")) is not None
    ]
    failed_api_calls = [
        _safe_float(task.get("api_calls"))
        for task in failed_tasks
        if _safe_float(task.get("api_calls")) is not None
    ]
    near_max_turn_tasks = [
        task
        for task in finished_tasks
        if (_safe_float(task.get("api_calls")) or 0) >= max_turns - 5
    ]
    total_finished = len(finished_tasks)
    return {
        "max_turns": max_turns,
        "total_tasks": total_finished,
        "completed_tasks": len(completed_tasks),
        "failed_tasks": len(failed_tasks),
        "running_tasks": len(merged_tasks) - total_finished,
        "completion_rate": round(len(completed_tasks) / max(total_finished, 1), 4)
        if total_finished
        else 0.0,
        "error_slugs": [
            {"slug": slug, "count": count}
            for slug, count in error_counter.most_common(10)
        ],
        "avg_api_calls_completed": _average(completed_api_calls),
        "avg_api_calls_failed": _average(failed_api_calls),
        "near_max_turn_tasks": len(near_max_turn_tasks),
        "near_max_turn_rate": round(len(near_max_turn_tasks) / max(total_finished, 1), 4)
        if total_finished
        else 0.0,
    }


def _analyze_model_efficiency(
    events: list[dict],
    model_roles: dict[str, Any],
    window_days: int,
) -> dict:
    model_call_events = [event for event in events if event.get("event") == "model_call"]
    per_model: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }
    )

    for event in model_call_events:
        model_name = str(event.get("model") or "unknown")
        stats = per_model[model_name]
        stats["calls"] += 1
        stats["input_tokens"] += _safe_int(event.get("input_tokens"), 0)
        stats["output_tokens"] += _safe_int(event.get("output_tokens"), 0)

    models = {}
    total_estimated_cost = 0.0
    for model_name, stats in sorted(per_model.items()):
        input_tokens = stats["input_tokens"]
        output_tokens = stats["output_tokens"]
        cost = _estimate_cost(model_name, input_tokens, output_tokens)
        total_estimated_cost += cost
        calls = stats["calls"]
        models[model_name] = {
            "calls": calls,
            "avg_input_tokens": round(input_tokens / max(calls, 1), 2),
            "avg_output_tokens": round(output_tokens / max(calls, 1), 2),
            "estimated_cost_usd": round(cost, 6),
            "token_efficiency": round(output_tokens / max(input_tokens, 1), 4)
            if input_tokens
            else None,
            "total_input_tokens": input_tokens,
            "total_output_tokens": output_tokens,
        }

    roles_empty = not any(str(value or "").strip() for value in (model_roles or {}).values())
    single_model_only = len(models) == 1 and bool(models)
    classifier_candidate = None
    if single_model_only and roles_empty:
        source_model = next(iter(models))
        recommended_model = _recommended_classifier_model(source_model)
        if recommended_model and recommended_model != source_model:
            stats = models[source_model]
            monthly_scale = 30 / max(window_days, 1)
            assumed_fraction = 0.10
            shifted_input = stats["total_input_tokens"] * monthly_scale * assumed_fraction
            shifted_output = stats["total_output_tokens"] * monthly_scale * assumed_fraction
            classifier_candidate = {
                "source_model": source_model,
                "recommended_model": recommended_model,
                "estimated_monthly_savings_usd": round(
                    max(
                        _estimate_cost(source_model, shifted_input, shifted_output)
                        - _estimate_cost(recommended_model, shifted_input, shifted_output),
                        0.0,
                    ),
                    4,
                ),
                "observed_calls": stats["calls"],
            }

    return {
        "model_roles": model_roles or {},
        "roles_empty": roles_empty,
        "single_model_only": single_model_only,
        "models": models,
        "total_estimated_cost_usd": round(total_estimated_cost, 6),
        "classifier_candidate": classifier_candidate,
    }


def _analyze_scheduler_health(events: list[dict], schedule_tasks: list[dict]) -> dict:
    per_task: dict[str, dict[str, Any]] = {}
    for task in schedule_tasks:
        task_id = str(task.get("id") or "")
        if not task_id:
            continue
        per_task[task_id] = {
            "name": task.get("name"),
            "enabled": bool(task.get("enabled")),
            "interval_hours": task.get("interval_hours"),
            "last_run": task.get("last_run"),
            "prompt_preview": _preview(task.get("prompt")),
            "successes": 0,
            "failures": 0,
            "timeouts": 0,
        }

    for event in events:
        task_id = str(event.get("task_id") or event.get("id") or "")
        if not task_id:
            continue
        stats = per_task.setdefault(
            task_id,
            {
                "name": event.get("name") or task_id,
                "enabled": False,
                "interval_hours": None,
                "last_run": None,
                "prompt_preview": "",
                "successes": 0,
                "failures": 0,
                "timeouts": 0,
            },
        )
        outcome = _scheduler_outcome(event)
        if outcome == "success":
            stats["successes"] += 1
        elif outcome == "timeout":
            stats["timeouts"] += 1
        elif outcome == "failure":
            stats["failures"] += 1

    enabled_tasks = sum(1 for task in per_task.values() if task.get("enabled"))
    disabled_tasks = sum(1 for task in per_task.values() if not task.get("enabled"))
    terminal_events = [
        task["successes"] + task["failures"] + task["timeouts"] for task in per_task.values()
    ]
    total_attempts = sum(terminal_events)
    total_successes = sum(task["successes"] for task in per_task.values())

    never_run_disabled_tasks = [
        {
            "task_id": task_id,
            "name": task.get("name"),
            "prompt_preview": task.get("prompt_preview"),
        }
        for task_id, task in per_task.items()
        if not task.get("enabled") and task.get("last_run") in (None, "", "null")
    ]
    timed_out_tasks = [
        {"task_id": task_id, "count": task.get("timeouts", 0)}
        for task_id, task in per_task.items()
        if task.get("timeouts", 0) > 0
    ]

    serialized_tasks = {}
    for task_id, task in sorted(per_task.items()):
        attempts = task["successes"] + task["failures"] + task["timeouts"]
        serialized_tasks[task_id] = {
            **task,
            "dispatch_success_rate": round(task["successes"] / max(attempts, 1), 4)
            if attempts
            else None,
        }

    return {
        "schedule_exists": SCHEDULE_PATH.exists(),
        "enabled_tasks": enabled_tasks,
        "disabled_tasks": disabled_tasks,
        "dispatch_success_rate": round(total_successes / max(total_attempts, 1), 4)
        if total_attempts
        else None,
        "never_run_disabled_tasks": never_run_disabled_tasks,
        "timed_out_tasks": timed_out_tasks,
        "tasks": serialized_tasks,
    }


def _read_jsonl(path: Path, cutoff: datetime | None = None) -> list[dict]:
    rows = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(item, dict):
                    continue
                if cutoff is not None:
                    ts = _parse_ts(item.get("ts"))
                    if ts is None or ts < cutoff:
                        continue
                rows.append(item)
    except Exception:
        return []
    return rows


def _load_db_tasks(cutoff: datetime | None = None) -> list[dict]:
    tasks = []
    try:
        with sqlite3.connect(STATE_DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            table_exists = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='tasks'"
            ).fetchone()
            if not table_exists:
                return []
            rows = conn.execute("SELECT * FROM tasks").fetchall()
    except Exception:
        return []

    for row in rows:
        item = dict(row)
        ts = (
            item.get("updated_at")
            or item.get("ended_at")
            or item.get("completed_at")
            or item.get("created_at")
            or item.get("ts")
        )
        parsed_ts = _parse_ts(ts)
        if cutoff is not None and parsed_ts is not None and parsed_ts < cutoff:
            continue
        tasks.append(
            {
                "task_id": item.get("task_id") or item.get("id"),
                "id": item.get("id"),
                "status": item.get("status"),
                "api_calls": item.get("api_calls"),
                "error_info": item.get("error_info"),
                "ts": ts,
            }
        )
    return tasks


def _load_schedule_tasks() -> list[dict]:
    raw = _read_yaml(SCHEDULE_PATH, default=[])
    _, tasks = _coerce_schedule_root(raw)
    return tasks


def _coerce_schedule_root(raw: Any) -> tuple[Any, list[dict]]:
    if isinstance(raw, list):
        tasks = [item for item in raw if isinstance(item, dict)]
        return tasks, tasks
    if isinstance(raw, dict):
        tasks = raw.get("tasks")
        if isinstance(tasks, list):
            return raw, [item for item in tasks if isinstance(item, dict)]
    return [], []


def _write_schedule_root(path: Path, schedule_root: Any, schedule_tasks: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        if isinstance(schedule_root, dict):
            schedule_root["tasks"] = schedule_tasks
            handle.write(_yaml_dump(schedule_root))
        else:
            handle.write(_yaml_dump(schedule_tasks))


def _read_yaml(path: Path, default: Any) -> Any:
    try:
        if not path.exists():
            return default
        text = path.read_text(encoding="utf-8")
        loaded = _yaml_load(text)
        return default if loaded is None else loaded
    except Exception:
        return default


def _append_jsonl(path: Path, payload: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
    except Exception:
        return


def _yaml_load(text: str) -> Any:
    if _yaml is not None:
        return _yaml.safe_load(text)
    return _basic_yaml_load(text)


def _yaml_dump(data: Any) -> str:
    if _yaml is not None:
        return _yaml.safe_dump(data, sort_keys=False, allow_unicode=False)
    return _basic_yaml_dump(data)


def _basic_yaml_load(text: str) -> Any:
    lines = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        lines.append((indent, line.lstrip(" ")))
    if not lines:
        return None
    parsed, _ = _parse_yaml_node(lines, 0, lines[0][0])
    return parsed


def _parse_yaml_node(
    lines: list[tuple[int, str]],
    index: int,
    indent: int,
) -> tuple[Any, int]:
    if index >= len(lines):
        return None, index
    _, content = lines[index]
    if content.startswith("- "):
        return _parse_yaml_list(lines, index, indent)
    return _parse_yaml_mapping(lines, index, indent)


def _parse_yaml_list(
    lines: list[tuple[int, str]],
    index: int,
    indent: int,
) -> tuple[list[Any], int]:
    items = []
    while index < len(lines):
        line_indent, content = lines[index]
        if line_indent < indent or line_indent != indent or not content.startswith("- "):
            break
        rest = content[2:].strip()
        index += 1
        if not rest:
            if index < len(lines) and lines[index][0] > indent:
                child, index = _parse_yaml_node(lines, index, lines[index][0])
                items.append(child)
            else:
                items.append(None)
            continue
        if ":" in rest and not rest.startswith(('"', "'")):
            key, value_text = rest.split(":", 1)
            item = {}
            value_text = value_text.strip()
            if value_text:
                item[key.strip()] = _parse_yaml_scalar(value_text)
            elif index < len(lines) and lines[index][0] > indent:
                child, index = _parse_yaml_node(lines, index, lines[index][0])
                item[key.strip()] = child
            else:
                item[key.strip()] = None
            if index < len(lines) and lines[index][0] > indent:
                extra, index = _parse_yaml_mapping(lines, index, lines[index][0])
                if isinstance(extra, dict):
                    item.update(extra)
            items.append(item)
        else:
            items.append(_parse_yaml_scalar(rest))
    return items, index


def _parse_yaml_mapping(
    lines: list[tuple[int, str]],
    index: int,
    indent: int,
) -> tuple[dict[str, Any], int]:
    mapping: dict[str, Any] = {}
    while index < len(lines):
        line_indent, content = lines[index]
        if line_indent < indent or line_indent != indent or content.startswith("- "):
            break
        if ":" not in content:
            break
        key, value_text = content.split(":", 1)
        key = key.strip()
        value_text = value_text.strip()
        index += 1
        if value_text:
            mapping[key] = _parse_yaml_scalar(value_text)
        elif index < len(lines) and lines[index][0] > indent:
            child, index = _parse_yaml_node(lines, index, lines[index][0])
            mapping[key] = child
        else:
            mapping[key] = None
    return mapping, index


def _parse_yaml_scalar(value: str) -> Any:
    if value in {"null", "Null", "NULL", "~"}:
        return None
    if value in {"true", "True", "TRUE"}:
        return True
    if value in {"false", "False", "FALSE"}:
        return False
    if value.startswith('"') and value.endswith('"'):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1].replace("''", "'")
    if value.startswith("[") or value.startswith("{"):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    if value.lstrip("-").isdigit():
        return int(value)
    try:
        return float(value)
    except ValueError:
        return value


def _basic_yaml_dump(data: Any) -> str:
    return "\n".join(_yaml_dump_lines(data, 0)) + "\n"


def _yaml_dump_lines(data: Any, indent: int) -> list[str]:
    prefix = " " * indent
    if isinstance(data, dict):
        lines: list[str] = []
        for key, value in data.items():
            if isinstance(value, dict):
                if value:
                    lines.append(f"{prefix}{key}:")
                    lines.extend(_yaml_dump_lines(value, indent + 2))
                else:
                    lines.append(f"{prefix}{key}: {{}}")
            elif isinstance(value, list):
                if value:
                    lines.append(f"{prefix}{key}:")
                    lines.extend(_yaml_dump_lines(value, indent + 2))
                else:
                    lines.append(f"{prefix}{key}: []")
            else:
                lines.append(f"{prefix}{key}: {_yaml_scalar(value)}")
        return lines
    if isinstance(data, list):
        lines = []
        for item in data:
            if isinstance(item, dict):
                if not item:
                    lines.append(f"{prefix}- {{}}")
                    continue
                first = True
                for key, value in item.items():
                    key_prefix = "- " if first else "  "
                    if isinstance(value, (dict, list)):
                        lines.append(f"{prefix}{key_prefix}{key}:")
                        lines.extend(_yaml_dump_lines(value, indent + 4))
                    else:
                        lines.append(f"{prefix}{key_prefix}{key}: {_yaml_scalar(value)}")
                    first = False
            elif isinstance(item, list):
                lines.append(f"{prefix}-")
                lines.extend(_yaml_dump_lines(item, indent + 2))
            else:
                lines.append(f"{prefix}- {_yaml_scalar(item)}")
        return lines
    return [f"{prefix}{_yaml_scalar(data)}"]


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return json.dumps(str(value))


def _estimate_cost(model: str, input_tokens: float, output_tokens: float) -> float:
    rates = KNOWN_COSTS.get(model)
    if not rates:
        return 0.0
    input_rate, output_rate = rates
    return (input_tokens / 1_000_000) * input_rate + (output_tokens / 1_000_000) * output_rate


def _recommended_classifier_model(source_model: str) -> str | None:
    if source_model.startswith("claude"):
        return "claude-haiku-4-5-20251001"
    if source_model in KNOWN_COSTS:
        return "gpt-4.1-nano"
    return None


def _recommended_compression_threshold(error_count: int) -> float:
    if error_count > 10:
        steps = 3
    elif error_count > 5:
        steps = 2
    else:
        steps = 1
    return round(max(0.50, DEFAULT_COMPRESSION_THRESHOLD - (0.05 * steps)), 2)


def _scheduler_outcome(event: dict) -> str | None:
    status = _normalize_status(event.get("status") or event.get("event"))
    success = event.get("success")
    error_text = _normalize_status(event.get("error") or event.get("error_info"))
    if success is True:
        return "success"
    if success is False:
        if "timeout" in error_text or "timeout" in status:
            return "timeout"
        return "failure"
    if "timeout" in status or "timed_out" in status:
        return "timeout"
    if status in {"success", "completed", "dispatch_success", "run_success"}:
        return "success"
    if status in {"failed", "error", "dispatch_failed", "run_failed"}:
        return "failure"
    return None


def _render_stdout_summary(proposals: dict, proposal_path: Path) -> str:
    lines = [f"Saved self-review proposal to {proposal_path}"]
    config_changes = proposals.get("proposals", [])
    scheduler_changes = proposals.get("scheduler_proposals", [])
    advisories = proposals.get("advisories", [])
    if config_changes:
        lines.append("Config changes:")
        for change in config_changes:
            lines.append(
                f"  - {change['key_path']}: {change['current_value']!r} -> "
                f"{change['proposed_value']!r} because {change['rationale']}"
            )
    if scheduler_changes:
        lines.append("Scheduler changes:")
        for change in scheduler_changes:
            lines.append(
                f"  - {change['task_id']}: enabled {change['current_enabled']} -> "
                f"{change['proposed_enabled']} because {change['rationale']}"
            )
    if not config_changes and not scheduler_changes:
        lines.append("No config or scheduler changes proposed.")
    if advisories:
        lines.append(f"Advisories: {len(advisories)}")
    return "\n".join(lines)


def _render_apply_preview(config_changes: list[dict], scheduler_changes: list[dict]) -> str:
    lines = ["Dry run only. Nothing was written."]
    for change in config_changes:
        lines.append(
            f"- config {change['key_path']}: {change['current_value']!r} -> {change['proposed_value']!r}"
        )
    for change in scheduler_changes:
        lines.append(
            f"- schedule {change['task_id']}: enabled {change['current_enabled']} -> "
            f"{change['proposed_enabled']}"
        )
    if len(lines) == 1:
        lines.append("- No changes to apply.")
    return "\n".join(lines)


def _render_apply_result(result: dict) -> str:
    lines = ["Applied self-review changes."]
    if result.get("backup_path"):
        lines.append(f"- Config backup: {result['backup_path']}")
    if result.get("schedule_backup_path"):
        lines.append(f"- Schedule backup: {result['schedule_backup_path']}")
    if result.get("applied"):
        for item in result["applied"]:
            if item.get("target") == "config":
                lines.append(f"- Updated {item['key_path']} -> {item['value']!r}")
            else:
                lines.append(f"- Updated schedule task {item['task_id']} enabled={item['enabled']}")
    if result.get("skipped"):
        for item in result["skipped"]:
            lines.append(f"- Skipped {item}")
    return "\n".join(lines)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (percentile / 100.0)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return round(ordered[int(rank)], 2)
    lower_value = ordered[lower]
    upper_value = ordered[upper]
    return round(lower_value + (upper_value - lower_value) * (rank - lower), 2)


def _average(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 2)


def _extract_error_slug(error_info: Any) -> str | None:
    if error_info is None:
        return None
    if isinstance(error_info, dict):
        for key in ("slug", "code", "type", "error", "message"):
            value = error_info.get(key)
            if value:
                return _slugify(value)
        return None
    if isinstance(error_info, str):
        text = error_info.strip()
        if text.startswith("{") and text.endswith("}"):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                return _extract_error_slug(parsed)
    return _slugify(str(error_info))


def _slugify(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    chars = []
    previous_underscore = False
    for char in text:
        if char.isalnum():
            chars.append(char)
            previous_underscore = False
        else:
            if not previous_underscore:
                chars.append("_")
                previous_underscore = True
    slug = "".join(chars).strip("_")
    return slug[:80] if slug else None


def _preview(value: Any, max_length: int = 80) -> str:
    text = str(value or "").strip().replace("\n", " ")
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def _parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _is_completed_status(value: Any) -> bool:
    return _normalize_status(value) in COMPLETED_STATUSES


def _is_failed_status(value: Any) -> bool:
    status = _normalize_status(value)
    return status in FAILURE_STATUSES or (
        bool(status)
        and status not in COMPLETED_STATUSES
        and status not in RUNNING_STATUSES
    )


def _normalize_status(value: Any) -> str:
    return str(value or "").strip().lower()


def _safe_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default


def _safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _deep_merge(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _get_nested(mapping: dict, key_path: str) -> Any:
    current = mapping
    for part in key_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _set_nested(mapping: dict, key_path: str, value: Any) -> None:
    current = mapping
    parts = key_path.split(".")
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = value


def _confidence_rank(value: str) -> int:
    return {"low": 1, "medium": 2, "high": 3}.get(str(value or "").lower(), 0)


def _fmt_ms(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.0f}ms"


def _fmt_float(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}"


def _fmt_percent(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.1%}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    print(run_full_review())

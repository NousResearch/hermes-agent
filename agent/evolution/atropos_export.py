"""Atropos Export — produces batch_runner-compatible ShareGPT records with failure labels and score deltas."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.evolution.evolution_store import get_evolution_store

logger = logging.getLogger(__name__)

# Matches the output schema from batch_runner.py exactly
ATROPOS_FORMAT_VERSION = "1.0"

# All possible tool names — mirrors batch_runner.ALL_POSSIBLE_TOOLS
# Kept small here; batch_runner auto-derives from TOOL_TO_TOOLSET_MAP
_COMMON_TOOLS = {
    "terminal", "read_file", "write_file", "search_files", "patch",
    "web_search", "web_extract", "browser_navigate", "browser_snapshot",
    "skill_view", "skill_manage", "memory_search", "memory_write",
    "delegate_task", "execute_code", "send_message", "read_terminal",
    "close_terminal", "image_generate", "text_to_speech", "todo_write",
    "kanban_block", "kanban_claim", "kanban_complete", "cronjob",
}


def export_run(run_id: str) -> List[Dict[str, Any]]:
    """Export evolution run as Atropos-compatible training records.

    Each record matches the EXACT format from batch_runner.py:
    {prompt_index, conversations, completed, api_calls, tool_stats,
     tool_error_counts, toolsets_used, metadata}

    Returns one record per iteration that has trajectory data.
    """
    store = get_evolution_store()
    run = store.get_run(run_id)
    if not run:
        return []

    iterations = store.get_iterations(run_id)
    records = []

    for iteration in iterations:
        record = _build_atropos_record(run, iteration)
        if record:
            records.append(record)

    return records


def export_all_runs(
    days: int = 30,
    status_filter: Optional[List[str]] = None,
    domain_filter: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Export all runs within a time window as Atropos training records.

    Args:
        days: Look back this many days.
        status_filter: Only export runs with these statuses.
        domain_filter: Only export runs from this domain.
        output_path: If set, write JSONL to this path.

    Returns:
        List of Atropos-compatible training records.
    """
    store = get_evolution_store()
    runs = store.list_runs(limit=1000)
    cutoff = datetime.now(timezone.utc).timestamp() - (days * 86400)
    records = []

    for run in runs:
        created_str = run.get("created_at", "")
        try:
            if datetime.fromisoformat(created_str).timestamp() < cutoff:
                continue
        except (ValueError, TypeError, OSError):
            continue
        if status_filter and run["status"] not in status_filter:
            continue
        if domain_filter and run.get("task_domain") != domain_filter:
            continue
        records.extend(export_run(run["run_id"]))

    if output_path and records:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info("Exported %d training records → %s", len(records), output_path)

    return records


# ---------------------------------------------------------------------------
# Record builder — matches batch_runner.py output schema EXACTLY
# ---------------------------------------------------------------------------


def _build_atropos_record(
    run: Dict[str, Any],
    iteration: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Build a single Atropos-compatible record from an evolution iteration.

    Output schema (identical to batch_runner.py):
      - prompt_index: int (iteration number)
      - conversations: [{from: "human"|"gpt"|"tool", value: "..."}]
      - completed: bool
      - api_calls: int
      - tool_stats: {tool_name: {count, success, failure}}
      - tool_error_counts: {tool_name: failure_count}
      - toolsets_used: [str]
      - metadata: {...}  # Evolution-specific enrichment
    """
    trace_json = iteration.get("trace_json")
    if not trace_json:
        return None

    try:
        trace = json.loads(trace_json)
    except (json.JSONDecodeError, TypeError):
        return None

    # Build ShareGPT conversations from trace steps
    conversations = _build_conversations(trace, run)

    # Extract tool stats (matching _extract_tool_stats from batch_runner.py)
    tool_stats, tool_error_counts = _extract_tool_stats(conversations)

    # Build metadata with evolution-specific labels
    metadata = _build_metadata(run, iteration)

    return {
        "prompt_index": iteration.get("iteration_num", 0),
        "conversations": conversations,
        "completed": run.get("status") == "succeeded",
        "api_calls": _count_api_calls(trace),
        "tool_stats": tool_stats,
        "tool_error_counts": tool_error_counts,
        "toolsets_used": _extract_toolsets(trace),
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Conversation builder — ShareGPT from/value format
# ---------------------------------------------------------------------------


def _build_conversations(trace: Dict[str, Any], run: Dict[str, Any]) -> List[Dict[str, str]]:
    """Build ShareGPT-format conversations from a trajectory trace.

    Format: [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."},
              {"from": "tool", "value": "..."}, ...]
    """
    conversations = []

    # Human: the task description
    task_desc = run.get("task_name", "unknown task")
    conversations.append({
        "from": "human",
        "value": f"Complete the task: {task_desc}",
    })

    # Assistant + tool turns from trace steps
    for step in trace.get("steps", []):
        if step.get("type") == "model_call":
            summary = step.get("summary", "")
            tool_calls = step.get("extra", {}).get("tool_calls", [])
            value = summary
            if tool_calls:
                value += f"\n[Tool calls: {', '.join(tool_calls)}]"
            conversations.append({"from": "gpt", "value": value[:2000]})

        elif step.get("type") == "tool_execution":
            tool_name = step.get("extra", {}).get("tool", "unknown")
            status = step.get("status", "success")
            summary = step.get("summary", "")
            value = f"[{tool_name}] [{status}] {summary}"
            conversations.append({"from": "tool", "value": value[:1000]})

    return conversations


# ---------------------------------------------------------------------------
# Tool stats — matches _extract_tool_stats + _normalize_tool_stats exactly
# ---------------------------------------------------------------------------


def _extract_tool_stats(
    conversations: List[Dict[str, str]],
) -> tuple:
    """Extract tool statistics matching batch_runner._extract_tool_stats format.

    Returns:
        (tool_stats, tool_error_counts)
        tool_stats: {tool_name: {count: int, success: int, failure: int}}
        tool_error_counts: {tool_name: failure_count}
    """
    tool_stats: Dict[str, Dict[str, int]] = {}

    for msg in conversations:
        if msg["from"] == "tool":
            value = msg.get("value", "")
            # Parse tool name from value: "[tool_name] [status] summary"
            if value.startswith("[") and "]" in value:
                tool_name = value[1:].split("]")[0]
                status = "success"
                if "[error]" in value or "[failure]" in value:
                    status = "failure"

                if tool_name not in tool_stats:
                    tool_stats[tool_name] = {"count": 0, "success": 0, "failure": 0}
                tool_stats[tool_name]["count"] += 1
                if status == "success":
                    tool_stats[tool_name]["success"] += 1
                else:
                    tool_stats[tool_name]["failure"] += 1

    # Normalize: include all common tools with zero defaults
    normalized = {}
    for tool_name in sorted(_COMMON_TOOLS):
        if tool_name in tool_stats:
            normalized[tool_name] = tool_stats[tool_name].copy()
        else:
            normalized[tool_name] = {"count": 0, "success": 0, "failure": 0}
    # Also include any tools not in the common set
    for tool_name, stats in tool_stats.items():
        if tool_name not in normalized:
            normalized[tool_name] = stats.copy()

    tool_error_counts = {
        tool_name: stats["failure"]
        for tool_name, stats in tool_stats.items()
    }

    return normalized, tool_error_counts


def _extract_toolsets(trace: Dict[str, Any]) -> List[str]:
    """Extract toolset names used in this trajectory."""
    tools_used = set()
    for step in trace.get("steps", []):
        if step.get("type") == "tool_execution":
            tool_name = step.get("extra", {}).get("tool", "")
            if tool_name:
                tools_used.add(tool_name)
    return sorted(tools_used)


def _count_api_calls(trace: Dict[str, Any]) -> int:
    """Count model API calls from trace steps."""
    return sum(1 for step in trace.get("steps", []) if step.get("type") == "model_call")


def _build_metadata(run: Dict[str, Any], iteration: Dict[str, Any]) -> Dict[str, Any]:
    """Build metadata with evolution-specific training labels.

    These enrich the standard Atropos record with labels that GRPO
    can use for targeted training — failure categories, fix types,
    score deltas, and improvement confidence.
    """
    metadata = {
        "source": "evolution_engine",
        "format_version": ATROPOS_FORMAT_VERSION,
        "run_id": run.get("run_id", ""),
        "task_name": run.get("task_name", ""),
        "task_domain": run.get("task_domain", "general"),
        "task_complexity": run.get("task_complexity", 1),
        "iteration_num": iteration.get("iteration_num", 0),
        "iteration_status": iteration.get("status", ""),
        "final_score": run.get("final_score"),
    }

    # Add analysis labels if available
    analysis_json = iteration.get("analysis_json")
    if analysis_json:
        try:
            analysis = json.loads(analysis_json)
            metadata["failure_categories"] = [
                f["category"] for f in analysis.get("findings", [])
            ]
            metadata["failure_confidence"] = max(
                (f.get("confidence", 0) for f in analysis.get("findings", [])),
                default=0,
            )
        except (json.JSONDecodeError, TypeError):
            pass

    # Add improvement labels if available
    if iteration.get("improvement_action"):
        metadata["improvement_action"] = iteration["improvement_action"]
        metadata["improvement_target"] = iteration.get("improvement_target", "")

    # Add score as reward signal
    if iteration.get("score") is not None:
        metadata["score"] = iteration["score"]
        # Score delta: how much this iteration improved over baseline
        if run.get("final_score") is not None:
            metadata["score_delta"] = run["final_score"] - iteration["score"]

    return metadata


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def export_run_to_jsonl(run_id: str, output_dir: Optional[Path] = None) -> Path:
    """Export a single run and return the output file path."""
    if output_dir is None:
        from hermes_constants import get_hermes_home
        output_dir = get_hermes_home() / "evolution" / "exports"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{run_id}.jsonl"
    records = export_run(run_id)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output_path


def get_export_stats(days: int = 30) -> Dict[str, Any]:
    """Get statistics on available training data."""
    store = get_evolution_store()
    runs = store.list_runs(limit=1000)
    cutoff = datetime.now(timezone.utc).timestamp() - (days * 86400)

    stats: Dict[str, Any] = {
        "total_runs": 0,
        "succeeded": 0,
        "failed": 0,
        "exhausted": 0,
        "total_iterations": 0,
        "total_proposals": 0,
        "by_domain": {},
        "estimated_training_records": 0,
        "period_days": days,
    }

    for run in runs:
        created_str = run.get("created_at", "")
        try:
            if datetime.fromisoformat(created_str).timestamp() < cutoff:
                continue
        except (ValueError, TypeError, OSError):
            continue
        stats["total_runs"] += 1
        status = run.get("status", "unknown")
        if status in stats:
            stats[status] += 1
        domain = run.get("task_domain", "general")
        stats["by_domain"][domain] = stats["by_domain"].get(domain, 0) + 1
        iterations = store.get_iterations(run["run_id"])
        stats["total_iterations"] += len(iterations)
        for it in iterations:
            if it.get("improvement_action"):
                stats["total_proposals"] += 1
            if it.get("trace_json"):
                stats["estimated_training_records"] += 1

    return stats

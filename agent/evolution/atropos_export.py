"""Atropos-compatible training data export from Evolution Engine runs.

Every evolution iteration contains structured, labeled data that directly
feeds the Atropos RL flywheel:

  - Failed trajectory → negative example with labeled failure mode
  - Applied fix → positive example showing the correction
  - Score delta → reward signal for GRPO advantage calculation
  - Failure category → training label for error-type classification

Output format matches what Atropos environments produce (ShareGPT-compatible
conversations, tool_stats, reasoning_stats, reward fields).

This is the flywheel: more HAEE usage → more labeled training data →
better Hermes models → more capable agents → more HAEE usage.

Usage:
    from agent.evolution.atropos_export import export_run, export_all_runs
    records = export_run(run_id)          # One run → training records
    records = export_all_runs(days=30)    # All recent runs → training records
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.evolution.evolution_store import get_evolution_store
from agent.evolution.task_definition import TaskStatus

logger = logging.getLogger(__name__)

# Atropos training record format version
ATROPOS_FORMAT_VERSION = "1.0"


def export_run(run_id: str) -> List[Dict[str, Any]]:
    """Export a single evolution run as Atropos-compatible training records.

    Each record is a ShareGPT-format conversation with additional training
    metadata fields that Atropos GRPO expects.

    Returns a list of records — one per evolution iteration that contains
    actionable improvement data.
    """
    store = get_evolution_store()
    run = store.get_run(run_id)
    if not run:
        logger.warning("Run not found: %s", run_id)
        return []

    iterations = store.get_iterations(run_id)
    if not iterations:
        return []

    records = []
    task_name = run["task_name"]
    task_domain = run.get("task_domain", "general")
    task_complexity = run.get("task_complexity", 1)

    for i, iteration in enumerate(iterations):
        # Build records from each iteration that has improvement data
        records_for_iter = _build_iteration_records(
            iteration, task_name, task_domain, task_complexity,
            is_first=(i == 0), is_last=(i == len(iterations) - 1),
            final_score=run.get("final_score"),
        )
        records.extend(records_for_iter)

    return records


def export_all_runs(
    days: int = 30,
    status_filter: Optional[List[str]] = None,
    domain_filter: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Export all evolution runs within a time window as training records.

    Args:
        days: Look back this many days (default 30).
        status_filter: Only export runs with these statuses.
        domain_filter: Only export runs from this task domain.
        output_path: If set, write JSONL to this path.

    Returns:
        List of training records in Atropos-compatible format.
    """
    store = get_evolution_store()
    runs = store.list_runs(limit=1000)

    # Filter by recency
    cutoff = datetime.now(timezone.utc).timestamp() - (days * 86400)
    records = []

    for run in runs:
        # Parse creation time
        created_str = run.get("created_at", "")
        try:
            created_dt = datetime.fromisoformat(created_str)
            if created_dt.timestamp() < cutoff:
                continue
        except (ValueError, TypeError, OSError):
            continue

        # Apply filters
        if status_filter and run["status"] not in status_filter:
            continue
        if domain_filter and run.get("task_domain") != domain_filter:
            continue

        # Export this run
        run_records = export_run(run["run_id"])
        records.extend(run_records)

    # Write if output path provided
    if output_path and records:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, default=str) + "\n")
        logger.info(
            "Exported %d training records to %s", len(records), output_path
        )

    return records


def export_benchmark_dataset(
    task_names: List[str],
    output_path: Path,
) -> Dict[str, Any]:
    """Export a dataset suitable for Atropos benchmark evaluation.

    Collects all runs for the given tasks, extracts scored trajectories,
    and produces a dataset manifest that Atropos can load for model eval.

    Args:
        task_names: List of task names to include.
        output_path: Where to write the dataset (JSONL).

    Returns:
        Dataset manifest with counts and summary statistics.
    """
    store = get_evolution_store()
    all_records = []

    for task_name in task_names:
        runs = store.list_runs(task_name=task_name, limit=100)
        for run in runs:
            records = export_run(run["run_id"])
            # Tag each record with its source task for dataset grouping
            for r in records:
                r["_source_task"] = task_name
            all_records.extend(records)

    # Write dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, default=str) + "\n")

    # Compute manifest stats
    by_status = {}
    by_domain = {}
    for r in all_records:
        status = r.get("status", "unknown")
        by_status[status] = by_status.get(status, 0) + 1
        domain = r.get("task_domain", "general")
        by_domain[domain] = by_domain.get(domain, 0) + 1

    manifest = {
        "format_version": ATROPOS_FORMAT_VERSION,
        "total_records": len(all_records),
        "tasks_included": len(task_names),
        "by_status": by_status,
        "by_domain": by_domain,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "output_path": str(output_path),
    }

    manifest_path = output_path.parent / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    logger.info(
        "Dataset exported: %d records from %d tasks → %s",
        len(all_records), len(task_names), output_path,
    )
    return manifest


# ---------------------------------------------------------------------------
# Record builders
# ---------------------------------------------------------------------------


def _build_iteration_records(
    iteration: Dict[str, Any],
    task_name: str,
    task_domain: str,
    task_complexity: int,
    is_first: bool,
    is_last: bool,
    final_score: Optional[float],
) -> List[Dict[str, Any]]:
    """Build training records from a single evolution iteration."""
    records = []
    base_record = {
        "format": "sharegpt",  # Atropos-compatible
        "format_version": ATROPOS_FORMAT_VERSION,
        "task_name": task_name,
        "task_domain": task_domain,
        "task_complexity": task_complexity,
        "iteration_num": iteration.get("iteration_num", 0),
        "iteration_status": iteration.get("status", "unknown"),
        "run_id": iteration.get("run_id", ""),
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }

    # Record 1: The trajectory (what the agent did)
    trace_json = iteration.get("trace_json")
    if trace_json:
        try:
            trace = json.loads(trace_json)
            record = dict(base_record)
            record["record_type"] = "trajectory"
            record["trajectory"] = trace
            record["score"] = iteration.get("score")
            _add_tool_stats(record, trace)
            records.append(record)
        except (json.JSONDecodeError, TypeError):
            pass

    # Record 2: The failure analysis (why it failed)
    analysis_json = iteration.get("analysis_json")
    if analysis_json:
        try:
            analysis = json.loads(analysis_json)
            record = dict(base_record)
            record["record_type"] = "failure_analysis"
            record["analysis"] = analysis
            # Extract failure categories as labels
            findings = analysis.get("findings", [])
            record["failure_categories"] = [f["category"] for f in findings]
            record["failure_confidence"] = max(
                (f.get("confidence", 0) for f in findings), default=0
            )
            # Extract implicated components for targeted training
            record["implicated_tools"] = list(set(
                tool for f in findings for tool in f.get("implicated_tools", [])
            ))
            records.append(record)
        except (json.JSONDecodeError, TypeError):
            pass

    # Record 3: The improvement proposal (how it was fixed)
    proposal_json = iteration.get("proposal_json")
    if proposal_json:
        try:
            proposal = json.loads(proposal_json)
            record = dict(base_record)
            record["record_type"] = "improvement"
            record["improvement_action"] = proposal.get("action_type", "")
            record["improvement_target"] = proposal.get("target", "")
            record["improvement_description"] = proposal.get("description", "")
            record["improvement_confidence"] = proposal.get("confidence", 0)
            record["is_destructive"] = proposal.get("is_destructive", False)
            # The actual fix content — valuable for supervised fine-tuning
            if proposal.get("content"):
                record["fix_content"] = proposal["content"][:5000]  # Truncate for training
            records.append(record)
        except (json.JSONDecodeError, TypeError):
            pass

    # Record 4: Composite — everything together (best for GRPO groups)
    if records and not is_first:
        composite = dict(base_record)
        composite["record_type"] = "composite"
        composite["score"] = iteration.get("score")
        if final_score is not None:
            composite["final_score"] = final_score
        composite["status"] = iteration.get("status", "")
        composite["improvement_applied"] = bool(iteration.get("improvement_action"))
        # Score delta from previous iteration for reward signal
        composite["score_delta"] = _compute_score_delta(iteration)
        records.append(composite)

    return records


def _add_tool_stats(record: Dict[str, Any], trace: Dict[str, Any]) -> None:
    """Add tool usage statistics from a trajectory, matching Atropos format."""
    tool_counts: Dict[str, Dict[str, int]] = {}
    total_success = 0
    total_failure = 0

    for step in trace.get("steps", []):
        if step.get("type") == "tool_execution":
            tool_name = step.get("extra", {}).get("tool", "unknown")
            if tool_name not in tool_counts:
                tool_counts[tool_name] = {"count": 0, "success": 0, "failure": 0}
            tool_counts[tool_name]["count"] += 1
            if step.get("status") == "success":
                tool_counts[tool_name]["success"] += 1
                total_success += 1
            else:
                tool_counts[tool_name]["failure"] += 1
                total_failure += 1

    record["tool_stats"] = tool_counts
    record["tool_success_total"] = total_success
    record["tool_failure_total"] = total_failure
    record["total_turns"] = trace.get("total_turns", 0)
    record["total_tokens"] = trace.get("total_tokens", 0)


def _compute_score_delta(iteration: Dict[str, Any]) -> Optional[float]:
    """Compute score improvement from this iteration's fix.

    A positive delta means the fix improved performance.
    This is the reward signal for GRPO advantage calculation.
    """
    score = iteration.get("score")
    if score is None:
        return None

    # The score_delta is positive when the fix worked
    # For now, use the improvement confidence as a proxy
    # In a full implementation, we'd compare with the previous iteration
    proposal_json = iteration.get("proposal_json")
    if proposal_json:
        try:
            proposal = json.loads(proposal_json)
            confidence = proposal.get("confidence", 0)
            # Higher confidence fix + higher score = better training example
            return score * confidence
        except (json.JSONDecodeError, TypeError):
            pass
    return score


# ---------------------------------------------------------------------------
# CLI integration helpers
# ---------------------------------------------------------------------------


def export_run_to_jsonl(run_id: str, output_dir: Optional[Path] = None) -> Path:
    """Export a run and return the output file path. For CLI use."""
    if output_dir is None:
        from hermes_constants import get_hermes_home
        output_dir = get_hermes_home() / "evolution" / "exports"

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{run_id}.jsonl"

    records = export_run(run_id)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, default=str) + "\n")

    return output_path


def get_export_stats(days: int = 30) -> Dict[str, Any]:
    """Get statistics on available training data. For CLI use."""
    store = get_evolution_store()
    runs = store.list_runs(limit=1000)

    cutoff = datetime.now(timezone.utc).timestamp() - (days * 86400)
    stats = {
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
            created_dt = datetime.fromisoformat(created_str)
            if created_dt.timestamp() < cutoff:
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
            # Each iteration with data produces ~3 training records
            if it.get("trace_json") or it.get("analysis_json"):
                stats["estimated_training_records"] += 3

    return stats

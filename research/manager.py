"""Decision helpers for autonomous research loops."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from research.loop_policy import diagnose_run_failure
from research.models import ResearchProject
from research.tinker_client import estimate_run_cost

_LOWER_IS_BETTER_MARKERS = (
    "loss",
    "perplexity",
    "ppl",
    "wer",
    "error",
    "ece",
    "brier",
    "mae",
    "mse",
    "rmse",
)

_STOPWORDS = {
    "with",
    "from",
    "that",
    "this",
    "into",
    "using",
    "llm",
    "llms",
    "model",
    "models",
    "agent",
    "agents",
    "research",
    "train",
    "training",
    "evaluation",
}


def _tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]{3,}", (text or "").lower())
        if token not in _STOPWORDS
    ]


def metric_direction(metric_name: str) -> int:
    """Return +1 when higher is better and -1 when lower is better."""
    lowered = (metric_name or "").lower()
    if any(marker in lowered for marker in _LOWER_IS_BETTER_MARKERS):
        return -1
    return 1


def aggregate_eval_score(metrics: dict[str, Any], preferred_metrics: list[str] | None = None) -> tuple[float, dict[str, float]]:
    """Aggregate numeric metrics into a scalar score for ranking."""
    numeric = {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float))
    }
    if preferred_metrics:
        numeric = {key: value for key, value in numeric.items() if key in preferred_metrics}
    if not numeric:
        return 0.0, {}

    adjusted = {
        key: value * metric_direction(key)
        for key, value in numeric.items()
    }
    score = sum(adjusted.values()) / len(adjusted)
    return round(score, 6), adjusted


def triage_literature(
    records: list[dict[str, Any]],
    objective: str = "",
    top_k: int = 10,
) -> dict[str, Any]:
    """Score literature for relevance, recency, and novelty."""
    objective_terms = set(_tokenize(objective))
    current_year = 2026
    ranked: list[dict[str, Any]] = []
    keyword_counter: Counter[str] = Counter()

    for record in records:
        title = str(record.get("title") or "")
        abstract = str(record.get("abstract") or record.get("summary") or "")
        joined = f"{title} {abstract} {' '.join(record.get('tags') or [])}".lower()
        tokens = _tokenize(joined)
        keyword_counter.update(tokens)
        overlap = len(objective_terms.intersection(tokens))
        relevance = overlap / max(len(objective_terms), 1) if objective_terms else min(len(tokens), 10) / 10.0

        year = record.get("year")
        citation_count = int(record.get("citation_count") or 0)
        recency = 0.0
        if isinstance(year, int):
            recency = max(0.0, 1.0 - max(current_year - year, 0) / 10.0)
        citation_score = min(math.log1p(citation_count) / 6.0, 1.0)

        frontier_score = (0.6 * relevance) + (0.3 * recency) + (0.1 * (1.0 - citation_score))
        baseline_score = (0.55 * relevance) + (0.45 * citation_score)
        if frontier_score >= 0.55:
            label = "frontier"
            score = frontier_score
        elif baseline_score >= 0.45:
            label = "baseline"
            score = baseline_score
        else:
            label = "related"
            score = max(frontier_score, baseline_score)

        ranked.append(
            {
                **record,
                "score": round(score, 4),
                "label": label,
                "relevance_score": round(relevance, 4),
                "recency_score": round(recency, 4),
                "citation_score": round(citation_score, 4),
            }
        )

    ranked.sort(key=lambda item: (item.get("score", 0), item.get("year") or 0, item.get("citation_count") or 0), reverse=True)
    gap_candidates = [
        paper for paper in ranked
        if paper.get("label") == "frontier" and int(paper.get("citation_count") or 0) < 75
    ][: min(top_k, 5)]
    focus_areas = [word for word, _count in keyword_counter.most_common(5)]

    return {
        "ranked_papers": ranked[:top_k],
        "gap_candidates": gap_candidates,
        "focus_areas": focus_areas,
        "paper_count": len(records),
    }


def assess_dataset_rows(
    rows: list[dict[str, Any]],
    method: str = "sft",
    field_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Compute a provenance/quality summary for a dataset."""
    field_map = dict(field_map or {})
    required_fields: dict[str, list[str]] = {
        "sft": ["prompt", "completion"],
        "dpo": ["prompt", "chosen", "rejected"],
        "ppo": ["prompt", "completion", "logprobs"],
    }
    required = required_fields.get(method, ["prompt"])
    required = [field_map.get(field, field) for field in required]

    duplicate_rows = 0
    seen: set[str] = set()
    schema_valid = 0
    prompt_lengths: list[int] = []
    response_lengths: list[int] = []
    reward_rows = 0

    for row in rows:
        key = json.dumps(row, sort_keys=True, ensure_ascii=False)
        if key in seen:
            duplicate_rows += 1
        else:
            seen.add(key)

        valid = True
        for field in required:
            value = row.get(field)
            if value in ("", None, [], {}):
                valid = False
                break
        if valid:
            schema_valid += 1

        prompt_value = row.get(field_map.get("prompt", "prompt"))
        if isinstance(prompt_value, str):
            prompt_lengths.append(len(prompt_value))

        response_field = field_map.get("completion", "completion")
        response_value = row.get(response_field) or row.get(field_map.get("chosen", "chosen"))
        if isinstance(response_value, str):
            response_lengths.append(len(response_value))

        if "reward" in row or field_map.get("reward") in row:
            reward_rows += 1

    num_rows = len(rows)
    schema_valid_ratio = schema_valid / num_rows if num_rows else 0.0
    duplicate_ratio = duplicate_rows / num_rows if num_rows else 0.0
    avg_prompt_chars = sum(prompt_lengths) / len(prompt_lengths) if prompt_lengths else 0.0
    avg_response_chars = sum(response_lengths) / len(response_lengths) if response_lengths else 0.0
    reward_coverage = reward_rows / num_rows if num_rows else 0.0

    quality_score = (
        (schema_valid_ratio * 60.0)
        + ((1.0 - duplicate_ratio) * 20.0)
        + (min(avg_prompt_chars / 200.0, 1.0) * 10.0)
        + (min(avg_response_chars / 400.0, 1.0) * 10.0)
    )
    if method == "ppo":
        quality_score += reward_coverage * 10.0

    return {
        "num_rows": num_rows,
        "required_fields": required,
        "schema_valid_rows": schema_valid,
        "schema_valid_ratio": round(schema_valid_ratio, 4),
        "duplicate_rows": duplicate_rows,
        "duplicate_ratio": round(duplicate_ratio, 4),
        "avg_prompt_chars": round(avg_prompt_chars, 2),
        "avg_response_chars": round(avg_response_chars, 2),
        "reward_coverage": round(reward_coverage, 4),
        "quality_score": round(min(quality_score, 100.0), 2),
    }


def rank_runs(
    runs: list[dict[str, Any]],
    eval_results: list[dict[str, Any]],
    preferred_metrics: list[str] | None = None,
    baseline_run_id: str | None = None,
) -> dict[str, Any]:
    """Rank runs by aggregated evaluation score and flag regressions."""
    evals_by_run: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eval_results:
        run_id = row.get("run_id")
        if run_id:
            evals_by_run[str(run_id)].append(row)

    completed_runs = [run for run in runs if run.get("status") == "completed"]
    sorted_completed = sorted(completed_runs, key=lambda item: item.get("created_at", ""))
    if not baseline_run_id and sorted_completed:
        baseline_run_id = sorted_completed[0].get("run_id")

    baseline_metrics: dict[str, float] = {}
    if baseline_run_id:
        for row in evals_by_run.get(baseline_run_id, []):
            for key, value in row.get("metrics", {}).items():
                if isinstance(value, (int, float)):
                    baseline_metrics[key] = float(value)

    rankings: list[dict[str, Any]] = []
    for run in runs:
        run_id = str(run.get("run_id"))
        metrics: dict[str, float] = {}
        eval_count = 0
        for row in evals_by_run.get(run_id, []):
            eval_count += 1
            for key, value in row.get("metrics", {}).items():
                if isinstance(value, (int, float)):
                    metrics[key] = float(value)
        score, adjusted = aggregate_eval_score(metrics, preferred_metrics)
        regressions = []
        for key, base_value in baseline_metrics.items():
            if key not in metrics:
                continue
            direction = metric_direction(key)
            delta = (metrics[key] - base_value) * direction
            if delta < 0:
                regressions.append({"metric": key, "baseline": base_value, "candidate": metrics[key], "delta": round(delta, 6)})
        rankings.append(
            {
                "run_id": run_id,
                "experiment_id": run.get("experiment_id"),
                "status": run.get("status", "unknown"),
                "score": score,
                "adjusted_metrics": adjusted,
                "raw_metrics": metrics,
                "eval_count": eval_count,
                "regressions": regressions,
                "is_baseline": run_id == baseline_run_id,
            }
        )

    rankings.sort(key=lambda item: (item["score"], item["status"] == "completed", item["eval_count"]), reverse=True)
    best = rankings[0] if rankings else None
    return {
        "baseline_run_id": baseline_run_id,
        "best_run": best,
        "rankings": rankings,
    }


def suggest_next_step(
    project: ResearchProject,
    status_summary: dict[str, Any],
    runs: list[dict[str, Any]],
    eval_results: list[dict[str, Any]],
    datasets: list[dict[str, Any]],
    literature_summary: dict[str, Any] | None = None,
    rankings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Choose the next best action for the research loop."""
    rankings = rankings or {}
    literature_summary = literature_summary or {}
    loop_state = status_summary.get("loop_state") or {}
    next_steps: list[str] = []
    reasons: list[str] = []

    active_runs = status_summary.get("active_run_ids") or []
    if active_runs:
        reasons.append("There are active training runs that still need monitoring.")
        return {
            "next_action": "monitor_active_runs",
            "summary": "Keep monitoring the active runs before launching more work.",
            "reasons": reasons,
            "next_steps": [f"Monitor run `{run_id}`." for run_id in active_runs],
        }

    if not status_summary.get("literature_notes", 0):
        reasons.append("No literature has been recorded yet.")
        return {
            "next_action": "triage_literature",
            "summary": "Collect and score literature before committing to the next experiment.",
            "reasons": reasons,
            "next_steps": ["Search ArXiv and Semantic Scholar for high-signal papers.", "Record the top papers into project state."],
        }

    if project.spec_level == "zero_spec" and not status_summary.get("hypotheses", 0):
        reasons.append("Zero-spec project still needs a concrete hypothesis.")
        return {
            "next_action": "form_hypothesis",
            "summary": "Convert the strongest literature-backed idea into a written hypothesis.",
            "reasons": reasons,
            "next_steps": ["Choose the highest-priority idea.", "Record success metric and stop condition."],
        }

    if not status_summary.get("experiments", 0):
        reasons.append("No experiment plan exists yet.")
        return {
            "next_action": "plan_experiment",
            "summary": "Write the first executable experiment plan before launching training.",
            "reasons": reasons,
            "next_steps": ["Choose method and dataset.", "Define evaluation plan and baseline."],
        }

    completed_runs = [run for run in runs if run.get("status") == "completed"]
    failed_runs = [run for run in runs if run.get("status") == "failed"]
    if failed_runs:
        reasons.append("There are failed runs that need diagnosis or recovery.")
        next_steps = [f"Diagnose failed run `{run.get('run_id')}`." for run in failed_runs[:3]]
        return {
            "next_action": "recover_failed_runs",
            "summary": "Diagnose the failed runs and queue patched retries where appropriate.",
            "reasons": reasons,
            "next_steps": next_steps,
        }

    if completed_runs and not eval_results:
        reasons.append("Completed runs exist but no evaluations have been recorded.")
        return {
            "next_action": "evaluate_completed_runs",
            "summary": "Run automated evaluations and qualitative checks before deciding what to do next.",
            "reasons": reasons,
            "next_steps": [f"Evaluate completed run `{run.get('run_id')}`." for run in completed_runs[:3]],
        }

    if loop_state.get("hyperparam_queue"):
        reasons.append("Queued hyperparameter candidates are available for launch.")
        return {
            "next_action": "launch_next_search_run",
            "summary": "Continue the queued search and compare candidates against the current best run.",
            "reasons": reasons,
            "next_steps": ["Start the next queued search config.", "Prune duplicates before launching more runs."],
        }

    best_run = (rankings or {}).get("best_run")
    if best_run and best_run.get("regressions"):
        reasons.append("The current leading run still regresses on at least one baseline metric.")
        return {
            "next_action": "plan_ablation",
            "summary": "The best-scoring run has regressions; run an ablation or mitigation pass before declaring success.",
            "reasons": reasons,
            "next_steps": ["Inspect regression metrics.", "Design a focused ablation to isolate the cause."],
        }

    if datasets:
        low_quality = [row for row in datasets if float(row.get("quality_report", {}).get("quality_score", 100.0)) < 60.0]
        if low_quality:
            reasons.append("At least one dataset quality report is below the quality threshold.")
            return {
                "next_action": "improve_dataset_quality",
                "summary": "Tighten dataset quality before more training iterations.",
                "reasons": reasons,
                "next_steps": [f"Curate dataset `{row.get('name') or row.get('dataset_id')}`." for row in low_quality[:3]],
            }

    focus_areas = literature_summary.get("focus_areas") or []
    if best_run:
        reasons.append("The project has a ranked best run and enough evidence for a decision memo.")
        if focus_areas:
            reasons.append(f"Literature focus areas: {', '.join(focus_areas[:3])}.")
        return {
            "next_action": "write_research_memo",
            "summary": "Summarize the current evidence and recommend the next experiment or stopping decision.",
            "reasons": reasons,
            "next_steps": ["Write a research memo.", "Capture the best run, regressions, and recommended follow-up."],
        }

    reasons.append("The project has enough setup to continue iterating.")
    return {
        "next_action": "continue_loop",
        "summary": "Continue the autonomous loop with the next most promising experiment.",
        "reasons": reasons,
        "next_steps": next_steps or ["Review current state and continue the loop."],
    }


def prune_search_queue(
    queue: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    max_queue: int = 10,
) -> dict[str, Any]:
    """Prune search configs by removing duplicates and capping queue length."""
    normalized_seen: set[str] = set()
    completed_configs = {
        json.dumps(
            {
                key: value
                for key, value in run.items()
                if key in {"method", "base_model", "dataset_path", "batch_size", "learning_rate", "lora_rank", "num_epochs", "max_steps", "beta"}
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        for run in runs
    }

    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for config in queue:
        normalized = json.dumps(
            {
                key: value
                for key, value in config.items()
                if not key.startswith("_")
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        if normalized in normalized_seen or normalized in completed_configs:
            dropped.append({**config, "_drop_reason": "duplicate"})
            continue
        normalized_seen.add(normalized)
        kept.append(config)

    kept.sort(key=lambda item: float(item.get("estimated_cost_usd") or estimate_run_cost(item)))
    if len(kept) > max_queue:
        dropped.extend([{**item, "_drop_reason": "queue_limit"} for item in kept[max_queue:]])
        kept = kept[:max_queue]

    return {"kept": kept, "dropped": dropped}


def build_recovery_plan(run: dict[str, Any], runtime_status: dict[str, Any]) -> dict[str, Any]:
    """Produce a retry patch and action recommendation for a failed run."""
    diagnosis = diagnose_run_failure(run, runtime_status)
    error_type = diagnosis.get("error_type", "unknown")
    patch: dict[str, Any] = {}
    next_steps: list[str] = []
    recommended_action = "inspect_logs"

    if error_type == "OOM":
        patch["batch_size"] = max(int(run.get("batch_size", 32)) // 2, 1)
        if int(run.get("lora_rank", 32)) > 8:
            patch["lora_rank"] = max(int(run.get("lora_rank", 32)) // 2, 8)
        recommended_action = "retry_with_smaller_footprint"
        next_steps = ["Reduce batch size.", "Reduce LoRA rank if memory is still tight."]
    elif error_type == "NaN_loss":
        patch["learning_rate"] = float(run.get("learning_rate", 1e-4)) / 10.0
        optimizer = dict(run.get("optimizer") or {})
        optimizer.setdefault("clip_grad_norm", 1.0)
        patch["optimizer"] = optimizer
        recommended_action = "retry_with_safer_optimizer"
        next_steps = ["Lower learning rate.", "Enable gradient clipping.", "Validate dataset examples."]
    elif error_type == "dataset_error":
        recommended_action = "validate_dataset"
        next_steps = ["Validate the dataset schema.", "Fix missing paths or malformed rows."]
    elif error_type == "timeout":
        patch["save_every"] = max(min(int(run.get("save_every", 50)), 25), 5)
        if runtime_status.get("last_checkpoint"):
            patch["resume_from"] = runtime_status["last_checkpoint"]
        recommended_action = "resume_from_checkpoint"
        next_steps = ["Resume from the last checkpoint.", "Increase checkpoint frequency."]
    elif error_type == "connection_error":
        if runtime_status.get("last_checkpoint"):
            patch["resume_from"] = runtime_status["last_checkpoint"]
        recommended_action = "resume_after_service_check"
        next_steps = ["Check Tinker connectivity.", "Resume from the last checkpoint if available."]
    else:
        next_steps = list(diagnosis.get("recovery_options") or [])

    return {
        "run_id": run.get("run_id"),
        "diagnosis": diagnosis,
        "config_patch": patch,
        "recommended_action": recommended_action,
        "summary": diagnosis.get("message", ""),
        "next_steps": next_steps,
    }


def queue_recovery_retry(
    loop_state: dict[str, Any],
    run: dict[str, Any],
    recovery_plan: dict[str, Any],
) -> dict[str, Any]:
    """Append a patched retry config to the search queue."""
    patched = {
        key: value
        for key, value in run.items()
        if key in {"method", "base_model", "dataset_path", "field_map", "batch_size", "learning_rate", "lora_rank", "num_epochs", "max_steps", "save_every", "reference_model_name", "beta", "sampling_params", "optimizer", "metadata", "experiment_id"}
    }
    patched.update(recovery_plan.get("config_patch") or {})
    metadata = dict(patched.get("metadata") or {})
    metadata["recovery_for"] = run.get("run_id")
    metadata["recovery_action"] = recovery_plan.get("recommended_action")
    patched["metadata"] = metadata

    queue = list(loop_state.get("hyperparam_queue") or [])
    queue.append(patched)
    loop_state["hyperparam_queue"] = queue
    return loop_state


def link_lineage_edges(
    graph,
    paper_ids: list[str] | None = None,
    idea_id: str | None = None,
    hypothesis_id: str | None = None,
    experiment_id: str | None = None,
    run_id: str | None = None,
    eval_id: str | None = None,
) -> list[dict[str, Any]]:
    """Create graph edges spanning the research lineage."""
    edges: list[dict[str, Any]] = []
    for paper_id in paper_ids or []:
        if idea_id:
            edges.append(graph.add_edge("paper", paper_id, "idea", idea_id, "inspires"))
        if hypothesis_id:
            edges.append(graph.add_edge("paper", paper_id, "hypothesis", hypothesis_id, "supports"))
        if experiment_id:
            edges.append(graph.add_edge("paper", paper_id, "experiment", experiment_id, "motivates"))
    if idea_id and hypothesis_id:
        edges.append(graph.add_edge("idea", idea_id, "hypothesis", hypothesis_id, "motivates"))
    if hypothesis_id and experiment_id:
        edges.append(graph.add_edge("hypothesis", hypothesis_id, "experiment", experiment_id, "tests"))
    if experiment_id and run_id:
        edges.append(graph.add_edge("experiment", experiment_id, "run", run_id, "launched"))
    if run_id and eval_id:
        edges.append(graph.add_edge("run", run_id, "eval", eval_id, "evaluated_by"))
    if experiment_id and eval_id:
        edges.append(graph.add_edge("experiment", experiment_id, "eval", eval_id, "measured_by"))
    return edges

"""Loop policy helpers for autonomous research execution."""

from __future__ import annotations

import itertools
import json
import random
from typing import Any

from research.models import ExperimentSpec, Hypothesis, ResearchIdea, ResearchProject, RunSpec, make_id

_FULL_SPEC_MARKERS = {
    "dataset",
    "benchmark",
    "model",
    "base model",
    "lora",
    "learning rate",
    "batch size",
    "epochs",
    "eval",
    "evaluation",
    "ablation",
    "train",
}

_PARTIAL_SPEC_MARKERS = {
    "post-train",
    "fine-tune",
    "alignment",
    "reasoning",
    "sft",
    "dpo",
    "ppo",
    "reward",
    "safety",
}


def detect_spec_level(user_brief: str) -> str:
    """Heuristically detect how specified the research request is."""
    brief = (user_brief or "").strip().lower()
    if not brief or len(brief.split()) < 8:
        return "zero_spec"

    full_hits = sum(1 for marker in _FULL_SPEC_MARKERS if marker in brief)
    partial_hits = sum(1 for marker in _PARTIAL_SPEC_MARKERS if marker in brief)

    if full_hits >= 4:
        return "full_spec"
    if full_hits >= 1 or partial_hits >= 1:
        return "partial_spec"
    return "zero_spec"


def bootstrap_idea_candidates(project_id: str, user_brief: str) -> list[ResearchIdea]:
    """Generate placeholder idea slots for zero-spec startup."""
    prompt = (user_brief or "autonomous llm post-training project").strip()
    return [
        ResearchIdea(
            idea_id=make_id("idea"),
            title=f"Candidate direction {idx}: derive from {prompt[:48]}",
            summary="Use literature search to sharpen this into a concrete, testable research idea.",
            rationale="Bootstrap placeholder created for a zero-spec project.",
            source="bootstrap-template",
        )
        for idx in range(1, 4)
    ]


def bootstrap_hypothesis_template(project_id: str, chosen_idea_id: str | None = None) -> Hypothesis:
    """Generate a default hypothesis template."""
    return Hypothesis(
        hypothesis_id=make_id("hyp"),
        statement="A targeted post-training intervention will outperform a strong baseline on the chosen evaluation suite.",
        success_metric="Define one primary automatic metric and one qualitative checkpoint before training.",
        stop_condition="Stop when the primary metric regresses for two consecutive checkpoints or cost exceeds the approved budget.",
        rationale="Bootstrap hypothesis for autonomous research startup.",
        linked_idea_id=chosen_idea_id,
        status="draft",
    )


def bootstrap_experiment_template(project_id: str, hypothesis_id: str, method: str = "sft") -> ExperimentSpec:
    """Generate a first-pass experiment template."""
    return ExperimentSpec(
        experiment_id=make_id("exp"),
        title="Bootstrap experiment",
        hypothesis_id=hypothesis_id,
        objective="Turn the chosen hypothesis into the first executable training/evaluation loop.",
        method=method,  # type: ignore[arg-type]
        evaluation_plan=[
            "Select one held-out benchmark aligned to the hypothesis.",
            "Run a smoke-test qualitative sample inspection.",
            "Compare against the current baseline and record ablations.",
        ],
        success_metric="Primary benchmark improves versus baseline without obvious qualitative regressions.",
        stop_condition="Abort if training diverges, metrics regress, or compute budget is exceeded.",
    )


def compose_continuation_prompt(
    project: ResearchProject,
    status_summary: dict[str, Any],
    reason: str,
    next_action: str = "",
) -> str:
    """Build a self-contained cron prompt for continuing a project."""
    summary_json = json.dumps(status_summary, indent=2, ensure_ascii=False)
    next_line = f"Prioritize this next action: {next_action}" if next_action else "Resume from the saved loop state and continue the research loop."
    return (
        "You are continuing an autonomous Hermes Research Agent project in a fresh session.\n"
        f"Working directory: {project.cwd}\n"
        f"Project id: {project.project_id}\n"
        f"Project name: {project.name}\n"
        f"Reason for continuation: {reason}\n"
        f"{next_line}\n"
        "First call research_loop(action='resume_project', project_id=..., cwd=...) to load the saved state.\n"
        "Then inspect active runs, literature, hypotheses, experiments, and unread inbox items.\n"
        "Continue the research loop until you either complete the current chunk, need user approval, or finish the project.\n"
        "Before finishing this session, call research_loop(action='checkpoint_loop', ...) if more work remains.\n"
        f"Saved status summary:\n{summary_json}"
    )


def generate_search_configs(
    strategy: str,
    param_grid: dict[str, list[Any]],
    base_config: dict[str, Any],
    max_configs: int = 50,
) -> list[dict[str, Any]]:
    """Generate hyperparameter search configurations.

    Args:
        strategy: 'grid' for exhaustive grid search, 'random' for random sampling.
        param_grid: Dict mapping parameter names to lists of values.
        base_config: Base run configuration to overlay on.
        max_configs: Maximum number of configs to return.
    """
    if strategy == "grid":
        keys = sorted(param_grid.keys())
        values = [param_grid[k] for k in keys]
        combos = list(itertools.product(*values))
        configs = []
        for combo in combos[:max_configs]:
            cfg = dict(base_config)
            for key, val in zip(keys, combo):
                cfg[key] = val
            configs.append(cfg)
        return configs
    elif strategy == "random":
        keys = sorted(param_grid.keys())
        total_possible = 1
        for vals in param_grid.values():
            total_possible *= len(vals)
        n_samples = min(max_configs, total_possible)
        all_combos = list(itertools.product(*[param_grid[k] for k in keys]))
        sampled = random.sample(all_combos, n_samples)
        configs = []
        for combo in sampled:
            cfg = dict(base_config)
            for key, val in zip(keys, combo):
                cfg[key] = val
            configs.append(cfg)
        return configs
    else:
        raise ValueError(f"Unsupported search strategy: {strategy}")


def diagnose_run_failure(run: dict[str, Any], runtime_status: dict[str, Any]) -> dict[str, Any]:
    """Pattern-match a failed run and suggest recovery actions.

    Returns a dict with 'error_type', 'pattern', 'message', and 'recovery_options'.
    """
    error = str(runtime_status.get("error", "") or run.get("error", "")).lower()
    traceback_str = str(runtime_status.get("traceback", "")).lower()
    combined = error + " " + traceback_str

    if "out of memory" in combined or "oom" in combined or "cuda out of memory" in combined:
        return {
            "error_type": "OOM",
            "pattern": "GPU out of memory",
            "message": "The training process ran out of GPU memory.",
            "recovery_options": [
                "Reduce batch_size (try halving the current value)",
                "Lower lora_rank (e.g., from 64 to 32 or 16)",
                "Enable gradient checkpointing if supported",
                "Use a smaller base model",
            ],
        }

    if "nan" in combined or "loss is nan" in combined or "diverge" in combined:
        return {
            "error_type": "NaN_loss",
            "pattern": "NaN or diverging loss",
            "message": "Training loss became NaN or diverged.",
            "recovery_options": [
                "Lower learning_rate (try 1/10 of current value)",
                "Add gradient clipping via optimizer config",
                "Check dataset for corrupted or extreme values",
                "Reduce batch_size for more stable gradients",
            ],
        }

    if "not found" in combined or "no such file" in combined or "filenotfounderror" in combined:
        return {
            "error_type": "dataset_error",
            "pattern": "File not found",
            "message": "A required file (dataset or checkpoint) was not found.",
            "recovery_options": [
                "Verify dataset_path exists and is accessible",
                "Run dataset validation before training",
                "Check that checkpoint paths are absolute",
            ],
        }

    if "timeout" in combined or "timed out" in combined:
        return {
            "error_type": "timeout",
            "pattern": "Process timeout",
            "message": "The training process timed out.",
            "recovery_options": [
                "Resume from the last saved checkpoint",
                "Increase max_steps or reduce save_every for more frequent checkpoints",
                "Check for infrastructure issues (GPU availability, network)",
            ],
        }

    if "connection" in combined or "api" in combined or "service" in combined:
        return {
            "error_type": "connection_error",
            "pattern": "Service/API connection failure",
            "message": "The training process lost connection to the backend service.",
            "recovery_options": [
                "Check that the Tinker service is running",
                "Verify TINKER_API_KEY is valid",
                "Resume from the last checkpoint",
            ],
        }

    return {
        "error_type": "unknown",
        "pattern": "unrecognized error",
        "message": error[:500] or "No error message available.",
        "recovery_options": [
            "Inspect the runner log for details",
            "Try resuming from the last checkpoint",
            "Re-run with verbose logging enabled",
        ],
    }


def compose_monitor_prompt(project: ResearchProject, run_id: str) -> str:
    """Build a self-contained cron prompt for run monitoring."""
    return (
        "You are monitoring a Hermes Research Agent training run in a fresh session.\n"
        f"Working directory: {project.cwd}\n"
        f"Project id: {project.project_id}\n"
        f"Run id: {run_id}\n"
        "First call research_loop(action='monitor_run', project_id=..., run_id=..., cwd=...).\n"
        "If the run is still active, let the monitoring tool schedule the next check.\n"
        "If the run finished or failed, leave a concise summary in the project inbox and continue the research loop if appropriate."
    )

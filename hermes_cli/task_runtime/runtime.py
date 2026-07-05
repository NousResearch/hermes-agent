"""Task Runtime coordinator.

Thin coordinator: each step delegates to a dedicated component module.
NO business logic here.

Pipeline:
    resolved_intent → context → skills → contract → pipeline → result.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hermes_cli.task_runtime.intent_resolver import resolve as resolve_intent, ResolvedIntent
from hermes_cli.task_runtime.context_resolver import resolve as resolve_context
from hermes_cli.task_runtime.skill_loader import load as load_skills
from hermes_cli.task_runtime.task_contract_builder import build as build_contract
from hermes_cli.task_runtime.execution_pipeline import run as run_pipeline, PipelineResult
from hermes_cli.task_runtime.result_consolidator import build as build_result, TaskResult


_VALID_MODES = ("dry-run", "shadow", "supervised", "enforce")


class TaskRuntime:
    """Thin coordinator: glue, no logic."""

    def __init__(
        self,
        *,
        hermes_home: Path | None = None,
        trace_dir: Path | None = None,
    ):
        self.hermes_home = hermes_home
        self.trace_dir = trace_dir

    def run(
        self,
        raw_text: str,
        *,
        execution_mode: str = "dry-run",
        source: str = "cli",
        source_id: str | None = None,
        confirmation_token: str | None = None,
    ) -> TaskResult:
        """Top-level: convert raw_text → execution → result.

        Pure orchestration. Each step delegates to a dedicated module.
        """
        if execution_mode not in _VALID_MODES:
            raise ValueError(
                f"invalid execution_mode: {execution_mode!r}; "
                f"must be one of {_VALID_MODES}"
            )

        resolved_intent: ResolvedIntent = resolve_intent(
            raw_text=raw_text,
            source=source,
            source_id=source_id,
        )
        context: dict[str, Any] = resolve_context(resolved_intent, self.hermes_home)
        skills = load_skills(resolved_intent, context)
        contract: dict[str, Any] = build_contract(
            resolved_intent, context, skills, execution_mode=execution_mode,
        )
        pipeline_result: PipelineResult = run_pipeline(
            contract,
            workdir=self.trace_dir,
            confirmation_token=confirmation_token,
        )
        return build_result(
            resolved_intent,
            contract,
            pipeline_result,
            trace_dir=self.trace_dir,
        )
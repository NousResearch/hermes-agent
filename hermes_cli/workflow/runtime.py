from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Callable

from hermes_cli.workflow.orchestrator import (
    EvaluationResult,
    Evaluator,
    FinalSynthesizer,
    Planner,
    Router,
    Subtask,
    TaskAnalysis,
    TaskAnalyzer,
    WorkerOutput,
    WorkerRunner,
    WorkflowConfig,
    WorkflowError,
    WorkflowLogger,
    WorkflowRun,
)
from hermes_cli.workflow.state import RunStore, analysis_from_record


class WorkflowPaused(WorkflowError):
    """Raised when a persisted run was paused by the user."""


class AgentExecutor:
    """Runs a subtask through the configured worker model/provider."""

    def __init__(self, config: WorkflowConfig, dry_run: bool):
        self.runner = WorkerRunner(config, dry_run=dry_run)

    def execute(
        self,
        subtask: Subtask,
        analysis: TaskAnalysis,
        previous_outputs: list[WorkerOutput],
    ) -> tuple[str, str, str]:
        return self.runner.run(subtask, analysis, previous_outputs)

    def revise(
        self,
        subtask: Subtask,
        analysis: TaskAnalysis,
        previous_outputs: list[WorkerOutput],
        first_output: str,
        evaluation: EvaluationResult,
    ) -> str:
        return self.runner.revise(
            subtask, analysis, previous_outputs, first_output, evaluation
        )


class EvaluatorLoop:
    """Evaluator-optimizer loop with one bounded revision cycle by default."""

    def __init__(self, config: WorkflowConfig):
        self.config = config
        self.evaluator = Evaluator()

    def evaluate(
        self,
        *,
        subtask: Subtask,
        analysis: TaskAnalysis,
        previous_outputs: list[WorkerOutput],
        output: str,
        executor: AgentExecutor,
    ) -> tuple[str, EvaluationResult, bool]:
        evaluation = self.evaluator.evaluate(subtask, output)
        revision_used = False
        if not evaluation.passed and self.config.max_revision_cycles > 0:
            output = executor.revise(
                subtask, analysis, previous_outputs, output, evaluation
            )
            evaluation = self.evaluator.evaluate(subtask, output)
            revision_used = True
        return output, evaluation, revision_used


class ContestHarness:
    """Runs competing workers for one subtask and selects the strongest output."""

    def __init__(
        self,
        *,
        config: WorkflowConfig,
        executor: AgentExecutor,
        evaluator_loop: EvaluatorLoop,
        record: Callable[[str, dict[str, Any]], None],
        cross_review: bool,
    ):
        self.config = config
        self.executor = executor
        self.evaluator_loop = evaluator_loop
        self.record = record
        self.cross_review = cross_review

    def run(
        self,
        subtask: Subtask,
        analysis: TaskAnalysis,
        previous_outputs: list[WorkerOutput],
    ) -> WorkerOutput:
        candidates: list[WorkerOutput] = []
        for worker_type in self._candidate_workers(subtask):
            candidate = self._candidate_subtask(subtask, worker_type)
            output, provider, model = self.executor.execute(
                candidate, analysis, previous_outputs
            )
            output, evaluation, revision_used = self.evaluator_loop.evaluate(
                subtask=candidate,
                analysis=analysis,
                previous_outputs=previous_outputs,
                output=output,
                executor=self.executor,
            )
            worker_output = WorkerOutput(
                subtask=candidate,
                worker_type=worker_type,
                provider=provider,
                model=model,
                output=output,
                evaluation=evaluation,
                revision_used=revision_used,
            )
            candidates.append(worker_output)
            self.record(
                "contest_candidate",
                {
                    "subtask_id": subtask.id,
                    "candidate_id": candidate.id,
                    "worker_type": worker_type,
                    "provider": provider,
                    "model": model,
                    "score": evaluation.score,
                    "passed": evaluation.passed,
                    "revision_used": revision_used,
                    "result": output,
                },
            )

        reviews = self._cross_review(subtask, analysis, previous_outputs, candidates)
        selected = self._select_winner(candidates, reviews)
        winner = WorkerOutput(
            subtask=subtask,
            worker_type=f"{selected.worker_type}:contest_winner",
            provider=selected.provider,
            model=selected.model,
            output=self._winner_output(selected, candidates, reviews),
            evaluation=selected.evaluation,
            revision_used=selected.revision_used,
        )
        self.record(
            "contest_selected",
            {
                "subtask_id": subtask.id,
                "winner_worker": winner.worker_type,
                "winner_candidate_id": selected.subtask.id,
                "winner_score": winner.evaluation.score,
                "candidate_count": len(candidates),
                "cross_review_count": len(reviews),
            },
        )
        return winner

    def _candidate_workers(self, subtask: Subtask) -> list[str]:
        workers = [subtask.worker_type or "researcher", *self.config.contest_workers]
        unique: list[str] = []
        for worker in workers:
            if worker not in unique:
                unique.append(worker)
        return unique[: self.config.contest_candidates]

    def _candidate_subtask(self, subtask: Subtask, worker_type: str) -> Subtask:
        return Subtask(
            id=f"{subtask.id}:candidate:{worker_type}",
            objective=(
                f"{subtask.objective}\n\n"
                "Contest mode: produce the strongest independent answer. "
                "Do not assume other candidates are correct."
            ),
            input_context=subtask.input_context,
            expected_output=subtask.expected_output,
            acceptance_criteria=subtask.acceptance_criteria,
            risk_level=subtask.risk_level,
            worker_type=worker_type,
            depends_on=list(subtask.depends_on),
            can_parallel=subtask.can_parallel,
        )

    def _cross_review(
        self,
        subtask: Subtask,
        analysis: TaskAnalysis,
        previous_outputs: list[WorkerOutput],
        candidates: list[WorkerOutput],
    ) -> list[WorkerOutput]:
        if not self.cross_review:
            return []

        candidate_context = "\n\n".join(
            f"{item.subtask.id} ({item.worker_type}, score={item.evaluation.score}):\n"
            f"{item.output}"
            for item in candidates
        )
        reviews: list[WorkerOutput] = []
        for worker_type in self.config.cross_review_workers:
            review_subtask = Subtask(
                id=f"{subtask.id}:cross-review:{worker_type}",
                objective=(
                    "Adversarially review contest candidates and identify the "
                    "best answer, gaps, contradictions, unsafe actions, and "
                    "unclear assumptions."
                ),
                input_context=(
                    f"Original subtask: {subtask.objective}\n\n"
                    f"Acceptance criteria:\n{json.dumps(subtask.acceptance_criteria, ensure_ascii=False)}\n\n"
                    f"Candidate outputs:\n{candidate_context}"
                ),
                expected_output="A ranked review of candidates with risks and a recommended winner.",
                acceptance_criteria=[
                    "Ranks or compares candidate outputs.",
                    "Calls out gaps, contradictions, unsafe actions, or unclear assumptions.",
                    "Recommends the strongest candidate or a merge direction.",
                ],
                risk_level=subtask.risk_level,
                worker_type=worker_type,
            )
            output, provider, model = self.executor.execute(
                review_subtask, analysis, previous_outputs + candidates
            )
            output, evaluation, revision_used = self.evaluator_loop.evaluate(
                subtask=review_subtask,
                analysis=analysis,
                previous_outputs=previous_outputs + candidates,
                output=output,
                executor=self.executor,
            )
            review = WorkerOutput(
                subtask=review_subtask,
                worker_type=worker_type,
                provider=provider,
                model=model,
                output=output,
                evaluation=evaluation,
                revision_used=revision_used,
            )
            reviews.append(review)
            self.record(
                "cross_review_result",
                {
                    "subtask_id": subtask.id,
                    "reviewer": worker_type,
                    "score": evaluation.score,
                    "passed": evaluation.passed,
                    "result": output,
                },
            )
        return reviews

    def _select_winner(
        self, candidates: list[WorkerOutput], reviews: list[WorkerOutput]
    ) -> WorkerOutput:
        review_text = "\n".join(item.output.lower() for item in reviews)

        def score(candidate: WorkerOutput) -> tuple[float, int, int]:
            mentions = review_text.count(candidate.subtask.id.lower())
            worker_mentions = review_text.count(candidate.worker_type.lower())
            return candidate.evaluation.score, mentions, worker_mentions

        return max(candidates, key=score)

    def _winner_output(
        self,
        winner: WorkerOutput,
        candidates: list[WorkerOutput],
        reviews: list[WorkerOutput],
    ) -> str:
        summary = [
            winner.output,
            "",
            "Contest harness",
            f"- Winner: {winner.subtask.id} via {winner.worker_type} "
            f"(score={winner.evaluation.score}).",
            "- Candidates: "
            + ", ".join(
                f"{item.subtask.id}/{item.worker_type}/score={item.evaluation.score}"
                for item in candidates
            ),
        ]
        if reviews:
            summary.append(
                "- Cross-reviewers: "
                + ", ".join(
                    f"{item.worker_type}/score={item.evaluation.score}" for item in reviews
                )
            )
        return "\n".join(summary)


class Scheduler:
    """Executes pending subtasks sequentially or with dependency-aware parallelism."""

    def __init__(
        self,
        *,
        store: RunStore,
        run_id: str,
        parallel: bool,
        max_concurrency: int,
    ):
        self.store = store
        self.run_id = run_id
        self.parallel = parallel
        self.max_concurrency = max(1, max_concurrency)

    def run(
        self,
        subtasks: list[Subtask],
        completed_outputs: list[WorkerOutput],
        execute_one: Callable[[Subtask, list[WorkerOutput]], WorkerOutput],
    ) -> list[WorkerOutput]:
        if not self.parallel:
            return self._run_sequential(subtasks, completed_outputs, execute_one)
        return self._run_parallel(subtasks, completed_outputs, execute_one)

    def _run_sequential(
        self,
        subtasks: list[Subtask],
        completed_outputs: list[WorkerOutput],
        execute_one: Callable[[Subtask, list[WorkerOutput]], WorkerOutput],
    ) -> list[WorkerOutput]:
        outputs = list(completed_outputs)
        completed_ids = {item.subtask.id for item in outputs}
        for subtask in subtasks:
            if subtask.id in completed_ids:
                continue
            self._ensure_active()
            worker_output = execute_one(subtask, outputs)
            outputs.append(worker_output)
            completed_ids.add(subtask.id)
        return outputs

    def _run_parallel(
        self,
        subtasks: list[Subtask],
        completed_outputs: list[WorkerOutput],
        execute_one: Callable[[Subtask, list[WorkerOutput]], WorkerOutput],
    ) -> list[WorkerOutput]:
        ordered_ids = [item.id for item in subtasks]
        outputs_by_id = {item.subtask.id: item for item in completed_outputs}
        remaining = {item.id: item for item in subtasks if item.id not in outputs_by_id}

        with ThreadPoolExecutor(max_workers=self.max_concurrency) as pool:
            futures: dict[Any, Subtask] = {}
            while remaining or futures:
                self._ensure_active()
                ready = [
                    item
                    for item in remaining.values()
                    if all(dep in outputs_by_id for dep in item.depends_on)
                ]
                if ready and not futures:
                    serial = [item for item in ready if not item.can_parallel]
                    batch = serial[:1] if serial else ready[: self.max_concurrency]
                    for subtask in batch:
                        previous = self._previous_context(subtask, outputs_by_id)
                        futures[pool.submit(execute_one, subtask, previous)] = subtask
                        remaining.pop(subtask.id, None)
                if not futures:
                    missing = {
                        dep
                        for item in remaining.values()
                        for dep in item.depends_on
                        if dep not in outputs_by_id
                    }
                    raise WorkflowError(
                        "No runnable workflow subtasks; unresolved dependencies: "
                        + ", ".join(sorted(missing))
                    )
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    subtask = futures.pop(future)
                    outputs_by_id[subtask.id] = future.result()

        return [outputs_by_id[item_id] for item_id in ordered_ids if item_id in outputs_by_id]

    def _previous_context(
        self, subtask: Subtask, outputs_by_id: dict[str, WorkerOutput]
    ) -> list[WorkerOutput]:
        if subtask.depends_on:
            return [outputs_by_id[item] for item in subtask.depends_on if item in outputs_by_id]
        return list(outputs_by_id.values())

    def _ensure_active(self) -> None:
        record = self.store.get_run(self.run_id)
        status = str((record or {}).get("status") or "")
        if status == "cancelled":
            raise WorkflowError(f"Workflow run {self.run_id} was cancelled.")
        if status == "paused":
            raise WorkflowPaused(f"Workflow run {self.run_id} was paused.")


class WorkflowRuntime:
    """Top-level runtime for local dynamic workflows."""

    def __init__(
        self,
        *,
        config: WorkflowConfig,
        dry_run: bool = False,
        max_subtasks: int | None = None,
        parallel: bool = False,
        codex_plan: bool = False,
        contest: bool = False,
        cross_review: bool = False,
    ):
        self.config = config
        self.dry_run = dry_run
        self.max_subtasks = max_subtasks or config.max_subtasks
        if parallel and not config.parallel_enabled:
            raise WorkflowError("Parallel workflow execution is disabled in config.")
        self.parallel = bool(parallel)
        self.codex_plan = bool(codex_plan)
        self.contest = bool(contest)
        self.cross_review = bool(cross_review)
        self.analyzer = TaskAnalyzer()
        self.planner = Planner(max_subtasks=self.max_subtasks)
        self.router = Router(config)
        self.executor = AgentExecutor(config, dry_run=dry_run)
        self.evaluator_loop = EvaluatorLoop(config)
        self.synthesizer = FinalSynthesizer()
        self.logger = WorkflowLogger(config)
        self.store = RunStore.from_config(config)

    def run(
        self,
        request: str,
        *,
        background: bool = False,
        run_id: str | None = None,
        preplanned_subtasks: list[Subtask] | None = None,
        analysis: TaskAnalysis | None = None,
    ) -> WorkflowRun:
        prepared = self.prepare_run(
            request,
            run_id=run_id,
            preplanned_subtasks=preplanned_subtasks,
            analysis=analysis,
            status="queued" if background else "running",
        )
        if background:
            return self._start_background(prepared)
        return self._execute_existing(prepared.run_id)

    def resume(self, run_id: str) -> WorkflowRun:
        return self._execute_existing(run_id)

    def prepare_run(
        self,
        request: str,
        *,
        run_id: str | None = None,
        preplanned_subtasks: list[Subtask] | None = None,
        analysis: TaskAnalysis | None = None,
        status: str = "queued",
    ) -> WorkflowRun:
        clean_request = request.strip()
        if not clean_request:
            raise WorkflowError("Workflow request cannot be empty.")

        run_id = run_id or _new_run_id()
        if self.store.get_run(run_id):
            raise WorkflowError(f"Workflow run already exists: {run_id}")

        analysis = analysis or self.analyzer.analyze(clean_request)
        subtasks = list(preplanned_subtasks or self.planner.plan(clean_request, analysis))
        for index, subtask in enumerate(subtasks):
            if index > 0 and not subtask.depends_on and analysis.task_type in {"code", "debug"}:
                subtask.depends_on = [subtasks[index - 1].id]
                subtask.can_parallel = False
            if not subtask.worker_type:
                subtask.worker_type = self.router.assign(subtask, analysis)

        mode_parts = ["parallel" if self.parallel else "sequential"]
        if self.contest:
            mode_parts.append("contest")
        if self.cross_review:
            mode_parts.append("cross-review")
        mode = "+".join(mode_parts)
        self.store.create_run(
            run_id=run_id,
            original_request=clean_request,
            analysis=analysis,
            mode=mode,
            dry_run=self.dry_run,
            codex_plan=self.codex_plan,
            contest=self.contest,
            cross_review=self.cross_review,
            status=status,
        )
        self._record(run_id, "request", {"original_request": clean_request})
        self._record(run_id, "analysis", analysis.to_dict())
        self._record(
            run_id,
            "planned",
            {"subtasks": [subtask.to_dict() for subtask in subtasks], "mode": mode},
        )
        for subtask in subtasks:
            self.store.save_subtask(run_id, subtask)
            agent_config = self.config.agent_for(subtask.worker_type or "researcher")
            self._record(
                run_id,
                "worker_selected",
                {
                    "subtask_id": subtask.id,
                    "selected_worker": subtask.worker_type,
                    "provider": agent_config.display_provider,
                    "model": agent_config.display_model,
                },
            )

        return WorkflowRun(
            run_id=run_id,
            original_request=clean_request,
            analysis=analysis,
            subtasks=subtasks,
            worker_outputs=[],
            final_response="Workflow run queued." if status == "queued" else "",
            log_path=self.logger.path,
            status=status,
            state_path=self.store.path,
            codex_plan=self.codex_plan,
            contest=self.contest,
            cross_review=self.cross_review,
        )

    def run_script(self, path: str | Path, *, background: bool = False) -> WorkflowRun:
        from hermes_cli.workflow.script_api import load_script

        script = load_script(path)
        if not script.subtasks:
            raise WorkflowError(f"Workflow script produced no tasks: {path}")
        if script.max_concurrency is not None:
            self.config.execution["max_concurrency"] = script.max_concurrency
        analysis = TaskAnalysis(
            task_type="planning",
            complexity="complex" if len(script.subtasks) > 3 else "medium",
            execute_directly=False,
            rationale=f"Loaded {len(script.subtasks)} subtask(s) from workflow script.",
        )
        return self.run(
            f"Workflow script: {Path(path).expanduser()}",
            background=background,
            preplanned_subtasks=script.subtasks,
            analysis=analysis,
        )

    def _execute_existing(self, run_id: str) -> WorkflowRun:
        record = self.store.get_run(run_id)
        if not record:
            raise WorkflowError(f"Workflow run not found: {run_id}")
        if record["status"] == "completed":
            return self._build_run_from_store(record)
        if record["status"] == "cancelled":
            raise WorkflowError(f"Workflow run {run_id} is cancelled.")

        self.store.update_run(run_id, status="running")
        self._record(run_id, "status", {"status": "running"})
        try:
            analysis = analysis_from_record(record)
            self.contest = bool(record["contest"])
            self.cross_review = bool(record["cross_review"])
            subtasks = self.store.load_subtasks(run_id)
            completed_outputs = self.store.load_outputs(run_id)
            scheduler = Scheduler(
                store=self.store,
                run_id=run_id,
                parallel=self.parallel,
                max_concurrency=self.config.max_concurrency,
            )
            outputs = scheduler.run(subtasks, completed_outputs, self._execute_subtask(run_id, analysis))
            self.codex_plan = bool(record["codex_plan"])
            final_response = self._final_response(analysis, outputs)
            self.store.update_run(run_id, status="completed", final_response=final_response)
            self._record(run_id, "final", {"result": final_response})
            return WorkflowRun(
                run_id=run_id,
                original_request=str(record["original_request"]),
                analysis=analysis,
                subtasks=subtasks,
                worker_outputs=outputs,
                final_response=final_response,
                log_path=self.logger.path,
                status="completed",
                state_path=self.store.path,
                codex_plan=bool(record["codex_plan"]),
                contest=bool(record["contest"]),
                cross_review=bool(record["cross_review"]),
            )
        except WorkflowPaused:
            self._record(run_id, "status", {"status": "paused"})
            return self._build_run_from_store(self.store.get_run(run_id) or record)
        except Exception as exc:
            self.store.update_run(run_id, status="failed", error=str(exc))
            self._record(run_id, "error", {"error": str(exc)})
            raise

    def _execute_subtask(
        self, run_id: str, analysis: TaskAnalysis
    ) -> Callable[[Subtask, list[WorkerOutput]], WorkerOutput]:
        def execute(subtask: Subtask, previous_outputs: list[WorkerOutput]) -> WorkerOutput:
            self.store.update_subtask_status(
                run_id, subtask.id, "running", subtask.worker_type
            )
            if self.contest:
                harness = ContestHarness(
                    config=self.config,
                    executor=self.executor,
                    evaluator_loop=self.evaluator_loop,
                    record=lambda event, payload: self._record(run_id, event, payload),
                    cross_review=self.cross_review,
                )
                worker_output = harness.run(subtask, analysis, previous_outputs)
                self.store.save_output(run_id, worker_output)
                self._record(
                    run_id,
                    "worker_result",
                    {
                        "subtask_id": subtask.id,
                        "selected_worker": worker_output.worker_type,
                        "provider": worker_output.provider,
                        "model": worker_output.model,
                        "result": worker_output.output,
                        "revision_used": worker_output.revision_used,
                        "contest": True,
                        "cross_review": self.cross_review,
                    },
                )
                self._record(
                    run_id,
                    "evaluation",
                    {
                        "subtask_id": subtask.id,
                        "evaluator_score": worker_output.evaluation.score,
                        "passed": worker_output.evaluation.passed,
                        "missing_steps": worker_output.evaluation.missing_steps,
                        "contradictions": worker_output.evaluation.contradictions,
                        "unsafe_actions": worker_output.evaluation.unsafe_actions,
                        "unclear_assumptions": worker_output.evaluation.unclear_assumptions,
                    },
                )
                return worker_output

            output, provider, model = self.executor.execute(
                subtask, analysis, previous_outputs
            )
            output, evaluation, revision_used = self.evaluator_loop.evaluate(
                subtask=subtask,
                analysis=analysis,
                previous_outputs=previous_outputs,
                output=output,
                executor=self.executor,
            )
            worker_output = WorkerOutput(
                subtask=subtask,
                worker_type=subtask.worker_type or "researcher",
                provider=provider,
                model=model,
                output=output,
                evaluation=evaluation,
                revision_used=revision_used,
            )
            self.store.save_output(run_id, worker_output)
            self._record(
                run_id,
                "worker_result",
                {
                    "subtask_id": subtask.id,
                    "selected_worker": worker_output.worker_type,
                    "provider": provider,
                    "model": model,
                    "result": output,
                    "revision_used": revision_used,
                },
            )
            self._record(
                run_id,
                "evaluation",
                {
                    "subtask_id": subtask.id,
                    "evaluator_score": evaluation.score,
                    "passed": evaluation.passed,
                    "missing_steps": evaluation.missing_steps,
                    "contradictions": evaluation.contradictions,
                    "unsafe_actions": evaluation.unsafe_actions,
                    "unclear_assumptions": evaluation.unclear_assumptions,
                },
            )
            return worker_output

        return execute

    def _start_background(self, prepared: WorkflowRun) -> WorkflowRun:
        if not self.config.background_enabled or not self.config.allow_background_execution:
            raise WorkflowError("Background workflow execution is disabled in config.")

        log_file = self.logger.path.parent / f"workflow_background_{prepared.run_id}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "-m",
            "hermes_cli.main",
            "workflow",
            "run",
            "--config",
            str(self.config.path),
            "--resume-run-id",
            prepared.run_id,
            "--internal-background-worker",
        ]
        if self.dry_run:
            cmd.append("--dry-run")
        if self.parallel:
            cmd.append("--parallel")
        if self.codex_plan:
            cmd.append("--codex-plan")
        if self.contest:
            cmd.append("--contest")
        if self.cross_review:
            cmd.append("--cross-review")
        with log_file.open("ab") as handle:
            process = subprocess.Popen(
                cmd,
                cwd=os.getcwd(),
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        self.store.update_run(prepared.run_id, pid=process.pid)
        self._record(
            prepared.run_id,
            "background_started",
            {"pid": process.pid, "background_log": str(log_file)},
        )
        prepared.final_response = (
            f"Workflow run queued in background.\n"
            f"Run id: {prepared.run_id}\n"
            f"Status: hermes workflow status {prepared.run_id}"
        )
        prepared.status = "queued"
        return prepared

    def _build_run_from_store(self, record: dict[str, Any]) -> WorkflowRun:
        analysis = analysis_from_record(record)
        return WorkflowRun(
            run_id=str(record["run_id"]),
            original_request=str(record["original_request"]),
            analysis=analysis,
            subtasks=self.store.load_subtasks(str(record["run_id"])),
            worker_outputs=self.store.load_outputs(str(record["run_id"])),
            final_response=str(record.get("final_response") or ""),
            log_path=self.logger.path,
            status=str(record["status"]),
            state_path=self.store.path,
            codex_plan=bool(record["codex_plan"]),
            contest=bool(record["contest"]),
            cross_review=bool(record["cross_review"]),
        )

    def _final_response(self, analysis: TaskAnalysis, outputs: list[WorkerOutput]) -> str:
        if self.codex_plan:
            return _codex_plan_response(analysis, outputs, dry_run=self.dry_run)
        return self.synthesizer.synthesize(analysis, outputs, dry_run=self.dry_run)

    def _record(self, run_id: str, event: str, payload: dict[str, Any]) -> None:
        self.logger.log(run_id, event, payload)
        self.store.append_event(run_id, event, payload)


def _codex_plan_response(
    analysis: TaskAnalysis, outputs: list[WorkerOutput], *, dry_run: bool
) -> str:
    lines = [
        "Codex Goal Plan",
        "",
        f"- Classification: {analysis.task_type} / {analysis.complexity}",
        f"- Execution source: {'dry-run' if dry_run else 'model-backed'} Hermes workflow",
        "",
        "Phases",
    ]
    for index, item in enumerate(outputs, start=1):
        status = "passed" if item.evaluation.passed else "needs attention"
        lines.append(
            f"{index}. {item.subtask.objective} "
            f"({item.worker_type}, score={item.evaluation.score}, {status})"
        )
    risks = [
        item
        for item in outputs
        if not item.evaluation.passed or item.subtask.risk_level == "high"
    ]
    lines.extend(["", "Risks"])
    if risks:
        for item in risks:
            lines.append(f"- {item.subtask.id}: {item.evaluation.notes or item.subtask.risk_level}")
    else:
        lines.append("- No evaluator-blocking risks were detected.")
    lines.extend(["", "Next action", "- Use this as the Codex Goal execution checklist."])
    return "\n".join(lines)


def _new_run_id() -> str:
    return f"wf-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"

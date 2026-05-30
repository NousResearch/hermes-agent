from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml


TASK_TYPES = (
    "code",
    "debug",
    "UI/UX",
    "product design",
    "research",
    "writing",
    "planning",
    "review",
)

WORKER_TYPES = (
    "coder",
    "debugger",
    "uiux_designer",
    "product_planner",
    "researcher",
    "reviewer",
    "tester",
)

_STOPWORDS = {
    "about",
    "acceptance",
    "addresses",
    "criteria",
    "deliverable",
    "expected",
    "include",
    "input",
    "output",
    "produce",
    "relevant",
    "request",
    "should",
    "subtask",
    "task",
    "that",
    "when",
    "with",
}


class WorkflowError(RuntimeError):
    """Raised when a workflow run cannot continue."""


@dataclass
class TaskAnalysis:
    task_type: str
    complexity: str
    execute_directly: bool
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type,
            "complexity": self.complexity,
            "execute_directly": self.execute_directly,
            "rationale": self.rationale,
        }


@dataclass
class Subtask:
    id: str
    objective: str
    input_context: str
    expected_output: str
    acceptance_criteria: list[str]
    risk_level: str
    worker_type: str | None = None
    depends_on: list[str] = field(default_factory=list)
    can_parallel: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "objective": self.objective,
            "input_context": self.input_context,
            "expected_output": self.expected_output,
            "acceptance_criteria": list(self.acceptance_criteria),
            "risk_level": self.risk_level,
            "worker_type": self.worker_type,
            "depends_on": list(self.depends_on),
            "can_parallel": self.can_parallel,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Subtask":
        return cls(
            id=str(data.get("id") or ""),
            objective=str(data.get("objective") or ""),
            input_context=str(data.get("input_context") or ""),
            expected_output=str(data.get("expected_output") or ""),
            acceptance_criteria=[str(item) for item in data.get("acceptance_criteria") or []],
            risk_level=str(data.get("risk_level") or "medium"),
            worker_type=str(data.get("worker_type") or "") or None,
            depends_on=[str(item) for item in data.get("depends_on") or []],
            can_parallel=bool(data.get("can_parallel", True)),
        )


@dataclass
class EvaluationResult:
    passed: bool
    score: float
    missing_steps: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    unsafe_actions: list[str] = field(default_factory=list)
    unclear_assumptions: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "score": self.score,
            "missing_steps": list(self.missing_steps),
            "contradictions": list(self.contradictions),
            "unsafe_actions": list(self.unsafe_actions),
            "unclear_assumptions": list(self.unclear_assumptions),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationResult":
        return cls(
            passed=bool(data.get("passed", False)),
            score=float(data.get("score", 0.0)),
            missing_steps=[str(item) for item in data.get("missing_steps") or []],
            contradictions=[str(item) for item in data.get("contradictions") or []],
            unsafe_actions=[str(item) for item in data.get("unsafe_actions") or []],
            unclear_assumptions=[str(item) for item in data.get("unclear_assumptions") or []],
            notes=str(data.get("notes") or ""),
        )


@dataclass
class WorkerOutput:
    subtask: Subtask
    worker_type: str
    provider: str
    model: str
    output: str
    evaluation: EvaluationResult
    revision_used: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "subtask": self.subtask.to_dict(),
            "worker_type": self.worker_type,
            "provider": self.provider,
            "model": self.model,
            "output": self.output,
            "evaluation": self.evaluation.to_dict(),
            "revision_used": self.revision_used,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkerOutput":
        return cls(
            subtask=Subtask.from_dict(data.get("subtask") or {}),
            worker_type=str(data.get("worker_type") or "researcher"),
            provider=str(data.get("provider") or ""),
            model=str(data.get("model") or ""),
            output=str(data.get("output") or ""),
            evaluation=EvaluationResult.from_dict(data.get("evaluation") or {}),
            revision_used=bool(data.get("revision_used", False)),
        )


@dataclass
class WorkflowRun:
    run_id: str
    original_request: str
    analysis: TaskAnalysis
    subtasks: list[Subtask]
    worker_outputs: list[WorkerOutput]
    final_response: str
    log_path: Path
    status: str = "completed"
    state_path: Path | None = None
    codex_plan: bool = False
    contest: bool = False
    cross_review: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "original_request": self.original_request,
            "status": self.status,
            "analysis": self.analysis.to_dict(),
            "subtasks": [item.to_dict() for item in self.subtasks],
            "worker_outputs": [item.to_dict() for item in self.worker_outputs],
            "final_response": self.final_response,
            "log_path": str(self.log_path),
            "state_path": str(self.state_path) if self.state_path else "",
            "codex_plan": self.codex_plan,
            "contest": self.contest,
            "cross_review": self.cross_review,
        }


@dataclass
class AgentConfig:
    worker_type: str
    provider: str = ""
    model: str = ""
    prompt: str = ""
    toolsets: list[str] = field(default_factory=list)
    description: str = ""
    tools: list[str] = field(default_factory=list)
    max_turns: int = 15
    can_write_files: bool = False

    @property
    def display_provider(self) -> str:
        return self.provider or "configured-default"

    @property
    def display_model(self) -> str:
        return self.model or "configured-default"


@dataclass
class WorkflowConfig:
    path: Path
    agents: dict[str, AgentConfig]
    routing: dict[str, Any]
    execution: dict[str, Any]
    safety: dict[str, Any]
    logging: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path | None = None) -> "WorkflowConfig":
        config_path = Path(path).expanduser() if path else default_config_path()
        if not config_path.exists():
            raise WorkflowError(f"Workflow config not found: {config_path}")

        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        agents: dict[str, AgentConfig] = {}
        agent_root = raw.get("agents") or {}
        for worker_type in WORKER_TYPES:
            item = agent_root.get(worker_type) or {}
            agents[worker_type] = AgentConfig(
                worker_type=worker_type,
                provider=str(item.get("provider") or ""),
                model=str(item.get("model") or ""),
                prompt=str(item.get("prompt") or f"prompts/worker_{worker_type}.md"),
                toolsets=list(item.get("toolsets") or item.get("tools") or []),
                description=str(item.get("description") or ""),
                tools=list(item.get("tools") or item.get("toolsets") or []),
                max_turns=int(item.get("max_turns") or 15),
                can_write_files=bool(item.get("can_write_files", False)),
            )

        return cls(
            path=config_path,
            agents=agents,
            routing=dict(raw.get("routing") or {}),
            execution=dict(raw.get("execution") or {}),
            safety=dict(raw.get("safety") or {}),
            logging=dict(raw.get("logging") or {}),
        )

    def agent_for(self, worker_type: str) -> AgentConfig:
        return self.agents.get(worker_type) or self.agents["researcher"]

    @property
    def max_revision_cycles(self) -> int:
        return int(self.execution.get("max_revision_cycles", 1))

    @property
    def max_subtasks(self) -> int:
        return int(self.execution.get("max_subtasks", 8))

    @property
    def allow_file_changes(self) -> bool:
        return bool(self.safety.get("allow_file_changes", False))

    @property
    def parallel_enabled(self) -> bool:
        return bool(self.execution.get("parallel_enabled", False))

    @property
    def background_enabled(self) -> bool:
        return bool(self.execution.get("background_enabled", False))

    @property
    def allow_background_execution(self) -> bool:
        value = self.safety.get("allow_background_execution", False)
        return value is True or str(value).lower() in {"true", "explicit-only"}

    @property
    def max_concurrency(self) -> int:
        return max(1, int(self.execution.get("max_concurrency", 2)))

    @property
    def contest_candidates(self) -> int:
        return max(2, min(5, int(self.execution.get("contest_candidates", 3))))

    @property
    def contest_workers(self) -> list[str]:
        workers = [str(item) for item in self.execution.get("contest_workers") or []]
        valid = [worker for worker in workers if worker in WORKER_TYPES]
        return valid or ["researcher", "product_planner", "reviewer"]

    @property
    def cross_review_workers(self) -> list[str]:
        workers = [str(item) for item in self.execution.get("cross_review_workers") or []]
        valid = [worker for worker in workers if worker in WORKER_TYPES]
        return valid or ["reviewer", "tester"]

    @property
    def default_permission_mode(self) -> str:
        return str(self.execution.get("default_permission_mode") or "read-only")

    @property
    def state_db(self) -> str:
        return str(self.execution.get("state_db") or "")


def package_root() -> Path:
    return Path(__file__).resolve().parent


def default_config_path() -> Path:
    return package_root() / "config" / "workflow.yaml"


def run_workflow(
    request: str,
    *,
    config_path: str | Path | None = None,
    dry_run: bool = False,
    max_subtasks: int | None = None,
    parallel: bool = False,
    background: bool = False,
    codex_plan: bool = False,
    contest: bool = False,
    cross_review: bool = False,
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
    run_id: str | None = None,
    resume: bool = False,
) -> WorkflowRun:
    from hermes_cli.workflow.runtime import WorkflowRuntime

    runtime = WorkflowRuntime(
        config=WorkflowConfig.load(config_path),
        dry_run=dry_run,
        max_subtasks=max_subtasks,
        parallel=parallel,
        codex_plan=codex_plan,
        contest=contest,
        cross_review=cross_review,
        progress_callback=progress_callback,
    )
    if resume:
        if not run_id:
            raise WorkflowError("resume requires a run_id.")
        return runtime.resume(run_id)
    return runtime.run(request, background=background, run_id=run_id)


class TaskAnalyzer:
    _KEYWORDS: dict[str, tuple[str, ...]] = {
        "code": (
            "add",
            "api",
            "build",
            "cli",
            "code",
            "command",
            "implement",
            "module",
            "python",
            "script",
            "test",
        ),
        "debug": (
            "bug",
            "debug",
            "error",
            "exception",
            "failing",
            "fix",
            "regression",
            "root cause",
            "traceback",
        ),
        "UI/UX": (
            "design system",
            "figma",
            "frontend",
            "screen",
            "ui",
            "ux",
            "visual",
            "wireframe",
        ),
        "product design": (
            "customer",
            "feature",
            "journey",
            "mvp",
            "prd",
            "product",
            "requirements",
            "user story",
        ),
        "research": (
            "best practice",
            "compare",
            "investigate",
            "latest",
            "look up",
            "research",
            "source",
            "study",
        ),
        "writing": (
            "article",
            "copy",
            "draft",
            "email",
            "essay",
            "readme",
            "rewrite",
            "write",
        ),
        "planning": (
            "architecture",
            "break down",
            "plan",
            "planning",
            "roadmap",
            "schedule",
            "strategy",
            "workflow",
        ),
        "review": (
            "audit",
            "evaluate",
            "feedback",
            "review",
            "risks",
            "security",
            "verify",
        ),
    }

    def analyze(self, request: str) -> TaskAnalysis:
        text = request.lower()
        scores = {task_type: 0 for task_type in TASK_TYPES}
        for task_type, keywords in self._KEYWORDS.items():
            for keyword in keywords:
                if keyword in text:
                    scores[task_type] += 2 if " " in keyword else 1

        task_type = max(scores, key=lambda item: scores[item])
        if scores[task_type] == 0:
            task_type = "planning"

        marker_count = len(
            re.findall(r"(?m)^\s*(?:[-*]|\d+\.|[A-Z][A-Za-z /&-]{2,}:)", request)
        )
        word_count = len(re.findall(r"\w+", request))
        complex_terms = (
            "architecture",
            "orchestrator",
            "required modules",
            "safety requirements",
            "subtasks",
            "system",
        )
        if word_count > 260 or marker_count >= 6 or any(term in text for term in complex_terms):
            complexity = "complex"
        elif word_count > 80 or marker_count >= 3:
            complexity = "medium"
        else:
            complexity = "simple"

        execute_directly = complexity == "simple" and not any(
            term in text for term in ("split", "subtask", "orchestrator", "workflow")
        )
        rationale = (
            f"Matched task type '{task_type}' from request keywords; "
            f"word_count={word_count}, requirement_markers={marker_count}."
        )
        return TaskAnalysis(task_type, complexity, execute_directly, rationale)


class Planner:
    def __init__(self, max_subtasks: int = 8):
        self.max_subtasks = max(1, min(8, max_subtasks))

    def plan(self, request: str, analysis: TaskAnalysis) -> list[Subtask]:
        if analysis.execute_directly:
            return [
                self._build_subtask(
                    "1",
                    "Complete the request directly",
                    request,
                    "A concise answer or implementation plan that satisfies the request.",
                    "low" if analysis.task_type in {"writing", "research"} else "medium",
                )
            ]

        objectives = self._objectives_for(analysis.task_type)
        if analysis.complexity == "medium":
            objectives = objectives[:4]
        objectives = objectives[: self.max_subtasks]
        return [
            self._build_subtask(
                str(index),
                objective,
                request,
                self._expected_output_for(objective),
                self._risk_for(objective, request),
            )
            for index, objective in enumerate(objectives, start=1)
        ]

    def _objectives_for(self, task_type: str) -> list[str]:
        if task_type in {"code", "debug"}:
            return [
                "Confirm scope, integration points, and safety constraints",
                "Design the minimal module and configuration structure",
                "Implement the core orchestration behavior",
                "Wire the optional command entrypoint",
                "Add focused tests and an example run",
                "Review evaluator results, risks, and next action",
            ]
        if task_type == "UI/UX":
            return [
                "Clarify target users, workflows, and screen states",
                "Define information architecture and interaction model",
                "Draft visual and component direction",
                "Review accessibility, responsiveness, and edge cases",
            ]
        if task_type == "product design":
            return [
                "Clarify user problem, scope, and success criteria",
                "Define MVP behavior and non-goals",
                "Break the product work into phased deliverables",
                "Review risks, dependencies, and validation steps",
            ]
        if task_type == "research":
            return [
                "Frame the research question and source needs",
                "Collect relevant findings and tradeoffs",
                "Compare options against the request constraints",
                "Summarize recommendations and open questions",
            ]
        if task_type == "review":
            return [
                "Identify the review target and risk model",
                "Inspect behavior against stated requirements",
                "List findings by severity with evidence",
                "Summarize residual risk and follow-up checks",
            ]
        return [
            "Clarify the goal, constraints, and assumptions",
            "Break the work into practical phases",
            "Draft the requested artifact or plan",
            "Review gaps, risks, and next action",
        ]

    def _build_subtask(
        self,
        subtask_id: str,
        objective: str,
        request: str,
        expected_output: str,
        risk_level: str,
    ) -> Subtask:
        return Subtask(
            id=f"wf-{subtask_id}",
            objective=objective,
            input_context=_truncate(
                "Original request:\n"
                f"{request.strip()}\n\n"
                "Use previous worker outputs as context when available.",
                2400,
            ),
            expected_output=expected_output,
            acceptance_criteria=[
                f"Addresses objective: {objective}",
                f"Produces expected output: {expected_output}",
                "Calls out assumptions, risks, or blockers when relevant.",
            ],
            risk_level=risk_level,
        )

    def _expected_output_for(self, objective: str) -> str:
        lower = objective.lower()
        if "test" in lower or "example" in lower:
            return "Verification steps, example output, and any failing checks."
        if "wire" in lower or "entrypoint" in lower or "command" in lower:
            return "A CLI integration plan or implementation notes."
        if "implement" in lower:
            return "Concrete implementation details with module responsibilities."
        if "review" in lower:
            return "Risks, evaluator observations, and recommended next action."
        return "A concise plan fragment that can be merged into the final answer."

    def _risk_for(self, objective: str, request: str) -> str:
        text = f"{objective} {request}".lower()
        if any(term in text for term in ("credential", "secret", "destructive", "delete", "payment")):
            return "high"
        if any(term in text for term in ("code", "command", "implement", "file", "debug", "security")):
            return "medium"
        return "low"


class Router:
    def __init__(self, config: WorkflowConfig):
        self.config = config

    def assign(self, subtask: Subtask, analysis: TaskAnalysis) -> str:
        text = f"{subtask.objective} {subtask.expected_output}".lower()
        for rule in self.config.routing.get("keywords", []) or []:
            worker = str(rule.get("worker") or "")
            matches = rule.get("match") or []
            if worker in WORKER_TYPES and any(str(item).lower() in text for item in matches):
                return worker

        task_routes = self.config.routing.get("task_type") or {}
        worker = str(task_routes.get(analysis.task_type) or "")
        if worker in WORKER_TYPES:
            return worker
        return "researcher"


class WorkerRunner:
    def __init__(self, config: WorkflowConfig, dry_run: bool = False):
        self.config = config
        self.dry_run = dry_run

    def run(
        self,
        subtask: Subtask,
        analysis: TaskAnalysis,
        previous_outputs: list[WorkerOutput],
    ) -> tuple[str, str, str]:
        agent_config = self.config.agent_for(subtask.worker_type or "researcher")
        provider = agent_config.display_provider
        model = agent_config.display_model

        if self.dry_run:
            return self._dry_run_output(subtask, analysis, previous_outputs), provider, model

        prompt = self._build_prompt(subtask, analysis, previous_outputs, agent_config)
        toolsets = (
            agent_config.toolsets
            if self.config.allow_file_changes and agent_config.can_write_files
            else []
        )
        try:
            from hermes_cli.oneshot import _run_agent

            output = _run_agent(
                prompt,
                model=agent_config.model or None,
                provider=agent_config.provider or None,
                toolsets=toolsets,
                use_config_toolsets=False,
            )
        except Exception as exc:
            raise WorkflowError(
                f"Worker {subtask.worker_type} failed using provider={provider}, "
                f"model={model}: {exc}"
            ) from exc
        return output.strip(), provider, model

    def revise(
        self,
        subtask: Subtask,
        analysis: TaskAnalysis,
        previous_outputs: list[WorkerOutput],
        first_output: str,
        evaluation: EvaluationResult,
    ) -> str:
        if self.dry_run:
            return first_output

        agent_config = self.config.agent_for(subtask.worker_type or "researcher")
        revision_prompt = (
            self._build_prompt(subtask, analysis, previous_outputs, agent_config)
            + "\n\nRevision required.\n"
            + f"Previous output:\n{first_output}\n\n"
            + f"Evaluator feedback:\n{json.dumps(evaluation.to_dict(), indent=2)}\n\n"
            + "Revise once. Keep the answer concise and address the missing criteria."
        )
        try:
            from hermes_cli.oneshot import _run_agent

            return _run_agent(
                revision_prompt,
                model=agent_config.model or None,
                provider=agent_config.provider or None,
                toolsets=[],
                use_config_toolsets=False,
            ).strip()
        except Exception as exc:
            raise WorkflowError(f"Revision failed for {subtask.id}: {exc}") from exc

    def _build_prompt(
        self,
        subtask: Subtask,
        analysis: TaskAnalysis,
        previous_outputs: list[WorkerOutput],
        agent_config: AgentConfig,
    ) -> str:
        template = _read_prompt(agent_config.prompt)
        return _render_prompt(
            template,
            {
                "worker_type": subtask.worker_type or agent_config.worker_type,
                "task_type": analysis.task_type,
                "complexity": analysis.complexity,
                "objective": subtask.objective,
                "input_context": subtask.input_context,
                "expected_output": subtask.expected_output,
                "acceptance_criteria": _format_list(subtask.acceptance_criteria),
                "risk_level": subtask.risk_level,
                "previous_outputs": _previous_output_context(previous_outputs),
            },
        )

    def _dry_run_output(
        self,
        subtask: Subtask,
        analysis: TaskAnalysis,
        previous_outputs: list[WorkerOutput],
    ) -> str:
        prior = "No prior worker outputs." if not previous_outputs else (
            f"Uses {len(previous_outputs)} prior worker output(s) as context."
        )
        return (
            f"Dry-run worker output for {subtask.worker_type}.\n"
            f"Task type: {analysis.task_type}; complexity: {analysis.complexity}.\n"
            f"Objective addressed: {subtask.objective}.\n"
            f"Expected output produced: {subtask.expected_output}.\n"
            f"Acceptance criteria addressed: {_format_list(subtask.acceptance_criteria)}\n"
            f"Risk note: {subtask.risk_level} risk. {prior}"
        )


class Evaluator:
    _UNSAFE_PATTERNS = (
        r"\brm\s+-rf\b",
        r"\bgit\s+reset\s+--hard\b",
        r"\bgit\s+checkout\s+--\b",
        r"\bprintenv\b.*(?:KEY|TOKEN|SECRET|PASSWORD)",
        r"\bcat\b.*(?:\.env|credentials|secrets?)",
    )

    def evaluate(self, subtask: Subtask, output: str) -> EvaluationResult:
        stripped = output.strip()
        if not stripped:
            return EvaluationResult(
                passed=False,
                score=0.0,
                missing_steps=list(subtask.acceptance_criteria),
                notes="Worker returned no output.",
            )

        lowered = stripped.lower()
        unsafe = [
            pattern for pattern in self._UNSAFE_PATTERNS if re.search(pattern, lowered)
        ]
        contradictions = self._detect_contradictions(lowered)
        unclear = []
        if "assume " in lowered and "assumption" not in lowered:
            unclear.append("Uses assumptions without clearly labelling them.")

        criterion_matches = 0
        missing: list[str] = []
        for criterion in subtask.acceptance_criteria:
            words = _keywords(criterion)
            if not words or any(word in lowered for word in words):
                criterion_matches += 1
            else:
                missing.append(criterion)

        objective_words = _keywords(subtask.objective)
        objective_hit = any(word in lowered for word in objective_words)
        score = 0.35
        score += 0.35 * (criterion_matches / max(1, len(subtask.acceptance_criteria)))
        score += 0.2 if objective_hit else 0.0
        score += 0.1 if not unsafe and not contradictions else 0.0
        score -= 0.2 * len(unsafe)
        score -= 0.15 * len(contradictions)
        score = max(0.0, min(1.0, score))

        passed = score >= 0.68 and not unsafe and not contradictions
        return EvaluationResult(
            passed=passed,
            score=round(score, 2),
            missing_steps=missing,
            contradictions=contradictions,
            unsafe_actions=unsafe,
            unclear_assumptions=unclear,
            notes="Heuristic MVP evaluator.",
        )

    def _detect_contradictions(self, lowered: str) -> list[str]:
        contradictions: list[str] = []
        if "file changes" in lowered and "no file changes" in lowered:
            contradictions.append("Mentions both file changes and no file changes.")
        if "background execution" in lowered and "no background execution" in lowered:
            contradictions.append(
                "Mentions both background execution and no background execution."
            )
        return contradictions


class FinalSynthesizer:
    def synthesize(
        self,
        analysis: TaskAnalysis,
        outputs: list[WorkerOutput],
        *,
        dry_run: bool,
    ) -> str:
        completed = [item for item in outputs if item.evaluation.passed]
        failed = [item for item in outputs if not item.evaluation.passed]
        high_risk = [item.subtask.id for item in outputs if item.subtask.risk_level == "high"]

        done_lines = [
            f"- {item.subtask.id} ({item.worker_type}): {item.subtask.objective}"
            for item in outputs
        ]
        risk_lines = []
        if failed:
            risk_lines.append(
                f"- {len(failed)} subtask(s) did not pass evaluator threshold."
            )
        if high_risk:
            risk_lines.append(f"- High-risk subtasks: {', '.join(high_risk)}.")
        if dry_run:
            risk_lines.append("- Dry-run mode did not call configured models.")
        if not risk_lines:
            risk_lines.append("- No evaluator-blocking risks were detected.")

        next_action = (
            "Review failed subtasks and rerun with clearer scope."
            if failed
            else "Use the synthesized plan as the next implementation step."
        )

        return "\n".join(
            [
                "What was done",
                *(done_lines or ["- No subtasks were produced."]),
                "",
                "Key decisions",
                f"- Classified request as {analysis.task_type} / {analysis.complexity}.",
                f"- Execution mode: {'dry-run' if dry_run else 'model-backed'}, sequential workers.",
                f"- Evaluator passed {len(completed)}/{len(outputs)} worker output(s).",
                "",
                "Risks",
                *risk_lines,
                "",
                "Next action",
                f"- {next_action}",
            ]
        )


class WorkflowLogger:
    def __init__(self, config: WorkflowConfig):
        self.path = self._resolve_path(config)

    def log(self, run_id: str, event: str, payload: dict[str, Any]) -> None:
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "run_id": run_id,
            "event": event,
            **_redact_payload(payload),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def _resolve_path(self, config: WorkflowConfig) -> Path:
        configured = str(config.logging.get("path") or "").strip()
        if configured:
            path = Path(configured).expanduser()
            if not path.is_absolute():
                path = config.path.parent / path
            return path
        try:
            from hermes_constants import get_hermes_home

            return get_hermes_home() / "logs" / "workflow_runs.jsonl"
        except Exception:
            return Path.home() / ".hermes" / "logs" / "workflow_runs.jsonl"


class WorkflowOrchestrator:
    def __init__(
        self,
        *,
        config: WorkflowConfig,
        dry_run: bool = False,
        max_subtasks: int | None = None,
    ):
        self.config = config
        self.dry_run = dry_run
        self.max_subtasks = max_subtasks or config.max_subtasks
        self.analyzer = TaskAnalyzer()
        self.planner = Planner(max_subtasks=self.max_subtasks)
        self.router = Router(config)
        self.runner = WorkerRunner(config, dry_run=dry_run)
        self.evaluator = Evaluator()
        self.synthesizer = FinalSynthesizer()
        self.logger = WorkflowLogger(config)

    def run(self, request: str) -> WorkflowRun:
        clean_request = request.strip()
        if not clean_request:
            raise WorkflowError("Workflow request cannot be empty.")

        run_id = f"wf-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        self.logger.log(run_id, "request", {"original_request": clean_request})

        analysis = self.analyzer.analyze(clean_request)
        self.logger.log(run_id, "analysis", analysis.to_dict())

        subtasks = self.planner.plan(clean_request, analysis)
        worker_outputs: list[WorkerOutput] = []

        for subtask in subtasks:
            worker_type = self.router.assign(subtask, analysis)
            subtask.worker_type = worker_type
            agent_config = self.config.agent_for(worker_type)
            self.logger.log(
                run_id,
                "worker_selected",
                {
                    "subtask_id": subtask.id,
                    "selected_worker": worker_type,
                    "provider": agent_config.display_provider,
                    "model": agent_config.display_model,
                },
            )

            output, provider, model = self.runner.run(subtask, analysis, worker_outputs)
            evaluation = self.evaluator.evaluate(subtask, output)
            revision_used = False
            if not evaluation.passed and self.config.max_revision_cycles > 0:
                output = self.runner.revise(
                    subtask, analysis, worker_outputs, output, evaluation
                )
                evaluation = self.evaluator.evaluate(subtask, output)
                revision_used = True

            worker_output = WorkerOutput(
                subtask=subtask,
                worker_type=worker_type,
                provider=provider,
                model=model,
                output=output,
                evaluation=evaluation,
                revision_used=revision_used,
            )
            worker_outputs.append(worker_output)
            self.logger.log(
                run_id,
                "worker_result",
                {
                    "subtask_id": subtask.id,
                    "selected_worker": worker_type,
                    "provider": provider,
                    "model": model,
                    "result": output,
                    "revision_used": revision_used,
                },
            )
            self.logger.log(
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

        final_response = self.synthesizer.synthesize(
            analysis, worker_outputs, dry_run=self.dry_run
        )
        self.logger.log(run_id, "final", {"result": final_response})
        return WorkflowRun(
            run_id=run_id,
            original_request=clean_request,
            analysis=analysis,
            subtasks=subtasks,
            worker_outputs=worker_outputs,
            final_response=final_response,
            log_path=self.logger.path,
        )


def _read_prompt(prompt_ref: str) -> str:
    path = Path(prompt_ref)
    if not path.is_absolute():
        path = package_root() / prompt_ref
    if not path.exists():
        path = package_root() / "prompts" / "worker_researcher.md"
    return path.read_text(encoding="utf-8")


def _render_prompt(template: str, values: dict[str, Any]) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{{ " + key + " }}", str(value))
        rendered = rendered.replace("{{" + key + "}}", str(value))
    return rendered


def _format_list(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def _previous_output_context(outputs: list[WorkerOutput]) -> str:
    if not outputs:
        return "No previous worker outputs."
    chunks = []
    for item in outputs:
        chunks.append(
            f"{item.subtask.id} ({item.worker_type}, score={item.evaluation.score}):\n"
            f"{_truncate(item.output, 900)}"
        )
    return "\n\n".join(chunks)


def _keywords(text: str) -> list[str]:
    return [
        word
        for word in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{3,}", text.lower())
        if word not in _STOPWORDS
    ][:8]


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 20)].rstrip() + "\n...[truncated]"


def _redact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        from agent.redact import redact_sensitive_text
    except Exception:
        redact_sensitive_text = None

    def redact(value: Any) -> Any:
        if isinstance(value, str):
            return redact_sensitive_text(value) if redact_sensitive_text else value
        if isinstance(value, list):
            return [redact(item) for item in value]
        if isinstance(value, dict):
            return {str(k): redact(v) for k, v in value.items()}
        return value

    return {str(key): redact(value) for key, value in payload.items()}

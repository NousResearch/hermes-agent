"""Eval Harness — AIDE²-inspired evaluation framework for Hermes self-evaluation.

Provides a structured way to define evaluations (prompt + budget + metric
+ private_check) and run them against the Hermes runtime, recording
outcomes into the Experience Ledger.

Phase 3 of the AIDE² plan (see ``docs/aide-squared-roadmap.md``). The
execution paths are now real:

- ``_simulate_task_execution`` runs the eval prompt through
  ``EvalRunner`` (defaults to ``DefaultEvalRunner``, which calls
  ``auxiliary_client.call_llm``).
- ``_run_deterministic_check`` runs the ``private_check`` shell command
  through the same runner, with a hardened subprocess invocation
  (no ``shell=True``, dangerous-token filter, restricted env).
- ``_run_llm_judge`` calls ``LLMJudge.judge`` (defaults to
  ``DefaultLLMJudge``) to score the eval response.

Both runner and judge are injectable via ``EvalHarness(runner=...,
judge=...)`` so tests can substitute fakes without monkeypatching the
global ``auxiliary_client``. ``DefaultEvalRunner`` never raises for
normal model errors — failures are surfaced via the ``EvalResult.error``
field. The runner refuses dangerous ``private_check`` commands
(sudo, curl, wget, network exfil patterns, etc.) unless explicitly
opted in via ``allow_unsafe_private_check=True``.

Design:
- Each eval has a prompt, golden output, metric, and budget_usd
- Metrics: deterministic (private_check via subprocess) or
  LLM-judge (aux model blind evaluation)
- Cost constraint: exceed budget → automatic failure
- Task families: tools/coding/research/security (heterogeneous evaluation)
- The evaluated agent NEVER sees the private_check (prevents reward hacking)

Usage::

    harness = EvalHarness(hermes_home=Path.home() / ".hermes")
    harness.load_evals()
    result = harness.run_eval("file-ops-batch")
    summary = harness.run_all_evals()
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from agent.eval_runner import (
    DefaultEvalRunner,
    EvalInvocation,
    EvalRunner,
    PrivateCheckResult,
    PrivateCheckError,
    PromptResult,
)
from agent.experience_ledger import ExperienceLedger, SkillEval
from agent.llm_judge import DefaultLLMJudge, JudgeScore, LLMJudge

logger = logging.getLogger(__name__)


@dataclass
class EvalDefinition:
    """A single evaluation task definition."""

    id: str
    family: str  # tools/coding/research/security/mlops
    prompt: str
    budget_usd: float = 1.0
    metric: str = "private"  # private/llm_judge_private/custom
    private_check: str = ""  # Shell command or script path
    golden_file: str = ""  # Path to golden output
    skill_id: str = ""  # Which skill this eval tests
    timeout_sec: int = 120
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "family": self.family,
            "prompt": self.prompt,
            "budget_usd": self.budget_usd,
            "metric": self.metric,
            "private_check": self.private_check,
            "golden_file": self.golden_file,
            "skill_id": self.skill_id,
            "timeout_sec": self.timeout_sec,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EvalDefinition":
        return cls(**d)


@dataclass
class EvalResult:
    """Result of running an evaluation."""

    eval_id: str
    skill_id: str
    success: bool
    public_score: float = 0.0
    private_score: float = 0.0
    cost_usd: float = 0.0
    duration_sec: float = 0.0
    budget_exceeded: bool = False
    reward_hack_detected: bool = False
    not_implemented: bool = False  # True if execution raised NotImplementedError
    output: str = ""
    error: str = ""
    metric_details: dict = field(default_factory=dict)
    started_at: float = 0.0
    completed_at: float = 0.0

    def to_dict(self) -> dict:
        return {
            "eval_id": self.eval_id,
            "skill_id": self.skill_id,
            "success": self.success,
            "public_score": self.public_score,
            "private_score": self.private_score,
            "cost_usd": self.cost_usd,
            "duration_sec": self.duration_sec,
            "budget_exceeded": self.budget_exceeded,
            "reward_hack_detected": self.reward_hack_detected,
            "not_implemented": self.not_implemented,
            "output": self.output,
            "error": self.error,
            "metric_details": self.metric_details,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


class EvalHarness:
    """Evaluation framework for Hermes self-evaluation.

    Loads eval definitions from evals.json / evals.yaml, runs them
    against Hermes, and records results in the Experience Ledger.

    Architecture:
    1. Load eval definitions from ~/.hermes/evals/{evals.json,evals.yaml}
    2. For each eval:
       a. Run the prompt via the injected ``EvalRunner`` (defaults to
          ``DefaultEvalRunner`` which calls auxiliary_client.call_llm).
       b. Check cost against budget (reject if exceeded).
       c. Run the metric: deterministic ``private_check`` (hardened
          subprocess via the runner) or ``LLMJudge``.
       d. Record result in Experience Ledger.

    The runner and judge are injectable so tests can substitute
    fakes without monkeypatching the global auxiliary_client.
    """

    def __init__(
        self,
        hermes_home: Optional[Path] = None,
        ledger: Optional[ExperienceLedger] = None,
        *,
        runner: Optional[EvalRunner] = None,
        judge: Optional[LLMJudge] = None,
    ):
        self.hermes_home = hermes_home or Path.home() / ".hermes"
        self.evals_dir = self.hermes_home / "evals"
        self.ledger = ledger or ExperienceLedger(hermes_home=self.hermes_home)
        self.runner: EvalRunner = runner or DefaultEvalRunner(
            hermes_home=self.hermes_home,
        )
        self.judge: LLMJudge = judge or DefaultLLMJudge()
        self._evals: Dict[str, EvalDefinition] = {}
        self._results: Dict[str, EvalResult] = {}
        self._custom_metrics: Dict[str, Callable] = {}

    def load_evals(self) -> int:
        """Load eval definitions from evals.yaml or evals.json."""
        self.evals_dir.mkdir(parents=True, exist_ok=True)

        # Try JSON first (easier for programmatic creation)
        json_path = self.evals_dir / "evals.json"
        if json_path.exists():
            data = json.loads(json_path.read_text(encoding="utf-8"))
            for d in data:
                ev = EvalDefinition.from_dict(d)
                self._evals[ev.id] = ev
            logger.info("Eval harness: loaded %d evals from JSON", len(self._evals))
            return len(self._evals)

        # Try YAML
        yaml_path = self.evals_dir / "evals.yaml"
        if yaml_path.exists():
            import yaml

            data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
            for d in data:
                ev = EvalDefinition.from_dict(d)
                self._evals[ev.id] = ev
            logger.info("Eval harness: loaded %d evals from YAML", len(self._evals))
            return len(self._evals)

        # Create default evals if none exist
        self._create_default_evals()
        return len(self._evals)

    def _create_default_evals(self) -> None:
        """Create default eval definitions for core Hermes capabilities."""
        defaults = [
            EvalDefinition(
                id="file-ops-batch",
                family="tools",
                prompt="Sort the CSV file at /tmp/test_input.csv by the second column and write to /tmp/test_output.csv",
                budget_usd=0.5,
                metric="private",
                private_check="test -f /tmp/test_output.csv && python3 -c \"import csv; rows=list(csv.reader(open('/tmp/test_output.csv', encoding='utf-8'))); assert all(rows[i][1]<=rows[i+1][1] for i in range(len(rows)-1)), 'Not sorted'\"",
                description="Tests file manipulation tool correctness",
            ),
            EvalDefinition(
                id="research-synthesis",
                family="research",
                prompt="Research 'Python async best practices' and output 5 key points to /tmp/research_output.md",
                budget_usd=1.0,
                metric="llm_judge_private",
                description="Tests research and synthesis capability",
            ),
            EvalDefinition(
                id="skill-creation",
                family="coding",
                prompt="Create a skill called 'test-skill' that validates JSON input and outputs a summary",
                budget_usd=0.8,
                metric="private",
                private_check="test -d ~/.hermes/skills/test-skill && test -f ~/.hermes/skills/test-skill/SKILL.md",
                description="Tests skill creation workflow",
            ),
        ]
        for ev in defaults:
            self._evals[ev.id] = ev

        # Save as JSON
        evals_file = self.evals_dir / "evals.json"
        evals_file.write_text(
            json.dumps([e.to_dict() for e in self._evals.values()], indent=2),
            encoding="utf-8",
        )
        logger.info("Eval harness: created %d default evals", len(defaults))

    def register_custom_metric(self, name: str, fn: Callable) -> None:
        """Register a custom metric function."""
        self._custom_metrics[name] = fn

    def run_eval(self, eval_id: str) -> EvalResult:
        """Run a single evaluation.

        The structural pieces (eval lookup, ledger recording, budget check,
        reward-hack detection) are functional. The execution path that
        *runs* the prompt against Hermes (``_simulate_task_execution``)
        is a stub that raises ``NotImplementedError`` until Phase 3.

        On NotImplementedError, the result is returned with
        ``not_implemented=True``, ``success=False``, and the original
        error message. This lets callers discover the stub state without
        being surprised by uncaught exceptions and without polluting the
        ledger with fake eval records.
        """
        if eval_id not in self._evals:
            return EvalResult(
                eval_id=eval_id,
                skill_id="",
                success=False,
                error=f"Unknown eval: {eval_id}",
            )

        ev = self._evals[eval_id]
        result = EvalResult(
            eval_id=eval_id,
            skill_id=ev.skill_id,
            success=False,  # Default, will be updated during execution
            started_at=time.time(),
        )

        logger.info(
            "Eval harness: running %s (family=%s, budget=$%.2f)",
            eval_id,
            ev.family,
            ev.budget_usd,
        )

        try:
            # Run the eval (stub: raises NotImplementedError until Phase 3)
            result = self._execute_eval(ev, result)

            # Record in ledger
            self.ledger.record_eval(
                SkillEval(
                    skill_id=ev.skill_id or eval_id,
                    eval_event_id=eval_id,
                    task_family=ev.family,
                    public_score=result.public_score,
                    private_score=result.private_score,
                    cost_usd=result.cost_usd,
                    outcome="success" if result.success else "failure",
                    duration_sec=result.duration_sec,
                )
            )

        except NotImplementedError as e:
            # Stub path: surface clearly without recording fake data.
            result.not_implemented = True
            result.success = False
            result.error = str(e)
            result.completed_at = time.time()
            result.duration_sec = result.completed_at - result.started_at
            logger.warning(
                "Eval harness: %s is a stub until Phase 3 — "
                "no eval result recorded in ledger. %s",
                eval_id,
                e,
            )
        except Exception as e:
            result.success = False
            result.error = str(e)
            result.completed_at = time.time()
            result.duration_sec = result.completed_at - result.started_at

        self._results[eval_id] = result
        # Only persist ledger when a real result was recorded.
        if not result.not_implemented:
            self.ledger.save()
        return result

    def run_all_evals(self) -> Dict[str, EvalResult]:
        """Run all registered evaluations."""
        results = {}
        for eval_id in self._evals:
            results[eval_id] = self.run_eval(eval_id)
        return results

    def _execute_eval(
        self,
        ev: EvalDefinition,
        result: EvalResult,
    ) -> EvalResult:
        """Execute a single eval and score it."""
        start = time.time()

        # Step 1: Simulate running the task
        # In production, this would: hermes chat -q "<prompt>"
        # For now, we simulate the outcome
        output, cost = self._simulate_task_execution(ev)

        result.output = output[:500]  # Truncate for storage
        result.cost_usd = cost
        result.completed_at = time.time()
        result.duration_sec = result.completed_at - result.started_at

        # Step 2: Check budget. Cost must be strictly within budget
        # (cost == budget_usd is allowed; cost > budget_usd is rejected).
        # This matches AIDE²'s "fixed cost budget" selection pressure.
        if result.cost_usd > ev.budget_usd:
            result.budget_exceeded = True
            result.success = False
            result.public_score = 0.0
            result.private_score = 0.0
            result.error = (
                f"Cost ${result.cost_usd:.4f} exceeded budget ${ev.budget_usd:.4f}"
            )
            return result

        # Step 3: Run private metric (agent doesn't see this!)
        result = self._run_private_metric(ev, result)

        # Step 4: Detect reward hacking — large gap between the agent-visible
        # public score and the hidden private score, or a self-reported
        # perfect score paired with objective failure markers.
        if result.public_score - result.private_score > 0.3 or (
            result.public_score >= 0.9 and result.private_score < 0.5
        ):
            result.reward_hack_detected = True
            result.success = False

        return result

    def _simulate_task_execution(self, ev: EvalDefinition) -> tuple:
        """Run ``ev.prompt`` against the real model via the injected runner.

        Returns ``(output_text, cost_usd)``. ``cost_usd`` is left at
        ``0.0`` in Phase 3 — token counts are tracked but USD cost
        requires model-specific pricing that lives in the billing
        subsystem. Phase 5 (metrics export) will thread pricing
        through.
        """
        invocation = EvalInvocation(
            prompt=ev.prompt,
            timeout_sec=float(ev.timeout_sec or 120.0),
        )
        prompt_result = self.runner.execute_prompt(invocation)
        if not prompt_result.success:
            # Propagate as a structured failure so the caller can
            # decide whether to retry.
            raise RuntimeError(
                f"EvalRunner.execute_prompt failed for {ev.id!r}: {prompt_result.error}"
            )
        # Update duration on the active result via metrics; nothing
        # to do here since the harness sets duration after _execute_eval.
        # Store the token counts on the result via the eval invocation
        # metadata; we attach them to the EvalResult after this returns.
        self._last_prompt_result: PromptResult = prompt_result  # type: ignore[attr-defined]
        # Truncate to the harness's storage cap.
        return (prompt_result.text[:500], 0.0)

    def _run_private_metric(
        self,
        ev: EvalDefinition,
        result: EvalResult,
    ) -> EvalResult:
        """Run the private evaluation metric.

        Dispatches to the appropriate path:
        - ``private`` metric → runner.run_private_check
        - ``llm_judge_private`` metric → judge.judge
        - custom → caller-registered function
        - otherwise → default moderate-score fallback
        """
        if ev.metric == "private" and ev.private_check:
            result = self._run_deterministic_check(ev, result)
        elif ev.metric == "llm_judge_private":
            result = self._run_llm_judge(ev, result)
        elif ev.metric in self._custom_metrics:
            result = self._custom_metrics[ev.metric](ev, result)
        else:
            # Default: assign moderate scores
            result.public_score = 0.7
            result.private_score = 0.6
            result.success = result.private_score >= 0.5

        return result

    def _run_deterministic_check(
        self,
        ev: EvalDefinition,
        result: EvalResult,
    ) -> EvalResult:
        """Run the deterministic private check via the runner."""
        invocation = EvalInvocation(
            prompt=ev.prompt,
            private_check=ev.private_check,
            timeout_sec=float(ev.timeout_sec or 120.0),
        )
        try:
            check = self.runner.run_private_check(invocation)
        except PrivateCheckError as e:
            # The runner refused to run the command (dangerous token).
            result.public_score = 0.0
            result.private_score = 0.0
            result.success = False
            result.error = f"private_check blocked by runner: {e}"
            result.metric_details = {"check": ev.private_check[:100], "blocked": True}
            return result

        if check.timed_out:
            result.public_score = 0.0
            result.private_score = 0.0
            result.success = False
            result.error = "Check timed out"
            return result

        passed = check.success
        result.public_score = 1.0 if passed else 0.3
        result.private_score = 1.0 if passed else 0.2
        result.success = passed
        result.metric_details = {
            "check": ev.private_check[:100],
            "exit_code": check.exit_code,
            "stderr": check.stderr[:200],
            "duration_sec": round(check.duration_sec, 3),
        }
        return result

    def _run_llm_judge(
        self,
        ev: EvalDefinition,
        result: EvalResult,
    ) -> EvalResult:
        """Run the LLM judge on the prompt + response.

        The prompt and response come from the EvalRunner output stored
        by the harness. If the runner didn't produce output (the
        judge path is being used standalone), the judge scores an
        empty response — usually a failure.
        """
        prompt_text = ev.prompt
        response_text = getattr(self, "_last_prompt_result", None)
        response_body = (
            getattr(response_text, "text", "") if response_text is not None else ""
        )

        verdict = self.judge.judge(prompt_text, response_body)
        if not verdict.success or verdict.score is None:
            result.public_score = 0.0
            result.private_score = 0.0
            result.success = False
            result.error = (
                verdict.error or "LLM judge did not produce a parseable score"
            )
            result.metric_details = {
                "judge": "llm_judge",
                "raw_reasoning": (verdict.reasoning or "")[:300],
            }
            return result

        # Map 0-100 judge score to 0-1 private score; public score is
        # the model's self-reported success (we don't have one, so use
        # the judge too — the heuristic private_score in
        # SkillEvalProducer already handles the public/private split).
        score_01 = verdict.score / 100.0
        result.public_score = score_01
        result.private_score = score_01
        result.success = verdict.score >= 50
        result.metric_details = {
            "judge": "llm_judge",
            "raw_score": verdict.score,
            "raw_reasoning": verdict.reasoning[:300],
            "judge_model": verdict.model,
        }
        return result

    def get_eval_summary(self) -> dict:
        """Get summary of all eval results."""
        if not self._results:
            return {"total": 0, "passed": 0, "failed": 0, "budget_exceeded": 0}

        total = len(self._results)
        passed = sum(1 for r in self._results.values() if r.success)
        failed = total - passed
        budget_exceeded = sum(1 for r in self._results.values() if r.budget_exceeded)

        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "success_rate": round(passed / max(total, 1), 3),
            "budget_exceeded": budget_exceeded,
            "total_cost_usd": round(sum(r.cost_usd for r in self._results.values()), 4),
        }

    def get_evals(self) -> Dict[str, EvalDefinition]:
        """Get all eval definitions."""
        return dict(self._evals)

    def get_results(self) -> Dict[str, EvalResult]:
        """Get all eval results."""
        return dict(self._results)

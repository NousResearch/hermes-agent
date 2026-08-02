"""Eval Harness — AIDE²-inspired evaluation framework for Hermes self-improvement.

Provides a structured way to evaluate Hermes skills, configs, and behaviors
against objective metrics. Inspired by AIDE²'s fixed-cost budget and
heterogeneous task evaluation protocol.

Key design:
- Each eval has a prompt, golden output, metric, and budget_usd
- Metrics: deterministic (diff/golden) or LLM-judge (aux model blind evaluation)
- Cost constraint: exceed budget → automatic failure
- Task families: tools/coding/research/security (heterogeneous evaluation)
- The evaluated agent NEVER sees the private_check (prevents reward hacking)

Usage:
    harness = EvalHarness(hermes_home=Path.home() / ".hermes")
    results = harness.run_eval("file-ops-batch")
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

from agent.experience_ledger import ExperienceLedger, SkillEval

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
            "output": self.output,
            "error": self.error,
            "metric_details": self.metric_details,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


class EvalHarness:
    """Evaluation framework for Hermes self-improvement.

    Loads eval definitions from evals.yaml, runs them against Hermes,
    and records results in the Experience Ledger.

    Architecture:
    1. Load eval definitions from ~/.hermes/evals/evals.yaml
    2. For each eval:
       a. Run Hermes with the eval prompt (isolated session)
       b. Check cost against budget (reject if exceeded)
       c. Run private_check (deterministic or LLM judge)
       d. Record result in Experience Ledger
    """

    def __init__(
        self,
        hermes_home: Optional[Path] = None,
        ledger: Optional[ExperienceLedger] = None,
    ):
        self.hermes_home = hermes_home or Path.home() / ".hermes"
        self.evals_dir = self.hermes_home / "evals"
        self.ledger = ledger or ExperienceLedger(hermes_home=self.hermes_home)
        self._evals: Dict[str, EvalDefinition] = {}
        self._results: Dict[str, EvalResult] = {}
        self._custom_metrics: Dict[str, Callable] = {}

    def load_evals(self) -> int:
        """Load eval definitions from evals.yaml or evals.json."""
        self.evals_dir.mkdir(parents=True, exist_ok=True)

        # Try JSON first (easier for programmatic creation)
        json_path = self.evals_dir / "evals.json"
        if json_path.exists():
            data = json.loads(json_path.read_text())
            for d in data:
                ev = EvalDefinition.from_dict(d)
                self._evals[ev.id] = ev
            logger.info("Eval harness: loaded %d evals from JSON", len(self._evals))
            return len(self._evals)

        # Try YAML
        yaml_path = self.evals_dir / "evals.yaml"
        if yaml_path.exists():
            import yaml
            data = yaml.safe_load(yaml_path.read_text())
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
                private_check="test -f /tmp/test_output.csv && python3 -c \"import csv; rows=list(csv.reader(open('/tmp/test_output.csv'))); assert all(rows[i][1]<=rows[i+1][1] for i in range(len(rows)-1)), 'Not sorted'\"",
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
            json.dumps([e.to_dict() for e in self._evals.values()], indent=2)
        )
        logger.info("Eval harness: created %d default evals", len(defaults))

    def register_custom_metric(self, name: str, fn: Callable) -> None:
        """Register a custom metric function."""
        self._custom_metrics[name] = fn

    def run_eval(self, eval_id: str) -> EvalResult:
        """Run a single evaluation."""
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
            eval_id, ev.family, ev.budget_usd,
        )

        try:
            # Run the eval (simulated — real implementation would invoke Hermes)
            result = self._execute_eval(ev, result)

            # Record in ledger
            self.ledger.record_eval(SkillEval(
                skill_id=ev.skill_id or eval_id,
                eval_event_id=eval_id,
                task_family=ev.family,
                public_score=result.public_score,
                private_score=result.private_score,
                cost_usd=result.cost_usd,
                outcome="success" if result.success else "failure",
                duration_sec=result.duration_sec,
            ))

        except Exception as e:
            result.success = False
            result.error = str(e)
            result.completed_at = time.time()
            result.duration_sec = result.completed_at - result.started_at

        self._results[eval_id] = result
        self.ledger.save()
        return result

    def run_all_evals(self) -> Dict[str, EvalResult]:
        """Run all registered evaluations."""
        results = {}
        for eval_id in self._evals:
            results[eval_id] = self.run_eval(eval_id)
        return results

    def _execute_eval(
        self, ev: EvalDefinition, result: EvalResult,
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

        # Step 2: Check budget
        if result.cost_usd > ev.budget_usd:
            result.budget_exceeded = True
            result.success = False
            result.public_score = 0.0
            result.private_score = 0.0
            return result

        # Step 3: Run private metric (agent doesn't see this!)
        result = self._run_private_metric(ev, result)

        # Step 4: Detect reward hacking
        if result.public_score > 0.8 and result.private_score < 0.5:
            result.reward_hack_detected = True
            result.success = False

        return result

    def _simulate_task_execution(self, ev: EvalDefinition) -> tuple:
        """Simulate task execution for testing.

        In production, this would call the actual Hermes gateway.
        """
        # Default simulation: partial success with moderate cost
        import random
        success = random.random() > 0.3  # 70% success rate
        cost = random.uniform(0.05, ev.budget_usd * 0.8)

        if success:
            output = f"Task '{ev.id}' completed successfully."
        else:
            output = f"Task '{ev.id}' completed with issues."

        return output, cost

    def _run_private_metric(
        self, ev: EvalDefinition, result: EvalResult,
    ) -> EvalResult:
        """Run the private evaluation metric."""
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
        self, ev: EvalDefinition, result: EvalResult,
    ) -> EvalResult:
        """Run a deterministic shell check."""
        try:
            proc = subprocess.run(
                ev.private_check,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            passed = proc.returncode == 0
            result.public_score = 1.0 if passed else 0.3
            result.private_score = 1.0 if passed else 0.2
            result.success = passed
            result.metric_details = {
                "check": ev.private_check[:100],
                "exit_code": proc.returncode,
                "stderr": proc.stderr[:200] if proc.stderr else "",
            }
        except subprocess.TimeoutExpired:
            result.public_score = 0.0
            result.private_score = 0.0
            result.success = False
            result.error = "Check timed out"

        return result

    def _run_llm_judge(
        self, ev: EvalDefinition, result: EvalResult,
    ) -> EvalResult:
        """Run an LLM judge for subjective evaluation.

        Uses auxiliary model for blind evaluation (agent doesn't see prompt).
        """
        # In production, this would call auxiliary_client.py
        # For now, simulate with moderate scores
        import random
        score = random.uniform(0.5, 0.9)
        result.public_score = min(score + random.uniform(-0.1, 0.1), 1.0)
        result.private_score = score
        result.success = score >= 0.5
        result.metric_details = {"judge": "simulated_llm", "raw_score": round(score, 3)}
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

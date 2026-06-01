"""swe_bench_adapter.py — SWE-bench integration for the HITL harness.

Brings real-world GitHub issue resolution into the fitness function.
The harness can now include "SWE-bench resolution rate" as a metric component.

Usage:
    from agent.hitl_harness.swe_bench_adapter import SWEBenchAdapter
    adapter = SWEBenchAdapter(repo_path="/path/to/repo", dataset="princeton-nlp/SWE-bench_Lite")
    result = adapter.evaluate_issue(issue_id, agent_or_harness)
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SWEBenchResult:
    """Result of evaluating a single SWE-bench-style issue."""

    issue_id: str
    resolved: bool
    patch_generated: bool
    tests_pass: bool
    lint_pass: bool
    steps_taken: int
    time_seconds: float
    error_message: str = ""
    trajectory: List[Dict[str, Any]] = field(default_factory=list)


class SWEBenchAdapter:
    """Adapter that runs SWE-bench-style tasks using Hermes execution environments."""

    def __init__(
        self,
        repo_path: Optional[str] = None,
        dataset: str = "princeton-nlp/SWE-bench_Lite",
        env_type: str = "local",
        max_iterations: int = 20,
    ):
        self.repo_path = Path(repo_path).expanduser() if repo_path else None
        self.dataset = dataset
        self.env_type = env_type
        self.max_iterations = max_iterations
        self._env = None

    def _get_env(self):
        """Lazy-load the execution environment."""
        if self._env is None:
            try:
                from tools.environments import create_environment
                self._env = create_environment(env_type=self.env_type)
            except Exception as e:
                logger.warning("Could not create %s environment: %s", self.env_type, e)
                self._env = _LocalFallbackEnv()
        return self._env

    def evaluate_issue(
        self,
        issue_id: str,
        issue_description: str,
        harness: Any,
    ) -> SWEBenchResult:
        """Run the agent against a single SWE-bench issue.

        Args:
            issue_id: Unique issue identifier
            issue_description: The GitHub issue text
            harness: HITLHarness instance (or agent with hitl_harness attached)

        Returns:
            SWEBenchResult with resolution status
        """
        import time
        start = time.time()

        env = self._get_env()
        steps = 0
        resolved = False
        patch_generated = False
        tests_pass = False
        lint_pass = False
        error_msg = ""
        trajectory: List[Dict[str, Any]] = []

        try:
            # Step 1: Set up the repo if provided
            if self.repo_path and self.repo_path.exists():
                env.execute(f"cd {self.repo_path}")

            # Step 2: Run the agent with the issue as the task
            prompt = (
                f"Fix the following GitHub issue:\n\n"
                f"Issue ID: {issue_id}\n"
                f"{issue_description}\n\n"
                f"Requirements:\n"
                f"1. Reproduce the issue first\n"
                f"2. Make minimal changes to fix it\n"
                f"3. Run tests to verify the fix\n"
                f"4. Run lint/type checks\n"
                f"5. Return a git diff of your changes"
            )

            # If harness has an agent, use it; otherwise we can't run
            agent = getattr(harness, "agent", None)
            if agent and hasattr(agent, "run_conversation"):
                result = agent.run_conversation(prompt)
                steps = getattr(agent, "api_call_count", 0)

                # Extract the response
                response_text = ""
                if isinstance(result, dict):
                    response_text = result.get("final_response", "")
                elif isinstance(result, str):
                    response_text = result

                trajectory.append({"role": "user", "content": prompt})
                trajectory.append({"role": "assistant", "content": response_text[:2000]})

                # Check if a patch was generated (simple heuristic)
                patch_generated = "diff --git" in response_text or "@@" in response_text

                # Try to apply and test the patch
                if patch_generated:
                    tests_pass = self._run_tests(env)
                    lint_pass = self._run_lint(env)
                    resolved = tests_pass and lint_pass
            else:
                error_msg = "No agent available in harness for SWE-bench evaluation"

        except Exception as e:
            error_msg = str(e)
            logger.exception("SWE-bench evaluation failed for issue %s", issue_id)

        elapsed = time.time() - start

        return SWEBenchResult(
            issue_id=issue_id,
            resolved=resolved,
            patch_generated=patch_generated,
            tests_pass=tests_pass,
            lint_pass=lint_pass,
            steps_taken=steps,
            time_seconds=elapsed,
            error_message=error_msg,
            trajectory=trajectory,
        )

    def evaluate_corpus(
        self,
        issues: List[Dict[str, str]],
        harness: Any,
    ) -> Dict[str, Any]:
        """Evaluate multiple issues and return aggregate metrics.

        Args:
            issues: List of {"issue_id": "...", "description": "..."} dicts
            harness: HITLHarness instance

        Returns:
            Dict with resolution_rate, avg_steps, avg_time, per_issue results
        """
        results: List[SWEBenchResult] = []
        for issue in issues:
            result = self.evaluate_issue(
                issue_id=issue["issue_id"],
                issue_description=issue["description"],
                harness=harness,
            )
            results.append(result)
            logger.info(
                "Issue %s: resolved=%s steps=%d time=%.1fs",
                result.issue_id, result.resolved, result.steps_taken, result.time_seconds,
            )

        resolved_count = sum(1 for r in results if r.resolved)
        total = len(results)
        resolution_rate = resolved_count / total if total > 0 else 0.0

        return {
            "resolution_rate": round(resolution_rate, 4),
            "resolved_count": resolved_count,
            "total_issues": total,
            "avg_steps": round(sum(r.steps_taken for r in results) / total, 2) if total > 0 else 0,
            "avg_time_seconds": round(sum(r.time_seconds for r in results) / total, 2) if total > 0 else 0,
            "per_issue": [
                {
                    "issue_id": r.issue_id,
                    "resolved": r.resolved,
                    "patch_generated": r.patch_generated,
                    "tests_pass": r.tests_pass,
                    "lint_pass": r.lint_pass,
                    "steps": r.steps_taken,
                    "time": r.time_seconds,
                    "error": r.error_message,
                }
                for r in results
            ],
        }

    def _run_tests(self, env) -> bool:
        """Run the test suite in the environment."""
        try:
            result = env.execute("python -m pytest --tb=short -q", timeout=120)
            return result.get("returncode", 1) == 0
        except Exception:
            return False

    def _run_lint(self, env) -> bool:
        """Run linting in the environment."""
        try:
            result = env.execute("python -m py_compile $(git diff --name-only)", timeout=60)
            return result.get("returncode", 1) == 0
        except Exception:
            return False


class _LocalFallbackEnv:
    """Minimal fallback environment when the full environment system isn't available."""

    def execute(self, command: str, timeout: int = 60) -> Dict[str, Any]:
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return {
                "output": result.stdout + result.stderr,
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {"output": "Command timed out", "returncode": -1}
        except Exception as e:
            return {"output": str(e), "returncode": -1}

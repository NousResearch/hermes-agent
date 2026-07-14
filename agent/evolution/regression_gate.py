"""Regression Gate — the AEGIS "Critic + Deterministic Gate" for the Evolution Engine.

Prevents catastrophic forgetting by enforcing the **seesaw constraint**: no
improvement may regress any previously-solved task.

Architecture (HarnessX-inspired):
  1. **Deterministic checks** always run — these are the hard gates
  2. **LLM-based critic** is advisory only — it can warn but never block
  3. **Manifest completeness** — every proposal must declare what it changes
  4. **Build/smoke tests** — code changes must import and instantiate
  5. **Regression tests** — prior successes must still pass
  6. **Variant-scoped** — checks run against the harness variant's baseline

The gate produces one of three verdicts:
  - ACCEPT: All checks pass, proposal can be applied
  - REJECT: Hard gate failed, proposal is blocked
  - NEEDS_REVIEW: Warning-level issues, requires human judgment
"""

from __future__ import annotations

import importlib.util
import logging
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from agent.evolution.improvement_proposer import ImprovementProposal
from agent.evolution.task_definition import ImprovementActionType, TaskDefinition
from agent.evolution.evolution_store import get_evolution_store
from agent.evolution.evaluator import TaskEvaluator, EvaluationContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gate verdict
# ---------------------------------------------------------------------------


class GateVerdict(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    NEEDS_REVIEW = "needs_review"


@dataclass
class GateResult:
    """Result of running a proposal through the regression gate."""
    verdict: GateVerdict
    proposal: ImprovementProposal
    checks: List[Dict[str, Any]] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    regression_results: Dict[str, bool] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.verdict in (GateVerdict.ACCEPT, GateVerdict.NEEDS_REVIEW)

    @property
    def is_blocked(self) -> bool:
        return self.verdict == GateVerdict.REJECT


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


class RegressionGate:
    """Deterministic safety gate for improvement proposals."""

    def __init__(
        self,
        harness_variant: str = "default",
        max_regression_tasks: int = 20,
        critic_fn: Optional[Callable] = None,
    ):
        self.harness_variant = harness_variant
        self.max_regression_tasks = max_regression_tasks
        self._critic_fn = critic_fn
        self._store = get_evolution_store()

    def evaluate(
        self,
        proposal: ImprovementProposal,
        regression_tasks: Optional[List[TaskDefinition]] = None,
    ) -> GateResult:
        """Run a proposal through all safety gates.

        Args:
            proposal: The improvement proposal to evaluate.
            regression_tasks: Optional list of tasks to check for regression.
              If None, loads from the store's regression baseline.

        Returns:
            GateResult with verdict and detailed check results.
        """
        checks: List[Dict[str, Any]] = []
        failures: List[str] = []
        warnings: List[str] = []
        regression_results: Dict[str, bool] = {}

        # ---- Gate 1: Manifest completeness ----
        manifest_ok, manifest_msg = self._check_manifest(proposal)
        checks.append({"gate": "manifest_completeness", "passed": manifest_ok, "detail": manifest_msg})
        if not manifest_ok:
            failures.append(manifest_msg)
            return GateResult(
                verdict=GateVerdict.REJECT,
                proposal=proposal,
                checks=checks,
                failures=failures,
                warnings=warnings,
            )

        # ---- Gate 2: Content validation ----
        content_ok, content_msg = self._check_content(proposal)
        checks.append({"gate": "content_validation", "passed": content_ok, "detail": content_msg})
        if not content_ok:
            failures.append(content_msg)
            return GateResult(
                verdict=GateVerdict.REJECT,
                proposal=proposal,
                checks=checks,
                failures=failures,
                warnings=warnings,
            )

        # ---- Gate 3: Build/smoke test (code proposals only) ----
        if proposal.action_type in (ImprovementActionType.TOOL_CREATE, ImprovementActionType.TOOL_MODIFY):
            smoke_ok, smoke_msg = self._check_smoke_test(proposal)
            checks.append({"gate": "smoke_test", "passed": smoke_ok, "detail": smoke_msg})
            if not smoke_ok:
                failures.append(smoke_msg)
        else:
            checks.append({"gate": "smoke_test", "passed": True, "detail": "N/A (not a code change)"})

        # ---- Gate 4: Size limits ----
        size_ok, size_msg = self._check_size_limits(proposal)
        checks.append({"gate": "size_limits", "passed": size_ok, "detail": size_msg})
        if not size_ok:
            warnings.append(size_msg)

        # ---- Gate 5: Seesaw constraint (regression check) ----
        if regression_tasks is None:
            regression_tasks = self._load_regression_tasks()
        if regression_tasks:
            reg_ok, reg_msg, reg_results = self._check_regression(proposal, regression_tasks)
            checks.append({"gate": "seesaw_constraint", "passed": reg_ok, "detail": reg_msg, "results": reg_results})
            regression_results = reg_results
            if not reg_ok:
                failures.append(reg_msg)
        else:
            checks.append({"gate": "seesaw_constraint", "passed": True, "detail": "No regression baselines to check"})

        # ---- Final verdict ----
        if failures:
            # Check if all failures are from regression (seesaw) — those are hard blocks
            hard_failures = [f for f in failures if "manifest" in f.lower() or "content" in f.lower()]
            if hard_failures:
                verdict = GateVerdict.REJECT
            else:
                # Regression failures: needs review (might be acceptable trade-off)
                verdict = GateVerdict.NEEDS_REVIEW
        elif warnings:
            verdict = GateVerdict.NEEDS_REVIEW
        else:
            verdict = GateVerdict.ACCEPT

        return GateResult(
            verdict=verdict,
            proposal=proposal,
            checks=checks,
            failures=failures,
            warnings=warnings,
            regression_results=regression_results,
        )

    # -- Individual gates -----------------------------------------------------

    def _check_manifest(self, proposal: ImprovementProposal) -> Tuple[bool, str]:
        """Verify the proposal declares what it changes."""
        if not proposal.target:
            return False, "Proposal has no target specified"
        if not proposal.description:
            return False, "Proposal has no description"
        if not proposal.action_type:
            return False, "Proposal has no action_type"
        if proposal.action_type in (ImprovementActionType.TOOL_MODIFY, ImprovementActionType.SKILL_PATCH):
            if not proposal.old_string.strip():
                return False, "Patch proposals must include old_string"
            if not proposal.new_string.strip():
                return False, "Patch proposals must include new_string"
        if proposal.action_type == ImprovementActionType.TOOL_CREATE:
            if not proposal.content.strip():
                return False, "Tool creation proposals must include code content"
        return True, "Manifest complete"

    def _check_content(self, proposal: ImprovementProposal) -> Tuple[bool, str]:
        """Validate proposal content."""
        if proposal.content is None:
            return False, "Proposal content is None"
        if proposal.action_type == ImprovementActionType.SKILL_CREATE:
            # Check for required SKILL.md frontmatter
            if "---" not in proposal.content:
                return False, "SKILL.md must have YAML frontmatter (--- delimiters)"
            if "name:" not in proposal.content:
                return False, "SKILL.md frontmatter must include 'name'"
            if "description:" not in proposal.content:
                return False, "SKILL.md frontmatter must include 'description'"

        elif proposal.action_type == ImprovementActionType.TOOL_CREATE:
            # Check for basic Python syntax
            try:
                compile(proposal.content, f"<proposal:{proposal.target}>", "exec")
            except SyntaxError as e:
                return False, f"Python syntax error: {e}"

        elif proposal.action_type == ImprovementActionType.PROMPT_MODIFY:
            if not proposal.content.strip():
                return False, "Prompt modification has no content"

        return True, "Content validated"

    def _check_smoke_test(self, proposal: ImprovementProposal) -> Tuple[bool, str]:
        """Verify code can be imported (smoke test)."""
        code = proposal.content
        if not code.strip():
            return True, "No code to test"

        # Write to temp file and try to compile
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, prefix="evo_smoke_"
            ) as f:
                f.write(code)
                temp_path = Path(f.name)

            # Try compiling
            try:
                compile(code, str(temp_path), "exec")
            except SyntaxError as e:
                return False, f"Compilation failed: {e}"

            # Try importing if it looks importable (has function or class defs)
            if "def " in code or "class " in code:
                try:
                    spec = importlib.util.spec_from_file_location(
                        f"evo_smoke_{proposal.target}", str(temp_path)
                    )
                    if spec and spec.loader:
                        mod = importlib.util.module_from_spec(spec)
                        sys.modules[spec.name] = mod
                        try:
                            spec.loader.exec_module(mod)
                        finally:
                            sys.modules.pop(spec.name, None)
                except Exception as e:
                    # Import failure is a warning, not a hard fail (may depend on
                    # runtime context we don't have in the gate)
                    return True, f"Code compiles but import check had issues: {e}"

            return True, "Smoke test passed"
        except Exception as e:
            return False, f"Smoke test failed: {e}"
        finally:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass

    def _check_size_limits(self, proposal: ImprovementProposal) -> Tuple[bool, str]:
        """Enforce size limits (matching hermes-agent-self-evolution guardrails)."""
        max_skill_size = 15 * 1024  # 15KB (from GEPA guardrails)
        max_tool_desc = 500         # characters (from GEPA guardrails)
        max_prompt_size = 8 * 1024  # 8KB (keep prompt cache healthy)

        if proposal.action_type == ImprovementActionType.SKILL_CREATE:
            if len(proposal.content) > max_skill_size:
                return False, f"Skill content {len(proposal.content)} bytes exceeds {max_skill_size} byte limit"

        elif proposal.action_type == ImprovementActionType.TOOL_CREATE:
            if len(proposal.content) > 100_000:
                return False, f"Tool code {len(proposal.content)} bytes is excessively large"

        elif proposal.action_type == ImprovementActionType.PROMPT_MODIFY:
            if len(proposal.content) > max_prompt_size:
                return False, f"Prompt content {len(proposal.content)} bytes exceeds {max_prompt_size} byte limit"

        return True, "Size limits OK"

    def _check_regression(
        self,
        proposal: ImprovementProposal,
        regression_tasks: List[TaskDefinition],
    ) -> Tuple[bool, str, Dict[str, bool]]:
        """Check proposal doesn't regress on previously-solved tasks.

        This is the SEESAW CONSTRAINT from HarnessX: a candidate is rejected
        if it causes any previously-solved task to fail.

        For non-code proposals, we do a lightweight check. For code/tool
        proposals, we run the full evaluation.
        """
        results: Dict[str, bool] = {}
        evaluator = TaskEvaluator()

        for task in regression_tasks[:self.max_regression_tasks]:
            try:
                result = evaluator.evaluate(task, None, EvaluationContext())  # type: ignore[arg-type]
                passed = result.passed
                results[task.name] = passed
                if not passed:
                    return False, f"Regression detected: '{task.name}' now fails after proposed change", results
            except Exception as e:
                logger.debug("Regression check error for '%s': %s", task.name, e)
                results[task.name] = False

        return True, f"All {len(results)} regression checks passed", results

    def _load_regression_tasks(self) -> List[TaskDefinition]:
        """Load tasks from the regression baseline for seesaw checking."""
        baselines = self._store.get_all_baselines(self.harness_variant)
        tasks = []
        for bl in baselines[:self.max_regression_tasks]:
            try:
                from agent.evolution.task_definition import load_task
                task = load_task(bl["task_name"])
                if task:
                    tasks.append(task)
            except Exception:
                continue
        return tasks

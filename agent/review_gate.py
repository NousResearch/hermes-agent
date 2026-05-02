"""Review Gate — quality gate before task delivery for Hermes 2.8.

Every task passes through Review Gate before results are delivered to the user.
Two layers:
  - Rule-based: deterministic structural checks (auto)
  - LLM-based: semantic quality checks (template questions, answered by reviewer)

Blocking rules (any of these blocks delivery):
  1. No Task Card
  2. No compiled_intent
  3. No result_summary
  4. Not addressing any success_criteria
  5. Any rule_check fails
  6. quality_score < 70 && revision_count < 1
  7. needs_revision = true && revision_count < 1

Degraded delivery (allowed with risk annotation):
  - LLM checks uncertain
  - Information insufficient but noted in result_summary
  - Sub-agent failed but main agent described risks/alternatives

Revision hard limit: max 1 retry → reviewer_exhausted → degraded delivery.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from agent.review_templates import get_checks_for_category, get_template_name

logger = logging.getLogger(__name__)

# ── Blocking rule IDs (structural checks that MUST pass) ──
BLOCKING_RULE_IDS = frozenset({
    "has_task_card",
    "has_compiled_intent",
    "has_result_summary",
    "success_criteria_addressed",
})

QUALITY_THRESHOLD = 70
MAX_REVISIONS = 1


@dataclass
class CheckResult:
    check_id: str
    question: str
    passed: bool
    check_type: str  # "rule" | "llm"
    detail: str = ""


@dataclass
class ReviewResult:
    task_id: str
    checked_at: str
    rule_checks: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    llm_checks: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    quality_score: int = 0
    risks: List[str] = field(default_factory=list)
    needs_revision: bool = False
    revision_instruction: str = ""
    revision_count: int = 0
    review_exhausted: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checked_at": self.checked_at,
            "rule_checks": self.rule_checks,
            "llm_checks": self.llm_checks,
            "quality_score": self.quality_score,
            "risks": self.risks,
            "needs_revision": self.needs_revision,
            "revision_instruction": self.revision_instruction,
            "revision_count": self.revision_count,
            "review_exhausted": self.review_exhausted,
        }


class ReviewGate:
    """Quality gate that inspects a TaskCard before results reach the user.

    Sprint 2 scope:
      - Rule-based checks are auto-executed against TaskCard fields.
      - LLM-based checks are defined as questions; answers are expected from
        the main agent or a reviewer model.
      - Blocking rules and revision limits are enforced.
    """

    def __init__(self):
        pass

    # ── Public API ──

    def check(
        self,
        task_card,  # TaskCard instance
        result_summary: str = "",
        llm_check_results: Optional[Dict[str, Dict[str, Any]]] = None,
        previous_review: Optional[ReviewResult] = None,
    ) -> ReviewResult:
        """Run the review gate against a task card.

        Args:
            task_card: TaskCard instance (has .task_id, .compiled_intent, etc.)
            result_summary: The agent's result text to review
            llm_check_results: Optional pre-computed LLM check answers
            previous_review: Previous ReviewResult if this is a re-review

        Returns:
            ReviewResult with rule_checks, llm_checks, quality_score, etc.
        """
        now = datetime.now(timezone.utc).isoformat()

        # Revision tracking
        revision_count = 0
        if previous_review is not None:
            revision_count = previous_review.revision_count + 1

        # Build check list for this task category
        category = getattr(task_card.compiled_intent, "task_category", "other")
        checks = get_checks_for_category(category)

        rule_checks: Dict[str, Dict[str, Any]] = {}
        llm_checks: Dict[str, Dict[str, Any]] = {}

        for check in checks:
            if check["type"] == "rule":
                rule_checks[check["id"]] = self._run_rule_check(
                    check, task_card, result_summary
                )
            else:
                # LLM checks: use provided results or mark as pending
                if llm_check_results and check["id"] in llm_check_results:
                    llm_checks[check["id"]] = llm_check_results[check["id"]]
                else:
                    llm_checks[check["id"]] = {
                        "id": check["id"],
                        "question": check["question"],
                        "pass": None,  # pending
                        "evidence": "",
                        "type": "llm",
                    }

        # Compute quality score
        quality_score = self._compute_score(rule_checks, llm_checks)

        # Determine if revision is needed
        needs_revision, revision_instruction = self._evaluate_blocking(
            rule_checks, llm_checks, quality_score, revision_count
        )

        # Check if review is exhausted
        review_exhausted = needs_revision and revision_count >= MAX_REVISIONS

        # Collect risks
        risks = self._collect_risks(rule_checks, llm_checks, review_exhausted)

        return ReviewResult(
            task_id=task_card.task_id,
            checked_at=now,
            rule_checks=rule_checks,
            llm_checks=llm_checks,
            quality_score=quality_score,
            risks=risks,
            needs_revision=needs_revision,
            revision_instruction=revision_instruction,
            revision_count=revision_count,
            review_exhausted=review_exhausted,
        )

    # ── Rule checks ──

    def _run_rule_check(
        self,
        check: dict,
        task_card,
        result_summary: str,
    ) -> Dict[str, Any]:
        check_id = check["id"]
        question = check["question"]

        if check_id == "has_task_card":
            passed = bool(task_card and task_card.task_id)
            detail = "" if passed else "Task Card 不存在"

        elif check_id == "has_compiled_intent":
            intent = getattr(task_card, "compiled_intent", None)
            passed = intent is not None and bool(getattr(intent, "real_task", ""))
            detail = "" if passed else "compiled_intent.real_task 为空"

        elif check_id == "has_result_summary":
            passed = bool(result_summary and result_summary.strip())
            detail = "" if passed else "result_summary 为空"

        elif check_id == "success_criteria_addressed":
            intent = getattr(task_card, "compiled_intent", None)
            criteria = getattr(intent, "success_criteria", []) if intent else []
            if not criteria:
                passed = True  # no criteria defined → nothing to miss
                detail = "无 success_criteria 定义，跳过"
            else:
                # Check if result addresses each criterion (simple keyword heuristic)
                summary_lower = (result_summary or "").lower()
                missed = [
                    c for c in criteria
                    if not any(word in summary_lower for word in c.lower().split()[:3])
                ]
                passed = len(missed) == 0
                detail = "" if passed else f"未覆盖的 success_criteria: {missed}"
        else:
            passed = True
            detail = ""

        return {
            "id": check_id,
            "question": question,
            "pass": passed,
            "detail": detail,
            "type": "rule",
        }

    # ── Scoring ──

    def _compute_score(
        self,
        rule_checks: Dict[str, Dict[str, Any]],
        llm_checks: Dict[str, Dict[str, Any]],
    ) -> int:
        total = 0
        count = 0

        for c in rule_checks.values():
            count += 1
            if c.get("pass"):
                total += 1

        for c in llm_checks.values():
            if c.get("pass") is True:
                count += 1
                total += 1
            elif c.get("pass") is False:
                count += 1
                # total += 0

        if count == 0:
            return 100
        return round((total / count) * 100)

    # ── Blocking logic ──

    def _evaluate_blocking(
        self,
        rule_checks: Dict[str, Dict[str, Any]],
        llm_checks: Dict[str, Dict[str, Any]],
        quality_score: int,
        revision_count: int,
    ) -> tuple:
        """Returns (needs_revision: bool, instruction: str)."""
        reasons = []

        # 1. No Task Card (structural)
        has_tc = rule_checks.get("has_task_card", {})
        if has_tc.get("pass") is False:
            reasons.append("缺少 Task Card")

        # 2. No compiled_intent
        has_ci = rule_checks.get("has_compiled_intent", {})
        if has_ci.get("pass") is False:
            reasons.append("缺少 compiled_intent")

        # 3. No result_summary
        has_rs = rule_checks.get("has_result_summary", {})
        if has_rs.get("pass") is False:
            reasons.append("缺少 result_summary")

        # 4. Not addressing success_criteria
        sc = rule_checks.get("success_criteria_addressed", {})
        if sc.get("pass") is False:
            reasons.append("未回应所有 success_criteria")

        # 5. Any rule check fails
        failed_rules = [
            c["id"] for c in rule_checks.values()
            if c.get("pass") is False
        ]
        if failed_rules:
            reasons.append(f"Rule checks 未通过: {failed_rules}")

        # 6. Quality score < threshold and no revision yet
        if quality_score < QUALITY_THRESHOLD and revision_count < 1:
            reasons.append(
                f"质量评分 {quality_score} < {QUALITY_THRESHOLD}，需要修改"
            )

        # 7. needs_revision = true (from previous review) and no revision yet
        # (handled by revision_count < 1 + quality_score logic)

        if reasons:
            return True, "; ".join(reasons)

        return False, ""

    # ── Risk collection ──

    def _collect_risks(
        self,
        rule_checks: Dict[str, Dict[str, Any]],
        llm_checks: Dict[str, Dict[str, Any]],
        review_exhausted: bool,
    ) -> List[str]:
        risks = []

        failed_llm = [
            c["id"] for c in llm_checks.values()
            if c.get("pass") is False
        ]
        if failed_llm:
            risks.append(f"LLM checks 未通过: {failed_llm}")

        pending_llm = [
            c["id"] for c in llm_checks.values()
            if c.get("pass") is None
        ]
        if pending_llm:
            risks.append(f"LLM checks 待评估: {pending_llm}")

        if review_exhausted:
            risks.append("Revision 已达上限，降级交付")

        return risks

    # ── Helpers ──

    @staticmethod
    def is_blocked(review_result: ReviewResult) -> bool:
        """Check if delivery should be blocked (hard block, not degraded)."""
        if review_result.review_exhausted:
            return False  # allow degraded delivery
        return review_result.needs_revision

    @staticmethod
    def allows_degraded_delivery(review_result: ReviewResult) -> bool:
        """Check if degraded delivery is appropriate."""
        if not review_result.needs_revision:
            return False  # fully passed, normal delivery
        return review_result.review_exhausted

"""Workflow lifecycle policy for TSK-58.

This module defines the boundary between autonomous internal work and
Sinclair decision/approval gates. It intentionally has no runtime
dependencies so the policy can be tested independently.
"""

from dataclasses import dataclass
from enum import Enum


class WorkflowNextAction(str, Enum):
    CONTINUE_AUTOMATION = "continue_automation"
    CALL_ADVISOR = "call_advisor"
    PREPARE_CUSTOMER_QUESTIONS = "prepare_customer_questions"
    PM_REVIEW = "pm_review"
    WAIT_SINCLAIR_DECISION = "wait_sinclair_decision"
    FINALIZE = "finalize"


@dataclass(frozen=True)
class WorkflowAssessment:
    internal_work_remaining: bool = False
    advisor_required: bool = False
    customer_information_required: bool = False
    commercial_decision_required: bool = False
    customer_questions_ready: bool = False
    pm_approved: bool = False
    sinclair_approved: bool = False


@dataclass(frozen=True)
class WorkflowDecision:
    action: WorkflowNextAction
    publish_result_report: bool = False
    result_report_kind: str | None = None
    approval_required: bool = False

    @property
    def result_response_kind(self) -> str | None:
        """Runtime response kind for Sinclair-facing result publication."""
        if not self.publish_result_report:
            return None
        return self.result_report_kind

    @property
    def allows_outbox_publication(self) -> bool:
        """Only Sinclair-facing lifecycle results may enter the result outbox."""
        return (
            self.publish_result_report
            and self.result_report_kind in {"decision_required", "final_result"}
        )


def decide_next_action(assessment: WorkflowAssessment) -> WorkflowDecision:
    """Return the next lifecycle action using the TSK-58 automation contract."""

    # Internal work always has priority over escalation to Sinclair.
    if assessment.advisor_required:
        return WorkflowDecision(
            action=WorkflowNextAction.CALL_ADVISOR,
        )

    if assessment.internal_work_remaining:
        return WorkflowDecision(
            action=WorkflowNextAction.CONTINUE_AUTOMATION,
        )

    # Do not escalate raw unknowns. First turn them into customer-ready
    # questions that Sinclair can actually use.
    if (
        assessment.customer_information_required
        and not assessment.customer_questions_ready
    ):
        return WorkflowDecision(
            action=WorkflowNextAction.PREPARE_CUSTOMER_QUESTIONS,
        )

    # A customer-facing/commercial decision is visible to Sinclair only
    # after Demian PM review has completed.
    decision_needed = (
        assessment.commercial_decision_required
        or assessment.customer_information_required
    )

    if decision_needed:
        if not assessment.pm_approved:
            return WorkflowDecision(
                action=WorkflowNextAction.PM_REVIEW,
            )

        if not assessment.sinclair_approved:
            return WorkflowDecision(
                action=WorkflowNextAction.WAIT_SINCLAIR_DECISION,
                publish_result_report=True,
                result_report_kind="decision_required",
                approval_required=True,
            )

    # No final publication before PM review.
    if not assessment.pm_approved:
        return WorkflowDecision(
            action=WorkflowNextAction.PM_REVIEW,
        )

    # Final result is published only after the Sinclair-facing approval
    # boundary has been satisfied.
    if assessment.sinclair_approved:
        return WorkflowDecision(
            action=WorkflowNextAction.FINALIZE,
            publish_result_report=True,
            result_report_kind="final_result",
            approval_required=False,
        )

    return WorkflowDecision(
        action=WorkflowNextAction.CONTINUE_AUTOMATION,
    )


@dataclass(frozen=True)
class OwnerWorkflowResult:
    """Structured lifecycle facts returned by an Owner after a work turn."""

    confirmed_information: tuple[str, ...] = ()
    internal_followups: tuple[str, ...] = ()
    customer_questions: tuple[str, ...] = ()
    advisor_required: bool = False
    commercial_decision_required: bool = False
    customer_ready: bool = False


def assessment_from_owner_result(
    result: OwnerWorkflowResult,
    *,
    pm_approved: bool = False,
    sinclair_approved: bool = False,
) -> WorkflowAssessment:
    """Convert structured Owner output into deterministic lifecycle facts."""

    customer_information_required = bool(result.customer_questions)

    return WorkflowAssessment(
        internal_work_remaining=bool(result.internal_followups),
        advisor_required=result.advisor_required,
        customer_information_required=customer_information_required,
        commercial_decision_required=result.commercial_decision_required,
        customer_questions_ready=(
            customer_information_required and result.customer_ready
        ),
        pm_approved=pm_approved,
        sinclair_approved=sinclair_approved,
    )


import json
import re


_WORKFLOW_BLOCK_RE = re.compile(
    r"<HERMES_WORKFLOW>\s*(.*?)\s*</HERMES_WORKFLOW>",
    re.DOTALL,
)


@dataclass(frozen=True)
class ParsedOwnerWorkflowResponse:
    """Visible Owner response plus optional internal lifecycle control data."""

    visible_text: str
    workflow: OwnerWorkflowResult | None = None


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(
        item.strip()
        for item in value
        if isinstance(item, str) and item.strip()
    )


def parse_owner_workflow_response(raw: str) -> ParsedOwnerWorkflowResponse:
    """Parse and strip the private HERMES_WORKFLOW control block.

    Missing or malformed control data is backward-compatible: the message
    remains a normal visible Staff response and no lifecycle facts are emitted.
    """

    text = str(raw or "")
    match = _WORKFLOW_BLOCK_RE.search(text)

    if match is None:
        return ParsedOwnerWorkflowResponse(
            visible_text=text,
            workflow=None,
        )

    visible_text = (
        text[: match.start()] + text[match.end() :]
    ).strip()

    try:
        payload = json.loads(match.group(1))
    except (TypeError, ValueError, json.JSONDecodeError):
        return ParsedOwnerWorkflowResponse(
            visible_text=visible_text,
            workflow=None,
        )

    if not isinstance(payload, dict):
        return ParsedOwnerWorkflowResponse(
            visible_text=visible_text,
            workflow=None,
        )

    workflow = OwnerWorkflowResult(
        confirmed_information=_string_tuple(
            payload.get("confirmed_information")
        ),
        internal_followups=_string_tuple(
            payload.get("internal_followups")
        ),
        customer_questions=_string_tuple(
            payload.get("customer_questions")
        ),
        advisor_required=payload.get("advisor_required") is True,
        commercial_decision_required=(
            payload.get("commercial_decision_required") is True
        ),
        customer_ready=payload.get("customer_ready") is True,
    )

    return ParsedOwnerWorkflowResponse(
        visible_text=visible_text,
        workflow=workflow,
    )


@dataclass(frozen=True)
class LifecycleRouteAction:
    """Runtime-neutral route intent produced by the workflow lifecycle."""

    kind: str
    target_profile: str | None = None
    response_kind: str = "silence"


def lifecycle_route_action(
    decision: WorkflowDecision,
    *,
    owner_profile: str,
    advisor_profile: str | None = None,
) -> LifecycleRouteAction:
    """Map a lifecycle decision to a runtime-neutral Work Router intent."""

    if decision.action is WorkflowNextAction.CONTINUE_AUTOMATION:
        return LifecycleRouteAction(
            kind="dispatch",
            target_profile=owner_profile,
            response_kind="workflow_continue",
        )

    if decision.action is WorkflowNextAction.CALL_ADVISOR:
        if not advisor_profile:
            raise ValueError("advisor_profile is required for CALL_ADVISOR")
        return LifecycleRouteAction(
            kind="dispatch",
            target_profile=advisor_profile,
            response_kind="advisor_request",
        )

    if decision.action is WorkflowNextAction.PREPARE_CUSTOMER_QUESTIONS:
        return LifecycleRouteAction(
            kind="dispatch",
            target_profile=owner_profile,
            response_kind="prepare_customer_questions",
        )

    if decision.action is WorkflowNextAction.PM_REVIEW:
        return LifecycleRouteAction(
            kind="dispatch",
            target_profile="Demian",
            response_kind="pm_review",
        )

    if decision.action is WorkflowNextAction.WAIT_SINCLAIR_DECISION:
        return LifecycleRouteAction(
            kind="result_report",
            response_kind="decision_required",
        )

    if decision.action is WorkflowNextAction.FINALIZE:
        return LifecycleRouteAction(
            kind="result_report",
            response_kind="final_result",
        )

    raise ValueError(f"unsupported lifecycle action: {decision.action!r}")


@dataclass(frozen=True)
class RoutedOwnerWorkflowResponse:
    """Owner response after optional TSK-58 lifecycle interpretation."""

    visible_text: str
    lifecycle_active: bool
    route: LifecycleRouteAction | None = None
    workflow: OwnerWorkflowResult | None = None


def route_owner_workflow_response(
    raw: str,
    *,
    owner_profile: str,
    advisor_profile: str | None = None,
    pm_approved: bool = False,
    sinclair_approved: bool = False,
) -> RoutedOwnerWorkflowResponse:
    """Activate TSK-58 only when an explicit workflow control block exists."""

    parsed = parse_owner_workflow_response(raw)

    # Backward compatibility:
    # ordinary Staff/Slack responses stay entirely on the legacy Router path.
    if parsed.workflow is None:
        return RoutedOwnerWorkflowResponse(
            visible_text=parsed.visible_text,
            lifecycle_active=False,
            route=None,
            workflow=None,
        )

    assessment = assessment_from_owner_result(
        parsed.workflow,
        pm_approved=pm_approved,
        sinclair_approved=sinclair_approved,
    )
    decision = decide_next_action(assessment)

    route = lifecycle_route_action(
        decision,
        owner_profile=owner_profile,
        advisor_profile=advisor_profile,
    )

    return RoutedOwnerWorkflowResponse(
        visible_text=parsed.visible_text,
        lifecycle_active=True,
        route=route,
        workflow=parsed.workflow,
    )


def should_activate_owner_lifecycle(
    *,
    author_profile: str | None,
    owner_profile: str | None,
    text: str,
) -> bool:
    """Activate TSK-58 only for the current Owner with explicit control data."""

    if not author_profile or not owner_profile:
        return False

    if author_profile.casefold() != owner_profile.casefold():
        return False

    return _WORKFLOW_BLOCK_RE.search(str(text or "")) is not None


@dataclass(frozen=True)
class PMReviewOutcome:
    """Deterministic result of Demian PM review."""

    next_state: str
    target_profile: str | None = None
    approval_required: bool = False
    publish_result_report: bool = False


def decide_pm_review_outcome(
    *,
    pm_approved: bool,
    rework_required: bool,
    decision_required: bool,
) -> PMReviewOutcome:
    """Map Demian PM review to the next durable work lifecycle state."""

    if rework_required:
        return PMReviewOutcome(
            next_state="OWNER_REWORK",
            target_profile="OWNER",
            approval_required=False,
            publish_result_report=False,
        )

    if not pm_approved:
        raise ValueError(
            "PM review must require rework when it is not approved"
        )

    if decision_required:
        return PMReviewOutcome(
            next_state="WAIT_SINCLAIR_DECISION",
            target_profile="Sinclair",
            approval_required=True,
            publish_result_report=True,
        )

    return PMReviewOutcome(
        next_state="FINAL_RESULT",
        target_profile=None,
        approval_required=False,
        publish_result_report=True,
    )


@dataclass(frozen=True)
class ParsedPMReviewResponse:
    """Visible Demian response plus private PM lifecycle control data."""

    visible_text: str
    pm_approved: bool = False
    rework_required: bool = False
    decision_required: bool = False
    control_found: bool = False


_PM_REVIEW_BLOCK_RE = re.compile(
    r"<HERMES_PM_REVIEW>\s*(.*?)\s*</HERMES_PM_REVIEW>",
    re.DOTALL,
)


def parse_pm_review_response(raw: str) -> ParsedPMReviewResponse:
    """Parse and strip Demian private PM lifecycle control data.

    Missing or malformed control data never creates lifecycle facts.
    The private block is never exposed through visible_text.
    """

    text = str(raw or "")
    match = _PM_REVIEW_BLOCK_RE.search(text)

    if match is None:
        return ParsedPMReviewResponse(
            visible_text=text.strip(),
        )

    visible_text = (
        text[: match.start()] + text[match.end() :]
    ).strip()

    try:
        payload = json.loads(match.group(1))
    except (TypeError, ValueError, json.JSONDecodeError):
        return ParsedPMReviewResponse(
            visible_text=visible_text,
        )

    if not isinstance(payload, dict):
        return ParsedPMReviewResponse(
            visible_text=visible_text,
        )

    required = (
        "pm_approved",
        "rework_required",
        "decision_required",
    )

    if any(
        key not in payload or not isinstance(payload[key], bool)
        for key in required
    ):
        return ParsedPMReviewResponse(
            visible_text=visible_text,
        )

    pm_approved = payload["pm_approved"]
    rework_required = payload["rework_required"]
    decision_required = payload["decision_required"]

    # Invalid combinations must not silently become workflow facts.
    if rework_required and pm_approved:
        return ParsedPMReviewResponse(
            visible_text=visible_text,
        )

    if rework_required and decision_required:
        return ParsedPMReviewResponse(
            visible_text=visible_text,
        )

    if not pm_approved and not rework_required:
        return ParsedPMReviewResponse(
            visible_text=visible_text,
        )

    return ParsedPMReviewResponse(
        visible_text=visible_text,
        pm_approved=pm_approved,
        rework_required=rework_required,
        decision_required=decision_required,
        control_found=True,
    )


@dataclass(frozen=True)
class PMReviewIngressAction:
    """Runtime-neutral action derived from a validated Demian PM review."""

    next_state: str
    target_profile: str | None
    response_kind: str
    review_generation: int
    approval_required: bool
    publish_result_report: bool


def pm_review_ingress_action(
    parsed: ParsedPMReviewResponse,
    *,
    owner_profile: str,
    review_generation: int,
) -> PMReviewIngressAction:
    """Convert validated PM control data into the next runtime intent."""

    if not parsed.control_found:
        raise ValueError("validated PM review control data is required")

    outcome = decide_pm_review_outcome(
        pm_approved=parsed.pm_approved,
        rework_required=parsed.rework_required,
        decision_required=parsed.decision_required,
    )

    if outcome.next_state == "OWNER_REWORK":
        return PMReviewIngressAction(
            next_state="OWNER_REWORK",
            target_profile=owner_profile,
            response_kind="owner_rework",
            review_generation=review_generation + 1,
            approval_required=False,
            publish_result_report=False,
        )

    if outcome.next_state == "WAIT_SINCLAIR_DECISION":
        return PMReviewIngressAction(
            next_state="WAIT_SINCLAIR_DECISION",
            target_profile=None,
            response_kind="decision_required",
            review_generation=review_generation,
            approval_required=True,
            publish_result_report=True,
        )

    if outcome.next_state == "FINAL_RESULT":
        return PMReviewIngressAction(
            next_state="FINAL_RESULT",
            target_profile=None,
            response_kind="final_result",
            review_generation=review_generation,
            approval_required=False,
            publish_result_report=True,
        )

    raise RuntimeError(
        f"unsupported PM review outcome: {outcome.next_state}"
    )

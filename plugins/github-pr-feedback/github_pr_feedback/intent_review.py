"""Per-PR operator-intent escalation for ambiguous review feedback."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .github_client import Feedback

_DISAGREEMENT = re.compile(
    r"\b(?:i\s+disagree|do\s+not\s+(?:approve|use|want)|don't\s+(?:apply|use)|not\s+approved|rather\s+use|use\s+.+\s+instead|instead\s+of)\b",
    re.IGNORECASE,
)
_ALTERNATIVE_VERB = (
    r"(?:allow|apply|exclude|include|preserve|reject|require|retain|use|validate)"
)
_UNRESOLVED_ALTERNATIVES = re.compile(
    rf"(?:\beither\s+(?:consistently\s+)?{_ALTERNATIVE_VERB}\b.{{1,600}}?"
    rf"\bor\s+(?:consistently\s+)?{_ALTERNATIVE_VERB}\b|"
    rf"\b{_ALTERNATIVE_VERB}\b.{{1,600}}?,\s+or\s+{_ALTERNATIVE_VERB}\b)",
    re.IGNORECASE,
)
_OPERATOR_COMMAND = re.compile(
    r"\b(?P<command>approve\s+original|dismiss|needs\s+more\s+evidence|use\s+alternative)\b",
    re.IGNORECASE,
)
_TARGET_COMMENT = re.compile(r"\bintent[-_ ]review\s*:\s*(\S+)", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class IntentReview:
    comment_id: str
    reason: str
    command: str | None = None


def classify_feedback(
    feedback: Feedback, *, owner_login: str | None = None
) -> IntentReview | None:
    """Escalate only explicit disagreement/change-of-approach comments."""

    if (
        owner_login is not None
        and feedback.reviewer.login.casefold() == owner_login.casefold()
    ):
        return None
    body = " ".join(str(feedback.body or "").split())
    if not body or not (
        _DISAGREEMENT.search(body) or _UNRESOLVED_ALTERNATIVES.search(body)
    ):
        return None
    return IntentReview(
        comment_id=feedback.feedback_id,
        reason="explicit disagreement or replacement approach",
    )


def operator_decision(feedback: Feedback, *, owner_login: str) -> str | None:
    """Read an operator-only decision; bot comments never clear intent review."""

    if feedback.reviewer.login.casefold() != owner_login.casefold():
        return None
    match = _OPERATOR_COMMAND.search(" ".join(str(feedback.body or "").split()))
    return match.group("command").casefold() if match else None


def pending_intent_review(feedback: tuple[Feedback, ...], *, owner_login: str) -> bool:
    """Return whether a high-signal disagreement lacks a later operator decision."""
    return bool(pending_intent_comment_ids(feedback, owner_login=owner_login))


def pending_intent_comment_ids(
    feedback: tuple[Feedback, ...], *, owner_login: str
) -> frozenset[str]:
    """Return only disagreement comments not superseded by an operator decision."""

    pending: dict[str, None] = {}
    for item in sorted(feedback, key=lambda value: value.created_at):
        if classify_feedback(item, owner_login=owner_login) is not None:
            pending[item.feedback_id] = None
        elif operator_decision(item, owner_login=owner_login) is not None:
            target = _TARGET_COMMENT.search(str(item.body or ""))
            if target is not None:
                pending.pop(target.group(1).rstrip(".,);"), None)
            elif len(pending) == 1:
                # A concise operator reply is safe only when this PR has one
                # unresolved intent item; multiple items require an ID.
                pending.clear()
    return frozenset(pending)

"""Spec critique / linter (05-final-plan.md §4 step 3).

Lints a SpecDocument for:
- Ambiguous or untestable acceptance criteria
- Missing auth/authorization or role-isolation (IDOR/RLS) rules when non-public roles exist
- Overlapping module ownership globs (which would break parallel swarm worktrees)
- Contradictions between stories and non-goals/assumptions
Returns structured issues and at most ONE blocking clarifying question.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any, Dict, List, Optional

from samagent.spec.models import SpecDocument

_VAGUE_PATTERNS = re.compile(
    r"\b(make it nice|looks? good|modern ui|fast enough|works? well|intuitive|seamless|etc\.?)\b",
    re.IGNORECASE,
)
_OBSERVABLE_SIGNAL = re.compile(
    r"(\bHTTP\s*\d{3}\b|\b200\b|\b201\b|\b400\b|\b401\b|\b403\b|\b404\b|\b409\b|\b422\b"
    r"|\bGET\b|\bPOST\b|\bPUT\b|\bDELETE\b|\bPATCH\b|\breturns?\b|\brejected\b|\bshows?\b|\b≥\s*\d+|\bat least \d+)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SpecIssue:
    code: str
    severity: str  # "error" | "warning"
    target: str
    message: str
    suggestion: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CritiqueReport:
    passed: bool
    issues: List[SpecIssue] = field(default_factory=list)
    blocking_question: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "issues": [i.to_dict() for i in self.issues],
            "blocking_question": self.blocking_question,
        }


def critique_spec(spec: SpecDocument) -> CritiqueReport:
    """Lint *spec* and return a CritiqueReport with at most one blocking question."""
    issues: List[SpecIssue] = []

    if not spec.goal or len(spec.goal.strip()) < 3:
        issues.append(
            SpecIssue(
                code="EMPTY_GOAL",
                severity="error",
                target="goal",
                message="Project goal is empty or too short.",
                suggestion="State what application is being built in one sentence.",
            )
        )

    if not spec.stories:
        issues.append(
            SpecIssue(
                code="NO_STORIES",
                severity="error",
                target="stories",
                message="Spec contains zero user stories.",
                suggestion="Add at least one user story with an executable acceptance criterion.",
            )
        )

    # 1. Check each story for testability and ambiguity
    for s in spec.stories:
        acc = (s.accept or "").strip()
        if not acc:
            issues.append(
                SpecIssue(
                    code="EMPTY_ACCEPTANCE",
                    severity="error",
                    target=s.id,
                    message=f"Story {s.id} has no acceptance criterion.",
                    suggestion="Specify an observable HTTP status, DOM element, or state transition.",
                )
            )
            continue
        if not _OBSERVABLE_SIGNAL.search(acc):
            issues.append(
                SpecIssue(
                    code="UNTESTABLE_ACCEPTANCE",
                    severity="error",
                    target=s.id,
                    message=f"Story {s.id} acceptance '{acc}' lacks a concrete observable check (status code, route, or count).",
                    suggestion="Rewrite with an explicit check, e.g. 'GET /api/items returns HTTP 200 with ≥1 item'.",
                )
            )
        if _VAGUE_PATTERNS.search(acc) and not _OBSERVABLE_SIGNAL.search(acc):
            issues.append(
                SpecIssue(
                    code="AMBIGUOUS_PHRASING",
                    severity="warning",
                    target=s.id,
                    message=f"Story {s.id} uses subjective phrasing in acceptance criteria.",
                    suggestion="Replace subjective adjectives with measurable assertions.",
                )
            )

    # 2. Check role security & row-level isolation (Lovable CVE-2025-48757 class prevention)
    non_public_roles = [r for r in spec.roles if r.lower() not in ("visitor", "anonymous", "public", "guest")]
    if non_public_roles:
        auth_stories = [s for s in spec.stories if s.auth_required or "401" in s.accept or "403" in s.accept or "login" in s.accept.lower()]
        if not auth_stories:
            issues.append(
                SpecIssue(
                    code="MISSING_AUTH_BOUNDARY",
                    severity="error",
                    target="roles",
                    message=f"Non-public roles {non_public_roles} are declared, but no story specifies authentication or authorization checks.",
                    suggestion="Mark protected stories with auth_required=True and specify HTTP 401/403 rejection for unauthorized callers.",
                )
            )

    # 3. Check module ownership glob collisions
    seen_globs: Dict[str, str] = {}
    for m in spec.modules:
        for g in m.owned_globs:
            norm = g.strip()
            if norm in seen_globs and seen_globs[norm] != m.name:
                issues.append(
                    SpecIssue(
                        code="OWNERSHIP_COLLISION",
                        severity="error",
                        target=f"modules.{m.name}",
                        message=f"Glob '{norm}' is claimed by both '{seen_globs[norm]}' and '{m.name}'.",
                        suggestion="Ensure every module has disjoint write-set globs before enabling parallel swarm fan-out.",
                    )
                )
            seen_globs[norm] = m.name

    # 4. Check contradictions between stories and non-goals/assumptions
    out_of_scope_text = " ".join(spec.non_goals + [a.text for a in spec.assumptions]).lower()
    if "payment" in out_of_scope_text and ("out of scope" in out_of_scope_text or "no live" in out_of_scope_text):
        for s in spec.stories:
            if any(k in (s.can + " " + s.accept).lower() for k in ("stripe live", "charge credit card", "real payment")):
                issues.append(
                    SpecIssue(
                        code="SPEC_CONTRADICTION",
                        severity="error",
                        target=s.id,
                        message=f"Story {s.id} requires live payments, which contradicts the non-goals/assumptions ledger.",
                        suggestion="Either use a local payment mock in Story {s.id} or remove payments from non-goals.",
                    )
                )

    errors = [i for i in issues if i.severity == "error"]
    blocking_q: Optional[str] = None
    if errors:
        top = errors[0]
        blocking_q = f"[{top.code} on {top.target}] {top.message} — Should I apply suggestion: {top.suggestion}?"

    return CritiqueReport(passed=(len(errors) == 0), issues=issues, blocking_question=blocking_q)

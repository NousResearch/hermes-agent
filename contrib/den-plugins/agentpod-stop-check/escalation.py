"""Escalation classification: when is pulling a human in actually required?

Motivating defect (EM regression requirement, 2026-09-17, card ``t_de12518a`` /
PR #4952): a supervision turn escalated to the human even though

  (a) a documented, authorised **opaque** (non-author) review identity existed,
  (b) an independent artifact review had been done at a pinned SHA, and
  (c) only a formal review gate remained.

Nothing about that situation is irreversible, financial, or external-policy.
The correct action was to route the scoped review itself. Escalating instead
costs a human round trip and is the failure this module classifies out.

The classification is deliberately **fail-closed**: anything this module cannot
positively prove is routable stays a human escalation. The three ways to lose
the capability are all explicit:

* the pinned review SHA no longer equals the current head -> the prior review
  is stale, the capability must NOT be used, a fresh independent review at the
  new head is required;
* no documented identity exists for the named capability, the identity does not
  positively declare ``opaque: true``, or using it would reveal or mint a
  credential -> emit the exact access blocker, do not invent a workaround;
* the escalation class is anything other than a review gate (financial,
  irreversible, external policy, or simply unstated) -> the human prompt is
  preserved untouched.

Nothing here performs a review, reads a credential, or contacts GitHub. It
classifies an escalation record that already exists on the card.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# --------------------------------------------------------------- markers ---
# Recorded on the card, e.g.:
#   STOP-CHECK-ESCALATION: PR #4952 needs a formal approving review
#     class=review_gate capability=app-review review_sha=326168c3 head_sha=326168c3
ESCALATION_RE = re.compile(
    r"STOP-CHECK-ESCALATION:\s*(?P<what>[^\n]+)", re.IGNORECASE
)
_FIELD_RE = re.compile(r"(?P<key>[a-z_]+)=(?P<val>[^\s]+)", re.IGNORECASE)

# The only class this module is allowed to route without a human. Every other
# value — and the absence of a value — preserves the human prompt.
CLASS_REVIEW_GATE = "review_gate"
HUMAN_PRESERVED_CLASSES = ("financial", "irreversible", "external_policy")

DECISION_ROUTE = "route_scoped_review"
DECISION_FRESH_REVIEW = "require_fresh_review"
DECISION_ACCESS_BLOCKER = "access_blocker"
DECISION_HUMAN = "human_required"


@dataclass
class Escalation:
    """An escalation record parsed off a card comment."""

    what: str
    author: str = ""
    written_at: int = 0
    klass: str = ""
    capability: str = ""
    review_sha: str = ""
    head_sha: str = ""

    @property
    def is_review_gate(self) -> bool:
        return self.klass.strip().lower() == CLASS_REVIEW_GATE


@dataclass
class EscalationVerdict:
    decision: str
    detail: str
    next_action: str

    @property
    def needs_human(self) -> bool:
        return self.decision == DECISION_HUMAN


def parse_escalation(body: str, *, author: str = "", written_at: int = 0):
    """Parse a STOP-CHECK-ESCALATION marker out of a comment body."""
    m = ESCALATION_RE.search(body or "")
    if not m:
        return None
    raw = m.group("what").strip()
    fields = {
        f.group("key").lower(): f.group("val") for f in _FIELD_RE.finditer(raw)
    }
    return Escalation(
        what=raw,
        author=author,
        written_at=int(written_at or 0),
        klass=fields.get("class", ""),
        capability=fields.get("capability", ""),
        review_sha=fields.get("review_sha", ""),
        head_sha=fields.get("head_sha", ""),
    )


def documented_capabilities(cfg: dict) -> dict:
    """Documented review identities, keyed by name.

    The config IS the documentation surface: an identity that is not declared
    here does not exist as far as this guard is concerned, and an undeclared
    capability fails CLOSED into an access blocker rather than being assumed
    usable. That direction is deliberate — the guard can only ever refuse a
    capability it has not been told about, never silently grant one.

    The bare-string shorthand (``review_capabilities: [app-review]``) is kept
    for config ergonomics but is **never routable**: it structurally cannot
    carry ``opaque: true``, and opacity must be positively asserted, so such an
    entry always yields an access blocker.
    """
    out: dict = {}
    for entry in cfg.get("review_capabilities") or []:
        if isinstance(entry, str):
            entry = {"name": entry}
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "").strip()
        if name:
            out[name.lower()] = entry
    return out


def classify_escalation(
    esc: Optional[Escalation],
    *,
    current_head: Optional[str],
    cfg: dict,
    task_id: str = "",
) -> EscalationVerdict:
    """Decide whether this escalation genuinely needs the human.

    ``current_head`` is the head the supervisor can observe right now. When it
    is unknown the escalation is NOT routed — an unobservable head cannot be
    matched against a pinned review.
    """
    if esc is None:
        return EscalationVerdict(
            DECISION_HUMAN,
            "no escalation record to classify",
            "record the escalation with class= and, for a review gate, "
            "capability=/review_sha=/head_sha= before routing anything",
        )

    if not esc.is_review_gate:
        why = (
            f"class '{esc.klass}'"
            if esc.klass
            else "no class recorded (fail-closed: unstated is not routable)"
        )
        return EscalationVerdict(
            DECISION_HUMAN,
            f"human escalation preserved: {why}",
            f"keep the human prompt for {task_id or 'this card'} — irreversible, "
            f"financial and external-policy decisions are not routable by an agent",
        )

    caps = documented_capabilities(cfg)
    entry = caps.get(esc.capability.strip().lower()) if esc.capability else None
    if entry is None:
        return EscalationVerdict(
            DECISION_ACCESS_BLOCKER,
            f"no documented review identity for capability "
            f"'{esc.capability or '(unnamed)'}' — {len(caps)} declared",
            f"ACCESS BLOCKER for {task_id or 'this card'}: no authorised "
            f"non-author review identity is documented for "
            f"'{esc.capability or '(unnamed)'}'. Ask the operator to document one; "
            f"do NOT mint, reveal or borrow a credential to work around it",
        )
    if entry.get("reveals_credential") or entry.get("mints_credential"):
        return EscalationVerdict(
            DECISION_ACCESS_BLOCKER,
            f"documented identity '{esc.capability}' would reveal or mint a "
            f"credential",
            f"ACCESS BLOCKER for {task_id or 'this card'}: using "
            f"'{esc.capability}' would reveal or mint a credential. Stop and ask "
            f"the operator; do NOT proceed",
        )
    if entry.get("opaque") is not True:
        # Opacity is the ONE field that must be positively asserted. Absence is
        # not "probably opaque": an entry that never declares it — including the
        # bare-string shorthand, which structurally cannot — could be the author
        # identity discharging its own review gate. reveals_credential /
        # mints_credential default false in the other direction on purpose
        # (absence there genuinely means "does not").
        declared = "is not opaque (it would review as the author)" if (
            entry.get("opaque") is False
        ) else "does not declare opaque: true (opacity must be asserted, not assumed)"
        return EscalationVerdict(
            DECISION_ACCESS_BLOCKER,
            f"documented identity '{esc.capability}' {declared}",
            f"ACCESS BLOCKER for {task_id or 'this card'}: '{esc.capability}' is "
            f"not a proven non-author identity ({declared}), so it cannot supply "
            f"a non-author review. Ask the operator to document it with "
            f"'opaque: true' or supply an authorised opaque identity",
        )

    head = str(current_head or "").strip()
    reviewed = esc.review_sha.strip()
    if not head:
        return EscalationVerdict(
            DECISION_FRESH_REVIEW,
            "current head is unknown, so the pinned review cannot be matched "
            "against it",
            f"read the current head for {task_id or 'this card'} and re-run an "
            f"independent review at that exact head before routing a review",
        )
    if not reviewed or not _same_sha(reviewed, head):
        return EscalationVerdict(
            DECISION_FRESH_REVIEW,
            f"independent review is pinned at "
            f"{reviewed or '(none)'} but the head is now {head[:12]} — the prior "
            f"review is stale",
            f"do NOT use '{esc.capability}' on {task_id or 'this card'}: run a "
            f"fresh independent review at {head[:12]} first, then re-classify",
        )

    return EscalationVerdict(
        DECISION_ROUTE,
        f"documented opaque identity '{esc.capability}' is authorised and the "
        f"independent review at {reviewed[:12]} matches the current head — only a "
        f"formal review gate remains",
        f"route the scoped review for {task_id or 'this card'} yourself via "
        f"'{esc.capability}'. Do NOT escalate to the human: nothing here is "
        f"irreversible, financial or external-policy",
    )


def _same_sha(a: str, b: str) -> bool:
    """Git SHAs compare by prefix, shortest wins — but never on empty input."""
    a, b = a.strip().lower(), b.strip().lower()
    if not a or not b:
        return False
    n = min(len(a), len(b))
    if n < 7:  # too short to identify a commit
        return False
    return a[:n] == b[:n]

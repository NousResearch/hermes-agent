"""Kanban size gate - deterministic pre-flight estimator for card complexity.

This module provides a cheap, deterministic estimator that does NOT invoke an LLM.
It measures card complexity using structural signals: acceptance criteria count,
distinct files/subsystems named, and distinct deliverables.

The gate hooks into every path that can land a card on ``ready``:

1. **Card creation** (``kanban_db.create_task``): every created card carries one
   ``size_estimate`` event. A card that would go straight to ``ready`` while
   oversized is diverted to ``triage``; one parked in ``todo``/``blocked`` gets an
   ``advisory`` estimate that :func:`gate_ready_entry` re-decides at promotion.
2. **Promotion** (``kanban_db.recompute_ready``, ``unblock_task``,
   ``reopen_review_task``, ``promote_task``): one decision point,
   :func:`gate_ready_entry`, so a card parked behind an open parent cannot reach
   ``ready`` unmeasured.
3. **Decomposition** (``kanban_db_graph.decompose_triage_task``): every emitted
   child is measured, oversized children are rejected with a typed error, and
   every accepted child records its estimate plus a one-line budget stamp in its
   body so the worker sees the budget it was sized for.

Estimator formula:
    score = (ac_count * 3) + (file_count * 2) + (deliverable_count * 1)

Where:
- ac_count: number of acceptance criteria items (numbered lists, bullet points
  in "Acceptance Criteria" sections)
- file_count: number of distinct files/subsystems named (paths with extensions
  like .py, .ts, .md, etc.)
- deliverable_count: number of distinct deliverables (sections with "##" headers
  that imply separate work units)

Default threshold: 30 (configurable via kanban.size_gate.threshold in config.yaml)

A card scoring above the threshold is considered oversized and must be decomposed
before dispatch.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# Default threshold for the size gate. Cards scoring above this are oversized.
# Configurable via kanban.size_gate.threshold in config.yaml.
DEFAULT_SIZE_GATE_THRESHOLD = 30

# Weights for the estimator formula
WEIGHT_ACCEPTANCE_CRITERIA = 3
WEIGHT_FILE = 2
WEIGHT_DELIVERABLE = 1

# One-line budget stamp prepended to a decomposed child's body. Estimator-neutral
# ON PURPOSE: no ``##`` header, no bullet/numbered-list lead and no filename, so
# re-measuring a stamped body returns the score that was measured before stamping.
BUDGET_ESTIMATE_STAMP = (
    "Budget estimate (size gate): score {score}/{threshold} - single-worker unit, "
    "measured at decomposition."
)


@dataclass
class SizeEstimate:
    """Result of estimating a card's size."""

    score: int
    acceptance_criteria_count: int
    file_count: int
    deliverable_count: int

    def is_oversized(self, threshold: int) -> bool:
        """Return True if this estimate exceeds the threshold."""
        return self.score > threshold


@dataclass(frozen=True)
class SizeGatePolicy:
    """Resolved size gate configuration (read once per call site, never per row)."""

    threshold: int = DEFAULT_SIZE_GATE_THRESHOLD

    def is_oversized(self, estimate: SizeEstimate) -> bool:
        return estimate.is_oversized(self.threshold)


class OversizedChildError(Exception):
    """Raised when a decomposed child exceeds the size gate threshold.

    This is a typed failure that the decomposer must raise (not silently pass)
    when a child task is too large for a single worker.
    """

    def __init__(self, child_index: int, child_title: str, estimate: SizeEstimate, threshold: int):
        self.child_index = child_index
        self.child_title = child_title
        self.estimate = estimate
        self.threshold = threshold
        super().__init__(
            f"Child task {child_index} ({child_title!r}) is oversized: "
            f"score={estimate.score} exceeds threshold={threshold}. "
            f"Break it down further before emitting."
        )


def _as_text(value: object) -> str:
    """Coerce a stored title/body to text.

    ``body`` is a TEXT column but a hand-written or tool-written row can carry a
    BLOB (one card on the live board did), and ``re.search`` raises on bytes.
    Decoding lossily keeps such a card measurable instead of silently unmeasured.
    """
    if value is None:
        return ""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value).decode("utf-8", "replace")
    return value if isinstance(value, str) else str(value)


def estimate_card_size(title: Optional[str], body: Optional[str]) -> SizeEstimate:
    """Estimate the size of a card from its title and body.

    This is a deterministic, LLM-free estimator that counts structural signals:
    - Acceptance criteria items (numbered lists, bullets in AC sections)
    - Distinct files/subsystems named (paths with extensions)
    - Distinct deliverables (## headers implying separate work units)

    Returns a SizeEstimate with the computed score and breakdown.
    """
    text = _as_text(body)
    if not text:
        # Minimal card with just a title
        return SizeEstimate(score=0, acceptance_criteria_count=0, file_count=0, deliverable_count=0)

    ac_count = 0
    file_count = 0
    deliverable_count = 0

    # Count acceptance criteria items
    # Look for numbered lists (1., 2., etc.) or bullet points (-, *) in AC sections
    # or standalone numbered lists that look like criteria
    ac_section_match = re.search(
        r"(?i)(acceptance\s+criteria|acceptance|criteria)\s*:?\s*\n(.*?)(?=\n##|\Z)",
        text,
        re.DOTALL,
    )
    if ac_section_match:
        ac_section = ac_section_match.group(2)
        # Count numbered items (1., 2., etc.) or bullets (-, *)
        ac_count = len(re.findall(r"^\s*(\d+\.|[-*])\s+", ac_section, re.MULTILINE))
    else:
        # No explicit AC section - count standalone numbered lists that look like criteria
        # Look for sequences of numbered items
        numbered_blocks = re.findall(r"(?:^|\n)(\s*\d+\..+(?:\n\s*\d+\..+)*)", text)
        for block in numbered_blocks:
            items = re.findall(r"^\s*\d+\.", block, re.MULTILINE)
            if len(items) >= 3:  # At least 3 items to count as AC
                ac_count += len(items)

    # Count distinct files/subsystems named
    # Look for file paths with extensions (.py, .ts, .md, .yaml, etc.)
    file_pattern = r"\b[\w\-/]+\.(?:py|ts|tsx|js|jsx|md|yaml|yml|json|toml|sql|sh|rb|go|rs|java|c|cpp|h|hpp)\b"
    files = set(re.findall(file_pattern, text, re.IGNORECASE))
    file_count = len(files)

    # Count distinct deliverables (## headers)
    # Each ## header implies a separate work unit or section
    headers = re.findall(r"^##\s+(.+)$", text, re.MULTILINE)
    deliverable_count = len(headers)

    # Compute the weighted score
    score = (
        ac_count * WEIGHT_ACCEPTANCE_CRITERIA
        + file_count * WEIGHT_FILE
        + deliverable_count * WEIGHT_DELIVERABLE
    )

    return SizeEstimate(
        score=score,
        acceptance_criteria_count=ac_count,
        file_count=file_count,
        deliverable_count=deliverable_count,
    )


def load_size_gate_policy() -> SizeGatePolicy:
    """Load the size gate policy from config, falling back to the default.

    Config key: kanban.size_gate.threshold

    Call sites that loop over many cards load this ONCE per sweep: the reader
    parses config.yaml, which must not happen per row on a dispatcher tick.
    """
    try:
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly()
        kanban_cfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
        threshold = kanban_cfg.get("size_gate", {}).get("threshold")
        if isinstance(threshold, (int, float)) and threshold > 0:
            return SizeGatePolicy(threshold=int(threshold))
    except Exception:
        # Config-read boundary: any failure must fall back to the shipped default
        # rather than block card creation, but it must not be a silent pass.
        logger.warning(
            "size gate: could not read kanban.size_gate.threshold; using the default %d",
            DEFAULT_SIZE_GATE_THRESHOLD, exc_info=True,
        )
    return SizeGatePolicy(threshold=DEFAULT_SIZE_GATE_THRESHOLD)


def load_size_gate_threshold() -> int:
    """Load the size gate threshold from config, falling back to the default.

    Config key: kanban.size_gate.threshold
    """
    return load_size_gate_policy().threshold


def should_route_to_triage(title: Optional[str], body: Optional[str]) -> tuple[bool, SizeEstimate]:
    """Determine if a card should be routed to triage based on its size.

    Returns (should_route, estimate) where should_route is True if the card
    is oversized and should be decomposed before dispatch.
    """
    threshold = load_size_gate_threshold()
    estimate = estimate_card_size(title, body)
    return estimate.is_oversized(threshold), estimate


def _event_payload(raw: object) -> dict:
    """Parse a stored event payload into a dict (empty on anything unexpected)."""
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _has_event(conn, task_id: str, kind: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM task_events WHERE task_id = ? AND kind = ? LIMIT 1", (task_id, kind),
    ).fetchone() is not None


def _already_decomposed(conn, task_id: str) -> bool:
    """True when ``task_id`` has a ``decomposed`` event (the root already fanned out).

    A decomposed root is intentionally oversized (its body is the original epic)
    and MUST NOT be re-triaged: a second decomposition is refused, so re-triaging
    would strand it forever. Used by both :func:`divert_promotion_to_triage` and
    :func:`promote_gate_verdict` for an early-exit check.
    """
    return _has_event(conn, task_id, "decomposed")


def latest_size_estimate(conn, task_id: str) -> Optional[dict]:
    """Newest ``size_estimate`` event payload for a card, or None when it has none."""
    row = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'size_estimate' "
        "ORDER BY id DESC LIMIT 1", (task_id,),
    ).fetchone()
    return None if row is None else _event_payload(row["payload"])


def record_size_estimate(
    conn, task_id: str, estimate: SizeEstimate, *,
    decision: str = "allowed", path: str = "creation",
    reason: Optional[str] = None, threshold: Optional[int] = None,
) -> None:
    """Record the size estimate on the card as a task event.

    ``decision`` (final for the card), ``path`` (which surface decided) and
    ``reason`` make the estimate machine-readable for later audits, the
    dispatcher and :func:`gate_ready_entry`.
    """
    from hermes_cli import kanban_db as kb

    if threshold is None:
        threshold = load_size_gate_threshold()
    try:
        kb._append_event(
            conn,
            task_id,
            "size_estimate",
            {
                "score": estimate.score,
                "acceptance_criteria_count": estimate.acceptance_criteria_count,
                "file_count": estimate.file_count,
                "deliverable_count": estimate.deliverable_count,
                "threshold": threshold,
                "decision": decision,
                "path": path,
                "reason": reason,
            },
        )
    except Exception:
        # The estimate is audit metadata: a failed event write must not block the
        # card, but it must not vanish silently either (a hole an audit would miss).
        logger.warning(
            "size gate: could not record the estimate for task %s (decision=%s)",
            task_id, decision, exc_info=True,
        )


def stamp_budget_estimate(body: Optional[str], estimate: SizeEstimate, threshold: int) -> str:
    """Prepend the one-line budget stamp to ``body`` (first line, blank line, body).

    Measure the UNSTAMPED body and stamp afterwards: the stamp is
    estimator-neutral, and a test pins that re-measuring it changes nothing.
    """
    line = BUDGET_ESTIMATE_STAMP.format(score=estimate.score, threshold=threshold)
    remainder = _as_text(body)
    return f"{line}\n\n{remainder}" if remainder else f"{line}\n"


def gate_ready_entry(
    conn, task_id: str, *, title: Optional[str], body: Optional[str],
    policy: Optional[SizeGatePolicy] = None, path: str = "promotion",
    record: bool = True,
) -> str:
    """Decide whether a card may land on ``ready``: ``"ready"`` or ``"triage"``.

    THE single decision point for every writer that parks a card on ``ready``
    (creation, ``recompute_ready``, ``unblock_task``, ``reopen_review_task``,
    ``promote_task``), so the rule cannot drift between surfaces. Rules, in order:

    1. The card has a ``claimed`` event -> ``"ready"``. A card that already ran is
       never re-litigated; this is what keeps review-reopen and dispatch-requeue
       safe.
    2. The newest ``size_estimate`` event decides:
       - ``triage`` / ``allowed`` are final -> ``"ready"``.
       - no event at all (legacy, pre-gate card) -> measure, record
         ``allowed``/``legacy`` -> ``"ready"``. Grandfathered ON PURPOSE: the live
         board's parked cards are not retro-diverted; every card this code creates
         carries an estimate from creation, so the hole closes forward.
       - ``advisory`` (created while parked) -> decide now: oversized and never
         decomposed -> record ``triage``/``oversized`` and return ``"triage"``;
         else record ``allowed`` (``epic_root`` when the card has a ``decomposed``
         event, else ``fits``) and return ``"ready"``. The epic-root exemption is
         load-bearing: an epic root is intentionally oversized and MUST wake so its
         orchestrator can judge completion, and a second decomposition is refused,
         so re-triaging it would strand it forever.
    3. Never raises: a gate bug must not stop a promotion. Falls back to
       ``"ready"`` with a logged warning.

    ``record=False`` is the dry-run switch (``promote_task(dry_run=True)``): the
    identical decision is reported and nothing is persisted.
    """
    try:
        if policy is None:
            policy = load_size_gate_policy()
        if _has_event(conn, task_id, "claimed"):
            return "ready"
        payload = latest_size_estimate(conn, task_id)
        if not payload:
            # Legacy / pre-gate card: measure it, but grandfather the decision.
            estimate = estimate_card_size(title, body)
            if record:
                record_size_estimate(
                    conn, task_id, estimate, decision="allowed", path=path,
                    reason="legacy", threshold=policy.threshold,
                )
            return "ready"
        decision = payload.get("decision")
        if decision in ("triage", "allowed", None):
            # Already final (``None`` = an estimate recorded before decisions
            # existed, i.e. the base gate's creation event).
            return "ready"
        if decision != "advisory":
            # Unknown decision value: never loop a card through triage forever.
            return "ready"
        estimate = estimate_card_size(title, body)
        if estimate.is_oversized(policy.threshold) and not _already_decomposed(conn, task_id):
            if record:
                record_size_estimate(
                    conn, task_id, estimate, decision="triage", path=path,
                    reason="oversized", threshold=policy.threshold,
                )
            return "triage"
        if record:
            record_size_estimate(
                conn, task_id, estimate, decision="allowed", path=path,
                reason="epic_root" if _already_decomposed(conn, task_id) else "fits", threshold=policy.threshold,
            )
        return "ready"
    except Exception:
        logger.warning(
            "size gate: ready-entry gate failed for task %s (path=%s); allowing the promotion",
            task_id, path, exc_info=True,
        )
        return "ready"


# --- Call-site wiring --------------------------------------------------------
# ``kanban_db`` is a facade at its line cap, so each writer of ``ready`` keeps
# ONE line of gate code here and the decision logic stays in this module. Every
# helper below delegates to :func:`gate_ready_entry` - never a second rule.


@dataclass(frozen=True)
class CreationGate:
    """Creation-time gate inputs: the resolved policy, the measurement, the verdict."""

    policy: SizeGatePolicy
    estimate: SizeEstimate
    oversized: bool


def prepare_creation_gate(title: Optional[str], body: Optional[str]) -> CreationGate:
    """Resolve the policy and measure the card ONCE per creation."""
    policy = load_size_gate_policy()
    estimate = estimate_card_size(title, body)
    return CreationGate(policy=policy, estimate=estimate, oversized=policy.is_oversized(estimate))


def creation_size_decision(status: str, *, oversized: bool) -> tuple[str, dict]:
    """Creation-time decision table, keyed on the status the card ACTUALLY lands in.

    Returns ``(status_after, record_kwargs)``: the status to insert, plus the
    ``decision``/``path``/``reason`` to record once the row exists.
    ``triage`` keeps its status (measured for a later audit), a ``ready`` card is
    diverted when oversized, and a parked card gets an ``advisory`` decision that
    :func:`gate_ready_entry` re-decides at promotion.
    """
    if status == "ready":
        if oversized:
            return "triage", {"decision": "triage", "path": "creation", "reason": "oversized"}
        return status, {"decision": "allowed", "path": "creation", "reason": "fits"}
    if status == "triage":
        return status, {
            "decision": "triage", "path": "creation_triage",
            "reason": "oversized" if oversized else "fits",
        }
    return status, {
        "decision": "advisory", "path": "creation_parked",
        "reason": "oversized" if oversized else "fits",
    }


def record_creation_gate(conn, task_id: str, gate: CreationGate, record_kwargs: dict) -> None:
    """Record the creation estimate under the decision table's verdict."""
    record_size_estimate(
        conn, task_id, gate.estimate, threshold=gate.policy.threshold, **record_kwargs,
    )


def divert_promotion_to_triage(
    conn, task_id: str, *, policy: Optional[SizeGatePolicy] = None, path: str = "promotion",
) -> bool:
    """Gate a promotion; True when the card was diverted to ``triage``.

    Callers must skip their own status write when this returns True: the gate
    already recorded ``decision=triage`` and this flipped the row. Measures the
    card's CURRENT title/body, which is also what the creation gate measured.
    A decomposed root is never diverted (early exit via :func:`_already_decomposed`),
    but the size estimate is still recorded for audit.
    """
    if policy is None:
        policy = load_size_gate_policy()
    title, body = _task_title_body(conn, task_id)
    verdict = gate_ready_entry(
        conn, task_id, title=title, body=body, policy=policy, path=path,
    )
    if verdict != "triage":
        return False
    conn.execute(
        "UPDATE tasks SET status = 'triage' WHERE id = ? AND status IN ('todo', 'blocked')",
        (task_id,),
    )
    return True


def promote_gate_verdict(
    conn, task_id: str, *, policy: Optional[SizeGatePolicy] = None, record: bool = True,
) -> Optional[str]:
    """Manual-promotion gate: ``None`` to promote, else the operator-facing refusal.

    When it diverts it moves the card to ``triage`` and says so; when it refuses
    it names the score and the threshold so the operator can act on the number.
    A decomposed root is never refused (handled by :func:`gate_ready_entry` via
    :func:`_already_decomposed`).
    """
    if policy is None:
        policy = load_size_gate_policy()
    title, body = _task_title_body(conn, task_id)
    if gate_ready_entry(
        conn, task_id, title=title, body=body, policy=policy, path="promote", record=record,
    ) != "triage":
        return None
    if record:
        conn.execute(
            "UPDATE tasks SET status = 'triage' WHERE id = ? AND status IN ('todo', 'blocked')",
            (task_id,),
        )
    score = estimate_card_size(title, body).score
    verb = "routed" if record else "would route"
    return (
        f"size gate: card is oversized (score {score} > threshold "
        f"{policy.threshold}); {verb} to triage for decomposition"
    )


def _task_title_body(conn, task_id: str) -> tuple[Optional[str], Optional[str]]:
    """``(title, body)`` for measuring ``task_id`` (missing row -> ``(None, None)``)."""
    row = conn.execute("SELECT title, body FROM tasks WHERE id = ?", (task_id,)).fetchone()
    return (row["title"], row["body"]) if row else (None, None)

"""Completion evidence gate shared by both Kanban lifecycle surfaces.

The CLI (``hermes kanban complete`` / ``request-review`` in ``hermes_cli/kanban.py``) and the
worker tools (``kanban_complete`` / ``kanban_request_review`` in ``tools/kanban_tools.py``) each
call the goal judge before a terminal handoff. This module owns the one decision both of them
must make identically: *which cards are judged at all*.

``kanban.evidence_gate`` in ``config.yaml``:

``goal_mode`` (default)
    Only ``goal_mode`` cards are judged — exactly the upstream behaviour.
``verify_line``
    ``goal_mode`` cards, plus any card whose body carries a ``verify:`` line (a line that starts,
    after optional whitespace, with ``verify:`` in any casing). Lines inside fenced code blocks are
    quoted material, not the card's own contract, and do not count.
``all``
    Every card.

An unknown value logs a warning and falls back to the default, so a typo never wedges the board.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

SCOPES = ("goal_mode", "verify_line", "all")
DEFAULT_SCOPE = "goal_mode"

_VERIFY_LINE = re.compile(r"^\s*verify:", re.IGNORECASE)
_FENCE = re.compile(r"^\s*(```|~~~)")


def configured_scope() -> str:
    """``kanban.evidence_gate`` from config, normalised; the default on any problem."""
    try:
        from hermes_cli.config import load_config_readonly

        kanban_cfg = (load_config_readonly() or {}).get("kanban") or {}
        raw = kanban_cfg.get("evidence_gate") if isinstance(kanban_cfg, dict) else None
    except Exception:
        # The gate runs on every terminal handoff; an unreadable config must not block them.
        logger.warning("kanban.evidence_gate: config unreadable; using %r", DEFAULT_SCOPE,
                       exc_info=True)
        return DEFAULT_SCOPE
    if raw is None:
        return DEFAULT_SCOPE
    scope = str(raw).strip().lower()
    if scope in SCOPES:
        return scope
    logger.warning("kanban.evidence_gate=%r is not one of %s; using %r", raw, SCOPES, DEFAULT_SCOPE)
    return DEFAULT_SCOPE


def has_verify_line(body: Optional[str]) -> bool:
    """True iff ``body`` has a ``verify:`` line outside fenced code blocks."""
    fence: Optional[str] = None
    for line in (body or "").splitlines():
        marker = _FENCE.match(line)
        if marker:
            if fence is None:
                fence = marker.group(1)
            elif marker.group(1) == fence:
                fence = None
            continue
        if fence is None and _VERIFY_LINE.match(line):
            return True
    return False


def gate_applies(task: Any, scope: Optional[str] = None) -> bool:
    """Whether a terminal handoff of ``task`` must pass the evidence judge."""
    if task is None:
        return False
    if getattr(task, "goal_mode", False):
        return True
    scope = scope or configured_scope()
    if scope == "all":
        return True
    if scope == "verify_line":
        return has_verify_line(getattr(task, "body", None))
    return False


# Upper bound on the metadata JSON shown to the judge: enough for shas, record names and test
# counts, small enough that a log dumped into metadata cannot crowd out the summary.
METADATA_EVIDENCE_LIMIT = 4096


def evidence_text(text: Optional[str], metadata: Any = None) -> str:
    """The handoff evidence the judge sees: summary/result plus compact, bounded metadata.

    Workers routinely put the proof (commit sha, record filenames, test counts) in structured
    metadata and keep the summary short; judging the prose alone rejected real evidence.
    """
    text = (text or "").strip()
    if not metadata:
        return text
    try:
        blob = json.dumps(metadata, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        blob = str(metadata)
    if len(blob) > METADATA_EVIDENCE_LIMIT:
        blob = blob[:METADATA_EVIDENCE_LIMIT] + "…[metadata truncated]"
    return f"{text}\n\nmetadata: {blob}".strip()


def dependency_wait_message(task_id: str, waits: list[str]) -> str:
    """What both surfaces tell the caller after a decomposition-only handoff was converted."""
    return (f"{task_id} converted to a dependency wait on {', '.join(waits)}; it completes when "
            f"they do (the dispatcher re-runs it once they are done, then hand off with evidence)")


def dependency_wait_reason(handoff: str, judge_reason: Optional[str], summary: Optional[str]) -> str:
    """Run summary recorded on the converted task so its re-run sees why it is waiting."""
    head = (summary or "").strip()[:1000]
    note = (f"[evidence gate] decomposition-only {handoff}: the judge found no evidence that the "
            f"work itself is done ({judge_reason or 'no reason given'}); waiting on the cards it "
            f"created instead of completing.")
    return f"{head}\n\n{note}".strip()


def decomposition_wait_enabled(scope: Optional[str] = None) -> bool:
    """Whether a rejected decomposition-only handoff is converted into a dependency wait.

    Only when the operator opted into a wider scope: the default (``goal_mode``) keeps upstream
    behaviour exactly, including refusing a goal_mode card's decomposition-only completion.
    """
    return (scope or configured_scope()) != DEFAULT_SCOPE

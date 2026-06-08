"""Kanban fan-in resolver — root-fix slice for review BLOCK/NEED_MORE.

When an implementation or review graph finishes at a final review task
with verdict ``BLOCK`` or ``NEED_MORE``, the originating operator can
be left waiting: ``notify-subscribe`` is a passive terminal delivery,
not an active fan-in/remediation resolver. This module classifies the
final task into three independent dimensions and, in apply mode,
creates at most one deduped fix card + one fix-review card linked
back to the final task so a remediation graph exists on the board.

Classification heuristics
-------------------------
* ``task_verdict``: ``GO`` | ``BLOCK`` | ``NEED_MORE`` — parsed from
  the final review task's summary / result / body / latest comments.
* ``ack_status``: ``PENDING`` | ``SENT`` | ``FAILED`` |
  ``WATCHDOG_FALLBACK`` | ``MANUAL_RELAYED`` — pulled from a comment
  whose body starts with ``ack-status:`` (the documented surface that
  gateway watchdogs / manual relays write). Default ``PENDING``.
* ``remediation_status``: ``NONE`` | ``REQUIRED`` | ``CREATED`` |
  ``BLOCKED``. ``BLOCKED`` is reserved for unsafe sentinel categories
  the resolver must never auto-fix.

Safety
------
The resolver only auto-creates remediation cards when the final review
task's combined text:

* contains a ``Verdict: BLOCK`` or ``Verdict: NEED_MORE`` line, AND
* references a ``review-required:`` or ``handoff:`` sentinel, AND
* stays within the caller's ``max_fan_in_threshold`` graph-size bound,
  when one is supplied, AND
* does NOT mention any unsafe blocker category — secret/token/key,
  destructive data loss, auth/credential/login, user input required,
  live money / trading / execution.

Card bodies are sanitised: only the verdict, a short reason summary,
and the originating task id are included. Raw transcript paths,
secrets, and full review text never reach the new task body.
"""

from __future__ import annotations

import json
import re
import sqlite3
from typing import Any, Iterable, Optional

from hermes_cli import kanban_db as kb


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_VERDICTS = ("GO", "BLOCK", "NEED_MORE")
VALID_ACK_STATUSES = (
    "PENDING", "SENT", "FAILED", "WATCHDOG_FALLBACK", "MANUAL_RELAYED",
)
VALID_REMEDIATION = ("NONE", "REQUIRED", "CREATED", "BLOCKED")

_VERDICT_RE = re.compile(r"verdict\s*[:=]\s*(GO|BLOCK|NEED_MORE)\b", re.IGNORECASE)
_ACK_RE = re.compile(
    r"ack[-_ ]?status\s*[:=]\s*(PENDING|SENT|FAILED|WATCHDOG_FALLBACK|MANUAL_RELAYED)\b",
    re.IGNORECASE,
)

# Sentinels that mean "an operator/worker explicitly handed back for human
# review/handoff" — the only categories we are allowed to auto-resolve.
_AUTORESOLVE_SENTINELS = (
    re.compile(r"review[-_]required", re.IGNORECASE),
    re.compile(r"\bhandoff\b", re.IGNORECASE),
)

# Unsafe blocker categories. Matching any one of these in the combined
# review text forces ``remediation_status=BLOCKED`` — no card creation,
# even in apply mode.
_UNSAFE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(secret|secrets|token|api[-_ ]?key|password|credential|cred)\b", re.IGNORECASE),
    re.compile(r"\bleak(ed|ing)?\b", re.IGNORECASE),
    re.compile(r"\b(rm\s+-rf|drop\s+table|truncate\s+table|destructive|data[-_ ]?loss|wipe)\b", re.IGNORECASE),
    re.compile(r"\b(auth|authentication|login|oauth|sso)\b", re.IGNORECASE),
    re.compile(r"\b(user[-_ ]?input|needs?\s+user\s+input|prompt\s+the\s+user)\b", re.IGNORECASE),
    re.compile(r"\b(live[-_ ]?(money|trading|exec(ution)?))\b", re.IGNORECASE),
    re.compile(r"\b(real[-_ ]?money|production[-_ ]?order|execute\s+trade)\b", re.IGNORECASE),
)

_IDEMPOTENCY_FIX_PREFIX = "fanin-fix:"
_IDEMPOTENCY_REVIEW_PREFIX = "fanin-fix-review:"
_IDEMPOTENCY_REPORTER_PREFIX = "fanin-reporter:"


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def _gather_review_text(conn: sqlite3.Connection, task_id: str) -> str:
    """Concatenate body / result / latest comments for parsing.

    Never include free-form payload bodies — only the text fields the
    operator can reasonably reason about. Cap each comment to its first
    400 chars so a runaway log dump can't blow up the regex pass.
    """
    row = conn.execute(
        "SELECT body, result FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()
    if row is None:
        return ""
    parts: list[str] = []
    for field in ("body", "result"):
        v = row[field]
        if v:
            parts.append(str(v))
    # Worker completions usually store the operator-facing Verdict in
    # task_runs.summary, not tasks.result. Include recent run summaries
    # and completed-event summaries so `kanban_complete(summary=...)`
    # handoffs classify correctly even when result is empty.
    for r in conn.execute(
        "SELECT summary, metadata, error FROM task_runs "
        "WHERE task_id = ? ORDER BY id DESC LIMIT 5",
        (task_id,),
    ).fetchall():
        for field in ("summary", "metadata", "error"):
            v = r[field]
            if v:
                parts.append(str(v)[:800])
    for ev in conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? "
        "ORDER BY id DESC LIMIT 10",
        (task_id,),
    ).fetchall():
        payload = ev["payload"]
        if not payload:
            continue
        try:
            data = json.loads(payload)
        except (TypeError, json.JSONDecodeError):
            parts.append(str(payload)[:400])
            continue
        summary = data.get("summary") if isinstance(data, dict) else None
        if summary:
            parts.append(str(summary)[:400])
    for c in kb.list_comments(conn, task_id):
        if c.body:
            parts.append(c.body[:400])
    return "\n".join(parts)


def _parse_verdict(text: str) -> str:
    m = _VERDICT_RE.search(text or "")
    if not m:
        # Default to GO when no verdict marker is present — callers can
        # override with an explicit verdict in the review summary.
        return "GO"
    return m.group(1).upper()


def _parse_ack_status(text: str) -> str:
    m = _ACK_RE.search(text or "")
    if not m:
        return "PENDING"
    return m.group(1).upper()


def _is_unsafe(text: str) -> Optional[str]:
    """Return the matching unsafe-category snippet, or ``None`` if safe."""
    for pat in _UNSAFE_PATTERNS:
        m = pat.search(text or "")
        if m:
            return m.group(0)
    return None


def _has_autoresolve_sentinel(text: str) -> bool:
    return any(p.search(text or "") for p in _AUTORESOLVE_SENTINELS)


def _walk_ancestors(conn: sqlite3.Connection, task_id: str) -> list[str]:
    """Return all transitive ancestors of ``task_id`` (excluding it)
    plus the task itself, deduped & sorted for determinism."""
    seen: set[str] = set()
    stack: list[str] = [task_id]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        for pid in kb.parent_ids(conn, cur):
            if pid not in seen:
                stack.append(pid)
    return sorted(seen)


def _root_ancestor(conn: sqlite3.Connection, task_id: str) -> Optional[str]:
    """Return the deterministic root ancestor (oldest, then min id).

    Used as the ``origin_return`` field — the originator the ledger
    points back to. When the task has multiple roots, prefer the
    earliest ``created_at`` then lowest id for stability.
    """
    ancestors = _walk_ancestors(conn, task_id)
    roots: list[str] = []
    for aid in ancestors:
        if not kb.parent_ids(conn, aid):
            roots.append(aid)
    if not roots:
        return None
    rows = conn.execute(
        "SELECT id, created_at FROM tasks WHERE id IN ("
        + ",".join("?" * len(roots)) + ")",
        roots,
    ).fetchall()
    rows = sorted(rows, key=lambda r: (r["created_at"] or 0, r["id"]))
    return rows[0]["id"] if rows else None


def _sanitize_snippet(text: str, *, max_len: int = 240) -> str:
    """Strip suspicious tokens (paths starting with ``/``, secret-like
    runs of hex) and clip to ``max_len`` characters. Always operates on
    the first non-empty line so we never include multi-line payloads.
    """
    if not text:
        return ""
    line = next((ln for ln in text.splitlines() if ln.strip()), "")
    # Scrub anything that looks like an absolute path.
    line = re.sub(r"(?:[a-zA-Z]:)?(?:/[\w.\-]+){2,}", "<path>", line)
    # Scrub long hex/base64-ish runs (likely tokens).
    line = re.sub(r"[A-Za-z0-9_\-]{32,}", "<redacted>", line)
    line = line.strip()
    if len(line) > max_len:
        line = line[: max_len - 1].rstrip() + "…"
    return line


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify(conn: sqlite3.Connection, final_task_id: str) -> dict[str, Any]:
    """Classify the final review task without mutating any state."""
    task = kb.get_task(conn, final_task_id)
    if task is None:
        raise ValueError(f"unknown task {final_task_id}")
    text = _gather_review_text(conn, final_task_id)
    verdict = _parse_verdict(text)
    ack = _parse_ack_status(text)

    if verdict == "GO":
        remediation = "NONE"
        unsafe_hit: Optional[str] = None
    else:
        unsafe_hit = _is_unsafe(text)
        if unsafe_hit:
            remediation = "BLOCKED"
        elif _has_autoresolve_sentinel(text):
            remediation = "REQUIRED"
        else:
            # No autoresolve sentinel + non-GO verdict => still
            # remediation-required, but we err on the side of safety
            # and refuse to auto-fix without an explicit handoff marker.
            remediation = "BLOCKED"
            unsafe_hit = "missing review-required/handoff sentinel"

    return {
        "task_verdict": verdict,
        "ack_status": ack,
        "remediation_status": remediation,
        "blocked_reason": unsafe_hit,
        "final_task": final_task_id,
        "final_status": task.status,
    }


def _existing_remediation_ids(
    conn: sqlite3.Connection, final_task_id: str
) -> tuple[Optional[str], Optional[str]]:
    """Look up previously created fix / fix-review cards for a final task
    by idempotency_key. Returns ``(fix_id, fix_review_id)`` (each may be
    ``None``).
    """
    fix_key = _IDEMPOTENCY_FIX_PREFIX + final_task_id
    review_key = _IDEMPOTENCY_REVIEW_PREFIX + final_task_id
    rows = conn.execute(
        "SELECT id, idempotency_key FROM tasks "
        "WHERE idempotency_key IN (?, ?) AND status != 'archived'",
        (fix_key, review_key),
    ).fetchall()
    fix_id: Optional[str] = None
    review_id: Optional[str] = None
    for r in rows:
        if r["idempotency_key"] == fix_key:
            fix_id = r["id"]
        elif r["idempotency_key"] == review_key:
            review_id = r["id"]
    return fix_id, review_id


def _existing_reporter_id(conn: sqlite3.Connection, final_task_id: str) -> Optional[str]:
    """Return the deduped final fan-in reporter card for a final task."""
    row = conn.execute(
        "SELECT id FROM tasks WHERE idempotency_key = ? "
        "AND status != 'archived' ORDER BY created_at DESC LIMIT 1",
        (_IDEMPOTENCY_REPORTER_PREFIX + final_task_id,),
    ).fetchone()
    return row["id"] if row else None


def resolve_fanin(
    conn: sqlite3.Connection,
    final_task_id: str,
    *,
    apply: bool = False,
    fix_assignee: Optional[str] = None,
    review_assignee: Optional[str] = None,
    reporter_assignee: Optional[str] = None,
    board: Optional[str] = None,
    max_fan_in_threshold: Optional[int] = None,
) -> dict[str, Any]:
    """Classify and (optionally) create deduped remediation cards.

    ``apply=False`` is dry-run: classification only, no writes.

    When ``apply=True`` and ``remediation_status == 'REQUIRED'``, create
    at most one fix card + one fix-review card. Both carry deterministic
    ``idempotency_key`` values, so re-running the resolver on the same
    final task is a safe no-op.

    ``max_fan_in_threshold`` bounds automatic remediation by graph size.
    If the final task's transitive dependency graph is larger than this
    positive threshold, the resolver returns ``BLOCKED`` and writes
    nothing. Operators can raise the threshold explicitly after reviewing
    the graph.

    Returns a machine-readable ledger:

    .. code-block:: json

        {
          "board": "default",
          "origin_return": "t_…",
          "graph_task_ids": ["t_…", ...],
          "final_task": "t_…",
          "task_verdict": "BLOCK",
          "ack_status": "PENDING",
          "remediation_status": "CREATED",
          "remediation_task_ids": {"fix": "t_…", "fix_review": "t_…"},
          "dry_run": false,
          "blocked_reason": null
        }
    """
    if max_fan_in_threshold is not None and max_fan_in_threshold < 1:
        raise ValueError("max_fan_in_threshold must be >= 1")

    base = classify(conn, final_task_id)
    verdict = base["task_verdict"]
    remediation = base["remediation_status"]
    graph_task_ids = _walk_ancestors(conn, final_task_id)
    graph_task_count = len(graph_task_ids)

    threshold_blocked_reason: Optional[str] = None
    if (
        max_fan_in_threshold is not None
        and graph_task_count > max_fan_in_threshold
        and remediation == "REQUIRED"
    ):
        remediation = "BLOCKED"
        threshold_blocked_reason = (
            f"fan-in graph size {graph_task_count} exceeds threshold "
            f"{max_fan_in_threshold}"
        )

    # Existing cards (if any) — surface them even on dry-run so the
    # operator can tell the resolver already ran.
    existing_fix, existing_review = _existing_remediation_ids(conn, final_task_id)
    if existing_fix and existing_review:
        # Treat a fully-formed prior run as the canonical CREATED state.
        remediation = "CREATED"

    rem_ids: dict[str, Optional[str]] = {"fix": existing_fix, "fix_review": existing_review}

    ledger = {
        "board": kb.get_current_board() if board is None else board,
        "origin_return": _root_ancestor(conn, final_task_id),
        "graph_task_ids": graph_task_ids,
        "graph_task_count": graph_task_count,
        "max_fan_in_threshold": max_fan_in_threshold,
        "final_task": final_task_id,
        "task_verdict": verdict,
        "ack_status": base["ack_status"],
        "remediation_status": remediation,
        "remediation_task_ids": rem_ids,
        "reporter_task_id": _existing_reporter_id(conn, final_task_id),
        "dry_run": not apply,
        "blocked_reason": threshold_blocked_reason or base["blocked_reason"],
    }

    if not apply:
        return ledger
    if remediation != "REQUIRED":
        # Either NONE / BLOCKED / already CREATED — nothing to do.
        return ledger

    # ----- Build sanitised card bodies. -----
    text = _gather_review_text(conn, final_task_id)
    verdict_summary = _sanitize_snippet(text)
    fix_body = (
        f"Auto-created from {final_task_id} (Verdict: {verdict}).\n"
        f"Summary: {verdict_summary or '(no summary)'}\n"
        f"Apply minimal remediation. See the parent review card for context."
    )
    review_body = (
        f"Auto-created fix review for {final_task_id} (Verdict: {verdict}).\n"
        f"Re-run reviewer flow against the fix output."
    )

    fix_key = _IDEMPOTENCY_FIX_PREFIX + final_task_id
    review_key = _IDEMPOTENCY_REVIEW_PREFIX + final_task_id

    fix_id = existing_fix or kb.create_task(
        conn,
        title=f"fix: remediate {final_task_id} ({verdict})",
        body=fix_body,
        assignee=fix_assignee,
        created_by="fanin-resolver",
        parents=[final_task_id],
        idempotency_key=fix_key,
        initial_status="running",
    )
    review_id = existing_review or kb.create_task(
        conn,
        title=f"fix-review: verify {final_task_id} remediation",
        body=review_body,
        assignee=review_assignee,
        created_by="fanin-resolver",
        parents=[fix_id],
        idempotency_key=review_key,
        initial_status="running",
    )

    # Final fan-in reporter (optional): if the caller named one and we
    # have no existing reporter card, gate it behind the fix-review.
    # Treated as best-effort linkage; we never duplicate it.
    if reporter_assignee:
        reporter_key = _IDEMPOTENCY_REPORTER_PREFIX + final_task_id
        reporter_id = _existing_reporter_id(conn, final_task_id)
        if reporter_id is None:
            reporter_id = kb.create_task(
                conn,
                title=f"fanin-report: {final_task_id}",
                body=f"Report back to originator once {review_id} clears.",
                assignee=reporter_assignee,
                created_by="fanin-resolver",
                parents=[review_id],
                idempotency_key=reporter_key,
                initial_status="running",
            )
        ledger["reporter_task_id"] = reporter_id

    ledger["remediation_task_ids"] = {"fix": fix_id, "fix_review": review_id}
    ledger["remediation_status"] = "CREATED"
    return ledger

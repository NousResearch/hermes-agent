"""Dependency gating and review provenance for kanban boards.

Split from ``hermes_cli.kanban_db`` (facade size cap); the facade re-exports every name
here. Late facade imports inside functions: the facade imports this module at its tail.

Parent gate: "terminal" is ONE predicate (:func:`_parent_terminal_sql`) shared by the
claim, ``recompute_ready``, ``unsatisfied_parents`` and ``link_tasks``. On a board that
opts in with ``"parent_gate": "merged"`` in its ``board.json``, a ``done`` parent whose
completion contract is a pull request also needs a ``merged`` event: an approved PR is
not yet on the base branch, and a child cut from that branch builds on code that is not
there. ``link_tasks`` used to test ``status in (done, archived)`` and left a child
``ready`` that the claim then refused (measured 2026-10-02: t_b7354264 under a done
parent whose PR #182 was still open).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Optional

# A parent whose completion contract is a PR URL (``prepare_acceptance`` binds
# the published PR there) is only merged once someone records a ``merged``
# event on it — the release tooling that performs the merge does.
_PR_CONTRACT_LIKE = "https://github.com/%/pull/%"
_parent_gate_cache: dict[str, tuple[float, bool]] = {}


def _parent_gate_is_merged(conn: sqlite3.Connection) -> bool:
    """True when *conn*'s board opts into ``"parent_gate": "merged"``.

    Read from the ``board.json`` beside the board's database file, cached by
    mtime. Any doubt (in-memory DB, missing or malformed file, other value) is
    the default gate — the behaviour every other board keeps.
    """
    from hermes_cli.kanban_db_connect import _main_db_file

    db_file = _main_db_file(conn)
    if not db_file:
        return False
    meta = Path(db_file).with_name("board.json")
    try:
        mtime = meta.stat().st_mtime
    except OSError:
        return False
    cached = _parent_gate_cache.get(str(meta))
    if cached is not None and cached[0] == mtime:
        return cached[1]
    try:
        merged = json.loads(meta.read_text(encoding="utf-8-sig")).get("parent_gate") == "merged"
    except (OSError, ValueError, AttributeError):
        merged = False
    _parent_gate_cache[str(meta)] = (mtime, merged)
    return merged


def _parent_terminal_sql(conn: sqlite3.Connection) -> str:
    """SQL predicate over parent row ``p``: the parent no longer gates its children."""
    if not _parent_gate_is_merged(conn):
        return "p.status IN ('done', 'archived')"
    return (
        "p.status = 'archived' OR (p.status = 'done' AND ("
        "p.completion_contract IS NULL "
        f"OR p.completion_contract NOT LIKE '{_PR_CONTRACT_LIKE}' "
        "OR EXISTS (SELECT 1 FROM task_events e WHERE e.task_id = p.id AND e.kind = 'merged')))"
    )


def record_merged(conn: sqlite3.Connection, task_id: str, payload: Optional[dict] = None) -> bool:
    """Record that ``task_id``'s PR landed on the base branch, then re-promote.

    Idempotent: a second call for the same task is a no-op returning False.
    Only meaningful on a ``"parent_gate": "merged"`` board, harmless elsewhere.
    """
    from hermes_cli.kanban_db import _append_event, recompute_ready, write_txn

    with write_txn(conn):
        if conn.execute("SELECT 1 FROM tasks WHERE id = ?", (task_id,)).fetchone() is None:
            return False
        if conn.execute(
            "SELECT 1 FROM task_events WHERE task_id = ? AND kind = 'merged' LIMIT 1", (task_id,),
        ).fetchone():
            return False
        _append_event(conn, task_id, "merged", payload)
    recompute_ready(conn)
    return True


def _parent_is_terminal(conn: sqlite3.Connection, parent_id: str) -> bool:
    """Whether ``parent_id`` no longer gates its children, under the claim's predicate."""
    return conn.execute(
        f"SELECT 1 FROM tasks p WHERE p.id = ? AND ({_parent_terminal_sql(conn)})", (parent_id,),
    ).fetchone() is not None


def implementer_from_run_history(
    conn: sqlite3.Connection, task_id: str, *,
    exclude_run_id: Optional[int], exclude_profile: Optional[str],
) -> Optional[str]:
    """Degraded implementer provenance for a card that reached ``review``
    without ever calling ``request_review`` — ``recompute_ready``'s
    ``promoted {"status": "review"}`` and a reconciler's raw ``UPDATE`` both
    skip it (#t_1e569887). Falls back to the profile of the latest ENDED run
    that is not the reviewer's own active run and whose ``metadata`` carries
    no ``review_outcome`` (the key the ``sdlc-review`` skill writes on
    approve) — that excludes a prior reviewer's approval run and lands on the
    implementer's own closing run. ``None`` when no such run exists; the
    caller then leaves ``assignee`` untouched, the same degradation
    :func:`reopen_review_task` already applies when its own event has no
    implementer.

    ``review_outcome`` alone is not enough: it is written only on the
    ``sdlc-review`` skill's APPROVAL path. A reviewer run that instead ends
    ``blocked``/``reclaimed``/``crashed``/``timed_out`` closes with
    ``metadata=NULL`` and, being the most recently ENDED run, would outrank
    the implementer's own closing run — electing the reviewer as the card's
    next assignee via the very rework path meant to route it away from them
    (measured on a synthetic board: the ``ready`` lane then dispatches
    ``profile=<reviewer>``, exactly the mix-up the review-lane guard family
    exists to prevent). ``exclude_profile`` — the caller's already-resolved
    reviewer — is the second, independent filter that closes this: any row
    whose ``profile`` matches it is skipped regardless of ``metadata``.
    """
    from hermes_cli.kanban_db import _canonical_assignee, _json_dict, _nonblank_str, _row_get

    rows = conn.execute(
        "SELECT id, profile, metadata FROM task_runs "
        "WHERE task_id = ? AND ended_at IS NOT NULL "
        "ORDER BY id DESC", (task_id,),
    ).fetchall()
    for row in rows:
        if exclude_run_id is not None and row["id"] == exclude_run_id:
            continue
        profile = _nonblank_str(_row_get(row, "profile"))
        if profile is not None and exclude_profile is not None:
            if _canonical_assignee(profile) == exclude_profile:
                continue
        if "review_outcome" in _json_dict(_row_get(row, "metadata")):
            continue
        if profile is not None:
            return profile
    return None

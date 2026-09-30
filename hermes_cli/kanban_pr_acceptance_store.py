"""Persist acceptance with the same ownership snapshot as the terminal write."""
from __future__ import annotations

import json
from dataclasses import dataclass

from hermes_cli.kanban_db_connect import write_txn
from hermes_cli.kanban_pr_acceptance import _PR, collect_acceptance


def _snapshot(conn, task_id):
    row = conn.execute("SELECT current_run_id, status, completion_contract FROM tasks WHERE id=?", (task_id,)).fetchone()
    return tuple(row) if row else None


def _historical_published_pr(conn, task_id, contract):
    """Omitted keys inherit; explicit invalid evidence stops the search."""
    rows = conn.execute(
        "SELECT metadata FROM task_runs WHERE task_id=? AND metadata IS NOT NULL ORDER BY id DESC",
        (task_id,),
    )
    for row in rows:
        try:
            metadata = json.loads(row["metadata"])
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(metadata, dict) or "published_pr" not in metadata:
            continue
        published_pr = metadata["published_pr"]
        match = _PR.fullmatch(published_pr) if isinstance(published_pr, str) else None
        if match and (contract == published_pr or contract == match[1]):
            return published_pr
        return None
    return None


def prepare_acceptance(conn, task_id, expected_run_id, metadata):
    snapshot = _snapshot(conn, task_id)
    if snapshot is None:
        return False
    run_id, status, contract = snapshot
    if not contract or contract == "local-only":
        return None
    if status not in {"running", "ready", "blocked", "review"} or (expected_run_id is not None and run_id != expected_run_id):
        return False
    published_pr = metadata.get("published_pr") if isinstance(metadata, dict) else None
    if not isinstance(metadata, dict) or "published_pr" not in metadata:
        published_pr = _historical_published_pr(conn, task_id, contract)
    match = _PR.fullmatch(published_pr) if isinstance(published_pr, str) else None
    # Publication binds once. Retrying cannot replace the task's PR with a green sibling.
    if match and contract == match[1]:
        with write_txn(conn):
            if _snapshot(conn, task_id) != snapshot:
                return False
            conn.execute("UPDATE tasks SET completion_contract=? WHERE id=?", (published_pr, task_id))
        snapshot = (run_id, status, published_pr)
        contract = published_pr
    # The assignee profile's gh login owns the repo: acceptance must not run as
    # the ambient login of whichever process completes the card (#122689).
    assignee = conn.execute("SELECT assignee FROM tasks WHERE id=?", (task_id,)).fetchone()["assignee"]
    return snapshot, collect_acceptance(contract, published_pr, assignee=assignee)


@dataclass(frozen=True)
class _ReviewSnapshot:
    task_id: str
    state: tuple
    handoffs: tuple
    event: tuple | None
    published_pr: str


@dataclass(frozen=True)
class _ReviewAcceptance:
    snapshot: _ReviewSnapshot
    receipt: dict


def _review_snapshot(conn, task_id):
    row = conn.execute(
        "SELECT current_run_id, status, completion_contract, assignee, claim_lock "
        "FROM tasks WHERE id=?", (task_id,),
    ).fetchone()
    if not row or row["status"] != "review" or row["claim_lock"] is not None:
        return None
    contract = row["completion_contract"]
    if not contract or contract == "local-only":
        return None
    # Include the newest event even when a metadata-free handoff made no run.
    event = conn.execute(
        "SELECT id, run_id, payload FROM task_events WHERE task_id=? "
        "AND kind='review_requested' ORDER BY id DESC LIMIT 1", (task_id,),
    ).fetchone()
    handoffs = []
    for run in conn.execute(
        "SELECT id, metadata FROM task_runs WHERE task_id=? "
        "AND outcome='review_requested' ORDER BY id DESC", (task_id,),
    ):
        handoffs.append(tuple(run))
        try:
            metadata = json.loads(run["metadata"]) if run["metadata"] is not None else {}
        except (TypeError, ValueError):
            return None
        if not isinstance(metadata, dict):
            return None
        if "published_pr" not in metadata:
            continue
        url = metadata["published_pr"]
        match = _PR.fullmatch(url) if isinstance(url, str) else None
        if not match or contract not in (url, match[1]):
            return None
        return _ReviewSnapshot(task_id, tuple(row), tuple(handoffs),
                               tuple(event) if event else None, url)
    return None


def record_acceptance(conn, task_id, acceptance):
    """Called under complete_task's write_txn, before its terminal UPDATE."""
    from hermes_cli.kanban_db import _append_event
    if isinstance(acceptance, _ReviewAcceptance):
        review = acceptance.snapshot
        receipt = acceptance.receipt
        if (review.task_id != task_id or _review_snapshot(conn, task_id) != review
                or not receipt.get("ok") or not receipt.get("merged")
                or receipt.get("pr_url") != review.published_pr):
            return False
        # Bind a repo-only contract only after successful evidence, in the
        # same transaction as completion. Failed polls never mutate the card.
        conn.execute("UPDATE tasks SET completion_contract=? WHERE id=?",
                     (review.published_pr, task_id))
        snapshot = review.state[:3]
    else:
        snapshot, receipt = acceptance
        if _snapshot(conn, task_id) != snapshot:
            return False
    _append_event(conn, task_id, "pr_acceptance", receipt, run_id=snapshot[0])
    if not receipt["ok"]:
        detail = f"PR acceptance {receipt['classification']}: {receipt.get('detail', '')} {receipt['recovery']}"
        conn.execute("UPDATE tasks SET last_failure_error=? WHERE id=?", (detail, task_id))
    return receipt["ok"]

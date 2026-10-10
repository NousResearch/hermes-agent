"""Deterministic merged-PR reconciliation, with no GitHub I/O under board locks."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import sqlite3
import threading
import time

from hermes_constants import hermes_home_key
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.kanban_pr_acceptance import collect_acceptance
from hermes_cli.kanban_pr_acceptance_store import _ReviewAcceptance, _review_snapshot


_CACHE_LIMIT = 256
_POLL_TTL = 60.0
_MAX_BACKOFF = 900.0
_MAX_POLLS_PER_TICK = 20
_CACHE_LOCK = threading.Lock()


@dataclass
class _CachedReceipt:
    receipt: dict | None = None  # None reserves a single in-flight query.
    expires: float = 0.0
    failures: int = 0


_CACHE: OrderedDict[tuple, _CachedReceipt] = OrderedDict()


def _cache_key(snapshot, db_path):
    # One board may be served from several homes with different gh identities.
    return (str(db_path), hermes_home_key(), snapshot.published_pr, snapshot.state[3])


def _cached_acceptance(snapshot, db_path, *, allow_query):
    key = _cache_key(snapshot, db_path)
    with _CACHE_LOCK:
        previous = _CACHE.get(key)
        if previous is not None:
            _CACHE.move_to_end(key)
            if previous.receipt is None or time.monotonic() < previous.expires:
                return previous.receipt, False
        if not allow_query:
            return None, False
        if previous is None and len(_CACHE) >= _CACHE_LIMIT:
            victim = next((k for k, v in _CACHE.items() if v.receipt is not None), None)
            if victim is None:
                return None, False
            del _CACHE[victim]
        pending = _CachedReceipt(failures=previous.failures if previous else 0)
        _CACHE[key] = pending
    try:
        receipt = collect_acceptance(snapshot.state[2], snapshot.published_pr,
                                     assignee=snapshot.state[3])
    except Exception:
        # Neither logs nor returned diagnostics may contain provider stderr.
        receipt = {"ok": False, "classification": "infra"}
    with _CACHE_LOCK:
        if receipt["classification"] in {"auth", "infra"}:
            pending.failures = min(pending.failures + 1, 5)
            delay = min(_POLL_TTL * 2 ** (pending.failures - 1), _MAX_BACKOFF)
        else:
            pending.failures = 0
            delay = _POLL_TTL
        pending.receipt = receipt
        pending.expires = time.monotonic() + delay
        _CACHE.move_to_end(key)
    return receipt, True


def snapshot_reviews(conn, db_path):
    # A caller can hand us a connection for a different board than its ambient
    # board selection. Its lock cannot protect this connection's mutations.
    try:
        if not any(row[1] == "main" and row[2] and Path(row[2]).resolve() == db_path
                   for row in conn.execute("PRAGMA database_list")):
            return []
    except (OSError, sqlite3.Error, ValueError):
        return []
    return [snapshot for row in conn.execute(
        "SELECT id FROM tasks WHERE status='review' AND completion_contract IS NOT NULL "
        "AND completion_contract != 'local-only' AND claim_lock IS NULL ORDER BY id"
    ) if (snapshot := _review_snapshot(conn, row["id"])) is not None]


def reconcile_reviews(conn, db_path, snapshots, result):
    # Missing and oldest receipts go first, even when a tick outlasts the TTL.
    # Otherwise a full poll budget could repeatedly starve the same tail.
    with _CACHE_LOCK:
        snapshots = sorted(snapshots, key=lambda snapshot: (
            _CACHE[_cache_key(snapshot, db_path)].expires
            if _cache_key(snapshot, db_path) in _CACHE else 0.0))
    polls = 0
    for snapshot in snapshots:
        receipt, queried = _cached_acceptance(snapshot, db_path, allow_query=polls < _MAX_POLLS_PER_TICK)
        polls += queried
        if receipt is None:
            continue
        classification = "open" if receipt.get("ok") and not receipt.get("merged") else receipt["classification"]
        if not receipt.get("ok") or not receipt.get("merged"):
            result.pr_reconciliation.append((snapshot.task_id, snapshot.published_pr, classification))
            continue
        with kbc._dispatch_tick_lock(db_path) as held:
            if not held or _review_snapshot(conn, snapshot.task_id) != snapshot:
                continue
            if kb._complete_task_with_acceptance(
                conn, snapshot.task_id,
                summary="Bound GitHub PR merged with required checks passing.",
                metadata={"published_pr": snapshot.published_pr, "source": "pr_reconciliation"},
                acceptance=_ReviewAcceptance(snapshot, receipt),
            ):
                result.completed_merged_prs.append(snapshot.task_id)

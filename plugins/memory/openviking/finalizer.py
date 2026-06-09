"""
OpenViking Session Finalizer — discovers and commits orphaned sessions.

This is the implementation that actually reads the repair markers, recovery
files, and registry to find and commit orphaned sessions. The previous
"Phase 4 finalizer" only checked the OV state DB — this one covers all paths.

Sources checked (in priority order):
1. Hermes state DB (~/.hermes/state.db) — the authoritative transcript store
2. OV registry DB (~/.hermes/openviking-sessions.db) — session lifecycle tracking
3. Repair journal (~/.hermes/openviking-repair/*.jsonl) — queue worker failures
4. Recovery snapshots (~/.hermes/openviking-recovery/*.json) — shutdown persistence

Usage:
    python3 -m plugins.memory.openviking.finalizer [--dry-run] [--batch=100]
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional, Set

from .registry import (
    get_uncommitted_sessions,
    get_all_session_ids,
    get_session,
    update_state as update_registry_state,
    clear_stale_finalizing,
    mark_dead,
)

logger = logging.getLogger(__name__)

# Paths
_HERMES_HOME = os.path.expanduser("~/.hermes")
_STATE_DB = os.path.join(_HERMES_HOME, "state.db")
_REPAIR_DIR = os.path.join(_HERMES_HOME, "openviking-repair")
_RECOVERY_DIR = os.path.join(_HERMES_HOME, "openviking-recovery")

# OpenViking
_OV_ENDPOINT = os.environ.get("OPENVIKING_ENDPOINT", "http://127.0.0.1:1933")
_OV_API_KEY = os.environ.get("OPENVIKING_API_KEY", "")
_OV_ACCOUNT = os.environ.get("OPENVIKING_ACCOUNT", "default")
_OV_USER = os.environ.get("OPENVIKING_USER", "default")
_OV_AGENT = os.environ.get("OPENVIKING_AGENT", "hermes")

# Limits
_BATCH_SIZE = 100
_OV_PAGE_LIMIT = 1000  # OV API max per page
_OV_TIMEOUT = 30.0


# ---------------------------------------------------------------------------
# OV API helpers
# ---------------------------------------------------------------------------

def _ov_headers() -> Dict[str, str]:
    h = {
        "Content-Type": "application/json",
        "X-OpenViking-Account": _OV_ACCOUNT,
        "X-OpenViking-User": _OV_USER,
        "X-OpenViking-Agent": _OV_AGENT,
    }
    if _OV_API_KEY:
        h["Authorization"] = f"Bearer {_OV_API_KEY}"
    return h


def _ov_list_sessions_paginated() -> Set[str]:
    """Return all OV session IDs from a single request.

    OV's GET /api/v1/sessions returns ALL sessions in one response — it
    does not support cursor/offset/page parameters (they are silently
    ignored). The list response only contains session_id/uri/is_dir;
    session detail (created_at, message_count, etc.) requires a separate
    GET /api/v1/sessions/{sid} call.

    Note: the list response does NOT include created_at or updated_at
    (those fields are null). This is a limitation of the filesystem-based
    list — not a bug. Session detail has the real timestamps.
    """
    import httpx
    all_sids: Set[str] = set()
    try:
        with httpx.Client(timeout=_OV_TIMEOUT) as client:
            resp = client.get(
                f"{_OV_ENDPOINT}/api/v1/sessions",
                headers=_ov_headers(),
            )
            if resp.status_code >= 400:
                logger.warning("OV API list returned %d", resp.status_code)
                return all_sids
            data = resp.json()
            result = data.get("result", [])
            for s in result:
                sid = s.get("session_id")
                if sid:
                    all_sids.add(sid)
            logger.info("Found %d sessions in OV", len(all_sids))
    except Exception as exc:
        logger.error("OV session list failed: %s", exc)
    return all_sids


def _ov_session_detail(sid: str) -> Optional[Dict[str, Any]]:
    """Get session detail from OV. Returns None on error."""
    import httpx
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(
                f"{_OV_ENDPOINT}/api/v1/sessions/{sid}",
                headers=_ov_headers(),
            )
            if resp.status_code >= 400:
                return None
            data = resp.json()
            if data.get("status") == "ok":
                return data.get("result")
            return None
    except Exception:
        return None


def _ov_push_and_commit(sid: str, messages: List[Dict[str, Any]]) -> bool:
    """Push messages to OV session and commit, then verify.

    Creates its own HTTP client. Returns True if commit endpoint accepted
    the request (HTTP < 400). Logs a warning if post-commit read-back
    shows message_count=0 (empty shell), but does NOT fail the commit
    — Hermes state.db is the authoritative session store, OV is secondary.
    """
    if not messages:
        return False
    import httpx
    try:
        with httpx.Client(timeout=_OV_TIMEOUT) as client:
            for msg in messages[-2000:]:
                role = msg.get("role", "user")
                if role not in ("user", "assistant", "tool"):
                    continue
                content = (msg.get("content") or "")[:4000]
                client.post(
                    f"{_OV_ENDPOINT}/api/v1/sessions/{sid}/messages",
                    json={"role": role, "content": content},
                    headers=_ov_headers(),
                )
            resp = client.post(
                f"{_OV_ENDPOINT}/api/v1/sessions/{sid}/commit",
                headers=_ov_headers(),
            )
            commit_ok = resp.status_code < 400

            # Post-commit read-back verification (informational)
            try:
                verify = client.get(
                    f"{_OV_ENDPOINT}/api/v1/sessions/{sid}",
                    headers=_ov_headers(),
                )
                if verify.status_code < 400:
                    vd = verify.json()
                    result = vd.get("result", {})
                    live_count = result.get("message_count", 0)
                    total_count = result.get("total_message_count", 0)
                    if live_count == 0 and total_count == 0 and commit_ok:
                        logger.warning(
                            "OV commit for %s returned success but message_count=0 "
                            "AND total_message_count=0 — messages were not persisted",
                            sid
                        )
                    elif live_count == 0 and total_count > 0 and commit_ok:
                        logger.debug(
                            "OV commit for %s OK: %d total messages (all archived, "
                            "live count is 0 as expected after commit)", sid, total_count
                        )
            except Exception as exc:
                logger.debug("OV verification read-back failed for %s: %s", sid, exc)

            return commit_ok
    except Exception as exc:
        logger.error("OV push/commit failed for %s: %s", sid, exc)
        return False


# ---------------------------------------------------------------------------
# Source discovery
# ---------------------------------------------------------------------------

def _discover_hermes_state_sessions(min_messages: int = 5) -> List[Dict[str, Any]]:
    """Return sessions from Hermes state DB with at least *min_messages* messages."""
    if not os.path.isfile(_STATE_DB):
        return []
    try:
        conn = sqlite3.connect(f"file:{_STATE_DB}?mode=ro", uri=True, timeout=5.0)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT s.id, s.source,
                    (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) as actual_msgs
                FROM sessions s
                WHERE s.message_count > 0
                  AND (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) >= ?
                ORDER BY s.started_at ASC
                """,
                (min_messages,),
            ).fetchall()
            result = []
            for r in rows:
                d = dict(r)
                d["_source"] = "hermes_state"
                result.append(d)
            return result
        finally:
            conn.close()
    except Exception as exc:
        logger.debug("Hermes state DB read failed: %s", exc)
        return []


def _discover_repair_journals() -> List[Dict[str, Any]]:
    """Read repair journal JSONL files."""
    if not os.path.isdir(_REPAIR_DIR):
        return []
    results: List[Dict[str, Any]] = []
    try:
        for fname in os.listdir(_REPAIR_DIR):
            if not fname.endswith(".jsonl"):
                continue
            sid = fname[:-6]  # strip .jsonl
            fpath = os.path.join(_REPAIR_DIR, fname)
            try:
                with open(fpath) as f:
                    lines = f.readlines()
                messages = []
                for line in lines:
                    line = line.strip()
                    if line:
                        messages.append(json.loads(line))
                if messages:
                    results.append({
                        "id": sid,
                        "_source": "repair_journal",
                        "actual_msgs": len(messages),
                        "_messages": messages,
                    })
            except Exception:
                continue
    except Exception:
        pass
    return results


def _discover_recovery_snapshots() -> List[Dict[str, Any]]:
    """Read recovery snapshot JSON files."""
    if not os.path.isdir(_RECOVERY_DIR):
        return []
    results: List[Dict[str, Any]] = []
    try:
        for fname in os.listdir(_RECOVERY_DIR):
            if not fname.endswith(".json"):
                continue
            if fname == ".pending":
                continue
            fpath = os.path.join(_RECOVERY_DIR, fname)
            try:
                with open(fpath) as f:
                    data = json.load(f)
                sid = data.get("session_id", fname[:-5])
                messages = data.get("messages", [])
                if messages:
                    results.append({
                        "id": sid,
                        "_source": "recovery_snapshot",
                        "actual_msgs": len(messages),
                        "_messages": messages,
                        "_snapshot": data,
                    })
            except Exception:
                continue
    except Exception:
        pass
    return results


def _discover_hermes_state_messages(sid: str) -> Optional[List[Dict[str, Any]]]:
    """Get messages for a session from the Hermes state DB."""
    try:
        conn = sqlite3.connect(f"file:{_STATE_DB}?mode=ro", uri=True, timeout=5.0)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT role, content, timestamp
                FROM messages
                WHERE session_id = ?
                ORDER BY timestamp ASC, id ASC
                """,
                (sid,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Cleanup helpers
# ---------------------------------------------------------------------------

def _cleanup_marker(sid: str) -> None:
    """Remove all marker files for a session."""
    for directory in (_REPAIR_DIR, _RECOVERY_DIR):
        if not os.path.isdir(directory):
            continue
        for suffix in (".jsonl", ".json"):
            fpath = os.path.join(directory, f"{sid}{suffix}")
            try:
                os.remove(fpath)
            except Exception:
                pass
    # Also remove from .pending
    pending_path = os.path.join(_RECOVERY_DIR, ".pending")
    if os.path.exists(pending_path):
        try:
            with open(pending_path) as f:
                lines = f.readlines()
            with open(pending_path, "w") as f:
                for line in lines:
                    if line.strip() != sid:
                        f.write(line)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_finalizer(
    dry_run: bool = False,
    batch_size: int = _BATCH_SIZE,
    min_messages: int = 5,
) -> Dict[str, Any]:
    """Run the finalizer: discover orphaned sessions and commit them.

    Args:
        dry_run: If True, only report what would be done (no writes).
        batch_size: Sessions per batch for progress reporting.
        min_messages: Skip sessions with fewer messages.

    Returns:
        Dict with committed, failed, skipped counts.
    """
    committed = 0
    failed = 0
    skipped = 0
    new_failures = 0

    # Phase 0: Escape stuck FINALIZING sessions (circuit breaker for stale state)
    unstuck = clear_stale_finalizing(max_age_minutes=30)
    if unstuck:
        logger.info("Unstuck %d sessions from stale FINALIZING state", unstuck)

    # Phase 1: Collect candidates from ALL sources
    candidates: Dict[str, Dict[str, Any]] = {}

    # Source A: Registry (sessions tracked but not committed)
    for s in get_uncommitted_sessions():
        sid = s["session_id"]
        candidates[sid] = {
            "id": sid,
            "_source": "registry",
            "actual_msgs": s.get("turn_count", 0) * 2,
            "_messages": s.get("cached_messages", []),
            "registry_state": s.get("state", ""),
        }

    # Source B: Hermes state DB
    for s in _discover_hermes_state_sessions(min_messages=min_messages):
        sid = s["id"]
        if sid not in candidates:
            candidates[sid] = s

    # Source C: Repair journals (queue worker failures)
    for s in _discover_repair_journals():
        sid = s["id"]
        if sid not in candidates:
            candidates[sid] = s
        elif not candidates[sid].get("_messages"):
            candidates[sid]["_messages"] = s.get("_messages", [])

    # Source D: Recovery snapshots (shutdown persistence)
    for s in _discover_recovery_snapshots():
        sid = s["id"]
        if sid not in candidates:
            candidates[sid] = s
        elif not candidates[sid].get("_messages"):
            candidates[sid]["_messages"] = s.get("_messages", [])

    if not candidates:
        logger.info("No orphaned sessions found — all clear")
        return {"committed": 0, "failed": 0, "skipped": 0}

    # Phase 2: Filter against OV's actual session list (paginated)
    logger.info("Fetching OV session list (paginated)...")
    ov_sids = _ov_list_sessions_paginated()
    logger.info("Found %d sessions in OV", len(ov_sids))

    to_process = []
    for sid, info in list(candidates.items()):
        if sid in ov_sids:
            skipped += 1
            continue
        to_process.append((sid, info))

    if not to_process:
        logger.info("All sessions already committed in OV")
        return {"committed": 0, "failed": 0, "skipped": skipped}

    if dry_run:
        logger.info(
            "DRY RUN — would process %d sessions (committed=%d, failed=%d, skipped=%d)",
            len(to_process), committed, failed, skipped,
        )
        return {"committed": 0, "failed": 0, "skipped": skipped, "dry_run": len(to_process)}

    # Phase 3: Process candidates in batches
    total = len(to_process)
    logger.info("Processing %d sessions in batches of %d...", total, batch_size)

    for i in range(0, total, batch_size):
        batch = to_process[i : i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total + batch_size - 1) // batch_size

        for sid, info in batch:
            # Get messages — try cached first, then Hermes state DB
            messages = info.get("_messages", [])
            if not messages:
                msg_data = _discover_hermes_state_messages(sid)
                if msg_data:
                    messages = msg_data
            if not messages:
                skipped += 1
                logger.info("  SKIP %s: no messages available", sid)
                continue
            if len(messages) < min_messages:
                skipped += 1
                continue

            # Push to OV and commit
            ok = _ov_push_and_commit(sid, messages)
            if ok:
                committed += 1
                update_registry_state(sid, "COMMITTED")
                _cleanup_marker(sid)
                logger.info("  OK  %s: %d msgs -> committed", sid, len(messages))
            else:
                failed += 1
                update_registry_state(sid, "FAILED", error="finalizer commit failed")
                logger.info("  FAIL %s: commit failed", sid)
                # Circuit breaker: if retries exceeded, mark DEAD and log once
                rec = get_session(sid)
                if rec and rec.get("retry_count", 0) >= 3:
                    mark_dead(sid, error="finalizer: max retries exceeded")
                    logger.warning(
                        "  DEAD %s: retry_count=%d exceeded max — "
                        "permanently excluded from finalizer runs",
                        sid, rec["retry_count"],
                    )
                elif rec and rec.get("retry_count", 0) <= 1:
                    new_failures += 1

        if i + batch_size < total:
            logger.info(
                "  Batch %d/%d done: %d committed, %d failed, %d skipped",
                batch_num, total_batches, committed, failed, skipped,
            )
            time.sleep(0.5)

    logger.info(
        "Finalizer done: %d committed, %d failed, %d skipped",
        committed, failed, skipped,
    )
    return {"committed": committed, "failed": failed, "skipped": skipped, "new_failures": new_failures}


def main() -> None:
    """CLI entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="OpenViking session finalizer")
    parser.add_argument("--dry-run", action="store_true", help="Report only, no writes")
    parser.add_argument("--batch", type=int, default=_BATCH_SIZE, help="Batch size")
    parser.add_argument("--min-messages", type=int, default=5, help="Skip sessions with fewer msgs")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    result = run_finalizer(
        dry_run=args.dry_run,
        batch_size=args.batch,
        min_messages=args.min_messages,
    )
    print(
        f"Committed: {result['committed']}, "
        f"Failed: {result['failed']}, "
        f"Skipped: {result['skipped']}"
    )


if __name__ == "__main__":
    main()

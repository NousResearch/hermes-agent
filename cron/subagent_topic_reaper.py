"""Subagent forum-topic reaper — delete idle topics older than TTL.

Phase B1 of the subagent → Telegram-topic pipeline.

Background
----------
:mod:`gateway.subagent_topic_router` lazily creates one Telegram forum topic
per parent session and forwards every ``subagent.*`` progress event into it.
Each successful forward bumps ``last_message_ts`` in
``~/.hermes/state/subagent_topics.json``.

Without periodic cleanup the forum accumulates stale topics for sessions that
finished hours or days ago.  This module is the cleanup half of the contract:
a small, idempotent sweep that

1. loads the router's state file,
2. finds every mapping whose ``last_message_ts`` is older than ``ttl_seconds``
   (default 24h),
3. asks Telegram to delete each expired topic, and
4. removes the deleted (or already-gone) entries from the state file.

It is intentionally **side-effect minimal**:

* Failed deletions stay in state and will be retried on the next tick.
* Topics Telegram reports as ``topic_not_found`` / ``chat_not_found`` are
  treated as already-gone and pruned from state.
* All exceptions are caught — the module **never** lets an error escape to
  its cron caller, because a crashing cron job would loop indefinitely.

How it runs
-----------
Designed to be invoked every ~15 minutes by a cron job (wired up in Phase
C2 — not this file).  Run manually with::

    python -m cron.subagent_topic_reaper --dry-run --json
    python -m cron.subagent_topic_reaper --ttl-seconds 86400

The script terminates after one sweep; there is no daemon loop.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home


__all__ = ["reap_expired_topics"]


_DEFAULT_TTL_SECONDS = 24 * 60 * 60  # 24h

# Error codes from tools.telegram_topic_tool._classify_error that we treat as
# "topic is already gone, prune from state and move on".
_GONE_CODES = frozenset({"topic_not_found", "chat_not_found"})


def _default_state_path() -> Path:
    """Return the canonical path to the router's JSON state file."""
    return get_hermes_home() / "state" / "subagent_topics.json"


def _empty_summary(*, dry_run: bool) -> Dict[str, Any]:
    """Build the zeroed summary dict used for early-exit paths."""
    return {
        "scanned": 0,
        "expired": 0,
        "deleted": 0,
        "skipped_gone": 0,
        "failed": 0,
        "dry_run": dry_run,
        "details": [],
    }


def reap_expired_topics(
    *,
    now: Optional[float] = None,
    ttl_seconds: int = _DEFAULT_TTL_SECONDS,
    state_path: Optional[Path] = None,
    dry_run: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Sweep the subagent-topic state file and delete topics idle past TTL.

    Parameters
    ----------
    now:
        Reference timestamp (seconds since epoch).  Defaults to ``time.time()``.
        Mostly useful for tests.
    ttl_seconds:
        Topics whose ``last_message_ts`` is older than this are considered
        expired.  Defaults to 24h.
    state_path:
        Override the state file location.  Defaults to
        ``~/.hermes/state/subagent_topics.json``.
    dry_run:
        When ``True``, do not call Telegram and do not modify the state file.
        Returned summary still reports what *would* have happened.
    logger:
        Logger to emit progress on.  Defaults to ``logging.getLogger`` for
        this module.

    Returns
    -------
    dict
        Summary of the sweep — see module docstring / task spec for the
        exact schema.  On hard-failure paths (no token, broken state file)
        the dict is still returned with ``"error"`` set instead of raising.
    """
    log = logger or logging.getLogger(__name__)
    ref_now = time.time() if now is None else float(now)
    path = state_path if state_path is not None else _default_state_path()
    summary: Dict[str, Any] = _empty_summary(dry_run=dry_run)

    # ------------------------------------------------------------------
    # Load state — defer the router import until call time so that just
    # importing this module never drags in the gateway / telegram stack.
    # ------------------------------------------------------------------
    try:
        from gateway.subagent_topic_router import _load_state, _save_state
    except Exception as exc:  # pragma: no cover — defensive only
        log.error("subagent_topic_reaper: cannot import router state helpers: %s", exc)
        summary["error"] = "import_failed"
        return summary

    try:
        state = _load_state(path)
    except Exception as exc:  # _load_state already swallows most errors
        log.error("subagent_topic_reaper: state load raised unexpectedly: %s", exc)
        summary["error"] = "state_load_failed"
        return summary

    topics: Dict[str, Any] = state.get("topics") or {}
    summary["scanned"] = len(topics)

    # ------------------------------------------------------------------
    # Find expired entries.
    # ------------------------------------------------------------------
    expired: List[Dict[str, Any]] = []
    for session_id, entry in topics.items():
        if not isinstance(entry, dict):
            continue
        last_ts_raw = entry.get("last_message_ts")
        try:
            last_ts = float(last_ts_raw)
        except (TypeError, ValueError):
            # Malformed entry — treat as immediately expired so the next
            # successful sweep prunes the garbage.
            last_ts = 0.0
        age = ref_now - last_ts
        if age > ttl_seconds:
            expired.append(
                {
                    "session_id": session_id,
                    "chat_id": str(entry.get("chat_id", "")),
                    "thread_id": str(entry.get("thread_id", "")),
                    "topic_name": str(entry.get("topic_name", "")),
                    "last_message_ts": last_ts,
                    "age_seconds": age,
                }
            )
    summary["expired"] = len(expired)

    log.info(
        "subagent_topic_reaper.start total=%d expired=%d ttl=%ds dry_run=%s",
        summary["scanned"],
        summary["expired"],
        ttl_seconds,
        dry_run,
    )

    if not expired:
        log.info(
            "subagent_topic_reaper.done scanned=%d expired=0 deleted=0 "
            "skipped_gone=0 failed=0 dry_run=%s",
            summary["scanned"],
            dry_run,
        )
        return summary

    # ------------------------------------------------------------------
    # Dry-run: report and bail without touching Telegram or state.
    # ------------------------------------------------------------------
    if dry_run:
        for item in expired:
            detail = dict(item)
            detail["status"] = "dry_run"
            detail["error"] = None
            summary["details"].append(detail)
        log.info(
            "subagent_topic_reaper.done scanned=%d expired=%d dry_run=True "
            "(no deletions performed)",
            summary["scanned"],
            summary["expired"],
        )
        return summary

    # ------------------------------------------------------------------
    # Load telegram token + delete coro factory (lazy import).
    # ------------------------------------------------------------------
    try:
        from tools.telegram_topic_tool import (
            _classify_error,
            _load_telegram_token,
            _run_topic_op,
        )
    except Exception as exc:  # pragma: no cover
        log.error("subagent_topic_reaper: cannot import telegram topic tool: %s", exc)
        summary["error"] = "import_failed"
        return summary

    token, token_err = _load_telegram_token()
    if not token:
        log.error(
            "subagent_topic_reaper: no Telegram token available (%s); skipping sweep",
            token_err or "unknown",
        )
        summary["error"] = "no_token"
        return summary

    # ------------------------------------------------------------------
    # Run all deletions on a single event loop (sequential, not gathered,
    # to respect Telegram rate limits).
    # ------------------------------------------------------------------
    async def _delete_one(item: Dict[str, Any]) -> Dict[str, Any]:
        detail: Dict[str, Any] = dict(item)
        chat_id = item["chat_id"]
        thread_id = item["thread_id"]
        if not chat_id or not thread_id:
            detail["status"] = "failed"
            detail["error"] = "missing chat_id or thread_id in state entry"
            return detail
        try:
            await _run_topic_op(
                token,
                "delete",
                chat_id=chat_id,
                thread_id=thread_id,
                name=None,
            )
            detail["status"] = "deleted"
            detail["error"] = None
            return detail
        except Exception as exc:  # noqa: BLE001 — classifier handles it
            try:
                code = _classify_error(exc)
            except Exception:  # pragma: no cover — classifier itself blew up
                code = "unknown"
            if code in _GONE_CODES:
                detail["status"] = "skipped_gone"
                detail["error"] = code
            else:
                detail["status"] = "failed"
                detail["error"] = f"{code}: {exc}"
            return detail

    async def _delete_all() -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for item in expired:
            result = await _delete_one(item)
            out.append(result)
        return out

    try:
        details = asyncio.run(_delete_all())
    except Exception as exc:  # absolute last-resort guard
        log.exception("subagent_topic_reaper: deletion loop crashed: %s", exc)
        summary["error"] = "delete_loop_crashed"
        return summary

    # ------------------------------------------------------------------
    # Tally results, log, and prune state of deleted/already-gone entries.
    # ------------------------------------------------------------------
    deletable_session_ids: List[str] = []
    for detail in details:
        summary["details"].append(detail)
        status = detail.get("status")
        sid = detail.get("session_id", "")
        topic_name = detail.get("topic_name", "")
        if status == "deleted":
            summary["deleted"] += 1
            deletable_session_ids.append(sid)
            log.info(
                "subagent_topic_reaper.deleted session=%s thread=%s name=%r age=%.0fs",
                sid,
                detail.get("thread_id"),
                topic_name,
                detail.get("age_seconds", 0.0),
            )
        elif status == "skipped_gone":
            summary["skipped_gone"] += 1
            deletable_session_ids.append(sid)
            log.info(
                "subagent_topic_reaper.skipped_gone session=%s thread=%s name=%r (%s)",
                sid,
                detail.get("thread_id"),
                topic_name,
                detail.get("error"),
            )
        else:
            summary["failed"] += 1
            log.warning(
                "subagent_topic_reaper.failed session=%s thread=%s name=%r error=%s",
                sid,
                detail.get("thread_id"),
                topic_name,
                detail.get("error"),
            )

    if deletable_session_ids:
        # Re-load state in case the router wrote new entries while we were
        # talking to Telegram, then drop only the sessions we successfully
        # cleared.  Failed entries are intentionally left for the next run.
        try:
            current = _load_state(path)
            current_topics = current.get("topics") or {}
            removed = 0
            for sid in deletable_session_ids:
                if sid in current_topics:
                    current_topics.pop(sid, None)
                    removed += 1
            current["topics"] = current_topics
            _save_state(path, current)
            log.info(
                "subagent_topic_reaper.state_pruned removed=%d remaining=%d",
                removed,
                len(current_topics),
            )
        except Exception as exc:  # noqa: BLE001
            log.error(
                "subagent_topic_reaper: state save failed (deletions stay in state, "
                "will retry next sweep): %s",
                exc,
            )
            summary["error"] = "state_save_failed"

    log.info(
        "subagent_topic_reaper.done scanned=%d expired=%d deleted=%d "
        "skipped_gone=%d failed=%d dry_run=%s",
        summary["scanned"],
        summary["expired"],
        summary["deleted"],
        summary["skipped_gone"],
        summary["failed"],
        dry_run,
    )
    return summary


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def _main() -> int:
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(
        description="Reap expired Telegram subagent progress topics.",
    )
    parser.add_argument(
        "--ttl-seconds",
        type=int,
        default=_DEFAULT_TTL_SECONDS,
        help="TTL in seconds (default 86400 = 24h).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be deleted without calling Telegram.",
    )
    parser.add_argument(
        "--json",
        dest="json_out",
        action="store_true",
        help="Emit the summary dict as JSON on stdout.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    try:
        summary = reap_expired_topics(
            ttl_seconds=args.ttl_seconds,
            dry_run=args.dry_run,
        )
    except Exception as exc:  # final safety net for cron loop
        logging.getLogger(__name__).exception(
            "subagent_topic_reaper: top-level crash: %s", exc
        )
        if args.json_out:
            json.dump(
                {"error": "top_level_crash", "message": str(exc)},
                sys.stdout,
                indent=2,
                default=str,
            )
            sys.stdout.write("\n")
        return 1

    if args.json_out:
        json.dump(summary, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
    else:
        print(
            f"Scanned: {summary['scanned']}  Expired: {summary['expired']}  "
            f"Deleted: {summary['deleted']}  Skipped: {summary['skipped_gone']}  "
            f"Failed: {summary['failed']}  DryRun: {summary['dry_run']}"
        )
        if summary.get("error"):
            print(f"Error: {summary['error']}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

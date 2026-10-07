"""Append-only lifecycle journal for the cron job store + its vanished-job guard.

Why this exists
---------------
``cron/jobs.json`` is a whole-file store. Every writer loads it, mutates an
in-memory copy and writes the whole thing back, so a lost job leaves **no
trace at all**: the record is simply absent from the next write, and the
store's own history is one file with no prior versions. On 2026-09-20 three
jobs vanished (or lost their run state) and the only way to tell "somebody
removed it" from "the store ate it" was to reconstruct intent from chat
scrollback.

``cron/jobs.py``'s by-id merge closes the race that caused those losses. This
module closes the *observability* half: an append-only journal of every
intentional create and remove, plus a guard that answers the one question the
store cannot answer by itself —

    a job is gone; did anyone actually ask for that?

The journal is written at the store's own choke points (``create_job`` and the
``removed_ids`` argument that every intentional deletion already threads
through ``save_jobs``), so it cannot drift from the store the way a log
scraper would: a removal that does not pass ``removed_ids`` is refused by the
shrink-merge guard, and therefore cannot happen silently.

Failure policy: journal writes are best-effort and never propagate. Losing an
audit record must not be able to break a cron write.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_time import now as _hermes_now

logger = logging.getLogger(__name__)

JOURNAL_FILENAME = "lifecycle.jsonl"

# How far back the guard looks. Matches the card's 24h window: a job created
# and removed longer ago than this is not evidence of anything current, and
# keeping the scan bounded keeps the journal from having to be read whole.
DEFAULT_WINDOW_HOURS = 24.0

# Entries older than this are pruned on write so the file cannot grow without
# bound. Deliberately several times the guard window: an entry must stay
# readable for the whole window even if nothing writes for days afterwards.
_RETENTION_DAYS = 14.0

EVENT_CREATED = "created"
EVENT_REMOVED = "removed"


def _journal_path() -> Path:
    from cron.jobs import _current_cron_store

    return _current_cron_store().cron_dir / JOURNAL_FILENAME


def _append(record: Dict[str, Any]) -> None:
    """Append one record. Best effort — never raises into a cron write."""
    try:
        path = _journal_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        logger.debug("cron lifecycle journal append failed", exc_info=True)


def record_created(job_id: str, *, name: Optional[str] = None,
                   actor: Optional[str] = None) -> None:
    """Journal an intentional job creation."""
    if not job_id:
        return
    _append({
        "event": EVENT_CREATED,
        "job_id": str(job_id),
        "name": name,
        "actor": actor or _default_actor(),
        "at": _hermes_now().isoformat(),
    })


def record_removed(job_id: str, *, reason: Optional[str] = None,
                   actor: Optional[str] = None) -> None:
    """Journal an intentional job removal."""
    if not job_id:
        return
    _append({
        "event": EVENT_REMOVED,
        "job_id": str(job_id),
        "reason": reason,
        "actor": actor or _default_actor(),
        "at": _hermes_now().isoformat(),
    })


def _default_actor() -> str:
    """Best-effort writer identity: which process asked for this."""
    return f"{os.path.basename(os.environ.get('HERMES_ACTOR', '') or 'hermes')}:{os.getpid()}"


def read_entries(*, window_hours: float = DEFAULT_WINDOW_HOURS) -> List[Dict[str, Any]]:
    """Return journal entries inside the window, oldest first.

    Malformed lines are skipped rather than failing the read: the journal is
    append-only from multiple processes, so a torn final line is expected
    after a crash and must not blind the guard to everything before it.
    """
    path = _journal_path()
    if not path.exists():
        return []
    cutoff = _hermes_now() - timedelta(hours=window_hours)
    entries: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    at = _parse_at(rec.get("at"))
                except Exception:
                    continue
                if at is None or at < cutoff:
                    continue
                entries.append(rec)
    except OSError:
        logger.debug("cron lifecycle journal unreadable", exc_info=True)
        raise
    return entries


def _parse_at(raw: Any):
    from datetime import datetime, timezone

    if not isinstance(raw, str):
        return None
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def prune() -> int:
    """Drop entries past retention. Returns the number of entries removed.

    Rewrites via a temp file + atomic replace so a concurrent reader never
    sees a half-written journal. Best effort: a failure leaves the journal
    intact and oversized, which is strictly better than losing audit records.
    """
    path = _journal_path()
    if not path.exists():
        return 0
    cutoff = _hermes_now() - timedelta(days=_RETENTION_DAYS)
    kept: List[str] = []
    dropped = 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    at = _parse_at(json.loads(stripped).get("at"))
                except Exception:
                    kept.append(stripped)  # unparseable: keep, never guess away
                    continue
                if at is not None and at < cutoff:
                    dropped += 1
                else:
                    kept.append(stripped)
        if not dropped:
            return 0
        from utils import atomic_replace

        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".lifecycle_",
                                   suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for line in kept:
                f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
        atomic_replace(tmp, path)
        return dropped
    except Exception:
        logger.debug("cron lifecycle journal prune failed", exc_info=True)
        return 0


# --------------------------------------------------------------------------
# The guard
# --------------------------------------------------------------------------

STATUS_OK = "ok"
STATUS_VANISHED = "vanished"
STATUS_UNAVAILABLE = "unavailable"


@dataclass
class VanishedJobReport:
    """Result of one expected-vs-present reconciliation.

    ``status`` is one of:

    * ``ok`` — every job created inside the window is either still present in
      ``jobs.json`` or has a matching removal record. Nothing to page.
    * ``vanished`` — at least one job was created, never removed, and is
      absent from the store. This is the incident shape; page #alerts.
    * ``unavailable`` — the journal or the store could not be read, so the
      reconciliation could not be performed. Explicitly NOT ``ok``: a guard
      that cannot see must never report green (it would have certified the
      very incident it exists to catch).
    """

    status: str
    vanished: List[Dict[str, Any]] = field(default_factory=list)
    created_count: int = 0
    removed_count: int = 0
    present_count: int = 0
    detail: Optional[str] = None

    @property
    def should_alert(self) -> bool:
        return self.status != STATUS_OK

    def summary(self) -> str:
        if self.status == STATUS_UNAVAILABLE:
            return f"cron vanished-job guard UNAVAILABLE: {self.detail}"
        if self.status == STATUS_OK:
            return (
                f"cron vanished-job guard ok: {self.created_count} created / "
                f"{self.removed_count} removed in window, "
                f"{self.present_count} present, 0 unaccounted"
            )
        names = ", ".join(
            f"{v['job_id']}({v.get('name') or '?'})" for v in self.vanished
        )
        return (
            f"cron jobs VANISHED with no matching removal: {names} — "
            f"created in the last window and absent from jobs.json"
        )


def check_vanished_jobs(
    *,
    window_hours: float = DEFAULT_WINDOW_HOURS,
    jobs: Optional[List[Dict[str, Any]]] = None,
) -> VanishedJobReport:
    """Reconcile journalled creates against what jobs.json actually holds.

    A job counts as vanished when it was created inside the window, has no
    removal record at or after that creation, and is not in the store. The
    "at or after" ordering matters: a job can legitimately be created,
    removed, and re-created under a fresh id — comparing against an older
    removal of the *same* id (a re-armed one-shot reusing its id) would
    wrongly excuse a real loss.

    Pass *jobs* to reconcile against an already-loaded snapshot; otherwise
    the live store is read.
    """
    try:
        entries = read_entries(window_hours=window_hours)
    except Exception as e:
        return VanishedJobReport(status=STATUS_UNAVAILABLE,
                                 detail=f"journal unreadable: {e}")

    if jobs is None:
        try:
            from cron.jobs import load_jobs

            jobs = load_jobs()
        except Exception as e:
            return VanishedJobReport(status=STATUS_UNAVAILABLE,
                                     detail=f"jobs.json unreadable: {e}")

    present = {
        str(j["id"]) for j in jobs
        if isinstance(j, dict) and j.get("id")
    }

    # Latest create, and every removal, per id.
    created: Dict[str, Dict[str, Any]] = {}
    removed: Dict[str, List[Any]] = {}
    for rec in entries:
        jid = rec.get("job_id")
        at = _parse_at(rec.get("at"))
        if not jid or at is None:
            continue
        jid = str(jid)
        if rec.get("event") == EVENT_CREATED:
            prior = created.get(jid)
            prior_at = _parse_at(prior["at"]) if prior else None
            if prior_at is None or at >= prior_at:
                created[jid] = rec
        elif rec.get("event") == EVENT_REMOVED:
            removed.setdefault(jid, []).append(at)

    vanished: List[Dict[str, Any]] = []
    for jid, rec in created.items():
        if jid in present:
            continue
        created_at = _parse_at(rec.get("at"))
        if created_at is None:
            continue
        if any(r >= created_at for r in removed.get(jid, [])):
            continue  # removed on purpose after this create
        vanished.append({
            "job_id": jid,
            "name": rec.get("name"),
            "created_at": rec.get("at"),
            "actor": rec.get("actor"),
        })

    vanished.sort(key=lambda v: v["created_at"])
    # Opportunistic retention: the guard is the one caller that has already
    # paid for a full journal read, so pruning here costs a rewrite only when
    # something is actually past retention, and never sits on a write path.
    prune()
    return VanishedJobReport(
        status=STATUS_VANISHED if vanished else STATUS_OK,
        vanished=vanished,
        created_count=len(created),
        removed_count=sum(len(v) for v in removed.values()),
        present_count=len(present),
    )


__all__ = [
    "DEFAULT_WINDOW_HOURS",
    "EVENT_CREATED",
    "EVENT_REMOVED",
    "JOURNAL_FILENAME",
    "STATUS_OK",
    "STATUS_UNAVAILABLE",
    "STATUS_VANISHED",
    "VanishedJobReport",
    "check_vanished_jobs",
    "prune",
    "read_entries",
    "record_created",
    "record_removed",
]

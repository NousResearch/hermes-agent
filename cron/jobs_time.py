"""Per-job ``timezone`` knob (sibling of ``cron/jobs.py``).

A job may pin its own IANA ``timezone``: its cron expression is then that zone's WALL CLOCK,
independent of the profile's configured zone. Absent/None means "inherit the profile", so a
store written before this field existed keeps byte-identical behaviour.

Behaviour lives here, not in the facade: ``cron/jobs.py`` is already past the file-lines
ratchet. The facade keeps the call sites (``compute_next_run``, the due scan, and every write
site that advances ``next_run_at``), which read the pins through this module.

Why the due scan needs the pin too: ``_ensure_aware`` normalizes a stored instant into a zone,
and the cron repair predicates then compare WALL CLOCKS. Normalizing a New-York job's instant
into a Los-Angeles profile zone makes the wall clock look like it is still in the future, so
``_repair_timezone_shifted_cron`` re-anchors to the NEXT occurrence and the due one is lost —
the "job timezone east of the profile timezone silently never fires" failure. See
``cron.jobs._evaluate_due_job``.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

logger = logging.getLogger(__name__)

TIMEZONE_KEY = "timezone"

# A corrupt persisted value would otherwise log on every 60s scan. Warn once per distinct value.
_warned: set[str] = set()


def reset_warning_state() -> None:
    """Forget which corrupt values were already warned about (tests, config reloads)."""
    _warned.clear()


def _warn_once(kind: str, value: Any, detail: str) -> None:
    key = f"{kind}:{value!r}"
    if key in _warned:
        return
    _warned.add(key)
    logger.warning(
        "Job %s %r is invalid (%s); falling back to the profile setting. Fix or clear it with "
        "`hermes cron edit`.", kind, value, detail)


def normalize_timezone(value: Any) -> Optional[str]:
    """Validate a job's ``timezone`` and return the canonical IANA key.

    ``None``/``""`` mean "inherit the profile timezone", stored as an ABSENT key. Anything that
    is not a resolvable IANA zone raises ValueError BEFORE the record is written, so a typo can
    never reach the tick loop.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(
            "timezone must be an IANA name such as 'Asia/Hong_Kong', "
            f"not {type(value).__name__}.")
    name = value.strip()
    if not name:
        return None
    try:
        return ZoneInfo(name).key
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(
            f"Unknown timezone {name!r}. Use an IANA key from "
            f"`zoneinfo.available_timezones()`, e.g. 'Asia/Hong_Kong'.") from exc


def zone_for_job(job_or_timezone: Any) -> Optional[ZoneInfo]:
    """The zone a job's schedule is expressed in, or None to inherit the profile's.

    Accepts a job record or a bare ``timezone`` value. Never raises: a corrupt stored value
    degrades to the profile zone (warned once per value) instead of breaking the tick it is
    read in — the 60s scan must not die on one bad record.
    """
    if isinstance(job_or_timezone, dict):
        value = job_or_timezone.get(TIMEZONE_KEY)
    else:
        value = job_or_timezone
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return ZoneInfo(value.strip())
    except (ZoneInfoNotFoundError, ValueError) as exc:
        _warn_once(TIMEZONE_KEY, value, str(exc))
        return None


def prune_inherited_timezone(record: dict[str, Any]) -> None:
    """``timezone: ''`` / null means "inherit the profile", so the key is DROPPED, not stored as
    null — an absent optional key is this repo's spelling for "unset" (see ``cron/jobs.py``'s
    optional-field write loop) and a stored null would render as a pinned-but-blank zone."""
    if TIMEZONE_KEY in record and not record.get(TIMEZONE_KEY):
        record.pop(TIMEZONE_KEY, None)


def oneshot_instant(value: Any, zone: Optional[ZoneInfo] = None) -> Optional[datetime]:
    """Parse a one-shot ``run_at`` into an aware instant, or None when unparseable.

    An offset-bearing value is already an instant. An offset-LESS value is the job's own wall
    clock, so a pinned zone interprets it in that zone (``2026-06-01T09:00:00`` means 09:00
    for THIS job) instead of in the host's local time.
    """
    from cron.jobs import _ensure_aware  # late import: the facade owns naive normalization

    try:
        dt = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None and zone is not None:
        return dt.replace(tzinfo=zone)
    return _ensure_aware(dt, zone)

"""Per-job ``timezone`` and ``grace_minutes`` knobs (sibling of ``cron/jobs.py``).

A job may pin its own IANA ``timezone``: its cron expression is then that zone's WALL CLOCK,
independent of the profile's configured zone. Absent/None means "inherit the profile", so a
store written before this field existed keeps byte-identical behaviour. ``grace_minutes``
overrides the computed half-period catch-up window for one job (``0`` = never catch up).

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
GRACE_KEY = "grace_minutes"

# ``0`` is a real value ("skip a missed occurrence instead of running it late"); a whole day is
# the most a late run can usefully be.
MIN_GRACE_MINUTES = 0
MAX_GRACE_MINUTES = 24 * 60

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


def normalize_grace_minutes(value: Any) -> Optional[int]:
    """Validate a job's ``grace_minutes`` override; None means "use the computed window".

    ``0`` is a real value (never catch up), so it must never be collapsed into absent:
    ``0 == False`` in Python, which is exactly the trap the ``workdir`` update lambda
    (``v in {None, "", False}``) would spring. Bools are therefore rejected explicitly.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(
            "grace_minutes must be a whole number of minutes (0 = never catch up), "
            "not a boolean.")
    if isinstance(value, str):
        if not value.strip():
            return None
        if not value.strip().lstrip("+-").isdigit():
            raise ValueError(
                f"grace_minutes must be a whole number of minutes, not {value!r}.")
    try:
        minutes = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"grace_minutes must be a whole number of minutes, not {value!r}.") from exc
    if isinstance(value, float) and value != minutes:
        raise ValueError(
            f"grace_minutes must be a whole number of minutes, not {value!r}.")
    if not MIN_GRACE_MINUTES <= minutes <= MAX_GRACE_MINUTES:
        raise ValueError(
            f"grace_minutes must be between {MIN_GRACE_MINUTES} and "
            f"{MAX_GRACE_MINUTES} (0 = never catch up), got {minutes}.")
    return minutes


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


def grace_seconds_override(grace_minutes: Any) -> Optional[int]:
    """Catch-up window in seconds from an explicit override, or None to compute it from cadence.

    Tolerant read (the record may have been hand-edited): a bad value is ignored rather than
    crashing the scan.
    """
    try:
        minutes = normalize_grace_minutes(grace_minutes)
    except ValueError as exc:
        _warn_once(GRACE_KEY, grace_minutes, str(exc))
        return None
    return None if minutes is None else minutes * 60


def prune_inherited_knobs(record: dict[str, Any]) -> None:
    """An explicitly cleared knob is stored by DROPPING the key, not as null: an absent optional
    key is this repo's spelling for "unset" (see ``cron/jobs.py``'s optional-field write loop),
    and a stored null would render as a pinned-but-blank zone.

    ``grace_minutes`` is tested with ``is None`` — ``0`` is the real "never catch up" value and
    must survive, which is exactly the ``0 == False`` trap an ``in {None, "", False}`` test
    would spring.
    """
    if TIMEZONE_KEY in record and not record.get(TIMEZONE_KEY):
        record.pop(TIMEZONE_KEY, None)
    if GRACE_KEY in record and record.get(GRACE_KEY) is None:
        record.pop(GRACE_KEY, None)


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

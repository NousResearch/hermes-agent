"""Per-action autonomy levels for guarded write tools.

A guarded profile (the Product Education teammate) reads, for each write
action, an **autonomy level** from its own ``config.yaml``::

    autonomy:
      paused: false            # kill switch: every action drops to `signoff`
      announce_channel: C0123  # where `veto` announces and `auto` reports
      veto_wait_working_hours: 4
      working_hours: {timezone: Asia/Jerusalem, start: 9, end: 18,
                      days: [sun, mon, tue, wed, thu]}
      actions:
        zendesk_create_draft: signoff
        zendesk_publish: off

Levels: ``off`` (refuse), ``signoff`` (a human approval signal is required
first), ``veto`` (announce, wait, act unless vetoed), ``auto`` (act, then
report). Graduating an action is a one-line config edit, picked up on the next
call because ``read_raw_config`` is mtime-cached — no code change, no deploy.

**Levels only unlock capabilities that exist.** ``BUILT_LEVELS`` lists, per
action, the levels whose code path is implemented; a configured level outside
that set raises ``AutonomyConfigError`` (the tool refuses with the message)
rather than being clamped, so a mis-graduation is loud. Actions not listed
here get ``off`` / ``signoff`` only.

Design: docs/superpowers/specs/2026-11-16-product-education-teammate-design.md
("Autonomy levels: how it graduates").
"""
from __future__ import annotations

import json
import os
import re
import secrets
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

LEVELS = ("off", "signoff", "veto", "auto")
DEFAULT_LEVEL = "signoff"
_BASE_LEVELS = frozenset({"off", "signoff"})

BUILT_LEVELS: dict[str, frozenset[str]] = {
    "zendesk_create_draft": frozenset(LEVELS),
    # Not built in v1: publishing is a human click in Zendesk. Built when
    # someone first decides to graduate it — its own small change.
    "zendesk_publish": frozenset({"off"}),
}


class AutonomyConfigError(ValueError):
    """The profile's ``autonomy`` block names a level that cannot be honoured."""


def _built(action: str) -> frozenset[str]:
    return BUILT_LEVELS.get(action, _BASE_LEVELS)


def _block(config: dict) -> dict:
    block = config.get("autonomy") if isinstance(config, dict) else None
    return block if isinstance(block, dict) else {}


def _load_config(config: Optional[dict]) -> dict:
    if config is not None:
        return config
    from hermes_cli.config import read_raw_config
    return read_raw_config()


def resolve_level(action: str, config: Optional[dict] = None) -> str:
    """Return the effective level for ``action`` in the active profile.

    ``config`` defaults to the active profile's raw ``config.yaml``. Raises
    ``AutonomyConfigError`` for a value that isn't a level, or a level this
    action has no code path for. ``paused: true`` caps every level at
    ``signoff`` (``off`` stays ``off``).
    """
    block = _block(_load_config(config))
    built = _built(action)
    actions = block.get("actions") if isinstance(block.get("actions"), dict) else {}
    raw = actions.get(action)
    if raw is None:
        level = DEFAULT_LEVEL if DEFAULT_LEVEL in built else "off"
    else:
        level = str(raw).strip().lower()
        if level not in built:
            what = "is not an autonomy level" if level not in LEVELS else "is not built for this action"
            raise AutonomyConfigError(
                f"autonomy.actions.{action}: {raw!r} {what}. Built levels for "
                f"{action}: {', '.join(lv for lv in LEVELS if lv in built)}. "
                f"An admin must fix the profile's config.yaml."
            )
    if block.get("paused") is True and level in ("veto", "auto"):
        return "signoff"
    return level


# ── working hours (the veto window) ──────────────────────────────────────

_DAY_NAMES = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}


@dataclass(frozen=True)
class WorkingHours:
    timezone: str
    start: int  # hour of day, inclusive
    end: int    # hour of day, exclusive
    days: frozenset[int]  # datetime.weekday() values


DEFAULT_WORKING_HOURS = WorkingHours("UTC", 9, 18, frozenset({0, 1, 2, 3, 4}))


def working_hours_from_config(config: Optional[dict] = None) -> WorkingHours:
    raw = _block(_load_config(config)).get("working_hours")
    if not isinstance(raw, dict):
        return DEFAULT_WORKING_HOURS
    d = DEFAULT_WORKING_HOURS
    tz = str(raw.get("timezone") or d.timezone)
    try:
        ZoneInfo(tz)
    except Exception:
        raise AutonomyConfigError(f"autonomy.working_hours.timezone: unknown timezone {tz!r}")
    start, end = int(raw.get("start", d.start)), int(raw.get("end", d.end))
    if not 0 <= start < end <= 24:
        raise AutonomyConfigError("autonomy.working_hours: need 0 <= start < end <= 24")
    days = d.days
    if raw.get("days") is not None:
        try:
            days = frozenset(_DAY_NAMES[str(x).strip().lower()[:3]] for x in raw["days"])
        except (KeyError, TypeError):
            raise AutonomyConfigError(
                "autonomy.working_hours.days: use day names like [mon, tue, wed, thu, fri]")
        if not days:
            raise AutonomyConfigError("autonomy.working_hours.days must not be empty")
    return WorkingHours(tz, start, end, days)


def add_working_hours(start: datetime, hours: float, wh: WorkingHours) -> datetime:
    """Return the instant ``hours`` working hours after ``start`` (aware).

    Time outside the configured days/hours does not count, so a veto window
    opened on Friday afternoon does not elapse over the weekend.
    """
    tz = ZoneInfo(wh.timezone)
    cur = start.astimezone(tz)
    remaining = timedelta(hours=hours)
    for _ in range(3660):  # ~10 years of days; guards a pathological config
        day_start = cur.replace(hour=wh.start, minute=0, second=0, microsecond=0)
        day_end = (cur.replace(hour=0, minute=0, second=0, microsecond=0)
                   + timedelta(hours=wh.end))
        if cur.weekday() in wh.days and cur < day_end:
            cur = max(cur, day_start)
            available = day_end - cur
            if remaining <= available:
                return (cur + remaining).astimezone(start.tzinfo)
            remaining -= available
        nxt = (cur + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        cur = nxt
    raise AutonomyConfigError("autonomy.working_hours: veto window never closes")


def veto_settings(config: Optional[dict] = None) -> tuple[str, float, WorkingHours]:
    """Return ``(announce_channel, wait_working_hours, working_hours)``."""
    config = _load_config(config)
    block = _block(config)
    channel = str(block.get("announce_channel") or "").strip()
    try:
        wait = float(block.get("veto_wait_working_hours", 4))
    except (TypeError, ValueError):
        raise AutonomyConfigError("autonomy.veto_wait_working_hours must be a number")
    if wait < 0:
        raise AutonomyConfigError("autonomy.veto_wait_working_hours must be >= 0")
    return channel, wait, working_hours_from_config(config)


# ── pending (veto-window) actions ────────────────────────────────────────

_PENDING_ID = re.compile(r"^[0-9a-f]{12}$")
_ACTION_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


def _pending_dir(action: str) -> Path:
    if not _ACTION_NAME.match(action):
        raise ValueError(f"invalid action name {action!r}")
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "autonomy" / "pending" / action


def _pending_path(action: str, pending_id: str) -> Path:
    if not isinstance(pending_id, str) or not _PENDING_ID.match(pending_id):
        raise ValueError(f"invalid pending id {pending_id!r}")
    return _pending_dir(action) / f"{pending_id}.json"


def _atomic_write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".pending-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def save_pending(action: str, record: dict) -> str:
    """Persist a new pending action; returns its ``pending_id``."""
    pending_id = secrets.token_hex(6)
    _atomic_write(_pending_path(action, pending_id), {**record, "pending_id": pending_id})
    return pending_id


def load_pending(action: str, pending_id: str) -> Optional[dict]:
    try:
        path = _pending_path(action, pending_id)
    except ValueError:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def update_pending(action: str, pending_id: str, **fields) -> None:
    path = _pending_path(action, pending_id)
    record = load_pending(action, pending_id) or {"pending_id": pending_id}
    record.update(fields)
    _atomic_write(path, record)


def list_pending(action: str) -> list[dict]:
    """All records for ``action`` (any status), oldest file first."""
    directory = _pending_dir(action)
    if not directory.is_dir():
        return []
    out = []
    for path in sorted(directory.glob("*.json"), key=lambda p: p.stat().st_mtime):
        rec = load_pending(action, path.stem)
        if rec is not None:
            out.append(rec)
    return out

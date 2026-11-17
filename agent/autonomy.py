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

import hashlib
import hmac
import json
import os
import re
import secrets
import tempfile
from contextlib import contextmanager
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


def veto_expiry_hours(config: Optional[dict] = None) -> float:
    """Wall-clock hours after ``not_before`` at which an unexecuted veto
    action expires (``autonomy.veto_expiry_hours``, default 72)."""
    raw = _block(_load_config(config)).get("veto_expiry_hours", 72)
    try:
        hours = float(raw)
    except (TypeError, ValueError):
        raise AutonomyConfigError("autonomy.veto_expiry_hours must be a number")
    if hours <= 0:
        raise AutonomyConfigError("autonomy.veto_expiry_hours must be > 0")
    return hours


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


# ── on-disk state: pending actions, ledger, locks ────────────────────────
#
# Everything lives under ``$HERMES_HOME/autonomy/``, which the file tools treat
# as a protected path (agent/file_safety.py) — the agent can neither read nor
# write it. Pending records are additionally HMAC-signed with a host-local key
# (``autonomy/.hmac_key``, 0600) so a hand-edited record is refused rather
# than executed. Not a boundary against a shell user on the host.

_PENDING_ID = re.compile(r"^[0-9a-f]{12}$")
_ACTION_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_KEY = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class AutonomyBusy(RuntimeError):
    """Another execution holds the lock for this key (or crashed holding it)."""


def autonomy_dir() -> Path:
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "autonomy"


def _check_action(action: str) -> str:
    if not _ACTION_NAME.match(action or ""):
        raise ValueError(f"invalid action name {action!r}")
    return action


def _check_key(key: str) -> str:
    if not isinstance(key, str) or not _KEY.match(key):
        raise ValueError(f"invalid key {key!r}")
    return key


def _atomic_write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _read_json(path: Path) -> Optional[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


# signing

def _hmac_key() -> bytes:
    path = autonomy_dir() / ".hmac_key"
    try:
        return bytes.fromhex(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:  # another process won the race
        return bytes.fromhex(path.read_text(encoding="utf-8").strip())
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(secrets.token_hex(32))
    return bytes.fromhex(path.read_text(encoding="utf-8").strip())


def _signature(pending_id: str, signed: dict) -> str:
    msg = json.dumps({"pending_id": pending_id, "signed": signed},
                     sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hmac.new(_hmac_key(), msg.encode("utf-8"), hashlib.sha256).hexdigest()


# pending (veto-window) actions

def new_pending_id() -> str:
    return secrets.token_hex(6)


def _pending_path(action: str, pending_id: str) -> Path:
    if not isinstance(pending_id, str) or not _PENDING_ID.match(pending_id):
        raise ValueError(f"invalid pending id {pending_id!r}")
    return autonomy_dir() / "pending" / _check_action(action) / f"{pending_id}.json"


def save_pending(action: str, pending_id: str, signed: dict, **state) -> None:
    """Persist a new pending action. ``signed`` holds everything execution
    trusts (what to do, when, where it was announced) and is HMAC-signed;
    ``state`` is the mutable status (``status``, outcome fields)."""
    path = _pending_path(action, pending_id)
    if path.exists():
        raise ValueError(f"pending id {pending_id} already exists")
    record = {"status": "pending", **state, "pending_id": pending_id,
              "signed": signed, "sig": _signature(pending_id, signed)}
    _atomic_write(path, record)


def load_pending(action: str, pending_id: str) -> Optional[dict]:
    try:
        path = _pending_path(action, pending_id)
    except ValueError:
        return None
    return _read_json(path)


def verify_pending(record: dict) -> bool:
    """True iff the record's signed part is untampered."""
    try:
        signed, sig, pid = record["signed"], record["sig"], record["pending_id"]
        if not isinstance(signed, dict) or not isinstance(sig, str):
            return False
        return hmac.compare_digest(_signature(pid, signed), sig)
    except Exception:
        return False


def update_pending(action: str, pending_id: str, **state) -> None:
    """Update a record's mutable state. The signed part cannot be changed."""
    if {"signed", "sig", "pending_id"} & set(state):
        raise ValueError("the signed part of a pending record is immutable")
    path = _pending_path(action, pending_id)
    record = _read_json(path)
    if record is None:
        raise ValueError(f"no pending record {pending_id}")
    record.update(state)
    _atomic_write(path, record)


def list_pending(action: str) -> list[dict]:
    """All records for ``action`` (any status), oldest file first."""
    directory = autonomy_dir() / "pending" / _check_action(action)
    if not directory.is_dir():
        return []
    out = []
    for path in sorted(directory.glob("*.json"), key=lambda p: p.stat().st_mtime):
        rec = load_pending(action, path.stem)
        if rec is not None:
            out.append(rec)
    return out


# ledger: what an action already did, per key (e.g. one draft per source row)

def _ledger_path(action: str, key: str) -> Path:
    return autonomy_dir() / "ledger" / _check_action(action) / f"{_check_key(key)}.json"


def ledger_get(action: str, key: str) -> Optional[dict]:
    return _read_json(_ledger_path(action, key))


def ledger_put(action: str, key: str, value: dict) -> None:
    _atomic_write(_ledger_path(action, key), value)


# locks

@contextmanager
def exclusive(action: str, key: str):
    """Hold an O_EXCL lock file for (action, key) for the duration.

    Raises ``AutonomyBusy`` if it is already held. A process that dies while
    holding it leaves the file behind ON PURPOSE: it may have died after an
    external write, so the key stays locked until an operator checks and
    removes ``autonomy/locks/<action>/<key>.lock``.
    """
    path = autonomy_dir() / "locks" / _check_action(action) / f"{_check_key(key)}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise AutonomyBusy(
            f"{action} for {key} is already running, or a previous run died mid-way "
            f"(lock {path}). An operator must check the target system, then remove the lock.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(f"{os.getpid()}\n")
        yield
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

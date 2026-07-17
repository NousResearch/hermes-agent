# SPDX-License-Identifier: Apache-2.0
# ---------------------------------------------------------------------------
# Portions of this file are adapted from BaiLongma
#   Upstream: https://github.com/xiaoyuanda666-ship-it/BaiLongma
#   Original: src/capabilities/tools/reminders.js
#             src/db/repositories/reminders.js (schema shape only)
#   Copyright (c) 2026 xiaoyuanda666-ship-it — Licensed under MIT
#   License text: see LICENSES/BaiLongma-MIT.txt
# ---------------------------------------------------------------------------
"""User-facing reminders: domain model + one-off/recurrence engine.

## Why this is not just cron

Hermes already ships :mod:`cron` for scheduled *agent jobs* — cron
runs prompts on a schedule against a full agent session. Reminders
solve a different problem:

* They belong to a *user* (or conversation party), not to the agent.
* They can be **merged** at the same wall-clock minute so a burst of
  "remind me at 6pm to X / to Y / to Z" collapses to one row.
* They cancel by id, list what's pending for a user, and advance
  their own ``due_at`` after firing when they're recurring.
* They persist their firing status so a late-woken agent knows what
  it missed (``fired`` / ``cancelled`` / ``pending``) instead of
  double-firing.

The module here is a **domain library** — no I/O, no tool wiring,
no message routing. The caller supplies a :class:`ReminderStore`
protocol implementation (SQLite, dict, ORM — whatever fits) and an
optional event hook. The manager handles all the recurrence maths
and merge logic in pure Python.

## Design invariants

* **Time is monotonic and injectable.** The manager takes a
  ``now_fn`` callable so tests aren't at the mercy of wall clock.
  All produced timestamps are ISO 8601 with an offset (never naive).
* **Merging is opt-in via minute-key.** A one-off reminder in the
  same UTC minute for the same user is a merge candidate; anything
  else creates a fresh row. This matches BaiLongma's SQL
  ``substr(due_at, 1, 16)`` semantics.
* **Recurrence math is pure.** :func:`calculate_next_due_at` is a
  standalone function with no state and no store dependency —
  callers can use it directly to preview "when is the next Tuesday
  09:00?" without persisting anything.
* **due_at must be in the future for one-offs.** The manager rejects
  past-times for ``once`` reminders (matches upstream); recurrence
  math always advances to the next matching slot.
* **Stdlib only.** No ``dateutil``, no ``pytz``.

Ported from BaiLongma's ``src/capabilities/tools/reminders.js`` +
``src/db/repositories/reminders.js`` (MIT).
"""
from __future__ import annotations

import calendar
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional, Protocol, Sequence


logger = logging.getLogger(__name__)


ISO_MINUTE_LEN = 16  # "YYYY-MM-DDTHH:MM"

RecurrenceKind = str  # "daily" | "weekly" | "monthly"
_VALID_RECURRENCE: frozenset[str] = frozenset({"daily", "weekly", "monthly"})

_HHMM_RE = re.compile(r"^(\d{1,2}):(\d{2})$")


# ── Data model ─────────────────────────────────────────────────────


@dataclass
class Reminder:
    """One reminder row.

    Field names mirror BaiLongma's SQLite schema so a straight
    port of the reminders table works as an in-memory implementation
    of :class:`ReminderStore`.
    """

    id: int
    user_id: str
    due_at: str  # ISO 8601 with offset
    task: str
    status: str = "pending"  # "pending" | "fired" | "cancelled"
    source: str = ""
    recurrence_type: Optional[RecurrenceKind] = None
    recurrence_config: Optional[dict[str, Any]] = None
    fired_at: Optional[str] = None
    cancelled_at: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ReminderStore(Protocol):
    """Persistence contract for the manager.

    A minimal implementation only needs the eight methods below; the
    manager never assumes anything about your indexing strategy or
    transaction model. Return values match BaiLongma's semantics:

    * ``create`` / ``append_task`` / ``cancel`` / ``mark_fired`` /
      ``advance_due_at`` return the persisted / updated row (or
      ``None`` if nothing changed).
    * ``find_mergeable_one_off`` returns ``None`` when no candidate
      exists.
    """

    def create(self, reminder: Reminder) -> Reminder:
        ...

    def find_mergeable_one_off(
        self, *, user_id: str, minute_key: str
    ) -> Optional[Reminder]:
        ...

    def append_task(
        self, *, reminder_id: int, additional_task: str
    ) -> Optional[Reminder]:
        ...

    def get_by_id(self, reminder_id: int) -> Optional[Reminder]:
        ...

    def cancel(
        self, *, reminder_id: int, cancelled_at: str
    ) -> Optional[Reminder]:
        ...

    def mark_fired(
        self, *, reminder_id: int, fired_at: str
    ) -> Optional[Reminder]:
        ...

    def advance_due_at(
        self, *, reminder_id: int, next_due_at: str
    ) -> Optional[Reminder]:
        ...

    def list_pending(self, limit: int = 50) -> Sequence[Reminder]:
        ...

    def list_due(
        self, *, now_iso: str, limit: int = 20
    ) -> Sequence[Reminder]:
        ...

    def next_pending(self) -> Optional[Reminder]:
        ...


# ── Recurrence engine (pure) ───────────────────────────────────────


def _parse_hour_minute(value: Any, label: str = "time") -> tuple[int, int]:
    text = str(value or "").strip()
    match = _HHMM_RE.match(text)
    if not match:
        raise ValueError(
            f"{label} must use HH:MM format, for example 09:00"
        )
    hour = int(match.group(1))
    minute = int(match.group(2))
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"{label} is outside the valid range")
    return hour, minute


def _iso_utc(dt: datetime) -> str:
    """Serialise as UTC ISO 8601 with a ``Z`` suffix (BaiLongma
    compatibility). Naive datetimes are treated as UTC."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (
        dt.astimezone(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _parse_iso(value: str) -> datetime:
    text = value.strip()
    # datetime.fromisoformat handles "Z" only on 3.11+; be explicit
    # so behaviour is consistent across supported versions.
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text)


def calculate_next_due_at(
    kind: RecurrenceKind,
    config: dict[str, Any],
    *,
    from_dt: Optional[datetime] = None,
) -> datetime:
    """Return the next matching wall-clock instant strictly after
    ``from_dt`` (default: :func:`datetime.now` in local tz).

    The returned datetime carries the same tzinfo as ``from_dt``.
    """
    if from_dt is None:
        from_dt = datetime.now().astimezone()
    hour, minute = _parse_hour_minute(config.get("time"), "time")

    if kind == "daily":
        next_dt = from_dt.replace(
            hour=hour, minute=minute, second=0, microsecond=0
        )
        if next_dt <= from_dt:
            next_dt = next_dt + timedelta(days=1)
        return next_dt

    if kind == "weekly":
        target_weekday_raw = config.get("weekday")
        try:
            target_weekday = int(target_weekday_raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            target_weekday = -1
        if not (0 <= target_weekday <= 6):
            raise ValueError(
                "weekday must be an integer from 0 to 6 (0=Sunday)"
            )
        # BaiLongma uses JS getDay(): 0=Sunday..6=Saturday.
        # Python weekday() is 0=Mon..6=Sun; isoweekday() is 1=Mon..7=Sun.
        # Convert to 0=Sun..6=Sat: (isoweekday % 7).
        js_weekday_now = from_dt.isoweekday() % 7
        candidate = from_dt.replace(
            hour=hour, minute=minute, second=0, microsecond=0
        )
        diff = (target_weekday - js_weekday_now + 7) % 7
        if diff == 0 and candidate <= from_dt:
            diff = 7
        return candidate + timedelta(days=diff)

    if kind == "monthly":
        target_day_raw = config.get("day_of_month")
        try:
            target_day = int(target_day_raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            target_day = -1
        if not (1 <= target_day <= 31):
            raise ValueError(
                "day_of_month must be an integer from 1 to 31"
            )
        year = from_dt.year
        month = from_dt.month
        for _ in range(12):
            _, last_day = calendar.monthrange(year, month)
            if target_day <= last_day:
                candidate = from_dt.replace(
                    year=year,
                    month=month,
                    day=target_day,
                    hour=hour,
                    minute=minute,
                    second=0,
                    microsecond=0,
                )
                if candidate > from_dt:
                    return candidate
            month += 1
            if month > 12:
                month = 1
                year += 1
        raise ValueError("Could not find the next matching month")

    raise ValueError(f"Unknown recurrence kind: {kind!r}")


# ── Formatting ─────────────────────────────────────────────────────


_WEEKDAY_ZH = ["周日", "周一", "周二", "周三", "周四", "周五", "周六"]


def format_reminder_row(row: Reminder) -> str:
    """Human-readable one-line rendering matching BaiLongma's format.

    Kept i18n-neutral apart from the weekday labels (upstream uses
    Chinese; this preserves them so ported tests stay behavioural).
    """
    if row.recurrence_type:
        config = row.recurrence_config or {}
        try:
            if isinstance(config, str):
                config = json.loads(config or "{}")
        except (TypeError, ValueError):
            config = {}
        if row.recurrence_type == "daily":
            recur_txt = f"每天 {config.get('time')}"
        elif row.recurrence_type == "weekly":
            weekday_idx = int(config.get("weekday", 0))
            weekday_name = (
                _WEEKDAY_ZH[weekday_idx]
                if 0 <= weekday_idx < len(_WEEKDAY_ZH)
                else str(weekday_idx)
            )
            recur_txt = f"每{weekday_name} {config.get('time')}"
        elif row.recurrence_type == "monthly":
            recur_txt = (
                f"每月 {config.get('day_of_month')} 号 "
                f"{config.get('time')}"
            )
        else:
            recur_txt = json.dumps(config, ensure_ascii=False)
        recurrence = f"[{row.recurrence_type}] {recur_txt}"
    else:
        recurrence = "[once]"
    return (
        f"#{row.id} {recurrence} 下次 {row.due_at} → "
        f"{row.user_id}：{row.task}"
    )


# ── Manager (write-side orchestration) ─────────────────────────────


@dataclass
class ReminderCreateRequest:
    """Structured input for :meth:`ReminderManager.create`.

    A dataclass instead of a bag of kwargs so callers can programmatically
    build requests without stringly-typed argument juggling. The tool
    layer (whichever surface exposes this to the model) is expected to
    parse the raw JSON into this shape before invoking the manager.
    """

    user_id: str
    task: str
    kind: RecurrenceKind = "once"  # "once" | "daily" | "weekly" | "monthly"
    due_at: Optional[str] = None
    time: Optional[str] = None  # HH:MM (recurrence only)
    weekday: Optional[int] = None  # 0=Sun..6=Sat (weekly only)
    day_of_month: Optional[int] = None  # 1..31 (monthly only)
    source: str = ""


class ReminderManager:
    """Domain-level orchestration for reminders."""

    def __init__(
        self,
        *,
        store: ReminderStore,
        now_fn: Optional[Callable[[], datetime]] = None,
        emit_event: Optional[Callable[[str, dict[str, Any]], None]] = None,
    ) -> None:
        self._store = store
        self._now_fn = now_fn or (lambda: datetime.now(tz=timezone.utc))
        self._emit_event = emit_event

    # ── Time / dispatch helpers ────────────────────────────────────

    def _now(self) -> datetime:
        return self._now_fn()

    def _emit(self, event: str, payload: dict[str, Any]) -> None:
        if self._emit_event is None:
            return
        try:
            self._emit_event(event, payload)
        except Exception as err:  # noqa: BLE001 — event emission is
            # advisory; a broken hook must not lose state changes.
            logger.warning(
                "[reminders] emit_event(%s) raised %s", event, err
            )

    # ── Public API ────────────────────────────────────────────────

    def create(self, request: ReminderCreateRequest) -> Reminder:
        """Create (or merge into) a reminder for ``request.user_id``.

        For ``kind='once'``, ``due_at`` is required and must be in
        the future. A one-off in the same wall-clock minute for the
        same user merges: the merge target's ``task`` is extended
        (``"task1; task2"``) and the original id is returned.

        For recurring kinds, ``due_at`` is ignored — the next slot
        is computed by :func:`calculate_next_due_at` using ``time``
        (and ``weekday`` / ``day_of_month`` where relevant).
        """
        task_text = (request.task or "").strip()
        if not task_text:
            raise ValueError("task must be a non-empty string")
        if not request.user_id:
            raise ValueError("user_id must be a non-empty string")

        kind = request.kind or "once"

        if kind == "once":
            if not request.due_at:
                raise ValueError(
                    "due_at is required for one-off reminders"
                )
            try:
                due_dt = _parse_iso(request.due_at)
            except ValueError as err:
                raise ValueError(
                    "due_at must be a valid ISO 8601 absolute time, "
                    "for example 2026-04-21T06:00:00+08:00"
                ) from err
            if due_dt.tzinfo is None:
                due_dt = due_dt.replace(tzinfo=timezone.utc)
            now = self._now()
            if due_dt <= now:
                raise ValueError(
                    "due_at must be strictly in the future"
                )
            iso_due_at = _iso_utc(due_dt)
            minute_key = iso_due_at[:ISO_MINUTE_LEN]

            merge_target = self._store.find_mergeable_one_off(
                user_id=request.user_id, minute_key=minute_key
            )
            if merge_target is not None:
                merged = self._store.append_task(
                    reminder_id=merge_target.id,
                    additional_task=task_text,
                )
                if merged is None:
                    raise RuntimeError(
                        f"failed to merge into reminder #{merge_target.id}"
                    )
                self._emit(
                    "reminder_merged",
                    {
                        "id": merge_target.id,
                        "user_id": request.user_id,
                        "due_at": merge_target.due_at,
                        "task": merged.task,
                    },
                )
                return merged

            new_row = Reminder(
                id=0,  # store assigns real id
                user_id=request.user_id,
                due_at=iso_due_at,
                task=task_text,
                status="pending",
                source=request.source,
                recurrence_type=None,
                recurrence_config=None,
            )
            created = self._store.create(new_row)
            self._emit(
                "reminder_created",
                {
                    "id": created.id,
                    "user_id": created.user_id,
                    "due_at": created.due_at,
                    "task": created.task,
                },
            )
            return created

        if kind not in _VALID_RECURRENCE:
            raise ValueError(
                f"Unknown kind {kind!r}: supports once/daily/weekly/monthly"
            )

        config: dict[str, Any] = {"time": request.time}
        if kind == "weekly":
            config["weekday"] = request.weekday
        elif kind == "monthly":
            config["day_of_month"] = request.day_of_month

        next_dt = calculate_next_due_at(kind, config, from_dt=self._now())
        iso_due_at = _iso_utc(next_dt)
        new_row = Reminder(
            id=0,
            user_id=request.user_id,
            due_at=iso_due_at,
            task=task_text,
            status="pending",
            source=request.source,
            recurrence_type=kind,
            recurrence_config=dict(config),
        )
        created = self._store.create(new_row)
        self._emit(
            "reminder_created",
            {
                "id": created.id,
                "user_id": created.user_id,
                "due_at": created.due_at,
                "task": created.task,
                "recurrence_type": kind,
                "recurrence_config": dict(config),
            },
        )
        return created

    def cancel(self, reminder_id: int) -> Reminder:
        """Cancel a pending reminder. Raises :class:`ValueError` on
        unknown / non-pending rows so tool layers can format a
        message from the exception."""
        existing = self._store.get_by_id(reminder_id)
        if existing is None:
            raise ValueError(f"reminder #{reminder_id} not found")
        if existing.status != "pending":
            raise ValueError(
                f"reminder #{reminder_id} is {existing.status}, "
                "cannot cancel"
            )
        cancelled = self._store.cancel(
            reminder_id=reminder_id,
            cancelled_at=_iso_utc(self._now()),
        )
        if cancelled is None:
            raise RuntimeError(
                f"failed to cancel reminder #{reminder_id}"
            )
        self._emit(
            "reminder_cancelled",
            {
                "id": cancelled.id,
                "user_id": cancelled.user_id,
                "task": cancelled.task,
            },
        )
        return cancelled

    def list_pending(self, limit: int = 50) -> list[Reminder]:
        return list(self._store.list_pending(limit=limit))

    def get(self, reminder_id: int) -> Optional[Reminder]:
        return self._store.get_by_id(reminder_id)

    def due_now(self, *, limit: int = 20) -> list[Reminder]:
        """Return reminders whose ``due_at`` has passed. Callers are
        expected to fire them and then invoke :meth:`mark_fired` or
        :meth:`advance_recurring`."""
        return list(
            self._store.list_due(
                now_iso=_iso_utc(self._now()), limit=limit
            )
        )

    def mark_fired(self, reminder_id: int) -> Reminder:
        """Mark a one-off reminder as fired. For recurring reminders,
        prefer :meth:`advance_recurring` which reschedules in place.
        """
        fired = self._store.mark_fired(
            reminder_id=reminder_id, fired_at=_iso_utc(self._now())
        )
        if fired is None:
            raise RuntimeError(
                f"reminder #{reminder_id} could not be marked fired "
                "(unknown id or not pending)"
            )
        return fired

    def advance_recurring(self, reminder_id: int) -> Reminder:
        """Compute the *next* ``due_at`` for a recurring reminder and
        update in place. No-op semantics for one-offs (raises)."""
        existing = self._store.get_by_id(reminder_id)
        if existing is None:
            raise ValueError(f"reminder #{reminder_id} not found")
        if not existing.recurrence_type or not existing.recurrence_config:
            raise ValueError(
                f"reminder #{reminder_id} is not recurring"
            )
        next_dt = calculate_next_due_at(
            existing.recurrence_type,
            existing.recurrence_config,
            from_dt=self._now(),
        )
        advanced = self._store.advance_due_at(
            reminder_id=reminder_id,
            next_due_at=_iso_utc(next_dt),
        )
        if advanced is None:
            raise RuntimeError(
                f"failed to advance reminder #{reminder_id}"
            )
        return advanced


__all__ = [
    "ISO_MINUTE_LEN",
    "Reminder",
    "ReminderCreateRequest",
    "ReminderManager",
    "ReminderStore",
    "RecurrenceKind",
    "calculate_next_due_at",
    "format_reminder_row",
]

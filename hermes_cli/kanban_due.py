"""Time-gated kanban cards: the ``due_at`` contract and the due-card waker.

A card parked in ``scheduled`` carries an absolute ``due_at`` (epoch seconds).
The dispatcher tick is then the ONLY thing that has to run for the card to wake:
:func:`wake_due_cards` unblocks every due card at the top of the tick, so no
external cron job is needed and nothing depends on a second scheduler being
alive. ``hermes kanban daemon`` is deprecated (the gateway owns the loop), so
in practice the tick that matters is the gateway's dispatcher watcher.

A wake is work -- unblocking a card hands it straight to the spawn pass in the
same tick -- so the wake is band-aware. ``execution_windows.json`` is the host's
machine source of truth for when scheduled work may run; it is read on every
tick that has work to do, and a due time landing inside a ``reserved`` or
``external`` band is DEFERRED to the end of that band rather than dropped. The
card keeps its place, gains a comment saying why, and is woken when the band
closes. A ``shared`` band accepts overlap, so it never holds a wake.

A card can opt out with ``due_window_policy='ambient'`` when its own wake is a
lightweight check that has to tick around the clock -- the map declares the
wake clock itself ambient for exactly that reason ("no reserved band can hold
it"). The default is to defer.

The waker fails CLOSED. If the map cannot be read it wakes nothing, because an
unreadable map could be hiding a reserved band; the reason lands on each
affected card and in the returned :class:`WakeOutcome`. What must never happen
silently is a card left parked past its due time with nothing waking it, so the
overdue case is also surfaced as a diagnostic
(``hermes_cli.kanban_diagnostics``); that diagnostic is the explicit fallback
for "the waker is not running".
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import time as dtime
from datetime import timedelta
from pathlib import Path
from typing import Any
from typing import Optional

from hermes_cli import kanban_db as kb

# Environment override, then ``kanban.execution_windows_map`` in config.yaml,
# then the host default. An explicit ``map_path`` argument beats all three (it
# is what tests and the checker use).
ENV_MAP_PATH = "HERMES_EXECUTION_WINDOWS_MAP"
CONFIG_KEY = "execution_windows_map"
DEFAULT_MAP_PATHS = ("/opt/hermes_prod/yoyodine-web-services/execution_windows.json",)

# Protections that HOLD a wake. "shared" means the band accepts overlapping
# work, so waking there starts nothing that the band's own rules forbid.
BLOCKING_PROTECTIONS = ("reserved", "external")

# A deferral must always move the card FORWARD, or the waker would re-defer the
# same card on the same tick forever. Band ends are strictly in the future
# whenever the band contains the due time (see :func:`band_end_epoch`), so this
# only catches clock skew and hand-edited maps.
MIN_DEFER_SECONDS = 60

# The author stamped on every comment the waker writes, so a deferral or a
# map failure is never mistaken for a peer's note.
WAKER_AUTHOR = "kanban-waker"

# Longest blame string we put in a comment/event; keeps a hand-edited map from
# dumping a paragraph of rationale into the board.
_MAX_LABEL = 120


class WindowMapError(RuntimeError):
    """The execution-window map is missing, unreadable, or malformed."""


def _parse_hhmm(value: Any) -> int:
    """``"01:00"`` -> 60. Raises :class:`WindowMapError` on anything else."""
    text = str(value or "").strip()
    if ":" not in text:
        raise WindowMapError(f"bad time {value!r}: expected HH:MM")
    hh, _, mm = text.partition(":")
    try:
        hour, minute = int(hh), int(mm)
    except ValueError as exc:
        raise WindowMapError(f"bad time {value!r}: expected HH:MM") from exc
    if not (0 <= hour <= 24) or not (0 <= minute < 60):
        raise WindowMapError(f"bad time {value!r}: out of range")
    return hour * 60 + minute


def _parse_days(value: Any) -> tuple[int, ...]:
    """ISO weekday numbers (1=Mon .. 7=Sun); 1-7 accepts every day."""
    if not value:
        return tuple(range(1, 8))
    out: list[int] = []
    for item in value:
        try:
            day = int(item)
        except (TypeError, ValueError) as exc:
            raise WindowMapError(f"bad day {item!r}: expected 1-7") from exc
        if not 1 <= day <= 7:
            raise WindowMapError(f"bad day {item!r}: expected 1-7")
        out.append(day)
    return tuple(sorted(set(out)))


@dataclass(frozen=True)
class Band:
    """One declared execution window (``windows`` or ``external`` entry)."""

    key: str
    label: str
    protection: str
    start_minute: int
    end_minute: int
    days: tuple[int, ...]
    priority: str = ""

    @property
    def spans_midnight(self) -> bool:
        """True for a band whose end is not later in the day than its start."""
        return self.end_minute <= self.start_minute

    @property
    def duration_minutes(self) -> int:
        span = self.end_minute - self.start_minute
        return span + 1440 if span <= 0 else span

    def contains(self, when: datetime) -> bool:
        """Is ``when`` (a local datetime) inside this band?"""
        if when.isoweekday() not in self.days:
            # A band that spans midnight also covers the early hours of the day
            # AFTER each of its declared days.
            if not self.spans_midnight:
                return False
            previous = (when - timedelta(days=1)).isoweekday()
            if previous not in self.days:
                return False
        minute = when.hour * 60 + when.minute
        if self.spans_midnight:
            return minute >= self.start_minute or minute < self.end_minute
        return self.start_minute <= minute < self.end_minute

    def to_dict(self) -> dict:
        return {
            "key": self.key, "label": self.label, "protection": self.protection,
            "start": f"{self.start_minute // 60:02d}:{self.start_minute % 60:02d}",
            "end": f"{self.end_minute // 60:02d}:{self.end_minute % 60:02d}",
            "days": list(self.days), "priority": self.priority,
        }


@dataclass(frozen=True)
class WindowMap:
    """Every band the host declares, plus where the map was read from."""

    path: Optional[Path]
    version: Any
    timezone: str
    bands: tuple[Band, ...]

    def blocking_band(self, when: datetime, tzinfo=None) -> Optional[Band]:
        """The band that holds a wake at ``when``, or ``None``.

        When several bands contain the instant, the one that CLOSES LATEST wins:
        deferring to an earlier band's end would land the wake inside a band that
        is still open. Comparing real close times (not raw ``HH:MM``) matters --
        a band ending at 01:00 and one ending at 23:30 on the same evening are
        not comparable as strings or as minutes-since-midnight.
        """
        hits = [
            b for b in self.bands
            if b.protection in BLOCKING_PROTECTIONS and b.contains(when)
        ]
        if not hits:
            return None
        tz = tzinfo or when.tzinfo
        return max(hits, key=lambda b: band_end_epoch(b, when, tz))

    def to_dict(self) -> dict:
        return {
            "path": str(self.path) if self.path else None,
            "version": self.version, "timezone": self.timezone,
            "bands": [b.to_dict() for b in self.bands],
        }


def _build_map(doc: dict, path: Optional[Path]) -> WindowMap:
    raw = list(doc.get("windows") or [])
    # ``external`` entries (operator crontab, launchd timers, OS events) are
    # fixed host time. They are declared in their own key but belong in the same
    # view, and where one collides with a reserved window the RESERVED window is
    # what moves -- so a wake must never be aimed into one either.
    for item in doc.get("external") or []:
        if isinstance(item, dict):
            raw.append({**item, "protection": "external"})
    bands: list[Band] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key") or "").strip()
        if not key:
            raise WindowMapError("window entry without a 'key'")
        bands.append(Band(
            key=key,
            label=str(item.get("label") or key),
            protection=str(item.get("protection") or "reserved").strip().lower(),
            start_minute=_parse_hhmm(item.get("start")),
            end_minute=_parse_hhmm(item.get("end")),
            days=_parse_days(item.get("days")),
            priority=str(item.get("priority") or ""),
        ))
    tz = str(doc.get("timezone") or "").strip() or (time.tzname[0] or "UTC")
    return WindowMap(path=path, version=doc.get("version"), timezone=tz, bands=tuple(bands))


def load_window_map(path: Optional[Path]) -> WindowMap:
    """Read and parse the map. Raises :class:`WindowMapError` on any problem.

    Every failure mode shares one exception type on purpose: callers must treat
    "no map", "unparseable map" and "map that is missing its windows" the same
    way (fail closed), and the exception text is what the operator sees.
    """
    if path is None:
        raise WindowMapError(
            "no execution-window map configured "
            f"(set {(ENV_MAP_PATH)} or kanban.{CONFIG_KEY} in config.yaml)"
        )
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise WindowMapError(f"cannot read {path}: {exc}") from exc
    try:
        doc = json.loads(raw)
    except ValueError as exc:
        raise WindowMapError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(doc, dict):
        raise WindowMapError(f"{path} must contain a JSON object")
    if not doc.get("windows") and not doc.get("external"):
        raise WindowMapError(f"{path} declares no windows")
    return _build_map(doc, Path(path))


def resolve_map_path(cfg: Optional[dict] = None) -> Optional[Path]:
    """Locate the map: env override, then config, then the host default."""
    env = os.environ.get(ENV_MAP_PATH)
    if env and env.strip():
        return Path(env.strip()).expanduser()
    if cfg is None:
        try:
            from hermes_cli.config import load_config
            cfg = load_config() or {}
        except Exception:
            cfg = {}
    configured = (cfg.get("kanban") or {}).get(CONFIG_KEY)
    if configured and str(configured).strip():
        return Path(str(configured).strip()).expanduser()
    for candidate in DEFAULT_MAP_PATHS:
        if Path(candidate).exists():
            return Path(candidate)
    return None


def _tzinfo(name: str):
    """ZoneInfo for the map's declared timezone; host local time as a fallback."""
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(name)
    except Exception:
        return datetime.now().astimezone().tzinfo


_DUE_RELATIVE_RE = re.compile(r"^\+\s*(\d+)\s*([smhd])$")
_DUE_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}
_DUE_FORMATS = (
    "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M%z",
)


def parse_due(value: Any, *, now: Optional[int] = None, tzinfo=None) -> int:
    """Epoch seconds from a due-time string.

    Accepts ``+30m`` / ``+2h`` / ``+1d`` / ``+45s`` (relative to ``now``), a bare
    epoch (9+ digits) or an ISO-8601 timestamp (``2026-09-16T01:40``).

    A naive timestamp is read on the MAP's clock (the same clock the bands use),
    so ``--due 01:40`` and a band that starts at 01:00 agree about when 01:40 is.
    Raises :class:`ValueError` on anything else -- a mistyped due time must fail
    loudly rather than silently park a card forever.
    """
    text = str(value or "").strip()
    if not text:
        raise ValueError("empty due time")
    base = int(now if now is not None else time.time())
    match = _DUE_RELATIVE_RE.match(text)
    if match:
        return base + int(match.group(1)) * _DUE_UNITS[match.group(2)]
    if text.isdigit() and len(text) >= 9:
        return int(text)
    for fmt in _DUE_FORMATS:
        try:
            parsed = datetime.strptime(text, fmt)
        except ValueError:
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=tzinfo or datetime.now().astimezone().tzinfo)
        return int(parsed.timestamp())
    raise ValueError(
        f"cannot read {value!r} as a due time -- use ISO-8601 (2026-09-16T01:40), "
        f"a relative offset (+30m, +2h, +1d) or epoch seconds"
    )


def format_due(epoch: Optional[int], tzinfo=None) -> str:
    """Human rendering of a due time, in the map's timezone when we know it."""
    if not epoch:
        return "no due time"
    tz = tzinfo or datetime.now().astimezone().tzinfo
    return datetime.fromtimestamp(int(epoch), tz).strftime("%Y-%m-%d %H:%M %Z")


def band_end_epoch(band: Band, when: datetime, tzinfo) -> int:
    """The epoch at which ``band`` closes relative to ``when``.

    ``when`` must be inside the band (the only caller checks). A band that spans
    midnight and was entered at or after its start closes on the FOLLOWING day;
    every other case closes later the same day. Either way the result is
    strictly after ``when``, which is what makes repeated deferral terminate.
    """
    day = when.date()
    if band.spans_midnight and (when.hour * 60 + when.minute) >= band.start_minute:
        day = day + timedelta(days=1)
    end_hour, end_minute = divmod(band.end_minute, 60)
    # 24:00 is a legal band end and means midnight of the following day.
    if end_hour >= 24:
        day = day + timedelta(days=1)
        end_hour = end_hour - 24
    close = datetime.combine(day, dtime(hour=end_hour, minute=end_minute), tzinfo=tzinfo)
    return int(close.timestamp())


@dataclass
class Deferral:
    """One card whose wake was pushed to the end of a band."""

    task_id: str
    due_at: int
    band: Band
    band_end: int
    reason: str = "band"

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id, "due_at": self.due_at, "band_end": self.band_end,
            "reason": self.reason, "band": self.band.key, "protection": self.band.protection,
        }


@dataclass
class WakeOutcome:
    """What one waker pass did. ``problems`` is operator-facing text."""

    woken: list[str] = field(default_factory=list)
    deferred: list[Deferral] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def touched(self) -> bool:
        return bool(self.woken or self.deferred or self.problems)


def _band_blame(band: Band) -> str:
    label = band.label if len(band.label) <= _MAX_LABEL else band.label[: _MAX_LABEL - 3] + "..."
    start = f"{band.start_minute // 60:02d}:{band.start_minute % 60:02d}"
    end = f"{band.end_minute // 60:02d}:{band.end_minute % 60:02d}"
    return f"{label} ({start}-{end}, {band.protection})"


def _clock(epoch: int, tzinfo) -> str:
    return datetime.fromtimestamp(int(epoch), tzinfo).strftime("%Y-%m-%d %H:%M %Z")


def wake_due_cards(
    conn,
    *,
    board: Optional[str] = None,
    now: Optional[int] = None,
    map_path: Optional[Path] = None,
    cfg: Optional[dict] = None,
    dry_run: bool = False,
    limit: int = 200,
) -> WakeOutcome:
    """Wake every due ``scheduled`` or time-fenced ``blocked`` card whose ``due_at`` has passed.

    Runs inside the board's single-writer dispatch critical section, ahead of
    the spawn pass, so a card woken here is spawnable on the SAME tick.

    Every deferral moves the card forward (never to a past time), so a pass is
    idempotent and cannot spin. ``dry_run`` reports what it would do and writes
    nothing.
    """
    now_ts = int(now if now is not None else time.time())
    outcome = WakeOutcome()
    due = kb.list_due_tasks(conn, now=now_ts, limit=limit, statuses=("scheduled", "blocked"))
    if not dry_run:
        # Heartbeat BEFORE the band check and before the early return: if the
        # window map turns out to be broken, "the tick ran and refused to wake"
        # is the useful story, not "nothing is ticking at all".
        kb.record_due_waker_tick(conn, now=now_ts)
    if not due:
        return outcome

    if map_path is None and cfg is None:
        map_path = resolve_map_path()
    try:
        window_map = load_window_map(map_path)
    except WindowMapError as exc:
        # Fail closed: wake nothing, say why on every affected card.
        outcome.problems.append(str(exc))
        if not dry_run:
            for task in due:
                kb.add_comment(
                    conn, task.id, WAKER_AUTHOR,
                    f"Wake blocked: the execution-window map could not be read "
                    f"({exc}). Nothing was woken -- an unreadable map could be "
                    f"holding a reserved band, and the card stays parked until "
                    f"the map is readable again.",
                )
        return outcome

    tzinfo = _tzinfo(window_map.timezone)
    for task in due:
        policy = (task.due_window_policy or "defer")
        when = datetime.fromtimestamp(now_ts, tzinfo)
        band = None if policy == "ambient" else window_map.blocking_band(when, tzinfo)
        if band is not None:
            target = band_end_epoch(band, when, tzinfo)
            if target <= now_ts:
                target = now_ts + MIN_DEFER_SECONDS
            outcome.deferred.append(Deferral(
                task_id=task.id, due_at=task.due_at, band=band, band_end=target,
            ))
            if dry_run:
                continue
            if task.status == "blocked":
                # A time-fenced hold must STAY blocked (never leak into
                # ``scheduled``): push its due time forward in place.
                kb.defer_due(conn, task.id, due_at=target)
            else:
                kb.schedule_task(
                    conn, task.id,
                    reason=(
                        f"deferred out of the {band.protection} band '{band.key}' "
                        f"({_band_blame(band)}) by the due-card waker"
                    ),
                    due_at=target,
                )
            kb.add_comment(
                conn, task.id, WAKER_AUTHOR,
                f"Wake deferred to {_clock(target, tzinfo)}: the due time "
                f"{_clock(task.due_at, tzinfo)} falls inside the {band.protection} "
                f"execution band '{band.key}' -- {_band_blame(band)}. A wake hands "
                f"this card to the spawn pass in the same tick, so it is held until "
                f"the band closes instead of starting work inside it. Re-arm with "
                f"`hermes kanban schedule {task.id} --due <when> --window-policy "
                f"ambient` if this wake is a lightweight check that must tick "
                f"around the clock.",
            )
            continue
        if dry_run:
            outcome.woken.append(task.id)
            continue
        if kb.unblock_task(conn, task.id, extra_event={
            "woken_by": "due-card-waker",
            "due_at": task.due_at,
            "window_policy": policy,
            "board": board,
        }):
            outcome.woken.append(task.id)
    return outcome

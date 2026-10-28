"""Advisory daily spend ledger for Mixture-of-Agents calls.

One SQLite ledger lives at the Hermes root shared by all profiles. The threshold
is deliberately advisory: recording an over-budget event emits one alarm per
Warsaw calendar day and never raises a budget veto into the model call.
"""

from __future__ import annotations

import hashlib
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

_WARSAW = ZoneInfo("Europe/Warsaw")


@dataclass(frozen=True)
class MoADailySpend:
    day: str
    total_usd: float
    budget_usd: float
    event_added: bool
    alarm_emitted: bool


def _ledger_path(home: Path | str | None = None) -> Path:
    from hermes_constants import get_default_hermes_root, get_hermes_home

    profile_home = Path(home) if home is not None else get_hermes_home()
    root = get_default_hermes_root(home=profile_home)
    return root / "usage" / "moa_daily_spend.sqlite3"


def record_moa_spend(
    cost_usd: float,
    budget_usd: float,
    event_id: str,
    *,
    home: Path | str | None = None,
    now: datetime | None = None,
) -> MoADailySpend:
    """Record one priced MoA response; atomically signal the first daily crossing.

    ``event_id`` is hashed before storage and protected by a unique index, so a
    retried accounting hook cannot inflate spend. The ledger is shared across
    profile homes beneath the same Hermes root.
    """
    cost = float(cost_usd)
    budget = float(budget_usd)
    if not math.isfinite(cost) or cost < 0:
        raise ValueError("cost_usd must be a finite non-negative number")
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError("budget_usd must be a finite positive number")
    if not str(event_id).strip():
        raise ValueError("event_id is required")

    local_now = now or datetime.now(_WARSAW)
    if local_now.tzinfo is None:
        local_now = local_now.replace(tzinfo=_WARSAW)
    local_now = local_now.astimezone(_WARSAW)
    day = local_now.date().isoformat()
    stamp = local_now.isoformat(timespec="seconds")
    event_hash = hashlib.sha256(str(event_id).encode("utf-8", "replace")).hexdigest()
    from hermes_constants import get_hermes_home

    profile_home = Path(home) if home is not None else get_hermes_home()
    path = _ledger_path(profile_home)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.touch(mode=0o600, exist_ok=True)

    with sqlite3.connect(path, timeout=10.0) as conn:
        conn.execute("PRAGMA busy_timeout=10000")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS moa_spend_events ("
            "event_hash TEXT PRIMARY KEY, day TEXT NOT NULL, occurred_at TEXT NOT NULL, "
            "profile_home TEXT NOT NULL, cost_usd REAL NOT NULL)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS moa_spend_events_day_idx ON moa_spend_events(day)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS moa_spend_alarms ("
            "day TEXT PRIMARY KEY, threshold_usd REAL NOT NULL, total_usd REAL NOT NULL, "
            "fired_at TEXT NOT NULL)"
        )
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.execute(
            "INSERT OR IGNORE INTO moa_spend_events(event_hash, day, occurred_at, profile_home, cost_usd) "
            "VALUES (?, ?, ?, ?, ?)",
            (event_hash, day, stamp, str(profile_home), cost),
        )
        event_added = cursor.rowcount == 1
        total = float(conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0.0) FROM moa_spend_events WHERE day = ?", (day,)
        ).fetchone()[0])
        alarm_emitted = False
        if total > budget:
            alarm = conn.execute(
                "INSERT OR IGNORE INTO moa_spend_alarms(day, threshold_usd, total_usd, fired_at) "
                "VALUES (?, ?, ?, ?)",
                (day, budget, total, stamp),
            )
            alarm_emitted = alarm.rowcount == 1
        conn.commit()
    return MoADailySpend(day, total, budget, event_added, alarm_emitted)

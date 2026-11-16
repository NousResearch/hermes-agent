"""Per-profile cost budget + alerts (fork-only).

Opt-in through a top-level ``cost_budget:`` block in the profile's
``config.yaml``. With the block absent every entry point returns before any
I/O, so a profile without it is unchanged.

* **Spend** comes from the ``hermes costs`` query layer
  (``agent.cost_attribution.attribute_sessions``) over the owning profile's
  ``state.db``, resolved through ``get_hermes_home()`` on every call — one
  multiplexed gateway ticks several homes — and cached per home for
  ``cache_seconds``.
* **Enforcement**: the cron scheduler asks ``preflight_cron_job`` before a run;
  a discretionary job is skipped while the profile is over its daily or
  monthly budget. Everything else keeps running, with a louder alert.
* **Alerts** go out once per key (threshold crossing per period, per costly
  run, unpriced tokens per day, overrun per job per period) through
  ``cron.scheduler._deliver_result`` with the adapters the caller passes — the
  owning profile's adapters, since both callers run in that profile's scope.

Every path fails open: a budget bug must never stop cron or the gateway.
Design: docs/superpowers/specs/2026-11-16-product-education-teammate-design.md
("Dependencies → 3. Cost budget + alerts").
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

DEFAULT_ALERT_AT = (0.5, 0.8, 1.0)
DEFAULT_CACHE_SECONDS = 300
MIN_CACHE_SECONDS = 30
_SLACK_CHANNEL_ID_RE = re.compile(r"^[CDG][A-Z0-9]{6,}$")


# ── config ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BudgetConfig:
    daily_usd: Optional[float] = None
    monthly_usd: Optional[float] = None
    alert_at: Tuple[float, ...] = DEFAULT_ALERT_AT
    per_run_usd: Optional[float] = None
    alert_channel: Optional[str] = None
    alert_mention: Optional[str] = None
    discretionary_jobs: FrozenSet[str] = frozenset()
    cache_seconds: int = DEFAULT_CACHE_SECONDS


def _positive_float(value: Any, key: str) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        logger.warning("cost_budget.%s: %r is not a number — ignored", key, value)
        return None
    return number if number > 0 else None


def _str_set(value: Any) -> FrozenSet[str]:
    if isinstance(value, str):
        value = value.split(",")
    if not isinstance(value, (list, tuple, set)):
        return frozenset()
    return frozenset(str(v).strip() for v in value if str(v).strip())


def _alert_channel(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    if not text:
        return None
    # A bare Slack channel id is the common case; anything else is already a
    # deliver target ("slack:C…", "telegram:-100…", "slack" = home channel).
    return f"slack:{text}" if _SLACK_CHANNEL_ID_RE.match(text) else text


def load_budget_config(raw: Optional[dict] = None) -> Optional[BudgetConfig]:
    """The profile's ``cost_budget:`` block, or None when absent / inert.

    ``raw`` is the whole config dict; by default the current profile's
    config.yaml (cached raw read — cheap enough for every cron tick).
    """
    if raw is None:
        from hermes_cli.config import read_raw_config_readonly

        raw = read_raw_config_readonly() or {}
    block = raw.get("cost_budget") if isinstance(raw, dict) else None
    if not isinstance(block, dict):
        return None
    daily = _positive_float(block.get("daily_usd"), "daily_usd")
    monthly = _positive_float(block.get("monthly_usd"), "monthly_usd")
    per_run = _positive_float(block.get("per_run_usd"), "per_run_usd")
    if daily is None and monthly is None and per_run is None:
        return None

    alert_at = DEFAULT_ALERT_AT
    if "alert_at" in block:
        raw_at = block.get("alert_at")
        items = raw_at if isinstance(raw_at, (list, tuple)) else [raw_at]
        parsed = {t for t in (_positive_float(v, "alert_at") for v in items) if t is not None}
        alert_at = tuple(sorted(parsed))

    try:
        cache_seconds = int(block.get("cache_seconds", DEFAULT_CACHE_SECONDS))
    except (TypeError, ValueError):
        cache_seconds = DEFAULT_CACHE_SECONDS
    mention = str(block.get("alert_mention") or "").strip() or None
    return BudgetConfig(
        daily_usd=daily, monthly_usd=monthly, alert_at=alert_at, per_run_usd=per_run,
        alert_channel=_alert_channel(block.get("alert_channel")), alert_mention=mention,
        discretionary_jobs=_str_set(block.get("discretionary_jobs")),
        cache_seconds=max(MIN_CACHE_SECONDS, cache_seconds),
    )


def is_discretionary(job: dict, cfg: BudgetConfig) -> bool:
    if job.get("discretionary") is True:
        return True
    return bool({str(job.get("id") or ""), str(job.get("name") or "")} & cfg.discretionary_jobs)


# ── spend ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RunSpend:
    """One cron run: its root session and every session in its lineage."""
    root_id: str
    job_id: str
    started_at: float
    cost_usd: float
    unpriced_tokens: int


@dataclass(frozen=True)
class SpendSnapshot:
    day_key: str
    month_key: str
    day_usd: float
    month_usd: float
    day_unpriced_tokens: int
    month_unpriced_tokens: int
    unpriced_models: Tuple[str, ...]
    runs: Tuple[RunSpend, ...]
    as_of: float

    @property
    def spend_is_lower_bound(self) -> bool:
        return self.month_unpriced_tokens > 0


def _period_starts(now: datetime) -> Tuple[datetime, datetime]:
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return day_start, day_start.replace(day=1)


def _keys(now: datetime) -> Tuple[str, str]:
    return now.strftime("%Y-%m-%d"), now.strftime("%Y-%m")


def compute_spend(home: Path, now: datetime) -> SpendSnapshot:
    """Spend for the day and month containing ``now``, from ``home``'s store."""
    day_key, month_key = _keys(now)
    db_path = Path(home) / "state.db"
    if not db_path.exists():
        return SpendSnapshot(day_key, month_key, 0.0, 0.0, 0, 0, (), (), now.timestamp())

    from agent.cost_attribution import attribute_sessions
    from hermes_state import SessionDB

    day_start, month_start = _period_starts(now)
    db = SessionDB(db_path=db_path, read_only=True)
    try:
        # No job/target resolution: the budget needs only the cron job id
        # (from the root id), and target resolution costs up to a second a job.
        sessions = attribute_sessions(db, since=month_start.timestamp(), until=now.timestamp() + 1,
                                      job_resolver=lambda _job_id: None,
                                      target_resolver=lambda _job: [])
    finally:
        db.close()

    day_ts = day_start.timestamp()
    day_usd = month_usd = 0.0
    day_unpriced = month_unpriced = 0
    unpriced_models = set()
    runs: Dict[str, Dict[str, Any]] = {}
    for s in sessions:
        month_usd += s.cost_usd
        month_unpriced += s.unpriced_tokens
        if s.started_at >= day_ts:
            day_usd += s.cost_usd
            day_unpriced += s.unpriced_tokens
        unpriced_models.update(m.model for m in s.models.values() if not m.priced and m.model)
        if s.job_id:
            run = runs.setdefault(s.root_id, {"job_id": s.job_id, "started_at": s.started_at,
                                              "cost_usd": 0.0, "unpriced_tokens": 0})
            run["started_at"] = min(run["started_at"], s.started_at)
            run["cost_usd"] += s.cost_usd
            run["unpriced_tokens"] += s.unpriced_tokens
    return SpendSnapshot(
        day_key=day_key, month_key=month_key, day_usd=day_usd, month_usd=month_usd,
        day_unpriced_tokens=day_unpriced, month_unpriced_tokens=month_unpriced,
        unpriced_models=tuple(sorted(unpriced_models)),
        runs=tuple(RunSpend(root_id=rid, **vals) for rid, vals in sorted(runs.items())),
        as_of=now.timestamp(),
    )


_cache_lock = threading.Lock()
_spend_cache: Dict[str, Tuple[float, SpendSnapshot]] = {}


def _home_key() -> str:
    return str(get_hermes_home().resolve())


def _now() -> datetime:
    from hermes_time import now as hermes_now

    return hermes_now()


def get_spend(cfg: BudgetConfig, *, now: Optional[datetime] = None) -> SpendSnapshot:
    """The current profile's spend, cached per home for ``cfg.cache_seconds``."""
    now = now or _now()
    key = _home_key()
    day_key, month_key = _keys(now)
    with _cache_lock:
        cached = _spend_cache.get(key)
    if cached is not None:
        stamp, snap = cached
        if (time.monotonic() - stamp < cfg.cache_seconds
                and (snap.day_key, snap.month_key) == (day_key, month_key)):
            return snap
    snap = compute_spend(Path(key), now)
    with _cache_lock:
        _spend_cache[key] = (time.monotonic(), snap)
    return snap


# ── status ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BudgetStatus:
    cfg: BudgetConfig
    snap: SpendSnapshot
    day_fraction: Optional[float]
    month_fraction: Optional[float]
    over_daily: bool
    over_monthly: bool

    @property
    def over_budget(self) -> bool:
        return self.over_daily or self.over_monthly


def budget_status(cfg: BudgetConfig, snap: SpendSnapshot) -> BudgetStatus:
    day_fraction = snap.day_usd / cfg.daily_usd if cfg.daily_usd else None
    month_fraction = snap.month_usd / cfg.monthly_usd if cfg.monthly_usd else None
    return BudgetStatus(
        cfg=cfg, snap=snap, day_fraction=day_fraction, month_fraction=month_fraction,
        over_daily=day_fraction is not None and day_fraction >= 1.0,
        over_monthly=month_fraction is not None and month_fraction >= 1.0,
    )


def _reset_for_tests() -> None:
    with _cache_lock:
        _spend_cache.clear()

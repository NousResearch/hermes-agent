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

import json
import logging
import os
import re
import tempfile
import threading
import time
from dataclasses import dataclass
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




# ── alerts ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Alert:
    key: str
    text: str
    loud: bool = False
    # Keys marked sent together with this one: the lower thresholds of the
    # same period, so a jump from 40% to 110% sends one alert, not three.
    covers: Tuple[str, ...] = ()


_STATE_RETENTION_SECONDS = 40 * 86400


def _fmt_threshold(t: float) -> str:
    return f"{t:g}"


def _money(value: float) -> str:
    return f"${value:,.2f}"


def _threshold_alert(status: BudgetStatus, *, period: str) -> Optional[Alert]:
    cfg, snap = status.cfg, status.snap
    if period == "daily":
        fraction, spent, budget, key = status.day_fraction, snap.day_usd, cfg.daily_usd, snap.day_key
        label, resets, unpriced = "Daily", "tomorrow", snap.day_unpriced_tokens
    else:
        fraction, spent, budget, key = status.month_fraction, snap.month_usd, cfg.monthly_usd, snap.month_key
        label, resets, unpriced = "Monthly", "next month", snap.month_unpriced_tokens
    if fraction is None:
        return None
    crossed = [t for t in cfg.alert_at if fraction >= t]
    if not crossed:
        return None
    top = max(crossed)
    keys = [f"{period}:{key}:{_fmt_threshold(t)}" for t in crossed]
    amount = f"at least {_money(spent)}" if unpriced else _money(spent)
    text = (f"{label} spend is {amount}, {fraction:.0%} of the {_money(budget)} "
            f"{period} budget ({key}).")
    if unpriced:
        text += f" {unpriced:,} tokens have no price, so the real figure is higher."
    loud = top >= 1.0
    if loud:
        text = (f"🚨 Over budget. {text} Discretionary jobs are skipped until {resets}; "
                "release (non-discretionary) work keeps running.")
    return Alert(key=keys[-1], text=text, loud=loud, covers=tuple(keys[:-1]))


def pending_alerts(status: BudgetStatus) -> List[Alert]:
    """Every alert the current spend warrants; ``send_once`` drops those already sent."""
    cfg, snap = status.cfg, status.snap
    alerts = [a for a in (_threshold_alert(status, period="daily"),
                          _threshold_alert(status, period="monthly")) if a]
    if cfg.per_run_usd:
        for run in snap.runs:
            if run.cost_usd >= cfg.per_run_usd:
                alerts.append(Alert(
                    key=f"run:{run.root_id}",
                    text=(f"A cron run of job `{run.job_id}` cost {_money(run.cost_usd)}, over the "
                          f"{_money(cfg.per_run_usd)} per-run limit (session `{run.root_id}`)."),
                ))
    if snap.day_unpriced_tokens:
        models = ", ".join(snap.unpriced_models) or "unknown model"
        alerts.append(Alert(
            key=f"unpriced:{snap.day_key}",
            text=(f"{snap.day_unpriced_tokens:,} tokens today have no price ({models}), so budget "
                  "spend is understated. Add per-million rates under pricing.overrides in "
                  "config.yaml, then run `hermes costs --reprice`."),
        ))
    return alerts


_state_lock = threading.RLock()
_logged_without_channel: set = set()


def _state_path() -> Path:
    return get_hermes_home() / "cost_budget" / "alerts.json"


def _load_sent(path: Path) -> Dict[str, float]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError):
        logger.warning("cost_budget: unreadable alert state %s — starting fresh", path)
        return {}
    sent = data.get("sent") if isinstance(data, dict) else None
    return {str(k): float(v) for k, v in sent.items()} if isinstance(sent, dict) else {}


def _save_sent(path: Path, sent: Dict[str, float]) -> None:
    cutoff = time.time() - _STATE_RETENTION_SECONDS
    sent = {k: v for k, v in sent.items() if v >= cutoff}
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".alerts-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump({"sent": sent}, fh, indent=1, sort_keys=True)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _deliver(target: str, text: str, *, adapters=None, loop=None) -> Optional[str]:
    """Send through the cron delivery path; None on success, else the error."""
    from agent.secret_scope import build_profile_secret_scope, reset_secret_scope, set_secret_scope
    from cron.scheduler import _deliver_result

    token = set_secret_scope(build_profile_secret_scope(get_hermes_home()))
    try:
        return _deliver_result({"id": "cost_budget", "name": "Cost budget", "deliver": target},
                               text, adapters=adapters, loop=loop)
    except Exception as exc:  # an adapter error must not escape into cron
        return f"{type(exc).__name__}: {exc}"
    finally:
        reset_secret_scope(token)


def send_once(alerts: List[Alert], cfg: BudgetConfig, *, adapters=None, loop=None) -> int:
    """Deliver each alert whose key was never sent from this profile; return how many went out.

    A key is recorded only after a successful delivery, so an alert raised
    while the adapter is down goes out on a later check.
    """
    if not alerts:
        return 0
    path = _state_path()
    with _state_lock:
        sent = _load_sent(path)
        fresh = [a for a in alerts if a.key not in sent]
        if not fresh:
            return 0
        if not cfg.alert_channel:
            for alert in fresh:
                marker = (str(path), alert.key)
                if marker not in _logged_without_channel:
                    _logged_without_channel.add(marker)
                    logger.warning("cost_budget alert (no cost_budget.alert_channel set): %s", alert.text)
            return 0
        delivered = 0
        for alert in fresh:
            if alert.key in sent:
                continue  # covered by an alert sent earlier in this batch
            text = f"{cfg.alert_mention} {alert.text}" if alert.loud and cfg.alert_mention else alert.text
            error = _deliver(cfg.alert_channel, text, adapters=adapters, loop=loop)
            if error is not None:
                logger.warning("cost_budget: alert %s not delivered to %s: %s",
                               alert.key, cfg.alert_channel, error)
                continue
            stamp = time.time()
            for key in (alert.key, *alert.covers):
                sent[key] = stamp
            delivered += 1
        if delivered:
            _save_sent(path, sent)
        return delivered


_watch_lock = threading.Lock()
_last_watch: Dict[str, float] = {}


def watch(*, adapters=None, loop=None) -> None:
    """Check the current profile's spend and send due alerts. Called from every cron tick.

    Returns before any I/O beyond the cached config read when the profile has
    no ``cost_budget:`` block; otherwise throttled to ``cache_seconds`` per home.
    Never raises.
    """
    try:
        cfg = load_budget_config()
        if cfg is None:
            return
        key = _home_key()
        now_m = time.monotonic()
        with _watch_lock:
            last = _last_watch.get(key)
            if last is not None and now_m - last < cfg.cache_seconds:
                return
            _last_watch[key] = now_m
        status = budget_status(cfg, get_spend(cfg))
        send_once(pending_alerts(status), cfg, adapters=adapters, loop=loop)
    except Exception:
        logger.warning("cost_budget: budget watch failed", exc_info=True)


def _reset_for_tests() -> None:
    with _cache_lock:
        _spend_cache.clear()
    with _watch_lock:
        _last_watch.clear()
    _logged_without_channel.clear()

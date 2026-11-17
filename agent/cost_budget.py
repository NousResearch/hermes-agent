"""Per-profile cost budget + alerts (fork-only).

Opt-in through a top-level ``cost_budget:`` block in the profile's
``config.yaml``. With the block absent every entry point returns after a cached
config read, so a profile without it is unchanged (no store opened, no state
written).

* **Spend** is attributed to *when it happened*. ``session_model_usage`` rows
  are cumulative upserts with no per-call timestamp, so windowing on
  ``sessions.started_at`` misses a long-lived session (a Slack channel root
  never resets) that keeps spending. Instead each scan reads every usage row
  touched since the last scan, takes the delta of its cumulative cost against
  the value recorded last time, and books that delta to the local day of the
  row's ``last_seen``. The deltas live in ``${HERMES_HOME}/cost_budget/spend.json``.
  The first scan of a profile (cold start) has no previous values, so it books
  each row's whole cumulative cost to its ``last_seen`` day and marks those
  days as an **upper bound** — it over-counts (fail closed) rather than missing
  spend. Resolved through ``get_hermes_home()`` per call (one multiplexed
  gateway ticks several homes) and cached per home for ``cache_seconds``.
* **Enforcement**: the cron scheduler asks ``preflight_cron_job`` before a run;
  a discretionary job is skipped while the profile is over its daily or
  monthly budget. Everything else keeps running.
* **Alerts**: one per key — a threshold crossing per period (``1.0`` is always a
  threshold and is the single over-budget alert, whoever trips it first), a
  batch of costly cron runs, unpriced tokens per day. Keys are claimed under a
  lock and the send happens outside it, on a thread carrying the caller's
  context (home override + the owning profile's adapters), so neither the
  state lock nor the cron tick lock is ever held across a network call. A
  failed send releases its claim and retries on a later check.

Every path fails open: a budget bug must never stop cron or the gateway.
Period boundaries use ``hermes_time``'s process-wide timezone.
Design: docs/superpowers/specs/2026-11-16-product-education-teammate-design.md
("Dependencies → 3. Cost budget + alerts").
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import os
import re
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterator, List, Optional, Tuple

try:
    import fcntl
except ImportError:  # Windows: in-process lock only
    fcntl = None  # type: ignore[assignment]

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

DEFAULT_ALERT_AT = (0.5, 0.8, 1.0)
MAX_THRESHOLD = 5.0
DEFAULT_CACHE_SECONDS = 300
MIN_CACHE_SECONDS = 30
ERROR_CACHE_SECONDS = 60
RETENTION_SECONDS = 40 * 86400
SCAN_OVERLAP_SECONDS = 600
MAX_RUNS_LISTED = 10
_SLACK_CHANNEL_ID_RE = re.compile(r"^[CDG][A-Z0-9]{6,}$")
_NON_TARGETS = frozenset({"local", "origin", "none", "off"})


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
    warnings: Tuple[str, ...] = ()
    # Opt-in fail-closed mode: every job is discretionary unless its id is in
    # essential_jobs. Both live only in config.yaml (operator-edited), so a
    # job the agent creates without the flag cannot escape the budget.
    default_discretionary: bool = False
    essential_jobs: FrozenSet[str] = frozenset()


_warned: set = set()


def _warn_once(message: str) -> None:
    marker = (str(get_hermes_home()), message)
    if marker not in _warned:
        _warned.add(marker)
        logger.error("cost_budget config: %s", message)


def _positive_float(value: Any, key: str, warnings: List[str]) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        warnings.append(f"{key}: {value!r} is not a number — ignored")
        return None
    return number if number > 0 else None


def _str_set(value: Any) -> FrozenSet[str]:
    if isinstance(value, str):
        value = value.split(",")
    if not isinstance(value, (list, tuple, set)):
        return frozenset()
    return frozenset(str(v).strip() for v in value if str(v).strip())


def _alert_channel(value: Any, warnings: List[str]) -> Optional[str]:
    text = str(value or "").strip()
    if not text:
        return None
    if text.lower() in _NON_TARGETS:
        warnings.append(f"alert_channel: {text!r} is not a place alerts can be sent — alerts are "
                        "only logged until a channel is set (e.g. slack:C0123ABCD)")
        return None
    # A bare Slack channel id is the common case; anything else is already a
    # deliver target ("slack:C…", "telegram:-100…", "slack" = home channel).
    return f"slack:{text}" if _SLACK_CHANNEL_ID_RE.match(text) else text


def _alert_at(raw: Any, *, has_period_budget: bool, warnings: List[str]) -> Tuple[float, ...]:
    items = raw if isinstance(raw, (list, tuple)) else [raw]
    parsed = set()
    for value in items:
        t = _positive_float(value, "alert_at", warnings)
        if t is None:
            continue
        if t > MAX_THRESHOLD:
            warnings.append(f"alert_at: {value!r} is a fraction of the budget (0.8 = 80%), not a "
                            f"percentage — ignored; did you mean {t / 100:g}?")
            continue
        parsed.add(t)
    if has_period_budget:
        # 1.0 is the over-budget alert. It must always exist: jobs are skipped
        # at 100% whatever the thresholds say, and a skip is never silent.
        parsed.add(1.0)
    return tuple(sorted(parsed))


def load_budget_config(raw: Optional[dict] = None) -> Optional[BudgetConfig]:
    """The profile's ``cost_budget:`` block, or None when absent / inert.

    ``raw`` is the whole config dict; by default the current profile's
    config.yaml (cached raw read — cheap enough for every cron tick). Bad
    values are dropped, logged once per home, and listed in ``warnings``.
    """
    if raw is None:
        from hermes_cli.config import read_raw_config_readonly

        raw = read_raw_config_readonly() or {}
    block = raw.get("cost_budget") if isinstance(raw, dict) else None
    if not isinstance(block, dict):
        return None
    warnings: List[str] = []
    daily = _positive_float(block.get("daily_usd"), "daily_usd", warnings)
    monthly = _positive_float(block.get("monthly_usd"), "monthly_usd", warnings)
    per_run = _positive_float(block.get("per_run_usd"), "per_run_usd", warnings)
    if daily is None and monthly is None and per_run is None:
        for w in warnings:
            _warn_once(w)
        return None
    has_period = daily is not None or monthly is not None
    alert_at = _alert_at(block.get("alert_at", list(DEFAULT_ALERT_AT)),
                         has_period_budget=has_period, warnings=warnings)
    try:
        cache_seconds = int(block.get("cache_seconds", DEFAULT_CACHE_SECONDS))
    except (TypeError, ValueError):
        cache_seconds = DEFAULT_CACHE_SECONDS
    mention = str(block.get("alert_mention") or "").strip() or None
    channel = _alert_channel(block.get("alert_channel"), warnings)
    for w in warnings:
        _warn_once(w)
    return BudgetConfig(
        daily_usd=daily, monthly_usd=monthly, alert_at=alert_at, per_run_usd=per_run,
        alert_channel=channel, alert_mention=mention,
        discretionary_jobs=_str_set(block.get("discretionary_jobs")),
        cache_seconds=max(MIN_CACHE_SECONDS, cache_seconds), warnings=tuple(warnings),
        default_discretionary=block.get("default_discretionary") is True,
        essential_jobs=_str_set(block.get("essential_jobs")),
    )


def is_discretionary(job: dict, cfg: BudgetConfig) -> bool:
    """Whether the budget may skip this job.

    An id in ``essential_jobs`` is never discretionary. Under
    ``default_discretionary`` every other job is. Otherwise a job is
    discretionary by its own ``discretionary: true`` field or its id in
    ``discretionary_jobs``. Ids only: a name is free text, so a renamed or
    look-alike job would silently change budget class.
    """
    job_id = str(job.get("id") or "")
    if job_id in cfg.essential_jobs:
        return False
    if cfg.default_discretionary:
        return True
    return job.get("discretionary") is True or job_id in cfg.discretionary_jobs


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
    day_upper_bound: bool = False
    month_upper_bound: bool = False
    tracking_since: float = 0.0

    @property
    def spend_is_lower_bound(self) -> bool:
        """Some tokens this month have no price, so real spend is higher."""
        return self.month_unpriced_tokens > 0

    @property
    def spend_is_upper_bound(self) -> bool:
        """Cold-start figures count whole sessions, so real spend may be lower."""
        return self.day_upper_bound or self.month_upper_bound


class SpendUnavailable(RuntimeError):
    """The spend scan failed recently; not retried until ``ERROR_CACHE_SECONDS`` pass."""


def _period_starts(now: datetime) -> Tuple[datetime, datetime]:
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return day_start, day_start.replace(day=1)


def _keys(now: datetime) -> Tuple[str, str]:
    return now.strftime("%Y-%m-%d"), now.strftime("%Y-%m")


def _day_of(ts: float, now: datetime) -> str:
    return datetime.fromtimestamp(ts, tz=now.tzinfo).strftime("%Y-%m-%d")


_USAGE_SQL = """
SELECT u.session_id, u.model, u.billing_provider, u.billing_base_url, u.billing_mode, u.task,
       u.input_tokens, u.output_tokens, u.estimated_cost_usd, u.actual_cost_usd, u.cost_status,
       COALESCE(u.first_seen, s.started_at) AS first_seen,
       COALESCE(u.last_seen, u.first_seen, s.started_at) AS last_seen
  FROM session_model_usage u JOIN sessions s ON s.id = u.session_id
 WHERE COALESCE(u.last_seen, u.first_seen, s.started_at) >= ?
"""


def _state_dir(home: Path) -> Path:
    return Path(home) / "cost_budget"


_file_locks: Dict[str, threading.RLock] = {}
_file_locks_guard = threading.Lock()


@contextlib.contextmanager
def _locked(path: Path) -> Iterator[None]:
    """In-process + cross-process (flock) lock for one state file."""
    with _file_locks_guard:
        lock = _file_locks.setdefault(str(path), threading.RLock())
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        if fcntl is None:
            yield
            return
        with open(path.with_suffix(path.suffix + ".lock"), "a", encoding="utf-8") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _read_json(path: Path) -> Optional[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        logger.warning("cost_budget: unreadable state %s — starting fresh", path)
        return None
    return data if isinstance(data, dict) else None


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=1, sort_keys=True)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _row_values(row) -> Tuple[float, int, int, bool]:
    """(cumulative cost, unpriced tokens, tokens, priced) for one usage row."""
    from agent.cost_attribution import PRICED_STATUSES, _model_usage_cost

    tokens = int(row["input_tokens"] or 0) + int(row["output_tokens"] or 0)
    priced = row["cost_status"] in PRICED_STATUSES or float(row["estimated_cost_usd"] or 0) > 0
    return (_model_usage_cost(row) if priced else 0.0), (0 if priced else tokens), tokens, priced


def _apply_scan(ledger: Optional[dict], rows, now: datetime, scanned_at: float,
                book_from: float) -> dict:
    """Book each usage row's growth since the last scan to its ``last_seen`` day.

    On a cold start, rows last seen before ``book_from`` (the month start) are
    only recorded as a baseline, so a long-lived session that spends later is
    booked by its delta, not its whole history.
    """
    cold = ledger is None
    ledger = ledger or {"since": scanned_at, "rows": {}, "days": {}}
    since = float(ledger.get("since") or scanned_at)
    stored: Dict[str, list] = ledger.setdefault("rows", {})
    days: Dict[str, dict] = ledger.setdefault("days", {})
    for row in rows:
        key = "|".join(str(row[c] or "") for c in ("session_id", "model", "billing_provider",
                                                   "billing_base_url", "billing_mode", "task"))
        cost, unpriced, tokens, priced = _row_values(row)
        last_seen = float(row["last_seen"])
        day = _day_of(last_seen, now)
        prev = stored.get(key)
        upper = False
        if prev is None and cold and last_seen < book_from:
            stored[key] = [cost, unpriced, tokens, last_seen]
            continue
        if prev is None:
            d_cost, d_unpriced = cost, unpriced
            # The whole cumulative lands on one day. Exact when the row began
            # that day (or after tracking began); otherwise it over-counts —
            # a cold start, or a session idle longer than the retention.
            first_seen = float(row["first_seen"] or last_seen)
            upper = _day_of(first_seen, now) != day and (cold or first_seen < since)
        else:
            d_cost, d_unpriced = cost - float(prev[0]), unpriced - int(prev[1])
        if d_cost or d_unpriced or upper:
            bucket = days.setdefault(day, {"usd": 0.0, "unpriced": 0, "models": [], "upper": False})
            bucket["usd"] = max(0.0, float(bucket["usd"]) + d_cost)
            bucket["unpriced"] = max(0, int(bucket["unpriced"]) + d_unpriced)
            bucket["upper"] = bool(bucket.get("upper")) or upper
            if not priced and row["model"] and row["model"] not in bucket["models"]:
                bucket["models"].append(row["model"])
        stored[key] = [cost, unpriced, tokens, last_seen]
    cutoff = scanned_at - RETENTION_SECONDS
    ledger["rows"] = {k: v for k, v in stored.items() if float(v[3]) >= cutoff}
    oldest_day = _day_of(cutoff, now)
    ledger["days"] = {d: v for d, v in days.items() if d >= oldest_day}
    ledger["scanned_at"] = scanned_at
    return ledger


def _cron_runs(db, since: float, until: float) -> Tuple[RunSpend, ...]:
    from agent.cost_attribution import attribute_sessions

    # Cron runs are always new root sessions, so windowing on started_at is
    # exact for them. No job/target resolution: only the job id is needed.
    sessions = attribute_sessions(db, since=since, until=until,
                                  job_resolver=lambda _job_id: None,
                                  target_resolver=lambda _job: [])
    runs: Dict[str, Dict[str, Any]] = {}
    for s in sessions:
        if not s.job_id:
            continue
        run = runs.setdefault(s.root_id, {"job_id": s.job_id, "started_at": s.started_at,
                                          "cost_usd": 0.0, "unpriced_tokens": 0})
        run["started_at"] = min(run["started_at"], s.started_at)
        run["cost_usd"] += s.cost_usd
        run["unpriced_tokens"] += s.unpriced_tokens
    return tuple(RunSpend(root_id=rid, **vals) for rid, vals in sorted(runs.items()))


def compute_spend(home: Path, now: datetime) -> SpendSnapshot:
    """Scan ``home``'s store, update its spend ledger, and total the day and month of ``now``."""
    home = Path(home)
    day_key, month_key = _keys(now)
    _day_start, month_start = _period_starts(now)
    db_path = home / "state.db"
    ledger_path = _state_dir(home) / "spend.json"
    runs: Tuple[RunSpend, ...] = ()
    with _locked(ledger_path):
        ledger = _read_json(ledger_path)
        scanned_at = now.timestamp()
        if db_path.exists():
            from hermes_state import SessionDB

            cutoff = (scanned_at - RETENTION_SECONDS if ledger is None
                      else float(ledger.get("scanned_at") or 0) - SCAN_OVERLAP_SECONDS)
            db = SessionDB(db_path=db_path, read_only=True)
            try:
                with db._read_ctx() as conn:
                    conn.execute("PRAGMA busy_timeout=5000")
                    rows = conn.execute(_USAGE_SQL, (cutoff,)).fetchall()
                ledger = _apply_scan(ledger, rows, now, scanned_at, month_start.timestamp())
                # Per-run alerts only cover runs since tracking began, so the
                # first enable does not replay a month of history.
                since = max(month_start.timestamp(), float(ledger["since"]))
                runs = _cron_runs(db, since, now.timestamp() + 1)
            finally:
                db.close()
        else:
            ledger = _apply_scan(ledger, [], now, scanned_at, month_start.timestamp())
        _write_json(ledger_path, ledger)

    day_usd = month_usd = 0.0
    day_unpriced = month_unpriced = 0
    day_upper = month_upper = False
    models = set()
    for key, bucket in ledger["days"].items():
        if not key.startswith(month_key):
            continue
        month_usd += float(bucket["usd"])
        month_unpriced += int(bucket["unpriced"])
        month_upper = month_upper or bool(bucket.get("upper"))
        if bucket["unpriced"]:
            models.update(bucket.get("models") or [])
        if key == day_key:
            day_usd, day_unpriced = float(bucket["usd"]), int(bucket["unpriced"])
            day_upper = bool(bucket.get("upper"))
    return SpendSnapshot(
        day_key=day_key, month_key=month_key, day_usd=day_usd, month_usd=month_usd,
        day_unpriced_tokens=day_unpriced, month_unpriced_tokens=month_unpriced,
        unpriced_models=tuple(sorted(models)), runs=runs, as_of=now.timestamp(),
        day_upper_bound=day_upper, month_upper_bound=month_upper,
        tracking_since=float(ledger["since"]),
    )


_cache_lock = threading.Lock()
_spend_cache: Dict[str, Tuple[float, Any]] = {}


def _home_key() -> str:
    return str(get_hermes_home().resolve())


def _now() -> datetime:
    from hermes_time import now as hermes_now

    return hermes_now()


def get_spend(cfg: BudgetConfig, *, now: Optional[datetime] = None) -> SpendSnapshot:
    """The current profile's spend, cached per home for ``cfg.cache_seconds``.

    A failed scan is remembered for ``ERROR_CACHE_SECONDS`` and re-raised as
    ``SpendUnavailable`` without touching the store again, so a locked or
    broken store is not re-queried on every preflight.
    """
    now = now or _now()
    key = _home_key()
    with _cache_lock:
        cached = _spend_cache.get(key)
    if cached is not None:
        stamp, value = cached
        age = time.monotonic() - stamp
        if isinstance(value, SpendUnavailable):
            if age < ERROR_CACHE_SECONDS:
                raise value
        elif age < cfg.cache_seconds and (value.day_key, value.month_key) == _keys(now):
            return value
    try:
        snap = compute_spend(Path(key), now)
    except Exception as exc:
        err = SpendUnavailable(f"{type(exc).__name__}: {exc}")
        with _cache_lock:
            _spend_cache[key] = (time.monotonic(), err)
        raise err from exc
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
    # Fresh alerts of one group go out as a single message.
    group: Optional[str] = None


@dataclass(frozen=True)
class Trigger:
    """The cron job whose preflight found the profile over budget."""
    name: str
    skipped: bool


def _fmt_threshold(t: float) -> str:
    return f"{t:g}"


def _money(value: float) -> str:
    return f"${value:,.2f}"


def _threshold_alert(status: BudgetStatus, *, period: str,
                     trigger: Optional[Trigger] = None) -> Optional[Alert]:
    cfg, snap = status.cfg, status.snap
    if period == "daily":
        fraction, spent, budget, key = status.day_fraction, snap.day_usd, cfg.daily_usd, snap.day_key
        label, resets, unpriced, upper = "Daily", "tomorrow", snap.day_unpriced_tokens, snap.day_upper_bound
    else:
        fraction, spent, budget, key = status.month_fraction, snap.month_usd, cfg.monthly_usd, snap.month_key
        label, resets, unpriced, upper = "Monthly", "next month", snap.month_unpriced_tokens, snap.month_upper_bound
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
    if upper:
        text += (" Part of this figure counts whole sessions that started before tracking "
                 "began, so the real figure may be lower.")
    loud = top >= 1.0
    if loud:
        text = (f"🚨 Over budget. {text} Discretionary jobs are skipped until {resets}; "
                "release (non-discretionary) work keeps running.")
        if trigger is not None:
            verb = "was skipped" if trigger.skipped else "is running anyway (not discretionary)"
            text += f" First noticed when job `{trigger.name}` {verb}."
    return Alert(key=keys[-1], text=text, loud=loud, covers=tuple(keys[:-1]))


def pending_alerts(status: BudgetStatus, *, trigger: Optional[Trigger] = None) -> List[Alert]:
    """Every alert the current spend warrants; ``send_once`` drops those already sent."""
    cfg, snap = status.cfg, status.snap
    alerts = [a for a in (_threshold_alert(status, period="daily", trigger=trigger),
                          _threshold_alert(status, period="monthly", trigger=trigger)) if a]
    if cfg.per_run_usd:
        for run in snap.runs:
            if run.cost_usd >= cfg.per_run_usd and run.started_at >= snap.tracking_since:
                alerts.append(Alert(
                    key=f"run:{run.root_id}", group="runs",
                    text=f"job `{run.job_id}` — {_money(run.cost_usd)} (session `{run.root_id}`)",
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


def _merge_groups(alerts: List[Alert], cfg: BudgetConfig) -> List[Tuple[Alert, Tuple[str, ...]]]:
    """[(alert to send, every key it accounts for)]; a group becomes one message."""
    out: List[Tuple[Alert, Tuple[str, ...]]] = []
    runs = [a for a in alerts if a.group == "runs"]
    for a in alerts:
        if a.group is None:
            out.append((a, (a.key, *a.covers)))
    if runs:
        lines = [f"• {a.text}" for a in runs[:MAX_RUNS_LISTED]]
        if len(runs) > MAX_RUNS_LISTED:
            lines.append(f"• …and {len(runs) - MAX_RUNS_LISTED} more")
        noun = "run" if len(runs) == 1 else "runs"
        text = (f"{len(runs)} cron {noun} went over the {_money(cfg.per_run_usd or 0)} "
                "per-run limit:\n" + "\n".join(lines))
        out.append((Alert(key=runs[0].key, text=text), tuple(a.key for a in runs)))
    return out


def _alerts_path() -> Path:
    return _state_dir(get_hermes_home()) / "alerts.json"


def _load_sent(path: Path) -> Dict[str, float]:
    data = _read_json(path) or {}
    sent = data.get("sent")
    return {str(k): float(v) for k, v in sent.items()} if isinstance(sent, dict) else {}


def _save_sent(path: Path, sent: Dict[str, float]) -> None:
    cutoff = time.time() - RETENTION_SECONDS
    _write_json(path, {"sent": {k: v for k, v in sent.items() if v >= cutoff}})


def _deliver(target: str, text: str, *, adapters=None, loop=None) -> Optional[str]:
    """Send through the cron delivery path; None only when a target received it."""
    from agent.secret_scope import build_profile_secret_scope, reset_secret_scope, set_secret_scope
    from cron import scheduler

    pseudo_job = {"id": "cost_budget", "name": "Cost budget", "deliver": target}
    token = set_secret_scope(build_profile_secret_scope(get_hermes_home()))
    try:
        # _deliver_result returns None for "nothing to deliver to" as well as
        # for success; an alert that went nowhere must not be marked sent.
        if not scheduler._resolve_delivery_targets(pseudo_job):
            return f"alert_channel {target!r} resolves to no delivery target"
        return scheduler._deliver_result(pseudo_job, text, adapters=adapters, loop=loop)
    except Exception as exc:  # an adapter error must not escape into cron
        return f"{type(exc).__name__}: {exc}"
    finally:
        reset_secret_scope(token)


_logged_without_channel: set = set()


def _claim(alerts: List[Alert], cfg: BudgetConfig) -> List[Tuple[Alert, Tuple[str, ...], float]]:
    """Mark fresh alerts sent (under the lock) and return what to deliver."""
    path = _alerts_path()
    with _locked(path):
        sent = _load_sent(path)
        fresh = [a for a in alerts if a.key not in sent]
        if not fresh:
            return []
        if not cfg.alert_channel:
            for alert in fresh:
                marker = (str(path), alert.key)
                if marker not in _logged_without_channel:
                    _logged_without_channel.add(marker)
                    logger.warning("cost_budget alert (no usable cost_budget.alert_channel): %s",
                                   alert.text)
            return []
        stamp = time.time()
        claimed = []
        for alert, keys in _merge_groups(fresh, cfg):
            for key in keys:
                sent[key] = stamp
            claimed.append((alert, keys, stamp))
        _save_sent(path, sent)
        return claimed


def _release(keys: Tuple[str, ...], stamp: float) -> None:
    path = _alerts_path()
    with _locked(path):
        sent = _load_sent(path)
        for key in keys:
            if sent.get(key) == stamp:
                sent.pop(key)
        _save_sent(path, sent)


def send_once(alerts: List[Alert], cfg: BudgetConfig, *, adapters=None, loop=None) -> int:
    """Deliver each alert whose key this profile never sent; return how many messages went out.

    Keys are claimed under the state lock and delivered after it is released;
    a failed delivery releases its claim so a later check retries it.
    """
    if not alerts:
        return 0
    delivered = 0
    for alert, keys, stamp in _claim(alerts, cfg):
        text = f"{cfg.alert_mention} {alert.text}" if alert.loud and cfg.alert_mention else alert.text
        error = _deliver(cfg.alert_channel or "", text, adapters=adapters, loop=loop)
        if error is None:
            delivered += 1
            continue
        logger.warning("cost_budget: alert %s not delivered to %s: %s",
                       alert.key, cfg.alert_channel, error)
        _release(keys, stamp)
    return delivered


_send_threads: List[threading.Thread] = []
_send_threads_lock = threading.Lock()


def _send_in_background(alerts: List[Alert], cfg: BudgetConfig, *, adapters=None,
                        loop=None) -> Optional[threading.Thread]:
    """``send_once`` on a daemon thread that carries the caller's context.

    The context copy is what keeps the profile's home override (and so its
    state file and secret scope) on the sending thread; the adapters are
    passed through. Nothing here holds the tick lock across the network.
    """
    if not alerts:
        return None
    ctx = contextvars.copy_context()

    def _run() -> None:
        try:
            send_once(alerts, cfg, adapters=adapters, loop=loop)
        except Exception:
            logger.warning("cost_budget: alert send failed", exc_info=True)

    thread = threading.Thread(target=ctx.run, args=(_run,), name="cost-budget-alert", daemon=True)
    with _send_threads_lock:
        _send_threads[:] = [t for t in _send_threads if t.is_alive()]
        _send_threads.append(thread)
    thread.start()
    return thread


_watch_lock = threading.Lock()
_last_watch: Dict[str, float] = {}


def watch(*, adapters=None, loop=None) -> Optional[threading.Thread]:
    """Check this profile's spend and send due alerts. Called from every cron tick.

    Returns after the cached config read when the profile has no
    ``cost_budget:`` block; otherwise throttled to ``cache_seconds`` per home.
    Alerts are sent on a background thread (returned, for tests). Never raises.
    """
    try:
        cfg = load_budget_config()
        if cfg is None:
            return None
        key = _home_key()
        now_m = time.monotonic()
        with _watch_lock:
            last = _last_watch.get(key)
            if last is not None and now_m - last < cfg.cache_seconds:
                return None
            _last_watch[key] = now_m
        status = budget_status(cfg, get_spend(cfg))
        return _send_in_background(pending_alerts(status), cfg, adapters=adapters, loop=loop)
    except SpendUnavailable as exc:
        logger.warning("cost_budget: spend unavailable: %s", exc)
    except Exception:
        logger.warning("cost_budget: budget watch failed", exc_info=True)
    return None


# ── enforcement ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Skip:
    reason: str
    resets_at: datetime


def preflight_cron_job(job: dict, *, adapters=None, loop=None) -> Optional[Skip]:
    """Called by the cron scheduler before a job's script or agent runs.

    Returns a ``Skip`` for a discretionary job while the profile is over its
    daily or monthly budget. Whenever the profile is over budget — skip or
    not — the (deduplicated) over-budget alert is sent, so a skip is never
    silent. None in every other case, including no ``cost_budget:`` block and
    any internal error (fail open: a budget bug must never stop release work).
    """
    try:
        cfg = load_budget_config()
        if cfg is None:
            return None
        now = _now()
        status = budget_status(cfg, get_spend(cfg, now=now))
        if not status.over_budget:
            return None
        snap = status.snap
        day_start, month_start = _period_starts(now)
        next_month = (month_start + timedelta(days=32)).replace(day=1)
        if status.over_daily:
            which, spent, budget = "daily", snap.day_usd, cfg.daily_usd
        else:
            which, spent, budget = "monthly", snap.month_usd, cfg.monthly_usd
        # A one-shot that is skipped re-arms when every exceeded budget has reset.
        resets = next_month if status.over_monthly else day_start + timedelta(days=1)
        name = str(job.get("name") or job.get("id") or "")
        skipped = is_discretionary(job, cfg)
        over_alerts = [a for a in pending_alerts(status, trigger=Trigger(name=name, skipped=skipped))
                       if a.loud]
        _send_in_background(over_alerts, cfg, adapters=adapters, loop=loop)
        if not skipped:
            return None
        reason = f"skipped: over budget ({which} {_money(spent)} of {_money(budget)})"
        logger.info("cost_budget: job '%s' %s", name, reason)
        return Skip(reason=reason, resets_at=resets)
    except SpendUnavailable as exc:
        logger.warning("cost_budget: preflight for job %s — spend unavailable (%s); running it",
                       job.get("id"), exc)
    except Exception:
        logger.warning("cost_budget: preflight for job %s failed — running it",
                       job.get("id"), exc_info=True)
    return None


def _wait_for_sends(timeout: float = 5.0) -> None:
    """Tests: join the background alert threads started so far."""
    with _send_threads_lock:
        threads = list(_send_threads)
    for t in threads:
        t.join(timeout)


def _reset_for_tests() -> None:
    _wait_for_sends()
    with _cache_lock:
        _spend_cache.clear()
    with _watch_lock:
        _last_watch.clear()
    _logged_without_channel.clear()
    _warned.clear()

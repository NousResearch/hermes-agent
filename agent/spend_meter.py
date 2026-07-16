"""Local spend metering, daily budgets, and dispatch throttling for the fleet.

Aggregates per-session token counts from the root and per-profile ``state.db``
files into USD cost (via ``agent.usage_pricing``), maintains a daily ledger
with watermark-delta accrual, evaluates budget thresholds, and exposes a tiny
throttle flag file the kanban dispatcher reads each tick.

Billing lanes: every profile maps to a lane — ``api_key`` (shared Anthropic
API key pinned via ANTHROPIC_TOKEN) or ``personal_oauth`` (keychain OAuth,
subscription-billed; costs are *estimated equivalents*, not real invoices).

Writers: only the spend poller (``scripts/spend_guard.py``) writes the ledger
and (together with the manual-override CLI) the throttle file. The gateway
imports only ``read_throttle``/``is_profile_paused``/``load_spend_config``.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

LEDGER_VERSION = 1
WATERMARK_RETENTION_SECONDS = 48 * 3600
HISTORY_DAYS = 30

_DEFAULT_LANES = {
    "api_key": {"daily_cap_usd": 10.0, "label": "shared API key"},
    "personal_oauth": {
        "daily_cap_usd": 5.0,
        "label": "personal OAuth — est. equivalent (subscription)",
    },
}


def hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")))


def default_ledger_path() -> Path:
    return hermes_home() / "data" / "spend_ledger.json"


def default_throttle_path() -> Path:
    return hermes_home() / "data" / "spend_throttle.json"


# ─── Config ──────────────────────────────────────────────────────────────────


@dataclass
class SpendConfig:
    enabled: bool = True
    timezone: str = "America/Montevideo"
    lanes: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {k: dict(v) for k, v in _DEFAULT_LANES.items()}
    )
    lane_overrides: Dict[str, str] = field(default_factory=dict)
    profile_caps: Dict[str, float] = field(default_factory=dict)
    exempt_profiles: List[str] = field(default_factory=lambda: ["pm"])
    slack_channel: str = ""
    thresholds: List[float] = field(default_factory=lambda: [0.5, 0.8, 1.0])
    throttle_enabled: bool = False

    def lane_cap(self, lane: str) -> Optional[Decimal]:
        cap = (self.lanes.get(lane) or {}).get("daily_cap_usd")
        if cap is None:
            return None
        return Decimal(str(cap))

    def lane_label(self, lane: str) -> str:
        return (self.lanes.get(lane) or {}).get("label") or lane


_config_cache: Dict[str, Any] = {}


def load_spend_config(cfg: Optional[dict] = None) -> SpendConfig:
    """Parse the ``spend:`` section of ``~/.hermes/config.yaml`` (mtime-cached)."""
    if cfg is None:
        path = hermes_home() / "config.yaml"
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return SpendConfig(enabled=False)
        cached = _config_cache.get(str(path))
        if cached and cached[0] == mtime:
            return cached[1]
        import yaml

        try:
            with open(path, encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh) or {}
        except Exception:
            return SpendConfig(enabled=False)
        parsed = _parse_spend_section(cfg.get("spend") or {})
        _config_cache[str(path)] = (mtime, parsed)
        return parsed
    return _parse_spend_section(cfg.get("spend") or {})


def _parse_spend_section(spend: dict) -> SpendConfig:
    out = SpendConfig()
    if not spend:
        out.enabled = False
        return out
    out.enabled = bool(spend.get("enabled", True))
    out.timezone = str(spend.get("timezone") or out.timezone)
    lanes = spend.get("lanes") or {}
    if lanes:
        out.lanes = {
            str(name): dict(vals or {}) for name, vals in lanes.items()
        }
    out.lane_overrides = {
        str(k): str(v) for k, v in (spend.get("lane_overrides") or {}).items()
    }
    out.profile_caps = {
        str(k): float(v) for k, v in (spend.get("profile_caps") or {}).items()
    }
    exempt = spend.get("exempt_profiles")
    if exempt is not None:
        out.exempt_profiles = [str(p) for p in exempt]
    alerts = spend.get("alerts") or {}
    out.slack_channel = str(alerts.get("slack_channel") or "")
    thresholds = alerts.get("thresholds")
    if thresholds:
        out.thresholds = sorted(float(t) for t in thresholds)
    throttle = spend.get("throttle") or {}
    out.throttle_enabled = bool(throttle.get("enabled", False))
    return out


# ─── Time windows ────────────────────────────────────────────────────────────


def day_window(now: float, tz_name: str) -> Tuple[float, float, str]:
    """Return (start_epoch, end_epoch, YYYY-MM-DD) for the local day of *now*."""
    tz = ZoneInfo(tz_name)
    local = datetime.fromtimestamp(now, tz)
    start = local.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    return start.timestamp(), end.timestamp(), start.strftime("%Y-%m-%d")


# ─── Lane resolution ─────────────────────────────────────────────────────────


def _read_env_value(path: Path, key: str) -> Optional[str]:
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line.startswith(f"{key}=") and not line.startswith("#"):
                    return line.split("=", 1)[1].strip().strip("'\"")
    except OSError:
        return None
    return None


def _looks_like_oauth(token: str) -> bool:
    try:
        from agent.anthropic_adapter import _is_oauth_token

        return _is_oauth_token(token)
    except Exception:
        return not token.startswith("sk-ant-api")


_lane_cache: Dict[str, str] = {}


def resolve_profile_lane(profile: str, cfg: SpendConfig) -> str:
    """Map a profile to its billing lane (config override first, then .env)."""
    override = cfg.lane_overrides.get(profile)
    if override:
        return override
    cached = _lane_cache.get(profile)
    if cached:
        return cached
    if profile == "default":
        env_path = hermes_home() / ".env"
    else:
        env_path = hermes_home() / "profiles" / profile / ".env"
    token = _read_env_value(env_path, "ANTHROPIC_TOKEN")
    if token and not _looks_like_oauth(token):
        lane = "api_key"
    elif token:
        lane = "personal_oauth"
    else:
        # No pinned token → the agent resolver falls through to keychain OAuth.
        lane = "personal_oauth"
    _lane_cache[profile] = lane
    return lane


# ─── Session DBs and costing ─────────────────────────────────────────────────


def iter_session_dbs() -> List[Tuple[str, Path]]:
    """(profile, state.db path) pairs — root DB is profile ``default``."""
    home = hermes_home()
    out: List[Tuple[str, Path]] = []
    root = home / "state.db"
    if root.exists():
        out.append(("default", root))
    profiles_dir = home / "profiles"
    if profiles_dir.is_dir():
        for entry in sorted(profiles_dir.iterdir()):
            db = entry / "state.db"
            if entry.is_dir() and db.exists():
                out.append((entry.name, db))
    return out


def _models_dev_fallback_entry(model: str):
    """PricingEntry from raw models.dev cost data, for slugs missing from the snapshot table."""
    try:
        from agent.models_dev import fetch_models_dev
        from agent.usage_pricing import PricingEntry

        models = (fetch_models_dev().get("anthropic") or {}).get("models") or {}
        cost = (models.get(model) or {}).get("cost") or {}
        if cost.get("input") is None or cost.get("output") is None:
            return None
        return PricingEntry(
            input_cost_per_million=Decimal(str(cost["input"])),
            output_cost_per_million=Decimal(str(cost["output"])),
            cache_read_cost_per_million=Decimal(str(cost.get("cache_read", 0))),
            cache_write_cost_per_million=Decimal(str(cost.get("cache_write", 0))),
            source="provider_models_api",
            pricing_version="models-dev-fallback",
        )
    except Exception:
        return None


def cumulative_cost(row: dict) -> Tuple[Decimal, str]:
    """(cumulative USD, status) for one sessions row. Status ``pricing_gap`` means no rate found.

    ``reasoning_tokens`` is intentionally excluded: Anthropic's output_tokens
    already include thinking tokens.
    """
    from agent.usage_pricing import CanonicalUsage, estimate_usage_cost

    model = (row.get("model") or "").strip()
    usage = CanonicalUsage(
        input_tokens=int(row.get("input_tokens") or 0),
        output_tokens=int(row.get("output_tokens") or 0),
        cache_read_tokens=int(row.get("cache_read_tokens") or 0),
        cache_write_tokens=int(row.get("cache_write_tokens") or 0),
        request_count=0,
    )
    if not model or usage.total_tokens == 0:
        return Decimal("0"), "empty"
    provider = (row.get("billing_provider") or "anthropic").strip() or "anthropic"
    result = estimate_usage_cost(model, usage, provider=provider)
    if result.amount_usd is not None and result.status != "unknown":
        return result.amount_usd, result.status
    entry = _models_dev_fallback_entry(model)
    if entry is None:
        return Decimal("0"), "pricing_gap"
    million = Decimal("1000000")
    amount = (
        Decimal(usage.input_tokens) * entry.input_cost_per_million
        + Decimal(usage.output_tokens) * entry.output_cost_per_million
        + Decimal(usage.cache_read_tokens) * (entry.cache_read_cost_per_million or 0)
        + Decimal(usage.cache_write_tokens) * (entry.cache_write_cost_per_million or 0)
    ) / million
    return amount, "estimated"


# ─── Ledger ──────────────────────────────────────────────────────────────────


def empty_ledger(date_str: str, tz_name: str) -> dict:
    return {
        "version": LEDGER_VERSION,
        "date": date_str,
        "tz": tz_name,
        "lanes": {},
        "profiles": {},
        "watermarks": {},
        "alerts_sent": {},
        "pricing_gaps": [],
        "history": {},
        "last_poll": 0,
    }


def load_ledger(path: Optional[Path] = None) -> Optional[dict]:
    path = path or default_ledger_path()
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def save_ledger(ledger: dict, path: Optional[Path] = None) -> None:
    path = path or default_ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(ledger, fh, indent=1, sort_keys=True)
    os.replace(tmp, path)


def _rollover(ledger: dict, date_str: str) -> None:
    old_date = ledger.get("date")
    if old_date and (ledger.get("lanes") or ledger.get("profiles")):
        ledger.setdefault("history", {})[old_date] = {
            "lanes": ledger.get("lanes", {}),
            "profiles": ledger.get("profiles", {}),
            "pricing_gaps": ledger.get("pricing_gaps", []),
        }
        history = ledger["history"]
        for key in sorted(history)[:-HISTORY_DAYS]:
            del history[key]
    ledger["date"] = date_str
    ledger["lanes"] = {}
    ledger["profiles"] = {}
    ledger["pricing_gaps"] = []
    alerts = ledger.get("alerts_sent") or {}
    ledger["alerts_sent"] = {k: v for k, v in alerts.items() if k >= date_str}


def accrue(
    ledger: Optional[dict],
    now: float,
    cfg: SpendConfig,
    dbs: Optional[List[Tuple[str, Path]]] = None,
) -> dict:
    """Poll all session DBs and accrue watermark deltas into today's totals."""
    window_start, _, date_str = day_window(now, cfg.timezone)
    if ledger is None:
        ledger = empty_ledger(date_str, cfg.timezone)
    if ledger.get("date") != date_str:
        _rollover(ledger, date_str)

    watermarks: Dict[str, dict] = ledger.setdefault("watermarks", {})
    lanes: Dict[str, dict] = ledger.setdefault("lanes", {})
    profiles: Dict[str, dict] = ledger.setdefault("profiles", {})
    gaps = set(ledger.get("pricing_gaps") or [])
    query_floor = window_start - WATERMARK_RETENTION_SECONDS

    for profile, db_path in dbs if dbs is not None else iter_session_dbs():
        lane = resolve_profile_lane(profile, cfg)
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT id, model, billing_provider, input_tokens, output_tokens,"
                " cache_read_tokens, cache_write_tokens, started_at, ended_at"
                " FROM sessions WHERE started_at >= ? OR ended_at IS NULL OR ended_at >= ?",
                (query_floor, query_floor),
            ).fetchall()
            conn.close()
        except sqlite3.Error:
            continue
        for row in rows:
            row = dict(row)
            cost_now, status = cumulative_cost(row)
            if status == "pricing_gap":
                gaps.add(row.get("model") or "?")
                continue
            key = f"{profile}:{row['id']}"
            mark = watermarks.get(key)
            if mark is None:
                started = float(row.get("started_at") or 0)
                delta = cost_now if started >= window_start else Decimal("0")
            else:
                delta = cost_now - Decimal(mark.get("usd", "0"))
                if delta < 0:
                    delta = Decimal("0")
            watermarks[key] = {
                "usd": str(cost_now),
                "ended_at": row.get("ended_at"),
                "seen_at": now,
            }
            if delta > 0:
                lane_entry = lanes.setdefault(lane, {"usd": 0.0})
                lane_entry["usd"] = float(Decimal(str(lane_entry["usd"])) + delta)
                prof_entry = profiles.setdefault(profile, {"usd": 0.0, "lane": lane})
                prof_entry["usd"] = float(Decimal(str(prof_entry["usd"])) + delta)
                prof_entry["lane"] = lane

    for key, mark in list(watermarks.items()):
        ended = mark.get("ended_at")
        seen = float(mark.get("seen_at") or 0)
        expired = ended is not None and float(ended) < now - WATERMARK_RETENTION_SECONDS
        if expired or (seen and seen < now - 7 * 24 * 3600):
            del watermarks[key]

    ledger["pricing_gaps"] = sorted(gaps)
    ledger["last_poll"] = now
    return ledger


# ─── Thresholds and alerts ───────────────────────────────────────────────────


@dataclass
class Alert:
    target: str  # lane name or "profile:<name>"
    threshold: float
    usd: Decimal
    cap: Decimal


def evaluate_thresholds(ledger: dict, cfg: SpendConfig) -> List[Alert]:
    """New (undeduped-today) threshold crossings; marks them as sent in the ledger."""
    date_str = ledger.get("date", "")
    sent_today = ledger.setdefault("alerts_sent", {}).setdefault(date_str, {})
    alerts: List[Alert] = []

    def check(target: str, usd: Decimal, cap: Optional[Decimal]) -> None:
        if cap is None or cap <= 0:
            return
        done = sent_today.setdefault(target, [])
        for threshold in cfg.thresholds:
            if usd / cap >= Decimal(str(threshold)) and threshold not in done:
                done.append(threshold)
                alerts.append(Alert(target, threshold, usd, cap))

    for lane, entry in (ledger.get("lanes") or {}).items():
        check(lane, Decimal(str(entry.get("usd", 0))), cfg.lane_cap(lane))
    for profile, cap in cfg.profile_caps.items():
        entry = (ledger.get("profiles") or {}).get(profile)
        if entry:
            check(
                f"profile:{profile}",
                Decimal(str(entry.get("usd", 0))),
                Decimal(str(cap)),
            )
    return alerts


# ─── Throttle flag file ──────────────────────────────────────────────────────


@dataclass
class ThrottleState:
    paused_lanes: Dict[str, dict] = field(default_factory=dict)
    paused_profiles: Dict[str, dict] = field(default_factory=dict)
    overrides: Dict[str, dict] = field(default_factory=dict)


def _load_throttle_raw(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}


def _save_throttle_raw(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1, sort_keys=True)
    os.replace(tmp, path)


def _override_active(overrides: dict, target: str, action: str, now: float) -> bool:
    entry = overrides.get(target)
    if not entry or entry.get("action") != action:
        return False
    until = entry.get("until")
    return until is None or float(until) > now


def compute_and_write_throttle(
    ledger: dict,
    cfg: SpendConfig,
    path: Optional[Path] = None,
    now: Optional[float] = None,
) -> ThrottleState:
    """Derive pause flags from the ledger and persist them (preserving overrides)."""
    path = path or default_throttle_path()
    now = now if now is not None else time.time()
    raw = _load_throttle_raw(path)
    overrides = {
        target: entry
        for target, entry in (raw.get("overrides") or {}).items()
        if entry.get("until") is None or float(entry["until"]) > now
    }
    paused: Dict[str, dict] = {}
    paused_profiles: Dict[str, dict] = {}
    if cfg.enabled and cfg.throttle_enabled:
        for lane, entry in (ledger.get("lanes") or {}).items():
            cap = cfg.lane_cap(lane)
            usd = Decimal(str(entry.get("usd", 0)))
            if cap and cap > 0 and usd >= cap and not _override_active(overrides, lane, "resume", now):
                paused[lane] = {"usd": float(usd), "cap": float(cap), "since": now}
        for profile, cap in cfg.profile_caps.items():
            entry = (ledger.get("profiles") or {}).get(profile) or {}
            usd = Decimal(str(entry.get("usd", 0)))
            if usd >= Decimal(str(cap)) and not _override_active(
                overrides, profile, "resume", now
            ):
                paused_profiles[profile] = {"usd": float(usd), "cap": float(cap), "since": now}
    for target, entry in overrides.items():
        if entry.get("action") == "pause" and (
            entry.get("until") is None or float(entry["until"]) > now
        ):
            bucket = paused if target in cfg.lanes else paused_profiles
            bucket.setdefault(target, {"since": now, "manual": True})
    _save_throttle_raw(
        {"paused": paused, "paused_profiles": paused_profiles, "overrides": overrides},
        path,
    )
    return ThrottleState(paused, paused_profiles, overrides)


_throttle_cache: Dict[str, Tuple[float, ThrottleState]] = {}


def read_throttle(path: Optional[Path] = None) -> ThrottleState:
    """Cheap mtime-cached read for the dispatcher hot path."""
    path = path or default_throttle_path()
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return ThrottleState()
    cached = _throttle_cache.get(str(path))
    if cached and cached[0] == mtime:
        return cached[1]
    raw = _load_throttle_raw(path)
    state = ThrottleState(
        paused_lanes=raw.get("paused") or {},
        paused_profiles=raw.get("paused_profiles") or {},
        overrides=raw.get("overrides") or {},
    )
    _throttle_cache[str(path)] = (mtime, state)
    return state


def is_profile_paused(
    profile: str,
    throttle: ThrottleState,
    cfg: SpendConfig,
    now: Optional[float] = None,
) -> Optional[str]:
    """Reason string if kanban dispatch for *profile* should be skipped, else None."""
    if not cfg.enabled or profile in cfg.exempt_profiles:
        return None
    now = now if now is not None else time.time()
    if _override_active(throttle.overrides, profile, "resume", now):
        return None
    if profile in throttle.paused_profiles:
        info = throttle.paused_profiles[profile]
        return _pause_reason(f"profile {profile}", info)
    lane = resolve_profile_lane(profile, cfg)
    if _override_active(throttle.overrides, lane, "resume", now):
        return None
    if lane in throttle.paused_lanes:
        return _pause_reason(f"lane {lane}", throttle.paused_lanes[lane])
    return None


def _pause_reason(target: str, info: dict) -> str:
    if info.get("manual"):
        return f"{target} manually paused"
    return (
        f"{target} over daily budget "
        f"(${info.get('usd', 0):.2f}/${info.get('cap', 0):.2f})"
    )


def set_override(
    target: str,
    action: str,
    until_epoch: Optional[float],
    path: Optional[Path] = None,
) -> None:
    path = path or default_throttle_path()
    raw = _load_throttle_raw(path)
    raw.setdefault("overrides", {})[target] = {"action": action, "until": until_epoch}
    if action == "resume":
        (raw.get("paused") or {}).pop(target, None)
        (raw.get("paused_profiles") or {}).pop(target, None)
    _save_throttle_raw(raw, path)
    _throttle_cache.clear()


def clear_override(target: str, path: Optional[Path] = None) -> None:
    path = path or default_throttle_path()
    raw = _load_throttle_raw(path)
    (raw.get("overrides") or {}).pop(target, None)
    _save_throttle_raw(raw, path)
    _throttle_cache.clear()


# ─── Formatting ──────────────────────────────────────────────────────────────


def _top_profiles(ledger: dict, lane: Optional[str] = None, limit: int = 3) -> str:
    entries = [
        (name, Decimal(str(info.get("usd", 0))))
        for name, info in (ledger.get("profiles") or {}).items()
        if lane is None or info.get("lane") == lane
    ]
    entries.sort(key=lambda item: item[1], reverse=True)
    return " | ".join(f"{name} ${usd:.2f}" for name, usd in entries[:limit])


def format_alert(alerts: List[Alert], ledger: dict, cfg: SpendConfig) -> str:
    lines: List[str] = []
    for alert in alerts:
        pct = int(alert.threshold * 100)
        if alert.target.startswith("profile:"):
            name = alert.target.split(":", 1)[1]
            label = f"profile {name}"
            top = ""
        else:
            label = f"{alert.target} lane ({cfg.lane_label(alert.target)})"
            top = _top_profiles(ledger, alert.target)
        lines.append(
            f"Hermes spend {pct}%: {label} ${alert.usd:.2f} / ${alert.cap:.2f} today"
            f" (resets 00:00 {cfg.timezone})"
        )
        if top:
            lines.append(f"Top: {top}")
        if alert.threshold >= 1.0 and cfg.throttle_enabled:
            target = alert.target.split(":", 1)[-1]
            lines.append(
                "Dispatch PAUSED for this "
                + ("profile" if alert.target.startswith("profile:") else "lane's profiles")
                + " — queued kanban tasks kept, Slack chat unaffected."
            )
            lines.append(f"Override: ~/.hermes/scripts/spend.sh resume {target} --for 4h")
        elif alert.threshold >= 1.0:
            lines.append("Throttle disabled (measurement mode) — dispatch continues.")
    return "\n".join(lines)


def format_status(ledger: dict, cfg: SpendConfig, throttle: Optional[ThrottleState] = None) -> str:
    lines = [f"Hermes spend — {ledger.get('date')} ({cfg.timezone})"]
    for lane, entry in sorted((ledger.get("lanes") or {}).items()):
        usd = Decimal(str(entry.get("usd", 0)))
        cap = cfg.lane_cap(lane)
        cap_txt = f" / ${cap:.2f} ({usd / cap * 100:.0f}%)" if cap else ""
        lines.append(f"  {lane}: ${usd:.2f}{cap_txt} — {cfg.lane_label(lane)}")
    if not (ledger.get("lanes") or {}):
        lines.append("  (no spend recorded today)")
    profiles = sorted(
        (ledger.get("profiles") or {}).items(),
        key=lambda item: item[1].get("usd", 0),
        reverse=True,
    )
    if profiles:
        lines.append("  Profiles: " + " | ".join(
            f"{name} ${Decimal(str(info.get('usd', 0))):.2f}" for name, info in profiles
        ))
    if ledger.get("pricing_gaps"):
        lines.append(f"  Pricing gaps (unpriced models): {', '.join(ledger['pricing_gaps'])}")
    if throttle and (throttle.paused_lanes or throttle.paused_profiles):
        paused = list(throttle.paused_lanes) + list(throttle.paused_profiles)
        lines.append(f"  THROTTLED: {', '.join(paused)}")
    return "\n".join(lines)


def format_digest(ledger: dict, cfg: SpendConfig) -> str:
    throttle = read_throttle()
    lines = [format_status(ledger, cfg, throttle)]
    history = ledger.get("history") or {}
    if history:
        recent = sorted(history.items())[-7:]
        week: List[str] = []
        for date_str, day in recent:
            total = sum(
                Decimal(str(entry.get("usd", 0)))
                for entry in (day.get("lanes") or {}).values()
            )
            week.append(f"{date_str[5:]} ${total:.2f}")
        lines.append("  Last days: " + " | ".join(week))
    return "\n".join(lines)

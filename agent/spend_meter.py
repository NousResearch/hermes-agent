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
    # Billing routing (account-first policy): in normal operation
    # ``personal_share`` of api_key-lane workers bill the personal account
    # (subscription) and the rest bill the API key. The poller watches the
    # account's /usage windows and fails everything over to the API key at
    # ``account_limit_threshold`` utilization, returning to the split below
    # ``resume_threshold`` (hysteresis).
    personal_share: float = 0.8
    account_limit_threshold: float = 0.80
    resume_threshold: float = 0.60
    # Extra-usage credit guards: the 5h plan window only triggers failover
    # when the credit pool can't absorb the spillover; credits themselves
    # trigger failover near exhaustion.
    credit_floor_usd: float = 5.0
    credit_limit_threshold: float = 0.95

    def lane_cap(self, lane: str) -> Optional[Decimal]:
        cap = (self.lanes.get(lane) or {}).get("daily_cap_usd")
        if cap is None:
            return None
        return Decimal(str(cap))

    def lane_label(self, lane: str) -> str:
        return (self.lanes.get(lane) or {}).get("label") or lane

    def lane_swap_target(self, lane: str) -> Optional[str]:
        """Failover lane when *lane* exhausts its daily cap (``swap_to``).

        When set, hitting 100% swaps billing to the target lane (the auth
        resolver skips the pinned API key and falls through to keychain
        OAuth) instead of pausing dispatch. Dispatch pauses only when the
        target lane is itself exhausted.
        """
        target = (self.lanes.get(lane) or {}).get("swap_to")
        if target and target != lane and target in self.lanes:
            return str(target)
        return None


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
    routing = spend.get("routing") or {}
    if routing.get("personal_share") is not None:
        out.personal_share = min(1.0, max(0.0, float(routing["personal_share"])))
    if routing.get("account_limit_threshold") is not None:
        out.account_limit_threshold = float(routing["account_limit_threshold"])
    if routing.get("resume_threshold") is not None:
        out.resume_threshold = float(routing["resume_threshold"])
    if routing.get("credit_floor_usd") is not None:
        out.credit_floor_usd = float(routing["credit_floor_usd"])
    if routing.get("credit_limit_threshold") is not None:
        out.credit_limit_threshold = float(routing["credit_limit_threshold"])
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


# ─── Personal account /usage windows ─────────────────────────────────────────


def fetch_personal_account_usage() -> Optional[dict]:
    """Query the Anthropic OAuth /usage API for the personal account's
    rate-limit windows (the same data Claude Code's /usage shows).

    Resolves the keychain OAuth token directly (never the API key — the
    windows belong to the subscription). Returns
    ``{"windows": {five_hour: {"pct": .., "resets_at": ..}, ...},
    "max_pct": .., "fetched_at": ..}`` or None on any failure.
    """
    try:
        import httpx

        from agent.anthropic_adapter import (
            _resolve_claude_code_token_from_credentials,
            read_claude_code_credentials,
        )

        token = _resolve_claude_code_token_from_credentials(
            read_claude_code_credentials()
        )
        if not token:
            return None
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "anthropic-beta": "oauth-2025-04-20",
            "User-Agent": "claude-code/2.1.0",
        }
        with httpx.Client(timeout=15.0) as client:
            response = client.get(
                "https://api.anthropic.com/api/oauth/usage", headers=headers
            )
            response.raise_for_status()
        payload = response.json() or {}
        windows: Dict[str, dict] = {}
        five_hour_pct = 0.0
        weekly_pct = 0.0
        for key in ("five_hour", "seven_day", "seven_day_opus", "seven_day_sonnet"):
            window = payload.get(key) or {}
            util = window.get("utilization")
            if util is None:
                continue
            pct = float(util) * 100 if float(util) <= 1 else float(util)
            windows[key] = {"pct": round(pct, 1), "resets_at": window.get("resets_at")}
            if key == "five_hour":
                five_hour_pct = pct
            elif key != "seven_day_opus":
                # Opus week doesn't gate sonnet workers; exclude it from the
                # failover signal (rule-engineer is personal-lane regardless).
                weekly_pct = max(weekly_pct, pct)
        if not windows:
            return None
        # Extra-usage credit pool (plan overflow bills these at API-like
        # rates). amount fields are already major-unit floats here; the
        # `spend` block mirrors them in minor units.
        extra = payload.get("extra_usage") or {}
        credits: Optional[dict] = None
        if extra.get("is_enabled"):
            used = extra.get("used_credits")
            limit = extra.get("monthly_limit")
            places = int(extra.get("decimal_places") or 0)
            if isinstance(used, (int, float)) and isinstance(limit, (int, float)) and limit:
                scale = 10 ** places
                credits = {
                    "used_usd": round(float(used) / scale, 2),
                    "limit_usd": round(float(limit) / scale, 2),
                    "remaining_usd": round((float(limit) - float(used)) / scale, 2),
                    "pct": round(float(used) / float(limit) * 100, 1),
                }
        return {
            "windows": windows,
            "five_hour_pct": round(five_hour_pct, 1),
            "weekly_pct": round(weekly_pct, 1),
            # Back-compat: max over the gating windows (legacy consumers).
            "max_pct": round(max(five_hour_pct, weekly_pct), 1),
            "credits": credits,
            "fetched_at": time.time(),
        }
    except Exception:
        return None


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
    throttle_path: Optional[Path] = None,
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
    # Billing attribution: read_throttle() here reflects the PREVIOUS poll's
    # routing decision — exactly the regime the newly observed tokens ran
    # under. api_key-lane deltas split across lanes by the active mode
    # (expected-value attribution: statistically right for daily
    # aggregates, which is what the caps compare against).
    prev_routing = read_throttle(throttle_path).routing
    mode = prev_routing.get("mode") or "api_key_only"
    share = float(prev_routing.get("personal_share") or 0.0)
    if mode == "personal_only":
        api_key_splits = [("personal_oauth", Decimal("1"))]
    elif mode == "split" and share > 0:
        api_key_splits = [
            ("personal_oauth", Decimal(str(share))),
            ("api_key", Decimal("1") - Decimal(str(share))),
        ]
    else:
        api_key_splits = [("api_key", Decimal("1"))]

    def _lane_shares(env_lane: str) -> List[Tuple[str, Decimal]]:
        if env_lane == "api_key":
            return api_key_splits
        return [(env_lane, Decimal("1"))]

    for profile, db_path in dbs if dbs is not None else iter_session_dbs():
        env_lane = resolve_profile_lane(profile, cfg)
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
                for share_lane, fraction in _lane_shares(env_lane):
                    part = delta * fraction
                    if part <= 0:
                        continue
                    lane_entry = lanes.setdefault(share_lane, {"usd": 0.0})
                    lane_entry["usd"] = float(Decimal(str(lane_entry["usd"])) + part)
                prof_entry = profiles.setdefault(profile, {"usd": 0.0, "lane": env_lane})
                prof_entry["usd"] = float(Decimal(str(prof_entry["usd"])) + delta)
                prof_entry["lane"] = env_lane

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
    swapped_lanes: Dict[str, dict] = field(default_factory=dict)
    """Lanes billing through their ``swap_to`` target (over cap, target has
    headroom). Entry: ``{"to": lane, "usd": .., "cap": .., "since": ..}``."""
    routing: Dict[str, Any] = field(default_factory=dict)
    """Billing routing directive for the auth resolver:
    ``{"mode": "split"|"api_key_only"|"personal_only", "personal_share": ..,
    "reason": .., "since": ..}``. Missing/empty → resolver default (API key)."""


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
    account: Optional[dict] = None,
) -> ThrottleState:
    """Derive pause flags + billing routing from the ledger and persist them.

    ``account`` is the latest :func:`fetch_personal_account_usage` snapshot
    (or None when unavailable). The routing ladder, account-first:

      normal              → ``split``: personal_share of workers bill the
                             account, the rest the API key
      account windows hot → ``api_key_only`` (protect the subscription;
                             hysteresis via resume_threshold)
      api_key lane capped → ``personal_only`` (the pre-existing swap)
      both exhausted      → api_key lane pauses dispatch

    Overrides are preserved; a manual pause pops swaps as before.
    """
    path = path or default_throttle_path()
    now = now if now is not None else time.time()
    raw = _load_throttle_raw(path)
    prev_routing = raw.get("routing") or {}
    overrides = {
        target: entry
        for target, entry in (raw.get("overrides") or {}).items()
        if entry.get("until") is None or float(entry["until"]) > now
    }
    paused: Dict[str, dict] = {}
    paused_profiles: Dict[str, dict] = {}
    swapped: Dict[str, dict] = {}
    routing: Dict[str, Any] = {
        "mode": "split",
        "personal_share": cfg.personal_share,
        "reason": f"normal split {int(cfg.personal_share * 100)}/"
        f"{int(round((1 - cfg.personal_share) * 100))} account/api-key",
    }
    if cfg.enabled and cfg.throttle_enabled:

        def _lane_over_cap(lane: str) -> Optional[Tuple[Decimal, Decimal]]:
            cap = cfg.lane_cap(lane)
            usd = Decimal(str(((ledger.get("lanes") or {}).get(lane) or {}).get("usd", 0)))
            if cap and cap > 0 and usd >= cap:
                return usd, cap
            return None

        # Account exhaustion signal, credits-aware (Diego 2026-07-16): the
        # weekly windows always gate; the 5h window gates only when the
        # extra-usage credit pool can't absorb the spillover (bursts within
        # credit headroom keep riding the account); the credit pool itself
        # gates near exhaustion. Hysteresis: enter at
        # account_limit_threshold, leave below resume_threshold.
        acc = account or {}
        five = float(acc.get("five_hour_pct") or acc.get("max_pct") or 0.0)
        weekly = float(acc.get("weekly_pct") or 0.0)
        credits = acc.get("credits")
        credits_hot = bool(credits) and (
            float(credits.get("remaining_usd") or 0) <= cfg.credit_floor_usd
            or float(credits.get("pct") or 0) >= cfg.credit_limit_threshold * 100
        )
        five_gates = credits is None or credits_hot
        was_limited = prev_routing.get("mode") == "api_key_only"
        bar = (cfg.resume_threshold if was_limited else cfg.account_limit_threshold) * 100
        hot_reasons = []
        if weekly >= bar:
            hot_reasons.append(f"weekly /usage window at {weekly:.0f}%")
        if five >= bar and five_gates:
            hot_reasons.append(f"5h /usage window at {five:.0f}% with no credit headroom")
        if credits_hot:
            hot_reasons.append(
                "extra-usage credits nearly exhausted "
                f"(${float((credits or {}).get('remaining_usd') or 0):.2f} left)"
            )
        account_hot = bool(hot_reasons)
        personal_over = _lane_over_cap("personal_oauth")
        account_exhausted = account_hot or bool(personal_over)
        api_over = _lane_over_cap("api_key")

        if account_exhausted and api_over and not _override_active(
            overrides, "api_key", "resume", now
        ):
            usd, cap = api_over
            paused["api_key"] = {"usd": float(usd), "cap": float(cap), "since": now}
        if account_exhausted:
            reason = (
                "; ".join(hot_reasons)
                if account_hot
                else "personal lane over its daily est-equivalent cap"
            )
            routing = {"mode": "api_key_only", "personal_share": 0.0, "reason": reason}
        elif api_over:
            usd, cap = api_over
            if not _override_active(overrides, "api_key", "resume", now):
                target = cfg.lane_swap_target("api_key")
                if target:
                    swapped["api_key"] = {
                        "to": target,
                        "usd": float(usd),
                        "cap": float(cap),
                        "since": now,
                    }
                    routing = {
                        "mode": "personal_only",
                        "personal_share": 1.0,
                        "reason": f"api-key lane over daily cap (${float(usd):.2f}/${float(cap):.2f})",
                    }
                else:
                    paused["api_key"] = {"usd": float(usd), "cap": float(cap), "since": now}
        if account_exhausted and personal_over and not _override_active(
            overrides, "personal_oauth", "resume", now
        ):
            usd, cap = personal_over
            paused.setdefault(
                "personal_oauth", {"usd": float(usd), "cap": float(cap), "since": now}
            )
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
            swapped.pop(target, None)
    routing["since"] = (
        prev_routing.get("since", now)
        if prev_routing.get("mode") == routing["mode"]
        else now
    )
    _save_throttle_raw(
        {
            "paused": paused,
            "paused_profiles": paused_profiles,
            "overrides": overrides,
            "swapped": swapped,
            "routing": routing,
        },
        path,
    )
    return ThrottleState(paused, paused_profiles, overrides, swapped, routing)


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
        swapped_lanes=raw.get("swapped") or {},
        routing=raw.get("routing") or {},
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
    # While a lane is swapped, its profiles bill through the failover lane —
    # so the failover lane's pause state is what gates them.
    swap = throttle.swapped_lanes.get(lane)
    effective_lane = (swap or {}).get("to") or lane
    if _override_active(throttle.overrides, effective_lane, "resume", now):
        return None
    if effective_lane in throttle.paused_lanes:
        reason = _pause_reason(
            f"lane {effective_lane}", throttle.paused_lanes[effective_lane]
        )
        if swap:
            reason = f"{lane} swapped to {effective_lane}; {reason}"
        return reason
    if effective_lane != lane and lane in throttle.paused_lanes:
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
        (raw.get("swapped") or {}).pop(target, None)
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
            swap_to = (
                cfg.lane_swap_target(alert.target)
                if not alert.target.startswith("profile:")
                else None
            )
            if swap_to:
                lines.append(
                    f"Billing SWAPPED to the {swap_to} lane "
                    f"({cfg.lane_label(swap_to)}) — dispatch continues; new worker"
                    " sessions bill the personal account until the window resets"
                    f" or the {swap_to} cap is reached (then dispatch pauses)."
                )
            else:
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
    account = ledger.get("account_usage") or {}
    if account.get("windows"):
        parts = []
        labels = {
            "five_hour": "5h",
            "seven_day": "week",
            "seven_day_opus": "opus wk",
            "seven_day_sonnet": "sonnet wk",
        }
        for key, info in account["windows"].items():
            parts.append(f"{labels.get(key, key)} {info.get('pct', 0):.0f}%")
        credits = account.get("credits")
        if credits:
            parts.append(
                f"credits ${credits.get('used_usd', 0):.2f}/${credits.get('limit_usd', 0):.2f}"
                f" ({credits.get('pct', 0):.0f}%)"
            )
        lines.append("  Account /usage: " + " | ".join(parts))
    if throttle and throttle.routing:
        mode = throttle.routing.get("mode", "?")
        share = throttle.routing.get("personal_share")
        if mode == "split" and share is not None:
            desc = f"split {int(float(share) * 100)}/{int(round((1 - float(share)) * 100))} account/api-key"
        else:
            desc = mode.replace("_", "-")
        lines.append(f"  Billing routing: {desc}")
    if ledger.get("pricing_gaps"):
        lines.append(f"  Pricing gaps (unpriced models): {', '.join(ledger['pricing_gaps'])}")
    if throttle and throttle.swapped_lanes:
        swaps = [f"{lane}→{info.get('to')}" for lane, info in throttle.swapped_lanes.items()]
        lines.append(f"  BILLING SWAPPED: {', '.join(swaps)}")
    if throttle and (throttle.paused_lanes or throttle.paused_profiles):
        paused = list(throttle.paused_lanes) + list(throttle.paused_profiles)
        lines.append(f"  THROTTLED: {', '.join(paused)}")
    return "\n".join(lines)


def format_routing_transition(
    old_mode: Optional[str], routing: dict, account: Optional[dict]
) -> str:
    """One-line Slack notification for a billing-routing switch."""
    mode = routing.get("mode", "?")
    share = float(routing.get("personal_share") or 0.0)
    if mode == "split":
        headline = (
            f"Billing routing → split {int(share * 100)}/{int(round((1 - share) * 100))}"
            " (account/api-key)"
        )
    elif mode == "api_key_only":
        headline = "Billing routing → API KEY ONLY (protecting the personal account)"
    elif mode == "personal_only":
        headline = "Billing routing → PERSONAL ACCOUNT ONLY (api-key budget exhausted)"
    else:
        headline = f"Billing routing → {mode}"
    reason = routing.get("reason")
    parts = [headline + (f" — {reason}" if reason else "")]
    windows = (account or {}).get("windows") or {}
    if windows:
        labels = {"five_hour": "5h", "seven_day": "week", "seven_day_sonnet": "sonnet wk"}
        util = " | ".join(
            f"{labels[k]} {v.get('pct', 0):.0f}%" for k, v in windows.items() if k in labels
        )
        credits = (account or {}).get("credits")
        if credits:
            util += (
                f" | credits ${credits.get('remaining_usd', 0):.2f} left"
                f" of ${credits.get('limit_usd', 0):.2f}"
            )
        parts.append(f"Account /usage: {util}")
    if old_mode:
        parts.append(f"(was: {old_mode.replace('_', '-')})")
    return "\n".join(parts)


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

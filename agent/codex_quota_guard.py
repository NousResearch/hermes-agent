"""Daily usage guard for ChatGPT/Codex subscription quota."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Optional

from hermes_constants import get_hermes_home

_STATE_FILE = "codex-daily-quota.json"


def _weekly_percent(snapshot: Any) -> Optional[float]:
    for window in getattr(snapshot, "windows", ()) or ():
        if str(getattr(window, "label", "")).strip().lower() == "weekly":
            value = getattr(window, "used_percent", None)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
    return None


def _state_path() -> Path:
    return Path(get_hermes_home()) / _STATE_FILE


def _load_state() -> dict:
    try:
        value = json.loads(_state_path().read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _save_state(value: dict) -> None:
    try:
        path = _state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(value), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        pass


def configured_daily_budget() -> Optional[float]:
    try:
        from hermes_cli.config import load_config_readonly
        cfg = (load_config_readonly() or {}).get("codex_quota_guard") or {}
    except Exception:
        return None
    if not isinstance(cfg, dict) or not cfg.get("enabled", False):
        return None
    try:
        value = float(cfg.get("daily_budget_percent", 14.0))
    except (TypeError, ValueError):
        value = 14.0
    return max(0.1, min(100.0, value))


def check_daily_budget(*, base_url: str = "", api_key: str = "") -> Optional[str]:
    """Return a terminal user-facing message when today's budget is exhausted.

    The baseline is persisted under HERMES_HOME, so restarting Hermes cannot bypass
    the daily guard. Provider telemetry failures fail open rather than disabling Codex.
    """
    budget = configured_daily_budget()
    if budget is None:
        return None
    try:
        from agent.account_usage import fetch_account_usage
        snapshot = fetch_account_usage("openai-codex", base_url=base_url or None, api_key=api_key or None)
    except Exception:
        return None
    used = _weekly_percent(snapshot)
    if used is None:
        return None

    today = date.today().isoformat()
    state = _load_state()
    if state.get("day") != today:
        state = {"day": today, "baseline_weekly_percent": used}
        _save_state(state)
    try:
        baseline = float(state.get("baseline_weekly_percent", used))
    except (TypeError, ValueError):
        baseline = used
    spent = max(0.0, used - baseline)
    if spent < budget:
        return None
    return (
        f"Codex daily quota guard: {spent:.1f}% of the weekly allowance has been used "
        f"today (daily limit {budget:.1f}%). New Codex requests are blocked until tomorrow. "
        "No fallback provider was selected."
    )

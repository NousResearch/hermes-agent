from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import httpx

CODEX_USAGE_URL = "https://chatgpt.com/backend-api/wham/usage"
WEEKLY_RESET_GAP_SECONDS = 3 * 24 * 60 * 60


@dataclass(frozen=True)
class CodexUsageWindow:
    label: str
    used_percent: float
    reset_at_ms: Optional[int] = None


@dataclass(frozen=True)
class CodexUsageSnapshot:
    available: bool = False
    plan: Optional[str] = None
    windows: list[CodexUsageWindow] = field(default_factory=list)
    error: Optional[str] = None


GetFn = Callable[..., Any]


def _clamp_percent(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(100.0, numeric))


def _coerce_reset_at_ms(window: dict[str, Any], now_ms: int) -> Optional[int]:
    reset_at = window.get("reset_at")
    if isinstance(reset_at, (int, float)) and reset_at > 0:
        return int(float(reset_at) * 1000)
    reset_after = window.get("reset_after_seconds")
    if isinstance(reset_after, (int, float)) and reset_after > 0:
        return int(now_ms + float(reset_after) * 1000)
    return None


def _resolve_secondary_window_label(*, window_hours: int, secondary_reset_at: Any, primary_reset_at: Any) -> str:
    if window_hours >= 168:
        return "Week"
    if window_hours < 24:
        return f"{window_hours}h"
    if isinstance(secondary_reset_at, (int, float)) and isinstance(primary_reset_at, (int, float)):
        if float(secondary_reset_at) - float(primary_reset_at) >= WEEKLY_RESET_GAP_SECONDS:
            return "Week"
    return "Day"


def _format_reset_remaining(target_ms: Optional[int], *, now_ms: Optional[int] = None) -> Optional[str]:
    if not target_ms:
        return None
    base = now_ms if now_ms is not None else int(datetime.now(timezone.utc).timestamp() * 1000)
    diff_ms = int(target_ms) - int(base)
    if diff_ms <= 0:
        return "now"

    diff_mins = diff_ms // 60000
    if diff_mins < 60:
        return f"{diff_mins}m"

    hours = diff_mins // 60
    mins = diff_mins % 60
    if hours < 24:
        return f"{hours}h {mins}m" if mins > 0 else f"{hours}h"

    days = hours // 24
    if days < 7:
        return f"{days}d {hours % 24}h"

    formatted = datetime.fromtimestamp(target_ms / 1000, tz=timezone.utc).strftime("%b %d")
    return formatted.replace(" 0", " ")


def _format_plan(data: dict[str, Any]) -> Optional[str]:
    plan = data.get("plan_type")
    credits = data.get("credits")
    if isinstance(credits, dict) and credits.get("balance") is not None:
        try:
            balance = float(credits.get("balance"))
        except (TypeError, ValueError):
            balance = 0.0
        if isinstance(plan, str) and plan.strip():
            return f"{plan.strip()} (${balance:.2f})"
        return f"${balance:.2f}"
    if isinstance(plan, str) and plan.strip():
        return plan.strip()
    return None


def fetch_codex_usage_snapshot(
    access_token: str,
    *,
    account_id: Optional[str] = None,
    timeout: float = 10.0,
    get: Optional[GetFn] = None,
    now_ms: Optional[int] = None,
) -> CodexUsageSnapshot:
    if not isinstance(access_token, str) or not access_token.strip():
        return CodexUsageSnapshot(available=False, error="Missing access token")

    headers = {
        "Authorization": f"Bearer {access_token.strip()}",
        "User-Agent": "CodexBar",
        "Accept": "application/json",
    }
    if account_id:
        headers["ChatGPT-Account-Id"] = account_id

    getter = get or httpx.get
    current_now_ms = now_ms if now_ms is not None else int(datetime.now(timezone.utc).timestamp() * 1000)

    try:
        response = getter(CODEX_USAGE_URL, headers=headers, timeout=timeout)
    except Exception as exc:  # pragma: no cover - exercised by tests with custom exception types
        return CodexUsageSnapshot(available=False, error=str(exc))

    status_code = getattr(response, "status_code", None)
    if status_code != 200:
        return CodexUsageSnapshot(available=False, error=f"HTTP {status_code}")

    try:
        data = response.json()
    except Exception:
        return CodexUsageSnapshot(available=False, error="Invalid JSON")

    if not isinstance(data, dict):
        return CodexUsageSnapshot(available=False, error="Invalid response")

    rate_limit = data.get("rate_limit") if isinstance(data.get("rate_limit"), dict) else {}
    windows: list[CodexUsageWindow] = []

    primary_window = rate_limit.get("primary_window") if isinstance(rate_limit.get("primary_window"), dict) else None
    if primary_window:
        try:
            window_hours = round(float(primary_window.get("limit_window_seconds") or 10_800) / 3600)
        except (TypeError, ValueError):
            window_hours = 3
        windows.append(
            CodexUsageWindow(
                label=f"{window_hours}h",
                used_percent=_clamp_percent(primary_window.get("used_percent")),
                reset_at_ms=_coerce_reset_at_ms(primary_window, current_now_ms),
            )
        )

    secondary_window = rate_limit.get("secondary_window") if isinstance(rate_limit.get("secondary_window"), dict) else None
    if secondary_window:
        try:
            window_hours = round(float(secondary_window.get("limit_window_seconds") or 86_400) / 3600)
        except (TypeError, ValueError):
            window_hours = 24
        windows.append(
            CodexUsageWindow(
                label=_resolve_secondary_window_label(
                    window_hours=window_hours,
                    secondary_reset_at=secondary_window.get("reset_at"),
                    primary_reset_at=primary_window.get("reset_at") if primary_window else None,
                ),
                used_percent=_clamp_percent(secondary_window.get("used_percent")),
                reset_at_ms=_coerce_reset_at_ms(secondary_window, current_now_ms),
            )
        )

    plan = _format_plan(data)
    return CodexUsageSnapshot(available=bool(windows or plan), plan=plan, windows=windows)


def get_current_codex_usage_snapshot(*, timeout: float = 10.0) -> CodexUsageSnapshot:
    try:
        from hermes_cli.auth import get_codex_auth_status

        status = get_codex_auth_status()
        api_key = status.get("api_key") if isinstance(status, dict) else None
        if not status or not status.get("logged_in") or not isinstance(api_key, str) or not api_key.strip():
            return CodexUsageSnapshot(available=False, error="Not logged in")
        account_id = status.get("account_id") if isinstance(status, dict) else None
        account_id = account_id.strip() if isinstance(account_id, str) and account_id.strip() else None
        return fetch_codex_usage_snapshot(api_key, account_id=account_id, timeout=timeout)
    except Exception as exc:  # pragma: no cover - defensive guard
        return CodexUsageSnapshot(available=False, error=str(exc))


def format_codex_usage_summary(
    snapshot: CodexUsageSnapshot,
    *,
    now_ms: Optional[int] = None,
    max_windows: Optional[int] = 2,
    include_resets: bool = True,
) -> Optional[str]:
    if not snapshot.available or not snapshot.windows:
        return None

    windows = snapshot.windows if not max_windows or max_windows <= 0 else snapshot.windows[:max_windows]
    parts: list[str] = []
    for window in windows:
        remaining = _clamp_percent(100.0 - window.used_percent)
        reset = _format_reset_remaining(window.reset_at_ms, now_ms=now_ms) if include_resets else None
        reset_suffix = f" ⏱{reset}" if reset else ""
        parts.append(f"{window.label} {remaining:.0f}% left{reset_suffix}")
    return " · ".join(parts)


def format_codex_usage_report_lines(
    snapshot: CodexUsageSnapshot,
    *,
    now_ms: Optional[int] = None,
) -> list[str]:
    if not snapshot.available or not snapshot.windows:
        return []

    lines = ["Codex Account Usage:"]
    if snapshot.plan:
        lines.append(f"  Plan: {snapshot.plan}")
    for window in snapshot.windows:
        remaining = _clamp_percent(100.0 - window.used_percent)
        reset = _format_reset_remaining(window.reset_at_ms, now_ms=now_ms)
        suffix = f" · resets {reset}" if reset else ""
        lines.append(f"  {window.label}: {remaining:.0f}% left{suffix}")
    return lines


def is_codex_provider(provider: Optional[str], base_url: Optional[str] = None) -> bool:
    if isinstance(provider, str) and provider.strip().lower() == "openai-codex":
        return True
    return isinstance(base_url, str) and "chatgpt.com/backend-api/codex" in base_url.lower()

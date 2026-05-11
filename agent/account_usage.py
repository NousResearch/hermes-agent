from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from agent.anthropic_adapter import _is_oauth_token, resolve_anthropic_token
from hermes_cli.auth import DEFAULT_CODEX_BASE_URL, _read_codex_tokens, resolve_codex_runtime_credentials
from hermes_cli.runtime_provider import resolve_runtime_provider


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class AccountUsageWindow:
    label: str
    used_percent: Optional[float] = None
    reset_at: Optional[datetime] = None
    detail: Optional[str] = None


@dataclass(frozen=True)
class AccountUsageSnapshot:
    provider: str
    source: str
    fetched_at: datetime
    title: str = "Account limits"
    plan: Optional[str] = None
    windows: tuple[AccountUsageWindow, ...] = ()
    details: tuple[str, ...] = ()
    unavailable_reason: Optional[str] = None

    @property
    def available(self) -> bool:
        return bool(self.windows or self.details) and not self.unavailable_reason


def _title_case_slug(value: Optional[str]) -> Optional[str]:
    cleaned = str(value or "").strip()
    if not cleaned:
        return None
    return cleaned.replace("_", " ").replace("-", " ").title()


def _parse_dt(value: Any) -> Optional[datetime]:
    if value in {None, ""}:
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def _format_reset(dt: Optional[datetime]) -> str:
    if not dt:
        return "unknown"
    local_dt = dt.astimezone()
    delta = dt - _utc_now()
    total_seconds = int(delta.total_seconds())
    if total_seconds <= 0:
        return f"now ({local_dt.strftime('%Y-%m-%d %H:%M %Z')})"
    hours, rem = divmod(total_seconds, 3600)
    minutes = rem // 60
    if hours >= 24:
        days, hours = divmod(hours, 24)
        rel = f"in {days}d {hours}h"
    elif hours > 0:
        rel = f"in {hours}h {minutes}m"
    else:
        rel = f"in {minutes}m"
    return f"{rel} ({local_dt.strftime('%Y-%m-%d %H:%M %Z')})"


def render_account_usage_lines(snapshot: Optional[AccountUsageSnapshot], *, markdown: bool = False) -> list[str]:
    if not snapshot:
        return []
    header = f"📈 {'**' if markdown else ''}{snapshot.title}{'**' if markdown else ''}"
    lines = [header]
    if snapshot.plan:
        lines.append(f"Provider: {snapshot.provider} ({snapshot.plan})")
    else:
        lines.append(f"Provider: {snapshot.provider}")
    for window in snapshot.windows:
        if window.used_percent is None:
            base = f"{window.label}: unavailable"
        else:
            remaining = max(0, round(100 - float(window.used_percent)))
            used = max(0, round(float(window.used_percent)))
            base = f"{window.label}: {remaining}% remaining ({used}% used)"
        if window.reset_at:
            base += f" • resets {_format_reset(window.reset_at)}"
        elif window.detail:
            base += f" • {window.detail}"
        lines.append(base)
    for detail in snapshot.details:
        lines.append(detail)
    if snapshot.unavailable_reason:
        lines.append(f"Unavailable: {snapshot.unavailable_reason}")
    return lines


def _resolve_codex_usage_url(base_url: str) -> str:
    normalized = (base_url or "").strip().rstrip("/")
    if not normalized:
        normalized = "https://chatgpt.com/backend-api/codex"
    if normalized.endswith("/codex"):
        normalized = normalized[: -len("/codex")]
    if "/backend-api" in normalized:
        return normalized + "/wham/usage"
    return normalized + "/api/codex/usage"


def _resolve_codex_usage_credentials() -> dict[str, Any]:
    """Resolve Codex usage credentials from singleton auth or the credential pool.

    ``resolve_codex_runtime_credentials`` only reads the singleton
    ``providers.openai-codex`` auth state. Joohyun/Mina often uses only pooled
    Codex OAuth credentials, so account usage needs the same pool fallback as
    runtime model calls.
    """
    try:
        creds = dict(resolve_codex_runtime_credentials(refresh_if_expiring=True))
        token_data = _read_codex_tokens()
        tokens = token_data.get("tokens") or {}
        creds["account_id"] = str(tokens.get("account_id", "") or "").strip() or None
        return creds
    except Exception as singleton_exc:
        try:
            from agent.credential_pool import load_pool

            pool = load_pool("openai-codex")
            # For monitoring, prefer the fill-first credential even if it is
            # currently marked exhausted: ChatGPT's usage endpoint can still
            # return the 5h/weekly windows and reset time for exhausted accounts.
            # ``select()`` would skip it and hide the account we most need to
            # observe.
            entry = next((candidate for candidate in pool.entries() if candidate.runtime_api_key), None)
            if entry is None:
                entry = pool.select()
            if entry is None or not entry.runtime_api_key:
                raise singleton_exc
            return {
                "provider": "openai-codex",
                "base_url": entry.runtime_base_url or DEFAULT_CODEX_BASE_URL,
                "api_key": entry.runtime_api_key,
                "source": f"credential-pool:{entry.label or entry.id}",
                "account_id": None,
            }
        except Exception:
            raise singleton_exc


def _fetch_codex_account_usage() -> Optional[AccountUsageSnapshot]:
    creds = _resolve_codex_usage_credentials()
    account_id = str(creds.get("account_id", "") or "").strip() or None
    headers = {
        "Authorization": f"Bearer {creds['api_key']}",
        "Accept": "application/json",
        "User-Agent": "codex-cli",
    }
    if account_id:
        headers["ChatGPT-Account-Id"] = account_id
    with httpx.Client(timeout=15.0) as client:
        response = client.get(_resolve_codex_usage_url(creds.get("base_url", "")), headers=headers)
        response.raise_for_status()
    payload = response.json() or {}
    def append_rate_limit_windows(rate_limit: Any, *, prefix: str = "") -> None:
        if not isinstance(rate_limit, dict):
            return
        for key, label in (("primary_window", "Session"), ("secondary_window", "Weekly")):
            window = rate_limit.get(key) or {}
            if not isinstance(window, dict):
                continue
            used = window.get("used_percent")
            if used is None:
                continue
            windows.append(
                AccountUsageWindow(
                    label=f"{prefix} {label}".strip(),
                    used_percent=float(used),
                    reset_at=_parse_dt(window.get("reset_at")),
                )
            )

    windows: list[AccountUsageWindow] = []
    append_rate_limit_windows(payload.get("rate_limit"))
    code_review_rate_limit = payload.get("code_review_rate_limit")
    if code_review_rate_limit:
        append_rate_limit_windows(code_review_rate_limit, prefix="Code review")
    for item in payload.get("additional_rate_limits") or []:
        if not isinstance(item, dict):
            continue
        label = str(item.get("limit_name") or item.get("metered_feature") or "Additional").strip()
        append_rate_limit_windows(item.get("rate_limit"), prefix=label)
    details: list[str] = []
    credits = payload.get("credits") or {}
    if credits.get("has_credits"):
        balance = credits.get("balance")
        if isinstance(balance, (int, float)):
            details.append(f"Credits balance: ${float(balance):.2f}")
        elif credits.get("unlimited"):
            details.append("Credits balance: unlimited")
    return AccountUsageSnapshot(
        provider="openai-codex",
        source="usage_api",
        fetched_at=_utc_now(),
        plan=_title_case_slug(payload.get("plan_type")),
        windows=tuple(windows),
        details=tuple(details),
    )


def _resolve_anthropic_usage_token() -> Optional[str]:
    """Resolve an Anthropic OAuth token for account usage monitoring.

    ``resolve_anthropic_token`` reads singleton/env/Claude Code auth. Joohyun/Mina
    often uses a Hermes credential-pool OAuth entry instead, so usage checks need
    the same pool fallback as runtime model calls.
    """
    token = (resolve_anthropic_token() or "").strip()
    if token:
        return token
    try:
        from agent.credential_pool import load_pool

        pool = load_pool("anthropic")
        for entry in pool.entries():
            candidate = str(getattr(entry, "access_token", "") or "").strip()
            if candidate:
                return candidate
    except Exception:
        return None
    return None


def _format_anthropic_extra_usage_detail(used_credits: float, monthly_limit: float, currency: str) -> str:
    # Anthropic currently reports extra-usage values in cent-like credits: UI
    # $200/month appears as monthly_limit=20000 and $0.76 used as 76 credits.
    return (
        f"Extra usage: ${used_credits / 100:.2f} / ${monthly_limit / 100:.2f} {currency} "
        f"({used_credits:.0f} / {monthly_limit:.0f} credits)"
    )


def _fetch_anthropic_account_usage() -> Optional[AccountUsageSnapshot]:
    token = _resolve_anthropic_usage_token()
    if not token:
        return None
    if not _is_oauth_token(token):
        return AccountUsageSnapshot(
            provider="anthropic",
            source="oauth_usage_api",
            fetched_at=_utc_now(),
            unavailable_reason="Anthropic account limits are only available for OAuth-backed Claude accounts.",
        )
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
        "anthropic-beta": "oauth-2025-04-20",
        "User-Agent": "claude-code/2.1.138",
    }
    with httpx.Client(timeout=15.0) as client:
        response = client.get("https://api.anthropic.com/api/oauth/usage", headers=headers)
        response.raise_for_status()
    payload = response.json() or {}
    windows: list[AccountUsageWindow] = []
    mapping = (
        ("five_hour", "Current session"),
        ("seven_day", "Current week"),
        ("seven_day_opus", "Opus week"),
        ("seven_day_sonnet", "Sonnet week"),
    )
    for key, label in mapping:
        window = payload.get(key) or {}
        util = window.get("utilization")
        if util is None:
            continue
        # Anthropic OAuth usage `utilization` is already a percentage value, not a
        # 0..1 fraction. Example: extra_usage reports 29 / 2000 USD as 1.45, and
        # five_hour can report 1.0 for ~1% used. Do not multiply values <= 1 by 100.
        used = float(util)
        windows.append(
            AccountUsageWindow(
                label=label,
                used_percent=used,
                reset_at=_parse_dt(window.get("resets_at")),
            )
        )
    details: list[str] = []
    extra = payload.get("extra_usage") or {}
    if extra.get("is_enabled"):
        used_credits = extra.get("used_credits")
        monthly_limit = extra.get("monthly_limit")
        currency = extra.get("currency") or "USD"
        if isinstance(used_credits, (int, float)) and isinstance(monthly_limit, (int, float)):
            details.append(
                _format_anthropic_extra_usage_detail(
                    float(used_credits), float(monthly_limit), str(currency)
                )
            )
    return AccountUsageSnapshot(
        provider="anthropic",
        source="oauth_usage_api",
        fetched_at=_utc_now(),
        windows=tuple(windows),
        details=tuple(details),
    )


def _fetch_openrouter_account_usage(base_url: Optional[str], api_key: Optional[str]) -> Optional[AccountUsageSnapshot]:
    runtime = resolve_runtime_provider(
        requested="openrouter",
        explicit_base_url=base_url,
        explicit_api_key=api_key,
    )
    token = str(runtime.get("api_key", "") or "").strip()
    if not token:
        return None
    normalized = str(runtime.get("base_url", "") or "").rstrip("/")
    credits_url = f"{normalized}/credits"
    key_url = f"{normalized}/key"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }
    with httpx.Client(timeout=10.0) as client:
        credits_resp = client.get(credits_url, headers=headers)
        credits_resp.raise_for_status()
        credits = (credits_resp.json() or {}).get("data") or {}
        try:
            key_resp = client.get(key_url, headers=headers)
            key_resp.raise_for_status()
            key_data = (key_resp.json() or {}).get("data") or {}
        except Exception:
            key_data = {}
    total_credits = float(credits.get("total_credits") or 0.0)
    total_usage = float(credits.get("total_usage") or 0.0)
    details = [f"Credits balance: ${max(0.0, total_credits - total_usage):.2f}"]
    windows: list[AccountUsageWindow] = []
    limit = key_data.get("limit")
    limit_remaining = key_data.get("limit_remaining")
    limit_reset = str(key_data.get("limit_reset") or "").strip()
    usage = key_data.get("usage")
    if (
        isinstance(limit, (int, float))
        and float(limit) > 0
        and isinstance(limit_remaining, (int, float))
        and 0 <= float(limit_remaining) <= float(limit)
    ):
        limit_value = float(limit)
        remaining_value = float(limit_remaining)
        used_percent = ((limit_value - remaining_value) / limit_value) * 100
        detail_parts = [f"${remaining_value:.2f} of ${limit_value:.2f} remaining"]
        if limit_reset:
            detail_parts.append(f"resets {limit_reset}")
        windows.append(
            AccountUsageWindow(
                label="API key quota",
                used_percent=used_percent,
                detail=" • ".join(detail_parts),
            )
        )
    if isinstance(usage, (int, float)):
        usage_parts = [f"API key usage: ${float(usage):.2f} total"]
        for value, label in (
            (key_data.get("usage_daily"), "today"),
            (key_data.get("usage_weekly"), "this week"),
            (key_data.get("usage_monthly"), "this month"),
        ):
            if isinstance(value, (int, float)) and float(value) > 0:
                usage_parts.append(f"${float(value):.2f} {label}")
        details.append(" • ".join(usage_parts))
    return AccountUsageSnapshot(
        provider="openrouter",
        source="credits_api",
        fetched_at=_utc_now(),
        windows=tuple(windows),
        details=tuple(details),
    )


def fetch_account_usage(
    provider: Optional[str],
    *,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Optional[AccountUsageSnapshot]:
    normalized = str(provider or "").strip().lower()
    if normalized in {"", "auto", "custom"}:
        return None
    try:
        if normalized == "openai-codex":
            return _fetch_codex_account_usage()
        if normalized == "anthropic":
            return _fetch_anthropic_account_usage()
        if normalized == "openrouter":
            return _fetch_openrouter_account_usage(base_url, api_key)
    except Exception:
        return None
    return None

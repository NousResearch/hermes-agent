from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Optional

import httpx

from agent.anthropic_adapter import _is_oauth_token, resolve_anthropic_token
from hermes_constants import get_hermes_home
from hermes_cli.auth import _read_codex_tokens, get_codex_auth_status, resolve_codex_runtime_credentials
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


def _account_usage_cache_path(provider: str) -> Path:
    safe_provider = str(provider or "unknown").replace("/", "_").replace(" ", "_")
    return Path(get_hermes_home()) / "cache" / "account_usage" / f"{safe_provider}.json"


def _snapshot_to_cache(snapshot: AccountUsageSnapshot) -> dict[str, Any]:
    return {
        "provider": snapshot.provider,
        "source": snapshot.source,
        "fetched_at": snapshot.fetched_at.isoformat(),
        "title": snapshot.title,
        "plan": snapshot.plan,
        "windows": [
            {
                "label": window.label,
                "used_percent": window.used_percent,
                "reset_at": window.reset_at.isoformat() if window.reset_at else None,
                "detail": window.detail,
            }
            for window in snapshot.windows
        ],
        "details": list(snapshot.details),
    }


def _snapshot_from_cache(
    data: dict[str, Any],
    *,
    unavailable_reason: Optional[str] = None,
) -> Optional[AccountUsageSnapshot]:
    fetched_at = _parse_dt(data.get("fetched_at"))
    if not fetched_at:
        return None
    windows: list[AccountUsageWindow] = []
    for raw in data.get("windows") or []:
        if not isinstance(raw, dict):
            continue
        used_raw = raw.get("used_percent")
        windows.append(
            AccountUsageWindow(
                label=str(raw.get("label") or "Limit"),
                used_percent=float(used_raw) if used_raw is not None else None,
                reset_at=_parse_dt(raw.get("reset_at")),
                detail=raw.get("detail"),
            )
        )
    if not windows and not data.get("details"):
        return None
    details = tuple(str(item) for item in (data.get("details") or []) if item)
    if unavailable_reason:
        details = (*details, f"Last live usage check unavailable: {unavailable_reason}")
    return AccountUsageSnapshot(
        provider=str(data.get("provider") or "openai-codex"),
        source=str(data.get("source") or "usage_api_cache"),
        fetched_at=fetched_at,
        title=str(data.get("title") or "Account limits"),
        plan=data.get("plan"),
        windows=tuple(windows),
        details=details,
    )


def _write_account_usage_cache(snapshot: AccountUsageSnapshot) -> None:
    try:
        path = _account_usage_cache_path(snapshot.provider)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_snapshot_to_cache(snapshot), ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        return


def _read_account_usage_cache(
    provider: str,
    *,
    unavailable_reason: Optional[str] = None,
) -> Optional[AccountUsageSnapshot]:
    try:
        path = _account_usage_cache_path(provider)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        return _snapshot_from_cache(data, unavailable_reason=unavailable_reason)
    except Exception:
        return None


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


def _fetch_codex_account_usage() -> Optional[AccountUsageSnapshot]:
    account_id = None
    try:
        creds = resolve_codex_runtime_credentials(refresh_if_expiring=True)
        token_data = _read_codex_tokens()
        tokens = token_data.get("tokens") or {}
        account_id = str(tokens.get("account_id", "") or "").strip() or None
    except Exception:
        # Profile setups commonly keep Codex credentials only in the credential
        # pool.  The active agent can use those credentials, so account usage
        # must use the same status/pool fallback instead of reporting
        # "not reported by provider".
        status = get_codex_auth_status()
        api_key = str(status.get("api_key", "") or "").strip()
        if not api_key:
            return None
        creds = {
            "api_key": api_key,
            "base_url": status.get("base_url") or "",
        }
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
    rate_limit = payload.get("rate_limit") or {}
    windows: list[AccountUsageWindow] = []
    for key, label in (("primary_window", "Session"), ("secondary_window", "Weekly")):
        window = rate_limit.get(key) or {}
        used = window.get("used_percent")
        if used is None:
            continue
        windows.append(
            AccountUsageWindow(
                label=label,
                used_percent=float(used),
                reset_at=_parse_dt(window.get("reset_at")),
            )
        )
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


def _fetch_anthropic_account_usage() -> Optional[AccountUsageSnapshot]:
    token = (resolve_anthropic_token() or "").strip()
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
        "User-Agent": "claude-code/2.1.0",
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
        used = float(util) * 100 if float(util) <= 1 else float(util)
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
                f"Extra usage: {used_credits:.2f} / {monthly_limit:.2f} {currency}"
            )
    return AccountUsageSnapshot(
        provider="anthropic",
        source="oauth_usage_api",
        fetched_at=_utc_now(),
        windows=tuple(windows),
        details=tuple(details),
    )


def _coerce_float_header(headers: Mapping[str, str], name: str) -> Optional[float]:
    value = headers.get(name) or headers.get(name.lower()) or headers.get(name.title())
    if value in {None, ""}:
        return None
    try:
        return float(str(value).strip())
    except Exception:
        return None


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


def _fetch_xai_oauth_account_usage() -> Optional[AccountUsageSnapshot]:
    try:
        from agent.credential_pool import load_pool

        pool = load_pool("xai-oauth")
        if not pool or not pool.has_credentials():
            return None
        entry = pool.peek()
        if not entry:
            return None
        token = str(getattr(entry, "access_token", "") or getattr(entry, "runtime_api_key", "") or "").strip()
        if not token:
            return None
        base_url = str(getattr(entry, "base_url", "") or "https://api.x.ai/v1").strip().rstrip("/")
    except Exception:
        return None

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    # xAI currently exposes rate-limit counters on inference responses, not on
    # the lightweight /models endpoint. Keep the probe tiny; response usage is
    # ignored except for optional diagnostics.
    payload = {
        "model": "grok-4.3-latest",
        "messages": [{"role": "user", "content": "OK"}],
        "max_tokens": 1,
    }
    with httpx.Client(timeout=15.0) as client:
        response = client.post(f"{base_url}/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
    resp_headers = {str(k).lower(): str(v) for k, v in response.headers.items()}
    req_limit = _coerce_float_header(resp_headers, "x-ratelimit-limit-requests")
    req_remaining = _coerce_float_header(resp_headers, "x-ratelimit-remaining-requests")
    tok_limit = _coerce_float_header(resp_headers, "x-ratelimit-limit-tokens")
    tok_remaining = _coerce_float_header(resp_headers, "x-ratelimit-remaining-tokens")
    windows: list[AccountUsageWindow] = []
    details: list[str] = []
    if req_limit:
        detail = f"Request rate-limit header: {int(req_limit):,}"
        if req_remaining is not None:
            detail += f"; remaining header currently reports {int(req_remaining):,}"
        details.append(detail)
    if tok_limit:
        detail = f"Token rate-limit header: {int(tok_limit):,}"
        if tok_remaining is not None:
            detail += f"; remaining header currently reports {int(tok_remaining):,}"
        details.append(detail)
    details.append("Note: xAI OAuth currently reports rate-limit headers, not reliable account consumption; observed remaining counters can stay static across calls.")
    usage = None
    try:
        usage = (response.json() or {}).get("usage") or {}
    except Exception:
        usage = {}
    cached = (((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens"))
    if isinstance(cached, (int, float)):
        details.append(f"Probe cache read: {int(cached):,} tokens")
    return AccountUsageSnapshot(
        provider="xai-oauth",
        source="inference_rate_limit_headers",
        fetched_at=_utc_now(),
        plan="SuperGrok OAuth",
        windows=tuple(windows),
        details=tuple(details),
        unavailable_reason=None if details else "xAI did not return rate-limit headers",
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
        snapshot: Optional[AccountUsageSnapshot]
        if normalized == "openai-codex":
            snapshot = _fetch_codex_account_usage()
        elif normalized == "anthropic":
            snapshot = _fetch_anthropic_account_usage()
        elif normalized == "openrouter":
            snapshot = _fetch_openrouter_account_usage(base_url, api_key)
        elif normalized == "xai-oauth":
            snapshot = _fetch_xai_oauth_account_usage()
        else:
            snapshot = None
        if snapshot:
            _write_account_usage_cache(snapshot)
            return snapshot
        return _read_account_usage_cache(normalized, unavailable_reason="not reported by provider")
    except Exception as exc:
        return _read_account_usage_cache(
            normalized,
            unavailable_reason=str(exc) or exc.__class__.__name__,
        )

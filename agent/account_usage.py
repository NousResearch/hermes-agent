from __future__ import annotations

import concurrent.futures
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import httpx

from agent.anthropic_adapter import _is_oauth_token, resolve_anthropic_token
from agent.credential_pool import load_pool
from hermes_cli.auth import (
    _decode_jwt_claims,
    _read_codex_tokens,
    read_credential_pool,
    resolve_codex_runtime_credentials,
)
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


@dataclass(frozen=True)
class ProviderAccountUsage:
    provider: str
    index: int
    label: str
    status: str
    snapshot: Optional[AccountUsageSnapshot] = None
    unavailable_reason: Optional[str] = None


@dataclass(frozen=True)
class AccountProviderChoice:
    slug: str
    name: str
    account_count: int
    is_current: bool = False


_ACCOUNT_USAGE_LOOKUP_PROVIDERS = {"anthropic", "openai-codex", "openrouter"}
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
_LEGACY_MARKDOWN_RE = re.compile(r"([_*\[\]()`])")


def safe_display_text(
    value: Any,
    *,
    fallback: str = "account",
    max_len: Optional[int] = 80,
    markdown: bool = False,
) -> str:
    """Return compact, user-facing text that cannot break platform renderers.

    This is intentionally conservative: credential labels and plan/provider
    names may be user-controlled, so strip control characters, collapse
    whitespace, shorten safely, and optionally escape legacy Markdown-sensitive
    characters used by Telegram account-picker messages.
    """
    text = str(value or "").strip() or fallback
    text = _CONTROL_CHAR_RE.sub("", text)
    text = " ".join(text.split())
    if max_len is not None and max_len > 1 and len(text) > max_len:
        text = text[: max_len - 1].rstrip() + "…"
    if markdown:
        text = _LEGACY_MARKDOWN_RE.sub(r"\\\1", text)
    return text


def safe_identifier(value: Any, *, fallback: str = "credential") -> str:
    """Display a credential id without exposing a long secret-bearing value."""
    text = safe_display_text(value, fallback=fallback, max_len=None)
    if len(text) <= 14:
        return text
    return f"{text[:6]}…{text[-4:]}"


def _normalize_provider(provider: Optional[str]) -> str:
    return str(provider or "").strip().lower()


def _account_unavailable_snapshot(provider: str, reason: str) -> AccountUsageSnapshot:
    return AccountUsageSnapshot(
        provider=provider,
        source="credential_pool",
        fetched_at=_utc_now(),
        unavailable_reason=reason,
    )


def _provider_display_name(provider: str) -> str:
    try:
        from hermes_cli.providers import get_label

        label = get_label(provider)
        if isinstance(label, str) and label.strip():
            return label.strip()
    except Exception:
        pass
    return _title_case_slug(provider) or provider


def _title_case_slug(value: Optional[str]) -> Optional[str]:
    cleaned = str(value or "").strip()
    if not cleaned:
        return None
    return cleaned.replace("_", " ").replace("-", " ").title()


def _parse_dt(value: Any) -> Optional[datetime]:
    if value in (None, ""):
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


def format_reset_countdown(
    dt: Optional[datetime],
    *,
    now: Optional[datetime] = None,
    verb: str = "renews",
) -> str:
    """Format reset/renewal time as a compact Hermes-style countdown."""
    if not dt:
        return "renewal time unknown"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    now = now or _utc_now()
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    now = now.astimezone(timezone.utc)
    delta = dt - now
    total_seconds = int(delta.total_seconds())
    if total_seconds <= 0:
        return f"{verb} now ✨"

    if dt.date() == (now + timedelta(days=1)).date():
        return f"{verb} tomorrow at {dt.strftime('%H:%M')} 🌙"

    minutes_total = max(1, (total_seconds + 59) // 60)
    hours, minutes = divmod(minutes_total, 60)
    if hours == 0:
        return f"{verb} in {minutes}m ⏳"
    if dt.date() == now.date():
        return f"{verb} in {hours}h {minutes}m ✨"
    days, hours = divmod(hours, 24)
    return f"{verb} in {days}d {hours}h 🗓️"


def _format_reset(dt: Optional[datetime]) -> str:
    return format_reset_countdown(dt, verb="resets")


def render_account_usage_lines(snapshot: Optional[AccountUsageSnapshot], *, markdown: bool = False) -> list[str]:
    if not snapshot:
        return []
    title = safe_display_text(snapshot.title, fallback="Account limits", markdown=markdown)
    provider = safe_display_text(snapshot.provider, fallback="provider", markdown=markdown)
    header = f"📈 {'**' if markdown else ''}{title}{'**' if markdown else ''}"
    lines = [header]
    if snapshot.plan:
        plan = safe_display_text(snapshot.plan, fallback="plan", markdown=markdown)
        lines.append(f"Provider: {provider} ({plan})")
    else:
        lines.append(f"Provider: {provider}")
    for window in snapshot.windows:
        if window.used_percent is None:
            label = safe_display_text(window.label, fallback="Limit", markdown=markdown)
            base = f"{label}: unavailable"
        else:
            remaining = max(0, round(100 - float(window.used_percent)))
            used = max(0, round(float(window.used_percent)))
            label = safe_display_text(window.label, fallback="Limit", markdown=markdown)
            base = f"{label}: {remaining}% left {_usage_bar(remaining)} ({used}% used)"
        if window.reset_at:
            base += f" · {format_reset_countdown(window.reset_at)}"
        elif window.detail:
            base += f" · {safe_display_text(window.detail, fallback='', markdown=markdown)}"
        lines.append(base)
    for detail in snapshot.details:
        lines.append(safe_display_text(detail, fallback="detail", markdown=markdown))
    if snapshot.unavailable_reason:
        reason = safe_display_text(snapshot.unavailable_reason, fallback="usage unavailable", markdown=markdown)
        lines.append(f"Unavailable: {reason}")
    return lines


def _remaining_percent(window: AccountUsageWindow) -> Optional[int]:
    if window.used_percent is None:
        return None
    return max(0, min(100, round(100 - float(window.used_percent))))


def _usage_bar(remaining_percent: Optional[int], *, width: int = 10) -> str:
    if remaining_percent is None:
        return "▱" * width
    filled = max(0, min(width, round((remaining_percent / 100) * width)))
    return "▰" * filled + "▱" * (width - filled)


def _account_health(snapshot: Optional[AccountUsageSnapshot]) -> tuple[str, str, Optional[int]]:
    if not snapshot or snapshot.unavailable_reason:
        return "⚪", "Unknown", None
    remaining_values = [
        remaining
        for remaining in (_remaining_percent(window) for window in snapshot.windows)
        if remaining is not None
    ]
    if not remaining_values:
        return "✅", "Available", None
    lowest_remaining = min(remaining_values)
    if lowest_remaining <= 5:
        return "⛔", "Limited", lowest_remaining
    if lowest_remaining <= 20:
        return "⚠️", "Low", lowest_remaining
    return "✅", "Healthy", lowest_remaining


def _format_account_window_line(window: AccountUsageWindow, *, markdown: bool = False) -> str:
    label = safe_display_text(window.label, fallback="Limit", markdown=markdown)
    remaining = _remaining_percent(window)
    if remaining is None or window.used_percent is None:
        base = f"   {label}: unavailable {_usage_bar(None)}"
    else:
        used = max(0, min(100, round(float(window.used_percent))))
        base = f"   {label}: {remaining}% left {_usage_bar(remaining)} ({used}% used)"
    if window.reset_at:
        base += f" · {format_reset_countdown(window.reset_at)}"
    elif window.detail:
        base += f" · {safe_display_text(window.detail, fallback='', markdown=markdown)}"
    return base


def compact_account_usage_description(
    snapshot: Optional[AccountUsageSnapshot],
    *,
    max_len: int = 100,
) -> str:
    """Discord/Telegram option description based on snapshot.windows."""
    if not snapshot:
        return "usage unavailable"
    if snapshot.unavailable_reason:
        return safe_display_text(snapshot.unavailable_reason, fallback="usage unavailable", max_len=max_len)
    parts: list[str] = []
    for window in snapshot.windows[:2]:
        remaining = _remaining_percent(window)
        label = safe_display_text(window.label, fallback="Limit", max_len=22)
        if remaining is not None:
            parts.append(f"{label}: {remaining}% left")
        elif window.detail:
            parts.append(f"{label}: {safe_display_text(window.detail, max_len=42)}")
    if not parts and snapshot.details:
        parts.append(safe_display_text(snapshot.details[0], fallback="usage detail", max_len=max_len))
    if not parts:
        parts.append("usage available")
    return safe_display_text(" · ".join(parts), fallback="usage unavailable", max_len=max_len)


def discord_account_option_parts(result: ProviderAccountUsage, *, active: bool = False) -> tuple[str, str]:
    """Return Discord-safe select option label and description."""
    _emoji, health, _lowest_remaining = _account_health(result.snapshot)
    raw_label = result.label or f"account-{result.index}"
    label = safe_display_text(raw_label, fallback=f"account-{result.index}", max_len=62)
    active_prefix = "✓ " if active else ""
    option_label = safe_display_text(
        f"{active_prefix}{result.index}. {label} · {health}",
        fallback=f"{result.index}. account · {health}",
        max_len=100,
    )
    desc = compact_account_usage_description(result.snapshot, max_len=100)
    if result.unavailable_reason and not result.snapshot:
        desc = safe_display_text(result.unavailable_reason, fallback="usage unavailable", max_len=100)
    return option_label, desc


def render_provider_account_usage_lines(
    provider: str,
    results: list[ProviderAccountUsage] | tuple[ProviderAccountUsage, ...],
    *,
    markdown: bool = False,
    active_index: Optional[int] = None,
    select_hint: Optional[str] = None,
) -> list[str]:
    normalized = _normalize_provider(provider)
    if not results:
        return []
    strong = "**" if markdown else ""
    lines = [f"📊 {strong}Hermes Account Center{strong}"]
    provider_label = safe_display_text(normalized, fallback="provider", markdown=markdown)
    lines.append(f"Provider: {provider_label} • Accounts: {len(results)}")
    hint = (select_hint or f"/account {normalized} <number>").strip()
    lines.append(f"Select: `{hint}`" if markdown else f"Select: {hint}")
    lines.append("")
    for result in results:
        label = safe_display_text(
            result.label or f"account-{result.index}",
            fallback=f"account-{result.index}",
            markdown=markdown,
        )
        snapshot = result.snapshot
        emoji, health, _lowest_remaining = _account_health(snapshot)
        active = active_index is not None and result.index == active_index
        active_suffix = " — active" if active else ""
        title = f"{result.index}. {label}"
        if markdown:
            title = f"{strong}{title}{strong}"
        lines.append(f"{emoji} {title}{active_suffix} · {health}")
        if result.status:
            lines.append(f"   Pool: {safe_display_text(result.status, fallback='available', markdown=markdown)}")
        if not snapshot:
            reason = safe_display_text(
                result.unavailable_reason or "usage unavailable",
                fallback="usage unavailable",
                markdown=markdown,
            )
            lines.append(f"   Unavailable: {reason}")
            lines.append("")
            continue
        if snapshot.plan:
            plan = safe_display_text(snapshot.plan, fallback="plan", markdown=markdown)
            lines.append(f"   Plan: {plan}")
        for window in snapshot.windows:
            lines.append(_format_account_window_line(window, markdown=markdown))
        for detail in snapshot.details:
            lines.append(f"   {safe_display_text(detail, fallback='detail', markdown=markdown)}")
        if snapshot.unavailable_reason:
            reason = safe_display_text(snapshot.unavailable_reason, fallback="usage unavailable", markdown=markdown)
            lines.append(f"   Unavailable: {reason}")
        lines.append("")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _resolve_codex_usage_url(base_url: str) -> str:
    normalized = (base_url or "").strip().rstrip("/")
    if not normalized:
        normalized = "https://chatgpt.com/backend-api/codex"
    if normalized.endswith("/api/codex"):
        normalized = normalized[: -len("/api/codex")]
    elif normalized.endswith("/codex"):
        normalized = normalized[: -len("/codex")]
    if "chatgpt.com" in normalized or "chat.openai.com" in normalized:
        if "/backend-api" not in normalized:
            normalized = normalized + "/backend-api"
        return normalized + "/wham/usage"
    if "/backend-api" in normalized:
        return normalized + "/wham/usage"
    return normalized + "/api/codex/usage"


def _codex_account_id_from_token(token: str) -> Optional[str]:
    claims = _decode_jwt_claims(token)
    nested = claims.get("https://api.openai.com/auth")
    if isinstance(nested, dict):
        value = nested.get("chatgpt_account_id") or nested.get("account_id")
        if isinstance(value, str) and value.strip():
            return value.strip()
    for key in ("chatgpt_account_id", "account_id"):
        value = claims.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _parse_codex_usage_snapshot(payload: dict[str, Any]) -> AccountUsageSnapshot:
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


def _fetch_codex_account_usage_for_token(
    access_token: str,
    *,
    base_url: Optional[str] = None,
    account_id: Optional[str] = None,
) -> AccountUsageSnapshot:
    token = str(access_token or "").strip()
    if not token:
        return _account_unavailable_snapshot("openai-codex", "missing access token")
    account_id = (str(account_id or "").strip() or _codex_account_id_from_token(token))
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": "hermes-agent",
    }
    if account_id:
        headers["ChatGPT-Account-Id"] = account_id
    with httpx.Client(timeout=15.0) as client:
        response = client.get(_resolve_codex_usage_url(base_url or ""), headers=headers)
        response.raise_for_status()
    payload = response.json() or {}
    return _parse_codex_usage_snapshot(payload)


def _fetch_codex_account_usage() -> Optional[AccountUsageSnapshot]:
    creds = resolve_codex_runtime_credentials(refresh_if_expiring=True)
    token_data = _read_codex_tokens()
    tokens = token_data.get("tokens") or {}
    account_id = str(
        tokens.get("chatgpt_account_id") or tokens.get("account_id") or ""
    ).strip() or None
    return _fetch_codex_account_usage_for_token(
        str(creds.get("api_key", "") or ""),
        base_url=creds.get("base_url", ""),
        account_id=account_id,
    )


def _fetch_anthropic_account_usage_for_token(token: str) -> Optional[AccountUsageSnapshot]:
    token = (token or "").strip()
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


def _fetch_anthropic_account_usage() -> Optional[AccountUsageSnapshot]:
    return _fetch_anthropic_account_usage_for_token(resolve_anthropic_token() or "")


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


def list_account_provider_choices(*, active_provider: Optional[str] = None) -> list[AccountProviderChoice]:
    """Return providers that have one or more persisted credentials/accounts."""
    choices: dict[str, AccountProviderChoice] = {}
    normalized_active = _normalize_provider(active_provider)
    try:
        pool = read_credential_pool(None)
    except Exception:
        pool = {}
    if isinstance(pool, dict):
        for provider, entries in pool.items():
            normalized = _normalize_provider(provider)
            if not normalized or not isinstance(entries, list) or not entries:
                continue
            choices[normalized] = AccountProviderChoice(
                slug=normalized,
                name=_provider_display_name(normalized),
                account_count=len(entries),
                is_current=normalized == normalized_active,
            )

    # If the active provider is backed by env/config singletons, load_pool()
    # may seed it into auth.json even when it was not in the raw store yet.
    if normalized_active and normalized_active not in choices:
        try:
            seeded_entries = load_pool(normalized_active).entries()
        except Exception:
            seeded_entries = []
        if seeded_entries:
            choices[normalized_active] = AccountProviderChoice(
                slug=normalized_active,
                name=_provider_display_name(normalized_active),
                account_count=len(seeded_entries),
                is_current=True,
            )

    return sorted(
        choices.values(),
        key=lambda choice: (not choice.is_current, choice.name.lower(), choice.slug),
    )


def list_account_usage_providers(*, active_provider: Optional[str] = None) -> list[str]:
    """Return provider slugs that have one or more credentials/accounts."""
    return [choice.slug for choice in list_account_provider_choices(active_provider=active_provider)]


def _entry_label(entry: Any, index: int) -> str:
    for attr in ("label", "source"):
        value = getattr(entry, attr, None)
        if isinstance(value, str) and value.strip():
            return safe_display_text(value, fallback=f"account-{index}", max_len=64)
    value = getattr(entry, "id", None)
    if isinstance(value, str) and value.strip():
        return safe_identifier(value, fallback=f"account-{index}")
    return f"account-{index}"


def _entry_status(entry: Any) -> str:
    status = getattr(entry, "last_status", None)
    if isinstance(status, str) and status.strip():
        return safe_display_text(status, fallback="available", max_len=40)
    return "available"


def _entry_runtime_api_key(entry: Any) -> str:
    for attr in ("runtime_api_key", "access_token", "api_key"):
        value = getattr(entry, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _entry_runtime_base_url(entry: Any) -> Optional[str]:
    for attr in ("runtime_base_url", "base_url", "inference_base_url"):
        value = getattr(entry, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _entry_account_id(entry: Any, token: str) -> Optional[str]:
    for attr in ("account_id", "chatgpt_account_id"):
        value = getattr(entry, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    extra = getattr(entry, "extra", None)
    if isinstance(extra, dict):
        for key in ("account_id", "chatgpt_account_id"):
            value = extra.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    id_token = getattr(entry, "id_token", None)
    if isinstance(id_token, str) and id_token.strip():
        from_id_token = _codex_account_id_from_token(id_token)
        if from_id_token:
            return from_id_token
    return _codex_account_id_from_token(token)


def _coerce_pool(provider: str, pool: Any = None) -> Any:
    if pool is not None:
        return pool
    return load_pool(provider)


def active_provider_account_index(provider: Optional[str], *, pool: Any = None) -> Optional[int]:
    """Return the 1-based index of the currently selected pool entry, if known."""
    normalized = _normalize_provider(provider)
    if not normalized:
        return None
    try:
        resolved_pool = _coerce_pool(normalized, pool)
        if not hasattr(resolved_pool, "current") or not hasattr(resolved_pool, "entries"):
            return None
        entries = list(resolved_pool.entries())
        if not entries:
            return None
        current = resolved_pool.current()
        current_id = getattr(current, "id", None)
        if not isinstance(current_id, str) or not current_id:
            return 1
        for index, entry in enumerate(entries, start=1):
            if getattr(entry, "id", None) == current_id:
                return index
    except Exception:
        return None
    return None


def select_provider_account(
    provider: Optional[str],
    target: str,
    *,
    pool: Any = None,
) -> tuple[Optional[int], Any, Optional[str]]:
    """Select one account in a provider credential pool by index, label, or id."""
    normalized = _normalize_provider(provider)
    if not normalized:
        return None, None, "missing provider"
    try:
        resolved_pool = _coerce_pool(normalized, pool)
    except Exception as exc:
        return None, None, f"could not load {normalized} credential pool: {type(exc).__name__}"
    if not hasattr(resolved_pool, "select_target"):
        return None, None, f"{normalized} credential pool cannot select accounts"
    try:
        index, entry, error = resolved_pool.select_target(target)
    except Exception as exc:
        return None, None, f"could not select account: {type(exc).__name__}"
    return index, entry, error


def account_choice_label(result: ProviderAccountUsage, *, active: bool = False) -> str:
    """Compact one-line label for terminal account pickers."""
    emoji, health, lowest_remaining = _account_health(result.snapshot)
    label = safe_display_text(result.label or f"account-{result.index}", fallback=f"account-{result.index}", max_len=48)
    suffix = "  ← active" if active else ""
    parts: list[str] = [f"{emoji} {result.index}. {label}{suffix}", health]
    if result.status:
        parts.append(f"status: {safe_display_text(result.status, fallback='available', max_len=32)}")
    if result.snapshot and result.snapshot.plan:
        parts.append(f"plan: {safe_display_text(result.snapshot.plan, fallback='plan', max_len=32)}")
    if lowest_remaining is not None:
        parts.append(f"limit: {lowest_remaining}% left {_usage_bar(lowest_remaining, width=8)}")
    if result.snapshot:
        window_bits: list[str] = []
        for window in result.snapshot.windows[:2]:
            remaining = _remaining_percent(window)
            window_label = safe_display_text(window.label, fallback="Limit", max_len=18)
            if remaining is not None and window.used_percent is not None:
                used = max(0, min(100, round(float(window.used_percent))))
                bit = f"{window_label} {remaining}% left / {used}% used"
                if window.reset_at:
                    bit += f" · {format_reset_countdown(window.reset_at)}"
                window_bits.append(bit)
            elif window.detail:
                window_bits.append(f"{window_label} {safe_display_text(window.detail, max_len=40)}")
        if window_bits:
            parts.append("; ".join(window_bits))
        elif result.snapshot.details:
            parts.append(safe_display_text(result.snapshot.details[0], fallback="detail", max_len=60))
    if result.unavailable_reason and not (result.snapshot and result.snapshot.windows):
        parts.append(safe_display_text(result.unavailable_reason, fallback="usage unavailable", max_len=60))
    return " · ".join(part for part in parts if part)


def _sync_entry_for_usage(provider: str, pool: Any, entry: Any) -> tuple[Any, Optional[str]]:
    """Best-effort OAuth sync before informational usage lookup."""
    try:
        if provider == "openai-codex" and (
            getattr(entry, "source", None) == "device_code"
            or str(getattr(entry, "source", "") or "").endswith(":device_code")
        ):
            sync = getattr(pool, "_sync_codex_entry_from_auth_store", None)
            if callable(sync):
                entry = sync(entry)
        elif provider == "anthropic" and getattr(entry, "source", None) == "claude_code":
            sync = getattr(pool, "_sync_anthropic_entry_from_credentials_file", None)
            if callable(sync):
                entry = sync(entry)
    except Exception as exc:
        return entry, f"credential sync failed: {type(exc).__name__}"
    return entry, None


def _fetch_provider_entry_usage(provider: str, index: int, entry: Any, pool: Any = None) -> ProviderAccountUsage:
    label = _entry_label(entry, index)
    status = _entry_status(entry)
    if provider not in _ACCOUNT_USAGE_LOOKUP_PROVIDERS:
        snapshot = _account_unavailable_snapshot(provider, "usage lookup not supported for this provider")
        return ProviderAccountUsage(provider, index, label, status, snapshot, snapshot.unavailable_reason)

    if pool is not None:
        entry, sync_error = _sync_entry_for_usage(provider, pool, entry)
        if sync_error:
            snapshot = _account_unavailable_snapshot(provider, sync_error)
            return ProviderAccountUsage(provider, index, label, status, snapshot, snapshot.unavailable_reason)

    token = _entry_runtime_api_key(entry)
    try:
        if provider == "openai-codex":
            snapshot = _fetch_codex_account_usage_for_token(
                token,
                base_url=_entry_runtime_base_url(entry),
                account_id=_entry_account_id(entry, token),
            )
        elif provider == "anthropic":
            snapshot = _fetch_anthropic_account_usage_for_token(token)
        elif provider == "openrouter":
            snapshot = _fetch_openrouter_account_usage(_entry_runtime_base_url(entry), token)
        else:
            snapshot = None
    except Exception as exc:
        snapshot = _account_unavailable_snapshot(provider, f"usage lookup failed: {type(exc).__name__}")
    if snapshot is None:
        snapshot = _account_unavailable_snapshot(provider, "usage unavailable")
    return ProviderAccountUsage(provider, index, label, status, snapshot, snapshot.unavailable_reason)


def fetch_provider_account_usages(
    provider: Optional[str],
    *,
    timeout: float = 20.0,
    max_concurrency: int = 4,
) -> list[ProviderAccountUsage]:
    """Fetch usage for every credential with bounded partial-failure safety."""
    normalized = _normalize_provider(provider)
    if not normalized:
        return []
    try:
        pool = load_pool(normalized)
        entries = list(pool.entries())
    except Exception:
        pool = None
        entries = []
    if not entries:
        return []

    results: list[Optional[ProviderAccountUsage]] = [None] * len(entries)
    workers = max(1, min(int(max_concurrency or 1), 4, len(entries)))
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
    future_to_pos: dict[concurrent.futures.Future, int] = {}
    try:
        for pos, entry in enumerate(entries):
            future = executor.submit(_fetch_provider_entry_usage, normalized, pos + 1, entry, pool)
            future_to_pos[future] = pos
        done, not_done = concurrent.futures.wait(future_to_pos, timeout=max(0.1, float(timeout)))
        for future in done:
            pos = future_to_pos[future]
            try:
                results[pos] = future.result()
            except Exception as exc:
                entry = entries[pos]
                snapshot = _account_unavailable_snapshot(normalized, f"usage lookup failed: {type(exc).__name__}")
                results[pos] = ProviderAccountUsage(
                    normalized,
                    pos + 1,
                    _entry_label(entry, pos + 1),
                    _entry_status(entry),
                    snapshot,
                    snapshot.unavailable_reason,
                )
        for future in not_done:
            pos = future_to_pos[future]
            future.cancel()
            entry = entries[pos]
            snapshot = _account_unavailable_snapshot(normalized, "usage lookup timed out")
            results[pos] = ProviderAccountUsage(
                normalized,
                pos + 1,
                _entry_label(entry, pos + 1),
                _entry_status(entry),
                snapshot,
                snapshot.unavailable_reason,
            )
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
    return [result for result in results if result is not None]


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

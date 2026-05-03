from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from hermes_cli.auth import _read_codex_tokens, resolve_codex_runtime_credentials

USAGE_URL = "https://chatgpt.com/backend-api/wham/usage"


@dataclass(frozen=True)
class CodexQuotaWindow:
    key: str
    label: str
    used_percent: float
    elapsed_percent: float
    limit_window_seconds: Optional[int] = None
    reset_at: Optional[datetime] = None


@dataclass(frozen=True)
class CodexQuotaResult:
    ok: bool
    fetched_at: datetime
    plan_type: Optional[str] = None
    windows: tuple[CodexQuotaWindow, ...] = ()
    error: Optional[str] = None
    status_code: Optional[int] = None
    payload: Optional[dict[str, Any]] = None


class CodexQuotaError(RuntimeError):
    def __init__(self, message: str, *, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _clamp_percent(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.0
    return max(0.0, min(100.0, numeric))


def parse_reset_at(value: Any) -> Optional[datetime]:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        if numeric <= 0:
            return None
        if numeric > 1_000_000_000_000:
            numeric /= 1000.0
        return datetime.fromtimestamp(numeric, tz=timezone.utc)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return parse_reset_at(float(text))
        except ValueError:
            pass
        try:
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            dt = datetime.fromisoformat(text)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def compute_elapsed_percent(
    *,
    used_percent: float,
    limit_window_seconds: Any,
    reset_at: Optional[datetime],
    now: Optional[datetime] = None,
) -> float:
    try:
        window_seconds = int(limit_window_seconds)
    except (TypeError, ValueError):
        window_seconds = 0
    if window_seconds <= 0 or reset_at is None:
        return _clamp_percent(used_percent)
    current = now or _now()
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    remaining = max(0.0, (reset_at - current).total_seconds())
    elapsed = ((window_seconds - remaining) / window_seconds) * 100.0
    return _clamp_percent(elapsed)


def classify_quota_usage(usage_percent: float, elapsed_percent: float) -> str:
    usage = _clamp_percent(usage_percent)
    elapsed = _clamp_percent(elapsed_percent)
    if usage >= 90:
        return "error"
    if usage >= 75:
        return "warning"
    if usage >= elapsed + 10:
        return "warning"
    if usage >= 50:
        return "accent"
    return "success"


def _bar_cells(usage_percent: float, elapsed_percent: float, width: int) -> tuple[int, int]:
    width = max(1, int(width))
    filled_end = round((_clamp_percent(usage_percent) / 100.0) * width)
    cursor_pos = round((_clamp_percent(elapsed_percent) / 100.0) * (width - 1))
    return max(0, min(width, filled_end)), max(0, min(width - 1, cursor_pos))


def render_quota_bar_text(
    usage_percent: float,
    elapsed_percent: float,
    *,
    width: int = 16,
    fill: str = "━",
    track: str = "━",
    cursor: str = "|",
) -> str:
    filled_end, cursor_pos = _bar_cells(usage_percent, elapsed_percent, width)
    cells: list[str] = []
    for index in range(max(1, int(width))):
        if index == cursor_pos:
            cells.append(cursor)
        elif index < filled_end:
            cells.append(fill)
        else:
            cells.append(track)
    return "".join(cells)


def render_quota_bar_fragments(
    usage_percent: float,
    elapsed_percent: float,
    *,
    width: int = 16,
) -> list[tuple[str, str]]:
    usage_style = classify_quota_usage(usage_percent, elapsed_percent)
    style_map = {
        "error": "class:status-bar-critical",
        "warning": "class:status-bar-bad",
        "accent": "class:status-bar-warn",
        "success": "class:status-bar-good",
    }
    fill_style = style_map[usage_style]
    filled_end, cursor_pos = _bar_cells(usage_percent, elapsed_percent, width)
    fragments: list[tuple[str, str]] = []
    for index in range(max(1, int(width))):
        if index == cursor_pos:
            fragments.append(("class:status-bar-strong", "|"))
        elif index < filled_end:
            fragments.append((fill_style, "━"))
        else:
            fragments.append(("class:status-bar-dim", "━"))
    return fragments


def format_remaining(reset_at: Optional[datetime], *, now: Optional[datetime] = None) -> str:
    if reset_at is None:
        return "unknown"
    current = now or _now()
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    seconds = int((reset_at - current).total_seconds())
    if seconds <= 0:
        return "now"
    minutes = max(1, seconds // 60)
    if minutes >= 24 * 60:
        days = minutes // (24 * 60)
        hours = (minutes % (24 * 60)) // 60
        return f"{days}d{hours}h"
    if minutes >= 60:
        hours = minutes // 60
        mins = minutes % 60
        return f"{hours}h{mins}m"
    return f"{minutes}m"


def _clock_no_leading_zero(dt: datetime) -> str:
    text = dt.strftime("%I:%M%p").lower()
    return text[1:] if text.startswith("0") else text


def format_reset_clock(reset_at: Optional[datetime], *, now: Optional[datetime] = None) -> str:
    if reset_at is None:
        return "unknown"
    current = now or _now()
    local_reset = reset_at.astimezone()
    local_now = current.astimezone()
    clock = _clock_no_leading_zero(local_reset)
    if local_reset.date() == local_now.date():
        return clock
    return f"{local_reset.day} {local_reset.strftime('%b')} {clock}"


def format_window_metadata(
    window: CodexQuotaWindow,
    *,
    now: Optional[datetime] = None,
    compact: bool = False,
) -> str:
    used = round(_clamp_percent(window.used_percent))
    sep = "·" if compact else " · "
    return f"{used}%{sep}{format_remaining(window.reset_at, now=now)}→{format_reset_clock(window.reset_at, now=now)}"


def parse_codex_quota_payload(payload: dict[str, Any], *, now: Optional[datetime] = None) -> CodexQuotaResult:
    rate_limit = payload.get("rate_limit") if isinstance(payload, dict) else None
    if not isinstance(rate_limit, dict):
        rate_limit = {}
    windows: list[CodexQuotaWindow] = []
    for key, label in (("primary_window", "Primary"), ("secondary_window", "Secondary")):
        raw = rate_limit.get(key)
        if not isinstance(raw, dict) or raw.get("used_percent") is None:
            continue
        used = _clamp_percent(raw.get("used_percent"))
        reset_at = parse_reset_at(raw.get("reset_at"))
        limit_window_seconds = raw.get("limit_window_seconds")
        elapsed = compute_elapsed_percent(
            used_percent=used,
            limit_window_seconds=limit_window_seconds,
            reset_at=reset_at,
            now=now,
        )
        try:
            window_seconds = int(limit_window_seconds) if limit_window_seconds is not None else None
        except (TypeError, ValueError):
            window_seconds = None
        windows.append(
            CodexQuotaWindow(
                key=key,
                label=label,
                used_percent=used,
                elapsed_percent=elapsed,
                limit_window_seconds=window_seconds,
                reset_at=reset_at,
            )
        )
    return CodexQuotaResult(
        ok=True,
        fetched_at=now or _now(),
        plan_type=str(payload.get("plan_type") or "").strip() or None,
        windows=tuple(windows),
        payload=payload,
    )


def _codex_usage_headers(api_key: str, account_id: Optional[str]) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "User-Agent": "codex-cli",
    }
    if account_id:
        headers["ChatGPT-Account-Id"] = account_id
    return headers


def _resolve_codex_quota_credentials() -> tuple[str, Optional[str]]:
    """Return (access_token, account_id) from Codex auth, including credential pools."""
    try:
        creds = resolve_codex_runtime_credentials(refresh_if_expiring=True)
        api_key = str(creds.get("api_key") or "").strip()
        if api_key:
            token_data = _read_codex_tokens()
            tokens = token_data.get("tokens") if isinstance(token_data, dict) else None
            account_id = None
            if isinstance(tokens, dict):
                account_id = str(tokens.get("account_id") or "").strip() or None
            return api_key, account_id
    except Exception:
        pass

    try:
        from agent.credential_pool import load_pool

        entry = load_pool("openai-codex").select()
        if entry and entry.runtime_api_key:
            return str(entry.runtime_api_key).strip(), None
    except Exception:
        pass
    return "", None


def fetch_codex_quota(*, timeout: float = 8.0) -> CodexQuotaResult:
    fetched_at = _now()
    api_key, account_id = _resolve_codex_quota_credentials()
    if not api_key:
        return CodexQuotaResult(ok=False, fetched_at=fetched_at, error="no codex auth")

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(USAGE_URL, headers=_codex_usage_headers(api_key, account_id))
        if response.status_code in (401, 403):
            return CodexQuotaResult(ok=False, fetched_at=fetched_at, error="codex auth expired", status_code=response.status_code)
        if response.status_code >= 400:
            return CodexQuotaResult(ok=False, fetched_at=fetched_at, error=f"codex quota: http {response.status_code}", status_code=response.status_code)
        payload = response.json()
        if not isinstance(payload, dict):
            return CodexQuotaResult(ok=False, fetched_at=fetched_at, error="codex quota: malformed response")
        return parse_codex_quota_payload(payload or {}, now=fetched_at)
    except httpx.TimeoutException:
        return CodexQuotaResult(ok=False, fetched_at=fetched_at, error="codex quota: timeout")
    except httpx.HTTPError as exc:
        return CodexQuotaResult(ok=False, fetched_at=fetched_at, error=f"codex quota: {exc.__class__.__name__}")
    except ValueError:
        return CodexQuotaResult(ok=False, fetched_at=fetched_at, error="codex quota: malformed response")


def format_codex_quota_compact(result: Optional[CodexQuotaResult], *, width: int = 12) -> str:
    if result is None:
        return "codex quota --"
    if not result.ok:
        return result.error or "codex quota unavailable"
    if not result.windows:
        return "codex quota unavailable"
    parts = ["codex"]
    for window in result.windows:
        parts.append(render_quota_bar_text(window.used_percent, window.elapsed_percent, width=width))
        parts.append(f"{round(window.used_percent)}%")
    return " ".join(parts)


def format_codex_quota_full(result: Optional[CodexQuotaResult], *, width: int = 16) -> str:
    if result is None:
        return "codex quota unavailable"
    if not result.ok:
        return result.error or "codex quota unavailable"
    if not result.windows:
        return "codex quota unavailable"
    title = "Codex quota"
    if result.plan_type:
        title += f" ({result.plan_type})"
    lines = [title]
    for window in result.windows:
        bar = render_quota_bar_text(window.used_percent, window.elapsed_percent, width=width)
        lines.append(f"  {window.label:<9} {bar} {format_window_metadata(window)}")
    return "\n".join(lines)

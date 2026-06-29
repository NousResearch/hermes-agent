"""Gateway runtime-metadata footer.

Renders a compact footer showing runtime state and account-limit telemetry, then
appends it to the FINAL message of an agent turn when enabled.

Config (``~/.hermes/config.yaml``)::

    display:
      runtime_footer:
        enabled: true
        fields: [model, context, session_limit, weekly_limit, cwd]
        usage_cache_seconds: 300
        usage_timeout_seconds: 2

Per-platform overrides live under ``display.platforms.<platform>.runtime_footer``.
Users can toggle the global setting with ``/footer on|off`` from both the CLI
and any gateway platform.
"""

from __future__ import annotations

import concurrent.futures
import math
import os
import time
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

_DEFAULT_FIELDS: tuple[str, ...] = ("model", "context", "session_limit", "weekly_limit", "cwd")
_SEP = " · "
_USAGE_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="runtime-footer-usage")
_USAGE_CACHE: dict[tuple[str, str, str], tuple[float, Any]] = {}
# When a usage fetch transiently fails (timeout / rate-limit / empty), retry
# again within this many seconds instead of caching the miss for the full TTL.
_USAGE_NEGATIVE_TTL_SECONDS = 20.0


def _home_relative_cwd(cwd: str) -> str:
    """Return *cwd* with ``$HOME`` collapsed to ``~``. Empty string if unset."""
    if not cwd:
        return ""
    try:
        home = os.path.expanduser("~")
        p = os.path.abspath(cwd)
        if home and (p == home or p.startswith(home + os.sep)):
            return "~" + p[len(home):]
        return p
    except Exception:
        return cwd


def _model_short(model: Optional[str]) -> str:
    """Drop ``vendor/`` prefix for readability (``openai/gpt-5.5`` → ``gpt-5.5``)."""
    if not model:
        return ""
    return str(model).rsplit("/", 1)[-1]


def _field_list(value: Any) -> list[str] | None:
    if isinstance(value, list) and value:
        return [str(f) for f in value]
    if isinstance(value, tuple) and value:
        return [str(f) for f in value]
    return None


def resolve_footer_config(
    user_config: dict[str, Any] | None,
    platform_key: str | None = None,
) -> dict[str, Any]:
    """Resolve effective runtime-footer config for *platform_key*.

    Merge order (later wins):
        1. Built-in defaults (enabled=False)
        2. ``display.runtime_footer``
        3. ``display.platforms.<platform_key>.runtime_footer``
    """
    resolved: dict[str, Any] = {
        "enabled": False,
        "fields": list(_DEFAULT_FIELDS),
        "usage_cache_seconds": 300,
        "usage_timeout_seconds": 2,
    }
    cfg = (user_config or {}).get("display") or {}

    def _merge(src: Any) -> None:
        if not isinstance(src, dict):
            return
        if "enabled" in src:
            resolved["enabled"] = bool(src.get("enabled"))
        fields = _field_list(src.get("fields"))
        if fields:
            resolved["fields"] = fields
        for key in ("usage_cache_seconds", "usage_timeout_seconds"):
            if key in src:
                raw_value = src.get(key)
                if raw_value is None:
                    continue
                try:
                    value = float(raw_value)
                    if value >= 0:
                        resolved[key] = value
                except (TypeError, ValueError):
                    pass

    _merge(cfg.get("runtime_footer"))

    if platform_key:
        platforms = cfg.get("platforms") or {}
        plat_cfg = platforms.get(platform_key)
        if isinstance(plat_cfg, dict):
            _merge(plat_cfg.get("runtime_footer"))

    return resolved


def _clamped_pct(numerator: int | float, denominator: int | float) -> int | None:
    try:
        n = float(numerator)
        d = float(denominator)
    except (TypeError, ValueError):
        return None
    if d <= 0 or n < 0 or not math.isfinite(n) or not math.isfinite(d):
        return None
    return max(0, min(100, round((n / d) * 100)))


def _format_count(value: Any) -> str:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return ""
    sign = "-" if n < 0 else ""
    n = abs(n)
    if n < 1_000:
        return f"{sign}{n}"
    if n < 1_000_000:
        if n % 1_000 == 0 or n >= 10_000:
            return f"{sign}{round(n / 1_000):g}k"
        return f"{sign}{n / 1_000:.1f}k"
    if n % 1_000_000 == 0 or n >= 10_000_000:
        return f"{sign}{round(n / 1_000_000):g}M"
    return f"{sign}{n / 1_000_000:.1f}M"


def _format_context(context_tokens: int, context_length: Optional[int]) -> str:
    pct = _clamped_pct(context_tokens, context_length or 0)
    if pct is None:
        return ""
    used = _format_count(context_tokens)
    limit = _format_count(context_length)
    if not used or not limit:
        return ""
    return f"ctx {used}/{limit} {pct}%"


def _window_attr(window: Any, name: str, default: Any = None) -> Any:
    if isinstance(window, dict):
        return window.get(name, default)
    return getattr(window, name, default)


def _parse_dt(value: Any) -> datetime | None:
    if value in {None, ""}:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
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


def _compact_reset(reset_at: Any) -> str:
    dt = _parse_dt(reset_at)
    if not dt:
        return ""
    seconds = math.ceil((dt - datetime.now(timezone.utc)).total_seconds())
    if seconds <= 0:
        return "now"
    if seconds >= 86400:
        return f"{math.ceil(seconds / 86400)}d"
    if seconds >= 3600:
        return f"{math.ceil(seconds / 3600)}h"
    if seconds >= 60:
        return f"{math.ceil(seconds / 60)}m"
    return f"{seconds}s"


def _usage_windows(account_usage: Any) -> list[Any]:
    if not account_usage:
        return []
    windows = _window_attr(account_usage, "windows", ()) or ()
    try:
        return list(windows)
    except TypeError:
        return []


def _select_usage_window(account_usage: Any, kind: str) -> Any | None:
    windows = _usage_windows(account_usage)
    if not windows:
        return None

    def label(window: Any) -> str:
        return str(_window_attr(window, "label", "") or "").strip().lower()

    if kind == "session":
        exact = {"session", "current session", "primary", "primary window", "five hour", "five-hour", "five_hour"}
        for window in windows:
            l = label(window)
            if l in exact:
                return window
        for window in windows:
            l = label(window)
            if "session" in l or "five" in l or "primary" in l:
                return window
        return None

    if kind == "weekly":
        exact = {"week", "weekly", "current week", "secondary", "secondary window", "seven day", "seven-day", "seven_day"}
        for window in windows:
            l = label(window)
            if l in exact:
                return window
        for window in windows:
            l = label(window)
            if ("week" in l or "seven" in l or "secondary" in l) and "opus" not in l and "sonnet" not in l:
                return window
        for window in windows:
            l = label(window)
            if "week" in l or "seven" in l or "secondary" in l:
                return window
    return None


def _format_limit_window(account_usage: Any, *, kind: str) -> str:
    window = _select_usage_window(account_usage, kind)
    if not window:
        return ""
    used_raw = _window_attr(window, "used_percent")
    try:
        used = float(used_raw)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(used):
        return ""
    remaining = max(0, min(100, round(100 - used)))
    prefix = "sess" if kind == "session" else "week"
    text = f"{prefix} {remaining}% left"
    reset = _compact_reset(_window_attr(window, "reset_at") or _window_attr(window, "resets_at"))
    if reset:
        text += f"/{reset}"
    return text


def _format_tokens(*, input_tokens: Any = None, output_tokens: Any = None, total_tokens: Any = None) -> str:
    total = _format_count(total_tokens)
    inp = _format_count(input_tokens)
    out = _format_count(output_tokens)
    if total and inp and out:
        return f"tok {inp}→{out}/{total}"
    if total:
        return f"tok {total}"
    return ""


def _format_cost(estimated_cost_usd: Any = None, cost_status: Optional[str] = None) -> str:
    try:
        cost = float(estimated_cost_usd)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(cost) or cost < 0:
        return ""
    if cost == 0 and cost_status and str(cost_status).lower() not in {"ok", "estimated", ""}:
        return ""
    if cost < 0.01:
        return f"${cost:.4f}"
    return f"${cost:.2f}"


def _needs_account_usage(fields: Iterable[str]) -> bool:
    wanted = {str(field) for field in fields}
    return bool(wanted & {"session_limit", "weekly_limit"})


def _config_model_value(user_config: dict[str, Any] | None, key: str) -> Any | None:
    """Return a model-route value from config.yaml when the agent result omitted it.

    Gateway footer rendering runs after the agent loop and should prefer the
    chat route from ``agent_result``.  Some older/fallback result paths may not
    include provider/base_url though, which prevents account-usage lookup and
    silently drops ``sess``/``week``.  Falling back to the configured model route
    keeps the footer resilient without exposing secrets.
    """
    model_cfg = (user_config or {}).get("model")
    if isinstance(model_cfg, dict):
        value = model_cfg.get(key)
        return None if value is None or value == "" else value
    return None


def _route_value_missing(value: Any) -> bool:
    return str(value or "").strip().lower() in {"", "auto", "custom"}


def _snapshot_is_definitive(snapshot: Any) -> bool:
    """True when a snapshot is a stable answer worth caching for the full TTL.

    A snapshot with real limit windows is good data. A snapshot that explicitly
    reports unavailability (e.g. a non-OAuth account) is also a definitive
    answer — not a transient miss — so it should be cached normally instead of
    hammering the provider every turn. Everything else (None, an exception, or
    an empty snapshot without a reason) is treated as a transient failure.
    """
    if snapshot is None:
        return False
    if getattr(snapshot, "windows", ()):
        return True
    return bool(getattr(snapshot, "unavailable_reason", None))


def _get_account_usage_cached(
    *,
    provider: Optional[str],
    base_url: Optional[str],
    api_key: Optional[str],
    cache_seconds: float,
    timeout_seconds: float,
) -> Any | None:
    provider_key = str(provider or "").strip().lower()
    if provider_key in {"", "auto", "custom"}:
        return None
    now = time.monotonic()
    key = (provider_key, str(base_url or ""), "api_key" if api_key else "")
    cached = _USAGE_CACHE.get(key)
    # Cache value is (expires_at_monotonic, snapshot).
    if cached and now < cached[0]:
        return cached[1]
    # Last snapshot that actually carried limit data — used to ride out a
    # transient miss instead of blanking the footer for the whole cache window.
    last_good = cached[1] if (cached and getattr(cached[1], "windows", ())) else None
    if timeout_seconds <= 0:
        return last_good
    try:
        from agent.account_usage import fetch_account_usage

        future = _USAGE_EXECUTOR.submit(
            fetch_account_usage,
            provider_key,
            base_url=base_url,
            api_key=api_key,
        )
        snapshot = future.result(timeout=timeout_seconds)
    except Exception:
        snapshot = None
    if _snapshot_is_definitive(snapshot):
        ttl = cache_seconds if cache_seconds > 0 else 0.0
        _USAGE_CACHE[key] = (now + ttl, snapshot)
        return snapshot
    # Transient miss (timeout / None / empty): keep serving the last good
    # snapshot and retry again soon rather than caching emptiness for 5 min.
    neg_ttl = min(_USAGE_NEGATIVE_TTL_SECONDS, cache_seconds) if cache_seconds > 0 else 0.0
    _USAGE_CACHE[key] = (now + neg_ttl, last_good)
    return last_good


def format_runtime_footer(
    *,
    model: Optional[str],
    context_tokens: int,
    context_length: Optional[int],
    cwd: Optional[str] = None,
    fields: Iterable[str] = _DEFAULT_FIELDS,
    account_usage: Any | None = None,
    input_tokens: Any | None = None,
    output_tokens: Any | None = None,
    total_tokens: Any | None = None,
    estimated_cost_usd: Any | None = None,
    cost_status: Optional[str] = None,
) -> str:
    """Render the footer line, or return "" if no fields have data.

    Fields are skipped silently when their underlying data is missing — a
    partially-populated footer is better than a line with ``?%`` or empty slots.
    """
    parts: list[str] = []
    for field in fields:
        field = str(field)
        if field == "model":
            m = _model_short(model)
            if m:
                parts.append(m)
        elif field == "context_pct":
            pct = _clamped_pct(context_tokens, context_length or 0)
            if pct is not None:
                parts.append(f"{pct}%")
        elif field in {"context", "ctx"}:
            ctx = _format_context(context_tokens, context_length)
            if ctx:
                parts.append(ctx)
        elif field == "session_limit":
            sess = _format_limit_window(account_usage, kind="session")
            if sess:
                parts.append(sess)
        elif field == "weekly_limit":
            week = _format_limit_window(account_usage, kind="weekly")
            if week:
                parts.append(week)
        elif field == "tokens":
            tok = _format_tokens(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
            )
            if tok:
                parts.append(tok)
        elif field == "cost":
            cost = _format_cost(estimated_cost_usd, cost_status)
            if cost:
                parts.append(cost)
        elif field == "cwd":
            rel = _home_relative_cwd(cwd or os.environ.get("TERMINAL_CWD", ""))
            if rel:
                parts.append(rel)
        # Unknown field names are silently ignored.

    if not parts:
        return ""
    return _SEP.join(parts)


def build_footer_line(
    *,
    user_config: dict[str, Any] | None,
    platform_key: str | None,
    model: Optional[str],
    context_tokens: int,
    context_length: Optional[int],
    cwd: Optional[str] = None,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    input_tokens: Any | None = None,
    output_tokens: Any | None = None,
    total_tokens: Any | None = None,
    estimated_cost_usd: Any | None = None,
    cost_status: Optional[str] = None,
    account_usage: Any | None = None,
) -> str:
    """Top-level entry point used by gateway/run.py.

    Returns the footer text (empty string when disabled or no data). Callers
    append this to the final response themselves, preserving a single blank
    line of separation.
    """
    cfg = resolve_footer_config(user_config, platform_key)
    if not cfg.get("enabled"):
        return ""
    fields = cfg.get("fields") or _DEFAULT_FIELDS
    if account_usage is None and _needs_account_usage(fields):
        if _route_value_missing(provider):
            provider = _config_model_value(user_config, "provider")
        if _route_value_missing(base_url):
            base_url = _config_model_value(user_config, "base_url")
        if not api_key:
            api_key = _config_model_value(user_config, "api_key")
        account_usage = _get_account_usage_cached(
            provider=provider,
            base_url=base_url,
            api_key=api_key,
            cache_seconds=float(cfg.get("usage_cache_seconds", 300) or 0),
            timeout_seconds=float(cfg.get("usage_timeout_seconds", 2) or 0),
        )
    return format_runtime_footer(
        model=model,
        context_tokens=context_tokens,
        context_length=context_length,
        cwd=cwd,
        fields=fields,
        account_usage=account_usage,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        estimated_cost_usd=estimated_cost_usd,
        cost_status=cost_status,
    )

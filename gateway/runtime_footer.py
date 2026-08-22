"""Gateway runtime-metadata footer.

Renders a compact footer showing runtime state (model, context %, cwd) and
appends it to the FINAL message of an agent turn when enabled.  Off by default
to keep replies minimal.

Config (``~/.hermes/config.yaml``)::

    display:
      runtime_footer:
        enabled: true                       # off by default
        fields: [model, context_pct, cwd]   # order shown; drop any to hide

Available fields:
    model        — bare model id, vendor prefix dropped (``gpt-5.4``)
    context_pct  — last-call context occupancy as a percent (``5%``)
    latency      — wall-clock duration of the turn (``22s``, ``1m05s``)
    cwd          — home-relative working dir (``~``)
    rate_tier    — billing tier for the current hour (``peak`` / ``off-peak``)
                   for providers with time-of-use pricing, e.g. DeepSeek's
                   peak/off-peak rates. Config-driven and model-agnostic — see
                   ``rate_windows`` below. Skipped silently when the active
                   model matches no configured window.

``latency`` and ``rate_tier`` are opt-in: they are NOT in the default field
set, so a footer whose ``fields`` are unset renders exactly as before.

``rate_windows`` (optional) — time-of-use pricing windows, keyed by model
substring (matched case-insensitively against the bare model id). Peak hours
are half-open ``[start, end)`` in the window's timezone::

    display:
      runtime_footer:
        enabled: true
        fields: [model, context_pct, rate_tier, cwd]
        rate_windows:
          deepseek:                       # model matcher
            tz: Asia/Singapore            # optional, default UTC
            peak: [[9, 12], [14, 18]]     # half-open hours in that tz

Built-in default (used when the key is absent): DeepSeek's published windows
(01-04 and 06-10 UTC). User entries deep-merge over the default, so a partial
``rate_windows`` map keeps the DeepSeek default for any matcher not listed.

Per-platform overrides live under ``display.platforms.<platform>.runtime_footer``.
Users can toggle the global setting with ``/footer on|off`` from both the CLI
and any gateway platform.

The footer is appended to the final response text in ``gateway/run.py`` right
before returning the response to the adapter send path — so it only lands on
the final message a user sees, not on tool-progress updates or streaming
partials.  When streaming is on and the final text has already been delivered
piecemeal, the footer is sent as a separate trailing message via
``send_trailing_footer()``.
"""

from __future__ import annotations

import datetime as _dt
import os
from typing import Any, Iterable, Optional

try:
    from zoneinfo import ZoneInfo as _ZoneInfo
except ImportError:  # pragma: no cover - py<3.9 fallback
    _ZoneInfo = None

_DEFAULT_FIELDS: tuple[str, ...] = ("model", "context_pct", "cwd")
_SEP = " · "

# Built-in time-of-use pricing windows. Keyed by model substring; peak hours
# are half-open [start, end) in the entry's timezone. Source for the DeepSeek
# default: https://api-docs.deepseek.com/quick_start/pricing (verified
# 2026-08-20) — peak 01:00-04:00 and 06:00-10:00 UTC, off-peak is half price.
_DEFAULT_RATE_WINDOWS: dict[str, dict[str, Any]] = {
    "deepseek": {
        "tz": "UTC",
        "peak": [(1, 4), (6, 10)],
    },
}


def _home_relative_cwd(cwd: str) -> str:
    """Return *cwd* with ``$HOME`` collapsed to ``~``.  Empty string if unset."""
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
    """Drop ``vendor/`` prefix for readability (``openai/gpt-5.4`` → ``gpt-5.4``)."""
    if not model:
        return ""
    return model.rsplit("/", 1)[-1]


# DeepSeek peak/off-peak billing windows, in UTC (source: api-docs.deepseek.com
# /quick_start/pricing, verified 2026-08-20). Peak hours are 01:00-04:00 and
# 06:00-10:00 UTC; all other hours are off-peak at half the price. SGT (UTC+8)
# equivalents: 09:00-12:00 and 14:00-18:00. Kept for backward-compatible
# helpers; the config-driven path is ``_DEFAULT_RATE_WINDOWS``.
_DEEPSEEK_PEAK_WINDOWS_UTC: tuple[tuple[int, int], ...] = (
    (1, 4),   # 01:00-04:00 UTC → 09:00-12:00 SGT
    (6, 10),  # 06:00-10:00 UTC → 14:00-18:00 SGT
)


def _is_deepseek_model(model: Optional[str]) -> bool:
    """True when *model* is a DeepSeek-hosted model (any vendor prefix form)."""
    return bool(model) and ("deepseek" in model.lower())


def deepseek_rate_tier(now_utc: Optional[_dt.datetime] = None) -> str:
    """Return the DeepSeek billing tier for the current UTC hour: ``peak`` or ``off-peak``.

    Peak windows per DeepSeek's published pricing (see ``_DEEPSEEK_PEAK_WINDOWS_UTC``).
    Uses UTC so the tier is independent of the host's local timezone.
    """
    now = now_utc or _dt.datetime.now(_dt.timezone.utc)
    hour = now.hour
    for start, end in _DEEPSEEK_PEAK_WINDOWS_UTC:
        if start <= hour < end:
            return "peak"
    return "off-peak"


def _match_rate_windows(
    model: Optional[str],
    rate_windows: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    """Return the first configured rate window whose key matches *model*.

    Keys are matched case-insensitively as substrings against the bare model
    id (vendor prefix stripped). None when no window matches — the caller
    then skips the ``rate_tier`` field silently.
    """
    if not model:
        return None
    windows = rate_windows or _DEFAULT_RATE_WINDOWS
    bare = model.rsplit("/", 1)[-1].lower()
    for key, win in windows.items():
        if key.lower() in bare:
            return win
    return None


def _hour_in_tz(now: _dt.datetime, tz_name: str) -> int:
    """Local hour of *now* in *tz_name*; falls back to UTC on bad tz data."""
    if _ZoneInfo is not None:
        try:
            return now.astimezone(_ZoneInfo(tz_name)).hour
        except Exception:
            pass
    return now.astimezone(_dt.timezone.utc).hour


def rate_tier_for_model(
    model: Optional[str],
    rate_windows: Optional[dict[str, Any]] = None,
    now: Optional[_dt.datetime] = None,
) -> Optional[str]:
    """Return ``peak``/``off-peak`` for *model* per configured windows, or None.

    Provider-agnostic: the window set comes from ``rate_windows`` (or the
    built-in ``_DEFAULT_RATE_WINDOWS`` DeepSeek default when none supplied).
    Returns None when the model matches no window — the footer field is then
    silently skipped.
    """
    win = _match_rate_windows(model, rate_windows)
    if win is None:
        return None
    now = now or _dt.datetime.now(_dt.timezone.utc)
    hour = _hour_in_tz(now, str(win.get("tz") or "UTC"))
    for start, end in win.get("peak") or ():
        try:
            start_i, end_i = int(start), int(end)
        except (TypeError, ValueError):
            continue
        if start_i <= hour < end_i:
            return "peak"
    return "off-peak"


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
    resolved = {
        "enabled": False,
        "fields": list(_DEFAULT_FIELDS),
        "rate_windows": _merge_rate_windows(None),
    }
    cfg = (user_config or {}).get("display") or {}

    global_cfg = cfg.get("runtime_footer")
    if isinstance(global_cfg, dict):
        if "enabled" in global_cfg:
            resolved["enabled"] = bool(global_cfg.get("enabled"))
        if isinstance(global_cfg.get("fields"), list) and global_cfg["fields"]:
            resolved["fields"] = [str(f) for f in global_cfg["fields"]]
        if isinstance(global_cfg.get("rate_windows"), dict):
            resolved["rate_windows"] = _merge_rate_windows(global_cfg["rate_windows"])

    if platform_key:
        platforms = cfg.get("platforms") or {}
        plat_cfg = platforms.get(platform_key)
        if isinstance(plat_cfg, dict):
            plat_footer = plat_cfg.get("runtime_footer")
            if isinstance(plat_footer, dict):
                if "enabled" in plat_footer:
                    resolved["enabled"] = bool(plat_footer.get("enabled"))
                if isinstance(plat_footer.get("fields"), list) and plat_footer["fields"]:
                    resolved["fields"] = [str(f) for f in plat_footer["fields"]]
                if isinstance(plat_footer.get("rate_windows"), dict):
                    resolved["rate_windows"] = _merge_rate_windows(
                        plat_footer["rate_windows"]
                    )

    return resolved


def _merge_rate_windows(
    user_windows: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """Merge user ``rate_windows`` over the built-in defaults (deep, per key).

    A partial user map keeps the DeepSeek default for matchers not listed.
    Non-dict values are replaced wholesale (a user entry is authoritative).
    """
    merged: dict[str, Any] = {}
    for key, val in _DEFAULT_RATE_WINDOWS.items():
        merged[key] = dict(val) if isinstance(val, dict) else val
    if not isinstance(user_windows, dict):
        return merged
    for key, val in user_windows.items():
        if isinstance(val, dict) and isinstance(merged.get(key), dict):
            base = dict(merged[key])
            base.update({k: v for k, v in val.items() if v is not None})
            merged[key] = base
        else:
            merged[key] = val
    return merged


def _format_latency(seconds: float) -> str:
    """Humanize a turn duration: ``<1s``, ``22s``, ``1m05s``."""
    if seconds < 1:
        return "<1s"
    total = int(round(seconds))
    if total < 60:
        return f"{total}s"
    m, sec = divmod(total, 60)
    return f"{m}m{sec:02d}s"


def format_runtime_footer(
    *,
    model: Optional[str],
    context_tokens: int,
    context_length: Optional[int],
    cwd: Optional[str] = None,
    turn_seconds: Optional[float] = None,
    fields: Iterable[str] = _DEFAULT_FIELDS,
    rate_windows: Optional[dict[str, Any]] = None,
) -> str:
    """Render the footer line, or return "" if no fields have data.

    Fields are skipped silently when their underlying data is missing — a
    partially-populated footer is better than a line with ``?%`` or empty slots.
    ``rate_windows`` is the resolved time-of-use window map (defaults merged);
    pass ``None`` to use the built-in DeepSeek default.
    """
    parts: list[str] = []
    for field in fields:
        if field == "model":
            m = _model_short(model)
            if m:
                parts.append(m)
        elif field == "context_pct":
            if context_length and context_length > 0 and context_tokens >= 0:
                pct = max(0, min(100, round((context_tokens / context_length) * 100)))
                parts.append(f"{pct}%")
        elif field == "latency":
            # Wall-clock turn duration. Skipped when the caller supplied no
            # timing (call sites that don't measure) or the value is negative.
            if turn_seconds is not None and turn_seconds >= 0:
                parts.append(_format_latency(turn_seconds))
        elif field == "cwd":
            rel = _home_relative_cwd(cwd or os.environ.get("TERMINAL_CWD", ""))
            if rel:
                parts.append(rel)
        elif field == "rate_tier":
            # Time-of-use billing awareness (e.g. DeepSeek peak/off-peak).
            # Rendered only when the active model matches a configured window.
            tier = rate_tier_for_model(model, rate_windows)
            if tier is not None:
                parts.append(tier)
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
    turn_seconds: Optional[float] = None,
) -> str:
    """Top-level entry point used by gateway/run.py.

    Returns the footer text (empty string when disabled or no data).  Callers
    append this to the final response themselves, preserving a single blank
    line of separation.

    ``turn_seconds`` is the wall-clock duration of the agent run, measured by
    the caller with ``time.monotonic()``.  Callers that don't measure it leave
    it ``None`` and the ``latency`` field is skipped.
    """
    cfg = resolve_footer_config(user_config, platform_key)
    if not cfg.get("enabled"):
        return ""
    return format_runtime_footer(
        model=model,
        context_tokens=context_tokens,
        context_length=context_length,
        cwd=cwd,
        turn_seconds=turn_seconds,
        fields=cfg.get("fields") or _DEFAULT_FIELDS,
        rate_windows=cfg.get("rate_windows"),
    )

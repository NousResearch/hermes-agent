"""Gateway runtime-metadata footer.

Renders a compact footer showing runtime state (model, context %, cwd, timing)
and appends it to the FINAL message of an agent turn when enabled. Off by
default to keep replies minimal.

Config (``~/.hermes/config.yaml``)::

    display:
      runtime_footer:
        enabled: true
        fields: [model, context_pct, cwd, turn_time, api_time, tool_time, overhead_time, api_calls]

Per-platform overrides live under ``display.platforms.<platform>.runtime_footer``.
Users can toggle the global setting with ``/footer on|off`` from both the CLI
and any gateway platform.

The footer is appended to the final response text in ``gateway/run.py`` right
before returning the response to the adapter send path, so it only lands on
the final message a user sees, not on tool-progress updates or streaming
partials. When streaming is on and the final text has already been delivered
piecemeal, the footer is sent as a separate trailing message via
``send_trailing_footer()``.
"""

from __future__ import annotations

import os
from typing import Any, Iterable, Optional

_DEFAULT_FIELDS: tuple[str, ...] = ("model", "context_pct", "cwd")
_SEP = " · "


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
    """Drop ``vendor/`` prefix for readability (``openai/gpt-5.4`` → ``gpt-5.4``)."""
    if not model:
        return ""
    return model.rsplit("/", 1)[-1]


def _format_duration(seconds: Optional[float]) -> str:
    """Human-friendly duration: 0.4s, 12s, 2m03s, 1h23m."""
    if seconds is None or seconds < 0:
        return ""
    if seconds < 10:
        return f"{seconds:.1f}s"
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


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
    resolved = {"enabled": False, "fields": list(_DEFAULT_FIELDS)}
    cfg = (user_config or {}).get("display") or {}

    global_cfg = cfg.get("runtime_footer")
    if isinstance(global_cfg, dict):
        if "enabled" in global_cfg:
            resolved["enabled"] = bool(global_cfg.get("enabled"))
        fields = global_cfg.get("fields")
        if isinstance(fields, list) and fields:
            resolved["fields"] = [str(f) for f in fields]

    if platform_key:
        platforms = cfg.get("platforms") or {}
        plat_cfg = platforms.get(platform_key)
        if isinstance(plat_cfg, dict):
            plat_footer = plat_cfg.get("runtime_footer")
            if isinstance(plat_footer, dict):
                if "enabled" in plat_footer:
                    resolved["enabled"] = bool(plat_footer.get("enabled"))
                fields = plat_footer.get("fields")
                if isinstance(fields, list) and fields:
                    resolved["fields"] = [str(f) for f in fields]

    return resolved


def format_runtime_footer(
    *,
    model: Optional[str],
    context_tokens: int,
    context_length: Optional[int],
    cwd: Optional[str] = None,
    turn_time: Optional[float] = None,
    api_time: Optional[float] = None,
    tool_time: Optional[float] = None,
    api_calls: Optional[int] = None,
    fields: Iterable[str] = _DEFAULT_FIELDS,
) -> str:
    """Render the footer line, or return "" if no fields have data.

    Fields are skipped silently when their underlying data is missing. A
    partially-populated footer is better than a line with ``?%`` or empty slots.
    """
    parts: list[str] = []
    overhead_time: Optional[float] = None
    if turn_time is not None and api_time is not None and tool_time is not None:
        overhead_time = max(
            0.0,
            float(turn_time) - float(api_time or 0.0) - float(tool_time or 0.0),
        )

    for field in fields:
        if field == "model":
            m = _model_short(model)
            if m:
                parts.append(m)
        elif field == "context_pct":
            if context_length and context_length > 0 and context_tokens >= 0:
                pct = max(0, min(100, round((context_tokens / context_length) * 100)))
                parts.append(f"{pct}%")
        elif field == "cwd":
            rel = _home_relative_cwd(cwd or os.environ.get("TERMINAL_CWD", ""))
            if rel:
                parts.append(rel)
        elif field == "turn_time":
            t = _format_duration(turn_time)
            if t:
                parts.append(t)
        elif field == "api_time":
            t = _format_duration(api_time)
            if t:
                parts.append(f"api {t}")
        elif field == "tool_time":
            t = _format_duration(tool_time)
            if t:
                parts.append(f"tools {t}")
        elif field == "overhead_time":
            t = _format_duration(overhead_time)
            if t:
                parts.append(f"other {t}")
        elif field == "api_calls":
            if api_calls is not None and api_calls > 0:
                label = "call" if api_calls == 1 else "calls"
                parts.append(f"{api_calls} {label}")
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
    turn_time: Optional[float] = None,
    api_time: Optional[float] = None,
    tool_time: Optional[float] = None,
    api_calls: Optional[int] = None,
) -> str:
    """Top-level entry point used by gateway/run.py.

    Returns the footer text (empty string when disabled or no data). Callers
    append this to the final response themselves, preserving a single blank
    line of separation.
    """
    cfg = resolve_footer_config(user_config, platform_key)
    if not cfg.get("enabled"):
        return ""
    return format_runtime_footer(
        model=model,
        context_tokens=context_tokens,
        context_length=context_length,
        cwd=cwd,
        turn_time=turn_time,
        api_time=api_time,
        tool_time=tool_time,
        api_calls=api_calls,
        fields=cfg.get("fields") or _DEFAULT_FIELDS,
    )

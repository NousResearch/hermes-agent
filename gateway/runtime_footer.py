"""Gateway runtime-metadata footer.

Renders a compact footer showing runtime state (model, context %, cwd) and
appends it to the FINAL message of an agent turn when enabled.  Off by default
to keep replies minimal.

Config (``~/.hermes/config.yaml``)::

    display:
      runtime_footer:
        enabled: true                       # off by default
        fields: [model, context_pct, quota, cwd]   # order shown; drop any to hide

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

import os
from datetime import datetime
from typing import Any, Iterable, Optional

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover - Python without zoneinfo
    ZoneInfo = None

_DEFAULT_FIELDS: tuple[str, ...] = ("model", "context_pct", "quota", "cwd")
_SEP = " · "


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


def _quota_window_label(label: str) -> str:
    """Human-friendly quota-window labels for the multiline footer."""
    text = str(label or "").strip()
    lowered = text.lower()
    if not lowered:
        return ""
    if lowered in {"current session", "session", "primary window"}:
        return "5h"
    if lowered in {"current week", "weekly", "secondary window"}:
        return "7d"
    if lowered.endswith(" week"):
        prefix = lowered[:-5].replace("current", "").strip()
        return f"{prefix.title()} 7d" if prefix else "7d"
    return text[:24]


def _quota_reset_short(value: Any) -> str:
    """Compact Eastern reset timestamp for a quota window."""
    if value is None:
        return ""
    try:
        dt = value
        if not isinstance(dt, datetime):
            return ""
        if ZoneInfo is not None:
            dt = dt.astimezone(ZoneInfo("America/Toronto"))
        else:
            dt = dt.astimezone()
        return dt.strftime("%b %-d %-I:%M %p %Z")
    except Exception:
        try:
            return value.strftime("%b %d %I:%M %p %Z")
        except Exception:
            return ""


def _quota_block(snapshot: Any) -> str:
    """Render provider quota/limit windows as a readable multiline block."""
    if not snapshot:
        return ""
    try:
        if getattr(snapshot, "unavailable_reason", None):
            return ""
        windows = list(getattr(snapshot, "windows", ()) or ())
    except Exception:
        return ""
    lines: list[str] = []
    for window in windows:
        try:
            used = getattr(window, "used_percent", None)
            if used is None:
                continue
            label = _quota_window_label(getattr(window, "label", ""))
            if not label:
                continue
            pct = max(0, min(100, round(float(used))))
            line = f"{label} - {pct}%"
            detail = str(getattr(window, "detail", "") or "").strip()
            if detail:
                line += f" - {detail}"
            reset = _quota_reset_short(getattr(window, "reset_at", None))
            if reset:
                line += f" - Reset {reset}"
            lines.append(line)
        except Exception:
            continue
        if len(lines) >= 3:
            break
    if not lines:
        return ""
    return "Quota Used:\n" + "\n".join(lines)


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
        if isinstance(global_cfg.get("fields"), list) and global_cfg["fields"]:
            resolved["fields"] = [str(f) for f in global_cfg["fields"]]

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

    return resolved


def format_runtime_footer(
    *,
    model: Optional[str],
    context_tokens: int,
    context_length: Optional[int],
    quota_snapshot: Any = None,
    cwd: Optional[str] = None,
    fields: Iterable[str] = _DEFAULT_FIELDS,
) -> str:
    """Render the footer line, or return "" if no fields have data.

    Fields are skipped silently when their underlying data is missing — a
    partially-populated footer is better than a line with ``?%`` or empty slots.
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
        elif field == "quota":
            q = _quota_block(quota_snapshot)
            if q:
                parts.append(q)
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
    quota_snapshot: Any = None,
    cwd: Optional[str] = None,
) -> str:
    """Top-level entry point used by gateway/run.py.

    Returns the footer text (empty string when disabled or no data).  Callers
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
        quota_snapshot=quota_snapshot,
        cwd=cwd,
        fields=cfg.get("fields") or _DEFAULT_FIELDS,
    )

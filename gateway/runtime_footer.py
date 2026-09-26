"""Gateway runtime-metadata footer (model · context % · cwd), off by default to keep replies
minimal. Config: ``display.runtime_footer: {enabled: bool, fields: [model, context_pct, cwd],
underline: bool}`` (order shown; drop any to hide), per-platform override
``display.platforms.<p>.runtime_footer``, toggled by ``/footer on|off``. Fields: ``model`` (vendor
prefix dropped), ``context_pct`` (last-call occupancy), ``context`` (compact ``ctx used/limit``),
``latency`` (turn wall-clock, opt-in — NOT in the default set so an unset ``fields`` renders exactly
as before), ``served_model`` (opt-in, ``alias → served``: the deployment a routing proxy reported via
``x-litellm-model-id`` / ``x-litellm-model-api-base``, or Hermes' own fallback route; skipped when
the served model is the requested one), ``reasoning`` (compact reasoning-effort label), ``provider``
(inference provider id), ``account`` (account/plan label, any ``<provider>-`` prefix stripped), and
``quota`` (one part per account-usage window — ``5h``/``7d N%`` plus a compact reset — followed by a
compact balance line when the snapshot carries one). ``underline: true`` prefixes the footer with a
separator line. ``gateway/run.py`` appends the footer to the final response only (never to
tool-progress or streaming partials); when streaming already delivered the text, it goes out as a
trailing message via ``send_trailing_footer()``."""

from __future__ import annotations

import math
import os
import re
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

_DEFAULT_FIELDS: tuple[str, ...] = ("model", "context_pct", "cwd")
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


def _env_cwd() -> str:
    try:
        from tools.terminal_scope import terminal_env
    except ImportError:
        return os.environ.get("TERMINAL_CWD", "")
    return terminal_env("TERMINAL_CWD", "")


# Compact labels for agent.reasoning_effort / runtime reasoning_config.
# Keep these short so the footer stays one line on mobile clients.
_REASONING_ABBREV = {
    "none": "off",
    "minimal": "min",
    "low": "low",
    "medium": "med",
    "high": "high",
    "xhigh": "xhi",
    "max": "max",
    "ultra": "ult",
}


def _reasoning_short(effort: Optional[str]) -> str:
    """Return a compact reasoning-effort label, or "" when unknown/empty."""
    raw = str(effort or "").strip().lower()
    if not raw:
        return ""
    if raw in {"false", "disabled", "off"}:
        return _REASONING_ABBREV["none"]
    if raw in _REASONING_ABBREV:
        return _REASONING_ABBREV[raw]
    # Unknown but non-empty values still surface compactly so a new level
    # is visible before we teach the map about it.
    return raw[:6]


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
    resolved = {"enabled": False, "fields": list(_DEFAULT_FIELDS), "underline": False}
    cfg = (user_config or {}).get("display") or {}

    global_cfg = cfg.get("runtime_footer")
    if isinstance(global_cfg, dict):
        if "enabled" in global_cfg:
            resolved["enabled"] = bool(global_cfg.get("enabled"))
        if "underline" in global_cfg:
            resolved["underline"] = bool(global_cfg.get("underline"))
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
                if "underline" in plat_footer:
                    resolved["underline"] = bool(plat_footer.get("underline"))
                if isinstance(plat_footer.get("fields"), list) and plat_footer["fields"]:
                    resolved["fields"] = [str(f) for f in plat_footer["fields"]]

    return resolved


def _format_latency(seconds: float) -> str:
    """Humanize a turn duration: ``<1s``, ``22s``, ``1m05s``."""
    if seconds < 1:
        return "<1s"
    total = int(round(seconds))
    if total < 60:
        return f"{total}s"
    m, sec = divmod(total, 60)
    return f"{m}m{sec:02d}s"


def _compact_number(value: int | float) -> str:
    try:
        n = float(value)
    except Exception:
        return str(value)
    if abs(n) >= 1_000_000:
        text = f"{n / 1_000_000:.1f}M"
    elif abs(n) >= 1_000:
        text = f"{n / 1_000:.1f}K"
    else:
        text = str(int(n))
    return text.replace(".0K", "K").replace(".0M", "M")


def _compact_reset(dt: Any) -> str:
    if not dt:
        return ""
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt.strip().replace("Z", "+00:00"))
        except Exception:
            return ""
    if not isinstance(dt, datetime):
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    seconds = int((dt - datetime.now(timezone.utc)).total_seconds())
    if seconds <= 0:
        return "now"
    hours, rem = divmod(seconds, 3600)
    minutes = rem // 60
    if hours >= 24:
        days, rem_hours = divmod(math.ceil(seconds / 3600), 24)
        return f"{days}d" + (f"{rem_hours}h" if rem_hours else "")
    if hours > 0:
        return f"{hours}h" + (f"{minutes}m" if minutes else "")
    return f"{minutes}m" if minutes else "<1m"


def _quota_label(window: Any, provider: Optional[str] = None, model: Optional[str] = None) -> str:
    """Return a compact quota-window label for footer display.

    The detailed ``/usage`` command keeps provider wording.  The footer is space
    constrained, so normalize the common OAuth/Codex rolling windows to the
    short labels users expect while leaving unknown provider windows intact.
    """
    raw = str(getattr(window, "label", "") or "quota").strip() or "quota"
    label = raw.lower().replace("_", "-")
    model_text = str(model or "").lower()

    if "opus" in label:
        return "opus7d"
    if "sonnet" in label:
        return "sonnet7d"
    if label in {
        "5h",
        "5-hour",
        "5 hour",
        "five-hour",
        "five hour",
        "current session",
        "session",
        "primary",
        "primary-window",
        "primary window",
    }:
        return "5h"
    if label in {
        "7d",
        "7-day",
        "7 day",
        "seven-day",
        "seven day",
        "current week",
        "week",
        "weekly",
        "secondary",
        "secondary-window",
        "secondary window",
    }:
        return "7d"
    if "week" in label:
        if "opus" in model_text:
            return "opus7d"
        if "sonnet" in model_text:
            return "sonnet7d"
    return raw


def _compact_quota_detail(detail: Any) -> str:
    text = str(detail or "").strip()
    if not text:
        return ""
    # Keep footer quota compact. Detailed breakdowns remain available via the
    # usage renderer; the footer only needs the immediately useful balance.
    if not re.match(r"^(credits\s+)?balance\s*:", text, flags=re.IGNORECASE):
        return ""
    text = re.sub(r"\s*\([^)]*\)\s*$", "", text).strip()
    text = re.sub(r"^(credits\s+)?balance\s*:", "balance", text, flags=re.IGNORECASE).strip()
    return text


def _format_quota(account_usage: Any, *, provider: Optional[str] = None, model: Optional[str] = None) -> list[str]:
    if not account_usage:
        return []
    provider = provider or getattr(account_usage, "provider", None)
    parts: list[str] = []
    for window in getattr(account_usage, "windows", ()) or ():
        used = getattr(window, "used_percent", None)
        if used is None:
            continue
        try:
            remaining = max(0, round(100 - float(used)))
        except Exception:
            continue
        label = _quota_label(window, provider=provider, model=model)
        text = f"{label} {remaining}%"
        reset = _compact_reset(getattr(window, "reset_at", None))
        if reset:
            text += f" {reset}"
        parts.append(text)
    for detail in getattr(account_usage, "details", ()) or ():
        compact = _compact_quota_detail(detail)
        if compact:
            parts.append(compact)
    return parts


def _account_short(account_label: Optional[str], provider: Optional[str]) -> str:
    raw = str(account_label or "").strip()
    if not raw:
        return ""
    prov = str(provider or "").strip()
    if prov:
        for prefix in (prov, prov.replace("-", "_"), prov.replace("_", "-")):
            for sep in ("-", "_"):
                marker = prefix + sep
                if raw.lower().startswith(marker.lower()):
                    return raw[len(marker):]
    return raw


def format_runtime_footer(*, model: Optional[str], context_tokens: int,
                          context_length: Optional[int], cwd: Optional[str] = None,
                          turn_seconds: Optional[float] = None,
                          requested_model: Optional[str] = None, served_model: Optional[str] = None,
                          fields: Iterable[str] = _DEFAULT_FIELDS,
                          provider: Optional[str] = None,
                          account_label: Optional[str] = None,
                          account_usage: Any = None,
                          reasoning_effort: Optional[str] = None,
                          underline: bool = False) -> str:
    """Render the footer line, or "" if no fields have data. Fields whose data is missing (and
    unknown field names) are skipped silently — a partial footer beats ``?%`` or empty slots."""
    def context_pct() -> str:
        if context_length and context_length > 0 and context_tokens >= 0:
            return f"{max(0, min(100, round((context_tokens / context_length) * 100)))}%"
        return ""

    def served() -> str:
        requested = requested_model or model
        alias = _model_short(requested)
        if served_model and served_model not in (alias, requested):
            return f"{alias} → {served_model}"
        return ""

    def context_compact() -> str:
        if context_length and context_length > 0 and context_tokens >= 0:
            return f"ctx {_compact_number(context_tokens)}/{_compact_number(context_length)}"
        return ""

    def account() -> str:
        label = (account_label
                 or getattr(account_usage, "account_label", None)
                 or getattr(account_usage, "plan", None))
        return _account_short(str(label), provider) if label else ""

    renderers = {
        "model": lambda: _model_short(model),
        "served_model": served,
        "reasoning": lambda: _reasoning_short(reasoning_effort),
        "reasoning_effort": lambda: _reasoning_short(reasoning_effort),
        "effort": lambda: _reasoning_short(reasoning_effort),
        "provider": lambda: str(provider) if provider else "",
        "account": account,
        "context": context_compact,
        "context_pct": context_pct,
        # Skipped when the caller did not measure (None) or the value is negative.
        "latency": lambda: _format_latency(turn_seconds) if turn_seconds is not None and turn_seconds >= 0 else "",
        "cwd": lambda: _home_relative_cwd(cwd or _env_cwd()),
    }

    parts: list[str] = []
    for field in fields:
        # ``quota`` expands to one part per account-usage window (plus any
        # compact balance line), so it cannot be a single-string renderer.
        if field == "quota":
            parts.extend(_format_quota(account_usage, provider=provider, model=model))
            continue
        render = renderers.get(field)
        if render is None:
            continue
        value = render()
        if value:
            parts.append(value)

    if not parts:
        return ""
    line = _SEP.join(parts)
    return f"──────────────\n{line}" if underline else line


def build_footer_line(*, user_config: dict[str, Any] | None, platform_key: str | None,
                      model: Optional[str], context_tokens: int, context_length: Optional[int],
                      cwd: Optional[str] = None, turn_seconds: Optional[float] = None,
                      requested_model: Optional[str] = None, served_model: Optional[str] = None,
                      provider: Optional[str] = None, account_label: Optional[str] = None,
                      account_usage: Any = None, reasoning_effort: Optional[str] = None,
                      resolved_config: Optional[dict[str, Any]] = None) -> str:
    """Entry point for gateway/run.py: footer text, or "" when disabled / no data. Callers append it
    to the final response themselves, preserving a single blank line of separation.
    ``turn_seconds`` is the caller-measured (``time.monotonic()``) run duration; ``None`` skips the
    ``latency`` field. ``resolved_config`` lets the turn runner hand over a footer config it already
    resolved, instead of resolving it a second time."""
    cfg = (
        resolved_config
        if isinstance(resolved_config, dict)
        else resolve_footer_config(user_config, platform_key)
    )
    if not cfg.get("enabled"):
        return ""
    return format_runtime_footer(model=model, context_tokens=context_tokens,
                                 context_length=context_length, cwd=cwd, turn_seconds=turn_seconds,
                                 requested_model=requested_model, served_model=served_model,
                                 fields=cfg.get("fields") or _DEFAULT_FIELDS,
                                 provider=provider, account_label=account_label,
                                 account_usage=account_usage,
                                 reasoning_effort=reasoning_effort,
                                 underline=bool(cfg.get("underline")))

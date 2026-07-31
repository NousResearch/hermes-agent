"""Gateway runtime-metadata footer.

Renders a compact footer showing runtime state (model, context %, cwd,
and optional provider/account/quota metadata) and appends it to the FINAL
message of an agent turn when enabled.  Off by default to keep replies minimal.

Config (``~/.hermes/config.yaml``)::

    display:
      runtime_footer:
        enabled: true                         # off by default
        fields: [model, context_pct, cwd]     # order shown; drop any to hide
        underline: false                      # optional separator before footer

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
        return f"{days}d{rem_hours}h"
    if hours > 0:
        return f"{hours}h{minutes}m"
    return f"{minutes}m"


def _quota_label(window: Any, provider: Optional[str] = None, model: Optional[str] = None) -> str:
    """Return a compact quota-window label for footer display.

    The detailed `/usage` command keeps provider wording.  The footer is space
    constrained, so normalize the common OAuth/Codex rolling windows to the
    short labels users expect while leaving unknown provider windows intact.
    """
    raw = str(getattr(window, "label", "") or "quota").strip() or "quota"
    label = raw.lower().replace("_", "-")
    provider_text = str(provider or getattr(window, "provider", "") or "").lower()
    model_text = str(model or "").lower()

    if "opus" in label:
        return "opus7d"
    if "sonnet" in label:
        return "sonnet7d"
    if (
        "five" in label
        or "5" in label
        or "current session" in label
        or label == "session"
        or "primary" in label
    ):
        return "5h"
    if (
        "seven" in label
        or "7" in label
        or "current week" in label
        or label == "weekly"
        or "secondary" in label
    ):
        return "7d"
    if "week" in label:
        if "opus" in model_text:
            return "opus7d"
        if "sonnet" in model_text:
            return "sonnet7d"
        return "7d"
    if "codex" in provider_text and label in {"session", "primary window"}:
        return "5h"
    if "codex" in provider_text and label in {"weekly", "secondary window"}:
        return "7d"
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


def format_runtime_footer(
    *,
    model: Optional[str],
    context_tokens: int,
    context_length: Optional[int],
    cwd: Optional[str] = None,
    fields: Iterable[str] = _DEFAULT_FIELDS,
    provider: Optional[str] = None,
    account_label: Optional[str] = None,
    account_usage: Any = None,
    underline: bool = False,
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
        elif field == "provider":
            if provider:
                parts.append(str(provider))
        elif field == "account":
            acct = account_label or getattr(account_usage, "account_label", None) or getattr(account_usage, "plan", None)
            if acct:
                parts.append(_account_short(str(acct), provider))
        elif field == "context":
            if context_length and context_length > 0 and context_tokens >= 0:
                parts.append(f"ctx {_compact_number(context_tokens)}/{_compact_number(context_length)}")
        elif field == "context_pct":
            if context_length and context_length > 0 and context_tokens >= 0:
                pct = max(0, min(100, round((context_tokens / context_length) * 100)))
                parts.append(f"{pct}%")
        elif field == "quota":
            parts.extend(_format_quota(account_usage, provider=provider, model=model))
        elif field == "cwd":
            rel = _home_relative_cwd(cwd or os.environ.get("TERMINAL_CWD", ""))
            if rel:
                parts.append(rel)
        # Unknown field names are silently ignored.

    if not parts:
        return ""
    line = _SEP.join(parts)
    return f"──────────────\n{line}" if underline else line


def build_footer_line(
    *,
    user_config: dict[str, Any] | None,
    platform_key: str | None,
    model: Optional[str],
    context_tokens: int,
    context_length: Optional[int],
    cwd: Optional[str] = None,
    provider: Optional[str] = None,
    account_label: Optional[str] = None,
    account_usage: Any = None,
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
        cwd=cwd,
        fields=cfg.get("fields") or _DEFAULT_FIELDS,
        provider=provider,
        account_label=account_label,
        account_usage=account_usage,
        underline=bool(cfg.get("underline")),
    )

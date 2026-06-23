"""Gateway runtime-metadata footer.

Renders a compact footer showing runtime state (model, context %, cwd) and
appends it to the FINAL message of an agent turn when enabled.  Off by default
to keep replies minimal.

Config (``~/.hermes/config.yaml``)::

    display:
      runtime_footer:
        enabled: true                       # off by default
        fields: [model, context_pct, cwd]   # order shown; drop any to hide

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

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

_DEFAULT_FIELDS: tuple[str, ...] = ("model", "context_pct", "cwd")
_SEP = " · "
_QUOTA_CACHE_REL = Path("cache") / "oneapi_comate_quota.json"
_QUOTA_MAX_AGE_SECONDS = 6 * 60 * 60


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


def _get_hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")).expanduser()


def _money_display(value: Any) -> str:
    try:
        return f"¥{float(value):,.2f}".replace(".00", "")
    except Exception:
        return str(value)


def _context_pct_text(context_tokens: int, context_length: Optional[int]) -> str:
    if context_length and context_length > 0 and context_tokens >= 0:
        pct = max(0, min(100, round((context_tokens / context_length) * 100)))
        return f"{pct}%"
    return ""


def _read_oneapi_quota_footer() -> str:
    """Return compact OneAPI monthly quota text from cache, or empty string.

    This deliberately never performs network I/O on the response path.  The
    cache is refreshed by ``~/.hermes/scripts/update_oneapi_quota.py`` using the
    user's logged-in Windows Edge CDP session.
    """
    path = _get_hermes_home() / _QUOTA_CACHE_REL
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        used = data.get("monthly_used_display") or data.get("monthly_used_quota")
        limit = data.get("monthly_limit_display") or data.get("monthly_quota_limit")
        if used is None:
            return ""
        text = f"OneAPI本月 {used}"
        if limit is not None:
            text += f"/{limit}"

        source_usage = data.get("source_usage") or {}
        details: list[str] = []
        for key, label in (("openclaw", "OpenClaw"), ("dodo", "DoDo")):
            if key in source_usage and source_usage.get(key) is not None:
                details.append(f"{label} {_money_display(source_usage.get(key))}")
        if details:
            text += f"（{'，'.join(details)}）"

        updated = data.get("updated_at")
        if updated:
            try:
                ts = datetime.fromisoformat(str(updated).replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                age = (datetime.now(timezone.utc) - ts.astimezone(timezone.utc)).total_seconds()
                if age > _QUOTA_MAX_AGE_SECONDS:
                    text += "(stale)"
            except Exception:
                pass
        return text
    except Exception:
        return ""


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
    cwd: Optional[str] = None,
    fields: Iterable[str] = _DEFAULT_FIELDS,
) -> str:
    """Render the footer line, or return "" if no fields have data.

    Fields are skipped silently when their underlying data is missing — a
    partially-populated footer is better than a line with ``?%`` or empty slots.
    """
    parts: list[str] = []
    fields = tuple(fields)
    folds_context_into_model = "model" in fields
    for field in fields:
        if field == "model":
            m = _model_short(model)
            pct = _context_pct_text(context_tokens, context_length)
            if m and pct:
                parts.append(f"模型：{m}（上下文 {pct}）")
            elif m:
                parts.append(f"模型：{m}")
        elif field == "context_pct":
            # Kept for backwards compatibility.  When the model field is present,
            # the percentage is folded into it to avoid a cryptic standalone "46%".
            if folds_context_into_model:
                continue
            pct = _context_pct_text(context_tokens, context_length)
            if pct:
                parts.append(pct)
        elif field == "cwd":
            rel = _home_relative_cwd(cwd or os.environ.get("TERMINAL_CWD", ""))
            if rel:
                parts.append(rel)
        elif field in {"oneapi_quota", "oneapi_comate_quota"}:
            quota = _read_oneapi_quota_footer()
            if quota:
                parts.append(quota)
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
) -> str:
    """Top-level entry point used by gateway/run.py.

    Returns the footer text (empty string when disabled or no data).  Callers
    append this to the final response themselves, preserving a single blank
    line of separation.
    """
    cfg = resolve_footer_config(user_config, platform_key)
    if not cfg.get("enabled"):
        return ""
    fields = list(cfg.get("fields") or _DEFAULT_FIELDS)
    try:
        display_cfg = (user_config or {}).get("display") or {}
        if bool(display_cfg.get("show_perf_footer", False)):
            # The agent-level perf footer already renders the model line and
            # context percentage.  Do not append a second gateway-only
            # "模型：..." footer; keep non-overlapping runtime fields such as
            # cwd or oneapi_quota if the user configured them.
            fields = [f for f in fields if f not in {"model", "context_pct"}]
    except Exception:
        pass
    return format_runtime_footer(
        model=model,
        context_tokens=context_tokens,
        context_length=context_length,
        cwd=cwd,
        fields=fields,
    )

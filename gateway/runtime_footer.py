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
    model           — bare model id, vendor prefix dropped (``gpt-5.4``)
    context_pct     — last-call context occupancy as a percent (``5%``)
    context_usage   — last-call context tokens + percent (``18.9K (9%)``)
    session_tokens  — cumulative session usage (``📊 Sesión · N req · In · Out · Total``)
    latency         — wall-clock duration of the turn (``22s``, ``1m05s``)
    cwd             — home-relative working dir (``~``)

``latency`` is opt-in: it is NOT in the default field set, so a footer whose
``fields`` are unset renders exactly as before.

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
from typing import Any, Iterable, Mapping, Optional

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


def _format_token_count(value: Any) -> str:
    """Format token counts with compact units (``18.9K``, ``2.9M``)."""
    try:
        number = max(0, int(value or 0))
    except (TypeError, ValueError):
        number = 0
    if number >= 1_000_000:
        return f"{number / 1_000_000:.1f}M"
    if number >= 1_000:
        return f"{number / 1_000:.1f}K"
    return str(number)


def format_session_token_usage(usage: Mapping[str, Any] | None) -> str:
    """Render cumulative token usage for one persisted Hermes session.

    ``input_tokens`` excludes cache tokens in Hermes' canonical accounting, so
    the displayed ``In`` value follows session-stats and includes read/write
    cache tokens while also showing the cache amount separately.
    """
    if not isinstance(usage, Mapping):
        return ""

    input_tokens = max(0, int(usage.get("input_tokens", 0) or 0))
    output_tokens = max(0, int(usage.get("output_tokens", 0) or 0))
    cache_tokens = max(
        0,
        int(usage.get("cache_read_tokens", 0) or 0)
        + int(usage.get("cache_write_tokens", 0) or 0),
    )
    request_count = max(0, int(usage.get("api_call_count", 0) or 0))
    total_tokens = max(
        0,
        int(usage.get("total_tokens", 0) or 0)
        or input_tokens + cache_tokens + output_tokens,
    )

    if not (request_count or input_tokens or output_tokens or cache_tokens or total_tokens):
        return ""

    parts = ["📊 Sesión"]
    if request_count:
        parts.append(f"{request_count} req")
    parts.append(f"In: {_format_token_count(input_tokens + cache_tokens)}")
    if cache_tokens:
        parts[-1] += f" (cache: {_format_token_count(cache_tokens)})"
    parts.append(f"Out: {_format_token_count(output_tokens)}")
    parts.append(f"Total: {_format_token_count(total_tokens)}")
    base = _SEP.join(parts)
    # Cost — append when the persisted session row carries a cost. Prefer
    # actual_cost_usd (provider-reported) over estimated_cost_usd so models
    # that return real cost are exact and estimated ones still show.
    try:
        cost_val: float | None = None
        actual = usage.get("actual_cost_usd")
        estimated = usage.get("estimated_cost_usd")
        if actual is not None and float(actual) > 0:
            cost_val = float(actual)
        elif estimated is not None and float(estimated) > 0:
            cost_val = float(estimated)
        # Fallback keys some call sites use
        if cost_val is None:
            for _k in ("cost", "total_cost", "cost_usd"):
                _v = usage.get(_k)
                if _v is not None and float(_v) > 0:
                    cost_val = float(_v)
                    break
        if cost_val is not None and cost_val > 0:
            cost_str = f"${cost_val:.4f}" if cost_val < 1 else f"${cost_val:.2f}"
            return f"{base} {cost_str}"
    except Exception:
        pass
    return base


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
    session_usage: Mapping[str, Any] | None = None,
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
        elif field == "context_usage":
            if context_length and context_length > 0 and context_tokens >= 0:
                pct = max(0, min(100, round((context_tokens / context_length) * 100)))
                parts.append(f"{_format_token_count(context_tokens)} ({pct}%)")
        elif field == "session_tokens":
            usage = format_session_token_usage(session_usage)
            if usage:
                parts.append(usage)
        elif field == "latency":
            # Wall-clock turn duration. Skipped when the caller supplied no
            # timing (call sites that don't measure) or the value is negative.
            if turn_seconds is not None and turn_seconds >= 0:
                parts.append(_format_latency(turn_seconds))
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
    turn_seconds: Optional[float] = None,
    session_usage: Mapping[str, Any] | None = None,
) -> str:
    """Top-level entry point used by gateway/run.py.

    Returns the footer text (empty string when disabled or no data).  Callers
    append this to the final response themselves, preserving a single blank
    line of separation.

    ``turn_seconds`` is the wall-clock duration of the agent run, measured by
    the caller with ``time.monotonic()``.  Callers that don't measure it leave
    it ``None`` and the ``latency`` field is skipped.

    ``session_usage`` is a mapping of persisted session counters (from
    ``state.db``) used by the ``session_tokens`` field; pass ``None`` to skip.
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
        session_usage=session_usage,
    )

"""Gateway runtime-metadata footer (model · context % · cwd), off by default to keep replies
minimal. Config: ``display.runtime_footer: {enabled: bool, fields: [model, context_pct, cwd]}``
(order shown; drop any to hide), per-platform override ``display.platforms.<p>.runtime_footer``,
toggled by ``/footer on|off``. Fields: ``model`` (vendor prefix dropped), ``context_pct`` (last-call
occupancy), ``tokens`` (compact total/prompt tokens, e.g. ``15.2k tokens``), ``in`` / ``out`` (input/output tokens,
e.g. ``in: 12.3k`` / ``out: 512``), ``cache`` (cache read tokens, e.g. ``cache: 224k``), ``reasoning``
(thinking tokens, e.g. ``reasoning: 408``), ``cost`` (estimated USD, e.g. ``~$0.0150``), ``api_calls``
(count for current session/turn), ``latency`` (turn wall-clock, opt-in — NOT in the default set so
an unset ``fields`` renders exactly as before), ``served_model`` (opt-in, ``alias → served``: the
deployment a routing proxy reported via ``x-litellm-model-id`` / ``x-litellm-model-api-base``, or Hermes'
own fallback route; skipped when the served model is the requested one), ``cwd`` (home-relative).
``gateway/run.py`` appends the footer to the final response only (never to tool-progress or streaming
partials); when streaming already delivered the text, it goes out as a trailing message via
``send_trailing_footer()``."""

from __future__ import annotations

import os
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
    """Drop ``vendor/`` prefix (``openai/gpt-5.4`` → ``gpt-5.4``)."""
    return model.rsplit("/", 1)[-1] if model else ""


def _env_cwd() -> str:
    try:
        from tools.terminal_scope import terminal_env
    except ImportError:
        return os.environ.get("TERMINAL_CWD", "")
    return terminal_env("TERMINAL_CWD", "")


def resolve_footer_config(user_config: dict[str, Any] | None, platform_key: str | None = None) -> dict[str, Any]:
    """Resolve effective footer config: defaults (enabled=False) <
    ``display.runtime_footer`` < ``display.platforms.<platform_key>.runtime_footer``."""
    resolved = {"enabled": False, "fields": list(_DEFAULT_FIELDS)}
    cfg = (user_config or {}).get("display") or {}
    plat_cfg = (cfg.get("platforms") or {}).get(platform_key) if platform_key else None
    sections = [cfg.get("runtime_footer"), plat_cfg.get("runtime_footer") if isinstance(plat_cfg, dict) else None]
    for section in sections:
        if not isinstance(section, dict):
            continue
        if "enabled" in section:
            resolved["enabled"] = bool(section.get("enabled"))
        if isinstance(section.get("fields"), list) and section["fields"]:
            resolved["fields"] = [str(f) for f in section["fields"]]
    return resolved


def _format_tokens(count: int) -> str:
    """Format token counts compactly: 120 -> '120', 15200 -> '15.2k', 1200000 -> '1.2M'."""
    if count < 1000:
        return str(count)
    if count < 1_000_000:
        val = count / 1000
        formatted = f"{val:.1f}".removesuffix(".0")
        return f"{formatted}k"
    val = count / 1_000_000
    formatted = f"{val:.2f}".rstrip("0").rstrip(".")
    return f"{formatted}M"


def _format_cost(cost: float) -> str:
    """Format estimated USD cost: 0.0 -> '$0.00', 0.015 -> '~$0.0150', < 0.0001 -> '<$0.0001'."""
    if cost == 0:
        return "$0.00"
    if cost < 0.0001:
        return "<$0.0001"
    return f"~${cost:.4f}"


def _format_latency(seconds: float) -> str:
    """Humanize a turn duration: ``<1s``, ``22s``, ``1m05s``."""
    if seconds < 1:
        return "<1s"
    total = round(seconds)
    if total < 60:
        return f"{total}s"
    m, sec = divmod(total, 60)
    return f"{m}m{sec:02d}s"


def format_runtime_footer(*, model: Optional[str], context_tokens: int,
                          context_length: Optional[int], cwd: Optional[str] = None,
                          turn_seconds: Optional[float] = None,
                          requested_model: Optional[str] = None, served_model: Optional[str] = None,
                          total_tokens: Optional[int] = None,
                          input_tokens: Optional[int] = None,
                          output_tokens: Optional[int] = None,
                          cache_read_tokens: Optional[int] = None,
                          cache_write_tokens: Optional[int] = None,
                          reasoning_tokens: Optional[int] = None,
                          estimated_cost_usd: Optional[float] = None,
                          api_calls: Optional[int] = None,
                          fields: Iterable[str] = _DEFAULT_FIELDS) -> str:
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

    def render_tokens() -> str:
        tok = total_tokens if (total_tokens is not None and total_tokens >= 0) else context_tokens
        if tok is not None and tok > 0:
            return f"{_format_tokens(tok)} tokens"
        return ""

    renderers = {
        "model": lambda: _model_short(model),
        "served_model": served,
        "context_pct": context_pct,
        "tokens": render_tokens,
        "in": lambda: f"in: {_format_tokens(input_tokens)}" if input_tokens is not None and input_tokens > 0 else "",
        "out": lambda: f"out: {_format_tokens(output_tokens)}" if output_tokens is not None and output_tokens > 0 else "",
        "cache": lambda: f"cache: {_format_tokens(cache_read_tokens)}" if cache_read_tokens is not None and cache_read_tokens > 0 else "",
        "reasoning": lambda: f"reasoning: {_format_tokens(reasoning_tokens)}" if reasoning_tokens is not None and reasoning_tokens > 0 else "",
        "cost": lambda: _format_cost(estimated_cost_usd) if estimated_cost_usd is not None and estimated_cost_usd >= 0 else "",
        "api_calls": lambda: (f"{api_calls} call" if api_calls == 1 else f"{api_calls} calls") if api_calls is not None and api_calls > 0 else "",
        # Skipped when the caller did not measure (None) or the value is negative.
        "latency": lambda: _format_latency(turn_seconds) if turn_seconds is not None and turn_seconds >= 0 else "",
        "cwd": lambda: _home_relative_cwd(cwd or _env_cwd()),
    }
    return _SEP.join(v for field in fields if (render := renderers.get(field)) and (v := render()))


def build_footer_line(*, user_config: dict[str, Any] | None, platform_key: str | None,
                      model: Optional[str], context_tokens: int, context_length: Optional[int],
                      cwd: Optional[str] = None, turn_seconds: Optional[float] = None,
                      requested_model: Optional[str] = None, served_model: Optional[str] = None,
                      total_tokens: Optional[int] = None,
                      input_tokens: Optional[int] = None,
                      output_tokens: Optional[int] = None,
                      cache_read_tokens: Optional[int] = None,
                      cache_write_tokens: Optional[int] = None,
                      reasoning_tokens: Optional[int] = None,
                      estimated_cost_usd: Optional[float] = None,
                      api_calls: Optional[int] = None,
                      **kwargs: Any) -> str:
    """Entry point for gateway/run.py: footer text, or "" when disabled / no data. Callers append it
    to the final response themselves, preserving a single blank line of separation.
    ``turn_seconds`` is the caller-measured (``time.monotonic()``) run duration; ``None`` skips the
    ``latency`` field."""
    cfg = resolve_footer_config(user_config, platform_key)
    if not cfg.get("enabled"):
        return ""
    return format_runtime_footer(model=model, context_tokens=context_tokens,
                                 context_length=context_length, cwd=cwd, turn_seconds=turn_seconds,
                                 requested_model=requested_model, served_model=served_model,
                                 total_tokens=total_tokens, input_tokens=input_tokens,
                                 output_tokens=output_tokens, cache_read_tokens=cache_read_tokens,
                                 cache_write_tokens=cache_write_tokens,
                                 reasoning_tokens=reasoning_tokens,
                                 estimated_cost_usd=estimated_cost_usd,
                                 api_calls=api_calls,
                                 fields=cfg.get("fields") or _DEFAULT_FIELDS)

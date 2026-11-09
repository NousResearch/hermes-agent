"""Gateway runtime-metadata footer (model · context % · cwd), off by default to keep replies
minimal. Config: ``display.runtime_footer: {enabled: bool, fields: [model, context_pct, cwd]}``
(order shown; drop any to hide), per-platform override ``display.platforms.<p>.runtime_footer``,
toggled by ``/footer on|off``. Fields: ``model`` (vendor prefix dropped), ``context_pct`` (last-call
occupancy), ``latency`` (turn wall-clock, opt-in — NOT in the default set so an unset ``fields``
renders exactly as before), ``served_model`` (opt-in, ``alias → served``: the deployment a routing
proxy reported via ``x-litellm-model-id`` / ``x-litellm-model-api-base``, or Hermes' own fallback
route; skipped when the served model is the requested one), ``cwd`` (home-relative). ``gateway/run.py`` appends the footer to the
final response only (never to tool-progress or streaming partials); when streaming already
delivered the text, it goes out as a trailing message via ``send_trailing_footer()``."""

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
    resolved = {"enabled": False, "fields": list(_DEFAULT_FIELDS), "usage_mode": "full"}
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
        if str(section.get("usage_mode") or "").strip().lower() in {"off", "tokens", "full", "cost"}:
            resolved["usage_mode"] = str(section["usage_mode"]).strip().lower()
    return resolved


def _format_latency(seconds: float) -> str:
    """Humanize a turn duration: ``<1s``, ``22s``, ``1m05s``."""
    if seconds < 1:
        return "<1s"
    total = round(seconds)
    if total < 60:
        return f"{total}s"
    m, sec = divmod(total, 60)
    return f"{m}m{sec:02d}s"


def _fmt_tokens(n: int) -> str:
    """Humanize a token count: 189k, 4.6k, 2.2m."""
    n = max(0, int(n or 0))
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}m"
    if n >= 10_000:
        return f"{n // 1_000}k"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def format_runtime_footer(*, model: Optional[str], context_tokens: int,
                          context_length: Optional[int], cwd: Optional[str] = None,
                          turn_seconds: Optional[float] = None,
                          requested_model: Optional[str] = None, served_model: Optional[str] = None,
                          in_tokens: int = 0, out_tokens: int = 0, cached_tokens: int = 0,
                          cost_usd: Optional[float] = None, usage_mode: str = "full",
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

    def usage() -> str:
        # Display modes: off | tokens | full | cost.
        mode = (usage_mode or "full").strip().lower()
        if mode == "off" or not (in_tokens or out_tokens or cached_tokens):
            return ""
        if mode == "tokens":
            return f"{_fmt_tokens(in_tokens)} in / {_fmt_tokens(out_tokens)} out"
        if mode == "cost":
            if cost_usd is None or cost_usd < 0:
                return ""
            return f"${cost_usd:.4f}"
        new_tokens = max(0, in_tokens - cached_tokens)
        return f"{_fmt_tokens(in_tokens)} in / {_fmt_tokens(out_tokens)} out · cache {_fmt_tokens(cached_tokens)} cached / {_fmt_tokens(new_tokens)} new"

    renderers = {
        "model": lambda: _model_short(model),
        "served_model": served,
        "context_pct": context_pct,
        # Skipped when the caller did not measure (None) or the value is negative.
        "latency": lambda: _format_latency(turn_seconds) if turn_seconds is not None and turn_seconds >= 0 else "",
        "cwd": lambda: _home_relative_cwd(cwd or _env_cwd()),
        "usage": usage,
    }
    return _SEP.join(v for field in fields if (render := renderers.get(field)) and (v := render()))


def build_footer_line(*, user_config: dict[str, Any] | None, platform_key: str | None,
                      model: Optional[str], context_tokens: int, context_length: Optional[int],
                      cwd: Optional[str] = None, turn_seconds: Optional[float] = None,
                      requested_model: Optional[str] = None, served_model: Optional[str] = None,
                      in_tokens: int = 0, out_tokens: int = 0, cached_tokens: int = 0,
                      cost_usd: Optional[float] = None) -> str:
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
                                 in_tokens=in_tokens, out_tokens=out_tokens, cached_tokens=cached_tokens,
                                 cost_usd=cost_usd, usage_mode=cfg.get("usage_mode") or "full",
                                 fields=cfg.get("fields") or _DEFAULT_FIELDS)

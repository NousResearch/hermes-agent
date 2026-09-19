"""Gateway runtime metadata and compact per-turn footer formatting.

The default footer remains ``model · context % · cwd``. Opt-in fields expose
turn latency, uncached provider input, cache-hit share, exact last-call context,
and requested reasoning effort. Unknown or untrustworthy usage facets are
omitted instead of rendered as zero.
"""

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


def _format_latency(seconds: float) -> str:
    """Humanize a turn duration: ``<1s``, ``22s``, ``1m05s``."""
    if seconds < 1:
        return "<1s"
    total = int(round(seconds))
    if total < 60:
        return f"{total}s"
    m, sec = divmod(total, 60)
    return f"{m}m{sec:02d}s"


def _format_token_count(value: int) -> str:
    """Render a compact, stable token count for the footer."""
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return str(value)


def _counter_delta(current: Any, baseline: Any) -> Optional[int]:
    """Return a non-negative cumulative-counter delta, or ``None`` if invalid."""
    if isinstance(current, bool) or isinstance(baseline, bool):
        return None
    try:
        current_int = int(current)
        baseline_int = int(baseline)
    except (TypeError, ValueError):
        return None
    if current_int < 0 or baseline_int < 0 or current_int < baseline_int:
        return None
    return current_int - baseline_int


def _requested_reasoning_effort(reasoning_config: Any) -> str:
    """Describe Hermes' active request intent without provider inference."""
    if not isinstance(reasoning_config, dict):
        return "default"
    if reasoning_config.get("enabled") is False:
        return "none"
    effort = str(reasoning_config.get("effort") or "").strip().lower()
    return effort or "default"


def _gateway_turn_runtime_metadata(
    agent: Any,
    *,
    uncached_input_tokens_start: Any,
    completion_tokens_start: Any,
    cache_read_tokens_start: Any,
    cache_write_tokens_start: Any,
    usage_report_calls_start: Any,
    cache_usage_report_calls_start: Any,
    context_usage_report_calls_start: Any,
    result_api_calls: Any,
) -> dict[str, Any]:
    """Snapshot final runtime identity and honest reported per-turn usage.

    Cached-agent counters are cumulative, so usage is the delta from the
    baseline captured immediately before ``run_conversation()``.
    """
    if agent is None:
        return {}

    compressor = getattr(agent, "context_compressor", None)
    last_prompt_tokens = getattr(agent, "session_last_prompt_tokens", 0) or 0
    context_length = getattr(compressor, "context_length", 0) or 0
    input_tokens = getattr(agent, "session_prompt_tokens", 0) or 0
    uncached_input_tokens = getattr(agent, "session_input_tokens", 0) or 0
    output_tokens = getattr(agent, "session_completion_tokens", 0) or 0
    cache_read_tokens = getattr(agent, "session_cache_read_tokens", 0) or 0
    cache_write_tokens = getattr(agent, "session_cache_write_tokens", 0) or 0
    usage_report_calls = getattr(agent, "session_usage_report_calls", 0) or 0
    cache_usage_report_calls = getattr(agent, "session_cache_usage_report_calls", 0) or 0
    context_usage_report_calls = getattr(agent, "session_context_usage_report_calls", 0) or 0

    turn_input_tokens = _counter_delta(uncached_input_tokens, uncached_input_tokens_start)
    turn_output_tokens = _counter_delta(output_tokens, completion_tokens_start)
    turn_cache_read_tokens = _counter_delta(cache_read_tokens, cache_read_tokens_start)
    turn_cache_write_tokens = _counter_delta(cache_write_tokens, cache_write_tokens_start)
    turn_usage_report_calls = _counter_delta(usage_report_calls, usage_report_calls_start)
    turn_cache_usage_report_calls = _counter_delta(
        cache_usage_report_calls, cache_usage_report_calls_start
    )
    turn_context_usage_report_calls = _counter_delta(
        context_usage_report_calls, context_usage_report_calls_start
    )
    try:
        expected_api_calls = None if isinstance(result_api_calls, bool) else int(result_api_calls)
    except (TypeError, ValueError):
        expected_api_calls = None

    token_usage_status = None
    if (
        isinstance(turn_usage_report_calls, int)
        and turn_usage_report_calls > 0
        and turn_input_tokens is not None
        and turn_output_tokens is not None
    ):
        token_usage_status = "reported"
        if expected_api_calls is None or expected_api_calls < 0 or turn_usage_report_calls != expected_api_calls:
            token_usage_status = "reported_partial"
    else:
        turn_input_tokens = None
        turn_output_tokens = None

    cache_usage_status = None
    if (
        isinstance(turn_cache_usage_report_calls, int)
        and turn_cache_usage_report_calls > 0
        and turn_cache_read_tokens is not None
        and turn_cache_write_tokens is not None
    ):
        cache_usage_status = "reported"
        if (
            expected_api_calls is None
            or expected_api_calls < 0
            or turn_cache_usage_report_calls != expected_api_calls
        ):
            cache_usage_status = "reported_partial"
    else:
        turn_cache_read_tokens = None
        turn_cache_write_tokens = None

    context_usage_status = None
    if (
        isinstance(turn_context_usage_report_calls, int)
        and turn_context_usage_report_calls > 0
        and expected_api_calls is not None
        and expected_api_calls >= 0
        and turn_context_usage_report_calls == expected_api_calls
        and last_prompt_tokens > 0
    ):
        context_usage_status = "reported"

    return {
        "last_prompt_tokens": last_prompt_tokens,
        "input_tokens": input_tokens,
        "uncached_input_tokens": uncached_input_tokens,
        "output_tokens": output_tokens,
        "cache_read_tokens": cache_read_tokens,
        "cache_write_tokens": cache_write_tokens,
        "usage_report_calls": usage_report_calls,
        "cache_usage_report_calls": cache_usage_report_calls,
        "context_usage_report_calls": context_usage_report_calls,
        "turn_input_tokens": turn_input_tokens,
        "turn_output_tokens": turn_output_tokens,
        "turn_cache_read_tokens": turn_cache_read_tokens,
        "turn_cache_write_tokens": turn_cache_write_tokens,
        "token_usage_status": token_usage_status,
        "cache_usage_status": cache_usage_status,
        "context_usage_status": context_usage_status,
        "reasoning_effort": _requested_reasoning_effort(getattr(agent, "reasoning_config", None)),
        "model": getattr(agent, "model", None),
        "context_length": context_length,
    }


def format_runtime_footer(
    *,
    model: Optional[str],
    context_tokens: int,
    context_length: Optional[int],
    cwd: Optional[str] = None,
    turn_seconds: Optional[float] = None,
    tokens_in: Optional[int] = None,
    tokens_out: Optional[int] = None,
    cache_read_tokens: Optional[int] = None,
    cache_write_tokens: Optional[int] = None,
    token_usage_status: Optional[str] = None,
    cache_usage_status: Optional[str] = None,
    context_usage_status: Optional[str] = "reported",
    reasoning_effort: Optional[str] = None,
    fields: Iterable[str] = _DEFAULT_FIELDS,
) -> str:
    """Render the footer line, skipping fields whose data is unavailable."""
    parts: list[str] = []
    for field in fields:
        if field == "model":
            if value := _model_short(model):
                parts.append(value)
        elif field in {"context_pct", "context_window"}:
            if (
                context_usage_status == "reported"
                and context_length
                and context_length > 0
                and context_tokens >= 0
            ):
                pct = max(0, min(100, round((context_tokens / context_length) * 100)))
                if field == "context_pct":
                    parts.append(f"{pct}%")
                else:
                    parts.append(
                        f"ctx(last):{_format_token_count(context_tokens)}/"
                        f"{_format_token_count(context_length)} ({pct}%)"
                    )
        elif field == "latency":
            if turn_seconds is not None and turn_seconds >= 0:
                parts.append(_format_latency(turn_seconds))
        elif field == "cwd":
            if value := _home_relative_cwd(cwd or _env_cwd()):
                parts.append(value)
        elif field == "tokens_turn":
            reported: list[str] = []
            if isinstance(tokens_in, int) and not isinstance(tokens_in, bool) and tokens_in >= 0:
                reported.append(f"{_format_token_count(tokens_in)} in")
            if isinstance(tokens_out, int) and not isinstance(tokens_out, bool) and tokens_out >= 0:
                reported.append(f"{_format_token_count(tokens_out)} out")
            if reported and token_usage_status in {"reported", "reported_partial"}:
                label = (
                    "tokens(turn,uncached,partial)"
                    if token_usage_status == "reported_partial"
                    else "tokens(turn,uncached)"
                )
                parts.append(f"{label}:{'/'.join(reported)}")
        elif field == "cache_hit":
            cache_buckets = (tokens_in, cache_read_tokens, cache_write_tokens)
            if (
                cache_usage_status in {"reported", "reported_partial"}
                and all(
                    isinstance(value, int) and not isinstance(value, bool) and value >= 0
                    for value in cache_buckets
                )
            ):
                prompt_tokens = sum(cache_buckets)
                if prompt_tokens > 0:
                    label = "cache(turn,partial)" if cache_usage_status == "reported_partial" else "cache(turn)"
                    parts.append(f"{label}:{round((cache_read_tokens / prompt_tokens) * 100)}%")
        elif field == "reasoning_effort" and reasoning_effort:
            parts.append(f"effort(req):{reasoning_effort}")
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
    tokens_in: Optional[int] = None,
    tokens_out: Optional[int] = None,
    cache_read_tokens: Optional[int] = None,
    cache_write_tokens: Optional[int] = None,
    token_usage_status: Optional[str] = None,
    cache_usage_status: Optional[str] = None,
    context_usage_status: Optional[str] = "reported",
    reasoning_effort: Optional[str] = None,
) -> str:
    """Entry point for gateway/run.py: footer text, or "" when disabled / no data. Callers append it
    to the final response themselves, preserving a single blank line of separation.
    ``turn_seconds`` is the caller-measured (``time.monotonic()``) run duration; ``None`` skips the
    ``latency`` field."""
    cfg = resolve_footer_config(user_config, platform_key)
    if not cfg.get("enabled"):
        return ""
    return format_runtime_footer(
        model=model,
        context_tokens=context_tokens,
        context_length=context_length,
        cwd=cwd,
        turn_seconds=turn_seconds,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        token_usage_status=token_usage_status,
        cache_usage_status=cache_usage_status,
        context_usage_status=context_usage_status,
        reasoning_effort=reasoning_effort,
        fields=cfg.get("fields") or _DEFAULT_FIELDS,
    )

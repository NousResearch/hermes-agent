"""Gateway runtime-metadata footer.

Renders a compact footer showing runtime state (model/provider, context, tool
calls, approximate cost, elapsed time, cwd) and appends it to the FINAL message
of an agent turn when enabled.  On by default so gateway users get lightweight
runtime provenance unless they opt out.

Config (``~/.hermes/config.yaml``)::

    display:
      runtime_footer:
        enabled: true
        style: khal_pulse_dev
        fields: [model, provider, context_bar, compressions, api_calls, cost, elapsed, cwd]

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
from typing import Any, Iterable, Optional

_DEFAULT_FIELDS: tuple[str, ...] = ("model", "context_pct", "cwd")
_KHAL_PULSE_FIELDS: tuple[str, ...] = (
    "model", "provider", "context_bar", "compressions",
    "api_calls", "cost", "elapsed", "cwd",
)
_DEFAULT_ENABLED = True
_DEFAULT_STYLE = "khal_pulse_dev"
_SEP = " · "
_ONE_MILLION = 1_000_000.0
_OPENAI_CODEX_SHADOW_SUBSIDY_DIVISOR = 20.0
# Footer-only shadow baseline for GPT-5.5 over the subscription Codex lane.
# It intentionally does not mutate session billing, which remains
# subscription-included.  The rendered "~" marks this as approximate.
_GPT55_SHADOW_INPUT_PER_M = 1.25
_GPT55_SHADOW_OUTPUT_PER_M = 10.00
_GPT55_SHADOW_CACHE_READ_PER_M = 0.125
_GPT55_SHADOW_CACHE_WRITE_PER_M = 1.25


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


def _context_pct(context_tokens: int, context_length: Optional[int]) -> Optional[int]:
    if context_length and context_length > 0 and context_tokens >= 0:
        return max(0, min(100, round((context_tokens / context_length) * 100)))
    return None


def _format_k_tokens(value: int | float | None) -> str:
    try:
        n = int(value or 0)
    except Exception:
        n = 0
    if n >= 1000:
        return f"{round(n / 1000):.0f}K"
    return str(max(0, n))


def _context_bar(pct: int) -> str:
    filled = max(0, min(10, round(pct / 10)))
    return "█" * filled + "░" * (10 - filled)


def _format_cost(value: float | int | None, *, approximate: bool = False) -> str:
    if value is None:
        return ""
    try:
        cost = float(value)
    except Exception:
        return ""
    if cost < 0:
        return ""
    prefix = "~" if approximate else ""
    if cost < 0.001 and cost > 0:
        return f"{prefix}${cost:.5f}"
    return f"{prefix}${cost:.3f}"


def _coerce_nonnegative_float(value: float | int | None) -> float:
    try:
        return max(0.0, float(value or 0))
    except Exception:
        return 0.0


def _openai_codex_shadow_subsidized_cost(
    *,
    model: Optional[str],
    provider: Optional[str],
    context_tokens: int,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    cache_read_tokens: Optional[int] = None,
    cache_write_tokens: Optional[int] = None,
) -> Optional[float]:
    """Approximate GPT-5.5 subscription-lane cost as retail-equivalent / 20.

    This is display-only.  The OpenAI Codex route remains billing-mode
    ``subscription_included`` in the session ledger; the footer shows this as
    an approximate operational cost with ``~``.
    """
    if (provider or "").strip().lower() != "openai-codex":
        return None
    if _model_short(model).lower() != "gpt-5.5":
        return None

    # Prefer an explicit usage bucket when the gateway provides one; otherwise
    # fall back to the context-token count shown in the same footer.  That makes
    # the estimate useful even on older runtime paths that only expose
    # last_prompt_tokens.
    prompt_basis = _coerce_nonnegative_float(input_tokens)
    if prompt_basis <= 0:
        prompt_basis = _coerce_nonnegative_float(context_tokens)
    output_basis = _coerce_nonnegative_float(output_tokens)
    cache_read_basis = _coerce_nonnegative_float(cache_read_tokens)
    cache_write_basis = _coerce_nonnegative_float(cache_write_tokens)
    if prompt_basis <= 0 and output_basis <= 0 and cache_read_basis <= 0 and cache_write_basis <= 0:
        return None

    # When prompt_basis includes cached input, subtract cache buckets so cached
    # tokens get the cheaper cached-input rate instead of full input rate.
    fresh_input = max(0.0, prompt_basis - cache_read_basis - cache_write_basis)
    retail = (
        fresh_input * _GPT55_SHADOW_INPUT_PER_M
        + output_basis * _GPT55_SHADOW_OUTPUT_PER_M
        + cache_read_basis * _GPT55_SHADOW_CACHE_READ_PER_M
        + cache_write_basis * _GPT55_SHADOW_CACHE_WRITE_PER_M
    ) / _ONE_MILLION
    return retail / _OPENAI_CODEX_SHADOW_SUBSIDY_DIVISOR


def _format_display_cost(
    *,
    model: Optional[str],
    provider: Optional[str],
    estimated_cost_usd: Optional[float],
    context_tokens: int,
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    cache_read_tokens: Optional[int],
    cache_write_tokens: Optional[int],
) -> str:
    if estimated_cost_usd and estimated_cost_usd > 0:
        return _format_cost(estimated_cost_usd)
    shadow = _openai_codex_shadow_subsidized_cost(
        model=model,
        provider=provider,
        context_tokens=context_tokens,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
    )
    if shadow is not None:
        return _format_cost(shadow, approximate=True)
    return _format_cost(estimated_cost_usd)


def _format_elapsed(seconds: float | int | None) -> str:
    if seconds is None:
        return ""
    try:
        s = max(0, float(seconds))
    except Exception:
        return ""
    if s < 60:
        return f"{round(s):.0f}s"
    m, sec = divmod(round(s), 60)
    if m < 60:
        return f"{m}m{sec:02d}s"
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
    resolved = {
        "enabled": _DEFAULT_ENABLED,
        "fields": list(_KHAL_PULSE_FIELDS),
        "style": _DEFAULT_STYLE,
    }
    cfg = (user_config or {}).get("display") or {}

    global_cfg = cfg.get("runtime_footer")
    if isinstance(global_cfg, dict):
        if "enabled" in global_cfg:
            resolved["enabled"] = bool(global_cfg.get("enabled"))
        if isinstance(global_cfg.get("fields"), list) and global_cfg["fields"]:
            resolved["fields"] = [str(f) for f in global_cfg["fields"]]
        if global_cfg.get("style"):
            resolved["style"] = str(global_cfg.get("style"))

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
                if plat_footer.get("style"):
                    resolved["style"] = str(plat_footer.get("style"))

    return resolved


def format_runtime_footer(
    *,
    model: Optional[str],
    context_tokens: int,
    context_length: Optional[int],
    cwd: Optional[str] = None,
    fields: Iterable[str] = _DEFAULT_FIELDS,
    style: str = "plain",
    provider: Optional[str] = None,
    api_calls: Optional[int] = None,
    estimated_cost_usd: Optional[float] = None,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    cache_read_tokens: Optional[int] = None,
    cache_write_tokens: Optional[int] = None,
    compression_count: Optional[int] = None,
    elapsed_seconds: Optional[float] = None,
    session_id: Optional[str] = None,
) -> str:
    """Render the footer line, or return "" if no fields have data."""
    if style == "khal_pulse_dev":
        return _format_khal_pulse_dev(
            model=model,
            provider=provider,
            context_tokens=context_tokens,
            context_length=context_length,
            cwd=cwd,
            fields=fields,
            api_calls=api_calls,
            estimated_cost_usd=estimated_cost_usd,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            compression_count=compression_count,
            elapsed_seconds=elapsed_seconds,
            session_id=session_id,
        )

    parts: list[str] = []
    for field in fields:
        if field == "model":
            m = _model_short(model)
            if m:
                parts.append(m)
        elif field == "context_pct":
            pct = _context_pct(context_tokens, context_length)
            if pct is not None:
                parts.append(f"{pct}%")
        elif field == "cwd":
            rel = _home_relative_cwd(cwd or os.environ.get("TERMINAL_CWD", ""))
            if rel:
                parts.append(rel)
        # Unknown field names are silently ignored.

    if not parts:
        return ""
    return _SEP.join(parts)


def _format_khal_pulse_dev(
    *,
    model: Optional[str],
    provider: Optional[str],
    context_tokens: int,
    context_length: Optional[int],
    cwd: Optional[str],
    fields: Iterable[str],
    api_calls: Optional[int],
    estimated_cost_usd: Optional[float],
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    cache_read_tokens: Optional[int],
    cache_write_tokens: Optional[int],
    compression_count: Optional[int],
    elapsed_seconds: Optional[float],
    session_id: Optional[str],
) -> str:
    fields = tuple(fields or _KHAL_PULSE_FIELDS)
    line1: list[str] = []
    line2: list[str] = []
    for field in fields:
        if field == "model":
            m = _model_short(model)
            if m:
                line1.append(f"⚕ {m}")
        elif field == "provider" and provider:
            line1.append(f"🧭 {provider}")
        elif field in {"context_bar", "context"}:
            pct = _context_pct(context_tokens, context_length)
            if pct is not None:
                line1.append(f"🧠{_format_k_tokens(context_tokens)}/{_format_k_tokens(context_length)} {_context_bar(pct)} {pct}%")
        elif field in {"compressions", "compression_count"}:
            try:
                n = int(compression_count or 0)
            except Exception:
                n = 0
            line1.append(f"🗜{n}")
        elif field in {"api_calls", "calls"} and api_calls is not None:
            try:
                line2.append(f"🔁{int(api_calls)}")
            except Exception:
                pass
        elif field == "cost":
            cost = _format_display_cost(
                model=model,
                provider=provider,
                estimated_cost_usd=estimated_cost_usd,
                context_tokens=context_tokens,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read_tokens,
                cache_write_tokens=cache_write_tokens,
            )
            if cost:
                line2.append(f"💸{cost}")
        elif field in {"elapsed", "elapsed_seconds"}:
            elapsed = _format_elapsed(elapsed_seconds)
            if elapsed:
                line2.append(f"✓{elapsed}")
        elif field == "cwd":
            rel = _home_relative_cwd(cwd or os.environ.get("TERMINAL_CWD", ""))
            if rel:
                line2.append(f"📁{rel}")
        elif field == "session_id" and session_id:
            line2.append(f"🧾{session_id}")
    if line1 and line2:
        return _SEP.join(line1) + "\n" + _SEP.join(line2)
    return _SEP.join(line1 or line2)


def build_footer_line(
    *,
    user_config: dict[str, Any] | None,
    platform_key: str | None,
    model: Optional[str],
    context_tokens: int,
    context_length: Optional[int],
    cwd: Optional[str] = None,
    provider: Optional[str] = None,
    api_calls: Optional[int] = None,
    estimated_cost_usd: Optional[float] = None,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    cache_read_tokens: Optional[int] = None,
    cache_write_tokens: Optional[int] = None,
    compression_count: Optional[int] = None,
    elapsed_seconds: Optional[float] = None,
    session_id: Optional[str] = None,
) -> str:
    """Top-level entry point used by gateway/run.py."""
    cfg = resolve_footer_config(user_config, platform_key)
    if not cfg.get("enabled"):
        return ""
    return format_runtime_footer(
        model=model,
        context_tokens=context_tokens,
        context_length=context_length,
        cwd=cwd,
        fields=cfg.get("fields") or _DEFAULT_FIELDS,
        style=cfg.get("style") or "plain",
        provider=provider,
        api_calls=api_calls,
        estimated_cost_usd=estimated_cost_usd,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        compression_count=compression_count,
        elapsed_seconds=elapsed_seconds,
        session_id=session_id,
    )

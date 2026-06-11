"""hexis_appraisal config — kill switch + appraisal tuning (APPR-06/07).

Reads ``plugins.entries.hexis_appraisal`` from the host config via a lazy
``hermes_cli.config.load_config`` import. On ANY failure (host absent,
config unreadable, malformed entry) every key falls back to its default —
get_cfg() never raises. The result is cached in a module global and reset
per session (on_session_start calls reset_cache(); long-lived gateway
processes serve many sessions — PITFALLS G6).

Plugin-own keys (read from the entry dict):
    enabled               bool, default True   — kill switch (APPR-07);
                          also gates reflection (checked in maybe_reflect)
    confidence_threshold  float, clamped [0,1], default 0.6 (APPR-03)
    deadline_seconds      float, clamped [0.5, 10.0], default 8.0 (R1)
    history_chars         int, default 4000
    max_tokens            int, default 700

Reflection keys (REFL-01, Phase 3):
    reflection_enabled        bool, default True — reflection-only switch
    reflect_every_n_turns     int, clamped [1, 50], default 5 — debounce
    reflect_max_tokens        int, default 700
    reflect_deadline_seconds  float, clamped [0.5, 10.0], default 8.0

Requested model (host trust gate — we only read WHICH model to request;
allow_model_override / allowed_models are enforced by the host):
    plugins.entries.hexis_appraisal.llm.model

Cheap-tier recommendations — documented only, NEVER auto-applied; the user
opts in by configuring the trust gate themselves:
    anthropic: claude-haiku-4-5
    openai:    gpt-4o-mini
    gemini:    gemini-2.5-flash
"""

import logging

logger = logging.getLogger("hermes.plugins.hexis_appraisal.config")

DEFAULT_ENABLED = True
DEFAULT_CONFIDENCE_THRESHOLD = 0.6
DEFAULT_DEADLINE_SECONDS = 8.0
DEFAULT_HISTORY_CHARS = 4000
DEFAULT_MODEL = None  # no override requested by default
DEFAULT_MAX_TOKENS = 700
DEFAULT_REFLECTION_ENABLED = True
DEFAULT_REFLECT_EVERY_N_TURNS = 5
DEFAULT_REFLECT_MAX_TOKENS = 700
DEFAULT_REFLECT_DEADLINE_SECONDS = 8.0

_cache = None


def _load_host_entry():
    """Return the plugins.entries.hexis_appraisal dict, or None.

    Lazy host import — keeps this module importable outside the host.
    Never raises.
    """
    try:
        from hermes_cli.config import load_config

        config = load_config() or {}
        plugins = config.get("plugins")
        if not isinstance(plugins, dict):
            return None
        entries = plugins.get("entries")
        if not isinstance(entries, dict):
            return None
        entry = entries.get("hexis_appraisal")
        return entry if isinstance(entry, dict) else None
    except Exception as exc:
        logger.debug("hexis config unavailable, using defaults: %s", exc)
        return None


def _coerce_bool(value, default):
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "yes", "on", "1"):
            return True
        if lowered in ("false", "no", "off", "0"):
            return False
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _coerce_float(value, default, lo, hi):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if result != result:  # NaN
        return default
    return max(lo, min(hi, result))


def _coerce_int(value, default, lo, hi=None):
    try:
        result = int(value)
    except (TypeError, ValueError):
        return default
    result = max(lo, result)
    return min(hi, result) if hi is not None else result


def get_cfg(force_reload=False) -> dict:
    """Return the effective config dict (cached). Never raises."""
    global _cache
    if _cache is not None and not force_reload:
        return _cache
    entry = _load_host_entry() or {}
    llm_cfg = entry.get("llm")
    if not isinstance(llm_cfg, dict):
        llm_cfg = {}
    model = llm_cfg.get("model")
    model = model.strip() if isinstance(model, str) and model.strip() else None
    _cache = {
        "enabled": _coerce_bool(entry.get("enabled"), DEFAULT_ENABLED),
        "confidence_threshold": _coerce_float(
            entry.get("confidence_threshold"),
            DEFAULT_CONFIDENCE_THRESHOLD, 0.0, 1.0,
        ),
        "deadline_seconds": _coerce_float(
            entry.get("deadline_seconds"), DEFAULT_DEADLINE_SECONDS, 0.5, 10.0
        ),
        "history_chars": _coerce_int(
            entry.get("history_chars"), DEFAULT_HISTORY_CHARS, 0
        ),
        "model": model,
        "max_tokens": _coerce_int(entry.get("max_tokens"), DEFAULT_MAX_TOKENS, 1),
        "reflection_enabled": _coerce_bool(
            entry.get("reflection_enabled"), DEFAULT_REFLECTION_ENABLED
        ),
        "reflect_every_n_turns": _coerce_int(
            entry.get("reflect_every_n_turns"),
            DEFAULT_REFLECT_EVERY_N_TURNS, 1, 50,
        ),
        "reflect_max_tokens": _coerce_int(
            entry.get("reflect_max_tokens"), DEFAULT_REFLECT_MAX_TOKENS, 1
        ),
        "reflect_deadline_seconds": _coerce_float(
            entry.get("reflect_deadline_seconds"),
            DEFAULT_REFLECT_DEADLINE_SECONDS, 0.5, 10.0,
        ),
    }
    return _cache


def reset_cache() -> None:
    """Clear the cached config (called from on_session_start)."""
    global _cache
    _cache = None

"""Live engine telemetry for the managed llama-server router (issue #123678).

The Desktop "Local Models" pane, the TUI status bar and CLI ``/status`` all read from here so
they cannot drift: pure parsers (``rates_from_timings``, ``slots_usage``, ``context_percent``)
plus a best-effort poller (``get_runtime_stats``) and a process-local last-turn store fed by
the shared response intake (``agent/turn_usage.py`` records llama-server ``timings`` there).

Every poller returns ``None`` on any failure — telemetry is garnish, never load-bearing — and
payloads carry pre-formatted labels the UI shows verbatim (no new i18n keys).
"""

from __future__ import annotations

import threading
import time
from contextlib import suppress
from typing import Any
from urllib.parse import quote

from hermes_cli.local_runtime.endpoint import managed_get_json, managed_root

# Per-model stats stay fresh across the status-bar repaint rate without polling the router
# on every frame (a repaint storm must not become an HTTP storm).
_STATS_TTL_S = 5.0

_lock = threading.Lock()
_last_turn: dict[str, dict[str, Any]] = {}
_stats_cache: dict[str, tuple[float, dict[str, Any] | None]] = {}


def _finite_positive(value: Any) -> float | None:
    """Positive finite float, or None (llama-server timings use ms floats; -0.8s-style
    garbage from the field must never render as a negative tok/s)."""
    with suppress(TypeError, ValueError, OverflowError):
        number = float(value)
        if number == number and 0 < number < 1e12:
            return number
    return None


def rates_from_timings(timings: Any) -> tuple[float | None, float | None]:
    """``(prompt_tps, gen_tps)`` from a llama-server chat-completion ``timings`` object.

    Handles both the count/duration shape (``prompt_n``/``prompt_ms``,
    ``predicted_n``/``predicted_ms``) and builds that already divide
    (``prompt_per_second``/``predicted_per_second``). Anything missing or
    non-positive reads as ``None``.
    """
    if not isinstance(timings, dict):
        return (None, None)

    def _rate(count_keys: tuple[str, ...], ms_keys: tuple[str, ...],
              direct_keys: tuple[str, ...]) -> float | None:
        for key in direct_keys:
            direct = _finite_positive(timings.get(key))
            if direct is not None:
                return direct
        count = None
        for key in count_keys:
            count = _finite_positive(timings.get(key))
            if count is not None:
                break
        ms = None
        for key in ms_keys:
            ms = _finite_positive(timings.get(key))
            if ms is not None:
                break
        if count is None or ms is None:
            return None
        return count / (ms / 1000.0)

    prompt_tps = _rate(("prompt_n", "n_prompt"), ("prompt_ms", "prompt_time_ms"),
                       ("prompt_per_second",))
    gen_tps = _rate(("predicted_n", "n_predicted"), ("predicted_ms", "predicted_time_ms"),
                    ("predicted_per_second",))
    return (prompt_tps, gen_tps)


def extract_timings(response: Any) -> dict | None:
    """The llama-server ``timings`` object off a completion response, whatever shape the
    provider SDK surfaced it in (attribute, ``model_extra`` passthrough, or raw dict)."""
    if response is None:
        return None
    direct = getattr(response, "timings", None)
    if isinstance(direct, dict):
        return direct
    extra = getattr(response, "model_extra", None)
    if isinstance(extra, dict) and isinstance(extra.get("timings"), dict):
        return extra["timings"]
    hidden = getattr(response, "_hidden_params", None)
    if isinstance(hidden, dict) and isinstance(hidden.get("timings"), dict):
        return hidden["timings"]
    if isinstance(response, dict) and isinstance(response.get("timings"), dict):
        return response["timings"]
    return None


def record_last_turn(model_id: str, prompt_tps: float | None, gen_tps: float | None,
                     prompt_tokens: int | None = None,
                     completion_tokens: int | None = None) -> None:
    """Remember the last turn's engine rates for ``model_id`` (in-memory only: a restart
    showing no rates beats showing a stale turn's)."""
    if not model_id:
        return
    with _lock:
        _last_turn[str(model_id)] = {
            "prompt_tps": prompt_tps, "gen_tps": gen_tps,
            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "ts": time.time(),
        }


def last_turn(model_id: str) -> dict[str, Any] | None:
    """Last recorded engine rates for ``model_id``, or None when no turn was seen."""
    if not model_id:
        return None
    with _lock:
        stored = _last_turn.get(str(model_id))
        return dict(stored) if stored is not None else None


def record_response_timings(model_id: str, response: Any) -> None:
    """One-guard intake helper: pull ``timings`` off ``response`` and stash this turn's
    rates. No-op unless the engine actually sent timings (only llama-server does)."""
    timings = extract_timings(response)
    if timings is None:
        return
    prompt_tps, gen_tps = rates_from_timings(timings)
    if prompt_tps is None and gen_tps is None:
        return
    prompt_tokens = completion_tokens = None
    with suppress(TypeError, ValueError):
        usage = getattr(response, "usage", None)
        if usage is not None:
            prompt_tokens = getattr(usage, "prompt_tokens", None) or getattr(
                usage, "input_tokens", None)
            completion_tokens = getattr(usage, "completion_tokens", None) or getattr(
                usage, "output_tokens", None)
            if isinstance(usage, dict):
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                completion_tokens = usage.get("completion_tokens", completion_tokens)
    record_last_turn(model_id, prompt_tps, gen_tps, prompt_tokens, completion_tokens)


def context_percent(used_tokens: Any, n_ctx: Any) -> int | None:
    """``used / n_ctx`` as a clamped 0-100 int, or None when the window is unknown."""
    with suppress(TypeError, ValueError, OverflowError):
        total = int(n_ctx)
        if total > 0:
            return max(0, min(100, round(int(used_tokens) / total * 100)))
    return None


def slots_usage(slots: Any) -> tuple[int | None, int | None]:
    """``(used_tokens, n_ctx)`` from a ``/slots`` payload, defensively.

    The busiest slot wins (a parallel small request idles while a live turn fills its
    window); ``n_ctx`` is the widest slot window seen. Unknown shapes read as
    ``(None, None)`` — the poller falls back to ``/props`` for the window.
    """
    if not isinstance(slots, list):
        return (None, None)
    used: int | None = None
    window: int | None = None
    for slot in slots:
        if not isinstance(slot, dict):
            continue
        with suppress(TypeError, ValueError):
            slot_ctx = slot.get("n_ctx")
            if slot_ctx is not None and int(slot_ctx) > 0:
                window = max(window or 0, int(slot_ctx))
        total = 0
        seen = False
        for key in ("n_prompt_tokens_processed", "n_tokens_predicted", "n_past", "n_prompt"):
            with suppress(TypeError, ValueError):
                value = slot.get(key)
                if value is not None:
                    total += int(value)
                    seen = True
        if seen:
            used = max(used or 0, total)
    return (used, window)


def _props_window(props: Any) -> int | None:
    """Granted ``n_ctx`` from a ``/props`` payload (the server's grant, not the request)."""
    if isinstance(props, dict):
        with suppress(TypeError, ValueError):
            n_ctx = (props.get("default_generation_settings") or {}).get("n_ctx")
            if n_ctx is not None and int(n_ctx) > 0:
                return int(n_ctx)
    return None


def _format_tps(value: float | None) -> str:
    return f"{value:.1f}" if value is not None else "--"


def _k_label(tokens: int) -> str:
    return f"{tokens // 1024}K"


def get_runtime_stats(model_id: str | None, *, _no_cache: bool = False) -> dict[str, Any] | None:
    """Live engine stats for one managed-router child: granted window, used context and
    last-turn throughput, with pre-formatted labels for direct display.

    ``None`` when there is no managed server, the model is unknown/unloaded, or any probe
    fails. Results are TTL-cached per model (status-bar repaints must not poll per frame).
    """
    if not model_id:
        return None
    key = str(model_id)
    now = time.monotonic()
    if not _no_cache:
        with _lock:
            cached = _stats_cache.get(key)
        if cached is not None and now - cached[0] < _STATS_TTL_S:
            return dict(cached[1]) if cached[1] is not None else None
    stats = _poll_runtime_stats(key)
    # A copied dict return so callers can never mutate the cache entry.
    result = dict(stats) if stats is not None else None
    with _lock:
        _stats_cache[key] = (now, result)
    return dict(result) if result is not None else None


def _poll_runtime_stats(model_id: str) -> dict[str, Any] | None:
    endpoint = managed_root()
    if endpoint is None:
        return None
    base, api_key = endpoint
    try:
        props = managed_get_json(base, api_key, f"/props?model={quote(model_id)}", timeout_s=3)
        slots = managed_get_json(base, api_key, f"/slots?model={quote(model_id)}", timeout_s=3)
    except Exception:
        return None
    n_ctx = _props_window(props)
    used, slots_ctx = slots_usage(slots)
    if n_ctx is None:
        n_ctx = slots_ctx
    if n_ctx is None:
        return None
    percent = context_percent(used if used is not None else 0, n_ctx)
    turn = last_turn(model_id) or {}
    prompt_tps = turn.get("prompt_tps")
    gen_tps = turn.get("gen_tps")
    used_label = f"{used:,}" if used is not None else "--"
    return {
        "model_id": model_id,
        "n_ctx": n_ctx,
        "n_ctx_label": _k_label(n_ctx),
        "used_tokens": used,
        "context_percent": percent,
        "context_label": (f"{used_label} / {n_ctx:,} tokens ({percent}% of {_k_label(n_ctx)})"
                          if percent is not None else f"ctx {_k_label(n_ctx)}"),
        "prompt_tps": prompt_tps,
        "gen_tps": gen_tps,
        "throughput_label": f"{_format_tps(prompt_tps)} prompt · {_format_tps(gen_tps)} gen tok/s",
    }

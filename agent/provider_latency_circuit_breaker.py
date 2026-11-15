"""provider_latency_circuit_breaker.py — autonomous circuit breaker for degraded LLM providers.

Tracks rolling latencies and error rates per (model, upstream_provider). When a provider's
rolling latency exceeds DEGRADED_LATENCY_THRESHOLD_S (default 20.0s) or incurs consecutive timeouts,
it is temporarily demoted in OpenRouter provider_routing.order for COOLDOWN_SECONDS (default 600s).
Once the cooldown expires, it is given an exploratory probe request to re-evaluate health.
"""
from __future__ import annotations

import collections
import logging
import threading
import time
from typing import Sequence

logger = logging.getLogger(__name__)

_DEGRADED_LATENCY_THRESHOLD_S = 20.0
_COOLDOWN_SECONDS = 600.0
_WINDOW_SIZE = 5

# DQ #500 (2026-10-21, rca-openrouter-shared-pool-429): the flat 20s base demoted
# HEALTHY long-context calls — census 21/10 over agent.log: google/gemini-4-argon at
# 100-200k input tokens p90 41.8s / 200k+ p90 52.7s are generation physics, yet
# 34.8% of those calls sat over the base → rolling demotion of both providers →
# all-demoted → glm-5.3 model-escalation flapping. Threshold now scales with input
# tokens; reasoning-heavy models get extra headroom on top (both axes from census).
_LATENCY_S_PER_1K_TOKENS = 0.25  # +0.25s per 1k input tokens (150k ctx → +37.5s)
_MODEL_LATENCY_MULTIPLIERS = {
    # high-effort reasoning model: healthy p95 at 100-200k ctx is 56.7s, far over
    # what context scaling alone covers at the low end of that bucket
    "google/gemini-4-argon": 1.25,
}

# DQ #500 del 2b: OpenRouter gemini endpoints — the SERVED provider name the API
# reports (e.g. "Google AI Studio", "Google" = Vertex) and the endpoint TAGS used
# in provider_routing.models order ("google-ai-studio", "google-ai-studio/flex",
# "google-vertex") must share ONE breaker key, else demotion never matches the
# pinned order entries and all_providers_demoted never fires for the pin.
# Kvitto 21/10 live probes: both google-ai-studio tags serve under the name
# "Google AI Studio" (flex at half price); Vertex serves as "Google".
_PROVIDER_KEY_ALIASES = {
    "google ai studio": "google-ai-studio",
    "google": "google-vertex",
}

_lock = threading.Lock()
# (model, provider) -> deque of float durations
_history: dict[tuple[str, str], collections.deque[float]] = {}
# (model, provider) -> deque of per-sample context-scaled thresholds matching _history
_thresholds: dict[tuple[str, str], collections.deque[float]] = {}
# (model, provider) -> demoted_until_timestamp
_demoted_until: dict[tuple[str, str], float] = {}
# (model, provider) -> explicit_failure_demoted_until_timestamp (immune to early latency recovery from in-flight calls)
_failure_demoted_until: dict[tuple[str, str], float] = {}
# (model, provider) -> warning_suppressed_until_timestamp (prevents flapping across the threshold within 600s from re-warning)
_warned_until: dict[tuple[str, str], float] = {}


def _breaker_provider_key(provider: str) -> str:
    """Canonical breaker key for an upstream provider (served name OR order tag)."""
    p = str(provider).strip().lower()
    if p in _PROVIDER_KEY_ALIASES:
        return _PROVIDER_KEY_ALIASES[p]
    if p.startswith("google-ai-studio"):
        return "google-ai-studio"  # covers the /flex endpoint tag
    if p.startswith("google-vertex"):
        return "google-vertex"
    return p


def _latency_threshold_s(model: str, context_tokens: int | None) -> float:
    """Context- and model-aware demotion threshold (DQ #500 del 2).

    Base 20s for small contexts; +0.25s per 1k input tokens so healthy long-context
    generation time does not demote; per-model multiplier for reasoning-heavy models.
    ``context_tokens`` None/0 falls back to the (multiplied) base.
    """
    threshold = _DEGRADED_LATENCY_THRESHOLD_S
    if context_tokens and context_tokens > 0:
        threshold += _LATENCY_S_PER_1K_TOKENS * (float(context_tokens) / 1000.0)
    mult = _MODEL_LATENCY_MULTIPLIERS.get(str(model or "").lower())
    if mult and mult > 0:
        threshold *= mult
    return threshold


def record_provider_latency(model: str, provider: str, duration_s: float,
                             context_tokens: int | None = None) -> None:
    if not model or not provider or duration_s <= 0:
        return
    key = (model.lower(), _breaker_provider_key(provider))
    threshold = _latency_threshold_s(model, context_tokens)
    now = time.monotonic()
    with _lock:
        if key not in _history:
            _history[key] = collections.deque(maxlen=_WINDOW_SIZE)
        if key not in _thresholds or len(_thresholds[key]) != len(_history[key]):
            _thresholds[key] = collections.deque(
                [_latency_threshold_s(model, None)] * len(_history[key]),
                maxlen=_WINDOW_SIZE,
            )
        _history[key].append(duration_s)
        _thresholds[key].append(threshold)

        # Check if rolling average exceeds rolling context-scaled window threshold
        avg = sum(_history[key]) / len(_history[key])
        eff_threshold = sum(_thresholds[key]) / len(_thresholds[key])
        if len(_history[key]) >= 2 and avg >= eff_threshold:
            prev_until = _demoted_until.get(key)
            warn_until = _warned_until.get(key)
            already_demoted = (prev_until is not None and now < prev_until) or (
                warn_until is not None and now < warn_until
            )
            until = now + _COOLDOWN_SECONDS
            _demoted_until[key] = until
            if not already_demoted:
                _warned_until[key] = until
            log_fn = logger.debug if already_demoted else logger.warning
            log_fn(
                "provider circuit-breaker: %s @ %s degraded (avg=%.1fs over %d calls, threshold=%.1fs); demoting for %.0fs",
                model, provider, avg, len(_history[key]), eff_threshold, _COOLDOWN_SECONDS,
            )
        elif (
            key in _demoted_until
            and duration_s < threshold
            and avg < eff_threshold
            and now >= _failure_demoted_until.get(key, 0.0)
        ):
            # Recovered early on probe (only for latency demotions; explicit failure/429
            # demotions hold until their cooldown expires so in-flight calls cannot wipe them)
            del _demoted_until[key]
            _failure_demoted_until.pop(key, None)
            logger.info("provider circuit-breaker: %s @ %s recovered (latest=%.1fs)", model, provider, duration_s)


def record_provider_failure(model: str, provider: str) -> None:
    if not model or not provider:
        return
    key = (model.lower(), _breaker_provider_key(provider))
    now = time.monotonic()
    with _lock:
        until = now + _COOLDOWN_SECONDS
        _demoted_until[key] = until
        _failure_demoted_until[key] = until
        logger.warning(
            "provider circuit-breaker: %s @ %s recorded failure/timeout; demoting for %.0fs",
            model, provider, _COOLDOWN_SECONDS,
        )


def is_provider_demoted(model: str, provider: str) -> bool:
    if not model or not provider:
        return False
    key = (model.lower(), _breaker_provider_key(provider))
    now = time.monotonic()
    with _lock:
        until = _demoted_until.get(key)
        if until is None:
            return False
        if now >= until:
            del _demoted_until[key]
            _failure_demoted_until.pop(key, None)
            return False
        return True


def adjust_provider_order(model: str, order: Sequence[str]) -> list[str]:
    """Sort provider order so healthy providers are tried before degraded ones."""
    if not order or not model:
        return list(order)
    healthy = []
    demoted = []
    for p in order:
        if is_provider_demoted(model, p):
            demoted.append(p)
        else:
            healthy.append(p)
    if demoted and healthy:
        logger.debug(
            "provider circuit-breaker adjusted order for %s: healthy=%s, demoted=%s",
            model, healthy, demoted,
        )
        return healthy + demoted
    return list(order)


def all_providers_demoted(model: str, order: Sequence[str] | None = None) -> bool:
    """True when every provider available to *model* is in latency cooldown.

    With a pinned provider ``order`` (``provider_routing.models.<id>.order``) the verdict
    covers that full population — including providers never observed this session. Without
    one, it covers the providers the breaker has actually seen for the model (history ∪
    cooldowns) and requires at least two, so a single flaky provider never escalates the
    model. Companion to the model-escalation guard (DQ #218): the breaker alone only
    reorders providers, so an all-demoted model would keep stalling.
    """
    if not model:
        return False
    now = time.monotonic()
    m = model.lower()
    with _lock:
        # Same lazy expiry as is_provider_demoted: drop lapsed cooldowns first.
        for key in [k for k, until in _demoted_until.items() if now >= until]:
            del _demoted_until[key]
            _failure_demoted_until.pop(key, None)
        if order:
            providers = [_breaker_provider_key(str(p)) for p in order if str(p)]
            if not providers:
                return False
            return all((m, p) in _demoted_until for p in providers)
        seen = {p for (mm, p) in set(_demoted_until) | set(_history) if mm == m}
        return len(seen) >= 2 and all((m, p) in _demoted_until for p in seen)


_ESCALATION_RETRY_S = 60.0
# model -> monotonic timestamp of the last model-escalation attempt
_last_escalation_attempt: dict[str, float] = {}


def latency_escalation_due(model: str) -> bool:
    """True when the model-escalation guard may attempt a fallback switch for *model*.

    Throttles repeat attempts (exhausted chains, flapping providers) to one per
    ``_ESCALATION_RETRY_S`` so the retry loop and parallel sessions don't walk a dead
    fallback chain on every pass.
    """
    if not model:
        return False
    with _lock:
        last = _last_escalation_attempt.get(model.lower())
        return last is None or (time.monotonic() - last) >= _ESCALATION_RETRY_S


def mark_latency_escalation(model: str) -> None:
    """Record that a model-escalation attempt was made for *model* now."""
    if model:
        with _lock:
            _last_escalation_attempt[model.lower()] = time.monotonic()


def reset_circuit_breaker() -> None:
    """Testing helper to clear all in-memory breaker state."""
    with _lock:
        _history.clear()
        _thresholds.clear()
        _demoted_until.clear()
        _failure_demoted_until.clear()
        _warned_until.clear()
        _last_escalation_attempt.clear()

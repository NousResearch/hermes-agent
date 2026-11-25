"""Cross-process token-bucket pacer for a free primary route whose 429s are token-budgeted.

Measured against the Nous free route on 2026-11-20: one 26k-token request exhausted the
allowance for 52s, one 84k-token request for 122s, and a 23-token request was accepted
immediately after a refill. Request count is irrelevant and a 1-token probe cannot tell you
whether a real turn will fit (see ``agent/model_health_ledger.py`` for the probe verdict).

Without pacing every lane aims a 40k-token call at the same allowance; most are refused, the
turn falls back, and the budget the primary would have served a minute later is wasted on a
429. This module meters the shared budget before a request is sent: it refills at a configured
rate, charges the request's estimated prompt tokens, and makes the caller wait for the
clearance. Two large turns therefore cannot collide — the second waits for what the first
consumed. State is one JSON object keyed ``provider/model`` beside the rate-limit ledger, with
the same advisory-lock read-modify-write so lanes in separate processes share one bucket.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import time
from typing import Any, Mapping

logger = logging.getLogger(__name__)

# Off unless a config opts in: most routes bill per token and must never be throttled here.
DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "providers": ["nous"],
    "tokens_per_minute": 33000.0,
    # One request's worth: Hermes refuses a context window below 64k, so this is the floor a
    # single turn can cost and the bucket must be able to hold at least that much.
    "burst_tokens": 64000.0,
    "max_wait_seconds": 240.0,
    # Skip (rather than sleep through) a provider window longer than this: it cannot open inside
    # the wait, so sleeping part of it only delays the 429 that sends the turn to its fallback.
    # 0 = inherit max_wait_seconds and wait it out, as before.
    "max_window_wait_seconds": 0.0,
}

_now = time.time


def _state_path() -> str:
    try:
        from hermes_constants import get_hermes_home
        base = str(get_hermes_home())
    except ImportError:
        base = os.path.join(os.path.expanduser("~"), ".hermes")
    return os.path.join(base, "rate_limits", "primary_budget.json")


def _key(provider: str, model: str) -> str:
    return f"{(provider or '').strip().lower()}/{(model or '').strip()}"


def _settings() -> dict[str, Any]:
    """``fallback.primary_budget`` merged over DEFAULTS; a broken config means disabled."""
    try:
        from hermes_cli.config import load_config
        raw = ((load_config() or {}).get("fallback") or {}).get("primary_budget") or {}
    except (ImportError, OSError, ValueError, TypeError, KeyError, AttributeError):
        return dict(DEFAULTS)
    cfg = dict(DEFAULTS)
    if isinstance(raw, Mapping):
        cfg.update({k: v for k, v in raw.items() if k in DEFAULTS})
    return cfg


@contextlib.contextmanager
def _locked():
    """Advisory lock around read-modify-write; a no-op where flock is unavailable."""
    lock_path = _state_path() + ".lock"
    handle = None
    try:
        os.makedirs(os.path.dirname(lock_path), exist_ok=True)
        handle = open(lock_path, "a+", encoding="utf-8")
    except OSError as exc:
        logger.debug("Primary-budget lock unavailable: %s", exc)
    try:
        if handle is not None:
            try:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            except (ImportError, OSError):
                pass
        yield
    finally:
        if handle is not None:
            with contextlib.suppress(OSError):
                handle.close()


def _read() -> dict:
    try:
        with open(_state_path(), encoding="utf-8-sig") as handle:
            data = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return {k: v for k, v in data.items() if isinstance(v, dict)} if isinstance(data, dict) else {}


def _write(state: dict) -> None:
    try:
        from utils import atomic_write_text

        atomic_write_text(_state_path(), json.dumps(state))
    except (OSError, ValueError, TypeError) as exc:  # never let bookkeeping break a turn
        logger.debug("Failed to write primary-budget state: %s", exc)


def _payload_chars(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value)
    if isinstance(value, Mapping):
        return sum(_payload_chars(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_payload_chars(v) for v in value)
    return len(str(value))


def estimate_prompt_tokens(api_kwargs: Mapping[str, Any]) -> int:
    """Rough prompt size of a chat request: ~4 characters per token over messages + tools.

    Deliberately cheap: the payload is already walked once by the SDK, and an estimate is all
    the bucket needs. Returns 0 when the shape is unfamiliar (the caller then sends as before).
    """
    try:
        chars = _payload_chars(api_kwargs.get("messages"))
        chars += _payload_chars(api_kwargs.get("tools"))
    except Exception:  # health: allow BLE001 -- walks caller-supplied SDK payloads whose shapes are open-ended; an unfamiliar shape must estimate as 0, never raise mid-request
        return 0
    return chars // 4


def reserve(provider: str, model: str, tokens: float, *, now: float | None = None,
            cfg: Mapping[str, Any] | None = None) -> tuple[bool, float]:
    """Charge ``tokens`` to the shared bucket. Returns ``(granted, seconds_until_clearance)``.

    A route with no state starts full. Refill is capped at the burst (one request's worth), so a
    request bigger than the burst reserves the whole bucket and waits for a full refill rather
    than for a balance that can never be reached.
    """
    cfg = cfg or _settings()
    burst = max(1.0, float(cfg.get("burst_tokens") or DEFAULTS["burst_tokens"]))
    rate = max(0.1, float(cfg.get("tokens_per_minute") or DEFAULTS["tokens_per_minute"]) / 60.0)
    charge = min(max(0.0, float(tokens)), burst)
    if charge <= 0:
        return True, 0.0
    at = _now() if now is None else float(now)
    key = _key(provider, model)
    with _locked():
        state = _read()
        entry = state.get(key) or {}
        if "tokens" in entry:
            available = min(burst, float(entry.get("tokens") or 0.0)
                            + max(0.0, at - float(entry.get("updated_at") or at)) * rate)
        else:
            available = burst
        if available >= charge:
            state[key] = {"tokens": available - charge, "updated_at": at}
            _write(state)
            return True, 0.0
        # Persist the refilled balance so the wait is not lost to the next reader.
        state = {k: v for k, v in state.items()
                 if k == key or float(v.get("tokens") or 0.0) > 0}
        state[key] = {"tokens": available, "updated_at": at}
        _write(state)
    return False, (charge - available) / rate


def pace_primary_request(agent, api_kwargs: Mapping[str, Any], *, sleep=time.sleep) -> float:
    """Wait until the shared bucket can pay for this request; return seconds waited.

    Applies only when the pacer is enabled, the live route is the configured primary (never a
    fallback, which has its own budget) and the provider is one the config names. Any failure
    here is skipped rather than raised: pacing must never fail a turn.
    """
    try:
        cfg = _settings()
        if not cfg.get("enabled"):
            return 0.0
        provider = str(getattr(agent, "provider", "") or "").strip().lower()
        allowed = {str(p).strip().lower() for p in (cfg.get("providers") or [])}
        if not provider or provider not in allowed:
            return 0.0
        if getattr(agent, "_fallback_activated", False):
            return 0.0
        from agent.fallback_cooldown import _on_primary_route
        if not _on_primary_route(agent):
            return 0.0
        tokens = estimate_prompt_tokens(api_kwargs)
        if tokens <= 0:
            return 0.0
        model = str(getattr(agent, "model", "") or "")
        budget = max(0.0, float(cfg.get("max_wait_seconds") or 0.0))
        # A provider window longer than this is not worth sleeping through at all. Unset (0)
        # keeps the older bounded-wait contract: sleep up to max_wait_seconds, then send.
        window_skip_budget = max(0.0, float(cfg.get("max_window_wait_seconds") or 0.0))
        window_budget = window_skip_budget or budget
        waited = 0.0

        def _window_remaining() -> float:
            try:
                from agent.model_health_ledger import model_window_remaining
                return max(0.0, float(model_window_remaining(provider, model) or 0.0))
            except (ImportError, OSError, ValueError, TypeError, KeyError):
                return 0.0

        def _adopt_after_wait() -> None:
            # A Nous agent key is a short-lived JWT, and the per-iteration pre-expiry adoption ran
            # BEFORE this wait. Without a second adoption a minute-long pace can outlive the key
            # and turn the request we just waited for into a 401 (observed 15:46: 158s wait → 401).
            try:
                adopt = getattr(agent, "_adopt_nous_key_before_expiry", None)
                if callable(adopt):
                    adopt()
            except Exception:
                logger.debug("Primary budget pacer: credential re-adoption skipped", exc_info=True)

        # The provider's own retry_after outranks our bucket: another lane may have recorded a
        # window while our local balance looked healthy, and a probe that passed a minute ago does
        # not reopen it (observed 13:00:27: 60s waited, sent into a 112s window, 429). Re-read the
        # ledger after every pause: a sibling lane can extend the window while we sleep.
        while True:
            remaining = _window_remaining()
            if remaining <= 0:
                break
            if window_skip_budget and remaining + 1.0 > window_skip_budget:
                # Waiting cannot open it inside the budget; send now and let the 429 (and the
                # retry budget in turn_api_error) move the turn on instead of idling here.
                logger.warning(
                    "Primary budget pacer: %s inside a recorded %.0fs window, window budget %.0fs — sending on",
                    model, remaining, window_skip_budget,
                )
                break
            pause = min(remaining + 1.0, window_budget - waited, budget - waited)
            if pause <= 0:
                logger.warning(
                    "Primary budget pacer: %s inside a recorded %.0fs window, max_wait %.0fs — sending on",
                    model, remaining, budget,
                )
                break
            sleep(pause)
            waited += pause
        while True:
            granted, clearance = reserve(provider, model, tokens, cfg=cfg)
            if granted:
                if waited >= 1.0:
                    logger.info(
                        "Primary budget pacer: waited %.0fs for a %d-token request on %s/%s",
                        waited, tokens, provider, model,
                    )
                    _adopt_after_wait()
                return waited
            if waited + clearance > budget:
                logger.warning(
                    "Primary budget pacer: %d tokens need %.0fs but max_wait is %.0fs — sending on",
                    tokens, clearance, budget,
                )
                _adopt_after_wait()
                return waited
            sleep(max(0.05, min(clearance, 5.0)))
            waited += max(0.05, min(clearance, 5.0))
    except Exception:
        logger.debug("Primary budget pacer skipped", exc_info=True)
        return 0.0


__all__ = ["estimate_prompt_tokens", "pace_primary_request", "reserve"]

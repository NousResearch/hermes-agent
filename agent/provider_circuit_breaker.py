"""Provider-level rolling-window circuit breaker for the fallback chain.

Two existing mechanisms cool things down, and neither answers the question this
module exists for:

* ``agent/credential_pool.py`` cools down individual **keys** inside one provider
  (a 429/402 on a credential), and
* ``agent/fallback_cooldown.py`` cools down the **primary** after *consecutive*
  rate-limit reasons.

Neither tracks "this fallback backend has failed N times in the last hour, so stop
wasting an RTT on it every single turn". A flapping fallback provider therefore gets
retried on every failover forever: it is skipped only for the session it already
failed in (``agent._unavailable_fallback_keys``), and that set is per-turn state.

This module adds exactly that missing piece: a durable, per-backend rolling-window
failure counter that quiesces a backend once it crosses a threshold, and clears the
window on a success so recovery is automatic.

Design notes
------------
* Identity is ``(provider, normalized base_url)`` — provider-level, not per-model,
  matching the intent ("this backend is unhealthy"), and reusing the same base_url
  normalization as ``hermes_cli.fallback_config``.
* State is wall-clock (``time.time``) because it is persisted and must survive
  restarts, mirroring ``CredentialPool`` (which also persists wall-clock timestamps).
* Storage lives under ``get_hermes_home()`` so profile isolation and native-Windows
  path handling are respected. Only provider/base_url/timestamps/counters are ever
  written — never credentials.
* The module imports nothing from ``agent``/``hermes_cli`` at import time; config is
  read lazily to avoid import cycles.

Configuration (config.yaml)::

    # The breaker auto-follows the project's API pool: it is active whenever a
    # fallback_providers (or legacy fallback_model) chain exists, and inert when
    # there is no pool. `enabled` is an optional hard kill-switch — set it to
    # false to force the feature off even when a pool exists. window/fail/cooldown
    # tune the rolling-window behaviour. Defaults are gentle (5 failures within 10
    # minutes -> 2 minute quiesce), the threshold/cooling can be tightened in config.
    fallback_circuit_breaker:
      window_seconds: 600       # rolling window the failure count is measured over (10m)
      fail_threshold: 5        # failures within the window before quiescing
      cooldown_seconds: 120    # how long a quiesced backend is skipped (2m)
      # Opt-in half-open self-healing. Default off (conservative): a quiesced
      # backend is retried only after the cooldown elapses. When enabled, it is
      # probed with ONE real attempt after `probe_interval_seconds` (default: half
      # the cooldown); a success recovers immediately, a failure re-arms cooldown.
      auto_probe: false
      probe_interval_seconds: 60  # optional; defaults to cooldown / 2

Public API::

    record_failure(provider, base_url)  -> bool   # True when this tripped the breaker
    record_success(provider, base_url)
    is_quiesced(provider, base_url)     -> bool
    remaining_cooldown(provider, base_url) -> float
    status() -> str
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── defaults ────────────────────────────────────────────────────────────────

DEFAULT_WINDOW_SECONDS = 600
DEFAULT_FAIL_THRESHOLD = 5
DEFAULT_COOLDOWN_SECONDS = 120

# Half-open auto-probe: OFF by default (conservative). When enabled, a quiesced
# backend is retried with a single real attempt after ``probe_interval_seconds``
# instead of waiting out the whole cooldown — a successful probe recovers
# immediately, a failed one re-arms a fresh cooldown.
DEFAULT_AUTO_PROBE = False

# Hard bounds so a pathological config cannot disable the breaker or wedge it for
# a week. ``window_seconds``/``cooldown_seconds`` floor at 1s; threshold floors at 1.
MIN_WINDOW_SECONDS = 1.0
MIN_COOLDOWN_SECONDS = 1.0
MIN_PROBE_INTERVAL_SECONDS = 1.0

# Cap the per-backend timestamp list so a long-lived process cannot grow it without
# bound. Only the window matters, so this is comfortably above any sane threshold.
MAX_TRACKED_FAILURES = 256

STATE_VERSION = 1


# ── settings ────────────────────────────────────────────────────────────────

_settings_cache: Optional[Dict[str, Any]] = None
_settings_lock = threading.Lock()

def _coerce_positive(raw: Any, default: float, floor: float) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(value) or value <= 0:
        return default
    return max(value, floor)


def _coerce_threshold(raw: Any, default: int) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError, OverflowError):
        return default
    return value if value >= 1 else default



def get_settings(force_reload: bool = False) -> Dict[str, Any]:
    """Return the effective breaker settings, defaulting when unconfigured.

    Never raises: a broken config must not take down the failover path.
    """
    global _settings_cache
    if _settings_cache is not None and not force_reload:
        return _settings_cache
    with _settings_lock:
        if _settings_cache is not None and not force_reload:
            return _settings_cache
        raw: Dict[str, Any] = {}
        try:
            from hermes_cli.config import load_config

            config = load_config() or {}
            block = config.get("fallback_circuit_breaker")
            if isinstance(block, dict):
                raw = block
        except Exception as exc:  # pragma: no cover - config layer is best-effort
            logger.debug("circuit breaker: config unavailable, using defaults: %s", exc)
        # The breaker rides on the project's existing "API pool": it only matters
        # when the user has actually configured a multi-provider fallback chain
        # (``fallback_providers``, plus legacy ``fallback_model``). No pool => the
        # feature is inert, so a single-provider user can never be "banned" by it.
        configured_chain = False
        try:
            from hermes_cli.fallback_config import get_fallback_chain

            configured_chain = bool(get_fallback_chain(config))
        except Exception:  # pragma: no cover - fallback config is best-effort
            configured_chain = False
        explicit_enabled = raw.get("enabled")
        if isinstance(explicit_enabled, bool) and not explicit_enabled:
            # An explicit `enabled: false` is honoured as a hard, conservative kill
            # switch even when a fallback chain exists.
            enabled = False
        else:
            # Otherwise auto-follow the pool: on iff a fallback chain is configured.
            enabled = configured_chain
        settings = {
            "enabled": enabled,
            "window_seconds": _coerce_positive(raw.get("window_seconds"), float(DEFAULT_WINDOW_SECONDS), MIN_WINDOW_SECONDS),
            "fail_threshold": _coerce_threshold(raw.get("fail_threshold"), DEFAULT_FAIL_THRESHOLD),
            "cooldown_seconds": _coerce_positive(raw.get("cooldown_seconds"), float(DEFAULT_COOLDOWN_SECONDS), MIN_COOLDOWN_SECONDS),
            # Half-open auto-probe: opt-in. When enabled, a quiesced backend is
            # retried with a single real attempt after ``probe_interval_seconds``
            # (default: half the cooldown) instead of waiting out the full cooldown.
            "auto_probe": bool(raw.get("auto_probe", DEFAULT_AUTO_PROBE)),
            "probe_interval_seconds": _coerce_positive(raw.get("probe_interval_seconds"), 0.0, MIN_PROBE_INTERVAL_SECONDS),
        }
        _settings_cache = settings
        return settings


def _effective_probe_interval(settings: Dict[str, Any]) -> float:
    """Seconds before a quiesced backend is probed (half-open gate).

    Explicit ``probe_interval_seconds`` wins; otherwise default to half the
    cooldown so a quiesced backend wakes up to a probe well before permanent
    break, but not so eagerly that it hammers a still-down provider.
    """
    explicit = float(settings.get("probe_interval_seconds") or 0.0)
    if explicit > 0:
        return max(explicit, MIN_PROBE_INTERVAL_SECONDS)
    cooldown = float(settings.get("cooldown_seconds") or DEFAULT_COOLDOWN_SECONDS)
    return max(cooldown / 2.0, MIN_PROBE_INTERVAL_SECONDS)


def _reset_settings_cache_for_tests() -> None:
    global _settings_cache
    with _settings_lock:
        _settings_cache = None


# ── identity ────────────────────────────────────────────────────────────────

def normalize_base_url(value: Any) -> str:
    return value.strip().rstrip("/").lower() if isinstance(value, str) else ""


def _backend_key(provider: str, base_url: str = "") -> Tuple[str, str]:
    return (str(provider or "").strip().lower(), normalize_base_url(base_url))


# ── durable state ───────────────────────────────────────────────────────────

_backends: Dict[str, Dict[str, Any]] = {}
_lock = threading.Lock()
_loaded_from: Optional[str] = None

# Backends currently in a half-open probe (memory-only, not persisted): while a
# probe is in flight we must not pass the gate again, or a burst of queued turns
# would all fire probes at once. Cleared on the probe's outcome (record_failure
# re-arms cooldown / record_success recovers).
_probe_inflight: set = set()


def _state_path() -> str:
    from hermes_constants import get_hermes_home

    return str(get_hermes_home() / "state" / "fallback_circuit_breaker.json")


def _key_to_str(key: Tuple[str, str]) -> str:
    provider, base_url = key
    return f"{provider}|{base_url}"


def _load_state_locked() -> None:
    """Populate ``_backends`` from disk. Caller holds ``_lock``."""
    global _loaded_from
    path = _state_path()
    if _loaded_from == path and _backends:
        return
    _backends.clear()
    _loaded_from = path
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        raw_backends = data.get("backends") if isinstance(data, dict) else None
        if not isinstance(raw_backends, dict):
            return
        for key, entry in raw_backends.items():
            if not isinstance(entry, dict):
                continue
            failures = entry.get("failures")
            if not isinstance(failures, list):
                failures = []
            _backends[str(key)] = {
                "provider": str(entry.get("provider") or ""),
                "base_url": str(entry.get("base_url") or ""),
                "failures": [float(t) for t in failures if isinstance(t, (int, float))],
                "quiesced_until": float(entry.get("quiesced_until") or 0.0),
                "total_failures": int(entry.get("total_failures") or 0),
                "total_successes": int(entry.get("total_successes") or 0),
            }
    except Exception as exc:
        logger.warning("circuit breaker: failed to load state from %s: %s", path, exc)
        _backends.clear()


def _save_state_locked() -> None:
    """Persist ``_backends`` atomically. Caller holds ``_lock``."""
    path = _state_path()
    try:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload = {
            "version": STATE_VERSION,
            "updated_at": time.time(),
            "backends": _backends,
        }
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(tmp, path)
    except Exception as exc:
        logger.warning("circuit breaker: failed to save state to %s: %s", path, exc)


# ── core logic ──────────────────────────────────────────────────────────────

def _prune(failures: List[float], now: float, window: float) -> List[float]:
    """Drop timestamps older than the window and cap the list length."""
    cutoff = now - window
    pruned = [t for t in failures if t > cutoff]
    if len(pruned) > MAX_TRACKED_FAILURES:
        pruned = pruned[-MAX_TRACKED_FAILURES:]
    return pruned


def _clock_expiry(entry: Dict[str, Any], now: float) -> None:
    """Lazily clear a quiesce whose cooldown has elapsed."""
    if entry.get("quiesced_until") and now >= float(entry["quiesced_until"]):
        entry["quiesced_until"] = 0.0
        entry["failures"] = []


def record_failure(provider: str, base_url: str = "", *, now: Optional[float] = None) -> bool:
    """Record one failure for a backend.

    Returns True when this failure *tripped* the breaker (the backend is now
    quiesced). Returns False when the breaker is disabled, the provider is empty,
    or the threshold has not been reached.

    Failure timestamps are pruned to the rolling window first, so the count is
    always "failures within the last ``window_seconds``", not a lifetime total.
    """
    settings = get_settings()
    if not settings["enabled"]:
        return False
    key = _backend_key(provider, base_url)
    if not key[0]:
        return False
    now = time.time() if now is None else now
    window = float(settings["window_seconds"])
    threshold = int(settings["fail_threshold"])
    with _lock:
        _load_state_locked()
        skey = _key_to_str(key)
        entry = _backends.get(skey)
        if entry is None:
            entry = {
                "provider": key[0],
                "base_url": key[1],
                "failures": [],
                "quiesced_until": 0.0,
                "total_failures": 0,
                "total_successes": 0,
            }
            _backends[skey] = entry
        _clock_expiry(entry, now)
        entry["total_failures"] = int(entry.get("total_failures") or 0) + 1
        failures = _prune(list(entry.get("failures") or []), now, window)
        failures.append(now)
        failures = _prune(failures, now, window)
        entry["failures"] = failures
        tripped = False
        if len(failures) >= threshold:
            cooldown = float(settings["cooldown_seconds"])
            entry["quiesced_until"] = now + cooldown
            tripped = True
            logger.warning(
                "Provider circuit breaker: %s%s failed %d time(s) within %.0fs -> quiesced for %.0fs",
                key[0], f" [{key[1]}]" if key[1] else "", len(failures), window, cooldown,
            )
        # A probe (failed) has settled — release its in-flight slot so the next
        # half-open window can fire again; the freshly re-armed cooldown above
        # naturally guards how soon that happens.
        _probe_inflight.discard(skey)
        _save_state_locked()
        return tripped


def record_success(provider: str, base_url: str = "") -> None:
    """Clear a backend's rolling window and any quiesce on a success.

    Recovery is automatic: the next time the backend is tried and answers, it is
    fully healthy again.
    """
    key = _backend_key(provider, base_url)
    if not key[0]:
        return
    with _lock:
        _load_state_locked()
        skey = _key_to_str(key)
        entry = _backends.get(skey)
        if entry is None:
            return
        was_quiesced = bool(entry.get("quiesced_until"))
        entry["failures"] = []
        entry["quiesced_until"] = 0.0
        entry["total_successes"] = int(entry.get("total_successes") or 0) + 1
        _probe_inflight.discard(skey)
        if was_quiesced:
            logger.info("Provider circuit breaker: %s%s recovered", key[0], f" [{key[1]}]" if key[1] else "")
        _save_state_locked()


def is_quiesced(provider: str, base_url: str = "", *, now: Optional[float] = None) -> bool:
    """True when the backend is currently quiesced and should be skipped.

    With ``auto_probe`` enabled this becomes a *half-open* gate: once the
    quiesce has lasted at least ``probe_interval_seconds``, a single real
    attempt is let through (the caller will then call ``record_failure`` or
    ``record_success`` to settle the probe). Only one probe per backend may be
    in flight at a time.
    """
    settings = get_settings()
    if not settings["enabled"]:
        return False
    key = _backend_key(provider, base_url)
    if not key[0]:
        return False
    now = time.time() if now is None else now
    skey = _key_to_str(key)
    with _lock:
        _load_state_locked()
        entry = _backends.get(skey)
        if entry is None:
            return False
        _clock_expiry(entry, now)
        quiesced_until = float(entry.get("quiesced_until") or 0.0)
        if now >= quiesced_until:
            # Cooldown elapsed: healthy again. Drop any stale probe marker so a
            # pass-through that never settled (candidate skipped for another
            # reason) cannot leak the backend's probe slot into the next cooldown.
            _probe_inflight.discard(skey)
            return False
        if not settings.get("auto_probe"):
            return True  # still in cooldown, no auto-probe configured
        # Half-open: allow one real probe once the probe interval has passed.
        if now >= quiesced_until - _effective_probe_interval(settings):
            if skey in _probe_inflight:
                # A probe is already running for this backend — hold the gate
                # until its outcome settles, don't stack another.
                return True
            _probe_inflight.add(skey)
            logger.info(
                "Provider circuit breaker: half-open probe for %s%s (cooldown %.0fs left)",
                key[0], f" [{key[1]}]" if key[1] else "", quiesced_until - now,
            )
            return False  # pass this single real attempt through
        return True  # still before the probe window


def remaining_cooldown(provider: str, base_url: str = "", *, now: Optional[float] = None) -> float:
    """Seconds left before a quiesced backend is retried (0.0 when not quiesced)."""
    key = _backend_key(provider, base_url)
    if not key[0]:
        return 0.0
    now = time.time() if now is None else now
    with _lock:
        _load_state_locked()
        entry = _backends.get(_key_to_str(key))
        if entry is None:
            return 0.0
        _clock_expiry(entry, now)
        return max(0.0, float(entry.get("quiesced_until") or 0.0) - now)


def status(*, now: Optional[float] = None) -> str:
    """Human-readable snapshot of every tracked backend.

    ``now`` is injectable for testing; defaults to the wall clock.
    """
    settings = get_settings()
    lines = [
        "Provider circuit breaker:",
        f"  enabled={settings['enabled']}  window={settings['window_seconds']:.0f}s  "
        f"threshold={settings['fail_threshold']}  cooldown={settings['cooldown_seconds']:.0f}s",
    ]
    now = time.time() if now is None else now
    with _lock:
        _load_state_locked()
        if not _backends:
            lines.append("  (no backends tracked)")
            return "\n".join(lines)
        for skey, entry in sorted(_backends.items()):
            _clock_expiry(entry, now)
            in_window = len(_prune(list(entry.get("failures") or []), now, settings["window_seconds"]))
            cd = float(entry.get("quiesced_until") or 0.0)
            state = f"QUIESCED ({cd - now:.0f}s left)" if now < cd else "ok"
            lines.append(
                f"  [{state}] {skey}  fails={in_window}/{settings['fail_threshold']}  "
                f"total={entry.get('total_successes', 0)}ok/{entry.get('total_failures', 0)}fail"
            )
    return "\n".join(lines)


def reset_all_for_tests() -> None:
    """Drop in-memory state so a test can re-read from a fresh ``HERMES_HOME``."""
    global _loaded_from
    with _lock:
        _backends.clear()
        _probe_inflight.clear()
        _loaded_from = None
    _reset_settings_cache_for_tests()

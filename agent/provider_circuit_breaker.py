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
import re
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit

try:  # POSIX advisory lock for the cross-process state merge (vault_store pattern).
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]

try:  # Windows fallback for the same lock.
    import msvcrt
except ImportError:  # pragma: no cover - POSIX
    msvcrt = None  # type: ignore[assignment]

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

_USERINFO_RE = re.compile(r"//[^/@\s]*@")


def normalize_base_url(value: Any) -> str:
    """Canonical identity for an endpoint, with any URL userinfo stripped.

    The identity is persisted (as the JSON key *and* the ``base_url`` field) in
    the breaker state file, so a credential embedded in a configured base_url —
    ``https://user:supersecret@example.com/v1`` — must never survive this
    function, or the supposedly secret-free state file leaks it.
    """
    if not isinstance(value, str):
        return ""
    raw = value.strip()
    if not raw:
        return ""
    try:
        parts = urlsplit(raw)
    except ValueError:  # pragma: no cover - malformed URLs
        return _USERINFO_RE.sub("//", raw).rstrip("/").lower()
    if parts.scheme and parts.netloc:
        try:
            # Rebuild without username/password: hostname + port only. ``.port``
            # raises ValueError for an out-of-range/garbage port, so a malformed
            # base_url must fall through to the string path instead of raising.
            host = parts.hostname or ""
            if parts.port:
                host = f"{host}:{parts.port}"
            return urlunsplit(parts._replace(netloc=host)).rstrip("/").lower()
        except ValueError:  # pragma: no cover - malformed authority
            pass
    # No authority to parse (bare host / path / placeholder): still drop anything
    # shaped like ``//user:pass@`` before canonicalising.
    return _USERINFO_RE.sub("//", raw).rstrip("/").lower()


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
    _backends.update(_read_state_file(path))


@contextmanager
def _state_lock(path: str):
    """Best-effort *cross-process* lock on a sibling ``.lock`` file.

    Several CLI/gateway processes can share one profile; without this, a
    read-modify-replace of the state file loses the other process's counters.
    Mirrors the flock/msvcrt pattern already used by ``agent/vault_store.py``.
    """
    lock_path = f"{path}.lock"
    handle = None
    try:
        directory = os.path.dirname(lock_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        handle = open(lock_path, "a+", encoding="utf-8")
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        elif msvcrt is not None:  # pragma: no cover - Windows
            handle.seek(0)
            handle.write(" ")
            handle.flush()
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
    except Exception as exc:  # pragma: no cover - lock is best-effort
        logger.debug("circuit breaker: state lock unavailable: %s", exc)
    try:
        yield
    finally:
        if handle is not None:
            try:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                elif msvcrt is not None:  # pragma: no cover - Windows
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            except Exception:  # pragma: no cover
                pass
            try:
                handle.close()
            except Exception:  # pragma: no cover
                pass


def _read_state_file(path: str) -> Dict[str, Any]:
    """Parse the state file into ``{skey: entry}``. Never raises."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        raw_backends = data.get("backends") if isinstance(data, dict) else None
        if not isinstance(raw_backends, dict):
            return {}
        out: Dict[str, Any] = {}
        for key, entry in raw_backends.items():
            if not isinstance(entry, dict):
                continue
            failures = entry.get("failures")
            if not isinstance(failures, list):
                failures = []
            out[str(key)] = {
                "provider": str(entry.get("provider") or ""),
                "base_url": str(entry.get("base_url") or ""),
                "failures": [float(t) for t in failures if isinstance(t, (int, float))],
                "quiesced_until": float(entry.get("quiesced_until") or 0.0),
                "reset_at": float(entry.get("reset_at") or 0.0),
                "total_failures": int(entry.get("total_failures") or 0),
                "total_successes": int(entry.get("total_successes") or 0),
            }
        return out
    except Exception as exc:
        logger.warning("circuit breaker: failed to load state from %s: %s", path, exc)
        return {}


def _merge_entry(old: Optional[Dict[str, Any]], new: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two views of one backend without resurrecting a cleared window.

    ``record_success`` (and a lazily expired cooldown) *clears* the window and
    stamps ``reset_at``; failures at or before that stamp are dropped, so a
    deliberate clear wins over a stale on-disk copy, while genuinely newer
    failures written by another process are still unioned in.
    """
    if not old:
        return dict(new)
    if not new:
        return dict(old)
    old_reset = float(old.get("reset_at") or 0.0)
    new_reset = float(new.get("reset_at") or 0.0)
    base, other = (new, old) if new_reset >= old_reset else (old, new)
    reset_at = max(old_reset, new_reset)
    failures = {
        float(t)
        for t in (list(base.get("failures") or []) + list(other.get("failures") or []))
        if float(t) > reset_at
    }
    # The side with the later reset owns the quiesce; equal stamps (the common
    # case) take the longer cooldown so neither process loses a trip.
    quiesced = float(base.get("quiesced_until") or 0.0)
    if old_reset == new_reset:
        quiesced = max(quiesced, float(other.get("quiesced_until") or 0.0))
    return {
        "provider": base.get("provider") or other.get("provider") or "",
        "base_url": base.get("base_url") or other.get("base_url") or "",
        "failures": sorted(failures),
        "quiesced_until": quiesced,
        "reset_at": reset_at,
        "total_failures": max(int(base.get("total_failures") or 0), int(other.get("total_failures") or 0)),
        "total_successes": max(int(base.get("total_successes") or 0), int(other.get("total_successes") or 0)),
    }


def _save_state_locked() -> None:
    """Persist ``_backends`` under a cross-process lock, merging foreign counters.

    Caller holds ``_lock`` (in-process); the sibling lock file serialises the
    read-merge-write against other processes sharing the profile, and
    ``utils.atomic_json_write`` (mkstemp + fsync + replace) keeps the write
    atomic without racing on a shared temp path.
    """
    path = _state_path()
    try:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with _state_lock(path):
            merged = _read_state_file(path)
            for key, entry in list(_backends.items()):
                merged[key] = _merge_entry(merged.get(key), entry)
            _backends.clear()
            _backends.update(merged)
            payload = {
                "version": STATE_VERSION,
                "updated_at": time.time(),
                "backends": merged,
            }
            from utils import atomic_json_write

            atomic_json_write(path, payload, mode=0o600)
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
        # Stamp the clear so a cross-process merge cannot resurrect the window.
        entry["reset_at"] = float(now)


# Only *backend-health* failures count against a provider endpoint. Model-,
# request- and policy-specific failures are excluded on purpose: the breaker
# identity is ``(provider, base_url)`` and does not include the model, so
# counting (for example) a ``context_overflow`` would let five bad requests for
# one model quiesce every model on that endpoint when it later appears as a
# fallback.
BACKEND_FAILURE_REASONS = frozenset({
    "rate_limit",
    "upstream_rate_limit",
    "billing",
    "overloaded",
    "server_error",
    "timeout",
    "ssl_cert_verification",
})


def _reason_name(reason: Any) -> str:
    if reason is None:
        return ""
    value = getattr(reason, "value", None)
    return str(value if value is not None else reason).strip().lower()


def record_failure(provider: str, base_url: str = "", *, reason: Any = None,
                   now: Optional[float] = None) -> bool:
    """Record one failure for a backend.

    ``reason`` (a ``FailoverReason`` or its string value) keeps model/request/
    policy failures out of the rolling window; pass ``None`` to count
    unconditionally.

    Returns True when this failure *tripped* the breaker (the backend is now
    quiesced). Returns False when the breaker is disabled, the provider is empty,
    the failure is not a backend-health class, or the threshold has not been
    reached.

    Failure timestamps are pruned to the rolling window first, so the count is
    always "failures within the last ``window_seconds``", not a lifetime total.
    """
    settings = get_settings()
    if not settings["enabled"]:
        return False
    key = _backend_key(provider, base_url)
    if not key[0]:
        return False
    if reason is not None and _reason_name(reason) not in BACKEND_FAILURE_REASONS:
        logger.debug(
            "circuit breaker: not counting %s for %s (not a backend-health failure)",
            _reason_name(reason), key[0],
        )
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
                "reset_at": 0.0,
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
        # Stamp the clear: a cross-process merge drops failures at/before this
        # instant instead of resurrecting the window we just cleared.
        entry["reset_at"] = time.time()
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

"""Opt-in lenient (rolling-window) cooldown policy for credential-pool keys.

Upstream ``credential_pool`` cools a key on a *single* failure: one 429/401/402
immediately benches that key for a fixed TTL. That is the "strict" behaviour and
stays the DEFAULT — this module does nothing unless the user opts in.

``provider_cooldown.mode: lenient`` layers a rolling window on top:

* a single failure only *parks* the key briefly (``park_seconds``) so the caller
  rotates away and retries soon, instead of a long bench;
* the key is only really benched once ``fail_threshold`` failures land inside
  ``window_seconds``;
* the bench length grows per consecutive trigger along a configurable backoff
  ladder (billing default: x3, x1.5, x1.25, ... saturating at x1.01, no cap);
* the bench IS the blackout: while it lasts the key is never retried;
* after the bench elapses the key is still held back for ``probe_requests``
  selections, then ONE probe is allowed through. The probe must SUCCEED to
  clear the state; a failed probe advances the ladder (a longer bench). This
  bounds how long a recharged key stays dark without hammering a dead one.

State lives in the pool entry's persisted ``extra`` dict (no new files), under a
namespaced key so it never collides with upstream fields:

    extra["provider_cooldown"] = {
        "failures": [<epoch>, ...],   # pruned to the window
        "step": <int>,                # consecutive-trigger index (ladder)
        "blackout_until": <epoch>,    # bench end (grows via the ladder)
        "probe_seen": <int>,          # selections counted since the blackout
        "probe_requests": <int>,      # selections needed before a probe
    }

Because the config layer may be unavailable (tests, partial installs), every
read falls back to the safe strict defaults and never raises.
"""

from __future__ import annotations

import copy
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Namespaced key inside a PooledCredential.extra dict.
STATE_KEY = "provider_cooldown"

MODE_STRICT = "strict"
MODE_LENIENT = "lenient"

# Failure classes we apply the rolling window to. Everything else keeps the
# regular upstream TTL (auth/401 and unknown statuses are deliberately excluded —
# a persistent auth failure should not be re-probed in a tight loop).
CLASS_RATE_LIMIT = "rate_limit"
CLASS_BILLING = "billing"

_DEFAULT_SETTINGS: Dict[str, Any] = {
    "mode": MODE_STRICT,
    "rate_limit": {
        "window_seconds": 1800.0,
        "fail_threshold": 5,
        "park_seconds": 30.0,
        "base_cooldown_seconds": 300.0,
        "backoff_multipliers": [1.0],
        # Curve selector: "auto" uses the built-in curve above; "custom" uses
        # whatever the user puts in backoff_multipliers.
        "curve": "auto",
        # Hard ceiling on the (ladder-grown) cooldown; 0 = no cap.
        "max_cooldown_seconds": 0.0,
        # Selections required AFTER the bench elapses before one probe is let
        # through (keeps a recharged key from being stuck behind a long bench).
        "probe_requests": 1,
    },
    "billing": {
        # First 402 immediately enters the special state (no window wait): a
        # worked-out quota won't recover on its own, so benching at once is
        # cheaper than burning the threshold. Only a success (after recharge)
        # clears it.
        "window_seconds": 3600.0,
        "fail_threshold": 1,
        "park_seconds": 30.0,
        "base_cooldown_seconds": 300.0,
        "backoff_multipliers": [3.0, 1.5, 1.25, 1.25, 1.2, 1.15, 1.11, 1.1, 1.01],
        "curve": "auto",
        "max_cooldown_seconds": 0.0,
        # Probe: after the bench elapses the key must see `probe_requests`
        # selections before ONE probe is allowed through. A successful probe
        # clears the state; a failed one advances the ladder (longer bench).
        "probe_requests": 1,
    },
}

# Built-in "auto" curves, used when `curve: auto`. Kept separate from the
# configurable multipliers so switching curve=custom never mutates the presets.
AUTO_CURVES: Dict[str, List[float]] = {
    "rate_limit": [1.0],
    "billing": [3.0, 1.5, 1.25, 1.25, 1.2, 1.15, 1.11, 1.1, 1.01],
}

_settings_cache: Optional[Dict[str, Any]] = None
_settings_lock = threading.Lock()


def _coerce_positive(raw: Any, default: float, floor: float = 1.0) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError):
        return default
    if value != value or value in (float("inf"), float("-inf")) or value <= 0:  # nan/inf guard
        return default
    return max(value, floor)


def _coerce_threshold(raw: Any, default: int) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError, OverflowError):
        return default
    return value if value >= 1 else default


def _coerce_nonneg(raw: Any, default: float) -> float:
    """Like _coerce_positive but 0 is a valid value (used for 'no cap')."""
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError):
        return default
    if value != value or value in (float("inf"), float("-inf")) or value < 0:
        return default
    return value


def _coerce_multipliers(raw: Any, default: List[float]) -> List[float]:
    if not isinstance(raw, (list, tuple)) or not raw:
        return list(default)
    out: List[float] = []
    for item in raw:
        try:
            value = float(item)
        except (TypeError, ValueError, OverflowError):
            continue
        if value != value or value <= 0:  # skip nan/<=0
            continue
        out.append(value)
    return out or list(default)


def _merge_settings(raw: Dict[str, Any]) -> Dict[str, Any]:
    # Deep-copy the class sub-dicts: a shallow dict() shares the nested
    # backoff_multipliers list with the module default, so a caller mutating the
    # returned settings would silently corrupt every later read.
    settings = {
        "mode": MODE_STRICT,
        "rate_limit": copy.deepcopy(_DEFAULT_SETTINGS["rate_limit"]),
        "billing": copy.deepcopy(_DEFAULT_SETTINGS["billing"]),
    }
    mode = raw.get("mode")
    if isinstance(mode, str) and mode.strip().lower() == MODE_LENIENT:
        settings["mode"] = MODE_LENIENT
    for cls in (CLASS_RATE_LIMIT, CLASS_BILLING):
        block = raw.get(cls)
        if not isinstance(block, dict):
            continue
        default = _DEFAULT_SETTINGS[cls]
        curve = block.get("curve")
        curve = curve.strip().lower() if isinstance(curve, str) else "auto"
        if curve == "custom":
            # Honour the user's own multipliers (falling back to the preset if
            # the list is missing/empty/invalid).
            multipliers = _coerce_multipliers(
                block.get("backoff_multipliers"), AUTO_CURVES[cls]
            )
        else:
            # "auto" (or anything unrecognised): use the built-in curve, ignoring
            # any stale multipliers the user may have left behind.
            curve = "auto"
            multipliers = list(AUTO_CURVES[cls])
        settings[cls] = {
            "window_seconds": _coerce_positive(block.get("window_seconds"), default["window_seconds"]),
            "fail_threshold": _coerce_threshold(block.get("fail_threshold"), default["fail_threshold"]),
            "park_seconds": _coerce_positive(block.get("park_seconds"), default["park_seconds"]),
            "base_cooldown_seconds": _coerce_positive(
                block.get("base_cooldown_seconds"), default["base_cooldown_seconds"]
            ),
            "curve": curve,
            "backoff_multipliers": multipliers,
            "max_cooldown_seconds": _coerce_nonneg(
                block.get("max_cooldown_seconds"), default["max_cooldown_seconds"]
            ),
            "probe_requests": _coerce_threshold(
                block.get("probe_requests"), default["probe_requests"]
            ),
        }
    return settings


def get_settings(force_reload: bool = False) -> Dict[str, Any]:
    """Return the effective cooldown policy. Never raises: defaults to strict."""
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
            block = config.get("provider_cooldown")
            if isinstance(block, dict):
                raw = block
        except Exception as exc:  # pragma: no cover - config layer is best-effort
            logger.debug("provider cooldown: config unavailable, using strict defaults: %s", exc)
        _settings_cache = _merge_settings(raw)
        return _settings_cache


def _reset_settings_cache_for_tests() -> None:
    global _settings_cache
    with _settings_lock:
        _settings_cache = None


def is_lenient() -> bool:
    """True when the opt-in lenient rolling-window policy is active."""
    return get_settings().get("mode") == MODE_LENIENT


def classify(status_code: Optional[int], failure_reason: Optional[str]) -> Optional[str]:
    """Map a failure to a rolling-window class, or None to keep upstream TTL.

    Only 429 (rate limit) and 402/billing are rolled; auth/401/unknown stay on
    the upstream path so a genuinely dead credential is not re-probed.
    """
    reason = (failure_reason or "").strip().lower()
    if status_code == 402 or reason in {"billing", "billing_unverified"}:
        return CLASS_BILLING
    if status_code == 429 or reason in {"rate_limit", "upstream_rate_limit"}:
        return CLASS_RATE_LIMIT
    return None


def _read_state(extra: Dict[str, Any]) -> Dict[str, Any]:
    state = extra.get(STATE_KEY)
    if not isinstance(state, dict):
        state = {}
    failures = [t for t in state.get("failures", []) if isinstance(t, (int, float))]
    return {
        "failures": failures,
        "step": int(state.get("step") or 0),
        "blackout_until": float(state.get("blackout_until") or 0.0),
        "probe_seen": int(state.get("probe_seen") or 0),
        "probe_requests": int(state.get("probe_requests") or 1),
    }


def _write_state(extra: Dict[str, Any], state: Dict[str, Any]) -> None:
    extra[STATE_KEY] = state


def clear_state(extra: Dict[str, Any]) -> bool:
    """Drop all rolling state for a key. Returns True if anything was cleared.

    Called on a real success so a recovered key starts its ladder over.
    """
    if STATE_KEY in extra:
        extra.pop(STATE_KEY, None)
        return True
    return False


def _ladder_cooldown(cls_settings: Dict[str, Any], step: int) -> float:
    """Bench length for a given ladder step.

    ``step`` is the 1-based consecutive-trigger index. The cooldown is the base
    multiplied by every ladder factor up to (and including) that step, with the
    final multiplier saturating (billing ends at x1.01 => slow growth). A
    positive ``max_cooldown_seconds`` clamps the result (0 = no cap).
    """
    base = float(cls_settings["base_cooldown_seconds"])
    multipliers = list(cls_settings["backoff_multipliers"])
    if step < 1:
        step = 1
    total = base
    for idx in range(step):
        factor = multipliers[idx] if idx < len(multipliers) else multipliers[-1]
        total *= factor
    cap = float(cls_settings.get("max_cooldown_seconds") or 0.0)
    if cap > 0:
        total = min(total, cap)
    return total


def has_rolling_state(extra: Dict[str, Any]) -> bool:
    """True when this credential carries opt-in rolling-window state."""
    return isinstance(extra.get(STATE_KEY), dict)


def probe_after_blackout(extra: Dict[str, Any], *, now: Optional[float] = None) -> bool:
    """Count one post-blackout selection; return True when a probe may run.

    Callers hold the credential back while this returns False. The *time* gate
    (the bench) is enforced upstream via the persisted ``last_error_reset_at``;
    this only adds the "N selections after the bench" hold so a recharged key is
    re-probed within a bounded number of calls without hammering a dead one.
    """
    state = extra.get(STATE_KEY)
    if not isinstance(state, dict):
        return True  # not a rolling credential — nothing to gate
    now = time.time() if now is None else now
    if now < float(state.get("blackout_until") or 0.0):
        return False  # still inside the blackout
    needed = max(1, int(state.get("probe_requests") or 1))
    seen = int(state.get("probe_seen") or 0) + 1
    state["probe_seen"] = seen
    extra[STATE_KEY] = state
    return seen >= needed


def record_lenient_failure(
    extra: Dict[str, Any],
    *,
    status_code: Optional[int],
    failure_reason: Optional[str] = None,
    now: Optional[float] = None,
) -> Optional[Tuple[float, bool]]:
    """Record one failure under the lenient policy.

    Returns ``(ttl_seconds, benched)``:
      * ``benched`` False  -> caller should only PARK the key for ttl seconds
        (rotate away, but retry soon); the window has not been crossed yet.
      * ``benched`` True   -> the window threshold was crossed; bench the key
        for the (grown) ttl.

    Returns ``None`` when the failure is not a rolled class (429/402) — the
    caller must then fall through to the upstream strict TTL.
    """
    cls = classify(status_code, failure_reason)
    if cls is None:
        return None
    settings = get_settings()
    cls_settings = settings[cls]
    now = time.time() if now is None else now
    window = float(cls_settings["window_seconds"])
    threshold = int(cls_settings["fail_threshold"])

    state = _read_state(extra)
    cutoff = now - window
    failures = [t for t in state["failures"] if t > cutoff]
    failures.append(now)

    if len(failures) >= threshold:
        # Ladder advances per CONSECUTIVE BENCH (not per raw failure): failures
        # that only parked never escalated, so the first real bench uses the
        # first multiplier (billing: x3 => 300s base becomes 900s), matching the
        # documented curve. The step persists until a success resets it.
        step = max(1, state["step"] + 1)
        ladder = _ladder_cooldown(cls_settings, step)
        # The bench IS the blackout: for its whole length the key is not
        # retried at all. Once it elapses the key is still held back for
        # ``probe_requests`` selections (see ``probe_after_blackout``), so a
        # recharged key is noticed within a bounded number of calls while a
        # still-dead one is not hammered.
        state["failures"] = failures
        state["step"] = step
        state["blackout_until"] = now + ladder
        state["probe_seen"] = 0
        state["probe_requests"] = int(cls_settings["probe_requests"])
        _write_state(extra, state)
        logger.info(
            "provider cooldown (lenient): %s x%d within %.0fs -> bench %.0fs "
            "(ladder step %d, probe after %d selection(s))",
            cls, len(failures), window, ladder, step,
            int(cls_settings["probe_requests"]),
        )
        return ladder, True

    park = float(cls_settings["park_seconds"])
    state["failures"] = failures
    state["parked_until"] = now + park
    _write_state(extra, state)
    logger.debug(
        "provider cooldown (lenient): %s x%d within %.0fs (< %d) -> park %.0fs",
        cls, len(failures), window, threshold, park,
    )
    return park, False


def record_success(extra: Dict[str, Any]) -> bool:
    """Clear rolling state after a successful call. Returns True if cleared."""
    return clear_state(extra)
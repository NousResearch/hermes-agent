"""Soft usage caps for pooled Anthropic subscription logins.

The pool benches a credential only after the provider rejects it. A soft cap moves a live session
off a subscription login earlier: before each request the plan windows of the credential in use
are read (cached), and once a window is at or above its cap that login is benched until the window
resets and the session continues on the next pooled login that is still under its own caps.

Config (``config.yaml``); no entry for the provider means the feature is off::

    credential_pool_usage_caps:
      anthropic:
        five_hour: 85        # percent of the 5-hour window
        seven_day: 80        # percent of the weekly window
        accounts:            # per pool label, overrides the provider-wide caps
          backup: {seven_day: 50}

The check fails open in every direction. Unknown usage never benches a login. When every login is
over its caps the session stays where it is. And the check itself may not hold up a request: its
locks and usage reads run on a helper thread with a time budget (see ``_guarded``); when the budget
runs out the request goes out on the credential it has, and the check stays off until the helper
returns. Rotation on a provider 429 is untouched by all of this.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
import traceback
from typing import Any, Optional

logger = logging.getLogger(__name__)

# cap key -> label of the account-wide window in the usage snapshot (agent.account_usage)
_WINDOW_LABELS = {"five_hour": "Current session", "seven_day": "Current week"}
_CACHE_TTL_SECONDS = 180.0
_ERROR_TTL_SECONDS = 600.0  # an unreadable usage endpoint is not asked again before this
_WARN_SECONDS = 900.0
_GUARD_BUDGET_SECONDS = 5.0
_GUARD_REOPEN_SECONDS = 600.0  # after a timeout, while the helper has still not returned
_GUARD_SETTLE_SECONDS = 60.0  # after a helper that was only slow
_GUARD_MAX_STUCK = 4  # helpers that never returned; at this count the check stays off

Usage = dict[str, tuple[float, Optional[float]]]  # window -> (used percent, reset epoch)

# Reentrant: a plain Lock would turn any helper that calls another one under it into a deadlock.
_lock = threading.RLock()
_cache: dict[str, tuple[float, Optional[Usage]]] = {}
_warned_at: dict[str, float] = {}
_guard_lock = threading.Lock()  # leaf lock: nothing is called while it is held
_guard_open_until = 0.0
_guard_stuck: list[threading.Thread] = []


def load_caps(provider: str, label: Optional[str] = None) -> dict[str, float]:
    """Caps in percent for one pool entry: the provider's, overridden by ``accounts.<label>``."""
    from agent.credential_pool import _load_config_safe

    section = ((_load_config_safe() or {}).get("credential_pool_usage_caps") or {}).get(provider)
    if not isinstance(section, dict):
        return {}

    def clean(raw: Any) -> dict[str, float]:
        out = {}
        for key, value in (raw.items() if isinstance(raw, dict) else ()):
            if key in _WINDOW_LABELS and isinstance(value, (int, float)) and not isinstance(value, bool):
                out[key] = float(value)
        return out

    caps = clean(section)
    accounts = section.get("accounts")
    if label and isinstance(accounts, dict):
        caps.update(clean(accounts.get(label)))
    return caps


def read_usage(entry: Any) -> Optional[Usage]:
    """Plan windows of an Anthropic OAuth pool entry, cached per entry; None when unknown."""
    key = str(getattr(entry, "id", "") or "")
    now = time.time()
    with _lock:
        hit = _cache.get(key)
    if hit and now - hit[0] < (_CACHE_TTL_SECONDS if hit[1] is not None else _ERROR_TTL_SECONDS):
        return hit[1]
    result: Optional[Usage] = None
    try:
        from agent.account_usage import _fetch_anthropic_account_usage

        token = str(getattr(entry, "runtime_api_key", "") or getattr(entry, "access_token", "") or "").strip()
        snapshot = _fetch_anthropic_account_usage(api_key=token) if token else None
        found: Usage = {}
        for window in getattr(snapshot, "windows", None) or ():
            name = next((k for k, label in _WINDOW_LABELS.items() if label == window.label), None)
            if name and window.used_percent is not None:
                found[name] = (float(window.used_percent), window.reset_at.timestamp() if window.reset_at else None)
        result = found or None
    except Exception:  # health: allow BLE001 -- an unreadable usage endpoint must read as "unknown", whatever failed
        logger.info("usage caps: usage read failed for %s", getattr(entry, "label", None) or key, exc_info=True)
    with _lock:
        _cache[key] = (now, result)
    return result


def over_cap(usage: Optional[Usage], caps: dict[str, float]) -> Optional[tuple[str, float, Optional[float]]]:
    """(window, used percent, reset epoch) of the first window at or above its cap, else None."""
    for window, cap in caps.items():
        if usage and window in usage and usage[window][0] >= cap:
            return window, usage[window][0], usage[window][1]
    return None


def guard_state() -> dict[str, Any]:
    """``off_for``: seconds the check is still switched off; ``stuck``: helpers that never returned."""
    with _guard_lock:
        stuck = sum(1 for thread in _guard_stuck if thread.is_alive())
        off_for = max(0.0, _guard_open_until - time.monotonic())
    return {"off_for": float("inf") if stuck >= _GUARD_MAX_STUCK else off_for, "stuck": stuck}


def _guarded(fn: Any, *args: Any, pool: Any = None) -> Any:
    """``fn(*args)`` on a helper thread, waited for up to the budget; None when it is late.

    A late result is dropped, so nothing the helper works out is applied after the caller has
    moved on. After a timeout no new helper is started until this one returns (plus a settle
    time), or for ``_GUARD_REOPEN_SECONDS`` when it never does."""
    global _guard_open_until
    owned = getattr(getattr(pool, "_lock", None), "_is_owned", None)
    if callable(owned) and owned():
        return fn(*args)  # the caller holds the pool lock; a helper would only wait for it
    with _guard_lock:
        _guard_stuck[:] = [thread for thread in _guard_stuck if thread.is_alive()]
        if time.monotonic() < _guard_open_until or len(_guard_stuck) >= _GUARD_MAX_STUCK:
            return None
    done = threading.Event()
    box: dict[str, Any] = {"result": None, "late": False}
    started = time.monotonic()

    def run() -> None:
        global _guard_open_until
        try:
            box["result"] = fn(*args)
        except Exception:  # health: allow BLE001 -- the check fails open; nothing it raises may reach the turn
            logger.warning("usage caps: check failed (fail open)", exc_info=True)
        finally:
            done.set()
            with _guard_lock:
                late = box["late"]
                if late:
                    _guard_open_until = min(_guard_open_until, time.monotonic() + _GUARD_SETTLE_SECONDS)
            if late:
                logger.warning("usage caps: check returned after %.1fs; back on in %.0fs",
                               time.monotonic() - started, _GUARD_SETTLE_SECONDS)

    from agent.memory_provider import spawn_context_thread

    helper = spawn_context_thread(run, name="usage-caps-check")
    try:
        helper.start()
    except RuntimeError:
        return None  # no thread to be had: skip the check rather than run it unguarded
    if done.wait(_GUARD_BUDGET_SECONDS):
        return box["result"]
    with _guard_lock:
        if done.is_set():
            return box["result"]  # finished while we were deciding
        box["late"] = True
        _guard_stuck.append(helper)
        _guard_open_until = time.monotonic() + _GUARD_REOPEN_SECONDS
    frame = sys._current_frames().get(helper.ident or 0)
    where = "".join(traceback.format_stack(frame)[-6:]).strip() if frame is not None else "unknown"
    logger.error("usage caps: check did not return within %.1fs; soft caps are switched off, requests "
                 "continue. The helper is at:\n%s", time.monotonic() - started, where)
    return None


def _plan(agent: Any) -> Optional[tuple]:
    """(pool, current, hit, cap, target, capped) when the live login is over its cap and another
    one is free, else None. Changes nothing."""
    pool = agent._credential_pool
    model = getattr(agent, "model", None)
    current_id = getattr(agent, "_credential_pool_entry_id", None) or getattr(pool, "_current_id", None)
    current = next((entry for entry in pool.entries() if entry.id == current_id), None)
    if current is None:
        return None
    caps = load_caps("anthropic", getattr(current, "label", None))
    hit = over_cap(read_usage(current), caps) if caps else None
    if hit is None:
        return None
    with pool._lock:
        available, _pending = pool._available_entries(model=model)
    capped = [(current, hit)]
    for candidate in available:
        if candidate.id == current.id:
            continue
        candidate_caps = load_caps("anthropic", getattr(candidate, "label", None))
        candidate_hit = over_cap(read_usage(candidate), candidate_caps) if candidate_caps else None
        if candidate_hit is None:
            return pool, current, hit, caps[hit[0]], candidate, capped
        capped.append((candidate, candidate_hit))
    now = time.time()
    if now - _warned_at.get("all", 0.0) > _WARN_SECONDS:
        _warned_at["all"] = now
        logger.warning("usage caps: every anthropic credential is over its cap; staying on %s",
                       getattr(current, "label", None) or current.id)
    return None


def _bench(pool: Any, entry: Any, hit: tuple[str, float, Optional[float]]) -> None:
    window, used, reset = hit
    now = time.time()
    context = {"reason": "usage_cap", "message": f"{window} usage {used:.0f}% >= cap",
               "reset_at": reset if reset and reset > now else now + 3600}
    with pool._lock:
        live = next((e for e in pool.entries() if e.id == entry.id), None)
        if live is not None:
            pool._mark_exhausted(live, 429, context, failure_reason="usage_cap")


def enforce_usage_caps(agent: Any) -> bool:
    """Move the live session off a login that is over its soft cap. True when it moved. Never
    raises and never waits longer than the guard's budget.

    The decision is made on a helper thread; the swap runs here, on the thread that owns the turn."""
    try:
        pool = getattr(agent, "_credential_pool", None)
        if pool is None or str(getattr(pool, "provider", "") or "").strip().lower() != "anthropic":
            return False
        from agent.credential_pool import _load_config_safe

        if not ((_load_config_safe() or {}).get("credential_pool_usage_caps") or {}).get("anthropic"):
            return False
        plan = _guarded(_plan, agent, pool=pool)
        if not plan:
            return False
        pool, current, hit, cap, target, capped = plan
        if agent._swap_credential(target) is False:
            return False  # the target's route cannot serve this model; nothing was changed
        for entry, entry_hit in capped:
            _bench(pool, entry, entry_hit)
        with pool._lock:
            pool._current_id = target.id
        logger.info("usage caps: %s at %s %.0f%% (cap %.0f%%); session moved to %s",
                    getattr(current, "label", None) or current.id, hit[0], hit[1], cap,
                    getattr(target, "label", None) or target.id)
        if not getattr(agent, "_credential_pool_revert_id", None) \
                and getattr(current, "priority", 0) < getattr(target, "priority", 0):
            # Same per-turn hook a quota 429 arms: go back to the preferred login once its window reopens.
            agent._credential_pool_revert_id = current.id
        return True
    except Exception:  # health: allow BLE001 -- the check fails open; nothing it raises may reach the turn
        logger.warning("usage caps: check failed (fail open)", exc_info=True)
        return False

"""Cross-process ledger of per-model retry windows.

A rate-limit window is a fact about the *model*, not about one session: when Qwen4 says
``retry_after=190``, every lane and chat sharing that account is inside the same window. Without a
shared record each process probes independently (and now, with the mid-turn re-probe, on its own
timer), so N sessions turn one closed window into N stampeding calls against a fair-share limiter.

State lives beside the Nous guard's file under ``$HERMES_HOME/rate_limits/``: one JSON object keyed
by ``provider/model`` holding the wall-clock ``reset_at``. Writes go through ``atomic_write_text``
under an advisory lock; readers fail open (no window) so a broken file never blocks a model.

An entry also carries the *probe verdict*: a 1-token probe says nothing about whether a real
(possibly 100k-token) request will be accepted, because fair-share limiters weigh the request. So
a successful probe restores the primary only tentatively (``mark_probe_restore``); if a real call
is refused shortly after, ``record_model_window`` resumes the streak from before the probe rather
than starting over at 1 — otherwise every bad restore resets the backoff to its base interval and
the lane flaps between the two models all day.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Ceiling mirrors the exponential backoff's own cap in fallback_cooldown (4h).
_MAX_WINDOW_SECONDS = 4 * 3600.0
# How long after a probe-driven restore a real 429 still counts against the pre-probe streak.
PROBE_VERIFY_SECONDS = 180.0


def _state_path() -> str:
    try:
        from hermes_constants import get_hermes_home
        base = str(get_hermes_home())
    except ImportError:
        base = os.path.join(os.path.expanduser("~"), ".hermes")
    return os.path.join(base, "rate_limits", "model_windows.json")


def _key(provider: str, model: str) -> str:
    return f"{(provider or '').strip().lower()}/{(model or '').strip()}"


@contextlib.contextmanager
def _locked():
    """Advisory lock around read-modify-write; a no-op where flock is unavailable."""
    path = _state_path()
    lock_path = path + ".lock"
    handle = None
    try:
        os.makedirs(os.path.dirname(lock_path), exist_ok=True)
        handle = open(lock_path, "a+", encoding="utf-8")
    except OSError as exc:
        logger.debug("Model-window lock unavailable: %s", exc)
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


def _read_windows() -> dict:
    try:
        with open(_state_path(), encoding="utf-8-sig") as handle:
            data = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {key: value for key, value in data.items() if isinstance(value, dict)}


def _prune(windows: dict, now: float) -> dict:
    """Drop expired entries so the file cannot grow without bound.

    Two exceptions outlive their window: one still carrying a failure streak (the streak is what
    widens the backoff after the window that produced it ended), and a pending probe verdict, so a
    real 429 seconds after the restore can still see it.
    """
    kept = {}
    for key, entry in windows.items():
        try:
            reset_at = float(entry.get("reset_at") or 0.0)
            failures = int(entry.get("failures") or 0)
            probe_at = float(entry.get("probe_at") or 0.0)
        except (TypeError, ValueError, AttributeError):
            continue
        if (reset_at > now or failures > 0
                or 0 <= now - probe_at <= PROBE_VERIFY_SECONDS):
            kept[key] = entry
    return kept


def _int_field(entry: dict, name: str) -> int:
    try:
        return max(0, int(entry.get(name) or 0))
    except (TypeError, ValueError):
        return 0


def _float_field(entry: dict, name: str) -> float:
    try:
        return float(entry.get(name) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def record_model_window(provider: str, model: str, seconds: float, *, probe_failure: bool = False) -> None:
    """Record that ``provider/model`` is unavailable for ``seconds`` from now.

    ``probe_failure`` marks a re-probe that came back 429 (as opposed to a normal call): the
    entry then carries a consecutive-failure count that callers use to back off exponentially,
    so a model throttled for hours is not probed on the base interval forever.

    A failure that lands within ``PROBE_VERIFY_SECONDS`` of a probe-driven restore is charged
    against the streak the probe claimed to disprove: the probe was not predictive, and letting
    that reset the count is what pinned the re-probe to its base interval while a model stayed
    throttled.
    """
    key = _key(provider, model)
    if not key.strip("/") or not seconds or seconds <= 0:
        return
    seconds = min(float(seconds), _MAX_WINDOW_SECONDS)
    now = time.time()
    with _locked():
        windows = _read_windows()
        existing = windows.get(key) or {}
        # Keep the later of the two: a shorter window must not shorten a still-valid one.
        previous = _float_field(existing, "reset_at")
        reset_at = max(previous, now + seconds)
        windows = _prune(windows, now)
        hits = _int_field(existing, "hits") + (1 if probe_failure else 0)
        failures = _int_field(existing, "failures") + (1 if probe_failure else 0)
        probe_at = _float_field(existing, "probe_at")
        probe_streak = _int_field(existing, "probe_streak")
        false_positive = bool(probe_at) and 0 <= now - probe_at <= PROBE_VERIFY_SECONDS
        if false_positive:
            failures = max(failures, probe_streak) + 1
        windows[key] = {"reset_at": reset_at, "recorded_at": now,
                        "seconds": reset_at - now, "hits": hits, "failures": failures}
        try:
            from utils import atomic_write_text

            atomic_write_text(_state_path(), json.dumps(windows))
        except Exception as exc:  # never let bookkeeping break a turn
            logger.debug("Failed to write model-window ledger: %s", exc)
            return
    if false_positive:
        logger.warning(
            "Primary probe was not predictive for %s: a real call was refused %.0fs after the "
            "probe passed (streak=%d, next probe backs off from there)",
            key, now - probe_at, failures,
        )
    logger.info("Model window recorded: %s unavailable for %.0fs", key, seconds)


def mark_probe_restore(provider: str, model: str, streak: int = 0) -> None:
    """Close ``provider/model``'s window because a cheap probe looked healthy.

    The window really is closed (``reset_at`` is now, so nothing gates on it), but the entry keeps
    the probe timestamp and the streak it was clearing: the next real refusal within
    ``PROBE_VERIFY_SECONDS`` resumes that streak instead of restarting at 1.
    """
    key = _key(provider, model)
    if not key.strip("/"):
        return
    now = time.time()
    with _locked():
        windows = _read_windows()
        existing = windows.get(key) or {}
        windows = _prune(windows, now)
        windows[key] = {"reset_at": now, "recorded_at": now, "seconds": 0.0,
                        "hits": _int_field(existing, "hits"), "failures": 0,
                        "probe_at": now, "probe_streak": max(0, int(streak))}
        try:
            from utils import atomic_write_text

            atomic_write_text(_state_path(), json.dumps(windows))
        except Exception as exc:
            logger.debug("Failed to write model-window ledger: %s", exc)


def model_window_remaining(provider: str, model: str) -> float:
    """Seconds left on the shared window for ``provider/model``; 0.0 when none/expired."""
    key = _key(provider, model)
    if not key.strip("/"):
        return 0.0
    entry = _read_windows().get(key) or {}
    try:
        remaining = float(entry.get("reset_at") or 0.0) - time.time()
    except (TypeError, ValueError):
        return 0.0
    return remaining if remaining > 0 else 0.0


def model_failure_streak(provider: str, model: str) -> int:
    """Consecutive failures recorded for ``provider/model`` (0 when none).

    Survives window expiry on purpose: the streak is what makes the re-probe interval grow
    while a model stays throttled. Only a successful *real* call (``clear_model_window``) resets
    it outright; a probe that merely looked healthy parks it in the entry (see
    ``mark_probe_restore``) so a refusal right after the restore continues from there.
    """
    entry = _read_windows().get(_key(provider, model)) or {}
    return _int_field(entry, "failures")


def model_windows_snapshot() -> dict:
    """Non-expired windows keyed ``provider/model``: ``{remaining, hits, failures, seconds}``.

    Read-only view for ``hermes fallback health``; expired entries are omitted rather
    than pruned so a status call never writes.
    """
    now = time.time()
    snapshot: dict = {}
    for key, entry in _read_windows().items():
        try:
            reset_at = float(entry.get("reset_at") or 0.0)
            hits = int(entry.get("hits") or 0)
            failures = int(entry.get("failures") or 0)
        except (TypeError, ValueError):
            continue
        remaining = reset_at - now
        if remaining > 0:
            snapshot[key] = {"remaining": remaining, "hits": hits, "failures": failures,
                             "seconds": float(entry.get("seconds") or 0.0)}
    return snapshot


def clear_model_window(provider: str, model: str) -> None:
    """Forget the window and every streak/marker (a successful call proves the model is back)."""
    key = _key(provider, model)
    with _locked():
        windows = _read_windows()
        if key not in windows:
            return
        windows.pop(key, None)
        try:
            from utils import atomic_write_text

            atomic_write_text(_state_path(), json.dumps(windows))
        except Exception as exc:
            logger.debug("Failed to clear model-window ledger: %s", exc)


__all__ = [
    "PROBE_VERIFY_SECONDS",
    "clear_model_window",
    "mark_probe_restore",
    "model_failure_streak",
    "model_window_remaining",
    "model_windows_snapshot",
    "record_model_window",
]

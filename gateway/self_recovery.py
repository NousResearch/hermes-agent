"""Self-recovery guards for the Hermes gateway (no gateway imports: stdlib only).

Three small, pure pieces that keep one failure from becoming a channel flood:

1. :class:`ErrorNoticeThrottle` — identical admission-error replies are posted at
   most 3 times in a 15-minute window; the 4th occurrence posts ONE diagnosis
   naming the error, and further identical occurrences are suppressed for 1 hour.
   A process restart resets the throttle (fresh process, fresh state) by design.
2. :func:`effective_restart_wait_timeout` — a service (SIGUSR1 / ``systemctl
   reload``) restart must never park the gateway in draining for the full
   1800s default while every new message earns a visible error post. The
   service-restart wait is capped at 300s; explicit operator restarts keep the
   configured value. A configured timeout of 0 still means "stop immediately".
3. :class:`PauseNoticeRegistry` — a durable, fail-open record of paused-session
   notices already delivered (``unknown_execution`` only), so an old-epoch
   paused session gets one resume/reset notice per 24h, not one per restart.
   Stored outside ``state.db`` (a small JSON sidecar under HERMES_HOME); any
   read/write failure fails open to posting the notice, never to dropping it.

All three are deliberately dependency-free so they can be unit-tested without
importing any gateway module.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

# Error-notice throttling (brief item 1).
ERROR_NOTICE_WINDOW_SECONDS = 900.0  # 15 minutes
ERROR_NOTICE_POST_LIMIT = 3
ERROR_NOTICE_DIAGNOSIS_SUPPRESS_SECONDS = 3600.0  # 1 hour

# Service-restart wait cap (brief item 2). The configured default is 1800s;
# a reload-triggered restart must not drain for that long.
SERVICE_RESTART_WAIT_CAP_SECONDS = 300.0

# Paused-notice dedup (brief item 3).
PAUSE_NOTICE_SUPPRESS_SECONDS = 86400.0  # 24 hours
PAUSE_NOTICE_SIDECAR_PRUNE_DAYS = 7.0
PAUSE_NOTICE_SIDECAR_MAX_ENTRIES = 1000
PAUSE_NOTICE_SIDECAR_FILENAME = ".pause_notices.json"


def error_diagnosis_text(reason: str, occurrences: int) -> str:
    """One diagnosis post replacing a run of identical error replies.

    Names the error and the count; never names a session, a user, or any
    internal identifier. The action line is honest for every throttled reason:
    the condition, not the notice, is what blocks new turns.
    """
    return (
        f"⚠️ Hermes diagnosis: {occurrences} identical {reason} refusals in "
        "this conversation in the last 15 minutes. Further identical notices "
        "are suppressed for 1 hour to stop the flood. New turns are still "
        "being refused while this condition holds; if it persists, send "
        "/reset to start a fresh conversation, or ask the operator to check "
        "gateway.log."
    )


class ErrorNoticeThrottle:
    """Per-scope throttle for identical admission-error replies.

    ``observe`` returns ``"post"`` (send the error reply as today),
    ``"diagnosis"`` (send :func:`error_diagnosis_text` instead), or
    ``"suppress"`` (send nothing). The second element of the returned tuple
    is the number of occurrences of this key inside the current window.
    """

    def __init__(self, clock=time.time):
        self._clock = clock
        # key -> {"occurrences": [float, ...], "diagnosed_at": float | None}
        self._state: dict = {}

    def _prune(self, now: float) -> None:
        for key in list(self._state):
            entry = self._state[key]
            entry["occurrences"] = [
                t for t in entry["occurrences"]
                if now - t <= ERROR_NOTICE_WINDOW_SECONDS
            ]
            diagnosed_at = entry["diagnosed_at"]
            if not entry["occurrences"] and (
                diagnosed_at is None
                or now - diagnosed_at >= ERROR_NOTICE_DIAGNOSIS_SUPPRESS_SECONDS
            ):
                del self._state[key]

    def observe(self, key, now: float | None = None):
        now = self._clock() if now is None else now
        self._prune(now)
        entry = self._state.setdefault(key, {"occurrences": [], "diagnosed_at": None})
        entry["occurrences"].append(now)
        count = len(entry["occurrences"])
        diagnosed_at = entry["diagnosed_at"]
        if (
            diagnosed_at is not None
            and now - diagnosed_at < ERROR_NOTICE_DIAGNOSIS_SUPPRESS_SECONDS
        ):
            return "suppress", count
        if count <= ERROR_NOTICE_POST_LIMIT:
            return "post", count
        entry["diagnosed_at"] = now
        return "diagnosis", count


def get_error_notice_throttle(holder):
    """Lazily attach (or return) an :class:`ErrorNoticeThrottle` on *holder*.

    The holder is the gateway runner: throttle state is process-local and
    resets on restart, which is the intended semantics. If the holder cannot
    carry attributes, a fresh throttle is returned (throttling degrades to
    per-call, never to an exception in the admission path).
    """
    try:
        throttle = getattr(holder, "_self_recovery_throttle", None)
        if throttle is None:
            throttle = ErrorNoticeThrottle()
            holder._self_recovery_throttle = throttle
        return throttle
    except Exception:
        return ErrorNoticeThrottle()


def effective_restart_wait_timeout(
    configured_seconds: float,
    via_service: bool,
    cap_seconds: float = SERVICE_RESTART_WAIT_CAP_SECONDS,
) -> float:
    """Effective after-turn wait for a pending restart.

    The wait itself was already bounded by the configured timeout (default
    1800s); the defect was the size of that bound for service reloads, which
    left the gateway draining — and refusing every new message visibly — for
    up to half an hour. A service restart caps the wait; anything else keeps
    the configured value. A configured value of 0 keeps its existing meaning:
    stop immediately.
    """
    configured = float(configured_seconds or 0.0)
    if configured <= 0.0:
        return 0.0
    if via_service:
        return min(configured, float(cap_seconds))
    return configured


def pause_sidecar_path(home) -> Path:
    """Sidecar path for delivered pause notices under a Hermes home."""
    return Path(home) / PAUSE_NOTICE_SIDECAR_FILENAME


class PauseNoticeRegistry:
    """Durable, fail-open record of delivered ``unknown_execution`` notices.

    Keyed by ``(session_key, reason)`` where ``session_key`` already includes
    the profile, so two profiles never share a suppression. ``should_notify``
    returns True the first time a key is seen (and records it), False while a
    notice for that key was delivered within
    :data:`PAUSE_NOTICE_SUPPRESS_SECONDS`. Any storage failure fails open
    (returns True) so a notice is never silently dropped by this guard.
    """

    def __init__(self, path, clock=time.time):
        self._path = Path(path)
        self._clock = clock

    def _load(self) -> dict:
        try:
            with open(self._path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save(self, data: dict) -> None:
        # Atomic within the sidecar directory; best effort only.
        directory = self._path.parent
        directory.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=self._path.name + ".", dir=str(directory)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, sort_keys=True)
            os.replace(tmp_name, self._path)
        except Exception:
            try:
                os.unlink(tmp_name)
            except Exception:
                pass
            raise

    def should_notify(self, session_key, reason, now: float | None = None) -> bool:
        now = self._clock() if now is None else now
        try:
            data = self._load()
        except Exception:
            return True
        # Prune entries older than the retention window and enforce the cap.
        cutoff = now - PAUSE_NOTICE_SIDECAR_PRUNE_DAYS * 86400.0
        pruned = {
            key: ts
            for key, ts in data.items()
            if isinstance(ts, (int, float)) and ts >= cutoff
        }
        key = f"{session_key}\x00{reason}"
        last = pruned.get(key)
        if isinstance(last, (int, float)) and now - last < PAUSE_NOTICE_SUPPRESS_SECONDS:
            return False
        pruned[key] = now
        if len(pruned) > PAUSE_NOTICE_SIDECAR_MAX_ENTRIES:
            # Drop the oldest entries first; the newest notice always records.
            for old_key in sorted(pruned, key=lambda k: pruned[k])[
                : len(pruned) - PAUSE_NOTICE_SIDECAR_MAX_ENTRIES
            ]:
                del pruned[old_key]
        try:
            self._save(pruned)
        except Exception:
            pass  # fail open: the notice still goes out
        return True

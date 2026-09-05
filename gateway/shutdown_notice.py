"""Durable cooldown for the home-channel "gateway shutting down" broadcast.

``GatewayRunner._notify_active_sessions_of_shutdown`` deduplicates its sends
through a ``notified`` set. That set is **per-process**, so it only ever
suppresses a duplicate *within a single shutdown*. Every fresh gateway process
starts with an empty set and is free to re-broadcast the identical message.

That is fine when a restart is deliberate and rare. It is not fine when
something outside Hermes is cycling the host: each activation starts a gateway,
the gateway is told to stop seconds later, and the home channel gets another
copy of

    ⚠️ Gateway shutting down — Your current task will be interrupted.

Observed on a WSL host where Windows Modern Standby repeatedly tore down and
re-activated the distro on a ~33 s cycle: **240** home-channel shutdown
notifications, 19 of them inside a single 10-minute window, against **1**
active-session notification. Nothing was wrong with the gateway — it was
started and stopped 240 times, and each process correctly believed it was
announcing its first shutdown.

This module makes that dedup survive process death by recording the last
broadcast time per destination under ``HERMES_HOME``, so the *next* gateway
process can see that the same channel was already told moments ago.

Scope is deliberately narrow — the **home-channel broadcast only**. The
per-active-session pings in the same function stay ungated, matching the
existing ``suppress_notification`` drain flag: those carry the genuinely useful
"your task was cut off, message me to resume" hint, and are empty by
construction on the idle shutdowns that produce this flood.

Fail-open everywhere: an unreadable, malformed, or unwritable state file must
never suppress a notification, and must never raise into the shutdown path.
Setting ``gateway.shutdown_notification_cooldown_seconds: 0`` restores the
original always-send behaviour.
"""
from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterator, Optional

from hermes_constants import get_hermes_home
from utils import atomic_json_write

_log = logging.getLogger(__name__)

_STATE_FILENAME = ".shutdown_notice_state.json"
_LOCK_FILENAME = ".shutdown_notice_state.lock"
_LOCK_TIMEOUT_SECONDS = 5.0
_LOCK_POLL_SECONDS = 0.020

# Entries older than this are dropped on write so the file cannot grow
# unbounded across a long-lived install with many home channels. Generous
# relative to any sane cooldown: an entry only matters while it is younger
# than the configured window.
_MAX_ENTRY_AGE_SECONDS = 7 * 24 * 60 * 60


def notice_state_path(home: Optional[Path] = None) -> Path:
    """Absolute path to the shutdown-notice state file, respecting HERMES_HOME."""
    base = home if home is not None else get_hermes_home()
    return Path(base) / _STATE_FILENAME


def _notice_lock_path(home: Optional[Path] = None) -> Path:
    return notice_state_path(home).with_name(_LOCK_FILENAME)


def destination_key(platform: str, chat_id: Any, thread_id: Any = None) -> str:
    """Return a canonical, one-to-one key for one delivery destination.

    The JSON array is intentionally structured rather than delimiter-joined:
    chat IDs and thread IDs can themselves contain colons.
    """
    thread = str(thread_id) if thread_id else None
    return json.dumps(
        [str(platform), str(chat_id), thread],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _read_state(path: Path) -> dict[str, float]:
    """Return ``{destination_key: last_sent_epoch}``; ``{}`` on any problem."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as e:  # malformed / unreadable / permission
        _log.debug("Ignoring unreadable shutdown-notice state %s: %s", path, e)
        return {}

    if not isinstance(raw, dict):
        return {}
    entries = raw.get("home_notices")
    if not isinstance(entries, dict):
        return {}

    out: dict[str, float] = {}
    for key, value in entries.items():
        if not isinstance(key, str):
            continue
        try:
            out[key] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def _should_send_from_state(
    entries: dict[str, float],
    key: str,
    cooldown_seconds: float,
    now_ts: float,
) -> bool:
    """Evaluate a loaded state snapshot without doing I/O."""
    if cooldown_seconds <= 0:
        return True
    last = entries.get(key)
    if last is None or last > now_ts:
        # A backwards clock step must not silence the channel indefinitely.
        return True
    return (now_ts - last) >= cooldown_seconds


def _fresh_state(entries: dict[str, float], now_ts: float) -> dict[str, float]:
    """Drop stale and wildly-future entries before persisting the state."""
    return {
        key: value
        for key, value in entries.items()
        if abs(value - now_ts) <= _MAX_ENTRY_AGE_SECONDS
    }


def _record_home_notice_unlocked(
    path: Path,
    key: str,
    now_ts: float,
) -> None:
    entries = _read_state(path)
    entries[key] = now_ts
    atomic_json_write(path, {"home_notices": _fresh_state(entries, now_ts)})


def _acquire_lock(handle: BinaryIO, timeout: float = _LOCK_TIMEOUT_SECONDS) -> bool:
    """Acquire the state lock with a bounded wait on POSIX and Windows."""
    deadline = time.monotonic() + timeout
    if os.name == "nt":
        import msvcrt

        try:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
        except OSError:
            return False

        while True:
            try:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                return True
            except OSError:
                if time.monotonic() >= deadline:
                    return False
                time.sleep(_LOCK_POLL_SECONDS)

    import fcntl

    while True:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except (BlockingIOError, OSError):
            if time.monotonic() >= deadline:
                return False
            time.sleep(_LOCK_POLL_SECONDS)


def _release_lock(handle: BinaryIO) -> None:
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except (OSError, AttributeError):
        pass


@dataclass
class HomeNoticeAdmission:
    """A cooldown decision that keeps the cross-process lock until send result."""

    allowed: bool
    key: str
    path: Path
    cooldown_seconds: float
    _lock_handle: Optional[BinaryIO] = None
    _recorded: bool = False

    def record_success(self, *, now: Optional[float] = None) -> None:
        """Record a successful send while the admission lock is still held."""
        if not self.allowed or self._recorded:
            return
        try:
            now_ts = time.time() if now is None else float(now)
            if self._lock_handle is not None:
                _record_home_notice_unlocked(self.path, self.key, now_ts)
            else:
                # Lock acquisition failure is fail-open for delivery. Make a
                # best-effort post-send record if the lock becomes available.
                record_home_notice(self.key, now=now_ts, home=self.path.parent)
            self._recorded = True
        except Exception as e:
            _log.debug("Failed recording shutdown notice for %s: %s", self.key, e)


@contextmanager
def home_notice_admission(
    key: str,
    *,
    cooldown_seconds: float,
    now: Optional[float] = None,
    home: Optional[Path] = None,
) -> Iterator[HomeNoticeAdmission]:
    """Atomically admit one home notice and hold the lock through its send.

    The lock spans the state read, awaited platform send, and successful state
    write. Two gateway processes therefore cannot both pass the cooldown check
    for the same destination. A lock or filesystem failure fails open: the
    caller may send, but bookkeeping never blocks a shutdown notification.
    """
    try:
        cooldown = float(cooldown_seconds or 0)
    except (TypeError, ValueError):
        cooldown = 0.0

    path = notice_state_path(home)
    handle: Optional[BinaryIO] = None
    admission: HomeNoticeAdmission
    try:
        if cooldown <= 0:
            admission = HomeNoticeAdmission(True, key, path, cooldown)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = _notice_lock_path(home).open("a+b")
            acquired = _acquire_lock(handle)
            if not acquired:
                _log.debug("Could not acquire shutdown-notice lock %s", handle.name)
                admission = HomeNoticeAdmission(True, key, path, cooldown)
            else:
                now_ts = time.time() if now is None else float(now)
                allowed = _should_send_from_state(
                    _read_state(path), key, cooldown, now_ts
                )
                admission = HomeNoticeAdmission(
                    allowed, key, path, cooldown, _lock_handle=handle
                )
    except Exception as e:
        _log.debug("Shutdown-notice admission failed for %s: %s", key, e)
        if handle is not None:
            handle.close()
            handle = None
        admission = HomeNoticeAdmission(True, key, path, cooldown)

    try:
        yield admission
    finally:
        if handle is not None:
            _release_lock(handle)
            handle.close()


def should_send_home_notice(
    key: str,
    *,
    cooldown_seconds: float,
    now: Optional[float] = None,
    home: Optional[Path] = None,
) -> bool:
    """Return True when the home-channel broadcast for *key* may be sent.

    This read-only helper remains available for callers that do not need
    cross-process admission. The shutdown send path uses
    :func:`home_notice_admission` instead.
    """
    try:
        cooldown = float(cooldown_seconds or 0)
        if cooldown <= 0:
            return True
        now_ts = time.time() if now is None else float(now)
        return _should_send_from_state(
            _read_state(notice_state_path(home)), key, cooldown, now_ts
        )
    except Exception as e:
        # Never let bookkeeping block a notification.
        _log.debug("shutdown-notice cooldown check failed for %s: %s", key, e)
        return True


def record_home_notice(
    key: str,
    *,
    now: Optional[float] = None,
    home: Optional[Path] = None,
) -> None:
    """Record that the home-channel broadcast for *key* was just sent.

    Best-effort and atomic: a failure to persist means the next process simply
    sends again (the pre-fix behaviour), which is the safe direction.
    """
    handle: Optional[BinaryIO] = None
    try:
        now_ts = time.time() if now is None else float(now)
        path = notice_state_path(home)
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = _notice_lock_path(home).open("a+b")
        if not _acquire_lock(handle):
            _log.debug("Could not acquire shutdown-notice lock for %s", key)
            return
        _record_home_notice_unlocked(path, key, now_ts)
    except Exception as e:
        _log.debug("Failed recording shutdown notice for %s: %s", key, e)
    finally:
        if handle is not None:
            _release_lock(handle)
            handle.close()

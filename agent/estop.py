"""Global emergency stop (ESTOP) — a resumable pause for NEW work only.

``hermes pause`` writes a sentinel file at ``$HERMES_HOME/ESTOP``;
``hermes resume`` removes it. While the sentinel exists:

* the cron scheduler skips dispatching due jobs (``cron/scheduler.py:tick``),
* the embedded kanban dispatcher skips spawning workers
  (``gateway/kanban_watchers.py``),
* new gateway turns get a brief "Hermes is paused" reply instead of an
  agent run (``gateway/run.py:_handle_message``).

In-flight work is NEVER killed — this is pause-new-work, not panic/exit.
The check is one or two ``os.stat`` calls (process home + fleet root when
they differ) so callers may run it every tick; no caching beyond the OS is
performed, so engaging/disengaging takes effect on the very next check.

The sentinel body is optional JSON ``{"reason": ..., "engaged_at": ...}``.
A corrupt or empty file still counts as engaged (fail safe): the pause must
hold even if the file was created by ``touch ~/.hermes/ESTOP``.

Ported from: gastownhall/gastown estop.go (MIT). Related prior art:
#26778 (/panic — kill/exit semantics; deliberately different, ours is
resumable) and #44617 (interrupting in-flight cron; deliberately out of
scope here).
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

SENTINEL_NAME = "ESTOP"

# Per-component "logged already for this engagement" flags so a paused
# dispatch loop logs once per engagement instead of once per tick.
_log_lock = threading.Lock()
_logged_components: set[str] = set()


def _hermes_home() -> Path:
    """Resolve the active HERMES_HOME (profile-aware) at call time."""
    try:
        from hermes_constants import get_hermes_home
        return get_hermes_home()
    except Exception:
        return Path(os.path.expanduser("~/.hermes"))


def _canonical_root() -> Path:
    """Fleet-wide Hermes root, even when this process is a profile gateway.

    Profile gateways launch with HERMES_HOME=~/.hermes/profiles/<name>.
    ``hermes pause`` from an operator seat writes ~/.hermes/ESTOP. If we
    only inspect the profile home, the emergency stop does not bind
    (jarvis-os/t_7b65ff88: fleet-analyst kept dispatching through pause).
    """
    try:
        from hermes_constants import get_default_hermes_root
        return Path(get_default_hermes_root())
    except Exception:
        return Path(os.path.expanduser("~/.hermes"))


def sentinel_path() -> Path:
    """Path of the fleet-wide ESTOP sentinel.

    ESTOP is a process-independent safety boundary.  Profile-local
    ``HERMES_HOME`` must never create a second stop state: a voice/profile
    gateway reads and writes the same root sentinel as the operator CLI.
    """
    return _canonical_root() / SENTINEL_NAME


def is_engaged() -> bool:
    """Cheap check: is the global emergency stop engaged?

    Engaged if the fleet canonical root (~/.hermes/ESTOP) exists. Fail SAFE on
    stat errors so an unreadable sentinel still holds the pause. A profile
    local sentinel is deliberately ignored; it is not an authoritative stop.
    """
    try:
        return sentinel_path().exists()
    except OSError:
        return True


def engage(reason: Optional[str] = None) -> Path:
    """Create the ESTOP sentinel. Idempotent; re-engaging updates the file."""
    path = sentinel_path()
    payload = {
        "engaged_at": datetime.now(timezone.utc).isoformat(),
        "reason": reason or None,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    except OSError:
        # Best effort: an empty/partial sentinel still pauses (fail safe).
        try:
            path.touch(exist_ok=True)
        except OSError:
            pass
    return path


def disengage() -> bool:
    """Remove the fleet-root ESTOP sentinel, if present."""
    try:
        sentinel_path().unlink()
        return True
    except FileNotFoundError:
        return False
    except (OSError, AttributeError):
        return False


def get_state() -> Optional[dict]:
    """Return the uncached fleet-root status envelope, or ``None`` if clear."""
    state = get_authoritative_state()
    return None if not state["engaged"] else state


def get_authoritative_state() -> dict:
    """Read the fleet-root ESTOP once and return an uncached status envelope.

    This is the machine-facing surface for status consumers such as the voice
    lane.  ``state=unknown`` is never presented as clear: ``engaged`` remains
    true on a root stat/read failure so callers can fail closed while showing
    an explicit error.  ``observed_at`` is generated for every read, including
    clear and error results, so a caller cannot mistake a cached snapshot for
    current state.
    """
    observed_at = datetime.now(timezone.utc).isoformat()
    path = sentinel_path()
    base = {
        "source": "fleet-root",
        "observed_at": observed_at,
        "freshness": "fresh",
        "path": str(path),
        "error": None,
    }
    try:
        exists = path.exists()
    except (OSError, AttributeError) as exc:
        return {
            **base,
            "state": "unknown",
            "engaged": True,
            "freshness": "unknown",
            "error": f"unable to read fleet-root ESTOP: {exc}",
        }
    if not exists:
        return {**base, "state": "clear", "engaged": False}

    reason = None
    engaged_at = None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            reason = raw.get("reason") or None
            engaged_at = raw.get("engaged_at") or None
    except (OSError, ValueError, AttributeError) as exc:
        return {
            **base,
            "state": "engaged",
            "engaged": True,
            "freshness": "unknown",
            "error": f"invalid fleet-root ESTOP metadata: {exc}",
        }
    return {
        **base,
        "state": "engaged",
        "engaged": True,
        "reason": reason,
        "engaged_at": engaged_at,
    }


def paused_reply() -> Optional[str]:
    """Short user-facing notice for new gateway turns, or None if not paused."""
    state = get_state()
    if state is None:
        return None
    reason = state.get("reason")
    if reason:
        return (
            f"⏸️ Hermes is paused ({reason}). New work is on hold; "
            "run `hermes resume` to pick things back up."
        )
    return (
        "⏸️ Hermes is paused. New work is on hold; "
        "run `hermes resume` to pick things back up."
    )


def check_paused(component: str, logger: logging.Logger) -> bool:
    """Return True when engaged, logging once per engagement per component.

    Dispatch loops call this every tick; the log fires on the disengaged→
    engaged transition for that component and re-arms after a resume, so a
    long pause doesn't spam one line per tick.
    """
    if not is_engaged():
        with _log_lock:
            _logged_components.discard(component)
        return False
    with _log_lock:
        first = component not in _logged_components
        if first:
            _logged_components.add(component)
    if first:
        state = get_state() or {}
        reason = state.get("reason")
        suffix = f" (reason: {reason})" if reason else ""
        logger.info(
            "%s dispatch paused by global emergency stop%s — remove with "
            "`hermes resume` (%s)",
            component,
            suffix,
            sentinel_path(),
        )
    return True


def _reset_log_state_for_tests() -> None:
    """Clear the log-once bookkeeping (test isolation helper)."""
    with _log_lock:
        _logged_components.clear()

"""Cross-process mutual exclusion for in-flight Hermes updates.

The marker file the Tauri updater writes (``UpdateMarkerGuard`` in
``apps/bootstrap-installer/src-tauri/src/update.rs``) and the Electron desktop reads
(``electron/update-marker.ts``) is the single lock for **all** update entrypoints.
Format and location are byte-compatible with both readers.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Keep in sync with UPDATE_MARKER_MAX_AGE_MS in apps/desktop/electron/update-marker.ts:
# a shorter ceiling here would let Python steal a lock Electron still considers live.
# A full update (git pull + uv sync + desktop rebuild) is minutes.
UPDATE_MARKER_MAX_AGE_SECONDS = 20 * 60

MARKER_NAME = ".hermes-update-in-progress"

# Set by an orchestrating updater (Tauri `hermes-setup --update`) to its own pid before
# spawning `hermes update` as a child stage; the parent holds the marker for its whole run,
# so without this the child would refuse its own parent's lock. Keep in sync with
# update_child_env in apps/bootstrap-installer/src-tauri/src/update.rs.
HANDOFF_PID_ENV = "HERMES_UPDATE_HANDOFF_PID"

# Exit code meaning "another updater/instance owns this install right now" — the same
# contract as the Windows shim / venv-holder guards in _cmd_update_impl, matched by the
# Tauri updater (UPDATE_EXIT_CONCURRENT in update.rs) to show "Hermes is still running".
UPDATE_EXIT_CONCURRENT = 2


def update_marker_path() -> Path:
    """Path of the shared update marker.

    Uses the *process* Hermes home (never the context-local profile override): the Rust
    updater resolves ``$HERMES_HOME`` or the platform default and the desktop pins that same
    value into the updater's env, so a profile-scoped path would be one the other owners never look at.
    """
    from hermes_constants import get_process_hermes_home
    return get_process_hermes_home() / MARKER_NAME


def _pid_alive(pid: int) -> bool:
    """Use the dependency-free, Windows-safe probe before PM is available."""
    if pid <= 0:
        return False
    try:
        from hermes_cli._early_recovery import _pid_is_running
        return _pid_is_running(pid)
    except Exception as exc:
        logger.debug("Could not probe pid %s: %s", pid, exc)
        return False


def _handoff_pid() -> int | None:
    """Pid of the orchestrating updater that spawned us (:data:`HANDOFF_PID_ENV`); malformed
    values count as absent so a broken handoff falls back to the normal refusal."""
    try:
        pid = int(os.environ.get(HANDOFF_PID_ENV, "").strip())
    except ValueError:
        return None
    return pid if pid > 0 else None


def _stdlib_parent_pid(pid: int) -> int | None:
    """The parent of ``pid`` without psutil, or ``None`` when unresolvable.

    The update-takeover child is spawned ``-I -S -B`` (hermes_cli/_old_updater.py) so
    psutil cannot import there — and that grandchild is exactly the process that most
    needs the two-hop ancestry walk to adopt the orchestrator's marker. /proc serves
    Linux; macOS keeps /proc absent, so shell out to ps once per hop.
    """
    try:
        if os.path.isdir("/proc"):
            with open(f"/proc/{pid}/stat", "rb") as fh:
                stat = fh.read()
        else:
            out = subprocess.run(
                ["ps", "-o", "ppid=", "-p", str(pid)],
                capture_output=True, text=True, check=True, timeout=5,
            ).stdout
            value = int(out.strip() or -1)
            return value if value > 0 else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    # Field 4 (1-indexed) is ppid, but comm may contain spaces/parens: split
    # after the closing paren of comm instead of on whitespace.
    try:
        return int(stat[stat.rindex(b")") + 2:].split()[1])
    except (ValueError, IndexError):
        return None


def _is_ancestor_pid(pid: int) -> bool:
    """True when ``pid`` is a live ancestor of this process.

    The orchestrating updater spawns ``hermes update`` as a (grand)child, so a live marker
    owned by an ancestor can only be the claim we already run under — an unrelated concurrent
    updater is never in our parent chain. Never our own pid; any failure is "not an ancestor".
    """
    if pid <= 0:
        return False
    if pid == os.getppid():
        return True
    try:
        import psutil
        return any(parent.pid == pid for parent in psutil.Process().parents())
    except ImportError:
        # -I -S -B takeover child: walk the same chain with stdlib probes.
        child = os.getpid()
        for _ in range(32):
            parent = _stdlib_parent_pid(child)
            if parent is None:
                return False
            if parent == pid:
                return True
            if parent == child:  # pid 1 re-parenting or a kernel loop guard
                return False
            child = parent
        return False
    except Exception as exc:
        logger.debug("Could not walk process ancestry for pid %s: %s", pid, exc)
        return False


@dataclass(frozen=True)
class UpdateHolder:
    """A confirmed-live update currently holding the lock."""

    pid: int
    age_seconds: float


def read_live_update(*, path: Path | None = None) -> UpdateHolder | None:
    """Return the live update holding the lock, or ``None``.

    Mirrors ``readLiveUpdateMarker`` in ``electron/update-marker.ts``: absent, unreadable,
    malformed, dead-pid, and past-the-ceiling all mean "no live update", and a stale marker
    file is deleted so it can't strand future runs. Never raises.
    """
    marker = path or update_marker_path()
    try:
        lines = marker.read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return None
    try:
        pid = int(lines[0].strip())
    except (IndexError, ValueError):
        pid = -1
    try:
        started_at = float(lines[1].strip())
    except (IndexError, ValueError):
        started_at = float("-inf")

    age = time.time() - started_at
    if not _pid_alive(pid) or age > UPDATE_MARKER_MAX_AGE_SECONDS:
        with suppress(OSError):
            marker.unlink()
        return None
    return UpdateHolder(pid=pid, age_seconds=age)


def describe_holder(holder: UpdateHolder | None) -> str:
    minutes, seconds = divmod(int(max(0 if holder is None else holder.age_seconds, 0)), 60)
    elapsed = f"{minutes}m {seconds}s" if minutes else f"{seconds}s"
    return (
        f"✗ Another Hermes update is already running {f'(PID {holder.pid})' if holder else ''}, "
        f"started {elapsed} ago).\n"
        "\n"
        "  Two updates running at once can corrupt Hermes. Wait for the update to finish, then try again."
    )


class UpdateLock:
    """Context manager owning the shared update marker for this process.

    ``acquired`` is False when another live update holds it; callers decide between hard
    refusal (CLI/dashboard) and waiting. Release only removes the marker when *we* still own
    it, so a marker rewritten by a handoff partner (the Tauri updater writes its own pid) is
    never deleted from under its new owner.
    """

    def __init__(self, *, path: Path | None = None) -> None:
        self.path = path or update_marker_path()
        self.acquired = False
        self.holder: UpdateHolder | None = None

    def acquire(self) -> bool:
        """Claim the lock. Returns False (and sets ``holder``) if it's taken.

        A live holder whose pid matches :data:`HANDOFF_PID_ENV` — or is an ancestor of ours —
        is our own orchestrating parent: run under ITS claim and leave its marker untouched on
        release. The ancestry path covers staged updaters older than the env-var export.
        """
        existing = read_live_update(path=self.path)
        if existing is not None:
            if existing.pid == _handoff_pid() or _is_ancestor_pid(existing.pid):
                return True
            self.holder = existing
            return False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(f"{os.getpid()}\n{int(time.time())}\n", encoding="utf-8")
        except OSError as exc:
            # Best-effort, like the Rust guard: an unwritable marker must not block the
            # update itself (worse than the race it prevents). Degrade to pre-lock behavior.
            logger.debug("Could not write update marker %s: %s", self.path, exc)
            return True
        self.acquired = True
        return True

    def release(self) -> None:
        """Drop the marker if this process still owns it. Never raises."""
        if not self.acquired:
            return
        self.acquired = False
        try:
            owner = int(self.path.read_text(encoding="utf-8-sig").splitlines()[0].strip())
        except (OSError, IndexError, ValueError):
            return
        if owner != os.getpid():
            return  # a handoff partner took ownership — still a live update
        with suppress(OSError):
            self.path.unlink()

    def __enter__(self) -> "UpdateLock":
        self.acquire()
        return self

    def __exit__(self, *_exc) -> None:
        self.release()

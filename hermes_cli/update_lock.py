"""Cross-process mutual exclusion for in-flight Hermes updates.

Three different surfaces can start an update of the same install tree:

* ``hermes update`` from a terminal,
* the dashboard's Update button (``POST /api/hermes/update`` →
  ``_spawn_hermes_action(["update"])``, detached),
* the desktop's Update button, which hands off to the Tauri
  ``hermes-setup --update`` and, on its failure screen, to install-mode
  bootstrap (``install.ps1`` / ``install.sh``).

Until now only the Tauri updater published an "update in progress" marker
(``UpdateMarkerGuard`` in ``apps/bootstrap-installer/src-tauri/src/update.rs``),
and only the Electron desktop consumed it (``electron/update-marker.ts``, to
gate local backend startup). Nothing stopped two *updaters* from running at
once — so a dashboard-spawned ``hermes update`` and an installer-driven
``git checkout`` could mutate the same checkout concurrently, rewriting source
under a live interpreter and leaving the tree half-updated.

This module makes that same marker the single lock for **all** update
entrypoints instead of adding a fourth mechanism. Format and location are
unchanged and remain byte-compatible with the Rust and Electron readers:

    <HERMES_HOME>/.hermes-update-in-progress   body: "<pid>\\n<started_at_unix>"

A marker only counts as a live update when its pid is alive AND it is younger
than :data:`UPDATE_MARKER_MAX_AGE_MS` — mirroring ``readLiveUpdateMarker`` so a
crashed updater self-heals instead of wedging every future update. A stale
marker is removed on read by whoever notices it first.

One layering wrinkle: the Tauri updater holds this marker for its WHOLE run and
then spawns ``hermes update`` as a child stage. Without a handoff the child
sees its own parent's live marker and refuses — the GUI update deadlocks
against itself on every attempt ("Hermes is still running", retry forever).
Two mechanisms recognize the orchestrating parent, and either suffices:

* The updater exports :data:`HANDOFF_PID_ENV` naming its own pid, and
  ``acquire`` treats a live holder matching that pid as the lock we are
  already running under. The env var alone grants nothing: the pid must also
  be the live marker owner, so a stale or forged value cannot bypass the lock.
* A live holder that is a *process ancestor* of ours is likewise our own
  orchestrator. This is the load-bearing path for the fleet: the staged
  ``hermes-setup`` binary under ``~/.hermes`` is only refreshed by a full
  installer run (``copy_self_to_hermes_home`` deliberately no-ops during
  ``--update``), so every desktop whose staged updater predates the
  HANDOFF_PID_ENV export runs an old parent against a new child. Without the
  ancestry check those users get exit 2 ("Hermes is still running") on every
  GUI update forever, with no Hermes process actually running.

The Windows console-shim re-exec is different: its parent must exit so the
child can replace ``hermes.exe``. That parent transfers its marker to the
spawned venv-Python child before returning the prompt. The child receives
:data:`UPDATE_LOCK_TAKEOVER_PID_ENV`, adopts the transferred claim, and owns
the marker until the update tail finishes. A second terminal update therefore
cannot enter during the handoff gap.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Keep in sync with UPDATE_MARKER_MAX_AGE_MS in
# apps/desktop/electron/update-marker.ts — the same marker is read by both, and
# a shorter ceiling here would let Python steal a lock Electron still considers
# live. A full update (git pull + uv sync + desktop rebuild) is minutes.
UPDATE_MARKER_MAX_AGE_SECONDS = 20 * 60

MARKER_NAME = ".hermes-update-in-progress"

# Set by an orchestrating updater (the Tauri `hermes-setup --update` flow) to
# its own pid before spawning `hermes update` as a child stage. The parent
# holds the marker for its whole run, so without this the child refuses its
# own parent's lock and the GUI update can never complete. See update_child_env
# in apps/bootstrap-installer/src-tauri/src/update.rs — keep the name in sync.
HANDOFF_PID_ENV = "HERMES_UPDATE_HANDOFF_PID"

# Set only by the Windows console-shim re-exec parent. Unlike
# HANDOFF_PID_ENV, this parent is about to exit and must transfer ownership to
# its child rather than keep the child running under the parent's claim.
UPDATE_LOCK_TAKEOVER_PID_ENV = "HERMES_UPDATE_LOCK_TAKEOVER_PID"

# Exit code meaning "another updater/instance owns this install right now".
# Already the de-facto contract: the Windows shim + venv-holder guards in
# _cmd_update_impl exit 2, and the Tauri updater matches on it
# (UPDATE_EXIT_CONCURRENT in apps/bootstrap-installer/src-tauri/src/update.rs)
# to show "Hermes is still running" instead of a generic failure. Naming it
# here keeps the concurrent-update refusal on that same understood contract.
UPDATE_EXIT_CONCURRENT = 2


def update_marker_path() -> Path:
    """Path of the shared update marker.

    Uses the *process* Hermes home (never the context-local profile override):
    the Rust updater resolves ``$HERMES_HOME`` or the platform default, and the
    desktop pins that same value into the updater's env. A profile-scoped path
    here would put the lock somewhere the other two owners never look.
    """
    from hermes_constants import get_process_hermes_home

    return get_process_hermes_home() / MARKER_NAME


def _pid_alive(pid: int) -> bool:
    """True when a process with ``pid`` currently exists.

    Delegates to :func:`gateway.status._pid_exists`, the project's existing
    no-kill probe. Do NOT hand-roll this with ``os.kill(pid, 0)``: on Windows
    that is not a no-op — CPython routes ``sig=0`` to
    ``GenerateConsoleCtrlEvent``, which Ctrl+C's the target's whole console
    process group (bpo-14484). A liveness check that killed the updater it was
    asking about would be a spectacular way to fix a concurrency bug.

    Any pid we cannot evaluate counts as dead: a corrupt marker must not wedge
    the lock forever.
    """
    if pid <= 0:
        return False
    try:
        from gateway.status import _pid_exists

        return bool(_pid_exists(pid))
    except Exception as exc:
        # Import failure or an unusable pid (e.g. larger than the platform's
        # pid_t). Treat the marker as stale rather than blocking updates.
        logger.debug("Could not probe pid %s: %s", pid, exc)
        return False


def _handoff_pid() -> int | None:
    """Pid of the orchestrating updater that spawned us, if any.

    Read from :data:`HANDOFF_PID_ENV`. Malformed values count as absent —
    a broken handoff must fall back to the normal refusal, never crash.
    """
    raw = os.environ.get(HANDOFF_PID_ENV, "").strip()
    if not raw:
        return None
    try:
        pid = int(raw)
    except ValueError:
        return None
    return pid if pid > 0 else None


def _takeover_pid() -> int | None:
    """Pid of the console-shim parent whose claim this child must adopt."""
    raw = os.environ.get(UPDATE_LOCK_TAKEOVER_PID_ENV, "").strip()
    if not raw:
        return None
    try:
        pid = int(raw)
    except ValueError:
        return None
    return pid if pid > 0 else None


def _replace_marker_owner(
    expected_pid: int,
    new_pid: int,
    *,
    path: Path,
) -> bool:
    """Replace only ``expected_pid`` with ``new_pid`` in a marker.

    The remaining marker fields are preserved so this stays compatible with
    readers that extend the two-line format. A temporary file plus
    :func:`os.replace` prevents another updater from observing a partial
    handoff marker.
    """
    if expected_pid <= 0 or new_pid <= 0:
        return False
    try:
        raw = path.read_text(encoding="utf-8")
        lines = raw.splitlines()
        owner = int(lines[0].strip())
    except (OSError, IndexError, ValueError):
        return False
    if owner == new_pid:
        return True
    if owner != expected_pid or len(lines) < 2:
        return False

    replacement = path.with_name(
        f"{path.name}.handoff-{os.getpid()}-{new_pid}-{time.time_ns()}"
    )
    try:
        replacement.write_text(
            "\n".join([str(new_pid), *lines[1:]]) + "\n",
            encoding="utf-8",
        )
        os.replace(replacement, path)
        return True
    except OSError as exc:
        logger.debug(
            "Could not transfer update marker %s from pid %s to %s: %s",
            path,
            expected_pid,
            new_pid,
            exc,
        )
        return False
    finally:
        try:
            replacement.unlink()
        except OSError:
            pass


def transfer_update_lock_to(pid: int, *, path: Path | None = None) -> bool:
    """Transfer this process's live update claim to ``pid``.

    Returns True when the marker now names the child or when another
    orchestrating ancestor owns it. The latter needs no transfer because that
    owner remains alive across the console-shim parent's exit.
    """
    marker = path or update_marker_path()
    try:
        raw = marker.read_text(encoding="utf-8")
        owner = int(raw.splitlines()[0].strip())
    except (OSError, IndexError, ValueError):
        # Marker publication is already best-effort. If this invocation has no
        # claim to transfer, preserve the existing behavior.
        return True
    if owner == pid:
        return True
    if owner != os.getpid():
        # Tauri/install-mode parents keep their own marker across both Python
        # processes, so the shim parent must not steal that claim.
        return True
    return _replace_marker_owner(os.getpid(), pid, path=marker)


def _is_ancestor_pid(pid: int) -> bool:
    """True when ``pid`` is a live ancestor (parent chain) of this process.

    The orchestrating updater spawns ``hermes update`` as a (grand)child, so a
    live marker owned by one of our ancestors can only be the claim we are
    already running under — an unrelated concurrent updater is never in our
    parent chain. This heals the fleet of staged ``hermes-setup`` binaries
    that predate the HANDOFF_PID_ENV export and can never send it.

    Never includes our own pid, and any failure counts as "not an ancestor":
    an unprovable ancestry must fall back to the normal refusal.
    """
    if pid <= 0:
        return False
    try:
        import psutil

        return any(parent.pid == pid for parent in psutil.Process().parents())
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

    Mirrors ``readLiveUpdateMarker`` in ``electron/update-marker.ts``: absent,
    unreadable, malformed, dead-pid, and past-the-ceiling all mean "no live
    update", and a stale marker file is deleted so it can't strand future runs.
    Never raises.
    """
    marker = path or update_marker_path()
    try:
        raw = marker.read_text(encoding="utf-8")
    except OSError:
        return None  # absent or unreadable => no live update

    lines = raw.splitlines()
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
        try:
            marker.unlink()
        except OSError:
            pass
        return None

    return UpdateHolder(pid=pid, age_seconds=age)


def describe_holder(holder: UpdateHolder) -> str:
    """One-line, user-facing explanation of who holds the update lock."""
    minutes, seconds = divmod(int(max(holder.age_seconds, 0)), 60)
    elapsed = f"{minutes}m {seconds}s" if minutes else f"{seconds}s"
    return (
        f"✗ Another Hermes update is already running (PID {holder.pid}, "
        f"started {elapsed} ago).\n"
        "\n"
        "  Two updates mutating the same checkout corrupt it: one rewrites\n"
        "  source while the other is mid-install. Wait for it to finish, or\n"
        "  close the window/dashboard tab that started it, then retry."
    )


class UpdateLock:
    """Context manager owning the shared update marker for this process.

    ``acquired`` is False when another live update already holds it — callers
    decide whether that's a hard refusal (CLI/dashboard) or a wait. Releasing
    only removes the marker when *we* still own it, so a marker rewritten by a
    handoff partner (the Tauri updater overwrites it with its own pid) is never
    deleted out from under its new owner.
    """

    def __init__(self, *, path: Path | None = None) -> None:
        self.path = path or update_marker_path()
        self.acquired = False
        self.holder: UpdateHolder | None = None

    def acquire(self) -> bool:
        """Claim the lock. Returns False (and sets ``holder``) if it's taken.

        A live holder whose pid matches :data:`HANDOFF_PID_ENV` — or is a
        process ancestor of ours — is our own orchestrating parent (the Tauri
        updater spawning `hermes update` as a stage): we run under ITS claim
        rather than refusing or re-writing the marker, and ``release`` leaves
        the parent's marker untouched. The ancestry path exists because staged
        updaters older than the HANDOFF_PID_ENV export never send the env var.
        """
        takeover_pid = _takeover_pid()
        os.environ.pop(UPDATE_LOCK_TAKEOVER_PID_ENV, None)
        existing = read_live_update(path=self.path)
        if existing is not None:
            if takeover_pid is not None and existing.pid == os.getpid():
                # The shim parent published our pid before returning. Adopt
                # that exact claim so our release removes it after the update.
                self.acquired = True
                return True
            if (
                takeover_pid is not None
                and existing.pid == takeover_pid
                and _is_ancestor_pid(existing.pid)
                and _replace_marker_owner(existing.pid, os.getpid(), path=self.path)
            ):
                # We reached the lock before the parent published our pid.
                # Taking it over here makes both scheduling orders safe.
                self.acquired = True
                return True
            if existing.pid == _handoff_pid() or _is_ancestor_pid(existing.pid):
                return True
            self.holder = existing
            return False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                f"{os.getpid()}\n{int(time.time())}\n", encoding="utf-8"
            )
        except OSError as exc:
            # Best-effort, exactly like the Rust guard: an unwritable marker
            # must not block the update itself (that would be a worse failure
            # than the race it prevents). Degrade to the pre-lock behavior.
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
            raw = self.path.read_text(encoding="utf-8")
            owner = int(raw.splitlines()[0].strip())
        except (OSError, IndexError, ValueError):
            return
        if owner != os.getpid():
            # A handoff partner took ownership (e.g. the Tauri updater wrote
            # its own pid). Leave it alone — it's still a live update.
            return
        try:
            self.path.unlink()
        except OSError:
            pass

    def __enter__(self) -> "UpdateLock":
        self.acquire()
        return self

    def __exit__(self, *_exc) -> None:
        self.release()

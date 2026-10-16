"""Launch-time stale-bytecode sweep, serialized by a single-winner lock."""
import json
import logging
import os
import socket
import time as _time
from pathlib import Path
logger = logging.getLogger(__name__)

_BYTECODE_FINGERPRINT_FILE = ".bytecode-fingerprint"

#: The single-winner lock lives in its own directory, and that directory ignores
#: itself (a ``.gitignore`` holding ``*``), so the lock never reads as an
#: untracked file to ``git status`` or ``hermes update``'s autostash and the
#: checkout's root ``.gitignore`` needs no entry for it.
_BYTECODE_SWEEP_DIR = ".bytecode-sweep"

_BYTECODE_SWEEP_LOCK_FILE = "lock"

#: Where the lock lived before it had a directory. A crashed sweeper could leave
#: one behind, and the root `.gitignore` no longer names it, so the lock path's
#: one reader deletes it and says so.
_LEGACY_BYTECODE_SWEEP_LOCK_FILE = ".bytecode-sweep.lock"

_BYTECODE_SWEEP_LOCK_WAIT_SECONDS = 20.0

_BYTECODE_SWEEP_LOCK_STALE_SECONDS = 120.0

_SWEEP_OUTCOME_SWEPT = "swept"

_SWEEP_OUTCOME_WAITED = "waited_for_winner"

_SWEEP_OUTCOME_UNSWEPT = "proceeded_unswept"

def _ensure_self_ignoring_dir(directory: Path) -> None:
    """Create *directory* with a ``.gitignore`` of ``*``. Never raises."""

    try:
        directory.mkdir(parents=True, exist_ok=True)
        marker = directory / ".gitignore"
        if not marker.exists():
            marker.write_text("*\n", encoding="utf-8", newline="\n")
    except OSError:
        pass


def _bytecode_sweep_lock_path() -> Path:
    from hermes_cli.main import PROJECT_ROOT
    directory = PROJECT_ROOT / _BYTECODE_SWEEP_DIR
    _ensure_self_ignoring_dir(directory)
    legacy = PROJECT_ROOT / _LEGACY_BYTECODE_SWEEP_LOCK_FILE
    try:
        legacy.unlink()
        logger.info("Removed the pre-2026-09-24 root bytecode-sweep lock: %s", legacy)
    except OSError:
        pass
    return directory / _BYTECODE_SWEEP_LOCK_FILE

def _bytecode_sweep_lock_payload() -> bytes:
    """What a claimed lock records: who holds it, provably, and where."""

    from hermes_cli.process_identity import _process_create_time

    payload = {
        "pid": os.getpid(),
        "create_time": _process_create_time(),
        "host": socket.gethostname(),
    }
    return (json.dumps(payload) + "\n").encode("utf-8")


def _bytecode_sweep_lock_holder_alive(lock_path: Path) -> bool:
    """True only when the lock's holder is on this host and provably running.

    Anything this process cannot verify — an unreadable or bare-pid lock,
    another host, ``psutil`` unable to say — answers False, which leaves the
    caller's age rule in charge.
    """

    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8-sig"))
        pid = int(payload["pid"])
        host = payload["host"]
        create_time = payload.get("create_time")
        if create_time is not None:
            create_time = float(create_time)
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return False
    if host != socket.gethostname():
        return False
    from hermes_cli.process_identity import _pid_alive_matches

    return _pid_alive_matches(pid, create_time) is True


def _break_stale_bytecode_sweep_lock(lock_path: Path) -> bool:
    """Remove a sweep lock whose holder is gone. Returns whether one was removed.

    A lock younger than :data:`_BYTECODE_SWEEP_LOCK_STALE_SECONDS` is never
    broken. Past that age, a lock written on THIS host is broken only when its
    holder is not provably running: a purge of a large checkout can outlast the
    age bound, and breaking a live holder's lock lets the next launch sweep on
    top of it. Age alone decides where liveness cannot be checked — a lock from
    another host (a checkout on a share), a lock in the old bare-pid format, or
    a host where ``psutil`` cannot answer — because there a crashed holder's
    lock would otherwise never be broken and no later launch could sweep that
    checkout again.
    """

    try:
        age = _time.time() - lock_path.stat().st_mtime
    except OSError:
        return False
    if age < _BYTECODE_SWEEP_LOCK_STALE_SECONDS:
        return False
    if _bytecode_sweep_lock_holder_alive(lock_path):
        return False
    try:
        lock_path.unlink()
    except OSError:
        # Somebody else broke it first, or the filesystem refused. Either way
        # this process is not the one that has to care.
        return False
    logger.debug(
        "Broke a stale bytecode-sweep lock (%.0fs old): %s", age, lock_path
    )
    return True

_SWEEP_CLAIM_CLAIMED = "claimed"

_SWEEP_CLAIM_CONTENDED = "contended"

_SWEEP_CLAIM_UNAVAILABLE = "unavailable"

def _claim_bytecode_sweep_lock(lock_path: Path) -> str:
    """Try to claim the right to sweep. See the three ``_SWEEP_CLAIM_*`` answers.

    ``O_EXCL`` and not a write-if-missing: two hermes processes booting against
    one checkout is a real concurrency, and the loser must defer to the winner rather than
    overwrite its claim.
    """

    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        if not _break_stale_bytecode_sweep_lock(lock_path):
            return _SWEEP_CLAIM_CONTENDED
        try:
            fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            # Another process claimed it in the gap after we broke the stale one.
            return _SWEEP_CLAIM_CONTENDED
        except OSError:
            return _SWEEP_CLAIM_UNAVAILABLE
    except OSError:
        return _SWEEP_CLAIM_UNAVAILABLE
    try:
        os.write(fd, _bytecode_sweep_lock_payload())
    except OSError:
        pass
    finally:
        os.close(fd)
    return _SWEEP_CLAIM_CLAIMED

def _release_bytecode_sweep_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink()
    except OSError:
        pass

def _await_bytecode_sweep_winner(lock_path: Path) -> bool:
    """Wait for the winner to release its lock. True if it did, in time.

    False means the wait expired (or the lock went stale under us), and the
    caller proceeds WITHOUT sweeping — see
    :data:`_BYTECODE_SWEEP_LOCK_WAIT_SECONDS` for why fail-open is the right
    direction here.
    """

    deadline = _time.monotonic() + _BYTECODE_SWEEP_LOCK_WAIT_SECONDS
    while _time.monotonic() < deadline:
        if not lock_path.exists():
            return True
        if _break_stale_bytecode_sweep_lock(lock_path):
            return False
        _time.sleep(0.05)
    return not lock_path.exists()

def _record_bytecode_fingerprint() -> None:
    """Persist the current checkout fingerprint after a bytecode sweep.

    Never raises. A failed write just means the next launch re-sweeps —
    safe, merely redundant.
    """
    from hermes_cli.main import PROJECT_ROOT, _read_git_revision_fingerprint
    try:
        fingerprint = _read_git_revision_fingerprint(PROJECT_ROOT)
        if not fingerprint:
            return
        stamp_path = PROJECT_ROOT / _BYTECODE_FINGERPRINT_FILE
        tmp_path = stamp_path.with_name(stamp_path.name + ".tmp")
        tmp_path.write_text(fingerprint, encoding="utf-8")
        tmp_path.replace(stamp_path)
    except OSError as exc:
        logger.debug("Could not record bytecode fingerprint: %s", exc)

def _log_bytecode_sweep_outcome(
    *,
    outcome: str,
    recorded: str,
    fingerprint: str,
    removed: int,
    started: float,
) -> None:
    """The one purge line, naming what this process actually did.

    Without the outcome, a boot where two processes contended was
    indistinguishable from a boot where one process swept twice: both looked
    like "two purge lines". Naming the outcome makes the lock's effect readable
    from the existing log.
    """

    logger.info(
        "Checkout changed since last launch (%s -> %s): cleared %d stale "
        "__pycache__ director%s outcome=%s swept_ms=%d",
        recorded or "unknown",
        fingerprint,
        removed,
        "y" if removed == 1 else "ies",
        outcome,
        int(max(0.0, _time.monotonic() - started) * 1000),
    )

def _sweep_stale_bytecode_if_checkout_changed() -> None:
    """Clear ``__pycache__`` at launch when the checkout changed underneath us.

    The stale-bytecode bug class (issues #6207, #60242; Dhruv's WhatsApp
    ``cannot import name 'parse_model_flags_detailed'`` report) has one
    shared shape: the checkout's ``.py`` files change (git pull inside
    ``hermes update``, a manual ``git pull``, a ZIP update, a file-sync
    restore) while ``__pycache__`` retains bytecode from the previous
    revision, and a later process trusts the stale ``.pyc`` instead of the
    fresh source.

    Update-time clears alone can never close this class: ``hermes update``
    always executes the PRE-pull updater code, so any hardening added to it
    only takes effect one update late, and manual ``git pull`` never runs
    the updater at all. This launch-time guard closes the loop: every
    ``hermes`` entry point compares the checkout fingerprint (cheap file
    reads, no git subprocess) against the last-validated stamp and sweeps
    the bytecode cache once when they diverge.

    Never raises — a failure here must not block launch.

    The sweep reports its own duration on the log line (``swept_ms=``).

    Exactly one process per checkout change does the work. Without the lock, two
    processes that start together (for example a gateway and a CLI) both reach
    this guard, each deletes every ``__pycache__`` directory, and each then
    recompiles the import set the other has just deleted. The loser now waits briefly on the winner's lock and proceeds
    WITHOUT sweeping — see :data:`_BYTECODE_SWEEP_LOCK_WAIT_SECONDS` for why
    fail-open rather than fail-closed. The outcome is named on the log line
    (``outcome=swept`` / ``waited_for_winner`` / ``proceeded_unswept``), because
    "how many purge lines appeared" was never a readable account of a
    multi-child boot.
    """
    from hermes_cli.main import PROJECT_ROOT, _read_git_revision_fingerprint, _clear_bytecode_cache
    _started = _time.monotonic()
    try:
        fingerprint = _read_git_revision_fingerprint(PROJECT_ROOT)
        if not fingerprint:
            return  # non-git install — the ZIP update path clears explicitly
        stamp_path = PROJECT_ROOT / _BYTECODE_FINGERPRINT_FILE
        try:
            recorded = stamp_path.read_text(encoding="utf-8-sig").strip()
        except OSError:
            recorded = ""
        if recorded == fingerprint:
            return
        # The fingerprint check is deliberately OUTSIDE the lock: it is two cheap
        # file reads, and on the overwhelmingly common path (nothing changed) it
        # returns before any process touches the lock at all. Only a genuine
        # divergence contends.
        lock_path = _bytecode_sweep_lock_path()
        claim = _claim_bytecode_sweep_lock(lock_path)
        if claim == _SWEEP_CLAIM_CONTENDED:
            waited = _await_bytecode_sweep_winner(lock_path)
            # Fail-open either way: the winner restamped (so a re-read would
            # return early) or it did not (so this process proceeds on possibly
            # stale bytecode, exactly as every launch did before this guard
            # existed). Not sweeping is the whole point — a second full purge is
            # what the lock exists to prevent.
            _log_bytecode_sweep_outcome(
                outcome=(
                    _SWEEP_OUTCOME_WAITED if waited else _SWEEP_OUTCOME_UNSWEPT
                ),
                recorded=recorded,
                fingerprint=fingerprint,
                removed=0,
                started=_started,
            )
            return
        try:
            # ``claimed`` or ``unavailable``. The second case sweeps too, and
            # that is deliberate — see the ``_SWEEP_CLAIM_*`` constants: a
            # filesystem nobody can lock must keep the guard, not lose it.
            removed = _clear_bytecode_cache(PROJECT_ROOT)
            if removed:
                _log_bytecode_sweep_outcome(
                    outcome=_SWEEP_OUTCOME_SWEPT,
                    recorded=recorded,
                    fingerprint=fingerprint,
                    removed=removed,
                    started=_started,
                )
            _record_bytecode_fingerprint()
        finally:
            # Released only after the restamp, so a loser that wakes up on the
            # released lock re-reads a fingerprint that already matches. Only the
            # process that CLAIMED releases — an ``unavailable`` claim holds
            # nothing, and unlinking on its behalf could take out a lock a
            # concurrent winner does hold.
            if claim == _SWEEP_CLAIM_CLAIMED:
                _release_bytecode_sweep_lock(lock_path)
    except Exception as exc:
        logger.debug("Stale-bytecode launch sweep failed: %s", exc, exc_info=True)
    finally:
        # Recorded even on the early returns and the exception path: "the sweep
        # decided in 4 ms that it had nothing to do" is exactly as much an
        # answer as "the sweep took 9 s", and a key that appears only on the
        # expensive path would make every cheap boot look unmeasured.
        logger.debug("Stale-bytecode launch sweep took %d ms", int(max(0.0, _time.monotonic() - _started) * 1000))

"""Cross-process locks for mutations of a profile's skill library.

Every Hermes-owned writer of ``~/.hermes/skills`` must go through this module.
The locks are advisory, so a writer that skips them is invisible to the ones
that cooperate.  The audited writer set is:

  * ``tools.skill_manager_tool``  — agent create/edit/patch/delete/write_file
  * ``tools.skill_usage``         — archive_skill / restore_skill
  * ``tools.skills_hub``          — install_from_quarantine / uninstall_skill
  * ``tools.skills_sync``         — bundled-skill seeding, reset, restore,
                                    prune of pristine bundled skills
  * ``tools.skills_sync_client``  — personal and org pulls (tree materialize,
                                    org-mirror replace)
  * ``agent.curator_backup``      — snapshot_skills / rollback (the rollback
                                    empties the tree and extracts an archive
                                    over it, so the lock spans the whole
                                    transaction, staging move included)

Deliberately NOT covered, so the claim above stays honest:

  * ``hermes_cli.profiles`` profile clone/delete, which copies or removes an
    entire profile directory.  These locks are profile-scoped, so covering a
    clone means locking the *source* profile's namespace from a process bound
    to a different one — a wider change than this protocol, and a different
    race (clone consistency, not concurrent mutation of one library).
  * Quarantine staging in ``hermes_cli.skills_hub``, which writes outside the
    skills tree until ``install_from_quarantine`` (which does take the lock).

Two lock scopes:

``skills_namespace_lock``
    Shared for "I am about to look up an existing skill and rewrite its
    contents"; exclusive for structural change — anything that creates,
    moves, replaces or removes a skill *directory*, or rewrites the sync
    manifest.  Holding it shared across lookup-then-write is what makes a
    resolved path safe to use: no structural writer can rename the directory
    out from under the transaction.

``skill_write_lock``
    Exclusive on one already-resolved skill directory, taken while holding
    the namespace lock shared.  Independent skills stay concurrent.

Backends
--------
The POSIX backend locks the directory inode itself with ``flock``, so there
is no lock file to leak into the skill tree.  Windows cannot lock a directory
handle, so the fallback serializes on a single sentinel file kept *outside*
the skills tree (``~/.hermes/locks/skills.lock``) — it must stay outside, or
the file would be picked up by the bundled-skill content hashes and reported
as a user modification.

The fallback is selected by :func:`_select_backend`, which honours an explicit
override before falling back to platform detection.  That override is the test
seam: :func:`use_lock_backend` lets the sentinel path be exercised on POSIX CI
instead of only on Windows, where this code would otherwise never be run by a
test.

Configuration
-------------
``skills.lock_timeout`` in ``config.yaml`` sets how long a blocking write waits
before reporting "skill library is busy" (default 30s).  It does not apply to
the opportunistic waits, which exist precisely to give up quickly and retry —
see :func:`_resolve_timeout`.
"""

from __future__ import annotations

import errno
import logging
import os
import threading
import time
import contextvars
from contextlib import ExitStack, contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, Iterator, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

try:  # Unix: lock the directory inode, leaving no lock-file to clean up.
    import fcntl
except ImportError:  # pragma: no cover - exercised on Windows only
    fcntl = None
else:
    pass

try:
    import msvcrt
except ImportError:
    msvcrt = None


DEFAULT_TIMEOUT = 30.0

#: Contention wait for background/startup passes that can safely run later.
OPPORTUNISTIC_TIMEOUT = 2.0

#: User-facing setting for the blocking wait, in seconds.  A deployment that
#: runs many Hermes processes against one profile may want longer than 30s
#: before a write reports "skill library is busy".  Invalid or non-positive
#: values are ignored with a warning.
TIMEOUT_CONFIG_KEYS = ("skills", "lock_timeout")

#: INTERNAL bridge only — not a user-facing setting, and deliberately not
#: documented in ``cli-config.yaml.example``.  Two callers need to set the wait
#: without a config file:  the installer/bootstrap path runs ``sync_skills``
#: before a profile config exists, and spawned test/worker subprocesses need a
#: short wait without mutating the user's config.  Users configure
#: ``skills.lock_timeout`` (see AGENTS.md: ``.env`` is for secrets only).
_INTERNAL_TIMEOUT_ENV = "HERMES_INTERNAL_SKILL_LOCK_TIMEOUT"

BACKEND_FLOCK = "flock"
BACKEND_SENTINEL = "sentinel"

_namespace_lock_mode: contextvars.ContextVar[tuple[int, str] | None] = contextvars.ContextVar(
    "skill_namespace_lock_mode", default=None
)
_backend_override: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "skill_lock_backend_override", default=None
)


class SkillLockTimeout(TimeoutError):
    """Raised when another Hermes process holds a skill-library lock too long."""


def _skills_dir() -> Path:
    return get_hermes_home() / "skills"


def _coerce_timeout(raw: Any, source: str) -> Optional[float]:
    """Validate one configured wait.  Returns None when unusable."""
    if raw is None or raw == "":
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning("ignoring invalid %s=%r (expected seconds)", source, raw)
        return None
    if value <= 0:
        logger.warning("ignoring non-positive %s=%r", source, raw)
        return None
    return value


def _configured_timeout() -> Optional[float]:
    """Read ``skills.lock_timeout`` from config, or None if unset/unreadable.

    Deliberately not cached: lock acquisitions are a handful per skill write,
    not a hot path, and a cache would make a config edit require a restart for
    no measurable gain.  ``load_config_readonly`` skips the deepcopy that the
    mutable loader performs.  Imported lazily — the installer runs
    ``sync_skills`` (and therefore this module) before the CLI config layer is
    necessarily importable.
    """
    try:
        from hermes_cli.config import cfg_get, load_config_readonly

        raw = cfg_get(load_config_readonly(), *TIMEOUT_CONFIG_KEYS)
    except Exception:
        logger.debug("skill lock timeout: config unavailable", exc_info=True)
        return None
    return _coerce_timeout(raw, "skills.lock_timeout")


def _resolve_timeout(
    timeout: Optional[float], default: float, *, configurable: bool = True
) -> float:
    """Caller argument, then the internal bridge, then config, then *default*.

    ``configurable=False`` skips the config lookup for the opportunistic waits.
    ``skills.lock_timeout`` is how long a user is willing to *wait* before a
    write reports "busy"; applying it to a path whose whole purpose is to give
    up quickly and retry later would invert the setting — a generous 60s would
    stall startup for a minute instead of deferring the pass.
    """
    if timeout is not None:
        return timeout
    from_env = _coerce_timeout(os.environ.get(_INTERNAL_TIMEOUT_ENV), _INTERNAL_TIMEOUT_ENV)
    if from_env is not None:
        return from_env
    if configurable:
        from_config = _configured_timeout()
        if from_config is not None:
            return from_config
    return default


def _select_backend() -> str:
    override = _backend_override.get()
    if override is not None:
        return override
    return BACKEND_FLOCK if fcntl is not None else BACKEND_SENTINEL


@contextmanager
def use_lock_backend(backend: str) -> Iterator[None]:
    """Force a lock backend for the current context (test seam).

    Lets the sentinel fallback — the Windows path — be exercised on any
    platform, so its cross-process behaviour is covered by ordinary CI rather
    than only on Windows runners.
    """
    if backend not in (BACKEND_FLOCK, BACKEND_SENTINEL):
        raise ValueError(f"unknown skill lock backend: {backend!r}")
    if backend == BACKEND_FLOCK and fcntl is None:  # pragma: no cover - Windows
        raise RuntimeError("flock backend is unavailable on this platform")
    token = _backend_override.set(backend)
    try:
        yield
    finally:
        _backend_override.reset(token)


# ---------------------------------------------------------------------------
# flock backend (POSIX)
# ---------------------------------------------------------------------------

@contextmanager
def _lock_flock(path: Path, *, exclusive: bool, deadline: float) -> Iterator[None]:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        while True:
            try:
                fcntl.flock(fd, mode | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN):
                    raise
                if time.monotonic() >= deadline:
                    raise SkillLockTimeout(f"timed out waiting for lock on {path}")
                time.sleep(0.05)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


# ---------------------------------------------------------------------------
# Sentinel-file backend (Windows fallback; forced in tests via use_lock_backend)
# ---------------------------------------------------------------------------

# The OS-level sentinel lock is not re-entrant: a Windows byte-range lock
# conflicts with the same process taking it again on a second handle, and this
# module legitimately nests (shared namespace lock → per-skill write lock,
# delete → archive).  Serialize in-process first and only let the outermost
# acquisition reach the filesystem, so nesting cannot self-deadlock.
_sentinel_guard = threading.RLock()
_sentinel_depth = 0


def _sentinel_path() -> Path:
    """Single profile-local sentinel, deliberately outside ``skills/``.

    Keeping it out of the skill tree means it is never hashed by the bundled
    sync (which would report every skill as user-modified), never copied by a
    skill install, and never swept by a recursive delete of a skill directory.
    """
    return get_hermes_home() / "locks" / "skills.lock"


def _os_lock_sentinel(handle, *, deadline: float, lock_file: Path) -> None:
    while True:
        try:
            handle.seek(0)
            if msvcrt is not None:  # pragma: no cover - Windows only
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                # POSIX stand-in with the same one-byte, exclusive-only,
                # whole-sentinel semantics, so the forced-backend tests
                # exercise the real contract rather than an approximation.
                fcntl.lockf(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB, 1, 0)
            return
        except OSError:
            if time.monotonic() >= deadline:
                raise SkillLockTimeout(f"timed out waiting for lock on {lock_file}")
            time.sleep(0.05)


def _os_unlock_sentinel(handle) -> None:
    handle.seek(0)
    if msvcrt is not None:  # pragma: no cover - Windows only
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        fcntl.lockf(handle.fileno(), fcntl.LOCK_UN, 1, 0)


@contextmanager
def _lock_sentinel(path: Path, *, exclusive: bool, deadline: float) -> Iterator[None]:
    """Serialize on one sentinel regardless of *path*.

    The sentinel cannot express shared mode or per-skill granularity, so this
    backend is strictly stronger than the flock one: every skill write in the
    profile is serialized.  That is a throughput cost on Windows, not a
    correctness gap, and it is the documented platform difference.
    """
    global _sentinel_depth

    # Bound the in-process wait by the same deadline the OS wait would use, so
    # a stuck sibling thread surfaces as SkillLockTimeout rather than hanging.
    remaining = max(0.0, deadline - time.monotonic())
    if not _sentinel_guard.acquire(timeout=remaining):
        raise SkillLockTimeout(f"timed out waiting for lock on {_sentinel_path()}")
    try:
        if _sentinel_depth > 0:
            _sentinel_depth += 1
            try:
                yield
            finally:
                _sentinel_depth -= 1
            return

        lock_file = _sentinel_path()
        lock_file.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_file, "a+b") as handle:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b" ")
                handle.flush()
            _os_lock_sentinel(handle, deadline=deadline, lock_file=lock_file)
            _sentinel_depth += 1
            try:
                yield
            finally:
                _sentinel_depth -= 1
                try:
                    _os_unlock_sentinel(handle)
                except OSError:  # pragma: no cover - handle already closing
                    pass
    finally:
        _sentinel_guard.release()


@contextmanager
def _lock_path(path: Path, *, exclusive: bool, timeout: float) -> Iterator[None]:
    """Lock *path*, polling so callers receive a useful bounded failure."""
    deadline = time.monotonic() + timeout
    if _select_backend() == BACKEND_FLOCK:
        with _lock_flock(path, exclusive=exclusive, deadline=deadline):
            yield
    else:
        with _lock_sentinel(path, exclusive=exclusive, deadline=deadline):
            yield


@contextmanager
def skills_namespace_lock(
    *, exclusive: bool = True, timeout: Optional[float] = None
) -> Iterator[None]:
    """Lock the profile skill namespace.

    Readers take this shared while locating and modifying an existing skill;
    structural writers take it exclusively before checking names or moving
    directories.  The lock therefore covers the lookup-to-mutation interval.
    """
    # ``flock`` is not re-entrant across distinct opens in one process.  A
    # structural operation may call another structural helper (delete →
    # archive), so inherit an already-held namespace lock.  No code path may
    # upgrade a shared lock to exclusive; that would defeat the protocol.
    timeout = _resolve_timeout(timeout, DEFAULT_TIMEOUT)
    held = _namespace_lock_mode.get()
    if held is not None and held[0] == os.getpid():
        if exclusive and held[1] != "exclusive":
            raise RuntimeError("cannot upgrade a shared skill namespace lock")
        yield
        return

    root = _skills_dir()
    root.mkdir(parents=True, exist_ok=True)
    with _lock_path(root, exclusive=exclusive, timeout=timeout):
        token = _namespace_lock_mode.set(
            (os.getpid(), "exclusive" if exclusive else "shared")
        )
        try:
            yield
        finally:
            _namespace_lock_mode.reset(token)


@contextmanager
def try_namespace_lock(
    *, exclusive: bool = True, timeout: Optional[float] = None
) -> Iterator[bool]:
    """Namespace lock for passes that may safely be deferred.

    Yields ``True`` when the lock was taken and ``False`` when another process
    held it past *timeout*.  Used by idempotent background/startup work — a
    bundled-skill sync or a remote pull that loses the race simply runs again
    on the next pass, which is preferable to blocking startup behind a long
    agent-side skill write.
    """
    # Only the *acquisition* may be swallowed.  Wrapping the caller's body in
    # the same ``try`` would convert an unrelated SkillLockTimeout raised
    # inside it into a second ``yield``, which a context manager cannot do.
    timeout = _resolve_timeout(timeout, OPPORTUNISTIC_TIMEOUT, configurable=False)
    stack = ExitStack()
    try:
        stack.enter_context(skills_namespace_lock(exclusive=exclusive, timeout=timeout))
    except SkillLockTimeout:
        yield False
        return
    with stack:
        yield True


@contextmanager
def skill_materialize_lock(
    dest: Path, *, replace: bool = False, timeout: Optional[float] = None
) -> Iterator[bool]:
    """Lock for writing ONE skill directory from a remote sync source.

    Scoped per skill on purpose.  A remote pull fetches blobs over the network
    while it writes, so holding the namespace lock across the whole pull would
    park every agent-side skill write behind an unbounded HTTP round trip —
    and ``DEFAULT_TIMEOUT`` would start failing real user writes.  Locking one
    destination at a time keeps each hold short.

    ``replace=True`` (the org mirror, which rmtrees and recreates the
    directory) escalates to an exclusive namespace lock: a per-skill lock is
    attached to the directory's inode, which does not survive the replace, so
    a third process could lock the freshly created directory while the pull is
    still filling it.  An in-place materialize keeps per-skill granularity.

    Yields ``True`` when the lock was taken, ``False`` when the caller should
    skip this skill and let the next pull retry it.
    """
    timeout = _resolve_timeout(timeout, OPPORTUNISTIC_TIMEOUT, configurable=False)
    stack = ExitStack()
    try:
        if replace or not dest.is_dir():
            stack.enter_context(skills_namespace_lock(exclusive=True, timeout=timeout))
        else:
            stack.enter_context(skills_namespace_lock(exclusive=False, timeout=timeout))
            stack.enter_context(skill_write_lock(dest, timeout=timeout))
    except (SkillLockTimeout, FileNotFoundError):
        stack.close()
        yield False
        return
    with stack:
        yield True


@contextmanager
def skill_write_lock(skill_dir: Path, *, timeout: Optional[float] = None) -> Iterator[None]:
    """Exclusively lock an already-resolved skill directory.

    Callers must hold a shared :func:`skills_namespace_lock` while resolving
    and using ``skill_dir``.  This keeps unrelated skill writes concurrent but
    prevents a structural writer from renaming the directory mid-transaction.
    """
    timeout = _resolve_timeout(timeout, DEFAULT_TIMEOUT)
    if not skill_dir.is_dir():
        raise FileNotFoundError(f"skill directory no longer exists: {skill_dir}")
    with _lock_path(skill_dir, exclusive=True, timeout=timeout):
        yield


def namespace_write_locked(func):
    """Decorator for a complete skill-library structural transaction.

    Safe to stack with an outer :func:`skills_namespace_lock`: a caller that
    already holds the namespace lock exclusively (``skill_manage`` delete →
    ``archive_skill``) is short-circuited by the re-entrancy check rather than
    deadlocking on a second acquisition.
    """
    @wraps(func)
    def wrapped(*args, **kwargs):
        with skills_namespace_lock():
            return func(*args, **kwargs)
    return wrapped

"""Scratch entries the prune cannot delete as this user: remembered once, skipped until fixed.

A container that ran as root and wrote into a bind mount under scratch (the Docker terminal
backend without ``docker_run_as_host_user``, a test or review lane's ``docker run -v``) leaves
files this user cannot unlink, or directories it cannot even read. ``rmtree(ignore_errors=True)``
removes what it can, the entry stays idle, and every hourly pass re-walked the whole subtree and
logged the same "could not fully remove" line again (218 times per entry on one host). Here such
an entry is recorded with the directories that refused access, the user gets ONE line naming the
command that clears it, and later passes skip it before walking it. A record is dropped (and
the entry judged normally again) when the entry is gone or its top directory changed, when
every recorded blocking directory is accessible now (``chown``/``chmod`` of a nested dir), when
it was made by another uid, or after :data:`RECHECK_AFTER_SECONDS` in any case. Nothing is
escalated or shelled out to: the hint is for the user to run.

A tree this user sealed read-only itself is not such residue: a release builder publishes
immutable generations as 0555 dirs / 0444 files, and ``rmtree`` cannot unlink the children of a
dir without ``w``. Before a refusal counts, the rmtree error handler unseals (``u+rwx``) the
refusing path's parent and the path itself when each is a real directory this user owns inside
the entry, and retries the failed operation once (:func:`rmtree_collecting_permission_errors`).
Every directory is reached from the entry's parent one ``O_NOFOLLOW`` component at a time and
changed through an fd verified against its lstat, so no symlink is followed, nothing outside the
entry (not even the scratch root) and nothing another uid owns is ever chmod'ed. Only what stays
denied is recorded, and the line says whose it is.

POSIX only: without ``os.getuid`` (Windows) a PermissionError is usually a transient open
handle, so the prune keeps retrying there exactly as before.
"""
from __future__ import annotations

import contextlib
import errno
import json
import os
import shlex
import shutil
import stat
import sys
import tempfile
import time
from pathlib import Path

STUCK_STATE_NAME = ".prune_stuck.json"
STUCK_LOCK_NAME = STUCK_STATE_NAME + ".lock"
# Bounded retry: even a record whose blockers still look inaccessible is re-judged weekly, so a
# fix this module cannot see (ACLs, a sticky bit cleared, a changed group) is never skipped forever.
RECHECK_AFTER_SECONDS = 7 * 24 * 3600
# A record keeps this many blocking dirs; checking a subset can only re-judge early, never skip longer.
_MAX_BLOCKERS = 16
_PERMISSION_ERRNOS = frozenset({errno.EACCES, errno.EPERM})
# Records filed since this user's own sealed dirs are unsealed in place carry this version. One
# without it may have called a 0555 dir of this user's "root-owned"; it is judged again at once.
_RECORD_VERSION = 2
_OWNER_RWX = stat.S_IRWXU
_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
# A component is opened as a real directory only: a symlink fails (ELOOP/ENOTDIR), never followed.
_DIR_NOFOLLOW = getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0) | _CLOEXEC
# Linux: an fd for a dir this user may not read (mode 0) — names resolve through it, and it is
# chmod'ed via /proc/self/fd. Elsewhere such a dir is left alone (and recorded as before).
_O_PATH = getattr(os, "O_PATH", 0)


def tracking_supported() -> bool:
    return os.name == "posix" and hasattr(os, "geteuid")


def is_permission_error(exc: BaseException) -> bool:
    return isinstance(exc, OSError) and exc.errno in _PERMISSION_ERRNOS


def _owned_by_this_user(st: os.stat_result) -> bool:
    """Whether this user owns the inode *st* describes: the only test for "ours to unseal".
    (A seam: tests mark a dir foreign without needing root to chown it.)"""
    return st.st_uid == os.geteuid()  # windows-footgun: ok — only reached when tracking_supported()


def _within(path: str, top: str) -> bool:
    return path == top or path.startswith(top.rstrip(os.sep) + os.sep)


def _open_dir(name: str, dir_fd: int | None) -> int:
    """An fd for the real directory *name* in *dir_fd* (O_NOFOLLOW + O_DIRECTORY: a symlink is
    refused, not followed). Read-only when the dir allows it, else ``O_PATH`` where available."""
    try:
        return os.open(name, os.O_RDONLY | _DIR_NOFOLLOW, dir_fd=dir_fd)
    except PermissionError:
        if not _O_PATH:
            raise
        return os.open(name, _O_PATH | _DIR_NOFOLLOW, dir_fd=dir_fd)


def _parent_fd(top: str, path: str) -> tuple[int, str] | None:
    """``(fd of path's parent, path's name)`` with the parent reached from the entry's own parent
    (the scratch root, which may itself be a configured symlink) one real directory at a time.
    None when *path* is not *top* or lexically below it, or any component below the root is not
    a real directory (a symlink, gone, unreadable). The caller closes the fd."""
    if os.path.normpath(top) != top or os.path.normpath(path) != path or not _within(path, top):
        return None
    names = [os.path.basename(top)]
    if path != top:
        names += path[len(top):].lstrip(os.sep).split(os.sep)
    try:
        fd = os.open(os.path.dirname(top) or os.curdir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | _CLOEXEC)
    except OSError:
        return None
    for name in names[:-1]:
        try:
            child = _open_dir(name, fd)
        except OSError:
            os.close(fd)
            return None
        os.close(fd)
        fd = child
    return fd, names[-1]


def _unseal_at(parent_fd: int, name: str) -> bool:
    """``u+rwx`` on *name* in *parent_fd* when it is a real directory this user owns that lacks
    it; True when the mode was changed. The change goes through an fd opened with O_NOFOLLOW and
    checked to be the same inode as the lstat, so a symlink (or a path swapped for one) and
    another uid's directory are never changed."""
    try:
        st = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError:
        return False
    mode = stat.S_IMODE(st.st_mode)
    if not stat.S_ISDIR(st.st_mode) or not _owned_by_this_user(st) or mode & _OWNER_RWX == _OWNER_RWX:
        return False
    try:
        fd = _open_dir(name, parent_fd)
    except OSError:
        return False
    try:
        if not os.path.samestat(st, os.fstat(fd)):
            return False
        try:
            os.fchmod(fd, mode | _OWNER_RWX)
        except OSError as exc:
            if exc.errno != errno.EBADF or not _O_PATH:
                raise
            os.chmod(f"/proc/self/fd/{fd}", mode | _OWNER_RWX)  # an O_PATH fd: fchmod is EBADF
    except OSError:
        return False
    finally:
        os.close(fd)
    return True


def _unseal_around(top: str, path: str) -> list[str]:
    """Unseal the parent of *path* and *path* itself (each only if it is a real directory this
    user owns, at or below the entry *top*, lacking ``u+rwx``); return the ones changed. The
    entry's own parent, the scratch root, is never a candidate."""
    if not tracking_supported():
        return []
    changed: list[str] = []
    for candidate in ([path] if path == top else [os.path.dirname(path), path]):
        located = _parent_fd(top, candidate)
        if located is None:
            continue
        fd, name = located
        try:
            if _unseal_at(fd, name):
                changed.append(candidate)
        finally:
            os.close(fd)
    return changed


def _owned_dir_blocker(path: str) -> bool:
    try:
        st = os.lstat(path)
    except OSError:
        return False
    return stat.S_ISDIR(st.st_mode) and _owned_by_this_user(st)


def _all_owned_by_this_user(blockers: list[str]) -> bool:
    """True when every blocker still there is this user's own (none foreign, none uninspectable)."""
    seen = False
    for b in blockers:
        try:
            st = os.lstat(b)
        except FileNotFoundError:
            continue
        except OSError:
            return False  # cannot even inspect it: not provably ours
        if not _owned_by_this_user(st):
            return False
        seen = True
    return seen


def _fingerprint(entry: Path) -> list[int] | None:
    try:
        st = os.lstat(entry)
    except OSError:
        return None
    # ctime too: ``sudo chown -R``/``chmod -R`` fixes the entry without touching its mtime.
    return [st.st_ino, st.st_mtime_ns, st.st_ctime_ns]


def _accessible(path: str) -> bool:
    """True when this user can now list and delete inside *path* (or it is gone)."""
    if not os.path.lexists(path):
        return True
    kw = {"effective_ids": True} if os.access in os.supports_effective_ids else {}
    return os.access(path, os.R_OK | os.W_OK | os.X_OK, **kw)


def _still_stuck(root: Path, name: str, rec: dict, now: float) -> bool:
    blockers = rec.get("blockers")
    if rec.get("v") != _RECORD_VERSION and isinstance(blockers, list) and any(
        isinstance(b, str) and _owned_dir_blocker(b) for b in blockers
    ):
        return False  # filed before own sealed dirs were unsealed in place: removable now
    return (
        rec.get("uid") == os.geteuid()  # windows-footgun: ok — only reached when tracking_supported()
        and now - float(rec.get("since") or 0) < RECHECK_AFTER_SECONDS
        and rec.get("fp") == _fingerprint(root / name)
        and isinstance(blockers, list) and bool(blockers)
        and not all(_accessible(b) for b in blockers if isinstance(b, str))
    )


def _read(root: Path) -> dict[str, dict]:
    # A torn or hand-edited file reads as empty: the worst case is one re-walk and one line.
    try:
        data = json.loads((root / STUCK_STATE_NAME).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}
    return {k: v for k, v in data.items() if isinstance(v, dict)} if isinstance(data, dict) else {}


def _write(root: Path, state: dict[str, dict]) -> None:
    """Publish *state* via a unique temp file + ``os.replace``: a reader sees old or new, never torn."""
    path = root / STUCK_STATE_NAME
    if not state:
        with contextlib.suppress(OSError):
            path.unlink(missing_ok=True)
        return
    try:
        fd, tmp = tempfile.mkstemp(dir=root, prefix=STUCK_STATE_NAME + ".", suffix=".tmp")
    except OSError:
        return
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=1, sort_keys=True)
        os.replace(tmp, path)
    except OSError:
        with contextlib.suppress(OSError):
            os.unlink(tmp)


@contextlib.contextmanager
def _locked(root: Path):
    """Exclusive cross-process lock around a read-merge-write of the state file. Two pruners of
    one home (separate processes past the ``.last_prune`` throttle together) otherwise overwrite
    each other's records from stale snapshots. POSIX only, like the tracking itself."""
    import fcntl

    fd = os.open(root / STUCK_LOCK_NAME, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)  # closing the descriptor releases the flock


def load_skips(root: Path, *, persist: bool = True) -> frozenset[str]:
    """Names to skip this pass. Records that no longer hold (see module doc) are dropped, so
    that entry is judged normally again."""
    if not tracking_supported() or not (root / STUCK_STATE_NAME).exists():
        return frozenset()
    now = time.time()
    if not persist:
        state = _read(root)
        return frozenset(n for n, rec in state.items() if _still_stuck(root, n, rec, now))
    try:
        with _locked(root):
            state = _read(root)
            live = {n: rec for n, rec in state.items() if _still_stuck(root, n, rec, now)}
            if live != state:
                _write(root, live)
    except OSError:
        return frozenset()
    return frozenset(live)


def rmtree_collecting_permission_errors(entry: Path, *, unsealed: list[str] | None = None) -> list[str]:
    """``rmtree`` that keeps going past errors (like ``ignore_errors=True``) and returns the
    directories that refused access with EACCES/EPERM (empty: nothing was denied).

    A refusal is first answered by unsealing: the refusing path's parent and the path itself,
    when each is a real directory this user owns inside *entry*, get ``u+rwx`` and the failed
    operation is retried once (see the module doc). Only what still refuses (another uid's
    directory, an immutable flag) is returned. Each directory unsealed is appended to
    *unsealed* when given."""
    blockers: list[str] = []
    changed_dirs: list[str] = [] if unsealed is None else unsealed
    tried: set[tuple[str, str]] = set()
    top = os.fspath(entry)

    def _rmtree(path: str, dir_fd: int | None = None, reported_as: str | None = None) -> None:
        def on(func, p, exc):
            # Under *dir_fd*, rmtree reports paths relative to it: name them as the entry sees them.
            _onexc(func, p if reported_as is None else os.path.join(os.path.dirname(reported_as), p), exc)

        if sys.version_info >= (3, 12):
            shutil.rmtree(path, onexc=on, dir_fd=dir_fd)
        else:  # pragma: no cover - 3.11
            shutil.rmtree(path, onerror=lambda f, p, info: on(f, p, info[1]), dir_fd=dir_fd)

    def _unseal_and_retry(func, path: str) -> bool:
        """True when *path* was unsealed and the failed operation retried (a new refusal of the
        retry is reported through ``_onexc`` again, which then records it)."""
        key = (getattr(func, "__name__", repr(func)), path)
        if key in tried:
            return False
        tried.add(key)
        changed = _unseal_around(top, path)
        if not changed:
            return False
        changed_dirs.extend(c for c in changed if c not in changed_dirs)
        located = _parent_fd(top, path)
        if located is None:
            return False
        fd, name = located
        try:
            st = os.stat(name, dir_fd=fd, follow_symlinks=False)
            if func is os.rmdir:
                os.rmdir(name, dir_fd=fd)
            elif stat.S_ISDIR(st.st_mode):
                _rmtree(name, dir_fd=fd, reported_as=path)  # a dir rmtree could not open or list
            else:
                os.unlink(name, dir_fd=fd)  # a file, or a symlink itself (never its target)
        except FileNotFoundError:
            pass
        except OSError as exc:
            _onexc(func, path, exc)  # same key: recorded as a blocker, not retried again
        except NotImplementedError:
            return False  # no dir_fd rmtree here: recorded as before
        finally:
            os.close(fd)
        return True

    def _onexc(func, path, exc):
        if not is_permission_error(exc):
            return
        if tracking_supported() and _unseal_and_retry(func, path):
            return
        # scandir/open of a dir fails on the dir itself; unlink/rmdir on its parent's perms.
        blocker = path if func in (os.scandir, os.open, os.lstat) else os.path.dirname(path)
        if blocker not in blockers:
            blockers.append(blocker)

    _rmtree(top)
    return blockers


def removal_hint(entry: Path) -> str:
    from hermes_platform.resolver import locate_command

    hint = f"sudo rm -rf {shlex.quote(os.fspath(entry))}"
    if locate_command("docker").command:
        hint += (
            f"  (or without sudo: docker run --rm -v {shlex.quote(os.fspath(entry.parent) + ':/s')}"
            f" alpine rm -rf {shlex.quote('/s/' + entry.name)})"
        )
    return hint


def own_removal_hint(entry: Path) -> str:
    q = shlex.quote(os.fspath(entry))
    return f"chmod -R u+rwx {q} && rm -rf {q}"


def log_unsealed(audit, entry: Path, unsealed: list[str], outcome: str) -> None:
    """The line for an entry whose own sealed dirs were unsealed: *outcome* is ``removed``,
    ``foreign`` (what is left belongs to another uid) or ``left`` (anything else). Nothing to say when
    nothing was unsealed, so callers need no guard of their own."""
    if not unsealed:
        return
    uid = os.geteuid()  # windows-footgun: ok — unsealing only happens when tracking_supported()
    if outcome == "removed":
        audit.info(
            "scratch prune: unsealed and removed %r: %d read-only dir(s) sealed by this user "
            "(uid %d) were made writable (u+rwx) so it could be deleted",
            os.fspath(entry), len(unsealed), uid,
        )
        return
    audit.info(
        "scratch prune: unsealed %d read-only dir(s) sealed by this user (uid %d) in %r; what is "
        "left %s",
        len(unsealed), uid, os.fspath(entry),
        "is owned by another user" if outcome == "foreign" else "could not be removed",
    )


def record(root: Path, entry: Path, blockers: list[str], audit, *, unsealed: list[str] | None = None) -> None:
    """Remember *entry* as not removable by this user and log the one actionable line. The line
    names whose the blockers are: another uid's (root/Docker residue: unchanged wording) or, when
    every one is this user's own and still refuses (mode 0 the scan cannot open, an immutable
    flag, an ACL), this user's, with a fix that needs no sudo."""
    fp = _fingerprint(entry)
    if fp is None or not blockers:
        return
    rec = {"v": _RECORD_VERSION, "fp": fp, "since": time.time(), "uid": os.geteuid(), "blockers": blockers[:_MAX_BLOCKERS]}  # windows-footgun: ok — only reached when tracking_supported()
    with contextlib.suppress(OSError), _locked(root):
        state = _read(root)  # re-read under the lock: merge, never publish a stale snapshot
        state[entry.name] = rec
        _write(root, state)
    own = _all_owned_by_this_user(blockers)
    if unsealed:
        log_unsealed(audit, entry, unsealed, "left" if own else "foreign")
    if own:
        audit.info(
            "scratch prune: cannot remove %r: it holds directories owned by this user (uid %d) "
            "that stay unreadable or undeletable (mode 0, an immutable flag or an ACL); skipping "
            "it until that is fixed (re-checked weekly). Remove it with: %s",
            os.fspath(entry), rec["uid"], own_removal_hint(entry),
        )
        return
    audit.info(
        "scratch prune: cannot remove %r: it holds files this user (uid %d) may not read or "
        "delete, usually root-owned output of a Docker container; skipping it until that is "
        "fixed (re-checked weekly). Remove it with: %s",
        os.fspath(entry), rec["uid"], removal_hint(entry),
    )


def stuck_entries(root: Path) -> list[Path]:
    """Entries the prune is currently skipping as not removable (for ``hermes doctor``)."""
    return [root / name for name in sorted(load_skips(root, persist=False))]

"""Startup requests for PM recovery and rescue of orphaned launchers."""

from __future__ import annotations

import contextlib
import errno
import os
import re
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath

# Older installers left renamed launchers behind on failure. The startup
# orphan sweep still restores them. Generation installs never rename live shims.
QUARANTINE_RESTORE_BACKOFF_MS: tuple[int, ...] = (0, 100, 250, 500, 1000)


def restore_quarantined_shims(
    moved: list[tuple[Path, Path]], *, stream=None,
    backoff_ms: tuple[int, ...] = QUARANTINE_RESTORE_BACKOFF_MS,
) -> list[tuple[Path, Path]]:
    """Rename quarantined shims back, retrying a lock instead of giving up.

    A pair is not a failure when ``original`` already exists or ``quarantined`` has gone: the
    installer wrote a fresh shim, or a concurrent sweep won the race. Both are silent, so two
    processes sweeping the same orphan cannot produce a spurious error.
    """
    if stream is None:
        stream = sys.stderr
    failed: list[tuple[Path, Path]] = []
    for original, quarantined in moved:
        last_exc: OSError | None = None
        for delay_ms in backoff_ms:
            try:
                if os.path.exists(original) or not os.path.exists(quarantined):
                    last_exc = None
                    break
                if delay_ms:
                    time.sleep(delay_ms / 1000.0)
                os.rename(quarantined, original)
                last_exc = None
                break
            except OSError as exc:
                last_exc = exc
        if last_exc is None:
            continue
        failed.append((original, quarantined))
        name = os.path.basename(str(original))
        stem = name[:-4] if name.lower().endswith(".exe") else name
        print(
            f"  ✖ FAILED to restore {name} "
            f"({last_exc.__class__.__name__}) — it is still quarantined "
            f"as {os.path.basename(str(quarantined))}.\n"
            f"    `{stem}` will NOT be on PATH until it is put back. Run this, "
            f"then re-run the update:\n"
            f'      move "{quarantined}" "{original}"',
            file=stream,
        )
    return failed


# Set only when this process successfully finishes a deferred core install for an ``update``
# invocation. The CLI import that follows must not resolve external secret sources: a configured
# source can map cryptography._rust and immediately recreate the self-lock marker this fresh
# process just consumed. Process-local on purpose so children do not inherit the exception.
_UPDATE_RETRY_RECOVERED = False


def _should_skip_external_secret_sources() -> bool:
    """True inside any ``hermes update`` process (and its import probes).

    Every dotenv load in the process — ``hermes_cli.main``, ``run_agent``, ``cli`` — consults
    this, so the updater never resolves external secret sources: on Windows they map
    ``cryptography._rust.pyd`` into the process replacing that venv, and everywhere a slow
    ``op``/``bws``/command helper (up to 120s per source) would run inside the updater's
    120s critical-module import probe and be reported as an import-health timeout.
    Profile flags are stripped before ``hermes_cli.main`` loads dotenv, so ``argv[1]`` is
    the authoritative subcommand.
    """
    return _UPDATE_RETRY_RECOVERED or sys.argv[1:2] == ["update"]


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _read_marker_attempts(marker_path: Path) -> int:
    """Attempt counter from a marker's opportunistic JSON body; corrupt/missing → 0."""
    try:
        raw = marker_path.read_text(encoding="utf-8-sig", errors="replace").strip()
    except OSError:
        return 0
    if not raw:
        return 0
    try:
        import json

        attempts = json.loads(raw).get("attempts", 0)
        if type(attempts) is not int:  # only our writers' ints count: Infinity/NaN/true never reach int()
            raise TypeError(attempts)
        return max(0, attempts)
    except (ValueError, AttributeError, TypeError):
        for line in reversed(raw.splitlines()):
            key, separator, value = line.partition("=")
            if separator and key.strip() == "attempts":
                try:
                    return max(0, int(value))
                except ValueError:
                    return 0
        return 0


def _process_state(pid: int) -> str | None:
    """Single-letter process state (``ps`` style), or ``None`` when unknowable.

    ``os.kill(pid, 0)`` also succeeds for a ZOMBIE — a process that exited but
    whose parent has not reaped it yet. Reading the state lets callers treat a
    zombie as dead, so a crashed update stage lingering under an un-reaping
    parent cannot keep a stale update marker "live" for the whole age ceiling
    (#77259, #120635, #125932). Best-effort and stdlib-only: on any failure the
    answer is ``None`` and callers keep their signal-0 verdict.
    """
    if sys.platform == "linux":
        try:
            with open(f"/proc/{pid}/stat", "rb") as fh:
                stat = fh.read()
        except OSError:
            return None
        # Field 3 is the state, but comm may contain spaces/parens: anchor on
        # the closing paren of comm instead of splitting on whitespace.
        comm_end = stat.rfind(b")")
        if comm_end < 0:
            return None
        return stat[comm_end + 2 : comm_end + 3].decode("ascii", "replace") or None
    if sys.platform == "darwin":
        try:
            out = subprocess.run(
                ["ps", "-o", "stat=", "-p", str(pid)],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=5, check=False,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return None
        return out.strip()[:1] or None
    return None


def _pid_is_running(pid: int) -> bool:
    """Best-effort stdlib-only process liveness probe.

    ``os.kill(pid, 0)`` is not a no-op on Windows, so use the Win32 process handle API there. An
    access-denied result counts as live: racing an elevated updater is worse than postponing
    recovery for one launch. A zombie (exited, un-reaped) counts as dead — see
    :func:`_process_state`.
    """
    if pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            import ctypes

            synchronize = 0x00100000
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
            kernel32.OpenProcess.restype = ctypes.c_void_p
            kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
            kernel32.WaitForSingleObject.restype = ctypes.c_ulong
            kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
            kernel32.CloseHandle.restype = ctypes.c_int
            handle = kernel32.OpenProcess(synchronize, False, pid)
            if not handle:
                return ctypes.get_last_error() == 5  # ERROR_ACCESS_DENIED
            try:
                return kernel32.WaitForSingleObject(handle, 0) == 258
            finally:
                kernel32.CloseHandle(handle)
        except Exception:
            return True
    try:
        os.kill(pid, 0)  # windows-footgun: ok — Windows returns above
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    state = _process_state(pid)
    if state is not None and state.upper().startswith("Z"):
        return False  # exited, unreaped — not a live owner
    return True


def _marker_owner_is_live(marker: Path) -> bool:
    """True when a legacy update marker names a process still running."""
    try:
        body = marker.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return False
    for line in body.splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() == "pid":
            try:
                return _pid_is_running(int(value.strip()))
            except ValueError:
                return False
    return False


# ``hermes update`` writes this into the git dir right before git moves the checkout and removes it
# once git has exited (a kill is the only exit that keeps it). Git rewrites the tree file by file and
# moves HEAD last, so an update killed in between leaves HEAD on the old commit with a prefix of the
# files already new; that mixed tree fails at import in every entry point, ``hermes update`` included.
INTERRUPTED_PULL_MARKER = "hermes-update-pull"
# A fast-forward takes seconds; past this a "live" owner pid is a recycled one.
_INTERRUPTED_PULL_MAX_AGE_SECONDS = 10 * 60
# The user (or a killed updater) is mid-operation: its own state files own the tree.
_GIT_OPERATION_IN_PROGRESS = ("MERGE_HEAD", "CHERRY_PICK_HEAD", "REVERT_HEAD", "rebase-merge", "rebase-apply")
_REGULAR_FILE_MODES = ("100644", "100755")
# What git writes into the tree itself: files and symlinks (a gitlink, 160000, is a submodule's own checkout).
_WORKTREE_BLOB_MODES = (*_REGULAR_FILE_MODES, "120000")


def git_operation_in_progress(root: Path) -> str | None:
    """Return the active Git operation currently controlling *root*, if any."""
    git_dir = _git_dir(root)
    for marker in _GIT_OPERATION_IN_PROGRESS:
        state = git_dir / marker
        if not state.exists():
            continue
        if marker == "rebase-apply":
            return "am" if (state / "applying").exists() else "rebase"
        if marker == "rebase-merge":
            return "rebase"
        return marker.removesuffix("_HEAD").lower().replace("_", "-")
    return None


def _git_dir(root: Path) -> Path:
    """``root``'s git dir: ``.git`` itself, or where a linked worktree's ``.git`` file points."""
    dot_git = root / ".git"
    if dot_git.is_file():
        text = dot_git.read_text(encoding="utf-8-sig").strip()
        if text.startswith("gitdir:"):
            return root / text[len("gitdir:"):].strip()
    return dot_git


def interrupted_pull_marker(root: Path) -> Path:
    return _git_dir(root) / INTERRUPTED_PULL_MARKER


# The repair's own code is in the tree a killed move tears (``hermes_bootstrap``, this module, the
# package initializer, the lock/custody modules). ``update_cmd_commit.arm_tree_move`` publishes this
# closure, as committed at the marker's ``pre``, beside the marker before git writes; a minted
# launcher whose checkout import fails runs the repair from there, reading the same files from git's
# objects first when no updater published them (``_launchers._CLOSURE_REPAIR``).
# The modules import only the stdlib and each other; the package initializer is published empty.
RECOVERY_CLOSURE_DIR = "hermes-update-recovery"
RECOVERY_CLOSURE = ("hermes_cli/_early_recovery.py", "hermes_cli/update_lock.py", "hermes_cli/update_custody.py")


# ``<blob id> <path>`` per published file (``RECOVERY_CLOSURE`` + the empty package initializer):
# a closure is used only when every file hashes to its id (a power loss can leave the renamed dir
# with files git's objects never had), else it is rebuilt from git's objects.
RECOVERY_CLOSURE_MANIFEST = "MANIFEST"
RECOVERY_CLOSURE_INIT = "hermes_cli/__init__.py"


def recovery_closure_dir(root: Path, pre: str) -> Path:
    if not is_object_id(pre):  # never a revision (``HEAD``) or a path (``../x``) under the git dir
        raise ValueError(f"not a commit id: {pre!r}")
    return _git_dir(root) / RECOVERY_CLOSURE_DIR / pre


def is_object_id(value: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", value or ""))


def blob_id(data: bytes, like: str) -> str:
    """Git's object id of a blob holding ``data``, in ``like``'s object format (SHA-1 or SHA-256)."""
    import hashlib

    return hashlib.new("sha1" if len(like) == 40 else "sha256", b"blob %d\0" % len(data) + data).hexdigest()


def recovery_closure_verified(closure: Path, pre: str) -> bool:
    """Every closure file present with exactly the bytes its manifest names, and nothing missing."""
    try:
        listed = dict(line.split(" ", 1)[::-1] for line in
                      (closure / RECOVERY_CLOSURE_MANIFEST).read_text(encoding="utf-8-sig").splitlines())
        return (set(listed) == {*RECOVERY_CLOSURE, RECOVERY_CLOSURE_INIT}
                and listed[RECOVERY_CLOSURE_INIT] == blob_id(b"", pre)
                and all(blob_id((closure / rel).read_bytes(), pre) == oid for rel, oid in listed.items()))
    except (OSError, ValueError):
        return False


def _git_executable(recorded_by_updater: str = "") -> str:
    """The git ``hermes update`` runs (``_subprocess_compat.expose_pm_git``), without installing it.

    The absolute git the killed updater recorded in its marker first: finding PM's copy needs
    ``pm``, which imports ``hermes_constants``, and a merge killed while writing that module (or
    anything else ``pm`` imports) leaves only this path to the repair. Then PATH's git. A Windows install whose only git is the one install.ps1 staged in PM's store
    has none on PATH until the updater exposes it, so this falls back to that copy: PM's recorded
    entry, or the lockfile's pinned entry the installer extracted without recording it. A bare
    ``git`` there dies with WinError 2 and the torn tree this repair exists for stays torn.
    """
    import shutil

    if recorded_by_updater and os.path.isfile(recorded_by_updater) and _is_git(recorded_by_updater):
        return recorded_by_updater
    found = shutil.which("git")
    if found:
        return found
    with contextlib.suppress(Exception):  # no PM, no store, no Windows git package: PATH's answer stands
        import pm
        from pm import paths
        from pm.lock import Lockfile

        recorded = pm.installed_package("git", allow_outdated=True)
        if recorded is not None and recorded.binary is not None and recorded.binary.is_file():
            return str(recorded.binary)
        package, target = pm.get_package("git"), pm.current_target()
        version = Lockfile(paths.lockfile_path()).version("git")
        staged = package.binary(paths.store_root() / package.store_entry(version, target), target)
        if staged is not None and staged.is_file():
            return str(staged)
    return "git"


def _is_git(path: str) -> bool:
    """``path`` still runs as git: ``--version`` exits 0 and says ``git version`` (m5). A recorded
    git that decayed into anything else (gone executable, a stub that exits 0 silently) must not
    answer the repair's questions: its empty ``rev-parse`` would read as "HEAD moved"."""
    try:
        probe = subprocess.run([path, "--version"], capture_output=True, text=True, encoding="utf-8",
                               errors="replace", timeout=30, stdin=subprocess.DEVNULL,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, ValueError, subprocess.SubprocessError):
        return False
    return probe.returncode == 0 and probe.stdout.startswith("git version")


# A full object name (SHA-1 or SHA-256): what `git rev-parse --verify HEAD` must print (m5).
_OID = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")


def _trees_git_could_write(git, pre: str, target: str) -> tuple[list[str], set[str]]:
    """The trees the killed git was moving the checkout to, and the paths whose new content is unknowable.

    A fast-forward or ``reset --hard`` writes ``target``. On a custom branch the updater runs
    ``git merge``, whose files are the merge of both sides: ``merge-tree`` computes the same tree. A
    conflicted path (its markers carry other labels) and, on git < 2.38 (no ``--write-tree``), every
    path both sides changed count as git's whatever their content.
    """
    base = git("merge-base", pre, target).stdout.strip()
    if not base or base == pre:  # fast-forward, or unrelated histories (only a reset can land those)
        return [target], set()
    merged = git("merge-tree", "--write-tree", "-z", "--name-only", "--no-messages", pre, target)
    if merged.returncode in (0, 1):  # 1: conflicts
        tree, *conflicted = merged.stdout.split("\0")
        return [target, tree], set(filter(None, conflicted))
    changed = [set(filter(None, git("diff", "--name-only", "-z", "--no-renames", base, side).stdout.split("\0")))
               for side in (pre, target)]
    return [target], changed[0] & changed[1]


def _hash_worktree(git, paths: list[str]) -> dict[str, str]:
    """Blob ids of the checkout's files, through the repo's clean filters, like ``git add`` would store."""
    listed = [p for p in paths if "\n" not in p]  # --stdin-paths is newline-delimited
    blobs = {}
    if listed:
        hashed = git("hash-object", "--stdin-paths", stdin="\n".join(listed) + "\n")
        if hashed.returncode != 0:
            raise subprocess.SubprocessError(hashed.stderr.strip())
        blobs.update(zip(listed, hashed.stdout.split()))
    for path in set(paths) - set(listed):
        blobs[path] = git("hash-object", "--", path).stdout.strip()
    return blobs


def _modes_git_wrote(git, root: Path, entries: dict, links: set[str]) -> set[str]:
    """Paths whose new entry keeps the ``pre`` blob under another mode, already in that mode.

    Equal bytes cannot tell a regular file holding ``original`` from a symlink to ``original`` (one
    blob id), nor 644 from 755: only the entry's mode can (review N07). The worktree shows a link
    (lstat), except under ``core.symlinks=false`` where git checks a link out as a plain file holding
    its target, and the executable bit, except on Windows. The index shows the mode git staged where
    the worktree cannot; git writes it after the files, so it covers a git killed after a whole tree.
    """
    wanted = {path: {m for m, b in new if b == old_blob and m != old_mode}
              for path, (old_mode, old_blob, new) in entries.items() if old_blob is not None}
    wanted = {path: modes for path, modes in wanted.items() if modes}
    if not wanted:
        return set()
    symlinks = git("config", "--type=bool", "core.symlinks").stdout.strip() != "false"
    staged = git("ls-files", "-s", "-z", "--", *wanted)
    if staged.returncode != 0:
        raise subprocess.SubprocessError(staged.stderr.strip())
    index = {path: meta.split()[0] for meta, _tab, path in
             (record.partition("\t") for record in staged.stdout.split("\0")) if path}
    written = set()
    for path, modes in wanted.items():
        if path in links:
            shown = {"120000"}
        elif not (root / path).is_file():
            shown = set()  # missing: the caller judges it by absence
        else:
            shown = set(_REGULAR_FILE_MODES) if sys.platform == "win32" else \
                {"100755" if (root / path).stat().st_mode & 0o100 else "100644"}
            if not symlinks:
                shown.add("120000")
        if (entries[path][0] not in shown and shown & modes) or index.get(path) in modes:
            written.add(path)
    return written


def _paths_git_wrote(git, root: Path, pre: str, target: str) -> tuple[list[str], list[str], set[str], set[str]] | None:
    """Paths the killed git already touched on the way to ``target``: (restore from HEAD, remove as added,
    directories git may have created for its added files, the added ones to keep aside, not delete).

    Git rewrites a file as unlink, create, write, so a kill leaves it missing, empty or cut short:
    all of those count as git's, like the full new blob. Content that matches neither side and is not
    the start of a new blob is the user's own edit (e.g. a re-applied stash) and is left alone. An
    added path holding less than a whole new blob may equally be the user's own file created there
    after the kill (a ``touch``, a first line): it leaves the tree, but renamed aside, never deleted.
    ``None``: git no longer knows ``target``.
    """
    if git("rev-parse", "-q", "--verify", f"{target}^{{commit}}").returncode != 0:
        return None
    trees, unknown = _trees_git_could_write(git, pre, target)
    entries = {}  # path -> (pre mode, pre blob or None when git adds it, [(new mode, new blob or None)])
    for tree in trees:
        diff = git("diff", "--raw", "-z", "--no-renames", "--no-abbrev", pre, tree)
        if diff.returncode != 0:
            raise subprocess.SubprocessError(diff.stderr.strip())
        parts = diff.stdout.split("\0")
        for meta, path in zip(parts[::2], parts[1::2]):
            old_mode, new_mode, old_blob, new_blob, status = meta.lstrip(":").split()
            if (old_mode if status == "D" else new_mode) not in _WORKTREE_BLOB_MODES:
                continue
            entry = entries.setdefault(path, (old_mode, None if status == "A" else old_blob, []))
            entry[2].append((new_mode, None if status == "D" else new_blob))
    # A symlink's blob is its target text, read without following it; core.symlinks=false checks a
    # link out as a plain file holding that text, which hash-object already matches.
    links = {path for path in entries if os.path.islink(root / path)}
    worktree_blob = _hash_worktree(git, [path for path in entries if path not in links and (root / path).is_file()])
    worktree_blob.update({path: blob_id(os.fsencode(os.readlink(root / path)), pre) for path in links})
    retyped = _modes_git_wrote(git, root, entries, links)
    restore, added, kept = [], [], set()
    for path, (old_mode, old_blob, new) in entries.items():
        file, blobs = root / path, {blob for _mode, blob in new if blob}
        if path not in worktree_blob:
            written = old_blob is not None  # unlinked (or deleted), not yet recreated
        elif worktree_blob[path] == old_blob:  # only the entry's mode tells whether git got here
            written = path in retyped
        elif worktree_blob[path] in blobs or path in unknown:
            written = True
        elif path in links:  # git creates a symlink whole: any other target is the user's
            written = False
        else:  # git's own file cut short starts one of the new blobs
            content = file.read_bytes()
            written = False
            for blob in blobs:
                shown = git("cat-file", "--filters", f"--path={path}", blob, text=False)
                if shown.returncode != 0:
                    raise subprocess.SubprocessError(shown.stderr.decode(errors="replace").strip())
                written = written or shown.stdout.startswith(content)
        if written:
            (added if old_blob is None else restore).append(path)
            if old_blob is None and worktree_blob.get(path) not in blobs:  # a whole blob is in git's objects
                kept.add(path)
    new_dirs = {str(parent) for path, (_m, old_blob, _n) in entries.items() if old_blob is None
                for parent in PurePosixPath(path).parents if parent.parts}
    if new_dirs:
        new_dirs -= set(git("ls-tree", "-r", "-d", "--name-only", "-z", pre).stdout.split("\0"))
    return restore, added, new_dirs, kept


# Launches that start together after a killed update (a restarting gateway or Desktop backend next to
# the user's CLI) restore one at a time: the claim holder owns the git index, the rest wait for it and
# relaunch from its tree. An OS lock, not a pid file: the kernel drops it with its owner, so a dead
# launch's claim is broken without a check-then-unlink race.
_RESTORE_CLAIM = "hermes-update-pull.claim"
_RESTORE_CLAIM_WAIT_SECONDS = 10.0
_merge_advice_shown = False


# flock's EWOULDBLOCK and msvcrt LK_NBLCK's EACCES/EDEADLOCK: another process holds the claim.
_CLAIM_HELD_ERRNOS = frozenset({errno.EAGAIN, errno.EWOULDBLOCK, errno.EACCES, errno.EDEADLK})


def _lock_fd(fd: int, lock: bool) -> bool:
    """False only while another launch holds the claim.

    A filesystem that cannot lock at all (ENOLCK on NFS without lockd, EOPNOTSUPP/EINVAL on some SMB
    shares) proceeds unguarded like a read-only git dir: counting it as "held" would stop every launch
    from ever repairing a checkout the restore handled fine before claims existed.
    """
    try:
        if sys.platform == "win32":
            import msvcrt

            os.lseek(fd, 0, os.SEEK_SET)  # msvcrt locks from the file position
            msvcrt.locking(fd, msvcrt.LK_NBLCK if lock else msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(fd, (fcntl.LOCK_EX | fcntl.LOCK_NB) if lock else fcntl.LOCK_UN)
    except OSError as exc:
        return not (lock and exc.errno in _CLAIM_HELD_ERRNOS)
    return True


@contextlib.contextmanager
def _restore_claim(git_dir: Path):
    """Yields True while this launch holds the restore claim, False when another held it past the wait."""
    try:
        fd = os.open(git_dir / _RESTORE_CLAIM, os.O_RDWR | os.O_CREAT, 0o644)
    except OSError:  # read-only git dir: no restore can write there either, the attempt reports why
        yield True
        return
    try:
        deadline = time.monotonic() + _RESTORE_CLAIM_WAIT_SECONDS
        while not _lock_fd(fd, True):
            if time.monotonic() > deadline:
                yield False
                return
            time.sleep(0.05)
        try:
            os.ftruncate(fd, 0)
            os.write(fd, f"pid={os.getpid()}\n".encode())
        except OSError:
            pass
        try:
            yield True
        finally:
            _lock_fd(fd, False)
    finally:
        os.close(fd)


class _Holder(str):
    """A process that may own a git lock, e.g. ``pid 4242 (git commit)``: truthy, and never ``is True``."""


# Git subcommands that only read: none takes ``index.lock`` and then waits with its fd CLOSED. Every
# other git (``commit``/``merge``/``rebase``/``am``/``stash``... in the editor or a hook, a dashed
# ``git-commit`` from git-core, an alias, a wrapper, a third-party ``git-<tool>``, a git whose
# subcommand cannot be read) may hold ``index.lock`` with no fd naming it, so it counts as a holder.
# A reader (a paged ``git log``, ``cat-file --batch``, a background ``fetch``) counts only while its
# fd is on the lock.
_READER_GIT = frozenset({
    "annotate", "blame", "cat-file", "check-attr", "check-ignore", "check-mailmap", "check-ref-format",
    "cherry", "count-objects", "describe", "diff", "diff-files", "diff-index", "diff-tree",
    "for-each-ref", "fsmonitor--daemon", "grep", "help", "log", "ls-files", "ls-remote", "ls-tree",
    "merge-base", "name-rev", "range-diff", "rev-list", "rev-parse", "shortlog", "show", "show-branch",
    "show-ref", "status", "var", "verify-commit", "verify-tag", "version", "whatchanged",
    "credential", "credential-cache", "credential-store",
    # Transfers write objects and refs, never the index.
    "fetch", "fetch-pack", "http-fetch", "index-pack", "pack-objects", "remote-http", "remote-https",
    "upload-pack",
})
# Global options that take the NEXT argument as their value (``git -C <dir> -c k=v commit``).
_GIT_VALUE_OPTIONS = frozenset({
    "-C", "-c", "--git-dir", "--work-tree", "--namespace", "--config-env", "--super-prefix", "--attr-source",
})
# What points a git at a repository other than its cwd.
_GIT_LOCATION_ENV = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR")


def _git_program(arg0: str) -> str | None:
    """``git`` or ``git-<sub>`` when ``arg0`` (an argv[0], a comm, a psutil name) runs git, else None.

    Any directory (``/usr/lib/git-core/git-commit``), either separator, ``.exe`` and case folded:
    Windows runs ``git.exe`` and ``git-commit.exe``."""
    name = re.split(r"[\\/]", arg0.strip())[-1].lower()
    name = name[:-4] if name.endswith(".exe") else name
    return name if name == "git" or name.startswith("git-") else None


def _git_argv(cmdline: bytes | list[str]) -> list[str]:
    if isinstance(cmdline, bytes):
        return [a.decode("utf-8", "replace") for a in cmdline.split(b"\0") if a]
    return [str(a) for a in cmdline if a]


def _git_subcommand_of(cmdline: bytes | list[str]) -> str | None:
    """The subcommand a git argv runs: ``commit`` for ``git -C x -c k=v commit`` and for a dashed
    ``/usr/lib/git-core/git-commit`` (``git-commit.exe``); None when argv[0] is not git or names none."""
    args = _git_argv(cmdline)
    program = _git_program(args[0]) if args else None
    if program is None:
        return None
    if program != "git":
        return program[len("git-"):]
    it = iter(args[1:])
    for arg in it:
        if arg in _GIT_VALUE_OPTIONS:
            next(it, None)
        elif not arg.startswith("-"):
            return arg
    return None


def _checkout_places(*paths: Path | str | None) -> tuple[str, ...]:
    """The checkout's own paths as given AND resolved, so another process's path need not be resolved."""
    places = {os.path.normcase(os.path.normpath(os.path.abspath(p))) for p in paths if p is not None}
    places |= {os.path.normcase(os.path.realpath(p)) for p in paths if p is not None}
    return tuple(sorted(places))


def _path_within(path: str, places: tuple[str, ...]) -> bool:
    """``path`` is one of ``places`` or under it, by PATH COMPONENTS (``/x/hermes-backup`` is not in ``/x/hermes``).

    Lexical: another process's cwd is already the kernel's resolved path, and resolving a stranger's
    path would touch whatever it names."""
    if not os.path.isabs(path):
        return False
    lexical = os.path.normcase(os.path.normpath(path))
    for place in places:
        try:
            if os.path.commonpath([lexical, place]) == place:
                return True
        except ValueError:  # another drive
            continue
    return False


def git_works_in(args: list[str], cwd: str | None, env: dict[str, str] | None, places: tuple[str, ...]) -> bool:
    """A git with this argv/cwd/environment works in ``places``: its cwd, a path argument (absolute,
    or relative to its cwd, ``--opt=<path>`` too) or a ``GIT_DIR``-style variable is inside one."""
    def inside(value: str) -> bool:
        if not value:
            return False
        if not os.path.isabs(value):
            if not cwd:
                return False
            value = os.path.join(cwd, value)
        return _path_within(value, places)

    if cwd and _path_within(cwd, places):
        return True
    for arg in args[1:]:
        if inside(arg.partition("=")[2] if arg.startswith("-") else arg):
            return True
    return any(inside((env or {}).get(key, "")) for key in _GIT_LOCATION_ENV)


def _could_write(uid: int | None, git_dir: Path) -> bool:
    """A process running as ``uid`` may write ``git_dir`` (root, its owner, or a group/world-writable dir)."""
    try:
        st = os.stat(git_dir)
    except OSError:
        return True
    return uid is None or uid in (0, st.st_uid) or bool(st.st_mode & 0o022)


def _held_open(path: Path, root: Path | None = None, *, any_git: bool = False) -> _Holder | bool | None:
    """Whether a running process may still own ``path`` (a git lock): the holder, False, or None (unknowable).

    The holder is a truthy :class:`_Holder` naming it (``pid 4242 (git commit)``). False is proof, not
    a guess: Linux reads every process's fds through /proc and, with ``root``, also counts any live git
    working in the checkout (cwd, a path argument or ``GIT_DIR``) that is not a pure reader
    (:data:`_READER_GIT`; ``any_git`` counts readers too), whatever form runs it: ``git commit``
    waiting in the editor has CLOSED its lock fd, and a dashed ``git-commit``, an alias or ``git.exe``
    is the same git. A git that could write the git dir but whose cwd or environment cannot be read
    makes the answer None. macOS/BSD ask ``lsof`` for open fds and ``ps`` for any such git (``ps``
    cannot say where it works, so any counts). Without either check the answer is None and the caller
    never deletes the lock. Windows needs no answer here: it refuses to unlink a file another process
    has open, so the caller's unlink is the probe.
    """
    if (Path("/proc") / "self" / "fd").is_dir():
        return _held_open_proc(path, root, any_git)
    return _held_open_lsof(path, root, any_git)


# Daemons a git spawns that outlive it, often with the checkout as cwd, and never take the index
# lock: even ``any_git`` ignores them, or one cached credential would keep a dead lock for hours.
_NEVER_LOCKS = frozenset({"credential-cache--daemon", "fsmonitor--daemon"})


def _counts_as_holder(args: list[str], any_git: bool) -> bool:
    sub = _git_subcommand_of(args)
    return sub not in _NEVER_LOCKS and (any_git or sub not in _READER_GIT)


def _held_open_proc(path: Path, root: Path | None, any_git: bool) -> _Holder | bool | None:
    proc = Path("/proc")
    target = os.path.realpath(path)
    places = _checkout_places(root, path.parent) if root is not None else None
    unknowable = False

    def name(args: list[str], pid_dir: Path) -> str:
        if args:
            return " ".join([os.path.basename(args[0]), *args[1:3]])
        try:
            return (pid_dir / "comm").read_bytes().decode("ascii", "replace").strip()  # /proc: Linux only
        except OSError:
            return "?"

    for pid_dir in proc.glob("[0-9]*"):
        try:
            if any(os.readlink(entry.path) == target for entry in os.scandir(pid_dir / "fd")):
                try:
                    args = _git_argv((pid_dir / "cmdline").read_bytes())
                except OSError:
                    args = []
                return _Holder(f"pid {pid_dir.name} ({name(args, pid_dir)})")
        except OSError:
            pass
        if places is None:
            continue
        try:
            uid = pid_dir.stat().st_uid
        except OSError:
            continue  # exited
        try:
            args = _git_argv((pid_dir / "cmdline").read_bytes())
            comm = (pid_dir / "comm").read_bytes().decode("ascii", "replace").strip()
        except OSError:
            if pid_dir.exists() and _could_write(uid, path.parent):
                unknowable = True  # hidden from us (hidepid) yet able to write here: it may be a git
            continue
        if not args:
            continue  # a zombie or a kernel thread holds nothing
        if _git_program(args[0]) is None and _git_program(comm) is None:
            continue
        if int(pid_dir.name) == os.getpid() or not _counts_as_holder(args, any_git):
            continue
        try:
            cwd = os.readlink(pid_dir / "cwd").removesuffix(" (deleted)")
            env = dict(entry.decode("utf-8", "replace").partition("=")[::2]
                       for entry in (pid_dir / "environ").read_bytes().split(b"\0") if entry)
        except OSError:
            if git_works_in(args, None, None, places):
                return _Holder(f"pid {pid_dir.name} ({name(args, pid_dir)})")
            if pid_dir.exists() and _could_write(uid, path.parent):
                unknowable = True  # a lock-keeping git we cannot place may be working here
            continue
        if git_works_in(args, cwd, env, places):
            return _Holder(f"pid {pid_dir.name} ({name(args, pid_dir)})")
    return None if unknowable else False


def _held_open_lsof(path: Path, root: Path | None, any_git: bool) -> _Holder | bool | None:
    import shutil

    lsof = shutil.which("lsof") or next((p for p in ("/usr/sbin/lsof", "/usr/bin/lsof") if os.path.isfile(p)), None)
    if lsof is None:
        return None
    try:
        found = subprocess.run([lsof, "-F", "pc", "--", str(path)], capture_output=True, text=True, encoding="utf-8",
                               errors="replace", timeout=20,
                               stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return None
    pids = [line[1:] for line in found.stdout.splitlines() if line.startswith("p")]
    if pids:
        names = [line[1:] for line in found.stdout.splitlines() if line.startswith("c")]
        return _Holder(f"pid {pids[0]} ({names[0] if names else '?'})")
    if found.returncode not in (0, 1):  # lsof exits 1 when nothing has the file open
        return None
    return False if root is None else _ps_git_holder(lsof, path.parent, root, any_git)


def _lsof_cwds(lsof: str, pids: list[str]) -> dict[str, str] | None:
    """Each pid's cwd from ``lsof -d cwd`` (macOS has no /proc); None when lsof cannot answer."""
    try:
        out = subprocess.run([lsof, "-a", "-d", "cwd", "-F", "pn", "-p", ",".join(pids)], capture_output=True,
                             text=True, encoding="utf-8", errors="replace", timeout=20, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode not in (0, 1):
        return None
    cwds: dict[str, str] = {}
    pid = None
    for line in out.stdout.splitlines():
        if line.startswith("p"):
            pid = line[1:]
        elif line.startswith("n") and pid is not None:
            cwds[pid] = line[1:]
    return cwds


def _still_running(pid: int) -> bool:
    """POSIX only (the lsof branch never runs on Windows); stdlib, as this launch-time repair must be."""
    try:
        os.kill(pid, 0)  # windows-footgun: ok — lsof/ps branch is macOS/BSD only, Windows never reaches it
    except ProcessLookupError:
        return False
    except OSError:  # EPERM: alive, another user's
        return True
    return True


def _ps_git_holder(lsof: str, git_dir: Path, root: Path, any_git: bool) -> _Holder | bool | None:
    """An empty ``lsof`` is no proof: ``git commit`` waiting in the editor has closed its lock fd.
    Every git (by executable name, dashed forms included) that could write ``git_dir``, is not a pure
    reader, and works in the checkout (its cwd from ``lsof -d cwd``, or a path argument) keeps the
    lock; one whose cwd cannot be read makes the answer None. Gits elsewhere on the machine do not
    count, or any commit open in another repository would block every launch-time repair."""
    import shutil

    # The launcher can run with a PATH that has neither git nor ps (a Windows-style install, a
    # stripped service env): ps lives at /bin/ps on macOS and every BSD.
    ps_bin = shutil.which("ps") or next((p for p in ("/bin/ps", "/usr/bin/ps") if os.path.isfile(p)), None)
    if ps_bin is None:
        return None

    def ps(*columns: str) -> list[list[str]] | None:
        try:
            out = subprocess.run([ps_bin, "-A", *(f"-o{c}=" for c in columns)], capture_output=True,
                                 text=True, encoding="utf-8", errors="replace",
                                 timeout=20, stdin=subprocess.DEVNULL)
        except (OSError, subprocess.SubprocessError):
            return None
        if out.returncode != 0:
            return None
        return [line.split(None, len(columns) - 1) for line in out.stdout.splitlines() if line.strip()]

    named = ps("pid", "uid", "comm")  # comm: the executable (``/usr/libexec/git-core/git-commit``)
    commands = ps("pid", "command")
    if named is None or commands is None:
        return None
    argv = {row[0]: row[1].split() for row in commands if len(row) == 2}
    candidates: dict[str, list[str]] = {}
    for row in named:
        if len(row) != 3 or not (row[0].isdigit() and row[1].isdigit()) or int(row[0]) == os.getpid():
            continue
        if _git_program(row[2]) is None and not (argv.get(row[0]) and _git_program(argv[row[0]][0])):
            continue
        args = argv.get(row[0]) or [row[2]]
        if _git_program(args[0]) is None:
            args = [row[2]]  # an argv[0] with spaces: no subcommand, so it counts
        if _could_write(int(row[1]), git_dir) and _counts_as_holder(args, any_git):
            candidates[row[0]] = args
    if not candidates:
        return False
    cwds = _lsof_cwds(lsof, sorted(candidates))
    if cwds is None:
        return None
    places = _checkout_places(root, git_dir)
    for pid, args in candidates.items():
        cwd = cwds.get(pid)
        if cwd is None and _still_running(int(pid)):
            return None  # a lock-keeping git we cannot place may be working here
        if git_works_in(args, cwd, None, places):
            return _Holder(f"pid {pid} ({' '.join([os.path.basename(args[0]), *args[1:3]])})")
    return False


def _index_lock_holder(git_dir: Path, root: Path | None) -> str:
    """Who keeps ``index.lock`` for the user-facing message: ``pid N (name)`` when it can be named."""
    held = None if sys.platform == "win32" else _held_open(git_dir / "index.lock", root)
    return f"a running git ({held})" if isinstance(held, _Holder) else "a running git"


def _release_dead_index_lock(git_dir: Path, root: Path | None = None, *, any_git: bool = False) -> bool:
    """Drop a killed git's ``index.lock`` (it refuses every git command) once its owner is PROVEN gone.

    False while a live git may hold it, or when this platform cannot prove it dead: the caller then
    keeps the interrupted-pull marker, so the next launch tries again instead of a rollback being lost.
    ``any_git``: every git working in ``root`` keeps it, readers included (:func:`_held_open`).
    """
    lock = git_dir / "index.lock"
    deadline = time.monotonic() + 5
    while lock.exists():
        held = None if sys.platform == "win32" else _held_open(lock, root, any_git=any_git)
        if held is False or sys.platform == "win32":
            try:
                lock.unlink()
                return True
            except FileNotFoundError:
                return True
            except PermissionError:  # Windows: open in a live process
                pass
        elif held is None:
            return False
        if time.monotonic() > deadline:
            return False
        time.sleep(0.1)
    return True


# The ZIP update's equivalent of the interrupted-pull marker (``_early_recovery_zip``): a journal in the
# install root naming every entry the swap stages/renames. Its name stays here: the launch fast path
# stats it without importing the ZIP code.
ZIP_SWAP_JOURNAL = ".hermes-update-zip-swap"


def write_durable_text(path: Path, text: str) -> None:
    """``text`` at ``path`` as one durable record, never a half-written one.

    The temp is an unpredictable name created exclusively (no-follow) beside ``path``: a pre-existing
    name there is never written through (its symlink or hardlink would carry the record onto another
    file) and never deleted (it may be a user's file: review Z5). fsync before the rename: an empty
    record over a mixed tree is no record.
    """
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    while True:
        tmp = path.with_name(f"{path.name}.{os.urandom(6).hex()}.tmp")
        try:
            fd = os.open(tmp, flags, 0o644)
            break
        except FileExistsError:
            continue  # 48 random bits taken: draw again, never reuse the name
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def restore_interrupted_pull(project_root: Path | None = None, *, after_failure: bool = False) -> bool:
    """Put back the files a killed ``hermes update`` had half-moved to the new commit.

    ``after_failure``: the updater itself calls this when git exited non-zero mid-move (a locked or
    read-only file): same restore, and the marker stays whenever the tree is not verified whole.

    Returns True when the tree changed under this process: modules it already imported may be the
    half-written ones, so the caller must relaunch (``relaunch_after_restore``).

    Fast path (no marker) is one or two ``stat`` calls. Acts only when the marker's owner is gone,
    HEAD is still the pre-pull commit and no merge/rebase is in progress; then every path git wrote
    (the target's content, or torn on the way there) returns to HEAD (the commit the venv was built
    for), so the install is whole again and ``hermes update`` redoes the update from the start. Local
    edits are never touched; the updater's autostash (if any) stays in ``git stash list``. Concurrent
    launches take turns (``_restore_claim``); a launch that waited out another's restore relaunches.

    A torn ``hermes_bootstrap.py``, ``hermes_cli/__init__.py`` or recovery/lock/custody module fails
    before (or inside) the checkout's copy of this code: a minted launcher then runs the copy the
    updater published beside the marker (``RECOVERY_CLOSURE``). Limits, by design: other entry points
    (``python -m``, the ``hermes-agent`` hook's ``agent/__init__.py``) have no such fallback, and a
    launcher minted before that fallback existed has none either. A file git also changes that the user deleted, emptied or cut to a prefix of git's version
    looks exactly like git's own half-written file and is restored too, as is a user edit to a
    conflicted path or, on git < 2.38, to a path both sides of a custom-branch merge changed.
    """
    try:
        root = _project_root() if project_root is None else project_root
        marker = interrupted_pull_marker(root)
        if not marker.is_file() and not (Path(root) / ZIP_SWAP_JOURNAL).is_file():
            return False  # fast path: nothing to repair, no lock taken
        if _pytest_owns_live_checkout(root):
            return False
        # A live `hermes update` (or its build/completion/git, after its updater died) owns the
        # checkout: its own transaction settles the tree, and repairing under it races its git.
        busy_note = "⚠ Not repairing the checkout now: {}. Launch again once it finishes."
        if (Path(root) / ZIP_SWAP_JOURNAL).is_file():
            with _checkout_custody(Path(root)) as busy:
                if busy:
                    print(busy_note.format(busy), file=sys.stderr)
                    return False
                from hermes_cli._early_recovery_zip import restore_interrupted_zip_swap

                if restore_interrupted_zip_swap(root):
                    return True
        if not marker.is_file():
            return False
        with _restore_claim(marker.parent) as claimed:
            if not claimed:
                if after_failure:
                    return False  # the marker stays: the updater reports the move as not settled
                # The tree is still torn (the marker is there): importing checkout code now would run
                # the half-written files. Fail closed like unprovable custody does, as one line and
                # exit 1 (SystemExit's message), never a traceback (F20).
                raise SystemExit("hermes: another Hermes launch is finishing an interrupted `hermes update`; "
                                 "launch again in a moment.")
            if not marker.is_file():
                return True  # another launch finished while this one started: rerun from its tree
            # The claim orders launches; the checkout lock keeps out an update tree. Claim first, so a
            # launch that waited out another's repair reruns from its tree without contending.
            with _checkout_custody(Path(root)) as busy:
                if busy:
                    print(busy_note.format(busy), file=sys.stderr)
                    return False
                return _restore_holding_claim(root, marker, after_failure=after_failure)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        # Never block launch: the import that follows surfaces any real breakage.
        print(f"⚠ Could not check for an interrupted `hermes update`: {exc}", file=sys.stderr)
    return False


@contextlib.contextmanager
def _checkout_custody(root: Path):
    """Hold the checkout kernel lock (``hermes_cli.update_lock``) for a repair, like an updater (R2).

    Yields ``""`` while this process holds or joined it (the updater's own ``after_failure``
    settle and its children join), else the busy reason: a live update tree owns the checkout and
    the repair leaves the tree alone. The restore's mutating git children inherit the lock (POSIX)
    or join this process's kill-on-close job (Windows) through ``update_custody.run_git``, so a
    launch killed mid-restore keeps the checkout locked until its git is gone.

    A torn lock module cannot prove custody: an already-running updater retains its imported
    copy and may still be writing this checkout. Refuse launch rather than race that writer.
    """
    try:
        from hermes_cli import update_lock
    except Exception as exc:
        raise RuntimeError(
            "Cannot safely repair the interrupted update: checkout custody is unavailable "
            f"({type(exc).__name__}: {exc}). The recovery marker was kept. "
            "Wait for any running update to finish; if this persists, repair the checkout "
            "before launching Hermes again."
        ) from exc
    holder = update_lock._acquire_checkout(Path(root))
    if holder is not None and not update_lock.checkout_lock_held(Path(root)):
        # Refused, then free by the probe: its holder just exited. Take it now rather than run the
        # repair unguarded in that window (m10).
        holder = update_lock._acquire_checkout(Path(root))
    if holder is not None:
        if not update_lock.checkout_lock_held(Path(root)):
            # Not a holder but no lock at all (a git dir without working locks, e.g. NFS without
            # lockd): no updater can hold it either (they fail closed), so the repair runs unguarded.
            yield ""
            return
        yield update_lock.describe_holder(holder) or "another update holds the checkout lock"
        return
    try:
        yield ""
    finally:
        update_lock._release_checkout()


def _claim_owner_alive(fields: dict[str, str], marker: Path) -> bool:
    """The updater that wrote ``marker`` is still running and the marker is fresh: hands off."""
    try:
        owner = int(fields.get("pid", ""))
    except ValueError:
        owner = -1
    # Our own pid is never the owner: this runs at startup, and containers hand a retry the
    # killed updater's pid.
    return (owner != os.getpid() and _pid_is_running(owner)
            and time.time() - marker.stat().st_mtime < _INTERRUPTED_PULL_MAX_AGE_SECONDS)


def _resume_killed_rollback(git, fields: dict[str, str], git_dir: Path, root: Path,
                            pre: str, rollback: str, *, after_failure: bool) -> str | None:
    """Redo a killed syntax rollback's HEAD move to ``pre``; the new HEAD, or None (marker kept)."""
    # Only the ref the rollback left HEAD on may be rewound: a branch the user checked out at
    # ``target`` since is theirs (and the update's branch would stay on the broken commit).
    named = git("symbolic-ref", "-q", "HEAD")
    if "ref" not in fields or (named.stdout.strip() if named.returncode == 0 else "") != fields["ref"].strip():
        # The marker stays: switching back lets the next launch resume. Never a reset recipe (F73):
        # a reset --hard would wipe whatever the user changed on the branch they checked out.
        ref = fields.get("ref", "").strip().removeprefix("refs/heads/")
        back = (f"`git -C {root} checkout {ref}`" if ref
                else f"`git -C {root} checkout --detach {fields.get('target', '').strip()[:10] or pre[:10]}`")
        print(f"⚠ An interrupted `hermes update` rollback to {pre[:10]} was not resumed: HEAD now names "
              f"another branch. Switch back with {back}, then launch `hermes` again to finish it.",
              file=sys.stderr)
        return None
    # A syntax rollback killed before it moved HEAD back: redo that step (HEAD and index, no file),
    # so the restore lands on ``pre`` (the code the update started from), not the broken tree.
    # Every step is checked, and the killed step's ``index.lock`` goes first, only once its git is
    # proven gone: until HEAD is on ``pre`` this marker is the rollback's only record.
    reason = _redo_rollback_head(git, git_dir, root, pre, rollback, after_failure=after_failure)
    if reason is not None:
        print(f"⚠ An interrupted `hermes update` rollback to {pre[:10]} cannot resume yet ({reason}); "
              "the next launch retries.", file=sys.stderr)
        return None
    return _read_head(git)


def _advise_killed_merge(git_dir: Path, root: Path, target: str, stash: str) -> None:
    """Once per process, tell the user to abort the killed updater's own unfinished merge."""
    global _merge_advice_shown
    merge_head = git_dir / "MERGE_HEAD"
    if (not _merge_advice_shown and merge_head.is_file()
            and merge_head.read_text(encoding="utf-8-sig").strip() == target):
        # The killed updater's own merge: its conflict markers may sit in startup modules.
        _merge_advice_shown = True
        print(f"⚠ A killed `hermes update` left its merge unfinished. Run `git -C {root} merge --abort`, "
              "then launch again." + (f" Your local changes are in its stash ({stash})." if stash else ""),
              file=sys.stderr)


def _custody_git(root: Path, recorded: str):
    """The restore's ``git(*args)``: the binary the updater recorded, every run under custody (``run_git``)."""
    executable = _git_executable(recorded)
    try:
        from hermes_cli.update_custody import run_git
    except Exception as exc:
        raise RuntimeError(
            "Cannot safely repair the interrupted update: child custody is unavailable "
            f"({type(exc).__name__}: {exc}). The recovery marker was kept. "
            "Repair the checkout before launching Hermes again."
        ) from exc

    def git(*args: str, stdin: str | None = None, text: bool = True) -> subprocess.CompletedProcess:
        base = [executable, "--literal-pathspecs", "-C", str(root)]
        kwargs = dict(input=stdin, cwd=str(root), capture_output=True, timeout=120,
                      stdin=None if stdin is not None else subprocess.DEVNULL,
                      **({"text": True, "encoding": "utf-8", "errors": "replace"} if text else {}))
        return run_git(base, list(args), **kwargs)

    return git


def _judge_index_lock(fields: dict[str, str], marker: Path, root: Path, *,
                      after_failure: bool) -> tuple[bool, bool, bool]:
    """``(foreign_lock, known_foreign, index_free)`` for the git dir's ``index.lock``, before any git runs.

    A killed git's lock goes first (an unresolvable or failing git would otherwise strand it, and it
    refuses every later git command). Proven-dead only; after a git that EXITED (``after_failure``) a
    lock now is another git's, never ours to drop."""
    git_dir = marker.parent
    lock = git_dir / "index.lock"
    foreign_lock = after_failure and lock.exists()
    if foreign_lock and _lock_predates_move(fields, lock):
        # Only the generation that was there before our git ran is another git's for sure. A lock
        # that appeared during the move can be our own SIGKILLed git's: never judged here (this
        # process cannot tell), but a later launch reclaims it once no git holds it (C15).
        _remember_foreign_lock(marker, lock)
    # The lock generation judged foreign when our git exited is never a killed git's to reclaim later:
    # a live `git commit` in the editor holds it with its fd closed, which only Linux can still see.
    known_foreign = bool(fields.get("foreign_lock", "").strip()) and \
        fields["foreign_lock"].strip() == _lock_identity(lock)
    index_free = foreign_lock or (not known_foreign and _release_dead_index_lock(git_dir, root))
    return foreign_lock, known_foreign, index_free


def _read_head(git) -> str | None:
    # Only a full object name is an answer: an empty or garbled one keeps the marker (m5).
    head = git("rev-parse", "HEAD")
    oid = head.stdout.strip() if head.returncode == 0 else ""
    if _OID.fullmatch(oid):
        return oid
    detail = head.stderr.strip() or f"git printed {oid!r}, exit {head.returncode}"
    print(f"⚠ Could not read HEAD to repair an interrupted `hermes update` ({detail}); "
          "the next launch retries.", file=sys.stderr)
    return None


def _retire_unattributable(git, git_dir: Path, root: Path, marker: Path, pre: str, target: str) -> None:
    """``target`` is gone (a gc or re-clone): nothing left to compare the tree against.

    Unattributable bytes are not proof of a whole tree: only a clean tracked tree at ``pre`` retires
    the record, rollback or not (review G2)."""
    if not _rollback_verified(git, git_dir, pre):
        print(f"⚠ An interrupted `hermes update` left tracked files that differ from {pre[:10]} and commit "
              f"{target[:10]} is gone; the marker was kept. Inspect `git -C {root} status`, then "
              f"`git -C {root} checkout {pre[:10]} -- <file>` for each file the update wrote.", file=sys.stderr)
        return
    marker.unlink()
    print(f"⚠ Ignoring a stale interrupted-update marker: commit {target[:10]} is gone.", file=sys.stderr)


def _restore_holding_claim(root: Path, marker: Path, *, after_failure: bool = False) -> bool:
    git_dir = marker.parent
    fields = dict(line.partition("=")[::2] for line in marker.read_text(encoding="utf-8-sig").splitlines())
    if _claim_owner_alive(fields, marker):
        return False
    pre, target = fields.get("pre", "").strip(), fields.get("target", "").strip()
    stash = fields.get("stash", "").strip()
    git = _custody_git(root, fields.get("git", "").strip())
    foreign_lock, known_foreign, index_free = _judge_index_lock(fields, marker, root, after_failure=after_failure)
    rollback = fields.get("rollback", "").strip()
    rollback = rollback if rollback in ("branch", "detach") else ""

    head = _read_head(git)
    if head is None:
        return False
    if rollback and pre and target and head == target:
        head = _resume_killed_rollback(git, fields, git_dir, root, pre, rollback,
                                       after_failure=after_failure or known_foreign)
        if head is None:
            return False
    if not pre or not target or head != pre:
        marker.unlink()  # git finished (HEAD moved) or the marker is unusable
        return False
    if any((git_dir / name).exists() for name in _GIT_OPERATION_IN_PROGRESS):
        _advise_killed_merge(git_dir, root, target, stash)
        return False
    # A killed claim holder's own git child can still be writing; scanning under it reads half a tree.
    # After a git that EXITED (``after_failure``) no git of ours is left: a lock now is another git's
    # (often the very reason ours failed), never ours to drop. The scan below only reads.
    if not index_free:
        print(f"⚠ {_index_lock_holder(git_dir, root)} holds the index after an interrupted `hermes update`; "
              "the next launch finishes the restore.", file=sys.stderr)
        return False
    written = _paths_git_wrote(git, root, pre, target)
    if written is None:
        _retire_unattributable(git, git_dir, root, marker, pre, target)
        return False
    return _put_back_written(git, root, marker, written, pre, stash, rollback,
                             foreign_lock=foreign_lock, after_failure=after_failure)


def _put_back_written(git, root: Path, marker: Path, written, pre: str, stash: str, rollback: str, *,
                      foreign_lock: bool, after_failure: bool) -> bool:
    """Return every path git wrote (``_paths_git_wrote``) to ``pre`` and retire the marker once verified."""
    restore, added, new_dirs, kept = written
    if (restore or added) and foreign_lock:
        print("⚠ Another git holds the index, so the files the update already wrote cannot be put back "
              "now; the next launch restores them.", file=sys.stderr)
        return False
    if restore or added:
        print(("⚠ git stopped partway through writing the new code — " if after_failure else
               "⚠ A previous `hermes update` was killed while git was writing the new code — ")
              + f"restoring the checkout to {pre[:10]}...", file=sys.stderr)
        failed = _put_back_paths(git, root, restore, added, kept)
        if failed:
            # No manual recipe: a reset would also wipe the edits this restore keeps, and the
            # marker stays so the next launch retries.
            reason = (failed.stderr.strip().splitlines() or ["git failed"])[-1]
            print(f"  ✗ Could not restore it automatically ({reason}); the next launch retries.",
                  file=sys.stderr)
            return False
    for rel in sorted(new_dirs, key=lambda d: d.count("/"), reverse=True):
        with contextlib.suppress(OSError):
            (root / rel).rmdir()  # only when empty: an untracked file inside keeps it
    if rollback and not _rollback_verified(git, marker.parent, pre, owned=set(restore)):
        # Only HEAD on ``pre`` with every path the update wrote back at ``pre`` is a finished rollback;
        # anything less keeps its only record. Other tracked edits (made after the kill, or autocrlf /
        # filemode noise) are not the rollback's to judge, and no reset is advised: it would wipe them.
        print(f"  ✗ The rollback to {pre[:10]} is not verified yet (files the update wrote still differ); "
              f"the next launch retries. Inspect `git -C {root} status`.", file=sys.stderr)
        return bool(restore or added)
    marker.unlink()
    if not restore and not added:
        return False  # the killed git never reached the tree: nothing to put back
    print("  ✓ Checkout restored; `hermes update` updates it again.", file=sys.stderr)
    if stash:
        print(f"  Your local changes are still in the update's stash ({stash}).", file=sys.stderr)
    return True


def _lock_identity(lock: Path) -> str:
    """``inode:mtime_ns`` of a lock file, "" when there is none: a recreated lock is a new generation."""
    try:
        st = lock.stat()
    except OSError:
        return ""
    return f"{st.st_ino}:{st.st_mtime_ns}"


def _lock_predates_move(fields: dict[str, str], lock: Path) -> bool:
    """``lock`` is the generation ``arm_tree_move`` saw before the move's git ran. A marker without
    the record (an older updater's, hand-written) cannot say: every lock counts as foreign."""
    if "index_lock" not in fields:
        return True
    before = fields["index_lock"].strip()
    return bool(before) and before == _lock_identity(lock)


def _remember_foreign_lock(marker: Path, lock: Path) -> None:
    """Record ``lock``'s identity in the marker (atomically: the marker is the restore's only record)."""
    identity = _lock_identity(lock)
    if not identity:
        return
    lines = [line for line in marker.read_text(encoding="utf-8-sig").splitlines()
             if not line.startswith("foreign_lock=")]
    write_durable_text(marker, "\n".join([*lines, f"foreign_lock={identity}"]) + "\n")


def _redo_rollback_head(git, git_dir: Path, root: Path, pre: str, mode: str, *, after_failure: bool) -> str | None:
    """Move HEAD (and the index) back to ``pre`` for a killed syntax rollback; None, or why not yet.

    The killed step's ``index.lock`` is reclaimed first, and only when its git is proven gone. After a
    git that exited in this very process (``after_failure``) a lock is never ours to judge.
    """
    lock = git_dir / "index.lock"
    if lock.exists() and (after_failure or not _release_dead_index_lock(git_dir, root)):
        return f"{_index_lock_holder(git_dir, root)} may still hold .git/index.lock"
    if mode == "detach":
        moved = git("update-ref", "--no-deref", "HEAD", pre)
        if moved.returncode != 0:
            return f"git update-ref failed: {(moved.stderr.strip().splitlines() or ['?'])[-1]}"
    reset = git("reset", "-q", pre)
    if reset.returncode != 0:
        return f"git reset failed: {(reset.stderr.strip().splitlines() or ['?'])[-1]}"
    head = git("rev-parse", "HEAD")
    if head.returncode != 0 or head.stdout.strip() != pre:
        return "HEAD did not move back"
    return None


def _rollback_verified(git, git_dir: Path, pre: str, owned: set[str] | None = None) -> bool:
    """The rollback landed: HEAD is ``pre``, no index lock, and no tracked change among ``owned`` (the
    paths the update wrote), or anywhere when they are unknown (``target`` gone)."""
    head = git("rev-parse", "HEAD")
    if head.returncode != 0 or head.stdout.strip() != pre or (git_dir / "index.lock").exists():
        return False
    status = git("status", "--porcelain", "-z", "--untracked-files=no", "--no-renames")
    changed = {entry[3:] for entry in status.stdout.split("\0") if entry}
    return status.returncode == 0 and not (changed if owned is None else changed & owned)


def _put_back_paths(git, root: Path, restore: list[str], added: list[str],
                    kept: set[str]) -> subprocess.CompletedProcess | None:
    """Return ``restore`` to HEAD and drop ``added``, renaming those in ``kept`` aside; the failed git run, or None."""
    if restore:
        run = git("restore", "--source=HEAD", "--staged", "--worktree", "--pathspec-from-file=-",
                  "--pathspec-file-nul", stdin="\0".join(restore))
        if run.returncode:
            return run
    if added:
        run = git("rm", "-q", "--cached", "--ignore-unmatch", "--pathspec-from-file=-",
                  "--pathspec-file-nul", stdin="\0".join(added))
        for rel in added:
            if rel in kept:
                print(f"  Kept {rel} as {_keep_aside(root / rel).name}: it may be your own file.", file=sys.stderr)
            else:
                (root / rel).unlink(missing_ok=True)
        if run.returncode:
            return run
    return None


def _keep_aside(file: Path) -> Path:
    """Rename ``file`` to a free ``<name>.hermes-update-kept[-N]`` beside it (never importable, never clobbered)."""
    n = 1
    while os.path.lexists(aside := file.with_name(f"{file.name}.hermes-update-kept" + (f"-{n}" if n > 1 else ""))):
        n += 1
    os.rename(file, aside)
    return aside


def relaunch_after_restore() -> None:
    """Re-run this command from the restored tree; never returns.

    Everything imported so far (this package, ``hermes_bootstrap``, ``hermes_cli.main`` itself) may
    be the killed git's new files, and they would run against the restored old tree.
    """
    argv = [sys.executable, *sys.orig_argv[1:]]
    sys.stdout.flush()
    sys.stderr.flush()
    if sys.platform == "win32":
        # os.execv on Windows spawns and exits, detaching the console's wait on us.
        sys.exit(subprocess.call(argv))  # windows-footgun: ok — interactive child keeps our console
    os.execv(sys.executable, argv)


def _pytest_owns_live_checkout(root: Path) -> bool:
    """Keep lifecycle tests from repairing the checkout that runs the suite.

    Copied installations in tmp_path remain eligible for real recovery tests.
    """
    return "PYTEST_CURRENT_TEST" in os.environ and root == Path(__file__).resolve().parent.parent


def _missing_environment(root: Path) -> bool:
    """True when PM recorded an environment whose site-packages is gone."""
    from pm.environments import runtime_facts_path, selected_venv, site_packages

    if not runtime_facts_path(root).is_file():
        return False
    try:
        return not site_packages(selected_venv(root)).is_dir()
    except RuntimeError:
        return True  # PM validates the recorded state before rebuilding.


def _count_failed_attempt(marker: Path) -> None:
    """Bump ``marker``'s attempt count in its own format so the retry limit can trip."""
    import json

    # Best-effort: an unwritable marker only means the limit trips later.
    with contextlib.suppress(OSError):
        attempts = _read_marker_attempts(marker) + 1
        # Our own record: undecodable bytes must not raise UnicodeDecodeError on every launch.
        body = marker.read_text(encoding="utf-8-sig", errors="replace")
        if any(line.startswith("pid=") for line in body.splitlines()):
            lines = [line for line in body.splitlines() if not line.startswith("attempts=")]
            body = "\n".join([*lines, f"attempts={attempts}"]) + "\n"
        else:
            body = json.dumps({"attempts": attempts})
        marker.write_text(body, encoding="utf-8")


def recover_if_needed(project_root: Path | None = None, argv: list[str] | None = None, *, explicit: bool = False) -> bool:
    """Ask PM to restore dependencies before activation; leave failed requests retryable."""
    global _UPDATE_RETRY_RECOVERED

    root = _project_root() if project_root is None else Path(project_root).resolve()
    if not explicit and _pytest_owns_live_checkout(root):
        return False
    from hermes_cli._parser import command_argv

    args = command_argv(sys.argv[1:] if argv is None else argv)
    if not explicit and args[:1] == ["pm"]:
        return False  # PM's command boundary owns the explicit repair.
    from pm.environments import install_state_dir

    missing_marker = install_state_dir(root) / ".repair-incomplete"
    marker_paths = (root / ".update-incomplete", root / ".lazy-refresh-incomplete", missing_marker)
    markers = [path for path in marker_paths if path.is_file()]

    if not (root / "pyproject.toml").is_file() or not (explicit or markers or _missing_environment(root)):
        return False
    lock = _claim_recovery_lock(root)
    if lock is None:
        return False
    try:
        # Recheck after locking: another launch can finish between discovery and claim.
        markers = [path for path in marker_paths if path.is_file()]
        if not markers:
            if not explicit and not _missing_environment(root):
                return False
            missing_marker.write_text('{"attempts": 0}', encoding="utf-8")
            markers = [missing_marker]
        if any(_marker_owner_is_live(marker) for marker in markers):
            return False
        if not explicit and any(_read_marker_attempts(marker) >= _EARLY_CORE_INSTALL_MAX_ATTEMPTS for marker in markers):
            print("hermes: automatic dependency repair retry limit reached; run `hermes pm repair`", file=sys.stderr)
            return False
        from pm.recovery import repair_dependencies

        print("hermes: repairing the recorded dependency environment...", file=sys.stderr)
        repair_dependencies(root)
        for marker in markers:
            marker.unlink(missing_ok=True)
        _UPDATE_RETRY_RECOVERED = args[:1] == ["update"]
        print("hermes: dependency environment repaired", file=sys.stderr)
        return True
    except Exception as exc:
        from pm.environments import install_state_permission_message

        if isinstance(exc, PermissionError) and install_state_permission_message(root, exc):
            raise  # The bootstrap or PM CLI reports the access error once.
        for marker in markers:
            _count_failed_attempt(marker)
        print(f"hermes: dependency repair failed: {exc}; run `hermes pm repair`", file=sys.stderr)
        return False
    finally:
        os.close(lock)


# A failed network or build must not retry on every launch forever.
_EARLY_CORE_INSTALL_MAX_ATTEMPTS = 3


def _claim_recovery_lock(root: Path) -> int | None:
    """Hold a kernel lock in writable state; process exit releases it."""
    from pm.environments import install_state_dir
    from hermes_cli.runtime_state import _lock

    state = install_state_dir(root)
    state.mkdir(parents=True, exist_ok=True)
    fd = os.open(state / ".recovery.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        if _lock(fd, wait=False):
            return fd
    except BaseException:
        os.close(fd)
        raise
    os.close(fd)
    return None

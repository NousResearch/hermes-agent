"""Dependency-light venv recovery that runs BEFORE hermes_cli.main's imports.

Deliberately **stdlib-only** so importing it can never fail on a corrupted venv. ``hermes_cli.main``
calls :func:`recover_if_needed` at the very top of its module body, before any third-party import.
Scope: repair only enough for ``hermes_cli.main`` to become importable again (force-reinstall of
the known-fragile core packages, using the pins from pyproject.toml).
"""

from __future__ import annotations

import contextlib
import errno
import importlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath

# Core packages a failed lazy ``uv pip install`` is known to leave with intact distribution
# metadata but wiped import files. ``module`` is probed via a real import; ``attr`` guards against
# an empty/stub module. main.py's marker-recovery path reuses these tables — keep them here so
# both layers probe and repair the same set.
# See #57828.
LAZY_REFRESH_IMPORT_PROBES: tuple[tuple[str, str], ...] = (
    ("yaml", "SafeDumper"), ("dotenv", "load_dotenv"), ("click", "Command"),
    ("certifi", "contents"), ("rich", "print"), ("cryptography", "__version__"),
    ("jwt", "encode"),
)

LAZY_REFRESH_REPAIR_PACKAGES: dict[str, str] = {
    "yaml": "PyYAML", "dotenv": "python-dotenv", "click": "click", "certifi": "certifi",
    "rich": "rich", "cryptography": "cryptography", "jwt": "PyJWT",
}

# ``hermes update`` renames the live ``hermes*.exe`` shims aside (``hermes.exe.old.<unix-ms>``) so
# uv can write replacements. Putting them BACK is the safety-critical direction: losing that rename
# leaves no ``hermes`` on PATH, and the command that would repair it IS ``hermes update``. The
# updater, the early-recovery installer and the startup orphan sweep all restore through this one
# stdlib-only helper so the retry ladder and the recovery wording cannot drift apart again.
# --- Windows entry-point shim quarantine ----------------------------------- They used to be separate
# one-shot renames with swallowed errors; the two that had messages had already drifted apart. The logic
# lives here, in the one stdlib-only module all of them can import, so the ladder and the recovery wording
# stay in lockstep. See #75584.
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


def _load_pyproject_project(root: Path) -> dict | None:
    """``[project]`` table of ``root/pyproject.toml``; ``None`` when missing/unreadable."""
    pyproject = root / "pyproject.toml"
    if not pyproject.is_file():
        return None
    try:
        import tomllib

        with open(pyproject, "rb") as f:
            project = tomllib.load(f).get("project", {})
    except Exception:
        return None
    return project if isinstance(project, dict) else None


def _read_marker_attempts(marker_path: Path) -> int:
    """Attempt counter from a marker's opportunistic JSON body; corrupt/missing → 0."""
    try:
        raw = marker_path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return 0
    if not raw:
        return 0
    try:
        import json

        return int(json.loads(raw).get("attempts", 0))
    except (ValueError, AttributeError):
        return 0


def _run_ensurepip(root: Path) -> None:
    """Best-effort pip bootstrap — a killed install can leave the venv with no pip module at all."""
    try:
        subprocess.run([sys.executable, "-m", "ensurepip", "--upgrade", "--default-pip"],
                       cwd=root, capture_output=True)
    except Exception:
        pass


def _pid_is_running(pid: int) -> bool:
    """Best-effort stdlib-only process liveness probe.

    ``os.kill(pid, 0)`` is not a no-op on Windows, so use the Win32 process handle API there. An
    access-denied result counts as live: racing an elevated updater is worse than postponing
    recovery for one launch.
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
    return True


def _marker_owner_is_live(marker: Path) -> bool:
    """True when a legacy update marker names a process still running."""
    try:
        body = marker.read_text(encoding="utf-8", errors="replace")
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


def _git_dir(root: Path) -> Path:
    """``root``'s git dir: ``.git`` itself, or where a linked worktree's ``.git`` file points."""
    dot_git = root / ".git"
    if dot_git.is_file():
        text = dot_git.read_text(encoding="utf-8").strip()
        if text.startswith("gitdir:"):
            return root / text[len("gitdir:"):].strip()
    return dot_git


def interrupted_pull_marker(root: Path) -> Path:
    return _git_dir(root) / INTERRUPTED_PULL_MARKER


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


def _paths_git_wrote(git, root: Path, pre: str, target: str) -> tuple[list[str], list[str], set[str]] | None:
    """Paths the killed git already touched on the way to ``target``: (restore from HEAD, delete as added,
    directories git may have created for its added files).

    Git rewrites a file as unlink, create, write, so a kill leaves it missing, empty or cut short:
    all of those count as git's, like the full new blob. Content that matches neither side and is not
    the start of a new blob is the user's own edit (e.g. a re-applied stash) and is left alone.
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
            if (old_mode if status == "D" else new_mode) not in _REGULAR_FILE_MODES:
                continue
            entry = entries.setdefault(path, (old_mode, None if status == "A" else old_blob, []))
            entry[2].append((new_mode, None if status == "D" else new_blob))
    worktree_blob = _hash_worktree(git, [path for path in entries if (root / path).is_file()])
    restore, added = [], []
    for path, (old_mode, old_blob, new) in entries.items():
        file, blobs = root / path, {blob for _mode, blob in new if blob}
        if path not in worktree_blob:
            written = old_blob is not None  # unlinked (or deleted), not yet recreated
        elif worktree_blob[path] == old_blob:  # only a mode change tells whether git got here
            written = (sys.platform != "win32" and any(b == old_blob and m != old_mode for m, b in new)
                       and bool(file.stat().st_mode & 0o100) != (old_mode == "100755"))
        elif worktree_blob[path] in blobs or path in unknown:
            written = True
        else:  # git's own file cut short starts one of the new blobs
            content = file.read_bytes()
            written = any(subprocess.run(["git", "-C", str(root), "cat-file", "--filters", f"--path={path}", blob],
                                         capture_output=True, check=True, timeout=120,
                                         stdin=subprocess.DEVNULL).stdout.startswith(content) for blob in blobs)
        if written:
            (added if old_blob is None else restore).append(path)
    new_dirs = {str(parent) for path, (_m, old_blob, _n) in entries.items() if old_blob is None
                for parent in PurePosixPath(path).parents if parent.parts}
    if new_dirs:
        new_dirs -= set(git("ls-tree", "-r", "-d", "--name-only", "-z", pre).stdout.split("\0"))
    return restore, added, new_dirs


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


def _held_open(path: Path) -> bool:
    """True when a running process has ``path`` open, the usual sign of a live git owning its lock.

    Best effort, not exact: Linux answers through /proc, but a live git that owns ``index.lock`` without
    an open fd (``commit`` waiting in the editor, or between closing the lock and renaming it) reads as
    dead; Windows refuses to unlink a file another process has open, so the caller's unlink is its probe;
    macOS/BSD have no portable check at all. The claim only orders Hermes launches, so on those paths a
    live git's lock can be removed; its command then fails and the marker stays for the next launch.
    """
    proc = Path("/proc")
    if not (proc / "self" / "fd").is_dir():
        return False
    target = os.path.realpath(path)
    for fd_dir in proc.glob("[0-9]*/fd"):
        try:
            if any(os.readlink(entry.path) == target for entry in os.scandir(fd_dir)):
                return True
        except OSError:
            continue
    return False


def _release_dead_index_lock(git_dir: Path) -> bool:
    """Drop the killed git's ``index.lock`` (it refuses every git command); False while a live git holds it."""
    lock = git_dir / "index.lock"
    deadline = time.monotonic() + 5
    while lock.exists():
        if not _held_open(lock):
            try:
                lock.unlink()
                return True
            except FileNotFoundError:
                return True
            except PermissionError:  # Windows: open in a live process
                pass
        if time.monotonic() > deadline:
            return False
        time.sleep(0.1)
    return True


def restore_interrupted_pull(project_root: Path | None = None) -> bool:
    """Put back the files a killed ``hermes update`` had half-moved to the new commit.

    Returns True when the tree changed under this process: modules it already imported may be the
    half-written ones, so the caller must relaunch (``relaunch_after_restore``).

    Fast path (no marker) is one or two ``stat`` calls. Acts only when the marker's owner is gone,
    HEAD is still the pre-pull commit and no merge/rebase is in progress; then every path git wrote
    (the target's content, or torn on the way there) returns to HEAD (the commit the venv was built
    for), so the install is whole again and ``hermes update`` redoes the update from the start. Local
    edits are never touched; the updater's autostash (if any) stays in ``git stash list``. Concurrent
    launches take turns (``_restore_claim``); a launch that waited out another's restore relaunches.

    Limits, by design: a torn ``hermes_cli/__init__.py``, ``hermes_bootstrap.py``, ``agent/__init__.py``
    or ``agent/jiter_preload.py`` (imported before the ``hermes-agent`` hook) fails before this runs. A file git also changes that the user deleted, emptied or cut to a prefix of git's version
    looks exactly like git's own half-written file and is restored too, as is a user edit to a
    conflicted path or, on git < 2.38, to a path both sides of a custom-branch merge changed.
    """
    try:
        root = _project_root() if project_root is None else project_root
        marker = interrupted_pull_marker(root)
        if not marker.is_file() or _pytest_owns_live_checkout(root):
            return False
        with _restore_claim(marker.parent) as claimed:
            if not claimed:
                print("⚠ Another Hermes launch is still repairing the checkout after an interrupted "
                      "`hermes update`; if this one fails, launch again in a moment.", file=sys.stderr)
                return False
            if not marker.is_file():
                return True  # another launch finished while this one started: rerun from its tree
            return _restore_holding_claim(root, marker)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        # Never block launch: the import that follows surfaces any real breakage.
        print(f"⚠ Could not check for an interrupted `hermes update`: {exc}", file=sys.stderr)
    return False


def _restore_holding_claim(root: Path, marker: Path) -> bool:
    global _merge_advice_shown
    git_dir = marker.parent
    fields = dict(line.partition("=")[::2] for line in marker.read_text(encoding="utf-8").splitlines())
    try:
        owner = int(fields.get("pid", ""))
    except ValueError:
        owner = -1
    # Our own pid is never the owner: this runs at startup, and containers hand a retry the
    # killed updater's pid.
    if (owner != os.getpid() and _pid_is_running(owner)
            and time.time() - marker.stat().st_mtime < _INTERRUPTED_PULL_MAX_AGE_SECONDS):
        return False
    pre, target = fields.get("pre", "").strip(), fields.get("target", "").strip()
    stash = fields.get("stash", "").strip()

    def git(*args: str, stdin: str | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(["git", "--literal-pathspecs", "-C", str(root), *args], input=stdin,
                              capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=120, stdin=None if stdin is not None else subprocess.DEVNULL)

    if not pre or not target or git("rev-parse", "HEAD").stdout.strip() != pre:
        marker.unlink()  # git finished (HEAD moved) or the marker is unusable
        return False
    if any((git_dir / name).exists() for name in _GIT_OPERATION_IN_PROGRESS):
        merge_head = git_dir / "MERGE_HEAD"
        if (not _merge_advice_shown and merge_head.is_file()
                and merge_head.read_text(encoding="utf-8").strip() == target):
            # The killed updater's own merge: its conflict markers may sit in startup modules.
            _merge_advice_shown = True
            print(f"⚠ A killed `hermes update` left its merge unfinished. Run `git -C {root} merge --abort`, "
                  "then launch again." + (f" Your local changes are in its stash ({stash})." if stash else ""),
                  file=sys.stderr)
        return False
    # A killed claim holder's own git child can still be writing; scanning under it reads half a tree.
    if not _release_dead_index_lock(git_dir):
        print("⚠ A running git holds the index after an interrupted `hermes update`; the next launch "
              "finishes the restore.", file=sys.stderr)
        return False
    written = _paths_git_wrote(git, root, pre, target)
    if written is None:  # after a gc or re-clone: nothing left to compare against
        marker.unlink()
        print(f"⚠ Ignoring a stale interrupted-update marker: commit {target[:10]} is gone.", file=sys.stderr)
        return False
    restore, added, new_dirs = written
    if restore or added:
        print("⚠ A previous `hermes update` was killed while git was writing the new code — "
              f"restoring the checkout to {pre[:10]}...", file=sys.stderr)
        failed = None
        if restore:
            run = git("restore", "--source=HEAD", "--staged", "--worktree", "--pathspec-from-file=-",
                      "--pathspec-file-nul", stdin="\0".join(restore))
            failed = run if run.returncode else None
        if added and not failed:
            run = git("rm", "-q", "--cached", "--ignore-unmatch", "--pathspec-from-file=-",
                      "--pathspec-file-nul", stdin="\0".join(added))
            failed = run if run.returncode else None
            for rel in added:
                (root / rel).unlink(missing_ok=True)
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
    marker.unlink()
    if not restore and not added:
        return False  # the killed git never reached the tree: nothing to put back
    print("  ✓ Checkout restored; `hermes update` updates it again.", file=sys.stderr)
    if stash:
        print(f"  Your local changes are still in the update's stash ({stash}).", file=sys.stderr)
    return True


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


def _pinned_specs(packages: list[str], project_root: Path) -> list[str]:
    """Map bare package names to their pinned specs from pyproject.toml.

    Naive requirement-head parsing on purpose — ``packaging`` may itself be broken in the failure
    state this module exists for. Unknown packages fall back to their bare name.
    """
    project = _load_pyproject_project(project_root)
    if project is None:
        return packages
    name_to_spec: dict[str, str] = {}
    for spec in project.get("dependencies", []) or []:
        head = spec.split(";", 1)[0].strip()
        bare = head
        for op in ("==", ">=", "<=", "~=", ">", "<", "!="):
            if op in bare:
                bare = bare.split(op, 1)[0]
                break
        key = bare.strip().split("[", 1)[0].strip().lower()
        if key:
            name_to_spec[key] = head
    return [name_to_spec.get(pkg.lower(), pkg) for pkg in packages]


def _certifi_bundle_broken() -> bool:
    """True when certifi imports but its ``cacert.pem`` is missing/corrupt.

    A brew Python upgrade or interrupted venv rebuild can leave certifi's metadata intact while
    ``cacert.pem`` is gone or a dangling symlink; an attribute probe passes in that state and every
    TLS connection then fails opaquely, so validate the bundle path itself.
    """
    try:
        import certifi

        bundle = Path(certifi.where())
        # <1 KiB cannot hold a single PEM certificate — treat as corrupt.
        return not bundle.is_file() or bundle.stat().st_size < 1024
    except Exception:
        # Import failure is caught by the regular probe table; failing to even stat is broken.
        return True


def _probe_broken_packages() -> list[str]:
    """Import-probe the fragile core packages in THIS process.

    Returns repair package names (deduped, probe order) for modules that fail to import or lack
    their sentinel attribute. Failed imports leave nothing in ``sys.modules``, so a post-repair
    retry in the same process works.
    """
    broken: list[str] = []
    for mod_name, attr in LAZY_REFRESH_IMPORT_PROBES:
        try:
            mod = importlib.import_module(mod_name)
            if not hasattr(mod, attr):
                raise ImportError(f"{mod_name} missing {attr}")
            if mod_name == "certifi" and _certifi_bundle_broken():
                raise ImportError("certifi cacert.pem missing or corrupt")
        except Exception:
            pkg = LAZY_REFRESH_REPAIR_PACKAGES.get(mod_name)
            if pkg and pkg not in broken:
                broken.append(pkg)
    return broken


def _hermes_home() -> Path:
    """Resolve the Hermes home without importing hermes_constants.

    This module is stdlib-only on purpose — it runs when the checkout is
    already broken enough that importing a Hermes module may fail. Mirrors
    hermes_constants._get_platform_default_hermes_home (LOCALAPPDATA on
    Windows, ~/.hermes elsewhere); if that logic ever changes, keep the two
    in sync.
    """
    hermes_env = os.environ.get("HERMES_HOME", "").strip()
    if hermes_env:
        return Path(hermes_env)
    if sys.platform == "win32":
        local_appdata = os.environ.get("LOCALAPPDATA", "").strip()
        base = Path(local_appdata) if local_appdata else Path.home() / "AppData" / "Local"
        return base / "hermes"
    return Path.home() / ".hermes"


def _hermes_root() -> Path:
    """Resolve the default Hermes root without importing hermes_constants.

    Mirrors :func:`hermes_constants.get_default_hermes_root`: the shared,
    per-machine managed uv lives here, so ``HERMES_HOME=<root>/profiles/<name>``
    must map back to ``<root>`` (a named profile shares the one uv binary).
    """
    home = _hermes_home()
    if home.parent.name == "profiles":
        return home.parent.parent
    return home


def _uv_isolation_env(base_env: dict | None = None) -> dict:
    """Stdlib-only twin of ``hermes_cli.managed_uv.managed_uv_env()`` (user layout).

    This module runs when the checkout is broken enough that importing
    ``managed_uv`` may fail, so it cannot reuse the runtime helper — it hand-
    rolls the same pins, OVERRIDING any inherited ``UV_*`` so a repair never
    writes the user's own uv state.  The two are kept in agreement by
    ``tests/scripts/install/test_uv_isolation_contract.py``.
    """
    env = dict(os.environ if base_env is None else base_env)
    home = _hermes_home()
    env.update({
        "UV_CACHE_DIR": str(home / "cache" / "uv"),
        "UV_TOOL_DIR": str(home / "uv" / "tools"),
        "UV_TOOL_BIN_DIR": str(home / "bin"),
        "UV_PYTHON_INSTALL_DIR": str(home / "python"),
        "UV_PYTHON_INSTALL_BIN": "0",
        "UV_PYTHON_INSTALL_REGISTRY": "0",
    })
    return env


def _find_uv_binary() -> str | None:
    """Locate a ``uv`` binary without importing third-party modules.

    uv-managed base interpreters carry an ``EXTERNALLY-MANAGED`` marker, so
    the stdlib ``pip`` fallback below refuses to touch them.  In that state
    the only sanctioned installer is uv itself, which Hermes already vendors
    (``~/.hermes/uv/uv.exe``) or the user has on PATH.  Stdlib-only.
    """
    exe = "uv.exe" if sys.platform == "win32" else "uv"
    hermes_root = _hermes_root()

    candidates = [
        # Shared private uv under the default root (%LOCALAPPDATA%/hermes on
        # Windows, ~/.hermes on POSIX, or a custom root under HERMES_HOME).
        hermes_root / "uv" / exe,
        # Legacy pre-isolation layout; migrated to the private dir on use.
        hermes_root / "bin" / exe,
        Path.home() / ".local" / "bin" / exe,
        Path.home() / ".cargo" / "bin" / exe,
    ]
    for path in candidates:
        if path.is_file():
            return str(path)
    return shutil.which(exe)


def _base_interpreter_is_externally_managed() -> bool:
    """True when ``sys.executable`` is a uv/standalone-builds managed install.

    Those ship an ``EXTERNALLY-MANAGED`` marker next to their stdlib (PEP 668), so
    ``python -m pip install`` aborts with ``externally-managed-environment``; the early repair
    must then go through uv (or explicitly override pip) or the venv stays broken.

    See #83569.
    """
    try:
        import sysconfig

        stdlib = Path(sysconfig.get_path("stdlib"))
        # uv 0.5+ moved the marker one level up next to a ``_uv_managed`` sentinel dir.
        return ((stdlib / "EXTERNALLY-MANAGED").exists()
                or (stdlib.parent / "EXTERNALLY-MANAGED").exists())
    except Exception:
        return False


def _run_installer(tool: str, cmd: list[str], root: Path, env: dict | None = None) -> bool:
    """Run one installer command; captured output is replayed to stderr only on failure."""
    try:
        result = subprocess.run(cmd, cwd=root, capture_output=True, text=True, encoding="utf-8",
                                errors="replace", env=env)
    except Exception as exc:
        print(f"  ✗ Early venv repair could not run {tool}: {exc}", file=sys.stderr)
        return False
    if result.returncode == 0:
        return True
    tail = (result.stderr or result.stdout or "")[-2000:]
    if tail:
        print(tail, file=sys.stderr)
    return False


def _run_repair_install(specs: list[str], project_root: Path) -> bool:
    """``uv pip`` (or stdlib ``pip``) force-reinstall of the given specs. Never raises.

    Streams nothing to stdout (``hermes acp`` speaks JSON-RPC on stdout). uv is preferred when the
    base interpreter is externally managed; without uv, pip runs with the PEP 668 override.
    """
    externally_managed = _base_interpreter_is_externally_managed()
    if externally_managed:
        uv = _find_uv_binary()
        if uv:
            # stdlib-only repair: pin uv's write dirs by hand (no managed_uv), or
            # these writes land in the user's own uv cache / tool store / python
            # store — the pollution the uv isolation removes.
            env = _uv_isolation_env({**os.environ, "VIRTUAL_ENV": str(project_root / "venv")})
            env.pop("PYTHONHOME", None)
            env.pop("PYTHONPATH", None)
            return _run_installer("uv", [uv, "pip", "install", "--force-reinstall", *specs],
                                  project_root, env)
        print("  ⚠ Base interpreter is externally managed and no uv binary was "
              "found; retrying repair via pip with PEP 668 override.", file=sys.stderr)
    _run_ensurepip(project_root)
    pip_cmd = [sys.executable, "-m", "pip", "install", "--force-reinstall"]
    if externally_managed:
        pip_cmd.append("--break-system-packages")
    return _run_installer("pip", pip_cmd + specs, project_root)


def _pytest_owns_live_checkout(root: Path) -> bool:
    """True under pytest when ``root`` is this module's own checkout — the venv running the suite.

    Lifecycle tests spawn real subprocesses that import ``hermes_cli.main`` with recovery armed and
    inherit ``PYTEST_CURRENT_TEST``; without this guard a broken dev venv would get a REAL
    ensurepip + force-reinstall from inside a running suite. tmp_path roots are unaffected.
    """
    return "PYTEST_CURRENT_TEST" in os.environ and root == Path(__file__).resolve().parent.parent


def recover_if_needed(project_root: Path | None = None, argv: list[str] | None = None) -> None:
    """Repair wiped core packages so ``hermes_cli.main`` can import at all.

    Fast path (no marker present) is two ``lstat`` calls. Only acts when a recovery marker from a
    prior ``hermes update`` exists AND an import probe confirms a core package is actually broken.
    Never raises: on any failure the import of main.py proceeds and surfaces the real error.
    """
    global _UPDATE_RETRY_RECOVERED

    try:
        args = sys.argv[1:] if argv is None else argv
        root = _project_root() if project_root is None else project_root
        if _pytest_owns_live_checkout(root):
            return
        core_marker = root / ".update-incomplete"
        lazy_marker = root / ".lazy-refresh-incomplete"
        if not core_marker.exists() and not lazy_marker.exists():
            return
        # Managed/Docker/PyPI installs have no source tree here — the marker is not ours to act
        # on; main.py's recovery clears it.
        if not (root / "pyproject.toml").is_file():
            return

        # Pending core install: complete it NOW, before any native extension is imported, so the
        # WHOLE dependency set is replaced while nothing pins venv .pyd files yet (deferring to
        # main()'s post-import recovery re-locks it on Windows). A live marker owner is another
        # updater inside the marker-to-install window — never race it. A dead owner MUST be
        # recovered even when this launch is itself `hermes update`: CLI and Desktop retries keep
        # that argv, and skipping solely on argv recreates the self-lock loop.
        # Bounded retries: a persistently failing install must not hammer every launch. The budget
        # is shared with main.py's post-import recovery, which past it prints the manual command
        # instead of reinstalling. See #83569, #81594.
        if core_marker.exists():
            if _marker_owner_is_live(core_marker):
                return
            if _complete_pending_core_install(root, core_marker) and "update" in args:
                _UPDATE_RETRY_RECOVERED = True
            return

        # The lazy-refresh marker keeps the update-argv exclusion: it is not a deferred native
        # install, and the active update flow owns its probe/repair lifecycle.
        if "update" in args:
            return

        broken = _probe_broken_packages()
        if not broken:
            return  # main.py will load and run full recovery.

        # Single-flight: share main.py's recovery lock so an early repair never races a
        # concurrent full recovery into the same shared venv.
        if not _claim_recovery_lock(root):
            return
        try:
            specs = _pinned_specs(broken, root)
            print("⚠ Core package(s) broken by an interrupted update — "
                  f"repairing before launch: {', '.join(broken)}", file=sys.stderr)
            if _run_repair_install(specs, root) and not _probe_broken_packages():
                print("  ✓ Core packages repaired.", file=sys.stderr)
            else:
                print("  ✗ Automatic repair incomplete. Recover manually with:", file=sys.stderr)
                print(f"    {sys.executable} -m pip install --force-reinstall " + " ".join(specs),
                      file=sys.stderr)
        finally:
            _release_recovery_lock(root)
    except Exception:
        pass  # Never block launch — the import of main.py will surface the truth.


# Cap on automatic core-install retries, shared by this early pass and main.py's post-import
# recovery: a persistently failing install (network down, a .pyd another Hermes process maps) must
# not reinstall-hammer every launch. Past it the post-import path presents the manual command. The
# counter lives in the marker's JSON body; an update that reaches the dependency sync rewrites
# the marker, re-arming it.
_CORE_INSTALL_MAX_ATTEMPTS = 3


def _claim_recovery_lock(root: Path) -> bool:
    """Single-flight claim on the shared recovery lock.  Never raises."""
    lock_path = root / ".update-incomplete.lock"
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.getpid()}\n".encode())
        os.close(fd)
        return True
    except FileExistsError:
        try:
            if time.time() - lock_path.stat().st_mtime > 3600:
                lock_path.unlink()
        except OSError:
            pass
        return False
    except OSError:
        # Read-only fs / perms — proceed unlocked; the install itself surfaces the real problem.
        return True


def _release_recovery_lock(root: Path) -> None:
    """Best-effort release of the shared recovery lock."""
    try:
        (root / ".update-incomplete.lock").unlink()
    except OSError:
        pass


def _complete_pending_core_install(root: Path, core_marker: Path) -> bool:
    """Run the pending core install BEFORE main.py can import native modules.

    Never raises: any failure leaves the marker for the post-import path and returns ``False``.
    Returns ``True`` only after the install succeeds.

    ``recover_if_needed`` invokes this when ``.update-incomplete`` exists — a prior ``hermes update`` (or
    the self-lock preflight, #83569) left the dependency sync deliberately unfinished. Completing it here
    matters on Windows: the deferral exists precisely because the process that wrote the marker had a native
    venv extension mapped; this process, running before ``hermes_cli.main``'s third-party imports, maps
    nothing yet, so the installer can replace ``.pyd`` files without hitting the lock.
    """
    try:
        from hermes_cli import _install_repair as ir

        # Read attempts before claiming the lock so a persistently-failing install stops early.
        attempts = _read_marker_attempts(core_marker)
        if attempts >= _CORE_INSTALL_MAX_ATTEMPTS:
            print("⚠ Pending interrupted-update install has already failed "
                  f"{attempts} times — skipping the automatic retry.", file=sys.stderr)
            return False
        if not _claim_recovery_lock(root):
            return False
        try:
            print("⚠ A previous `hermes update` was interrupted mid-install — "
                  "finishing dependency installation now (before any native "
                  "extensions load)...", file=sys.stderr)
            ir.run_core_install(root)
        except Exception as exc:
            new_attempts = ir.bump_marker_attempts(core_marker)
            print(f"  ✗ Early interrupted-install completion failed (attempt "
                  f"{new_attempts}/{_CORE_INSTALL_MAX_ATTEMPTS}): {exc}", file=sys.stderr)
            if new_attempts < _CORE_INSTALL_MAX_ATTEMPTS:
                print("  The next launch will retry; hermes will keep working from "
                      "the current venv in the meantime.", file=sys.stderr)
            return False
        finally:
            _release_recovery_lock(root)

        try:
            core_marker.unlink()
        except OSError:
            pass
        print("  ✓ Dependency installation completed in the early pass.", file=sys.stderr)
        return True
    except Exception:
        return False  # Never block launch — the marker stays for the post-import path.

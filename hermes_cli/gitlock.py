"""Stale git lock-file and aborted-fetch pack-debris recovery for update/check paths.

A killed ``git fetch`` can leave ``.git/shallow.lock`` behind (every later fetch fails with "Unable to
create '.../shallow.lock': File exists") and ``tmp_pack_*`` files git itself never cleans up."""

from __future__ import annotations

import logging
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Callable, Iterable, List, Optional

from runtime.git_subprocess import NO_LAZY_FETCH_ENV, noninteractive_git_env
from runtime.subprocess_compat import bounded_probe_run, windows_hide_flags
from hermes_cli.update_custody import run_git

logger = logging.getLogger(__name__)

# Folding ~100 small packs takes seconds, but the checkouts this exists for (thousands of packs,
# tens of GiB) need a full repack. A killed fold restarts from scratch on every update and never
# converges, so the bound is generous and a timeout is reported, not swallowed.

# Files younger than this are presumed live (a fetch may be in flight) and are never removed. Lock
# files live for seconds and a healthy fetch completes in minutes; 10 minutes is abandoned.
STALE_LOCK_MIN_AGE_SECONDS = 10 * 60
STALE_TMP_PACK_MIN_AGE_SECONDS = STALE_LOCK_MIN_AGE_SECONDS
# ``shallow.lock`` is the one observed in the wild; the others are the same class of failure
# (interrupted git operation). Locks held by a live git process are protected by the process guard.
LOCK_NAMES = ("shallow.lock", "index.lock", "HEAD.lock", "MERGE_HEAD.lock")
# Temp-file prefixes git writes into .git/objects/pack during a transfer and renames away on
# success; anything left with these names after a fetch died is garbage by definition.
# ``.tmp-<pid>-pack*`` is the same thing from ``pack-objects`` (repack/gc) killed mid-write.
_TMP_PACK_PREFIXES = ("tmp_pack_", "tmp_idx_", "tmp_rev_", "tmp_mtimes_", ".tmp-")


def _git_proc_running() -> bool:
    """True when a ``git`` process is running — the check that stops us yanking a lock a live fetch holds.

    A failed probe logs and returns False; the age floor in the sweep still applies.
    """
    try:
        # One shared spawn for both probes: the argv is the platform fork, the
        # hide-flags wiring is common, so a console-less parent (pythonw backend)
        # never flashes a console for either (#117781).
        argv = (["tasklist", "/FI", "IMAGENAME eq git.exe", "/FO", "CSV"]
                if os.name == "nt" else ["pgrep", "-x", "git"])
        proc = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", errors="replace",
                               timeout=10, creationflags=windows_hide_flags())
        if os.name == "nt":
            return "git.exe" in proc.stdout.lower()
        return proc.returncode == 0
    except Exception:
        logger.debug("git process probe failed; assuming no git running", exc_info=True)
        return False


def _sweep_stale(directory: Path, candidates: Callable[[], Iterable[Path]], *, min_age_seconds: Optional[int],
                 default_age: int, skip_msg: str, log_removed: Callable[[Path, int], None]) -> list[str]:
    """Shared guard + age-floor sweep. Never raises; skips anything it cannot stat/unlink."""
    if not directory.is_dir():
        return []
    if _git_proc_running():
        logger.debug(skip_msg)
        return []
    cutoff = time.time() - (min_age_seconds if min_age_seconds is not None else default_age)
    removed: list[str] = []
    for entry in candidates():
        try:
            if entry.is_file() and (st := entry.stat()).st_mtime < cutoff:
                if os.name == "nt":
                    # git renames its transfer temps into place read-only; Windows refuses to
                    # unlink a read-only file (EACCES/13), so without clearing the write bit
                    # this sweep silently removes nothing on real debris (#116384).
                    try:
                        os.chmod(entry, 0o666)
                    except OSError:
                        pass
                entry.unlink()
                removed.append(str(entry))
                log_removed(entry, st.st_size)
        except OSError as exc:
            # A cleaner that fails silently is worse than none: debug-level skips hid the
            # Windows read-only unlink failure for months while debris grew to gigabytes.
            logger.warning("Could not clear %s (skipping): %s", entry, exc)
    return removed


def clear_stale_git_locks(repo_root: Path, *, min_age_seconds: Optional[int] = None) -> list[str]:
    """Remove abandoned ``.git`` lock files under ``repo_root``; returns the removed paths.

    Removes only when older than the age floor AND no git process is running. Never raises: a lock we cannot
    stat/unlink is skipped (it may have been re-created between the age check and the unlink; skipping is safe).
    """
    git_dir = Path(repo_root) / ".git"
    return _sweep_stale(
        git_dir, lambda: [git_dir / name for name in LOCK_NAMES],
        min_age_seconds=min_age_seconds, default_age=STALE_LOCK_MIN_AGE_SECONDS,
        skip_msg="git process running; skipping stale-lock sweep",
        log_removed=lambda p, _size: logger.info("Removed stale git lock %s", p),
    )


def release_dead_index_lock(repo_root: Path) -> bool:
    """Drop ``.git/index.lock`` when the git that took it is proven gone, whatever its age.

    A git killed while it held the index lock (an update's tree killed by the user, a probe killed
    by its own timeout) leaves it behind, and every later merge/stash/reset refuses with "File
    exists". The age floor above keeps such a lock for 10 minutes, so the next ``hermes update``
    died at its fast-forward (#132089; the Windows crash cell ``mid_fetch``).

    Two conditions, both required. A killed update run must be behind the lock
    (:func:`_killed_update_owns`): necessary, never sufficient, since a user's ``git commit`` can take
    the lock after that run died. And no git of ANY form may be working in this checkout: a
    ``git commit`` waiting in the editor has closed its lock fd, so an absent fd proves nothing, and
    it runs as ``git``, a dashed ``git-commit``, an alias or ``git.exe``. Linux reads every process's
    cwd, path arguments and ``GIT_DIR`` through /proc (``_early_recovery._held_open`` with
    ``any_git``); macOS keeps the lock while any git runs (``ps`` cannot say where); Windows scans
    with psutil (:func:`_windows_git_in_checkout`). Whatever cannot be read keeps the lock. An
    interrupted tree move owns its own lock judgement, so its marker defers to that repair.
    """
    from hermes_cli._early_recovery import _git_dir, _release_dead_index_lock, interrupted_pull_marker

    root = Path(repo_root)
    git_dir = _git_dir(root)
    lock = git_dir / "index.lock"
    if not lock.exists() or interrupted_pull_marker(root).exists():
        return False
    if not _killed_update_owns(lock):
        return False
    if os.name == "nt" and _windows_git_in_checkout(root) is not False:
        return False
    return _release_dead_index_lock(git_dir, root, any_git=True)


def _receipt_time(value: object) -> Optional[float]:
    from datetime import datetime

    try:
        return datetime.fromisoformat(str(value)).timestamp()
    except (TypeError, ValueError):
        return None


def _killed_update_owns(lock: Path) -> bool:
    """The last update run that started before ``lock`` was written died unfinished, its processes gone.

    Read from the durable per-run records (read-only), not only ``latest.json``: a retry that kept
    the then-live lock and failed replaced ``latest.json``, and the killed run's evidence must
    still stand once the lock's git is gone. A run that started after the lock was written cannot
    have left it, so a later run never discharges an earlier killed one.
    """
    import json

    from hermes_cli import update_receipt

    try:
        written = lock.stat().st_mtime
    except OSError:
        return False
    records: list[dict] = []
    try:
        for directory in update_receipt._receipt_dirs():
            for path in directory.glob("update_*.json"):
                try:
                    record = json.loads(path.read_text(encoding="utf-8-sig"))
                except (OSError, ValueError):
                    continue
                if isinstance(record, dict):
                    records.append(record)
    except OSError:
        pass
    latest = update_receipt.read_latest_receipt()
    if isinstance(latest, dict):
        records.append(latest)  # a store from before per-run records
    runs: dict[str, tuple[float, dict]] = {}
    for record in records:
        started = _receipt_time(record.get("started_at"))
        if started is None or started > written:
            continue
        key = str(record.get("update_id") or id(record))
        held = runs.get(key)
        # One run, several copies (its archive, latest.json, a profile mirror): a terminal copy wins.
        if held is None or (held[1].get("outcome") in ("running", "interrupted")
                            and record.get("outcome") not in ("running", "interrupted")):
            runs[key] = (started, record)
    if not runs:
        return False
    _started, last = max(runs.values(), key=lambda run: run[0])
    return last.get("outcome") in ("running", "interrupted") and not update_receipt._owner_alive(last)


def _windows_git_in_checkout(root: Path) -> "bool | None":
    """Whether any git (``git.exe``, a dashed ``git-<sub>.exe``) works in ``root``; None when unknowable.

    Works in: its cwd, a path argument or a ``GIT_DIR``-style variable is inside the checkout or
    its git dir, compared by path components (``hermes-backup`` is not inside ``hermes``). Scoped to
    the checkout: a system-wide check would keep the lock whenever any editor or terminal elsewhere
    runs git, which on Windows is most of the time. A git whose cwd cannot be read keeps the lock.
    """
    from hermes_cli._early_recovery import _NEVER_LOCKS, _checkout_places, _git_dir, _git_program, \
        _git_subcommand_of, git_works_in

    try:
        import psutil
    except ImportError:
        return None
    places = _checkout_places(root, _git_dir(Path(root)))
    try:
        for proc in psutil.process_iter(["pid", "name", "cwd", "cmdline", "environ"]):
            info = proc.info
            if info.get("pid") == os.getpid():
                continue
            args = [str(a) for a in info.get("cmdline") or () if a]
            if _git_program(info.get("name") or "") is None and not (args and _git_program(args[0])):
                continue
            if _git_subcommand_of(args or [info.get("name") or ""]) in _NEVER_LOCKS:
                continue
            cwd = info.get("cwd")
            if git_works_in(args or [info.get("name") or "git"], cwd, info.get("environ"), places):
                return True
            if not cwd and proc.is_running():
                return None  # a git we cannot inspect may be working here
    except Exception:  # health: allow BLE001 -- a failed scan keeps the lock, never drops it
        return None
    return False


def _pack_dir(repo_root: Path) -> Path:
    git_dir = Path(repo_root) / ".git"
    return (git_dir if git_dir.is_dir() else Path(repo_root)) / "objects" / "pack"


def mark_unmarked_packs_promisor(repo_root: Path) -> int:
    """Give every pack without a ``.promisor`` marker one; returns how many were marked.

    git 2.53+ ``index-pack --promisor`` hands objects that sit outside promisor packs to
    pack-objects, which BUG()s on the missing objects they lead to (#124272). Unmarked packs in a
    partial clone come from a filtered fetch that turned a full or depth-limited clone partial (the
    old packs keep no marker) and from checkouts that lost their markers. The marker only tells git
    that objects those packs link to may be fetched on demand; nothing is rewritten or deleted.
    """
    marked = 0
    for pack in _pack_dir(repo_root).glob("pack-*.pack"):
        marker = pack.with_suffix(".promisor")
        try:
            with open(marker, "xb"):
                marked += 1
        except FileExistsError:
            pass
        except OSError as exc:
            # A read-only store must not turn a fetch failure into a traceback.
            logger.warning("Could not mark %s as a partial-clone pack: %s", pack, exc)
    if marked:
        logger.info("Marked %d pack(s) in %s as partial-clone packs", marked, repo_root)
    return marked


# In a partial clone whose commit-graph already carries changed-path (Bloom) data, any commit-graph
# write over commits it has not seen yet needs their trees: one lazy fetch (and one pack) per commit,
# and each lazy fetch spawns ``git maintenance`` again, which is the unbounded loop of #127711. gc,
# ``maintenance run --task=commit-graph`` and ``fetch.writeCommitGraph`` all write one (git
# 2.50.1, 2.53.0 and 2.55.0 alike). So a promisor checkout never writes the graph. Automatic
# maintenance itself stays on: on git <= 2.53 its post-fetch ``gc --auto`` is what keeps the
# lazy-fetch packs folded between updates. ``gc.auto`` stays at its default for the same reason
# (``gc.auto=0`` turns that fold into a no-op).
_TREE0_MAINTENANCE_OFF = (
    ("maintenance.commit-graph.enabled", "false"),
    ("gc.writeCommitGraph", "false"),
    ("fetch.writeCommitGraph", "false"),
)


def disable_tree0_auto_maintenance(repo_root: Path) -> None:
    """Keep git from writing a commit-graph in a partial clone, from gc, fetch or maintenance.

    Idempotent (a value already in place is not rewritten, so concurrent git never meets a config
    lock from this) and never raises: a read-only config must not turn fetch recovery into a
    traceback, matching mark_unmarked_packs_promisor above.
    """
    _migrate_earlier_maintenance_keys(repo_root)
    for key, value in _TREE0_MAINTENANCE_OFF:
        try:
            current = run_git(
                ["git"], ["config", "--local", "--get", key],
                cwd=str(repo_root), capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=30,
                creationflags=windows_hide_flags(),
            ).stdout.strip()
            if current == value:
                continue
            run_git(
                ["git"], ["config", "--local", key, value],
                cwd=str(repo_root), check=True,
                capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=30,
                creationflags=windows_hide_flags(),
            )
        except Exception:
            logger.warning("Could not set %s=%s in %s", key, value, repo_root)


def _migrate_earlier_maintenance_keys(repo_root: Path) -> None:
    """Undo what earlier cuts of _TREE0_MAINTENANCE_OFF persisted, once.

    ``maintenance.auto=false`` switched off git's own post-fetch fold (git <= 2.53) and the first
    cut's ``gc.auto=0`` turns ``gc --auto``, the update's own fold, into a no-op. Nothing records
    who wrote them, so they are removed only under the earlier cuts' fingerprint: both wrote
    ``maintenance.auto=false`` together with ``fetch.writeCommitGraph=false``, and neither wrote
    ``maintenance.commit-graph.enabled``. The first pass of disable_tree0_auto_maintenance writes
    that key, so this runs once; an operator's own lone setting never matches.
    """
    try:
        local = dict(line.split(None, 1) for line in _git_stdout_lines(repo_root, [
            "config", "--local", "--get-regexp",
            r"^(maintenance\.auto|maintenance\.commit-graph\.enabled|fetch\.writecommitgraph|gc\.auto)$"]))
        if (local.get("maintenance.auto") != "false" or local.get("fetch.writecommitgraph") != "false"
                or "maintenance.commit-graph.enabled" in local):
            return
        for key in ("maintenance.auto", "gc.auto") if local.get("gc.auto") == "0" else ("maintenance.auto",):
            run_git(
                ["git"], ["config", "--local", "--unset", key],
                cwd=str(repo_root), check=True, capture_output=True, timeout=30,
                creationflags=windows_hide_flags(),
            )
    except Exception:
        logger.warning("Could not migrate earlier maintenance keys in %s", repo_root)


def clear_stale_tmp_packs(repo_root: Path, *, min_age_seconds: Optional[int] = None) -> list[str]:
    """Remove aborted-transfer temp pack files; same contract as clear_stale_git_locks.

    Resolves ``.git/objects/pack`` for a checkout and ``objects/pack`` for a bare repo such as
    the checkpoint store — a ``git gc`` killed by a timeout strands the same debris there."""
    pack_dir = _pack_dir(repo_root)

    def _candidates():
        try:
            return [e for e in pack_dir.iterdir() if e.name.startswith(_TMP_PACK_PREFIXES)]
        except OSError:
            return []

    return _sweep_stale(
        pack_dir, _candidates,
        min_age_seconds=min_age_seconds, default_age=STALE_TMP_PACK_MIN_AGE_SECONDS,
        skip_msg="git process running; skipping tmp-pack sweep",
        log_removed=lambda p, size: logger.info("Removed aborted-fetch pack debris %s (%d bytes)", p, size),
    )


def _git_stdout_lines(repo_root: Path, args: list[str]) -> list[str]:
    """Run a read-only git query in ``repo_root``; [] on any failure."""
    try:
        result = run_git(
            ["git"], [*args], cwd=str(repo_root),
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10,
            creationflags=windows_hide_flags(),
        )
        if result.returncode != 0:
            return []
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except Exception:
        logger.debug("git query failed: %s", args, exc_info=True)
        return []


def _batch_missing_parents(repo_root: Path, candidates: list[str]) -> set[str]:
    """Return local commit objects whose parent objects are missing.

    Parents are read from the commit *header* only (lines before the first blank
    line) — a ``parent <sha>`` line inside a commit message body is prose, not
    an edge.
    """
    if not candidates:
        return set()
    try:
        parents_by_commit = {}
        parents = set()
        request = "\n".join(candidates) + "\n"
        result = run_git(
            ["git"], ["cat-file", "--batch"],
            cwd=str(repo_root),
            input=request.encode(),
            capture_output=True,
            timeout=30,
            creationflags=windows_hide_flags(),
        )
        if result.returncode != 0:
            return set()
        data = result.stdout
        cursor = 0
        for candidate in candidates:
            header_end = data.find(b"\n", cursor)
            if header_end < 0:
                return set()
            header = data[cursor:header_end].split()
            cursor = header_end + 1
            if len(header) >= 3 and header[1] == b"commit":
                size = int(header[2])
                body = data[cursor:cursor + size]
                cursor += size
                if data[cursor:cursor + 1] != b"\n":
                    return set()
                cursor += 1
                commit_parents = set()
                for line in body.split(b"\n"):
                    if not line:  # blank line ends the commit header block
                        break
                    fields = line.split()
                    if len(fields) >= 2 and fields[0] == b"parent":
                        commit_parents.add(fields[1].decode())
                parents_by_commit[candidate] = commit_parents
                parents.update(commit_parents)
            elif len(header) < 2 or header[1] != b"missing":
                return set()
        if not parents:
            return set()
        check = run_git(
            ["git"], ["cat-file", "--batch-check"],
            cwd=str(repo_root),
            input=("\n".join(sorted(parents)) + "\n").encode(),
            capture_output=True,
            timeout=10,
            creationflags=windows_hide_flags(),
        )
        if check.returncode != 0:
            return set()
        missing = {
            line.split()[0]
            for line in check.stdout.decode(errors="replace").splitlines()
            if line.endswith(" missing")
        }
        return {commit for commit, commit_parents in parents_by_commit.items() if commit_parents & missing}
    except Exception:
        logger.debug("parent-object probe failed for %s", repo_root, exc_info=True)
        return set()


def _shallow_file_path(repo_root: Path) -> Optional[Path]:
    """Resolve ``.git/shallow`` via git, or None when the repo has none."""
    shallow_rel = _git_stdout_lines(repo_root, ["rev-parse", "--git-path", "shallow"])
    if not shallow_rel:
        return None
    shallow_path = Path(shallow_rel[0])
    if not shallow_path.is_absolute():
        shallow_path = Path(repo_root) / shallow_rel[0]
    return shallow_path if shallow_path.is_file() else None


class _ShallowLock:
    """Git's own ``shallow.lock`` protocol, so a concurrent ``git fetch`` that
    writes ``.git/shallow`` between our read and our write is never clobbered:
    the fetch fails fast on the lock and we re-read before writing."""

    def __init__(self, shallow_path: Path):
        self._path = shallow_path
        self._lock_path = shallow_path.with_name(shallow_path.name + ".lock")

    def __enter__(self) -> "_ShallowLock":
        try:
            fd = os.open(self._lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise RuntimeError(f"shallow lock held: {self._lock_path}")
        except OSError as exc:
            raise RuntimeError(f"cannot create shallow lock: {exc}") from exc
        try:
            os.write(fd, b"hermes shallow maintenance\n")
        finally:
            os.close(fd)
        return self

    def __exit__(self, *exc_info) -> None:
        try:
            self._lock_path.unlink()
        except FileNotFoundError:
            pass


def _write_shallow(shallow_path: Path, content: str, *, suffix: str) -> None:
    """Atomically replace ``.git/shallow`` (temp file + os.replace)."""
    tmp_path = shallow_path.with_name(shallow_path.name + suffix)
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(tmp_path, shallow_path)


def repair_broken_shallow_boundaries(repo_root: Path) -> int:
    """Re-append shallow boundaries for reflog-reachable commits whose parents
    were never fetched (#108286).

    A reflog-only commit whose graft was pruned leaves the repo unwalkable
    (``gc``/``fsck``/``fetch`` fail) and unable to self-heal: reflog expiry
    happens during ``git gc``, which is exactly what the corruption breaks.
    Only reflog-reachable commits are considered, so unrelated object loss is
    never re-labelled as shallow history. Returns the number of boundaries
    appended; never raises.
    """
    try:
        shallow_path = _shallow_file_path(repo_root)
        if shallow_path is None:
            return 0
        # Cheap gate: repair only when the walk the corruption breaks already fails.
        probe = run_git(
            ["git"], ["rev-list", "--count", "--all", "--reflog"],
            cwd=str(repo_root), capture_output=True, timeout=10,
            creationflags=windows_hide_flags(),
        )
        if probe.returncode == 0:
            return 0
        with _ShallowLock(shallow_path):
            # Keep the rollback image byte-exact, including BOM and line endings.
            original = shallow_path.read_bytes()
            existing = {line for line in original.decode("utf-8-sig").splitlines() if line}
            if not existing:
                return 0
            # Boundary candidates: commits recorded as *fetch tips* in remote-tracking
            # refs' reflogs (NOT --batch-all-objects, and not HEAD's reflog): the bug
            # class is fetched tips whose graft the prune dropped, and restricting to
            # fetch-recorded tips is what keeps unrelated object loss (a deleted parent
            # of a locally-created commit) from being re-labelled as shallow history.
            # On an already-corrupted repo a rev-list --reflog walk is exactly what
            # fails, so read the reflog hash list directly. Scope: fetch-by-SHA
            # installs (scripts/install.sh) record their tip only in HEAD's reflog
            # and are NOT candidates — new corruption of that shape is prevented by
            # the prune's reflog fail-safe instead.
            reflog = _git_stdout_lines(
                repo_root, ["reflog", "show", "--all", "--format=%H%x00%gD"])
            candidates = sorted({
                sha
                for entry in reflog
                for sha, selector in [entry.split("\x00", 1)]
                if selector.startswith("refs/remotes/")
            })
            broken = _batch_missing_parents(repo_root, candidates)
            repaired = broken - existing
            if not repaired:
                return 0
            _write_shallow(shallow_path, "\n".join(sorted(existing | repaired)) + "\n",
                           suffix=".hermes-repair")
            # Self-check under the same lock hold (rev-list never takes
            # shallow.lock): the rollback cannot be defeated by lock contention.
            if not _git_stdout_lines(repo_root, ["rev-list", "--count", "--all", "--reflog"]):
                shallow_path.write_bytes(original)
                logger.debug("shallow boundary repair self-check failed; file restored")
                return 0
        logger.info("Restored %d broken shallow boundary(ies) in %s", len(repaired), repo_root)
        return len(repaired)
    except Exception:
        logger.debug("shallow boundary repair failed for %s", repo_root, exc_info=True)
        return 0


def prune_stale_shallow_grafts(repo_root: Path) -> int:
    """Drop ``.git/shallow`` graft lines no live ref or reflog still points at (#105951).

    Every ``git fetch --depth 1`` appends the fetched tip to ``.git/shallow`` as a new
    graft and never removes the previous one, so a long-lived shallow installer checkout
    accumulates one graft per update check (57 observed in the wild). The stale grafts
    break ``merge-base`` and push ``hermes update`` into the orphan-divergence reset path
    on every run. Keep only the boundaries that still protect referenced tips (HEAD,
    FETCH_HEAD, and every ref tip): the dropped commits are already unreachable and their
    objects are left for ``git gc``. Fetch reflogs naming a dropped graft are expired
    first — otherwise they pin it and the prune rolls back forever (#124645). Returns
    the number of graft lines removed; never raises, and restores the original file if
    the trimmed set breaks history walking (including the ``--reflog`` walk, so a graft
    a reflog-only commit still needs is never dropped, #108286).
    """
    try:
        shallow_path = _shallow_file_path(repo_root)
        if shallow_path is None:
            return 0
        with _ShallowLock(shallow_path):
            # Decode for pruning, but retain the same capture for a lossless rollback.
            original = shallow_path.read_bytes()
            lines = [line for line in original.decode("utf-8-sig").splitlines() if line]
            if not lines:
                return 0
            keep = set(lines) & {
                *(_git_stdout_lines(repo_root, ["rev-parse", "HEAD"]) or []),
                *(_git_stdout_lines(repo_root, ["rev-parse", "--verify", "--quiet", "FETCH_HEAD"]) or []),
                *_git_stdout_lines(repo_root, ["for-each-ref", "--format=%(objectname)"]),
            }
            if len(keep) == len(lines):
                return 0
            # A dropped graft's parent was never fetched (depth-1), so a fetch reflog
            # still naming it makes the fail-safe walk below fail and rolls the prune
            # back on every run — grafts keep accumulating and the next update falls
            # into orphan divergence (#124645). Fetch-history reflogs are the only safe
            # ones to expire; reflogs users read (HEAD, local branches) keep their
            # entries, and the fail-safe still rolls the prune back for those.
            dropped = set(lines) - keep
            for ref in _git_stdout_lines(
                repo_root, ["for-each-ref", "--format=%(refname)", "refs/remotes/"]
            ):
                entries = _git_stdout_lines(repo_root, ["reflog", "show", "--format=%H", ref])
                if set(entries) & dropped:
                    run_git(
                        ["git"], ["reflog", "expire", "--expire=now", ref],
                        cwd=str(repo_root), capture_output=True, timeout=10,
                        creationflags=windows_hide_flags(),
                    )
            _write_shallow(shallow_path, "\n".join(sorted(keep)) + "\n", suffix=".hermes-prune")
            # Fail-safe: if any reachable walk now crosses a boundary we wrongly
            # removed, put the grafts back — a growing file beats a broken repo.
            # Runs under the same lock hold (rev-list never takes shallow.lock) so
            # the rollback cannot be defeated by lock contention.
            still_walks = _git_stdout_lines(repo_root, ["rev-list", "--count", "HEAD"]) and \
                _git_stdout_lines(repo_root, ["rev-list", "--count", "--all"]) and \
                _git_stdout_lines(repo_root, ["rev-list", "--count", "--all", "--reflog"])
            if not still_walks:
                shallow_path.write_bytes(original)
                logger.debug("shallow prune self-check failed; grafts restored")
                return 0
        logger.info("Pruned %d stale shallow graft(s) in %s", len(lines) - len(keep), repo_root)
        return len(lines) - len(keep)
    except Exception:
        logger.debug("shallow graft prune failed for %s", repo_root, exc_info=True)
        return 0


def _partial_clone_filter(repo_root: Path, **run_kwargs) -> "str | None":
    """The checkout's own ``remote.origin.partialclonefilter``, or None for a non-partial clone."""
    result = run_git(
        ["git"], ["config", "--get", "remote.origin.promisor"],
        cwd=str(repo_root), capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=30, **run_kwargs,
    )
    if result.returncode != 0 or result.stdout.strip().lower() != "true":
        return None
    configured = run_git(
        ["git"], ["config", "--get", "remote.origin.partialclonefilter"],
        cwd=str(repo_root), capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=30, **run_kwargs,
    )
    return configured.stdout.strip() or None


def heal_shallow_history(repo_root: Path, branch: str, **run_kwargs) -> bool:
    """Unshallow a stale installer checkout before the updater's bounded fetch (#123254).

    From a depth-1 clone far behind ``branch``, a plain ``git fetch origin <branch>`` makes the
    server send the full ancestry of every side branch merged past the shallow boundary, which
    cannot finish inside the 300s network cap; the post-update unshallow never got to run. Fetch
    the commit graph first (900s budget), pulling ``branch`` along so the bounded fetch that
    follows is small. Returns whether the checkout was shallow; raises on fetch failure.
    """
    return _shallow_file_path(repo_root) is not None and fetch_full_commit_graph(repo_root, branch, **run_kwargs)


def fetch_full_commit_graph(repo_root: Path, *extra_refspecs: str, **run_kwargs) -> bool:
    """Refresh release tags and fill shallow history before publishing identity.

    A full commit graph does not imply current tags, especially after a --no-tags
    clone. Fetch version tags explicitly without fetching every remote branch or
    replacing existing tags. The fetch never changes the clone's mode: ``--filter`` makes git
    write ``remote.origin.promisor``/``partialclonefilter``, so a full clone fetches unfiltered
    (#122353) and a partial clone repeats its own filter. The one conversion is deliberate: a
    depth-limited full clone whose history is really missing unshallows as a partial clone, because
    an unfiltered ``--unshallow`` downloads every file version ever committed; a full clone grafted by a
    ``--depth`` fetch already has its history and stays full. The converted clone's existing packs
    are marked as partial-clone packs, or every later fetch crashes on git 2.53+ (#124272).
    It converts to ``blob:none``, the layout installers make (see ``convert_treeless_checkout``).
    Returns whether the checkout was unshallowed; fetch failures raise subprocess errors.
    """
    shallow_path = _shallow_file_path(repo_root)
    shallow = shallow_path is not None
    # Callers already inject creationflags via no_prompt_git_kwargs(); OR the
    # hide flag into the shared kwargs instead of passing the keyword twice
    # (TypeError: got multiple values) — the config probes below inherit it.
    run_kwargs["creationflags"] = run_kwargs.get("creationflags", 0) | windows_hide_flags()
    fetch_filter = _partial_clone_filter(repo_root, **run_kwargs)
    converts = fetch_filter is None and shallow and bool(_batch_missing_parents(
        repo_root, shallow_path.read_text(encoding="utf-8-sig").split()))
    if converts:
        fetch_filter = "blob:none"
    try:
        run_git(
            ["git"], ["fetch", "--quiet", *(["--unshallow"] if shallow else []),
             *([f"--filter={fetch_filter}"] if fetch_filter else []),
             "--no-tags", "origin", "refs/tags/v*:refs/tags/v*", *extra_refspecs],
            cwd=str(repo_root), check=True, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=900, **run_kwargs,
        )
    finally:
        # git writes the partial-clone config before it fetches, so a failed fetch converts too.
        if converts:
            mark_unmarked_packs_promisor(repo_root)
            disable_tree0_auto_maintenance(repo_root)
    return shallow


# git 2.53+ promisor fetches run index-pack --promisor, whose repack_local_links() BUG()s in
# pack-objects (should_include_obj) when an object outside the promisor packs leads to a
# promisor-missing one (#124272). The unmarked packs stay, so fetches keep dying until they are
# marked. The assertion is the fingerprint; a terminator line confirms the helper aborted. Those
# lines vary by build (POSIX "died of signal 6", Windows "could not finish pack-objects to repack
# local links", git 2.55 "fetch-pack: invalid index-pack output", #125138), so a new git wording
# needs a new terminator here.
_PACK_OBJECTS_CRASH_FINGERPRINT = "should_include_obj should only be called on existing objects"
_PACK_OBJECTS_CRASH_TERMINATORS = (
    "pack-objects died of signal 6",
    "could not finish pack-objects to repack local links",
    "fetch-pack: invalid index-pack output",
)


def is_partial_clone_pack_objects_crash(stderr: str) -> bool:
    """True when a fetch failure is the git 2.53+ partial-clone pack-objects BUG (#124272)."""
    text = stderr or ""
    if _PACK_OBJECTS_CRASH_FINGERPRINT not in text:
        return False
    return any(terminator in text for terminator in _PACK_OBJECTS_CRASH_TERMINATORS)


def fetch_with_partial_clone_recovery(runner: Callable[..., subprocess.CompletedProcess],
                                      git_cmd: list[str], fetch_args: list[str],
                                      repo_root: Path) -> subprocess.CompletedProcess:
    """Run a fetch; on the pack-objects BUG, mark the unmarked packs and retry it once.

    ``runner(git_cmd, args) -> CompletedProcess`` and ``git_cmd + fetch_args`` is the plain
    fetch argv. On the crash, ``repo_root``'s unmarked packs get a ``.promisor`` file and the
    identical fetch runs once more. The retry's result is returned whatever its exit code, so the caller keeps its
    normal failure handling. (A ``-c remote.origin.promisor=`` override does not help: git
    registers promisor remotes additively, so the repo's own ``true`` still wins.)
    """
    result = runner(git_cmd, fetch_args)
    if result.returncode == 0 or not is_partial_clone_pack_objects_crash(getattr(result, "stderr", "") or ""):
        return result
    mark_unmarked_packs_promisor(repo_root)
    logger.info("pack-objects crash on a partial clone; retrying the fetch")
    return runner(git_cmd, fetch_args)


def convert_treeless_checkout(repo_root: Path, **run_kwargs) -> bool:
    """Turn a treeless (``tree:0``) checkout into the blobless layout installers now make (#129712).

    A treeless checkout holds no trees, and git asks for a missing tree without telling the server
    which ones it already has, so every checkout and path-filtered history walk downloads complete
    directory snapshots again: hundreds of GB on some installs. One ``--refetch`` of the clone's
    refspec and its tags brings every commit and tree (about 120 MB for this repo); file contents
    stay on demand. The checked-out commit can sit outside both (a branch fetched by hand, a
    release ref the refspec does not name), so its history is checked offline and refetched by
    commit when trees are still missing. The new filter is recorded only once that history is
    whole: a failed or partial conversion leaves the checkout treeless, so the next update
    retries. That check, not the fetch's exit status, is the verdict: a refused tag update (a
    local tag that would be clobbered) fails the fetch after its objects have landed. Git before
    2.36 has no ``--refetch`` and is left as it is. Returns whether it converted; fetch failures
    raise subprocess errors.
    """
    run_kwargs["creationflags"] = run_kwargs.get("creationflags", 0) | windows_hide_flags()
    if _partial_clone_filter(repo_root, **run_kwargs) != "tree:0" or _git_version(**run_kwargs) < (2, 36):
        return False
    # A fetch spawns a detached gc/maintenance that would repack the whole refetch outside the
    # update's time limit; the per-command keys leave the user's own settings alone.
    refetch = ["git", "-c", "gc.auto=0", "-c", "maintenance.auto=false", "fetch", "--quiet", "--refetch",
               "--filter=blob:none", "origin"]
    fetch_kwargs = dict(cwd=str(repo_root), capture_output=True, text=True,
                        encoding="utf-8", errors="replace", timeout=900, **run_kwargs)
    subprocess.run([*refetch, "--tags"], check=False, **fetch_kwargs)
    if not _history_trees_complete(repo_root, **run_kwargs):
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo_root), check=True,
                              capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30,
                              **run_kwargs).stdout.strip()
        subprocess.run([*refetch, head], check=True, **fetch_kwargs)
        if not _history_trees_complete(repo_root, **run_kwargs):
            raise subprocess.CalledProcessError(1, refetch, stderr="the checked-out history is still missing trees")
    # A refetch into an existing partial clone leaves its configured filter alone; record the
    # new one only now, so a failed or interrupted refetch is retried by the next update.
    subprocess.run(
        ["git", "config", "remote.origin.partialclonefilter", "blob:none"],
        cwd=str(repo_root), check=True, capture_output=True, timeout=30, **run_kwargs,
    )
    return True


def convert_treeless_checkout_first(repo_root: Path) -> None:
    """Run the one-time conversion as the update's first new-code step. Never raises.

    Both update hand-offs (``update_completion._prepare`` and an older updater's
    ``_update_takeover.prepare``) call this before minutes of dependency work: a Desktop built before
    the fix keeps walking history during that window, and on a treeless checkout every walk
    downloads trees again (#129514: 434 GB, disk full mid-update). Runs in the dependency-free
    bootstrap interpreter.
    """
    from runtime.git_subprocess import no_prompt_git_kwargs

    try:
        if convert_treeless_checkout(repo_root, **no_prompt_git_kwargs()):
            print("  ✓ Fetched this checkout's directory history once; updates stop re-downloading it",
                  flush=True)
    except (OSError, subprocess.SubprocessError) as exc:
        detail = (getattr(exc, "stderr", None) or str(exc)).strip().splitlines()[-1:] or [type(exc).__name__]
        print(f"  ⚠ Could not fetch this checkout's directory history ({detail[0]}); retrying next update",
              flush=True)


def _git_version(**run_kwargs) -> tuple:
    out = subprocess.run(["git", "--version"], capture_output=True, text=True, encoding="utf-8",
                         errors="replace", timeout=30, **run_kwargs).stdout
    match = re.search(r"(\d+)\.(\d+)", out)
    return (int(match[1]), int(match[2])) if match else (0, 0)


def _history_trees_complete(repo_root: Path, **run_kwargs) -> bool:
    """Whether every tree in HEAD's history is local, checked without fetching (~3 s for this repo)."""
    env = dict(run_kwargs.pop("env", None) or noninteractive_git_env(), GIT_NO_LAZY_FETCH="1")
    walk = subprocess.run(
        ["git", "rev-list", "--objects", "--filter=blob:none", "--missing=print", "HEAD"],
        cwd=str(repo_root), capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=300, env=env, **run_kwargs,
    )
    return walk.returncode == 0 and not any(line.startswith("?") for line in walk.stdout.splitlines())


def settle_partial_clone_maintenance(repo_root: Path) -> None:
    """Persist the commit-graph-off keys on a partial clone; a full clone keeps stock maintenance.

    Runs on every update so checkouts that predate the installer change converge. Never raises.
    """
    try:
        if _partial_clone_filter(repo_root, creationflags=windows_hide_flags()) is not None:
            disable_tree0_auto_maintenance(repo_root)
    except Exception:
        logger.debug("partial-clone maintenance settings failed for %s", repo_root, exc_info=True)

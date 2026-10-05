"""Durable record of the gateways and services ``hermes update`` paused on Windows.

The pause token used to live only in the updater's memory (plus an ``atexit`` resume), so a
killed updater (console closed, ``taskkill``, power loss) left every paused gateway and SCM
service stopped with nothing on disk saying so. The record is written under the ROOT Hermes
home BEFORE anything is stopped, names its owner (pid + creation time, the update-marker
identity format), is rewritten as the token changes, and is removed only after a verified
resume. A later launch whose owner check finds the updater dead (and no update live) resumes it.

One record per checkout (the file name carries a key of the install root): a checkout sharing
the home never imports, relabels or certifies another checkout's paused set — its tree gate
says nothing about the other tree.

Resuming is gated on a whole tree: no interrupted-pull marker; at the pre-update HEAD, no tracked
change beyond the ones present at pause time (git died before moving HEAD); at a moved HEAD,
dependencies current for it where a launch syncs them (a build step may rewrite tracked files there). Every pause folded into
a record keeps its own baseline (``baselines``): each one taken at the current HEAD must hold.
Otherwise the record stays for the next launch, which runs after the interrupted-pull restore and
the dependency sync.

Custody: every mutation (write, claim, retire, discharge) happens while holding a kernel lock on
``<record dir>/.hermes-update-paused-gateways.lock`` (flock / msvcrt, released by the kernel when
the holder dies; the file is never deleted), and the read → judge → mutate decision is made
inside that hold. A recovering launch claims a record by publishing ``<record>.<pid>.<nonce>.claim``
with its own identity already inside (atomic replace), then retires the source; a crash between the
two leaves two files with ONE obligation id (``pause_id``), and an update that folds orphaned sets
into its own record lists their ids under ``absorbed`` before retiring them. Readers treat a file
whose id another file carries as a copy, never as a second obligation, and the next mutator
retires it. Completing an obligation first adds its ids to ``<record stem>.retired`` (atomic), then
unlinks its files: an unlink Windows refuses (a reader holding the file without
FILE_SHARE_DELETE) leaves a copy that is redundant by id, never an obligation that executes again.
The list is read only under the mutex, and only its absence means "nothing retired": a list that
cannot be read means unknown, so nothing is claimed and the list is never rewritten from that read.
A list that reads but does not parse is unknown too while any record or claim exists; with none
left it guards nothing and is set aside as ``<record stem>.corrupt``.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from contextlib import contextmanager, suppress
from pathlib import Path

RECORD_STEM = ".hermes-update-paused-gateways"
MUTEX_NAME = RECORD_STEM + ".lock"
_MUTEX_WAIT_S = 10.0
# Re-entry depth per THREAD: another thread of this process (a gateway's consumer, a watcher)
# must wait for the kernel lock like any other holder, never ride on this thread's hold.
_mutex_held = threading.local()


def install_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _install_key(root: Path | None = None) -> str:
    return hashlib.sha256(os.path.normcase(str(root or install_root())).encode("utf-8")).hexdigest()[:12]


def record_path() -> Path:
    from hermes_constants import get_default_hermes_root
    return get_default_hermes_root() / f"{RECORD_STEM}.{_install_key()}.json"


def identity(pid: int | None = None) -> dict:
    """``{"pid", "ct"}`` with ``ct`` spelled like the update marker's line 3 (``ct:<s.3f>``).

    The creation time comes from the probe :func:`identity_is_live` judges with
    (``update_lock.process_create_time``: psutil, else stdlib), so a host without psutil still
    writes a provable identity instead of ``ct=None``."""
    from hermes_cli import update_lock
    pid = os.getpid() if pid is None else int(pid)
    created = update_lock.process_create_time(pid)
    return {"pid": pid, "ct": None if created is None else f"ct:{created:.3f}"}


def identity_is_live(ident: dict | None) -> bool:
    """The update marker's incarnation rule (``update_lock.incarnation_live``) for a recorded identity.

    Unprovable (alive, no readable creation time) counts as live: a record is never taken from, nor
    a paused process restarted over, a process that may still be the one recorded.
    """
    from hermes_cli import update_lock
    ident = ident or {}
    return update_lock.incarnation_live(ident.get("pid") or 0, ident.get("ct") or None) is not False


class RecordConflict(OSError):
    """The record names another pause whose owner is still live."""


class RecordBusy(OSError):
    """Another process held the record mutex past the bounded wait."""


class RetiredUnknown(OSError):
    """The retired-id list exists but cannot be read: which obligations are complete is unknown."""


@contextmanager
def _mutex(wait_s: float = _MUTEX_WAIT_S):
    """Exclusive kernel lock on the record directory's sidecar (A7). Re-entrant per thread; each
    thread's first entry opens its own descriptor, so threads exclude each other like processes."""
    depth = getattr(_mutex_held, "depth", 0)
    if depth:
        _mutex_held.depth = depth + 1
        try:
            yield
        finally:
            _mutex_held.depth -= 1
        return
    from hermes_cli import update_lock
    path = record_path().with_name(MUTEX_NAME)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0), 0o644)
    try:
        deadline = time.monotonic() + wait_s
        while not update_lock._try_lock(fd):
            if time.monotonic() > deadline:
                raise RecordBusy(f"{path} is held by another process or thread")
            time.sleep(0.05)
        _mutex_held.depth = 1
        try:
            yield
        finally:
            _mutex_held.depth = 0
            update_lock._unlock(fd)
    finally:
        os.close(fd)


def _git_executable() -> str:
    """The git ``hermes update`` runs, found without installing anything: PATH's, else the copy
    install.ps1 staged in PM's store (``_subprocess_compat.expose_pm_git`` puts it on PATH only
    inside the updater). A bare ``git`` on such an install dies with WinError 2, so every gate
    read would fail and the paused gateways would never be restarted."""
    found = shutil.which("git")
    if found:
        return found
    with suppress(Exception):  # health: allow BLE001 -- no PM, no store, no Windows git package: PATH's answer ("git") stands
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


def _git(root: Path, *args: str) -> subprocess.CompletedProcess | None:
    """A read-only git query in the updater's custody (``update_custody.run_git``: job-bound on
    Windows inside an update, custody config in argv); ``None`` when git cannot run."""
    from hermes_cli.update_custody import run_git
    try:
        return run_git([_git_executable(), "-C", str(root)], args, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL, timeout=30, check=False)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def head_sha(root: Path) -> str | None:
    result = _git(root, "rev-parse", "--verify", "HEAD")
    if result is None or result.returncode != 0:
        return None
    return result.stdout.strip() or None


def tracked_changes(root: Path) -> list[str] | None:
    """Tracked paths that differ from HEAD (index or worktree); ``None`` when git cannot say."""
    result = _git(root, "status", "--porcelain=v1", "-z", "--untracked-files=no")
    if result is None or result.returncode != 0:
        return None
    paths = set()
    entries = result.stdout.split("\0")
    index = 0
    while index < len(entries):
        entry = entries[index]
        index += 1
        if len(entry) < 4:
            continue
        paths.add(entry[3:])
        if entry[0] in "RC":  # rename/copy: the source path follows as its own entry
            index += 1
    return sorted(paths)


def stamp_tree(token: dict, root: Path | None = None) -> dict:
    """Record what "the tree before this update" was, for the whole-tree gate: one baseline per
    pause (idempotent for this token's ``pause_id``), next to the baselines of the pauses it absorbed."""
    root = root or install_root()
    pause_id = token.setdefault("pause_id", uuid.uuid4().hex)
    baselines = token.setdefault("baselines", [])
    if (root / ".git").exists() and not any(b.get("pause_id") == pause_id for b in baselines):
        baselines.append({"pause_id": pause_id, "pre_sha": head_sha(root), "dirty_at_pause": tracked_changes(root)})
    return token


_MAX_MOVE_TARGETS = 16


def _paths_a_move_could_write(root: Path) -> set[str] | None:
    """Tracked paths an update's checkout move could have written while HEAD stayed put: the ones
    that differ between HEAD and a commit the checkout fetched (``FETCH_HEAD``) or tracks
    (``refs/remotes``) — a fast-forward, reset or merge only writes those. ``None`` when git cannot
    say or nothing was fetched (the move's target is then unknown)."""
    where = _git(root, "rev-parse", "--git-path", "FETCH_HEAD")
    refs = _git(root, "for-each-ref", "--format=%(objectname)", "refs/remotes")
    if where is None or where.returncode != 0 or refs is None or refs.returncode != 0:
        return None
    try:
        fetched = (root / where.stdout.strip()).read_text(encoding="utf-8-sig", errors="replace").split("\n")
    except FileNotFoundError:
        fetched = []
    except OSError:
        return None
    targets = {line.split()[0] for line in fetched if line.split()} | set(refs.stdout.split())
    if not targets or len(targets) > _MAX_MOVE_TARGETS:
        return None
    paths: set[str] = set()
    for target in sorted(targets):
        diff = _git(root, "diff", "--name-only", "-z", "--no-renames", "HEAD", target, "--")
        if diff is None or diff.returncode != 0:
            return None
        paths.update(filter(None, diff.stdout.split("\0")))
    return paths


def _half_written(root: Path, changes: list[str], at_head: list[dict]) -> list[str]:
    """Tracked changes at an unmoved HEAD that git may have left half-written, for the first
    baseline that has any. A change the pause did not see counts only on a path the update's move
    could write: a build/sync step (or the user) rewriting any other tracked file is not git's
    doing, and holding the set for it would keep gateways stopped until someone cleans the file.
    When the move's target is unknown every unseen change still counts."""
    writable: set[str] | None = None
    probed = False
    for baseline in at_head:
        unexpected = sorted(set(changes) - set(baseline.get("dirty_at_pause") or []))
        if unexpected and not probed:
            writable, probed = _paths_a_move_could_write(root), True
        torn = unexpected if writable is None else [path for path in unexpected if path in writable]
        if torn:
            return torn
    return []


def _deps_hold_resume(root: Path) -> bool:
    """At a moved HEAD: are stale dependencies a reason to keep the paused set stopped?

    Only where waiting converges. A self-managed install (``updateMechanism: self``) syncs its
    dependencies at the start of every launch (``venv_sync.prepare_launch``), before startup
    recovery runs, so a deferred set comes back on the next launch. Anywhere else nothing a launch
    does makes them current (a developer checkout owns its own venv): the set resumes now, as it
    did before the pause was durable, and a currency the probe cannot establish never holds it
    either — the restarted gateway's own launch syncs or reports the remedy.
    """
    if (os.environ.get("HERMES_DISABLE_LAZY_INSTALLS", "").lower() in ("1", "true", "yes")
            or not (root / ".git").exists() or not (root / "pyproject.toml").is_file()):
        return False  # prepare_launch skips the sync: waiting would never end
    try:
        from hermes_cli.steward import read_install_stamp
        if read_install_stamp(root).get("updateMechanism") != "self":
            return False
        import pm
        return not pm.venv_is_current(project_root=root)
    except Exception:  # health: allow BLE001 -- unknown currency must not strand the set: resume, the gateway's own launch syncs or names the remedy
        return False


def tree_is_whole(token: dict, root: Path | None = None) -> tuple[bool, str]:
    """May paused gateways start on this checkout now? ``(verdict, reason when not)``."""
    from hermes_cli._early_recovery import interrupted_pull_marker
    root = root or install_root()
    if (root / ".git").exists():
        if interrupted_pull_marker(root).exists():
            return False, "the checkout is mid-pull (interrupted-pull marker present)"
        head = head_sha(root)
        if head is None:
            return False, "the checkout HEAD is unreadable"
        at_head = [b for b in token.get("baselines") or [] if b.get("pre_sha") == head]
        if at_head:
            # HEAD never moved for these pauses: a tracked change one of them did not see is git's
            # half-written checkout. Each is judged on its own set — a killed run that moved HEAD
            # elsewhere vouches for nothing here, and this run's set never certifies an older one's.
            changes = tracked_changes(root)
            if changes is None:
                return False, "git cannot read the checkout state"
            torn = _half_written(root, changes, at_head)
            if torn:
                return False, f"the checkout has {len(torn)} file(s) git left half-written (e.g. {torn[0]})"
            return True, ""
    if _deps_hold_resume(root):
        return False, "dependencies are not current for the updated code yet"
    return True, ""


def _atomic_write(path: Path, body: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(body, fh, indent=1, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        # A refused replace (a Windows reader without FILE_SHARE_DELETE) or a failed write must
        # not leave the half-published temp beside the record; the error itself still propagates.
        with suppress(OSError):
            tmp.unlink()
        raise


def read(path: Path | None = None) -> dict | None:
    path = path or record_path()
    try:
        body = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    return body if isinstance(body, dict) and isinstance(body.get("token"), dict) else None


UNOWNED = {"pid": 0, "ct": None}


def write(token: dict, *, owner: dict | None = None, path: Path | None = None) -> None:
    """Persist *token*. The owner of the same pause is kept; another pause's orphaned set is merged
    into *token* (in place, so this run's resume brings it back); a live one refuses."""
    path = path or record_path()
    with _mutex():
        existing = read(path)
        same = existing is not None and existing["token"].get("pause_id") == token.get("pause_id")
        if existing is not None and not same:
            if existing.get("install_root") != str(install_root()):
                raise RecordConflict(f"{path} holds gateways paused for another checkout ({existing.get('install_root')})")
            if identity_is_live(existing.get("owner")):
                raise RecordConflict(f"{path} holds gateways paused by live process {existing['owner'].get('pid')}")
            merge_into(token, drop_never_stopped(dict(existing["token"])))
        if owner is None:
            owner = existing["owner"] if same and existing is not None else identity()
        _atomic_write(path, {"schema": 1, "owner": owner, "install_root": str(install_root()), "token": token})


def mark_stop_requested(token: dict, pids, markers: dict | None = None) -> None:
    """Persist, BEFORE the first stop request, the stops this update intends and where each mapped
    gateway's request (its planned-stop marker) will be written. Intent is not a request: a live
    process counts as asked only once the producer or consumer checkpoints it, or while the
    valid request this update wrote is still on disk (:func:`_asked`)."""
    token["stop_requested"] = sorted({*map(str, token.get("stop_requested") or []), *(str(int(p)) for p in pids)})
    token["stop_sent"] = sorted(map(str, token.get("stop_sent") or []))
    token["stopper_pid"] = os.getpid()
    token["stop_markers"] = {**(token.get("stop_markers") or {}), **{str(int(p)): str(m) for p, m in (markers or {}).items()}}
    write(token, owner=identity())


def mark_stop_sent(token: dict, pid) -> None:
    """Persist that *pid*'s stop request was issued (its planned-stop marker is on disk): from here a
    live *pid* is draining and keeps its restart debt until it exits. Best effort: the marker itself
    is evidence until this lands; its consumer checkpoints before unlinking it."""
    token["stop_sent"] = sorted({*map(str, token.get("stop_sent") or []), str(int(pid))})
    try:
        write(token, owner=identity())
    except OSError as exc:
        print(f"  ⚠ Could not record the stop request for gateway {pid}: {exc}")


def mark_stop_consumed(path: Path, marker: dict) -> None:
    """Checkpoint a validated gateway consume BEFORE its marker disappears.

    Called under the pause mutex by the consumer. Preserve the holder's identity;
    this is evidence of a stop, not a transfer of the updater's custody.
    """
    from gateway.status import _same_hermes_home
    pid = str(marker["target_pid"])
    for src, body in _files(record_path()):
        token = body["token"]
        recorded_path = (token.get("stop_markers") or {}).get(pid)
        if (str(token.get("stopper_pid")) != str(marker.get("stopper_pid"))
                or pid not in (token.get("stop_requested") or [])
                or not recorded_path or not _same_hermes_home(recorded_path, path)):
            continue
        token["stop_sent"] = sorted({*map(str, token.get("stop_sent") or []), pid})
        _atomic_write(src, body)  # Failure leaves the request intact for recovery.


def discharge(token: dict, path: Path | None = None) -> None:
    """Remove the record only while it still names this pause (decided under the mutex)."""
    path = path or record_path()
    with _mutex():
        existing = read(path)
        if existing is not None and existing["token"].get("pause_id") == token.get("pause_id"):
            _retire(path, [(path, existing)])


def sync(token: dict) -> None:
    """Mirror the token after a resume attempt: done → delete; anything still owed → rewrite."""
    if not token.get("pause_id") or token.get("recovery"):
        return  # a recovering launch keeps what it owes in its own claim (_resume_claimed)
    try:
        if token.get("resume_needed") or token.get("resume_deferred"):
            owed = {**token, "resume_needed": True}
            owed.pop("resume_deferred", None)
            write(owed)
        else:
            discharge(token)
    except OSError as exc:
        print(f"  ⚠ Could not update the paused-gateway record {record_path()}: {exc}")


def _live_update_elsewhere() -> bool:
    """A live update other than the one this process belongs to. The update tree holding the
    checkout lock (this process plus the partner whose marker it adopted) is not "elsewhere"."""
    from hermes_cli import update_lock
    root = install_root()
    return not update_lock.holds_checkout_lock(root) and update_lock.update_in_progress(root)


def _claims(path: Path) -> list[Path]:
    return sorted(path.parent.glob(f"{glob.escape(path.name)}.*.claim"))


def _retired_path(path: Path) -> Path:
    return path.with_suffix(".retired")


def _retired(path: Path) -> set[str]:
    """The completed obligation ids (call under the mutex). Only an absent list is empty: one that
    exists but cannot be read raises :class:`RetiredUnknown`, so a completed obligation is never
    claimed again and the list is never rewritten from a read that failed."""
    target = _retired_path(path)
    try:
        text = target.read_text(encoding="utf-8-sig", errors="replace")
    except FileNotFoundError:
        return set()
    except OSError as exc:  # a refused read (sharing violation, AV scanner): unknown
        raise RetiredUnknown(f"cannot read the retired paused-gateway list {target}: {exc}") from exc
    try:
        ids = json.loads(text).get("ids")
    except (ValueError, AttributeError):  # not JSON / not an object: malformed, like a non-list
        ids = None
    if not isinstance(ids, list):
        return _quarantine_retired(path, target)
    return {str(i) for i in ids}


def _quarantine_retired(path: Path, target: Path) -> set[str]:
    """A retired list that reads but does not parse. While a record or claim of this checkout
    exists the list may be the only proof it is complete: unknown, fail closed. With none left it
    guards nothing, and keeping it would abort every Windows update: set it aside and go on."""
    if path.exists() or _claims(path):
        raise RetiredUnknown(f"the retired paused-gateway list {target} is malformed")
    corrupt = target.with_suffix(".corrupt")
    try:
        os.replace(target, corrupt)
    except OSError as exc:
        raise RetiredUnknown(f"cannot set aside the malformed retired paused-gateway list {target}: {exc}") from exc
    print(f"  ⚠ Set aside a malformed paused-gateway history (no paused set refers to it): {corrupt}",
          file=sys.stderr)
    return set()


def _retire(path: Path, carriers: list[tuple[Path, dict]]) -> None:
    """Complete the obligations *carriers* hold (call under the mutex): their ids (and every id
    they absorbed) go on the durable retired list FIRST; a failed read or write raises and deletes
    nothing. Then the files go; one that cannot be deleted is a redundant copy from here on."""
    ids = {str(i) for _src, body in carriers
           for i in [body["token"].get("pause_id"), *(body["token"].get("absorbed") or [])] if i}
    retired = _retired(path)
    if not ids <= retired:
        _atomic_write(_retired_path(path), {"schema": 1, "ids": sorted(retired | ids)})
    for src, _body in carriers:
        with suppress(OSError):
            src.unlink()


def _holder(src: Path, body: dict) -> dict:
    """Who holds *src*: a record's owner, a claim's claimer (published with the claim itself)."""
    held_by = body.get("owner") if src.suffix == ".json" else body.get("claimer")
    return held_by if isinstance(held_by, dict) else UNOWNED


def _files(path: Path) -> list[tuple[Path, dict]]:
    """This checkout's record and claims, readable ones only."""
    found = []
    for src in (path, *_claims(path)):
        body = read(src)
        if body is not None and body.get("install_root") == str(install_root()):
            found.append((src, body))
    return found


def _redundant(found: list[tuple[Path, dict]], held: set[Path], retired: set[str] = frozenset()) -> set[Path]:
    """Files that only copy an obligation another file carries or that was completed: a source an
    update absorbed but died before retiring, one of two same-id files a claim transfer left (the
    furthest-progressed copy is kept), a retired id's leftover. Never a held file."""
    carriers: dict[str, Path] = {}
    for src, body in found:
        for oid in body["token"].get("absorbed") or []:
            carriers.setdefault(str(oid), src)
    redundant: set[Path] = set()
    by_id: dict[str, list[Path]] = {}
    for src, body in found:
        oid = str(body["token"].get("pause_id") or "")
        if not oid:
            continue
        if oid in retired or (oid in carriers and carriers[oid] != src):
            redundant.add(src)
        else:
            by_id.setdefault(oid, []).append(src)
    rev = {src: int(body.get("rev") or 0) for src, body in found}
    for copies in by_id.values():
        keep = next((s for s in copies if s in held), max(copies, key=lambda s: rev[s]))
        redundant.update(s for s in copies if s != keep)
    return redundant - held


def _survey(path: Path) -> tuple[list[tuple[Path, dict]], set[Path], set[Path]]:
    found = _files(path)
    held = {src for src, body in found if identity_is_live(_holder(src, body))}
    return found, held, _redundant(found, held, _retired(path))


def orphans(path: Path | None = None) -> list[tuple[Path, dict]]:
    """``(file, body)`` for every obligation of this checkout whose holder is dead — one file per
    obligation id; empty while another update is live (it adopts them itself). Surveyed under the
    mutex (a mutator replaces the retired list meanwhile); raises when the list cannot be read."""
    path = path or record_path()
    if not path.exists() and not _claims(path):
        return []
    with _mutex():
        found, held, redundant = _survey(path)
    found = [(src, body) for src, body in found if src not in held and src not in redundant]
    if not found or _live_update_elsewhere():
        return []
    return found


def orphaned_record(path: Path | None = None) -> dict | None:
    found = orphans(path)
    return found[0][1] if found else None


def retire_redundant(path: Path | None = None) -> None:
    """Delete copies of obligations another file already carries (crash leftovers of a transfer)."""
    path = path or record_path()
    if not path.exists() and not _claims(path) and not _retired_path(path).exists():
        return
    with suppress(RecordBusy), _mutex():
        _found, _held, redundant = _survey(path)
        for src in redundant:
            with suppress(OSError):
                src.unlink()
        _prune_retired(path)


def _prune_retired(path: Path) -> None:
    """Forget retired ids no file carries any more — only when every candidate file was readable
    (one Windows would not let us read may still carry a retired id)."""
    retired = _retired(path)
    candidates = [p for p in (path, *_claims(path)) if p.exists()]
    bodies = [read(p) for p in candidates]
    if not retired or None in bodies:
        return
    carried = {str(b["token"].get("pause_id")) for b in bodies}
    keep = retired & carried
    with suppress(OSError):
        if keep:
            _atomic_write(_retired_path(path), {"schema": 1, "ids": sorted(keep)})
        else:
            _retired_path(path).unlink()


def claim(src: Path) -> tuple[Path, dict] | None:
    """Take *src* (an orphaned record or a dead claimer's claim) for this process; ``None`` when a
    concurrent launch won it or holds the mutex. The claim is published with our identity inside,
    then the source is retired; a crash between leaves a same-id copy :func:`_redundant` drops."""
    path = record_path()
    mine = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.claim")
    try:
        with _mutex():
            body = read(src)
            if body is None or body.get("install_root") != str(install_root()):
                return None
            _found, held, redundant = _survey(path)
            if src in held or src in redundant or src not in {s for s, _ in _found}:
                return None
            body["claimer"] = identity()
            body["rev"] = int(body.get("rev") or 0) + 1  # a copy left behind is never the newer one
            _atomic_write(mine, body)
            with suppress(OSError):
                src.unlink()
    except OSError:
        return None
    return mine, body


def adopt_orphans() -> tuple[dict | None, list[Path]]:
    """For ``hermes update``: claim every orphaned pause and merge their sets. The caller records the
    merged set durably (its ``absorbed`` ids name the claims), then :func:`release_claims`."""
    retire_redundant()
    adopted, claims = None, []
    for src, _body in orphans():
        won = claim(src)
        if won is None:
            continue
        claims.append(won[0])
        with _mutex():
            body = read(won[0]) or won[1]
            adopted = merge_into(adopted, drop_never_stopped(dict(body["token"])))
    return adopted, claims


def release_claims(claims: list[Path]) -> None:
    with suppress(RecordBusy), _mutex():
        for path in claims:
            with suppress(OSError):
                path.unlink()


def record_pause(token: dict, adopted: dict | None, claims: list[Path]) -> dict:
    """Merge the adopted set, stamp the tree and persist under this process BEFORE the first stop.
    Publish then retire: a crash between leaves claims whose ids the record lists as absorbed."""
    if adopted is not None:
        merge_into(token, adopted)
    stamp_tree(token)
    token.setdefault("stop_requested", [])
    write(token, owner=identity())
    release_claims(claims)
    return token


_CARRIED = ("pause_id", "baselines", "identities", "stop_requested", "stop_sent", "stopper_pid",
            "stop_markers", "absorbed")


def finish_pause(token: dict, intended: dict, adopted: dict | None) -> dict:
    """Carry the recorded pause identity onto the final token and mirror it to disk."""
    if adopted is not None:
        merge_into(token, adopted)
    token.update({key: intended[key] for key in _CARRIED if key in intended})
    sync(token)
    return token


def abandon_pause(intended: dict, adopted: dict | None) -> None:
    """This run's own stops were rolled back in-line: only an adopted orphan set is still owed,
    and it goes back on disk unowned for the next launch."""
    if adopted is None:
        discharge(intended)
        return
    carried = {key: intended[key] for key in ("pause_id", "baselines", "absorbed") if key in intended}
    write({**adopted, **carried}, owner=UNOWNED)


def _live_pids(token: dict) -> set[str]:
    return {str(pid) for pid, ct in (token.get("identities") or {}).items()
            if identity_is_live({"pid": pid, "ct": ct})}


def _without(token: dict, pids: set[str]) -> dict:
    token["profiles"] = {p: pid for p, pid in (token.get("profiles") or {}).items() if str(pid) not in pids}
    token["unmapped"] = [u for u in (token.get("unmapped") or []) if str(u.get("pid")) not in pids]
    return token


def _request_on_disk(token: dict, pid: str) -> bool:
    """The planned-stop marker this update's stopper wrote for *pid* is still on disk (the gateway's
    watcher has not consumed it yet): the request was issued even if the updater died before
    :func:`mark_stop_sent`. A marker naming another stopper (a user's ``hermes gateway stop``) is not."""
    path = (token.get("stop_markers") or {}).get(str(pid))
    if not path:
        return False
    from gateway.status import _marker_is_stale, _PLANNED_STOP_MARKER_TTL_S, get_process_start_time
    try:
        marker = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if (int(marker["target_pid"]) != int(pid)
                or int(marker["stopper_pid"]) != int(token["stopper_pid"])
                or _marker_is_stale(marker.get("written_at") or "", _PLANNED_STOP_MARKER_TTL_S)):
            return False
        expected, actual = marker.get("target_start_time"), get_process_start_time(int(pid))
        # Match the consumer's optional birth fingerprint, including unavailable clocks.
        return None in (expected, actual) or expected == actual
    except (OSError, ValueError, TypeError, KeyError):
        return False


def _asked(token: dict) -> set[str]:
    """Pids this update actually asked to stop: recorded as sent, or whose request is still on disk."""
    sent = {str(p) for p in token.get("stop_sent") or []}
    return sent | {str(p) for p in token.get("stop_requested") or [] if str(p) not in sent and _request_on_disk(token, str(p))}


def drop_never_stopped(token: dict) -> dict:
    """Drop entries whose recorded process is the same live incarnation AND was never actually asked
    to stop (the update died before issuing its request, even with the intent on disk): it is still
    serving, and a later exit — a user's ``hermes gateway stop`` included — is not this update's to
    undo. One it asked may be draining: it keeps its restart debt (:func:`split_draining`). The
    evidence is resolved here, once, into ``stop_sent`` (a merge carries it under another stopper).
    No stop record at all (a set from before stop tracking): nothing can be told apart."""
    if token.get("stop_requested") is None:
        return token
    asked = _asked(token)
    token["stop_sent"] = sorted(asked)
    return _without(token, _live_pids(token) - asked)


def split_draining(token: dict) -> dict:
    """Remove and return ``{"profiles", "unmapped"}`` whose recorded process is still running: it was
    asked to stop and has not exited yet, so it can only be restarted once it has."""
    live = _live_pids(token)
    draining = {"profiles": {p: pid for p, pid in (token.get("profiles") or {}).items() if str(pid) in live},
                "unmapped": [u for u in (token.get("unmapped") or []) if str(u.get("pid")) in live]}
    _without(token, live)
    return draining


def merge_into(token: dict | None, adopted: dict) -> dict:
    """Fold an orphaned pause into this update's token so its resume brings both sets back; its
    obligation id (and every id it absorbed) is recorded so a leftover copy is never resumed twice."""
    token = token if token is not None else {"resume_needed": True, "profiles": {}, "unmapped_pids": [], "unmapped": []}
    token["resume_needed"] = True
    # Every pause keeps its own tree evidence: the adopted baseline stops this run's stamp from
    # certifying what a killed update left half-written as "dirty before the pause", and this
    # run's own baseline still guards the HEAD a killed update moved to.
    baselines = token.setdefault("baselines", [])
    baselines.extend(b for b in adopted.get("baselines") or [] if b not in baselines)
    absorbed = token.setdefault("absorbed", [])
    absorbed.extend(i for i in [adopted.get("pause_id"), *(adopted.get("absorbed") or [])] if i and i not in absorbed)
    profiles = token.setdefault("profiles", {})
    for name, pid in (adopted.get("profiles") or {}).items():
        profiles.setdefault(name, pid)
    unmapped = token.setdefault("unmapped", [])
    unmapped.extend(u for u in adopted.get("unmapped") or [] if u not in unmapped)
    identities = token.setdefault("identities", {})
    for pid, ct in (adopted.get("identities") or {}).items():
        identities.setdefault(pid, ct)
    if "stop_requested" in adopted or "stop_requested" in token:
        requested, sent = adopted.get("stop_requested"), adopted.get("stop_sent")
        if requested is None:  # a set with no stop record keeps every live entry: all count as asked
            requested = sent = list(adopted.get("identities") or {})
        token["stop_requested"] = sorted({*map(str, token.get("stop_requested") or []), *map(str, requested)})
        token["stop_sent"] = sorted({*map(str, token.get("stop_sent") or []), *map(str, sent or [])})
    cold = token.setdefault("cold_start_profiles", {})
    for name, generation in (adopted.get("cold_start_profiles") or {}).items():
        cold.setdefault(name, generation)
    if adopted.get("cold_start_if_installed"):
        token["cold_start_if_installed"] = True
        if "attested_generation" in adopted:
            token.setdefault("attested_generation", adopted["attested_generation"])
    services = token.setdefault("services", [])
    services.extend(s for s in adopted.get("services") or [] if s not in services)
    if services:
        token.setdefault("expected_services", []).extend(s for s in services if s not in token["expected_services"])
        token.setdefault("restarted_services", [])
        token.setdefault("service_profiles", {}).update(adopted.get("service_profiles") or {})
    return token


def _has_work(token: dict) -> bool:
    return bool(token.get("profiles") or any(u.get("argv") for u in token.get("unmapped") or [])
                or token.get("services") or token.get("cold_start_if_installed") or token.get("cold_start_profiles"))


def _resume_claimed(claim_path: Path, body: dict) -> None:
    """Resume under the checkout lock, so no update mutates the tree between the gate and the
    verified start; an update that took the checkout after our claim gets the set back untouched."""
    from hermes_cli.update_lock import UpdateLock
    fence = UpdateLock()
    if not fence.acquire_checkout(install_root()):
        with suppress(OSError), _mutex():
            _atomic_write(claim_path, {**(read(claim_path) or body), "claimer": UNOWNED})
        return
    try:
        _resume_fenced(claim_path, body)
    finally:
        fence.release()


def _resume_fenced(claim_path: Path, body: dict) -> None:
    with _mutex():
        # A live consumer may have checkpointed after claim() returned its snapshot.
        body = read(claim_path) or body
        token = drop_never_stopped(dict(body["token"]))
        draining = split_draining(token)
    token.update(resume_needed=True, recovery=True)
    try:
        if _has_work(token):
            print("→ Restarting gateway(s) paused by an interrupted `hermes update`...", file=sys.stderr)
            from hermes_cli.update_cmd_windows import _resume_windows_gateways_after_update
            _resume_windows_gateways_after_update(token)
        else:
            token["resume_needed"] = False
    except Exception as exc:  # health: allow BLE001 -- recovery boundary: any resume error is reported and the claim handed back (finally), never raised into the launch
        print(f"  ⚠ Could not restart every paused gateway: {exc}. Run `hermes update` or "
              "`hermes gateway start`.", file=sys.stderr)
    finally:
        _hand_back(claim_path, body, token, draining)


def _hand_back(claim_path: Path, body: dict, token: dict, draining: dict) -> None:
    """Retire the claim when nothing is owed; else hand it back unowned so the next launch retries
    at once — this launch may live for hours (a chat). A draining process stays owed until it has
    exited and been restarted. A failed rewrite leaves our claimer line, dead once we exit."""
    token["profiles"] = {**(token.get("profiles") or {}), **draining["profiles"]}
    token["unmapped"] = [*(token.get("unmapped") or []), *draining["unmapped"]]
    owed = bool(token.get("resume_needed") or token.get("resume_deferred") or draining["profiles"] or draining["unmapped"])
    with suppress(OSError), _mutex():
        if owed:
            kept = {key: value for key, value in token.items() if key not in ("recovery", "resume_deferred")}
            _atomic_write(claim_path, {**body, "token": {**kept, "resume_needed": True}, "claimer": UNOWNED})
        else:
            _retire(record_path(), [(claim_path, read(claim_path) or body)])


def recover(argv: list[str] | None = None) -> None:
    """Start-of-run recovery: resume every orphaned pause. Never raises."""
    from hermes_cli._parser import command_argv
    command = command_argv(list(sys.argv[1:] if argv is None else argv))
    if command[:1] == ["update"] or command[:2] == ["gateway", "run"]:
        return  # the update adopts it itself; a booting gateway must not block on its siblings
    try:
        retire_redundant()
        for src, _body in orphans():
            won = claim(src)
            if won is not None:
                _resume_claimed(*won)
    except Exception as exc:  # health: allow BLE001 -- never brick a launch on recovery; the record stays for the next one
        print(f"  ⚠ Paused-gateway recovery skipped: {exc}", file=sys.stderr)

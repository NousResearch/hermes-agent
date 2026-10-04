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
dependencies current for it (a build step may rewrite tracked files there). Otherwise the record
stays for the next launch, which runs after the interrupted-pull restore and the dependency sync.

Custody: every mutation (write, claim, retire, discharge) happens while holding a kernel lock on
``<record dir>/.hermes-update-paused-gateways.lock`` (flock / msvcrt, released by the kernel when
the holder dies; the file is never deleted), and the read → judge → mutate decision is made
inside that hold. A recovering launch claims a record by publishing ``<record>.<pid>.<nonce>.claim``
with its own identity already inside (atomic replace), then retires the source; a crash between the
two leaves two files with ONE obligation id (``pause_id``), and an update that folds orphaned sets
into its own record lists their ids under ``absorbed`` before retiring them. Readers treat a file
whose id another file carries as a copy, never as a second obligation, and the next mutator
retires it.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager, suppress
from pathlib import Path

RECORD_STEM = ".hermes-update-paused-gateways"
MUTEX_NAME = RECORD_STEM + ".lock"
_MUTEX_WAIT_S = 10.0
_mutex_depth = 0


def install_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _install_key(root: Path | None = None) -> str:
    return hashlib.sha256(os.path.normcase(str(root or install_root())).encode("utf-8")).hexdigest()[:12]


def record_path() -> Path:
    from hermes_constants import get_default_hermes_root
    return get_default_hermes_root() / f"{RECORD_STEM}.{_install_key()}.json"


def _create_time(pid: int) -> float | None:
    from hermes_cli.process_identity import _process_create_time
    return _process_create_time(pid)


def identity(pid: int | None = None) -> dict:
    """``{"pid", "ct"}`` with ``ct`` spelled like the update marker's line 3 (``ct:<s.3f>``)."""
    pid = os.getpid() if pid is None else int(pid)
    created = _create_time(pid)
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


@contextmanager
def _mutex():
    """Exclusive kernel lock on the record directory's sidecar (A7). Re-entrant in-process."""
    global _mutex_depth
    if _mutex_depth:
        _mutex_depth += 1
        try:
            yield
        finally:
            _mutex_depth -= 1
        return
    from hermes_cli import update_lock
    path = record_path().with_name(MUTEX_NAME)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0), 0o644)
    try:
        deadline = time.monotonic() + _MUTEX_WAIT_S
        while not update_lock._try_lock(fd):
            if time.monotonic() > deadline:
                raise RecordBusy(f"{path} is held by another process")
            time.sleep(0.05)
        _mutex_depth = 1
        try:
            yield
        finally:
            _mutex_depth = 0
            update_lock._unlock(fd)
    finally:
        os.close(fd)


def _git(root: Path, *args: str) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, encoding="utf-8",
            errors="replace", stdin=subprocess.DEVNULL, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
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
    """Record what "the tree before this update" was, for the whole-tree gate."""
    root = root or install_root()
    if (root / ".git").exists():
        token.setdefault("pre_sha", head_sha(root))
        token.setdefault("dirty_at_pause", tracked_changes(root))
    token.setdefault("pause_id", uuid.uuid4().hex)
    return token


def _venv_is_current(root: Path) -> bool:
    try:
        import pm
        return bool(pm.venv_is_current(project_root=root))
    except Exception:
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
        if head == token.get("pre_sha"):
            # HEAD never moved: a new tracked change is git's half-written checkout.
            changes = tracked_changes(root)
            if changes is None:
                return False, "git cannot read the checkout state"
            unexpected = sorted(set(changes) - set(token.get("dirty_at_pause") or []))
            if unexpected:
                return False, f"the checkout has {len(unexpected)} file(s) git left half-written (e.g. {unexpected[0]})"
            return True, ""
    if not _venv_is_current(root):
        return False, "dependencies are not current for the updated code yet"
    return True, ""


def _atomic_write(path: Path, body: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(body, fh, indent=1, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


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


def mark_stop_requested(token: dict, pids) -> None:
    """Persist, BEFORE the first stop request, that these processes are being asked to stop: one
    that acknowledged and is still draining keeps its restart obligation after a crash."""
    token["stop_requested"] = sorted({*map(str, token.get("stop_requested") or []), *(str(int(p)) for p in pids)})
    write(token, owner=identity())


def discharge(token: dict, path: Path | None = None) -> None:
    """Remove the record only while it still names this pause (decided under the mutex)."""
    path = path or record_path()
    with _mutex():
        existing = read(path)
        if existing is not None and existing["token"].get("pause_id") == token.get("pause_id"):
            with suppress(OSError):
                path.unlink()


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


def _redundant(found: list[tuple[Path, dict]], held: set[Path]) -> set[Path]:
    """Files that only copy an obligation another file carries: a source an update absorbed but
    died before retiring, or one of two same-id files a claim transfer left. Never a held file."""
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
        if oid in carriers and carriers[oid] != src:
            redundant.add(src)
        else:
            by_id.setdefault(oid, []).append(src)
    for copies in by_id.values():
        keep = next((s for s in copies if s in held), copies[0])
        redundant.update(s for s in copies if s != keep)
    return redundant - held


def _survey(path: Path) -> tuple[list[tuple[Path, dict]], set[Path], set[Path]]:
    found = _files(path)
    held = {src for src, body in found if identity_is_live(_holder(src, body))}
    return found, held, _redundant(found, held)


def orphans(path: Path | None = None) -> list[tuple[Path, dict]]:
    """``(file, body)`` for every obligation of this checkout whose holder is dead — one file per
    obligation id; empty while another update is live (it adopts them itself)."""
    path = path or record_path()
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
    if not path.exists() and not _claims(path):
        return
    with suppress(RecordBusy), _mutex():
        _found, _held, redundant = _survey(path)
        for src in redundant:
            with suppress(OSError):
                src.unlink()


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
        adopted = merge_into(adopted, drop_never_stopped(dict(won[1]["token"])))
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


_CARRIED = ("pause_id", "pre_sha", "dirty_at_pause", "identities", "stop_requested", "absorbed")


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
    carried = {key: intended[key] for key in ("pause_id", "pre_sha", "dirty_at_pause", "absorbed") if key in intended}
    write({**adopted, **carried}, owner=UNOWNED)


def _live_pids(token: dict) -> set[str]:
    return {str(pid) for pid, ct in (token.get("identities") or {}).items()
            if identity_is_live({"pid": pid, "ct": ct})}


def _without(token: dict, pids: set[str]) -> dict:
    token["profiles"] = {p: pid for p, pid in (token.get("profiles") or {}).items() if str(pid) not in pids}
    token["unmapped"] = [u for u in (token.get("unmapped") or []) if str(u.get("pid")) not in pids]
    return token


def drop_never_stopped(token: dict) -> dict:
    """Drop entries whose recorded process is the same live incarnation AND was never asked to stop
    (the update died before its first stop request). One it asked to stop may be draining: it keeps
    its restart debt (:func:`split_draining`). No stop record at all: nothing can be told apart."""
    if token.get("stop_requested") is None:
        return token
    return _without(token, _live_pids(token) - {str(p) for p in token["stop_requested"]})


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
        requested = adopted.get("stop_requested")
        if requested is None:  # a set with no stop record keeps every live entry: all count as asked
            requested = list(adopted.get("identities") or {})
        token["stop_requested"] = sorted({*map(str, token.get("stop_requested") or []), *map(str, requested)})
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
    except Exception as exc:
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
            claim_path.unlink()


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
    except Exception as exc:  # never brick a launch on recovery
        print(f"  ⚠ Paused-gateway recovery skipped: {exc}", file=sys.stderr)

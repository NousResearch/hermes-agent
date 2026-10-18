"""Settles a ZIP ``hermes update`` swap whose owner died: its journal, owner lock and recovery.

Split out of ``_early_recovery`` (stdlib-only, like it). ``_early_recovery.restore_interrupted_pull``
imports this only when the journal exists, so the git-marker repair and its published closure
(``RECOVERY_CLOSURE``) never need it.
"""

from __future__ import annotations

import contextlib
import os
import re
import stat
import sys
import time
from pathlib import Path

from hermes_cli._early_recovery import (
    _CLAIM_HELD_ERRNOS, ZIP_SWAP_JOURNAL, _keep_aside, _lock_fd, _project_root, _pytest_owns_live_checkout,
    write_durable_text)

# A journal in the install root naming every entry the swap stages/renames and whether it existed
# before, plus an OS lock its owner holds for the whole stage+swap (the kernel drops it with the
# owner: liveness without pids or ages).
_ZIP_STAGING_SUFFIX, _ZIP_OLD_SUFFIX = ".hermes-update-staging", ".hermes-update-old"


class ZipSwapLock:
    """``zip_swap_owner_lock``'s verdict: truthy only while this process holds the kernel lock.

    ``reason`` names a refusal: ``busy`` (a live owner holds it) or why no lock could be taken at all.
    """

    __slots__ = ("owned", "reason")

    def __init__(self, owned: bool, reason: str = "") -> None:
        self.owned, self.reason = owned, reason

    def __bool__(self) -> bool:
        return self.owned


@contextlib.contextmanager
def zip_swap_owner_lock(root: Path, *, wait: float = 0.0):
    """Yields a truthy ``ZipSwapLock`` while this process owns the ZIP swap lock, a falsy one otherwise.

    The lock file is a stable sidecar: never unlinked, so every process locks the same inode (an
    unlink lets a waiter keep the old inode while a newcomer locks a fresh one). No lock, no admission:
    a root where the file cannot be opened or locked refuses with the reason (fail closed)."""
    lock_path = Path(root) / (ZIP_SWAP_JOURNAL + ".lock")
    try:
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    except OSError as exc:
        yield ZipSwapLock(False, f"cannot open the ZIP swap lock {lock_path}: {exc.strerror or exc}")
        return
    try:
        deadline = time.monotonic() + wait
        while True:
            try:
                if sys.platform == "win32":
                    import msvcrt

                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno not in _CLAIM_HELD_ERRNOS:
                    yield ZipSwapLock(False, f"cannot lock the ZIP swap lock {lock_path}: {exc.strerror or exc}")
                    return
            if time.monotonic() >= deadline:
                yield ZipSwapLock(False, "busy")
                return
            time.sleep(0.05)
        try:
            yield ZipSwapLock(True)
        finally:
            _lock_fd(fd, False)
    finally:
        os.close(fd)


def write_zip_swap_journal(root: Path, phase: str, entries: list, gen: str, temps: dict | None = None) -> None:
    """Publish the swap journal: ``entries`` are ``[name, existed, staged_id, live_id]`` (ids from
    ``zip_entry_identity``, "" while unknown); ``gen`` is the run's random tag (its backup temps);
    ``temps`` maps each backup temp the swap created (its file name) to its ``filling`` identity."""
    import json

    write_durable_text(Path(root) / ZIP_SWAP_JOURNAL, json.dumps(
        {"pid": os.getpid(), "gen": gen, "phase": phase, "entries": entries, "temps": temps or {}}))


def zip_entry_identity(path, *, filling: bool = False) -> str:
    """The entry's own identity (lstat, never its target); "" when absent or unidentifiable.

    The ZIP journal's provenance: recovery deletes only an entry whose identity it recorded when the
    swap made it (a staging copy, or the live entry its backup links/renames), never a lookalike. A
    non-directory adds size and mtime: a file deleted and recreated often gets the same inode back.
    A directory, or a file still being written (``filling``), is ``dev:ino:type`` only: its own size
    and mtime move while it is being filled."""
    try:
        st = os.lstat(path)
    except OSError:
        return ""
    if not st.st_ino:
        return ""
    base = f"{st.st_dev}:{st.st_ino}:{stat.S_IFMT(st.st_mode):o}"
    return base if filling or stat.S_ISDIR(st.st_mode) else f"{base}:{st.st_size}:{st.st_mtime_ns}"


def _drop_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        import shutil

        try:
            shutil.rmtree(path)
        except OSError:
            # A staged copy keeps its source's modes: a read-only directory refuses to give up its
            # entries. These trees are the update's own copies, so make their directories writable and
            # retry. Files keep their modes: a grafted artifact is a hardlink to the LIVE file, and POSIX
            # deletion needs only the parent's write bit. Windows refuses to delete a read-only file, so
            # there only a file no other link shares has its read-only bit cleared.
            rwx = stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR
            os.chmod(path, os.stat(path).st_mode | rwx)
            for dirpath, dirnames, files in os.walk(path):
                for name in dirnames:
                    child = os.path.join(dirpath, name)
                    if not os.path.islink(child):
                        os.chmod(child, os.stat(child).st_mode | rwx)
                for name in files if sys.platform == "win32" else ():
                    child = os.path.join(dirpath, name)
                    st = os.lstat(child)
                    if st.st_nlink == 1 and not stat.S_ISLNK(st.st_mode):
                        os.chmod(child, st.st_mode | stat.S_IWUSR)
            shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def _parse_zip_swap_journal(raw: str) -> tuple[str, str, list[tuple[str, bool, str, str]], dict[str, str]] | None:
    """``(phase, gen, [(entry name, existed, staged_id, live_id)], temps)`` from a journal this code wrote,
    else None. A journal without ``temps`` (an older writer's) vouches for no backup temp."""
    import json

    try:
        data = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(data, dict) or data.get("phase") not in ("staging", "swapping", "committed"):
        return None
    entries, gen = data.get("entries"), data.get("gen")
    if not isinstance(entries, list) or not isinstance(gen, str) or not re.fullmatch(r"[0-9a-f]{12}", gen):
        return None
    if not all(isinstance(e, list) and len(e) == 4 and all(isinstance(v, str) for v in (e[0], e[2], e[3]))
               and isinstance(e[1], bool) and e[0] not in ("", ".", "..") and "/" not in e[0] and "\\" not in e[0]
               for e in entries):
        return None
    temps = data.get("temps", {})
    if not isinstance(temps, dict) or not all(isinstance(v, str) for v in temps.values()):
        return None
    return data["phase"], gen, [(name, existed, staged, live) for name, existed, staged, live in entries], temps


def _discard_unless_owned(path: Path, identity: str, kept: list[Path]) -> None:
    """Delete ``path`` only when it is provably the entry the swap recorded; anything else (a later
    user file at the suffix, a foreign journal's victim) is renamed aside, never deleted (review Z1)."""
    if not os.path.lexists(path):
        return
    if identity and zip_entry_identity(path) == identity:
        _drop_path(path)
    else:
        kept.append(_keep_aside(path))


def _settle_zip_entry(root: Path, phase: str, gen: str, entry: tuple[str, bool, str, str], temps: dict[str, str],
                      kept: list[Path]) -> bool:
    """Finish or roll back one journaled entry; True when the live tree changed.

    Presence is the ENTRY's (lexists/lstat), never its target's: a dangling symlink backup is the only
    copy of a tracked symlink, not "no backup" (review Z3). Restoring a backup never deletes bytes, so
    it needs no provenance; every deletion does (``_discard_unless_owned``)."""
    name, existed, staged_id, live_id = entry
    here = os.path.lexists
    dst = root / name
    staging, old = Path(f"{dst}{_ZIP_STAGING_SUFFIX}"), Path(f"{dst}{_ZIP_OLD_SUFFIX}")
    changed = False
    if phase == "swapping":
        # Staging finished before this phase, so a backup now is the one the swap made from the old
        # entry: it existed, whatever an older journal recorded.
        existed = existed or here(old)
        if existed and here(old) and not (live_id and zip_entry_identity(dst) == live_id):
            if (old.is_file() and not old.is_symlink() and staged_id and zip_entry_identity(dst) == staged_id
                    and not dst.is_dir()):
                os.replace(old, dst)  # a file entry never goes missing, not even here
            else:
                _discard_unless_owned(dst, staged_id, kept)
                os.rename(old, dst)  # moves a symlink itself, never its target
            changed = True
        elif not existed and here(dst) and not here(staging):
            _discard_unless_owned(dst, staged_id, kept)
            changed = True
    elif existed and not here(dst) and here(old):
        os.rename(old, dst)  # the backup is the only copy left
        changed = True
    for leftover, identity in ((staging, staged_id), (old, live_id)):
        _discard_unless_owned(leftover, identity, kept)
    # ``<old>.<gen>.tmp``: a backup copy of the live file killed before its rename (no-hardlink file
    # systems). Neither the tag nor the bytes prove a file there is that copy (F78/F78-R): only the
    # identity the swap journaled the moment it created the temp does. A file system that hands a
    # recreated name its old inode (FAT, where this copy runs) would let that record vouch for a
    # later file too, so the bytes must also be what a killed copy is: a prefix of the live file.
    tmp = Path(f"{old}.{gen}.tmp")
    if _is_killed_backup_copy(tmp, dst, temps.get(tmp.name, "")):
        _drop_path(tmp)
    elif here(tmp):
        kept.append(_keep_aside(tmp))
    return changed


def _is_killed_backup_copy(tmp: Path, dst: Path, identity: str) -> bool:
    """The identity matches the regular file the swap recorded (type included), so the bytes are readable."""
    return (bool(identity) and zip_entry_identity(tmp, filling=True) == identity
            and dst.is_file() and not dst.is_symlink() and dst.read_bytes().startswith(tmp.read_bytes()))


def restore_interrupted_zip_swap(project_root: Path | None = None) -> bool:
    """Finish or roll back a ZIP swap whose owner died; True when the tree changed (relaunch).

    ``committed``: every rename landed, only backups remain -> drop them (finish). ``swapping``: put
    each moved-aside entry back and remove entries the swap added (roll back to the old tree, which
    the venv was built for; ``hermes update`` redoes it). ``staging``: nothing live moved -> drop the
    staging copies. Either way no ``*.hermes-update-staging``/``-old`` sibling is left to wedge the
    next run's free-space or dirty-tree checks. A live owner (lock held) is never second-guessed.
    Provenance (review Z1): a sibling or entry is deleted only when its identity is the one the journal
    recorded; anything else is kept aside under a ``.hermes-update-kept`` name and reported.
    """
    root = _project_root() if project_root is None else Path(project_root)
    journal = root / ZIP_SWAP_JOURNAL
    if not journal.is_file() or _pytest_owns_live_checkout(root):
        return False
    with zip_swap_owner_lock(root) as owned:
        if not owned and owned.reason != "busy":
            print(f"⚠ An interrupted ZIP update cannot be settled: {owned.reason}.", file=sys.stderr)
        if not owned or not journal.is_file():
            return False
        try:
            raw = journal.read_text(encoding="utf-8-sig")
        except OSError as exc:  # transient (AV, sharing violation): the journal is still the record
            print(f"⚠ Could not read the interrupted ZIP update's journal ({exc}); the next launch retries.",
                  file=sys.stderr)
            return False
        parsed = _parse_zip_swap_journal(raw)
        if parsed is None:
            # Never guess which live entry to move, and never retire the only record of a mixed tree.
            print(f"⚠ The interrupted ZIP update's journal {journal} is unreadable or from another version; "
                  "it and every `*.hermes-update-old` backup were kept. Put back what each backup replaced "
                  "(or reinstall), then delete the journal.", file=sys.stderr)
            return False
        phase, gen, entries, temps = parsed
        changed = False
        failed = False
        kept: list[Path] = []
        for entry in reversed(entries):
            try:
                changed = _settle_zip_entry(root, phase, gen, entry, temps, kept) or changed
            except OSError as exc:
                failed = True
                print(f"⚠ Could not settle {entry[0]} after an interrupted ZIP update: {exc}", file=sys.stderr)
        if kept:
            print("⚠ An interrupted ZIP update's recovery found entries it could not prove were its own and "
                  f"kept them aside instead of deleting them: {', '.join(map(str, kept))}. Delete each once "
                  "you know it is not yours.", file=sys.stderr)
        # Retire the journal only on a verified terminal state, not an exception-free loop: no sibling
        # (staging copy, backup, backup temp) left that only this journal could still explain.
        siblings = (_ZIP_STAGING_SUFFIX, _ZIP_OLD_SUFFIX, f"{_ZIP_OLD_SUFFIX}.{gen}.tmp")
        unsettled = [e[0] for e in entries if any(os.path.lexists(f"{root / e[0]}{x}") for x in siblings)]
        if failed or unsettled:
            if unsettled and not failed:
                print(f"⚠ The interrupted ZIP update left {', '.join(unsettled)} unsettled; its journal "
                      f"{journal} stays and the next launch retries.", file=sys.stderr)
            return changed  # the journal stays: the next launch retries
        journal.unlink(missing_ok=True)
    if changed:
        print("⚠ A previous ZIP `hermes update` was interrupted mid-swap; the old install was put back. "
              "`hermes update` updates it again.", file=sys.stderr)
    return changed

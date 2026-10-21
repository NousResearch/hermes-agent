"""SQLite-safe restore and archive-member publish plumbing for backups.

Owns the restore side of ``hermes_cli.backup``: the page-copy SQLite restore
(``_safe_restore_db`` plus the foreign-holder scan) and the zip-member publish
helpers used by ``hermes import`` and ``/snapshot restore``.  Backup
*creation* (``run_backup``, full-zip writing) stays in ``hermes_cli.backup``,
which composes these helpers.
"""

import json
import logging
import os
import shutil
import sqlite3
import stat
import sys
import tempfile
import zipfile
from contextlib import closing, contextmanager
from pathlib import Path
from typing import List, Optional, Tuple

from hermes_state_holders import read_only_db_uri
from hermes_cli.backup_sqlite import _safe_copy_db
from utils import (
    _preserve_file_mode, _preserve_file_owner, _restore_file_mode, _restore_file_owner, atomic_replace,
    mkstemp_beside,
)

logger = logging.getLogger(__name__)

def _foreign_db_holder_pids(db_path: Path) -> Optional[list[int]]:
    """PIDs of OTHER processes holding *db_path* or its WAL/SHM open.

    Linux-only ``/proc/<pid>/fd`` scan (no psutil dependency), preserving the
    kernel's ``(deleted)`` suffix so an already-unlinked sidecar generation —
    the #90950 split-brain fingerprint — still counts as held. Returns
    ``None`` when the scan is unavailable (non-Linux, or /proc unreadable);
    callers must treat ``None`` as "unknown", not as "no holders".
    """
    if not sys.platform.startswith("linux"):
        return None

    def _canonical(path: str) -> str:
        return os.path.normcase(
            os.path.abspath(path.removesuffix(" (deleted)"))
        )

    canonical_db = _canonical(os.fspath(db_path))
    watched = {canonical_db, canonical_db + "-wal", canonical_db + "-shm"}
    pids: list[int] = []
    try:
        own_pid = os.getpid()
        for pid_str in os.listdir("/proc"):
            if not pid_str.isdigit():
                continue
            pid = int(pid_str)
            if pid == own_pid:
                continue
            fd_dir = f"/proc/{pid}/fd"
            try:
                fds = os.listdir(fd_dir)
            except OSError:
                continue
            for fd in fds:
                try:
                    target = os.readlink(f"{fd_dir}/{fd}")
                except OSError:
                    continue
                if _canonical(target) in watched:
                    pids.append(pid)
                    break
    except OSError:
        return None
    return pids


def _auth_restore_target(dst: Path) -> Optional[Path]:
    """Writable auth-store path whose lock and publish name the same underlying file.

    A file symlink is resolved so refresh writers and restore take the same auth.lock. A hard-linked
    auth.json cannot be atomically replaced without splitting a deliberately shared store into two
    inodes, so that topology fails closed instead of silently breaking credential sharing.
    """
    try:
        if dst.exists():
            links = dst.stat().st_nlink
            if links > 1:
                logger.error(
                    "Refusing auth.json restore to %s: it has %d hard links; atomic replacement "
                    "would split a shared auth store",
                    dst,
                    links,
                )
                return None
        return dst.resolve(strict=False) if dst.is_symlink() else dst
    except (OSError, RuntimeError) as exc:
        logger.error("Refusing auth.json restore to %s: cannot resolve store identity: %s", dst, exc)
        return None


def _restore_auth_json(src: Path, dst: Path) -> bool:
    """Restore auth.json without rolling back a live single-use OAuth generation.

    The live read and final write share the canonical auth-store lock, so a concurrent refresh
    cannot land between preservation and publish.
    """
    try:
        snapshot_store = json.loads(src.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, UnicodeError) as exc:
        logger.error("Refusing auth.json restore from %s: %s", src, exc)
        return False
    if not isinstance(snapshot_store, dict):
        logger.error("Refusing auth.json restore from %s: top level is not an object", src)
        return False
    if not (
        isinstance(snapshot_store.get("providers"), dict)
        or isinstance(snapshot_store.get("credential_pool"), dict)
        or isinstance(snapshot_store.get("systems"), dict)
    ):
        logger.error("Refusing auth.json restore from %s: unrecognized auth-store shape", src)
        return False
    # Legacy "systems" stores are still valid snapshot input. Mirror auth._load_auth_store's
    # migration without asking that loader to create a .corrupt sidecar inside the snapshot.
    if (
        not isinstance(snapshot_store.get("providers"), dict)
        and not isinstance(snapshot_store.get("credential_pool"), dict)
        and isinstance(snapshot_store.get("systems"), dict)
    ):
        systems = snapshot_store["systems"]
        providers = {"nous": systems["nous_portal"]} if "nous_portal" in systems else {}
        # Match auth._load_auth_store's migration shape exactly: the legacy
        # "systems" container itself is obsolete and must not be written back.
        snapshot_store = {
            "providers": providers,
            "active_provider": "nous" if providers else None,
        }

    try:
        from hermes_cli.auth import _auth_store_lock, _load_auth_store, _save_auth_store
        from hermes_cli.auth_oauth_grants import (
            merge_snapshot_auth_preserving_live_single_use_grants,
        )

        target = _auth_restore_target(dst)
        if target is None:
            return False
        with _auth_store_lock(target_path=target):
            live_store = _load_auth_store(target)
            restored = merge_snapshot_auth_preserving_live_single_use_grants(
                snapshot_store, live_store
            )
            _save_auth_store(restored, target_path=target)
        return True
    except Exception as exc:
        logger.error("Failed to restore %s safely: %s", dst, exc)
        return False


def _safe_restore_db(src: Path, dst: Path) -> bool:
    """Restore only while holding the destination authority's maintenance reservation.

    Returns ``False``, without touching *dst*, when *src* fails the SQLite
    integrity check, when the runtime owns the profile, or when a canonical
    ``state.db`` epoch floor cannot be proven.
    """
    from gateway.runtime_ownership import OwnershipConflict, exclusive_maintenance
    from hermes_cli.backup import verify_sqlite_integrity

    # backup() copies pages without validating their contents; its fallback
    # copies bytes even when SQLite rejected the source. Neither may touch the
    # destination until the snapshot passes the existing bounded integrity policy.
    source_check = verify_sqlite_integrity(src)
    if not source_check["valid"]:
        logger.error("Refusing SQLite restore from %s: %s", src, source_check["message"])
        return False
    try:
        with exclusive_maintenance([dst.absolute().parent, dst.resolve().parent]):
            with _restore_epoch_source(src, dst) as prepared:
                return _restore_db_pages(prepared, dst)
    except (OwnershipConflict, OSError, sqlite3.Error) as exc:
        logger.error("%s", exc)
        return False


def _restore_epoch(path: Path) -> int:
    if not path.exists():
        return 0
    with closing(sqlite3.connect(read_only_db_uri(path), uri=True)) as conn:
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='runtime_epoch'").fetchone():
            return 0
        row = conn.execute("SELECT epoch FROM runtime_epoch WHERE singleton=1").fetchone()
        return int(row[0]) if row else 0


def _restore_destination_epoch(dst: Path) -> int:
    try:
        return _restore_epoch(dst)
    except (OSError, sqlite3.Error, ValueError) as exc:
        raise OSError(
            f"Cannot read the canonical runtime epoch at {dst}; refusing in-place restore. "
            "Use `hermes sessions recover --source <snapshot> --output <separate-path>` "
            "to salvage transcripts into a separate output."
        ) from exc


@contextmanager
def _restore_epoch_source(src: Path, dst: Path):
    """Stage an epoch floor into the image BEFORE publishing it atomically.

    Updating the destination after backup() would leave a crash window that
    reuses an older worker epoch. Never modify the user's source snapshot.
    Full restores retain all admissions/worker receipts; normal startup advances
    this floor and reconciles started work to unknown, not replayable queued work.
    An unreadable destination cannot prove its epoch floor: use transcript salvage
    into a separate output rather than silently restoring with a recycled epoch.
    """
    if dst.name != 'state.db' and dst.resolve().name != 'state.db':
        yield src
        return
    floor = _restore_destination_epoch(dst)
    if floor <= _restore_epoch(src):
        yield src
        return
    with tempfile.TemporaryDirectory(prefix='.restore-epoch-', dir=dst.parent) as work:
        prepared = Path(work) / 'state.db'
        if not _safe_copy_db(src, prepared):
            raise OSError('Unable to stage a restore with a safe runtime epoch')
        with closing(sqlite3.connect(prepared)) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS runtime_epoch (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                epoch INTEGER NOT NULL CHECK (epoch > 0), instance_id TEXT NOT NULL)""")
            conn.execute("""INSERT INTO runtime_epoch VALUES(1,?, 'offline-restore')
                ON CONFLICT(singleton) DO UPDATE SET epoch=excluded.epoch,
                instance_id=excluded.instance_id""", (floor,))
            conn.commit()
        prepared.chmod(src.stat().st_mode)
        yield prepared


def _restore_db_pages(src: Path, dst: Path) -> bool:
    """Restore a SQLite database from snapshot *src* into live *dst*.

    Uses SQLite's backup() API to write snapshot pages into the live
    database file, preserving the file's inode and WAL state so that
    any other process still holding the DB open (gateway, dashboard,
    another CLI session) sees the restored data on the next read —
    instead of continuing to serve stale cached pages from a replaced
    inode.

    The old approach was ``unlink() + move()``, which replaced the file
    under any live connection.  SQLite connections cache pages in
    per-connection page caches keyed by inode; after an unlink+move the
    old inode still existed (the live connection held a reference), so
    that connection continued serving the pre-restore data while new
    connections saw the restored snapshot — a partial/inconsistent
    state (issue #65942).

    By writing pages through the backup API the file inode is preserved,
    the WAL journal is updated correctly, and all connections (old and
    new) converge on the restored data.

    Falls back to the unlink+move approach on failure ONLY when no other
    process or in-process connection holds the file: replacing the inode
    under a live holder is the #90950 split-brain, so that branch fails
    closed (returns ``False``) and the caller reports the file as skipped.
    Callers go through ``_safe_restore_db``, which validates *src* and holds
    the maintenance reservation and epoch floor around this publication.
    """
    dst_conn: Optional[sqlite3.Connection] = None
    try:
        dst_conn = sqlite3.connect(str(dst))
        try:
            # Force a WAL checkpoint so the backup starts from a clean
            # state rather than writing on top of a deep WAL.
            dst_conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.Error as exc:
            # Optional: backup() still publishes over a deep WAL, just less tidily.
            logger.debug("Pre-restore WAL checkpoint of %s skipped: %s", dst, exc)
        src_conn = sqlite3.connect(read_only_db_uri(src), uri=True)
        try:
            src_conn.backup(dst_conn)
        finally:
            src_conn.close()
        dst_conn.close()
        # Restore original file permissions from the snapshot
        try:
            mode = src.stat().st_mode
            dst.chmod(mode)
        except OSError as exc:
            # The data is restored; only the snapshot's file mode could not be carried over.
            logger.warning("Restored %s but could not apply the snapshot's file mode: %s", dst, exc)
        return True
    except (sqlite3.Error, OSError, ValueError) as exc:  # ValueError: an unrepresentable path
        logger.warning("SQLite safe restore failed for %s -> %s: %s", src, dst, exc)
        # Release our own handle on *dst* before the fallback: on Windows an
        # open connection blocks unlink() with WinError 32, which would make
        # the fallback below dead code exactly when it is needed.
        if dst_conn is not None:
            try:
                dst_conn.close()
            except sqlite3.Error as close_exc:
                logger.debug("Closing %s before the fallback restore failed: %s", dst, close_exc)
        # Fallback: unlink+move (the old approach).  This still works for
        # the common case where no other process holds the DB open.
        from hermes_cli.sqlite_safe_read import (
            LiveConnectionError,
            offline_file_access,
        )

        try:
            holders = _foreign_db_holder_pids(dst)
            if holders:
                # Replacing the inode under a live holder is the #90950
                # corruption class: the holder keeps writing through a
                # deleted-inode fd (split brain), and removing its sidecars
                # detaches the WAL index it is checkpointing through. The
                # backup-API path above is the live-safe route; if it failed,
                # fail closed rather than corrupt.
                logger.error(
                    "Refusing unlink+move restore of %s: process(es) %s still "
                    "hold the database or its WAL open. Stop them and retry.",
                    dst, holders,
                )
                return False
            # The foreign-pid scan above deliberately excludes THIS process,
            # but an in-process SessionDB (the agent's own handle during
            # /snapshot restore, a second SessionDB instance, a read pool)
            # is exactly as much of a live holder: unlinking the DB and its
            # sidecars under it leaves this process on deleted-inode fds —
            # the same #90950 split brain, produced first-party (proven live
            # on main: `/proc/self/fd` shows `state.db-wal (deleted)` right
            # after this fallback ran under a tracked connection).
            # ``offline_file_access`` fails CLOSED when any tracked
            # connection to *dst* is live and holds the connection-lifecycle
            # lock across the whole swap so no new connection can appear
            # mid-replace.
            with offline_file_access(dst, what="unlink+move restore of"):
                tmp = dst.parent / f".{dst.name}.snap_restore"
                shutil.copy2(src, tmp)
                dst.unlink(missing_ok=True)
                # Drop the destination's sidecars before installing the
                # snapshot. The snapshot is a checkpointed ``sqlite3.backup()``
                # image (see ``_safe_copy_db``) that owns no WAL, so any
                # ``-wal``/``-shm`` still sitting here describes the database we
                # just unlinked — an ungracefully killed gateway leaves them
                # behind, which is exactly when a restore gets run. SQLite
                # replays that foreign WAL over the restored file on the next
                # open and the database comes up "malformed" (or silently
                # resurrects post-snapshot rows). Same reasoning as
                # ``_EXCLUDED_SUFFIXES``, applied to the restore destination.
                for _sidecar_suffix in ("-wal", "-shm", "-journal"):
                    dst.with_name(dst.name + _sidecar_suffix).unlink(missing_ok=True)
                shutil.move(str(tmp), str(dst))
            return True
        except LiveConnectionError as exc2:
            logger.error(
                "Refusing unlink+move restore of %s: %s Close the in-process "
                "database handles (or restart Hermes) and retry.",
                dst, exc2,
            )
            return False
        except (OSError, ValueError) as exc2:  # ValueError: a path the OS cannot represent
            logger.error("Fallback restore also failed for %s -> %s: %s", src, dst, exc2)
            return False


def _validate_backup_zip(zf: zipfile.ZipFile) -> tuple[bool, str]:
    """Check that a zip looks like a Hermes backup.

    Returns (ok, reason).
    """
    names = zf.namelist()
    if not names:
        return False, "zip archive is empty"

    # Look for telltale files that a hermes home would have
    markers = {"config.yaml", ".env", "state.db"}
    found = set()
    for n in names:
        # Could be at the root or one level deep (if someone zipped the directory)
        basename = Path(n).name
        if basename in markers:
            found.add(basename)

    if not found:
        return False, (
            "zip does not appear to be a Hermes backup "
            "(no config.yaml, .env, or state databases found)"
        )

    return True, ""


def _detect_prefix(zf: zipfile.ZipFile) -> str:
    """Detect if the zip has a common directory prefix wrapping all entries.

    Some tools zip as `.hermes/config.yaml` instead of `config.yaml`.
    Returns the prefix to strip (empty string if none).
    """
    names = [n for n in zf.namelist() if not n.endswith("/")]
    if not names:
        return ""

    # Find common prefix
    parts_list = [Path(n).parts for n in names]

    # Check if all entries share a common first directory
    first_parts = {p[0] for p in parts_list if len(p) > 1}
    if len(first_parts) == 1:
        prefix = first_parts.pop()
        # Only strip if it looks like a hermes dir name
        if prefix in {".hermes", "hermes"}:
            return prefix + "/"

    return ""


def _default_new_file_mode() -> Optional[int]:
    """Return the mode ``open(path, "wb")`` gives a file it has to create.

    ``tempfile.mkstemp`` always creates at 0600, so staging an import through a
    temp file would tighten every *newly created* file to owner-only — the same
    hazard ``utils._restore_file_mode`` documents for Docker/NAS volume mounts
    that rely on broader permissions.  The umask can only be read by setting it,
    so this is resolved once per import rather than once per member.  The probe
    installs a *restrictive* mask rather than 0 so that anything another thread
    creates inside the two-syscall window is owner-only, never world-writable.
    Returns ``None`` if the umask cannot be read, in which case the caller
    leaves mkstemp's mode alone.
    """
    try:
        current = os.umask(0o077)
        os.umask(current)
    except OSError:
        return None
    return 0o666 & ~current


def _extract_member_atomically(
    zf: zipfile.ZipFile,
    member: str,
    target: Path,
    new_file_mode: Optional[int] = None,
) -> None:
    """Restore one zip member onto *target* with no truncation window.

    ``open(target, "wb")`` truncates the user's existing file to zero *before*
    any replacement bytes exist.  A Ctrl-C, an ENOSPC, a corrupt member, or a
    crash between the truncate and the write therefore leaves that file empty
    with nothing behind it — during ``hermes import``, which is the
    disaster-recovery path a user reaches for *because* they already lost
    something.  Staging into the target's own directory and publishing with a
    rename means the target only ever moves from its old contents to the
    complete new contents.

    ``atomic_replace`` rather than a bare ``os.replace``: it resolves a
    symlinked target first, so a deployment that links ``config.yaml`` into a
    dotfiles repo keeps the link instead of having it silently swapped for a
    regular file (GitHub #16743), and it falls back to copy/fsync/unlink on
    ``EXDEV``/``EBUSY`` for cross-device and bind-mount installs.  That
    fallback uses ``shutil.copyfile``, which does truncate in place, so on the
    cross-device path the guarantee above degrades to today's behaviour rather
    than improving on it; closing that belongs in ``utils.atomic_replace``,
    where every atomic writer in the repo would benefit, not here.

    Permission bits *and* ownership are carried across the replace so routing
    through mkstemp does not change the file the caller would otherwise have
    produced.  ``os.replace`` swaps in a temp file owned by the *writing* user,
    so without the chown a ``sudo hermes import`` would silently re-own every
    restored file to root — on the disaster-recovery path, and on exactly the
    Docker/NAS installs ``utils._restore_file_owner`` documents.  Both concerns
    delegate to the shared ``utils`` helpers rather than being re-derived here.
    The temp file is removed on any failure so a partial import leaves no
    residue.

    The one bit of the old file *not* carried across is setuid/setgid.  The
    replacement bytes come out of the zip, so preserving those would let an
    archive take over the identity an existing privileged file executes as —
    and unlike the other ``utils`` writers, which re-serialize content this
    process produced, the trust boundary here is an untrusted archive.  The
    mask is applied once, before the temp file is chmod'd, so neither the
    pre-replace ``fchmod`` nor the post-replace restore can re-elevate the
    target.
    """
    # ``_preserve_file_mode`` returns None when the target does not exist (or
    # cannot be stat'd), in which case the umask-derived create-mode applies —
    # the same shape as ``atomic_yaml_write``'s ``create_mode`` fallback.
    mode = _preserve_file_mode(target)
    owner = _preserve_file_owner(target)
    if mode is None:
        mode = new_file_mode
    else:
        # Deliberately NOT a faithful mode copy: setuid/setgid are dropped.
        # ``_preserve_file_mode`` returns ``stat.S_IMODE``, i.e. all twelve
        # bits, and the content replacing this file comes from the archive.
        # Carrying the elevated bits across would let archive-controlled bytes
        # take over an existing setuid/setgid file, so ``hermes import`` would
        # hand whoever produced the zip the identity that file runs as.  Nothing
        # constrains that to Hermes' own state either: the ``_external/`` branch
        # of ``run_import`` publishes members anywhere under ``$HOME``.  The
        # sticky bit is kept — it is inert on a regular file.
        mode &= ~(stat.S_ISUID | stat.S_ISGID)

    # Truncate the stem: mkstemp adds ~16 characters, and a member already near
    # NAME_MAX would otherwise fail here on a write that used to succeed.
    fd, tmp_name = mkstemp_beside(target, prefix=f".{target.name[:80]}.", suffix=".partial")
    try:
        with os.fdopen(fd, "wb") as dst:
            if mode is not None:
                # Apply the mode to the temp file BEFORE the replace so the
                # target never transits through mkstemp's 0600, and so
                # ``atomic_replace``'s EXDEV/EBUSY ``shutil.copystat`` fallback
                # copies the intended bits rather than 0600.  fchmod is
                # Unix-only; Windows takes the path-based chmod.
                if hasattr(os, "fchmod"):
                    os.fchmod(dst.fileno(), mode)
                else:
                    os.chmod(tmp_name, mode)
            # Stream instead of ``src.read()``: a multi-gigabyte state.db member
            # must not be held in memory in one piece.
            with zf.open(member) as src:
                shutil.copyfileobj(src, dst)
            dst.flush()
            os.fsync(dst.fileno())
        real_path = Path(atomic_replace(tmp_name, target))
        # Owner first, mode second — the ordering ``atomic_yaml_write`` uses,
        # because chown drops setuid/setgid and a mode restore that ran first
        # would be partly undone.  Here ``mode`` no longer carries those bits,
        # so the two agree: neither step can re-elevate the restored file.
        _restore_file_owner(real_path, owner)
        _restore_file_mode(real_path, mode)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _count_session_rows(path: Path) -> Optional[tuple[int, int]]:
    """Return ``(sessions, messages)`` stored in the session database *path*.

    Read-only and best effort.  ``None`` means "unknown" — a missing file, a
    database that is not a Hermes session store, or one that cannot be read.
    Callers must never read ``None`` as "zero rows": acting on an unreadable
    database would mask the very loss this count exists to surface.  Same
    contract as :func:`_count_cron_jobs`.
    """
    if not path.is_file():
        return None
    try:
        conn = sqlite3.connect(read_only_db_uri(path), uri=True)
    except sqlite3.Error:
        return None
    try:
        sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        messages = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        return int(sessions), int(messages)
    except (sqlite3.Error, TypeError, ValueError):
        return None
    finally:
        conn.close()


def _import_db_member(
    zf: zipfile.ZipFile,
    member: str,
    target: Path,
    new_file_mode: Optional[int] = None,
) -> None:
    from gateway.runtime_ownership import OwnershipConflict, exclusive_maintenance
    try:
        with exclusive_maintenance([target.absolute().parent, target.resolve().parent]):
            _import_db_member_exclusive(zf, member, target, new_file_mode)
    except OwnershipConflict as exc:
        raise OSError(str(exc)) from exc


def _import_db_member_exclusive(
    zf: zipfile.ZipFile,
    member: str,
    target: Path,
    new_file_mode: Optional[int] = None,
) -> None:
    """Publish a SQLite ``.db`` member onto *target* without replacing its inode.

    ``_extract_member_atomically`` publishes with a rename.  For an ordinary
    file that is the safest write available; for a live SQLite database it is
    the #65942 / #90950 corruption class.  A gateway, dashboard, or WebUI
    process holding the database open keeps its descriptor on the now-unlinked
    inode: it goes on serving pre-import pages and writing sessions that no
    other process will ever see, and any sidecar WAL left beside the new file
    describes the database that was just unlinked.  Nothing fails, so nothing
    is reported — the sessions simply are not there afterwards (issue #100960).

    ``hermes import`` is the disaster-recovery path, so that failure mode lands
    on users who have already lost something once.  Route the member through
    the same ``_safe_restore_db`` page copy that ``/snapshot restore`` has used
    since #65942: the live inode is preserved, every open connection converges
    on the imported data, and the sidecars are handled there.

    Raises ``OSError`` when the database could not be replaced safely, so the
    caller reports a skipped file instead of counting a silent success.
    """
    if not target.exists():
        # "Missing" is not "unheld": a gateway or dashboard that had the database open when it
        # was unlinked still writes the deleted inode (the ``(deleted)`` fingerprint of #90950).
        # Publishing a fresh inode here re-creates the same split brain the branch below exists
        # to prevent, so refuse and name the holders instead (#110179).
        holders = _foreign_db_holder_pids(target)
        if holders:
            raise OSError(
                f"{target.name} was deleted but is still open in PID(s) "
                f"{', '.join(str(pid) for pid in sorted(holders))}; publishing a new file would "
                "leave them writing an invisible database. Stop those processes and re-run the import."
            )
        _extract_member_atomically(zf, member, target, new_file_mode)
        return

    # The database keeps its own mode/ownership: the bytes come from the
    # archive but the file does not, so the archive has no say in either.
    mode = _preserve_file_mode(target)
    owner = _preserve_file_owner(target)

    fd, tmp_name = tempfile.mkstemp(
        dir=str(target.parent), prefix=f".{target.name[:80]}.", suffix=".dbimport"
    )
    try:
        with os.fdopen(fd, "wb") as dst:
            # Stream: a multi-gigabyte state.db member must not be held in
            # memory in one piece.
            with zf.open(member) as src:
                shutil.copyfileobj(src, dst)
            dst.flush()
            os.fsync(dst.fileno())
        if not _safe_restore_db(Path(tmp_name), target):
            from hermes_cli.backup import verify_sqlite_integrity

            # Re-check only on failure so the user gets the real cause; the
            # detailed integrity message was already logged by _safe_restore_db.
            if not verify_sqlite_integrity(Path(tmp_name))["valid"]:
                raise OSError(
                    "the archived database failed its integrity check; the existing "
                    "database was left untouched."
                )
            raise OSError(
                "live-safe restore refused or failed; the existing database was "
                "left untouched. Stop the gateway/dashboard processes holding it "
                "open and re-run the import."
            )
        _restore_file_owner(target, owner)
        _restore_file_mode(target, mode)
    finally:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass

"""Publish a prepared distribution without replacing the profile root."""

from __future__ import annotations

import errno
import logging
import os
from pathlib import Path
import shutil
import stat
import tempfile
from collections.abc import Sequence


logger = logging.getLogger(__name__)


def _checked_paths(paths: Sequence[tuple[str, ...]]) -> tuple[tuple[str, ...], ...]:
    result = tuple(tuple(parts) for parts in paths)
    for parts in result:
        if not parts or any(
            not part or part in (".", "..") or "\0" in part or Path(part).name != part
            or (os.name == "nt" and ":" in part)
            for part in parts
        ):
            raise OSError("Distribution publication requires relative path components")
    return result


def _is_reparse(info) -> bool:
    return bool(getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def _ensure_directory(target: Path, parts: tuple[str, ...], created: list[Path]) -> None:
    current = target
    for part in (None, *parts):
        if part is not None:
            current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            current.mkdir()
            created.append(current)
        else:
            if not stat.S_ISDIR(info.st_mode) or _is_reparse(info):
                raise OSError(f"Distribution destination parent is not a plain directory: {current}")


def _remove_backup(path: Path) -> None:
    from hermes_cli.profiles import _rmtree_make_writable

    shutil.rmtree(path, onerror=_rmtree_make_writable)


def commit_owned_payload(
    target: Path, prepared_root: Path, relative_paths: Sequence[tuple[str, ...]],
    *, empty_dirs: Sequence[tuple[str, ...]] = (),
) -> None:
    """Commit disjoint, fully validated entries under the profile lifecycle lock.

    Renames preserve originals for rollback on an exception, including cancellation.
    This does not provide isolation from external readers or recovery after power loss.
    """
    paths = _checked_paths(relative_paths)
    directories = _checked_paths(empty_dirs)
    ordered = sorted(tuple(os.path.normcase(part) for part in parts) for parts in paths)
    if any(right[:len(left)] == left for left, right in zip(ordered, ordered[1:])):
        raise OSError("Distribution publication paths must be disjoint")
    device = target.parent.stat().st_dev
    if prepared_root.stat().st_dev != device or (
        os.path.lexists(target) and target.lstat().st_dev != device
    ):
        raise OSError("Distribution publication requires preparation on the profile filesystem")
    for parts in paths:
        info = prepared_root.joinpath(*parts).lstat()
        if info.st_dev != device or _is_reparse(info) or not (
            stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)
        ):
            raise OSError("Prepared distribution entries must be local regular files or directories")

    # Keep backups outside the caller's temporary tree: a failed rollback must
    # never let its cleanup delete the only recoverable copy of an original.
    backup_root = Path(tempfile.mkdtemp(prefix=".hermes-dist-rollback-", dir=target.parent))
    created: list[Path] = []
    journal: list[tuple[Path, Path, Path, bool]] = []
    try:
        _ensure_directory(target, (), created)
        for parts in paths:
            _ensure_directory(target, parts[:-1], created)
            source = prepared_root.joinpath(*parts)
            destination = target.joinpath(*parts)
            backup = backup_root.joinpath(*parts)
            existed = os.path.lexists(destination)
            # Register before either rename. Presence of source/backup also covers
            # an interrupt delivered immediately after a successful native rename.
            journal.append((source, destination, backup, existed))
            if existed:
                backup.parent.mkdir(parents=True, exist_ok=True)
                os.replace(destination, backup)
            os.replace(source, destination)
        for parts in directories:
            # Existing bootstrap dirs can be user-owned links. Reuse them without
            # writing through them; parents of published entries remain no-follow.
            if target.joinpath(*parts).is_dir():
                continue
            _ensure_directory(target, parts, created)
    except BaseException as exc:
        failures = []
        for source, destination, backup, existed in reversed(journal):
            try:
                backed_up = os.path.lexists(backup)
                if backed_up or not existed:
                    # Move new payloads back into the caller's scratch tree. This
                    # also avoids deleting read-only Windows files during rollback.
                    if not os.path.lexists(source) and os.path.lexists(destination):
                        os.replace(destination, source)
                    if backed_up:
                        os.replace(backup, destination)
            except OSError as rollback_error:
                failures.append(f"{destination}: {rollback_error}")
        for directory in reversed(created):
            try:
                directory.rmdir()
            except FileNotFoundError:
                continue
            except OSError as rollback_error:
                # A concurrent writer's new data is never ours to remove.
                if rollback_error.errno not in (errno.ENOTEMPTY, errno.EEXIST):
                    failures.append(f"{directory}: {rollback_error}")
        if not failures:
            try:
                _remove_backup(backup_root)
            except OSError as cleanup_error:
                failures.append(f"backup cleanup: {cleanup_error}")
        if failures:
            message = (
                f"Distribution rollback could not finish; recovery files retained at {backup_root}: "
                + "; ".join(failures)
            )
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                exc.add_note(message)
                raise
            raise OSError(message) from exc
        raise
    else:
        try:
            _remove_backup(backup_root)
        except OSError:
            # The new payload is committed; rollback is no longer possible once
            # cleanup has removed some originals. Keep and report any leftovers.
            logger.warning("Distribution committed; could not remove backups at %s", backup_root, exc_info=True)

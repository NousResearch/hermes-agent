"""Publish a prepared distribution through retained destination directories."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import secrets
import stat


logger = logging.getLogger(__name__)


@dataclass
class _CreatedDirectory:
    parent: object
    name: str
    identity: tuple[int, int] | None = None
    anchor: object | None = None


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


def _exists(parent, name: str) -> bool:
    try:
        parent.stat(name)
    except FileNotFoundError:
        return False
    return True


def _descend(parent, parts):
    for name in parts:
        parent = parent.child(name)
    return parent


def _create_directory(parent, name, created, mode=0o777):
    parent.mkdir(name, mode=mode)
    # Even stat/pin acquisition can fail after mkdir succeeds. Keep the location
    # first; an unknown or replaced identity must be reported, never removed.
    record = _CreatedDirectory(parent, name)
    created.append(record)
    info = parent.stat(name)
    record.identity = info.st_dev, info.st_ino
    child = parent.child(name)
    record.anchor = child
    if child.identity != record.identity:
        raise OSError(f"Created distribution directory changed before opening: {parent.path / name}")
    return child


def _ensure_child(parent, name, created):
    parent.verify()
    child = parent.child(name) if _exists(parent, name) else _create_directory(parent, name, created)
    child.verify()
    return child


def _ensure_directory(parent, parts, created):
    for name in parts:
        parent = _ensure_child(parent, name, created)
    return parent


def _replace(source, name, destination):
    source.verify()
    destination.verify()
    source.replace(name, destination, name)
    source.verify()
    destination.verify()


def _remove_backup(parent, name, backup):
    backup.close()
    parent.remove_tree(name)


def _undo_created(record, failures):
    path = record.parent.path / record.name
    try:
        if record.anchor is not None:
            record.anchor.close()
        current = record.parent.stat(record.name)
        if (
            record.identity is None or record.identity != (current.st_dev, current.st_ino)
            or not stat.S_ISDIR(current.st_mode)
            or getattr(current, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
        ):
            failures.append(f"Created directory identity could not be verified; left {path} untouched")
            return
        record.parent.rmdir(record.name)
    except FileNotFoundError:
        return
    except OSError as cleanup_error:
        failures.append(f"Created directory retained at {path}: {cleanup_error}")


def commit_owned_payload(
    target: Path, prepared_root: Path, relative_paths: Sequence[tuple[str, ...]],
    *, empty_dirs: Sequence[tuple[str, ...]] = (),
) -> None:
    """Commit under the lifecycle lock, rolling back our changes on an exception.

    Rollback uses the same directory objects even if another process renames them;
    it never undoes that process's rename. External-reader isolation and recovery
    after power loss are outside this in-process transaction.
    """
    from hermes_cli.profile_distribution_destination import open_directory

    paths = _checked_paths(relative_paths)
    directories = _checked_paths(empty_dirs)
    ordered = sorted(tuple(os.path.normcase(part) for part in parts) for parts in paths)
    if any(right[:len(left)] == left for left, right in zip(ordered, ordered[1:])):
        raise OSError("Distribution publication paths must be disjoint")

    with ExitStack() as handles:
        parent = open_directory(target.parent)
        handles.callback(parent.close)
        prepared = open_directory(prepared_root)
        handles.callback(prepared.close)
        if prepared.identity[0] != parent.identity[0]:
            raise OSError("Distribution publication requires preparation on the profile filesystem")
        sources = []
        for parts in paths:
            source = _descend(prepared, parts[:-1])
            info = source.stat(parts[-1])
            if info.st_dev != parent.identity[0] or (
                getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
            ) or not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
                raise OSError("Prepared distribution entries must be local regular files or directories")
            sources.append(source)

        # A separate private sibling survives the caller's scratch cleanup if a
        # restoration fails. Create it through the pinned parent too.
        backup_name = ".hermes-dist-rollback-" + secrets.token_hex(16)
        backup_path = parent.path / backup_name
        backup = None
        backup_created = []
        created = []
        journal = []
        try:
            parent.verify()
            backup = _create_directory(parent, backup_name, backup_created, mode=0o700)
            backup.verify()
            destination_root = _ensure_child(parent, target.name, created)
            if destination_root.identity[0] != prepared.identity[0]:
                raise OSError("Distribution publication requires the profile filesystem")
            for parts, source in zip(paths, sources):
                destination = _ensure_directory(destination_root, parts[:-1], created)
                saved = _ensure_directory(backup, parts[:-1], [])
                name = parts[-1]
                existed = _exists(destination, name)
                # Register before either rename, including exceptions raised by
                # the post-rename identity check when a parent was moved.
                journal.append((source, destination, saved, name, existed))
                if existed:
                    _replace(destination, name, saved)
                _replace(source, name, destination)
            for parts in directories:
                # Reuse existing user-owned bootstrap links without writing through them.
                if target.joinpath(*parts).is_dir():
                    continue
                _ensure_directory(destination_root, parts, created)
            destination_root.verify()
        except BaseException as exc:
            failures = []
            for source, destination, saved, name, existed in reversed(journal):
                try:
                    backed_up = _exists(saved, name)
                    if backed_up or not existed:
                        # Deliberately skip logical-path verification here: these
                        # anchors still refer to the directories we actually changed.
                        if not _exists(source, name) and _exists(destination, name):
                            destination.replace(name, source, name)
                        if backed_up:
                            saved.replace(name, destination, name)
                except OSError as rollback_error:
                    failures.append(f"{destination.path / name}: {rollback_error}")
            for record in reversed(created):
                _undo_created(record, failures)
            if backup is None:
                for record in reversed(backup_created):
                    _undo_created(record, failures)
            elif not failures:
                try:
                    _remove_backup(parent, backup_name, backup)
                except OSError as cleanup_error:
                    failures.append(f"backup cleanup: {cleanup_error}")
            if failures:
                message = (
                    f"Distribution rollback could not finish; recovery files retained at {backup_path}: "
                    + "; ".join(failures)
                )
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    exc.add_note(message)
                    raise
                raise OSError(message) from exc
            raise
        else:
            try:
                _remove_backup(parent, backup_name, backup)
            except OSError:
                logger.warning("Distribution committed; could not remove backups at %s", backup_path, exc_info=True)

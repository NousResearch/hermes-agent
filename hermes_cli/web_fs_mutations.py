"""Synchronous admission and effects for the browser's entry mutations.

These are accident-prevention guards, not an OS containment boundary. The SQLite
registry serializes tracked connections in this process, not other processes.
"""

from contextlib import ExitStack, contextmanager
from pathlib import Path
import errno
import os
import shutil
import stat

from fastapi import HTTPException

from hermes_cli.sqlite_safe_read import LiveConnectionError, offline_file_access


def _entry_guard(target: Path) -> None:
    from agent.file_safety import get_write_denied_error, is_write_approval_required
    from hermes_cli.web_routers import files

    files._refuse_sensitive_write(target, str(target))
    error = get_write_denied_error(str(target), entry=True)
    if error or is_write_approval_required(str(target)):
        raise HTTPException(status_code=403, detail="Mutating protected files is not allowed")


def _footprint(target: Path):
    """Walk entries without following links or junctions; traversal failures refuse the operation."""
    yield target
    mode = target.lstat().st_mode
    if not stat.S_ISDIR(mode) or target.is_junction():
        return
    with os.scandir(target) as entries:
        for entry in entries:
            yield from _footprint(Path(entry.path))


@contextmanager
def _offline_mutation(target: Path, destination: Path | None, *, scan: bool):
    try:
        # The first admission takes the canonical registry lock before traversal. Every
        # descendant is checked under that same reentrant lock, held through the effect.
        with ExitStack() as stack:
            stack.enter_context(offline_file_access(target, what="mutate"))
            for entry in _footprint(target) if scan else (target,):
                _entry_guard(entry)
                stack.enter_context(offline_file_access(entry, what="mutate"))
                if destination is not None:
                    mapped = destination / entry.relative_to(target)
                    _entry_guard(mapped)
                    stack.enter_context(offline_file_access(mapped, what="mutate"))
            yield
    except LiveConnectionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except OSError as exc:
        if isinstance(exc, PermissionError):
            raise HTTPException(status_code=403, detail="Mutation footprint is not accessible") from exc
        raise HTTPException(status_code=409, detail="Could not verify or mutate the path") from exc


def mutate_local(target: Path, request, profile: str | None, operation: str, *,
                 destination: Path | None = None, directory: bool = False, recursive: bool = False) -> None:
    from hermes_cli.web_routers import files

    with files._profile_scope(profile):
        files._fs_mutation_policy_guard(target, request)
        policy = files._managed_files_policy(request, create_root=False)
        for protected in (Path(target.anchor), Path.home(), policy.locked_root):
            if protected is not None and (target == protected or target in protected.parents):
                raise HTTPException(status_code=400, detail="Cannot mutate a protected root or its ancestor")
        if destination is not None:
            files._fs_mutation_policy_guard(destination, request)
        if operation != "create" and not os.path.lexists(target):
            raise HTTPException(status_code=404, detail="Path not found")
        scan = operation == "rename" or (operation == "delete" and recursive)
        with _offline_mutation(target, destination, scan=scan):
            if operation == "create":
                _create(target, directory)
            elif operation == "rename":
                _rename(target, destination)
            else:
                _delete(target, recursive)


def _create(target: Path, directory: bool) -> None:
    try:
        if directory:
            target.mkdir()
        else:
            fd = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
            os.close(fd)
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail="Path already exists") from exc


def _rename(target: Path, destination: Path) -> None:
    from hermes_cli.web_routers import files

    if destination == target:
        return
    try:
        files._rename_noreplace(target, destination)
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail="Path already exists") from exc
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except OSError as exc:
        if exc.errno in {errno.ENOSYS, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EINVAL}:
            raise HTTPException(status_code=501, detail="Atomic no-replace rename is unavailable") from exc
        raise


def _delete(target: Path, recursive: bool) -> None:
    if target.is_symlink():
        target.unlink()
    elif target.is_junction():
        target.rmdir()
    elif target.is_dir():
        if recursive:
            shutil.rmtree(target)
        else:
            try:
                target.rmdir()
            except OSError as exc:
                if exc.errno in (errno.ENOTEMPTY, errno.EEXIST):
                    raise HTTPException(status_code=409, detail="Directory is not empty (enable recursive delete)") from exc
                raise
    else:
        target.unlink()

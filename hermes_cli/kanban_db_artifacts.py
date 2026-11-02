"""Workspace-confined durable completion artifacts (PR #86610)."""
import contextlib
import inspect
import os
import re
import sqlite3
import stat
from pathlib import Path
from typing import Optional

def _connection_board(conn: sqlite3.Connection) -> str:
    """Derive board ownership from an open connection's main DB path."""
    from hermes_cli.kanban_db import DEFAULT_BOARD, kanban_home, boards_root, _normalize_board_slug, ArtifactPreservationError
    try:
        row = conn.execute("PRAGMA database_list").fetchone()
        db_path = Path(row["file"]).resolve(strict=False)
        default_db = (kanban_home() / "kanban.db").resolve(strict=False)
        if db_path == default_db:
            return DEFAULT_BOARD
        try:
            relative = db_path.relative_to(boards_root().resolve(strict=False))
        except ValueError:
            relative = None
        if relative is not None and len(relative.parts) == 2 and relative.parts[1] == "kanban.db":
            board = _normalize_board_slug(relative.parts[0])
            if board:
                return board
        # A nonstandard database needs an explicit pair, not ambient UI state.
        # Canonical paths above remain authoritative even with conflicting pins.
        pinned_db = os.environ.get("HERMES_KANBAN_DB", "").strip()
        pinned_board = _normalize_board_slug(os.environ.get("HERMES_KANBAN_BOARD"))
        if pinned_db and pinned_board and db_path == Path(pinned_db).expanduser().resolve(strict=False):
            return pinned_board
    except (OSError, RuntimeError, TypeError, ValueError):
        pass
    raise ArtifactPreservationError(
        "cannot determine artifact board ownership from the connection"
    )

def _scope_completion_artifacts(
    conn: sqlite3.Connection,
    task_id: str,
    metadata: Optional[dict],
) -> Optional[dict]:
    """Canonicalize declared artifacts below the task's workspace."""
    from hermes_cli.kanban_db import ArtifactPreservationError
    if not isinstance(metadata, dict) or "artifacts" not in metadata:
        return metadata
    raw_artifacts = metadata.get("artifacts")
    if not isinstance(raw_artifacts, (list, tuple)):
        raise ArtifactPreservationError("metadata.artifacts must be a list")
    if not raw_artifacts:
        return metadata

    row = conn.execute(
        "SELECT workspace_path FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    if not row or not row["workspace_path"]:
        raise ArtifactPreservationError(
            "completion artifacts require an available task workspace"
        )
    try:
        workspace_root = Path(row["workspace_path"]).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ArtifactPreservationError(
            "completion artifacts require an available task workspace"
        ) from exc
    if not workspace_root.is_dir():
        raise ArtifactPreservationError(
            "completion artifacts require an available task workspace"
        )

    canonical: list[str] = []
    seen: set[str] = set()
    for item in raw_artifacts:
        if not isinstance(item, str) or not item.strip():
            raise ArtifactPreservationError(
                "every completion artifact must be a non-empty path string"
            )
        candidate = Path(item).expanduser()
        if not candidate.is_absolute():
            candidate = workspace_root / candidate
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(workspace_root)
        except (OSError, RuntimeError, ValueError) as exc:
            raise ArtifactPreservationError(
                f"declared artifact is outside its task workspace: {item}"
            ) from exc
        if not resolved.is_file():
            raise ArtifactPreservationError(
                f"declared artifact is not a regular file: {item}"
            )
        normalized = str(resolved)
        if normalized not in seen:
            seen.add(normalized)
            canonical.append(normalized)

    updated = dict(metadata)
    updated["artifacts"] = canonical
    return updated

_SECRET_SUFFIXES = (".pem", ".key", ".p12", ".pfx", ".jks", ".keystore")
_SECRET_NAMES = {"credentials", "credentials.json", "auth.json", "secrets.json",
                 "secrets.yaml", "secrets.yml", "vault.key"}
_SECRET_PREFIXES = ("id_rsa", "id_dsa", "id_ecdsa", "id_ed25519")


def _looks_secret(relative: Path) -> bool:
    """Hidden path component or a conventional key/credential file name."""
    if any(part.startswith(".") for part in relative.parts):
        return True
    name = relative.name.lower()
    return (name in _SECRET_NAMES or name.endswith(_SECRET_SUFFIXES)
            or name.startswith(_SECRET_PREFIXES))


def _merge_completion_prose_artifacts(
    conn: sqlite3.Connection,
    task_id: str,
    metadata: Optional[dict],
    *,
    summary: Optional[str],
    result: Optional[str],
) -> Optional[dict]:
    """Promote existing workspace files named in legacy completion prose.

    ``artifacts=[...]`` is preferred. Older workers only wrote an absolute
    deliverable path in ``summary``/``result``; discover it before completion
    so it can be staged outside the worker-controlled workspace.
    """
    row = conn.execute(
        "SELECT workspace_path, workspace_kind FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    # A repository or persistent directory contains inputs and secrets, not
    # only deliverables. Mentioning a path is not consent to attach it.
    if not row or row["workspace_kind"] != "scratch" or not row["workspace_path"]:
        return metadata
    workspace = Path(row["workspace_path"]).expanduser()
    # Same guard as scratch cleanup (#28818): a board default_workdir on a
    # real source tree can still be paired with workspace_kind='scratch'.
    from hermes_cli.kanban_db_workspace import _is_managed_scratch_path
    if not _is_managed_scratch_path(workspace):
        return metadata
    try:
        workspace_root = workspace.resolve(strict=True)
    except (OSError, RuntimeError):
        return metadata
    text = "\n".join(part for part in (summary, result) if part)
    if not text:
        return metadata
    prefix = re.escape(str(workspace))
    discovered: list[str] = []
    for match in re.finditer(prefix + r"(?:[/\\][^\s`\"'<>]+)", text):
        raw = match.group(0).rstrip(".,;:!?)]}")
        candidate = Path(raw)
        try:
            resolved = candidate.resolve(strict=True)
            relative = resolved.relative_to(workspace_root)
        except (OSError, RuntimeError, ValueError):
            continue
        # Prose is not a declaration: never infer hidden or credential files.
        # An explicit ``artifacts=[...]`` entry remains the way to ship one.
        if _looks_secret(relative):
            continue
        if resolved.is_file():
            discovered.append(str(resolved))
    if not discovered:
        return metadata
    updated = dict(metadata) if isinstance(metadata, dict) else {}
    existing = updated.get("artifacts")
    merged = list(existing) if isinstance(existing, (list, tuple)) else []
    seen = {str(path) for path in merged}
    for path in discovered:
        if path not in seen:
            merged.append(path)
            seen.add(path)
    updated["artifacts"] = merged
    return updated

def _persist_completion_artifacts(
    conn: sqlite3.Connection,
    task_id: str,
    metadata: dict,
) -> None:
    """Stage completion artifacts outside the worker-controlled workspace."""
    from hermes_cli.kanban_db import ArtifactPreservationError, task_attachments_dir
    raw_artifacts = metadata.get("artifacts")
    if not isinstance(raw_artifacts, (list, tuple)):
        return
    if not raw_artifacts:
        return

    row = conn.execute(
        "SELECT workspace_path FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    if not row or not row["workspace_path"]:
        return

    workspace = Path(row["workspace_path"]).expanduser()
    board = _connection_board(conn)
    if (
        not task_id
        or task_id in {".", ".."}
        or "/" in task_id
        or "\\" in task_id
        or Path(task_id).name != task_id
    ):
        raise ArtifactPreservationError("task id cannot name an attachment directory")

    try:
        workspace_root = workspace.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ArtifactPreservationError(
            "completion artifacts require an available task workspace"
        ) from exc

    attachment_dir = task_attachments_dir(task_id, board=board)
    persisted: list[str] = []
    used_destinations: set[Path] = set()
    changed = False

    try:
        attachment_dir = anchor_to_configured_root(attachment_dir)
        with _open_pinned_attachment_directory(attachment_dir) as (
            pinned_attachment_dir,
            attachment_dir_fd,
        ):
            def _discard_copies() -> None:
                for copied in used_destinations:
                    try:
                        if attachment_dir_fd is None:
                            copied.unlink(missing_ok=True)
                        else:
                            os.unlink(copied.name, dir_fd=attachment_dir_fd)
                    except OSError:
                        pass

            for item in raw_artifacts:
                artifact = str(item).strip() if isinstance(item, str) else ""
                if not artifact:
                    continue
                dest = _stage_one_artifact(
                    artifact, workspace_root, pinned_attachment_dir,
                    attachment_dir_fd, used_destinations, _discard_copies,
                )
                used_destinations.add(dest)
                persisted.append(str(dest))
                changed = True
    except ArtifactPreservationError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise ArtifactPreservationError(
            f"could not prepare the durable artifact directory: {exc}"
        ) from exc

    if changed:
        metadata["artifacts"] = persisted
        metadata["_staged_artifacts"] = list(persisted)

def _stage_one_artifact(artifact, workspace_root, pinned_attachment_dir,
                        attachment_dir_fd, used_destinations, _discard_copies):
    """Validate, open and copy one workspace file without following a swapped leaf."""
    from hermes_cli.kanban_db import ArtifactPreservationError, KANBAN_ATTACHMENT_MAX_BYTES
    src = Path(artifact).expanduser()
    try:
        resolved_src = src.resolve(strict=True)
    except OSError as exc:
        _discard_copies()
        raise ArtifactPreservationError(
            f"declared completion artifact is unavailable: {artifact}"
        ) from exc

    if not resolved_src.is_relative_to(workspace_root):
        _discard_copies()
        raise ArtifactPreservationError(
            f"declared artifact is outside its task workspace: {artifact}"
        )

    try:
        expected_stat = os.stat(resolved_src, follow_symlinks=False)
    except OSError as exc:
        _discard_copies()
        raise ArtifactPreservationError(
            f"declared completion artifact is unavailable or not a regular file: {artifact}"
        ) from exc
    if not stat.S_ISREG(expected_stat.st_mode):
        _discard_copies()
        raise ArtifactPreservationError(
            f"declared completion artifact is unavailable or not a regular file: {artifact}"
        )
    if expected_stat.st_size > KANBAN_ATTACHMENT_MAX_BYTES:
        _discard_copies()
        raise ArtifactPreservationError(
            f"declared completion artifact exceeds the "
            f"{KANBAN_ATTACHMENT_MAX_BYTES}-byte limit: {artifact}"
        )

    dest: Optional[Path] = None
    destination_fd: Optional[int] = None
    try:
        dest, destination_fd = _open_unique_attachment_file(
            pinned_attachment_dir,
            attachment_dir_fd,
            resolved_src.name,
            used_destinations,
        )
        source_fd = _open_workspace_source(workspace_root, resolved_src)
        with (
            os.fdopen(source_fd, "rb") as source_file,
            os.fdopen(destination_fd, "wb") as destination_file,
        ):
            opened_stat = os.fstat(source_file.fileno())
            if not stat.S_ISREG(opened_stat.st_mode):
                raise ArtifactPreservationError(
                    f"declared completion artifact is not a regular file: {artifact}"
                )
            if (
                opened_stat.st_dev != expected_stat.st_dev
                or opened_stat.st_ino != expected_stat.st_ino
            ):
                raise ArtifactPreservationError(
                    f"declared completion artifact changed before it could be copied: {artifact}"
                )
            if opened_stat.st_size > KANBAN_ATTACHMENT_MAX_BYTES:
                raise ArtifactPreservationError(
                    f"declared completion artifact exceeds the "
                    f"{KANBAN_ATTACHMENT_MAX_BYTES}-byte limit: {artifact}"
                )
            copied = 0
            while chunk := source_file.read(1024 * 1024):
                copied += len(chunk)
                if copied > KANBAN_ATTACHMENT_MAX_BYTES:
                    raise ArtifactPreservationError(
                        f"declared completion artifact grew beyond the size limit: {artifact}"
                    )
                destination_file.write(chunk)
    except Exception as exc:
        if destination_fd is not None:
            with contextlib.suppress(OSError):
                os.close(destination_fd)
        if dest is not None:
            try:
                if attachment_dir_fd is None:
                    dest.unlink(missing_ok=True)
                else:
                    os.unlink(dest.name, dir_fd=attachment_dir_fd)
            except OSError:
                pass
        _discard_copies()
        if isinstance(exc, ArtifactPreservationError):
            raise
        raise ArtifactPreservationError(
            f"could not preserve declared completion artifact {artifact}: {exc}"
        ) from exc

    return dest


def _open_workspace_source(workspace_root: Path, source: Path) -> int:
    """Open a confined source without following a symlink below the workspace.

    ``source`` was resolved inside ``workspace_root``, but O_NOFOLLOW only
    covers the leaf: a parent directory swapped for a symlink after that
    check would redirect the open outside the workspace. Walking again from
    pinned directory descriptors makes such a swap fail instead. O_NONBLOCK
    keeps a leaf swapped for a FIFO from blocking; the caller still requires
    a regular file with the expected identity.
    """
    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    if not _attachment_dir_fd_supported():
        return _open_source_checked_after_open(workspace_root, source, flags)
    directory_flags = (
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    )
    current_fd = os.open(workspace_root, directory_flags)
    try:
        for component in source.relative_to(workspace_root).parts[:-1]:
            next_fd = os.open(component, directory_flags, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        return os.open(source.name, flags, dir_fd=current_fd)
    finally:
        os.close(current_fd)


def _open_source_checked_after_open(workspace_root: Path, source: Path, flags: int) -> int:
    """Path-based open for platforms without descriptor-relative IO (Windows).

    Refusing every artifact there would remove the feature it protects.
    Instead the opened descriptor must still be the file that the confined
    path names after the open: a parent swapped beforehand resolves outside
    the workspace or to another inode. On Windows a directory holding an
    open file cannot be renamed, so the path cannot be swapped back while
    the descriptor is held.
    """
    fd = os.open(source, flags)
    try:
        opened = os.fstat(fd)
        current = source.resolve(strict=True)
        named = os.stat(current, follow_symlinks=False)
        if (
            not current.is_relative_to(workspace_root)
            or (named.st_dev, named.st_ino) != (opened.st_dev, opened.st_ino)
        ):
            raise OSError(f"workspace source changed while it was opened: {source}")
    except BaseException:
        os.close(fd)
        raise
    return fd


def _unique_attachment_path(directory: Path, filename: str, used: set[Path]) -> Path:
    """Return the next candidate path under ``directory`` for ``filename``."""
    safe_name = Path(filename).name or "artifact"
    candidate = directory / safe_name
    if candidate not in used:
        return candidate

    stem = Path(safe_name).stem or "artifact"
    suffix = Path(safe_name).suffix
    idx = 1
    while True:
        candidate = directory / f"{stem}_{idx}{suffix}"
        if candidate not in used:
            return candidate
        idx += 1

def anchor_to_configured_root(path: Path) -> Path:
    """Return ``path`` with only the configured Kanban root canonicalised.

    Configuration owns the root, not the worker: an alias above it (e.g.
    /var -> /private/var on macOS) is resolved once, while every board,
    attachments and task component below it is left for no-follow checks.
    """
    from hermes_cli.kanban_db import kanban_home
    configured_root = Path(os.path.abspath(kanban_home().expanduser()))
    absolute = Path(os.path.abspath(path.expanduser()))
    if absolute.is_relative_to(configured_root):
        return configured_root.resolve(strict=True) / absolute.relative_to(configured_root)
    return absolute

def _attachment_dir_fd_supported() -> bool:
    """Return whether this platform supports descriptor-relative directory IO.

    ``supports_dir_fd`` holds the os builtins themselves; a pass-through
    wrapper (an I/O audit hook, the test suite's home guard) must not
    silently downgrade every caller to the weaker lstat fallback.
    """
    supported = getattr(os, "supports_dir_fd", set())
    return (
        hasattr(os, "O_DIRECTORY")
        and hasattr(os, "O_NOFOLLOW")
        and all(inspect.unwrap(fn) in supported for fn in (os.open, os.mkdir, os.unlink))
    )

@contextlib.contextmanager
def _open_pinned_attachment_directory(directory: Path):
    """Create and pin every destination component without following symlinks.

    On POSIX, every component is opened relative to its already-pinned parent
    using ``O_NOFOLLOW``. The returned descriptor remains the authority for
    file creation and cleanup even if the visible pathname is swapped later.
    Platforms without descriptor-relative IO still reject every symlink via
    ``lstat`` before opening the leaf; file creation remains ``O_EXCL`` and
    ``O_NOFOLLOW`` where available.
    """
    absolute = Path(os.path.abspath(os.fspath(directory.expanduser())))
    if not absolute.is_absolute() or not absolute.anchor:
        raise ValueError("attachment directory must resolve to an absolute path")

    if not _attachment_dir_fd_supported():
        current = Path(absolute.anchor)
        for component in absolute.parts[1:]:
            current /= component
            try:
                os.mkdir(current, mode=0o700)
            except FileExistsError:
                pass
            info = os.lstat(current)
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise OSError(f"unsafe attachment directory component: {current}")
        yield absolute, None
        return

    directory_flags = (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )
    current_fd = os.open(absolute.anchor, directory_flags)
    try:
        for component in absolute.parts[1:]:
            try:
                os.mkdir(component, mode=0o700, dir_fd=current_fd)
            except FileExistsError:
                pass
            next_fd = os.open(component, directory_flags, dir_fd=current_fd)
            try:
                if not stat.S_ISDIR(os.fstat(next_fd).st_mode):
                    raise NotADirectoryError(component)
            except Exception:
                os.close(next_fd)
                raise
            previous_fd = current_fd
            current_fd = next_fd
            os.close(previous_fd)
        yield absolute, current_fd
    finally:
        os.close(current_fd)

def _open_unique_attachment_file(
    directory: Path,
    directory_fd: Optional[int],
    filename: str,
    used: set[Path],
) -> tuple[Path, int]:
    """Atomically create a collision-free regular file in a pinned directory."""
    unavailable: set[Path] = set()
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    while True:
        destination = _unique_attachment_path(
            directory,
            filename,
            used | unavailable,
        )
        try:
            if directory_fd is None:
                fd = os.open(destination, flags, 0o600)
            else:
                fd = os.open(destination.name, flags, 0o600, dir_fd=directory_fd)
        except FileExistsError:
            unavailable.add(destination)
            continue
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode):
            os.close(fd)
            raise OSError(f"attachment destination is not a regular file: {destination}")
        return destination, fd

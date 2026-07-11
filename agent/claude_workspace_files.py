"""Race-safe workspace file capabilities for Claude subscription workers."""

from __future__ import annotations

import os
import stat
import errno
import threading
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Iterator


class WorkspaceFileBroker:
    """Read/write files relative to an immutable root directory descriptor."""

    MAX_FILE_BYTES = 2 * 1024 * 1024
    MAX_TURN_WRITE_BYTES = 8 * 1024 * 1024

    def __init__(self, workspace: str | Path) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self._root_fd = os.open(
            self.workspace, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
        )
        self._write_lock = threading.Lock()
        self._turn_written_bytes = 0

    def begin_turn(self) -> None:
        """Reset the bounded write budget for one SDK query/response turn."""
        with self._write_lock:
            self._turn_written_bytes = 0

    def close(self) -> None:
        if self._root_fd >= 0:
            os.close(self._root_fd)
            self._root_fd = -1

    def _parts(self, raw: Any) -> tuple[str, ...]:
        path = PurePosixPath(str(raw or ""))
        if path.is_absolute() or not path.parts or any(
            part in {"", ".", ".."} for part in path.parts
        ):
            raise RuntimeError("Workspace file path must be a relative path")
        return path.parts

    @contextmanager
    def _parent(self, parts: tuple[str, ...]) -> Iterator[tuple[int, str]]:
        current = os.dup(self._root_fd)
        try:
            for part in parts[:-1]:
                next_fd = os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                    dir_fd=current,
                )
                os.close(current)
                current = next_fd
            yield current, parts[-1]
        finally:
            os.close(current)

    def _read(self, arguments: dict[str, Any]) -> str:
        parts = self._parts(arguments.get("path"))
        with self._parent(parts) as (parent_fd, name):
            fd = os.open(
                name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent_fd
            )
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise RuntimeError("Workspace reads reject non-regular or linked files")
            if info.st_size > self.MAX_FILE_BYTES:
                raise RuntimeError("Workspace read exceeds the 2 MiB safety limit")
            with os.fdopen(fd, "r", encoding="utf-8", errors="replace") as handle:
                fd = -1
                lines = handle.read(self.MAX_FILE_BYTES + 1).splitlines()
        finally:
            if fd >= 0:
                os.close(fd)
        offset = max(int(arguments.get("offset") or 1), 1)
        limit = max(min(int(arguments.get("limit") or 500), 5000), 1)
        selected = lines[offset - 1 : offset - 1 + limit]
        return "\n".join(
            f"{line_number}: {line}"
            for line_number, line in enumerate(selected, start=offset)
        )

    def _write(self, arguments: dict[str, Any]) -> dict[str, Any]:
        parts = self._parts(arguments.get("path"))
        content = str(arguments.get("content") or "").encode("utf-8")
        if len(content) > self.MAX_FILE_BYTES:
            raise RuntimeError("Workspace write exceeds the 2 MiB safety limit")
        with self._write_lock:
            next_total = self._turn_written_bytes + len(content)
            if next_total > self.MAX_TURN_WRITE_BYTES:
                raise RuntimeError("Workspace writes exceed the 8 MiB per-turn safety limit")
            self._turn_written_bytes = next_total
        with self._parent(parts) as (parent_fd, name):
            flags = os.O_WRONLY | os.O_NOFOLLOW | os.O_CLOEXEC
            try:
                fd = os.open(name, flags, dir_fd=parent_fd)
            except FileNotFoundError:
                try:
                    fd = os.open(
                        name,
                        flags | os.O_CREAT | os.O_EXCL,
                        0o600,
                        dir_fd=parent_fd,
                    )
                except FileExistsError as exc:
                    raise RuntimeError("Workspace path changed during write") from exc
            except OSError as exc:
                if exc.errno == errno.ELOOP:
                    raise RuntimeError("Workspace writes reject symbolic links") from exc
                raise
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise RuntimeError("Workspace writes reject non-regular or linked files")
            os.ftruncate(fd, 0)
            view = memoryview(content)
            while view:
                view = view[os.write(fd, view) :]
            os.fsync(fd)
        finally:
            os.close(fd)
        return {"success": True, "path": "/".join(parts), "bytes": len(content)}

    def handle(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        if tool_name == "read_file":
            return self._read(arguments)
        if tool_name == "write_file":
            return self._write(arguments)
        raise RuntimeError(f"Unsupported workspace file capability: {tool_name}")


__all__ = ["WorkspaceFileBroker"]

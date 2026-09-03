"""Descriptor-relative, no-follow builder file tools."""

from __future__ import annotations

import os
import secrets
import fnmatch
from pathlib import Path, PurePosixPath

from .errors import AdapterError
from .gitops import AllowedPathManifest, safe_relative_path


class ConfinedTools:
    def __init__(
        self,
        root: str | Path,
        manifest: AllowedPathManifest,
        readable_paths: frozenset[str] | set[str],
    ):
        self.root = Path(root).resolve(strict=True)
        self.manifest = manifest
        self.readable_paths = frozenset(readable_paths)

    def _parent_fd(self, raw: str, *, create: bool) -> tuple[int, str]:
        relative = safe_relative_path(raw)
        parts = PurePosixPath(relative).parts
        fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            for component in parts[:-1]:
                try:
                    child = os.open(
                        component,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                        dir_fd=fd,
                    )
                except FileNotFoundError:
                    if not create:
                        raise
                    os.mkdir(component, mode=0o700, dir_fd=fd)
                    child = os.open(
                        component,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                        dir_fd=fd,
                    )
                os.close(fd)
                fd = child
            return fd, parts[-1]
        except OSError as exc:
            os.close(fd)
            raise AdapterError(
                "MANIFEST_MISMATCH", "unsafe or symlinked directory component"
            ) from exc
        except Exception:
            os.close(fd)
            raise

    def read_file(self, path: str, *, max_bytes: int = 1_000_000) -> str:
        relative = safe_relative_path(path)
        if relative not in self.readable_paths:
            raise AdapterError("MANIFEST_MISMATCH", "file is not readable")
        parent_fd, name = self._parent_fd(relative, create=False)
        try:
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd)
            try:
                stat = os.fstat(fd)
                if stat.st_nlink != 1 or stat.st_size > max_bytes:
                    raise AdapterError(
                        "MANIFEST_MISMATCH", "file unavailable, linked, or oversized"
                    )
                data = os.read(fd, max_bytes + 1)
                if len(data) > max_bytes:
                    raise AdapterError("MANIFEST_MISMATCH", "file oversized")
                return data.decode("utf-8")
            finally:
                os.close(fd)
        except (OSError, UnicodeDecodeError) as exc:
            raise AdapterError("MANIFEST_MISMATCH", "file read rejected") from exc
        finally:
            os.close(parent_fd)

    def write_file(self, path: str, content: str) -> None:
        relative = safe_relative_path(path)
        if not self.manifest.permits(relative):
            raise AdapterError("MANIFEST_MISMATCH", "write path is not permitted")
        parent_fd, name = self._parent_fd(relative, create=True)
        temporary = f".builder-{secrets.token_hex(16)}"
        try:
            try:
                existing = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd)
            except FileNotFoundError:
                existing = None
            if existing is not None:
                try:
                    if os.fstat(existing).st_nlink != 1:
                        raise AdapterError("MANIFEST_MISMATCH", "hard link forbidden")
                finally:
                    os.close(existing)
            fd = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=parent_fd,
            )
            try:
                data = content.encode("utf-8")
                offset = 0
                while offset < len(data):
                    offset += os.write(fd, data[offset:])
                os.fsync(fd)
            finally:
                os.close(fd)
            os.replace(
                temporary,
                name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
            os.fsync(parent_fd)
        except OSError as exc:
            try:
                os.unlink(temporary, dir_fd=parent_fd)
            except OSError:
                pass
            raise AdapterError("MANIFEST_MISMATCH", "file write rejected") from exc
        finally:
            os.close(parent_fd)

    def search_files(self, pattern: str) -> list[str]:
        if "/" in pattern or "\\" in pattern or pattern in {"", ".", ".."}:
            raise AdapterError("INVALID_REQUEST", "unsafe search pattern")
        results = []
        for relative in sorted(self.readable_paths):
            if not fnmatch.fnmatchcase(PurePosixPath(relative).name, pattern):
                continue
            try:
                parent_fd, name = self._parent_fd(relative, create=False)
                try:
                    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd)
                    try:
                        stat = os.fstat(fd)
                        if stat.st_nlink == 1:
                            results.append(relative)
                    finally:
                        os.close(fd)
                finally:
                    os.close(parent_fd)
            except OSError:
                continue
        return results

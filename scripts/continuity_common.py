"""Shared primitives for the local Hermes continuity pilot."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import platform
import re
import secrets
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import psutil
import yaml


class ContinuityError(RuntimeError):
    """A safe, user-readable continuity failure."""


@dataclass(frozen=True)
class GitState:
    root: str
    branch: str
    commit: str
    dirty: bool
    changed_files: tuple[str, ...]
    fingerprint: str
    integration_ref: str | None = None
    integration_sha: str | None = None
    merge_base: str | None = None

    def as_dict(self) -> dict[str, Any]:
        value = {
            "root": self.root,
            "branch": self.branch,
            "commit": self.commit,
            "dirty": self.dirty,
            "changed_files": list(self.changed_files),
            "fingerprint": self.fingerprint,
        }
        if self.integration_ref is not None:
            value.update({
                "integration_ref": self.integration_ref,
                "integration_sha": self.integration_sha,
                "merge_base": self.merge_base,
            })
        return value


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContinuityError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContinuityError(f"expected a JSON object in {path}")
    return value


def read_markdown_frontmatter(path: Path) -> tuple[dict[str, Any], str]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ContinuityError(f"cannot read authority {path}: {exc}") from exc
    return parse_markdown_frontmatter(text, path)


def parse_markdown_frontmatter(text: str, path: Path) -> tuple[dict[str, Any], str]:
    if len(text) > 256_000:
        raise ContinuityError(f"authority is unexpectedly large: {path}")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ContinuityError(f"missing YAML front matter in {path}")
    try:
        end = next(
            index
            for index, line in enumerate(lines[1:], start=1)
            if line.strip() == "---"
        )
    except StopIteration as exc:
        raise ContinuityError(f"unterminated YAML front matter in {path}") from exc
    try:
        data = yaml.safe_load("\n".join(lines[1:end])) or {}
    except yaml.YAMLError as exc:
        raise ContinuityError(f"invalid YAML front matter in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ContinuityError(f"expected YAML mapping in {path}")
    return data, "\n".join(lines[end + 1 :]).lstrip("\n")


def render_markdown_frontmatter(data: dict[str, Any], body: str) -> str:
    header = yaml.safe_dump(data, sort_keys=False, allow_unicode=True).strip()
    suffix = body.rstrip() + "\n" if body else ""
    return f"---\n{header}\n---\n\n{suffix}"


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ContinuityError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def minimal_child_env(
    extra: dict[str, str] | None = None,
    *,
    state_root: str | Path,
    external_volume: str | Path,
) -> dict[str, str]:
    """Return a credential-minimized, state-confined child environment."""
    allowed = {
        "PATH",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TERM",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "SYSTEMROOT",
        "WINDIR",
    }
    env = {key: value for key, value in os.environ.items() if key in allowed}
    env["PATH"] = os.defpath
    resolved_state = validate_state_storage(external_volume, state_root)
    expected_temp_root = resolved_state / "tmp"
    try:
        temp_root = expected_temp_root.resolve(strict=True)
    except OSError as exc:
        raise ContinuityError(f"child temp root is unavailable: {exc}") from exc
    if temp_root != expected_temp_root or not temp_root.is_dir():
        raise ContinuityError(
            f"child temp root is not a confined direct directory: {temp_root}"
        )
    env.update({
        "HOME": str(resolved_state),
        "TMPDIR": str(temp_root),
        "TEMP": str(temp_root),
        "TMP": str(temp_root),
    })
    if extra:
        env.update(extra)
    for name in ("HOME", "TMPDIR", "TEMP", "TMP"):
        path = Path(env[name]).resolve()
        if not path.is_relative_to(resolved_state):
            raise ContinuityError(f"child {name} escapes pilot state root: {path}")
    return env


def validate_state_storage(external_volume: str | Path, state_root: str | Path) -> Path:
    """Validate a real mount and an existing state directory beneath it."""
    secure_dirfd = (
        os.open in os.supports_dir_fd
        and os.mkdir in os.supports_dir_fd
        and os.unlink in os.supports_dir_fd
        and bool(getattr(os, "O_DIRECTORY", 0))
        and bool(getattr(os, "O_NOFOLLOW", 0))
    )
    if not secure_dirfd and os.name != "nt":
        raise ContinuityError(
            "secure confined state access is unavailable on this platform"
        )
    volume = Path(external_volume)
    if not external_volume_available(volume):
        raise ContinuityError(f"external volume is not mounted: {volume}")
    try:
        mounted = volume.resolve(strict=True)
        raw_state = Path(os.path.abspath(state_root))
        state = raw_state.resolve(strict=True)
    except OSError as exc:
        raise ContinuityError(f"pilot state storage is unavailable: {exc}") from exc
    if not state.is_dir() or not state.is_relative_to(mounted):
        raise ContinuityError(f"pilot state root escapes external volume: {state}")
    if os.name == "nt":
        _windows_require_plain_path(raw_state, mounted)
    return state


def _windows_require_plain_path(path: Path, root: Path) -> None:
    """Reject Windows reparse points from root through an existing path."""
    reparse_point = 0x400
    normalized = Path(os.path.abspath(path))
    try:
        relative = normalized.relative_to(root)
    except ValueError as exc:
        raise ContinuityError(f"state path escapes pilot state root: {path}") from exc
    candidates = (
        root,
        *(
            root / Path(*relative.parts[:index])
            for index in range(1, len(relative.parts) + 1)
        ),
    )
    for current in candidates:
        try:
            metadata = os.lstat(current)
        except OSError as exc:
            raise ContinuityError(
                f"state path is unavailable: {current}: {exc}"
            ) from exc
        if getattr(metadata, "st_file_attributes", 0) & reparse_point:
            raise ContinuityError(f"state path contains a reparse point: {current}")


def _windows_reject_reparse_target(path: Path) -> None:
    """Reject an existing Windows target leaf when it is a reparse point."""
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise ContinuityError(f"state target is unavailable: {path}: {exc}") from exc
    if getattr(metadata, "st_file_attributes", 0) & 0x400:
        raise ContinuityError(f"state target is a reparse point: {path}")


@contextmanager
def _windows_hold_directory_chain(path: Path, root: Path) -> Iterator[None]:
    """Retain non-delete-shared handles so checked directories cannot be swapped."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    invalid = ctypes.c_void_p(-1).value
    relative = path.relative_to(root)
    candidates = (
        root,
        *(
            root / Path(*relative.parts[:index])
            for index in range(1, len(relative.parts) + 1)
        ),
    )
    handles: list[int] = []
    try:
        for candidate in candidates:
            handle = kernel32.CreateFileW(
                str(candidate),
                0x0080,  # FILE_READ_ATTRIBUTES
                0x0001 | 0x0002,  # share read/write, deliberately not delete
                None,
                3,  # OPEN_EXISTING
                0x02000000 | 0x00200000,  # BACKUP_SEMANTICS | OPEN_REPARSE_POINT
                None,
            )
            if handle == invalid:
                raise ContinuityError(
                    "cannot retain checked Windows state directory: "
                    f"{candidate}: {ctypes.get_last_error()}"
                )
            handles.append(int(handle))
        _windows_require_plain_path(path, root)
        yield
    finally:
        for handle in reversed(handles):
            kernel32.CloseHandle(wintypes.HANDLE(handle))


def _windows_open_regular_fd(path: Path, *, append: bool = False) -> int:
    """Open a Windows leaf itself, never its reparse target."""
    import ctypes
    import msvcrt
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.GetFileAttributesW.argtypes = [wintypes.LPCWSTR]
    kernel32.GetFileAttributesW.restype = wintypes.DWORD
    kernel32.GetFileType.argtypes = [wintypes.HANDLE]
    kernel32.GetFileType.restype = wintypes.DWORD
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    invalid = ctypes.c_void_p(-1).value
    desired_access = 0x0004 if append else 0x80000000
    disposition = 4 if append else 3  # OPEN_ALWAYS or OPEN_EXISTING
    handle = kernel32.CreateFileW(
        str(path),
        desired_access,
        0x0001 | 0x0002,
        None,
        disposition,
        0x00200000,  # FILE_FLAG_OPEN_REPARSE_POINT
        None,
    )
    if handle == invalid:
        error = ctypes.get_last_error()
        if error in {2, 3}:
            raise FileNotFoundError(error, "Windows state target is unavailable", path)
        raise ContinuityError(f"cannot open Windows state target safely: {error}")
    try:
        attributes = kernel32.GetFileAttributesW(str(path))
        if attributes == 0xFFFFFFFF or attributes & 0x400:
            raise ContinuityError(f"state target is a reparse point: {path}")
        if attributes & 0x10 or kernel32.GetFileType(handle) != 0x0001:
            raise ContinuityError(f"state target is not a regular file: {path}")
        flags = (os.O_APPEND | os.O_WRONLY) if append else os.O_RDONLY
        descriptor = msvcrt.open_osfhandle(int(handle), flags)
        handle = None
        return descriptor
    finally:
        if handle is not None:
            kernel32.CloseHandle(handle)


def _secure_dirfd_available() -> bool:
    return (
        os.open in os.supports_dir_fd
        and os.mkdir in os.supports_dir_fd
        and os.unlink in os.supports_dir_fd
        and bool(getattr(os, "O_DIRECTORY", 0))
        and bool(getattr(os, "O_NOFOLLOW", 0))
    )


@contextmanager
def confined_parent(
    config: dict[str, Any], path: Path
) -> Iterator[tuple[int | None, str | Path]]:
    """Open a target parent relative to an already-validated state root."""
    state = validate_state_storage(config["external_volume"], config["state_root"])
    normalized = Path(os.path.abspath(path))
    try:
        relative = normalized.relative_to(state)
    except ValueError as exc:
        raise ContinuityError(f"state path escapes pilot state root: {path}") from exc
    if len(relative.parts) < 1:
        raise ContinuityError(f"state path has no target name: {path}")
    parent_parts = relative.parts[:-1]
    if not _secure_dirfd_available():
        if os.name != "nt":
            raise ContinuityError(
                "secure confined state access requires directory descriptor support"
            )
        parent = normalized.parent
        with _windows_hold_directory_chain(parent, state):
            if parent.resolve(strict=True) != parent:
                raise ContinuityError(f"state path parent is not direct: {parent}")
            _windows_reject_reparse_target(normalized)
            validate_state_storage(config["external_volume"], config["state_root"])
            yield None, normalized
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(state, flags)
    except OSError as exc:
        raise ContinuityError(f"cannot open confined state root: {exc}") from exc
    try:
        try:
            for part in parent_parts:
                next_descriptor = os.open(part, flags, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = next_descriptor
        except OSError as exc:
            raise ContinuityError(
                f"cannot open confined state parent for {path}: {exc}"
            ) from exc
        validate_state_storage(config["external_volume"], config["state_root"])
        yield descriptor, relative.name
    finally:
        os.close(descriptor)


def confined_atomic_write_text(
    config: dict[str, Any], path: Path, content: str
) -> None:
    data = content.encode("utf-8")
    with confined_parent(config, path) as (parent_fd, target):
        _require_regular_leaf(parent_fd, target, missing_ok=True)
        if parent_fd is None:
            target_path = Path(target)
            temporary = target_path.parent / (
                f".{target_path.name}.{os.getpid()}.{time.time_ns()}.tmp"
            )
            try:
                descriptor = os.open(
                    temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
                )
                try:
                    _write_all(descriptor, data)
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
                validate_state_storage(config["external_volume"], config["state_root"])
                _windows_reject_reparse_target(target_path)
                os.replace(temporary, target_path)
            except BaseException as exc:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass
                if isinstance(exc, OSError):
                    raise ContinuityError(
                        f"cannot write confined state file {path}: {exc}"
                    ) from exc
                raise
            return
        temporary = f".{target}.{os.getpid()}.{time.time_ns()}.tmp"
        try:
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=parent_fd,
            )
            try:
                _write_all(descriptor, data)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.replace(temporary, target, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
            os.fsync(parent_fd)
        except BaseException as exc:
            try:
                os.unlink(temporary, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
            if isinstance(exc, OSError):
                raise ContinuityError(
                    f"cannot write confined state file {path}: {exc}"
                ) from exc
            raise


def confined_atomic_write_json(
    config: dict[str, Any], path: Path, value: dict[str, Any]
) -> None:
    confined_atomic_write_text(
        config, path, json.dumps(value, indent=2, sort_keys=True) + "\n"
    )


def confined_append_text(config: dict[str, Any], path: Path, content: str) -> None:
    data = content.encode("utf-8")
    with confined_parent(config, path) as (parent_fd, target):
        flags = (
            os.O_WRONLY
            | os.O_APPEND
            | os.O_CREAT
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        try:
            descriptor = (
                _windows_open_regular_fd(Path(target), append=True)
                if parent_fd is None and os.name == "nt"
                else os.open(target, flags, 0o600)
                if parent_fd is None
                else os.open(target, flags, 0o600, dir_fd=parent_fd)
            )
        except OSError as exc:
            raise ContinuityError(
                f"cannot open confined append file {path}: {exc}"
            ) from exc
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ContinuityError(
                    f"confined append target is not a regular file: {path}"
                )
            _write_all(descriptor, data)
            os.fsync(descriptor)
        except OSError as exc:
            raise ContinuityError(
                f"cannot append confined state file {path}: {exc}"
            ) from exc
        finally:
            os.close(descriptor)


def confined_read_bytes(config: dict[str, Any], path: Path) -> bytes:
    with confined_parent(config, path) as (parent_fd, target):
        flags = (
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        )
        try:
            descriptor = (
                _windows_open_regular_fd(Path(target))
                if parent_fd is None and os.name == "nt"
                else os.open(target, flags)
                if parent_fd is None
                else os.open(target, flags, dir_fd=parent_fd)
            )
        except FileNotFoundError:
            raise
        except OSError as exc:
            raise ContinuityError(
                f"cannot open confined state file {path}: {exc}"
            ) from exc
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ContinuityError(
                    f"confined state file is not a regular file: {path}"
                )
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    return b"".join(chunks)
                chunks.append(chunk)
        finally:
            os.close(descriptor)


def confined_file_checkpoint(
    config: dict[str, Any], path: Path
) -> tuple[int, tuple[int, int]]:
    with confined_parent(config, path) as (parent_fd, target):
        flags = (
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        )
        try:
            descriptor = (
                _windows_open_regular_fd(Path(target))
                if parent_fd is None and os.name == "nt"
                else os.open(target, flags)
                if parent_fd is None
                else os.open(target, flags, dir_fd=parent_fd)
            )
        except FileNotFoundError:
            raise
        except OSError as exc:
            raise ContinuityError(
                f"cannot checkpoint confined state file {path}: {exc}"
            ) from exc
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise ContinuityError(
                    f"confined state file is not a regular file: {path}"
                )
            return metadata.st_size, (metadata.st_dev, metadata.st_ino)
        finally:
            os.close(descriptor)


def confined_read_range(
    config: dict[str, Any], path: Path, *, offset: int, max_bytes: int
) -> tuple[bytes, int, tuple[int, int]]:
    if offset < 0 or max_bytes < 1:
        raise ContinuityError("confined read range is invalid")
    with confined_parent(config, path) as (parent_fd, target):
        flags = (
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        )
        try:
            descriptor = (
                _windows_open_regular_fd(Path(target))
                if parent_fd is None and os.name == "nt"
                else os.open(target, flags)
                if parent_fd is None
                else os.open(target, flags, dir_fd=parent_fd)
            )
        except FileNotFoundError:
            raise
        except OSError as exc:
            raise ContinuityError(
                f"cannot open confined state range {path}: {exc}"
            ) from exc
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise ContinuityError(
                    f"confined state file is not a regular file: {path}"
                )
            if metadata.st_size < offset:
                raise ContinuityError("confined state file was truncated")
            available = metadata.st_size - offset
            if available > max_bytes:
                raise ContinuityError(f"confined state range exceeds {max_bytes} bytes")
            os.lseek(descriptor, offset, os.SEEK_SET)
            chunks: list[bytes] = []
            remaining = available
            while remaining:
                chunk = os.read(descriptor, min(remaining, 64 * 1024))
                if not chunk:
                    raise ContinuityError(
                        "confined state file changed during ranged read"
                    )
                chunks.append(chunk)
                remaining -= len(chunk)
            return (
                b"".join(chunks),
                metadata.st_size,
                (metadata.st_dev, metadata.st_ino),
            )
        finally:
            os.close(descriptor)


def _require_regular_leaf(
    parent_fd: int | None, target: str | Path, *, missing_ok: bool
) -> None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = (
            _windows_open_regular_fd(Path(target))
            if parent_fd is None and os.name == "nt"
            else os.open(target, flags)
            if parent_fd is None
            else os.open(target, flags, dir_fd=parent_fd)
        )
    except FileNotFoundError:
        if missing_ok:
            return
        raise
    except OSError as exc:
        raise ContinuityError(f"state target cannot be opened safely: {exc}") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ContinuityError("state target is not a regular file")
    finally:
        os.close(descriptor)


def confined_read_text(config: dict[str, Any], path: Path) -> str:
    try:
        return confined_read_bytes(config, path).decode("utf-8")
    except FileNotFoundError:
        raise
    except (OSError, UnicodeDecodeError) as exc:
        raise ContinuityError(f"cannot read confined state file {path}: {exc}") from exc


def confined_load_json(config: dict[str, Any], path: Path) -> dict[str, Any]:
    try:
        value = json.loads(confined_read_text(config, path))
    except json.JSONDecodeError as exc:
        raise ContinuityError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContinuityError(f"expected a JSON object in {path}")
    return value


def confined_ensure_dir(config: dict[str, Any], name: str) -> Path:
    if not name or "/" in name or "\\" in name or name in {".", ".."}:
        raise ContinuityError(f"invalid state directory name: {name!r}")
    state = validate_state_storage(config["external_volume"], config["state_root"])
    if not _secure_dirfd_available():
        if os.name != "nt":
            raise ContinuityError(
                "secure confined state directory creation requires directory descriptor support"
            )
        target = state / name
        with _windows_hold_directory_chain(state, state):
            _windows_reject_reparse_target(target)
            validate_state_storage(config["external_volume"], config["state_root"])
            try:
                os.mkdir(target, mode=0o700)
            except FileExistsError:
                pass
            with _windows_hold_directory_chain(target, state):
                if not target.is_dir():
                    raise ContinuityError(
                        f"state directory is not a confined direct directory: {target}"
                    )
        return target
    try:
        descriptor = os.open(
            state,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise ContinuityError(f"cannot open confined state root: {exc}") from exc
    try:
        validate_state_storage(config["external_volume"], config["state_root"])
        try:
            os.mkdir(name, mode=0o700, dir_fd=descriptor)
        except FileExistsError:
            pass
        try:
            child_descriptor = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=descriptor,
            )
        except OSError as exc:
            raise ContinuityError(
                f"state directory is not a confined direct directory: {state / name}"
            ) from exc
        else:
            os.close(child_descriptor)
    finally:
        os.close(descriptor)
    return state / name


def _write_all(descriptor: int, data: bytes) -> None:
    offset = 0
    while offset < len(data):
        written = os.write(descriptor, data[offset:])
        if written <= 0:
            raise OSError("confined write made no progress")
        offset += written


def verify_pinned_executable(
    config: dict[str, Any], repo_root: Path, tool_name: str, executable: Path
) -> str:
    """Bind an authority-bearing executable to the committed platform digest."""
    lock_path = repo_root / config.get(
        "toolchain_lock", ".continuity/toolchain.lock.json"
    )
    lock = load_json(lock_path)
    tool = (lock.get("tools") or {}).get(tool_name)
    if not isinstance(tool, dict):
        raise ContinuityError(f"toolchain lock has no {tool_name!r} entry")
    system = platform.system().lower()
    machine = (
        platform
        .machine()
        .lower()
        .replace("aarch64", "arm64")
        .replace("x86_64", "amd64")
    )
    digest_key = f"{system}-{machine}-sha256"
    expected = tool.get(digest_key)
    if not isinstance(expected, str) or len(expected) != 64:
        raise ContinuityError(
            f"toolchain lock has no valid {digest_key} for {tool_name}"
        )
    try:
        resolved = executable.resolve(strict=True)
    except OSError as exc:
        raise ContinuityError(
            f"cannot resolve pinned {tool_name} executable: {exc}"
        ) from exc
    state_root = Path(config["state_root"]).resolve()
    if not resolved.is_relative_to(state_root):
        raise ContinuityError(
            f"{tool_name} executable escapes pilot state root: {resolved}"
        )
    actual = sha256_file(resolved)
    if actual != expected:
        raise ContinuityError(
            f"{tool_name} executable digest mismatch: expected {expected}, got {actual}"
        )
    return actual


def external_input_digests(config: dict[str, Any], repo_root: Path) -> dict[str, str]:
    inputs: dict[str, str] = {}
    for item in config.get("external_instructions", []):
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise ContinuityError(
                "external instruction pins require exactly path and sha256"
            )
        path = Path(str(item["path"]))
        if not path.is_absolute():
            path = repo_root / path
        path = path.resolve()
        expected = item["sha256"]
        if not isinstance(expected, str) or len(expected) != 64:
            raise ContinuityError(f"external instruction has invalid SHA-256: {path}")
        actual = sha256_file(path)
        if not hmac.compare_digest(actual, expected):
            raise ContinuityError(
                f"external instruction digest mismatch: expected {expected}, got {actual}"
            )
        inputs[str(path)] = actual
    for item in config.get("evidence_executables", []):
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise ContinuityError(
                "evidence executable pins require exactly path and sha256"
            )
        path = Path(str(item["path"]))
        if not path.is_absolute():
            path = repo_root / path
        path = path.resolve()
        expected = item["sha256"]
        actual = sha256_file(path)
        if not isinstance(expected, str) or not hmac.compare_digest(actual, expected):
            raise ContinuityError(
                f"evidence executable digest mismatch: expected {expected}, got {actual}"
            )
        inputs[str(path)] = actual
    beads_path = Path(config["beads"]["binary"])
    inputs[str(beads_path.resolve())] = verify_pinned_executable(
        config, repo_root, "beads", beads_path
    )
    return dict(sorted(inputs.items()))


def _receipt_key_path(config: dict[str, Any]) -> Path:
    return Path(config["state_root"]) / "receipt-signing.key"


def _load_receipt_key(config: dict[str, Any], *, create: bool) -> bytes:
    path = _receipt_key_path(config)
    with confined_parent(config, path) as (parent_fd, target):
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        if create:
            try:
                descriptor = (
                    os.open(target, flags, 0o600)
                    if parent_fd is None
                    else os.open(target, flags, 0o600, dir_fd=parent_fd)
                )
            except FileExistsError:
                pass
            except OSError as exc:
                raise ContinuityError(
                    f"cannot create receipt signing key safely: {exc}"
                ) from exc
            else:
                try:
                    _write_all(descriptor, secrets.token_bytes(32))
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
                if parent_fd is not None:
                    os.fsync(parent_fd)
        read_flags = (
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        )
        try:
            descriptor = (
                _windows_open_regular_fd(Path(target))
                if parent_fd is None and os.name == "nt"
                else os.open(target, read_flags)
                if parent_fd is None
                else os.open(target, read_flags, dir_fd=parent_fd)
            )
            try:
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode):
                    raise ContinuityError("receipt signing key must be a regular file")
                key = os.read(descriptor, 33)
                mode = metadata.st_mode & 0o777
            finally:
                os.close(descriptor)
        except OSError as exc:
            raise ContinuityError(f"cannot read receipt signing key: {exc}") from exc
    if len(key) != 32 or mode & 0o077:
        raise ContinuityError("receipt signing key must be 32 bytes with mode 0600")
    return key


def _receipt_auth_payload(receipt: dict[str, Any]) -> bytes:
    unsigned = {key: value for key, value in receipt.items() if key != "auth"}
    return json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def receipt_signing_key(config: dict[str, Any]) -> bytes:
    return _load_receipt_key(config, create=True)


def sign_receipt(
    config: dict[str, Any], receipt: dict[str, Any], *, key: bytes | None = None
) -> dict[str, str]:
    key = key or _load_receipt_key(config, create=True)
    return {
        "algorithm": "hmac-sha256",
        "key_id": hashlib.sha256(key).hexdigest()[:16],
        "digest": hmac.new(
            key, _receipt_auth_payload(receipt), hashlib.sha256
        ).hexdigest(),
    }


def receipt_signature_errors(
    config: dict[str, Any], receipt: dict[str, Any]
) -> list[str]:
    auth = receipt.get("auth")
    if not isinstance(auth, dict) or set(auth) != {"algorithm", "key_id", "digest"}:
        return ["receipt authentication record is absent or malformed"]
    if auth.get("algorithm") != "hmac-sha256":
        return ["receipt authentication algorithm is invalid"]
    try:
        key = _load_receipt_key(config, create=False)
    except ContinuityError as exc:
        return [str(exc)]
    expected_id = hashlib.sha256(key).hexdigest()[:16]
    expected_digest = hmac.new(
        key, _receipt_auth_payload(receipt), hashlib.sha256
    ).hexdigest()
    errors: list[str] = []
    if not hmac.compare_digest(str(auth.get("key_id")), expected_id):
        errors.append("receipt signing key identity does not match")
    if not hmac.compare_digest(str(auth.get("digest")), expected_digest):
        errors.append("receipt signature is invalid")
    return errors


def _terminate_process_tree(
    process: subprocess.Popen[str],
    *,
    process_group: int | None = None,
    tracked_descendants: tuple[psutil.Process, ...] = (),
) -> None:
    """Boundedly terminate a command and its snapshotted descendants.

    ``psutil.Process`` retains the process identity (PID plus creation time),
    so the escalation pass cannot accidentally target a recycled PID.  Walk
    children before the parent to prevent a live parent from replacing exited
    descendants while cleanup is in progress.  This is the repository's
    cross-platform process-tree primitive.  A retained POSIX group identity
    additionally covers descendants after their leader exits; Windows callers
    use a kill-on-close Job Object established before the child is resumed.
    """
    try:
        parent = psutil.Process(process.pid)
        current_descendants = parent.children(recursive=True)
    except psutil.NoSuchProcess:
        current_descendants = []
        parent = None
    except (psutil.AccessDenied, OSError):
        current_descendants = []
        parent = None
    descendants = list(dict.fromkeys((*tracked_descendants, *current_descendants)))

    if process_group is not None and os.name == "posix":
        try:
            kill_process_group = os.killpg  # windows-footgun: ok -- POSIX gate
            kill_process_group(process_group, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            pass

    def is_alive(target: psutil.Process) -> bool:
        try:
            return target.is_running() and target.status() != psutil.STATUS_ZOMBIE
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            return False

    descendants.reverse()
    for target in descendants:
        try:
            target.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            pass
    # Escalate descendants before stopping the parent.  Once the parent exits,
    # its live children are reparented and can no longer be rediscovered by a
    # process-tree walk.
    for target in descendants:
        if not is_alive(target):
            continue
        try:
            target.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            pass

    if parent is not None:
        try:
            parent.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            pass
        deadline = time.monotonic() + 0.75
        while time.monotonic() < deadline and is_alive(parent):
            time.sleep(0.025)
        if is_alive(parent):
            try:
                parent.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                pass

    if process.poll() is None:
        try:
            process.kill()
        except (ProcessLookupError, PermissionError, OSError):
            pass
    if process_group is not None and os.name == "posix":
        try:
            os.killpg(  # windows-footgun: ok -- POSIX gate
                process_group, getattr(signal, "SIGKILL", signal.SIGTERM)
            )
        except (ProcessLookupError, PermissionError, OSError):
            pass
    try:
        process.communicate(timeout=0.75)
    except subprocess.TimeoutExpired:
        # A process that survives the platform's hard-kill primitive is no
        # longer safe to wait for. The caller still fails closed.
        pass


def _isolated_process_options(*, suspended: bool = False) -> dict[str, Any]:
    """Return host-native options that isolate a spawned command."""
    if os.name == "posix":
        return {"start_new_session": True}
    if os.name == "nt":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP
        if suspended:
            flags |= 0x00000004  # CREATE_SUSPENDED
        return {"creationflags": flags}
    return {}


_LINUX_SUBREAPER = r"""
import ctypes, os, signal, subprocess, sys, time

expected_parent_pid = int(sys.argv[1])
cleanup_ack_fd = int(sys.argv[2])

def acknowledge(status):
    try:
        os.write(cleanup_ack_fd, (status + '\n').encode('ascii'))
    except OSError:
        pass
    try:
        os.close(cleanup_ack_fd)
    except OSError:
        pass

libc = ctypes.CDLL(None, use_errno=True)
if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
    acknowledge('CONTAINMENT_ERROR')
    raise SystemExit(125)

child = None
pending_signal = None

def direct_children():
    found = []
    for name in os.listdir('/proc'):
        if not name.isdigit() or int(name) == os.getpid():
            continue
        try:
            raw = open(f'/proc/{name}/stat', encoding='utf-8').read()
            fields = raw[raw.rfind(')') + 2:].split()
            if int(fields[1]) == os.getpid():
                found.append(int(name))
        except (FileNotFoundError, PermissionError, ValueError, IndexError):
            pass
    return found

def signal_children(signum):
    for pid in direct_children():
        try:
            os.kill(pid, signum)
        except ProcessLookupError:
            pass

def reap_exited():
    while True:
        try:
            pid, _status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            return

def cleanup():
    soft_deadline = time.monotonic() + 0.2
    while time.monotonic() < soft_deadline:
        signal_children(signal.SIGTERM)
        reap_exited()
        if not direct_children():
            return
        time.sleep(0.01)
    # Killing one generation can cause its still-living descendants to be
    # adopted by this subreaper.  Rescan after every reap until the complete
    # adopted tree is gone, rather than assuming one kill pass is sufficient.
    hard_deadline = time.monotonic() + 2.0
    while time.monotonic() < hard_deadline:
        signal_children(signal.SIGKILL)
        reap_exited()
        if not direct_children():
            reap_exited()
            if not direct_children():
                return
        time.sleep(0.01)
    signal_children(signal.SIGKILL)
    reap_exited()
    if direct_children():
        acknowledge('CLEANUP_ERROR')
        raise SystemExit(125)

def request_termination(signum, _frame):
    global pending_signal
    if pending_signal is None:
        pending_signal = signum

def terminate(signum):
    if child is not None and child.poll() is None:
        try:
            child.terminate()
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=0.2)
        except subprocess.TimeoutExpired:
            try:
                child.kill()
            except ProcessLookupError:
                pass
            child.wait()
    cleanup()
    acknowledge('CLEANUP_OK')
    raise SystemExit(128 + signum)

signal.signal(signal.SIGTERM, request_termination)
signal.signal(signal.SIGINT, request_termination)
# A caller may block termination signals to close its spawn-to-handler race.
# The wrapper inherits that mask, so unblock only after its passive handlers
# are installed; otherwise cooperative cleanup can never receive termination.
signal.pthread_sigmask(signal.SIG_UNBLOCK, (signal.SIGTERM, signal.SIGINT))
# If the controlling Hermes process is killed, retain the wrapper long enough
# to run its bounded adopted-descendant cleanup. Check for the standard race in
# which the parent exits between recording its PID and arming PDEATHSIG.
if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
    acknowledge('CONTAINMENT_ERROR')
    raise SystemExit(125)
if os.getppid() != expected_parent_pid:
    terminate(signal.SIGTERM)
# Keep the acknowledgement capability private from the target. Linux resets
# dumpability for an exec'd target, but the wrapper itself remains protected.
if libc.prctl(4, 0, 0, 0, 0) != 0:  # PR_SET_DUMPABLE
    acknowledge('CONTAINMENT_ERROR')
    raise SystemExit(125)
if pending_signal is not None:
    terminate(pending_signal)
try:
    child = subprocess.Popen(
        sys.argv[3:], env=os.environ.copy(), close_fds=True
    )
except OSError:
    acknowledge('CONTAINMENT_ERROR')
    raise SystemExit(126)
while child.poll() is None:
    # The Python signal handler only records intent. It never touches Popen,
    # so an interruption while poll() holds its waitpid lock cannot re-enter it.
    if pending_signal is not None:
        terminate(pending_signal)
    time.sleep(0.05)
returncode = child.returncode
cleanup()
acknowledge('CLEANUP_OK')
raise SystemExit(returncode)
"""


def _native_posix_containment_command(
    args: list[str], *, cleanup_ack_fd: int | None = None
) -> list[str]:
    system = platform.system().lower()
    if system == "darwin":
        raise ContinuityError(
            "native macOS descendant containment is unavailable without a "
            "privileged helper"
        )
    if system == "linux" and Path("/proc").is_dir():
        if cleanup_ack_fd is None:
            raise ContinuityError(
                "Linux native containment requires a cleanup acknowledgement channel"
            )
        return [
            sys.executable,
            "-I",
            "-c",
            _LINUX_SUBREAPER,
            str(os.getpid()),
            str(cleanup_ack_fd),
            *args,
        ]
    raise ContinuityError(
        f"native descendant containment is unavailable on POSIX {system or 'unknown'}"
    )


def _create_windows_kill_job(process: subprocess.Popen[str]) -> int:
    """Contain a suspended Windows child in a kill-on-close Job Object."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    ntdll = ctypes.WinDLL("ntdll", use_last_error=True)
    kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    ]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]
    ntdll.NtResumeProcess.restype = wintypes.LONG

    class IoCounters(ctypes.Structure):
        _fields_ = [
            (name, ctypes.c_ulonglong)
            for name in (
                "read_operations",
                "write_operations",
                "other_operations",
                "read_bytes",
                "write_bytes",
                "other_bytes",
            )
        ]

    class BasicLimits(ctypes.Structure):
        _fields_ = [
            ("per_process_time", ctypes.c_longlong),
            ("per_job_time", ctypes.c_longlong),
            ("limit_flags", wintypes.DWORD),
            ("minimum_working_set", ctypes.c_size_t),
            ("maximum_working_set", ctypes.c_size_t),
            ("active_process_limit", wintypes.DWORD),
            ("affinity", ctypes.c_size_t),
            ("priority_class", wintypes.DWORD),
            ("scheduling_class", wintypes.DWORD),
        ]

    class ExtendedLimits(ctypes.Structure):
        _fields_ = [
            ("basic", BasicLimits),
            ("io", IoCounters),
            ("process_memory", ctypes.c_size_t),
            ("job_memory", ctypes.c_size_t),
            ("peak_process_memory", ctypes.c_size_t),
            ("peak_job_memory", ctypes.c_size_t),
        ]

    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")
    try:
        limits = ExtendedLimits()
        limits.basic.limit_flags = 0x00002000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(
            job, 9, ctypes.byref(limits), ctypes.sizeof(limits)
        ):
            raise OSError(ctypes.get_last_error(), "SetInformationJobObject failed")
        process_handle = wintypes.HANDLE(int(process._handle))
        if not kernel32.AssignProcessToJobObject(job, process_handle):
            raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject failed")
        status = ntdll.NtResumeProcess(process_handle)
        if status != 0:
            raise OSError(int(status), "NtResumeProcess failed")
        return int(job)
    except BaseException:
        kernel32.CloseHandle(job)
        raise


def _close_windows_job(handle: int) -> None:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    if not kernel32.CloseHandle(wintypes.HANDLE(handle)):
        raise OSError(ctypes.get_last_error(), "CloseHandle failed for process job")


@dataclass
class _ContainedProcess:
    process: subprocess.Popen[str]
    family_token: str
    process_group: int | None = None
    windows_job: int | None = None
    windows_directory_lease: ExitStack | None = None
    linux_subreaper: bool = False
    linux_cleanup_ack_fd: int | None = None
    terminated: bool = False
    tracked_descendants: dict[tuple[int, float], psutil.Process] = field(
        default_factory=dict
    )
    descendant_lock: threading.Lock = field(default_factory=threading.Lock)
    monitor_stop: threading.Event = field(default_factory=threading.Event)
    monitor_thread: threading.Thread | None = None

    def start_descendant_monitor(self) -> None:
        if os.name != "posix":
            return
        self.snapshot_descendants()
        self.monitor_thread = threading.Thread(
            target=self._monitor_descendants,
            name=f"continuity-descendants-{self.process.pid}",
            daemon=True,
        )
        self.monitor_thread.start()

    def _monitor_descendants(self) -> None:
        while not self.monitor_stop.wait(5.0):
            self.snapshot_descendants()
            if self.process.poll() is not None:
                return

    def snapshot_process_family(self) -> None:
        if os.name != "posix":
            return
        ps_binary = Path("/bin/ps")
        if not ps_binary.is_file():
            return
        scanner_env = dict(os.environ)
        scanner_env.pop("CONTINUITY_PROCESS_FAMILY", None)
        try:
            listing = subprocess.run(
                [str(ps_binary), "eww", "-axo", "pid=,command="],
                capture_output=True,
                check=False,
                env=scanner_env,
                text=True,
                timeout=0.2,
            )
        except (OSError, subprocess.TimeoutExpired):
            return
        needle = f"CONTINUITY_PROCESS_FAMILY={self.family_token}"
        for line in listing.stdout.splitlines()[:512]:
            if needle not in line:
                continue
            try:
                pid = int(line.lstrip().split(None, 1)[0])
            except (ValueError, IndexError):
                continue
            if pid in {os.getpid(), self.process.pid}:
                continue
            try:
                candidate = psutil.Process(pid)
                identity = (candidate.pid, candidate.create_time())
            except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                continue
            with self.descendant_lock:
                self.tracked_descendants[identity] = candidate

    def snapshot_descendants(self) -> None:
        try:
            descendants = psutil.Process(self.process.pid).children(recursive=True)
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            return
        with self.descendant_lock:
            for descendant in descendants:
                try:
                    identity = (descendant.pid, descendant.create_time())
                except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                    continue
                self.tracked_descendants[identity] = descendant

    def _finish_descendant_monitor(self) -> tuple[psutil.Process, ...]:
        self.snapshot_descendants()
        self.monitor_stop.set()
        if self.monitor_thread is not None:
            self.monitor_thread.join(timeout=0.1)
            self.monitor_thread = None
        self.snapshot_descendants()
        if os.name == "posix" and self.process_group is None:
            self.snapshot_process_family()
        with self.descendant_lock:
            return tuple(self.tracked_descendants.values())

    def _require_linux_cleanup_ack(self, *, fallback_kill: bool) -> None:
        descriptor = self.linux_cleanup_ack_fd
        self.linux_cleanup_ack_fd = None
        if descriptor is None:
            raise ContinuityError(
                "Linux native containment cleanup acknowledgement is unavailable"
            )
        if fallback_kill:
            os.close(descriptor)
            raise ContinuityError(
                "Linux native containment required a last-resort wrapper kill"
            )
        try:
            import select

            readable, _writable, _exceptional = select.select(
                [descriptor], [], [descriptor], 0.1
            )
            if not readable:
                raise ContinuityError(
                    "Linux native containment cleanup acknowledgement timed out"
                )
            acknowledgement = os.read(descriptor, 64)
        except OSError as exc:
            raise ContinuityError(
                "Linux native containment cleanup acknowledgement could not be read"
            ) from exc
        finally:
            os.close(descriptor)
        if acknowledgement == b"CLEANUP_OK\n":
            return
        if acknowledgement in {
            b"CLEANUP_ERROR\n",
            b"CONTAINMENT_ERROR\n",
        }:
            raise ContinuityError("Linux native containment reported cleanup failure")
        raise ContinuityError(
            "Linux native containment cleanup acknowledgement is missing or invalid"
        )

    def terminate_tree(self) -> None:
        if self.terminated:
            return
        self.terminated = True
        try:
            self._terminate_tree_owned()
        finally:
            descriptor = self.linux_cleanup_ack_fd
            self.linux_cleanup_ack_fd = None
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass

    def _terminate_tree_owned(self) -> None:
        tracked_descendants = self._finish_descendant_monitor()
        with self.descendant_lock:
            prior_identities = set(self.tracked_descendants)
        job_error: OSError | None = None
        linux_fallback_kill = False
        if self.windows_job is not None:
            try:
                _close_windows_job(self.windows_job)
            except OSError as exc:
                job_error = exc
            self.windows_job = None
        try:
            if self.linux_subreaper:
                if self.process.poll() is None:
                    try:
                        self.process.terminate()
                    except (ProcessLookupError, PermissionError, OSError):
                        pass
                    try:
                        # The wrapper reserves 0.2s for cooperative child exit
                        # and 2.2s for iterative adopted-generation cleanup.
                        # Leave scheduling margin before the last-resort kill.
                        self.process.wait(timeout=3.5)
                    except subprocess.TimeoutExpired:
                        linux_fallback_kill = True
                        _terminate_process_tree(
                            self.process,
                            process_group=self.process_group,
                            tracked_descendants=tracked_descendants,
                        )
                # A completed wrapper already reaped its adopted tree. Do not
                # signal its released process-group number, which may be reused.
                self._require_linux_cleanup_ack(fallback_kill=linux_fallback_kill)
            else:
                _terminate_process_tree(
                    self.process,
                    process_group=self.process_group,
                    tracked_descendants=tracked_descendants,
                )
            if os.name == "posix" and self.process_group is None:
                self.snapshot_process_family()
                with self.descendant_lock:
                    late_descendants = tuple(
                        candidate
                        for identity, candidate in self.tracked_descendants.items()
                        if identity not in prior_identities
                    )
                if late_descendants:
                    _terminate_process_tree(
                        self.process,
                        tracked_descendants=late_descendants,
                    )
        finally:
            if self.windows_directory_lease is not None:
                self.windows_directory_lease.close()
                self.windows_directory_lease = None
        if job_error is not None:
            raise ContinuityError(
                f"Windows process containment could not be released: {job_error}"
            ) from job_error


def _spawn_contained_process(
    args: list[str],
    *,
    popen_factory: Any | None = None,
    require_native_containment: bool = False,
    **kwargs: Any,
) -> _ContainedProcess:
    """Spawn with native containment when the caller requires that boundary."""
    suspended = os.name == "nt"
    factory = popen_factory or subprocess.Popen
    family_token = secrets.token_hex(16)
    requested_env = kwargs.get("env")
    spawn_env = dict(os.environ if requested_env is None else requested_env)
    spawn_env["CONTINUITY_PROCESS_FAMILY"] = family_token
    spawn_args = args
    process_options = _isolated_process_options(suspended=suspended)
    linux_subreaper = False
    linux_cleanup_ack_read: int | None = None
    linux_cleanup_ack_write: int | None = None
    if os.name == "posix":
        if require_native_containment:
            linux_subreaper = platform.system().lower() == "linux"
            if linux_subreaper:
                linux_cleanup_ack_read, linux_cleanup_ack_write = os.pipe()
                os.set_inheritable(linux_cleanup_ack_write, True)
                process_options["pass_fds"] = (linux_cleanup_ack_write,)
            try:
                spawn_args = _native_posix_containment_command(
                    args, cleanup_ack_fd=linux_cleanup_ack_write
                )
            except BaseException:
                if linux_cleanup_ack_read is not None:
                    os.close(linux_cleanup_ack_read)
                if linux_cleanup_ack_write is not None:
                    os.close(linux_cleanup_ack_write)
                raise
    kwargs["env"] = spawn_env
    directory_lease: ExitStack | None = None
    if os.name == "nt":
        directory_lease = ExitStack()
        try:
            env = kwargs.get("env")
            if isinstance(env, dict):
                directory_paths = {
                    Path(os.path.abspath(value))
                    for name, value in env.items()
                    if name in {"HOME", "TMPDIR", "TEMP", "TMP"}
                    and isinstance(value, str)
                    and value
                }
                for directory in sorted(directory_paths, key=str):
                    directory_lease.enter_context(
                        _windows_hold_directory_chain(directory, Path(directory.anchor))
                    )
        except BaseException:
            directory_lease.close()
            raise
    try:
        process = factory(
            spawn_args,
            **kwargs,
            **process_options,
        )
    except BaseException as exc:
        if linux_cleanup_ack_read is not None:
            os.close(linux_cleanup_ack_read)
        if linux_cleanup_ack_write is not None:
            os.close(linux_cleanup_ack_write)
        if directory_lease is not None:
            directory_lease.close()
        if isinstance(exc, OSError):
            raise ContinuityError(f"command failed to start: {args[0]}: {exc}") from exc
        raise
    if linux_cleanup_ack_write is not None:
        os.close(linux_cleanup_ack_write)
    if os.name == "nt":
        try:
            job = _create_windows_kill_job(process)
        except BaseException as exc:
            try:
                process.kill()
            except OSError:
                pass
            try:
                process.communicate(timeout=0.75)
            except (OSError, subprocess.TimeoutExpired):
                pass
            if directory_lease is not None:
                directory_lease.close()
            raise ContinuityError(
                f"command containment failed before start: {args[0]}: {exc}"
            ) from exc
        return _ContainedProcess(
            process=process,
            family_token=family_token,
            windows_job=job,
            windows_directory_lease=directory_lease,
        )
    contained = _ContainedProcess(
        process=process,
        family_token=family_token,
        process_group=process.pid if os.name == "posix" else None,
        linux_subreaper=linux_subreaper,
        linux_cleanup_ack_fd=linux_cleanup_ack_read,
    )
    try:
        contained.start_descendant_monitor()
    except BaseException:
        try:
            contained.terminate_tree()
        except BaseException:
            pass
        raise
    return contained


def run_command(
    args: list[str],
    *,
    cwd: Path,
    timeout: float,
    env: dict[str, str] | None = None,
    required_mount: tuple[str | Path, str | Path] | None = None,
    require_native_containment: bool = False,
) -> subprocess.CompletedProcess[str]:
    if required_mount is not None:
        validate_state_storage(*required_mount)
    contained = _spawn_contained_process(
        args,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        require_native_containment=require_native_containment,
    )
    process = contained.process
    deadline = time.monotonic() + timeout
    while True:
        if required_mount is not None:
            try:
                validate_state_storage(*required_mount)
            except ContinuityError as exc:
                contained.terminate_tree()
                raise ContinuityError(
                    f"required external storage became unavailable during command: {args[0]}"
                ) from exc
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            contained.terminate_tree()
            raise ContinuityError(f"command timed out after {timeout:g}s: {args[0]}")
        try:
            stdout, stderr = process.communicate(timeout=min(0.1, remaining))
            break
        except subprocess.TimeoutExpired:
            continue
        except BaseException:
            contained.terminate_tree()
            raise
    contained.terminate_tree()
    return subprocess.CompletedProcess(
        args=args,
        returncode=process.returncode,
        stdout=stdout,
        stderr=stderr,
    )


def git_state(
    cwd: Path, timeout: float = 10, integration_ref: str | None = None
) -> GitState:
    def git(*args: str) -> str:
        result = run_command(["git", *args], cwd=cwd, timeout=timeout)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise ContinuityError(f"git {' '.join(args)} failed: {detail}")
        return result.stdout.strip()

    root = str(Path(git("rev-parse", "--show-toplevel")).resolve())
    branch = git("branch", "--show-current") or "DETACHED"
    commit = git("rev-parse", "HEAD")
    status_result = run_command(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=cwd,
        timeout=timeout,
    )
    if status_result.returncode != 0:
        detail = (status_result.stderr or status_result.stdout).strip()
        raise ContinuityError(f"git status failed: {detail}")
    status = status_result.stdout.rstrip("\r\n")
    changed = tuple(
        line[3:] if len(line) >= 4 else line for line in status.splitlines()
    )
    digest = hashlib.sha256()
    diff_result = subprocess.run(
        ["git", "diff", "--binary", "HEAD", "--"],
        cwd=cwd,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if diff_result.returncode != 0:
        raise ContinuityError("git diff failed while fingerprinting exact state")
    digest.update(diff_result.stdout)
    untracked_result = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=cwd,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if untracked_result.returncode != 0:
        raise ContinuityError("git ls-files failed while fingerprinting exact state")
    for encoded in sorted(
        item for item in untracked_result.stdout.split(b"\0") if item
    ):
        relative = os.fsdecode(encoded)
        candidate = Path(root) / relative
        digest.update(encoded)
        if candidate.is_symlink():
            digest.update(
                os.readlink(candidate).encode("utf-8", errors="surrogateescape")
            )
        elif candidate.is_file():
            digest.update(candidate.read_bytes())
    integration_sha = git("rev-parse", integration_ref) if integration_ref else None
    merge_base = git("merge-base", integration_ref, "HEAD") if integration_ref else None
    return GitState(
        root,
        branch,
        commit,
        bool(status),
        changed,
        digest.hexdigest(),
        integration_ref,
        integration_sha,
        merge_base,
    )


def external_volume_available(path: Path) -> bool:
    """Require a mounted volume, not merely a same-named internal directory."""
    try:
        return path.exists() and path.is_mount()
    except OSError:
        return False


def compact_json(value: dict[str, Any], max_chars: int) -> str:
    rendered = json.dumps(value, sort_keys=True, separators=(",", ":"))
    if len(rendered) <= max_chars:
        return rendered
    minimal = {
        "schema_version": value.get("schema_version", 1),
        "status": "BLOCKED",
        "completion_allowed": False,
        "errors": ["preflight output exceeded configured bound"],
    }
    return json.dumps(minimal, sort_keys=True, separators=(",", ":"))


def receipt_errors(receipt: dict[str, Any], current: GitState) -> list[str]:
    """Return every reason an exact-state receipt cannot authorize promotion."""
    errors: list[str] = []
    allowed_receipt = {
        "schema_version",
        "gate",
        "project",
        "repo_id",
        "lifecycle_target",
        "risk_tier",
        "created_at",
        "git",
        "authority_check",
        "external_inputs",
        "dependency_identity",
        "commands",
        "runtime_checks",
        "rollback",
        "result",
        "auth",
    }
    unknown_receipt = sorted(set(receipt) - allowed_receipt)
    if unknown_receipt:
        errors.append("receipt contains unknown keys: " + ", ".join(unknown_receipt))
    if receipt.get("gate") != "hermes-continuity-gate/2":
        errors.append("receipt was not emitted by hermes-continuity-gate/2")
    auth = receipt.get("auth")
    if not isinstance(auth, dict) or set(auth) != {"algorithm", "key_id", "digest"}:
        errors.append("receipt authentication record is absent or malformed")
    external_inputs = receipt.get("external_inputs")
    if not isinstance(external_inputs, dict) or not external_inputs:
        errors.append("receipt has no external input digests")
    dependency_identity = receipt.get("dependency_identity")
    dependency_keys = {
        "schema_version",
        "requirements_sha256",
        "python_launcher_sha256",
        "python_executable_sha256",
        "python_version",
        "packages_sha256",
        "package_count",
    }
    if (
        not isinstance(dependency_identity, dict)
        or set(dependency_identity) != dependency_keys
        or dependency_identity.get("schema_version") != 1
        or not isinstance(dependency_identity.get("package_count"), int)
        or dependency_identity.get("package_count", 0) < 1
        or any(
            not isinstance(dependency_identity.get(key), str)
            or not dependency_identity[key]
            for key in dependency_keys - {"schema_version", "package_count"}
        )
    ):
        errors.append("receipt dependency identity is absent or malformed")
    target = receipt.get("lifecycle_target")
    if target not in {"TESTED", "ENFORCED"}:
        errors.append(f"invalid lifecycle target: {target!r}")
    expected_git = current.as_dict()
    if receipt.get("git") != expected_git:
        errors.append("receipt repository, branch, commit, or dirty state is stale")
    if current.dirty:
        errors.append("terminal lifecycle receipt requires a clean repository")
    if current.integration_ref and current.merge_base != current.integration_sha:
        errors.append("current branch is not based on the configured integration SHA")
    authority = receipt.get("authority_check")
    if not isinstance(authority, dict) or authority.get("passed") is not True:
        errors.append("strict authority check is absent or failed")

    tier = receipt.get("risk_tier")
    if tier not in {"T0", "T1", "T2", "T3"}:
        errors.append(f"invalid risk tier: {tier!r}")
        tier = "T3"
    commands = receipt.get("commands")
    if not isinstance(commands, list) or (tier != "T0" and not commands):
        errors.append("required command evidence is absent")
        commands = []
    has_positive_test = False
    for command in commands:
        if not isinstance(command, dict):
            errors.append("command evidence is malformed")
            continue
        name = command.get("name", "unnamed")
        allowed_command = {
            "name",
            "kind",
            "command_sha256",
            "cwd",
            "started_at",
            "completed_at",
            "duration_ms",
            "exit_code",
            "output_sha256",
            "scope",
            "test_count",
            "failed_count",
            "skipped",
            "flaky",
        }
        unknown = sorted(set(command) - allowed_command)
        if unknown:
            errors.append(
                f"command contains unknown keys ({name}): {', '.join(unknown)}"
            )
        if command.get("kind") != "command":
            errors.append(f"command has invalid kind: {name}")
        if command.get("exit_code") != 0:
            errors.append(f"command failed: {name}")
        if command.get("failed_count") != 0:
            errors.append(f"command reports failed tests: {name}")
        test_count = command.get("test_count", 0)
        skipped = command.get("skipped", 0)
        if not isinstance(test_count, int) or test_count < 0:
            errors.append(f"command has invalid test count: {name}")
        elif test_count > 0:
            has_positive_test = True
        if not isinstance(skipped, int) or skipped < 0:
            errors.append(f"command has invalid skip count: {name}")
        elif skipped != 0:
            errors.append(f"command contains skips: {name}")
        if command.get("flaky") is True:
            errors.append(f"command is flaky: {name}")
    if tier != "T0" and not has_positive_test:
        errors.append("receipt has no command with a positive test count")
    if target == "ENFORCED" and not any(
        isinstance(command, dict)
        and command.get("scope") == "full"
        and command.get("exit_code") == 0
        for command in commands
    ):
        errors.append("ENFORCED requires a passing full-suite or CI command")

    runtime_checks = receipt.get("runtime_checks")
    if tier in {"T2", "T3"} and (
        not isinstance(runtime_checks, list) or not runtime_checks
    ):
        errors.append("runtime checks are absent for T2/T3")
        runtime_checks = []
    for check in runtime_checks or []:
        if isinstance(check, dict):
            allowed_runtime = {
                "name",
                "kind",
                "command_sha256",
                "cwd",
                "started_at",
                "completed_at",
                "duration_ms",
                "exit_code",
                "output_sha256",
                "surface",
                "passed",
                "native_observation",
            }
            unknown = sorted(set(check) - allowed_runtime)
            if unknown:
                errors.append(
                    f"runtime check contains unknown keys ({check.get('name', 'unnamed')}): "
                    + ", ".join(unknown)
                )
            observation = check.get("native_observation")
            observation_keys = {
                "surface",
                "event",
                "commit",
                "adapter_path",
                "adapter_sha256",
                "bound_artifacts",
                "host_identity",
                "observed_at",
            }
            bound_artifacts = (
                observation.get("bound_artifacts")
                if isinstance(observation, dict)
                else None
            )
            host_identity = (
                observation.get("host_identity")
                if isinstance(observation, dict)
                else None
            )
            if (
                not isinstance(observation, dict)
                or set(observation) != observation_keys
                or any(
                    not isinstance(observation.get(key), str) or not observation[key]
                    for key in observation_keys - {"bound_artifacts", "host_identity"}
                )
                or not isinstance(bound_artifacts, dict)
                or not bound_artifacts
                or any(
                    not isinstance(path, str)
                    or not path
                    or not isinstance(digest, str)
                    or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                    for path, digest in (bound_artifacts or {}).items()
                )
                or not isinstance(host_identity, dict)
                or set(host_identity) != {"path", "sha256"}
                or not isinstance(host_identity.get("path"), str)
                or not host_identity["path"]
                or not isinstance(host_identity.get("sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", host_identity["sha256"]) is None
            ):
                errors.append(
                    f"runtime native observation is malformed: {check.get('name', 'unnamed')}"
                )
        if (
            not isinstance(check, dict)
            or check.get("kind") != "runtime"
            or check.get("passed") is not True
            or check.get("exit_code") != 0
        ):
            errors.append(
                f"runtime check failed: {check.get('name', 'unnamed') if isinstance(check, dict) else 'malformed'}"
            )

    rollback = receipt.get("rollback")
    if tier == "T3" and (
        not isinstance(rollback, dict)
        or rollback.get("kind") != "rollback"
        or rollback.get("passed") is not True
        or rollback.get("exit_code") != 0
    ):
        errors.append("rollback proof is absent or failed for T3")
    if isinstance(rollback, dict) and rollback:
        allowed_rollback = {
            "name",
            "kind",
            "command_sha256",
            "cwd",
            "started_at",
            "completed_at",
            "duration_ms",
            "exit_code",
            "output_sha256",
            "mode",
            "passed",
        }
        unknown = sorted(set(rollback) - allowed_rollback)
        if unknown:
            errors.append("rollback contains unknown keys: " + ", ".join(unknown))
    return errors

"""Machine-global, durable away-from-keyboard state."""

from __future__ import annotations

import errno
import json
import os
import secrets
import stat
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STATE_NAME = "afk.json"
LOCK_NAME = "afk.lock"
STATE_MODE = 0o600
MAX_REASON_CHARS = 200
MAX_STATE_BYTES = 16 * 1024


class AfkStateError(RuntimeError):
    def __init__(self, message: str, *, changed: bool = False):
        super().__init__(message)
        self.changed = changed


class AfkStateChangedUnconfirmed(AfkStateError):
    """A mutation happened, but the validated root no longer matches its path."""


def _root() -> Path:
    try:
        from hermes_constants import get_default_hermes_root

        return Path(get_default_hermes_root())
    except Exception as exc:
        raise AfkStateError(
            f"could not resolve the machine-global Hermes root: {exc}"
        ) from exc


def state_path() -> Path:
    return _root() / STATE_NAME


def _lock_path() -> Path:
    return _root() / LOCK_NAME


def _flags(*names: str) -> int:
    return sum(getattr(os, name, 0) for name in names)


def _allowed_owner(info: os.stat_result) -> bool:
    return info.st_uid in (0, os.geteuid()) if hasattr(os, "geteuid") else True


def _check_directory(fd: int, *, final: bool) -> None:
    info = os.fstat(fd)
    if not stat.S_ISDIR(info.st_mode):
        raise AfkStateError("AFK root path component is not a directory")
    if not _allowed_owner(info):
        raise AfkStateError("AFK root path component has an unsafe owner")
    mode = stat.S_IMODE(info.st_mode)
    if mode & 0o022 and not (mode & stat.S_ISVTX and not final):
        raise AfkStateError("AFK root path component is writable by group or other")


class _Root:
    def __init__(self, path: Path, fd: int | None):
        self.path, self.fd = path, fd
        self.mutated = False
        self.identity = (
            os.fstat(fd) if fd is not None else os.stat(path, follow_symlinks=False)
        )

    def close(self) -> None:
        fd = self.fd
        self.fd = None
        if fd is not None:
            os.close(fd)

    def revalidate(self, *, changed: bool = False) -> None:
        try:
            current = os.stat(self.path, follow_symlinks=False)
        except OSError as exc:
            raise AfkStateChangedUnconfirmed(
                f"AFK root identity changed: {exc}", changed=changed
            ) from exc
        if (current.st_dev, current.st_ino) != (
            self.identity.st_dev,
            self.identity.st_ino,
        ):
            raise AfkStateChangedUnconfirmed(
                "AFK root identity changed; durability could not be confirmed",
                changed=changed,
            )


def _path_value(root: _Root, name: str) -> str | Path:
    return name if root.fd is not None else root.path / name


def _path_open(root: _Root, name: str, flags: int, mode: int = 0) -> int:
    kwargs = {"dir_fd": root.fd} if root.fd is not None else {}
    args = (
        (_path_value(root, name), flags)
        if not mode
        else (_path_value(root, name), flags, mode)
    )
    return os.open(*args, **kwargs)


def _path_stat(root: _Root, name: str) -> os.stat_result:
    kwargs = (
        {"dir_fd": root.fd, "follow_symlinks": False}
        if root.fd is not None
        else {"follow_symlinks": False}
    )
    return os.stat(_path_value(root, name), **kwargs)


def _path_replace(root: _Root, source: str, destination: str) -> None:
    kwargs = (
        {"src_dir_fd": root.fd, "dst_dir_fd": root.fd} if root.fd is not None else {}
    )
    os.replace(_path_value(root, source), _path_value(root, destination), **kwargs)


def _path_unlink(root: _Root, name: str) -> None:
    kwargs = {"dir_fd": root.fd} if root.fd is not None else {}
    os.unlink(_path_value(root, name), **kwargs)


def _open_root() -> _Root:
    path = Path(os.path.abspath(os.fspath(_root().expanduser())))
    if os.name == "nt":
        try:
            for component in (path, *path.parents):
                try:
                    if stat.S_ISLNK(component.lstat().st_mode):
                        raise AfkStateError("refusing symlinked AFK root component")
                except FileNotFoundError:
                    pass
            path.mkdir(parents=True, exist_ok=True)
            return _Root(path, None)
        except OSError as exc:
            raise AfkStateError(f"could not set up AFK root: {exc}") from exc

    parts = path.parts
    try:
        fd = os.open(parts[0], _flags("O_RDONLY", "O_DIRECTORY", "O_CLOEXEC"))
    except OSError as exc:
        raise AfkStateError(f"could not set up AFK root: {exc}") from exc
    child = -1
    primary_error = None
    try:
        for index, component in enumerate(parts[1:], 1):
            child = -1
            try:
                if stat.S_ISLNK(
                    os.stat(component, dir_fd=fd, follow_symlinks=False).st_mode
                ):
                    raise AfkStateError("refusing symlinked AFK root path component")
            except FileNotFoundError:
                pass
            try:
                child = os.open(
                    component,
                    _flags("O_RDONLY", "O_DIRECTORY", "O_CLOEXEC", "O_NOFOLLOW"),
                    dir_fd=fd,
                )
            except FileNotFoundError:
                _check_directory(fd, final=False)
                os.mkdir(component, 0o700, dir_fd=fd)
                child = os.open(
                    component,
                    _flags("O_RDONLY", "O_DIRECTORY", "O_CLOEXEC", "O_NOFOLLOW"),
                    dir_fd=fd,
                )
            _check_directory(child, final=index == len(parts) - 1)
            old_fd = fd
            fd = child
            child = -1
            os.close(old_fd)
        _check_directory(fd, final=True)
        root = _Root(path, fd)
        fd = -1
        return root
    except OSError as exc:
        primary_error = exc
        raise AfkStateError(f"could not set up AFK root: {exc}") from exc
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        cleanup_error = None
        for owned_fd in (child, fd):
            if owned_fd >= 0:
                if owned_fd == child:
                    child = -1
                else:
                    fd = -1
                try:
                    os.close(owned_fd)
                except OSError as exc:
                    if cleanup_error is None:
                        cleanup_error = exc
        if primary_error is None and cleanup_error is not None:
            raise AfkStateError(
                f"could not close AFK root: {cleanup_error}"
            ) from cleanup_error


class _FileLock:
    def __init__(self, root: _Root | Path):
        self.root = root
        self.file = None

    def __enter__(self):
        compat = isinstance(self.root, Path)
        compat_root_fd = -1
        self._compat = compat
        fd = -1
        try:
            if compat:
                compat_root_fd = os.open(self.root.parent, os.O_RDONLY)
                self.root = _Root(self.root.parent, compat_root_fd)
                compat_root_fd = -1
            fd = _path_open(
                self.root,
                LOCK_NAME,
                _flags("O_RDWR", "O_CREAT", "O_CLOEXEC", "O_NOFOLLOW"),
                STATE_MODE,
            )
            info = os.fstat(fd)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
                or not _allowed_owner(info)
            ):
                raise AfkStateError("AFK lock is not a safe regular file")
            self.file = os.fdopen(fd, "a+b")
            fd = -1
            if os.name == "nt":
                import msvcrt

                self.file.seek(0, os.SEEK_END)
                if self.file.tell() == 0:
                    self.file.write(b"\0")
                    self.file.flush()
                self.file.seek(0)
                msvcrt.locking(self.file.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(self.file.fileno(), fcntl.LOCK_EX)
            return self
        except BaseException as primary:
            file = self.file
            self.file = None
            cleanup_error = None
            if file is not None:
                try:
                    file.close()
                except OSError as exc:
                    cleanup_error = exc
            if fd >= 0:
                owned_fd = fd
                fd = -1
                try:
                    os.close(owned_fd)
                except OSError as exc:
                    if cleanup_error is None:
                        cleanup_error = exc
            if compat_root_fd >= 0:
                owned_fd = compat_root_fd
                compat_root_fd = -1
                try:
                    os.close(owned_fd)
                except OSError as exc:
                    if cleanup_error is None:
                        cleanup_error = exc
            if compat and isinstance(self.root, _Root):
                try:
                    self.root.close()
                except OSError as exc:
                    if cleanup_error is None:
                        cleanup_error = exc
            if isinstance(primary, OSError):
                raise AfkStateError(
                    f"could not acquire AFK lock: {primary}"
                ) from primary
            raise

    def __exit__(self, exc_type, exc, tb):
        if self.file is None:
            return False
        cleanup_error = None
        file = self.file
        try:
            try:
                if os.name == "nt":
                    import msvcrt

                    file.seek(0)
                    msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(file.fileno(), fcntl.LOCK_UN)
            except OSError as error:
                cleanup_error = error
        finally:
            self.file = None
            try:
                file.close()
            except OSError as error:
                if cleanup_error is None:
                    cleanup_error = error
            finally:
                if getattr(self, "_compat", False):
                    try:
                        self.root.close()
                    except OSError as error:
                        if cleanup_error is None:
                            cleanup_error = error
        if exc_type is not None:
            return False
        if cleanup_error is not None:
            raise AfkStateError(
                f"could not release AFK lock: {cleanup_error}",
                changed=self.root.mutated,
            ) from cleanup_error
        return False


_mutex = threading.RLock()
_local = threading.local()


@contextmanager
def _transaction():
    with _mutex:
        if getattr(_local, "depth", 0):
            _local.depth += 1
            try:
                yield _local.root
            finally:
                _local.depth -= 1
            return
        root = _open_root()
        primary_error = None
        try:
            try:
                with _FileLock(root):
                    _local.root, _local.depth = root, 1
                    try:
                        yield root
                    finally:
                        _local.root, _local.depth = None, 0
            except BaseException as exc:
                primary_error = exc
                raise
        finally:
            try:
                root.close()
            except OSError as exc:
                if primary_error is None:
                    raise AfkStateError(
                        f"could not close AFK root: {exc}",
                        changed=root.mutated,
                    ) from exc


def _neutralize(value: Any, limit: int = MAX_REASON_CHARS) -> str | None:
    if not isinstance(value, str):
        return None
    value = "".join(" " if ord(ch) < 32 or ord(ch) == 127 else ch for ch in value)
    value = " ".join(value.replace("[", "(").replace("]", ")").split())
    if not value:
        return None
    return value[: limit - 1].rstrip() + "…" if len(value) > limit else value


def _unverifiable() -> dict:
    return {"unverifiable": True}


def _reject_state_symlink(root: _Root) -> None:
    try:
        info = _path_stat(root, STATE_NAME)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise AfkStateError(f"could not inspect AFK state: {exc}") from exc
    if stat.S_ISLNK(info.st_mode):
        raise AfkStateError("refusing symlinked AFK state")


def _read_state(root: _Root) -> dict | None:
    _reject_state_symlink(root)
    primary_error = None
    try:
        fd = _path_open(root, STATE_NAME, _flags("O_RDONLY", "O_CLOEXEC", "O_NOFOLLOW"))
    except FileNotFoundError:
        return None
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.ENOTDIR):
            raise AfkStateError("refusing symlinked AFK state") from exc
        return _unverifiable()
    try:
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or not _allowed_owner(info)
            or info.st_size > MAX_STATE_BYTES
            or (os.name != "nt" and stat.S_IMODE(info.st_mode) & 0o077)
        ):
            return _unverifiable()
        raw = os.read(fd, MAX_STATE_BYTES + 1)
        if len(raw) > MAX_STATE_BYTES:
            return _unverifiable()
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeError, TypeError, ValueError):
            return _unverifiable()
    except OSError:
        return _unverifiable()
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        owned_fd = fd
        fd = -1
        try:
            os.close(owned_fd)
        except OSError as exc:
            if primary_error is None:
                raise AfkStateError(
                    f"could not close AFK state: {exc}"
                ) from exc
    if not isinstance(data, dict) or not isinstance(data.get("engaged_at"), str):
        return _unverifiable()
    try:
        timestamp = datetime.fromisoformat(data["engaged_at"])
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            return _unverifiable()
    except ValueError:
        return _unverifiable()
    reason = data.get("reason")
    if reason is not None and (
        not isinstance(reason, str) or len(reason) > MAX_REASON_CHARS * 4
    ):
        return _unverifiable()
    return {
        "engaged_at": _neutralize(data["engaged_at"], 64),
        "reason": _neutralize(reason),
    }


def get_state() -> dict | None:
    with locked_state() as state:
        return state


@contextmanager
def locked_state():
    """Yield one validated AFK snapshot while holding the AFK file lock."""
    with _transaction() as root:
        state = _read_state(root)
        root.revalidate()
        yield state


def is_afk() -> bool:
    return get_state() is not None


def _atomic_replace_json(root: _Root, payload: dict) -> None:
    temporary, fd, primary_error = None, -1, None
    try:
        for _ in range(20):
            temporary = f".afk.{secrets.token_hex(8)}"
            try:
                fd = _path_open(
                    root,
                    temporary,
                    _flags("O_WRONLY", "O_CREAT", "O_EXCL", "O_CLOEXEC", "O_NOFOLLOW"),
                    STATE_MODE,
                )
                break
            except FileExistsError:
                temporary = None
        if fd < 0:
            raise AfkStateError("could not create a unique AFK temporary file")
        if os.name != "nt":
            os.fchmod(fd, STATE_MODE)
        raw = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        written = 0
        while written < len(raw):
            count = os.write(fd, raw[written:])
            if count <= 0:
                raise AfkStateError("AFK state write made no progress")
            written += count
        os.fsync(fd)
        owned_fd = fd
        fd = -1
        os.close(owned_fd)
        _path_replace(root, temporary, STATE_NAME)
        root.mutated = True
        temporary = None
        _sync_parent(root)
    except AfkStateError as exc:
        primary_error = exc
        raise
    except (OSError, TypeError) as exc:
        primary_error = exc
        raise AfkStateError(
            f"could not write AFK state: {exc}", changed=root.mutated
        ) from exc
    finally:
        cleanup_error = None
        if fd >= 0:
            owned_fd = fd
            fd = -1
            try:
                os.close(owned_fd)
            except OSError as exc:
                cleanup_error = exc
        if temporary is not None:
            try:
                _path_unlink(root, temporary)
            except FileNotFoundError:
                pass
            except OSError as exc:
                if cleanup_error is None:
                    cleanup_error = exc
        if primary_error is None and cleanup_error is not None:
            raise AfkStateError(
                f"could not clean up AFK temporary state: {cleanup_error}",
                changed=root.mutated,
            ) from cleanup_error


def _sync_parent(root: _Root) -> None:
    if os.name != "nt" and root.fd is not None:
        os.fsync(root.fd)


def _verify_owner_only(root: _Root) -> None:
    fd = _path_open(root, STATE_NAME, _flags("O_RDONLY", "O_CLOEXEC", "O_NOFOLLOW"))
    primary_error = None
    try:
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or not _allowed_owner(info)
        ):
            raise AfkStateError("AFK state is not a safe regular file")
        if os.name != "nt":
            os.fchmod(fd, STATE_MODE)
            if stat.S_IMODE(info.st_mode) & 0o077:
                raise AfkStateError("AFK state is not owner-only")
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        owned_fd = fd
        fd = -1
        try:
            os.close(owned_fd)
        except OSError as exc:
            if primary_error is None:
                raise AfkStateError(
                    f"could not close AFK state: {exc}"
                ) from exc


def engage(reason: str | None = None) -> dict:
    payload = {
        "engaged_at": datetime.now(timezone.utc).isoformat(),
        "reason": _neutralize(reason),
    }
    with _transaction() as root:
        _reject_state_symlink(root)
        _atomic_replace_json(root, payload)
        try:
            _verify_owner_only(root)
        except AfkStateError as exc:
            raise AfkStateChangedUnconfirmed(str(exc), changed=True) from exc
        except OSError as exc:
            raise AfkStateChangedUnconfirmed(
                f"could not verify AFK state: {exc}", changed=True
            ) from exc
        result = _read_state(root)
        root.revalidate(changed=True)
        if result is None or result.get("unverifiable"):
            raise AfkStateChangedUnconfirmed(
                "AFK state could not be verified", changed=True
            )
    # Wake approval waiters only after releasing the AFK file lock.  The
    # approval queue takes its own lock and must never invert AFK->queue order.
    try:
        from tools.approval import cancel_gateway_approvals_for_afk
        cancel_gateway_approvals_for_afk()
    except Exception:
        pass
    return result


def clear() -> bool:
    with _transaction() as root:
        try:
            if stat.S_ISLNK(_path_stat(root, STATE_NAME).st_mode):
                raise AfkStateError("refusing symlinked AFK state")
            _path_unlink(root, STATE_NAME)
            root.mutated = True
        except FileNotFoundError:
            root.revalidate()
            return False
        except OSError as exc:
            raise AfkStateError(f"could not clear AFK state: {exc}") from exc
        try:
            _sync_parent(root)
            root.revalidate(changed=True)
        except AfkStateChangedUnconfirmed:
            raise
        except OSError as exc:
            raise AfkStateChangedUnconfirmed(
                f"could not sync cleared AFK state: {exc}", changed=True
            ) from exc
        return True


def _reply(*lines: str) -> str:
    return "\n".join((
        *lines,
        "AFK never widens approvals or authorizes consequential work.",
    ))


def handle_command(args: str = "") -> str:
    parts = (args or "").strip().split(None, 1)
    verb = parts[0].lower() if parts else ""
    rest = parts[1].strip() if len(parts) > 1 else ""
    try:
        if not args.strip():
            state = get_state()
            if state and state.get("unverifiable"):
                return _reply(
                    "AFK state is present but unreadable/unverifiable; remaining fail-closed."
                )
            if state is not None:
                return _reply("Already AFK. Use `/afk off` when you're back.")
        if not args.strip() or verb == "on":
            state = engage(rest or None)
            suffix = f" ({state['reason']})" if state.get("reason") else ""
            return _reply(f"AFK recorded since {state['engaged_at']}{suffix}.")
        if verb == "off" and not rest:
            return _reply("AFK cleared." if clear() else "You weren't marked AFK.")
        if verb == "status" and not rest:
            state = get_state()
            if not state:
                return _reply("Not AFK.")
            if state.get("unverifiable"):
                return _reply(
                    "AFK state is present but unreadable/unverifiable; remaining fail-closed."
                )
            suffix = f" ({state['reason']})" if state.get("reason") else ""
            return _reply(f"AFK since {state['engaged_at']}{suffix}.")
    except AfkStateError as exc:
        if exc.changed:
            return _reply(
                "AFK state changed, but durability could not be confirmed; check /afk status."
            )
        return _reply("Couldn't safely change the machine-global AFK state.")
    return _reply("Usage: `/afk [on [reason] | off | status]`")

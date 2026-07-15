"""Dedicated no-secret local browser controller.

The gateway talks to this process over one peer-authenticated AF_UNIX
connection per canonical conversation lease.  Only this process invokes the
release-local Node/agent-browser/Chrome binaries.  It never inherits operator
credentials and never exposes CDP or a generic subprocess surface.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import ipaddress
import json
import os
import re
import shutil
import signal
import socket
import stat
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol
from urllib.parse import urlsplit

from gateway.browser_controller_protocol import (
    BrowserCommand,
    BrowserControllerProtocolError,
    MAX_ARTIFACT_BYTES,
    MAX_REQUEST_BYTES,
    MAX_REQUESTS_PER_SESSION,
    MAX_RESPONSE_BYTES,
    MAX_RESULT_BYTES,
    PeerCredentials,
    decode_json,
    error_response,
    ok_response,
    receive_frame,
    send_frame,
    validate_command_request,
    validate_session_close,
    validate_session_open,
    unix_peer_credentials,
)


CONFIG_SCHEMA = "hermes-browser-controller-service.v1"
SYSTEMD_READY_STATUS = "hermes-browser-controller-ready-v1"
_TRUSTED_AGENT_BROWSER_CONFIG_UID = 0
_EMPTY_AGENT_BROWSER_CONFIG = b"{}\n"
_MAX_STDERR_BYTES = 64 * 1024
_MAX_PROCESS_CMDLINE_BYTES = 256 * 1024
_QUOTA_POLL_SECONDS = 0.05
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SESSION_HASH = re.compile(r"[0-9a-f]{64}")
_METADATA_HOSTS = frozenset(
    {
        "metadata.google.internal",
        "metadata.goog",
        "instance-data",
        "instance-data.ec2.internal",
    }
)
_FORBIDDEN_NETWORKS = tuple(
    ipaddress.ip_network(value)
    for value in (
        "0.0.0.0/8",
        "10.0.0.0/8",
        "100.64.0.0/10",
        "127.0.0.0/8",
        "169.254.0.0/16",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "224.0.0.0/4",
        "240.0.0.0/4",
        "::/128",
        "::1/128",
        "fc00::/7",
        "fe80::/10",
        "ff00::/8",
    )
)


class BrowserControllerError(RuntimeError):
    """Stable controller/runtime boundary failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def notify_systemd_ready(*, _notify_socket: str | None = None) -> bool:
    """Notify systemd only after the controller has bound its AF_UNIX socket.

    The notification is a mechanical process-readiness signal.  It carries no
    configuration, credential, request text, task metadata, or semantic state.
    Local/non-systemd launches intentionally remain supported and return
    ``False`` when no inherited notify socket exists.
    """

    raw_address = (
        os.environ.get("NOTIFY_SOCKET", "")
        if _notify_socket is None
        else _notify_socket
    )
    if not raw_address:
        return False
    if (
        not isinstance(raw_address, str)
        or len(raw_address.encode("utf-8")) > 100
        or any(character in raw_address for character in ("\x00", "\n", "\r"))
        or raw_address[0] not in {"/", "@"}
    ):
        raise BrowserControllerError("browser_controller_notify_socket_invalid")
    address = "\x00" + raw_address[1:] if raw_address.startswith("@") else raw_address
    payload = (
        f"READY=1\nSTATUS={SYSTEMD_READY_STATUS}\n"
    ).encode("ascii", errors="strict")
    channel = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    channel.set_inheritable(False)
    try:
        sent = channel.sendto(payload, address)
    except OSError as exc:
        raise BrowserControllerError("browser_controller_notify_failed") from exc
    finally:
        channel.close()
    if sent != len(payload):
        raise BrowserControllerError("browser_controller_notify_incomplete")
    return True


def _exact_mapping(
    value: Any,
    *,
    required: frozenset[str],
    code: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(k, str) for k in value):
        raise BrowserControllerError(code)
    result = dict(value)
    if set(result) != required:
        raise BrowserControllerError(code)
    return result


def _absolute_path(value: Any, code: str) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise BrowserControllerError(code)
    path = Path(value)
    if not path.is_absolute() or str(path) != value:
        raise BrowserControllerError(code)
    return path


def _bounded_int(value: Any, *, low: int, high: int, code: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise BrowserControllerError(code)
    return value


@dataclass(frozen=True)
class BrowserControllerConfig:
    socket_path: Path
    socket_runtime_root: Path
    socket_gid: int
    allowed_client_uid: int
    session_root: Path
    release_root: Path
    node_path: Path
    node_sha256: str
    wrapper_path: Path
    wrapper_sha256: str
    native_path: Path
    native_sha256: str
    chrome_path: Path
    chrome_sha256: str
    agent_browser_config_path: Path
    agent_browser_config_sha256: str
    command_timeout_seconds: int
    idle_timeout_seconds: int
    max_connections: int
    max_sessions: int
    session_quota_bytes: int
    session_quota_entries: int

    @classmethod
    def from_mapping(cls, value: Any) -> "BrowserControllerConfig":
        keys = frozenset(
            {
                "schema",
                "socket_path",
                "socket_runtime_root",
                "socket_gid",
                "allowed_client_uid",
                "session_root",
                "release_root",
                "node_path",
                "node_sha256",
                "wrapper_path",
                "wrapper_sha256",
                "native_path",
                "native_sha256",
                "chrome_path",
                "chrome_sha256",
                "agent_browser_config_path",
                "agent_browser_config_sha256",
                "command_timeout_seconds",
                "idle_timeout_seconds",
                "max_connections",
                "max_sessions",
                "session_quota_bytes",
                "session_quota_entries",
            }
        )
        raw = _exact_mapping(value, required=keys, code="browser_controller_config_invalid")
        if raw["schema"] != CONFIG_SCHEMA:
            raise BrowserControllerError("browser_controller_config_schema_invalid")
        socket_runtime_root = _absolute_path(
            raw["socket_runtime_root"],
            "browser_controller_socket_runtime_root_invalid",
        )
        session_root = _absolute_path(
            raw["session_root"], "browser_controller_session_root_invalid"
        )
        socket_path = _absolute_path(
            raw["socket_path"], "browser_controller_socket_path_invalid"
        )
        release_root = _absolute_path(
            raw["release_root"], "browser_controller_release_root_invalid"
        )
        if (
            socket_path.parent != socket_runtime_root
            or socket_path.name != "controller.sock"
            or session_root == socket_runtime_root
        ):
            raise BrowserControllerError("browser_controller_socket_path_invalid")
        executable_paths: dict[str, Path] = {}
        for key in ("node", "wrapper", "native", "chrome"):
            path = _absolute_path(
                raw[f"{key}_path"], f"browser_controller_{key}_path_invalid"
            )
            try:
                path.relative_to(release_root)
            except ValueError as exc:
                raise BrowserControllerError(
                    f"browser_controller_{key}_path_escaped"
                ) from exc
            digest = raw[f"{key}_sha256"]
            if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
                raise BrowserControllerError(
                    f"browser_controller_{key}_digest_invalid"
                )
            executable_paths[key] = path
        if executable_paths["native"].parent != executable_paths["wrapper"].parent:
            # The attested Node wrapper resolves its native launcher relative
            # to its own directory.  Pinning an unrelated native file would
            # otherwise attest one path while the wrapper executes another.
            raise BrowserControllerError("browser_controller_native_path_invalid")
        agent_browser_config_path = _absolute_path(
            raw["agent_browser_config_path"],
            "browser_controller_agent_browser_config_path_invalid",
        )
        try:
            agent_browser_config_path.relative_to(release_root)
        except ValueError as exc:
            raise BrowserControllerError(
                "browser_controller_agent_browser_config_path_escaped"
            ) from exc
        agent_browser_config_sha256 = raw["agent_browser_config_sha256"]
        if (
            not isinstance(agent_browser_config_sha256, str)
            or _SHA256.fullmatch(agent_browser_config_sha256) is None
        ):
            raise BrowserControllerError(
                "browser_controller_agent_browser_config_digest_invalid"
            )
        max_connections = _bounded_int(
            raw["max_connections"],
            low=1,
            high=128,
            code="browser_controller_max_connections_invalid",
        )
        max_sessions = _bounded_int(
            raw["max_sessions"],
            low=1,
            high=64,
            code="browser_controller_max_sessions_invalid",
        )
        if max_sessions > max_connections:
            raise BrowserControllerError("browser_controller_capacity_invalid")
        return cls(
            socket_path=socket_path,
            socket_runtime_root=socket_runtime_root,
            socket_gid=_bounded_int(
                raw["socket_gid"],
                low=0,
                high=2**31 - 1,
                code="browser_controller_socket_gid_invalid",
            ),
            allowed_client_uid=_bounded_int(
                raw["allowed_client_uid"],
                low=0,
                high=2**31 - 1,
                code="browser_controller_client_uid_invalid",
            ),
            session_root=session_root,
            release_root=release_root,
            node_path=executable_paths["node"],
            node_sha256=raw["node_sha256"],
            wrapper_path=executable_paths["wrapper"],
            wrapper_sha256=raw["wrapper_sha256"],
            native_path=executable_paths["native"],
            native_sha256=raw["native_sha256"],
            chrome_path=executable_paths["chrome"],
            chrome_sha256=raw["chrome_sha256"],
            agent_browser_config_path=agent_browser_config_path,
            agent_browser_config_sha256=agent_browser_config_sha256,
            command_timeout_seconds=_bounded_int(
                raw["command_timeout_seconds"],
                low=5,
                high=300,
                code="browser_controller_command_timeout_invalid",
            ),
            idle_timeout_seconds=_bounded_int(
                raw["idle_timeout_seconds"],
                low=30,
                high=3600,
                code="browser_controller_idle_timeout_invalid",
            ),
            max_connections=max_connections,
            max_sessions=max_sessions,
            session_quota_bytes=_bounded_int(
                raw["session_quota_bytes"],
                low=MAX_ARTIFACT_BYTES + MAX_RESULT_BYTES + _MAX_STDERR_BYTES,
                high=8 * 1024 * 1024 * 1024,
                code="browser_controller_session_quota_bytes_invalid",
            ),
            session_quota_entries=_bounded_int(
                raw["session_quota_entries"],
                low=64,
                high=100_000,
                code="browser_controller_session_quota_entries_invalid",
            ),
        )


@dataclass(frozen=True)
class FileIdentity:
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int


def _identity(state: os.stat_result) -> FileIdentity:
    return FileIdentity(
        state.st_dev,
        state.st_ino,
        state.st_size,
        state.st_mtime_ns,
        state.st_ctime_ns,
    )


def _attest_regular_file(
    path: Path,
    expected_sha256: str,
    *,
    maximum: int,
    executable: bool,
) -> FileIdentity:
    try:
        before = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise BrowserControllerError("browser_controller_executable_unavailable") from exc
    if (
        resolved != path
        or stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or not 0 < before.st_size <= maximum
        or stat.S_IMODE(before.st_mode) & 0o022
        or (executable and not os.access(path, os.X_OK))
    ):
        raise BrowserControllerError("browser_controller_executable_invalid")
    hasher = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                hasher.update(chunk)
        after = path.lstat()
    except OSError as exc:
        raise BrowserControllerError("browser_controller_executable_unavailable") from exc
    if _identity(before) != _identity(after):
        raise BrowserControllerError("browser_controller_executable_raced")
    if hasher.hexdigest() != expected_sha256:
        raise BrowserControllerError("browser_controller_executable_digest_mismatch")
    return _identity(after)


def _attest_agent_browser_config(
    path: Path,
    expected_sha256: str,
) -> FileIdentity:
    try:
        before = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise BrowserControllerError(
            "browser_controller_agent_browser_config_unavailable"
        ) from exc
    if (
        resolved != path
        or stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_uid != _TRUSTED_AGENT_BROWSER_CONFIG_UID
        or stat.S_IMODE(before.st_mode) & 0o022
        or before.st_size != len(_EMPTY_AGENT_BROWSER_CONFIG)
    ):
        raise BrowserControllerError(
            "browser_controller_agent_browser_config_invalid"
        )
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
        try:
            opened = os.fstat(fd)
            payload = os.read(fd, len(_EMPTY_AGENT_BROWSER_CONFIG) + 1)
            after = os.fstat(fd)
        finally:
            os.close(fd)
        path_after = path.lstat()
    except OSError as exc:
        raise BrowserControllerError(
            "browser_controller_agent_browser_config_unavailable"
        ) from exc
    if (
        _identity(before) != _identity(opened)
        or _identity(opened) != _identity(after)
        or _identity(after) != _identity(path_after)
        or payload != _EMPTY_AGENT_BROWSER_CONFIG
        or hashlib.sha256(payload).hexdigest() != expected_sha256
    ):
        raise BrowserControllerError(
            "browser_controller_agent_browser_config_invalid"
        )
    return _identity(after)


class ExecutableAttestor:
    def __init__(self, config: BrowserControllerConfig) -> None:
        self.config = config
        self._identities = self._attest_all()

    def _attest_all(self) -> dict[Path, FileIdentity]:
        values = (
            (self.config.node_path, self.config.node_sha256, 256 * 1024 * 1024, True),
            (self.config.wrapper_path, self.config.wrapper_sha256, 2 * 1024 * 1024, False),
            (self.config.native_path, self.config.native_sha256, 128 * 1024 * 1024, True),
            (self.config.chrome_path, self.config.chrome_sha256, 512 * 1024 * 1024, True),
        )
        identities = {
            path: _attest_regular_file(
                path, digest, maximum=maximum, executable=executable
            )
            for path, digest, maximum, executable in values
        }
        identities[self.config.agent_browser_config_path] = (
            _attest_agent_browser_config(
                self.config.agent_browser_config_path,
                self.config.agent_browser_config_sha256,
            )
        )
        return identities

    def validate_unchanged(self) -> None:
        for path, expected in self._identities.items():
            try:
                state = path.lstat()
            except OSError as exc:
                raise BrowserControllerError(
                    "browser_controller_executable_drifted"
                ) from exc
            if (
                stat.S_ISLNK(state.st_mode)
                or not stat.S_ISREG(state.st_mode)
                or _identity(state) != expected
            ):
                raise BrowserControllerError("browser_controller_executable_drifted")


class PublicURLPolicy:
    """Application preflight for non-public browser navigation targets.

    This is not the production network authority: DNS can change between this
    check and Chrome's connection.  The controller's systemd sandbox MUST
    provide the real public-only kernel egress boundary and a constrained DNS
    stub.  A release is not ready without that external service-unit contract.
    """

    def __init__(
        self,
        *,
        resolver: Callable[..., Any] = socket.getaddrinfo,
        website_checker: Callable[[str], Any] | None = None,
    ) -> None:
        self.resolver = resolver
        if website_checker is None:
            from tools.website_policy import check_website_access

            website_checker = check_website_access
        self.website_checker = website_checker

    @staticmethod
    def _forbidden_ip(value: str) -> bool:
        try:
            address = ipaddress.ip_address(value)
        except ValueError as exc:
            raise BrowserControllerError("browser_controller_url_probe_invalid") from exc
        return (
            any(address in network for network in _FORBIDDEN_NETWORKS)
            or address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_multicast
            or address.is_reserved
            or address.is_unspecified
        )

    def validate(self, url: Any, *, allow_about_blank: bool = False) -> str:
        if url == "about:blank" and allow_about_blank:
            return url
        if not isinstance(url, str) or not url or len(url.encode("utf-8")) > 8192:
            raise BrowserControllerError("browser_controller_url_invalid")
        try:
            parsed = urlsplit(url)
            port = parsed.port
        except (TypeError, ValueError) as exc:
            raise BrowserControllerError("browser_controller_url_invalid") from exc
        hostname = (parsed.hostname or "").rstrip(".").lower()
        if (
            parsed.scheme not in {"http", "https"}
            or not hostname
            or parsed.username is not None
            or parsed.password is not None
            or port is not None and not 1 <= port <= 65535
            or hostname in _METADATA_HOSTS
            or hostname == "localhost"
            or hostname.endswith((".localhost", ".local", ".internal", ".lan"))
        ):
            raise BrowserControllerError("browser_controller_url_not_public")
        try:
            records = self.resolver(
                hostname,
                port or (443 if parsed.scheme == "https" else 80),
                socket.AF_UNSPEC,
                socket.SOCK_STREAM,
            )
        except Exception as exc:
            raise BrowserControllerError("browser_controller_url_probe_failed") from exc
        if not records:
            raise BrowserControllerError("browser_controller_url_probe_failed")
        addresses: list[str] = []
        try:
            for record in records:
                sockaddr = record[4]
                addresses.append(str(sockaddr[0]))
        except (IndexError, TypeError, ValueError) as exc:
            raise BrowserControllerError("browser_controller_url_probe_invalid") from exc
        if not addresses or any(self._forbidden_ip(address) for address in addresses):
            raise BrowserControllerError("browser_controller_url_not_public")
        try:
            blocked = self.website_checker(url)
        except Exception as exc:
            raise BrowserControllerError("browser_controller_app_policy_failed") from exc
        if blocked:
            raise BrowserControllerError("browser_controller_app_policy_blocked")
        return url


@dataclass
class BrowserSession:
    session_id_sha256: str
    agent_session_name: str
    root: Path
    last_activity: float
    quota_error: str | None = None


@dataclass(frozen=True)
class SessionUsage:
    bytes: int
    entries: int


class BrowserExecutor(Protocol):
    def execute(self, session: BrowserSession, command: BrowserCommand) -> dict[str, Any]: ...

    def close(self, session: BrowserSession) -> None: ...

    def cleanup_stale_sessions(self) -> None: ...

    def enforce_session_quota(self, session: BrowserSession) -> SessionUsage: ...


class AgentBrowserExecutor:
    """Exact absolute release-local agent-browser mechanical executor."""

    def __init__(
        self,
        config: BrowserControllerConfig,
        attestor: ExecutableAttestor,
        *,
        popen: Callable[..., subprocess.Popen] = subprocess.Popen,
    ) -> None:
        self.config = config
        self.attestor = attestor
        self._popen = popen
        self._process_lock = threading.Lock()
        self._active_processes: dict[str, subprocess.Popen] = {}

    def _private_dir(self, session: BrowserSession, path: Path) -> Path:
        try:
            path.relative_to(session.root)
            path.mkdir(mode=0o700, parents=True, exist_ok=True)
            state = path.lstat()
            resolved = path.resolve(strict=True)
        except (OSError, ValueError) as exc:
            raise BrowserControllerError(
                "browser_controller_session_directory_invalid"
            ) from exc
        if (
            resolved != path
            or stat.S_ISLNK(state.st_mode)
            or not stat.S_ISDIR(state.st_mode)
            or state.st_uid != os.geteuid()  # windows-footgun: ok — POSIX AF_UNIX controller boundary
            or stat.S_IMODE(state.st_mode) != 0o700
        ):
            raise BrowserControllerError(
                "browser_controller_session_directory_invalid"
            )
        return path

    def _environment(self, session: BrowserSession) -> dict[str, str]:
        home = session.root / "home"
        socket_dir = session.root / "socket"
        tmp = session.root / "tmp"
        for path in (home, socket_dir, tmp):
            self._private_dir(session, path)
        return {
            "AGENT_BROWSER_CONFIG": str(self.config.agent_browser_config_path),
            "AGENT_BROWSER_EXECUTABLE_PATH": str(self.config.chrome_path),
            "AGENT_BROWSER_IDLE_TIMEOUT_MS": str(
                self.config.idle_timeout_seconds * 1000
            ),
            "AGENT_BROWSER_SOCKET_DIR": str(socket_dir),
            "HOME": str(home),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/bin:/bin",
            "TMPDIR": str(tmp),
            "XDG_CACHE_HOME": str(home / ".cache"),
            "XDG_CONFIG_HOME": str(home / ".config"),
        }

    @staticmethod
    def _kill_process(proc: subprocess.Popen) -> None:
        try:
            if proc.poll() is not None:
                proc.wait(timeout=1)
                return
        except Exception:
            pass
        try:
            pgid = os.getpgid(proc.pid)
            if (
                pgid != proc.pid
                or pgid <= 1
                or pgid == os.getpgrp()
                or proc.poll() is not None
                or os.getpgid(proc.pid) != pgid
            ):
                raise ProcessLookupError
            os.killpg(pgid, signal.SIGKILL)  # windows-footgun: ok — POSIX AF_UNIX controller boundary
        except (ProcessLookupError, PermissionError, OSError):
            try:
                if proc.poll() is None:
                    proc.kill()
            except (ProcessLookupError, OSError):
                pass
        try:
            proc.wait(timeout=5)
        except Exception:
            pass

    @staticmethod
    def _read_bounded_regular(
        path: Path,
        *,
        maximum: int,
        minimum: int = 0,
        code: str,
    ) -> bytes:
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            path_before = path.lstat()
            fd = os.open(path, flags)
        except OSError as exc:
            raise BrowserControllerError(code) from exc
        try:
            before = os.fstat(fd)
            if (
                stat.S_ISLNK(path_before.st_mode)
                or not stat.S_ISREG(path_before.st_mode)
                or not stat.S_ISREG(before.st_mode)
                or path_before.st_nlink != 1
                or before.st_nlink != 1
                or path_before.st_uid != os.geteuid()  # windows-footgun: ok — POSIX AF_UNIX controller boundary
                or before.st_uid != os.geteuid()  # windows-footgun: ok — POSIX AF_UNIX controller boundary
                or _identity(path_before) != _identity(before)
                or not minimum <= before.st_size <= maximum
            ):
                raise BrowserControllerError(code)
            chunks: list[bytes] = []
            remaining = maximum + 1
            while remaining:
                chunk = os.read(fd, min(1024 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            payload = b"".join(chunks)
            after = os.fstat(fd)
            path_after = path.lstat()
        except BrowserControllerError:
            raise
        except OSError as exc:
            raise BrowserControllerError(code) from exc
        finally:
            os.close(fd)
        if (
            len(payload) > maximum
            or not minimum <= len(payload)
            or _identity(before) != _identity(after)
            or _identity(after) != _identity(path_after)
        ):
            raise BrowserControllerError(code)
        return payload

    def _scan_session_tree(self, session: BrowserSession) -> SessionUsage:
        directory_flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            directory_flags |= os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            directory_flags |= os.O_NOFOLLOW
        if hasattr(os, "O_CLOEXEC"):
            directory_flags |= os.O_CLOEXEC
        root_fd: int | None = None
        try:
            root_state = session.root.lstat()
            root_resolved = session.root.resolve(strict=True)
            root_fd = os.open(session.root, directory_flags)
            root_opened = os.fstat(root_fd)
        except OSError as exc:
            if root_fd is not None:
                os.close(root_fd)
            raise BrowserControllerError("browser_controller_session_tree_raced") from exc
        assert root_fd is not None
        if (
            root_resolved != session.root
            or stat.S_ISLNK(root_state.st_mode)
            or not stat.S_ISDIR(root_state.st_mode)
            or not stat.S_ISDIR(root_opened.st_mode)
            or root_state.st_uid != os.geteuid()  # windows-footgun: ok — POSIX AF_UNIX controller boundary
            or root_opened.st_uid != os.geteuid()  # windows-footgun: ok — POSIX AF_UNIX controller boundary
            or (root_state.st_dev, root_state.st_ino)
            != (root_opened.st_dev, root_opened.st_ino)
            or stat.S_IMODE(root_state.st_mode) != 0o700
        ):
            os.close(root_fd)
            raise BrowserControllerError("browser_controller_session_tree_invalid")
        total_bytes = 0
        total_entries = 0
        stack: list[tuple[Path, int]] = [(session.root, root_fd)]
        try:
            while stack:
                directory, directory_fd = stack.pop()
                try:
                    with os.scandir(directory_fd) as iterator:
                        for entry in iterator:
                            path = directory / entry.name
                            try:
                                state = entry.stat(follow_symlinks=False)
                            except FileNotFoundError:
                                continue
                            except OSError as exc:
                                raise BrowserControllerError(
                                    "browser_controller_session_tree_raced"
                                ) from exc
                            total_entries += 1
                            if total_entries > self.config.session_quota_entries:
                                raise BrowserControllerError(
                                    "browser_controller_session_entry_quota_exceeded"
                                )
                            if state.st_uid != os.geteuid() or stat.S_ISLNK(  # windows-footgun: ok — POSIX AF_UNIX controller boundary
                                state.st_mode
                            ):
                                raise BrowserControllerError(
                                    "browser_controller_session_entry_invalid"
                                )
                            if stat.S_ISDIR(state.st_mode):
                                child_fd: int | None = None
                                try:
                                    child_fd = os.open(
                                        entry.name,
                                        directory_flags,
                                        dir_fd=directory_fd,
                                    )
                                    opened = os.fstat(child_fd)
                                except FileNotFoundError:
                                    if child_fd is not None:
                                        os.close(child_fd)
                                    continue
                                except OSError as exc:
                                    if child_fd is not None:
                                        os.close(child_fd)
                                    raise BrowserControllerError(
                                        "browser_controller_session_tree_raced"
                                    ) from exc
                                if (
                                    not stat.S_ISDIR(opened.st_mode)
                                    or opened.st_uid != os.geteuid()  # windows-footgun: ok — POSIX AF_UNIX controller boundary
                                    or (state.st_dev, state.st_ino)
                                    != (opened.st_dev, opened.st_ino)
                                ):
                                    os.close(child_fd)
                                    raise BrowserControllerError(
                                        "browser_controller_session_tree_raced"
                                    )
                                stack.append((path, child_fd))
                                continue
                            if stat.S_ISREG(state.st_mode):
                                if state.st_nlink != 1:
                                    raise BrowserControllerError(
                                        "browser_controller_session_entry_invalid"
                                    )
                                total_bytes += state.st_size
                            elif not (
                                stat.S_ISSOCK(state.st_mode)
                                and path.parent == session.root / "socket"
                                and path.name
                                == f"{session.agent_session_name}.sock"
                            ):
                                raise BrowserControllerError(
                                    "browser_controller_session_entry_invalid"
                                )
                            if total_bytes > self.config.session_quota_bytes:
                                raise BrowserControllerError(
                                    "browser_controller_session_byte_quota_exceeded"
                                )
                finally:
                    os.close(directory_fd)
        finally:
            for _path, pending_fd in stack:
                try:
                    os.close(pending_fd)
                except OSError:
                    pass
        return SessionUsage(bytes=total_bytes, entries=total_entries)

    def enforce_session_quota(self, session: BrowserSession) -> SessionUsage:
        try:
            return self._scan_session_tree(session)
        except BrowserControllerError:
            with self._process_lock:
                proc = self._active_processes.get(session.agent_session_name)
            if proc is not None:
                self._kill_process(proc)
            self._kill_daemon(session)
            raise

    @staticmethod
    def _output_size(path: Path, maximum: int) -> None:
        try:
            state = path.lstat()
        except OSError as exc:
            raise BrowserControllerError(
                "browser_controller_command_output_unavailable"
            ) from exc
        if (
            stat.S_ISLNK(state.st_mode)
            or not stat.S_ISREG(state.st_mode)
            or state.st_nlink != 1
            or state.st_uid != os.geteuid()  # windows-footgun: ok — POSIX AF_UNIX controller boundary
            or state.st_size > maximum
        ):
            raise BrowserControllerError(
                "browser_controller_command_output_oversized"
            )

    def _monitor_process(
        self,
        proc: subprocess.Popen,
        session: BrowserSession,
        stdout_path: Path,
        stderr_path: Path,
    ) -> None:
        deadline = time.monotonic() + self.config.command_timeout_seconds
        while True:
            try:
                self._output_size(stdout_path, MAX_RESULT_BYTES)
                self._output_size(stderr_path, _MAX_STDERR_BYTES)
                self.enforce_session_quota(session)
            except BrowserControllerError:
                self._kill_process(proc)
                raise
            if proc.poll() is not None:
                break
            if time.monotonic() >= deadline:
                self._kill_process(proc)
                raise BrowserControllerError("browser_controller_command_timeout")
            time.sleep(_QUOTA_POLL_SECONDS)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired as exc:
            self._kill_process(proc)
            raise BrowserControllerError("browser_controller_command_timeout") from exc
        self._output_size(stdout_path, MAX_RESULT_BYTES)
        self._output_size(stderr_path, _MAX_STDERR_BYTES)
        self.enforce_session_quota(session)

    def _run(self, session: BrowserSession, command: BrowserCommand) -> dict[str, Any]:
        self.attestor.validate_unchanged()
        self.enforce_session_quota(session)
        artifact_path: Path | None = None
        token = uuid.uuid4().hex
        work_dir = session.root / "work" / token
        self._private_dir(session, work_dir)
        argv = [
            str(self.config.node_path),
            str(self.config.wrapper_path),
            "--config",
            str(self.config.agent_browser_config_path),
            "--session",
            session.agent_session_name,
            "--json",
            command.argv_command,
            *command.argv,
        ]
        if command.screenshot:
            artifact_dir = session.root / "artifacts"
            self._private_dir(session, artifact_dir)
            artifact_path = artifact_dir / f"{uuid.uuid4().hex}.png"
            argv.append(str(artifact_path))
        io_dir = session.root / "io"
        self._private_dir(session, io_dir)
        stdout_path = io_dir / f"{token}.stdout"
        stderr_path = io_dir / f"{token}.stderr"
        stdout_fd: int | None = None
        stderr_fd: int | None = None
        proc: subprocess.Popen | None = None
        try:
            stdout_fd = os.open(
                stdout_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            stderr_fd = os.open(
                stderr_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            try:
                proc = self._popen(
                    argv,
                    cwd=str(work_dir),
                    env=self._environment(session),
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_fd,
                    stderr=stderr_fd,
                    close_fds=True,
                    start_new_session=True,
                )
            except Exception as exc:
                raise BrowserControllerError("browser_controller_spawn_failed") from exc
            finally:
                os.close(stdout_fd)
                stdout_fd = None
                os.close(stderr_fd)
                stderr_fd = None
            with self._process_lock:
                self._active_processes[session.agent_session_name] = proc
            self._monitor_process(proc, session, stdout_path, stderr_path)
            stdout = self._read_bounded_regular(
                stdout_path,
                maximum=MAX_RESULT_BYTES,
                code="browser_controller_command_output_invalid",
            )
            self._read_bounded_regular(
                stderr_path,
                maximum=_MAX_STDERR_BYTES,
                code="browser_controller_command_output_invalid",
            )
            if proc.returncode != 0:
                return {"success": False, "error": "browser_controller_command_failed"}
            parsed = decode_json(stdout, maximum=MAX_RESULT_BYTES)
            if type(parsed.get("success")) is not bool:
                raise BrowserControllerError("browser_controller_command_result_invalid")
            if command.screenshot and parsed.get("success") is True:
                artifact = self._read_artifact(artifact_path)
                data = parsed.get("data")
                if not isinstance(data, Mapping):
                    data = {}
                parsed["data"] = {
                    key: value for key, value in dict(data).items() if key != "path"
                }
                parsed["artifact"] = artifact
            return parsed
        finally:
            with self._process_lock:
                if self._active_processes.get(session.agent_session_name) is proc:
                    self._active_processes.pop(session.agent_session_name, None)
            for fd in (stdout_fd, stderr_fd):
                if fd is not None:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
            for path in (stdout_path, stderr_path, artifact_path):
                if path is not None:
                    try:
                        path.unlink()
                    except OSError:
                        pass
            shutil.rmtree(work_dir, ignore_errors=True)

    def _read_artifact(self, path: Path | None) -> dict[str, Any]:
        if path is None:
            raise BrowserControllerError("browser_controller_artifact_missing")
        payload = self._read_bounded_regular(
            path,
            maximum=MAX_ARTIFACT_BYTES,
            minimum=8,
            code="browser_controller_artifact_invalid",
        )
        if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
            raise BrowserControllerError("browser_controller_artifact_invalid")
        return {
            "encoding": "base64",
            "media_type": "image/png",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size": len(payload),
            "data": base64.b64encode(payload).decode("ascii"),
        }

    def execute(self, session: BrowserSession, command: BrowserCommand) -> dict[str, Any]:
        return self._run(session, command)

    def _pid_bound_to_session(self, pid: int, session: BrowserSession) -> bool:
        if pid <= 1:
            return False
        try:
            with Path(f"/proc/{pid}/cmdline").open("rb") as handle:
                payload = handle.read(_MAX_PROCESS_CMDLINE_BYTES + 1)
        except OSError:
            return False
        if len(payload) > _MAX_PROCESS_CMDLINE_BYTES:
            return False
        decoded = payload.replace(b"\x00", b" ").decode("utf-8", errors="replace")
        return "agent-browser" in decoded and session.agent_session_name in decoded

    def _process_group_bound_to_session(
        self,
        pgid: int,
        session: BrowserSession,
    ) -> bool:
        """Re-authenticate a process group before an irreversible group signal."""

        try:
            iterator = os.scandir("/proc")
        except OSError:
            return False
        session_markers = (session.agent_session_name, str(session.root))
        with iterator:
            for entry in iterator:
                if not entry.name.isdigit():
                    continue
                pid = int(entry.name)
                try:
                    if os.getpgid(pid) != pgid:
                        continue
                    with Path("/proc", entry.name, "cmdline").open("rb") as handle:
                        payload = handle.read(_MAX_PROCESS_CMDLINE_BYTES + 1)
                except (OSError, ValueError):
                    continue
                if len(payload) > _MAX_PROCESS_CMDLINE_BYTES:
                    continue
                decoded = payload.replace(b"\x00", b" ").decode(
                    "utf-8", errors="replace"
                )
                if (
                    any(marker in decoded for marker in session_markers)
                    and ("agent-browser" in decoded or "chrome" in decoded.lower())
                ):
                    return True
        return False

    def _kill_daemon(self, session: BrowserSession) -> None:
        pid_path = session.root / "socket" / f"{session.agent_session_name}.pid"
        try:
            payload = self._read_bounded_regular(
                pid_path,
                maximum=32,
                minimum=1,
                code="browser_controller_daemon_pid_invalid",
            )
            pid = int(payload.decode("ascii", errors="strict").strip())
        except (BrowserControllerError, UnicodeError, ValueError):
            return
        if not self._pid_bound_to_session(pid, session):
            return
        try:
            pgid = os.getpgid(pid)
            if pgid <= 1 or pgid == os.getpgrp():
                return
            # Re-check after resolving the group so PID reuse cannot redirect a
            # group signal away from the session-bound daemon we authenticated.
            if not self._pid_bound_to_session(pid, session) or os.getpgid(pid) != pgid:
                return
            os.killpg(pgid, signal.SIGTERM)  # windows-footgun: ok — POSIX AF_UNIX controller boundary
        except (ProcessLookupError, PermissionError, OSError):
            return
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            try:
                os.killpg(pgid, 0)  # windows-footgun: ok — POSIX AF_UNIX controller boundary
            except ProcessLookupError:
                return
            time.sleep(0.05)
        if not self._process_group_bound_to_session(pgid, session):
            # The original leader may have exited and its numeric PGID may have
            # been reused. Leak rather than signal an unrelated process group.
            return
        try:
            os.killpg(pgid, signal.SIGKILL)  # windows-footgun: ok — POSIX AF_UNIX controller boundary
        except (ProcessLookupError, PermissionError, OSError):
            pass

    def close(self, session: BrowserSession) -> None:
        try:
            self._run(session, BrowserCommand("close", "close", ()))
        except Exception:
            pass
        self._kill_daemon(session)
        self._remove_session_root(session.root)

    @staticmethod
    def _remove_session_root(root: Path) -> None:
        try:
            state = root.lstat()
        except FileNotFoundError:
            return
        except OSError:
            return
        if stat.S_ISLNK(state.st_mode):
            try:
                root.unlink()
            except OSError:
                pass
            return
        try:
            resolved = root.resolve(strict=True)
        except OSError:
            return
        if (
            resolved == root
            and stat.S_ISDIR(state.st_mode)
            and state.st_uid == os.geteuid()  # windows-footgun: ok — POSIX AF_UNIX controller boundary
        ):
            shutil.rmtree(root, ignore_errors=True)

    def cleanup_stale_sessions(self) -> None:
        try:
            entries = list(self.config.session_root.iterdir())
        except OSError:
            return
        for entry in entries:
            try:
                state = entry.lstat()
            except OSError as exc:
                raise BrowserControllerError(
                    "browser_controller_stale_session_entry_invalid"
                ) from exc
            token = entry.name.removeprefix("session-")
            if (
                entry.name != f"session-{token}"
                or re.fullmatch(r"[0-9a-f]{32}", token) is None
                or stat.S_ISLNK(state.st_mode)
                or not stat.S_ISDIR(state.st_mode)
                or state.st_uid != os.geteuid()  # windows-footgun: ok — POSIX AF_UNIX controller boundary
            ):
                raise BrowserControllerError(
                    "browser_controller_stale_session_entry_invalid"
                )
            session = BrowserSession(
                session_id_sha256="",
                agent_session_name=f"hbc_{token}",
                root=entry,
                last_activity=0.0,
            )
            # Reissue the exact close command and verify/terminate the
            # session-bound daemon group before deleting its private profile.
            self.close(session)


class BrowserControllerServer:
    def __init__(
        self,
        config: BrowserControllerConfig,
        *,
        executor: BrowserExecutor | None = None,
        peer_getter: Callable[[socket.socket], PeerCredentials] = unix_peer_credentials,
        url_policy: PublicURLPolicy | None = None,
    ) -> None:
        self.config = config
        self.attestor = ExecutableAttestor(config)
        self.executor = executor or AgentBrowserExecutor(config, self.attestor)
        self.peer_getter = peer_getter
        self.url_policy = url_policy or PublicURLPolicy()
        self._listener: socket.socket | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._connections: set[socket.socket] = set()
        self._sessions: dict[int, BrowserSession] = {}
        self._threads: set[threading.Thread] = set()

    def _validate_roots(self) -> None:
        try:
            socket_state = self.config.socket_runtime_root.lstat()
            socket_resolved = self.config.socket_runtime_root.resolve(strict=True)
            session_state = self.config.session_root.lstat()
            session_resolved = self.config.session_root.resolve(strict=True)
        except OSError as exc:
            raise BrowserControllerError("browser_controller_runtime_root_unavailable") from exc
        if (
            socket_resolved != self.config.socket_runtime_root
            or stat.S_ISLNK(socket_state.st_mode)
            or not stat.S_ISDIR(socket_state.st_mode)
            or socket_state.st_uid != os.geteuid()  # windows-footgun: ok — POSIX AF_UNIX controller boundary
            or socket_state.st_gid != self.config.socket_gid
            or stat.S_IMODE(socket_state.st_mode) != 0o750
        ):
            raise BrowserControllerError(
                "browser_controller_socket_runtime_root_invalid"
            )
        if (
            session_resolved != self.config.session_root
            or stat.S_ISLNK(session_state.st_mode)
            or not stat.S_ISDIR(session_state.st_mode)
            or session_state.st_uid != os.geteuid()  # windows-footgun: ok — POSIX AF_UNIX controller boundary
            or stat.S_IMODE(session_state.st_mode) != 0o700
        ):
            raise BrowserControllerError("browser_controller_session_root_invalid")

    def bind(self) -> None:
        self._validate_roots()
        self.executor.cleanup_stale_sessions()
        try:
            state = self.config.socket_path.lstat()
        except FileNotFoundError:
            state = None
        except OSError as exc:
            raise BrowserControllerError("browser_controller_socket_unavailable") from exc
        if state is not None:
            if not stat.S_ISSOCK(state.st_mode) or stat.S_ISLNK(state.st_mode):
                raise BrowserControllerError("browser_controller_socket_invalid")
            self.config.socket_path.unlink()
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            listener.bind(str(self.config.socket_path))
            os.chown(self.config.socket_path, -1, self.config.socket_gid)
            os.chmod(self.config.socket_path, 0o660)
            listener.listen(self.config.max_connections)
            listener.settimeout(0.25)
        except Exception:
            listener.close()
            try:
                self.config.socket_path.unlink()
            except OSError:
                pass
            raise
        self._listener = listener

    def serve_forever(self) -> None:
        if self._listener is None:
            self.bind()
        listener = self._listener
        assert listener is not None
        try:
            while not self._stop.is_set():
                try:
                    conn, _address = listener.accept()
                except socket.timeout:
                    continue
                except OSError:
                    if self._stop.is_set():
                        break
                    raise
                thread = threading.Thread(
                    target=self._serve_connection,
                    args=(conn,),
                    name="browser-controller-client",
                    daemon=True,
                )
                with self._lock:
                    if len(self._connections) >= self.config.max_connections:
                        rejected = True
                    else:
                        rejected = False
                        self._connections.add(conn)
                        self._threads.add(thread)
                        # Starting while holding the registry lock prevents
                        # stop() from observing an unstarted registered worker.
                        thread.start()
                if rejected:
                    try:
                        conn.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                    conn.close()
        finally:
            self.stop()

    def _new_session(self, identity: str) -> BrowserSession:
        if _SESSION_HASH.fullmatch(identity) is None:
            raise BrowserControllerError("browser_controller_session_identity_invalid")
        token = uuid.uuid4().hex
        root = self.config.session_root / f"session-{token}"
        root.mkdir(mode=0o700)
        return BrowserSession(
            session_id_sha256=identity,
            agent_session_name=f"hbc_{token}",
            root=root,
            last_activity=time.monotonic(),
        )

    @staticmethod
    def _result_url(result: Mapping[str, Any]) -> str | None:
        data = result.get("data")
        if isinstance(data, Mapping):
            value = data.get("url") if "url" in data else data.get("result")
            if isinstance(value, str):
                return value.strip().strip('"').strip("'")
        return None

    def _probe_current_url(self, session: BrowserSession) -> str:
        result = self.executor.execute(
            session,
            BrowserCommand("current_url", "eval", ("window.location.href",)),
        )
        if result.get("success") is not True:
            raise BrowserControllerError("browser_controller_page_url_probe_failed")
        url = self._result_url(result)
        if not url:
            raise BrowserControllerError("browser_controller_page_url_probe_failed")
        return url

    def _blank_page(self, session: BrowserSession) -> None:
        try:
            self.executor.execute(
                session, BrowserCommand("internal.blank", "open", ("about:blank",))
            )
        except Exception:
            pass

    def _execute(self, session: BrowserSession, command: BrowserCommand) -> dict[str, Any]:
        if command.name == "open":
            self.url_policy.validate(command.argv[0])
        elif command.name != "current_url":
            # A failed/aborted prior navigation can leave the browser on a
            # target whose content was never released.  Validate the live page
            # before allowing any subsequent input or content extraction.
            current_url = self._probe_current_url(session)
            try:
                self.url_policy.validate(current_url, allow_about_blank=True)
            except BrowserControllerError:
                self._blank_page(session)
                raise
        result = self.executor.execute(session, command)
        if result.get("success") is not True:
            return result
        if command.name == "current_url":
            page_url = self._result_url(result)
            if not page_url:
                raise BrowserControllerError("browser_controller_page_url_probe_failed")
        else:
            # Never trust a command's reported target as the post-action page
            # identity. Probe the live page after open/click/fill/back/press
            # and before releasing any snapshot/log/screenshot content.
            page_url = self._probe_current_url(session)
        try:
            self.url_policy.validate(page_url, allow_about_blank=True)
        except BrowserControllerError:
            self._blank_page(session)
            raise
        session.last_activity = time.monotonic()
        return result

    def _send_error(self, conn: socket.socket, request_id: str, code: str) -> None:
        try:
            send_frame(
                conn,
                error_response(request_id, code),
                maximum=MAX_RESPONSE_BYTES,
            )
        except Exception:
            pass

    def _monitor_session_quota(
        self,
        session: BrowserSession,
        conn: socket.socket,
        stopped: threading.Event,
    ) -> None:
        enforce = getattr(self.executor, "enforce_session_quota", None)
        if not callable(enforce):
            return
        while not stopped.wait(_QUOTA_POLL_SECONDS):
            try:
                enforce(session)
            except BrowserControllerError as exc:
                session.quota_error = exc.code
                try:
                    conn.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                return
            except Exception:
                session.quota_error = "browser_controller_session_quota_failed"
                try:
                    conn.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                return

    def _serve_connection(self, conn: socket.socket) -> None:
        session: BrowserSession | None = None
        quota_stop = threading.Event()
        quota_thread: threading.Thread | None = None
        key = id(conn)
        try:
            peer = self.peer_getter(conn)
            if peer.uid != self.config.allowed_client_uid:
                raise BrowserControllerError("browser_controller_peer_forbidden")
            # Poll the process stop event while retaining the configured idle
            # deadline.  Some Unix kernels do not wake a blocking recv in one
            # thread when another thread closes the same Python socket object.
            conn.settimeout(0.25)
            opening = receive_frame(
                conn,
                maximum=MAX_REQUEST_BYTES,
                deadline=time.monotonic() + self.config.idle_timeout_seconds,
                stop_requested=self._stop.is_set,
            )
            opening = validate_session_open(opening)
            seen_request_ids = {opening["request_id"]}
            session = self._new_session(opening["session_id_sha256"])
            with self._lock:
                if len(self._sessions) >= self.config.max_sessions:
                    raise BrowserControllerError("browser_controller_session_limit")
                self._sessions[key] = session
            quota_thread = threading.Thread(
                target=self._monitor_session_quota,
                args=(session, conn, quota_stop),
                name="browser-controller-quota",
                daemon=True,
            )
            quota_thread.start()
            send_frame(
                conn,
                ok_response(opening["request_id"], {"session": "ready"}),
                maximum=MAX_RESPONSE_BYTES,
            )
            idle_deadline = time.monotonic() + self.config.idle_timeout_seconds
            while not self._stop.is_set():
                try:
                    request = receive_frame(
                        conn,
                        maximum=MAX_REQUEST_BYTES,
                        deadline=idle_deadline,
                        stop_requested=self._stop.is_set,
                    )
                except socket.timeout:
                    break
                except EOFError:
                    break
                idle_deadline = time.monotonic() + self.config.idle_timeout_seconds
                request_id = str(request.get("request_id") or "")
                try:
                    if session.quota_error is not None:
                        raise BrowserControllerError(session.quota_error)
                    if request.get("op") == "session.close":
                        validated = validate_session_close(request)
                        request_id = validated["request_id"]
                        if request_id in seen_request_ids:
                            raise BrowserControllerProtocolError(
                                "browser_controller_request_replayed"
                            )
                        if len(seen_request_ids) >= MAX_REQUESTS_PER_SESSION:
                            raise BrowserControllerProtocolError(
                                "browser_controller_request_limit"
                            )
                        seen_request_ids.add(request_id)
                        send_frame(
                            conn,
                            ok_response(request_id, {"closed": True}),
                            maximum=MAX_RESPONSE_BYTES,
                        )
                        break
                    raw, command = validate_command_request(request)
                    request_id = raw["request_id"]
                    if request_id in seen_request_ids:
                        raise BrowserControllerProtocolError(
                            "browser_controller_request_replayed"
                        )
                    if len(seen_request_ids) >= MAX_REQUESTS_PER_SESSION:
                        raise BrowserControllerProtocolError(
                            "browser_controller_request_limit"
                        )
                    seen_request_ids.add(request_id)
                    result = self._execute(session, command)
                    response = ok_response(request_id, result)
                    send_frame(conn, response, maximum=MAX_RESPONSE_BYTES)
                except (BrowserControllerProtocolError, BrowserControllerError) as exc:
                    self._send_error(conn, request_id, getattr(exc, "code", str(exc)))
        except (
            BrowserControllerProtocolError,
            BrowserControllerError,
            EOFError,
            socket.timeout,
            OSError,
        ):
            pass
        finally:
            quota_stop.set()
            if quota_thread is not None:
                quota_thread.join(timeout=2)
            if session is not None:
                self.executor.close(session)
            with self._lock:
                self._sessions.pop(key, None)
                self._connections.discard(conn)
                self._threads.discard(threading.current_thread())
            try:
                conn.close()
            except OSError:
                pass

    def stop(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        listener, self._listener = self._listener, None
        if listener is not None:
            try:
                listener.close()
            except OSError:
                pass
        with self._lock:
            connections = list(self._connections)
            threads = list(self._threads)
        for conn in connections:
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                conn.close()
            except OSError:
                pass
        for thread in threads:
            if thread is not threading.current_thread():
                thread.join(timeout=5)
        try:
            state = self.config.socket_path.lstat()
            if stat.S_ISSOCK(state.st_mode):
                self.config.socket_path.unlink()
        except OSError:
            pass


def _load_config(path: Path) -> BrowserControllerConfig:
    try:
        state = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise BrowserControllerError("browser_controller_config_unavailable") from exc
    if (
        resolved != path
        or stat.S_ISLNK(state.st_mode)
        or not stat.S_ISREG(state.st_mode)
        or state.st_nlink != 1
        or not 0 < state.st_size <= 64 * 1024
        or stat.S_IMODE(state.st_mode) & 0o022
        or state.st_uid not in {0, os.geteuid()}  # windows-footgun: ok — POSIX AF_UNIX controller boundary
    ):
        raise BrowserControllerError("browser_controller_config_invalid")
    raw = path.read_bytes()
    after = path.lstat()
    if _identity(state) != _identity(after):
        raise BrowserControllerError("browser_controller_config_raced")
    return BrowserControllerConfig.from_mapping(decode_json(raw, maximum=64 * 1024))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hermes browser controller")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    server = BrowserControllerServer(_load_config(args.config))
    try:
        server.bind()
        notify_systemd_ready()
        server.serve_forever()
    except KeyboardInterrupt:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AgentBrowserExecutor",
    "BrowserControllerConfig",
    "BrowserControllerError",
    "BrowserControllerServer",
    "BrowserSession",
    "CONFIG_SCHEMA",
    "ExecutableAttestor",
    "PeerCredentials",
    "PublicURLPolicy",
    "SYSTEMD_READY_STATUS",
    "notify_systemd_ready",
    "unix_peer_credentials",
]

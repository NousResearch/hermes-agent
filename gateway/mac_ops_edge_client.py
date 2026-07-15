"""Credential-free client for the privileged Mac operations Unix edge."""

from __future__ import annotations

import hashlib
import os
import re
import socket
import stat
import struct
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from gateway.mac_ops_edge_protocol import (
    MAX_REQUEST_BYTES,
    MAX_RESPONSE_BYTES,
    MacOpsEdgeOperation,
    MacOpsEdgeReceipt,
    MacOpsEdgeRequest,
    MacOpsPing,
    MacOpsReadOnlyClass,
    MacOpsReadOnlySubmit,
    MacOpsTaskRead,
    canonical_json_bytes,
    decode_json_object,
    validate_response,
)
from gateway.mac_ops_edge_service import DEFAULT_SOCKET_PATH, MacOpsEdgePeer


DEFAULT_SERVICE_UNIT = "muncho-mac-ops-edge.service"
_FRAME_HEADER = struct.Struct("!I")
_PEER_CREDENTIALS = struct.Struct("3i")
_UNIT_RE = re.compile(r"^[A-Za-z0-9_.@:-]+\.service$")
_BOUNDARY_LOCK = threading.Lock()
_FROZEN_CONFIG: "MacOpsEdgeClientConfig | None" = None
_FROZEN_ERROR: str | None = None
_BOUNDARY_FROZEN = False
_CLIENTS: dict["MacOpsEdgeClientConfig", "MacOpsEdgeClient"] = {}


class MacOpsEdgeClientError(RuntimeError):
    def __init__(self, code: str, *, dispatch_uncertain: bool = False) -> None:
        self.code = code
        self.dispatch_uncertain = dispatch_uncertain
        super().__init__(code)


@dataclass(frozen=True)
class MacOpsEdgeClientConfig:
    socket_path: Path
    service_unit: str
    service_uid: int
    socket_gid: int
    service_identity_sha256: str
    connect_timeout_seconds: float = 2.0
    request_timeout_seconds: float = 30.0

    @classmethod
    def from_mapping(cls, value: Any) -> "MacOpsEdgeClientConfig":
        if not isinstance(value, Mapping):
            raise ValueError("mac_ops_edge_config_invalid")
        expected = {
            "enabled",
            "socket_path",
            "service_unit",
            "service_uid",
            "socket_gid",
            "service_identity_sha256",
            "connect_timeout_seconds",
            "request_timeout_seconds",
        }
        if set(value) != expected or value.get("enabled") is not True:
            raise ValueError("mac_ops_edge_config_invalid")
        path = Path(value.get("socket_path", ""))
        if (
            path != DEFAULT_SOCKET_PATH
            or not path.is_absolute()
            or path != Path(os.path.normpath(path))
        ):
            raise ValueError("mac_ops_edge_socket_not_pinned")
        unit = value.get("service_unit")
        if unit != DEFAULT_SERVICE_UNIT or _UNIT_RE.fullmatch(str(unit)) is None:
            raise ValueError("mac_ops_edge_unit_not_pinned")
        service_uid = value.get("service_uid")
        socket_gid = value.get("socket_gid")
        if (
            type(service_uid) is not int
            or type(socket_gid) is not int
            or service_uid < 1
            or socket_gid < 1
        ):
            raise ValueError("mac_ops_edge_identity_invalid")
        identity = value.get("service_identity_sha256")
        if (
            not isinstance(identity, str)
            or len(identity) != 64
            or any(char not in "0123456789abcdef" for char in identity)
        ):
            raise ValueError("mac_ops_edge_identity_invalid")
        connect = value.get("connect_timeout_seconds")
        request = value.get("request_timeout_seconds")
        if (
            isinstance(connect, bool)
            or not isinstance(connect, (int, float))
            or not 0.1 <= float(connect) <= 10
            or isinstance(request, bool)
            or not isinstance(request, (int, float))
            or not 1 <= float(request) <= 30
        ):
            raise ValueError("mac_ops_edge_timeout_invalid")
        return cls(
            socket_path=path,
            service_unit=str(unit),
            service_uid=service_uid,
            socket_gid=socket_gid,
            service_identity_sha256=identity,
            connect_timeout_seconds=float(connect),
            request_timeout_seconds=float(request),
        )


class MainPidProvider(Protocol):
    def main_pid(self, service_unit: str) -> int: ...


class SystemctlMainPidProvider:
    def main_pid(self, service_unit: str) -> int:
        if _UNIT_RE.fullmatch(service_unit) is None:
            raise ValueError("mac_ops_edge_unit_invalid")
        try:
            result = subprocess.run(
                [
                    "systemctl",
                    "show",
                    service_unit,
                    "--property=MainPID",
                    "--value",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
                env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
            )
            pid = int(result.stdout.strip())
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            raise MacOpsEdgeClientError("mac_ops_edge_identity_unavailable") from exc
        if pid < 1:
            raise MacOpsEdgeClientError("mac_ops_edge_not_running")
        return pid


PeerGetter = Callable[[socket.socket], MacOpsEdgePeer]


def linux_server_peer(sock: socket.socket) -> MacOpsEdgePeer:
    option = getattr(socket, "SO_PEERCRED", None)
    if option is None:
        raise OSError("peer_credentials_unavailable")
    raw = sock.getsockopt(socket.SOL_SOCKET, option, _PEER_CREDENTIALS.size)
    if len(raw) != _PEER_CREDENTIALS.size:
        raise OSError("peer_credentials_invalid")
    return MacOpsEdgePeer(*_PEER_CREDENTIALS.unpack(raw))


def _receive_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise OSError("connection_closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class MacOpsEdgeClient:
    def __init__(
        self,
        config: MacOpsEdgeClientConfig,
        *,
        main_pid_provider: MainPidProvider | None = None,
        peer_getter: PeerGetter = linux_server_peer,
    ) -> None:
        self.config = config
        self.main_pid_provider = main_pid_provider or SystemctlMainPidProvider()
        self.peer_getter = peer_getter
        self._sequence = 0
        self._lock = threading.Lock()

    def _next_sequence(self) -> int:
        with self._lock:
            value = self._sequence
            self._sequence += 1
            return value

    def _validate_socket(self) -> None:
        try:
            state = os.lstat(self.config.socket_path)
        except OSError as exc:
            raise MacOpsEdgeClientError("mac_ops_edge_unavailable") from exc
        if (
            not stat.S_ISSOCK(state.st_mode)
            or state.st_uid != self.config.service_uid
            or state.st_gid != self.config.socket_gid
            or stat.S_IMODE(state.st_mode) != 0o660
        ):
            raise MacOpsEdgeClientError("mac_ops_edge_socket_identity_invalid")

    def _call(self, request: MacOpsEdgeRequest) -> dict[str, Any]:
        self._validate_socket()
        expected_pid = self.main_pid_provider.main_pid(self.config.service_unit)
        body = canonical_json_bytes(request.to_mapping())
        if len(body) > MAX_REQUEST_BYTES:
            raise ValueError("mac_ops_edge_request_too_large")
        reached_edge = False
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.settimeout(self.config.connect_timeout_seconds)
                sock.connect(str(self.config.socket_path))
                peer = self.peer_getter(sock)
                if (
                    peer.pid != expected_pid
                    or peer.uid != self.config.service_uid
                    or peer.gid < 0
                ):
                    raise MacOpsEdgeClientError("mac_ops_edge_server_unauthorized")
                sock.settimeout(self.config.request_timeout_seconds)
                sock.sendall(_FRAME_HEADER.pack(len(body)) + body)
                reached_edge = True
                header = _receive_exact(sock, _FRAME_HEADER.size)
                (size,) = _FRAME_HEADER.unpack(header)
                if size == 0 or size > MAX_RESPONSE_BYTES:
                    raise ValueError("mac_ops_edge_response_frame_invalid")
                value = decode_json_object(
                    _receive_exact(sock, size), maximum=MAX_RESPONSE_BYTES
                )
        except MacOpsEdgeClientError:
            raise
        except (OSError, ValueError) as exc:
            raise MacOpsEdgeClientError(
                "mac_ops_edge_transport_failed",
                dispatch_uncertain=reached_edge
                and request.operation is MacOpsEdgeOperation.READONLY_SUBMIT,
            ) from exc
        if set(value) == {"protocol", "error"}:
            error = value.get("error")
            if not isinstance(error, str) or not error:
                error = "mac_ops_edge_rejected"
            raise MacOpsEdgeClientError(error)
        response = validate_response(value, request=request)
        receipt = MacOpsEdgeReceipt.from_mapping(
            response["receipt"], request=request
        )
        if (
            receipt.value["service_identity_sha256"]
            != self.config.service_identity_sha256
        ):
            raise MacOpsEdgeClientError("mac_ops_edge_service_identity_mismatch")
        return response

    def submit_readonly(
        self,
        *,
        title: str,
        task_class: MacOpsReadOnlyClass | str,
        contract: str,
        idempotency_key: str,
    ) -> dict[str, Any]:
        payload = MacOpsReadOnlySubmit.from_mapping(
            {
                "title": title,
                "task_class": MacOpsReadOnlyClass(task_class).value,
                "contract": contract,
                "contract_sha256": hashlib.sha256(
                    contract.encode("utf-8")
                ).hexdigest(),
            }
        )
        request = MacOpsEdgeRequest.from_mapping(
            {
                "protocol": "muncho-mac-ops-edge.v1",
                "request_id": str(uuid.uuid4()),
                "sequence": self._next_sequence(),
                "deadline_unix_ms": int(time.time() * 1000)
                + int(self.config.request_timeout_seconds * 1000),
                "operation": MacOpsEdgeOperation.READONLY_SUBMIT.value,
                "idempotency_key": idempotency_key,
                "payload": payload.to_mapping(),
            }
        )
        return self._call(request)

    def ping(self, *, nonce: str) -> dict[str, Any]:
        """Exercise the authenticated socket protocol without external I/O."""

        payload = MacOpsPing.from_mapping({"nonce": nonce})
        request = MacOpsEdgeRequest.from_mapping(
            {
                "protocol": "muncho-mac-ops-edge.v1",
                "request_id": str(uuid.uuid4()),
                "sequence": self._next_sequence(),
                "deadline_unix_ms": int(time.time() * 1000)
                + int(self.config.connect_timeout_seconds * 1000)
                + 5_000,
                "operation": MacOpsEdgeOperation.PING.value,
                "idempotency_key": f"prerequisite:ping:{nonce[:24]}",
                "payload": payload.to_mapping(),
            }
        )
        return self._call(request)

    def read_task(
        self,
        *,
        issue_iid: int,
        idempotency_key: str,
    ) -> dict[str, Any]:
        payload = MacOpsTaskRead.from_mapping({"issue_iid": issue_iid})
        request = MacOpsEdgeRequest.from_mapping(
            {
                "protocol": "muncho-mac-ops-edge.v1",
                "request_id": str(uuid.uuid4()),
                "sequence": self._next_sequence(),
                "deadline_unix_ms": int(time.time() * 1000)
                + int(self.config.request_timeout_seconds * 1000),
                "operation": MacOpsEdgeOperation.TASK_READ.value,
                "idempotency_key": idempotency_key,
                "payload": payload.to_mapping(),
            }
        )
        return self._call(request)


def _root_config(config: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if config is not None:
        return config
    try:
        from hermes_cli.config import load_config

        loaded = load_config() or {}
    except Exception:
        loaded = {}
    return loaded if isinstance(loaded, Mapping) else {}


def load_mac_ops_edge_client_config(
    config: Mapping[str, Any] | None = None,
) -> MacOpsEdgeClientConfig | None:
    """Load the non-secret, restart-only client boundary from config.yaml."""

    root = _root_config(config)
    raw = root.get("mac_ops_edge")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("mac_ops_edge_config_invalid")
    if raw.get("enabled") is not True:
        if set(raw) != {"enabled"} or raw.get("enabled") is not False:
            raise ValueError("disabled_mac_ops_edge_config_not_exact")
        return None
    return MacOpsEdgeClientConfig.from_mapping(raw)


def frozen_mac_ops_edge_client_config(
    config: Mapping[str, Any] | None = None,
) -> MacOpsEdgeClientConfig | None:
    """Freeze static capability presence for prompt-cache stability."""

    global _BOUNDARY_FROZEN, _FROZEN_CONFIG, _FROZEN_ERROR
    with _BOUNDARY_LOCK:
        if not _BOUNDARY_FROZEN:
            try:
                _FROZEN_CONFIG = load_mac_ops_edge_client_config(config)
            except Exception as exc:
                _FROZEN_ERROR = type(exc).__name__
                _FROZEN_CONFIG = None
            _BOUNDARY_FROZEN = True
        if _FROZEN_ERROR is not None:
            raise RuntimeError("mac_ops_edge_config_invalid")
        return _FROZEN_CONFIG


def mac_ops_edge_configured(config: Mapping[str, Any] | None = None) -> bool:
    try:
        return frozen_mac_ops_edge_client_config(config) is not None
    except Exception:
        return False


def privileged_mac_ops_edge_client() -> MacOpsEdgeClient:
    config = frozen_mac_ops_edge_client_config()
    if config is None:
        raise RuntimeError("mac_ops_edge_not_configured")
    with _BOUNDARY_LOCK:
        client = _CLIENTS.get(config)
        if client is None:
            client = MacOpsEdgeClient(config)
            _CLIENTS[config] = client
        return client


def _reset_mac_ops_edge_boundary_for_tests() -> None:
    global _BOUNDARY_FROZEN, _FROZEN_CONFIG, _FROZEN_ERROR
    with _BOUNDARY_LOCK:
        _BOUNDARY_FROZEN = False
        _FROZEN_CONFIG = None
        _FROZEN_ERROR = None
        _CLIENTS.clear()


__all__ = [
    "DEFAULT_SERVICE_UNIT",
    "MacOpsEdgeClient",
    "MacOpsEdgeClientConfig",
    "MacOpsEdgeClientError",
    "MainPidProvider",
    "SystemctlMainPidProvider",
    "_reset_mac_ops_edge_boundary_for_tests",
    "frozen_mac_ops_edge_client_config",
    "linux_server_peer",
    "load_mac_ops_edge_client_config",
    "mac_ops_edge_configured",
    "privileged_mac_ops_edge_client",
]

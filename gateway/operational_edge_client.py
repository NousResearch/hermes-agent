"""Credential-free client for exact operational edge operations."""

from __future__ import annotations

import hashlib
import json
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
from typing import Any, Mapping, Protocol

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from gateway.operational_edge_catalog import operation_catalog
from gateway.operational_edge_protocol import (
    MAX_REQUEST_BYTES,
    MAX_RESPONSE_BYTES,
    PREDISPATCH_MUTATION_BLOCKERS,
    RECEIPT_SCHEMA,
    OperationalIntent,
    OperationalProtocolError,
    OperationalRequest,
    SignedEnvelope,
    canonical_json_bytes,
    decode_json_object,
    sha256_json,
    verify_envelope,
)
from gateway.operational_edge_service import OperationalEdgePeer


_FRAME = struct.Struct("!I")
_PEER = struct.Struct("3i")
_UNIT = re.compile(r"^muncho-operational-edge-[a-z][a-z0-9_-]{0,31}\.service$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
DEFAULT_CLIENT_CONFIG_PATH = Path("/etc/muncho/operational-edge-client.json")
SYSTEMCTL = Path("/usr/bin/systemctl")
CLIENT_CONFIG_SCHEMA = "muncho-operational-edge-client-config.v3"


class OperationalEdgeClientError(RuntimeError):
    def __init__(self, code: str, *, dispatch_uncertain: bool = False) -> None:
        self.code = code
        self.dispatch_uncertain = dispatch_uncertain
        super().__init__(code)


def _predispatch_capability_truth_matches(
    blocker_code: Any,
    *,
    capability_present: bool,
) -> bool:
    """Preserve the exact difference between absent and invalid authority."""

    if blocker_code == "mutation_capability_required":
        return not capability_present
    if blocker_code == "mutation_capability_invalid":
        return capability_present
    return False


def _stable_root_file(path: Path, *, maximum: int, mode: int = 0o444) -> bytes:
    descriptor = -1
    try:
        before = os.lstat(path)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or before.st_uid != 0
            or before.st_gid != 0
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != mode
            or not 0 < before.st_size <= maximum
        ):
            raise ValueError
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(descriptor)
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
    except (OSError, ValueError) as exc:
        raise OperationalEdgeClientError(
            "operational_edge_trust_file_invalid"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_uid,
        item.st_gid,
        item.st_nlink,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    raw = b"".join(chunks)
    if (
        len(raw) != before.st_size
        or identity(before) != identity(opened)
        or identity(before) != identity(after)
    ):
        raise OperationalEdgeClientError(
            "operational_edge_trust_file_changed"
        )
    return raw


class MainPidProvider(Protocol):
    def main_pid(self, unit: str) -> int: ...


class SystemctlMainPidProvider:
    def main_pid(self, unit: str) -> int:
        if _UNIT.fullmatch(unit) is None:
            raise OperationalEdgeClientError("operational_edge_unit_invalid")
        try:
            completed = subprocess.run(
                [str(SYSTEMCTL), "show", unit, "--property=MainPID", "--value"],
                env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                check=True,
                timeout=2,
            )
            pid = int(completed.stdout.strip())
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            raise OperationalEdgeClientError(
                "operational_edge_main_pid_unavailable"
            ) from exc
        if pid < 1:
            raise OperationalEdgeClientError("operational_edge_not_running")
        return pid


class AttestedMainPidFileProvider:
    """Read a short-lived root-owned MainPID observation inside the worker."""

    def __init__(self, path: Path, *, domain: str, maximum_age_seconds: int = 120) -> None:
        expected = Path("/run/muncho-operational-edge") / domain / "mainpid.json"
        if path != expected or not 5 <= maximum_age_seconds <= 300:
            raise ValueError("operational_edge_main_pid_attestation_invalid")
        self.path = path
        self.domain = domain
        self.maximum_age_seconds = maximum_age_seconds

    def main_pid(self, unit: str) -> int:
        try:
            raw = _stable_root_file(self.path, maximum=4096)
            value = decode_json_object(raw, maximum=4096)
        except (OperationalEdgeClientError, OperationalProtocolError) as exc:
            raise OperationalEdgeClientError(
                "operational_edge_main_pid_attestation_invalid"
            ) from exc
        if not isinstance(value, Mapping):
            raise OperationalEdgeClientError(
                "operational_edge_main_pid_attestation_invalid"
            )
        expected_fields = {
            "schema", "domain", "service_unit", "main_pid",
            "observed_at_unix", "attestation_sha256",
        }
        unsigned = {key: item for key, item in value.items() if key != "attestation_sha256"}
        expected_digest = hashlib.sha256(
            json.dumps(unsigned, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest()
        now = int(time.time())
        if (
            set(value) != expected_fields
            or value.get("schema") != "muncho-operational-edge-mainpid.v1"
            or value.get("domain") != self.domain
            or value.get("service_unit") != unit
            or raw
            != json.dumps(
                value,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("ascii")
            + b"\n"
            or type(value.get("main_pid")) is not int
            or value["main_pid"] < 1
            or type(value.get("observed_at_unix")) is not int
            or not 0 <= now - value["observed_at_unix"] <= self.maximum_age_seconds
            or value.get("attestation_sha256") != expected_digest
        ):
            raise OperationalEdgeClientError(
                "operational_edge_main_pid_attestation_invalid"
            )
        return value["main_pid"]


def linux_server_peer(sock: socket.socket) -> OperationalEdgePeer:
    option = getattr(socket, "SO_PEERCRED", None)
    if option is None:
        raise OSError("peer_credentials_unavailable")
    raw = sock.getsockopt(socket.SOL_SOCKET, option, _PEER.size)
    if len(raw) != _PEER.size:
        raise OSError("peer_credentials_invalid")
    return OperationalEdgePeer(*_PEER.unpack(raw))


@dataclass(frozen=True)
class OperationalEdgeClientConfig:
    domain: str
    socket_path: Path
    service_unit: str
    service_uid: int
    service_gid: int
    socket_gid: int
    probe_uid: int
    probe_gid: int
    probe_supplementary_gids: tuple[int, ...]
    receipt_public_key_file: Path
    receipt_key_id: str
    connect_timeout_seconds: float = 2.0
    request_timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        expected = Path("/run/muncho-operational-edge") / self.domain / "edge.sock"
        if (
            self.domain not in {item.domain for item in operation_catalog().values()}
            or self.socket_path != expected
            or self.service_unit != f"muncho-operational-edge-{self.domain}.service"
            or _UNIT.fullmatch(self.service_unit) is None
            or type(self.service_uid) is not int
            or type(self.service_gid) is not int
            or type(self.socket_gid) is not int
            or type(self.probe_uid) is not int
            or type(self.probe_gid) is not int
            or not isinstance(self.probe_supplementary_gids, tuple)
            or not self.probe_supplementary_gids
            or tuple(sorted(set(self.probe_supplementary_gids)))
            != self.probe_supplementary_gids
            or any(
                type(gid) is not int or gid < 1
                for gid in self.probe_supplementary_gids
            )
            or self.service_uid < 1
            or self.service_gid < 1
            or self.socket_gid < 1
            or self.probe_uid < 1
            or self.probe_gid < 1
            or self.probe_uid == self.service_uid
            or self.socket_gid not in self.probe_supplementary_gids
            or not self.receipt_public_key_file.is_absolute()
            or _SHA256.fullmatch(self.receipt_key_id or "") is None
            or not 0.1 <= float(self.connect_timeout_seconds) <= 10
            or not 1 <= float(self.request_timeout_seconds) <= 60
        ):
            raise ValueError("operational_edge_client_config_invalid")


def parse_operational_edge_client_configs(
    value: Any,
) -> Mapping[str, OperationalEdgeClientConfig]:
    """Parse one already provenance-checked canonical client config value."""

    if (
        not isinstance(value, Mapping)
        or set(value) != {"schema", "domains"}
        or value.get("schema") != CLIENT_CONFIG_SCHEMA
        or not isinstance(value.get("domains"), Mapping)
    ):
        raise OperationalEdgeClientError(
            "operational_edge_client_config_invalid"
        )
    configs: dict[str, OperationalEdgeClientConfig] = {}
    expected = {
        "socket_path", "service_unit", "service_uid", "service_gid", "socket_gid",
        "probe_uid", "probe_gid", "probe_supplementary_gids",
        "receipt_public_key_file", "receipt_key_id",
    }
    for domain, row in value["domains"].items():
        if not isinstance(domain, str) or not isinstance(row, Mapping) or set(row) != expected:
            raise OperationalEdgeClientError(
                "operational_edge_client_config_invalid"
            )
        configs[domain] = OperationalEdgeClientConfig(
            domain=domain,
            socket_path=Path(row["socket_path"]),
            service_unit=str(row["service_unit"]),
            service_uid=row["service_uid"],
            service_gid=row["service_gid"],
            socket_gid=row["socket_gid"],
            probe_uid=row["probe_uid"],
            probe_gid=row["probe_gid"],
            probe_supplementary_gids=tuple(row["probe_supplementary_gids"]),
            receipt_public_key_file=Path(row["receipt_public_key_file"]),
            receipt_key_id=str(row["receipt_key_id"]),
        )
    expected_domains = {item.domain for item in operation_catalog().values()}
    service_uids = {item.service_uid for item in configs.values()}
    service_gids = {item.service_gid for item in configs.values()}
    socket_gids = {item.socket_gid for item in configs.values()}
    probe_identities = {
        (item.probe_uid, item.probe_gid, item.probe_supplementary_gids)
        for item in configs.values()
    }
    if (
        set(configs) != expected_domains
        or len(service_uids) != len(expected_domains)
        or len(service_gids) != len(expected_domains)
        or len(socket_gids) != len(expected_domains)
        or service_gids & socket_gids
        or len(probe_identities) != 1
        or next(iter(probe_identities))[2] != tuple(sorted(socket_gids))
        or next(iter(probe_identities))[0] in service_uids
        or next(iter(probe_identities))[1] in service_gids | socket_gids
    ):
        raise OperationalEdgeClientError(
            "operational_edge_client_config_incomplete"
        )
    return configs


def load_operational_edge_client_configs(
    path: Path = DEFAULT_CLIENT_CONFIG_PATH,
) -> Mapping[str, OperationalEdgeClientConfig]:
    if path != DEFAULT_CLIENT_CONFIG_PATH:
        raise OperationalEdgeClientError("operational_edge_client_config_invalid")
    try:
        raw = _stable_root_file(path, maximum=256 * 1024)
        value = decode_json_object(raw, maximum=256 * 1024)
    except (OperationalEdgeClientError, OperationalProtocolError) as exc:
        raise OperationalEdgeClientError(
            "operational_edge_client_config_invalid"
        ) from exc
    if raw != json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii") + b"\n":
        raise OperationalEdgeClientError(
            "operational_edge_client_config_invalid"
        )
    return parse_operational_edge_client_configs(value)


def _public_key(path: Path) -> Ed25519PublicKey:
    try:
        raw = _stable_root_file(path, maximum=16 * 1024)
        key = serialization.load_pem_public_key(raw)
    except (OperationalEdgeClientError, TypeError, ValueError) as exc:
        raise OperationalEdgeClientError(
            "operational_edge_receipt_key_invalid"
        ) from exc
    if (
        not isinstance(key, Ed25519PublicKey)
        or raw
        != key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    ):
        raise OperationalEdgeClientError("operational_edge_receipt_key_invalid")
    return key


def _receive_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    while size:
        chunk = sock.recv(size)
        if not chunk:
            raise OSError("connection_closed")
        chunks.append(chunk)
        size -= len(chunk)
    return b"".join(chunks)


class OperationalEdgeClient:
    def __init__(
        self,
        config: OperationalEdgeClientConfig,
        *,
        main_pid_provider: MainPidProvider | None = None,
    ) -> None:
        self.config = config
        self.main_pid_provider = main_pid_provider or SystemctlMainPidProvider()
        self.receipt_public_key = _public_key(config.receipt_public_key_file)
        observed_key_id = hashlib.sha256(
            self.receipt_public_key.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
        ).hexdigest()
        if observed_key_id != config.receipt_key_id:
            raise OperationalEdgeClientError(
                "operational_edge_receipt_key_identity_invalid"
            )
        self._sequence = 0
        self._lock = threading.Lock()

    def _socket_identity(self) -> None:
        try:
            item = os.lstat(self.config.socket_path)
        except OSError as exc:
            raise OperationalEdgeClientError("operational_edge_unavailable") from exc
        if (
            not stat.S_ISSOCK(item.st_mode)
            or item.st_uid != self.config.service_uid
            or item.st_gid != self.config.socket_gid
            or stat.S_IMODE(item.st_mode) != 0o660
        ):
            raise OperationalEdgeClientError(
                "operational_edge_socket_identity_invalid"
            )

    def invoke(
        self,
        operation_id: str,
        arguments: Mapping[str, Any],
        *,
        idempotency_key: str,
        capability: Mapping[str, Any] | None = None,
        timeout_seconds: int = 60,
        _preserve_verified_envelope: bool = False,
    ) -> Mapping[str, Any]:
        operation = operation_catalog().get(operation_id)
        if operation is None or operation.domain != self.config.domain:
            raise OperationalEdgeClientError("operational_edge_operation_invalid")
        if not operation.available:
            raise OperationalEdgeClientError(operation.blocker_code)
        if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 60:
            raise OperationalEdgeClientError("operational_edge_timeout_invalid")
        intent = OperationalIntent(
            operation_id=operation_id,
            arguments=dict(arguments),
            arguments_sha256=sha256_json(dict(arguments)),
            idempotency_key=idempotency_key,
        )
        envelope = (
            None
            if capability is None
            else SignedEnvelope.from_mapping(
                capability, code="operational_edge_capability_invalid"
            )
        )
        with self._lock:
            sequence = self._sequence
            self._sequence += 1
        request = OperationalRequest(
            request_id=str(uuid.uuid4()),
            sequence=sequence,
            deadline_unix_ms=int(time.time() * 1000) + timeout_seconds * 1000,
            intent=intent,
            capability=envelope,
        )
        request_sha256 = hashlib.sha256(
            canonical_json_bytes(request.to_mapping())
        ).hexdigest()
        body = canonical_json_bytes(request.to_mapping())
        if len(body) > MAX_REQUEST_BYTES:
            raise OperationalEdgeClientError("operational_edge_request_oversized")
        self._socket_identity()
        expected_pid = self.main_pid_provider.main_pid(self.config.service_unit)
        reached = False
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.settimeout(self.config.connect_timeout_seconds)
                sock.connect(str(self.config.socket_path))
                peer = linux_server_peer(sock)
                if (
                    peer.pid != expected_pid
                    or peer.uid != self.config.service_uid
                    or peer.gid != self.config.service_gid
                ):
                    raise OperationalEdgeClientError(
                        "operational_edge_server_unauthorized"
                    )
                sock.settimeout(min(timeout_seconds, self.config.request_timeout_seconds))
                sock.sendall(_FRAME.pack(len(body)) + body)
                reached = True
                (size,) = _FRAME.unpack(_receive_exact(sock, _FRAME.size))
                if not 0 < size <= MAX_RESPONSE_BYTES:
                    raise OperationalEdgeClientError(
                        "operational_edge_response_invalid",
                        dispatch_uncertain=True,
                    )
                raw = _receive_exact(sock, size)
        except OperationalEdgeClientError:
            raise
        except (OSError, TimeoutError) as exc:
            raise OperationalEdgeClientError(
                "operational_edge_transport_failed",
                dispatch_uncertain=reached,
            ) from exc
        response = decode_json_object(raw, maximum=MAX_RESPONSE_BYTES)
        payload = verify_envelope(
            response,
            key_id=self.config.receipt_key_id,
            public_key=self.receipt_public_key,
            code="operational_edge_receipt_signature_invalid",
        )
        required = {
            "schema", "request_id", "operation_id", "arguments_sha256",
            "idempotency_key", "domain", "access", "outcome", "service_pid",
            "service_unit", "release_revision", "request_sha256",
            "executable_sha256", "return_code", "stdout_b64", "stderr_b64",
            "started_at_unix_ms", "finished_at_unix_ms", "blocker_code",
            "dispatched", "executable_started", "mutation_performed",
            "readback_verified", "secret_material_recorded",
        }
        try:
            receipt_request_id = uuid.UUID(str(payload.get("request_id")))
        except (ValueError, TypeError, AttributeError) as exc:
            raise OperationalEdgeClientError(
                "operational_edge_receipt_mismatch", dispatch_uncertain=True
            ) from exc
        outcome = payload.get("outcome")
        if (
            set(payload) != required
            or payload.get("schema") != RECEIPT_SCHEMA
            or receipt_request_id.version != 4
            or str(receipt_request_id) != payload.get("request_id")
            or payload.get("operation_id") != operation_id
            or payload.get("arguments_sha256") != intent.arguments_sha256
            or payload.get("idempotency_key") != idempotency_key
            or payload.get("domain") != self.config.domain
            or payload.get("service_unit") != self.config.service_unit
            or not isinstance(payload.get("release_revision"), str)
            or _REVISION.fullmatch(payload["release_revision"]) is None
            or payload.get("request_sha256") != request_sha256
            or payload.get("access") != operation.access.value
            or type(payload.get("service_pid")) is not int
            or payload["service_pid"] != expected_pid
            or outcome not in {"succeeded", "blocked", "dispatch_uncertain"}
            or (
                outcome == "succeeded"
                and (
                    payload.get("return_code") != 0
                    or payload.get("readback_verified") is not True
                    or payload.get("blocker_code") is not None
                )
            )
            or (
                outcome != "succeeded"
                and payload.get("readback_verified") is not False
            )
            or payload.get("secret_material_recorded") is not False
        ):
            raise OperationalEdgeClientError(
                "operational_edge_receipt_mismatch", dispatch_uncertain=True
            )
        dispatched = payload["dispatched"]
        executable_started = payload["executable_started"]
        mutation_performed = payload["mutation_performed"]
        predispatch_denial = (
            operation.access.value == "mutation"
            and outcome == "blocked"
            and payload["blocker_code"] in PREDISPATCH_MUTATION_BLOCKERS
        )
        if (
            type(dispatched) is not bool
            or type(executable_started) is not bool
            or (
                mutation_performed is not None
                and type(mutation_performed) is not bool
            )
            or executable_started and not dispatched
            or (
                predispatch_denial
                and (
                    not _predispatch_capability_truth_matches(
                        payload["blocker_code"],
                        capability_present=request.capability is not None,
                    )
                    or dispatched is not False
                    or executable_started is not False
                    or mutation_performed is not False
                    or payload["return_code"] is not None
                    or payload["executable_sha256"] != "0" * 64
                    or payload["stdout_b64"] != ""
                    or payload["stderr_b64"] != ""
                )
            )
            or (
                outcome == "succeeded"
                and (
                    dispatched is not True
                    or executable_started is not True
                    or mutation_performed
                    is not (operation.access.value == "mutation")
                )
            )
            or (
                operation.access.value != "mutation"
                and mutation_performed is not False
            )
        ):
            raise OperationalEdgeClientError(
                "operational_edge_receipt_mismatch",
                dispatch_uncertain=True,
            )
        result = dict(payload)
        if not _preserve_verified_envelope:
            return result
        return {
            "schema": "muncho-operational-edge-verified-evidence.v1",
            "payload": result,
            "signed_envelope": dict(response),
            "signed_envelope_sha256": hashlib.sha256(raw).hexdigest(),
            "request_sha256": request_sha256,
            "peer": {
                "pid": peer.pid,
                "uid": peer.uid,
                "gid": peer.gid,
                "service_unit": self.config.service_unit,
            },
        }

    def invoke_verified_evidence(
        self,
        operation_id: str,
        arguments: Mapping[str, Any],
        *,
        idempotency_key: str,
        expected_release_revision: str,
        capability: Mapping[str, Any] | None = None,
        timeout_seconds: int = 60,
    ) -> Mapping[str, Any]:
        """Return the native signed envelope after all client checks pass."""

        if _REVISION.fullmatch(expected_release_revision or "") is None:
            raise OperationalEdgeClientError("operational_edge_release_invalid")
        evidence = self.invoke(
            operation_id,
            arguments,
            idempotency_key=idempotency_key,
            capability=capability,
            timeout_seconds=timeout_seconds,
            _preserve_verified_envelope=True,
        )
        payload = evidence.get("payload")
        if (
            not isinstance(payload, Mapping)
            or payload.get("release_revision") != expected_release_revision
        ):
            raise OperationalEdgeClientError(
                "operational_edge_release_mismatch",
                dispatch_uncertain=True,
            )
        return evidence


__all__ = [
    "AttestedMainPidFileProvider",
    "CLIENT_CONFIG_SCHEMA",
    "MainPidProvider",
    "OperationalEdgeClient",
    "OperationalEdgeClientConfig",
    "OperationalEdgeClientError",
    "SystemctlMainPidProvider",
    "load_operational_edge_client_configs",
    "linux_server_peer",
    "parse_operational_edge_client_configs",
]

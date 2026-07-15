#!/usr/bin/env python3
"""Credential-scoped AF_UNIX executor for explicit operational operations."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import signal
import socket
import sqlite3
import stat
import struct
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from gateway.operational_edge_assets import (
    ASSET_MANIFEST_RELATIVE,
    validate_operational_asset_manifest,
)
from gateway.operational_edge_catalog import (
    HERMES_HOME,
    asset_catalog,
    build_operation_argv,
    catalog_public_contract,
    operation_catalog,
)
from gateway.operational_edge_protocol import (
    MAX_REQUEST_BYTES,
    MAX_RESPONSE_BYTES,
    PREDISPATCH_MUTATION_BLOCKERS,
    OperationalAccess,
    OperationalOutcome,
    OperationalProtocolError,
    OperationalRequest,
    canonical_json_bytes,
    decode_json_object,
    receipt_payload,
    sign_envelope,
    verify_mutation_capability,
)


CONFIG_SCHEMA = "muncho-operational-edge-service-config.v1"
DEFAULT_SOCKET_ROOT = Path("/run/muncho-operational-edge")
DEFAULT_STATE_ROOT = Path("/var/lib/muncho-operational-edge")
MAX_CONFIG_BYTES = 512 * 1024
MAX_OUTPUT_BYTES = 1024 * 1024

_SHA256 = re.compile(r"^[0-9a-f]{64}$")

_FRAME = struct.Struct("!I")
_PEER = struct.Struct("3i")


class OperationalEdgeServiceError(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class OperationalEdgePeer:
    pid: int
    uid: int
    gid: int


@dataclass(frozen=True)
class OperationalEdgeServiceConfig:
    domain: str
    release_revision: str
    release_root: Path
    release_owner_uid: int
    release_owner_gid: int
    socket_path: Path
    socket_gid: int
    service_uid: int
    service_gid: int
    allowed_read_peer_uids: frozenset[int]
    mutation_peer_uid: int
    journal_path: Path
    subprocess_home: Path
    receipt_private_key_file: Path
    receipt_key_id: str
    writer_public_key_file: Path
    writer_key_id: str
    maximum_output_bytes: int
    maximum_connections: int


def linux_peer_credentials(sock: socket.socket) -> OperationalEdgePeer:
    option = getattr(socket, "SO_PEERCRED", None)
    if option is None:
        raise OSError("peer_credentials_unavailable")
    raw = sock.getsockopt(socket.SOL_SOCKET, option, _PEER.size)
    if len(raw) != _PEER.size:
        raise OSError("peer_credentials_invalid")
    return OperationalEdgePeer(*_PEER.unpack(raw))


def _stable_regular(
    path: Path,
    *,
    maximum: int,
    allowed_modes: frozenset[int],
    expected_uid: int | None,
    expected_gid: int | None = None,
) -> bytes:
    descriptor = -1
    try:
        before = os.lstat(path)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) not in allowed_modes
            or expected_uid is not None and before.st_uid != expected_uid
            or expected_gid is not None and before.st_gid != expected_gid
            or not 0 < before.st_size <= maximum
        ):
            raise OperationalEdgeServiceError("protected_file_invalid")
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
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
    except OperationalEdgeServiceError:
        raise
    except OSError as exc:
        raise OperationalEdgeServiceError("protected_file_unavailable") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    identity = lambda item: (
        item.st_dev, item.st_ino, item.st_mode, item.st_uid, item.st_gid,
        item.st_nlink, item.st_size, item.st_mtime_ns, item.st_ctime_ns,
    )
    raw = b"".join(chunks)
    if len(raw) != before.st_size or identity(before) != identity(opened) or identity(before) != identity(after):
        raise OperationalEdgeServiceError("protected_file_changed")
    return raw


def _path(value: Any, *, expected: Path | None = None) -> Path:
    path = Path(value) if isinstance(value, str) else Path("")
    if not path.is_absolute() or ".." in path.parts or expected is not None and path != expected:
        raise OperationalEdgeServiceError("service_config_path_invalid")
    return path


def load_config(
    path: Path,
    *,
    expected_owner_uid: int | None = 0,
    require_service_credential_path: bool = False,
) -> OperationalEdgeServiceConfig:
    owners = (
        (expected_owner_uid,)
        if expected_owner_uid is not None
        else tuple(dict.fromkeys((os.geteuid(), 0)))
    )
    raw: bytes | None = None
    last_error: OperationalEdgeServiceError | None = None
    for owner_uid in owners:
        try:
            raw = _stable_regular(
                path,
                maximum=MAX_CONFIG_BYTES,
                allowed_modes=frozenset({0o400}),
                expected_uid=owner_uid,
            )
            break
        except OperationalEdgeServiceError as exc:
            last_error = exc
            if exc.code != "protected_file_invalid":
                raise
    if raw is None:
        assert last_error is not None
        raise last_error
    value = decode_json_object(raw, maximum=MAX_CONFIG_BYTES)
    fields = {
        "schema", "domain", "release_revision", "release_root", "socket_path",
        "release_owner_uid", "release_owner_gid",
        "socket_gid", "service_uid", "service_gid", "allowed_read_peer_uids",
        "mutation_peer_uid", "journal_path", "subprocess_home",
        "receipt_private_key_file", "receipt_key_id", "writer_public_key_file",
        "writer_key_id", "maximum_output_bytes", "maximum_connections",
        "catalog_sha256",
    }
    if set(value) != fields or value.get("schema") != CONFIG_SCHEMA:
        raise OperationalEdgeServiceError("service_config_invalid")
    domain = value.get("domain")
    domains = {item.domain for item in operation_catalog().values()}
    release_revision = value.get("release_revision")
    release_root = _path(value.get("release_root"))
    socket_path = _path(
        value.get("socket_path"),
        expected=DEFAULT_SOCKET_ROOT / str(domain) / "edge.sock",
    )
    journal_path = _path(
        value.get("journal_path"),
        expected=DEFAULT_STATE_ROOT / str(domain) / "journal.sqlite3",
    )
    subprocess_home = _path(value.get("subprocess_home"))
    revision_text = release_revision if isinstance(release_revision, str) else ""
    production_release_root = (
        Path("/opt/adventico-ai-platform/hermes-agent-releases")
        / f"hermes-agent-{revision_text[:12]}"
    )
    canary_release_root = (
        Path("/opt/muncho-canary-releases") / revision_text
    )
    if (
        domain not in domains
        or not isinstance(release_revision, str)
        or len(release_revision) != 40
        or any(character not in "0123456789abcdef" for character in release_revision)
        or release_root not in {production_release_root, canary_release_root}
        or subprocess_home != Path("/opt/adventico-ai-platform")
        or value.get("catalog_sha256")
        != hashlib.sha256(
            json.dumps(catalog_public_contract(), ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest()
    ):
        raise OperationalEdgeServiceError("service_config_identity_invalid")
    if require_service_credential_path and path != (
        Path("/run/credentials")
        / f"muncho-operational-edge-{domain}.service"
        / "service-config"
    ):
        raise OperationalEdgeServiceError("service_config_path_invalid")
    integer_fields = (
        "socket_gid", "service_uid", "service_gid", "mutation_peer_uid",
        "maximum_output_bytes", "maximum_connections",
    )
    if any(type(value.get(name)) is not int or value[name] < 1 for name in integer_fields):
        raise OperationalEdgeServiceError("service_config_identity_invalid")
    release_owner = (
        value.get("release_owner_uid"), value.get("release_owner_gid")
    )
    if (
        release_root == canary_release_root
        and release_owner != (0, 0)
        or release_root == production_release_root
        and (
            any(type(item) is not int or item < 1 for item in release_owner)
        )
    ):
        # Canary artifacts are installed by the owner launcher as immutable
        # root:root bytes.  The long-lived production release tree retains its
        # existing non-root owner invariant.  No mixed or caller-selected
        # ownership is accepted on either exact release-root branch.
        raise OperationalEdgeServiceError("service_config_identity_invalid")
    peers = value.get("allowed_read_peer_uids")
    if (
        not isinstance(peers, list)
        or not peers
        or len(peers) > 16
        or any(type(item) is not int or item < 1 for item in peers)
        or len(peers) != len(set(peers))
        or peers != sorted(peers)
        or value["mutation_peer_uid"] not in peers
        or value["mutation_peer_uid"] == value["service_uid"]
        or value["socket_gid"] == value["service_gid"]
        or _SHA256.fullmatch(str(value.get("receipt_key_id") or "")) is None
        or _SHA256.fullmatch(str(value.get("writer_key_id") or "")) is None
        or not 4096 <= value["maximum_output_bytes"] <= MAX_OUTPUT_BYTES
        or not 1 <= value["maximum_connections"] <= 64
    ):
        raise OperationalEdgeServiceError("service_config_identity_invalid")
    return OperationalEdgeServiceConfig(
        domain=domain,
        release_revision=release_revision,
        release_root=release_root,
        release_owner_uid=value["release_owner_uid"],
        release_owner_gid=value["release_owner_gid"],
        socket_path=socket_path,
        socket_gid=value["socket_gid"],
        service_uid=value["service_uid"],
        service_gid=value["service_gid"],
        allowed_read_peer_uids=frozenset(peers),
        mutation_peer_uid=value["mutation_peer_uid"],
        journal_path=journal_path,
        subprocess_home=subprocess_home,
        receipt_private_key_file=_path(
            value["receipt_private_key_file"],
            expected=(
                Path("/run/credentials")
                / f"muncho-operational-edge-{domain}.service"
                / "receipt-private-key"
            ),
        ),
        receipt_key_id=str(value["receipt_key_id"]),
        writer_public_key_file=_path(
            value["writer_public_key_file"],
            expected=(
                Path("/run/credentials")
                / f"muncho-operational-edge-{domain}.service"
                / "writer-public-key"
            ),
        ),
        writer_key_id=str(value["writer_key_id"]),
        maximum_output_bytes=value["maximum_output_bytes"],
        maximum_connections=value["maximum_connections"],
    )


def _load_private(path: Path, service_uid: int) -> Ed25519PrivateKey:
    # systemd credentials are owned either by root or by the service UID,
    # depending on the manager/kernel mount path.  No other owner is valid.
    try:
        raw = _stable_regular(
            path,
            maximum=16 * 1024,
            allowed_modes=frozenset({0o400}),
            expected_uid=service_uid,
        )
    except OperationalEdgeServiceError as exc:
        if exc.code != "protected_file_invalid":
            raise
        raw = _stable_regular(
            path,
            maximum=16 * 1024,
            allowed_modes=frozenset({0o400}),
            expected_uid=0,
        )
    try:
        key = serialization.load_pem_private_key(raw, password=None)
    except (TypeError, ValueError) as exc:
        raise OperationalEdgeServiceError("receipt_private_key_invalid") from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise OperationalEdgeServiceError("receipt_private_key_invalid")
    if raw != key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ):
        raise OperationalEdgeServiceError("receipt_private_key_invalid")
    return key


def _load_public(path: Path, service_uid: int) -> Ed25519PublicKey:
    try:
        raw = _stable_regular(
            path,
            maximum=16 * 1024,
            allowed_modes=frozenset({0o400, 0o440, 0o444}),
            expected_uid=service_uid,
        )
    except OperationalEdgeServiceError as exc:
        if exc.code != "protected_file_invalid":
            raise
        raw = _stable_regular(
            path,
            maximum=16 * 1024,
            allowed_modes=frozenset({0o400, 0o440, 0o444}),
            expected_uid=0,
        )
    try:
        key = serialization.load_pem_public_key(raw)
    except (TypeError, ValueError) as exc:
        raise OperationalEdgeServiceError("writer_public_key_invalid") from exc
    if not isinstance(key, Ed25519PublicKey):
        raise OperationalEdgeServiceError("writer_public_key_invalid")
    if raw != key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ):
        raise OperationalEdgeServiceError("writer_public_key_invalid")
    return key


def _key_id(key: Ed25519PublicKey) -> str:
    return hashlib.sha256(
        key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
    ).hexdigest()


class OperationalEdgeJournal:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._db = sqlite3.connect(path, check_same_thread=False, timeout=5)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS receipts ("
            "idempotency_key TEXT PRIMARY KEY, request_sha256 TEXT NOT NULL, "
            "response_json BLOB NOT NULL, created_unix_ms INTEGER NOT NULL)"
        )
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS predispatch_denials ("
            "idempotency_key TEXT NOT NULL, intent_sha256 TEXT NOT NULL, "
            "capability_state TEXT NOT NULL, request_sha256 TEXT NOT NULL, "
            "response_json BLOB NOT NULL, created_unix_ms INTEGER NOT NULL, "
            "PRIMARY KEY(idempotency_key,intent_sha256,capability_state))"
        )
        self._db.commit()
        self._lock = threading.Lock()

    def read(self, key: str, request_sha256: str) -> bytes | None:
        with self._lock:
            row = self._db.execute(
                "SELECT request_sha256,response_json FROM receipts WHERE idempotency_key=?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        if row[0] != request_sha256:
            raise OperationalEdgeServiceError("idempotency_conflict")
        return bytes(row[1])

    def store(self, key: str, request_sha256: str, response: bytes) -> None:
        with self._lock:
            try:
                self._db.execute("BEGIN IMMEDIATE")
                existing = self._db.execute(
                    "SELECT request_sha256,response_json FROM receipts WHERE idempotency_key=?",
                    (key,),
                ).fetchone()
                if existing is None:
                    self._db.execute(
                        "INSERT INTO receipts VALUES (?,?,?,?)",
                        (key, request_sha256, response, int(time.time() * 1000)),
                    )
                elif existing[0] != request_sha256 or bytes(existing[1]) != response:
                    raise OperationalEdgeServiceError("idempotency_conflict")
                self._db.commit()
            except BaseException:
                self._db.rollback()
                raise

    @staticmethod
    def _capability_state(value: str) -> str:
        if value not in {"absent", "invalid"}:
            raise OperationalEdgeServiceError(
                "predispatch_capability_state_invalid"
            )
        return value

    def read_predispatch_denial(
        self,
        key: str,
        intent_sha256: str,
        capability_state: str,
        request_sha256: str,
    ) -> bytes | None:
        state = self._capability_state(capability_state)
        with self._lock:
            row = self._db.execute(
                "SELECT request_sha256,response_json FROM "
                "predispatch_denials WHERE idempotency_key=? "
                "AND intent_sha256=? AND capability_state=?",
                (key, intent_sha256, state),
            ).fetchone()
        if row is None:
            return None
        if row[0] != request_sha256:
            raise OperationalEdgeServiceError("idempotency_conflict")
        return bytes(row[1])

    def store_predispatch_denial(
        self,
        key: str,
        intent_sha256: str,
        capability_state: str,
        request_sha256: str,
        response: bytes,
    ) -> None:
        state = self._capability_state(capability_state)
        with self._lock:
            try:
                self._db.execute("BEGIN IMMEDIATE")
                existing = self._db.execute(
                    "SELECT request_sha256,response_json FROM "
                    "predispatch_denials WHERE idempotency_key=? "
                    "AND intent_sha256=? AND capability_state=?",
                    (key, intent_sha256, state),
                ).fetchone()
                if existing is None:
                    self._db.execute(
                        "INSERT INTO predispatch_denials VALUES (?,?,?,?,?,?)",
                        (
                            key,
                            intent_sha256,
                            state,
                            request_sha256,
                            response,
                            int(time.time() * 1000),
                        ),
                    )
                elif (
                    existing[0] != request_sha256
                    or bytes(existing[1]) != response
                ):
                    raise OperationalEdgeServiceError("idempotency_conflict")
                self._db.commit()
            except BaseException:
                self._db.rollback()
                raise

    def close(self) -> None:
        self._db.close()


class OperationalEdgeService:
    def __init__(self, config: OperationalEdgeServiceConfig) -> None:
        if os.getuid() != config.service_uid or os.getgid() != config.service_gid:
            raise OperationalEdgeServiceError("service_process_identity_invalid")
        try:
            resolved = config.release_root.resolve(strict=True)
            release_item = config.release_root.lstat()
        except OSError as exc:
            raise OperationalEdgeServiceError(
                "release_root_identity_invalid"
            ) from exc
        if (
            resolved != config.release_root
            or not stat.S_ISDIR(release_item.st_mode)
            or stat.S_ISLNK(release_item.st_mode)
            or release_item.st_uid != config.release_owner_uid
            or release_item.st_gid != config.release_owner_gid
            or stat.S_IMODE(release_item.st_mode) & 0o022
        ):
            raise OperationalEdgeServiceError(
                "release_root_identity_invalid"
            )
        self.config = config
        self.operations = {
            key: item for key, item in operation_catalog().items()
            if item.domain == config.domain
        }
        manifest_raw = _stable_regular(
            config.release_root / ASSET_MANIFEST_RELATIVE,
            maximum=4 * 1024 * 1024,
            allowed_modes=frozenset({0o400, 0o440, 0o444}),
            expected_uid=config.release_owner_uid,
            expected_gid=config.release_owner_gid,
        )
        try:
            manifest_value = decode_json_object(
                manifest_raw,
                maximum=4 * 1024 * 1024,
            )
        except OperationalProtocolError as exc:
            raise OperationalEdgeServiceError("asset_manifest_invalid") from exc
        if manifest_raw != (
            json.dumps(
                manifest_value,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("ascii")
            + b"\n"
        ):
            raise OperationalEdgeServiceError("asset_manifest_invalid")
        self.manifest = validate_operational_asset_manifest(
            manifest_value, revision=config.release_revision
        )
        self.assets = {row["asset_id"]: row for row in self.manifest["assets"]}
        self.receipt_private_key = _load_private(
            config.receipt_private_key_file, config.service_uid
        )
        self.writer_public_key = _load_public(
            config.writer_public_key_file,
            config.service_uid,
        )
        if (
            _key_id(self.receipt_private_key.public_key())
            != config.receipt_key_id
            or _key_id(self.writer_public_key) != config.writer_key_id
        ):
            raise OperationalEdgeServiceError("service_key_identity_invalid")
        self.journal = OperationalEdgeJournal(config.journal_path)

    def close(self) -> None:
        self.journal.close()

    def _asset(self, asset_id: str) -> tuple[Path, str]:
        row = self.assets.get(asset_id)
        catalog = asset_catalog().get(asset_id)
        if row is None or catalog is None:
            raise OperationalEdgeServiceError("operation_asset_unavailable")
        path = self.config.release_root / catalog.packaged_relative
        raw = _stable_regular(
            path,
            maximum=16 * 1024 * 1024,
            allowed_modes=frozenset({0o555}),
            expected_uid=self.config.release_owner_uid,
            expected_gid=self.config.release_owner_gid,
        )
        digest = hashlib.sha256(raw).hexdigest()
        if digest != row["sha256"]:
            raise OperationalEdgeServiceError("operation_asset_digest_mismatch")
        return path, digest

    def _argv(self, operation: Any, arguments: Mapping[str, Any]) -> tuple[list[str], str]:
        asset, digest = self._asset(operation.asset_id)
        dynamic = build_operation_argv(operation, arguments)
        interpreter = self.config.release_root / (
            "venv/bin/python"
            if self.config.release_root.parent
            == Path("/opt/muncho-canary-releases")
            else ".venv/bin/python"
        )
        if operation.runner_module:
            argv = [
                str(interpreter), "-I", "-B", "-m", operation.runner_module,
                "--asset", str(asset), *dynamic,
            ]
        elif asset.suffix == ".py":
            argv = [str(interpreter), "-I", "-B", str(asset), *dynamic]
        else:
            argv = [str(asset), *dynamic]
        return argv, digest

    @staticmethod
    def _meaningful(stdout: bytes, return_code: int) -> bool:
        if return_code != 0 or not stdout:
            return False
        try:
            value = json.loads(stdout.decode("utf-8", errors="strict"))
        except (UnicodeError, json.JSONDecodeError):
            # Canonical/observer scripts may intentionally emit bounded text.
            return bool(stdout.strip())
        if not isinstance(value, Mapping):
            return False
        status = value.get("status")
        return status not in {"BLOCKED", "ERROR", "FAILED"}

    def dispatch(self, request: OperationalRequest, peer: OperationalEdgePeer) -> bytes:
        operation = self.operations.get(request.intent.operation_id)
        if operation is None:
            raise OperationalEdgeServiceError("operation_not_allowlisted")
        if not operation.available:
            raise OperationalEdgeServiceError(operation.blocker_code)
        if peer.uid not in self.config.allowed_read_peer_uids:
            raise OperationalEdgeServiceError("peer_unauthorized")
        request_sha256 = hashlib.sha256(
            canonical_json_bytes(request.to_mapping())
        ).hexdigest()
        intent_sha256 = hashlib.sha256(
            canonical_json_bytes(request.intent.to_mapping())
        ).hexdigest()
        if operation.access is OperationalAccess.MUTATION:
            if peer.uid != self.config.mutation_peer_uid:
                raise OperationalEdgeServiceError("mutation_peer_unauthorized")
            capability_blocker: str | None = None
            if request.capability is None:
                capability_blocker = "mutation_capability_required"
            else:
                try:
                    verify_mutation_capability(
                        request,
                        key_id=self.config.writer_key_id,
                        public_key=self.writer_public_key,
                    )
                except OperationalProtocolError:
                    capability_blocker = "mutation_capability_invalid"
            if capability_blocker is not None:
                if capability_blocker not in PREDISPATCH_MUTATION_BLOCKERS:
                    raise OperationalEdgeServiceError(
                        "mutation_capability_denial_invalid"
                    )
                capability_state = (
                    "absent"
                    if capability_blocker == "mutation_capability_required"
                    else "invalid"
                )
                replay = self.journal.read_predispatch_denial(
                    request.intent.idempotency_key,
                    intent_sha256,
                    capability_state,
                    request_sha256,
                )
                if replay is not None:
                    return replay
                occurred = int(time.time() * 1000)
                denial = receipt_payload(
                    request=request,
                    domain=self.config.domain,
                    service_unit=(
                        f"muncho-operational-edge-{self.config.domain}.service"
                    ),
                    release_revision=self.config.release_revision,
                    request_sha256=request_sha256,
                    access=operation.access,
                    outcome=OperationalOutcome.BLOCKED,
                    service_pid=os.getpid(),
                    executable_sha256="0" * 64,
                    return_code=None,
                    stdout_b64="",
                    stderr_b64="",
                    started_at_unix_ms=occurred,
                    finished_at_unix_ms=occurred,
                    blocker_code=capability_blocker,
                    dispatched=False,
                    executable_started=False,
                    mutation_performed=False,
                    readback_verified=False,
                )
                response = canonical_json_bytes(
                    sign_envelope(
                        denial,
                        key_id=self.config.receipt_key_id,
                        private_key=self.receipt_private_key,
                    ).to_mapping()
                )
                self.journal.store_predispatch_denial(
                    request.intent.idempotency_key,
                    intent_sha256,
                    capability_state,
                    request_sha256,
                    response,
                )
                return response
        elif request.capability is not None:
            raise OperationalEdgeServiceError("read_capability_not_accepted")
        replay = self.journal.read(
            request.intent.idempotency_key, intent_sha256
        )
        if replay is not None:
            return replay
        argv, executable_sha256 = self._argv(operation, request.intent.arguments)
        started = int(time.time() * 1000)
        timeout = max(
            1,
            min(operation.timeout_seconds, (request.deadline_unix_ms - started) // 1000),
        )
        try:
            completed = subprocess.run(
                argv,
                cwd=str(self.config.release_root),
                env={
                    "HOME": str(self.config.subprocess_home),
                    "HERMES_HOME": str(HERMES_HOME),
                    "LANG": "C.UTF-8",
                    "LC_ALL": "C.UTF-8",
                    "PATH": "/usr/bin:/bin",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONNOUSERSITE": "1",
                    "TZ": "UTC",
                },
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=timeout,
            )
            stdout = completed.stdout[: self.config.maximum_output_bytes]
            stderr = completed.stderr[: self.config.maximum_output_bytes]
            meaningful = self._meaningful(stdout, completed.returncode)
            outcome = OperationalOutcome.SUCCEEDED if meaningful else OperationalOutcome.BLOCKED
            blocker = None if meaningful else "operation_result_not_meaningful"
            return_code: int | None = completed.returncode
        except subprocess.TimeoutExpired as exc:
            stdout = bytes(exc.stdout or b"")[: self.config.maximum_output_bytes]
            stderr = bytes(exc.stderr or b"")[: self.config.maximum_output_bytes]
            outcome = (
                OperationalOutcome.DISPATCH_UNCERTAIN
                if operation.access is OperationalAccess.MUTATION
                else OperationalOutcome.BLOCKED
            )
            blocker = "operation_timeout"
            return_code = None
        finished = int(time.time() * 1000)
        payload = receipt_payload(
            request=request,
            domain=self.config.domain,
            service_unit=f"muncho-operational-edge-{self.config.domain}.service",
            release_revision=self.config.release_revision,
            request_sha256=request_sha256,
            access=operation.access,
            outcome=outcome,
            service_pid=os.getpid(),
            executable_sha256=executable_sha256,
            return_code=return_code,
            stdout_b64=base64.b64encode(stdout).decode("ascii"),
            stderr_b64=base64.b64encode(stderr).decode("ascii"),
            started_at_unix_ms=started,
            finished_at_unix_ms=finished,
            blocker_code=blocker,
            dispatched=True,
            executable_started=True,
            mutation_performed=(
                True
                if operation.access is OperationalAccess.MUTATION
                and outcome is OperationalOutcome.SUCCEEDED
                else None
                if operation.access is OperationalAccess.MUTATION
                else False
            ),
            readback_verified=outcome is OperationalOutcome.SUCCEEDED,
        )
        response = canonical_json_bytes(
            sign_envelope(
                payload,
                key_id=self.config.receipt_key_id,
                private_key=self.receipt_private_key,
            ).to_mapping()
        )
        if len(response) > MAX_RESPONSE_BYTES:
            raise OperationalEdgeServiceError("operation_response_oversized")
        self.journal.store(
            request.intent.idempotency_key, intent_sha256, response
        )
        return response


def _receive_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    while size:
        chunk = sock.recv(size)
        if not chunk:
            raise OSError("connection_closed")
        chunks.append(chunk)
        size -= len(chunk)
    return b"".join(chunks)


def _receive(sock: socket.socket) -> bytes:
    (size,) = _FRAME.unpack(_receive_exact(sock, _FRAME.size))
    if not 0 < size <= MAX_REQUEST_BYTES:
        raise OperationalEdgeServiceError("request_frame_invalid")
    return _receive_exact(sock, size)


def _send(sock: socket.socket, value: bytes) -> None:
    if not 0 < len(value) <= MAX_RESPONSE_BYTES:
        raise OperationalEdgeServiceError("response_frame_invalid")
    sock.sendall(_FRAME.pack(len(value)) + value)


def serve(config: OperationalEdgeServiceConfig) -> None:
    if (
        os.geteuid() == 0
        or os.geteuid() != config.service_uid
        or os.getegid() != config.service_gid
    ):
        raise OperationalEdgeServiceError("service_process_identity_invalid")
    service = OperationalEdgeService(config)
    stop = threading.Event()
    previous_term = signal.signal(signal.SIGTERM, lambda *_args: stop.set())
    previous_int = signal.signal(signal.SIGINT, lambda *_args: stop.set())
    config.socket_path.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    try:
        config.socket_path.unlink(missing_ok=True)
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(config.socket_path))
        os.chown(config.socket_path, config.service_uid, config.socket_gid)
        os.chmod(config.socket_path, 0o660)
        listener.listen(config.maximum_connections)
        listener.settimeout(0.2)
        try:
            while not stop.is_set():
                try:
                    connection, _address = listener.accept()
                except socket.timeout:
                    continue
                with connection:
                    try:
                        peer = linux_peer_credentials(connection)
                        raw = _receive(connection)
                        request = OperationalRequest.from_mapping(
                            decode_json_object(raw, maximum=MAX_REQUEST_BYTES)
                        )
                        _send(connection, service.dispatch(request, peer))
                    except (OSError, ValueError, OperationalProtocolError, OperationalEdgeServiceError):
                        # No raw exception text or request data crosses the edge.
                        continue
        finally:
            listener.close()
    finally:
        config.socket_path.unlink(missing_ok=True)
        service.close()
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGINT, previous_int)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="muncho-operational-edge-service")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    serve(
        load_config(
            args.config,
            expected_owner_uid=None,
            require_service_credential_path=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONFIG_SCHEMA",
    "DEFAULT_SOCKET_ROOT",
    "DEFAULT_STATE_ROOT",
    "OperationalEdgePeer",
    "OperationalEdgeService",
    "OperationalEdgeServiceConfig",
    "OperationalEdgeServiceError",
    "linux_peer_credentials",
    "load_config",
    "serve",
]

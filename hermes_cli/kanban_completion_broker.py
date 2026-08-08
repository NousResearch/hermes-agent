"""Least-privilege Unix-socket broker for governed Kanban completion.

The intended production topology is:

* the Kanban database is owned and writable only by a dedicated broker UID;
* worker/coordinator processes can reach a group-owned Unix socket but cannot
  open the database or its WAL/SHM files for writing;
* Linux ``SO_PEERCRED`` binds each request to an explicitly configured UID and
  profile before the broker constructs :class:`CompletionContext`;
* the broker calls the same transactional kernel guard as every in-process
  completion path.

This module is an opt-in candidate. Merely importing it does not create sockets,
change permissions, migrate a database, or start a service.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import stat
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from hermes_cli import kanban_completion_guard as guard
from hermes_cli import kanban_db


_PROTOCOL_VERSION = "1.0.0"
_MAX_REQUEST_BYTES = 1_048_576
_REQUEST_FIELDS = {
    "version",
    "operation",
    "request_id",
    "profile",
    "task_id",
    "run_id",
    "result",
    "summary",
    "metadata",
    "created_cards",
}


class BrokerProtocolError(ValueError):
    """A request was malformed or not authorized for the connected peer."""


@dataclass(frozen=True)
class PeerCredentials:
    pid: int
    uid: int
    gid: int


@dataclass(frozen=True)
class BrokerConfig:
    db_path: Path
    socket_path: Path
    uid_profiles: Mapping[int, frozenset[str]]
    socket_mode: int = 0o660
    request_timeout_seconds: float = 10.0

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "BrokerConfig":
        expected = {
            "schema_version",
            "db_path",
            "socket_path",
            "uid_profiles",
            "socket_mode",
            "request_timeout_seconds",
        }
        if set(value) != expected or value.get("schema_version") != _PROTOCOL_VERSION:
            raise BrokerProtocolError("broker config does not match schema 1.0.0")
        raw_profiles = value.get("uid_profiles")
        if not isinstance(raw_profiles, dict) or not raw_profiles:
            raise BrokerProtocolError("uid_profiles must be a non-empty object")
        uid_profiles: dict[int, frozenset[str]] = {}
        for raw_uid, raw_names in raw_profiles.items():
            try:
                uid = int(raw_uid)
            except (TypeError, ValueError) as exc:
                raise BrokerProtocolError("uid_profiles contains an invalid UID") from exc
            if uid < 0 or not isinstance(raw_names, list) or not raw_names:
                raise BrokerProtocolError("uid_profiles contains an invalid profile list")
            names = frozenset(name for name in raw_names if isinstance(name, str) and name)
            if len(names) != len(raw_names):
                raise BrokerProtocolError("uid_profiles contains an invalid profile name")
            uid_profiles[uid] = names
        mode_raw = value.get("socket_mode")
        try:
            mode = int(str(mode_raw), 8)
        except (TypeError, ValueError) as exc:
            raise BrokerProtocolError("socket_mode must be an octal string") from exc
        timeout = value.get("request_timeout_seconds")
        if not isinstance(timeout, (int, float)) or timeout <= 0 or timeout > 60:
            raise BrokerProtocolError("request timeout is outside 0..60 seconds")
        db_path = Path(str(value.get("db_path", ""))).expanduser().resolve()
        socket_path = Path(str(value.get("socket_path", ""))).expanduser().resolve()
        if not db_path.is_absolute() or not socket_path.is_absolute():
            raise BrokerProtocolError("broker paths must be absolute")
        return cls(
            db_path=db_path,
            socket_path=socket_path,
            uid_profiles=uid_profiles,
            socket_mode=mode,
            request_timeout_seconds=float(timeout),
        )

    @classmethod
    def load(cls, path: Path) -> "BrokerConfig":
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise BrokerProtocolError("broker config is unreadable or malformed") from exc
        if not isinstance(value, dict):
            raise BrokerProtocolError("broker config must be an object")
        return cls.from_json(value)


def peer_credentials(conn: socket.socket) -> PeerCredentials:
    """Return Linux peer credentials or fail closed on unsupported platforms."""

    if not hasattr(socket, "SO_PEERCRED"):
        raise BrokerProtocolError("SO_PEERCRED is unavailable")
    raw = conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
    pid, uid, gid = struct.unpack("3i", raw)
    return PeerCredentials(pid=pid, uid=uid, gid=gid)


def _read_request(conn: socket.socket) -> dict[str, Any]:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = conn.recv(min(65536, _MAX_REQUEST_BYTES + 1 - total))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        if total > _MAX_REQUEST_BYTES:
            raise BrokerProtocolError("request exceeds the maximum size")
        if b"\n" in chunk:
            break
    raw = b"".join(chunks)
    line, separator, trailing = raw.partition(b"\n")
    if not separator or trailing:
        raise BrokerProtocolError("request must be exactly one newline-terminated JSON object")
    try:
        value = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise BrokerProtocolError("request JSON is malformed") from exc
    if not isinstance(value, dict) or set(value) != _REQUEST_FIELDS:
        raise BrokerProtocolError("request fields do not match protocol 1.0.0")
    return value


def _validate_request(
    request: Mapping[str, Any],
    peer: PeerCredentials,
    config: BrokerConfig,
) -> None:
    if request.get("version") != _PROTOCOL_VERSION:
        raise BrokerProtocolError("unsupported protocol version")
    if request.get("operation") != "complete":
        raise BrokerProtocolError("unsupported broker operation")
    for name in ("request_id", "profile", "task_id", "result"):
        if not isinstance(request.get(name), str) or not request[name]:
            raise BrokerProtocolError(f"{name} is required")
    if not isinstance(request.get("run_id"), int) or int(request["run_id"]) <= 0:
        raise BrokerProtocolError("run_id must be a positive integer")
    if request.get("summary") is not None and not isinstance(request["summary"], str):
        raise BrokerProtocolError("summary must be a string or null")
    metadata = request.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        raise BrokerProtocolError("metadata must be an object or null")
    if isinstance(metadata, dict) and "artifacts" in metadata:
        # Never let a lower-privilege caller turn the broker UID into a host-file
        # copy oracle. Artifact ingestion needs a separately owned staging API.
        raise BrokerProtocolError("artifact promotion is not supported by this broker")
    created_cards = request.get("created_cards")
    if created_cards is not None and (
        not isinstance(created_cards, list)
        or any(not isinstance(item, str) or not item for item in created_cards)
    ):
        raise BrokerProtocolError("created_cards must be a list of task ids or null")
    allowed = config.uid_profiles.get(peer.uid, frozenset())
    if request["profile"] not in allowed:
        raise BrokerProtocolError("peer UID is not authorized for the requested profile")


def _response(request_id: Optional[str], **payload: Any) -> bytes:
    value = {
        "version": _PROTOCOL_VERSION,
        "request_id": request_id,
        **payload,
    }
    return (guard.canonical_json(value) + "\n").encode("utf-8")


def handle_connection(conn: socket.socket, config: BrokerConfig) -> None:
    """Handle one authenticated request; intended for tests and serve_forever."""

    request_id: Optional[str] = None
    try:
        conn.settimeout(config.request_timeout_seconds)
        peer = peer_credentials(conn)
        request = _read_request(conn)
        raw_id = request.get("request_id")
        request_id = raw_id if isinstance(raw_id, str) else None
        _validate_request(request, peer, config)
        context = guard.CompletionContext(
            caller_profile=str(request["profile"]),
            native_task_id=str(request["task_id"]),
            native_run_id=int(request["run_id"]),
            source="broker",
            peer_uid=peer.uid,
        )
        with kanban_db.connect(db_path=config.db_path) as db:
            completed = kanban_db.complete_task(
                db,
                str(request["task_id"]),
                result=str(request["result"]),
                summary=request["summary"],
                metadata=request["metadata"],
                created_cards=request["created_cards"],
                expected_run_id=int(request["run_id"]),
                completion_context=context,
            )
            receipt = db.execute(
                "SELECT receipt_sha256 FROM completion_governance_receipts "
                "WHERE native_task_id = ?",
                (str(request["task_id"]),),
            ).fetchone()
        conn.sendall(
            _response(
                request_id,
                ok=True,
                completed=bool(completed),
                receipt_sha256=receipt["receipt_sha256"] if receipt else None,
            )
        )
    except guard.CompletionGovernanceDenied as exc:
        conn.sendall(_response(request_id, ok=False, code="denied", error=str(exc)))
    except BrokerProtocolError as exc:
        conn.sendall(_response(request_id, ok=False, code="invalid_request", error=str(exc)))
    except Exception:
        # Do not expose DB paths, SQL, or tracebacks to a lower-privilege peer.
        conn.sendall(
            _response(
                request_id,
                ok=False,
                code="internal_error",
                error="broker request failed",
            )
        )


def _prepare_socket_path(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        existing = path.lstat()
    except FileNotFoundError:
        return
    if not stat.S_ISSOCK(existing.st_mode):
        raise BrokerProtocolError("refusing to replace a non-socket path")
    path.unlink()


def serve_forever(config: BrokerConfig) -> None:
    """Serve serial completion requests on a local Unix socket."""

    if not config.db_path.is_file():
        raise BrokerProtocolError("configured Kanban database does not exist")
    _prepare_socket_path(config.socket_path)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(str(config.socket_path))
        os.chmod(config.socket_path, config.socket_mode)
        server.listen(16)
        while True:
            client, _ = server.accept()
            with client:
                handle_connection(client, config)
    finally:
        server.close()


def request_completion(socket_path: Path, request: Mapping[str, Any]) -> dict[str, Any]:
    """Small strict client used by trusted completion adapters."""

    raw = (guard.canonical_json(dict(request)) + "\n").encode("utf-8")
    if len(raw) > _MAX_REQUEST_BYTES:
        raise BrokerProtocolError("request exceeds the maximum size")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(str(socket_path))
        client.sendall(raw)
        response = _read_response(client)
    return response


def _read_response(conn: socket.socket) -> dict[str, Any]:
    raw = bytearray()
    while b"\n" not in raw:
        chunk = conn.recv(65536)
        if not chunk:
            break
        raw.extend(chunk)
        if len(raw) > _MAX_REQUEST_BYTES:
            raise BrokerProtocolError("response exceeds the maximum size")
    line, separator, trailing = bytes(raw).partition(b"\n")
    if not separator or trailing:
        raise BrokerProtocolError("broker response framing is invalid")
    try:
        value = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise BrokerProtocolError("broker response JSON is malformed") from exc
    if not isinstance(value, dict) or value.get("version") != _PROTOCOL_VERSION:
        raise BrokerProtocolError("broker response protocol is invalid")
    return value


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Hermes governed Kanban completion broker")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    config = BrokerConfig.load(args.config)
    serve_forever(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

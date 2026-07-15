"""Token-owning public Discord connector service and durable journal.

The service accepts one strict request per authenticated Unix connection.  It
reciprocally binds the peer to the exact current gateway systemd ``MainPID``;
the credential-free client performs the mirror check for the connector
``MainPID``.  Discord-specific network I/O is injected through the narrow
``DiscordPublicConnectorBackend`` protocol.
"""

from __future__ import annotations

import json
import errno
import os
import re
import socket
import sqlite3
import stat
import struct
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from gateway.discord_connector_protocol import (
    MAX_FRAME_BYTES,
    DiscordConnectorEvent,
    DiscordConnectorHistoryAuthority,
    DiscordConnectorHistoryPage,
    DiscordConnectorKind,
    DiscordConnectorProtocolError,
    DiscordConnectorRequest,
    DiscordConnectorTarget,
    canonical_json_bytes,
    decode_frame,
    parse_request,
    receipt,
    sha256_json,
)
from gateway.discord_edge_service import (
    DiscordEdgeMainPidProvider,
    DiscordEdgePeerCredentials,
    SystemctlDiscordEdgeMainPidProvider,
    linux_discord_edge_peer_credentials,
)

DEFAULT_DISCORD_CONNECTOR_SOCKET = Path("/run/muncho-discord-connector/connector.sock")
DEFAULT_DISCORD_CONNECTOR_UNIT = "muncho-discord-connector.service"
DEFAULT_DISCORD_CONNECTOR_USER = "muncho-discord-connector"
DEFAULT_DISCORD_CONNECTOR_JOURNAL = Path(
    "/var/lib/muncho-discord-connector/connector.sqlite3"
)
DEFAULT_GATEWAY_UNIT = "hermes-cloud-gateway.service"

MAX_RESPONSE_BYTES = 128 * 1024
SOCKET_MODE = 0o660
EVENT_LEASE_MS = 15_000
_FRAME_HEADER = struct.Struct("!I")
_SNOWFLAKE_RE = re.compile(r"^[1-9][0-9]{0,24}$")


class DiscordConnectorServiceError(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class DiscordConnectorHistoryReaderPeer:
    """One canary-only peer allowed to perform an exact user history read."""

    service_unit: str
    expected_uid: int
    requester_user_id: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.service_unit, str)
            or re.fullmatch(r"[a-z0-9][a-z0-9@_.-]{0,126}\.service", self.service_unit)
            is None
            or isinstance(self.expected_uid, bool)
            or not isinstance(self.expected_uid, int)
            or self.expected_uid < 1
            or not isinstance(self.requester_user_id, str)
            or _SNOWFLAKE_RE.fullmatch(self.requester_user_id) is None
        ):
            raise ValueError("connector history-reader peer is invalid")

    @property
    def authority(self) -> DiscordConnectorHistoryAuthority:
        return DiscordConnectorHistoryAuthority.authenticated_user(
            self.requester_user_id
        )

    def readiness_mapping(self) -> dict[str, Any]:
        return {
            "service_unit": self.service_unit,
            "expected_uid": self.expected_uid,
            "authority_sha256": self.authority.sha256,
            "operation": DiscordConnectorKind.HISTORY_FETCH.value,
        }


@dataclass(frozen=True)
class DiscordConnectorAcceptedMessage:
    message_id: str
    readback_verified: bool

    def __post_init__(self) -> None:
        if (
            not isinstance(self.message_id, str)
            or _SNOWFLAKE_RE.fullmatch(self.message_id) is None
        ):
            raise ValueError("connector backend message id is invalid")
        if type(self.readback_verified) is not bool:
            raise ValueError("connector backend readback marker is invalid")


class DiscordPublicConnectorBackend(Protocol):
    """Fixed Discord I/O surface; no raw URL, method, token, or dispatcher."""

    def prove_public_target(self, channel_id: str) -> DiscordConnectorTarget: ...

    def fetch_guild_history(
        self,
        channel_id: str,
        *,
        limit: int,
        before_message_id: str | None,
        after_message_id: str | None,
        authority: DiscordConnectorHistoryAuthority,
    ) -> DiscordConnectorHistoryPage: ...

    def send_public_message(
        self,
        target: DiscordConnectorTarget,
        content: str,
        *,
        reply_to_message_id: str | None,
        deadline_unix_ms: int,
    ) -> DiscordConnectorAcceptedMessage: ...


def _receipt_result(
    *,
    target: DiscordConnectorTarget,
    content_sha256: str,
    idempotency_key: str,
    message_id: str | None,
    readback_verified: bool,
) -> dict[str, Any]:
    return {
        "target": target.to_mapping(),
        "content_sha256": content_sha256,
        "idempotency_key": idempotency_key,
        "message_id": message_id,
        "readback_verified": readback_verified,
    }


class DurableDiscordConnectorJournal:
    """SQLite first-wins event/send journal with explicit bootstrap."""

    _SCHEMA = "discord-public-connector-journal.v1"

    def __init__(self, path: str | os.PathLike[str], *, busy_timeout_ms: int = 5_000):
        raw = Path(path)
        normalized = Path(os.path.normpath(os.fspath(raw)))
        if not raw.is_absolute() or normalized != raw:
            raise ValueError("connector journal path must be absolute and normalized")
        if not 1 <= busy_timeout_ms <= 30_000:
            raise ValueError("connector journal busy timeout is invalid")
        self.path = raw
        self.busy_timeout_ms = busy_timeout_ms
        self._lock = threading.Lock()
        self._validate_schema()

    @classmethod
    def bootstrap(
        cls, path: str | os.PathLike[str], *, busy_timeout_ms: int = 5_000
    ) -> "DurableDiscordConnectorJournal":
        target = Path(path)
        if target.exists() or target.is_symlink():
            raise FileExistsError("refusing to replace connector journal")
        if not target.parent.is_dir():
            raise ValueError("connector journal parent must already exist")
        conn = sqlite3.connect(target)
        try:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA synchronous=FULL;
                CREATE TABLE connector_meta_v1 (
                    schema_version TEXT PRIMARY KEY
                ) STRICT;
                INSERT INTO connector_meta_v1(schema_version)
                VALUES ('discord-public-connector-journal.v1');
                CREATE TABLE connector_events_v1 (
                    event_id TEXT PRIMARY KEY,
                    event_sha256 TEXT NOT NULL,
                    event_json TEXT NOT NULL,
                    state TEXT NOT NULL CHECK(state IN ('pending','delivering','acked')),
                    delivery_id TEXT,
                    lease_until_unix_ms INTEGER,
                    offered_at_unix_ms INTEGER NOT NULL,
                    acked_at_unix_ms INTEGER
                ) STRICT;
                CREATE TABLE connector_sends_v1 (
                    idempotency_key TEXT PRIMARY KEY,
                    request_sha256 TEXT NOT NULL,
                    state TEXT NOT NULL CHECK(
                        state IN ('prepared','dispatching','verified','blocked','uncertain')
                    ),
                    result_json TEXT,
                    updated_at_unix_ms INTEGER NOT NULL
                ) STRICT;
                """
            )
            conn.commit()
        finally:
            conn.close()
        return cls(target, busy_timeout_ms=busy_timeout_ms)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.path,
            timeout=self.busy_timeout_ms / 1000,
            isolation_level=None,
        )
        conn.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _validate_schema(self) -> None:
        if not self.path.is_file() or self.path.is_symlink():
            raise ValueError("connector journal must be an existing regular file")
        with self._connect() as conn:
            row = conn.execute(
                "SELECT schema_version FROM connector_meta_v1"
            ).fetchone()
            if row != (self._SCHEMA,):
                raise ValueError("connector journal schema is invalid")
            expected = {
                "connector_meta_v1",
                "connector_events_v1",
                "connector_sends_v1",
            }
            actual = {
                str(row[0])
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
                if not str(row[0]).startswith("sqlite_")
            }
            if actual != expected:
                raise ValueError("connector journal table set is invalid")

    def offer_event(self, event: DiscordConnectorEvent) -> bool:
        raw = canonical_json_bytes(event.to_mapping()).decode("utf-8")
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT event_sha256 FROM connector_events_v1 WHERE event_id=?",
                (event.event_id,),
            ).fetchone()
            if row is not None:
                if row[0] != event.sha256:
                    conn.rollback()
                    raise DiscordConnectorServiceError("event_idempotency_conflict")
                conn.commit()
                return False
            conn.execute(
                """
                INSERT INTO connector_events_v1(
                    event_id,event_sha256,event_json,state,delivery_id,
                    lease_until_unix_ms,offered_at_unix_ms,acked_at_unix_ms
                ) VALUES (?,?,?,'pending',NULL,NULL,?,NULL)
                """,
                (event.event_id, event.sha256, raw, int(time.time() * 1000)),
            )
            conn.commit()
            return True

    def next_event(self, *, now_unix_ms: int) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT event_id,event_sha256,event_json
                  FROM connector_events_v1
                 WHERE state='pending'
                    OR (state='delivering' AND lease_until_unix_ms<=?)
                 ORDER BY offered_at_unix_ms,event_id
                 LIMIT 1
                """,
                (now_unix_ms,),
            ).fetchone()
            if row is None:
                conn.commit()
                return None
            delivery_id = str(uuid.uuid4())
            conn.execute(
                """
                UPDATE connector_events_v1
                   SET state='delivering',delivery_id=?,lease_until_unix_ms=?
                 WHERE event_id=?
                """,
                (delivery_id, now_unix_ms + EVENT_LEASE_MS, row[0]),
            )
            conn.commit()
            value = json.loads(row[2])
            event = DiscordConnectorEvent.from_mapping(value)
            if event.event_id != row[0] or event.sha256 != row[1]:
                raise DiscordConnectorServiceError("event_journal_binding_invalid")
            return {
                "delivery_id": delivery_id,
                "event_id": event.event_id,
                "event_sha256": event.sha256,
                "event": event.to_mapping(),
            }

    def ack_event(
        self,
        *,
        delivery_id: str,
        event_id: str,
        event_sha256: str,
        now_unix_ms: int,
    ) -> bool:
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT event_sha256,state,delivery_id
                  FROM connector_events_v1 WHERE event_id=?
                """,
                (event_id,),
            ).fetchone()
            if row is None or row[0] != event_sha256:
                conn.rollback()
                raise DiscordConnectorServiceError("event_ack_binding_invalid")
            if row[1] == "acked":
                if row[2] != delivery_id:
                    conn.rollback()
                    raise DiscordConnectorServiceError("event_ack_binding_invalid")
                conn.commit()
                return True
            if row[1] != "delivering" or row[2] != delivery_id:
                conn.rollback()
                raise DiscordConnectorServiceError("event_ack_binding_invalid")
            conn.execute(
                """
                UPDATE connector_events_v1
                   SET state='acked',lease_until_unix_ms=NULL,acked_at_unix_ms=?
                 WHERE event_id=?
                """,
                (now_unix_ms, event_id),
            )
            conn.commit()
            return False

    def prepare_send(
        self, *, idempotency_key: str, request_sha256: str, now_unix_ms: int
    ) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT request_sha256,state,result_json
                  FROM connector_sends_v1 WHERE idempotency_key=?
                """,
                (idempotency_key,),
            ).fetchone()
            if row is not None:
                if row[0] != request_sha256:
                    conn.rollback()
                    raise DiscordConnectorServiceError("send_idempotency_conflict")
                conn.commit()
                return {
                    "state": row[1],
                    "result": json.loads(row[2]) if row[2] else {},
                }
            conn.execute(
                """
                INSERT INTO connector_sends_v1(
                    idempotency_key,request_sha256,state,result_json,updated_at_unix_ms
                ) VALUES (?,?,'prepared',NULL,?)
                """,
                (idempotency_key, request_sha256, now_unix_ms),
            )
            conn.commit()
            return None

    def set_send_state(
        self,
        *,
        idempotency_key: str,
        request_sha256: str,
        from_state: str,
        to_state: str,
        result: Mapping[str, Any] | None,
        now_unix_ms: int,
    ) -> None:
        raw = (
            canonical_json_bytes(result).decode("utf-8") if result is not None else None
        )
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute(
                """
                UPDATE connector_sends_v1
                   SET state=?,result_json=?,updated_at_unix_ms=?
                 WHERE idempotency_key=? AND request_sha256=? AND state=?
                """,
                (
                    to_state,
                    raw,
                    now_unix_ms,
                    idempotency_key,
                    request_sha256,
                    from_state,
                ),
            )
            if cursor.rowcount != 1:
                conn.rollback()
                raise DiscordConnectorServiceError("send_state_conflict")
            conn.commit()

    def cleanup_snapshot(self) -> dict[str, Any]:
        """Return exact state counts and fail closed on unresolved dispatches.

        The snapshot contains no message text, target, or request payload.  It
        exists solely so a bounded canary lifecycle can prove that stopping
        the connector will not silently abandon a possibly-sent operation.
        """

        with self._lock, self._connect() as conn:
            event_rows = conn.execute(
                """
                SELECT state, COUNT(*)
                  FROM connector_events_v1
                 GROUP BY state
                 ORDER BY state
                """
            ).fetchall()
            send_rows = conn.execute(
                """
                SELECT state, COUNT(*)
                  FROM connector_sends_v1
                 GROUP BY state
                 ORDER BY state
                """
            ).fetchall()
        events = {str(state): int(count) for state, count in event_rows}
        sends = {str(state): int(count) for state, count in send_rows}
        unresolved = sum(
            sends.get(state, 0) for state in ("prepared", "dispatching", "uncertain")
        )
        unacked_events = sum(
            events.get(state, 0) for state in ("pending", "delivering")
        )
        return {
            "schema": "discord-public-connector-cleanup-snapshot.v1",
            "event_state_counts": events,
            "send_state_counts": sends,
            "unresolved_dispatch_count": unresolved,
            "unacked_event_count": unacked_events,
            "safe_to_retire": unresolved == 0 and unacked_events == 0,
        }


class DiscordConnectorRuntime:
    """Mechanical request dispatcher over the fixed protocol enum."""

    def __init__(
        self,
        *,
        backend: DiscordPublicConnectorBackend,
        journal: DurableDiscordConnectorJournal,
        event_wait_sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if not callable(getattr(backend, "prove_public_target", None)) or not callable(
            getattr(backend, "send_public_message", None)
        ) or not callable(getattr(backend, "fetch_guild_history", None)):
            raise TypeError("connector backend is invalid")
        if not isinstance(journal, DurableDiscordConnectorJournal):
            raise TypeError("connector journal is invalid")
        self.backend = backend
        self.journal = journal
        self._sleep = event_wait_sleeper

    @staticmethod
    def descriptor() -> dict[str, Any]:
        return {
            "contract_version": 1,
            "platform": "discord",
            "label": "Discord (privileged connector)",
            "max_message_length": 2_000,
            "supports_draft_streaming": False,
            "supports_edit": False,
            "supports_threads": True,
            "markdown_dialect": "discord",
            "len_unit": "chars",
            "emoji": "🎮",
            "platform_hint": "",
            "pii_safe": False,
        }

    def handle(self, request: DiscordConnectorRequest) -> dict[str, Any]:
        if request.kind is DiscordConnectorKind.HELLO:
            return receipt(
                request=request,
                status="ok",
                result={"descriptor": self.descriptor()},
            )
        if request.kind is DiscordConnectorKind.EVENT_NEXT:
            deadline = time.monotonic() + request.payload["wait_ms"] / 1000
            item = self.journal.next_event(now_unix_ms=int(time.time() * 1000))
            while item is None and time.monotonic() < deadline:
                self._sleep(min(0.05, max(0.0, deadline - time.monotonic())))
                item = self.journal.next_event(now_unix_ms=int(time.time() * 1000))
            return receipt(
                request=request,
                status="ok" if item is not None else "idle",
                result=item or {},
            )
        if request.kind is DiscordConnectorKind.EVENT_ACK:
            replayed = self.journal.ack_event(
                **request.payload,
                now_unix_ms=int(time.time() * 1000),
            )
            return receipt(
                request=request,
                status="ok",
                result={
                    "event_id": request.payload["event_id"],
                    "event_sha256": request.payload["event_sha256"],
                    "acked": True,
                },
                replayed=replayed,
            )
        if request.kind is DiscordConnectorKind.TARGET_GET:
            try:
                target = self.backend.prove_public_target(
                    str(request.payload["channel_id"])
                )
            except Exception:
                return receipt(request=request, status="blocked", result={})
            return receipt(
                request=request,
                status="ok",
                result={"target": target.to_mapping()},
            )
        if request.kind is DiscordConnectorKind.HISTORY_FETCH:
            return self._fetch_history(request)
        return self._send(request)

    def _fetch_history(self, request: DiscordConnectorRequest) -> dict[str, Any]:
        payload = dict(request.payload)
        authority = DiscordConnectorHistoryAuthority.from_mapping(
            payload["authority"]
        )
        try:
            page = self.backend.fetch_guild_history(
                str(payload["channel_id"]),
                limit=int(payload["limit"]),
                before_message_id=payload["before_message_id"],
                after_message_id=payload["after_message_id"],
                authority=authority,
            )
            if not isinstance(page, DiscordConnectorHistoryPage):
                raise DiscordConnectorServiceError("invalid_backend_history_page")
            # Reparse the complete page so a forged/deserialized dataclass cannot
            # cross the connector boundary with invalid public-target or content
            # fields.
            validated = DiscordConnectorHistoryPage.from_mapping(page.to_mapping())
            if (
                validated.target.channel_id != payload["channel_id"]
                or validated.limit != payload["limit"]
                or validated.before_message_id != payload["before_message_id"]
                or validated.after_message_id != payload["after_message_id"]
            ):
                raise DiscordConnectorServiceError("history_query_binding_mismatch")
        except Exception:
            return receipt(request=request, status="blocked", result={})
        return receipt(
            request=request,
            status="ok",
            result={
                "page": validated.to_mapping(),
                "page_sha256": validated.sha256,
                # Bind the receipt to the internal requester/job authority
                # without returning the raw non-secret identity to the model.
                "authority_sha256": authority.sha256,
            },
        )

    def _send(self, request: DiscordConnectorRequest) -> dict[str, Any]:
        payload = dict(request.payload)
        target = DiscordConnectorTarget.from_mapping(payload["target"])
        request_sha256 = sha256_json(payload)
        idempotency_key = str(payload["idempotency_key"])
        content_sha256 = sha256_json({"content": payload["content"]})
        existing = self.journal.prepare_send(
            idempotency_key=idempotency_key,
            request_sha256=request_sha256,
            now_unix_ms=int(time.time() * 1000),
        )
        if existing is not None:
            state = str(existing["state"])
            status = {
                "verified": "ok",
                "blocked": "blocked",
                "dispatching": "dispatch_uncertain",
                "uncertain": "dispatch_uncertain",
                "prepared": "dispatch_uncertain",
            }[state]
            return receipt(
                request=request,
                status=status,
                result=dict(existing["result"]),
                replayed=True,
            )

        try:
            proven = self.backend.prove_public_target(target.channel_id)
            if proven != target:
                raise DiscordConnectorServiceError("public_target_binding_mismatch")
        except Exception:
            result = _receipt_result(
                target=target,
                content_sha256=content_sha256,
                idempotency_key=idempotency_key,
                message_id=None,
                readback_verified=False,
            )
            self.journal.set_send_state(
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                from_state="prepared",
                to_state="blocked",
                result=result,
                now_unix_ms=int(time.time() * 1000),
            )
            return receipt(request=request, status="blocked", result=result)

        self.journal.set_send_state(
            idempotency_key=idempotency_key,
            request_sha256=request_sha256,
            from_state="prepared",
            to_state="dispatching",
            result=None,
            now_unix_ms=int(time.time() * 1000),
        )
        try:
            accepted = self.backend.send_public_message(
                target,
                str(payload["content"]),
                reply_to_message_id=payload["reply_to_message_id"],
                deadline_unix_ms=int(payload["deadline_unix_ms"]),
            )
            if not isinstance(accepted, DiscordConnectorAcceptedMessage):
                raise DiscordConnectorServiceError("invalid_backend_receipt")
            # Revalidate even a forged/deserialized dataclass instance before
            # any verified state can enter the durable journal.
            if (
                not isinstance(accepted.message_id, str)
                or _SNOWFLAKE_RE.fullmatch(accepted.message_id) is None
                or type(accepted.readback_verified) is not bool
            ):
                raise DiscordConnectorServiceError("invalid_backend_receipt")
            result = _receipt_result(
                target=target,
                content_sha256=content_sha256,
                idempotency_key=idempotency_key,
                message_id=accepted.message_id,
                readback_verified=accepted.readback_verified,
            )
            final_state = "verified" if accepted.readback_verified else "uncertain"
            self.journal.set_send_state(
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                from_state="dispatching",
                to_state=final_state,
                result=result,
                now_unix_ms=int(time.time() * 1000),
            )
            return receipt(
                request=request,
                status="ok" if accepted.readback_verified else "dispatch_uncertain",
                result=result,
            )
        except Exception:
            result = _receipt_result(
                target=target,
                content_sha256=content_sha256,
                idempotency_key=idempotency_key,
                message_id=None,
                readback_verified=False,
            )
            self.journal.set_send_state(
                idempotency_key=idempotency_key,
                request_sha256=request_sha256,
                from_state="dispatching",
                to_state="uncertain",
                result=result,
                now_unix_ms=int(time.time() * 1000),
            )
            return receipt(
                request=request,
                status="dispatch_uncertain",
                result=result,
            )


def _recv_exact(conn: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = conn.recv(remaining)
        if not chunk:
            raise OSError("connector frame closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _recv_frame(conn: socket.socket) -> bytes:
    (size,) = _FRAME_HEADER.unpack(_recv_exact(conn, _FRAME_HEADER.size))
    if size == 0 or size > MAX_FRAME_BYTES:
        raise DiscordConnectorProtocolError("invalid_frame_size")
    return _recv_exact(conn, size)


class DiscordConnectorUnixServer:
    """One-frame AF_UNIX server with exact, operation-scoped peer identity."""

    def __init__(
        self,
        socket_path: str | os.PathLike[str],
        *,
        runtime: DiscordConnectorRuntime,
        expected_gateway_uid: int,
        gateway_unit: str = DEFAULT_GATEWAY_UNIT,
        main_pid_provider: DiscordEdgeMainPidProvider | None = None,
        peer_getter: Callable[[socket.socket], DiscordEdgePeerCredentials] = (
            linux_discord_edge_peer_credentials
        ),
        history_reader_peer: DiscordConnectorHistoryReaderPeer | None = None,
        connection_timeout_seconds: float = 10,
    ) -> None:
        if not isinstance(runtime, DiscordConnectorRuntime):
            raise TypeError("connector runtime is invalid")
        if isinstance(expected_gateway_uid, bool) or expected_gateway_uid < 1:
            raise ValueError("connector gateway UID is invalid")
        if history_reader_peer is not None and not isinstance(
            history_reader_peer, DiscordConnectorHistoryReaderPeer
        ):
            raise TypeError("connector history-reader peer is invalid")
        if (
            history_reader_peer is not None
            and history_reader_peer.expected_uid == expected_gateway_uid
        ):
            raise ValueError("connector peer identities must be isolated")
        if not 0 < connection_timeout_seconds <= 30:
            raise ValueError("connector timeout is invalid")
        self.socket_path = Path(socket_path)
        if not self.socket_path.is_absolute() or self.socket_path != Path(
            os.path.normpath(os.fspath(self.socket_path))
        ):
            raise ValueError("connector socket path is invalid")
        parent = self.socket_path.parent.resolve(strict=True)
        parent_stat = os.lstat(parent)
        geteuid = getattr(os, "geteuid", None)
        if (
            parent != self.socket_path.parent
            or not stat.S_ISDIR(parent_stat.st_mode)
            or not callable(geteuid)
            or parent_stat.st_uid != geteuid()
            or parent_stat.st_mode & 0o022
        ):
            raise PermissionError("connector socket parent is untrusted")
        self._parent_identity = (parent_stat.st_dev, parent_stat.st_ino)
        self.runtime = runtime
        self.expected_gateway_uid = expected_gateway_uid
        self.gateway_unit = gateway_unit
        self.main_pid_provider = (
            main_pid_provider or SystemctlDiscordEdgeMainPidProvider()
        )
        self.peer_getter = peer_getter
        self.history_reader_peer = history_reader_peer
        self.connection_timeout_seconds = connection_timeout_seconds
        self._listener: socket.socket | None = None
        self._socket_identity: tuple[int, int] | None = None
        self._stop = threading.Event()
        self._handler_lock = threading.Lock()
        self._handlers: set[threading.Thread] = set()

    def _authorized(self, peer: object) -> bool:
        if (
            not isinstance(peer, DiscordEdgePeerCredentials)
            or peer.uid != self.expected_gateway_uid
            or peer.pid <= 1
        ):
            return False
        try:
            main_pid = self.main_pid_provider.main_pid(self.gateway_unit)
        except Exception:
            return False
        return (
            isinstance(main_pid, int)
            and not isinstance(main_pid, bool)
            and peer.pid == main_pid
        )

    def _history_reader_authorized(self, peer: object) -> bool:
        authority = self.history_reader_peer
        if (
            authority is None
            or not isinstance(peer, DiscordEdgePeerCredentials)
            or peer.uid != authority.expected_uid
            or peer.pid <= 1
        ):
            return False
        try:
            main_pid = self.main_pid_provider.main_pid(authority.service_unit)
        except Exception:
            return False
        return (
            isinstance(main_pid, int)
            and not isinstance(main_pid, bool)
            and peer.pid == main_pid
        )

    def _peer_scope(self, peer: object) -> str | None:
        if self._authorized(peer):
            return "gateway"
        if self._history_reader_authorized(peer):
            return "history_reader"
        return None

    def history_reader_identity(self) -> dict[str, Any] | None:
        authority = self.history_reader_peer
        return None if authority is None else authority.readiness_mapping()

    def _history_reader_request_allowed(
        self,
        request: DiscordConnectorRequest,
    ) -> bool:
        authority = self.history_reader_peer
        if authority is None or request.kind is not DiscordConnectorKind.HISTORY_FETCH:
            return False
        try:
            requested = DiscordConnectorHistoryAuthority.from_mapping(
                request.payload["authority"]
            )
        except (KeyError, DiscordConnectorProtocolError):
            return False
        return requested == authority.authority

    def start(self) -> None:
        if self._listener is not None:
            return
        parent_stat = os.lstat(self.socket_path.parent)
        if (parent_stat.st_dev, parent_stat.st_ino) != self._parent_identity:
            raise PermissionError("connector socket parent identity changed")
        if self.socket_path.exists() or self.socket_path.is_symlink():
            raise FileExistsError("refusing to replace connector socket")
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.set_inheritable(False)
        listener.bind(str(self.socket_path))
        os.chmod(self.socket_path, SOCKET_MODE)
        socket_stat = self.socket_path.lstat()
        if (
            not stat.S_ISSOCK(socket_stat.st_mode)
            or stat.S_IMODE(socket_stat.st_mode) != SOCKET_MODE
        ):
            listener.close()
            raise PermissionError("connector socket identity is invalid")
        self._socket_identity = (socket_stat.st_dev, socket_stat.st_ino)
        listener.listen(16)
        listener.settimeout(0.2)
        self._listener = listener

    def readiness_identity(self) -> dict[str, Any]:
        """Return the exact non-secret identity of the bound Unix listener."""

        listener = self._listener
        identity = self._socket_identity
        if listener is None or identity is None:
            raise DiscordConnectorServiceError("connector_socket_not_ready")
        parent = os.lstat(self.socket_path.parent)
        current = self.socket_path.lstat()
        try:
            bound_path = listener.getsockname()
            descriptor_open = listener.fileno() >= 0
        except OSError as exc:
            raise DiscordConnectorServiceError("connector_socket_not_ready") from exc
        try:
            accepting = listener.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN)
        except OSError as exc:
            # Darwin does not expose SO_ACCEPTCONN for AF_UNIX.  Production is
            # Linux and must prove the live kernel flag; the portable fallback
            # still requires the exact open bound descriptor created only
            # after this class successfully called listen().
            if sys.platform == "darwin" and exc.errno == errno.ENOPROTOOPT:
                accepting = 1
            else:
                raise DiscordConnectorServiceError(
                    "connector_socket_not_ready"
                ) from exc
        if (
            (parent.st_dev, parent.st_ino) != self._parent_identity
            or not stat.S_ISSOCK(current.st_mode)
            or (current.st_dev, current.st_ino) != identity
            or stat.S_IMODE(current.st_mode) != SOCKET_MODE
            or not descriptor_open
            or accepting != 1
            or bound_path != str(self.socket_path)
        ):
            raise DiscordConnectorServiceError("connector_socket_identity_changed")
        return {
            "socket_path": str(self.socket_path),
            "socket_device": current.st_dev,
            "socket_inode": current.st_ino,
            "socket_uid": current.st_uid,
            "socket_gid": current.st_gid,
            "socket_mode": f"{stat.S_IMODE(current.st_mode):04o}",
            "listening": True,
        }

    def serve_forever(self) -> None:
        self.start()
        assert self._listener is not None
        try:
            while not self._stop.is_set():
                try:
                    conn, _ = self._listener.accept()
                except socket.timeout:
                    continue
                except OSError:
                    if self._stop.is_set():
                        break
                    raise
                thread = threading.Thread(
                    target=self._handle_connection_tracked,
                    args=(conn,),
                    daemon=False,
                    name="discord-connector-request",
                )
                with self._handler_lock:
                    self._handlers.add(thread)
                thread.start()
        finally:
            self.shutdown()

    def _handle_connection_tracked(self, conn: socket.socket) -> None:
        try:
            self._handle_connection(conn)
        finally:
            current = threading.current_thread()
            with self._handler_lock:
                self._handlers.discard(current)

    def _handle_connection(self, conn: socket.socket) -> None:
        try:
            conn.settimeout(self.connection_timeout_seconds)
            peer = self.peer_getter(conn)
            scope = self._peer_scope(peer)
            if scope is None:
                return
            request = parse_request(decode_frame(_recv_frame(conn)))
            if (
                scope == "history_reader"
                and not self._history_reader_request_allowed(request)
            ):
                return
            if self._peer_scope(peer) != scope:
                return
            response = self.runtime.handle(request)
            body = canonical_json_bytes(response)
            if not body or len(body) > MAX_RESPONSE_BYTES:
                return
            if self._peer_scope(peer) != scope:
                return
            conn.sendall(_FRAME_HEADER.pack(len(body)) + body)
        except (
            OSError,
            TypeError,
            ValueError,
            DiscordConnectorProtocolError,
            DiscordConnectorServiceError,
        ):
            return
        finally:
            conn.close()

    def shutdown(self) -> None:
        self._stop.set()
        listener = self._listener
        self._listener = None
        if listener is not None:
            listener.close()
        deadline = time.monotonic() + min(
            45.0,
            self.connection_timeout_seconds + 15.0,
        )
        while True:
            with self._handler_lock:
                handlers = [
                    item
                    for item in self._handlers
                    if item is not threading.current_thread() and item.is_alive()
                ]
            if not handlers:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise DiscordConnectorServiceError("connector_request_drain_timeout")
            handlers[0].join(timeout=min(0.2, remaining))
        identity = self._socket_identity
        self._socket_identity = None
        if identity is None:
            return
        try:
            current = self.socket_path.lstat()
            if (
                stat.S_ISSOCK(current.st_mode)
                and (
                    current.st_dev,
                    current.st_ino,
                )
                == identity
            ):
                self.socket_path.unlink()
        except FileNotFoundError:
            pass


__all__ = [
    "DEFAULT_DISCORD_CONNECTOR_JOURNAL",
    "DEFAULT_DISCORD_CONNECTOR_SOCKET",
    "DEFAULT_DISCORD_CONNECTOR_UNIT",
    "DEFAULT_DISCORD_CONNECTOR_USER",
    "DiscordConnectorAcceptedMessage",
    "DiscordConnectorHistoryReaderPeer",
    "DiscordConnectorRuntime",
    "DiscordConnectorServiceError",
    "DiscordConnectorUnixServer",
    "DiscordPublicConnectorBackend",
    "DurableDiscordConnectorJournal",
]

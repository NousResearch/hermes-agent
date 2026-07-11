"""Authenticated per-board single-writer service for Kanban mutations.

The protocol is length-delimited JSON over a local Unix socket. Only an
explicit tagged value domain is accepted after the peer proves possession of
the board-local token; neither requests nor responses deserialize executable
objects.
"""

from __future__ import annotations

import argparse
import contextlib
import contextvars
import dataclasses
import functools
import hashlib
import hmac
import inspect
import json
import math
import os
import secrets
import signal
import socket
import sqlite3
import stat
import struct
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any, Iterator

from hermes_cli import kanban_safety


DEFAULT_TIMEOUT_SECONDS = 5.0
PROTOCOL_VERSION = 2
MUTATION_SOURCES = frozenset({"cli", "tool", "gateway", "worker", "dashboard"})
_ACTIVATION_MARKER = "kanban-writer-required-v1"
_SERVER_PEER_TIMEOUT_SECONDS = 0.25
_MAX_MESSAGE_BYTES = 16 * 1024 * 1024
_WRITER_DB: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "kanban_writer_db", default=None
)
_PRIVILEGED_DB: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "kanban_privileged_db", default=None
)
_MUTATION_SOURCE: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "kanban_mutation_source", default=None
)


class KanbanWriterError(RuntimeError):
    """Base class for writer boundary failures."""


class WriterOwnershipError(KanbanWriterError):
    """A non-owner attempted a direct SQLite write transaction."""


class WriterUnavailableError(KanbanWriterError):
    """The board writer cannot be reached within the bounded timeout."""


class WriterAuthenticationError(KanbanWriterError):
    """The peer did not possess the board-local authentication token."""


class WriterProtocolError(KanbanWriterError):
    """The local IPC request or response violated the protocol."""


RUNTIME_MUTATIONS = {
    "create_task", "assign_task", "assign_default_assignee", "link_tasks", "unlink_tasks", "add_comment",
    "add_attachment", "delete_attachment", "recompute_ready", "claim_task",
    "claim_review_task", "heartbeat_claim", "complete_task", "edit_completed_task_result", "block_task", "promote_task",
    "unblock_task", "specify_triage_task", "decompose_triage_task", "archive_task",
    "delete_archived_task", "delete_task", "edit_task_fields", "set_task_status_direct",
    "set_workspace_path", "set_branch_name",
    "schedule_task", "heartbeat_worker", "detect_crashed_workers",
    "_extend_stale_claim", "_defer_reclaim_for_live_worker", "_finalize_stale_reclaim",
    "_finalize_manual_reclaim", "_finalize_max_runtime", "_finalize_stale_running",
    "_record_spawn_failure", "_set_worker_pid",
    "record_respawn_guarded", "emit_scratch_tip_event", "add_notify_sub", "remove_notify_sub",
    "claim_unseen_events_for_sub", "advance_notify_cursor", "rewind_notify_cursor",
    "gc_events",
}


def _resolved(path: Path) -> str:
    return str(Path(path).expanduser().resolve())


def _infer_board(db_path: Path) -> str:
    path = Path(db_path).expanduser().resolve()
    if path.name == "kanban.db" and path.parent.parent.name == "boards":
        return path.parent.name
    explicit = os.environ.get("HERMES_KANBAN_BOARD", "").strip()
    if explicit:
        return explicit
    return "default"


def _protocol_label(value: str, field: str) -> str:
    if not isinstance(value, str):
        raise WriterProtocolError(f"{field} must be a string")
    value = value.strip()
    if not value or len(value) > 200 or any(ord(char) < 32 for char in value):
        raise WriterProtocolError(
            f"{field} is required, must be <= 200 chars, and cannot contain controls"
        )
    return value


def mutation_source_label(value: str) -> str:
    source = _protocol_label(value, "source")
    if source not in MUTATION_SOURCES:
        allowed = ", ".join(sorted(MUTATION_SOURCES))
        raise WriterProtocolError(f"source must be one of: {allowed}")
    return source


def current_mutation_source() -> str:
    return _MUTATION_SOURCE.get() or "worker"


@contextlib.contextmanager
def mutation_source(source: str) -> Iterator[None]:
    token = _MUTATION_SOURCE.set(mutation_source_label(source))
    try:
        yield
    finally:
        _MUTATION_SOURCE.reset(token)


def default_mutation_source(source: str):
    source = mutation_source_label(source)

    def decorate(function):
        @functools.wraps(function)
        def attributed(*args, **kwargs):
            if _MUTATION_SOURCE.get() is not None:
                return function(*args, **kwargs)
            with mutation_source(source):
                return function(*args, **kwargs)

        return attributed

    return decorate


def is_writer_owner(db_path: Path) -> bool:
    resolved = _resolved(db_path)
    return _WRITER_DB.get() == resolved or _PRIVILEGED_DB.get() == resolved


def is_writer_required(db_path: Path) -> bool:
    """Return true after an explicit, durable single-writer activation."""
    path = writer_activation_path(db_path)
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o077:
        # An unsafe marker must fail closed rather than silently re-enable writes.
        return True
    return True


def route_runtime_mutation(operation: str, function):
    """Route every non-owner physical-DB mutation through local IPC."""
    signature = inspect.signature(function)

    @functools.wraps(function)
    def routed(conn, *args, **kwargs):
        row = next((row for row in conn.execute("PRAGMA database_list") if row[1] == "main"), None)
        db_path = Path(row[2]).expanduser().resolve() if row is not None and row[2] else None
        if (
            db_path is None
            or is_writer_owner(db_path)
            or not is_writer_required(db_path)
        ):
            return function(conn, *args, **kwargs)
        bound = signature.bind(conn, *args, **kwargs)
        arguments = dict(bound.arguments)
        arguments.pop(next(iter(signature.parameters)))
        result = KanbanWriterClient(
            db_path,
            board=getattr(conn, "_kanban_board", None),
            actor_profile=getattr(conn, "_kanban_actor_profile", None),
            source=getattr(conn, "_kanban_source", None),
        ).mutate(operation, arguments)
        if (
            operation == "detect_crashed_workers"
            and isinstance(result, dict)
            and set(result) == {"result", "side_channels"}
        ):
            side_channels = result["side_channels"]
            if isinstance(side_channels, dict):
                for name, value in side_channels.items():
                    if name in {"_last_auto_blocked", "_last_rate_limited"}:
                        setattr(routed, name, value)
            return result["result"]
        return result

    return routed


@contextlib.contextmanager
def writer_owner(db_path: Path) -> Iterator[None]:
    token = _WRITER_DB.set(_resolved(db_path))
    try:
        yield
    finally:
        _WRITER_DB.reset(token)


@contextlib.contextmanager
def privileged_maintenance(db_path: Path) -> Iterator[None]:
    """Narrow bypass for schema bootstrap/migration, never runtime fallback."""
    token = _PRIVILEGED_DB.set(_resolved(db_path))
    try:
        yield
    finally:
        _PRIVILEGED_DB.reset(token)


def writer_socket_path(db_path: Path) -> Path:
    path = Path(db_path).expanduser().resolve()
    digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16]
    endpoint = path.parent / f".kanban-writer-{digest}.sock"
    # sockaddr_un.sun_path is only 108 bytes on Linux (and shorter on some
    # BSDs). Deep worktrees and pytest temp directories can exceed that even
    # though the board path itself is valid. Keep the board-local endpoint
    # when possible, otherwise use a private, deterministic runtime directory.
    if len(os.fsencode(endpoint)) <= 100:
        return endpoint
    uid = os.getuid() if hasattr(os, "getuid") else os.getpid()
    runtime_dir = Path(tempfile.gettempdir()) / f"hermes-kanban-{uid}"
    try:
        runtime_dir.mkdir(mode=0o700, parents=False, exist_ok=True)
        metadata = runtime_dir.lstat()
        if not stat.S_ISDIR(metadata.st_mode):
            raise WriterUnavailableError(
                f"writer runtime path is not a directory: {runtime_dir}"
            )
        if hasattr(os, "getuid") and metadata.st_uid != os.getuid():
            raise WriterUnavailableError(
                f"writer runtime directory has the wrong owner: {runtime_dir}"
            )
        if metadata.st_mode & 0o077:
            os.chmod(runtime_dir, 0o700)
    except OSError as exc:
        raise WriterUnavailableError(
            f"cannot prepare private writer runtime directory {runtime_dir}"
        ) from exc
    return runtime_dir / f"{digest}.sock"


def writer_token_path(db_path: Path) -> Path:
    path = Path(db_path).expanduser().resolve()
    return path.with_name(path.name + ".writer.token")


def writer_activation_path(db_path: Path) -> Path:
    path = Path(db_path).expanduser().resolve()
    return path.with_name(path.name + ".writer.required")


def writer_lock_path(db_path: Path) -> Path:
    path = Path(db_path).expanduser().resolve()
    return path.with_name(path.name + ".writer.lock")


def _encode_value(value: Any) -> Any:
    """Encode the small, explicit value domain allowed on writer IPC."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise WriterProtocolError("non-finite floats are not supported")
        return value
    if isinstance(value, Path):
        return {"__type__": "path", "value": str(value)}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        name = type(value).__name__
        if name not in {"Task", "Attachment", "Event"}:
            raise WriterProtocolError(f"unsupported dataclass on writer IPC: {name}")
        return {
            "__type__": "dataclass",
            "class": name,
            "fields": {
                field.name: _encode_value(getattr(value, field.name))
                for field in dataclasses.fields(value)
            },
        }
    if isinstance(value, tuple):
        return {"__type__": "tuple", "items": [_encode_value(item) for item in value]}
    if isinstance(value, list):
        return [_encode_value(item) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise WriterProtocolError("writer IPC dictionaries require string keys")
        return {key: _encode_value(item) for key, item in value.items()}
    raise WriterProtocolError(
        f"unsupported value on writer IPC: {type(value).__module__}.{type(value).__name__}"
    )


def _decode_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise WriterProtocolError("non-finite floats are not supported")
        return value
    if isinstance(value, list):
        return [_decode_value(item) for item in value]
    if not isinstance(value, dict):
        raise WriterProtocolError("invalid tagged writer IPC value")
    value_type = value.get("__type__")
    if value_type is None:
        if not all(isinstance(key, str) for key in value):
            raise WriterProtocolError("writer IPC dictionaries require string keys")
        return {key: _decode_value(item) for key, item in value.items()}
    if value_type == "path" and set(value) == {"__type__", "value"}:
        return Path(str(value["value"]))
    if value_type == "tuple" and set(value) == {"__type__", "items"}:
        items = value["items"]
        if not isinstance(items, list):
            raise WriterProtocolError("tuple items must be a list")
        return tuple(_decode_value(item) for item in items)
    if value_type == "dataclass" and set(value) == {"__type__", "class", "fields"}:
        from hermes_cli import kanban_db as kb

        classes = {"Task": kb.Task, "Attachment": kb.Attachment, "Event": kb.Event}
        cls = classes.get(str(value["class"]))
        fields = value["fields"]
        if cls is None or not isinstance(fields, dict):
            raise WriterProtocolError("unsupported writer IPC dataclass")
        return cls(**{key: _decode_value(item) for key, item in fields.items()})
    raise WriterProtocolError(f"unsupported tagged writer IPC value: {value_type!r}")


def _json_text(value: Any) -> str:
    try:
        return json.dumps(
            _encode_value(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise WriterProtocolError("writer IPC value is not JSON encodable") from exc


def _json_value(text: str) -> Any:
    try:
        return _decode_value(json.loads(text))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise WriterProtocolError("invalid writer IPC value JSON") from exc


def _send_message(sock: socket.socket, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(encoded) > _MAX_MESSAGE_BYTES:
        raise WriterProtocolError("writer IPC message exceeds size limit")
    sock.sendall(struct.pack("!I", len(encoded)) + encoded)


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise WriterProtocolError("writer IPC peer closed an incomplete message")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _recv_message(sock: socket.socket) -> dict[str, Any]:
    size = struct.unpack("!I", _recv_exact(sock, 4))[0]
    if size <= 0 or size > _MAX_MESSAGE_BYTES:
        raise WriterProtocolError("writer IPC message exceeds size limit")

    def reject_duplicate_keys(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise WriterProtocolError(f"duplicate writer IPC key: {key}")
            value[key] = item
        return value

    def reject_constant(value):
        raise WriterProtocolError(f"invalid writer IPC numeric constant: {value}")

    try:
        value = json.loads(
            _recv_exact(sock, size).decode("utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WriterProtocolError("invalid writer IPC JSON") from exc
    if not isinstance(value, dict):
        raise WriterProtocolError("writer IPC payload must be an object")
    return value


def _write_private_token(path: Path, token: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(temporary, flags, 0o600)
    try:
        os.write(fd, (token + "\n").encode("ascii"))
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        if os.name != "nt":
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def activate_writer_requirement(db_path: Path) -> None:
    """Durably and irreversibly require writer IPC for a board."""
    path = writer_activation_path(db_path)
    with kanban_safety.maintenance_lock(db_path, exclusive=True):
        try:
            path.lstat()
        except FileNotFoundError:
            _write_private_token(path, _ACTIVATION_MARKER)
            return
        marker = _read_private_token(path)
        if marker != _ACTIVATION_MARKER:
            raise WriterUnavailableError(
                f"single-writer activation marker has unsupported content at {path}"
            )


def _read_private_token(path: Path) -> str:
    try:
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_mode & 0o077:
            raise WriterUnavailableError("single-writer token is not a private regular file")
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(path, flags)
        try:
            token = os.read(fd, 1024).decode("ascii").strip()
        finally:
            os.close(fd)
    except (OSError, UnicodeDecodeError) as exc:
        raise WriterUnavailableError(f"single-writer token unavailable at {path}") from exc
    if not token:
        raise WriterUnavailableError(f"single-writer token is empty at {path}")
    return token


def _remove_stale_socket(path: Path) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return
    if not stat.S_ISSOCK(metadata.st_mode):
        raise WriterUnavailableError(f"refusing to unlink non-socket endpoint {path}")
    path.unlink()


class KanbanWriterService:
    """One sequential mutation executor and local IPC listener for one board."""

    def __init__(self, db_path: Path, *, board: str | None = None):
        self.db_path = Path(db_path).expanduser().resolve()
        self.board = _protocol_label(board or _infer_board(self.db_path), "board")
        self.socket_path = writer_socket_path(self.db_path)
        self.token_path = writer_token_path(self.db_path)
        self._token = secrets.token_hex(32)
        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._lock_handle = None
        self._stopping = threading.Event()
        self._stop_lock = threading.Lock()
        self._mutation_lock = threading.Lock()
        self._peer_slots = threading.BoundedSemaphore(64)
        self._peer_threads: set[threading.Thread] = set()
        self._peer_threads_lock = threading.Lock()
        self._finalize_lock = threading.Lock()
        self._finalized = threading.Event()
        self._finalize_error: BaseException | None = None
        self.service_generation: int | None = None
        self.board_generation: int | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        if self._lock_handle is not None:
            raise WriterUnavailableError("writer shutdown is still draining mutations")
        self._finalized.clear()
        self._finalize_error = None
        if not hasattr(socket, "AF_UNIX"):
            raise WriterUnavailableError("local Unix sockets are unavailable")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.socket_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._acquire_owner_lock()
        try:
            from hermes_cli import kanban_db as kb

            with kanban_safety.maintenance_lock(self.db_path, exclusive=True):
                kanban_safety.assert_board_not_quarantined(self.db_path)
                with writer_owner(self.db_path):
                    conn = kb._connect_under_maintenance_lock(self.db_path)
                    conn.close()
                self.service_generation = (
                    kanban_safety._bump_service_generation_locked(self.db_path)
                )
                self.board_generation = kanban_safety.read_generations(
                    self.db_path
                ).board_generation
                _write_private_token(self.token_path, self._token)
                _remove_stale_socket(self.socket_path)
                listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                listener.bind(str(self.socket_path))
                os.chmod(self.socket_path, 0o600)
                listener.listen(64)
                listener.settimeout(0.1)
                self._socket = listener
                # Activation is durable before the serving thread can accept a
                # mutation.  Once marked, every non-owner write fails closed
                # until this or a replacement writer is available.
                activate_writer_requirement(self.db_path)
            self._stopping.clear()
            self._thread = threading.Thread(
                target=self._serve, name=f"kanban-writer-{self.db_path.name}", daemon=True
            )
            self._thread.start()
        except Exception:
            with contextlib.suppress(Exception):
                self.stop()
            self._release_owner_lock()
            raise

    def _acquire_owner_lock(self) -> None:
        lock_path = writer_lock_path(self.db_path)
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            fd = os.open(lock_path, flags, 0o600)
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                os.close(fd)
                raise WriterOwnershipError(f"writer lock is not regular: {lock_path}")
            handle = os.fdopen(fd, "a+b")
        except OSError as exc:
            raise WriterOwnershipError(f"cannot open writer owner lock {lock_path}") from exc
        try:
            if os.name == "nt":  # pragma: no cover - Windows adapter
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as exc:
            handle.close()
            raise WriterOwnershipError(
                f"another single-writer service owns {self.db_path}"
            ) from exc
        self._lock_handle = handle

    def _release_owner_lock(self) -> None:
        handle, self._lock_handle = self._lock_handle, None
        if handle is None:
            return
        with contextlib.suppress(OSError):
            if os.name == "nt":  # pragma: no cover
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()

    def _finalize_stop_if_drained(self) -> bool:
        if not self._stopping.is_set():
            return False
        with self._peer_threads_lock:
            if any(peer.is_alive() for peer in self._peer_threads):
                return False
        with self._finalize_lock:
            if self._finalized.is_set():
                return True
            with self._peer_threads_lock:
                if any(peer.is_alive() for peer in self._peer_threads):
                    return False
            checkpoint_error: BaseException | None = None
            try:
                if (
                    self._lock_handle is not None
                    and self.db_path.exists()
                    and kanban_safety.active_quarantine(self.db_path) is None
                ):
                    from hermes_cli import kanban_db as kb

                    with writer_owner(self.db_path), kb.connect(self.db_path) as conn:
                        checkpoint = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
                    if checkpoint is not None and checkpoint[0] != 0:
                        raise WriterUnavailableError(
                            "writer shutdown WAL checkpoint was busy"
                        )
            except BaseException as exc:
                checkpoint_error = exc
            finally:
                with contextlib.suppress(FileNotFoundError, WriterUnavailableError):
                    _remove_stale_socket(self.socket_path)
                self._release_owner_lock()
                self._finalize_error = checkpoint_error
                self._finalized.set()
            return True

    def stop(self) -> None:
        with self._stop_lock:
            self._stopping.set()
            listener, self._socket = self._socket, None
            if listener is not None:
                listener.close()
            thread, self._thread = self._thread, None
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=2)

            with self._peer_threads_lock:
                peers = list(self._peer_threads)
            for peer_thread in peers:
                if peer_thread is not threading.current_thread():
                    peer_thread.join(timeout=DEFAULT_TIMEOUT_SECONDS)
            with self._peer_threads_lock:
                live_peers = [peer for peer in self._peer_threads if peer.is_alive()]
            if live_peers:
                raise WriterUnavailableError(
                    f"writer shutdown timed out draining {len(live_peers)} mutation(s)"
                )

            self._finalize_stop_if_drained()
            if self._finalize_error is not None:
                raise self._finalize_error

    def _serve(self) -> None:
        while not self._stopping.is_set():
            listener = self._socket
            if listener is None:
                break
            try:
                peer, _ = listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            if not self._peer_slots.acquire(blocking=False):
                peer.close()
                continue
            peer_thread = threading.Thread(
                target=self._serve_peer,
                args=(peer,),
                name=f"kanban-writer-peer-{self.db_path.name}",
                daemon=True,
            )
            with self._peer_threads_lock:
                self._peer_threads.add(peer_thread)
            peer_thread.start()

    def _serve_peer(self, peer: socket.socket) -> None:
        try:
            with peer:
                peer.settimeout(_SERVER_PEER_TIMEOUT_SECONDS)
                try:
                    request = _recv_message(peer)
                    with self._mutation_lock:
                        response = self._handle(request)
                except BaseException as exc:
                    response = self._error_response(exc)
                with contextlib.suppress(OSError, WriterProtocolError):
                    _send_message(peer, response)
        finally:
            self._peer_slots.release()
            with self._peer_threads_lock:
                self._peer_threads.discard(threading.current_thread())
            self._finalize_stop_if_drained()

    def _handle(self, request: dict[str, Any]) -> dict[str, Any]:
        expected_fields = {
            "authentication_token",
            "actor_profile",
            "board",
            "expected_board_generation",
            "expected_service_generation",
            "operation",
            "payload",
            "request_id",
            "source",
            "version",
        }
        if set(request) != expected_fields:
            raise WriterProtocolError("writer request fields are missing or unsupported")
        supplied = request["authentication_token"]
        if not isinstance(supplied, str):
            raise WriterProtocolError("authentication_token must be a string")
        if not hmac.compare_digest(supplied, self._token):
            raise WriterAuthenticationError("writer authentication rejected")
        if type(request["version"]) is not int or request["version"] != PROTOCOL_VERSION:
            raise WriterProtocolError("unsupported writer protocol version")
        board = _protocol_label(request["board"], "board")
        actor_profile = _protocol_label(request["actor_profile"], "actor_profile")
        source = mutation_source_label(request["source"])
        if board != self.board:
            raise WriterProtocolError(
                f"writer board mismatch: expected {self.board!r}, received {board!r}"
            )
        operation = request["operation"]
        if not isinstance(operation, str):
            raise WriterProtocolError("operation must be a string")
        if operation not in RUNTIME_MUTATIONS:
            raise WriterProtocolError(f"unsupported writer mutation: {operation}")
        request_id = request["request_id"]
        if not isinstance(request_id, str):
            raise WriterProtocolError("request_id must be a string")
        if not request_id or len(request_id) > 200:
            raise WriterProtocolError("request_id is required and must be <= 200 chars")
        encoded_payload = request.get("payload")
        kwargs = _decode_value(encoded_payload)
        if not isinstance(kwargs, dict):
            raise WriterProtocolError("mutation payload must decode to a dict")
        expected_service = request["expected_service_generation"]
        expected_board = request["expected_board_generation"]
        if (
            type(expected_service) is not int
            or type(expected_board) is not int
            or expected_service < 0
            or expected_board < 0
        ):
            raise WriterProtocolError(
                "expected service and board generations must be non-negative integers"
            )
        identity_text = json.dumps(
            {
                "payload": encoded_payload,
                "expected_service_generation": expected_service,
                "expected_board_generation": expected_board,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        payload_digest = hashlib.sha256(identity_text.encode("utf-8")).hexdigest()

        from hermes_cli import kanban_db as kb
        with writer_owner(self.db_path), kb.connect(self.db_path) as conn:
            with kb.write_txn(conn):
                replay = conn.execute(
                    "SELECT board, actor_profile, source, operation, payload_digest, response "
                    "FROM kanban_writer_requests WHERE request_id=?",
                    (request_id,),
                ).fetchone()
                if replay is not None:
                    if (
                        replay["board"] != board
                        or replay["actor_profile"] != actor_profile
                        or replay["source"] != source
                        or replay["operation"] != operation
                        or replay["payload_digest"] != payload_digest
                    ):
                        raise WriterProtocolError(
                            "request_id was already used with another envelope, operation, or payload"
                        )
                    return {
                        "ok": True,
                        "result": json.loads(replay["response"]),
                        "replayed": True,
                    }
                kanban_safety.validate_generations(
                    self.db_path,
                    expected_service_generation=expected_service,
                    expected_board_generation=expected_board,
                )
                kanban_safety.assert_board_not_quarantined(self.db_path)
                operation_function = getattr(kb, operation)
                try:
                    inspect.signature(operation_function).bind(conn, **kwargs)
                except TypeError as exc:
                    raise WriterProtocolError(
                        f"invalid payload for writer mutation {operation}: {exc}"
                    ) from exc
                result = operation_function(conn, **kwargs)
                if operation == "detect_crashed_workers":
                    result = {
                        "result": result,
                        "side_channels": {
                            "_last_auto_blocked": getattr(
                                operation_function, "_last_auto_blocked", []
                            ),
                            "_last_rate_limited": getattr(
                                operation_function, "_last_rate_limited", []
                            ),
                        },
                    }
                encoded = _json_text(result)
                conn.execute(
                    "INSERT INTO kanban_writer_requests"
                    "(request_id, board, actor_profile, source, writer_pid, "
                    "operation, payload_digest, response, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, strftime('%s','now'))",
                    (
                        request_id,
                        board,
                        actor_profile,
                        source,
                        os.getpid(),
                        operation,
                        payload_digest,
                        encoded,
                    ),
                )
            return {"ok": True, "result": json.loads(encoded), "replayed": False}

    @staticmethod
    def _error_response(exc: BaseException) -> dict[str, Any]:
        return {
            "ok": False,
            "error_module": type(exc).__module__,
            "error_type": type(exc).__name__,
            "message": str(exc)[:2000],
        }


class KanbanWriterClient:
    """Bounded, fail-closed client for a board's writer service."""

    def __init__(
        self,
        db_path: Path,
        *,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        authentication_token: str | None = None,
        board: str | None = None,
        actor_profile: str | None = None,
        source: str | None = None,
    ):
        self.db_path = Path(db_path).expanduser().resolve()
        self.timeout_seconds = max(0.001, float(timeout_seconds))
        self.authentication_token = authentication_token
        self.board = _protocol_label(board or _infer_board(self.db_path), "board")
        self.actor_profile = _protocol_label(
            actor_profile or os.environ.get("HERMES_PROFILE") or "default",
            "actor_profile",
        )
        self.source = mutation_source_label(source or "worker")

    def mutate(
        self,
        operation: str,
        kwargs: dict[str, Any],
        *,
        request_id: str | None = None,
        expected_service_generation: int | None = None,
        expected_board_generation: int | None = None,
    ) -> Any:
        token = self.authentication_token
        if token is None:
            token = _read_private_token(writer_token_path(self.db_path))
        generations = kanban_safety.read_generations(self.db_path)
        if expected_service_generation is None:
            expected_service_generation = generations.service_generation
        if expected_board_generation is None:
            expected_board_generation = generations.board_generation
        request = {
            "authentication_token": token,
            "actor_profile": self.actor_profile,
            "board": self.board,
            "expected_board_generation": expected_board_generation,
            "expected_service_generation": expected_service_generation,
            "operation": operation,
            "payload": _encode_value(kwargs),
            "request_id": request_id or uuid.uuid4().hex,
            "source": self.source,
            "version": PROTOCOL_VERSION,
        }
        response: dict[str, Any] | None = None
        last_error: BaseException | None = None
        for _attempt in range(2):
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as peer:
                    peer.settimeout(self.timeout_seconds)
                    peer.connect(str(writer_socket_path(self.db_path)))
                    _send_message(peer, request)
                    response = _recv_message(peer)
                break
            except (OSError, TimeoutError, WriterProtocolError) as exc:
                last_error = exc
        if response is None:
            raise WriterUnavailableError(
                f"single-writer service unavailable for {self.db_path}"
            ) from last_error
        if response.get("ok") is True:
            return _decode_value(response["result"])
        error_type = str(response.get("error_type") or "")
        message = str(response.get("message") or "writer mutation failed")
        if error_type == "BoardQuarantinedError":
            kanban_safety.assert_board_not_quarantined(self.db_path)
        known_errors: dict[str, type[BaseException]] = {
            "WriterAuthenticationError": WriterAuthenticationError,
            "WriterOwnershipError": WriterOwnershipError,
            "WriterProtocolError": WriterProtocolError,
            "WriterUnavailableError": WriterUnavailableError,
            "GenerationFencedError": kanban_safety.GenerationFencedError,
            "MaintenanceLockError": kanban_safety.MaintenanceLockError,
            "QuarantinePersistenceError": kanban_safety.QuarantinePersistenceError,
            "ValueError": ValueError,
            "TypeError": TypeError,
            "KeyError": KeyError,
            "DatabaseError": sqlite3.DatabaseError,
            "IntegrityError": sqlite3.IntegrityError,
            "OperationalError": sqlite3.OperationalError,
        }
        error_class = known_errors.get(error_type)
        if error_class is not None:
            raise error_class(message)
        raise WriterProtocolError(f"{error_type or 'writer error'}: {message}")


def _serve_process(db_path: Path, board: str | None) -> int:
    stopped = threading.Event()

    def request_stop(_signum, _frame) -> None:
        stopped.set()

    previous_handlers = {
        signum: signal.signal(signum, request_stop)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    service = KanbanWriterService(db_path, board=board)
    try:
        service.start()
        stopped.wait()
    finally:
        service.stop()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hermes Kanban single-writer service")
    subparsers = parser.add_subparsers(dest="command", required=True)
    serve = subparsers.add_parser("serve", help="serve one board over local IPC")
    serve.add_argument("--db", required=True, type=Path)
    serve.add_argument("--board")
    args = parser.parse_args(argv)
    if args.command == "serve":
        return _serve_process(args.db, args.board)
    parser.error(f"unsupported command: {args.command}")


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess
    # ``python -m`` executes this file as ``__main__``. Delegate to the
    # canonical module instance so kanban_db and the service share the same
    # ContextVars used for writer ownership.
    from hermes_cli.kanban_writer import main as _canonical_main

    raise SystemExit(_canonical_main())

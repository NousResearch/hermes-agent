"""Strict local protocol for the no-secret browser controller.

The protocol is deliberately smaller than the agent-browser CLI.  It carries
only bounded, structured mechanical browser actions.  It has no shell, PATH,
CDP, profile-path, arbitrary output-path, install, or raw JavaScript surface.
"""

from __future__ import annotations

import json
import re
import socket
import struct
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Mapping


PROTOCOL_VERSION = "hermes-browser-controller.v1"
MAX_REQUEST_BYTES = 64 * 1024
MAX_RESPONSE_BYTES = 12 * 1024 * 1024
MAX_RESULT_BYTES = 2 * 1024 * 1024
MAX_ARTIFACT_BYTES = 8 * 1024 * 1024
MAX_TEXT_ARGUMENT_BYTES = 32 * 1024
MAX_REQUESTS_PER_SESSION = 1024
_FRAME = struct.Struct("!I")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_REF = re.compile(r"@[A-Za-z][A-Za-z0-9_.:-]{0,63}")
_KEY = re.compile(r"[A-Za-z0-9][A-Za-z0-9+_.:-]{0,63}")


class BrowserControllerProtocolError(ValueError):
    """One stable protocol validation failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class PeerCredentials:
    pid: int
    uid: int
    gid: int


def unix_peer_credentials(sock: socket.socket) -> PeerCredentials:
    """Read authenticated kernel credentials for one connected Unix peer."""

    if hasattr(socket, "SO_PEERCRED"):
        raw = sock.getsockopt(
            socket.SOL_SOCKET,
            socket.SO_PEERCRED,
            struct.calcsize("3i"),
        )
        pid, uid, gid = struct.unpack("3i", raw)
        return PeerCredentials(pid=pid, uid=uid, gid=gid)
    getpeereid = getattr(sock, "getpeereid", None)
    if callable(getpeereid):
        uid, gid = getpeereid()
        return PeerCredentials(pid=-1, uid=int(uid), gid=int(gid))
    # CPython on macOS exposes LOCAL_PEERCRED but not getpeereid(). Darwin's
    # xucred layout is: uint version, uid_t uid, short ngroups, 2-byte pad,
    # then gid_t groups[16]. SOL_LOCAL is the literal level 0 on Darwin.
    if hasattr(socket, "LOCAL_PEERCRED"):
        try:
            raw = sock.getsockopt(0, socket.LOCAL_PEERCRED, 76)
            _version, uid, groups = struct.unpack_from("=IIh", raw)
            gid = struct.unpack_from("=I", raw, 12)[0] if groups > 0 else -1
            return PeerCredentials(pid=-1, uid=int(uid), gid=int(gid))
        except (OSError, struct.error, ValueError) as exc:
            raise BrowserControllerProtocolError(
                "browser_controller_peer_credentials_unavailable"
            ) from exc
    raise BrowserControllerProtocolError(
        "browser_controller_peer_credentials_unavailable"
    )


def _duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def canonical_json(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            dict(value),
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise BrowserControllerProtocolError("browser_controller_json_invalid") from exc


def decode_json(payload: bytes, *, maximum: int) -> dict[str, Any]:
    if not isinstance(payload, bytes) or not 0 < len(payload) <= maximum:
        raise BrowserControllerProtocolError("browser_controller_frame_size_invalid")
    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=_duplicates,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise BrowserControllerProtocolError("browser_controller_json_invalid") from exc
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise BrowserControllerProtocolError("browser_controller_frame_shape_invalid")
    return value


def _receive_exact(
    sock: socket.socket,
    size: int,
    *,
    deadline: float | None = None,
    stop_requested: Callable[[], bool] | None = None,
) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        try:
            chunk = sock.recv(remaining)
        except socket.timeout:
            # The controller polls so stop() can reliably wake a worker on all
            # supported Unix kernels.  Preserve partial frame bytes across
            # those polls; restarting framing after a fragmented write would
            # let one peer desynchronise the stream.
            if deadline is None:
                raise
            if (
                stop_requested is not None and stop_requested()
            ) or time.monotonic() >= deadline:
                raise
            continue
        if not chunk:
            raise EOFError("browser_controller_connection_closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def receive_frame(
    sock: socket.socket,
    *,
    maximum: int,
    deadline: float | None = None,
    stop_requested: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    (size,) = _FRAME.unpack(
        _receive_exact(
            sock,
            _FRAME.size,
            deadline=deadline,
            stop_requested=stop_requested,
        )
    )
    if size == 0 or size > maximum:
        raise BrowserControllerProtocolError("browser_controller_frame_size_invalid")
    return decode_json(
        _receive_exact(
            sock,
            size,
            deadline=deadline,
            stop_requested=stop_requested,
        ),
        maximum=maximum,
    )


def send_frame(sock: socket.socket, value: Mapping[str, Any], *, maximum: int) -> None:
    payload = canonical_json(value)
    if not payload or len(payload) > maximum:
        raise BrowserControllerProtocolError("browser_controller_frame_size_invalid")
    sock.sendall(_FRAME.pack(len(payload)) + payload)


def _exact(
    value: Any,
    *,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
    code: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(k, str) for k in value):
        raise BrowserControllerProtocolError(code)
    result = dict(value)
    if set(result) - required - optional or required - set(result):
        raise BrowserControllerProtocolError(code)
    return result


def _request_id(value: Any) -> str:
    if not isinstance(value, str):
        raise BrowserControllerProtocolError("browser_controller_request_id_invalid")
    try:
        parsed = uuid.UUID(value)
    except (TypeError, ValueError, AttributeError) as exc:
        raise BrowserControllerProtocolError(
            "browser_controller_request_id_invalid"
        ) from exc
    if str(parsed) != value:
        raise BrowserControllerProtocolError("browser_controller_request_id_invalid")
    return value


def _base_request(value: Any) -> dict[str, Any]:
    raw = _exact(
        value,
        required=frozenset({"version", "request_id", "op"}),
        optional=frozenset({"session_id_sha256", "command", "args"}),
        code="browser_controller_request_shape_invalid",
    )
    if raw["version"] != PROTOCOL_VERSION:
        raise BrowserControllerProtocolError("browser_controller_version_invalid")
    _request_id(raw["request_id"])
    if not isinstance(raw["op"], str):
        raise BrowserControllerProtocolError("browser_controller_operation_invalid")
    return raw


def validate_session_open(value: Any) -> dict[str, Any]:
    raw = _base_request(value)
    if raw["op"] != "session.open" or set(raw) != {
        "version",
        "request_id",
        "op",
        "session_id_sha256",
    }:
        raise BrowserControllerProtocolError("browser_controller_session_open_invalid")
    identity = raw["session_id_sha256"]
    if not isinstance(identity, str) or _SHA256.fullmatch(identity) is None:
        raise BrowserControllerProtocolError("browser_controller_session_identity_invalid")
    return raw


def validate_session_close(value: Any) -> dict[str, Any]:
    raw = _base_request(value)
    if raw["op"] != "session.close" or set(raw) != {
        "version",
        "request_id",
        "op",
    }:
        raise BrowserControllerProtocolError("browser_controller_session_close_invalid")
    return raw


@dataclass(frozen=True)
class BrowserCommand:
    name: str
    argv_command: str
    argv: tuple[str, ...]
    screenshot: bool = False


def _args(value: Any) -> list[str]:
    if not isinstance(value, list) or len(value) > 8:
        raise BrowserControllerProtocolError("browser_controller_command_args_invalid")
    if any(not isinstance(item, str) for item in value):
        raise BrowserControllerProtocolError("browser_controller_command_args_invalid")
    if sum(len(item.encode("utf-8")) for item in value) > MAX_TEXT_ARGUMENT_BYTES:
        raise BrowserControllerProtocolError("browser_controller_command_args_oversized")
    if any("\x00" in item for item in value):
        raise BrowserControllerProtocolError("browser_controller_command_args_invalid")
    return value


def normalize_command(name: Any, value: Any) -> BrowserCommand:
    """Map one structured command to fixed agent-browser argv."""

    if not isinstance(name, str):
        raise BrowserControllerProtocolError("browser_controller_command_invalid")
    args = _args(value)
    if name == "open":
        if len(args) != 1 or not args[0]:
            raise BrowserControllerProtocolError("browser_controller_open_args_invalid")
        return BrowserCommand(name, "open", tuple(args))
    if name == "snapshot":
        if args not in ([], ["-c"]):
            raise BrowserControllerProtocolError("browser_controller_snapshot_args_invalid")
        return BrowserCommand(name, name, tuple(args))
    if name == "click":
        if len(args) != 1 or _REF.fullmatch(args[0]) is None:
            raise BrowserControllerProtocolError("browser_controller_ref_invalid")
        return BrowserCommand(name, name, tuple(args))
    if name == "fill":
        if len(args) != 2 or _REF.fullmatch(args[0]) is None:
            raise BrowserControllerProtocolError("browser_controller_fill_args_invalid")
        return BrowserCommand(name, name, tuple(args))
    if name == "scroll":
        if (
            len(args) != 2
            or args[0] not in {"up", "down"}
            or not args[1].isdigit()
            or not 1 <= int(args[1]) <= 5000
        ):
            raise BrowserControllerProtocolError("browser_controller_scroll_args_invalid")
        return BrowserCommand(name, name, tuple(args))
    if name == "back":
        if args:
            raise BrowserControllerProtocolError("browser_controller_back_args_invalid")
        return BrowserCommand(name, name, ())
    if name == "press":
        if len(args) != 1 or _KEY.fullmatch(args[0]) is None:
            raise BrowserControllerProtocolError("browser_controller_press_args_invalid")
        return BrowserCommand(name, name, tuple(args))
    if name in {"console", "errors"}:
        if args not in ([], ["--clear"]):
            raise BrowserControllerProtocolError("browser_controller_log_args_invalid")
        return BrowserCommand(name, name, tuple(args))
    if name == "screenshot":
        if (
            len(args) != len(set(args))
            or any(item not in {"--annotate", "--full"} for item in args)
        ):
            raise BrowserControllerProtocolError(
                "browser_controller_screenshot_args_invalid"
            )
        return BrowserCommand(name, name, tuple(args), screenshot=True)
    if name == "current_url":
        if args:
            raise BrowserControllerProtocolError(
                "browser_controller_current_url_args_invalid"
            )
        return BrowserCommand(name, "eval", ("window.location.href",))
    # Raw eval, CDP, cookies, tabs, downloads, install and arbitrary CLI verbs
    # intentionally have no protocol representation.
    raise BrowserControllerProtocolError("browser_controller_command_forbidden")


def validate_command_request(value: Any) -> tuple[dict[str, Any], BrowserCommand]:
    raw = _base_request(value)
    if raw["op"] != "command" or set(raw) != {
        "version",
        "request_id",
        "op",
        "command",
        "args",
    }:
        raise BrowserControllerProtocolError("browser_controller_command_request_invalid")
    return raw, normalize_command(raw["command"], raw["args"])


def ok_response(request_id: str, result: Mapping[str, Any]) -> dict[str, Any]:
    _request_id(request_id)
    if not isinstance(result, Mapping):
        raise BrowserControllerProtocolError("browser_controller_result_invalid")
    return {
        "version": PROTOCOL_VERSION,
        "request_id": request_id,
        "status": "ok",
        "result": dict(result),
    }


def error_response(request_id: str, code: str) -> dict[str, Any]:
    try:
        _request_id(request_id)
    except BrowserControllerProtocolError:
        request_id = str(uuid.UUID(int=0))
    if not isinstance(code, str) or not code or len(code) > 160:
        code = "browser_controller_request_failed"
    return {
        "version": PROTOCOL_VERSION,
        "request_id": request_id,
        "status": "error",
        "error": code,
    }


def validate_response(value: Any, request_id: str) -> dict[str, Any]:
    raw = _exact(
        value,
        required=frozenset({"version", "request_id", "status"}),
        optional=frozenset({"result", "error"}),
        code="browser_controller_response_shape_invalid",
    )
    if raw["version"] != PROTOCOL_VERSION or raw["request_id"] != request_id:
        raise BrowserControllerProtocolError("browser_controller_response_binding_invalid")
    if raw["status"] == "ok":
        if set(raw) != {"version", "request_id", "status", "result"} or not isinstance(
            raw["result"], Mapping
        ):
            raise BrowserControllerProtocolError("browser_controller_response_shape_invalid")
    elif raw["status"] == "error":
        if set(raw) != {"version", "request_id", "status", "error"} or not isinstance(
            raw["error"], str
        ):
            raise BrowserControllerProtocolError("browser_controller_response_shape_invalid")
    else:
        raise BrowserControllerProtocolError("browser_controller_response_status_invalid")
    return raw


__all__ = [
    "BrowserCommand",
    "BrowserControllerProtocolError",
    "MAX_ARTIFACT_BYTES",
    "MAX_REQUEST_BYTES",
    "MAX_REQUESTS_PER_SESSION",
    "MAX_RESPONSE_BYTES",
    "MAX_RESULT_BYTES",
    "PeerCredentials",
    "PROTOCOL_VERSION",
    "canonical_json",
    "decode_json",
    "error_response",
    "normalize_command",
    "ok_response",
    "receive_frame",
    "send_frame",
    "unix_peer_credentials",
    "validate_command_request",
    "validate_response",
    "validate_session_close",
    "validate_session_open",
]

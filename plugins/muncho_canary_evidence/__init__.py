"""Sealed evidence observer for the isolated Muncho full canary.

This bundled plugin is deliberately opt-in.  It registers observer hooks only:
no model tool, middleware, prompt/context injection, classifier, dispatcher, or
persistent writer.  The model remains the sole semantic decision maker.

Evidence is sent synchronously as bounded canonical-JSON frames to one
plan-fixed root collector over AF_UNIX.  Both sides authenticate the other by
Linux peer credentials.  The collector independently verifies that this
client is the exact gateway MainPID; this plugin reciprocally pins the root
collector PID/UID/GID from a root-owned runtime materialization.

``on_session_start`` also performs two mechanical boundary actions before the
first model request:

* bind the exact API-server session-key digest to the owner-published fixture;
  and
* send one deliberately invalid private-target frame to the privileged Discord
  edge.  A pre-probe collector ACK is a barrier for the collector-owned journal
  snapshot.  The plugin never opens or reads the edge journal.

Hook exceptions are observationally isolated by Hermes.  The Canonical Writer
remains the privileged mechanical mutation boundary, while the model authors
all task meaning in the Canonical Task Workspace.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import socket
import stat
import struct
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping


CONFIG_SCHEMA = "muncho-canary-evidence-config.v1"
FRAME_SCHEMA = "muncho-canary-evidence-frame.v1"
ACK_SCHEMA = "muncho-canary-evidence-ack.v1"
GOAL_CONFIG_SCHEMA = "muncho-capability-goal-observer-config.v1"
GOAL_FRAME_SCHEMA = "muncho-capability-goal-observer-frame.v1"
API_OBSERVER_RETIREMENT_SCHEMA = (
    "muncho-capability-api-observer-retirement.v1"
)

DEFAULT_CONFIG_PATH = Path("/etc/muncho/full-canary/observer.json")
DEFAULT_COLLECTOR_SOCKET_PATH = Path("/run/muncho-full-canary/collector.sock")
DEFAULT_GOAL_CONFIG_PATH = Path(
    "/etc/muncho/capability-canary/goal-observer.json"
)
DEFAULT_GOAL_COLLECTOR_SOCKET_PATH = Path(
    "/run/muncho-capability-goal/collector.sock"
)
DEFAULT_API_OBSERVER_RETIREMENT_PATH = Path(
    "/run/muncho-capability-goal/api-observer-retired.json"
)
DEFAULT_DISCORD_EDGE_SOCKET_PATH = Path(
    "/run/muncho-discord-egress/edge.sock"
)

MAX_CONFIG_BYTES = 64 * 1024
MAX_FIXTURE_BYTES = 128 * 1024
MAX_MODULE_BYTES = 4 * 1024 * 1024
MAX_FRAME_BYTES = 2 * 1024 * 1024
MAX_ACK_BYTES = 16 * 1024
MAX_TOOL_PROJECTION_BYTES = 768 * 1024
MAX_TOOL_CALLS = 256
MAX_MODEL_CALLS = 128
MAX_IDENTIFIER_CHARS = 240

_FRAME_HEADER = struct.Struct("!I")
_PEER_CREDENTIALS = struct.Struct("3i")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_CASE_ID_RE = re.compile(r"^case:[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}$")
_SNOWFLAKE_RE = re.compile(r"^[1-9][0-9]{5,24}$")

_GOAL_CONFIG_FIELDS = frozenset(
    {
        "schema",
        "release_sha",
        "release_sha256",
        "run_id",
        "fixture_sha256",
        "valid_from_unix_ms",
        "valid_until_unix_ms",
        "public_target",
        "owner_user_id",
        "model_route",
        "collector",
        "api_observer_retirement",
    }
)
_GOAL_MODEL_ROUTE_FIELDS = frozenset(
    {"provider", "api_mode", "base_url", "model", "fallback_configured"}
)
_API_OBSERVER_RETIREMENT_CONFIG_FIELDS = frozenset(
    {
        "marker_path",
        "marker_sha256",
        "marker_file_sha256",
        "api_observer_config_path",
        "api_observer_config_sha256",
        "goal_config_authority_sha256",
    }
)
_API_OBSERVER_RETIREMENT_FIELDS = frozenset(
    {
        "schema",
        "release_sha",
        "release_sha256",
        "run_id",
        "fixture_sha256",
        "api_observer_config_path",
        "api_observer_config_sha256",
        "goal_config_authority_sha256",
        "historical_api_observer_terminal",
        "message_content_recorded",
        "marker_sha256",
    }
)
_GOAL_EVENTS = frozenset(
    {
        "goal_plugin_ready",
        "goal_pre_api_request",
        "goal_post_api_request",
        "goal_model_outcome",
        "goal_canonical_event",
        "goal_canonical_readback",
        "goal_turn_end",
    }
)


def _process_gid() -> int:
    """Return the POSIX process GID or reject this Linux-only plugin boundary."""

    getter = getattr(os, "getgid", None)
    if not callable(getter):
        raise CanaryEvidenceError("config_invalid")
    return int(getter())

_CONFIG_FIELDS = frozenset(
    {
        "schema",
        "release_sha",
        "release_sha256",
        "canary_run_id",
        "case_id",
        "fixture_path",
        "fixture_sha256",
        "collector",
        "discord_edge",
    }
)
_COLLECTOR_FIELDS = frozenset(
    {
        "socket_path",
        "expected_pid",
        "expected_uid",
        "expected_gid",
        "socket_owner_uid",
        "socket_owner_gid",
        "socket_mode",
        "service_identity_sha256",
        "connect_timeout_ms",
        "ack_timeout_ms",
    }
)
_EDGE_FIELDS = frozenset(
    {
        "socket_path",
        "expected_pid",
        "expected_uid",
        "expected_gid",
        "socket_owner_uid",
        "socket_owner_gid",
        "socket_mode",
        "service_identity_sha256",
        "connect_timeout_ms",
        "response_timeout_ms",
    }
)
_FIXTURE_FIELDS = frozenset(
    {
        "schema",
        "canary_run_id",
        "release_sha",
        "release_artifact_sha256",
        "api_session_key_sha256",
        "valid_from_unix_ms",
        "valid_until_unix_ms",
        "case_id",
        "owner_discord_user_id",
        "source",
        "model_route",
        "task_policy",
        "public_routeback",
        "discord_public_keys",
    }
)

_READY_EVENTS = frozenset({"plugin_ready"})
_START_EVENTS = frozenset(
    {
        "api_session_bound",
        "private_target_probe_ready",
        "private_target_probe_result",
    }
)
_EVENTS = _READY_EVENTS | _START_EVENTS | frozenset(
    {
        "pre_api_request",
        "post_api_request",
        "post_tool_call",
        "canonical_case_readback",
        "session_end",
    }
)
_PROJECTED_TOOL_RESULTS = frozenset(
    {
        "canonical_event_append",
        "canonical_brain_query",
        "route_back_state",
        "route_back_execute",
    }
)


class CanaryEvidenceError(RuntimeError):
    """Stable, secret-free plugin failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class PeerIdentity:
    pid: int
    uid: int
    gid: int


@dataclass(frozen=True)
class SocketIdentity:
    owner_uid: int
    owner_gid: int
    mode: int


@dataclass(frozen=True)
class CollectorEndpoint:
    socket_path: Path
    expected_peer: PeerIdentity
    socket_identity: SocketIdentity
    service_identity_sha256: str
    connect_timeout_ms: int
    ack_timeout_ms: int


@dataclass(frozen=True)
class EdgeEndpoint:
    socket_path: Path
    expected_peer: PeerIdentity
    socket_identity: SocketIdentity
    service_identity_sha256: str
    connect_timeout_ms: int
    response_timeout_ms: int


@dataclass(frozen=True)
class CanaryFixture:
    value: Mapping[str, Any]
    sha256: str

    @property
    def valid_from_unix_ms(self) -> int:
        return int(self.value["valid_from_unix_ms"])

    @property
    def valid_until_unix_ms(self) -> int:
        return int(self.value["valid_until_unix_ms"])

    @property
    def model_route(self) -> Mapping[str, Any]:
        return self.value["model_route"]

    @property
    def public_target(self) -> Mapping[str, Any]:
        return self.value["public_routeback"]["target"]


@dataclass(frozen=True)
class CanaryEvidenceConfig:
    config_sha256: str
    release_sha: str
    release_sha256: str
    canary_run_id: str
    case_id: str
    fixture: CanaryFixture
    collector: CollectorEndpoint
    discord_edge: EdgeEndpoint


@dataclass(frozen=True)
class GoalObserverConfig:
    """Credential-free, exact admission for the Discord goal observer.

    This is intentionally separate from ``CanaryEvidenceConfig``.  The API
    canary keeps its historical single-session/single-process contract while
    the goal observer writes a second segmented chain that can span a full
    gateway restart.
    """

    config_sha256: str
    release_sha: str
    release_sha256: str
    run_id: str
    fixture_sha256: str
    valid_from_unix_ms: int
    valid_until_unix_ms: int
    public_target: Mapping[str, str]
    owner_user_id: str
    model_route: Mapping[str, Any]
    collector: CollectorEndpoint
    api_observer_retirement: Mapping[str, Any]


PeerGetter = Callable[[socket.socket], PeerIdentity]
CollectorTransport = Callable[[CollectorEndpoint, bytes], Mapping[str, Any]]
SocketInspector = Callable[[Path, SocketIdentity], str]
EdgeProbe = Callable[[EdgeEndpoint, CanaryFixture, str, int], Mapping[str, Any]]
WriterCall = Callable[[Any, Mapping[str, Any]], Mapping[str, Any]]
RuntimeEnvelope = Callable[[], Mapping[str, Any]]


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8", errors="strict")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise CanaryEvidenceError("non_canonical_json") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _bounded_sha256_json(value: Any, *, maximum: int, code: str) -> str:
    body = _canonical_bytes(value)
    if not body or len(body) > maximum:
        raise CanaryEvidenceError(code)
    return _sha256_bytes(body)


def _strict_json(body: bytes, *, code: str) -> Mapping[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate")
            result[key] = value
        return result

    try:
        value = json.loads(
            body.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda _token: (_ for _ in ()).throw(
                ValueError("constant")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise CanaryEvidenceError(code) from exc
    if not isinstance(value, Mapping):
        raise CanaryEvidenceError(code)
    return value


def _strict_mapping(
    value: Any, *, fields: frozenset[str], code: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise CanaryEvidenceError(code)
    return value


def _digest(value: Any, *, code: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise CanaryEvidenceError(code)
    return value


def _safe_id(value: Any, *, code: str) -> str:
    if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
        raise CanaryEvidenceError(code)
    return value


def _uuid(value: Any, *, code: str) -> str:
    if not isinstance(value, str):
        raise CanaryEvidenceError(code)
    try:
        parsed = uuid.UUID(value)
    except (ValueError, TypeError, AttributeError) as exc:
        raise CanaryEvidenceError(code) from exc
    if parsed.int == 0 or str(parsed) != value:
        raise CanaryEvidenceError(code)
    return value


def _positive_int(value: Any, *, code: str, maximum: int = (1 << 63) - 1) -> int:
    if type(value) is not int or not 0 < value <= maximum:
        raise CanaryEvidenceError(code)
    return value


def _nonnegative_int(
    value: Any, *, code: str, maximum: int = (1 << 31) - 1
) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise CanaryEvidenceError(code)
    return value


def _timeout(value: Any, *, low: int, high: int, code: str) -> int:
    observed = _positive_int(value, code=code, maximum=high)
    if observed < low:
        raise CanaryEvidenceError(code)
    return observed


def _socket_mode(value: Any, *, code: str) -> int:
    if value != "0660":
        raise CanaryEvidenceError(code)
    return 0o660


def _absolute_path(value: Any, *, code: str) -> Path:
    if not isinstance(value, str):
        raise CanaryEvidenceError(code)
    path = Path(value)
    if (
        not path.is_absolute()
        or str(path) != value
        or path != Path(os.path.normpath(value))
        or ".." in path.parts
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
    ):
        raise CanaryEvidenceError(code)
    return path


def _read_owned_json(
    path: Path,
    *,
    expected_owner_uid: int,
    expected_owner_gid: int,
    expected_mode: int,
    maximum: int,
    expected_sha256: str | None,
    code: str,
) -> tuple[Mapping[str, Any], str]:
    path = _absolute_path(str(path), code=code)
    try:
        resolved_parent = path.parent.resolve(strict=True)
        parent_item = os.stat(resolved_parent, follow_symlinks=False)
        before = path.lstat()
    except OSError as exc:
        raise CanaryEvidenceError(code) from exc
    if (
        resolved_parent != path.parent
        or not stat.S_ISDIR(parent_item.st_mode)
        or parent_item.st_uid != expected_owner_uid
        or parent_item.st_mode & 0o022
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_uid != expected_owner_uid
        or before.st_gid != expected_owner_gid
        or stat.S_IMODE(before.st_mode) != expected_mode
        or not 1 < before.st_size <= maximum
    ):
        raise CanaryEvidenceError(code)

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise CanaryEvidenceError(code) from exc
    try:
        opened = os.fstat(fd)
        identity = (
            "st_dev",
            "st_ino",
            "st_uid",
            "st_gid",
            "st_mode",
            "st_size",
            "st_mtime_ns",
        )
        if any(getattr(before, name) != getattr(opened, name) for name in identity):
            raise CanaryEvidenceError(code)
        chunks: list[bytes] = []
        remaining = maximum + 1
        while remaining:
            chunk = os.read(fd, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(fd)
        if any(getattr(opened, name) != getattr(after, name) for name in identity):
            raise CanaryEvidenceError(code)
    finally:
        os.close(fd)
    body = b"".join(chunks)
    if len(body) != before.st_size or len(body) > maximum:
        raise CanaryEvidenceError(code)
    observed_digest = _sha256_bytes(body)
    if expected_sha256 is not None and observed_digest != expected_sha256:
        raise CanaryEvidenceError(code)
    return _strict_json(body, code=code), observed_digest


def _module_identity(path: Path | None = None) -> tuple[str, str]:
    """Return a stable digest for the exact loaded observer module.

    The deployed release is independently digest-bound by the full-canary
    controller.  This local read prevents a path/content race while producing
    the module evidence that the root collector joins to that release proof.
    """

    source = Path(__file__) if path is None else path
    try:
        resolved = source.resolve(strict=True)
        parent = resolved.parent.resolve(strict=True)
        before = resolved.lstat()
    except OSError as exc:
        raise CanaryEvidenceError("plugin_module_invalid") from exc
    if (
        not resolved.is_absolute()
        or parent != resolved.parent
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_mode & 0o022
        or not 1 < before.st_size <= MAX_MODULE_BYTES
    ):
        raise CanaryEvidenceError("plugin_module_invalid")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    try:
        fd = os.open(resolved, flags)
    except OSError as exc:
        raise CanaryEvidenceError("plugin_module_invalid") from exc
    try:
        opened = os.fstat(fd)
        identity = (
            "st_dev",
            "st_ino",
            "st_uid",
            "st_gid",
            "st_mode",
            "st_size",
            "st_mtime_ns",
        )
        if any(getattr(before, name) != getattr(opened, name) for name in identity):
            raise CanaryEvidenceError("plugin_module_invalid")
        chunks: list[bytes] = []
        remaining = MAX_MODULE_BYTES + 1
        while remaining:
            chunk = os.read(fd, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(fd)
        if any(getattr(opened, name) != getattr(after, name) for name in identity):
            raise CanaryEvidenceError("plugin_module_invalid")
    finally:
        os.close(fd)
    body = b"".join(chunks)
    if len(body) != before.st_size or len(body) > MAX_MODULE_BYTES:
        raise CanaryEvidenceError("plugin_module_invalid")
    return str(resolved), _sha256_bytes(body)


def _endpoint_peer(value: Mapping[str, Any], *, code: str) -> PeerIdentity:
    return PeerIdentity(
        pid=_positive_int(value["expected_pid"], code=code, maximum=(1 << 31) - 1),
        uid=_nonnegative_int(value["expected_uid"], code=code),
        gid=_nonnegative_int(value["expected_gid"], code=code),
    )


def _validate_fixture(value: Mapping[str, Any], digest: str) -> CanaryFixture:
    fixture = _strict_mapping(value, fields=_FIXTURE_FIELDS, code="fixture_invalid")
    if fixture["schema"] != "muncho-full-canary-e2e-fixture.v1":
        raise CanaryEvidenceError("fixture_invalid")
    _uuid(fixture["canary_run_id"], code="fixture_invalid")
    if (
        not isinstance(fixture["release_sha"], str)
        or _GIT_SHA_RE.fullmatch(fixture["release_sha"]) is None
        or not isinstance(fixture["case_id"], str)
        or _CASE_ID_RE.fullmatch(fixture["case_id"]) is None
        or not isinstance(fixture["owner_discord_user_id"], str)
        or _SNOWFLAKE_RE.fullmatch(fixture["owner_discord_user_id"]) is None
    ):
        raise CanaryEvidenceError("fixture_invalid")
    _digest(fixture["release_artifact_sha256"], code="fixture_invalid")
    _digest(fixture["api_session_key_sha256"], code="fixture_invalid")
    valid_from = _positive_int(fixture["valid_from_unix_ms"], code="fixture_invalid")
    valid_until = _positive_int(fixture["valid_until_unix_ms"], code="fixture_invalid")
    if valid_until <= valid_from or valid_until - valid_from > 3_600_000:
        raise CanaryEvidenceError("fixture_invalid")

    source = _strict_mapping(
        fixture["source"],
        fields=frozenset(
            {
                "platform",
                "control_protocol",
                "host",
                "port",
                "session_create_endpoint",
                "chat_stream_endpoint_template",
            }
        ),
        code="fixture_invalid",
    )
    if source != {
        "platform": "api_server",
        "control_protocol": "authenticated_loopback_api_server.v1",
        "host": "127.0.0.1",
        "port": 8642,
        "session_create_endpoint": "/api/sessions",
        "chat_stream_endpoint_template": "/api/sessions/{session_id}/chat/stream",
    }:
        raise CanaryEvidenceError("fixture_invalid")

    route = _strict_mapping(
        fixture["model_route"],
        fields=frozenset(
            {"provider", "api_mode", "base_url", "model", "initial_effort", "elevated_effort"}
        ),
        code="fixture_invalid",
    )
    for name in ("provider", "api_mode", "base_url", "model"):
        if not isinstance(route[name], str) or not route[name]:
            raise CanaryEvidenceError("fixture_invalid")
    if route["initial_effort"] != "high" or route["elevated_effort"] != "max":
        raise CanaryEvidenceError("fixture_invalid")

    task = _strict_mapping(
        fixture["task_policy"],
        fields=frozenset({"minimum_completed_steps", "prompt", "prompt_sha256"}),
        code="fixture_invalid",
    )
    minimum = task["minimum_completed_steps"]
    prompt = task["prompt"]
    if (
        type(minimum) is not int
        or not 3 <= minimum <= 64
        or not isinstance(prompt, str)
        or not prompt.strip()
        or len(prompt) > 16_000
        or _sha256_bytes(prompt.encode("utf-8", errors="strict"))
        != _digest(task["prompt_sha256"], code="fixture_invalid")
    ):
        raise CanaryEvidenceError("fixture_invalid")

    routeback = _strict_mapping(
        fixture["public_routeback"],
        fields=frozenset({"target", "canonical_idempotency_key"}),
        code="fixture_invalid",
    )
    _safe_id(routeback["canonical_idempotency_key"], code="fixture_invalid")
    try:
        from gateway.discord_edge_protocol import DiscordPublicTarget

        DiscordPublicTarget.from_mapping(routeback["target"])
    except (ImportError, TypeError, ValueError) as exc:
        raise CanaryEvidenceError("fixture_invalid") from exc

    keys = _strict_mapping(
        fixture["discord_public_keys"],
        fields=frozenset(
            {"writer_capability_ed25519_hex", "edge_receipt_ed25519_hex"}
        ),
        code="fixture_invalid",
    )
    if any(
        not isinstance(item, str) or re.fullmatch(r"[0-9a-f]{64}", item) is None
        for item in keys.values()
    ) or len(set(keys.values())) != 2:
        raise CanaryEvidenceError("fixture_invalid")
    return CanaryFixture(value=dict(fixture), sha256=digest)


def load_config(
    path: Path = DEFAULT_CONFIG_PATH,
    *,
    expected_owner_uid: int = 0,
    expected_owner_gid: int | None = None,
) -> CanaryEvidenceConfig:
    owner_gid = _process_gid() if expected_owner_gid is None else expected_owner_gid
    _nonnegative_int(expected_owner_uid, code="config_invalid")
    _nonnegative_int(owner_gid, code="config_invalid")
    raw, config_digest = _read_owned_json(
        path,
        expected_owner_uid=expected_owner_uid,
        expected_owner_gid=owner_gid,
        expected_mode=0o440,
        maximum=MAX_CONFIG_BYTES,
        expected_sha256=None,
        code="config_invalid",
    )
    root = _strict_mapping(raw, fields=_CONFIG_FIELDS, code="config_invalid")
    if root["schema"] != CONFIG_SCHEMA:
        raise CanaryEvidenceError("config_invalid")
    release_sha = root["release_sha"]
    if not isinstance(release_sha, str) or _GIT_SHA_RE.fullmatch(release_sha) is None:
        raise CanaryEvidenceError("config_invalid")
    release_sha256 = _digest(root["release_sha256"], code="config_invalid")
    canary_run_id = _uuid(root["canary_run_id"], code="config_invalid")
    case_id = root["case_id"]
    if not isinstance(case_id, str) or _CASE_ID_RE.fullmatch(case_id) is None:
        raise CanaryEvidenceError("config_invalid")
    fixture_sha256 = _digest(root["fixture_sha256"], code="config_invalid")
    fixture_path = _absolute_path(root["fixture_path"], code="config_invalid")
    fixture_raw, observed_fixture_digest = _read_owned_json(
        fixture_path,
        expected_owner_uid=expected_owner_uid,
        expected_owner_gid=owner_gid,
        expected_mode=0o440,
        maximum=MAX_FIXTURE_BYTES,
        expected_sha256=fixture_sha256,
        code="fixture_invalid",
    )
    fixture = _validate_fixture(fixture_raw, observed_fixture_digest)
    if (
        fixture.value["release_sha"] != release_sha
        or fixture.value["release_artifact_sha256"] != release_sha256
        or fixture.value["canary_run_id"] != canary_run_id
        or fixture.value["case_id"] != case_id
    ):
        raise CanaryEvidenceError("config_fixture_binding_invalid")

    collector_raw = _strict_mapping(
        root["collector"], fields=_COLLECTOR_FIELDS, code="config_invalid"
    )
    collector_path = _absolute_path(
        collector_raw["socket_path"], code="config_invalid"
    )
    if collector_path != DEFAULT_COLLECTOR_SOCKET_PATH:
        raise CanaryEvidenceError("config_invalid")
    collector_peer = _endpoint_peer(collector_raw, code="config_invalid")
    if collector_peer.uid != 0:
        raise CanaryEvidenceError("config_invalid")
    collector = CollectorEndpoint(
        socket_path=collector_path,
        expected_peer=collector_peer,
        socket_identity=SocketIdentity(
            owner_uid=_nonnegative_int(
                collector_raw["socket_owner_uid"], code="config_invalid"
            ),
            owner_gid=_nonnegative_int(
                collector_raw["socket_owner_gid"], code="config_invalid"
            ),
            mode=_socket_mode(collector_raw["socket_mode"], code="config_invalid"),
        ),
        service_identity_sha256=_digest(
            collector_raw["service_identity_sha256"], code="config_invalid"
        ),
        connect_timeout_ms=_timeout(
            collector_raw["connect_timeout_ms"],
            low=50,
            high=5_000,
            code="config_invalid",
        ),
        ack_timeout_ms=_timeout(
            collector_raw["ack_timeout_ms"],
            low=100,
            high=10_000,
            code="config_invalid",
        ),
    )
    if collector.socket_identity.owner_uid != 0:
        raise CanaryEvidenceError("config_invalid")

    edge_raw = _strict_mapping(
        root["discord_edge"], fields=_EDGE_FIELDS, code="config_invalid"
    )
    edge_path = _absolute_path(edge_raw["socket_path"], code="config_invalid")
    if edge_path != DEFAULT_DISCORD_EDGE_SOCKET_PATH:
        raise CanaryEvidenceError("config_invalid")
    edge = EdgeEndpoint(
        socket_path=edge_path,
        expected_peer=_endpoint_peer(edge_raw, code="config_invalid"),
        socket_identity=SocketIdentity(
            owner_uid=_nonnegative_int(
                edge_raw["socket_owner_uid"], code="config_invalid"
            ),
            owner_gid=_nonnegative_int(
                edge_raw["socket_owner_gid"], code="config_invalid"
            ),
            mode=_socket_mode(edge_raw["socket_mode"], code="config_invalid"),
        ),
        service_identity_sha256=_digest(
            edge_raw["service_identity_sha256"], code="config_invalid"
        ),
        connect_timeout_ms=_timeout(
            edge_raw["connect_timeout_ms"],
            low=50,
            high=5_000,
            code="config_invalid",
        ),
        response_timeout_ms=_timeout(
            edge_raw["response_timeout_ms"],
            low=100,
            high=5_000,
            code="config_invalid",
        ),
    )
    return CanaryEvidenceConfig(
        config_sha256=config_digest,
        release_sha=release_sha,
        release_sha256=release_sha256,
        canary_run_id=canary_run_id,
        case_id=case_id,
        fixture=fixture,
        collector=collector,
        discord_edge=edge,
    )


def load_goal_config(
    path: Path = DEFAULT_GOAL_CONFIG_PATH,
    *,
    expected_owner_uid: int = 0,
    expected_owner_gid: int | None = None,
) -> GoalObserverConfig:
    """Load the separate exact Discord-goal observer admission.

    The file contains no credential and no task text.  It is installed only
    for the bounded capability run and pins the release, fixture, public
    owner lane, model route, and root collector endpoint.
    """

    owner_gid = _process_gid() if expected_owner_gid is None else expected_owner_gid
    _nonnegative_int(expected_owner_uid, code="goal_config_invalid")
    _nonnegative_int(owner_gid, code="goal_config_invalid")
    raw, config_sha256 = _read_owned_json(
        path,
        expected_owner_uid=expected_owner_uid,
        expected_owner_gid=owner_gid,
        expected_mode=0o440,
        maximum=MAX_CONFIG_BYTES,
        expected_sha256=None,
        code="goal_config_invalid",
    )
    root = _strict_mapping(
        raw,
        fields=_GOAL_CONFIG_FIELDS,
        code="goal_config_invalid",
    )
    if root["schema"] != GOAL_CONFIG_SCHEMA:
        raise CanaryEvidenceError("goal_config_invalid")
    release_sha = root["release_sha"]
    if not isinstance(release_sha, str) or _GIT_SHA_RE.fullmatch(release_sha) is None:
        raise CanaryEvidenceError("goal_config_invalid")
    release_sha256 = _digest(root["release_sha256"], code="goal_config_invalid")
    run_id = _uuid(root["run_id"], code="goal_config_invalid")
    fixture_sha256 = _digest(root["fixture_sha256"], code="goal_config_invalid")
    valid_from = _positive_int(
        root["valid_from_unix_ms"], code="goal_config_invalid"
    )
    valid_until = _positive_int(
        root["valid_until_unix_ms"], code="goal_config_invalid"
    )
    if valid_until <= valid_from or valid_until - valid_from > 3_600_000:
        raise CanaryEvidenceError("goal_config_invalid")
    owner_user_id = root["owner_user_id"]
    if not isinstance(owner_user_id, str) or _SNOWFLAKE_RE.fullmatch(
        owner_user_id
    ) is None:
        raise CanaryEvidenceError("goal_config_invalid")
    target = _strict_mapping(
        root["public_target"],
        fields=frozenset({"target_type", "guild_id", "channel_id"}),
        code="goal_config_invalid",
    )
    if (
        target["target_type"] != "public_guild_channel"
        or any(
            not isinstance(target[name], str)
            or _SNOWFLAKE_RE.fullmatch(target[name]) is None
            for name in ("guild_id", "channel_id")
        )
    ):
        raise CanaryEvidenceError("goal_config_invalid")
    route = _strict_mapping(
        root["model_route"],
        fields=_GOAL_MODEL_ROUTE_FIELDS,
        code="goal_config_invalid",
    )
    if route != {
        "provider": "openai-codex",
        "api_mode": "codex_responses",
        "base_url": "https://chatgpt.com/backend-api/codex",
        "model": "gpt-5.6-sol",
        "fallback_configured": False,
    }:
        raise CanaryEvidenceError("goal_config_invalid")
    collector_raw = _strict_mapping(
        root["collector"], fields=_COLLECTOR_FIELDS, code="goal_config_invalid"
    )
    collector_path = _absolute_path(
        collector_raw["socket_path"], code="goal_config_invalid"
    )
    if collector_path != DEFAULT_GOAL_COLLECTOR_SOCKET_PATH:
        raise CanaryEvidenceError("goal_config_invalid")
    collector_peer = _endpoint_peer(collector_raw, code="goal_config_invalid")
    if collector_peer.uid != 0:
        raise CanaryEvidenceError("goal_config_invalid")
    collector = CollectorEndpoint(
        socket_path=collector_path,
        expected_peer=collector_peer,
        socket_identity=SocketIdentity(
            owner_uid=_nonnegative_int(
                collector_raw["socket_owner_uid"], code="goal_config_invalid"
            ),
            owner_gid=_nonnegative_int(
                collector_raw["socket_owner_gid"], code="goal_config_invalid"
            ),
            mode=_socket_mode(
                collector_raw["socket_mode"], code="goal_config_invalid"
            ),
        ),
        service_identity_sha256=_digest(
            collector_raw["service_identity_sha256"],
            code="goal_config_invalid",
        ),
        connect_timeout_ms=_timeout(
            collector_raw["connect_timeout_ms"],
            low=50,
            high=5_000,
            code="goal_config_invalid",
        ),
        ack_timeout_ms=_timeout(
            collector_raw["ack_timeout_ms"],
            low=100,
            high=10_000,
            code="goal_config_invalid",
        ),
    )
    if collector.socket_identity.owner_uid != 0:
        raise CanaryEvidenceError("goal_config_invalid")
    retirement = _strict_mapping(
        root["api_observer_retirement"],
        fields=_API_OBSERVER_RETIREMENT_CONFIG_FIELDS,
        code="goal_config_invalid",
    )
    marker_path = _absolute_path(
        retirement["marker_path"], code="goal_config_invalid"
    )
    api_config_path = _absolute_path(
        retirement["api_observer_config_path"], code="goal_config_invalid"
    )
    marker_sha256 = _digest(
        retirement["marker_sha256"], code="goal_config_invalid"
    )
    marker_file_sha256 = _digest(
        retirement["marker_file_sha256"], code="goal_config_invalid"
    )
    api_config_sha256 = _digest(
        retirement["api_observer_config_sha256"],
        code="goal_config_invalid",
    )
    goal_authority_sha256 = _digest(
        retirement["goal_config_authority_sha256"],
        code="goal_config_invalid",
    )
    core = {
        key: value
        for key, value in root.items()
        if key != "api_observer_retirement"
    }
    marker_unsigned = {
        "schema": API_OBSERVER_RETIREMENT_SCHEMA,
        "release_sha": release_sha,
        "release_sha256": release_sha256,
        "run_id": run_id,
        "fixture_sha256": fixture_sha256,
        "api_observer_config_path": str(api_config_path),
        "api_observer_config_sha256": api_config_sha256,
        "goal_config_authority_sha256": goal_authority_sha256,
        "historical_api_observer_terminal": True,
        "message_content_recorded": False,
    }
    if (
        marker_path != DEFAULT_API_OBSERVER_RETIREMENT_PATH
        or api_config_path != DEFAULT_CONFIG_PATH
        or goal_authority_sha256 != _sha256_json(core)
        or marker_sha256 != _sha256_json(marker_unsigned)
        or marker_file_sha256
        != _sha256_bytes(
            _canonical_bytes(
                {**marker_unsigned, "marker_sha256": marker_sha256}
            )
        )
    ):
        raise CanaryEvidenceError("goal_config_invalid")
    return GoalObserverConfig(
        config_sha256=config_sha256,
        release_sha=release_sha,
        release_sha256=release_sha256,
        run_id=run_id,
        fixture_sha256=fixture_sha256,
        valid_from_unix_ms=valid_from,
        valid_until_unix_ms=valid_until,
        public_target=dict(target),
        owner_user_id=owner_user_id,
        model_route=dict(route),
        collector=collector,
        api_observer_retirement=dict(retirement),
    )


def validate_api_observer_retirement(
    config: GoalObserverConfig,
    *,
    expected_owner_uid: int = 0,
    expected_owner_gid: int | None = None,
) -> Mapping[str, Any]:
    """Validate the exact root-authored stage marker before API skip."""

    owner_gid = _process_gid() if expected_owner_gid is None else expected_owner_gid
    _nonnegative_int(expected_owner_uid, code="api_observer_retirement_invalid")
    retirement = config.api_observer_retirement
    marker_path = Path(retirement["marker_path"])
    api_config_path = Path(retirement["api_observer_config_path"])
    _read_owned_json(
        api_config_path,
        expected_owner_uid=expected_owner_uid,
        expected_owner_gid=owner_gid,
        expected_mode=0o440,
        maximum=MAX_CONFIG_BYTES,
        expected_sha256=retirement["api_observer_config_sha256"],
        code="api_observer_retirement_invalid",
    )
    marker, marker_sha256 = _read_owned_json(
        marker_path,
        expected_owner_uid=expected_owner_uid,
        expected_owner_gid=owner_gid,
        expected_mode=0o440,
        maximum=MAX_CONFIG_BYTES,
        expected_sha256=retirement["marker_file_sha256"],
        code="api_observer_retirement_invalid",
    )
    marker = _strict_mapping(
        marker,
        fields=_API_OBSERVER_RETIREMENT_FIELDS,
        code="api_observer_retirement_invalid",
    )
    unsigned = {
        "schema": API_OBSERVER_RETIREMENT_SCHEMA,
        "release_sha": config.release_sha,
        "release_sha256": config.release_sha256,
        "run_id": config.run_id,
        "fixture_sha256": config.fixture_sha256,
        "api_observer_config_path": str(api_config_path),
        "api_observer_config_sha256": retirement[
            "api_observer_config_sha256"
        ],
        "goal_config_authority_sha256": retirement[
            "goal_config_authority_sha256"
        ],
        "historical_api_observer_terminal": True,
        "message_content_recorded": False,
    }
    expected = {**unsigned, "marker_sha256": _sha256_json(unsigned)}
    if (
        marker != expected
        or marker_sha256 != retirement["marker_file_sha256"]
        or marker["marker_sha256"] != retirement["marker_sha256"]
    ):
        raise CanaryEvidenceError("api_observer_retirement_invalid")
    return dict(marker)


def linux_peer_identity(sock: socket.socket) -> PeerIdentity:
    peercred = getattr(socket, "SO_PEERCRED", None)
    if peercred is None:
        raise CanaryEvidenceError("peer_credentials_unavailable")
    try:
        raw = sock.getsockopt(socket.SOL_SOCKET, peercred, _PEER_CREDENTIALS.size)
    except OSError as exc:
        raise CanaryEvidenceError("peer_credentials_unavailable") from exc
    if len(raw) != _PEER_CREDENTIALS.size:
        raise CanaryEvidenceError("peer_credentials_unavailable")
    return PeerIdentity(*_PEER_CREDENTIALS.unpack(raw))


def _require_peer(observed: PeerIdentity, expected: PeerIdentity, *, code: str) -> None:
    if not isinstance(observed, PeerIdentity) or observed != expected:
        raise CanaryEvidenceError(code)


def _socket_identity(path: Path, expected: SocketIdentity) -> str:
    try:
        resolved_parent = path.parent.resolve(strict=True)
        item = path.lstat()
    except OSError as exc:
        raise CanaryEvidenceError("socket_identity_invalid") from exc
    if (
        resolved_parent != path.parent
        or not stat.S_ISSOCK(item.st_mode)
        or item.st_nlink != 1
        or item.st_uid != expected.owner_uid
        or item.st_gid != expected.owner_gid
        or stat.S_IMODE(item.st_mode) != expected.mode
    ):
        raise CanaryEvidenceError("socket_identity_invalid")
    return _sha256_json(
        {
            "device": int(item.st_dev),
            "inode": int(item.st_ino),
            "mode": f"{expected.mode:04o}",
            "owner_uid": int(item.st_uid),
            "owner_gid": int(item.st_gid),
        }
    )


def _receive_exact(sock: socket.socket, size: int, *, code: str) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        try:
            chunk = sock.recv(remaining)
        except (OSError, socket.timeout) as exc:
            raise CanaryEvidenceError(code) from exc
        if not chunk:
            raise CanaryEvidenceError(code)
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _collector_exchange(
    endpoint: CollectorEndpoint,
    body: bytes,
    *,
    peer_getter: PeerGetter = linux_peer_identity,
) -> Mapping[str, Any]:
    if not body or len(body) > MAX_FRAME_BYTES:
        raise CanaryEvidenceError("collector_frame_invalid")
    before_identity = _socket_identity(
        endpoint.socket_path, endpoint.socket_identity
    )
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.set_inheritable(False)
    try:
        sock.settimeout(endpoint.connect_timeout_ms / 1000)
        sock.connect(str(endpoint.socket_path))
        _require_peer(
            peer_getter(sock), endpoint.expected_peer, code="collector_peer_invalid"
        )
        sock.settimeout(endpoint.ack_timeout_ms / 1000)
        sock.sendall(_FRAME_HEADER.pack(len(body)) + body)
        raw_header = _receive_exact(sock, _FRAME_HEADER.size, code="collector_ack_invalid")
        (size,) = _FRAME_HEADER.unpack(raw_header)
        if not 1 < size <= MAX_ACK_BYTES:
            raise CanaryEvidenceError("collector_ack_invalid")
        raw_ack = _receive_exact(sock, size, code="collector_ack_invalid")
        _require_peer(
            peer_getter(sock), endpoint.expected_peer, code="collector_peer_invalid"
        )
    except CanaryEvidenceError:
        raise
    except (OSError, socket.timeout) as exc:
        raise CanaryEvidenceError("collector_transport_failed") from exc
    finally:
        sock.close()
    after_identity = _socket_identity(endpoint.socket_path, endpoint.socket_identity)
    if before_identity != after_identity:
        raise CanaryEvidenceError("collector_socket_replaced")
    return _strict_json(raw_ack, code="collector_ack_invalid")


def _validate_ack(
    value: Mapping[str, Any], *, sequence: int, frame_sha256: str
) -> str:
    ack = _strict_mapping(
        value,
        fields=frozenset(
            {
                "schema",
                "sequence",
                "accepted",
                "frame_sha256",
                "collector_receipt_sha256",
            }
        ),
        code="collector_ack_invalid",
    )
    if (
        ack["schema"] != ACK_SCHEMA
        or ack["sequence"] != sequence
        or ack["accepted"] is not True
        or ack["frame_sha256"] != frame_sha256
    ):
        raise CanaryEvidenceError("collector_ack_invalid")
    return _digest(ack["collector_receipt_sha256"], code="collector_ack_invalid")


def _private_probe_frame(fixture: CanaryFixture, now_unix_ms: int) -> Mapping[str, Any]:
    target = fixture.public_target
    return {
        "protocol": "discord-edge.v1",
        "request_id": str(uuid.uuid4()),
        "sequence": 1,
        "deadline_unix_ms": now_unix_ms + 5_000,
        "operation": "public.message.send",
        "target": {
            "target_type": "private_channel",
            "guild_id": target["guild_id"],
            "channel_id": target["channel_id"],
        },
        "payload": {"content": "muncho-canary-private-target-probe"},
        "idempotency_key": "private-probe:" + fixture.value["canary_run_id"],
        # Parsing rejects the forbidden target before capability parsing.  No
        # signing material exists in this process and none is needed for this
        # negative protocol-boundary probe.
        "capability": {},
    }


def _run_private_probe(
    endpoint: EdgeEndpoint,
    fixture: CanaryFixture,
    expected_socket_identity_sha256: str,
    now_unix_ms: int,
    *,
    peer_getter: PeerGetter = linux_peer_identity,
) -> Mapping[str, Any]:
    observed_identity = _socket_identity(
        endpoint.socket_path, endpoint.socket_identity
    )
    if observed_identity != expected_socket_identity_sha256:
        raise CanaryEvidenceError("edge_socket_replaced")
    frame = _private_probe_frame(fixture, now_unix_ms)
    body = _canonical_bytes(frame)
    if not body or len(body) > 64 * 1024:
        raise CanaryEvidenceError("private_probe_frame_invalid")
    attempt_sha256 = _sha256_bytes(body)

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.set_inheritable(False)
    try:
        sock.settimeout(endpoint.connect_timeout_ms / 1000)
        sock.connect(str(endpoint.socket_path))
        _require_peer(peer_getter(sock), endpoint.expected_peer, code="edge_peer_invalid")
        sock.settimeout(endpoint.response_timeout_ms / 1000)
        sock.sendall(_FRAME_HEADER.pack(len(body)) + body)
        try:
            first_response_byte = sock.recv(1)
        except (OSError, socket.timeout) as exc:
            raise CanaryEvidenceError("private_probe_no_close") from exc
        if first_response_byte != b"":
            raise CanaryEvidenceError("private_probe_response_observed")
        _require_peer(peer_getter(sock), endpoint.expected_peer, code="edge_peer_invalid")
    except CanaryEvidenceError:
        raise
    except (OSError, socket.timeout) as exc:
        raise CanaryEvidenceError("private_probe_transport_failed") from exc
    finally:
        sock.close()
    after_identity = _socket_identity(endpoint.socket_path, endpoint.socket_identity)
    if after_identity != expected_socket_identity_sha256:
        raise CanaryEvidenceError("edge_socket_replaced")
    return {
        "discord_edge_service_identity_sha256": endpoint.service_identity_sha256,
        "socket_identity_sha256": after_identity,
        "attempt_frame_sha256": attempt_sha256,
        "attempted_operation": "public.message.send",
        "attempted_target_type": "private_channel",
        "connection_closed_without_response": True,
        "signed_receipt_observed": False,
        "observed_at_unix_ms": now_unix_ms,
    }


def _default_writer_call(operation: Any, payload: Mapping[str, Any], **kwargs: Any) -> Mapping[str, Any]:
    from gateway.canonical_writer_boundary import canonical_writer_call

    return canonical_writer_call(operation, payload, **kwargs)


def _default_runtime_envelope() -> Mapping[str, Any]:
    from gateway.canonical_writer_boundary import trusted_runtime_envelope

    return trusted_runtime_envelope()


def _finite_timestamp_ms(value: Any, *, code: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CanaryEvidenceError(code)
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise CanaryEvidenceError(code)
    return int(number * 1000)


def _reasoning_effort(request: Mapping[str, Any]) -> str:
    body = request.get("body")
    if not isinstance(body, Mapping):
        raise CanaryEvidenceError("api_request_projection_invalid")
    reasoning = body.get("reasoning")
    effort = reasoning.get("effort") if isinstance(reasoning, Mapping) else None
    if effort not in {"high", "max"}:
        raise CanaryEvidenceError("api_request_effort_invalid")
    return str(effort)


def _assistant_tool_call_ids(response: Mapping[str, Any]) -> list[str]:
    assistant = response.get("assistant_message")
    if not isinstance(assistant, Mapping):
        raise CanaryEvidenceError("api_response_projection_invalid")
    raw_calls = assistant.get("tool_calls")
    if not isinstance(raw_calls, list) or len(raw_calls) > MAX_TOOL_CALLS:
        raise CanaryEvidenceError("api_response_projection_invalid")
    result: list[str] = []
    for item in raw_calls:
        if not isinstance(item, Mapping):
            raise CanaryEvidenceError("api_response_projection_invalid")
        call_id = item.get("id")
        if call_id is None and isinstance(item.get("provider_data"), Mapping):
            call_id = item["provider_data"].get("call_id")
        result.append(_safe_id(call_id, code="api_response_projection_invalid"))
    if len(set(result)) != len(result):
        raise CanaryEvidenceError("api_response_projection_invalid")
    return result


def _parsed_result(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str):
        return None
    raw = value.encode("utf-8", errors="strict")
    if not raw or len(raw) > MAX_TOOL_PROJECTION_BYTES:
        return None
    try:
        return _strict_json(raw, code="tool_result_invalid")
    except CanaryEvidenceError:
        return None


class CanaryEvidencePlugin:
    """Single-process observer state for one isolated canary session."""

    def __init__(
        self,
        config: CanaryEvidenceConfig,
        *,
        collector_transport: CollectorTransport = _collector_exchange,
        socket_inspector: SocketInspector = _socket_identity,
        edge_probe: EdgeProbe = _run_private_probe,
        writer_call: Callable[..., Mapping[str, Any]] = _default_writer_call,
        runtime_envelope: RuntimeEnvelope = _default_runtime_envelope,
        clock_ms: Callable[[], int] = lambda: int(time.time() * 1000),
    ) -> None:
        self.config = config
        self._collector_transport = collector_transport
        self._socket_inspector = socket_inspector
        self._edge_probe = edge_probe
        self._writer_call = writer_call
        self._runtime_envelope = runtime_envelope
        self._clock_ms = clock_ms
        self._owner_pid = os.getpid()
        self._lock = threading.RLock()
        self._sequence = 0
        self._ready = False
        self._session_started = False
        self._session_ended = False
        self._session_id: str | None = None
        self._turn_id: str | None = None
        self._session_bound = False
        self._model_calls = 0
        self._tool_calls = 0
        self._api_ordinals: dict[str, int] = {}
        self._pending_api: set[str] = set()
        self._assistant_tool_ids: set[str] = set()
        self._observed_tool_ids: set[str] = set()

    def _require_owner(self) -> None:
        if os.getpid() != self._owner_pid:
            raise CanaryEvidenceError("plugin_wrong_process")

    def _now(self) -> int:
        value = self._clock_ms()
        if type(value) is not int or not (
            self.config.fixture.valid_from_unix_ms
            <= value
            <= self.config.fixture.valid_until_unix_ms
        ):
            raise CanaryEvidenceError("observation_outside_fixture_window")
        return value

    def _bind_session(self, session_id: Any) -> str:
        observed = _safe_id(session_id, code="session_binding_invalid")
        if self._session_id is None:
            self._session_id = observed
        elif self._session_id != observed:
            raise CanaryEvidenceError("second_session_forbidden")
        return observed

    def _bind_turn(self, turn_id: Any) -> str:
        observed = _safe_id(turn_id, code="turn_binding_invalid")
        if self._turn_id is None:
            self._turn_id = observed
        elif self._turn_id != observed:
            raise CanaryEvidenceError("second_turn_forbidden")
        return observed

    def _emit(
        self,
        event: str,
        *,
        session_id: str | None,
        turn_id: str | None,
        payload: Mapping[str, Any],
        observed_at_unix_ms: int | None = None,
    ) -> str:
        if event not in _EVENTS:
            raise CanaryEvidenceError("frame_event_invalid")
        if event in _READY_EVENTS:
            correlation_valid = session_id is None and turn_id is None
        elif event in _START_EVENTS:
            correlation_valid = isinstance(session_id, str) and turn_id is None
        else:
            correlation_valid = isinstance(session_id, str) and isinstance(
                turn_id, str
            )
        if not correlation_valid:
            raise CanaryEvidenceError("frame_event_invalid")
        observed_at = self._now() if observed_at_unix_ms is None else observed_at_unix_ms
        if not (
            self.config.fixture.valid_from_unix_ms
            <= observed_at
            <= self.config.fixture.valid_until_unix_ms
        ):
            raise CanaryEvidenceError("observation_outside_fixture_window")
        with self._lock:
            self._sequence += 1
            sequence = self._sequence
            frame = {
                "schema": FRAME_SCHEMA,
                "sequence": sequence,
                "event": event,
                "release_sha": self.config.release_sha,
                "release_sha256": self.config.release_sha256,
                "canary_run_id": self.config.canary_run_id,
                "case_id": self.config.case_id,
                "fixture_sha256": self.config.fixture.sha256,
                "collector_service_identity_sha256": (
                    self.config.collector.service_identity_sha256
                ),
                "discord_edge_service_identity_sha256": (
                    self.config.discord_edge.service_identity_sha256
                ),
                "session_id": session_id,
                "turn_id": turn_id,
                "observed_at_unix_ms": observed_at,
                "payload": dict(payload),
            }
            body = _canonical_bytes(frame)
            if not body or len(body) > MAX_FRAME_BYTES:
                raise CanaryEvidenceError("collector_frame_invalid")
            frame_sha256 = _sha256_bytes(body)
            ack = self._collector_transport(self.config.collector, body)
            return _validate_ack(
                ack, sequence=sequence, frame_sha256=frame_sha256
            )

    def start(self, *, module_origin: str, module_sha256: str) -> None:
        """Prove plugin/config/socket readiness before any hook is admitted."""

        with self._lock:
            self._require_owner()
            if self._ready or self._sequence != 0 or self._session_id is not None:
                raise CanaryEvidenceError("plugin_start_replayed")
            origin = _absolute_path(module_origin, code="plugin_module_invalid")
            module_digest = _digest(module_sha256, code="plugin_module_invalid")
            collector_socket = self._socket_inspector(
                self.config.collector.socket_path,
                self.config.collector.socket_identity,
            )
            edge_socket = self._socket_inspector(
                self.config.discord_edge.socket_path,
                self.config.discord_edge.socket_identity,
            )
            self._emit(
                "plugin_ready",
                session_id=None,
                turn_id=None,
                payload={
                    "plugin_name": "muncho_canary_evidence",
                    "gateway_pid": self._owner_pid,
                    "config_sha256": self.config.config_sha256,
                    "fixture_sha256": self.config.fixture.sha256,
                    "release_sha": self.config.release_sha,
                    "release_sha256": self.config.release_sha256,
                    "api_session_key_sha256": (
                        self.config.fixture.value["api_session_key_sha256"]
                    ),
                    "collector_service_identity_sha256": (
                        self.config.collector.service_identity_sha256
                    ),
                    "collector_socket_identity_sha256": collector_socket,
                    "discord_edge_service_identity_sha256": (
                        self.config.discord_edge.service_identity_sha256
                    ),
                    "discord_edge_socket_identity_sha256": edge_socket,
                    "module_origin": str(origin),
                    "module_sha256": module_digest,
                },
            )
            self._ready = True

    def _bind_api_session(self, session_id: str) -> bool:
        """Prove only the mechanical API-session/fixture digest binding."""

        try:
            runtime = self._runtime_envelope()
            if not isinstance(runtime, Mapping):
                raise CanaryEvidenceError("api_session_binding_invalid")
            observed_session_digest = _digest(
                runtime.get("session_key_sha256"),
                code="api_session_binding_invalid",
            )
            if (
                runtime.get("platform") != "api_server"
                or runtime.get("session_id") != session_id
                or observed_session_digest
                != self.config.fixture.value["api_session_key_sha256"]
            ):
                raise CanaryEvidenceError("api_session_binding_invalid")
            payload = {
                "success": True,
                "runtime_platform": "api_server",
                "runtime_session_id": session_id,
                "session_key_sha256": observed_session_digest,
                "fixture_sha256": self.config.fixture.sha256,
            }
        except Exception:
            self._emit(
                "api_session_bound",
                session_id=session_id,
                turn_id=None,
                payload={"success": False, "failure_code": "session_binding_failed"},
            )
            return False
        self._emit(
            "api_session_bound",
            session_id=session_id,
            turn_id=None,
            payload=payload,
        )
        return True

    def on_session_start(
        self,
        *,
        session_id: str = "",
        model: str = "",
        platform: str = "",
        **_kwargs: Any,
    ) -> None:
        with self._lock:
            self._require_owner()
            if not self._ready or self._session_started:
                raise CanaryEvidenceError("session_start_replayed")
            session = self._bind_session(session_id)
            if platform != "api_server" or model != self.config.fixture.model_route["model"]:
                raise CanaryEvidenceError("session_route_invalid")
            self._session_started = True
            self._session_bound = self._bind_api_session(session)
            if not self._session_bound:
                return
            edge_identity = self._socket_inspector(
                self.config.discord_edge.socket_path,
                self.config.discord_edge.socket_identity,
            )
            self._emit(
                "private_target_probe_ready",
                session_id=session,
                turn_id=None,
                payload={
                    "discord_edge_service_identity_sha256": (
                        self.config.discord_edge.service_identity_sha256
                    ),
                    "socket_identity_sha256": edge_identity,
                    "attempted_operation": "public.message.send",
                    "attempted_target_type": "private_channel",
                    "collector_snapshot_barrier": "before_probe",
                },
            )
            probe = self._edge_probe(
                self.config.discord_edge,
                self.config.fixture,
                edge_identity,
                self._now(),
            )
            required = {
                "discord_edge_service_identity_sha256",
                "socket_identity_sha256",
                "attempt_frame_sha256",
                "attempted_operation",
                "attempted_target_type",
                "connection_closed_without_response",
                "signed_receipt_observed",
                "observed_at_unix_ms",
            }
            if (
                not isinstance(probe, Mapping)
                or set(probe) != required
                or probe["discord_edge_service_identity_sha256"]
                != self.config.discord_edge.service_identity_sha256
                or probe["socket_identity_sha256"] != edge_identity
                or probe["attempted_operation"] != "public.message.send"
                or probe["attempted_target_type"] != "private_channel"
                or probe["connection_closed_without_response"] is not True
                or probe["signed_receipt_observed"] is not False
            ):
                raise CanaryEvidenceError("private_probe_result_invalid")
            _digest(probe["attempt_frame_sha256"], code="private_probe_result_invalid")
            observed = _positive_int(
                probe["observed_at_unix_ms"], code="private_probe_result_invalid"
            )
            self._emit(
                "private_target_probe_result",
                session_id=session,
                turn_id=None,
                payload=dict(probe),
                observed_at_unix_ms=observed,
            )

    def pre_api_request(
        self,
        *,
        session_id: str = "",
        turn_id: str = "",
        task_id: str = "",
        api_request_id: str = "",
        platform: str = "",
        model: str = "",
        provider: str = "",
        base_url: str = "",
        api_mode: str = "",
        api_call_count: Any = None,
        started_at: Any = None,
        request: Any = None,
        **_kwargs: Any,
    ) -> None:
        with self._lock:
            self._require_owner()
            if not self._ready or not self._session_started or self._session_ended:
                raise CanaryEvidenceError("api_request_projection_invalid")
            session = self._bind_session(session_id)
            turn = self._bind_turn(turn_id)
            _safe_id(task_id, code="api_request_projection_invalid")
            api_id = _safe_id(api_request_id, code="api_request_projection_invalid")
            if not self._session_bound or api_id in self._api_ordinals:
                raise CanaryEvidenceError("api_request_projection_invalid")
            if self._model_calls >= MAX_MODEL_CALLS:
                raise CanaryEvidenceError("model_call_limit_exceeded")
            route = self.config.fixture.model_route
            if (
                platform != "api_server"
                or model != route["model"]
                or provider != route["provider"]
                or base_url != route["base_url"]
                or api_mode != route["api_mode"]
                or type(api_call_count) is not int
                or not isinstance(request, Mapping)
            ):
                raise CanaryEvidenceError("api_request_projection_invalid")
            self._model_calls += 1
            ordinal = self._model_calls
            self._api_ordinals[api_id] = ordinal
            self._pending_api.add(api_id)
            started_ms = _finite_timestamp_ms(
                started_at, code="api_request_projection_invalid"
            )
            payload = {
                "task_id": task_id,
                "api_request_id": api_id,
                "request_ordinal": ordinal,
                "runtime_api_call_count": api_call_count,
                "provider": provider,
                "api_mode": api_mode,
                "base_url": base_url,
                "model": model,
                "reasoning_effort": _reasoning_effort(request),
                "api_request_sha256": _bounded_sha256_json(
                    request,
                    maximum=MAX_FRAME_BYTES,
                    code="api_request_projection_invalid",
                ),
                "started_at_unix_ms": started_ms,
            }
            self._emit(
                "pre_api_request",
                session_id=session,
                turn_id=turn,
                payload=payload,
                observed_at_unix_ms=started_ms,
            )

    def post_api_request(
        self,
        *,
        session_id: str = "",
        turn_id: str = "",
        task_id: str = "",
        api_request_id: str = "",
        platform: str = "",
        model: str = "",
        provider: str = "",
        base_url: str = "",
        api_mode: str = "",
        ended_at: Any = None,
        finish_reason: Any = None,
        response_model: Any = None,
        response: Any = None,
        **_kwargs: Any,
    ) -> None:
        with self._lock:
            self._require_owner()
            if not self._ready or not self._session_started or self._session_ended:
                raise CanaryEvidenceError("api_response_projection_invalid")
            session = self._bind_session(session_id)
            turn = self._bind_turn(turn_id)
            _safe_id(task_id, code="api_response_projection_invalid")
            api_id = _safe_id(api_request_id, code="api_response_projection_invalid")
            if api_id not in self._pending_api or not isinstance(response, Mapping):
                raise CanaryEvidenceError("api_response_projection_invalid")
            route = self.config.fixture.model_route
            if (
                platform != "api_server"
                or model != route["model"]
                or provider != route["provider"]
                or base_url != route["base_url"]
                or api_mode != route["api_mode"]
                or response_model != route["model"]
                or not isinstance(finish_reason, str)
                or not finish_reason
            ):
                raise CanaryEvidenceError("api_response_projection_invalid")
            call_ids = _assistant_tool_call_ids(response)
            if self._assistant_tool_ids.intersection(call_ids):
                raise CanaryEvidenceError("duplicate_assistant_tool_call_id")
            self._assistant_tool_ids.update(call_ids)
            self._pending_api.remove(api_id)
            ended_ms = _finite_timestamp_ms(
                ended_at, code="api_response_projection_invalid"
            )
            self._emit(
                "post_api_request",
                session_id=session,
                turn_id=turn,
                payload={
                    "task_id": task_id,
                    "api_request_id": api_id,
                    "request_ordinal": self._api_ordinals[api_id],
                    "finish_reason": finish_reason,
                    "response_model": response_model,
                    "response_payload_sha256": _bounded_sha256_json(
                        response,
                        maximum=MAX_FRAME_BYTES,
                        code="api_response_projection_invalid",
                    ),
                    "assistant_tool_call_ids": call_ids,
                    "response_observed_at_unix_ms": ended_ms,
                },
                observed_at_unix_ms=ended_ms,
            )

    def post_tool_call(
        self,
        *,
        session_id: str = "",
        turn_id: str = "",
        task_id: str = "",
        api_request_id: str = "",
        tool_call_id: str = "",
        tool_name: str = "",
        args: Any = None,
        result: Any = None,
        duration_ms: Any = 0,
        status: Any = None,
        error_type: Any = None,
        **_kwargs: Any,
    ) -> None:
        with self._lock:
            self._require_owner()
            if not self._ready or not self._session_started or self._session_ended:
                raise CanaryEvidenceError("tool_observation_invalid")
            session = self._bind_session(session_id)
            turn = self._bind_turn(turn_id)
            _safe_id(task_id, code="tool_observation_invalid")
            api_id = _safe_id(api_request_id, code="tool_observation_invalid")
            call_id = _safe_id(tool_call_id, code="tool_observation_invalid")
            if (
                api_id not in self._api_ordinals
                or call_id not in self._assistant_tool_ids
                or call_id in self._observed_tool_ids
                or not isinstance(tool_name, str)
                or not tool_name
                or not isinstance(args, Mapping)
                or type(duration_ms) is not int
                or duration_ms < 0
                or not isinstance(status, str)
                or not status
            ):
                raise CanaryEvidenceError("tool_observation_invalid")
            if self._tool_calls >= MAX_TOOL_CALLS:
                raise CanaryEvidenceError("tool_call_limit_exceeded")
            self._tool_calls += 1
            self._observed_tool_ids.add(call_id)
            parsed = _parsed_result(result)
            payload: dict[str, Any] = {
                "task_id": task_id,
                "api_request_id": api_id,
                "produced_by_model_call_ordinal": self._api_ordinals[api_id],
                "tool_call_ordinal": self._tool_calls,
                "tool_call_id": call_id,
                "tool_name": tool_name,
                "duration_ms": duration_ms,
                "status": status,
                "error_type": error_type if isinstance(error_type, str) else None,
                "args_sha256": _bounded_sha256_json(
                    args,
                    maximum=MAX_FRAME_BYTES,
                    code="tool_observation_invalid",
                ),
                "result_sha256": _bounded_sha256_json(
                    parsed if parsed is not None else result,
                    maximum=MAX_FRAME_BYTES,
                    code="tool_observation_invalid",
                ),
            }
            if tool_name == "todo":
                directive = args.get("reasoning")
                control = parsed.get("reasoning_control") if parsed is not None else None
                if directive is not None:
                    if not isinstance(directive, Mapping) or not isinstance(control, Mapping):
                        raise CanaryEvidenceError("todo_reasoning_receipt_invalid")
                    payload["reasoning_directive"] = dict(directive)
                    payload["reasoning_control"] = dict(control)
            elif tool_name in _PROJECTED_TOOL_RESULTS:
                if parsed is None:
                    raise CanaryEvidenceError("tool_result_invalid")
                projection = dict(parsed)
                if len(_canonical_bytes(projection)) > MAX_TOOL_PROJECTION_BYTES:
                    raise CanaryEvidenceError("tool_result_invalid")
                payload["result_projection"] = projection
            self._emit(
                "post_tool_call",
                session_id=session,
                turn_id=turn,
                payload=payload,
            )

    def on_session_end(
        self,
        *,
        session_id: str = "",
        turn_id: str = "",
        task_id: str = "",
        completed: Any = None,
        interrupted: Any = None,
        model: str = "",
        platform: str = "",
        **_kwargs: Any,
    ) -> None:
        with self._lock:
            self._require_owner()
            if not self._ready or not self._session_started or self._session_ended:
                raise CanaryEvidenceError("session_end_invalid")
            session = self._bind_session(session_id)
            turn = self._bind_turn(turn_id)
            _safe_id(task_id, code="session_end_invalid")
            if (
                not self._session_bound
                or platform != "api_server"
                or model != self.config.fixture.model_route["model"]
                or type(completed) is not bool
                or type(interrupted) is not bool
                or self._pending_api
                or self._assistant_tool_ids != self._observed_tool_ids
            ):
                raise CanaryEvidenceError("session_end_invalid")
            from gateway.canonical_writer_protocol import CanonicalWriterOperation

            readback = self._writer_call(
                CanonicalWriterOperation.CASE_QUERY,
                {
                    "case_id": self.config.case_id,
                    "thread_id": "",
                    "limit": 200,
                    "view": "resume_bundle",
                },
            )
            if not isinstance(readback, Mapping):
                raise CanaryEvidenceError("canonical_readback_invalid")
            readback_value = dict(readback)
            if len(_canonical_bytes(readback_value)) > MAX_TOOL_PROJECTION_BYTES:
                raise CanaryEvidenceError("canonical_readback_invalid")
            self._emit(
                "canonical_case_readback",
                session_id=session,
                turn_id=turn,
                payload={
                    "writer_request_id": _uuid(
                        readback_value.get("request_id"),
                        code="canonical_readback_invalid",
                    ),
                    "query_view": "resume_bundle",
                    "query_limit": 200,
                    "readback_sha256": _sha256_json(readback_value),
                    "readback": readback_value,
                },
            )
            self._emit(
                "session_end",
                session_id=session,
                turn_id=turn,
                payload={
                    "task_id": task_id,
                    "completed": completed,
                    "interrupted": interrupted,
                    "model": model,
                    "platform": platform,
                    "model_call_count": self._model_calls,
                    "tool_call_count": self._tool_calls,
                    # The existing hook does not expose partial/failed,
                    # turn_exit_reason, API run/message IDs, or final response.
                    # The trusted collector must join those exact values from
                    # the authenticated loopback SSE terminal event rather
                    # than this plugin fabricating them.
                    "terminal_fields_source": "authenticated_loopback_sse",
                },
            )
            self._session_ended = True


def _goal_request_identity(request: Any) -> tuple[str, str]:
    """Return only system-prompt and tool-schema digests for one request."""

    if not isinstance(request, Mapping):
        raise CanaryEvidenceError("goal_api_request_invalid")
    body = request.get("body")
    if not isinstance(body, Mapping):
        raise CanaryEvidenceError("goal_api_request_invalid")
    instructions = body.get("instructions")
    tools = body.get("tools")
    if not isinstance(instructions, str) or not instructions or not isinstance(
        tools, list
    ):
        raise CanaryEvidenceError("goal_api_request_invalid")
    return (
        _sha256_bytes(instructions.encode("utf-8", errors="strict")),
        _bounded_sha256_json(
            tools,
            maximum=MAX_FRAME_BYTES,
            code="goal_api_request_invalid",
        ),
    )


class GoalContinuationEvidencePlugin:
    """Observer-only Discord goal projection with one chain per gateway PID.

    The root collector admits each process segment against systemd.  This
    process emits no user/model prose and never interprets completion from
    text: a goal outcome exists only when the model called ``todo`` with the
    structured ``goal_outcome`` object and the tool result says that exact
    write was recorded.
    """

    def __init__(
        self,
        config: GoalObserverConfig,
        *,
        collector_transport: CollectorTransport = _collector_exchange,
        socket_inspector: SocketInspector = _socket_identity,
        writer_call: WriterCall = _default_writer_call,
        clock_ms: Callable[[], int] = lambda: int(time.time() * 1000),
    ) -> None:
        self.config = config
        self._collector_transport = collector_transport
        self._socket_inspector = socket_inspector
        self._writer_call = writer_call
        self._clock_ms = clock_ms
        self._owner_pid = os.getpid()
        self._segment_id = uuid.uuid4().hex
        self._sequence = 0
        self._ready = False
        self._session_id: str | None = None
        self._model_calls = 0
        self._api_calls: dict[str, dict[str, Any]] = {}
        self._tool_authority: dict[str, tuple[str, str]] = {}
        self._observed_tool_ids: set[str] = set()
        self._canonical_case_ids: set[str] = set()
        self._lock = threading.RLock()

    def _require_owner(self) -> None:
        if os.getpid() != self._owner_pid:
            raise CanaryEvidenceError("goal_plugin_wrong_process")

    def _now(self) -> int:
        value = self._clock_ms()
        if type(value) is not int or not (
            self.config.valid_from_unix_ms
            <= value
            <= self.config.valid_until_unix_ms
        ):
            raise CanaryEvidenceError("goal_observation_outside_window")
        return value

    def _bind_session(self, session_id: Any) -> str:
        observed = _safe_id(session_id, code="goal_session_binding_invalid")
        if self._session_id is None:
            self._session_id = observed
        elif self._session_id != observed:
            raise CanaryEvidenceError("goal_second_session_forbidden")
        return observed

    def _emit(
        self,
        event: str,
        *,
        session_id: str | None,
        turn_id: str | None,
        payload: Mapping[str, Any],
        observed_at_unix_ms: int | None = None,
    ) -> str:
        if event not in _GOAL_EVENTS:
            raise CanaryEvidenceError("goal_frame_event_invalid")
        if event == "goal_plugin_ready":
            if session_id is not None or turn_id is not None:
                raise CanaryEvidenceError("goal_frame_event_invalid")
        elif not isinstance(session_id, str) or not isinstance(turn_id, str):
            raise CanaryEvidenceError("goal_frame_event_invalid")
        observed_at = self._now() if observed_at_unix_ms is None else observed_at_unix_ms
        if not (
            self.config.valid_from_unix_ms
            <= observed_at
            <= self.config.valid_until_unix_ms
        ):
            raise CanaryEvidenceError("goal_observation_outside_window")
        with self._lock:
            self._require_owner()
            self._sequence += 1
            frame = {
                "schema": GOAL_FRAME_SCHEMA,
                "segment_id": self._segment_id,
                "sequence": self._sequence,
                "event": event,
                "release_sha": self.config.release_sha,
                "release_sha256": self.config.release_sha256,
                "run_id": self.config.run_id,
                "fixture_sha256": self.config.fixture_sha256,
                "collector_service_identity_sha256": (
                    self.config.collector.service_identity_sha256
                ),
                "session_id": session_id,
                "turn_id": turn_id,
                "observed_at_unix_ms": observed_at,
                "payload": dict(payload),
            }
            body = _canonical_bytes(frame)
            if not 1 < len(body) <= MAX_FRAME_BYTES:
                raise CanaryEvidenceError("goal_collector_frame_invalid")
            frame_sha256 = _sha256_bytes(body)
            ack = self._collector_transport(self.config.collector, body)
            return _validate_ack(
                ack,
                sequence=self._sequence,
                frame_sha256=frame_sha256,
            )

    def start(self, *, module_origin: str, module_sha256: str) -> None:
        with self._lock:
            self._require_owner()
            if self._ready or self._sequence != 0:
                raise CanaryEvidenceError("goal_plugin_start_replayed")
            origin = _absolute_path(module_origin, code="plugin_module_invalid")
            module_digest = _digest(module_sha256, code="plugin_module_invalid")
            socket_identity = self._socket_inspector(
                self.config.collector.socket_path,
                self.config.collector.socket_identity,
            )
            self._emit(
                "goal_plugin_ready",
                session_id=None,
                turn_id=None,
                payload={
                    "plugin_name": "muncho_canary_evidence.goal_continuation",
                    "gateway_pid": self._owner_pid,
                    "config_sha256": self.config.config_sha256,
                    "fixture_sha256": self.config.fixture_sha256,
                    "collector_service_identity_sha256": (
                        self.config.collector.service_identity_sha256
                    ),
                    "collector_socket_identity_sha256": socket_identity,
                    "module_origin": str(origin),
                    "module_sha256": module_digest,
                },
            )
            self._ready = True

    @staticmethod
    def _is_goal_platform(platform: Any) -> bool:
        return platform == "discord"

    def on_session_start(
        self,
        *,
        session_id: str = "",
        model: str = "",
        platform: str = "",
        **_kwargs: Any,
    ) -> None:
        if not self._is_goal_platform(platform):
            return
        self._require_owner()
        if model != self.config.model_route["model"]:
            raise CanaryEvidenceError("goal_session_route_invalid")
        self._bind_session(session_id)

    def pre_api_request(
        self,
        *,
        session_id: str = "",
        turn_id: str = "",
        task_id: str = "",
        api_request_id: str = "",
        platform: str = "",
        model: str = "",
        provider: str = "",
        base_url: str = "",
        api_mode: str = "",
        api_call_count: Any = None,
        started_at: Any = None,
        request: Any = None,
        **_kwargs: Any,
    ) -> None:
        if not self._is_goal_platform(platform):
            return
        with self._lock:
            self._require_owner()
            if not self._ready:
                raise CanaryEvidenceError("goal_api_request_invalid")
            session = self._bind_session(session_id)
            turn = _safe_id(turn_id, code="goal_api_request_invalid")
            task = _safe_id(task_id, code="goal_api_request_invalid")
            api_id = _safe_id(api_request_id, code="goal_api_request_invalid")
            route = self.config.model_route
            if (
                model != route["model"]
                or provider != route["provider"]
                or base_url != route["base_url"]
                or api_mode != route["api_mode"]
                or type(api_call_count) is not int
                or api_id in self._api_calls
                or self._model_calls >= MAX_MODEL_CALLS
            ):
                raise CanaryEvidenceError("goal_api_request_invalid")
            prompt_sha256, tools_sha256 = _goal_request_identity(request)
            started_ms = _finite_timestamp_ms(
                started_at, code="goal_api_request_invalid"
            )
            self._model_calls += 1
            ordinal = self._model_calls
            projection = {
                "request_ordinal": ordinal,
                "task_id_sha256": _sha256_bytes(task.encode("utf-8")),
                "api_request_id_sha256": _sha256_bytes(api_id.encode("utf-8")),
                "runtime_api_call_count": api_call_count,
                "provider": provider,
                "api_mode": api_mode,
                "model": model,
                "base_url_sha256": _sha256_bytes(base_url.encode("utf-8")),
                "system_prompt_sha256": prompt_sha256,
                "tool_schema_sha256": tools_sha256,
                "reasoning_effort": _reasoning_effort(request),
                "started_at_unix_ms": started_ms,
            }
            self._api_calls[api_id] = {
                **projection,
                "session_id": session,
                "turn_id": turn,
                "completed": False,
            }
            self._emit(
                "goal_pre_api_request",
                session_id=session,
                turn_id=turn,
                payload=projection,
                observed_at_unix_ms=started_ms,
            )

    def post_api_request(
        self,
        *,
        session_id: str = "",
        turn_id: str = "",
        api_request_id: str = "",
        platform: str = "",
        model: str = "",
        provider: str = "",
        base_url: str = "",
        api_mode: str = "",
        ended_at: Any = None,
        finish_reason: Any = None,
        response_model: Any = None,
        response: Any = None,
        **_kwargs: Any,
    ) -> None:
        if not self._is_goal_platform(platform):
            return
        with self._lock:
            self._require_owner()
            session = self._bind_session(session_id)
            turn = _safe_id(turn_id, code="goal_api_response_invalid")
            api_id = _safe_id(api_request_id, code="goal_api_response_invalid")
            call = self._api_calls.get(api_id)
            route = self.config.model_route
            if (
                not isinstance(call, Mapping)
                or call.get("completed") is not False
                or call.get("session_id") != session
                or call.get("turn_id") != turn
                or model != route["model"]
                or provider != route["provider"]
                or base_url != route["base_url"]
                or api_mode != route["api_mode"]
                or response_model != route["model"]
                or not isinstance(finish_reason, str)
                or not finish_reason
                or not isinstance(response, Mapping)
            ):
                raise CanaryEvidenceError("goal_api_response_invalid")
            call_ids = _assistant_tool_call_ids(response)
            if any(
                call_id in self._tool_authority
                or call_id in self._observed_tool_ids
                for call_id in call_ids
            ):
                raise CanaryEvidenceError("goal_api_response_invalid")
            ended_ms = _finite_timestamp_ms(
                ended_at, code="goal_api_response_invalid"
            )
            if ended_ms < call["started_at_unix_ms"]:
                raise CanaryEvidenceError("goal_api_response_invalid")
            call["completed"] = True
            for call_id in call_ids:
                self._tool_authority[call_id] = (api_id, turn)
            self._emit(
                "goal_post_api_request",
                session_id=session,
                turn_id=turn,
                payload={
                    "request_ordinal": call["request_ordinal"],
                    "api_request_id_sha256": call["api_request_id_sha256"],
                    "finish_reason_sha256": _sha256_bytes(
                        finish_reason.encode("utf-8", errors="strict")
                    ),
                    "response_model_sha256": _sha256_bytes(
                        response_model.encode("utf-8", errors="strict")
                    ),
                    "response_payload_sha256": _bounded_sha256_json(
                        response,
                        maximum=MAX_FRAME_BYTES,
                        code="goal_api_response_invalid",
                    ),
                    "assistant_tool_call_id_sha256s": [
                        _sha256_bytes(item.encode("utf-8", errors="strict"))
                        for item in call_ids
                    ],
                    "response_observed_at_unix_ms": ended_ms,
                },
                observed_at_unix_ms=ended_ms,
            )

    def post_tool_call(
        self,
        *,
        session_id: str = "",
        turn_id: str = "",
        api_request_id: str = "",
        tool_call_id: str = "",
        tool_name: str = "",
        args: Any = None,
        result: Any = None,
        status: Any = None,
        **_kwargs: Any,
    ) -> None:
        if self._session_id is None or session_id != self._session_id:
            return
        self._require_owner()
        session = self._bind_session(session_id)
        turn = _safe_id(turn_id, code="goal_tool_observation_invalid")
        api_id = _safe_id(api_request_id, code="goal_tool_observation_invalid")
        tool_id = _safe_id(tool_call_id, code="goal_tool_observation_invalid")
        if not isinstance(args, Mapping):
            raise CanaryEvidenceError("goal_tool_observation_invalid")
        authority = self._tool_authority.get(tool_id)
        if (
            authority != (api_id, turn)
            or tool_id in self._observed_tool_ids
        ):
            raise CanaryEvidenceError("goal_tool_observation_invalid")
        parsed = _parsed_result(result)
        self._observed_tool_ids.add(tool_id)
        if tool_name == "todo" and args.get("goal_outcome") is not None:
            outcome = _strict_mapping(
                args["goal_outcome"],
                fields=frozenset({"status", "reason"}),
                code="goal_outcome_observation_invalid",
            )
            reason = outcome["reason"]
            recorded = (
                parsed.get("goal_outcome", {}).get("recorded")
                if isinstance(parsed, Mapping)
                and isinstance(parsed.get("goal_outcome"), Mapping)
                else None
            )
            if (
                outcome["status"] not in {"continue", "complete", "blocked"}
                or not isinstance(reason, str)
                or not reason.strip()
                or status != "ok"
                or recorded is not True
            ):
                raise CanaryEvidenceError("goal_outcome_observation_invalid")
            self._emit(
                "goal_model_outcome",
                session_id=session,
                turn_id=turn,
                payload={
                    "api_request_id_sha256": _sha256_bytes(api_id.encode("utf-8")),
                    "tool_call_id_sha256": _sha256_bytes(tool_id.encode("utf-8")),
                    "outcome": outcome["status"],
                    "reason_sha256": _sha256_bytes(
                        reason.encode("utf-8", errors="strict")
                    ),
                    "recorded": True,
                    "result_sha256": _bounded_sha256_json(
                        parsed,
                        maximum=MAX_FRAME_BYTES,
                        code="goal_outcome_observation_invalid",
                    ),
                },
            )
            return
        canonical_result: Mapping[str, Any] | None = None
        if tool_name == "canonical_event_append" and isinstance(parsed, Mapping):
            canonical_result = parsed
        elif tool_name == "route_back_execute" and isinstance(parsed, Mapping):
            route_record = parsed.get("route_back_record")
            if (
                parsed.get("success") is True
                and parsed.get("status")
                in {
                    "ROUTE_BACK_EXECUTE_SENT",
                    "ROUTE_BACK_EXECUTE_SENT_RECONCILED",
                }
                and isinstance(route_record, Mapping)
                and route_record.get("event_type") == "route_back.sent"
            ):
                canonical_result = route_record
        if canonical_result is not None:
            event_id = canonical_result.get("event_id")
            case_id = canonical_result.get("case_id")
            event_type = canonical_result.get("event_type")
            if (
                status != "ok"
                or canonical_result.get("readback_verified") is not True
                or any(
                    not isinstance(item, str) or not item
                    for item in (event_id, case_id, event_type)
                )
            ):
                raise CanaryEvidenceError("goal_canonical_event_invalid")
            self._emit(
                "goal_canonical_event",
                session_id=session,
                turn_id=turn,
                payload={
                    "api_request_id_sha256": _sha256_bytes(api_id.encode("utf-8")),
                    "tool_call_id_sha256": _sha256_bytes(tool_id.encode("utf-8")),
                    "event_id": event_id,
                    "case_id": case_id,
                    "event_type": event_type,
                    "canonical_content_sha256": _digest(
                        canonical_result.get("canonical_content_sha256"),
                        code="goal_canonical_event_invalid",
                    ),
                    "idempotency_key_sha256": _sha256_bytes(
                        str(
                            canonical_result.get("idempotency_key") or ""
                        ).encode("utf-8")
                    ),
                    "readback_verified": True,
                    "result_sha256": _bounded_sha256_json(
                        parsed,
                        maximum=MAX_FRAME_BYTES,
                        code="goal_canonical_event_invalid",
                    ),
                },
            )
            self._canonical_case_ids.add(case_id)

    @staticmethod
    def _readback_plan_projection(readback: Mapping[str, Any]) -> Mapping[str, Any]:
        """Project typed plan identities from a real resume-bundle readback."""

        events = readback.get("events")
        support = readback.get("support_events")
        if (
            readback.get("status") != "ok"
            or readback.get("view") != "resume_bundle"
            or not isinstance(events, list)
            or not isinstance(support, list)
        ):
            raise CanaryEvidenceError("goal_canonical_readback_invalid")
        plans: list[dict[str, Any]] = []
        for item in [*events, *support]:
            if not isinstance(item, Mapping) or item.get("event_type") != "task.plan.updated":
                continue
            body = item.get("body")
            payload = body.get("payload") if isinstance(body, Mapping) else None
            plan = payload.get("plan") if isinstance(payload, Mapping) else None
            if not isinstance(plan, Mapping):
                raise CanaryEvidenceError("goal_canonical_readback_invalid")
            plan_id = plan.get("plan_id")
            revision = plan.get("revision")
            state = plan.get("state")
            cursor = plan.get("resume_cursor")
            next_step_id = (
                cursor.get("next_step_id") if isinstance(cursor, Mapping) else None
            )
            if (
                not isinstance(plan_id, str)
                or not plan_id
                or type(revision) is not int
                or revision <= 0
                or state not in {"active", "blocked", "completed", "cancelled"}
                or next_step_id is not None
                and (not isinstance(next_step_id, str) or not next_step_id)
            ):
                raise CanaryEvidenceError("goal_canonical_readback_invalid")
            event_id = item.get("event_id")
            if not isinstance(event_id, str) or not event_id:
                raise CanaryEvidenceError("goal_canonical_readback_invalid")
            plans.append(
                {
                    "event_id": event_id,
                    "plan_id": plan_id,
                    "revision": revision,
                    "state": state,
                    "next_step_id": next_step_id,
                }
            )
        if not plans:
            raise CanaryEvidenceError("goal_canonical_readback_invalid")
        return {
            "plan_identities": plans,
            "support_incomplete_reasons_sha256": _bounded_sha256_json(
                readback.get("support_incomplete_reasons"),
                maximum=MAX_TOOL_PROJECTION_BYTES,
                code="goal_canonical_readback_invalid",
            ),
            "missing_verification_event_ids_sha256": _bounded_sha256_json(
                readback.get("missing_verification_event_ids"),
                maximum=MAX_TOOL_PROJECTION_BYTES,
                code="goal_canonical_readback_invalid",
            ),
        }

    def on_session_end(
        self,
        *,
        session_id: str = "",
        turn_id: str = "",
        completed: Any = None,
        interrupted: Any = None,
        model: str = "",
        platform: str = "",
        **_kwargs: Any,
    ) -> None:
        if not self._is_goal_platform(platform):
            return
        self._require_owner()
        session = self._bind_session(session_id)
        turn = _safe_id(turn_id, code="goal_turn_end_invalid")
        if (
            model != self.config.model_route["model"]
            or type(completed) is not bool
            or type(interrupted) is not bool
            or any(
                call["turn_id"] == turn and call["completed"] is not True
                for call in self._api_calls.values()
            )
            or any(
                authority_turn == turn and tool_id not in self._observed_tool_ids
                for tool_id, (_api_id, authority_turn) in self._tool_authority.items()
            )
        ):
            raise CanaryEvidenceError("goal_turn_end_invalid")
        from gateway.canonical_writer_protocol import CanonicalWriterOperation

        for case_id in sorted(self._canonical_case_ids):
            readback = self._writer_call(
                CanonicalWriterOperation.CASE_QUERY,
                {
                    "case_id": case_id,
                    "thread_id": "",
                    "limit": 200,
                    "view": "resume_bundle",
                },
            )
            if not isinstance(readback, Mapping):
                raise CanaryEvidenceError("goal_canonical_readback_invalid")
            projection = self._readback_plan_projection(readback)
            self._emit(
                "goal_canonical_readback",
                session_id=session,
                turn_id=turn,
                payload={
                    "case_id": case_id,
                    "query_view": "resume_bundle",
                    "query_limit": 200,
                    "readback_sha256": _bounded_sha256_json(
                        readback,
                        maximum=MAX_TOOL_PROJECTION_BYTES,
                        code="goal_canonical_readback_invalid",
                    ),
                    **projection,
                },
            )
        self._emit(
            "goal_turn_end",
            session_id=session,
            turn_id=turn,
            payload={
                "completed": completed,
                "interrupted": interrupted,
                "model_sha256": _sha256_bytes(model.encode("utf-8")),
            },
        )


class CanaryEvidenceHookMultiplexer:
    """Exact five-hook, observer-only dispatch surface.

    Capability startup attests that every registered callback is a bound
    method on the single module ``_PLUGIN`` object.  Keep that identity stable
    while routing only by the gateway-provided platform/session envelope; no
    task text or model output is inspected here.
    """

    def __init__(
        self,
        api_plugin: CanaryEvidencePlugin | None,
        goal_plugin: GoalContinuationEvidencePlugin | None,
    ) -> None:
        if api_plugin is None and goal_plugin is None:
            raise CanaryEvidenceError("observer_mode_missing")
        self.api_plugin = api_plugin
        self.goal_plugin = goal_plugin

    def pre_api_request(self, **kwargs: Any) -> None:
        platform = kwargs.get("platform")
        if platform == "api_server" and self.api_plugin is not None:
            self.api_plugin.pre_api_request(**kwargs)
        elif platform == "discord" and self.goal_plugin is not None:
            self.goal_plugin.pre_api_request(**kwargs)

    def post_api_request(self, **kwargs: Any) -> None:
        platform = kwargs.get("platform")
        if platform == "api_server" and self.api_plugin is not None:
            self.api_plugin.post_api_request(**kwargs)
        elif platform == "discord" and self.goal_plugin is not None:
            self.goal_plugin.post_api_request(**kwargs)

    def post_tool_call(self, **kwargs: Any) -> None:
        session_id = kwargs.get("session_id")
        if (
            self.api_plugin is not None
            and self.api_plugin._session_id == session_id
        ):
            self.api_plugin.post_tool_call(**kwargs)
        if (
            self.goal_plugin is not None
            and self.goal_plugin._session_id == session_id
        ):
            self.goal_plugin.post_tool_call(**kwargs)

    def on_session_start(self, **kwargs: Any) -> None:
        platform = kwargs.get("platform")
        if platform == "api_server" and self.api_plugin is not None:
            self.api_plugin.on_session_start(**kwargs)
        elif platform == "discord" and self.goal_plugin is not None:
            self.goal_plugin.on_session_start(**kwargs)

    def on_session_end(self, **kwargs: Any) -> None:
        platform = kwargs.get("platform")
        if platform == "api_server" and self.api_plugin is not None:
            self.api_plugin.on_session_end(**kwargs)
        elif platform == "discord" and self.goal_plugin is not None:
            self.goal_plugin.on_session_end(**kwargs)


_PLUGIN: CanaryEvidenceHookMultiplexer | None = None
_API_PLUGIN: CanaryEvidencePlugin | None = None
_GOAL_PLUGIN: GoalContinuationEvidencePlugin | None = None


def register(ctx: Any) -> None:
    """Load independently admitted API/Discord observers and multiplex hooks."""

    global _PLUGIN, _API_PLUGIN, _GOAL_PLUGIN
    if _PLUGIN is not None or _API_PLUGIN is not None or _GOAL_PLUGIN is not None:
        raise CanaryEvidenceError("plugin_already_registered")
    module_origin, module_sha256 = _module_identity()
    api_plugin: CanaryEvidencePlugin | None = None
    goal_plugin: GoalContinuationEvidencePlugin | None = None
    goal_config_present = os.path.lexists(DEFAULT_GOAL_CONFIG_PATH)
    goal_config = load_goal_config() if goal_config_present else None
    retirement_present = os.path.lexists(DEFAULT_API_OBSERVER_RETIREMENT_PATH)
    api_config_present = os.path.lexists(DEFAULT_CONFIG_PATH)
    if retirement_present:
        if goal_config is None or not api_config_present:
            raise CanaryEvidenceError("api_observer_retirement_invalid")
        validate_api_observer_retirement(goal_config)
    if (api_config_present or not goal_config_present) and not retirement_present:
        api_plugin = CanaryEvidencePlugin(load_config())
        api_plugin.start(module_origin=module_origin, module_sha256=module_sha256)
    if goal_config is not None:
        goal_plugin = GoalContinuationEvidencePlugin(goal_config)
        goal_plugin.start(module_origin=module_origin, module_sha256=module_sha256)
    if api_plugin is None and goal_plugin is None:
        raise CanaryEvidenceError("observer_mode_missing")

    multiplexer = CanaryEvidenceHookMultiplexer(api_plugin, goal_plugin)
    ctx.register_hook("pre_api_request", multiplexer.pre_api_request)
    ctx.register_hook("post_api_request", multiplexer.post_api_request)
    ctx.register_hook("post_tool_call", multiplexer.post_tool_call)
    ctx.register_hook("on_session_start", multiplexer.on_session_start)
    ctx.register_hook("on_session_end", multiplexer.on_session_end)
    _PLUGIN = multiplexer
    _API_PLUGIN = api_plugin
    _GOAL_PLUGIN = goal_plugin


__all__ = [
    "ACK_SCHEMA",
    "API_OBSERVER_RETIREMENT_SCHEMA",
    "CONFIG_SCHEMA",
    "FRAME_SCHEMA",
    "GOAL_CONFIG_SCHEMA",
    "GOAL_FRAME_SCHEMA",
    "CanaryEvidenceConfig",
    "CanaryEvidenceError",
    "CanaryEvidencePlugin",
    "CanaryEvidenceHookMultiplexer",
    "CanaryFixture",
    "CollectorEndpoint",
    "EdgeEndpoint",
    "PeerIdentity",
    "GoalContinuationEvidencePlugin",
    "GoalObserverConfig",
    "SocketIdentity",
    "linux_peer_identity",
    "validate_api_observer_retirement",
    "load_config",
    "load_goal_config",
    "register",
]

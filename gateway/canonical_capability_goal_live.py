"""Native, segmented evidence collector for the live Discord goal canary.

This module is intentionally mechanical.  It admits hook frames only from the
exact current systemd MainPID, keeps one hash chain per gateway invocation,
and reads the connector/GoalManager journals through read-only SQLite
transactions.  It never interprets message, task, goal, plan, or model prose.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import socket
import sqlite3
import stat
import struct
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from gateway.canonical_capability_canary_runtime import (
    DEFAULT_DISCORD_CONNECTOR_JOURNAL,
    DEFAULT_GATEWAY_PROFILE_HOME,
    DEFAULT_GOAL_COLLECTOR_SOCKET,
    DEFAULT_GOAL_OBSERVER_CONFIG,
    GATEWAY_UNIT_NAME,
)
from gateway.canonical_projection_export import (
    ProjectionExportError,
    projection_provenance_sha256,
    validate_projection_export,
)
from gateway.discord_connector_protocol import DiscordConnectorEvent
from gateway.posix_identity import effective_uid, real_gid, real_uid
from plugins.muncho_canary_evidence import (
    ACK_SCHEMA,
    DEFAULT_CONFIG_PATH as DEFAULT_API_OBSERVER_CONFIG_PATH,
    GOAL_CONFIG_SCHEMA,
    GOAL_FRAME_SCHEMA,
)


GOAL_COLLECTOR_READINESS_SCHEMA = "muncho-capability-goal-collector-readiness.v1"
GOAL_COLLECTOR_CHAIN_SCHEMA = "muncho-capability-goal-collector-chain.v1"
GOAL_CHALLENGE_SCHEMA = "muncho-capability-goal-owner-challenge.v1"
API_OBSERVER_RETIREMENT_SCHEMA = (
    "muncho-capability-api-observer-retirement.v1"
)
API_OBSERVER_RETIREMENT_PATH = Path(
    "/run/muncho-capability-goal/api-observer-retired.json"
)
GOAL_COLLECTOR_READINESS_PATH = Path(
    "/run/muncho-capability-goal/readiness.json"
)
GOAL_CHALLENGE_FILENAME = "goal-owner-challenge.json"
GOAL_STATE_DB = DEFAULT_GATEWAY_PROFILE_HOME / "state.db"
GOAL_RECOVERY_SCHEMA = "muncho-capability-goal-lineage-recovery.v1"
GOAL_PREEMPTION_SCHEMA = "muncho-capability-goal-preemption.v1"
GOAL_FINALIZATION_INTENT_SCHEMA = (
    "muncho-capability-goal-finalization-intent.v1"
)
GOAL_FINALIZATION_SCHEMA = "muncho-capability-goal-finalization.v1"
MAX_FRAME_BYTES = 2 * 1024 * 1024
MAX_FRAMES = 512
_HEADER = struct.Struct("!I")
_PEERCRED = struct.Struct("3i")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_INVOCATION = re.compile(r"^[0-9a-f]{32}$")
_UUID = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,239}$")
_ZERO_CHAIN = "0" * 64
_MODULE_MAX_BYTES = 8 * 1024 * 1024
_PROJECTION_MAX_BYTES = 256 * 1024 * 1024
_BOOT_ID = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)


class GoalLiveEvidenceError(RuntimeError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _fail(code: str) -> None:
    raise GoalLiveEvidenceError(code)


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8", errors="strict")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise GoalLiveEvidenceError("goal_non_canonical_json") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _strict_json(raw: bytes, code: str) -> Mapping[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate")
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=pairs)
    except (UnicodeError, ValueError, TypeError) as exc:
        raise GoalLiveEvidenceError(code) from exc
    if not isinstance(value, Mapping) or raw != _canonical_bytes(value):
        _fail(code)
    return value


def normalize_goal_observer_public_target(
    fixture_target: Mapping[str, Any],
) -> Mapping[str, str]:
    """Mechanical vocabulary bridge for the same reviewed public target."""

    if (
        not isinstance(fixture_target, Mapping)
        or set(fixture_target) != {"target_type", "guild_id", "channel_id"}
        or fixture_target.get("target_type") != "public_channel"
        or any(
            not isinstance(fixture_target.get(field), str)
            or re.fullmatch(r"[1-9][0-9]{5,24}", fixture_target[field]) is None
            for field in ("guild_id", "channel_id")
        )
    ):
        _fail("goal_collector_public_target_invalid")
    return {
        "target_type": "public_guild_channel",
        "guild_id": str(fixture_target["guild_id"]),
        "channel_id": str(fixture_target["channel_id"]),
    }


@dataclass(frozen=True)
class GatewayServiceIdentity:
    service_unit: str
    main_pid: int
    invocation_id: str
    active_enter_timestamp_monotonic: int
    n_restarts: int
    identity_sha256: str

    def to_mapping(self) -> dict[str, Any]:
        return {
            "service_unit": self.service_unit,
            "main_pid": self.main_pid,
            "invocation_id": self.invocation_id,
            "active_enter_timestamp_monotonic": (
                self.active_enter_timestamp_monotonic
            ),
            "n_restarts": self.n_restarts,
            "identity_sha256": self.identity_sha256,
        }


def read_gateway_service_identity(
    unit: str = GATEWAY_UNIT_NAME,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> GatewayServiceIdentity:
    if unit != GATEWAY_UNIT_NAME:
        _fail("goal_gateway_service_unit_invalid")
    try:
        completed = runner(
            [
                "/usr/bin/systemctl",
                "show",
                unit,
                "--property=MainPID",
                "--property=InvocationID",
                "--property=ActiveEnterTimestampMonotonic",
                "--property=NRestarts",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        values: dict[str, str] = {}
        for line in completed.stdout.splitlines():
            name, separator, value = line.partition("=")
            if not separator or name in values:
                _fail("goal_gateway_service_identity_invalid")
            values[name] = value
        if set(values) != {
            "MainPID",
            "InvocationID",
            "ActiveEnterTimestampMonotonic",
            "NRestarts",
        }:
            _fail("goal_gateway_service_identity_invalid")
        main_pid = int(values["MainPID"])
        invocation_id = values["InvocationID"]
        active_enter = int(values["ActiveEnterTimestampMonotonic"])
        n_restarts = int(values["NRestarts"])
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise GoalLiveEvidenceError(
            "goal_gateway_service_identity_invalid"
        ) from exc
    unsigned = {
        "service_unit": unit,
        "main_pid": main_pid,
        "invocation_id": invocation_id,
        "active_enter_timestamp_monotonic": active_enter,
        "n_restarts": n_restarts,
    }
    if (
        main_pid <= 1
        or _INVOCATION.fullmatch(invocation_id) is None
        or active_enter <= 0
        or n_restarts < 0
    ):
        _fail("goal_gateway_service_identity_invalid")
    return GatewayServiceIdentity(
        **unsigned,
        identity_sha256=_sha256_json(unsigned),
    )


@dataclass(frozen=True)
class GoalPeer:
    pid: int
    uid: int
    gid: int


@dataclass(frozen=True)
class GoalCollectedFrame:
    value: Mapping[str, Any]
    frame_sha256: str
    segment_chain_head_sha256: str
    previous_segment_terminal_sha256: str
    peer: GoalPeer
    service_identity: GatewayServiceIdentity


@dataclass(frozen=True)
class GoalOwnerChallenge:
    """Transient owner text paired with its prose-free durable receipt."""

    transient_input: Mapping[str, str]
    receipt: Mapping[str, Any]
    path: Path


@dataclass(frozen=True)
class GatewayRestartObservation:
    pre_service_identity: GatewayServiceIdentity
    post_service_identity: GatewayServiceIdentity
    restart_requested_at_unix_ms: int
    restart_completed_at_unix_ms: int


@dataclass(frozen=True)
class GoalCanonicalProjectionBinding:
    """Exact writer-owned event/provenance joins used by the goal gate."""

    case_id: str
    canonical_event_pairs: Mapping[
        str, tuple[Mapping[str, Any], Mapping[str, Any]]
    ]
    readback_plan_pairs: Mapping[
        str, tuple[Mapping[str, Any], Mapping[str, Any]]
    ]
    readback_frames: tuple[GoalCollectedFrame, ...]


@dataclass(frozen=True)
class GoalNativeReceiptBundle:
    """One stable SQLite snapshot of native restart/preemption/finalization."""

    recovery: Mapping[str, Any]
    preemption: Mapping[str, Any]
    finalizations: Mapping[
        str, tuple[Mapping[str, Any], Mapping[str, Any]]
    ]


def linux_peer(sock: socket.socket) -> GoalPeer:
    option = getattr(socket, "SO_PEERCRED", None)
    if option is None:
        _fail("goal_collector_peer_credentials_unavailable")
    raw = sock.getsockopt(socket.SOL_SOCKET, option, _PEERCRED.size)
    if len(raw) != _PEERCRED.size:
        _fail("goal_collector_peer_credentials_unavailable")
    return GoalPeer(*_PEERCRED.unpack(raw))


def _receive_exact(connection: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            _fail("goal_collector_frame_truncated")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _safe_owned_directory(path: Path, *, gid: int, mode: int) -> bool:
    """Create one exact root directory; return True only when created here."""

    if not path.is_absolute() or ".." in path.parts:
        _fail("goal_collector_directory_invalid")
    try:
        parent_before = path.parent.lstat()
    except OSError as exc:
        raise GoalLiveEvidenceError("goal_collector_directory_invalid") from exc
    if (
        not stat.S_ISDIR(parent_before.st_mode)
        or stat.S_ISLNK(parent_before.st_mode)
        or parent_before.st_uid != 0
        or stat.S_IMODE(parent_before.st_mode) & 0o022
    ):
        _fail("goal_collector_directory_invalid")
    created = False
    if not os.path.lexists(path):
        path.mkdir(mode=mode, parents=False)
        created = True
        os.chown(path, 0, gid)
        os.chmod(path, mode)
        _fsync_directory(path.parent)
    item = path.lstat()
    parent_after = path.parent.lstat()
    if (
        (parent_before.st_dev, parent_before.st_ino)
        != (parent_after.st_dev, parent_after.st_ino)
        or not stat.S_ISDIR(item.st_mode)
        or stat.S_ISLNK(item.st_mode)
        or item.st_nlink < 2
        or item.st_uid != 0
        or item.st_gid != gid
        or stat.S_IMODE(item.st_mode) != mode
    ):
        _fail("goal_collector_directory_invalid")
    return created


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_exclusive(
    path: Path,
    payload: bytes,
    *,
    gid: int,
    mode: int,
) -> tuple[int, ...]:
    if (
        path.is_absolute() is False
        or ".." in path.parts
        or path.parent.is_symlink()
        or not payload
    ):
        _fail("goal_collector_publication_invalid")
    parent_before = path.parent.lstat()
    if not stat.S_ISDIR(parent_before.st_mode) or stat.S_ISLNK(
        parent_before.st_mode
    ):
        _fail("goal_collector_publication_invalid")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags, mode)
    try:
        os.fchown(descriptor, 0, gid)
        os.fchmod(descriptor, mode)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                _fail("goal_collector_publication_write_stalled")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)
    parent_after = path.parent.lstat()
    item = path.lstat()
    if (
        (parent_before.st_dev, parent_before.st_ino)
        != (parent_after.st_dev, parent_after.st_ino)
        or not stat.S_ISREG(item.st_mode)
        or stat.S_ISLNK(item.st_mode)
        or item.st_nlink != 1
        or item.st_uid != 0
        or item.st_gid != gid
        or stat.S_IMODE(item.st_mode) != mode
        or item.st_size != len(payload)
    ):
        _fail("goal_collector_publication_invalid")
    return (
        item.st_dev,
        item.st_ino,
        item.st_uid,
        item.st_gid,
        stat.S_IMODE(item.st_mode),
        item.st_size,
    )


def _identity(path: Path) -> tuple[int, ...]:
    item = path.lstat()
    return (
        item.st_dev,
        item.st_ino,
        item.st_uid,
        item.st_gid,
        stat.S_IMODE(item.st_mode),
        item.st_size,
    )


def _exact_stat_identity(
    item: os.stat_result,
    *,
    uid: int,
    gid: int,
    mode: int,
    directory: bool,
    code: str,
) -> tuple[int, ...]:
    expected_kind = stat.S_ISDIR if directory else stat.S_ISREG
    if (
        not expected_kind(item.st_mode)
        or stat.S_ISLNK(item.st_mode)
        or (not directory and item.st_nlink != 1)
        or item.st_uid != uid
        or item.st_gid != gid
        or stat.S_IMODE(item.st_mode) != mode
    ):
        _fail(code)
    return (
        item.st_dev,
        item.st_ino,
        item.st_uid,
        item.st_gid,
        stat.S_IMODE(item.st_mode),
    )


def _sqlite_identity_snapshot(
    path: Path,
    *,
    expected_uid: int,
    expected_gid: int,
    expected_mode: int,
    expected_parent_uid: int,
    expected_parent_gid: int,
    expected_parent_mode: int,
    code: str,
) -> tuple[tuple[int, ...], tuple[tuple[str, tuple[int, ...]], ...]]:
    """Pin main DB plus every allowed WAL sidecar under one exact parent."""

    if not path.is_absolute() or ".." in path.parts:
        _fail(code)
    try:
        parent = path.parent.lstat()
        names = os.listdir(path.parent)
    except OSError as exc:
        raise GoalLiveEvidenceError(code) from exc
    parent_identity = _exact_stat_identity(
        parent,
        uid=expected_parent_uid,
        gid=expected_parent_gid,
        mode=expected_parent_mode,
        directory=True,
        code=code,
    )
    prefix = path.name + "-"
    observed_sidecars = {
        name for name in names if name.startswith(prefix)
    }
    allowed_sidecars = {path.name + "-wal", path.name + "-shm"}
    if not observed_sidecars.issubset(allowed_sidecars):
        _fail(code)
    entries: list[tuple[str, tuple[int, ...]]] = []
    for name in (path.name, *sorted(observed_sidecars)):
        entry_path = path.parent / name
        try:
            item = entry_path.lstat()
        except OSError as exc:
            raise GoalLiveEvidenceError(code) from exc
        entries.append(
            (
                name,
                _exact_stat_identity(
                    item,
                    uid=expected_uid,
                    gid=expected_gid,
                    mode=expected_mode,
                    directory=False,
                    code=code,
                ),
            )
        )
    return parent_identity, tuple(entries)


def _sqlite_read_rows(
    path: Path,
    *,
    query: str,
    parameters: Sequence[Any] = (),
    expected_uid: int,
    expected_gid: int,
    expected_mode: int = 0o600,
    expected_parent_uid: int,
    expected_parent_gid: int,
    expected_parent_mode: int = 0o700,
    row_factory: bool = False,
    code: str,
) -> list[Any]:
    expected = _sqlite_identity_snapshot(
        path,
        expected_uid=expected_uid,
        expected_gid=expected_gid,
        expected_mode=expected_mode,
        expected_parent_uid=expected_parent_uid,
        expected_parent_gid=expected_parent_gid,
        expected_parent_mode=expected_parent_mode,
        code=code,
    )
    parent_descriptor = -1
    database_descriptor = -1
    connection: sqlite3.Connection | None = None
    try:
        parent_descriptor = os.open(
            path.parent,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        database_descriptor = os.open(
            path.name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_descriptor,
        )
        parent_fd_identity = _exact_stat_identity(
            os.fstat(parent_descriptor),
            uid=expected_parent_uid,
            gid=expected_parent_gid,
            mode=expected_parent_mode,
            directory=True,
            code=code,
        )
        database_fd_identity = _exact_stat_identity(
            os.fstat(database_descriptor),
            uid=expected_uid,
            gid=expected_gid,
            mode=expected_mode,
            directory=False,
            code=code,
        )
        if (
            parent_fd_identity != expected[0]
            or database_fd_identity != dict(expected[1]).get(path.name)
        ):
            _fail(code)
        connection = sqlite3.connect(
            f"file:{path}?mode=ro",
            uri=True,
            timeout=5,
        )
        if row_factory:
            connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA trusted_schema=OFF")
        query_only = connection.execute("PRAGMA query_only").fetchone()
        if query_only is None or query_only[0] != 1:
            _fail(code)
        connection.execute("BEGIN")
        rows = connection.execute(query, tuple(parameters)).fetchall()
        connection.execute("COMMIT")
        during = _sqlite_identity_snapshot(
            path,
            expected_uid=expected_uid,
            expected_gid=expected_gid,
            expected_mode=expected_mode,
            expected_parent_uid=expected_parent_uid,
            expected_parent_gid=expected_parent_gid,
            expected_parent_mode=expected_parent_mode,
            code=code,
        )
        if during != expected:
            _fail(code)
    except (OSError, sqlite3.Error) as exc:
        raise GoalLiveEvidenceError(code) from exc
    finally:
        if connection is not None:
            connection.close()
        if database_descriptor >= 0:
            os.close(database_descriptor)
        if parent_descriptor >= 0:
            os.close(parent_descriptor)
    if _sqlite_identity_snapshot(
        path,
        expected_uid=expected_uid,
        expected_gid=expected_gid,
        expected_mode=expected_mode,
        expected_parent_uid=expected_parent_uid,
        expected_parent_gid=expected_parent_gid,
        expected_parent_mode=expected_parent_mode,
        code=code,
    ) != expected:
        _fail(code)
    return rows


def _stable_read_bytes(
    path: Path,
    *,
    maximum: int,
    code: str,
    allow_pseudo_file: bool = False,
    expected_uid: int = 0,
    expected_gid: int | None = None,
    expected_mode: int | None = None,
) -> bytes:
    try:
        before = path.lstat()
        if (
            not path.is_absolute()
            or ".." in path.parts
            or not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != expected_uid
            or (expected_gid is not None and before.st_gid != expected_gid)
            or (
                expected_mode is not None
                and stat.S_IMODE(before.st_mode) != expected_mode
            )
            or stat.S_IMODE(before.st_mode) & 0o022
            or (
                not allow_pseudo_file
                and not 0 < before.st_size <= maximum
            )
        ):
            _fail(code)
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise GoalLiveEvidenceError(code) from exc
    try:
        opened = os.fstat(descriptor)
        chunks: list[bytes] = []
        remaining = maximum + 1
        while remaining:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after_fd = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        after_path = path.lstat()
    except OSError as exc:
        raise GoalLiveEvidenceError(code) from exc

    def identity(value: os.stat_result) -> tuple[int, ...]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_nlink,
            value.st_uid,
            value.st_gid,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    raw = b"".join(chunks)
    if (
        not raw
        or len(raw) > maximum
        or (not allow_pseudo_file and len(raw) != before.st_size)
        or identity(before) != identity(opened)
        or identity(before) != identity(after_fd)
        or identity(before) != identity(after_path)
    ):
        _fail(code)
    return raw


def read_collector_process_identity(
    *,
    module_path: Path = Path(__file__).resolve(),
    boot_id_path: Path = Path("/proc/sys/kernel/random/boot_id"),
    process_stat_path: Path = Path("/proc/self/stat"),
) -> Mapping[str, Any]:
    """Bind the root collector to exact code, boot, and process lifetime."""

    module_raw = _stable_read_bytes(
        module_path,
        maximum=_MODULE_MAX_BYTES,
        code="goal_collector_process_identity_invalid",
    )
    boot_raw = _stable_read_bytes(
        boot_id_path,
        maximum=128,
        code="goal_collector_process_identity_invalid",
        allow_pseudo_file=True,
    )
    stat_raw = _stable_read_bytes(
        process_stat_path,
        maximum=8 * 1024,
        code="goal_collector_process_identity_invalid",
        allow_pseudo_file=True,
    )
    try:
        boot_id = boot_raw.decode("ascii", errors="strict").strip()
        process_stat = stat_raw.decode("ascii", errors="strict").strip()
        suffix = process_stat[process_stat.rindex(")") + 2 :].split()
        start_time_ticks = int(suffix[19])
    except (UnicodeError, ValueError, IndexError) as exc:
        raise GoalLiveEvidenceError(
            "goal_collector_process_identity_invalid"
        ) from exc
    if _BOOT_ID.fullmatch(boot_id) is None or start_time_ticks <= 0:
        _fail("goal_collector_process_identity_invalid")
    return {
        "collector_pid": os.getpid(),
        "collector_uid": real_uid(),
        "collector_gid": real_gid(),
        "module_origin_sha256": _sha256_bytes(str(module_path).encode("utf-8")),
        "module_sha256": _sha256_bytes(module_raw),
        "boot_id_sha256": _sha256_bytes(boot_id.encode("ascii")),
        "process_start_time_ticks": start_time_ticks,
    }


def validate_packaged_goal_plugin_module(
    module_origin: str,
    module_sha256: str,
    revision: str,
    *,
    release_base: Path = Path("/opt/muncho-canary-releases"),
    expected_uid: int = 0,
) -> None:
    """Independently attest the exact root-owned plugin loaded by gateway."""

    if (
        not isinstance(module_origin, str)
        or not isinstance(module_sha256, str)
        or _SHA256.fullmatch(module_sha256) is None
        or _REVISION.fullmatch(revision) is None
    ):
        _fail("goal_collector_plugin_module_invalid")
    path = Path(module_origin)
    release_root = release_base / revision
    try:
        resolved_release_root = release_root.resolve(strict=True)
        resolved_path = path.resolve(strict=True)
        relative = resolved_path.relative_to(resolved_release_root)
    except (OSError, ValueError) as exc:
        raise GoalLiveEvidenceError(
            "goal_collector_plugin_module_invalid"
        ) from exc
    if (
        resolved_path != path
        or resolved_release_root != release_root
        or relative.parts[-3:] != (
        "plugins",
        "muncho_canary_evidence",
        "__init__.py",
        )
    ):
        _fail("goal_collector_plugin_module_invalid")
    raw = _stable_read_bytes(
        resolved_path,
        maximum=_MODULE_MAX_BYTES,
        code="goal_collector_plugin_module_invalid",
        expected_uid=expected_uid,
    )
    if _sha256_bytes(raw) != module_sha256:
        _fail("goal_collector_plugin_module_invalid")


class SegmentedGoalEvidenceCollector:
    """Root in-process collector with exact systemd admission per segment."""

    _FRAME_FIELDS = frozenset(
        {
            "schema",
            "segment_id",
            "sequence",
            "event",
            "release_sha",
            "release_sha256",
            "run_id",
            "fixture_sha256",
            "collector_service_identity_sha256",
            "session_id",
            "turn_id",
            "observed_at_unix_ms",
            "payload",
        }
    )
    _PROCESS_IDENTITY_FIELDS = frozenset(
        {
            "collector_pid",
            "collector_uid",
            "collector_gid",
            "module_origin_sha256",
            "module_sha256",
            "boot_id_sha256",
            "process_start_time_ticks",
        }
    )
    _PAYLOAD_FIELDS = {
        "goal_plugin_ready": frozenset(
            {
                "plugin_name",
                "gateway_pid",
                "config_sha256",
                "fixture_sha256",
                "collector_service_identity_sha256",
                "collector_socket_identity_sha256",
                "module_origin",
                "module_sha256",
            }
        ),
        "goal_pre_api_request": frozenset(
            {
                "request_ordinal",
                "task_id_sha256",
                "api_request_id_sha256",
                "runtime_api_call_count",
                "provider",
                "api_mode",
                "model",
                "base_url_sha256",
                "system_prompt_sha256",
                "tool_schema_sha256",
                "reasoning_effort",
                "started_at_unix_ms",
            }
        ),
        "goal_post_api_request": frozenset(
            {
                "request_ordinal",
                "api_request_id_sha256",
                "finish_reason_sha256",
                "response_model_sha256",
                "response_payload_sha256",
                "assistant_tool_call_id_sha256s",
                "response_observed_at_unix_ms",
            }
        ),
        "goal_model_outcome": frozenset(
            {
                "api_request_id_sha256",
                "tool_call_id_sha256",
                "outcome",
                "reason_sha256",
                "recorded",
                "result_sha256",
            }
        ),
        "goal_canonical_event": frozenset(
            {
                "api_request_id_sha256",
                "tool_call_id_sha256",
                "event_id",
                "case_id",
                "event_type",
                "canonical_content_sha256",
                "idempotency_key_sha256",
                "readback_verified",
                "result_sha256",
            }
        ),
        "goal_canonical_readback": frozenset(
            {
                "case_id",
                "query_view",
                "query_limit",
                "readback_sha256",
                "plan_identities",
                "support_incomplete_reasons_sha256",
                "missing_verification_event_ids_sha256",
            }
        ),
        "goal_turn_end": frozenset(
            {"completed", "interrupted", "model_sha256"}
        ),
    }

    def __init__(
        self,
        *,
        revision: str,
        release_sha256: str,
        run_id: str,
        fixture_sha256: str,
        valid_from_unix_ms: int,
        valid_until_unix_ms: int,
        public_target: Mapping[str, Any],
        owner_user_id: str,
        api_observer_config_sha256: str,
        gateway_uid: int,
        gateway_gid: int,
        service_identity_reader: Callable[[], GatewayServiceIdentity] = (
            read_gateway_service_identity
        ),
        peer_reader: Callable[[socket.socket], GoalPeer] = linux_peer,
        collector_process_identity_reader: Callable[[], Mapping[str, Any]] = (
            read_collector_process_identity
        ),
        plugin_module_validator: Callable[[str, str, str], None] = (
            validate_packaged_goal_plugin_module
        ),
        now_ms: Callable[[], int] = lambda: int(time.time() * 1000),
        socket_path: Path = DEFAULT_GOAL_COLLECTOR_SOCKET,
        config_path: Path = DEFAULT_GOAL_OBSERVER_CONFIG,
        readiness_path: Path = GOAL_COLLECTOR_READINESS_PATH,
    ) -> None:
        if (
            _REVISION.fullmatch(revision) is None
            or _SHA256.fullmatch(release_sha256) is None
            or _UUID.fullmatch(run_id) is None
            or _SHA256.fullmatch(fixture_sha256) is None
            or type(valid_from_unix_ms) is not int
            or type(valid_until_unix_ms) is not int
            or not 0 < valid_from_unix_ms < valid_until_unix_ms
            or valid_until_unix_ms - valid_from_unix_ms > 3_600_000
            or type(gateway_uid) is not int
            or type(gateway_gid) is not int
            or gateway_uid <= 0
            or gateway_gid <= 0
            or not isinstance(public_target, Mapping)
            or dict(public_target)
            != {
                "target_type": "public_guild_channel",
                "guild_id": public_target.get("guild_id"),
                "channel_id": public_target.get("channel_id"),
            }
            or any(
                not isinstance(public_target.get(field), str)
                or re.fullmatch(r"[1-9][0-9]{5,24}", public_target[field]) is None
                for field in ("guild_id", "channel_id")
            )
            or not isinstance(owner_user_id, str)
            or re.fullmatch(r"[1-9][0-9]{5,24}", owner_user_id) is None
            or not isinstance(api_observer_config_sha256, str)
            or _SHA256.fullmatch(api_observer_config_sha256) is None
            or not callable(service_identity_reader)
            or not callable(peer_reader)
            or not callable(collector_process_identity_reader)
            or not callable(plugin_module_validator)
        ):
            _fail("goal_collector_configuration_invalid")
        self.revision = revision
        self.release_sha256 = release_sha256
        self.run_id = run_id
        self.fixture_sha256 = fixture_sha256
        self.valid_from_unix_ms = valid_from_unix_ms
        self.valid_until_unix_ms = valid_until_unix_ms
        self.public_target = copy.deepcopy(dict(public_target))
        self.owner_user_id = owner_user_id
        self.api_observer_config_sha256 = api_observer_config_sha256
        self.gateway_uid = gateway_uid
        self.gateway_gid = gateway_gid
        self._service_identity_reader = service_identity_reader
        self._peer_reader = peer_reader
        self._plugin_module_validator = plugin_module_validator
        self._now_ms = now_ms
        self.socket_path = socket_path
        self.config_path = config_path
        self.readiness_path = readiness_path
        self._listener: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._closed = threading.Event()
        self._ready = threading.Event()
        self._segment_ready = threading.Condition()
        self._frame_ready = threading.Condition()
        self._lock = threading.RLock()
        self._error: BaseException | None = None
        self._frames: list[GoalCollectedFrame] = []
        self._segment_heads: dict[str, str] = {}
        self._segment_peers: dict[str, GoalPeer] = {}
        self._segment_services: dict[str, GatewayServiceIdentity] = {}
        self._segment_order: list[str] = []
        self._segment_requests: dict[str, dict[str, dict[str, Any]]] = {}
        self._segment_tool_ids: dict[str, set[str]] = {}
        self._segment_pre_count: dict[str, int] = {}
        self._goal_session_id: str | None = None
        self._ended_turn_ids: set[str] = set()
        self._canonical_case_ids: set[str] = set()
        process_identity = collector_process_identity_reader()
        if (
            not isinstance(process_identity, Mapping)
            or set(process_identity) != self._PROCESS_IDENTITY_FIELDS
            or process_identity.get("collector_pid") != os.getpid()
            or process_identity.get("collector_uid") != real_uid()
            or process_identity.get("collector_gid") != real_gid()
            or any(
                _SHA256.fullmatch(str(process_identity.get(field) or "")) is None
                for field in (
                    "module_origin_sha256",
                    "module_sha256",
                    "boot_id_sha256",
                )
            )
            or type(process_identity.get("process_start_time_ticks")) is not int
            or process_identity["process_start_time_ticks"] <= 0
        ):
            _fail("goal_collector_process_identity_invalid")
        self._collector_process_identity = copy.deepcopy(dict(process_identity))
        self._collector_service_identity_sha256 = _sha256_json(
            {
                "process_identity": self._collector_process_identity,
                "run_id": run_id,
                "fixture_sha256": fixture_sha256,
            }
        )
        self._created_dirs: set[Path] = set()
        self._published: dict[Path, tuple[int, ...]] = {}

    def _observer_config_core(self) -> dict[str, Any]:
        return {
            "schema": GOAL_CONFIG_SCHEMA,
            "release_sha": self.revision,
            "release_sha256": self.release_sha256,
            "run_id": self.run_id,
            "fixture_sha256": self.fixture_sha256,
            "valid_from_unix_ms": self.valid_from_unix_ms,
            "valid_until_unix_ms": self.valid_until_unix_ms,
            "public_target": copy.deepcopy(self.public_target),
            "owner_user_id": self.owner_user_id,
            "model_route": {
                "provider": "openai-codex",
                "api_mode": "codex_responses",
                "base_url": "https://chatgpt.com/backend-api/codex",
                "model": "gpt-5.6-sol",
                "fallback_configured": False,
            },
            "collector": {
                "socket_path": str(self.socket_path),
                "expected_pid": os.getpid(),
                "expected_uid": 0,
                "expected_gid": real_gid(),
                "socket_owner_uid": 0,
                "socket_owner_gid": self.gateway_gid,
                "socket_mode": "0660",
                "service_identity_sha256": (
                    self._collector_service_identity_sha256
                ),
                "connect_timeout_ms": 1_000,
                "ack_timeout_ms": 3_000,
            },
        }

    def _api_observer_retirement_marker(self) -> Mapping[str, Any]:
        goal_authority_sha256 = _sha256_json(self._observer_config_core())
        unsigned = {
            "schema": API_OBSERVER_RETIREMENT_SCHEMA,
            "release_sha": self.revision,
            "release_sha256": self.release_sha256,
            "run_id": self.run_id,
            "fixture_sha256": self.fixture_sha256,
            "api_observer_config_path": str(DEFAULT_API_OBSERVER_CONFIG_PATH),
            "api_observer_config_sha256": self.api_observer_config_sha256,
            "goal_config_authority_sha256": goal_authority_sha256,
            "historical_api_observer_terminal": True,
            "message_content_recorded": False,
        }
        return {**unsigned, "marker_sha256": _sha256_json(unsigned)}

    @property
    def frames(self) -> tuple[GoalCollectedFrame, ...]:
        with self._lock:
            return tuple(self._frames)

    @property
    def segment_service_identities(self) -> tuple[GatewayServiceIdentity, ...]:
        with self._lock:
            return tuple(
                self._segment_services[item] for item in self._segment_order
            )

    def _prepare_paths(self) -> None:
        runtime_parent = self.socket_path.parent
        config_parent = self.config_path.parent
        if _safe_owned_directory(runtime_parent, gid=self.gateway_gid, mode=0o750):
            self._created_dirs.add(runtime_parent)
        if not config_parent.exists():
            if _safe_owned_directory(config_parent, gid=self.gateway_gid, mode=0o750):
                self._created_dirs.add(config_parent)
        else:
            _safe_owned_directory(config_parent, gid=self.gateway_gid, mode=0o750)
        for path in (
            self.socket_path,
            self.config_path,
            self.readiness_path,
            API_OBSERVER_RETIREMENT_PATH,
        ):
            if os.path.lexists(path):
                _fail("goal_collector_runtime_not_fresh")

    def _bind(self) -> None:
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.set_inheritable(False)
        try:
            listener.bind(str(self.socket_path))
            os.chown(self.socket_path, 0, self.gateway_gid)
            os.chmod(self.socket_path, 0o660)
            listener.listen(8)
            listener.settimeout(0.25)
        except BaseException:
            listener.close()
            raise
        self._listener = listener
        self._published[self.socket_path] = _identity(self.socket_path)

    def observer_config(self) -> Mapping[str, Any]:
        """Return the exact prose-free config consumed by the gateway hook."""

        marker = self._api_observer_retirement_marker()
        return {
            **self._observer_config_core(),
            "api_observer_retirement": {
                "marker_path": str(API_OBSERVER_RETIREMENT_PATH),
                "marker_sha256": marker["marker_sha256"],
                "marker_file_sha256": _sha256_bytes(
                    _canonical_bytes(marker)
                ),
                "api_observer_config_path": str(
                    DEFAULT_API_OBSERVER_CONFIG_PATH
                ),
                "api_observer_config_sha256": (
                    self.api_observer_config_sha256
                ),
                "goal_config_authority_sha256": marker[
                    "goal_config_authority_sha256"
                ],
            },
        }

    def publish_api_observer_retirement(self) -> Mapping[str, Any]:
        """Retire only the completed historical API observer for restart."""

        marker = self._api_observer_retirement_marker()
        if os.path.lexists(API_OBSERVER_RETIREMENT_PATH):
            _fail("goal_api_observer_retirement_replayed")
        self._published[API_OBSERVER_RETIREMENT_PATH] = _publish_exclusive(
            API_OBSERVER_RETIREMENT_PATH,
            _canonical_bytes(marker),
            gid=self.gateway_gid,
            mode=0o440,
        )
        return copy.deepcopy(marker)

    def _publish_config_and_readiness(self) -> None:
        item = self.socket_path.lstat()
        config = self.observer_config()
        config_raw = _canonical_bytes(config)
        self._published[self.config_path] = _publish_exclusive(
            self.config_path,
            config_raw,
            gid=self.gateway_gid,
            mode=0o440,
        )
        readiness_unsigned = {
            "schema": GOAL_COLLECTOR_READINESS_SCHEMA,
            "run_id": self.run_id,
            "release_sha": self.revision,
            "fixture_sha256": self.fixture_sha256,
            "collector_pid": os.getpid(),
            "collector_uid": real_uid(),
            "collector_gid": real_gid(),
            "collector_service_identity_sha256": (
                self._collector_service_identity_sha256
            ),
            "collector_process_identity": copy.deepcopy(
                self._collector_process_identity
            ),
            "config_sha256": _sha256_bytes(config_raw),
            "socket": {
                "path": str(self.socket_path),
                "device": item.st_dev,
                "inode": item.st_ino,
                "uid": item.st_uid,
                "gid": item.st_gid,
                "mode": "0660",
            },
            "observed_at_unix_ms": self._now_ms(),
        }
        readiness = {
            **readiness_unsigned,
            "receipt_sha256": _sha256_json(readiness_unsigned),
        }
        self._published[self.readiness_path] = _publish_exclusive(
            self.readiness_path,
            _canonical_bytes(readiness),
            gid=0,
            mode=0o400,
        )

    def start(self) -> None:
        if effective_uid() != 0 or self._thread is not None:
            _fail("goal_collector_start_invalid")
        self._prepare_paths()
        self._bind()
        self._publish_config_and_readiness()
        self._thread = threading.Thread(
            target=self._serve,
            name="muncho-capability-goal-root-collector",
            daemon=True,
        )
        self._thread.start()
        self._ready.set()

    @staticmethod
    def _payload_digest(payload: Mapping[str, Any], field: str) -> str:
        value = payload.get(field)
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            _fail("goal_collector_event_payload_invalid")
        return value

    @staticmethod
    def _payload_safe_id(payload: Mapping[str, Any], field: str) -> str:
        value = payload.get(field)
        if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
            _fail("goal_collector_event_payload_invalid")
        return value

    def _validate_event_payload(
        self,
        frame: Mapping[str, Any],
        *,
        peer: GoalPeer,
    ) -> None:
        event = frame["event"]
        payload = frame["payload"]
        expected_fields = self._PAYLOAD_FIELDS.get(event)
        if expected_fields is None or set(payload) != expected_fields:
            _fail("goal_collector_event_payload_invalid")
        segment_id = str(frame["segment_id"])
        session_id = frame.get("session_id")
        turn_id = frame.get("turn_id")
        observed_at = frame["observed_at_unix_ms"]
        if event == "goal_plugin_ready":
            if (
                payload.get("plugin_name")
                != "muncho_canary_evidence.goal_continuation"
                or payload.get("gateway_pid") != peer.pid
                or payload.get("fixture_sha256") != self.fixture_sha256
                or payload.get("collector_service_identity_sha256")
                != self._collector_service_identity_sha256
                or not isinstance(payload.get("module_origin"), str)
                or not str(payload["module_origin"]).startswith("/")
            ):
                _fail("goal_collector_event_payload_invalid")
            for field in (
                "config_sha256",
                "collector_socket_identity_sha256",
                "module_sha256",
            ):
                self._payload_digest(payload, field)
            self._plugin_module_validator(
                str(payload["module_origin"]),
                str(payload["module_sha256"]),
                self.revision,
            )
            return

        if self._goal_session_id is None:
            self._goal_session_id = str(session_id)
        elif session_id != self._goal_session_id:
            _fail("goal_collector_session_rotated")
        if turn_id in self._ended_turn_ids:
            _fail("goal_collector_turn_replayed")
        requests = self._segment_requests[segment_id]
        observed_tool_ids = self._segment_tool_ids[segment_id]

        if event == "goal_pre_api_request":
            ordinal = payload.get("request_ordinal")
            api_digest = self._payload_digest(payload, "api_request_id_sha256")
            expected_ordinal = self._segment_pre_count[segment_id] + 1
            if (
                ordinal != expected_ordinal
                or type(payload.get("runtime_api_call_count")) is not int
                or payload["runtime_api_call_count"] < 0
                or payload.get("provider") != "openai-codex"
                or payload.get("api_mode") != "codex_responses"
                or payload.get("model") != "gpt-5.6-sol"
                or payload.get("base_url_sha256")
                != _sha256_bytes(
                    b"https://chatgpt.com/backend-api/codex"
                )
                or payload.get("reasoning_effort") not in {"high", "max"}
                or payload.get("started_at_unix_ms") != observed_at
                or api_digest in requests
            ):
                _fail("goal_collector_event_payload_invalid")
            for field in (
                "task_id_sha256",
                "system_prompt_sha256",
                "tool_schema_sha256",
            ):
                self._payload_digest(payload, field)
            requests[api_digest] = {
                "ordinal": ordinal,
                "turn_id": turn_id,
                "completed": False,
                "assistant_tool_call_id_sha256s": set(),
            }
            self._segment_pre_count[segment_id] = expected_ordinal
            return

        if event == "goal_post_api_request":
            api_digest = self._payload_digest(payload, "api_request_id_sha256")
            request = requests.get(api_digest)
            tool_ids = payload.get("assistant_tool_call_id_sha256s")
            if (
                not isinstance(request, Mapping)
                or request.get("completed") is not False
                or request.get("turn_id") != turn_id
                or payload.get("request_ordinal") != request.get("ordinal")
                or payload.get("response_observed_at_unix_ms") != observed_at
                or not isinstance(tool_ids, list)
                or len(tool_ids) > 256
                or any(
                    not isinstance(value, str)
                    or _SHA256.fullmatch(value) is None
                    for value in tool_ids
                )
                or len(set(tool_ids)) != len(tool_ids)
            ):
                _fail("goal_collector_event_payload_invalid")
            for field in (
                "finish_reason_sha256",
                "response_model_sha256",
                "response_payload_sha256",
            ):
                self._payload_digest(payload, field)
            request["completed"] = True
            request["assistant_tool_call_id_sha256s"] = set(tool_ids)
            return

        if event in {"goal_model_outcome", "goal_canonical_event"}:
            api_digest = self._payload_digest(payload, "api_request_id_sha256")
            tool_digest = self._payload_digest(payload, "tool_call_id_sha256")
            request = requests.get(api_digest)
            if (
                not isinstance(request, Mapping)
                or request.get("completed") is not True
                or request.get("turn_id") != turn_id
                or tool_digest
                not in request.get("assistant_tool_call_id_sha256s", set())
                or tool_digest in observed_tool_ids
            ):
                _fail("goal_collector_event_payload_invalid")
            self._payload_digest(payload, "result_sha256")
            if event == "goal_model_outcome":
                if (
                    payload.get("outcome")
                    not in {"continue", "complete", "blocked"}
                    or payload.get("recorded") is not True
                ):
                    _fail("goal_collector_event_payload_invalid")
                self._payload_digest(payload, "reason_sha256")
            else:
                case_id = self._payload_safe_id(payload, "case_id")
                if (
                    not case_id.startswith("case:")
                    or payload.get("readback_verified") is not True
                ):
                    _fail("goal_collector_event_payload_invalid")
                self._payload_safe_id(payload, "event_id")
                self._payload_safe_id(payload, "event_type")
                self._payload_digest(payload, "canonical_content_sha256")
                self._payload_digest(payload, "idempotency_key_sha256")
                self._canonical_case_ids.add(case_id)
            observed_tool_ids.add(tool_digest)
            return

        if event == "goal_canonical_readback":
            case_id = self._payload_safe_id(payload, "case_id")
            plans = payload.get("plan_identities")
            if (
                case_id not in self._canonical_case_ids
                or payload.get("query_view") != "resume_bundle"
                or payload.get("query_limit") != 200
                or not isinstance(plans, list)
                or not plans
            ):
                _fail("goal_collector_event_payload_invalid")
            for field in (
                "readback_sha256",
                "support_incomplete_reasons_sha256",
                "missing_verification_event_ids_sha256",
            ):
                self._payload_digest(payload, field)
            for plan in plans:
                if (
                    not isinstance(plan, Mapping)
                    or set(plan)
                    != {
                        "event_id",
                        "plan_id",
                        "revision",
                        "state",
                        "next_step_id",
                    }
                    or type(plan.get("revision")) is not int
                    or plan["revision"] <= 0
                    or plan.get("state")
                    not in {"active", "blocked", "completed", "cancelled"}
                    or plan.get("next_step_id") is not None
                    and (
                        not isinstance(plan.get("next_step_id"), str)
                        or _SAFE_ID.fullmatch(plan["next_step_id"]) is None
                    )
                ):
                    _fail("goal_collector_event_payload_invalid")
                self._payload_safe_id(plan, "event_id")
                self._payload_safe_id(plan, "plan_id")
            return

        if event == "goal_turn_end":
            if (
                type(payload.get("completed")) is not bool
                or type(payload.get("interrupted")) is not bool
                or any(
                    request.get("turn_id") == turn_id
                    and request.get("completed") is not True
                    for request in requests.values()
                )
            ):
                _fail("goal_collector_event_payload_invalid")
            self._payload_digest(payload, "model_sha256")
            self._ended_turn_ids.add(str(turn_id))
            return
        _fail("goal_collector_event_payload_invalid")

    def _validate_frame(
        self,
        frame: Mapping[str, Any],
        peer: GoalPeer,
        service: GatewayServiceIdentity,
    ) -> None:
        if set(frame) != self._FRAME_FIELDS:
            _fail("goal_collector_frame_invalid")
        segment_id = frame.get("segment_id")
        sequence = frame.get("sequence")
        expected_sequence = 1 + sum(
            item.value["segment_id"] == segment_id for item in self._frames
        )
        if (
            frame.get("schema") != GOAL_FRAME_SCHEMA
            or not isinstance(segment_id, str)
            or re.fullmatch(r"[0-9a-f]{32}", segment_id) is None
            or sequence != expected_sequence
            or frame.get("release_sha") != self.revision
            or frame.get("release_sha256") != self.release_sha256
            or frame.get("run_id") != self.run_id
            or frame.get("fixture_sha256") != self.fixture_sha256
            or frame.get("collector_service_identity_sha256")
            != self._collector_service_identity_sha256
            or type(frame.get("observed_at_unix_ms")) is not int
            or not self.valid_from_unix_ms
            <= frame["observed_at_unix_ms"]
            <= self.valid_until_unix_ms
            or not isinstance(frame.get("event"), str)
            or frame.get("event") not in self._PAYLOAD_FIELDS
            or not isinstance(frame.get("payload"), Mapping)
            or peer.pid != service.main_pid
            or peer.uid != self.gateway_uid
            or peer.gid != self.gateway_gid
        ):
            _fail("goal_collector_frame_binding_invalid")
        if sequence == 1:
            if (
                frame.get("event") != "goal_plugin_ready"
                or frame.get("session_id") is not None
                or frame.get("turn_id") is not None
                or frame["payload"].get("gateway_pid") != peer.pid
                or segment_id in self._segment_heads
                or any(
                    identity.invocation_id == service.invocation_id
                    for identity in self._segment_services.values()
                )
            ):
                _fail("goal_collector_segment_ready_invalid")
        else:
            if (
                self._segment_peers.get(segment_id) != peer
                or self._segment_services.get(segment_id) != service
                or frame.get("event") == "goal_plugin_ready"
                or not isinstance(frame.get("session_id"), str)
                or not isinstance(frame.get("turn_id"), str)
            ):
                _fail("goal_collector_segment_binding_invalid")
        self._validate_event_payload(frame, peer=peer)

    def _accept(self, connection: socket.socket) -> None:
        connection.settimeout(4.0)
        peer = self._peer_reader(connection)
        header = _receive_exact(connection, _HEADER.size)
        (size,) = _HEADER.unpack(header)
        if not 1 < size <= MAX_FRAME_BYTES:
            _fail("goal_collector_frame_size_invalid")
        raw = _receive_exact(connection, size)
        if self._peer_reader(connection) != peer:
            _fail("goal_collector_peer_rotated")
        frame = _strict_json(raw, "goal_collector_frame_invalid")
        service = self._service_identity_reader()
        with self._lock:
            if len(self._frames) >= MAX_FRAMES:
                _fail("goal_collector_frame_limit")
            self._validate_frame(frame, peer, service)
            segment_id = str(frame["segment_id"])
            if frame["sequence"] == 1:
                previous_segment_terminal = (
                    self._segment_heads[self._segment_order[-1]]
                    if self._segment_order
                    else _ZERO_CHAIN
                )
                self._segment_order.append(segment_id)
                self._segment_peers[segment_id] = peer
                self._segment_services[segment_id] = service
                self._segment_requests[segment_id] = {}
                self._segment_tool_ids[segment_id] = set()
                self._segment_pre_count[segment_id] = 0
                previous = _ZERO_CHAIN
            else:
                previous_segment_terminal = (
                    self._segment_heads[self._segment_order[-2]]
                    if len(self._segment_order) > 1
                    and self._segment_order[-1] == segment_id
                    else _ZERO_CHAIN
                )
                previous = self._segment_heads[segment_id]
            frame_sha256 = _sha256_bytes(raw)
            chain_unsigned = {
                "schema": GOAL_COLLECTOR_CHAIN_SCHEMA,
                "segment_id": segment_id,
                "previous_segment_terminal_sha256": previous_segment_terminal,
                "previous_sha256": previous,
                "sequence": frame["sequence"],
                "frame_sha256": frame_sha256,
                "gateway_service_identity_sha256": service.identity_sha256,
                "peer_pid": peer.pid,
                "peer_uid": peer.uid,
                "peer_gid": peer.gid,
            }
            chain_head = _sha256_json(chain_unsigned)
            self._segment_heads[segment_id] = chain_head
            self._frames.append(
                GoalCollectedFrame(
                    value=copy.deepcopy(dict(frame)),
                    frame_sha256=frame_sha256,
                    segment_chain_head_sha256=chain_head,
                    previous_segment_terminal_sha256=(
                        previous_segment_terminal
                    ),
                    peer=peer,
                    service_identity=service,
                )
            )
            ack = {
                "schema": ACK_SCHEMA,
                "sequence": frame["sequence"],
                "accepted": True,
                "frame_sha256": frame_sha256,
                "collector_receipt_sha256": chain_head,
            }
            with self._frame_ready:
                self._frame_ready.notify_all()
            if frame["sequence"] == 1:
                with self._segment_ready:
                    self._segment_ready.notify_all()
        payload = _canonical_bytes(ack)
        connection.sendall(_HEADER.pack(len(payload)) + payload)

    def _serve(self) -> None:
        try:
            assert self._listener is not None
            while not self._closed.is_set():
                try:
                    connection, _address = self._listener.accept()
                except socket.timeout:
                    continue
                with connection:
                    self._accept(connection)
        except BaseException as exc:
            with self._lock:
                self._error = exc
            with self._frame_ready:
                self._frame_ready.notify_all()
            with self._segment_ready:
                self._segment_ready.notify_all()

    def wait_for_segments(self, count: int, *, deadline: float) -> None:
        while True:
            with self._lock:
                if self._error is not None:
                    raise GoalLiveEvidenceError(
                        "goal_collector_segment_failed"
                    ) from self._error
                if len(self._segment_order) >= count:
                    return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _fail("goal_collector_segment_timeout")
            with self._segment_ready:
                self._segment_ready.wait(min(remaining, 0.25))

    def wait_for_frame(
        self,
        predicate: Callable[[Sequence[GoalCollectedFrame]], bool],
        *,
        deadline: float,
    ) -> tuple[GoalCollectedFrame, ...]:
        if not callable(predicate):
            _fail("goal_collector_wait_invalid")
        while True:
            frames = self.frames
            if predicate(frames):
                return frames
            with self._lock:
                if self._error is not None:
                    raise GoalLiveEvidenceError(
                        "goal_collector_frame_failed"
                    ) from self._error
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _fail("goal_collector_frame_timeout")
            with self._frame_ready:
                self._frame_ready.wait(min(remaining, 0.25))

    def controlled_gateway_restart(
        self,
        *,
        deadline: float,
        pre_restart_validator: Callable[
            [Sequence[GoalCollectedFrame]], bool
        ],
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> GatewayRestartObservation:
        """Restart only after two continues and their durable native receipts."""

        if not callable(pre_restart_validator):
            _fail("goal_gateway_restart_validator_invalid")

        frames = self.wait_for_frame(
            lambda values: sum(
                item.value.get("event") == "goal_model_outcome"
                and item.value.get("payload", {}).get("outcome") == "continue"
                for item in values
            )
            >= 2,
            deadline=deadline,
        )
        while True:
            validated = pre_restart_validator(frames)
            if type(validated) is not bool:
                _fail("goal_gateway_restart_validator_invalid")
            if validated:
                break
            if time.monotonic() >= deadline:
                _fail("goal_gateway_restart_native_receipt_timeout")
            time.sleep(0.05)
            frames = self.frames
        with self._lock:
            if len(self._segment_order) != 1:
                _fail("goal_gateway_restart_phase_invalid")
            first_segment = self._segment_order[0]
            segment_service = self._segment_services.get(first_segment)
        pre = self._service_identity_reader()
        marker_identity = self._published.get(API_OBSERVER_RETIREMENT_PATH)
        if (
            segment_service != pre
            or frames[-1].value["observed_at_unix_ms"] > self._now_ms()
            or marker_identity is None
        ):
            _fail("goal_gateway_restart_precondition_invalid")
        try:
            if _identity(API_OBSERVER_RETIREMENT_PATH) != marker_identity:
                _fail("goal_gateway_restart_precondition_invalid")
        except OSError as exc:
            raise GoalLiveEvidenceError(
                "goal_gateway_restart_precondition_invalid"
            ) from exc
        requested_at = self._now_ms()
        try:
            runner(
                ["/usr/bin/systemctl", "restart", GATEWAY_UNIT_NAME],
                check=True,
                capture_output=True,
                text=True,
                timeout=max(1.0, min(60.0, deadline - time.monotonic())),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise GoalLiveEvidenceError("goal_gateway_restart_failed") from exc
        while True:
            post = self._service_identity_reader()
            if (
                post.main_pid != pre.main_pid
                and post.invocation_id != pre.invocation_id
                and post.active_enter_timestamp_monotonic
                > pre.active_enter_timestamp_monotonic
                and post.n_restarts >= pre.n_restarts
            ):
                break
            if time.monotonic() >= deadline:
                _fail("goal_gateway_restart_identity_timeout")
            time.sleep(0.05)
        self.wait_for_segments(2, deadline=deadline)
        completed_at = self._now_ms()
        return GatewayRestartObservation(
            pre_service_identity=pre,
            post_service_identity=post,
            restart_requested_at_unix_ms=requested_at,
            restart_completed_at_unix_ms=completed_at,
        )

    def publish_challenge(
        self,
        *,
        receipt_root: Path,
        goal_command: str,
        kickoff_message: str,
        preemption_message: str,
    ) -> GoalOwnerChallenge:
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (goal_command, kickoff_message, preemption_message)
        ):
            _fail("goal_owner_challenge_invalid")
        path = receipt_root / self.run_id / GOAL_CHALLENGE_FILENAME
        if not path.parent.exists():
            _fail("goal_owner_challenge_directory_unavailable")
        challenge_id = str(uuid.uuid4())
        unsigned = {
            "schema": GOAL_CHALLENGE_SCHEMA,
            "run_id": self.run_id,
            "fixture_sha256": self.fixture_sha256,
            "challenge_id": challenge_id,
            "public_target": copy.deepcopy(self.public_target),
            "owner_user_id": self.owner_user_id,
            "goal_command_sha256": _sha256_bytes(goal_command.encode()),
            "kickoff_message_sha256": _sha256_bytes(kickoff_message.encode()),
            "preemption_message_sha256": _sha256_bytes(
                preemption_message.encode()
            ),
            "message_content_recorded": False,
            "published_at_unix_ms": self._now_ms(),
        }
        receipt = {**unsigned, "challenge_sha256": _sha256_json(unsigned)}
        _publish_exclusive(path, _canonical_bytes(receipt), gid=0, mode=0o400)
        return GoalOwnerChallenge(
            transient_input={
                "goal_command": goal_command,
                "kickoff_message": kickoff_message,
                "preemption_message": preemption_message,
            },
            receipt=copy.deepcopy(receipt),
            path=path,
        )

    def close(self) -> None:
        self._closed.set()
        if self._listener is not None:
            try:
                self._listener.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=5)
        replaced = False
        for path, expected in reversed(tuple(self._published.items())):
            try:
                if _identity(path) != expected:
                    replaced = True
                    continue
                path.unlink()
            except FileNotFoundError:
                pass
        for path in sorted(self._created_dirs, key=lambda item: len(item.parts), reverse=True):
            try:
                path.rmdir()
            except OSError:
                pass
        if replaced:
            _fail("goal_collector_runtime_replaced")


def _native_goal_receipt(
    raw: Any,
    *,
    schema: str,
    digest_field: str,
) -> Mapping[str, Any]:
    if not isinstance(raw, str) or not raw:
        _fail("goal_native_receipt_invalid")
    value = _strict_json(
        raw.encode("utf-8", errors="strict"),
        "goal_native_receipt_invalid",
    )
    return _validate_native_goal_receipt(
        value,
        schema=schema,
        digest_field=digest_field,
    )


def _validate_native_goal_receipt(
    value: Any,
    *,
    schema: str,
    digest_field: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail("goal_native_receipt_invalid")
    unsigned = {
        key: copy.deepcopy(item)
        for key, item in value.items()
        if key != digest_field
    }
    if (
        value.get("schema") != schema
        or set(value) != {*unsigned, digest_field}
        or value.get(digest_field) != _sha256_json(unsigned)
    ):
        _fail("goal_native_receipt_invalid")
    return copy.deepcopy(dict(value))


def read_goal_native_finalizations(
    session_id: str,
    turn_ids: Sequence[str],
    *,
    path: Path = GOAL_STATE_DB,
    expected_uid: int,
    expected_gid: int,
    expected_parent_uid: int | None = None,
    expected_parent_gid: int | None = None,
) -> Mapping[str, tuple[Mapping[str, Any], Mapping[str, Any]]]:
    """Read exact before/after seals without requiring post-restart receipts."""

    if (
        _SAFE_ID.fullmatch(session_id) is None
        or not isinstance(turn_ids, Sequence)
        or isinstance(turn_ids, (str, bytes, bytearray))
        or not turn_ids
        or len(turn_ids) > 128
        or any(
            not isinstance(turn_id, str)
            or _SAFE_ID.fullmatch(turn_id) is None
            for turn_id in turn_ids
        )
        or len(set(turn_ids)) != len(turn_ids)
    ):
        _fail("goal_native_receipt_invalid")
    pair_keys: dict[str, tuple[str, str]] = {}
    for turn_id in turn_ids:
        turn_sha256 = _sha256_bytes(turn_id.encode("utf-8", errors="strict"))
        prefix = f"capability_goal_finalization:{session_id}:{turn_sha256}:"
        pair_keys[turn_id] = (prefix + "before", prefix + "after")
    expected_keys = {
        key for pair in pair_keys.values() for key in pair
    }
    ordered_keys = tuple(sorted(expected_keys))
    placeholders = ",".join("?" for _ in ordered_keys)
    parent_uid = expected_uid if expected_parent_uid is None else expected_parent_uid
    parent_gid = expected_gid if expected_parent_gid is None else expected_parent_gid
    rows = _sqlite_read_rows(
        path,
        query=(
            "SELECT key,value FROM state_meta WHERE key IN ("
            + placeholders
            + ") ORDER BY key"
        ),
        parameters=ordered_keys,
        expected_uid=expected_uid,
        expected_gid=expected_gid,
        expected_parent_uid=parent_uid,
        expected_parent_gid=parent_gid,
        code="goal_native_receipt_invalid",
    )
    observed = {str(key): value for key, value in rows}
    if len(observed) != len(rows):
        _fail("goal_native_receipt_invalid")
    if set(observed) != expected_keys:
        _fail("goal_native_receipt_unavailable")
    return {
        turn_id: (
            _native_goal_receipt(
                observed[before],
                schema=GOAL_FINALIZATION_INTENT_SCHEMA,
                digest_field="intent_sha256",
            ),
            _native_goal_receipt(
                observed[after],
                schema=GOAL_FINALIZATION_SCHEMA,
                digest_field="finalization_sha256",
            ),
        )
        for turn_id, (before, after) in pair_keys.items()
    }


def read_goal_native_receipts(
    session_id: str,
    turn_ids: Sequence[str],
    *,
    path: Path = GOAL_STATE_DB,
    expected_uid: int,
    expected_gid: int,
    expected_parent_uid: int | None = None,
    expected_parent_gid: int | None = None,
) -> GoalNativeReceiptBundle:
    """Read all goal authority receipts in one stable read-only transaction."""

    if (
        _SAFE_ID.fullmatch(session_id) is None
        or not isinstance(turn_ids, Sequence)
        or isinstance(turn_ids, (str, bytes, bytearray))
        or not turn_ids
        or len(turn_ids) > 128
        or any(
            not isinstance(turn_id, str)
            or _SAFE_ID.fullmatch(turn_id) is None
            for turn_id in turn_ids
        )
        or len(set(turn_ids)) != len(turn_ids)
    ):
        _fail("goal_native_receipt_invalid")
    recovery_key = f"capability_goal_lineage_recovery:{session_id}"
    preemption_key = f"capability_goal_preemption:{session_id}"
    expected_keys = {recovery_key, preemption_key}
    pair_keys: dict[str, tuple[str, str]] = {}
    for turn_id in turn_ids:
        turn_sha256 = _sha256_bytes(turn_id.encode("utf-8", errors="strict"))
        prefix = f"capability_goal_finalization:{session_id}:{turn_sha256}:"
        pair_keys[turn_id] = (prefix + "before", prefix + "after")
        expected_keys.update(pair_keys[turn_id])
    parent_uid = expected_uid if expected_parent_uid is None else expected_parent_uid
    parent_gid = expected_gid if expected_parent_gid is None else expected_parent_gid
    ordered_keys = tuple(sorted(expected_keys))
    placeholders = ",".join("?" for _ in ordered_keys)
    rows = _sqlite_read_rows(
        path,
        query=(
            "SELECT key,value FROM state_meta WHERE key IN ("
            + placeholders
            + ") ORDER BY key"
        ),
        parameters=ordered_keys,
        expected_uid=expected_uid,
        expected_gid=expected_gid,
        expected_parent_uid=parent_uid,
        expected_parent_gid=parent_gid,
        code="goal_native_receipt_invalid",
    )
    observed = {str(key): value for key, value in rows}
    if len(observed) != len(rows):
        _fail("goal_native_receipt_invalid")
    if set(observed) != expected_keys:
        _fail("goal_native_receipt_unavailable")
    finalizations = {
        turn_id: (
            _native_goal_receipt(
                observed[before],
                schema=GOAL_FINALIZATION_INTENT_SCHEMA,
                digest_field="intent_sha256",
            ),
            _native_goal_receipt(
                observed[after],
                schema=GOAL_FINALIZATION_SCHEMA,
                digest_field="finalization_sha256",
            ),
        )
        for turn_id, (before, after) in pair_keys.items()
    }
    return GoalNativeReceiptBundle(
        recovery=_native_goal_receipt(
            observed[recovery_key],
            schema=GOAL_RECOVERY_SCHEMA,
            digest_field="recovery_sha256",
        ),
        preemption=_native_goal_receipt(
            observed[preemption_key],
            schema=GOAL_PREEMPTION_SCHEMA,
            digest_field="preemption_sha256",
        ),
        finalizations=finalizations,
    )


def read_writer_projection_export(
    path: Path,
    *,
    expected_writer_uid: int,
    expected_projector_gid: int,
    export_receipt: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Read and bind the exact one-shot writer-owned v2 projection file."""

    if (
        type(expected_writer_uid) is not int
        or type(expected_projector_gid) is not int
        or expected_writer_uid <= 0
        or expected_projector_gid <= 0
        or not isinstance(export_receipt, Mapping)
        or set(export_receipt)
        != {
            "event_count",
            "provenance_count",
            "provenance_sha256",
            "sha256",
            "size",
            "owner_uid",
            "group_gid",
            "mode",
            "stdout_receipt",
        }
    ):
        _fail("goal_projection_export_invalid")
    raw = _stable_read_bytes(
        path,
        maximum=_PROJECTION_MAX_BYTES,
        code="goal_projection_export_invalid",
        expected_uid=expected_writer_uid,
        expected_gid=expected_projector_gid,
        expected_mode=0o640,
    )
    document_raw = raw[:-1] if raw.endswith(b"\n") else raw
    if not document_raw or b"\n" in raw[:-1]:
        _fail("goal_projection_export_invalid")
    value = _strict_json(document_raw, "goal_projection_export_invalid")
    try:
        events, provenance = validate_projection_export(
            value,
            maximum_events=1_000_000,
        )
    except ProjectionExportError as exc:
        raise GoalLiveEvidenceError("goal_projection_export_invalid") from exc
    stdout = export_receipt.get("stdout_receipt")
    expected = {
        "event_count": len(events),
        "provenance_count": len(provenance),
        "provenance_sha256": projection_provenance_sha256(provenance),
        "sha256": _sha256_bytes(raw),
        "size": len(raw),
        "owner_uid": expected_writer_uid,
        "group_gid": expected_projector_gid,
        "mode": "0640",
        "stdout_receipt": {
            "event_count": len(events),
            "success": True,
        },
    }
    if dict(export_receipt) != expected or stdout != expected["stdout_receipt"]:
        _fail("goal_projection_export_invalid")
    return copy.deepcopy(dict(value))


def validate_goal_canonical_projection_binding(
    frames: Sequence[GoalCollectedFrame],
    projection_export: Mapping[str, Any],
    *,
    maximum_events: int = 1_000_000,
) -> GoalCanonicalProjectionBinding:
    """Join observer frames to the privileged writer's immutable export.

    This is intentionally a mechanical identity check.  It never classifies
    event prose or infers plan meaning: the event id, case, typed event name,
    canonical content digest, idempotency digest, writer provenance, and typed
    plan readback projection must all agree exactly.
    """

    if (
        not isinstance(frames, Sequence)
        or isinstance(frames, (str, bytes, bytearray))
        or not frames
        or isinstance(maximum_events, bool)
        or not 1 <= maximum_events <= 1_000_000
    ):
        _fail("goal_projection_binding_invalid")
    try:
        events, provenance = validate_projection_export(
            projection_export,
            maximum_events=maximum_events,
        )
    except ProjectionExportError as exc:
        raise GoalLiveEvidenceError("goal_projection_binding_invalid") from exc
    pairs_by_id = {
        event["event_id"]: (event, proof)
        for event, proof in zip(events, provenance, strict=True)
    }
    canonical_pairs: dict[
        str, tuple[Mapping[str, Any], Mapping[str, Any]]
    ] = {}
    readback_pairs: dict[
        str, tuple[Mapping[str, Any], Mapping[str, Any]]
    ] = {}
    readback_frames: list[GoalCollectedFrame] = []
    case_ids: set[str] = set()

    for collected in frames:
        if not isinstance(collected, GoalCollectedFrame):
            _fail("goal_projection_binding_invalid")
        frame = collected.value
        event_name = frame.get("event")
        if event_name == "goal_canonical_event":
            payload = frame.get("payload")
            if not isinstance(payload, Mapping):
                _fail("goal_projection_binding_invalid")
            event_id = payload.get("event_id")
            pair = pairs_by_id.get(event_id)
            if pair is None or event_id in canonical_pairs:
                _fail("goal_projection_binding_invalid")
            event, proof = pair
            event_payload = event.get("payload")
            idempotency_key = (
                event_payload.get("idempotency_key")
                if isinstance(event_payload, Mapping)
                else None
            )
            if (
                event.get("case_id") != payload.get("case_id")
                or event.get("event_type") != payload.get("event_type")
                or not isinstance(event_payload, Mapping)
                or event_payload.get("canonical_content_sha256")
                != payload.get("canonical_content_sha256")
                or proof.get("canonical_content_sha256")
                != payload.get("canonical_content_sha256")
                or not isinstance(idempotency_key, str)
                or not idempotency_key
                or _sha256_bytes(idempotency_key.encode("utf-8", errors="strict"))
                != payload.get("idempotency_key_sha256")
                or proof.get("trusted_runtime")
                != event.get("source", {}).get("observed_session")
                or proof.get("origin")
                != event.get("decision", {}).get("decided_by")
                or proof.get("appended_at") != event.get("occurred_at")
            ):
                _fail("goal_projection_binding_invalid")
            canonical_pairs[str(event_id)] = (event, proof)
            case_ids.add(str(event["case_id"]))
        elif event_name == "goal_canonical_readback":
            payload = frame.get("payload")
            if not isinstance(payload, Mapping):
                _fail("goal_projection_binding_invalid")
            case_id = payload.get("case_id")
            identities = payload.get("plan_identities")
            if not isinstance(case_id, str) or not isinstance(identities, list):
                _fail("goal_projection_binding_invalid")
            case_ids.add(case_id)
            readback_frames.append(collected)
            for identity in identities:
                if not isinstance(identity, Mapping):
                    _fail("goal_projection_binding_invalid")
                event_id = identity.get("event_id")
                pair = pairs_by_id.get(event_id)
                if pair is None:
                    _fail("goal_projection_binding_invalid")
                event, proof = pair
                event_payload = event.get("payload")
                plan = (
                    event_payload.get("plan")
                    if isinstance(event_payload, Mapping)
                    else None
                )
                cursor = (
                    plan.get("resume_cursor")
                    if isinstance(plan, Mapping)
                    else None
                )
                expected_identity = {
                    "event_id": event_id,
                    "plan_id": plan.get("plan_id") if isinstance(plan, Mapping) else None,
                    "revision": plan.get("revision") if isinstance(plan, Mapping) else None,
                    "state": plan.get("state") if isinstance(plan, Mapping) else None,
                    "next_step_id": (
                        cursor.get("next_step_id")
                        if isinstance(cursor, Mapping)
                        else None
                    ),
                }
                if (
                    event.get("case_id") != case_id
                    or event.get("event_type") != "task.plan.updated"
                    or dict(identity) != expected_identity
                ):
                    _fail("goal_projection_binding_invalid")
                prior = readback_pairs.get(str(event_id))
                if prior is not None and prior != pair:
                    _fail("goal_projection_binding_invalid")
                readback_pairs[str(event_id)] = pair

    if not canonical_pairs or not readback_pairs or len(case_ids) != 1:
        _fail("goal_projection_binding_invalid")
    return GoalCanonicalProjectionBinding(
        case_id=next(iter(case_ids)),
        canonical_event_pairs=copy.deepcopy(canonical_pairs),
        readback_plan_pairs=copy.deepcopy(readback_pairs),
        readback_frames=tuple(readback_frames),
    )


def _goal_bound_receipt(
    *,
    schema: str,
    digest_field: str,
    fixture: Mapping[str, Any],
    fixture_sha256: str,
    owner_approval_receipt_sha256: str,
    fields: Mapping[str, Any],
) -> Mapping[str, Any]:
    unsigned = {
        "schema": schema,
        "run_id": fixture["run_id"],
        "release_sha": fixture["release_sha"],
        "capability_plan_sha256": fixture["capability_plan_sha256"],
        "full_canary_plan_sha256": fixture["full_canary_plan_sha256"],
        "fixture_sha256": fixture_sha256,
        "owner_approval_receipt_sha256": owner_approval_receipt_sha256,
        **copy.deepcopy(dict(fields)),
    }
    return {**unsigned, digest_field: _sha256_json(unsigned)}


def _exact_connector_row(
    rows: Sequence[ConnectorJournalRow],
    *,
    content: str,
    public_target: Mapping[str, Any],
    owner_user_id: str,
    not_before_unix_ms: int,
) -> ConnectorJournalRow:
    if type(not_before_unix_ms) is not int or not_before_unix_ms <= 0:
        _fail("goal_connector_ingress_invalid")
    matches = [
        row
        for row in rows
        if row.event.content == content
        and row.event.target.to_mapping() == dict(public_target)
        and row.event.author_id == owner_user_id
        and row.event.author_is_bot is False
        and row.event.created_at_unix_ms >= not_before_unix_ms
        and row.offered_at_unix_ms >= not_before_unix_ms
    ]
    if len(matches) != 1:
        _fail("goal_connector_ingress_invalid")
    row = matches[0]
    if (
        row.state != "acked"
        or not isinstance(row.delivery_id, str)
        or not row.delivery_id
        or row.lease_until_unix_ms is not None
        or type(row.offered_at_unix_ms) is not int
        or type(row.acked_at_unix_ms) is not int
        or row.offered_at_unix_ms <= 0
        or row.acked_at_unix_ms < row.offered_at_unix_ms
    ):
        _fail("goal_connector_ingress_invalid")
    return row


def _goal_fixture_native_binding(
    value: Mapping[str, Any],
    *,
    fixture: Mapping[str, Any],
    fixture_sha256: str,
) -> None:
    if (
        value.get("run_id") != fixture.get("run_id")
        or value.get("fixture_sha256") != fixture_sha256
        or value.get("release_sha") != fixture.get("release_sha")
        or value.get("capability_plan_sha256")
        != fixture.get("capability_plan_sha256")
        or value.get("full_canary_plan_sha256")
        != fixture.get("full_canary_plan_sha256")
    ):
        _fail("goal_native_receipt_invalid")


def _service_process_sha256(identity: GatewayServiceIdentity) -> str:
    return _sha256_json(
        {
            "service_unit": identity.service_unit,
            "invocation_id": identity.invocation_id,
            "main_pid": identity.main_pid,
        }
    )


def _service_mapping(identity: GatewayServiceIdentity) -> Mapping[str, Any]:
    return identity.to_mapping()


@dataclass(frozen=True)
class ConnectorJournalRow:
    event: DiscordConnectorEvent
    event_sha256: str
    state: str
    delivery_id: str | None
    lease_until_unix_ms: int | None
    offered_at_unix_ms: int
    acked_at_unix_ms: int | None
    row_sha256: str


def read_connector_rows(
    path: Path = DEFAULT_DISCORD_CONNECTOR_JOURNAL,
    *,
    expected_uid: int,
    expected_gid: int,
    expected_parent_uid: int | None = None,
    expected_parent_gid: int | None = None,
) -> tuple[ConnectorJournalRow, ...]:
    parent_uid = expected_uid if expected_parent_uid is None else expected_parent_uid
    parent_gid = expected_gid if expected_parent_gid is None else expected_parent_gid
    rows = _sqlite_read_rows(
        path,
        query="""
        SELECT event_id,event_sha256,event_json,state,delivery_id,
               lease_until_unix_ms,offered_at_unix_ms,acked_at_unix_ms
          FROM connector_events_v1
         ORDER BY offered_at_unix_ms,event_id
        """,
        expected_uid=expected_uid,
        expected_gid=expected_gid,
        expected_parent_uid=parent_uid,
        expected_parent_gid=parent_gid,
        row_factory=True,
        code="goal_connector_journal_read_failed",
    )
    result: list[ConnectorJournalRow] = []
    for row in rows:
        event_raw = str(row["event_json"]).encode("utf-8", errors="strict")
        event_mapping = _strict_json(event_raw, "goal_connector_event_invalid")
        event = DiscordConnectorEvent.from_mapping(event_mapping)
        raw = {name: row[name] for name in row.keys()}
        if event.event_id != row["event_id"] or event.sha256 != row["event_sha256"]:
            _fail("goal_connector_event_binding_invalid")
        result.append(
            ConnectorJournalRow(
                event=event,
                event_sha256=str(row["event_sha256"]),
                state=str(row["state"]),
                delivery_id=(
                    str(row["delivery_id"])
                    if row["delivery_id"] is not None
                    else None
                ),
                lease_until_unix_ms=row["lease_until_unix_ms"],
                offered_at_unix_ms=int(row["offered_at_unix_ms"]),
                acked_at_unix_ms=row["acked_at_unix_ms"],
                row_sha256=_sha256_json(raw),
            )
        )
    return tuple(result)


def read_state_meta(
    key: str,
    *,
    path: Path = GOAL_STATE_DB,
    expected_uid: int,
    expected_gid: int,
    expected_parent_uid: int | None = None,
    expected_parent_gid: int | None = None,
) -> tuple[str | None, str]:
    if not isinstance(key, str) or not key:
        _fail("goal_state_meta_key_invalid")
    parent_uid = expected_uid if expected_parent_uid is None else expected_parent_uid
    parent_gid = expected_gid if expected_parent_gid is None else expected_parent_gid
    rows = _sqlite_read_rows(
        path,
        query="SELECT value FROM state_meta WHERE key=?",
        parameters=(key,),
        expected_uid=expected_uid,
        expected_gid=expected_gid,
        expected_parent_uid=parent_uid,
        expected_parent_gid=parent_gid,
        code="goal_state_meta_read_failed",
    )
    if len(rows) > 1:
        _fail("goal_state_meta_read_failed")
    row = rows[0] if rows else None
    value = str(row[0]) if row is not None else None
    projection = {"key": key, "value": value}
    return value, _sha256_json(projection)


def read_goal_state_projection(
    session_id: str,
    *,
    path: Path = GOAL_STATE_DB,
    expected_uid: int,
    expected_gid: int,
) -> Mapping[str, Any]:
    if _SAFE_ID.fullmatch(session_id) is None:
        _fail("goal_state_session_invalid")
    value, state_sha256 = read_state_meta(
        f"goal:{session_id}",
        path=path,
        expected_uid=expected_uid,
        expected_gid=expected_gid,
    )
    if value is None:
        return {
            "state_sha256": state_sha256,
            "present": False,
            "status": None,
            "generation_id": None,
            "active_model_turn_id": None,
            "pending_model_outcome": None,
            "pending_model_reason_sha256": None,
            "pending_model_turn_id": None,
            "pending_model_generation_id": None,
            "max_turns": None,
            "created_at_unix_ms": None,
            "turns_used": None,
        }
    try:
        state = json.loads(value)
    except (TypeError, ValueError) as exc:
        raise GoalLiveEvidenceError("goal_state_value_invalid") from exc
    if not isinstance(state, Mapping):
        _fail("goal_state_value_invalid")
    reason = state.get("pending_model_reason")
    projection = {
        "state_sha256": state_sha256,
        "present": True,
        "status": state.get("status"),
        "generation_id": state.get("generation_id"),
        "active_model_turn_id": state.get("active_model_turn_id"),
        "pending_model_outcome": state.get("pending_model_outcome"),
        "pending_model_reason_sha256": (
            _sha256_bytes(reason.encode("utf-8", errors="strict"))
            if isinstance(reason, str) and reason
            else None
        ),
        "pending_model_turn_id": state.get("pending_model_turn_id"),
        "pending_model_generation_id": state.get(
            "pending_model_generation_id"
        ),
        "max_turns": state.get("max_turns"),
        "created_at_unix_ms": (
            int(float(state.get("created_at")) * 1000)
            if isinstance(state.get("created_at"), (int, float))
            and not isinstance(state.get("created_at"), bool)
            else None
        ),
        "turns_used": state.get("turns_used"),
    }
    if (
        projection["status"] not in {"active", "paused", "done", "blocked"}
        or not isinstance(projection["generation_id"], str)
        or re.fullmatch(r"[0-9a-f]{32}", projection["generation_id"]) is None
        or type(projection["max_turns"]) is not int
        or projection["max_turns"] < 0
        or type(projection["turns_used"]) is not int
        or projection["turns_used"] < 0
    ):
        _fail("goal_state_projection_invalid")
    return projection


def build_goal_continuation_evidence(
    *,
    fixture: Mapping[str, Any],
    fixture_sha256: str,
    owner_approval_receipt_sha256: str,
    challenge: GoalOwnerChallenge,
    connector_rows: Sequence[ConnectorJournalRow],
    frames: Sequence[GoalCollectedFrame],
    native_receipts: GoalNativeReceiptBundle,
    restart: GatewayRestartObservation,
    projection_binding: GoalCanonicalProjectionBinding,
    production_diff: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Assemble the live goal proof from independent mechanical authorities."""

    from gateway import canonical_capability_canary_e2e as evidence_contract

    if (
        not isinstance(fixture, Mapping)
        or _SHA256.fullmatch(fixture_sha256) is None
        or _SHA256.fullmatch(owner_approval_receipt_sha256) is None
        or not isinstance(challenge, GoalOwnerChallenge)
        or not isinstance(native_receipts, GoalNativeReceiptBundle)
        or not isinstance(restart, GatewayRestartObservation)
        or not isinstance(projection_binding, GoalCanonicalProjectionBinding)
        or not isinstance(production_diff, Mapping)
    ):
        _fail("goal_continuation_evidence_invalid")
    try:
        fixture_target = normalize_goal_observer_public_target(
            fixture["public_discord_target"]
        )
        owner_user_id = str(fixture["owner_id"])
        connector_bot_id = str(
            fixture["discord_bot_identities"]["connector_bot_user_id"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise GoalLiveEvidenceError(
            "goal_continuation_evidence_invalid"
        ) from exc
    challenge_receipt = challenge.receipt
    transient = challenge.transient_input
    if (
        not isinstance(challenge_receipt, Mapping)
        or not isinstance(transient, Mapping)
        or set(transient)
        != {"goal_command", "kickoff_message", "preemption_message"}
        or set(challenge_receipt)
        != {
            "schema",
            "run_id",
            "fixture_sha256",
            "challenge_id",
            "public_target",
            "owner_user_id",
            "goal_command_sha256",
            "kickoff_message_sha256",
            "preemption_message_sha256",
            "message_content_recorded",
            "published_at_unix_ms",
            "challenge_sha256",
        }
    ):
        _fail("goal_continuation_evidence_invalid")
    challenge_unsigned = {
        key: copy.deepcopy(value)
        for key, value in challenge_receipt.items()
        if key != "challenge_sha256"
    }
    goal_command = transient["goal_command"]
    kickoff_message = transient["kickoff_message"]
    preemption_message = transient["preemption_message"]
    if (
        challenge_receipt.get("schema") != GOAL_CHALLENGE_SCHEMA
        or challenge_receipt.get("run_id") != fixture.get("run_id")
        or challenge_receipt.get("fixture_sha256") != fixture_sha256
        or challenge_receipt.get("public_target") != fixture_target
        or challenge_receipt.get("owner_user_id") != owner_user_id
        or challenge_receipt.get("message_content_recorded") is not False
        or challenge_receipt.get("challenge_sha256")
        != _sha256_json(challenge_unsigned)
        or any(
            not isinstance(value, str) or not value.strip()
            for value in (goal_command, kickoff_message, preemption_message)
        )
        or goal_command != f"/goal {kickoff_message}"
        or challenge_receipt.get("goal_command_sha256")
        != _sha256_bytes(goal_command.encode("utf-8", errors="strict"))
        or challenge_receipt.get("kickoff_message_sha256")
        != _sha256_bytes(kickoff_message.encode("utf-8", errors="strict"))
        or challenge_receipt.get("preemption_message_sha256")
        != _sha256_bytes(preemption_message.encode("utf-8", errors="strict"))
        or type(challenge_receipt.get("published_at_unix_ms")) is not int
    ):
        _fail("goal_continuation_evidence_invalid")

    goal_row = _exact_connector_row(
        connector_rows,
        content=goal_command,
        public_target=fixture_target,
        owner_user_id=owner_user_id,
        not_before_unix_ms=challenge_receipt["published_at_unix_ms"],
    )
    preemption_row = _exact_connector_row(
        connector_rows,
        content=preemption_message,
        public_target=fixture_target,
        owner_user_id=owner_user_id,
        not_before_unix_ms=challenge_receipt["published_at_unix_ms"],
    )
    if (
        goal_row.event.event_id == preemption_row.event.event_id
        or not challenge_receipt["published_at_unix_ms"]
        <= goal_row.offered_at_unix_ms
        <= goal_row.acked_at_unix_ms
        <= preemption_row.offered_at_unix_ms
        <= preemption_row.acked_at_unix_ms
    ):
        _fail("goal_continuation_evidence_invalid")
    ingress = _goal_bound_receipt(
        schema=evidence_contract.DISCORD_OWNER_INGRESS_SCHEMA,
        digest_field="receipt_sha256",
        fixture=fixture,
        fixture_sha256=fixture_sha256,
        owner_approval_receipt_sha256=owner_approval_receipt_sha256,
        fields={
            "challenge_id": challenge_receipt["challenge_id"],
            "challenge_sha256": challenge_receipt["challenge_sha256"],
            "wait_started_at_unix_ms": challenge_receipt[
                "published_at_unix_ms"
            ],
            "discord_event_id": goal_row.event.event_id,
            "discord_event_sha256": goal_row.event.sha256,
            "target_type": "public_guild_channel",
            "guild_id": fixture_target["guild_id"],
            "channel_id": fixture_target["channel_id"],
            "owner_user_id": owner_user_id,
            "connector_bot_user_id": connector_bot_id,
            "delivery_id_sha256": _sha256_bytes(
                str(goal_row.delivery_id).encode("utf-8", errors="strict")
            ),
            "connector_journal_row_sha256": goal_row.row_sha256,
            "journal_state": "acked",
            "offered_at_unix_ms": goal_row.offered_at_unix_ms,
            "acked_at_unix_ms": goal_row.acked_at_unix_ms,
            "acked": True,
            "message_content_recorded": False,
        },
    )

    frame_list = tuple(frames)
    outcome_frames = tuple(
        frame
        for frame in frame_list
        if frame.value.get("event") == "goal_model_outcome"
    )
    turn_end_frames = tuple(
        frame
        for frame in frame_list
        if frame.value.get("event") == "goal_turn_end"
    )
    if (
        not 3 <= len(outcome_frames) <= 128
        or len(turn_end_frames) != len(outcome_frames)
        or any(
            frame.value.get("payload", {}).get("outcome") != "continue"
            for frame in outcome_frames[:-1]
        )
        or outcome_frames[-1].value.get("payload", {}).get("outcome")
        != "complete"
        or len(
            {
                frame.value.get("turn_id")
                for frame in outcome_frames
            }
        )
        != len(outcome_frames)
        or {
            frame.value.get("turn_id") for frame in outcome_frames
        }
        != {frame.value.get("turn_id") for frame in turn_end_frames}
        or any(
            frame.value.get("payload", {}).get("completed") is not True
            or frame.value.get("payload", {}).get("interrupted") is not False
            for frame in turn_end_frames
        )
    ):
        _fail("goal_continuation_evidence_invalid")
    sessions = {frame.value.get("session_id") for frame in outcome_frames}
    if len(sessions) != 1 or not isinstance(next(iter(sessions)), str):
        _fail("goal_continuation_evidence_invalid")
    session_id = str(next(iter(sessions)))

    def one_api_frame(
        event_name: str,
        outcome_frame: GoalCollectedFrame,
    ) -> GoalCollectedFrame:
        outcome = outcome_frame.value
        matches = [
            frame
            for frame in frame_list
            if frame.value.get("event") == event_name
            and frame.value.get("segment_id") == outcome.get("segment_id")
            and frame.value.get("turn_id") == outcome.get("turn_id")
            and frame.value.get("payload", {}).get("api_request_id_sha256")
            == outcome.get("payload", {}).get("api_request_id_sha256")
        ]
        if len(matches) != 1:
            _fail("goal_continuation_evidence_invalid")
        return matches[0]

    model_outcomes: list[Mapping[str, Any]] = []
    generation_id: str | None = None
    prior_state_after: str | None = None
    for ordinal, frame in enumerate(outcome_frames, start=1):
        turn_id = str(frame.value["turn_id"])
        pair = native_receipts.finalizations.get(turn_id)
        if pair is None:
            _fail("goal_continuation_evidence_invalid")
        intent, finalization = pair
        intent = _validate_native_goal_receipt(
            intent,
            schema=GOAL_FINALIZATION_INTENT_SCHEMA,
            digest_field="intent_sha256",
        )
        finalization = _validate_native_goal_receipt(
            finalization,
            schema=GOAL_FINALIZATION_SCHEMA,
            digest_field="finalization_sha256",
        )
        _goal_fixture_native_binding(
            intent,
            fixture=fixture,
            fixture_sha256=fixture_sha256,
        )
        _goal_fixture_native_binding(
            finalization,
            fixture=fixture,
            fixture_sha256=fixture_sha256,
        )
        state_before = intent.get("state_before")
        state_after = finalization.get("state_after")
        payload = frame.value["payload"]
        pre_frame = one_api_frame("goal_pre_api_request", frame)
        post_frame = one_api_frame("goal_post_api_request", frame)
        service = frame.service_identity
        process_sha256 = _service_process_sha256(service)
        observed_generation = intent.get("goal_generation_id")
        if generation_id is None:
            generation_id = str(observed_generation)
        expected_verdict = (
            "continue" if payload["outcome"] == "continue" else "done"
        )
        expected_status = (
            "active" if payload["outcome"] == "continue" else "done"
        )
        if (
            _INVOCATION.fullmatch(str(observed_generation or "")) is None
            or generation_id != observed_generation
            or intent.get("session_id") != session_id
            or finalization.get("session_id") != session_id
            or intent.get("originating_turn_id") != turn_id
            or finalization.get("originating_turn_id") != turn_id
            or intent.get("intent_sha256")
            != finalization.get("intent_sha256")
            or intent.get("pending_outcome_exact") is not True
            or intent.get("model_outcome") != payload.get("outcome")
            or intent.get("model_reason_sha256")
            != payload.get("reason_sha256")
            or finalization.get("model_outcome") != payload.get("outcome")
            or finalization.get("model_reason_sha256")
            != payload.get("reason_sha256")
            or not isinstance(state_before, Mapping)
            or not isinstance(state_after, Mapping)
            or intent.get("state_before_sha256") != _sha256_json(state_before)
            or finalization.get("state_before") != state_before
            or finalization.get("state_before_sha256")
            != intent.get("state_before_sha256")
            or finalization.get("state_after_sha256")
            != _sha256_json(state_after)
            or state_before.get("session_id") != session_id
            or state_after.get("session_id") != session_id
            or state_before.get("generation_id") != generation_id
            or state_after.get("generation_id") != generation_id
            or state_before.get("max_turns") != 0
            or state_after.get("max_turns") != 0
            or state_before.get("status") != "active"
            or state_after.get("status") != expected_status
            or finalization.get("decision_verdict") != expected_verdict
            or finalization.get("should_continue")
            != (payload["outcome"] == "continue")
            or finalization.get("gateway_service_unit")
            != service.service_unit
            or finalization.get("gateway_invocation_id")
            != service.invocation_id
            or finalization.get("gateway_main_pid") != service.main_pid
            or finalization.get("gateway_process_identity_sha256")
            != process_sha256
            or intent.get("gateway_process_identity_sha256") != process_sha256
            or type(finalization.get("observed_after_unix_ms")) is not int
            or finalization["observed_after_unix_ms"]
            < frame.value["observed_at_unix_ms"]
            or (
                prior_state_after is not None
                and intent.get("state_before_sha256") != prior_state_after
            )
        ):
            _fail("goal_continuation_evidence_invalid")
        prior_state_after = str(finalization["state_after_sha256"])
        model_outcomes.append(
            _goal_bound_receipt(
                schema=evidence_contract.GOAL_MODEL_OUTCOME_SCHEMA,
                digest_field="receipt_sha256",
                fixture=fixture,
                fixture_sha256=fixture_sha256,
                owner_approval_receipt_sha256=(
                    owner_approval_receipt_sha256
                ),
                fields={
                    "session_id": session_id,
                    "goal_generation_id": generation_id,
                    "turn_id": turn_id,
                    "ordinal": ordinal,
                    "outcome": payload["outcome"],
                    "reason_sha256": payload["reason_sha256"],
                    "todo_tool_call_id_sha256": payload[
                        "tool_call_id_sha256"
                    ],
                    "goal_state_before_sha256": intent[
                        "state_before_sha256"
                    ],
                    "goal_state_after_sha256": finalization[
                        "state_after_sha256"
                    ],
                    "turn_started_at_unix_ms": pre_frame.value["payload"][
                        "started_at_unix_ms"
                    ],
                    "observed_at_unix_ms": finalization[
                        "observed_after_unix_ms"
                    ],
                    "structured_outcome_source": "todo.goal_outcome",
                    "goal_manager": "hermes_cli.goals.GoalManager",
                    "goal_max_turns": 0,
                },
            )
        )
    if generation_id is None:
        _fail("goal_continuation_evidence_invalid")

    continue_before_restart = [
        (frame, receipt)
        for frame, receipt in zip(outcome_frames, model_outcomes, strict=True)
        if receipt["outcome"] == "continue"
        and receipt["observed_at_unix_ms"] <= restart.restart_requested_at_unix_ms
    ]
    if (
        len(continue_before_restart) < 2
        or restart.pre_service_identity
        == restart.post_service_identity
        or not any(
            frame.service_identity == restart.post_service_identity
            for frame in outcome_frames
        )
    ):
        _fail("goal_continuation_evidence_invalid")
    restart_turn = continue_before_restart[-1][1]
    gateway_restart = _goal_bound_receipt(
        schema=evidence_contract.GATEWAY_RESTART_RECEIPT_SCHEMA,
        digest_field="receipt_sha256",
        fixture=fixture,
        fixture_sha256=fixture_sha256,
        owner_approval_receipt_sha256=owner_approval_receipt_sha256,
        fields={
            "service_unit": GATEWAY_UNIT_NAME,
            "restart_count": 1,
            "continuation_turn_id": restart_turn["turn_id"],
            "continuation_started_at_unix_ms": restart_turn[
                "turn_started_at_unix_ms"
            ],
            "restart_requested_at_unix_ms": (
                restart.restart_requested_at_unix_ms
            ),
            "restart_completed_at_unix_ms": (
                restart.restart_completed_at_unix_ms
            ),
            "pre_service_identity": _service_mapping(
                restart.pre_service_identity
            ),
            "post_service_identity": _service_mapping(
                restart.post_service_identity
            ),
            "controlled_restart": True,
        },
    )

    recovery = native_receipts.recovery
    preemption = native_receipts.preemption
    recovery = _validate_native_goal_receipt(
        recovery,
        schema=GOAL_RECOVERY_SCHEMA,
        digest_field="recovery_sha256",
    )
    preemption = _validate_native_goal_receipt(
        preemption,
        schema=GOAL_PREEMPTION_SCHEMA,
        digest_field="preemption_sha256",
    )
    _goal_fixture_native_binding(
        recovery,
        fixture=fixture,
        fixture_sha256=fixture_sha256,
    )
    _goal_fixture_native_binding(
        preemption,
        fixture=fixture,
        fixture_sha256=fixture_sha256,
    )
    if (
        recovery.get("session_id") != session_id
        or recovery.get("goal_generation_id") != generation_id
        or recovery.get("connector_event_id") != goal_row.event.event_id
        or recovery.get("connector_event_sha256") != goal_row.event.sha256
        or recovery.get("connector_delivery_id") != goal_row.delivery_id
        or recovery.get("connector_journal_state") != "acked"
        or recovery.get("gateway_service_unit") != GATEWAY_UNIT_NAME
        or recovery.get("gateway_invocation_id")
        != restart.post_service_identity.invocation_id
        or recovery.get("gateway_main_pid")
        != restart.post_service_identity.main_pid
        or type(recovery.get("restored_at_unix_ms")) is not int
        or recovery["restored_at_unix_ms"]
        <= restart.restart_completed_at_unix_ms
        or preemption.get("session_id") != session_id
        or preemption.get("goal_generation_id") != generation_id
        or preemption.get("queued_event_id")
        != preemption_row.event.event_id
        or preemption.get("queued_event_sha256") != preemption_row.event.sha256
        or preemption.get("queued_delivery_id") != preemption_row.delivery_id
        or preemption.get("queued_owner_user_id") != owner_user_id
        or preemption.get("queued_guild_id") != fixture_target["guild_id"]
        or preemption.get("queued_channel_id") != fixture_target["channel_id"]
        or preemption.get("automatic_continuation_was_pending") is not True
        or preemption.get("automatic_continuation_duplicate_count") != 0
        or preemption.get("originating_turn_id")
        not in {item["turn_id"] for item in model_outcomes[:-1]}
        or preemption.get("queue_path")
        not in {"adapter.pending", "runner.fifo_overflow"}
    ):
        _fail("goal_continuation_evidence_invalid")

    canonical_frames = {
        str(frame.value["payload"]["event_id"]): frame
        for frame in frame_list
        if frame.value.get("event") == "goal_canonical_event"
    }
    plan_candidates: list[
        tuple[Mapping[str, Any], Mapping[str, Any], GoalCollectedFrame]
    ] = []
    for event_id, pair in projection_binding.canonical_event_pairs.items():
        event, proof = pair
        frame = canonical_frames.get(event_id)
        payload = event.get("payload")
        plan = payload.get("plan") if isinstance(payload, Mapping) else None
        if (
            frame is not None
            and event.get("event_type") == "task.plan.updated"
            and isinstance(plan, Mapping)
        ):
            plan_candidates.append((event, proof, frame))
    checkpoints = [
        item
        for item in plan_candidates
        if item[0]["payload"]["plan"].get("state") == "active"
        and item[2].service_identity == restart.pre_service_identity
    ]
    terminals = [
        item
        for item in plan_candidates
        if item[0]["payload"]["plan"].get("state") == "completed"
        and item[2].service_identity == restart.post_service_identity
    ]
    if not checkpoints or len(terminals) != 1:
        _fail("goal_continuation_evidence_invalid")
    checkpoint_event, checkpoint_proof, _checkpoint_frame = max(
        checkpoints,
        key=lambda item: int(item[0]["payload"]["plan"].get("revision", 0)),
    )
    terminal_event, terminal_proof, terminal_frame = terminals[0]
    checkpoint_plan = checkpoint_event["payload"]["plan"]
    terminal_plan = terminal_event["payload"]["plan"]
    checkpoint_cursor = checkpoint_plan.get("resume_cursor")
    next_step_id = (
        checkpoint_cursor.get("next_step_id")
        if isinstance(checkpoint_cursor, Mapping)
        else None
    )
    terminal_readbacks = [
        frame
        for frame in projection_binding.readback_frames
        if frame.service_identity == restart.post_service_identity
        and any(
            identity.get("event_id") == terminal_event["event_id"]
            for identity in frame.value.get("payload", {}).get(
                "plan_identities", []
            )
        )
    ]
    if (
        checkpoint_event.get("case_id") != projection_binding.case_id
        or terminal_event.get("case_id") != projection_binding.case_id
        or checkpoint_plan.get("plan_id") != terminal_plan.get("plan_id")
        or type(checkpoint_plan.get("revision")) is not int
        or type(terminal_plan.get("revision")) is not int
        or terminal_plan["revision"] <= checkpoint_plan["revision"]
        or not isinstance(next_step_id, str)
        or not next_step_id
        or len(terminal_readbacks) != 1
        or terminal_frame.value["observed_at_unix_ms"]
        <= restart.restart_completed_at_unix_ms
    ):
        _fail("goal_continuation_evidence_invalid")
    terminal_readback = terminal_readbacks[0]
    ctw_recovery = _goal_bound_receipt(
        schema=evidence_contract.CTW_RECOVERY_RECEIPT_SCHEMA,
        digest_field="receipt_sha256",
        fixture=fixture,
        fixture_sha256=fixture_sha256,
        owner_approval_receipt_sha256=owner_approval_receipt_sha256,
        fields={
            "session_id": session_id,
            "goal_generation_id": generation_id,
            "case_id": projection_binding.case_id,
            "plan_id": terminal_plan["plan_id"],
            "plan_revision": terminal_plan["revision"],
            "checkpoint_event_id": checkpoint_event["event_id"],
            "checkpoint_event_sha256": checkpoint_proof[
                "canonical_content_sha256"
            ],
            "recovery_readback_sha256": terminal_readback.value["payload"][
                "readback_sha256"
            ],
            "next_step_id": next_step_id,
            "terminal_event_id": terminal_event["event_id"],
            "terminal_event_sha256": terminal_proof[
                "canonical_content_sha256"
            ],
            "terminal_state": "completed",
            "recovery_boundary": "gateway_restart_resume",
            "resumed_after_restart": True,
            "replayed_mutation_count": 0,
            "observed_at_unix_ms": max(
                terminal_frame.value["observed_at_unix_ms"],
                terminal_readback.value["observed_at_unix_ms"],
                recovery["restored_at_unix_ms"],
            ),
        },
    )

    routeback_candidates: list[
        tuple[Mapping[str, Any], Mapping[str, Any], GoalCollectedFrame]
    ] = []
    for event_id, pair in projection_binding.canonical_event_pairs.items():
        event, proof = pair
        frame = canonical_frames.get(event_id)
        if (
            frame is not None
            and frame.service_identity == restart.post_service_identity
            and event.get("event_type") == "route_back.sent"
            and event.get("case_id") == projection_binding.case_id
        ):
            routeback_candidates.append((event, proof, frame))
    if len(routeback_candidates) != 1:
        _fail("goal_continuation_evidence_invalid")
    routeback_event, routeback_proof, routeback_frame = routeback_candidates[0]
    routeback_payload = routeback_event.get("payload")
    routeback = (
        routeback_payload.get("route_back")
        if isinstance(routeback_payload, Mapping)
        else None
    )
    routeback_receipt = (
        routeback_payload.get("receipt")
        if isinstance(routeback_payload, Mapping)
        else None
    )
    target_ref = (
        routeback.get("target_ref") if isinstance(routeback, Mapping) else None
    )
    routeback_nested_receipt = (
        routeback.get("receipt") if isinstance(routeback, Mapping) else None
    )
    execution_binding = (
        routeback.get("execution_binding")
        if isinstance(routeback, Mapping)
        else None
    )
    decision = routeback_event.get("decision")
    safety = routeback_event.get("safety")
    expected_receipt_fields = {
        "platform",
        "adapter_receipt",
        "receipt_readback_verified",
        "message_id",
        "channel_id",
        "content_sha256",
        "public_receipt_sha256",
    }
    if (
        not isinstance(routeback_payload, Mapping)
        or not isinstance(routeback, Mapping)
        or not isinstance(routeback_receipt, Mapping)
        or set(routeback_receipt) != expected_receipt_fields
        or routeback_nested_receipt != routeback_receipt
        or not isinstance(target_ref, Mapping)
        or not isinstance(execution_binding, Mapping)
        or set(execution_binding) != {"target_channel_id", "content_sha256"}
        or routeback_receipt.get("platform") != "discord"
        or routeback_receipt.get("adapter_receipt") is not True
        or routeback_receipt.get("receipt_readback_verified") is not True
        or not isinstance(routeback_receipt.get("message_id"), str)
        or not routeback_receipt["message_id"].isdigit()
        or routeback_receipt.get("channel_id") != fixture_target["channel_id"]
        or _SHA256.fullmatch(
            str(routeback_receipt.get("content_sha256") or "")
        )
        is None
        or _SHA256.fullmatch(
            str(routeback_receipt.get("public_receipt_sha256") or "")
        )
        is None
        or target_ref.get("target_type") != "public_guild_channel"
        or target_ref.get("guild_id") != fixture_target["guild_id"]
        or target_ref.get("channel_id") != fixture_target["channel_id"]
        or execution_binding.get("target_channel_id")
        != routeback_receipt["channel_id"]
        or execution_binding.get("content_sha256")
        != routeback_receipt["content_sha256"]
        or not isinstance(decision, Mapping)
        or decision.get("decided_by") != "routeback_finalize_sent"
        or routeback_proof.get("origin") != "routeback_finalize_sent"
        or not isinstance(safety, Mapping)
        or safety.get("outbound") is not True
        or routeback_event.get("occurred_at")
        != routeback_proof.get("appended_at")
        or routeback_payload.get("canonical_content_sha256")
        != routeback_proof.get("canonical_content_sha256")
        or routeback_frame.value["observed_at_unix_ms"]
        <= restart.restart_completed_at_unix_ms
    ):
        _fail("goal_continuation_evidence_invalid")
    terminal_routeback = _goal_bound_receipt(
        schema=evidence_contract.GOAL_TERMINAL_ROUTEBACK_RECEIPT_SCHEMA,
        digest_field="receipt_sha256",
        fixture=fixture,
        fixture_sha256=fixture_sha256,
        owner_approval_receipt_sha256=owner_approval_receipt_sha256,
        fields={
            "session_id": session_id,
            "goal_generation_id": generation_id,
            "case_id": projection_binding.case_id,
            "event_id": routeback_event["event_id"],
            "event_sha256": _sha256_json(routeback_event),
            "canonical_content_sha256": routeback_proof[
                "canonical_content_sha256"
            ],
            "message_id": routeback_receipt["message_id"],
            "channel_id": routeback_receipt["channel_id"],
            "content_sha256": routeback_receipt["content_sha256"],
            "public_receipt_sha256": routeback_receipt[
                "public_receipt_sha256"
            ],
            "adapter_receipt": True,
            "receipt_readback_verified": True,
            "writer_origin": routeback_proof["origin"],
            "writer_appended_at": routeback_proof["appended_at"],
            "observed_at_unix_ms": routeback_frame.value[
                "observed_at_unix_ms"
            ],
        },
    )

    api_pairs: list[tuple[GoalCollectedFrame, GoalCollectedFrame]] = []
    for frame in frame_list:
        if frame.value.get("event") != "goal_pre_api_request":
            continue
        matches = [
            candidate
            for candidate in frame_list
            if candidate.value.get("event") == "goal_post_api_request"
            and candidate.value.get("segment_id")
            == frame.value.get("segment_id")
            and candidate.value.get("turn_id") == frame.value.get("turn_id")
            and candidate.value.get("payload", {}).get(
                "api_request_id_sha256"
            )
            == frame.value.get("payload", {}).get("api_request_id_sha256")
        ]
        if len(matches) != 1:
            _fail("goal_continuation_evidence_invalid")
        api_pairs.append((frame, matches[0]))
    call_observations: list[Mapping[str, Any]] = []
    for ordinal, (pre_frame, post_frame) in enumerate(api_pairs, start=1):
        if pre_frame.service_identity == restart.pre_service_identity:
            phase = "pre_restart"
        elif pre_frame.service_identity == restart.post_service_identity:
            phase = "post_restart"
        else:
            _fail("goal_continuation_evidence_invalid")
        if post_frame.service_identity != pre_frame.service_identity:
            _fail("goal_continuation_evidence_invalid")
        call_observations.append(
            {
                "ordinal": ordinal,
                "phase": phase,
                "turn_id": pre_frame.value["turn_id"],
                "gateway_process_identity_sha256": (
                    pre_frame.service_identity.identity_sha256
                ),
                "pre_api_request_frame_sha256": pre_frame.frame_sha256,
                "post_api_request_frame_sha256": post_frame.frame_sha256,
                "system_prompt_sha256": pre_frame.value["payload"][
                    "system_prompt_sha256"
                ],
                "tool_schema_sha256": pre_frame.value["payload"][
                    "tool_schema_sha256"
                ],
                "response_model_sha256": post_frame.value["payload"][
                    "response_model_sha256"
                ],
                "started_at_unix_ms": pre_frame.value["payload"][
                    "started_at_unix_ms"
                ],
                "completed_at_unix_ms": post_frame.value["payload"][
                    "response_observed_at_unix_ms"
                ],
            }
        )
    if (
        len(api_pairs) < len(model_outcomes)
        or not any(item["phase"] == "pre_restart" for item in call_observations)
        or not any(item["phase"] == "post_restart" for item in call_observations)
        or any(
            item["system_prompt_sha256"]
            != call_observations[0]["system_prompt_sha256"]
            or item["tool_schema_sha256"]
            != call_observations[0]["tool_schema_sha256"]
            for item in call_observations[1:]
        )
    ):
        _fail("goal_continuation_evidence_invalid")
    model_route = _goal_bound_receipt(
        schema=evidence_contract.MODEL_ROUTE_RECEIPT_SCHEMA,
        digest_field="receipt_sha256",
        fixture=fixture,
        fixture_sha256=fixture_sha256,
        owner_approval_receipt_sha256=owner_approval_receipt_sha256,
        fields={
            "provider": "openai-codex",
            "api_mode": "codex_responses",
            "model": "gpt-5.6-sol",
            "base_url_sha256": _sha256_bytes(
                b"https://chatgpt.com/backend-api/codex"
            ),
            "fallback_configured": False,
            "fallback_used": False,
            "model_call_count": len(call_observations),
            "response_model_sha256s": [
                item["response_model_sha256"] for item in call_observations
            ],
            "api_call_observations": call_observations,
            "observed_at_unix_ms": max(
                item["completed_at_unix_ms"] for item in call_observations
            ),
        },
    )
    prompt_stability = _goal_bound_receipt(
        schema=evidence_contract.PROMPT_TOOL_STABILITY_RECEIPT_SCHEMA,
        digest_field="receipt_sha256",
        fixture=fixture,
        fixture_sha256=fixture_sha256,
        owner_approval_receipt_sha256=owner_approval_receipt_sha256,
        fields={
            "pre_restart_system_prompt_sha256": call_observations[0][
                "system_prompt_sha256"
            ],
            "post_restart_system_prompt_sha256": call_observations[-1][
                "system_prompt_sha256"
            ],
            "pre_restart_tool_schema_sha256": call_observations[0][
                "tool_schema_sha256"
            ],
            "post_restart_tool_schema_sha256": call_observations[-1][
                "tool_schema_sha256"
            ],
            "prompt_cache_stable": True,
            "tool_schema_stable": True,
            "process_reconstruction_exact": True,
            "observed_at_unix_ms": model_route["observed_at_unix_ms"],
        },
    )
    preemption_e2e = _goal_bound_receipt(
        schema=evidence_contract.USER_PREEMPTION_QUEUE_E2E_RECEIPT_SCHEMA,
        digest_field="receipt_sha256",
        fixture=fixture,
        fixture_sha256=fixture_sha256,
        owner_approval_receipt_sha256=owner_approval_receipt_sha256,
        fields={
            "queue_path": "gateway.GatewayRunner._post_turn_goal_continuation",
            "owner_event_id": preemption_row.event.event_id,
            "owner_event_sha256": preemption_row.event.sha256,
            "goal_ingress_connector_row_sha256": goal_row.row_sha256,
            "preemption_ingress_connector_row_sha256": (
                preemption_row.row_sha256
            ),
            "automatic_continuation_was_pending": True,
            "real_user_event_preempted_automatic_continuation": True,
            "queued_user_event_identity_preserved": True,
            "connector_ack_readback_after_restart": True,
            "transport_redelivery_required": False,
            "duplicate_transport_delivery_policy": (
                "same_delivery_id_exact_ack_replay_only"
            ),
            "duplicate_model_turn_count": 0,
            "live_observed": True,
            "observed_at_unix_ms": max(
                recovery["restored_at_unix_ms"],
                preemption["preempted_at_unix_ms"],
            ),
        },
    )
    isolation_projection = (
        evidence_contract.normalize_goal_continuation_isolation_projection(
            fixture=fixture
        )
    )
    terminal_completed_at = max(
        model_outcomes[-1]["observed_at_unix_ms"],
        ctw_recovery["observed_at_unix_ms"],
        model_route["observed_at_unix_ms"],
        prompt_stability["observed_at_unix_ms"],
        preemption_e2e["observed_at_unix_ms"],
        terminal_routeback["observed_at_unix_ms"],
    )
    terminal = _goal_bound_receipt(
        schema=evidence_contract.GOAL_CONTINUATION_TERMINAL_SCHEMA,
        digest_field="terminal_sha256",
        fixture=fixture,
        fixture_sha256=fixture_sha256,
        owner_approval_receipt_sha256=owner_approval_receipt_sha256,
        fields={
            "discord_owner_ingress_receipt_sha256": ingress["receipt_sha256"],
            "session_id": session_id,
            "goal_generation_id": generation_id,
            "continue_outcome_receipt_sha256s": [
                item["receipt_sha256"]
                for item in model_outcomes
                if item["outcome"] == "continue"
            ],
            "completion_outcome_receipt_sha256": model_outcomes[-1][
                "receipt_sha256"
            ],
            "gateway_restart_receipt_sha256": gateway_restart[
                "receipt_sha256"
            ],
            "ctw_recovery_receipt_sha256": ctw_recovery["receipt_sha256"],
            "model_route_receipt_sha256": model_route["receipt_sha256"],
            "prompt_tool_stability_receipt_sha256": prompt_stability[
                "receipt_sha256"
            ],
            "user_preemption_queue_e2e_receipt_sha256": preemption_e2e[
                "receipt_sha256"
            ],
            "terminal_routeback_receipt_sha256": terminal_routeback[
                "receipt_sha256"
            ],
            "production_diff_sha256": production_diff.get("diff_sha256"),
            "isolation_equivalence_projection": isolation_projection,
            "isolation_equivalence_projection_sha256": _sha256_json(
                isolation_projection
            ),
            "completed_at_unix_ms": terminal_completed_at,
        },
    )
    evidence = _goal_bound_receipt(
        schema=evidence_contract.GOAL_CONTINUATION_EVIDENCE_SCHEMA,
        digest_field="evidence_sha256",
        fixture=fixture,
        fixture_sha256=fixture_sha256,
        owner_approval_receipt_sha256=owner_approval_receipt_sha256,
        fields={
            "discord_owner_ingress": ingress,
            "model_outcomes": model_outcomes,
            "gateway_restart": gateway_restart,
            "ctw_recovery": ctw_recovery,
            "model_route": model_route,
            "prompt_tool_stability": prompt_stability,
            "user_preemption_queue_e2e": preemption_e2e,
            "terminal_routeback": terminal_routeback,
            "terminal": terminal,
        },
    )
    try:
        validated = evidence_contract._validate_goal_continuation_evidence(
            evidence,
            fixture=fixture,
            fixture_sha256=fixture_sha256,
            owner_approval_receipt_sha256=owner_approval_receipt_sha256,
        )
    except Exception as exc:
        raise GoalLiveEvidenceError(
            "goal_continuation_evidence_invalid"
        ) from exc
    return copy.deepcopy(dict(validated))


__all__ = [
    "API_OBSERVER_RETIREMENT_PATH",
    "API_OBSERVER_RETIREMENT_SCHEMA",
    "ConnectorJournalRow",
    "GatewayRestartObservation",
    "GatewayServiceIdentity",
    "GoalCanonicalProjectionBinding",
    "GoalCollectedFrame",
    "GoalLiveEvidenceError",
    "GoalNativeReceiptBundle",
    "GoalOwnerChallenge",
    "SegmentedGoalEvidenceCollector",
    "build_goal_continuation_evidence",
    "normalize_goal_observer_public_target",
    "read_collector_process_identity",
    "read_connector_rows",
    "read_gateway_service_identity",
    "read_goal_native_finalizations",
    "read_goal_native_receipts",
    "read_state_meta",
    "read_writer_projection_export",
    "validate_packaged_goal_plugin_module",
    "validate_goal_canonical_projection_binding",
]

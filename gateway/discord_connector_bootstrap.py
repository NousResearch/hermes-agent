#!/usr/bin/env python3
"""Fail-closed bootstrap for the privileged public Discord connector.

Normal service startup opens an already bootstrapped journal and reads the bot
token from one connector-owned credential file.  The gateway never imports
this module and never receives the credential.  Journal creation is an
explicit operator action, not a startup side effect.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pwd
import signal
import stat
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import Any, Mapping, Sequence

from gateway.discord_connector_service import (
    DEFAULT_DISCORD_CONNECTOR_JOURNAL,
    DEFAULT_DISCORD_CONNECTOR_SOCKET,
    DEFAULT_DISCORD_CONNECTOR_UNIT,
    DEFAULT_GATEWAY_UNIT,
    DiscordConnectorHistoryReaderPeer,
    DiscordConnectorRuntime,
    DiscordConnectorUnixServer,
    DurableDiscordConnectorJournal,
    SystemctlDiscordEdgeMainPidProvider,
)
from gateway.discord_history_authority import (
    CANARY_HISTORY_READER_SERVICE_UNIT,
    CANARY_HISTORY_READER_SERVICE_USER,
    CANARY_REQUESTER_USER_ID,
)
from gateway.canonical_writer_readiness import (
    module_file_identity,
    notify_systemd_attestation,
    process_start_time_ticks,
)
from gateway.discord_connector_protocol import (
    DiscordConnectorTarget,
    canonical_json_bytes,
    sha256_json,
)
from plugins.platforms.discord.public_connector import (
    DiscordPublicConnectorClient,
    DiscordPublicConnectorPolicy,
)

DEFAULT_CONFIG_PATH = Path("/etc/muncho/discord-public-connector.json")
DEFAULT_READINESS_PATH = Path("/run/muncho-discord-connector/readiness.json")
READINESS_SCHEMA = "muncho-discord-public-connector-readiness.v2"
_MAX_CONFIG_BYTES = 32 * 1024
_MAX_READINESS_BYTES = 64 * 1024
_MAX_TOKEN_BYTES = 512
_ROOT_KEYS = frozenset({"service", "discord", "journal"})
_SERVICE_KEYS = frozenset({
    "socket_path",
    "gateway_unit",
    "connector_unit",
    "gateway_uid",
    "connector_uid",
    "connector_gid",
    "canary_history_reader",
    "connection_timeout_seconds",
})
_DISCORD_KEYS = frozenset({
    "token_file",
    "credentials_directory",
    "allowed_guild_ids",
    "allowed_channel_ids",
    "allowed_user_ids",
    "allowed_role_ids",
    "free_response_channel_ids",
    "public_only",
    "author_policy",
    "allow_bot_authors",
    "require_mention",
    "auto_thread",
    "thread_require_mention",
    "reviewed_cron_history_targets",
    "ready_timeout_seconds",
    "request_timeout_seconds",
})
_JOURNAL_KEYS = frozenset({"path", "busy_timeout_ms"})


@dataclass(frozen=True)
class DiscordConnectorConfig:
    config_path: Path
    socket_path: Path
    gateway_unit: str
    connector_unit: str
    gateway_uid: int
    connector_uid: int
    connector_gid: int
    connection_timeout_seconds: float
    token_file: Path
    credentials_directory: Path
    policy: DiscordPublicConnectorPolicy
    ready_timeout_seconds: float
    request_timeout_seconds: float
    journal_path: Path
    journal_busy_timeout_ms: int
    canary_history_reader: DiscordConnectorHistoryReaderPeer | None = None


@dataclass
class DiscordConnectorBootstrap:
    config: DiscordConnectorConfig
    journal: DurableDiscordConnectorJournal
    client: DiscordPublicConnectorClient
    runtime: DiscordConnectorRuntime
    server: DiscordConnectorUnixServer

    def close(self) -> None:
        self.server.shutdown()
        self.client.stop()


def _duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Discord connector config has duplicate keys")
        result[key] = value
    return result


def _no_constant(value: str) -> None:
    raise ValueError(f"Discord connector config has non-JSON constant: {value}")


def _strict(value: Any, *, keys: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be an object")
    actual = set(value)
    if actual != set(keys):
        raise ValueError(f"{label} fields are invalid")
    return value


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        raise ValueError(f"{label} is invalid")
    return value


def _number(value: Any, label: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is invalid")
    result = float(value)
    if not minimum <= result <= maximum:
        raise ValueError(f"{label} is invalid")
    return result


def _path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value or any(ord(char) < 32 for char in value):
        raise ValueError(f"{label} is invalid")
    raw = Path(value)
    if (
        not raw.is_absolute()
        or raw != Path(os.path.normpath(os.fspath(raw)))
        or ".." in raw.parts
    ):
        raise ValueError(f"{label} is invalid")
    return raw


def _snowflake_list(
    value: Any,
    label: str,
    *,
    allow_empty: bool = False,
) -> list[str]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ValueError(f"{label} is invalid")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.isdigit() or item.startswith("0"):
            raise ValueError(f"{label} is invalid")
        result.append(item)
    if len(set(result)) != len(result):
        raise ValueError(f"{label} has duplicates")
    return result


def _reviewed_cron_history_targets(value: Any) -> dict[str, list[str]]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise ValueError("reviewed_cron_history_targets is invalid")
    result: dict[str, list[str]] = {}
    for job_id, targets in sorted(value.items()):
        if (
            len(job_id) != 12
            or any(char not in "0123456789abcdef" for char in job_id)
        ):
            raise ValueError("reviewed_cron_history_targets is invalid")
        result[job_id] = _snowflake_list(
            targets,
            f"reviewed_cron_history_targets.{job_id}",
        )
    return result


def _canary_history_reader(value: Any) -> DiscordConnectorHistoryReaderPeer | None:
    if value is None:
        return None
    raw = _strict(
        value,
        keys=frozenset({"service_unit", "service_user", "requester_user_id"}),
        label="canary_history_reader",
    )
    if raw != {
        "service_unit": CANARY_HISTORY_READER_SERVICE_UNIT,
        "service_user": CANARY_HISTORY_READER_SERVICE_USER,
        "requester_user_id": CANARY_REQUESTER_USER_ID,
    }:
        raise ValueError("canary_history_reader is not the pinned canary peer")
    try:
        user = pwd.getpwnam(CANARY_HISTORY_READER_SERVICE_USER)
    except (KeyError, OSError) as exc:
        raise PermissionError("canary history-reader identity is unavailable") from exc
    if (
        user.pw_name != CANARY_HISTORY_READER_SERVICE_USER
        or user.pw_uid < 1
        or user.pw_gid < 1
        or user.pw_dir != "/nonexistent"
        or user.pw_shell != "/usr/sbin/nologin"
    ):
        raise PermissionError("canary history-reader identity is invalid")
    return DiscordConnectorHistoryReaderPeer(
        service_unit=CANARY_HISTORY_READER_SERVICE_UNIT,
        expected_uid=user.pw_uid,
        requester_user_id=CANARY_REQUESTER_USER_ID,
    )


def _trusted_file(
    path: Path,
    *,
    owner_uid: int,
    owner_gid: int | None,
    mode: int,
    max_bytes: int,
) -> bytes:
    expected = os.lstat(path)
    if (
        stat.S_ISLNK(expected.st_mode)
        or not stat.S_ISREG(expected.st_mode)
        or expected.st_nlink != 1
        or expected.st_uid != owner_uid
        or (owner_gid is not None and expected.st_gid != owner_gid)
        or stat.S_IMODE(expected.st_mode) != mode
        or expected.st_size <= 0
        or expected.st_size > max_bytes
    ):
        raise PermissionError("Discord connector trusted file identity is invalid")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        actual = os.fstat(descriptor)
        if (
            (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino)
            or actual.st_uid != expected.st_uid
            or actual.st_gid != expected.st_gid
            or actual.st_nlink != 1
            or stat.S_IMODE(actual.st_mode) != mode
            or actual.st_size != expected.st_size
        ):
            raise PermissionError("Discord connector trusted file changed")
        chunks: list[bytes] = []
        remaining = expected.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                raise PermissionError("Discord connector trusted file was truncated")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise PermissionError("Discord connector trusted file grew")
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        path_after = os.lstat(path)
        identity = (
            expected.st_dev,
            expected.st_ino,
            expected.st_size,
            expected.st_mtime_ns,
            expected.st_ctime_ns,
        )
        if any(
            (
                item.st_dev,
                item.st_ino,
                item.st_size,
                item.st_mtime_ns,
                item.st_ctime_ns,
            )
            != identity
            for item in (after, path_after)
        ):
            raise PermissionError("Discord connector trusted file changed")
        if not raw or len(raw) > max_bytes:
            raise ValueError("Discord connector trusted file size is invalid")
        return raw
    finally:
        os.close(descriptor)


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> DiscordConnectorConfig:
    raw = _trusted_file(
        path,
        owner_uid=0,
        owner_gid=None,
        mode=0o440,
        max_bytes=_MAX_CONFIG_BYTES,
    )
    try:
        decoded = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_duplicates,
            parse_constant=_no_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Discord connector config JSON is invalid") from exc
    root = _strict(decoded, keys=_ROOT_KEYS, label="config")
    if not isinstance(root["service"], Mapping):
        raise ValueError("service must be an object")
    service_value = dict(root["service"])
    service_value.setdefault("canary_history_reader", None)
    service = _strict(service_value, keys=_SERVICE_KEYS, label="service")
    discord_value = dict(root["discord"])
    discord_value.setdefault("free_response_channel_ids", [])
    discord_value.setdefault("public_only", True)
    discord_value.setdefault("author_policy", "exact_ids_or_roles")
    discord_value.setdefault("reviewed_cron_history_targets", {})
    discord_config = _strict(discord_value, keys=_DISCORD_KEYS, label="discord")
    journal = _strict(root["journal"], keys=_JOURNAL_KEYS, label="journal")

    socket_path = _path(service["socket_path"], "socket_path")
    journal_path = _path(journal["path"], "journal.path")
    if socket_path != DEFAULT_DISCORD_CONNECTOR_SOCKET:
        raise ValueError("Discord connector socket is not pinned")
    if journal_path != DEFAULT_DISCORD_CONNECTOR_JOURNAL:
        raise ValueError("Discord connector journal is not pinned")
    if service["gateway_unit"] != DEFAULT_GATEWAY_UNIT:
        raise ValueError("Discord connector gateway unit is not pinned")
    if service["connector_unit"] != DEFAULT_DISCORD_CONNECTOR_UNIT:
        raise ValueError("Discord connector unit is not pinned")

    connector_uid = _integer(
        service["connector_uid"], "connector_uid", 1, (1 << 31) - 1
    )
    connector_gid = _integer(
        service["connector_gid"], "connector_gid", 1, (1 << 31) - 1
    )
    if os.lstat(path).st_gid != connector_gid:
        raise PermissionError("Discord connector config group is not trusted")
    credentials_directory = _path(
        discord_config["credentials_directory"], "credentials_directory"
    )
    token_file = _path(discord_config["token_file"], "token_file")
    if token_file.parent != credentials_directory:
        raise ValueError("Discord connector token must be a direct credential file")
    if type(discord_config["allow_bot_authors"]) is not bool:
        raise ValueError("allow_bot_authors must be boolean")
    for flag in (
        "require_mention",
        "auto_thread",
        "thread_require_mention",
        "public_only",
    ):
        if type(discord_config[flag]) is not bool:
            raise ValueError(f"{flag} must be boolean")

    policy = DiscordPublicConnectorPolicy.build(
        allowed_guild_ids=_snowflake_list(
            discord_config["allowed_guild_ids"], "allowed_guild_ids"
        ),
        allowed_channel_ids=_snowflake_list(
            discord_config["allowed_channel_ids"], "allowed_channel_ids"
        ),
        allowed_user_ids=_snowflake_list(
            discord_config["allowed_user_ids"],
            "allowed_user_ids",
            allow_empty=discord_config["author_policy"] == "guild_acl",
        ),
        allowed_role_ids=_snowflake_list(
            discord_config["allowed_role_ids"],
            "allowed_role_ids",
            allow_empty=discord_config["author_policy"] == "guild_acl",
        ),
        free_response_channel_ids=_snowflake_list(
            discord_config["free_response_channel_ids"],
            "free_response_channel_ids",
            allow_empty=True,
        ),
        public_only=discord_config["public_only"],
        author_policy=discord_config["author_policy"],
        allow_bot_authors=discord_config["allow_bot_authors"],
        require_mention=discord_config["require_mention"],
        auto_thread=discord_config["auto_thread"],
        thread_require_mention=discord_config["thread_require_mention"],
        reviewed_cron_history_targets=_reviewed_cron_history_targets(
            discord_config["reviewed_cron_history_targets"]
        ),
    )
    return DiscordConnectorConfig(
        config_path=path,
        socket_path=socket_path,
        gateway_unit=str(service["gateway_unit"]),
        connector_unit=str(service["connector_unit"]),
        gateway_uid=_integer(service["gateway_uid"], "gateway_uid", 1, (1 << 31) - 1),
        connector_uid=connector_uid,
        connector_gid=connector_gid,
        connection_timeout_seconds=_number(
            service["connection_timeout_seconds"],
            "connection_timeout_seconds",
            1,
            30,
        ),
        token_file=token_file,
        credentials_directory=credentials_directory,
        policy=policy,
        ready_timeout_seconds=_number(
            discord_config["ready_timeout_seconds"],
            "ready_timeout_seconds",
            1,
            120,
        ),
        request_timeout_seconds=_number(
            discord_config["request_timeout_seconds"],
            "request_timeout_seconds",
            1,
            30,
        ),
        journal_path=journal_path,
        journal_busy_timeout_ms=_integer(
            journal["busy_timeout_ms"], "busy_timeout_ms", 1, 30_000
        ),
        canary_history_reader=_canary_history_reader(
            service["canary_history_reader"]
        ),
    )


def _require_service_identity(config: DiscordConnectorConfig) -> None:
    if os.geteuid() != config.connector_uid or os.getegid() != config.connector_gid:
        raise PermissionError("Discord connector process identity is invalid")
    credential_stat = os.lstat(config.credentials_directory)
    if (
        stat.S_ISLNK(credential_stat.st_mode)
        or not stat.S_ISDIR(credential_stat.st_mode)
        or credential_stat.st_uid != config.connector_uid
        or credential_stat.st_gid != config.connector_gid
        or stat.S_IMODE(credential_stat.st_mode) & 0o077
    ):
        raise PermissionError("Discord connector credential directory is invalid")


def _require_existing_journal(config: DiscordConnectorConfig) -> None:
    journal = os.lstat(config.journal_path)
    if (
        stat.S_ISLNK(journal.st_mode)
        or not stat.S_ISREG(journal.st_mode)
        or journal.st_nlink != 1
        or journal.st_uid != config.connector_uid
        or journal.st_gid != config.connector_gid
        or stat.S_IMODE(journal.st_mode) != 0o600
    ):
        raise PermissionError("Discord connector journal identity is invalid")


def _load_token(config: DiscordConnectorConfig) -> str:
    raw = _trusted_file(
        config.token_file,
        owner_uid=config.connector_uid,
        owner_gid=config.connector_gid,
        mode=0o400,
        max_bytes=_MAX_TOKEN_BYTES,
    )
    try:
        token = raw.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise ValueError("Discord connector token encoding is invalid") from exc
    if (
        not token
        or len(token) > _MAX_TOKEN_BYTES
        or any(char.isspace() for char in token)
    ):
        raise ValueError("Discord connector token is invalid")
    return token


def bootstrap(config: DiscordConnectorConfig) -> DiscordConnectorBootstrap:
    _require_service_identity(config)
    _require_existing_journal(config)
    token = _load_token(config)
    journal = DurableDiscordConnectorJournal(
        config.journal_path,
        busy_timeout_ms=config.journal_busy_timeout_ms,
    )
    client = DiscordPublicConnectorClient(
        token,
        policy=config.policy,
        event_sink=journal.offer_event,
        ready_timeout_seconds=config.ready_timeout_seconds,
        request_timeout_seconds=config.request_timeout_seconds,
    )
    runtime = DiscordConnectorRuntime(backend=client, journal=journal)
    server = DiscordConnectorUnixServer(
        config.socket_path,
        runtime=runtime,
        expected_gateway_uid=config.gateway_uid,
        gateway_unit=config.gateway_unit,
        main_pid_provider=SystemctlDiscordEdgeMainPidProvider(),
        history_reader_peer=config.canary_history_reader,
        connection_timeout_seconds=config.connection_timeout_seconds,
    )
    client.start()
    return DiscordConnectorBootstrap(config, journal, client, runtime, server)


def _bootstrap_journal(config: DiscordConnectorConfig) -> None:
    _require_service_identity(config)
    parent = os.lstat(config.journal_path.parent)
    if (
        stat.S_ISLNK(parent.st_mode)
        or not stat.S_ISDIR(parent.st_mode)
        or parent.st_uid != config.connector_uid
        or parent.st_gid != config.connector_gid
        or stat.S_IMODE(parent.st_mode) & 0o077
    ):
        raise PermissionError("Discord connector journal parent is invalid")
    created = DurableDiscordConnectorJournal.bootstrap(
        config.journal_path,
        busy_timeout_ms=config.journal_busy_timeout_ms,
    )
    os.chmod(created.path, 0o600)


def _file_identity(
    path: Path,
    *,
    mode: int,
    uid: int,
    gid: int,
) -> dict[str, Any]:
    item = os.lstat(path)
    if (
        stat.S_ISLNK(item.st_mode)
        or not stat.S_ISREG(item.st_mode)
        or item.st_nlink != 1
        or item.st_uid != uid
        or item.st_gid != gid
        or stat.S_IMODE(item.st_mode) != mode
    ):
        raise PermissionError("Discord connector runtime file identity is invalid")
    return {
        "path": str(path),
        "device": item.st_dev,
        "inode": item.st_ino,
        "uid": item.st_uid,
        "gid": item.st_gid,
        "mode": f"{mode:04o}",
    }


def _write_readiness_receipt(
    path: Path,
    value: Mapping[str, Any],
    *,
    uid: int,
    gid: int,
) -> tuple[int, int]:
    parent = os.lstat(path.parent)
    if (
        stat.S_ISLNK(parent.st_mode)
        or not stat.S_ISDIR(parent.st_mode)
        or parent.st_uid != uid
        or parent.st_gid != gid
        or stat.S_IMODE(parent.st_mode) & 0o007
    ):
        raise PermissionError("Discord connector readiness parent is invalid")
    if os.path.lexists(path):
        raise FileExistsError("Discord connector readiness already exists")
    payload = canonical_json_bytes(value)
    if not payload or len(payload) > _MAX_READINESS_BYTES:
        raise ValueError("Discord connector readiness is oversized")
    temporary = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o400)
    try:
        view = memoryview(payload)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise OSError("Discord connector readiness write made no progress")
            written += count
        os.fchmod(descriptor, 0o400)
        os.fchown(descriptor, uid, gid)
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise
    os.close(descriptor)
    installed_identity: tuple[int, int] | None = None
    try:
        os.link(temporary, path, follow_symlinks=False)
        linked = os.lstat(path)
        installed_identity = (linked.st_dev, linked.st_ino)
        temporary.unlink()
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        if installed_identity is not None:
            _unlink_exact_and_fsync(path, installed_identity)
        raise
    installed = os.lstat(path)
    if (
        stat.S_ISLNK(installed.st_mode)
        or not stat.S_ISREG(installed.st_mode)
        or installed.st_nlink != 1
        or installed.st_uid != uid
        or installed.st_gid != gid
        or stat.S_IMODE(installed.st_mode) != 0o400
        or installed.st_size != len(payload)
    ):
        raise PermissionError("Discord connector readiness install is invalid")
    return installed.st_dev, installed.st_ino


def _fsync_directory(path: Path) -> None:
    directory = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _unlink_exact_and_fsync(path: Path, identity: tuple[int, int]) -> None:
    try:
        current = os.lstat(path)
    except FileNotFoundError:
        return
    if (current.st_dev, current.st_ino) != identity:
        raise RuntimeError("Discord connector readiness identity changed")
    path.unlink()
    _fsync_directory(path.parent)


def build_readiness_receipt(
    running: DiscordConnectorBootstrap,
    *,
    now_unix: int | None = None,
    pid: int | None = None,
    start_time_reader=process_start_time_ticks,
) -> dict[str, Any]:
    """Build one exact, secret-free proof after real Discord and socket ready."""

    config = running.config
    process_id = os.getpid() if pid is None else pid
    if type(process_id) is not int or process_id < 2:
        raise ValueError("Discord connector MainPID is invalid")
    if os.geteuid() != config.connector_uid or os.getegid() != config.connector_gid:
        raise PermissionError("Discord connector readiness process identity drifted")
    client = running.client.readiness_identity()
    listener = running.server.readiness_identity()
    cleanup = running.journal.cleanup_snapshot()
    if cleanup.get("unresolved_dispatch_count") != 0:
        raise RuntimeError("Discord connector has unresolved dispatch state")
    config_raw = _trusted_file(
        config.config_path,
        owner_uid=0,
        owner_gid=config.connector_gid,
        mode=0o440,
        max_bytes=_MAX_CONFIG_BYTES,
    )
    module_path, module_sha256 = module_file_identity(
        str(Path(__file__).resolve(strict=True))
    )
    ready_at = int(time.time()) if now_unix is None else now_unix
    if type(ready_at) is not int or ready_at <= 0:
        raise ValueError("Discord connector readiness time is invalid")
    unsigned = {
        "schema": READINESS_SCHEMA,
        "service_unit": config.connector_unit,
        "operation_class": "ordinary_public_ingress_and_session_replies",
        "main_pid": process_id,
        "process_start_time_ticks": start_time_reader(process_id),
        "process_uid": os.geteuid(),
        "process_gid": os.getegid(),
        "config_path": str(config.config_path),
        "config_sha256": hashlib.sha256(config_raw).hexdigest(),
        "module_path": module_path,
        "module_sha256": module_sha256,
        "socket": listener,
        "journal": _file_identity(
            config.journal_path,
            mode=0o600,
            uid=config.connector_uid,
            gid=config.connector_gid,
        ),
        "journal_cleanup": cleanup,
        "token_lease": _file_identity(
            config.token_file,
            mode=0o400,
            uid=config.connector_uid,
            gid=config.connector_gid,
        ),
        "canary_history_reader": running.server.history_reader_identity(),
        "discord": client,
        "allowed_guild_ids": sorted(config.policy.allowed_guild_ids),
        "allowed_channel_ids": sorted(config.policy.allowed_channel_ids),
        "allowed_user_ids": sorted(config.policy.allowed_user_ids),
        "allowed_role_ids": sorted(config.policy.allowed_role_ids),
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
        "ready_at_unix": ready_at,
    }
    return {**unsigned, "receipt_sha256": sha256_json(unsigned)}


_READINESS_FIELDS = frozenset({
    "schema",
    "service_unit",
    "operation_class",
    "main_pid",
    "process_start_time_ticks",
    "process_uid",
    "process_gid",
    "config_path",
    "config_sha256",
    "module_path",
    "module_sha256",
    "socket",
    "journal",
    "journal_cleanup",
    "token_lease",
    "canary_history_reader",
    "discord",
    "allowed_guild_ids",
    "allowed_channel_ids",
    "allowed_user_ids",
    "allowed_role_ids",
    "secret_material_recorded",
    "secret_digest_recorded",
    "ready_at_unix",
    "receipt_sha256",
})


def load_readiness_receipt(
    config: DiscordConnectorConfig,
    path: Path = DEFAULT_READINESS_PATH,
    *,
    start_time_reader=process_start_time_ticks,
) -> dict[str, Any]:
    raw = _trusted_file(
        path,
        owner_uid=config.connector_uid,
        owner_gid=config.connector_gid,
        mode=0o400,
        max_bytes=_MAX_READINESS_BYTES,
    )
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_duplicates,
            parse_constant=_no_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Discord connector readiness JSON is invalid") from exc
    receipt = _strict(value, keys=_READINESS_FIELDS, label="readiness")
    unsigned = {key: item for key, item in receipt.items() if key != "receipt_sha256"}
    config_raw = _trusted_file(
        config.config_path,
        owner_uid=0,
        owner_gid=config.connector_gid,
        mode=0o440,
        max_bytes=_MAX_CONFIG_BYTES,
    )
    discord = receipt.get("discord")
    token = receipt.get("token_lease")
    journal = receipt.get("journal")
    socket_identity = receipt.get("socket")
    cleanup = receipt.get("journal_cleanup")
    history_reader = receipt.get("canary_history_reader")
    identity_fields = frozenset({"path", "device", "inode", "uid", "gid", "mode"})
    token_value = _strict(token, keys=identity_fields, label="token_lease")
    journal_value = _strict(journal, keys=identity_fields, label="journal")
    socket_value = _strict(
        socket_identity,
        keys=frozenset({
            "socket_path",
            "socket_device",
            "socket_inode",
            "socket_uid",
            "socket_gid",
            "socket_mode",
            "listening",
        }),
        label="socket",
    )
    discord_value = _strict(
        discord,
        keys=frozenset({
            "discord_gateway_ready",
            "bot_user_id",
            "intents",
            "dm_messages",
            "require_mention",
            "auto_thread",
            "thread_require_mention",
            "public_only",
            "author_policy",
            "free_response_channel_ids",
            "allowed_user_ids",
            "allowed_role_ids",
            "allowed_channel_ids",
            "reviewed_cron_history_targets_sha256",
            "public_target_proofs",
        }),
        label="discord",
    )
    cleanup_value = _strict(
        cleanup,
        keys=frozenset({
            "schema",
            "event_state_counts",
            "send_state_counts",
            "unresolved_dispatch_count",
            "unacked_event_count",
            "safe_to_retire",
        }),
        label="journal_cleanup",
    )
    expected_token = _file_identity(
        config.token_file,
        mode=0o400,
        uid=config.connector_uid,
        gid=config.connector_gid,
    )
    expected_journal = _file_identity(
        config.journal_path,
        mode=0o600,
        uid=config.connector_uid,
        gid=config.connector_gid,
    )
    module_path, module_sha256 = module_file_identity(
        str(Path(__file__).resolve(strict=True))
    )
    current_socket = os.lstat(config.socket_path)
    event_counts = cleanup_value.get("event_state_counts")
    send_counts = cleanup_value.get("send_state_counts")
    target_proofs = discord_value.get("public_target_proofs")

    def _counts_are_exact(value: Any, allowed: frozenset[str]) -> bool:
        return bool(
            isinstance(value, Mapping)
            and set(value).issubset(allowed)
            and all(type(count) is int and count >= 0 for count in value.values())
        )

    try:
        parsed_target_proofs = [
            DiscordConnectorTarget.from_mapping(item).to_mapping()
            for item in target_proofs
        ]
    except (TypeError, ValueError):
        parsed_target_proofs = []
    event_counts_exact = _counts_are_exact(
        event_counts, frozenset({"pending", "delivering", "acked"})
    )
    send_counts_exact = _counts_are_exact(
        send_counts,
        frozenset({"prepared", "dispatching", "verified", "blocked", "uncertain"}),
    )
    expected_unresolved = (
        sum(
            send_counts.get(state, 0)
            for state in ("prepared", "dispatching", "uncertain")
        )
        if send_counts_exact
        else -1
    )
    expected_unacked = (
        sum(event_counts.get(state, 0) for state in ("pending", "delivering"))
        if event_counts_exact
        else -1
    )
    if (
        receipt.get("schema") != READINESS_SCHEMA
        or receipt.get("receipt_sha256") != sha256_json(unsigned)
        or receipt.get("service_unit") != config.connector_unit
        or receipt.get("operation_class")
        != "ordinary_public_ingress_and_session_replies"
        or receipt.get("config_path") != str(config.config_path)
        or receipt.get("config_sha256") != hashlib.sha256(config_raw).hexdigest()
        or receipt.get("module_path") != module_path
        or receipt.get("module_sha256") != module_sha256
        or receipt.get("process_uid") != config.connector_uid
        or receipt.get("process_gid") != config.connector_gid
        or type(receipt.get("main_pid")) is not int
        or receipt["main_pid"] < 2
        or type(receipt.get("process_start_time_ticks")) is not int
        or receipt["process_start_time_ticks"] < 1
        or receipt["process_start_time_ticks"] != start_time_reader(receipt["main_pid"])
        or discord_value.get("discord_gateway_ready") is not True
        or discord_value.get("dm_messages") is not False
        or discord_value.get("require_mention") is not config.policy.require_mention
        or discord_value.get("auto_thread") is not config.policy.auto_thread
        or discord_value.get("thread_require_mention")
        is not config.policy.thread_require_mention
        or discord_value.get("public_only") is not config.policy.public_only
        or discord_value.get("author_policy") != config.policy.author_policy
        or discord_value.get("free_response_channel_ids")
        != sorted(config.policy.free_response_channel_ids)
        or discord_value.get("allowed_user_ids")
        != sorted(config.policy.allowed_user_ids)
        or discord_value.get("allowed_role_ids")
        != sorted(config.policy.allowed_role_ids)
        or discord_value.get("allowed_channel_ids")
        != sorted(config.policy.allowed_channel_ids)
        or discord_value.get("reviewed_cron_history_targets_sha256")
        != config.policy.reviewed_cron_history_targets_sha256
        or discord_value.get("intents")
        != ["guilds", "guild_messages", "message_content"]
        or not str(discord_value.get("bot_user_id") or "").isdigit()
        or str(discord_value.get("bot_user_id")).startswith("0")
        or not isinstance(target_proofs, list)
        or len(target_proofs) != len(config.policy.allowed_channel_ids)
        or parsed_target_proofs != target_proofs
        or sorted(item["channel_id"] for item in parsed_target_proofs)
        != sorted(config.policy.allowed_channel_ids)
        or any(
            item["guild_id"] not in config.policy.allowed_guild_ids
            for item in parsed_target_proofs
        )
        or token_value != expected_token
        or journal_value != expected_journal
        or not stat.S_ISSOCK(current_socket.st_mode)
        or socket_value.get("socket_path") != str(config.socket_path)
        or socket_value.get("socket_device") != current_socket.st_dev
        or socket_value.get("socket_inode") != current_socket.st_ino
        or socket_value.get("socket_uid") != current_socket.st_uid
        or socket_value.get("socket_gid") != current_socket.st_gid
        or socket_value.get("socket_mode")
        != f"{stat.S_IMODE(current_socket.st_mode):04o}"
        or socket_value.get("listening") is not True
        or cleanup_value.get("schema") != "discord-public-connector-cleanup-snapshot.v1"
        or not event_counts_exact
        or not send_counts_exact
        or cleanup_value.get("unresolved_dispatch_count") != expected_unresolved
        or cleanup_value.get("unacked_event_count") != expected_unacked
        or cleanup_value.get("safe_to_retire")
        is not (expected_unresolved == 0 and expected_unacked == 0)
        or expected_unresolved != 0
        or history_reader
        != (
            None
            if config.canary_history_reader is None
            else config.canary_history_reader.readiness_mapping()
        )
        or receipt.get("allowed_guild_ids") != sorted(config.policy.allowed_guild_ids)
        or receipt.get("allowed_channel_ids")
        != sorted(config.policy.allowed_channel_ids)
        or receipt.get("allowed_user_ids") != sorted(config.policy.allowed_user_ids)
        or receipt.get("allowed_role_ids") != sorted(config.policy.allowed_role_ids)
        or receipt.get("secret_material_recorded") is not False
        or receipt.get("secret_digest_recorded") is not False
        or type(receipt.get("ready_at_unix")) is not int
        or receipt["ready_at_unix"] <= 0
    ):
        raise ValueError("Discord connector readiness receipt is invalid")
    return dict(receipt)


def publish_readiness(
    running: DiscordConnectorBootstrap,
    *,
    path: Path = DEFAULT_READINESS_PATH,
    now_unix: int | None = None,
    start_time_reader=process_start_time_ticks,
    notifier=notify_systemd_attestation,
) -> dict[str, Any]:
    running.server.start()
    receipt = build_readiness_receipt(
        running,
        now_unix=now_unix,
        start_time_reader=start_time_reader,
    )
    identity = _write_readiness_receipt(
        path,
        receipt,
        uid=running.config.connector_uid,
        gid=running.config.connector_gid,
    )
    if not notifier(
        READINESS_SCHEMA,
        receipt["receipt_sha256"],
        ready=True,
    ):
        _unlink_exact_and_fsync(path, identity)
        raise RuntimeError("Discord connector requires systemd Type=notify")
    return receipt


def _watch_discord_health(
    running: DiscordConnectorBootstrap,
    stop: threading.Event,
    health_failed: threading.Event,
) -> None:
    """Turn a post-ready Discord loss into a bounded service failure."""

    while not stop.is_set():
        if running.client.wait_for_health_failure(0.2):
            if not stop.is_set():
                health_failed.set()
                running.server.shutdown()
            return


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Public Discord connector service")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--bootstrap-journal", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.bootstrap_journal:
        _bootstrap_journal(config)
        return 0

    running = bootstrap(config)
    stop = threading.Event()
    health_failed = threading.Event()

    def _stop(_signum: int, _frame: FrameType | None) -> None:
        stop.set()
        running.server.shutdown()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    monitor: threading.Thread | None = None
    try:
        publish_readiness(running)
        monitor = threading.Thread(
            target=_watch_discord_health,
            args=(running, stop, health_failed),
            name="discord-connector-health",
            daemon=False,
        )
        monitor.start()
        running.server.serve_forever()
    finally:
        stop.set()
        running.close()
        if monitor is not None:
            monitor.join(timeout=2)
    return 1 if health_failed.is_set() else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

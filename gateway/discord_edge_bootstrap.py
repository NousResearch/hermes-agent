#!/usr/bin/env python3
"""Fail-closed executable bootstrap for the privileged Discord egress edge.

The edge consumes one explicit root-owned JSON configuration file. Key and bot
credential material remain in separately owned local files; nothing is loaded
from environment variables, Secret Manager, or a model-controlled payload.

Normal startup opens an already initialized durable journal. Journal creation
is a separate, explicit ``--bootstrap-journal`` action and never occurs as a
side effect of service startup.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import stat
import threading
from dataclasses import dataclass, field
from pathlib import Path
from types import FrameType
from typing import Any, Callable, Mapping, Sequence

from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from gateway.canonical_writer_boundary import (
    DEFAULT_DISCORD_EDGE_SOCKET_PATH,
    DEFAULT_DISCORD_EDGE_UNIT,
    DEFAULT_GATEWAY_UNIT,
)
from gateway.discord_edge_protocol import ed25519_public_key_id
from gateway.discord_edge_runtime import (
    DiscordEdgeRuntime,
    DurableDiscordEdgeJournal,
)
from gateway.discord_edge_service import (
    DiscordEdgeUnixServer,
    SystemctlDiscordEdgeMainPidProvider,
)
from gateway.discord_rest_edge import DiscordRestEdgeAdapter


DEFAULT_JOURNAL_PATH = Path(
    "/var/lib/muncho-discord-egress/discord-edge-journal.sqlite3"
)

_MAX_CONFIG_BYTES = 32 * 1024
_MAX_KEY_BYTES = 8 * 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SYSTEMD_UNIT_RE = re.compile(r"^[A-Za-z0-9_.@:-]+\.service$")
_ROOT_KEYS = frozenset({"service", "keys", "discord", "journal", "runtime"})
_SERVICE_KEYS = frozenset(
    {
        "socket_path",
        "gateway_unit",
        "edge_unit",
        "gateway_uid",
        "edge_uid",
        "edge_gid",
        "connection_timeout_seconds",
        "max_connections",
    }
)
_KEY_KEYS = frozenset(
    {
        "writer_capability_public_key_file",
        "writer_capability_public_key_id",
        "edge_receipt_private_key_file",
        "edge_receipt_public_key_id",
    }
)
_DISCORD_KEYS = frozenset(
    {
        "token_file",
        "credentials_directory",
        "api_timeout_seconds",
    }
)
_JOURNAL_KEYS = frozenset({"path", "busy_timeout_ms"})
_RUNTIME_KEYS = frozenset({"max_proof_age_ms"})
_FORBIDDEN_EMBEDDED_SECRET_KEYS = frozenset(
    {
        "token",
        "bot_token",
        "private_key",
        "secret",
        "password",
        "credential",
        "credential_value",
    }
)


@dataclass(frozen=True)
class DiscordEdgeServiceConfig:
    socket_path: Path
    gateway_unit: str
    edge_unit: str
    gateway_uid: int
    edge_uid: int
    edge_gid: int
    connection_timeout_seconds: float
    max_connections: int
    writer_capability_public_key_file: Path
    writer_capability_public_key_id: str
    edge_receipt_private_key_file: Path
    edge_receipt_public_key_id: str
    writer_public_key: Ed25519PublicKey = field(repr=False, compare=False)
    edge_private_key: Ed25519PrivateKey = field(repr=False, compare=False)
    token_file: Path
    credentials_directory: Path
    api_timeout_seconds: float
    journal_path: Path
    journal_busy_timeout_ms: int
    max_proof_age_ms: int


@dataclass
class DiscordEdgeBootstrap:
    config: DiscordEdgeServiceConfig
    journal: DurableDiscordEdgeJournal
    adapter: DiscordRestEdgeAdapter
    runtime: DiscordEdgeRuntime
    server: DiscordEdgeUnixServer
    _closed: bool = field(default=False, init=False, repr=False)
    _close_lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
    )

    def close(self) -> None:
        """Idempotently stop local IPC before closing the HTTPS adapter."""

        with self._close_lock:
            if self._closed:
                return
            try:
                self.server.shutdown()
            finally:
                self.adapter.close()
            self._closed = True


def _strict_mapping(
    value: Any,
    *,
    label: str,
    allowed: frozenset[str],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} keys must be strings")
    unknown = sorted(set(value) - allowed)
    missing = sorted(allowed - set(value))
    if unknown:
        raise ValueError(f"{label} contains unknown fields: {','.join(unknown)}")
    if missing:
        raise ValueError(f"{label} is missing fields: {','.join(missing)}")
    return value


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    result = value.strip()
    if not result or any(ord(char) < 32 for char in result):
        raise ValueError(f"{label} is invalid")
    return result


def _integer(value: Any, label: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{label} is outside its bound")
    return value


def _number(value: Any, label: str, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    result = float(value)
    if not minimum <= result <= maximum:
        raise ValueError(f"{label} is outside its bound")
    return result


def _absolute_path(value: Any, label: str) -> Path:
    raw = Path(_required_text(value, label))
    normalized = Path(os.path.normpath(os.fspath(raw)))
    if not raw.is_absolute() or normalized != raw or ".." in raw.parts:
        raise ValueError(f"{label} must be an absolute normalized path")
    return raw


def _reject_embedded_secrets(value: Any) -> None:
    if isinstance(value, Mapping):
        for raw_key, nested in value.items():
            key = str(raw_key).strip().casefold()
            if key in _FORBIDDEN_EMBEDDED_SECRET_KEYS:
                raise ValueError("Discord edge config must not embed secret material")
            _reject_embedded_secrets(nested)
    elif isinstance(value, list):
        for nested in value:
            _reject_embedded_secrets(nested)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Discord edge config contains duplicate JSON keys")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Discord edge config contains non-JSON constant: {value}")


def _validate_config_path(
    path: Path,
    *,
    expected_owner_uid: int,
    require_root_owned_parents: bool,
) -> os.stat_result:
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("Discord edge config path must be absolute")
    try:
        file_stat = os.lstat(path)
    except OSError as exc:
        raise ValueError("Discord edge config is unavailable") from exc
    if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(file_stat.st_mode):
        raise ValueError("Discord edge config must be a regular non-symlink file")
    if file_stat.st_nlink != 1:
        raise ValueError("Discord edge config must have exactly one link")
    if file_stat.st_uid != expected_owner_uid:
        raise ValueError("Discord edge config owner is not trusted")
    if stat.S_IMODE(file_stat.st_mode) != 0o440:
        raise ValueError("Discord edge config mode must be exactly 0440")
    if require_root_owned_parents:
        current = path.parent
        while True:
            try:
                parent_stat = os.lstat(current)
            except OSError as exc:
                raise ValueError("Discord edge config parent is unavailable") from exc
            if (
                stat.S_ISLNK(parent_stat.st_mode)
                or not stat.S_ISDIR(parent_stat.st_mode)
                or parent_stat.st_uid != expected_owner_uid
                or stat.S_IMODE(parent_stat.st_mode) & 0o022
            ):
                raise ValueError("Discord edge config parent is not root-controlled")
            if current == current.parent:
                break
            current = current.parent
    return file_stat


def _read_config(path: Path, expected: os.stat_result) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError("Discord edge config cannot be opened") from exc
    try:
        actual = os.fstat(descriptor)
        if (
            (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino)
            or actual.st_uid != expected.st_uid
            or actual.st_gid != expected.st_gid
            or actual.st_nlink != 1
            or stat.S_IMODE(actual.st_mode) != stat.S_IMODE(expected.st_mode)
        ):
            raise ValueError("Discord edge config identity changed during open")
        chunks: list[bytes] = []
        total = 0
        while total <= _MAX_CONFIG_BYTES:
            chunk = os.read(
                descriptor,
                min(4096, _MAX_CONFIG_BYTES + 1 - total),
            )
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        path_after = os.lstat(path)
        expected_identity = (
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
            != expected_identity
            for item in (actual, after, path_after)
        ):
            raise ValueError("Discord edge config changed during read")
    finally:
        os.close(descriptor)
    if not raw or len(raw) > _MAX_CONFIG_BYTES:
        raise ValueError("Discord edge config size is invalid")
    return raw


def _validate_key_path(
    path: Path,
    *,
    label: str,
    expected_owner_uid: int,
    expected_gid: int,
    expected_mode: int,
    trusted_parent_owner_uid: int,
    require_trusted_parents: bool,
) -> os.stat_result:
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be an absolute normalized path")
    try:
        file_stat = os.lstat(path)
    except OSError as exc:
        raise ValueError(f"{label} is unavailable") from exc
    if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(file_stat.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file")
    if file_stat.st_nlink != 1:
        raise ValueError(f"{label} must have exactly one link")
    if file_stat.st_uid != expected_owner_uid:
        raise ValueError(f"{label} owner is not trusted")
    if file_stat.st_gid != expected_gid:
        raise ValueError(f"{label} group is not the edge service group")
    if stat.S_IMODE(file_stat.st_mode) != expected_mode:
        raise ValueError(f"{label} mode must be {expected_mode:04o}")
    if require_trusted_parents:
        current = path.parent
        while True:
            parent_stat = os.lstat(current)
            if (
                stat.S_ISLNK(parent_stat.st_mode)
                or not stat.S_ISDIR(parent_stat.st_mode)
                or parent_stat.st_uid != trusted_parent_owner_uid
                or stat.S_IMODE(parent_stat.st_mode) & 0o022
            ):
                raise ValueError(f"{label} parent path is not root-controlled")
            if current == current.parent:
                break
            current = current.parent
    return file_stat


def _read_key(
    path: Path,
    expected: os.stat_result,
    *,
    label: str,
    expected_owner_uid: int,
    expected_gid: int,
    expected_mode: int,
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"{label} cannot be opened") from exc
    try:
        actual = os.fstat(descriptor)
        expected_identity = (
            expected.st_dev,
            expected.st_ino,
            expected.st_size,
            expected.st_mtime_ns,
            expected.st_ctime_ns,
        )
        if (
            (
                actual.st_dev,
                actual.st_ino,
                actual.st_size,
                actual.st_mtime_ns,
                actual.st_ctime_ns,
            )
            != expected_identity
            or not stat.S_ISREG(actual.st_mode)
            or actual.st_nlink != 1
            or actual.st_uid != expected_owner_uid
            or actual.st_gid != expected_gid
            or stat.S_IMODE(actual.st_mode) != expected_mode
        ):
            raise ValueError(f"{label} identity or policy changed during open")
        chunks: list[bytes] = []
        total = 0
        while total <= _MAX_KEY_BYTES:
            chunk = os.read(
                descriptor,
                min(4096, _MAX_KEY_BYTES + 1 - total),
            )
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        path_after = os.lstat(path)
        if any(
            (
                item.st_dev,
                item.st_ino,
                item.st_size,
                item.st_mtime_ns,
                item.st_ctime_ns,
            )
            != expected_identity
            for item in (after, path_after)
        ):
            raise ValueError(f"{label} identity changed during read")
        for item in (after, path_after):
            if (
                not stat.S_ISREG(item.st_mode)
                or item.st_nlink != 1
                or item.st_uid != expected_owner_uid
                or item.st_gid != expected_gid
                or stat.S_IMODE(item.st_mode) != expected_mode
            ):
                raise ValueError(f"{label} policy changed during read")
    finally:
        os.close(descriptor)
    if not raw or len(raw) > _MAX_KEY_BYTES:
        raise ValueError(f"{label} size is invalid")
    return raw


def _load_keys(
    raw: Mapping[str, Any],
    *,
    edge_uid: int,
    edge_gid: int,
    trusted_config_owner_uid: int,
    require_trusted_parents: bool,
) -> tuple[Path, str, Ed25519PublicKey, Path, str, Ed25519PrivateKey]:
    writer_path = _absolute_path(
        raw["writer_capability_public_key_file"],
        "keys.writer_capability_public_key_file",
    )
    edge_path = _absolute_path(
        raw["edge_receipt_private_key_file"],
        "keys.edge_receipt_private_key_file",
    )
    if writer_path == edge_path:
        raise ValueError("writer and edge keys require distinct files")
    writer_key_id = _required_text(
        raw["writer_capability_public_key_id"],
        "keys.writer_capability_public_key_id",
    )
    edge_key_id = _required_text(
        raw["edge_receipt_public_key_id"],
        "keys.edge_receipt_public_key_id",
    )
    if not _SHA256_RE.fullmatch(writer_key_id) or not _SHA256_RE.fullmatch(
        edge_key_id
    ):
        raise ValueError("Discord edge key IDs must be lowercase SHA-256")
    if writer_key_id == edge_key_id:
        raise ValueError("writer and edge signing identities must be distinct")
    writer_stat = _validate_key_path(
        writer_path,
        label="writer capability public key",
        expected_owner_uid=trusted_config_owner_uid,
        expected_gid=edge_gid,
        expected_mode=0o440,
        trusted_parent_owner_uid=trusted_config_owner_uid,
        require_trusted_parents=require_trusted_parents,
    )
    edge_stat = _validate_key_path(
        edge_path,
        label="edge receipt private key",
        expected_owner_uid=edge_uid,
        expected_gid=edge_gid,
        expected_mode=0o400,
        trusted_parent_owner_uid=trusted_config_owner_uid,
        require_trusted_parents=require_trusted_parents,
    )
    writer_bytes = _read_key(
        writer_path,
        writer_stat,
        label="writer capability public key",
        expected_owner_uid=trusted_config_owner_uid,
        expected_gid=edge_gid,
        expected_mode=0o440,
    )
    edge_bytes = _read_key(
        edge_path,
        edge_stat,
        label="edge receipt private key",
        expected_owner_uid=edge_uid,
        expected_gid=edge_gid,
        expected_mode=0o400,
    )
    try:
        writer_key = serialization.load_pem_public_key(writer_bytes)
    except (TypeError, ValueError, UnsupportedAlgorithm) as exc:
        raise ValueError("writer capability public key is not PEM") from exc
    try:
        edge_key = serialization.load_pem_private_key(edge_bytes, password=None)
    except (TypeError, ValueError, UnsupportedAlgorithm) as exc:
        raise ValueError("edge receipt private key is not unencrypted PEM") from exc
    if not isinstance(writer_key, Ed25519PublicKey):
        raise ValueError("writer capability public key must be Ed25519")
    if not isinstance(edge_key, Ed25519PrivateKey):
        raise ValueError("edge receipt private key must be Ed25519")
    if writer_bytes != writer_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ):
        raise ValueError("writer capability key must use exact SPKI PEM encoding")
    if edge_bytes != edge_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ):
        raise ValueError("edge receipt key must use exact PKCS#8 PEM encoding")
    if ed25519_public_key_id(writer_key) != writer_key_id:
        raise ValueError("writer capability public key does not match pinned key ID")
    if ed25519_public_key_id(edge_key.public_key()) != edge_key_id:
        raise ValueError("edge receipt private key does not match pinned key ID")
    return writer_path, writer_key_id, writer_key, edge_path, edge_key_id, edge_key


def load_service_config(
    path: str | os.PathLike[str],
    *,
    _expected_owner_uid: int = 0,
    _require_root_owned_parents: bool = True,
    _expected_socket_path: Path | None = DEFAULT_DISCORD_EDGE_SOCKET_PATH,
    _expected_journal_path: Path | None = DEFAULT_JOURNAL_PATH,
) -> DiscordEdgeServiceConfig:
    """Load one strict local edge configuration and its pinned key material."""

    config_path = Path(path)
    trusted_stat = _validate_config_path(
        config_path,
        expected_owner_uid=_expected_owner_uid,
        require_root_owned_parents=_require_root_owned_parents,
    )
    raw_bytes = _read_config(config_path, trusted_stat)
    try:
        value = json.loads(
            raw_bytes.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("Discord edge config is not strict UTF-8 JSON") from exc
    root = _strict_mapping(value, label="config", allowed=_ROOT_KEYS)
    _reject_embedded_secrets(root)
    service = _strict_mapping(
        root["service"],
        label="service",
        allowed=_SERVICE_KEYS,
    )
    keys = _strict_mapping(root["keys"], label="keys", allowed=_KEY_KEYS)
    discord = _strict_mapping(
        root["discord"],
        label="discord",
        allowed=_DISCORD_KEYS,
    )
    journal = _strict_mapping(
        root["journal"],
        label="journal",
        allowed=_JOURNAL_KEYS,
    )
    runtime = _strict_mapping(
        root["runtime"],
        label="runtime",
        allowed=_RUNTIME_KEYS,
    )

    edge_uid = _integer(
        service["edge_uid"],
        "service.edge_uid",
        minimum=1,
        maximum=(1 << 31) - 1,
    )
    edge_gid = _integer(
        service["edge_gid"],
        "service.edge_gid",
        minimum=1,
        maximum=(1 << 31) - 1,
    )
    gateway_uid = _integer(
        service["gateway_uid"],
        "service.gateway_uid",
        minimum=1,
        maximum=(1 << 31) - 1,
    )
    if gateway_uid == edge_uid:
        raise ValueError("gateway and Discord edge UIDs must be distinct")
    if trusted_stat.st_gid != edge_gid:
        raise ValueError("group-readable Discord edge config must use edge GID")
    gateway_unit = _required_text(
        service["gateway_unit"],
        "service.gateway_unit",
    )
    edge_unit = _required_text(service["edge_unit"], "service.edge_unit")
    if (
        not _SYSTEMD_UNIT_RE.fullmatch(gateway_unit)
        or gateway_unit != DEFAULT_GATEWAY_UNIT
    ):
        raise ValueError("service.gateway_unit must match the pinned gateway unit")
    if (
        not _SYSTEMD_UNIT_RE.fullmatch(edge_unit)
        or edge_unit != DEFAULT_DISCORD_EDGE_UNIT
    ):
        raise ValueError("service.edge_unit must match the pinned edge unit")
    socket_path = _absolute_path(service["socket_path"], "service.socket_path")
    if _expected_socket_path is not None and socket_path != _expected_socket_path:
        raise ValueError("service.socket_path must match the pinned edge socket")

    (
        writer_key_path,
        writer_key_id,
        writer_public_key,
        edge_key_path,
        edge_key_id,
        edge_private_key,
    ) = _load_keys(
        keys,
        edge_uid=edge_uid,
        edge_gid=edge_gid,
        trusted_config_owner_uid=_expected_owner_uid,
        require_trusted_parents=_require_root_owned_parents,
    )

    credentials_directory = _absolute_path(
        discord["credentials_directory"],
        "discord.credentials_directory",
    )
    token_file = _absolute_path(discord["token_file"], "discord.token_file")
    if token_file.parent != credentials_directory:
        raise ValueError(
            "discord.token_file must be directly inside credentials_directory"
        )
    journal_path = _absolute_path(journal["path"], "journal.path")
    if _expected_journal_path is not None and journal_path != _expected_journal_path:
        raise ValueError("journal.path must match the pinned edge journal")
    if len(
        {credentials_directory, journal_path.parent, socket_path.parent}
    ) != 3:
        raise ValueError(
            "credential, journal, and socket directories must be distinct"
        )
    protected_paths = {
        config_path,
        writer_key_path,
        edge_key_path,
        token_file,
        journal_path,
        Path(f"{journal_path}.initialized"),
        socket_path,
    }
    if len(protected_paths) != 7:
        raise ValueError("Discord edge config paths must be pairwise distinct")
    return DiscordEdgeServiceConfig(
        socket_path=socket_path,
        gateway_unit=gateway_unit,
        edge_unit=edge_unit,
        gateway_uid=gateway_uid,
        edge_uid=edge_uid,
        edge_gid=edge_gid,
        connection_timeout_seconds=_number(
            service["connection_timeout_seconds"],
            "service.connection_timeout_seconds",
            minimum=1,
            maximum=300,
        ),
        max_connections=_integer(
            service["max_connections"],
            "service.max_connections",
            minimum=1,
            maximum=64,
        ),
        writer_capability_public_key_file=writer_key_path,
        writer_capability_public_key_id=writer_key_id,
        edge_receipt_private_key_file=edge_key_path,
        edge_receipt_public_key_id=edge_key_id,
        writer_public_key=writer_public_key,
        edge_private_key=edge_private_key,
        token_file=token_file,
        credentials_directory=credentials_directory,
        api_timeout_seconds=_number(
            discord["api_timeout_seconds"],
            "discord.api_timeout_seconds",
            minimum=0.1,
            maximum=15,
        ),
        journal_path=journal_path,
        journal_busy_timeout_ms=_integer(
            journal["busy_timeout_ms"],
            "journal.busy_timeout_ms",
            minimum=1,
            maximum=30_000,
        ),
        max_proof_age_ms=_integer(
            runtime["max_proof_age_ms"],
            "runtime.max_proof_age_ms",
            minimum=1,
            maximum=30_000,
        ),
    )


def _assert_service_identity(config: DiscordEdgeServiceConfig) -> None:
    geteuid = getattr(os, "geteuid", None)
    getegid = getattr(os, "getegid", None)
    if not callable(geteuid) or not callable(getegid):
        raise RuntimeError("privileged Discord edge requires POSIX UID/GID")
    if geteuid() != config.edge_uid or getegid() != config.edge_gid:
        raise PermissionError("Discord edge process UID/GID does not match config")


def bootstrap_journal(
    config: DiscordEdgeServiceConfig,
    *,
    _journal_bootstrap: Callable[..., DurableDiscordEdgeJournal] = (
        DurableDiscordEdgeJournal.bootstrap
    ),
) -> DurableDiscordEdgeJournal:
    """Explicitly create a fresh journal; never adopt or replace any path."""

    _assert_service_identity(config)
    return _journal_bootstrap(
        config.journal_path,
        busy_timeout_ms=config.journal_busy_timeout_ms,
    )


def build_service(
    config: DiscordEdgeServiceConfig,
    *,
    _journal_factory: Callable[..., DurableDiscordEdgeJournal] = (
        DurableDiscordEdgeJournal
    ),
    _adapter_factory: Callable[..., DiscordRestEdgeAdapter] = (
        DiscordRestEdgeAdapter.from_credential_file
    ),
    _runtime_factory: Callable[..., DiscordEdgeRuntime] = DiscordEdgeRuntime,
    _server_factory: Callable[..., DiscordEdgeUnixServer] = DiscordEdgeUnixServer,
    _main_pid_provider_factory: Callable[[], Any] = (
        SystemctlDiscordEdgeMainPidProvider
    ),
) -> DiscordEdgeBootstrap:
    """Open the existing journal and assemble the live edge without serving."""

    _assert_service_identity(config)
    journal = _journal_factory(
        config.journal_path,
        busy_timeout_ms=config.journal_busy_timeout_ms,
    )
    adapter: DiscordRestEdgeAdapter | None = None
    try:
        adapter = _adapter_factory(
            config.token_file,
            credentials_directory=config.credentials_directory,
            expected_owner_uid=config.edge_uid,
            timeout_seconds=config.api_timeout_seconds,
        )
        runtime = _runtime_factory(
            writer_public_key=config.writer_public_key,
            edge_private_key=config.edge_private_key,
            journal=journal,
            target_prover=adapter,
            transport=adapter,
            max_proof_age_ms=config.max_proof_age_ms,
        )
        server = _server_factory(
            config.socket_path,
            runtime=runtime,
            expected_client_uid=config.gateway_uid,
            gateway_unit=config.gateway_unit,
            main_pid_provider=_main_pid_provider_factory(),
            connection_timeout_seconds=config.connection_timeout_seconds,
            max_connections=config.max_connections,
        )
    except BaseException:
        if adapter is not None:
            adapter.close()
        raise
    return DiscordEdgeBootstrap(
        config=config,
        journal=journal,
        adapter=adapter,
        runtime=runtime,
        server=server,
    )


def serve_service(bootstrap: DiscordEdgeBootstrap) -> None:
    """Serve until shutdown and always close socket and HTTPS resources."""

    if not isinstance(bootstrap, DiscordEdgeBootstrap):
        raise TypeError("bootstrap must be DiscordEdgeBootstrap")
    previous_handlers: dict[int, Any] = {}

    def _shutdown(_signum: int, _frame: FrameType | None) -> None:
        bootstrap.server.shutdown()

    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, _shutdown)
        bootstrap.server.serve_forever()
    finally:
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)
        bootstrap.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="absolute root-owned Discord edge JSON config",
    )
    parser.add_argument(
        "--bootstrap-journal",
        action="store_true",
        help="create a fresh durable journal and exit; refuses existing paths",
    )
    arguments = parser.parse_args(argv)
    config = load_service_config(arguments.config)
    if arguments.bootstrap_journal:
        bootstrap_journal(config)
        return 0
    bootstrap = build_service(config)
    try:
        serve_service(bootstrap)
    except KeyboardInterrupt:
        bootstrap.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_JOURNAL_PATH",
    "DiscordEdgeBootstrap",
    "DiscordEdgeServiceConfig",
    "bootstrap_journal",
    "build_service",
    "load_service_config",
    "main",
    "serve_service",
]

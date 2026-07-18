#!/usr/bin/env python3
"""Fixed owner-side recoverability gate for the production PostgreSQL instance.

The public boundary has no target, network, path, host, or command arguments.
It can only back up ``adventico-ai-platform/ai-platform-postgres``, restore that
backup into the release-bound scratch instance on ``muncho-canary-vpc``, run
the fixed read-only probe, and delete that scratch instance.  The backup is
retained.  Every mutation is preceded by a durable journal intent and all
provider operations are reconciled by deterministic provider identity so a
crash can resume without creating a second backup or scratch instance.

The private probe reuses the release-bound IAP/OS Login transport and reads one
fixed Secret Manager version only after a fresh stopped-release, VM, network,
scratch-address, and server-CA gate.  It sends one bounded secret frame to the
fixed remote module, which exposes no caller-selected target, path, command, or
SQL.  Availability is proven before the first Cloud mutation.
"""

from __future__ import annotations

import copy
import hashlib
import ipaddress
import json
import os
import re
import ssl
import stat
import struct
import time
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from gateway import canonical_writer_production_cutover as cutover
from scripts.canary import full_canary_owner_launcher as owner_transport
from scripts.canary import production_database_recovery_probe as remote_probe


JOURNAL_SCHEMA = "muncho-production-database-recovery-journal.v1"
JOURNAL_ENTRY_SCHEMA = "muncho-production-database-recovery-journal-entry.v1"
MAX_JSON = 4 * 1024 * 1024
BACKUP_DESCRIPTION_PREFIX = "muncho-production-recovery"
MAX_BACKUP_AGE_SECONDS = 6 * 60 * 60
MAX_BACKUP_RECHECK_AGE_SECONDS = 120
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_OPERATION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,254}$")
_BACKUP_ID = re.compile(r"^[1-9][0-9]{0,19}$")
_RFC1918 = tuple(
    ipaddress.ip_network(value)
    for value in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")
)
_SOURCE_PRIVATE_NETWORK = re.compile(
    rf"^projects/{re.escape(cutover.PROJECT)}/global/networks/"
    r"[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?$"
)
_PROBE_RECEIPT_FIELDS = frozenset({
    "schema",
    "ok",
    "release_revision",
    "scratch_instance",
    "database",
    "probe_contract_sha256",
    "transaction_read_only",
    "schema_sha256",
    "content_sha256",
    "canonical_event_row_count",
    "scratch_private_ip",
    "server_ca_sha256",
    "tls_mode",
    "tls_ca_verified",
    "tls_hostname_verified",
    "canary_instance_id",
    "canary_network",
    "canary_subnetwork",
    "canary_private_ip",
    "release_manifest_file_sha256",
    "stopped_units_sha256",
    "psql_executable_sha256",
    "probed_at_unix",
    "secret_material_recorded",
    "secret_digest_recorded",
    "receipt_sha256",
})
_STAGES = (
    "manifest",
    "source_readback",
    "backup_intent",
    "backup_ready",
    "scratch_intent",
    "scratch_ready",
    "restore_intent",
    "restore_ready",
    "probe_receipt",
    "scratch_delete_intent",
    "scratch_deleted",
    "backup_rechecked",
    "terminal_receipt",
)


def _stage_for_sequence(sequence: int) -> str:
    if 0 <= sequence < len(_STAGES):
        return _STAGES[sequence]
    offset = sequence - len(_STAGES)
    if offset < 0:
        raise ProductionDatabaseRecoveryError(
            "production_database_recovery_journal_order_invalid"
        )
    generation = offset // 2 + 1
    prefix = "backup_rechecked_refresh" if offset % 2 == 0 else "terminal_receipt_refresh"
    return f"{prefix}_{generation}"


class ProductionDatabaseRecoveryError(RuntimeError):
    """Stable, secret-free recovery-gate failure."""


def _canonical(value: Any) -> bytes:
    try:
        raw = json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii", errors="strict")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ProductionDatabaseRecoveryError(
            "production_database_recovery_json_invalid"
        ) from exc
    if len(raw) > MAX_JSON:
        raise ProductionDatabaseRecoveryError(
            "production_database_recovery_json_oversized"
        )
    return raw


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_json(value: Any) -> str:
    return _sha(_canonical(value))


def _release(revision: str) -> str:
    if not isinstance(revision, str) or _REVISION.fullmatch(revision) is None:
        raise ProductionDatabaseRecoveryError(
            "production_database_recovery_revision_invalid"
        )
    return revision


def _backup_description(revision: str) -> str:
    return f"{BACKUP_DESCRIPTION_PREFIX}-{_release(revision)[:20]}"


def _unix_time(value: Any, code: str) -> int:
    if isinstance(value, int) and value > 0:
        return value
    if not isinstance(value, str) or not value:
        raise ProductionDatabaseRecoveryError(code)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("timestamp is not timezone-aware")
        timestamp = int(parsed.timestamp())
    except (ValueError, OverflowError):
        raise ProductionDatabaseRecoveryError(code) from None
    if timestamp <= 0:
        raise ProductionDatabaseRecoveryError(code)
    return timestamp


def _operation_id(value: Any) -> str:
    if not isinstance(value, str) or _OPERATION_ID.fullmatch(value) is None:
        raise ProductionDatabaseRecoveryError(
            "production_database_recovery_operation_invalid"
        )
    return value


def _backup_id(value: Any) -> str:
    normalized = str(value)
    if _BACKUP_ID.fullmatch(normalized) is None:
        raise ProductionDatabaseRecoveryError(
            "production_database_recovery_backup_invalid"
        )
    return normalized


def _journal_root(revision: str) -> Path:
    return (
        Path.home()
        / ".hermes"
        / "owner-gates"
        / "production-database-recovery"
        / _release(revision)
    )


class _Journal:
    def __init__(self, root: Path, revision: str, *, now_unix: int) -> None:
        self.root = root
        self.revision = _release(revision)
        self._prepare_root()
        self._entries = self._load()
        manifest = {
            "schema": JOURNAL_SCHEMA,
            "release_revision": self.revision,
            "source_project": cutover.PROJECT,
            "source_instance": cutover.PRODUCTION_SQL_INSTANCE,
            "source_region": cutover.PRODUCTION_SQL_REGION,
            "scratch_instance": cutover.database_recovery_scratch_instance(
                self.revision
            ),
            "scratch_network": cutover.DATABASE_RECOVERY_SCRATCH_NETWORK,
            "backup_description": _backup_description(self.revision),
            "started_at_unix": now_unix,
            "secret_material_recorded": False,
            "secret_digest_recorded": False,
        }
        if not self._entries:
            self.record("manifest", manifest, now_unix=now_unix)
        elif self.payload("manifest") != manifest:
            existing = self.payload("manifest")
            comparable = dict(manifest)
            comparable["started_at_unix"] = existing.get("started_at_unix")
            if existing != comparable:
                raise ProductionDatabaseRecoveryError(
                    "production_database_recovery_journal_conflict"
                )

    def _prepare_root(self) -> None:
        try:
            if not self.root.is_absolute():
                raise ProductionDatabaseRecoveryError(
                    "production_database_recovery_journal_identity_invalid"
                )
            self.root.mkdir(parents=True, mode=0o700, exist_ok=True)
            metadata = self.root.lstat()
            if (
                self.root.resolve(strict=True) != self.root
                or stat.S_ISLNK(metadata.st_mode)
                or not stat.S_ISDIR(metadata.st_mode)
                or metadata.st_uid != os.getuid()
                or stat.S_IMODE(metadata.st_mode) & 0o077
            ):
                raise ProductionDatabaseRecoveryError(
                    "production_database_recovery_journal_identity_invalid"
                )
        except ProductionDatabaseRecoveryError:
            raise
        except OSError as exc:
            raise ProductionDatabaseRecoveryError(
                "production_database_recovery_journal_unavailable"
            ) from exc

    @staticmethod
    def _read_entry(path: Path) -> bytes:
        descriptor: int | None = None
        try:
            before = path.lstat()
            if (
                stat.S_ISLNK(before.st_mode)
                or not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or before.st_uid != os.getuid()
                or stat.S_IMODE(before.st_mode) != 0o600
                or not 0 < before.st_size <= MAX_JSON
            ):
                raise ProductionDatabaseRecoveryError(
                    "production_database_recovery_journal_identity_invalid"
                )
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(path, flags)
            opened = os.fstat(descriptor)
            chunks: list[bytes] = []
            remaining = MAX_JSON + 1
            while remaining:
                chunk = os.read(descriptor, min(remaining, 64 * 1024))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            after = os.fstat(descriptor)
            reachable = path.lstat()
        except ProductionDatabaseRecoveryError:
            raise
        except OSError as exc:
            raise ProductionDatabaseRecoveryError(
                "production_database_recovery_journal_unavailable"
            ) from exc
        finally:
            if descriptor is not None:
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
            identity(before) != identity(opened)
            or identity(before) != identity(after)
            or identity(before) != identity(reachable)
            or len(raw) != before.st_size
        ):
            raise ProductionDatabaseRecoveryError(
                "production_database_recovery_journal_changed"
            )
        return raw

    def _load(self) -> list[Mapping[str, Any]]:
        entries: list[Mapping[str, Any]] = []
        previous = "0" * 64
        try:
            paths = sorted(
                self.root.glob("[0-9][0-9][0-9][0-9]-*.json")
            )
        except OSError as exc:
            raise ProductionDatabaseRecoveryError(
                "production_database_recovery_journal_unavailable"
            ) from exc
        for sequence, path in enumerate(paths):
            try:
                raw = self._read_entry(path)
                value = json.loads(raw.decode("ascii", errors="strict"))
            except ProductionDatabaseRecoveryError:
                raise
            except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
                raise ProductionDatabaseRecoveryError(
                    "production_database_recovery_journal_invalid"
                ) from exc
            if not isinstance(value, Mapping) or raw != _canonical(value):
                raise ProductionDatabaseRecoveryError(
                    "production_database_recovery_journal_invalid"
                )
            unsigned = {
                name: item for name, item in value.items() if name != "entry_sha256"
            }
            if (
                set(value)
                != {
                    "schema",
                    "sequence",
                    "stage",
                    "previous_entry_sha256",
                    "payload",
                    "recorded_at_unix",
                    "entry_sha256",
                }
                or value["schema"] != JOURNAL_ENTRY_SCHEMA
                or value["sequence"] != sequence
                or value["stage"] != _stage_for_sequence(sequence)
                or value["previous_entry_sha256"] != previous
                or type(value["recorded_at_unix"]) is not int
                or value["recorded_at_unix"] <= 0
                or not isinstance(value["payload"], Mapping)
                or value["entry_sha256"] != _sha_json(unsigned)
            ):
                raise ProductionDatabaseRecoveryError(
                    "production_database_recovery_journal_invalid"
                )
            entries.append(copy.deepcopy(dict(value)))
            previous = str(value["entry_sha256"])
        return entries

    def has(self, stage: str) -> bool:
        return any(entry["stage"] == stage for entry in self._entries)

    def payload(self, stage: str) -> Mapping[str, Any]:
        for entry in self._entries:
            if entry["stage"] == stage:
                return copy.deepcopy(dict(entry["payload"]))
        raise ProductionDatabaseRecoveryError(
            "production_database_recovery_journal_stage_missing"
        )

    def latest_payload(self, prefix: str) -> Mapping[str, Any]:
        for entry in reversed(self._entries):
            if str(entry["stage"]).startswith(prefix):
                return copy.deepcopy(dict(entry["payload"]))
        raise ProductionDatabaseRecoveryError(
            "production_database_recovery_journal_stage_missing"
        )

    def refresh_generation(self) -> int:
        return sum(
            str(entry["stage"]).startswith("terminal_receipt_refresh_")
            for entry in self._entries
        ) + 1

    @property
    def prefix_sha256(self) -> str:
        if not self._entries:
            return "0" * 64
        return str(self._entries[-1]["entry_sha256"])

    def record(
        self,
        stage: str,
        payload: Mapping[str, Any],
        *,
        now_unix: int,
    ) -> Mapping[str, Any]:
        if self.has(stage):
            existing = self.payload(stage)
            if _canonical(existing) != _canonical(payload):
                raise ProductionDatabaseRecoveryError(
                    "production_database_recovery_journal_conflict"
                )
            return existing
        sequence = len(self._entries)
        if stage != _stage_for_sequence(sequence):
            raise ProductionDatabaseRecoveryError(
                "production_database_recovery_journal_order_invalid"
            )
        unsigned = {
            "schema": JOURNAL_ENTRY_SCHEMA,
            "sequence": sequence,
            "stage": stage,
            "previous_entry_sha256": self.prefix_sha256,
            "payload": copy.deepcopy(dict(payload)),
            "recorded_at_unix": now_unix,
        }
        entry = {**unsigned, "entry_sha256": _sha_json(unsigned)}
        if sequence > 9_999:
            raise ProductionDatabaseRecoveryError(
                "production_database_recovery_journal_exhausted"
            )
        path = self.root / f"{sequence:04d}-{stage}.json"
        temporary = self.root / f".{path.name}.{os.getpid()}.tmp"
        descriptor: int | None = None
        try:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(temporary, flags, 0o600)
            payload_bytes = _canonical(entry)
            view = memoryview(payload_bytes)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short recovery journal write")
                view = view[written:]
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = None
            os.link(temporary, path, follow_symlinks=False)
            temporary.unlink()
            directory = os.open(self.root, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except FileExistsError:
            raise ProductionDatabaseRecoveryError(
                "production_database_recovery_journal_conflict"
            ) from None
        except OSError as exc:
            raise ProductionDatabaseRecoveryError(
                "production_database_recovery_journal_write_failed"
            ) from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        self._entries.append(entry)
        return copy.deepcopy(dict(payload))


class RecoveryProvider(Protocol):
    def source_readback(self) -> Mapping[str, Any]: ...

    def ensure_backup(
        self, *, release_revision: str, not_before_unix: int
    ) -> Mapping[str, Any]: ...

    def ensure_scratch(
        self, *, release_revision: str, source: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...

    def ensure_restore(
        self,
        *,
        release_revision: str,
        backup: Mapping[str, Any],
        scratch: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

    def delete_scratch(
        self, *, release_revision: str
    ) -> Mapping[str, Any]: ...

    def backup_readback(self, *, backup_id: str) -> Mapping[str, Any]: ...

    def scratch_readback(self, *, release_revision: str) -> Mapping[str, Any]: ...


class RecoveryProbe(Protocol):
    def require_available(self) -> None: ...

    def probe(
        self,
        *,
        release_revision: str,
        scratch: Mapping[str, Any],
        now_unix: int,
    ) -> Mapping[str, Any]: ...


class SecretAccessor(Protocol):
    def access(self) -> tuple[str, bytearray]: ...


class PrivateProbeTransport(Protocol):
    def require_available(self, release_revision: str) -> Mapping[str, Any]: ...

    def open_probe(self, release_revision: str) -> Any: ...


def _zeroize(value: bytearray | memoryview | None) -> None:
    if value is None:
        return
    try:
        view = value if isinstance(value, memoryview) else memoryview(value)
        view.cast("B")[:] = b"\x00" * view.nbytes
        if not isinstance(value, memoryview):
            view.release()
    except (TypeError, ValueError, BufferError):
        pass


def _crc32c(value: bytearray) -> int:
    crc = 0xFFFFFFFF
    for octet in value:
        crc ^= octet
        for _ in range(8):
            crc = (crc >> 1) ^ (0x82F63B78 if crc & 1 else 0)
    return (~crc) & 0xFFFFFFFF


_BASE64_VALUES = {
    character: index
    for index, character in enumerate(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    )
}


def _decode_secret_base64(value: str) -> bytearray:
    """Strictly decode into the only wipeable plaintext representation."""

    maximum_encoded = ((remote_probe.MAX_PASSWORD_BYTES + 2) // 3) * 4
    if (
        type(value) is not str
        or not 4 <= len(value) <= maximum_encoded
        or len(value) % 4 != 0
    ):
        raise ProductionDatabaseRecoveryError(
            "production_database_recovery_secret_shape_invalid"
        )
    result = bytearray()
    try:
        for offset in range(0, len(value), 4):
            quartet = value[offset : offset + 4]
            final = offset + 4 == len(value)
            if quartet[0] == "=" or quartet[1] == "=":
                raise ValueError("invalid base64 padding")
            first = _BASE64_VALUES.get(quartet[0])
            second = _BASE64_VALUES.get(quartet[1])
            third = None if quartet[2] == "=" else _BASE64_VALUES.get(quartet[2])
            fourth = None if quartet[3] == "=" else _BASE64_VALUES.get(quartet[3])
            if (
                first is None
                or second is None
                or (quartet[2] != "=" and third is None)
                or (quartet[3] != "=" and fourth is None)
                or (quartet[2] == "=" and quartet[3] != "=")
                or ((quartet[2] == "=" or quartet[3] == "=") and not final)
                or (quartet[2] == "=" and second & 0x0F != 0)
                or (quartet[3] == "=" and third is not None and third & 0x03 != 0)
            ):
                raise ValueError("invalid base64 encoding")
            result.append((first << 2) | (second >> 4))
            if third is not None:
                result.append(((second & 0x0F) << 4) | (third >> 2))
            if fourth is not None and third is not None:
                result.append(((third & 0x03) << 6) | fourth)
            if len(result) > remote_probe.MAX_PASSWORD_BYTES:
                raise ValueError("decoded secret is oversized")
        return result
    except (KeyError, TypeError, ValueError):
        _zeroize(result)
        raise ProductionDatabaseRecoveryError(
            "production_database_recovery_secret_shape_invalid"
        ) from None


class FixedSecretManagerAccess:
    """Read only the fixed built-in postgres password through owner REST.

    CPython's JSON response and base64 text are immutable transient objects;
    their references are dropped immediately after decoding.  The plaintext
    itself is decoded directly into a bytearray and is explicitly wiped by
    this boundary on failure and by its caller after the one framed write.
    """

    _RESOURCE = (
        "projects/adventico-ai-platform/secrets/ai-platform-db-password"
    )
    _URL = (
        "https://secretmanager.googleapis.com/v1/"
        f"{_RESOURCE}/versions/latest:access"
    )
    _VERSION_NAME = re.compile(
        r"^projects/(?:adventico-ai-platform|39589465056)/secrets/"
        r"ai-platform-db-password/versions/([1-9][0-9]{0,18})$"
    )

    def __init__(
        self,
        token_provider: Callable[[], str],
        *,
        requester: Any = owner_transport._default_http_request,
        timeout_seconds: float = 20.0,
    ) -> None:
        self._token_provider = token_provider
        self._requester = requester
        self._timeout_seconds = timeout_seconds

    def access(self) -> tuple[str, bytearray]:
        password: bytearray | None = None
        token: str | None = None
        headers: dict[str, str] = {}
        response: Any = None
        value: Mapping[str, Any] | None = None
        payload: Mapping[str, Any] | None = None
        encoded_data: str | None = None
        checksum_text: str | None = None
        try:
            owner_transport._reject_custom_ca_environment()
            token = self._token_provider()
            if (
                not isinstance(token, str)
                or not token
                or len(token) > 16 * 1024
                or any(not 0x21 <= ord(character) <= 0x7E for character in token)
            ):
                raise ProductionDatabaseRecoveryError(
                    "production_database_recovery_secret_access_failed"
                )
            headers.update(
                {
                    "Accept": "application/json",
                    "Authorization": f"Bearer {token}",
                }
            )
            try:
                response = self._requester(
                    "GET",
                    self._URL,
                    headers,
                    None,
                    self._timeout_seconds,
                )
            finally:
                headers.clear()
                token = None
            if response.status != 200 or not isinstance(response.body, bytes):
                raise ProductionDatabaseRecoveryError(
                    "production_database_recovery_secret_access_failed"
                )
            value = owner_transport._decode_json_object(
                response.body,
                maximum=128 * 1024,
            )
            payload = value.get("payload")
            name = value.get("name")
            if (
                set(value) != {"name", "payload"}
                or not isinstance(name, str)
                or not isinstance(payload, Mapping)
                or set(payload) != {"data", "dataCrc32c"}
                or not isinstance(payload.get("data"), str)
                or not isinstance(payload.get("dataCrc32c"), str)
            ):
                raise ProductionDatabaseRecoveryError(
                    "production_database_recovery_secret_shape_invalid"
                )
            matched = self._VERSION_NAME.fullmatch(name)
            if matched is None:
                raise ProductionDatabaseRecoveryError(
                    "production_database_recovery_secret_shape_invalid"
                )
            encoded_data = payload["data"]
            checksum_text = payload["dataCrc32c"]
            try:
                password = _decode_secret_base64(encoded_data)
                checksum = int(checksum_text, 10)
            except (UnicodeError, ValueError, TypeError):
                raise ProductionDatabaseRecoveryError(
                    "production_database_recovery_secret_shape_invalid"
                ) from None
            version = matched.group(1)
            response = None
            value = None
            payload = None
            encoded_data = None
            checksum_text = None
            if (
                not 1 <= len(password) <= remote_probe.MAX_PASSWORD_BYTES
                or not 0 <= checksum <= 0xFFFFFFFF
                or _crc32c(password) != checksum
                or any(value in {0, 10, 13} for value in password)
            ):
                raise ProductionDatabaseRecoveryError(
                    "production_database_recovery_secret_shape_invalid"
                )
            result = password
            password = None
            return version, result
        except ProductionDatabaseRecoveryError:
            raise
        except Exception:
            raise ProductionDatabaseRecoveryError(
                "production_database_recovery_secret_access_failed"
            ) from None
        finally:
            headers.clear()
            token = None
            response = None
            value = None
            payload = None
            encoded_data = None
            checksum_text = None
            _zeroize(password)


_REMOTE_PREFLIGHT_FIELDS = frozenset({
    "schema", "ok", "release_revision", "canary_instance_id",
    "canary_host_identity_sha256", "canary_network", "canary_subnetwork",
    "canary_private_ip", "network_identity_sha256",
    "release_manifest_file_sha256", "stopped_units_sha256", "psql_executable",
    "psql_executable_sha256", "database", "probe_contract_sha256",
    "accepts_caller_target", "accepts_caller_sql", "accepts_caller_command",
    "preflight_sha256",
})
_REMOTE_GATE_FIELDS = frozenset({
    "schema", "ok", "release_revision", "preflight_sha256", "challenge",
    "issued_at_unix", "expires_at_unix", "gate_sha256",
})


def _validate_remote_preflight(
    value: Mapping[str, Any], *, release_revision: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProductionDatabaseRecoveryError(
            "production_database_recovery_probe_preflight_invalid"
        )
    unsigned = {name: item for name, item in value.items() if name != "preflight_sha256"}
    if (
        set(value) != _REMOTE_PREFLIGHT_FIELDS
        or value.get("schema") != remote_probe.PREFLIGHT_SCHEMA
        or value.get("ok") is not True
        or value.get("release_revision") != release_revision
        or value.get("canary_instance_id") != owner_transport.VM_INSTANCE_ID
        or value.get("canary_network") != remote_probe.EXPECTED_NETWORK
        or value.get("canary_subnetwork") != remote_probe.EXPECTED_SUBNETWORK
        or value.get("canary_private_ip") != remote_probe.EXPECTED_PRIVATE_IP
        or value.get("psql_executable") != str(remote_probe.PSQL)
        or value.get("psql_executable_sha256")
        != remote_probe.EXPECTED_PSQL_SHA256
        or value.get("database") != cutover.DATABASE
        or value.get("probe_contract_sha256")
        != _sha_json(cutover.DATABASE_RECOVERY_PROBE_CONTRACT)
        or value.get("accepts_caller_target") is not False
        or value.get("accepts_caller_sql") is not False
        or value.get("accepts_caller_command") is not False
        or any(
            _SHA256.fullmatch(str(value.get(name))) is None
            for name in (
                "canary_host_identity_sha256", "network_identity_sha256",
                "release_manifest_file_sha256", "stopped_units_sha256",
                "psql_executable_sha256",
            )
        )
        or value.get("preflight_sha256") != _sha_json(unsigned)
    ):
        raise ProductionDatabaseRecoveryError(
            "production_database_recovery_probe_preflight_invalid"
        )
    return copy.deepcopy(dict(value))


def _validate_remote_gate(
    value: Mapping[str, Any],
    *,
    release_revision: str,
    preflight: Mapping[str, Any],
    now_unix: int,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProductionDatabaseRecoveryError(
            "production_database_recovery_probe_gate_invalid"
        )
    unsigned = {name: item for name, item in value.items() if name != "gate_sha256"}
    if (
        set(value) != _REMOTE_GATE_FIELDS
        or value.get("schema") != remote_probe.GATE_SCHEMA
        or value.get("ok") is not True
        or value.get("release_revision") != release_revision
        or value.get("preflight_sha256") != preflight["preflight_sha256"]
        or not isinstance(value.get("challenge"), str)
        or re.fullmatch(r"[0-9a-f]{64}", value["challenge"]) is None
        or type(value.get("issued_at_unix")) is not int
        or type(value.get("expires_at_unix")) is not int
        or not value["issued_at_unix"] <= now_unix < value["expires_at_unix"]
        or value["expires_at_unix"] - value["issued_at_unix"]
        != remote_probe.GATE_LIFETIME_SECONDS
        or value.get("gate_sha256") != _sha_json(unsigned)
    ):
        raise ProductionDatabaseRecoveryError(
            "production_database_recovery_probe_gate_invalid"
        )
    return copy.deepcopy(dict(value))


def _remote_failure(value: Mapping[str, Any], *, revision: str) -> bool:
    if not isinstance(value, Mapping) or value.get("schema") != remote_probe.FAILURE_SCHEMA:
        return False
    unsigned = {name: item for name, item in value.items() if name != "receipt_sha256"}
    if (
        set(value)
        != {
            "schema", "ok", "release_revision", "error_code",
            "secret_material_recorded", "secret_digest_recorded", "receipt_sha256",
        }
        or value.get("ok") is not False
        or value.get("release_revision") not in {None, revision}
        or value.get("error_code") != "production_database_recovery_probe_failed"
        or value.get("secret_material_recorded") is not False
        or value.get("secret_digest_recorded") is not False
        or value.get("receipt_sha256") != _sha_json(unsigned)
    ):
        raise ProductionDatabaseRecoveryError(
            "production_database_recovery_probe_failure_invalid"
        )
    return True


class FixedDatabaseRecoveryIapTransport(owner_transport.IapCoordinatorTransport):
    """Existing pinned IAP transport specialized to the fixed probe module."""

    _MODULE = "scripts.canary.production_database_recovery_probe"
    _COMMANDS = frozenset({"preflight", "probe"})

    def require_available(self, release_revision: str) -> Mapping[str, Any]:
        session = self._open(release_revision, "preflight", approved=False)
        primary: BaseException | None = None
        try:
            value = session.read_gate()
            if _remote_failure(value, revision=release_revision):
                session.mark_validated(value)
                raise ProductionDatabaseRecoveryError(
                    "production_database_recovery_probe_preflight_failed"
                )
            validated = _validate_remote_preflight(
                value, release_revision=release_revision
            )
            session.complete_read_only()
            return validated
        except BaseException as exc:
            primary = exc
            if not session.termination_proven:
                try:
                    session.abort_and_prove_terminated()
                except BaseException:
                    pass
            raise
        finally:
            try:
                session.close()
            except BaseException:
                if primary is None:
                    raise

    def open_probe(self, release_revision: str) -> Any:
        return self._open(
            release_revision,
            "probe",
            approved=True,
            post_frame_timeout_seconds=300.0,
            maximum_line_bytes=owner_transport.PHASE_B_MAX_RESPONSE_BYTES,
        )


class FixedPrivateReadOnlyProbe:
    """Owner edge for one exact secret frame and one exact remote SQL probe."""

    def __init__(
        self,
        release_revision: str,
        *,
        transport: PrivateProbeTransport,
        secret_accessor: SecretAccessor,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._revision = _release(release_revision)
        self._transport = transport
        self._secret_accessor = secret_accessor
        self._clock = clock
        self._preflight: Mapping[str, Any] | None = None

    def require_available(self) -> None:
        observed = self._transport.require_available(self._revision)
        self._preflight = _validate_remote_preflight(
            observed, release_revision=self._revision
        )

    @staticmethod
    def _frame(
        *,
        revision: str,
        gate: Mapping[str, Any],
        scratch: Mapping[str, Any],
        secret_version: str,
        password: bytearray,
    ) -> bytearray:
        metadata = {
            "schema": remote_probe.FRAME_SCHEMA,
            "release_revision": revision,
            "gate_sha256": gate["gate_sha256"],
            "scratch_instance": scratch["instance"],
            "scratch_private_ip": scratch["private_ip"],
            "server_ca_pem": scratch["server_ca_pem"],
            "server_ca_sha256": scratch["server_ca_sha256"],
            "tls_mode": remote_probe.TLS_MODE,
            "secret_resource": FixedSecretManagerAccess._RESOURCE,
            "secret_version": secret_version,
        }
        metadata_raw = _canonical(metadata)
        if (
            len(metadata_raw) > remote_probe.MAX_METADATA_BYTES
            or not 1 <= len(password) <= remote_probe.MAX_PASSWORD_BYTES
        ):
            raise ProductionDatabaseRecoveryError(
                "production_database_recovery_probe_frame_invalid"
            )
        frame = bytearray(12 + len(metadata_raw) + len(password))
        struct.pack_into(
            ">4sII", frame, 0, remote_probe.FRAME_MAGIC, len(metadata_raw), len(password)
        )
        frame[12 : 12 + len(metadata_raw)] = metadata_raw
        frame[12 + len(metadata_raw) :] = password
        return frame

    def probe(
        self,
        *,
        release_revision: str,
        scratch: Mapping[str, Any],
        now_unix: int,
    ) -> Mapping[str, Any]:
        if release_revision != self._revision or type(now_unix) is not int:
            raise ProductionDatabaseRecoveryError(
                "production_database_recovery_probe_invalid"
            )
        if self._preflight is None:
            self.require_available()
        preflight = copy.deepcopy(dict(self._preflight or {}))
        private_ip = scratch.get("private_ip")
        ca_pem = scratch.get("server_ca_pem")
        try:
            address = ipaddress.ip_address(str(private_ip))
            ca_raw = str(ca_pem).encode("ascii", errors="strict")
            ssl.PEM_cert_to_DER_cert(str(ca_pem))
        except (ValueError, UnicodeError, ssl.SSLError) as exc:
            raise ProductionDatabaseRecoveryError(
                "production_database_recovery_probe_scratch_invalid"
            ) from exc
        if (
            scratch.get("project") != cutover.PROJECT
            or scratch.get("instance")
            != cutover.database_recovery_scratch_instance(self._revision)
            or scratch.get("region") != cutover.PRODUCTION_SQL_REGION
            or scratch.get("private_network")
            != cutover.DATABASE_RECOVERY_SCRATCH_NETWORK
            or not isinstance(scratch.get("database_version"), str)
            or not scratch["database_version"].startswith("POSTGRES_")
            or type(private_ip) is not str
            or address.version != 4
            or not any(address in network for network in _RFC1918)
            or type(ca_pem) is not str
            or not 1 <= len(ca_raw) <= remote_probe.MAX_CA_BYTES
            or scratch.get("server_ca_sha256") != _sha(ca_raw)
            or scratch.get("ssl_mode") != "ENCRYPTED_ONLY"
            or scratch.get("server_ca_mode") != "GOOGLE_MANAGED_INTERNAL_CA"
            or scratch.get("connection_name")
            != (
                f"{cutover.PROJECT}:{cutover.PRODUCTION_SQL_REGION}:"
                f"{cutover.database_recovery_scratch_instance(self._revision)}"
            )
        ):
            raise ProductionDatabaseRecoveryError(
                "production_database_recovery_probe_scratch_invalid"
            )
        session = self._transport.open_probe(self._revision)
        password: bytearray | None = None
        frame: bytearray | None = None
        primary: BaseException | None = None
        try:
            gate_raw = session.read_gate()
            if _remote_failure(gate_raw, revision=self._revision):
                session.mark_validated(gate_raw)
                raise ProductionDatabaseRecoveryError(
                    "production_database_recovery_probe_remote_failed"
                )
            gate = _validate_remote_gate(
                gate_raw,
                release_revision=self._revision,
                preflight=preflight,
                now_unix=int(self._clock()),
            )
            # This is deliberately the last operation before the one bounded
            # secret frame crosses the already-validated IAP session.
            secret_version, password = self._secret_accessor.access()
            frame = self._frame(
                revision=self._revision,
                gate=gate,
                scratch=scratch,
                secret_version=secret_version,
                password=password,
            )
            value = session.finish(frame)
            _zeroize(frame)
            frame = None
            _zeroize(password)
            password = None
            if _remote_failure(value, revision=self._revision):
                session.mark_validated(value)
                raise ProductionDatabaseRecoveryError(
                    "production_database_recovery_probe_remote_failed"
                )
            validated = _validated_probe_receipt_for_delete(
                value,
                release_revision=self._revision,
                scratch_instance=str(scratch["instance"]),
                backup_completed_at_unix=1,
                now_unix=int(self._clock()),
                expected_scratch=scratch,
                expected_preflight=preflight,
            )
            session.mark_validated(validated)
            return validated
        except BaseException as exc:
            primary = exc
            if not getattr(session, "termination_proven", False):
                try:
                    session.abort_and_prove_terminated()
                except BaseException:
                    pass
            raise
        finally:
            _zeroize(frame)
            _zeroize(password)
            try:
                session.close()
            except BaseException:
                if primary is None:
                    raise


class CloudSqlRecoveryProvider:
    """Cloud SQL Admin REST boundary pinned to the recovery contract."""

    _BASE = f"https://sqladmin.googleapis.com/sql/v1beta4/projects/{cutover.PROJECT}"

    def __init__(
        self,
        client: owner_transport.GoogleRestClient,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
        operation_timeout_seconds: float = 45 * 60,
    ) -> None:
        self._client = client
        self._monotonic = monotonic
        self._sleeper = sleeper
        self._operation_timeout_seconds = operation_timeout_seconds

    def _url(self, suffix: str, **query: Any) -> str:
        if not suffix.startswith("/") or ".." in suffix:
            raise ProductionDatabaseRecoveryError(
                "production_database_recovery_provider_contract_invalid"
            )
        url = f"{self._BASE}{suffix}"
        if query:
            url += "?" + urllib.parse.urlencode(query)
        return url

    def _request(
        self,
        method: str,
        suffix: str,
        *,
        body: Mapping[str, Any] | None = None,
        query: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        try:
            return self._client.request_json(
                method,
                self._url(suffix, **dict(query or {})),
                body=body,
            )
        except owner_transport.OwnerLauncherError as exc:
            raise ProductionDatabaseRecoveryError(
                "production_database_recovery_provider_failed"
            ) from exc

    def _pages(self, suffix: str, *, item_kind: str) -> list[Mapping[str, Any]]:
        items: list[Mapping[str, Any]] = []
        token: str | None = None
        visited: set[str] = set()
        for _ in range(100):
            query: dict[str, Any] = {"maxResults": 500}
            if token is not None:
                query["pageToken"] = token
            page = self._request("GET", suffix, query=query)
            raw_items = page.get("items", [])
            if not isinstance(raw_items, list):
                raise ProductionDatabaseRecoveryError(
                    "production_database_recovery_provider_evidence_invalid"
                )
            for item in raw_items:
                if not isinstance(item, Mapping) or item.get("kind") != item_kind:
                    raise ProductionDatabaseRecoveryError(
                        "production_database_recovery_provider_evidence_invalid"
                    )
                items.append(copy.deepcopy(dict(item)))
            next_token = page.get("nextPageToken")
            if next_token is None:
                return items
            if (
                not isinstance(next_token, str)
                or not next_token
                or next_token in visited
            ):
                raise ProductionDatabaseRecoveryError(
                    "production_database_recovery_provider_evidence_invalid"
                )
            visited.add(next_token)
            token = next_token
        raise ProductionDatabaseRecoveryError(
            "production_database_recovery_provider_evidence_invalid"
        )

    def _instances(self) -> list[Mapping[str, Any]]:
        return self._pages("/instances", item_kind="sql#instance")

    def _databases(self, instance: str) -> list[Mapping[str, Any]]:
        quoted = urllib.parse.quote(instance, safe="")
        return self._pages(
            f"/instances/{quoted}/databases",
            item_kind="sql#database",
        )

    def _operations(self) -> list[Mapping[str, Any]]:
        return self._pages("/operations", item_kind="sql#operation")

    def _wait_operation(
        self,
        operation_id: str,
        *,
        target: str,
        allowed_types: frozenset[str],
    ) -> Mapping[str, Any]:
        operation = _operation_id(operation_id)
        deadline = self._monotonic() + self._operation_timeout_seconds
        while True:
            raw = self._request(
                "GET",
                f"/operations/{urllib.parse.quote(operation, safe='')}",
            )
            if (
                raw.get("kind") != "sql#operation"
                or raw.get("name") != operation
                or raw.get("targetProject") != cutover.PROJECT
                or raw.get("targetId") != target
                or raw.get("operationType") not in allowed_types
                or raw.get("status") not in {"PENDING", "RUNNING", "DONE"}
            ):
                raise ProductionDatabaseRecoveryError(
                    "production_database_recovery_operation_invalid"
                )
            if raw["status"] == "DONE":
                if raw.get("error") is not None:
                    raise ProductionDatabaseRecoveryError(
                        "production_database_recovery_operation_failed"
                    )
                return copy.deepcopy(dict(raw))
            if self._monotonic() >= deadline:
                raise ProductionDatabaseRecoveryError(
                    "production_database_recovery_operation_timeout"
                )
            self._sleeper(2.0)

    def _find_operation(
        self,
        *,
        target: str,
        allowed_types: frozenset[str],
        backup_id: str | None = None,
    ) -> Mapping[str, Any] | None:
        matches = []
        for operation in self._operations():
            context = operation.get("backupContext")
            observed_backup = (
                str(context.get("backupId"))
                if isinstance(context, Mapping) and context.get("backupId") is not None
                else None
            )
            if (
                operation.get("targetProject") == cutover.PROJECT
                and operation.get("targetId") == target
                and operation.get("operationType") in allowed_types
                and (backup_id is None or observed_backup == backup_id)
            ):
                matches.append(operation)
        if len(matches) > 1:
            matches.sort(key=lambda item: str(item.get("insertTime") or ""))
        return copy.deepcopy(dict(matches[-1])) if matches else None

    @staticmethod
    def _instance_projection(
        raw: Mapping[str, Any],
        *,
        expected_name: str,
        expected_network: str | None,
        scratch: bool,
    ) -> Mapping[str, Any]:
        settings = raw.get("settings")
        ip_configuration = (
            settings.get("ipConfiguration") if isinstance(settings, Mapping) else None
        )
        backup = (
            settings.get("backupConfiguration") if isinstance(settings, Mapping) else None
        )
        addresses = raw.get("ipAddresses")
        private_addresses = [
            item
            for item in addresses
            if isinstance(item, Mapping) and item.get("type") == "PRIVATE"
        ] if isinstance(addresses, list) else []
        private_ip = (
            private_addresses[0].get("ipAddress")
            if len(private_addresses) == 1
            else None
        )
        try:
            parsed_private_ip = ipaddress.ip_address(str(private_ip))
        except ValueError:
            parsed_private_ip = None
        if (
            raw.get("kind") != "sql#instance"
            or raw.get("project") != cutover.PROJECT
            or raw.get("name") != expected_name
            or raw.get("region") != cutover.PRODUCTION_SQL_REGION
            or raw.get("state") != "RUNNABLE"
            or not isinstance(settings, Mapping)
            or not isinstance(ip_configuration, Mapping)
            or not isinstance(backup, Mapping)
            or not isinstance(raw.get("databaseVersion"), str)
            or not raw["databaseVersion"].startswith("POSTGRES_")
            or ip_configuration.get("ipv4Enabled") is not False
            or (ip_configuration.get("authorizedNetworks") or []) != []
            or ip_configuration.get("sslMode") != "ENCRYPTED_ONLY"
            or not isinstance(ip_configuration.get("privateNetwork"), str)
            or len(private_addresses) != 1
            or parsed_private_ip is None
            or parsed_private_ip.version != 4
            or not any(parsed_private_ip in network for network in _RFC1918)
            or any(
                isinstance(item, Mapping) and item.get("type") == "PRIMARY"
                for item in (addresses or [])
            )
            or (
                expected_network is not None
                and ip_configuration.get("privateNetwork") != expected_network
            )
            or (
                scratch
                and (
                    backup.get("enabled") is not False
                    or backup.get("pointInTimeRecoveryEnabled") not in {None, False}
                    or settings.get("deletionProtectionEnabled") is not False
                    or ip_configuration.get("serverCaMode")
                    != "GOOGLE_MANAGED_INTERNAL_CA"
                )
            )
        ):
            raise ProductionDatabaseRecoveryError(
                "production_database_recovery_instance_readback_invalid"
            )
        configuration = {
            "database_version": raw["databaseVersion"],
            "settings_version": str(settings.get("settingsVersion") or ""),
            "tier": settings.get("tier"),
            "edition": settings.get("edition"),
            "availability_type": settings.get("availabilityType"),
            "activation_policy": settings.get("activationPolicy"),
            "data_disk_type": settings.get("dataDiskType"),
            "data_disk_size_gb": str(settings.get("dataDiskSizeGb") or ""),
            "storage_auto_resize": settings.get("storageAutoResize"),
            "backup_configuration": copy.deepcopy(dict(backup)),
            "ip_configuration": copy.deepcopy(dict(ip_configuration)),
        }
        projection = {
            "project": cutover.PROJECT,
            "instance": expected_name,
            "region": cutover.PRODUCTION_SQL_REGION,
            "private_network": str(ip_configuration["privateNetwork"]),
            "database_version": str(raw["databaseVersion"]),
            "configuration": configuration,
            "configuration_sha256": _sha_json(configuration),
            "readback_sha256": _sha_json(raw),
        }
        if scratch:
            ca = raw.get("serverCaCert")
            connection_name = raw.get("connectionName")
            if not isinstance(ca, Mapping):
                raise ProductionDatabaseRecoveryError(
                    "production_database_recovery_server_ca_invalid"
                )
            ca_pem = ca.get("cert")
            try:
                ca_raw = str(ca_pem).encode("ascii", errors="strict")
                ssl.PEM_cert_to_DER_cert(str(ca_pem))
            except (UnicodeError, ValueError, ssl.SSLError) as exc:
                raise ProductionDatabaseRecoveryError(
                    "production_database_recovery_server_ca_invalid"
                ) from exc
            if (
                ca.get("kind") != "sql#sslCert"
                or ca.get("instance") != expected_name
                or not 1 <= len(ca_raw) <= remote_probe.MAX_CA_BYTES
                or connection_name
                != f"{cutover.PROJECT}:{cutover.PRODUCTION_SQL_REGION}:{expected_name}"
            ):
                raise ProductionDatabaseRecoveryError(
                    "production_database_recovery_server_ca_invalid"
                )
            projection.update({
                "private_ip": str(parsed_private_ip),
                "server_ca_pem": str(ca_pem),
                "server_ca_sha256": _sha(ca_raw),
                "ssl_mode": "ENCRYPTED_ONLY",
                "server_ca_mode": "GOOGLE_MANAGED_INTERNAL_CA",
                "connection_name": str(connection_name),
            })
        return projection

    def source_readback(self) -> Mapping[str, Any]:
        matches = [
            item
            for item in self._instances()
            if item.get("name") == cutover.PRODUCTION_SQL_INSTANCE
        ]
        if len(matches) != 1:
            raise ProductionDatabaseRecoveryError(
                "production_database_recovery_source_missing"
            )
        projection = self._instance_projection(
            matches[0],
            expected_name=cutover.PRODUCTION_SQL_INSTANCE,
            expected_network=None,
            scratch=False,
        )
        network = str(projection["private_network"])
        databases = self._databases(cutover.PRODUCTION_SQL_INSTANCE)
        names = [item.get("name") for item in databases]
        if (
            _SOURCE_PRIVATE_NETWORK.fullmatch(network) is None
            or names.count(cutover.DATABASE) != 1
        ):
            raise ProductionDatabaseRecoveryError(
                "production_database_recovery_source_configuration_invalid"
            )
        result = {
            name: item
            for name, item in projection.items()
            if name != "configuration"
        }
        result["database"] = cutover.DATABASE
        return result

    def _backup_runs(self) -> list[Mapping[str, Any]]:
        source = urllib.parse.quote(cutover.PRODUCTION_SQL_INSTANCE, safe="")
        return self._pages(
            f"/instances/{source}/backupRuns",
            item_kind="sql#backupRun",
        )

    def _backup_projection(
        self,
        raw: Mapping[str, Any],
        *,
        operation_id: str,
    ) -> Mapping[str, Any]:
        completed = _unix_time(
            raw.get("endTime") or raw.get("startTime"),
            "production_database_recovery_backup_invalid",
        )
        if (
            raw.get("instance") != cutover.PRODUCTION_SQL_INSTANCE
            or raw.get("status") != "SUCCESSFUL"
            or raw.get("type") != "ON_DEMAND"
        ):
            raise ProductionDatabaseRecoveryError(
                "production_database_recovery_backup_invalid"
            )
        return {
            "backup_id": _backup_id(raw.get("id")),
            "operation_id": _operation_id(operation_id),
            "status": "SUCCESSFUL",
            "type": "ON_DEMAND",
            "source_instance": cutover.PRODUCTION_SQL_INSTANCE,
            "completed_at_unix": completed,
            "retained": True,
            "readback_sha256": _sha_json(raw),
        }

    def ensure_backup(
        self, *, release_revision: str, not_before_unix: int
    ) -> Mapping[str, Any]:
        description = _backup_description(release_revision)
        matches = [
            item for item in self._backup_runs() if item.get("description") == description
        ]
        if len(matches) > 1:
            raise ProductionDatabaseRecoveryError(
                "production_database_recovery_backup_ambiguous"
            )
        if matches:
            backup_id = _backup_id(matches[0].get("id"))
            operation = self._find_operation(
                target=cutover.PRODUCTION_SQL_INSTANCE,
                allowed_types=frozenset({"BACKUP_VOLUME"}),
                backup_id=backup_id,
            )
            if operation is None:
                raise ProductionDatabaseRecoveryError(
                    "production_database_recovery_backup_operation_missing"
                )
            done = self._wait_operation(
                _operation_id(operation.get("name")),
                target=cutover.PRODUCTION_SQL_INSTANCE,
                allowed_types=frozenset({"BACKUP_VOLUME"}),
            )
            fresh = [
                item
                for item in self._backup_runs()
                if str(item.get("id")) == backup_id
            ]
            if len(fresh) != 1 or fresh[0].get("description") != description:
                raise ProductionDatabaseRecoveryError(
                    "production_database_recovery_backup_invalid"
                )
            result = self._backup_projection(
                fresh[0], operation_id=str(done["name"])
            )
        else:
            source = urllib.parse.quote(cutover.PRODUCTION_SQL_INSTANCE, safe="")
            operation = self._request(
                "POST",
                f"/instances/{source}/backupRuns",
                body={"description": description},
            )
            done = self._wait_operation(
                _operation_id(operation.get("name")),
                target=cutover.PRODUCTION_SQL_INSTANCE,
                allowed_types=frozenset({"BACKUP_VOLUME"}),
            )
            context = done.get("backupContext")
            if not isinstance(context, Mapping):
                raise ProductionDatabaseRecoveryError(
                    "production_database_recovery_backup_operation_invalid"
                )
            backup_id = _backup_id(context.get("backupId"))
            fresh = [
                item for item in self._backup_runs() if str(item.get("id")) == backup_id
            ]
            if len(fresh) != 1 or fresh[0].get("description") != description:
                raise ProductionDatabaseRecoveryError(
                    "production_database_recovery_backup_invalid"
                )
            result = self._backup_projection(
                fresh[0], operation_id=str(done["name"])
            )
        if result["completed_at_unix"] < not_before_unix:
            raise ProductionDatabaseRecoveryError(
                "production_database_recovery_backup_not_fresh"
            )
        return result

    @staticmethod
    def _scratch_body(source: Mapping[str, Any], revision: str) -> Mapping[str, Any]:
        database_version = source.get("database_version")
        if (
            source.get("project") != cutover.PROJECT
            or source.get("instance") != cutover.PRODUCTION_SQL_INSTANCE
            or source.get("region") != cutover.PRODUCTION_SQL_REGION
            or not isinstance(database_version, str)
            or not database_version.startswith("POSTGRES_")
        ):
            raise ProductionDatabaseRecoveryError(
                "production_database_recovery_source_configuration_invalid"
            )
        return {
            "name": cutover.database_recovery_scratch_instance(revision),
            "project": cutover.PROJECT,
            "region": cutover.PRODUCTION_SQL_REGION,
            "databaseVersion": database_version,
            "settings": {
                "tier": "db-custom-1-3840",
                "edition": "ENTERPRISE",
                "availabilityType": "ZONAL",
                "activationPolicy": "ALWAYS",
                "dataDiskType": "PD_SSD",
                "dataDiskSizeGb": "10",
                "storageAutoResize": True,
                "deletionProtectionEnabled": False,
                "backupConfiguration": {
                    "enabled": False,
                    "pointInTimeRecoveryEnabled": False,
                },
                "ipConfiguration": {
                    "ipv4Enabled": False,
                    "privateNetwork": cutover.DATABASE_RECOVERY_SCRATCH_NETWORK,
                    "authorizedNetworks": [],
                    "sslMode": "ENCRYPTED_ONLY",
                    "serverCaMode": "GOOGLE_MANAGED_INTERNAL_CA",
                },
            },
        }

    def ensure_scratch(
        self, *, release_revision: str, source: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        scratch = cutover.database_recovery_scratch_instance(release_revision)
        matches = [item for item in self._instances() if item.get("name") == scratch]
        if len(matches) > 1:
            raise ProductionDatabaseRecoveryError(
                "production_database_recovery_scratch_ambiguous"
            )
        if matches:
            operation = self._find_operation(
                target=scratch,
                allowed_types=frozenset({"CREATE"}),
            )
            if operation is None:
                raise ProductionDatabaseRecoveryError(
                    "production_database_recovery_scratch_operation_missing"
                )
            done = self._wait_operation(
                _operation_id(operation.get("name")),
                target=scratch,
                allowed_types=frozenset({"CREATE"}),
            )
        else:
            operation = self._request(
                "POST", "/instances", body=self._scratch_body(source, release_revision)
            )
            done = self._wait_operation(
                _operation_id(operation.get("name")),
                target=scratch,
                allowed_types=frozenset({"CREATE"}),
            )
        fresh = [item for item in self._instances() if item.get("name") == scratch]
        if len(fresh) != 1:
            raise ProductionDatabaseRecoveryError(
                "production_database_recovery_scratch_invalid"
            )
        raw = fresh[0]
        projection = self._instance_projection(
            raw,
            expected_name=scratch,
            expected_network=cutover.DATABASE_RECOVERY_SCRATCH_NETWORK,
            scratch=True,
        )
        return {**projection, "create_operation_id": str(done["name"])}

    def ensure_restore(
        self,
        *,
        release_revision: str,
        backup: Mapping[str, Any],
        scratch: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        name = cutover.database_recovery_scratch_instance(release_revision)
        backup_id = _backup_id(backup.get("backup_id"))
        if scratch.get("instance") != name:
            raise ProductionDatabaseRecoveryError(
                "production_database_recovery_scratch_invalid"
            )
        operation = self._find_operation(
            target=name,
            allowed_types=frozenset({"RESTORE_VOLUME"}),
            backup_id=backup_id,
        )
        if operation is None:
            operation = self._request(
                "POST",
                f"/instances/{urllib.parse.quote(name, safe='')}/restoreBackup",
                body={
                    "restoreBackupContext": {
                        "kind": "sql#restoreBackupContext",
                        "backupRunId": backup_id,
                        "instanceId": cutover.PRODUCTION_SQL_INSTANCE,
                        "project": cutover.PROJECT,
                    }
                },
            )
        done = self._wait_operation(
            _operation_id(operation.get("name")),
            target=name,
            allowed_types=frozenset({"RESTORE_VOLUME"}),
        )
        matches = [item for item in self._instances() if item.get("name") == name]
        databases = self._databases(name)
        if len(matches) != 1 or [item.get("name") for item in databases].count(
            cutover.DATABASE
        ) != 1:
            raise ProductionDatabaseRecoveryError(
                "production_database_recovery_restore_invalid"
            )
        projection = self._instance_projection(
            matches[0],
            expected_name=name,
            expected_network=cutover.DATABASE_RECOVERY_SCRATCH_NETWORK,
            scratch=True,
        )
        return {
            **projection,
            "create_operation_id": scratch["create_operation_id"],
            "restore_operation_id": str(done["name"]),
            "restored_backup_id": backup_id,
        }

    def scratch_readback(self, *, release_revision: str) -> Mapping[str, Any]:
        name = cutover.database_recovery_scratch_instance(release_revision)
        matches = [item for item in self._instances() if item.get("name") == name]
        databases = self._databases(name)
        if (
            len(matches) != 1
            or [item.get("name") for item in databases].count(cutover.DATABASE) != 1
        ):
            raise ProductionDatabaseRecoveryError(
                "production_database_recovery_probe_readback_invalid"
            )
        return self._instance_projection(
            matches[0],
            expected_name=name,
            expected_network=cutover.DATABASE_RECOVERY_SCRATCH_NETWORK,
            scratch=True,
        )

    def delete_scratch(
        self, *, release_revision: str
    ) -> Mapping[str, Any]:
        name = cutover.database_recovery_scratch_instance(release_revision)
        matches = [item for item in self._instances() if item.get("name") == name]
        if len(matches) > 1:
            raise ProductionDatabaseRecoveryError(
                "production_database_recovery_scratch_ambiguous"
            )
        if matches:
            operation = self._find_operation(
                target=name,
                allowed_types=frozenset({"DELETE"}),
            )
            if operation is None:
                operation = self._request(
                    "DELETE",
                    f"/instances/{urllib.parse.quote(name, safe='')}",
                    query={"enableFinalBackup": "false"},
                )
        else:
            operation = self._find_operation(
                target=name,
                allowed_types=frozenset({"DELETE"}),
            )
            if operation is None:
                raise ProductionDatabaseRecoveryError(
                    "production_database_recovery_scratch_delete_unproven"
                )
        done = self._wait_operation(
            _operation_id(operation.get("name")),
            target=name,
            allowed_types=frozenset({"DELETE"}),
        )
        if any(item.get("name") == name for item in self._instances()):
            raise ProductionDatabaseRecoveryError(
                "production_database_recovery_scratch_delete_unproven"
            )
        return {
            "instance": name,
            "delete_operation_id": str(done["name"]),
            "deleted": True,
        }

    def backup_readback(self, *, backup_id: str) -> Mapping[str, Any]:
        expected = _backup_id(backup_id)
        matches = [
            item for item in self._backup_runs() if str(item.get("id")) == expected
        ]
        if len(matches) != 1 or matches[0].get("status") != "SUCCESSFUL":
            raise ProductionDatabaseRecoveryError(
                "production_database_recovery_backup_recheck_failed"
            )
        return {
            "backup_id": expected,
            "status": "SUCCESSFUL",
            "type": matches[0].get("type"),
            "source_instance": matches[0].get("instance"),
            "readback_sha256": _sha_json(matches[0]),
        }


def build_probe_receipt(
    *,
    release_revision: str,
    scratch_instance: str,
    transaction_read_only: bool,
    schema_sha256: str,
    content_sha256: str,
    canonical_event_row_count: int,
    probed_at_unix: int,
    scratch_private_ip: str = "10.0.0.2",
    server_ca_sha256: str = "a" * 64,
    release_manifest_file_sha256: str = "b" * 64,
    stopped_units_sha256: str = "c" * 64,
    psql_executable_sha256: str = remote_probe.EXPECTED_PSQL_SHA256,
) -> Mapping[str, Any]:
    revision = _release(release_revision)
    unsigned = {
        "schema": cutover.DATABASE_RECOVERY_PROBE_RECEIPT_SCHEMA,
        "ok": True,
        "release_revision": revision,
        "scratch_instance": scratch_instance,
        "database": cutover.DATABASE,
        "probe_contract_sha256": _sha_json(
            cutover.DATABASE_RECOVERY_PROBE_CONTRACT
        ),
        "transaction_read_only": transaction_read_only,
        "schema_sha256": schema_sha256,
        "content_sha256": content_sha256,
        "canonical_event_row_count": canonical_event_row_count,
        "scratch_private_ip": scratch_private_ip,
        "server_ca_sha256": server_ca_sha256,
        "tls_mode": remote_probe.TLS_MODE,
        "tls_ca_verified": True,
        "tls_hostname_verified": False,
        "canary_instance_id": owner_transport.VM_INSTANCE_ID,
        "canary_network": remote_probe.EXPECTED_NETWORK,
        "canary_subnetwork": remote_probe.EXPECTED_SUBNETWORK,
        "canary_private_ip": remote_probe.EXPECTED_PRIVATE_IP,
        "release_manifest_file_sha256": release_manifest_file_sha256,
        "stopped_units_sha256": stopped_units_sha256,
        "psql_executable_sha256": psql_executable_sha256,
        "probed_at_unix": probed_at_unix,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    receipt = {**unsigned, "receipt_sha256": _sha_json(unsigned)}
    try:
        cutover._validate_database_recovery_receipt
    except AttributeError as exc:  # pragma: no cover - packaging contract guard
        raise ProductionDatabaseRecoveryError(
            "production_database_recovery_contract_unavailable"
        ) from exc
    if (
        scratch_instance != cutover.database_recovery_scratch_instance(revision)
        or transaction_read_only is not True
        or _SHA256.fullmatch(schema_sha256 or "") is None
        or _SHA256.fullmatch(content_sha256 or "") is None
        or type(canonical_event_row_count) is not int
        or canonical_event_row_count < 0
        or type(probed_at_unix) is not int
        or probed_at_unix <= 0
        or not isinstance(scratch_private_ip, str)
        or psql_executable_sha256 != remote_probe.EXPECTED_PSQL_SHA256
        or any(
            _SHA256.fullmatch(value or "") is None
            for value in (
                server_ca_sha256,
                release_manifest_file_sha256,
                stopped_units_sha256,
                psql_executable_sha256,
            )
        )
    ):
        raise ProductionDatabaseRecoveryError(
            "production_database_recovery_probe_invalid"
        )
    return receipt


def _validated_backup_recheck(
    provider: RecoveryProvider,
    *,
    backup_id: str,
) -> Mapping[str, Any]:
    rechecked = provider.backup_readback(backup_id=backup_id)
    if (
        rechecked.get("backup_id") != backup_id
        or rechecked.get("status") != "SUCCESSFUL"
        or rechecked.get("type") != "ON_DEMAND"
        or rechecked.get("source_instance") != cutover.PRODUCTION_SQL_INSTANCE
        or _SHA256.fullmatch(str(rechecked.get("readback_sha256"))) is None
    ):
        raise ProductionDatabaseRecoveryError(
            "production_database_recovery_backup_recheck_failed"
        )
    return copy.deepcopy(dict(rechecked))


def _validated_probe_receipt_for_delete(
    value: Mapping[str, Any],
    *,
    release_revision: str,
    scratch_instance: str,
    backup_completed_at_unix: int,
    now_unix: int,
    expected_scratch: Mapping[str, Any] | None = None,
    expected_preflight: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProductionDatabaseRecoveryError(
            "production_database_recovery_probe_invalid"
        )
    unsigned = {
        name: item for name, item in value.items() if name != "receipt_sha256"
    }
    if (
        set(value) != _PROBE_RECEIPT_FIELDS
        or value.get("schema")
        != cutover.DATABASE_RECOVERY_PROBE_RECEIPT_SCHEMA
        or value.get("ok") is not True
        or value.get("release_revision") != release_revision
        or value.get("scratch_instance") != scratch_instance
        or value.get("database") != cutover.DATABASE
        or value.get("probe_contract_sha256")
        != _sha_json(cutover.DATABASE_RECOVERY_PROBE_CONTRACT)
        or value.get("transaction_read_only") is not True
        or _SHA256.fullmatch(str(value.get("schema_sha256"))) is None
        or _SHA256.fullmatch(str(value.get("content_sha256"))) is None
        or type(value.get("canonical_event_row_count")) is not int
        or value["canonical_event_row_count"] < 0
        or not isinstance(value.get("scratch_private_ip"), str)
        or value.get("tls_mode") != remote_probe.TLS_MODE
        or value.get("tls_ca_verified") is not True
        or value.get("tls_hostname_verified") is not False
        or value.get("canary_instance_id") != owner_transport.VM_INSTANCE_ID
        or value.get("canary_network") != remote_probe.EXPECTED_NETWORK
        or value.get("canary_subnetwork") != remote_probe.EXPECTED_SUBNETWORK
        or value.get("canary_private_ip") != remote_probe.EXPECTED_PRIVATE_IP
        or value.get("psql_executable_sha256")
        != remote_probe.EXPECTED_PSQL_SHA256
        or type(value.get("probed_at_unix")) is not int
        or not backup_completed_at_unix
        <= value["probed_at_unix"]
        <= now_unix
        or value.get("secret_material_recorded") is not False
        or value.get("secret_digest_recorded") is not False
        or value.get("receipt_sha256") != _sha_json(unsigned)
        or any(
            _SHA256.fullmatch(str(value.get(name))) is None
            for name in (
                "server_ca_sha256",
                "release_manifest_file_sha256",
                "stopped_units_sha256",
                "psql_executable_sha256",
            )
        )
    ):
        raise ProductionDatabaseRecoveryError(
            "production_database_recovery_probe_invalid"
        )
    try:
        address = ipaddress.ip_address(str(value["scratch_private_ip"]))
    except ValueError as exc:
        raise ProductionDatabaseRecoveryError(
            "production_database_recovery_probe_invalid"
        ) from exc
    if address.version != 4 or not any(address in network for network in _RFC1918):
        raise ProductionDatabaseRecoveryError(
            "production_database_recovery_probe_invalid"
        )
    if expected_scratch is not None and (
        value["scratch_private_ip"] != expected_scratch.get("private_ip")
        or value["server_ca_sha256"] != expected_scratch.get("server_ca_sha256")
    ):
        raise ProductionDatabaseRecoveryError(
            "production_database_recovery_probe_invalid"
        )
    if expected_preflight is not None and any(
        value[name] != expected_preflight.get(name)
        for name in (
            "canary_instance_id",
            "canary_network",
            "canary_subnetwork",
            "canary_private_ip",
            "release_manifest_file_sha256",
            "stopped_units_sha256",
            "psql_executable_sha256",
        )
    ):
        raise ProductionDatabaseRecoveryError(
            "production_database_recovery_probe_invalid"
        )
    return copy.deepcopy(dict(value))


def _execute_gate(
    *,
    release_revision: str,
    provider: RecoveryProvider,
    probe: RecoveryProbe,
    journal_root: Path,
    clock: Callable[[], float] = time.time,
) -> Mapping[str, Any]:
    revision = _release(release_revision)
    if not callable(clock):
        raise ProductionDatabaseRecoveryError(
            "production_database_recovery_clock_invalid"
        )
    probe.require_available()
    now = int(clock())
    journal = _Journal(journal_root, revision, now_unix=now)
    manifest = journal.payload("manifest")

    if journal.has("terminal_receipt"):
        receipt = journal.latest_payload("terminal_receipt")
        try:
            validated = cutover._validate_database_recovery_receipt(
                receipt,
                revision=revision,
            )
        except ValueError as exc:
            raise ProductionDatabaseRecoveryError(
                "production_database_recovery_receipt_invalid"
            ) from exc
        rechecked_at = int(validated["backup_rechecked_at_unix"])
        if 0 <= now - rechecked_at <= MAX_BACKUP_RECHECK_AGE_SECONDS:
            return validated
        backup = validated["backup"]
        if (
            now < rechecked_at
            or now - int(backup["completed_at_unix"])
            > MAX_BACKUP_AGE_SECONDS
        ):
            raise ProductionDatabaseRecoveryError(
                "production_database_recovery_receipt_stale"
            )
        generation = journal.refresh_generation()
        refresh_stage = f"backup_rechecked_refresh_{generation}"
        if journal.has(refresh_stage):
            refresh = journal.payload(refresh_stage)
        else:
            refresh = {
                **dict(
                    _validated_backup_recheck(
                        provider,
                        backup_id=str(backup["backup_id"]),
                    )
                ),
                "rechecked_at_unix": int(clock()),
            }
            journal.record(
                refresh_stage,
                refresh,
                now_unix=int(clock()),
            )
        if (
            set(refresh)
            != {
                "backup_id",
                "status",
                "type",
                "source_instance",
                "readback_sha256",
                "rechecked_at_unix",
            }
            or refresh.get("backup_id") != backup["backup_id"]
            or refresh.get("status") != "SUCCESSFUL"
            or refresh.get("type") != "ON_DEMAND"
            or refresh.get("source_instance")
            != cutover.PRODUCTION_SQL_INSTANCE
            or _SHA256.fullmatch(str(refresh.get("readback_sha256"))) is None
            or type(refresh.get("rechecked_at_unix")) is not int
            or refresh["rechecked_at_unix"] < rechecked_at
        ):
            raise ProductionDatabaseRecoveryError(
                "production_database_recovery_backup_recheck_failed"
            )
        unsigned = {
            name: copy.deepcopy(item)
            for name, item in validated.items()
            if name != "receipt_sha256"
        }
        unsigned["backup_rechecked_at_unix"] = refresh[
            "rechecked_at_unix"
        ]
        unsigned["journal_prefix_sha256"] = journal.prefix_sha256
        refreshed_receipt = {
            **unsigned,
            "receipt_sha256": _sha_json(unsigned),
        }
        try:
            refreshed = cutover._validate_database_recovery_receipt(
                refreshed_receipt,
                revision=revision,
            )
        except ValueError as exc:
            raise ProductionDatabaseRecoveryError(
                "production_database_recovery_receipt_invalid"
            ) from exc
        journal.record(
            f"terminal_receipt_refresh_{generation}",
            refreshed,
            now_unix=int(clock()),
        )
        final_now = int(clock())
        if (
            final_now - int(refreshed["backup_rechecked_at_unix"])
            > MAX_BACKUP_RECHECK_AGE_SECONDS
        ):
            return _execute_gate(
                release_revision=revision,
                provider=provider,
                probe=probe,
                journal_root=journal_root,
                clock=clock,
            )
        try:
            return cutover._validate_database_recovery_receipt(
                refreshed,
                revision=revision,
                now_unix=final_now,
                max_recheck_age_seconds=MAX_BACKUP_RECHECK_AGE_SECONDS,
            )
        except ValueError as exc:
            raise ProductionDatabaseRecoveryError(
                "production_database_recovery_receipt_invalid"
            ) from exc

    live_source = provider.source_readback()
    if journal.has("source_readback"):
        source = journal.payload("source_readback")
        if live_source != source:
            raise ProductionDatabaseRecoveryError(
                "production_database_recovery_source_drifted"
            )
    else:
        source = journal.record("source_readback", live_source, now_unix=int(clock()))
    if (
        source.get("project") != cutover.PROJECT
        or source.get("instance") != cutover.PRODUCTION_SQL_INSTANCE
        or source.get("region") != cutover.PRODUCTION_SQL_REGION
        or source.get("database") != cutover.DATABASE
        or not isinstance(source.get("private_network"), str)
        or _SOURCE_PRIVATE_NETWORK.fullmatch(source["private_network"]) is None
        or any(
            _SHA256.fullmatch(str(source.get(name))) is None
            for name in ("configuration_sha256", "readback_sha256")
        )
    ):
        raise ProductionDatabaseRecoveryError(
            "production_database_recovery_source_configuration_invalid"
        )

    backup_intent = {
        "source_instance": cutover.PRODUCTION_SQL_INSTANCE,
        "description": _backup_description(revision),
        "retain_backup": True,
    }
    journal.record("backup_intent", backup_intent, now_unix=int(clock()))
    if journal.has("backup_ready"):
        backup = journal.payload("backup_ready")
    else:
        backup = journal.record(
            "backup_ready",
            provider.ensure_backup(
                release_revision=revision,
                not_before_unix=int(manifest["started_at_unix"]),
            ),
            now_unix=int(clock()),
        )
    if (
        backup.get("source_instance") != cutover.PRODUCTION_SQL_INSTANCE
        or backup.get("status") != "SUCCESSFUL"
        or backup.get("type") != "ON_DEMAND"
        or backup.get("retained") is not True
        or _BACKUP_ID.fullmatch(str(backup.get("backup_id"))) is None
        or _OPERATION_ID.fullmatch(str(backup.get("operation_id"))) is None
        or type(backup.get("completed_at_unix")) is not int
        or backup["completed_at_unix"] < manifest["started_at_unix"]
    ):
        raise ProductionDatabaseRecoveryError(
            "production_database_recovery_backup_invalid"
        )

    scratch_name = cutover.database_recovery_scratch_instance(revision)
    journal.record(
        "scratch_intent",
        {
            "instance": scratch_name,
            "network": cutover.DATABASE_RECOVERY_SCRATCH_NETWORK,
            "private_only": True,
            "backup_enabled": False,
            "deletion_protection_enabled": False,
        },
        now_unix=int(clock()),
    )
    if journal.has("scratch_ready"):
        scratch = journal.payload("scratch_ready")
    else:
        scratch = journal.record(
            "scratch_ready",
            provider.ensure_scratch(release_revision=revision, source=source),
            now_unix=int(clock()),
        )
    if (
        scratch.get("project") != cutover.PROJECT
        or scratch.get("instance") != scratch_name
        or scratch.get("region") != cutover.PRODUCTION_SQL_REGION
        or scratch.get("private_network")
        != cutover.DATABASE_RECOVERY_SCRATCH_NETWORK
        or not isinstance(scratch.get("database_version"), str)
        or not scratch["database_version"].startswith("POSTGRES_")
        or _SHA256.fullmatch(str(scratch.get("configuration_sha256"))) is None
        or _SHA256.fullmatch(str(scratch.get("readback_sha256"))) is None
        or _OPERATION_ID.fullmatch(str(scratch.get("create_operation_id"))) is None
        or not isinstance(scratch.get("private_ip"), str)
        or scratch.get("ssl_mode") != "ENCRYPTED_ONLY"
        or scratch.get("server_ca_mode") != "GOOGLE_MANAGED_INTERNAL_CA"
        or _SHA256.fullmatch(str(scratch.get("server_ca_sha256"))) is None
        or not isinstance(scratch.get("server_ca_pem"), str)
        or scratch.get("server_ca_sha256")
        != _sha(scratch["server_ca_pem"].encode("ascii", errors="strict"))
        or scratch.get("connection_name")
        != f"{cutover.PROJECT}:{cutover.PRODUCTION_SQL_REGION}:{scratch_name}"
    ):
        raise ProductionDatabaseRecoveryError(
            "production_database_recovery_scratch_invalid"
        )

    journal.record(
        "restore_intent",
        {"instance": scratch_name, "backup_id": str(backup["backup_id"])},
        now_unix=int(clock()),
    )
    if journal.has("restore_ready"):
        restored = journal.payload("restore_ready")
    else:
        restored = journal.record(
            "restore_ready",
            provider.ensure_restore(
                release_revision=revision,
                backup=backup,
                scratch=scratch,
            ),
            now_unix=int(clock()),
        )
    if (
        restored.get("project") != cutover.PROJECT
        or restored.get("instance") != scratch_name
        or restored.get("region") != cutover.PRODUCTION_SQL_REGION
        or restored.get("private_network")
        != cutover.DATABASE_RECOVERY_SCRATCH_NETWORK
        or restored.get("restored_backup_id") != backup["backup_id"]
        or restored.get("create_operation_id")
        != scratch["create_operation_id"]
        or not isinstance(restored.get("database_version"), str)
        or not restored["database_version"].startswith("POSTGRES_")
        or _SHA256.fullmatch(str(restored.get("configuration_sha256"))) is None
        or _SHA256.fullmatch(str(restored.get("readback_sha256"))) is None
        or _OPERATION_ID.fullmatch(
            str(restored.get("restore_operation_id"))
        )
        is None
        or not isinstance(restored.get("private_ip"), str)
        or restored.get("ssl_mode") != "ENCRYPTED_ONLY"
        or restored.get("server_ca_mode") != "GOOGLE_MANAGED_INTERNAL_CA"
        or _SHA256.fullmatch(str(restored.get("server_ca_sha256"))) is None
        or not isinstance(restored.get("server_ca_pem"), str)
        or restored.get("server_ca_sha256")
        != _sha(restored["server_ca_pem"].encode("ascii", errors="strict"))
        or restored.get("connection_name")
        != f"{cutover.PROJECT}:{cutover.PRODUCTION_SQL_REGION}:{scratch_name}"
    ):
        raise ProductionDatabaseRecoveryError(
            "production_database_recovery_restore_invalid"
        )

    if journal.has("probe_receipt"):
        probe_receipt = journal.payload("probe_receipt")
    else:
        fresh_probe_readback = provider.scratch_readback(
            release_revision=revision
        )
        if (
            fresh_probe_readback.get("project") != cutover.PROJECT
            or fresh_probe_readback.get("instance") != scratch_name
            or fresh_probe_readback.get("region") != cutover.PRODUCTION_SQL_REGION
            or fresh_probe_readback.get("private_network")
            != cutover.DATABASE_RECOVERY_SCRATCH_NETWORK
            or fresh_probe_readback.get("ssl_mode") != "ENCRYPTED_ONLY"
            or fresh_probe_readback.get("server_ca_mode")
            != "GOOGLE_MANAGED_INTERNAL_CA"
            or not isinstance(fresh_probe_readback.get("private_ip"), str)
            or not isinstance(fresh_probe_readback.get("server_ca_pem"), str)
            or fresh_probe_readback.get("server_ca_sha256")
            != _sha(
                fresh_probe_readback["server_ca_pem"].encode(
                    "ascii", errors="strict"
                )
            )
            or fresh_probe_readback.get("connection_name")
            != f"{cutover.PROJECT}:{cutover.PRODUCTION_SQL_REGION}:{scratch_name}"
            or _SHA256.fullmatch(
                str(fresh_probe_readback.get("readback_sha256"))
            )
            is None
        ):
            raise ProductionDatabaseRecoveryError(
                "production_database_recovery_probe_readback_invalid"
            )
        probe_scratch = {
            **dict(restored),
            **dict(fresh_probe_readback),
            "create_operation_id": scratch["create_operation_id"],
            "restore_operation_id": restored["restore_operation_id"],
            "restored_backup_id": backup["backup_id"],
        }
        candidate_probe_receipt = probe.probe(
            release_revision=revision,
            scratch=probe_scratch,
            now_unix=int(clock()),
        )
        candidate_probe_receipt = _validated_probe_receipt_for_delete(
            candidate_probe_receipt,
            release_revision=revision,
            scratch_instance=scratch_name,
            backup_completed_at_unix=int(backup["completed_at_unix"]),
            now_unix=int(clock()),
            expected_scratch=probe_scratch,
        )
        probe_receipt = journal.record(
            "probe_receipt",
            candidate_probe_receipt,
            now_unix=int(clock()),
        )
    probe_receipt = _validated_probe_receipt_for_delete(
        probe_receipt,
        release_revision=revision,
        scratch_instance=scratch_name,
        backup_completed_at_unix=int(backup["completed_at_unix"]),
        now_unix=int(clock()),
    )

    # The durable, self-hashed probe receipt above is a hard prerequisite for
    # the only delete boundary in this module.
    journal.record(
        "scratch_delete_intent",
        {
            "instance": scratch_name,
            "probe_receipt_sha256": probe_receipt["receipt_sha256"],
        },
        now_unix=int(clock()),
    )
    if journal.has("scratch_deleted"):
        deleted = journal.payload("scratch_deleted")
    else:
        delete_result = dict(
            provider.delete_scratch(release_revision=revision)
        )
        delete_result["deleted_at_unix"] = int(clock())
        deleted = journal.record(
            "scratch_deleted",
            delete_result,
            now_unix=int(clock()),
        )
    if (
        deleted.get("instance") != scratch_name
        or deleted.get("deleted") is not True
        or _OPERATION_ID.fullmatch(
            str(deleted.get("delete_operation_id"))
        )
        is None
        or type(deleted.get("deleted_at_unix")) is not int
        or deleted["deleted_at_unix"] <= 0
    ):
        raise ProductionDatabaseRecoveryError(
            "production_database_recovery_scratch_delete_unproven"
        )

    rechecked = _validated_backup_recheck(
        provider,
        backup_id=str(backup["backup_id"]),
    )
    recheck_payload = {
        **dict(rechecked),
        "rechecked_at_unix": int(clock()),
    }
    journal.record("backup_rechecked", recheck_payload, now_unix=int(clock()))
    journal_prefix = journal.prefix_sha256
    deleted_at = int(deleted["deleted_at_unix"])
    unsigned = {
        "schema": cutover.DATABASE_RECOVERY_RECEIPT_SCHEMA,
        "release_revision": revision,
        "source": {
            "project": source["project"],
            "instance": source["instance"],
            "region": source["region"],
            "database": source["database"],
            "private_network": source["private_network"],
            "database_version": source["database_version"],
            "configuration_sha256": source["configuration_sha256"],
            "readback_sha256": source["readback_sha256"],
        },
        "backup": copy.deepcopy(dict(backup)),
        "scratch": {
            "project": cutover.PROJECT,
            "instance": scratch_name,
            "region": cutover.PRODUCTION_SQL_REGION,
            "network": cutover.DATABASE_RECOVERY_SCRATCH_NETWORK,
            "create_operation_id": scratch["create_operation_id"],
            "restore_operation_id": restored["restore_operation_id"],
            "delete_operation_id": deleted["delete_operation_id"],
            "restored_backup_id": backup["backup_id"],
            "private_only": True,
            "backup_enabled": False,
            "deletion_protection_enabled": False,
            "deleted": True,
            "readback_sha256": restored["readback_sha256"],
            "deleted_at_unix": deleted_at,
            "private_ip": probe_receipt["scratch_private_ip"],
            "server_ca_sha256": probe_receipt["server_ca_sha256"],
            "cloud_sql_ssl_mode": "ENCRYPTED_ONLY",
            "tls_mode": probe_receipt["tls_mode"],
            "tls_ca_verified": probe_receipt["tls_ca_verified"],
            "tls_hostname_verified": probe_receipt["tls_hostname_verified"],
        },
        "probe_receipt": copy.deepcopy(dict(probe_receipt)),
        "backup_rechecked_at_unix": recheck_payload["rechecked_at_unix"],
        "journal_prefix_sha256": journal_prefix,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    receipt = {**unsigned, "receipt_sha256": _sha_json(unsigned)}
    try:
        validated = cutover._validate_database_recovery_receipt(
            receipt,
            revision=revision,
            now_unix=int(clock()),
            max_recheck_age_seconds=MAX_BACKUP_RECHECK_AGE_SECONDS,
        )
    except ValueError as exc:
        raise ProductionDatabaseRecoveryError(
            "production_database_recovery_receipt_invalid"
        ) from exc
    journal.record("terminal_receipt", validated, now_unix=int(clock()))
    return validated


def run_for_owner(
    release_revision: str,
    owner_identity: Any,
    expected_owner_subject_sha256: str,
) -> Mapping[str, Any]:
    """Run the sole production boundary with no caller-controlled target."""

    revision = _release(release_revision)
    transport = FixedDatabaseRecoveryIapTransport(owner_identity)
    probe = FixedPrivateReadOnlyProbe(
        revision,
        transport=transport,
        secret_accessor=FixedSecretManagerAccess(owner_identity),
    )
    # Prove the exact release-bound remote runtime before binding mutation
    # authority, minting a token, fetching a secret, or opening a journal.
    probe.require_available()
    if (
        _SHA256.fullmatch(expected_owner_subject_sha256 or "") is None
        or not callable(getattr(owner_identity, "bind_approved_subject", None))
    ):
        raise ProductionDatabaseRecoveryError(
            "production_database_recovery_owner_identity_invalid"
        )
    owner_identity.bind_approved_subject(expected_owner_subject_sha256)
    client = owner_transport.GoogleRestClient(owner_identity)
    return _execute_gate(
        release_revision=revision,
        provider=CloudSqlRecoveryProvider(client),
        probe=probe,
        journal_root=_journal_root(revision),
    )


def validate_receipt_for_freeze(
    value: Mapping[str, Any],
    *,
    release_revision: str,
    now_unix: int,
) -> Mapping[str, Any]:
    try:
        validated = cutover._validate_database_recovery_receipt(
            value,
            revision=_release(release_revision),
            now_unix=now_unix,
            max_recheck_age_seconds=MAX_BACKUP_RECHECK_AGE_SECONDS,
        )
    except ValueError as exc:
        raise ProductionDatabaseRecoveryError(
            "production_database_recovery_receipt_invalid"
        ) from exc
    completed = int(validated["backup"]["completed_at_unix"])
    if not 0 <= now_unix - completed <= MAX_BACKUP_AGE_SECONDS:
        raise ProductionDatabaseRecoveryError(
            "production_database_recovery_receipt_invalid"
        )
    return validated


__all__ = [
    "CloudSqlRecoveryProvider",
    "FixedDatabaseRecoveryIapTransport",
    "FixedPrivateReadOnlyProbe",
    "FixedSecretManagerAccess",
    "MAX_BACKUP_RECHECK_AGE_SECONDS",
    "ProductionDatabaseRecoveryError",
    "build_probe_receipt",
    "run_for_owner",
    "validate_receipt_for_freeze",
]

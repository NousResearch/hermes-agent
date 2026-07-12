"""Root-only, digest-bound activation for the isolated Canonical Writer canary.

This module is deliberately packaged with the runtime wheel.  It performs only
mechanical host operations described by an exact owner-reviewed plan: trusted
file reads and installs, systemd verification/start/stop, a temporary bounded
projection export, and invocation of the separately packaged root collector.
It never interprets user text, chooses a route, enables a service, invokes a
shell, reads a secret value into a receipt, or grants approval to itself.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import grp
import pwd
import ipaddress
import re
import stat
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

from gateway.canonical_writer_root_collector import (
    DEFAULT_ROOT_EVIDENCE_ROOT,
    TrustedDeploymentManifest,
)
from gateway.canonical_writer_config_collector import (
    ConfigCollectorReceipt,
    load_config_collector_receipt,
)
from gateway.canonical_writer_host_authority import (
    DEFAULT_EXTERNAL_IAM_RECEIPT_ROOT,
    DEFAULT_EXTERNAL_IAM_LIVE_PATH,
    DEFAULT_NATIVE_OBSERVATION_EVIDENCE_ROOT,
    DEFAULT_NATIVE_OBSERVATION_STAGE_ROOT,
    DEFAULT_OWNER_APPROVAL_ROOT,
    ExternalIAMReceipt,
    LEGACY_CLOUD_SQL_HELPER_PATH,
    NativeObservationPlan,
    NativeObservationReceipt,
    NATIVE_OBSERVATION_STAGE_SCHEMA,
    OWNER_APPROVAL_RECEIPT_SCHEMA,
    OwnerApprovalReceipt,
    current_host_identity_sha256,
    finalize_native_observation_stage,
    load_external_iam_receipt as load_trusted_external_iam_receipt,
    owner_approval_receipt_path,
    write_native_observation_stage,
)


ACTIVATION_PLAN_SCHEMA = "muncho-writer-only-activation-plan.v3"
ACTIVATION_RECEIPT_SCHEMA = "muncho-writer-only-activation-receipt.v1"
ACTIVATION_FAILURE_SCHEMA = "muncho-writer-only-activation-failure.v1"
OWNER_APPROVAL_SCHEMA = OWNER_APPROVAL_RECEIPT_SCHEMA
SYSTEMD_BUNDLE_SCHEMA = "muncho-writer-only-systemd-bundle.v2"
RELEASE_SCHEMA = "muncho-writer-only-release.v1"
NATIVE_OBSERVATION_SCHEMA = "muncho-writer-native-observation.v1"

WRITER_UNIT = "muncho-canonical-writer.service"
GATEWAY_UNIT = "hermes-cloud-gateway.service"
EXPORTER_UNIT = "muncho-canonical-writer-export.service"
DISCORD_UNIT = "muncho-discord-egress.service"
ROOT_COLLECTOR_MODULE = "gateway.canonical_writer_root_collector"

GATEWAY_USER = "muncho-gateway"
GATEWAY_GROUP = "muncho-gateway"
WRITER_USER = "muncho-canonical-writer"
WRITER_GROUP = "muncho-canonical-writer"
SOCKET_CLIENT_GROUP = "muncho-writer-client"
PROJECTOR_GROUP = "muncho-projector"
NOLOGIN_SHELL = "/usr/sbin/nologin"
GATEWAY_HOME = "/var/lib/hermes-gateway"
GROUPADD = "/usr/sbin/groupadd"
USERMOD = "/usr/sbin/usermod"

CANARY_GATEWAY_UID = 993
CANARY_GATEWAY_GID = 992
CANARY_WRITER_UID = 999
CANARY_WRITER_GID = 994
CANARY_SOCKET_CLIENT_GID = 990
CANARY_PROJECTOR_GID = 991
CANARY_PROJECTOR_UID = 992
PROJECTOR_USER = "muncho-projector"
ACTIVATION_LOCK_PATH = Path("/run/muncho-writer-activation.lock")

DEFAULT_PLAN_PATH = Path("/etc/muncho/writer-activation/activation-plan.json")
DEFAULT_STAGED_PLAN_PATH = Path(
    "/etc/muncho/writer-activation/staged/activation-plan.json"
)
DEFAULT_NATIVE_PLAN_PATH = Path(
    "/etc/muncho/writer-activation/native-observation-plan.json"
)
DEFAULT_STAGED_NATIVE_PLAN_PATH = Path(
    "/etc/muncho/writer-activation/staged/native-observation-plan.json"
)
DEFAULT_STAGED_OWNER_APPROVAL_PATH = Path(
    "/etc/muncho/writer-activation/staged/owner-approval.json"
)
DEFAULT_STAGED_EXTERNAL_IAM_PATH = Path(
    "/etc/muncho/writer-activation/staged/external-iam-receipt.json"
)
DEFAULT_STAGED_WRITER_UNIT_PATH = Path(
    "/etc/muncho/writer-activation/staged/muncho-canonical-writer.service"
)
DEFAULT_STAGED_GATEWAY_UNIT_PATH = Path(
    "/etc/muncho/writer-activation/staged/hermes-cloud-gateway.service"
)
DEFAULT_MANIFEST_PATH = Path("/etc/muncho/writer-activation/deployment-manifest.json")
DEFAULT_ROOT_RECEIPT_PATH = Path("/run/muncho-canonical-preflight/root-preflight.json")
DEFAULT_WRITER_UNIT_PATH = Path("/etc/systemd/system") / WRITER_UNIT
DEFAULT_GATEWAY_UNIT_PATH = Path("/etc/systemd/system") / GATEWAY_UNIT
DEFAULT_EXPORTER_UNIT_PATH = Path("/etc/systemd/system") / EXPORTER_UNIT
DEFAULT_TMPFILES_PATH = Path("/etc/tmpfiles.d/muncho-canonical-writer.conf")
DEFAULT_WRITER_CONFIG_SOURCE_PATH = Path(
    "/etc/muncho/writer-activation/staged/writer.json"
)
DEFAULT_GATEWAY_CONFIG_SOURCE_PATH = Path(
    "/etc/muncho/writer-activation/staged/gateway.yaml"
)
DEFAULT_WRITER_CONFIG_PATH = Path("/etc/muncho-canonical-writer/writer.json")
DEFAULT_GATEWAY_CONFIG_PATH = Path("/etc/hermes/config.yaml")
DEFAULT_DATABASE_CA_PATH = Path("/etc/muncho/trust/cloudsql-server-ca.pem")
DEFAULT_EVIDENCE_ROOT = Path("/var/lib/muncho-writer-activation")
DEFAULT_QUARANTINE_PATH = Path("/var/lib/muncho-writer-activation/quarantine.json")
DEFAULT_PROJECTION_PATH = Path(
    "/var/lib/muncho-canonical-writer/projection/canonical-events.json"
)
DEFAULT_WRITER_RUNTIME = Path("/run/muncho-canonical-writer")
DEFAULT_PROJECTION_DIRECTORY = DEFAULT_PROJECTION_PATH.parent
DEFAULT_GATEWAY_RUNTIME = Path("/run/hermes-cloud-gateway")
DEFAULT_GATEWAY_HOME = Path(GATEWAY_HOME)
DEFAULT_GATEWAY_LOGS = Path("/var/log/hermes-gateway")
DEFAULT_EXTERNAL_IAM_RECEIPT_PATH = DEFAULT_EXTERNAL_IAM_LIVE_PATH

SYSTEMCTL = "/usr/bin/systemctl"
SYSTEMD_ANALYZE = "/usr/bin/systemd-analyze"
SYSTEMD_TMPFILES = "/usr/bin/systemd-tmpfiles"
JOURNALCTL = "/usr/bin/journalctl"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_MAX_PLAN_BYTES = 4 * 1024 * 1024
_MAX_MANIFEST_BYTES = 8 * 1024 * 1024
_MAX_CONFIG_BYTES = 2 * 1024 * 1024
_MAX_RELEASE_FILE_BYTES = 128 * 1024 * 1024
_MAX_COMMAND_OUTPUT_BYTES = 1024 * 1024
_MAX_EXPORT_BYTES = 256 * 1024 * 1024
_COMMAND_TIMEOUT_SECONDS = 360


def _canonical_bytes(value: Any) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("activation value is not canonical JSON") from exc
    return encoded.encode("utf-8", errors="strict")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be lowercase SHA-256")
    return value


def _positive_id(value: Any, label: str) -> int:
    if type(value) is not int or not 0 < value < 1 << 31:
        raise ValueError(f"{label} must be an exact positive numeric identity")
    return value


def _strict_mapping(
    value: Any,
    *,
    fields: frozenset[str],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{label} fields are not exact")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("activation JSON contains duplicate keys")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"activation JSON contains non-JSON constant: {value}")


def _absolute_path(value: Any, label: str) -> Path:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an absolute normalized path")
    path = Path(value)
    if (
        not path.is_absolute()
        or ".." in path.parts
        or str(path) != value
        or _CONTROL_RE.search(value) is not None
    ):
        raise ValueError(f"{label} must be an absolute normalized path")
    return path


def _effective_uid() -> int:
    getter = getattr(os, "geteuid", None)
    return int(getter()) if callable(getter) else -1


def _current_boot_id_sha256() -> str:
    raw = Path("/proc/sys/kernel/random/boot_id").read_bytes()
    if len(raw) > 128:
        raise RuntimeError("activation boot identity is oversized")
    try:
        value = raw.decode("ascii", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise RuntimeError("activation boot identity is invalid") from exc
    if (
        re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            value,
        )
        is None
    ):
        raise RuntimeError("activation boot identity is invalid")
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _current_boottime_ns() -> int:
    clock = getattr(time, "CLOCK_BOOTTIME", None)
    if clock is None:
        raise RuntimeError("activation boot clock is unavailable")
    value = time.clock_gettime_ns(clock)
    if type(value) is not int or value <= 0:
        raise RuntimeError("activation boot clock is invalid")
    return value


def _require_root_linux() -> None:
    if _effective_uid() != 0:
        raise PermissionError("canonical_writer_activation_requires_uid_0")
    if sys.platform != "linux":
        raise RuntimeError("canonical_writer_activation_requires_linux")


@contextmanager
def _host_activation_lock():
    """Serialize every identity/systemd/evidence lifecycle on this host."""

    _require_root_linux()
    _validate_root_parent_chain(ACTIVATION_LOCK_PATH.parent)
    base_flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        base_flags |= os.O_NOFOLLOW
    created = False
    try:
        descriptor = os.open(
            ACTIVATION_LOCK_PATH,
            base_flags | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        created = True
    except FileExistsError:
        descriptor = os.open(ACTIVATION_LOCK_PATH, base_flags)
    try:
        if created:
            os.fchown(descriptor, 0, 0)
            os.fchmod(descriptor, 0o600)
            os.fsync(descriptor)
            _fsync_directory(ACTIVATION_LOCK_PATH.parent)
        item = os.fstat(descriptor)
        reached = os.lstat(ACTIVATION_LOCK_PATH)
        if (
            not stat.S_ISREG(item.st_mode)
            or item.st_nlink != 1
            or item.st_uid != 0
            or item.st_gid != 0
            or stat.S_IMODE(item.st_mode) != 0o600
            or (item.st_dev, item.st_ino) != (reached.st_dev, reached.st_ino)
            or _list_xattrs(ACTIVATION_LOCK_PATH)
        ):
            raise PermissionError("activation lock identity is invalid")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                "another writer activation lifecycle is running"
            ) from exc
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _validate_root_parent_chain(path: Path) -> None:
    current = path
    while True:
        try:
            item = os.lstat(current)
        except OSError as exc:
            raise ValueError("activation parent path is unavailable") from exc
        if (
            stat.S_ISLNK(item.st_mode)
            or not stat.S_ISDIR(item.st_mode)
            or item.st_uid != 0
            or item.st_gid != 0
            or stat.S_IMODE(item.st_mode) & 0o022
            or _list_xattrs(current)
        ):
            raise PermissionError("activation parent path is not root-controlled")
        if current == current.parent:
            return
        current = current.parent


def _list_xattrs(path: Path) -> tuple[str, ...]:
    lister = getattr(os, "listxattr", None)
    if not callable(lister):
        if sys.platform == "linux":
            raise RuntimeError("activation xattr inspection is unavailable")
        return ()
    try:
        return tuple(lister(path, follow_symlinks=False))
    except OSError as exc:
        raise RuntimeError("activation xattrs are unavailable") from exc


def _read_fd_bounded(
    descriptor: int,
    *,
    maximum: int,
    allow_empty: bool = False,
) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while total <= maximum:
        chunk = os.read(descriptor, min(64 * 1024, maximum + 1 - total))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
    raw = b"".join(chunks)
    if (not raw and not allow_empty) or len(raw) > maximum:
        raise ValueError("trusted activation file size is invalid")
    return raw


def _read_trusted_file(
    path_value: str | os.PathLike[str],
    *,
    expected_uid: int,
    expected_gid: int,
    allowed_modes: frozenset[int],
    maximum: int,
    trusted_parents: bool = True,
    allow_empty: bool = False,
) -> bytes:
    path = _absolute_path(os.fspath(path_value), "trusted file path")
    if trusted_parents:
        _validate_root_parent_chain(path.parent)
    try:
        before = os.lstat(path)
    except OSError as exc:
        raise ValueError("trusted activation file is unavailable") from exc
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_uid != expected_uid
        or before.st_gid != expected_gid
        or stat.S_IMODE(before.st_mode) not in allowed_modes
        or _list_xattrs(path)
    ):
        raise PermissionError("trusted activation file identity is invalid")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (
            (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
            or not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_uid != expected_uid
            or opened.st_gid != expected_gid
            or stat.S_IMODE(opened.st_mode) not in allowed_modes
        ):
            raise RuntimeError("trusted activation file changed during open")
        raw = _read_fd_bounded(
            descriptor,
            maximum=maximum,
            allow_empty=allow_empty,
        )
        after = os.fstat(descriptor)
        if (
            after.st_size != opened.st_size
            or after.st_mtime_ns != opened.st_mtime_ns
            or after.st_ctime_ns != opened.st_ctime_ns
        ):
            raise RuntimeError("trusted activation file changed during read")
        reached = os.lstat(path)
        if (
            (reached.st_dev, reached.st_ino) != (after.st_dev, after.st_ino)
            or reached.st_nlink != 1
            or reached.st_uid != expected_uid
            or reached.st_gid != expected_gid
            or stat.S_IMODE(reached.st_mode) not in allowed_modes
        ):
            raise RuntimeError("trusted activation file changed after read")
        if trusted_parents:
            _validate_root_parent_chain(path.parent)
        return raw
    finally:
        os.close(descriptor)


def _decode_strict_json(raw: bytes, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} root must be an object")
    if raw != _canonical_bytes(value) and raw != _canonical_bytes(value) + b"\n":
        raise ValueError(f"{label} is not canonical JSON")
    return value


@dataclass(frozen=True)
class NumericIdentities:
    gateway_uid: int
    gateway_gid: int
    gateway_home: str
    writer_uid: int
    writer_gid: int
    writer_home: str
    socket_client_gid: int
    projector_uid: int
    projector_gid: int
    projector_home: str

    @classmethod
    def from_mapping(cls, value: Any) -> "NumericIdentities":
        raw = _strict_mapping(
            value,
            fields=frozenset({
                "gateway_uid",
                "gateway_gid",
                "gateway_home",
                "writer_uid",
                "writer_gid",
                "writer_home",
                "socket_client_gid",
                "projector_uid",
                "projector_gid",
                "projector_home",
                "gateway_supplementary_gids",
                "writer_supplementary_gids",
            }),
            label="activation identities",
        )
        result = cls(
            gateway_uid=_positive_id(raw["gateway_uid"], "gateway_uid"),
            gateway_gid=_positive_id(raw["gateway_gid"], "gateway_gid"),
            gateway_home=raw["gateway_home"],
            writer_uid=_positive_id(raw["writer_uid"], "writer_uid"),
            writer_gid=_positive_id(raw["writer_gid"], "writer_gid"),
            writer_home=raw["writer_home"],
            socket_client_gid=_positive_id(
                raw["socket_client_gid"], "socket_client_gid"
            ),
            projector_uid=_positive_id(raw["projector_uid"], "projector_uid"),
            projector_gid=_positive_id(raw["projector_gid"], "projector_gid"),
            projector_home=raw["projector_home"],
        )
        if (
            not all(
                isinstance(path, str)
                for path in (
                    result.gateway_home,
                    result.writer_home,
                    result.projector_home,
                )
            )
            or len({result.gateway_uid, result.writer_uid, result.projector_uid}) != 3
            or len({
                result.gateway_gid,
                result.writer_gid,
                result.socket_client_gid,
                result.projector_gid,
            })
            != 4
        ):
            raise ValueError("activation identities are not isolated")
        if result != cls(
            gateway_uid=CANARY_GATEWAY_UID,
            gateway_gid=CANARY_GATEWAY_GID,
            gateway_home=GATEWAY_HOME,
            writer_uid=CANARY_WRITER_UID,
            writer_gid=CANARY_WRITER_GID,
            writer_home="/nonexistent",
            socket_client_gid=CANARY_SOCKET_CLIENT_GID,
            projector_uid=CANARY_PROJECTOR_UID,
            projector_gid=CANARY_PROJECTOR_GID,
            projector_home="/nonexistent",
        ):
            raise ValueError("activation identities are not canary-pinned")
        if raw["gateway_supplementary_gids"] != sorted((
            result.gateway_gid,
            result.socket_client_gid,
        )) or raw["writer_supplementary_gids"] != sorted((
            result.writer_gid,
            result.projector_gid,
        )):
            raise ValueError("activation supplementary groups are not exact")
        return result


@dataclass(frozen=True)
class ActivationPaths:
    plan_path: Path = DEFAULT_PLAN_PATH
    manifest_path: Path = DEFAULT_MANIFEST_PATH
    root_receipt_path: Path = DEFAULT_ROOT_RECEIPT_PATH
    writer_unit_path: Path = DEFAULT_WRITER_UNIT_PATH
    gateway_unit_path: Path = DEFAULT_GATEWAY_UNIT_PATH
    exporter_unit_path: Path = DEFAULT_EXPORTER_UNIT_PATH
    tmpfiles_path: Path = DEFAULT_TMPFILES_PATH
    writer_config_source_path: Path = DEFAULT_WRITER_CONFIG_SOURCE_PATH
    gateway_config_source_path: Path = DEFAULT_GATEWAY_CONFIG_SOURCE_PATH
    writer_config_path: Path = DEFAULT_WRITER_CONFIG_PATH
    gateway_config_path: Path = DEFAULT_GATEWAY_CONFIG_PATH
    database_ca_path: Path = DEFAULT_DATABASE_CA_PATH
    external_iam_receipt_path: Path = DEFAULT_EXTERNAL_IAM_RECEIPT_PATH
    evidence_root: Path = DEFAULT_EVIDENCE_ROOT
    quarantine_path: Path = DEFAULT_QUARANTINE_PATH
    projection_export_path: Path = DEFAULT_PROJECTION_PATH

    @classmethod
    def from_mapping(cls, value: Any) -> "ActivationPaths":
        names = frozenset({
            "plan_path",
            "manifest_path",
            "root_receipt_path",
            "writer_unit_path",
            "gateway_unit_path",
            "exporter_unit_path",
            "tmpfiles_path",
            "writer_config_source_path",
            "gateway_config_source_path",
            "writer_config_path",
            "gateway_config_path",
            "database_ca_path",
            "external_iam_receipt_path",
            "evidence_root",
            "quarantine_path",
            "projection_export_path",
        })
        raw = _strict_mapping(value, fields=names, label="activation paths")
        result = cls(**{
            name: _absolute_path(raw[name], f"activation {name}") for name in names
        })
        pinned = {
            "plan_path": DEFAULT_PLAN_PATH,
            "manifest_path": DEFAULT_MANIFEST_PATH,
            "root_receipt_path": DEFAULT_ROOT_RECEIPT_PATH,
            "writer_unit_path": DEFAULT_WRITER_UNIT_PATH,
            "gateway_unit_path": DEFAULT_GATEWAY_UNIT_PATH,
            "exporter_unit_path": DEFAULT_EXPORTER_UNIT_PATH,
            "tmpfiles_path": DEFAULT_TMPFILES_PATH,
            "writer_config_source_path": DEFAULT_WRITER_CONFIG_SOURCE_PATH,
            "gateway_config_source_path": DEFAULT_GATEWAY_CONFIG_SOURCE_PATH,
            "writer_config_path": DEFAULT_WRITER_CONFIG_PATH,
            "gateway_config_path": DEFAULT_GATEWAY_CONFIG_PATH,
            "database_ca_path": DEFAULT_DATABASE_CA_PATH,
            "external_iam_receipt_path": DEFAULT_EXTERNAL_IAM_RECEIPT_PATH,
            "evidence_root": DEFAULT_EVIDENCE_ROOT,
            "quarantine_path": DEFAULT_QUARANTINE_PATH,
            "projection_export_path": DEFAULT_PROJECTION_PATH,
        }
        for name, expected in pinned.items():
            if getattr(result, name) != expected:
                raise ValueError(f"activation {name} is not production-pinned")
        all_paths = list(result.__dict__.values())
        if len(all_paths) != len(set(all_paths)):
            raise ValueError("activation paths must be distinct")
        return result

    def to_mapping(self) -> dict[str, str]:
        return {name: str(value) for name, value in self.__dict__.items()}


@dataclass(frozen=True)
class ActivationDigests:
    native_observation_plan_sha256: str
    native_observation_receipt_sha256: str
    release_manifest_file_sha256: str
    database_ca_sha256: str
    external_iam_policy_sha256: str
    deployment_manifest_sha256: str
    writer_unit_sha256: str
    gateway_unit_sha256: str
    exporter_unit_sha256: str
    tmpfiles_sha256: str
    writer_config_sha256: str
    gateway_config_sha256: str

    @classmethod
    def from_mapping(cls, value: Any) -> "ActivationDigests":
        names = frozenset(cls.__dataclass_fields__)
        raw = _strict_mapping(value, fields=names, label="activation digests")
        return cls(**{name: _digest(raw[name], name) for name in names})

    def to_mapping(self) -> dict[str, str]:
        return dict(self.__dict__)


def load_owner_approval_receipt(
    path_value: str | os.PathLike[str],
    *,
    scope: str,
    plan_sha256: str,
) -> OwnerApprovalReceipt:
    _require_root_linux()
    _digest(plan_sha256, "owner approval path plan sha256")
    if scope not in {"native_observation", "activation"}:
        raise ValueError("owner approval path scope is invalid")
    path = _absolute_path(os.fspath(path_value), "owner approval receipt path")
    raw = _read_trusted_file(
        path,
        expected_uid=0,
        expected_gid=0,
        allowed_modes=frozenset({0o400}),
        maximum=64 * 1024,
    )
    receipt = OwnerApprovalReceipt.from_mapping(
        _decode_strict_json(raw, label="owner approval receipt")
    )
    if path != owner_approval_receipt_path(receipt):
        raise ValueError("owner approval receipt path is not receipt-addressed")
    receipt.require(
        scope=scope,
        plan_sha256=plan_sha256,
        now_unix=int(time.time()),
    )
    return receipt


def install_staged_owner_approval(
    path_value: str | os.PathLike[str],
) -> tuple[OwnerApprovalReceipt, Path]:
    """Install one fresh approval into its immutable receipt-addressed path."""

    _require_root_linux()
    source = _absolute_path(os.fspath(path_value), "staged owner approval path")
    if source != DEFAULT_STAGED_OWNER_APPROVAL_PATH:
        raise ValueError("staged owner approval path is not production-pinned")
    raw = _read_trusted_file(
        source,
        expected_uid=0,
        expected_gid=0,
        allowed_modes=frozenset({0o400}),
        maximum=64 * 1024,
    )
    receipt = OwnerApprovalReceipt.from_mapping(
        _decode_strict_json(raw, label="staged owner approval receipt")
    )
    receipt.require(
        scope=str(receipt.value["scope"]),
        plan_sha256=str(receipt.value["plan_sha256"]),
        now_unix=int(time.time()),
    )
    canonical = _canonical_bytes(receipt.to_mapping())
    if raw not in {canonical, canonical + b"\n"}:
        raise ValueError("staged owner approval changed during validation")
    target = owner_approval_receipt_path(receipt)
    if DEFAULT_OWNER_APPROVAL_ROOT not in target.parents:
        raise ValueError("owner approval target escaped the approval root")
    _ensure_root_directory(target.parent)
    _install_exact_bytes(target, canonical, uid=0, gid=0, mode=0o400)
    return receipt, target


@dataclass(frozen=True)
class InstallArtifact:
    source_path: Path | None
    target_path: Path
    sha256: str
    mode: int
    uid: int
    gid: int
    maximum_bytes: int

    @classmethod
    def from_mapping(cls, value: Any, *, label: str) -> "InstallArtifact":
        raw = _strict_mapping(
            value,
            fields=frozenset({
                "source_path",
                "target_path",
                "sha256",
                "mode",
                "uid",
                "gid",
                "maximum_bytes",
            }),
            label=f"{label} artifact",
        )
        source_raw = raw["source_path"]
        source = (
            None
            if source_raw is None
            else _absolute_path(source_raw, f"{label} source path")
        )
        mode_raw = raw["mode"]
        if (
            not isinstance(mode_raw, str)
            or re.fullmatch(r"0[0-7]{3}", mode_raw) is None
        ):
            raise ValueError(f"{label} mode is invalid")
        mode = int(mode_raw, 8)
        maximum = raw["maximum_bytes"]
        if type(maximum) is not int or not 0 < maximum <= _MAX_MANIFEST_BYTES:
            raise ValueError(f"{label} maximum size is invalid")
        uid = raw["uid"]
        gid = raw["gid"]
        if type(uid) is not int or type(gid) is not int or uid < 0 or gid < 0:
            raise ValueError(f"{label} ownership is invalid")
        return cls(
            source_path=source,
            target_path=_absolute_path(raw["target_path"], f"{label} target path"),
            sha256=_digest(raw["sha256"], f"{label} sha256"),
            mode=mode,
            uid=uid,
            gid=gid,
            maximum_bytes=maximum,
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "source_path": None if self.source_path is None else str(self.source_path),
            "target_path": str(self.target_path),
            "sha256": self.sha256,
            "mode": f"{self.mode:04o}",
            "uid": self.uid,
            "gid": self.gid,
            "maximum_bytes": self.maximum_bytes,
        }


@dataclass(frozen=True)
class SystemdBundle:
    writer_service: str
    gateway_service: str
    exporter_service: str
    tmpfiles: str
    contract: Mapping[str, str]
    sha256: str
    schema: str = SYSTEMD_BUNDLE_SCHEMA

    @classmethod
    def from_mapping(cls, value: Any) -> "SystemdBundle":
        raw = _strict_mapping(
            value,
            fields=frozenset({
                "schema",
                "writer_service",
                "gateway_service",
                "exporter_service",
                "tmpfiles",
                "contract",
                "sha256",
            }),
            label="systemd bundle",
        )
        if raw["schema"] != SYSTEMD_BUNDLE_SCHEMA:
            raise ValueError("systemd bundle schema is invalid")
        for name in (
            "writer_service",
            "gateway_service",
            "exporter_service",
            "tmpfiles",
        ):
            text = raw[name]
            if (
                not isinstance(text, str)
                or not text.endswith("\n")
                or "\x00" in text
                or len(text.encode("utf-8")) > 256 * 1024
            ):
                raise ValueError(f"systemd bundle {name} is invalid")
        contract = raw["contract"]
        if (
            not isinstance(contract, Mapping)
            or not contract
            or any(
                not isinstance(key, str) or not isinstance(item, str)
                for key, item in contract.items()
            )
        ):
            raise ValueError("systemd bundle contract is invalid")
        unsigned = {name: copy.deepcopy(raw[name]) for name in raw if name != "sha256"}
        digest = _digest(raw["sha256"], "systemd bundle sha256")
        if _sha256_json(unsigned) != digest:
            raise ValueError("systemd bundle self-digest is invalid")
        forbidden = re.compile(
            r"(?im)^(?:EnvironmentFile|PassEnvironment|LoadCredential)="
        )
        if any(
            forbidden.search(raw[name])
            for name in (
                "writer_service",
                "gateway_service",
                "exporter_service",
            )
        ):
            raise ValueError("systemd bundle contains credential injection")
        if (
            "[Install]" in raw["exporter_service"]
            or ".timer" in raw["exporter_service"]
        ):
            raise ValueError("temporary exporter cannot be installable or scheduled")
        return cls(
            writer_service=raw["writer_service"],
            gateway_service=raw["gateway_service"],
            exporter_service=raw["exporter_service"],
            tmpfiles=raw["tmpfiles"],
            contract=copy.deepcopy(dict(contract)),
            sha256=digest,
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "writer_service": self.writer_service,
            "gateway_service": self.gateway_service,
            "exporter_service": self.exporter_service,
            "tmpfiles": self.tmpfiles,
            "contract": copy.deepcopy(dict(self.contract)),
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class ActivationPlan:
    revision: str
    identities: NumericIdentities
    paths: ActivationPaths
    digests: ActivationDigests
    deployment_manifest: Mapping[str, Any]
    native_observation_receipt: Mapping[str, Any]
    unit_bundle: SystemdBundle
    install_artifacts: Mapping[str, InstallArtifact]
    collector_argv: tuple[str, ...]
    validator_argv: tuple[str, ...]
    sha256: str
    schema: str = ACTIVATION_PLAN_SCHEMA

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ActivationPlan":
        fields = frozenset({
            "schema",
            "revision",
            "identities",
            "paths",
            "digests",
            "deployment_manifest",
            "native_observation_receipt",
            "systemd_bundle",
            "install_artifacts",
            "collector_argv",
            "validator_argv",
            "activation_plan_sha256",
        })
        raw = _strict_mapping(value, fields=fields, label="activation plan")
        if raw["schema"] != ACTIVATION_PLAN_SCHEMA:
            raise ValueError("activation plan schema is invalid")
        revision = raw["revision"]
        if not isinstance(revision, str) or _REVISION_RE.fullmatch(revision) is None:
            raise ValueError("activation revision is invalid")
        identities = NumericIdentities.from_mapping(raw["identities"])
        paths = ActivationPaths.from_mapping(raw["paths"])
        digests = ActivationDigests.from_mapping(raw["digests"])
        trusted = TrustedDeploymentManifest.from_mapping(raw["deployment_manifest"])
        if trusted.revision != revision:
            raise ValueError("activation manifest revision drifted")
        native_raw = raw["native_observation_receipt"]
        if not isinstance(native_raw, Mapping):
            raise ValueError("native observation receipt must be an object")
        native = NativeObservationReceipt.from_mapping(
            native_raw,
            expected_plan_sha256=digests.native_observation_plan_sha256,
        )
        if native.sha256 != digests.native_observation_receipt_sha256:
            raise ValueError("native observation receipt digest is invalid")
        bundle = SystemdBundle.from_mapping(raw["systemd_bundle"])
        artifact_raw = _strict_mapping(
            raw["install_artifacts"],
            fields=frozenset({
                "manifest",
                "writer_unit",
                "gateway_unit",
                "tmpfiles",
                "writer_config",
                "gateway_config",
            }),
            label="install artifacts",
        )
        artifacts = {
            name: InstallArtifact.from_mapping(item, label=name)
            for name, item in artifact_raw.items()
        }
        collector = _argv(raw["collector_argv"], "collector argv")
        validator = _argv(raw["validator_argv"], "validator argv")
        unsigned = _plan_digest_projection(raw)
        digest = _digest(raw["activation_plan_sha256"], "activation plan sha256")
        if _sha256_json(unsigned) != digest:
            raise ValueError("activation plan self-digest is invalid")
        result = cls(
            revision=revision,
            identities=identities,
            paths=paths,
            digests=digests,
            deployment_manifest=trusted.to_mapping(),
            native_observation_receipt=native.to_mapping(),
            unit_bundle=bundle,
            install_artifacts=artifacts,
            collector_argv=collector,
            validator_argv=validator,
            sha256=digest,
        )
        result._validate_bindings()
        return result

    def _validate_bindings(self) -> None:
        interpreter = self.deployment_manifest["snapshot_template"][
            "writer_deployment"
        ]["policy"]["interpreter"]
        expected_collector = (
            interpreter,
            "-I",
            "-m",
            ROOT_COLLECTOR_MODULE,
            "collect",
            "--manifest",
            str(self.paths.manifest_path),
            "--receipt",
            str(self.paths.root_receipt_path),
        )
        expected_validator = (
            *expected_collector[:4],
            "validate",
            *expected_collector[5:],
        )
        if (
            self.collector_argv != expected_collector
            or self.validator_argv != expected_validator
        ):
            raise ValueError("activation collector argv is not exact")
        expected = {
            "manifest": (
                self.paths.manifest_path,
                self.digests.deployment_manifest_sha256,
                0,
                0,
                0o400,
            ),
            "writer_unit": (
                self.paths.writer_unit_path,
                self.digests.writer_unit_sha256,
                0,
                0,
                0o644,
            ),
            "gateway_unit": (
                self.paths.gateway_unit_path,
                self.digests.gateway_unit_sha256,
                0,
                0,
                0o644,
            ),
            "tmpfiles": (
                self.paths.tmpfiles_path,
                self.digests.tmpfiles_sha256,
                0,
                0,
                0o644,
            ),
            "writer_config": (
                self.paths.writer_config_path,
                self.digests.writer_config_sha256,
                0,
                self.identities.writer_gid,
                0o440,
            ),
            "gateway_config": (
                self.paths.gateway_config_path,
                self.digests.gateway_config_sha256,
                0,
                0,
                0o444,
            ),
        }
        for name, (target, digest, uid, gid, mode) in expected.items():
            artifact = self.install_artifacts[name]
            if (
                artifact.target_path != target
                or artifact.sha256 != digest
                or artifact.uid != uid
                or artifact.gid != gid
                or artifact.mode != mode
            ):
                raise ValueError(f"activation {name} artifact binding drifted")
        if self.install_artifacts["manifest"].source_path is not None:
            raise ValueError(
                "deployment manifest must come from the digest-bound approved plan"
            )
        source_bindings = {
            "writer_config": self.paths.writer_config_source_path,
            "gateway_config": self.paths.gateway_config_source_path,
        }
        for name, source in source_bindings.items():
            if self.install_artifacts[name].source_path != source:
                raise ValueError(f"activation {name} staging path drifted")
        for name in ("writer_unit", "gateway_unit", "tmpfiles"):
            if self.install_artifacts[name].source_path is not None:
                raise ValueError(
                    f"activation {name} must come from the digest-bound approved plan"
                )
        bundle_values = {
            "writer_unit": self.unit_bundle.writer_service.encode("utf-8"),
            "gateway_unit": self.unit_bundle.gateway_service.encode("utf-8"),
            "tmpfiles": self.unit_bundle.tmpfiles.encode("utf-8"),
        }
        for name, payload in bundle_values.items():
            if _sha256_bytes(payload) != self.install_artifacts[name].sha256:
                raise ValueError(f"activation {name} content digest drifted")
        if _sha256_bytes(self.unit_bundle.exporter_service.encode("utf-8")) != (
            self.digests.exporter_unit_sha256
        ):
            raise ValueError("activation exporter unit digest drifted")
        if _sha256_bytes(_canonical_bytes(self.deployment_manifest)) != (
            self.install_artifacts["manifest"].sha256
        ):
            raise ValueError("activation deployment manifest digest drifted")
        native = NativeObservationReceipt.from_mapping(
            self.native_observation_receipt,
            expected_plan_sha256=self.digests.native_observation_plan_sha256,
        )
        native_plan = NativeObservationPlan.from_mapping(native.value["plan"])
        snapshot = self.deployment_manifest["snapshot_template"]
        observed = native.value["observation"]
        observed_native = {
            "writer": observed["writer_service"]["external_native_mappings"],
            "gateway": observed["gateway_service"]["external_native_mappings"],
        }
        expected_native = {
            "writer": snapshot["writer_deployment"]["policy"][
                "preapproved_external_native_executable_mappings"
            ],
            "gateway": snapshot["gateway_deployment"]["policy"][
                "preapproved_external_native_executable_mappings"
            ],
        }
        if observed_native != expected_native:
            raise ValueError("final native mapping policy differs from stopped receipt")
        observed_kernel = {
            "writer": observed["writer_service"]["kernel_executable_mappings"],
            "gateway": observed["gateway_service"]["kernel_executable_mappings"],
        }
        expected_kernel = {
            "writer": snapshot["writer_deployment"]["policy"].get(
                "preapproved_kernel_executable_mappings"
            ),
            "gateway": snapshot["gateway_deployment"]["policy"].get(
                "preapproved_kernel_executable_mappings"
            ),
        }
        if observed_kernel != expected_kernel:
            raise ValueError("final kernel mapping policy differs from stopped receipt")
        writer_policy = snapshot["writer_deployment"]["policy"]
        gateway_policy = snapshot["gateway_deployment"]["policy"]
        connection = snapshot["database"]["connection"]
        native_identities = native_plan.value["identities"]
        expected_identities = {
            "gateway_uid": self.identities.gateway_uid,
            "gateway_gid": self.identities.gateway_gid,
            "gateway_home": self.identities.gateway_home,
            "gateway_supplementary_gids": sorted((
                self.identities.gateway_gid,
                self.identities.socket_client_gid,
            )),
            "writer_uid": self.identities.writer_uid,
            "writer_gid": self.identities.writer_gid,
            "writer_home": self.identities.writer_home,
            "writer_supplementary_gids": sorted((
                self.identities.writer_gid,
                self.identities.projector_gid,
            )),
            "socket_group_gid": self.identities.socket_client_gid,
            "projector_uid": self.identities.projector_uid,
            "projector_gid": self.identities.projector_gid,
            "projector_home": self.identities.projector_home,
        }
        cross_bindings = (
            native_plan.value["revision"] == self.revision,
            native_plan.value["artifact_root"] == writer_policy["artifact_root"],
            native_plan.value["artifact_root"] == gateway_policy["artifact_root"],
            native_plan.value["artifact_sha256"]
            == writer_policy["artifact_digest_sha256"],
            native_plan.value["artifact_sha256"]
            == gateway_policy["artifact_digest_sha256"],
            native_plan.value["release_manifest_file_sha256"]
            == self.digests.release_manifest_file_sha256,
            native_plan.value["writer_unit"]
            == {
                "name": WRITER_UNIT,
                "path": str(self.paths.writer_unit_path),
                "sha256": self.digests.writer_unit_sha256,
            },
            native_plan.value["gateway_unit"]
            == {
                "name": GATEWAY_UNIT,
                "path": str(self.paths.gateway_unit_path),
                "sha256": self.digests.gateway_unit_sha256,
            },
            native_plan.value["writer_argv"] == writer_policy["exec_start"],
            native_plan.value["gateway_argv"] == gateway_policy["exec_start"],
            native_plan.value["writer_config"]
            == {
                "path": str(self.paths.writer_config_path),
                "sha256": self.digests.writer_config_sha256,
            },
            native_plan.value["gateway_config"]
            == {
                "path": str(self.paths.gateway_config_path),
                "sha256": self.digests.gateway_config_sha256,
            },
            native_identities == expected_identities,
            native_plan.value["database"]
            == {
                "ip_network": f"{connection['host']}/32",
                "tls_server_name": connection["tls_server_name"],
                "ca_path": str(self.paths.database_ca_path),
                "ca_sha256": self.digests.database_ca_sha256,
            },
            native_plan.value["discord"]
            == {
                "unit_name": DISCORD_UNIT,
                "config_path": snapshot["discord_edge"]["config_path"],
                "token_path": snapshot["discord_edge"]["token_path"],
                "socket_path": snapshot["discord_edge"]["socket_path"],
                "required_absent": True,
            },
            native_plan.value["legacy_helper_path"]
            == str(LEGACY_CLOUD_SQL_HELPER_PATH),
            native_plan.value["external_iam_policy_sha256"]
            == self.digests.external_iam_policy_sha256,
        )
        if not all(cross_bindings):
            raise ValueError("native observation plan does not match final activation")
        host = self.deployment_manifest["host_contract"]
        required_host = {
            "writer_unit_fragment_path": str(self.paths.writer_unit_path),
            "writer_unit_fragment_sha256": self.digests.writer_unit_sha256,
            "gateway_unit_fragment_path": str(self.paths.gateway_unit_path),
            "gateway_unit_fragment_sha256": self.digests.gateway_unit_sha256,
            "writer_config_sha256": self.digests.writer_config_sha256,
            "writer_config_path": str(self.paths.writer_config_path),
            "gateway_config_path": str(self.paths.gateway_config_path),
            "gateway_config_sha256": self.digests.gateway_config_sha256,
            "projection_export_path": str(self.paths.projection_export_path),
            "native_observation_receipt_sha256": native.sha256,
            "native_observation_plan_sha256": native_plan.sha256,
            "native_observation_receipt_path": str(_native_receipt_path(native_plan)),
            "legacy_helper_path": str(LEGACY_CLOUD_SQL_HELPER_PATH),
            "external_iam_receipt_path": str(self.paths.external_iam_receipt_path),
            "external_iam_policy_sha256": (self.digests.external_iam_policy_sha256),
        }
        for name, expected_value in required_host.items():
            if host.get(name) != expected_value:
                raise ValueError(f"activation host contract {name} drifted")

    def unsigned_mapping(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "revision": self.revision,
            "identities": {
                **self.identities.__dict__,
                "gateway_supplementary_gids": sorted((
                    self.identities.gateway_gid,
                    self.identities.socket_client_gid,
                )),
                "writer_supplementary_gids": sorted((
                    self.identities.writer_gid,
                    self.identities.projector_gid,
                )),
            },
            "paths": self.paths.to_mapping(),
            "digests": self.digests.to_mapping(),
            "deployment_manifest": copy.deepcopy(dict(self.deployment_manifest)),
            "native_observation_receipt": copy.deepcopy(
                dict(self.native_observation_receipt)
            ),
            "systemd_bundle": self.unit_bundle.to_mapping(),
            "install_artifacts": {
                name: artifact.to_mapping()
                for name, artifact in sorted(self.install_artifacts.items())
            },
            "collector_argv": list(self.collector_argv),
            "validator_argv": list(self.validator_argv),
        }

    def to_mapping(self) -> dict[str, Any]:
        return {**self.unsigned_mapping(), "activation_plan_sha256": self.sha256}


def _argv(value: Any, label: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(
            not isinstance(item, str)
            or not item
            or _CONTROL_RE.search(item) is not None
            for item in value
        )
        or value[0] in {"sh", "bash", "/bin/sh", "/bin/bash"}
    ):
        raise ValueError(f"{label} is invalid")
    return tuple(value)


def _plan_digest_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return every plan byte except the independent self-digest field."""

    return {
        name: copy.deepcopy(item)
        for name, item in value.items()
        if name != "activation_plan_sha256"
    }


def load_activation_plan(path_value: str | os.PathLike[str]) -> ActivationPlan:
    _require_root_linux()
    path = _absolute_path(os.fspath(path_value), "activation plan path")
    if path != DEFAULT_PLAN_PATH:
        raise ValueError("activation plan path is not production-pinned")
    raw = _read_trusted_file(
        path,
        expected_uid=0,
        expected_gid=0,
        allowed_modes=frozenset({0o400}),
        maximum=_MAX_PLAN_BYTES,
    )
    plan = ActivationPlan.from_mapping(
        _decode_strict_json(raw, label="activation plan")
    )
    if plan.paths.plan_path != path:
        raise ValueError("activation plan internal path drifted")
    return plan


def install_staged_activation_plan(
    path_value: str | os.PathLike[str],
) -> ActivationPlan:
    """Atomically install the one production-pinned staged owner plan."""

    _require_root_linux()
    source = _absolute_path(os.fspath(path_value), "staged activation plan path")
    if source != DEFAULT_STAGED_PLAN_PATH:
        raise ValueError("staged activation plan path is not production-pinned")
    raw = _read_trusted_file(
        source,
        expected_uid=0,
        expected_gid=0,
        allowed_modes=frozenset({0o400}),
        maximum=_MAX_PLAN_BYTES,
    )
    plan = ActivationPlan.from_mapping(
        _decode_strict_json(raw, label="staged activation plan")
    )
    canonical = _canonical_bytes(plan.to_mapping())
    if raw not in {canonical, canonical + b"\n"}:
        raise ValueError("staged activation plan changed during validation")
    _install_exact_bytes(DEFAULT_PLAN_PATH, canonical, uid=0, gid=0, mode=0o400)
    return plan


def load_native_observation_plan(
    path_value: str | os.PathLike[str],
) -> NativeObservationPlan:
    _require_root_linux()
    path = _absolute_path(os.fspath(path_value), "native observation plan path")
    if path != DEFAULT_NATIVE_PLAN_PATH:
        raise ValueError("native observation plan path is not production-pinned")
    raw = _read_trusted_file(
        path,
        expected_uid=0,
        expected_gid=0,
        allowed_modes=frozenset({0o400}),
        maximum=_MAX_PLAN_BYTES,
    )
    return NativeObservationPlan.from_mapping(
        _decode_strict_json(raw, label="native observation plan")
    )


def install_staged_native_observation_plan(
    path_value: str | os.PathLike[str],
) -> NativeObservationPlan:
    _require_root_linux()
    source = _absolute_path(
        os.fspath(path_value),
        "staged native observation plan path",
    )
    if source != DEFAULT_STAGED_NATIVE_PLAN_PATH:
        raise ValueError("staged native plan path is not production-pinned")
    raw = _read_trusted_file(
        source,
        expected_uid=0,
        expected_gid=0,
        allowed_modes=frozenset({0o400}),
        maximum=_MAX_PLAN_BYTES,
    )
    plan = NativeObservationPlan.from_mapping(
        _decode_strict_json(raw, label="staged native observation plan")
    )
    canonical = _canonical_bytes(plan.to_mapping())
    if raw not in {canonical, canonical + b"\n"}:
        raise ValueError("staged native observation plan changed during validation")
    _install_exact_bytes(
        DEFAULT_NATIVE_PLAN_PATH,
        canonical,
        uid=0,
        gid=0,
        mode=0o400,
    )
    return plan


def load_external_iam_receipt(
    plan: NativeObservationPlan,
    *,
    path_value: str | os.PathLike[str],
    now_unix: int | None = None,
    minimum_remaining_seconds: int = 0,
    expected_source_approval_sha256: str | None = None,
) -> ExternalIAMReceipt:
    if not isinstance(plan, NativeObservationPlan):
        raise TypeError("native observation plan is required for IAM evidence")
    _require_root_linux()
    path = _absolute_path(os.fspath(path_value), "external IAM receipt path")
    if path != DEFAULT_EXTERNAL_IAM_LIVE_PATH:
        raise ValueError("external IAM live receipt path is not pinned")
    current = int(time.time()) if now_unix is None else now_unix
    receipt = load_trusted_external_iam_receipt(
        path,
        expected_policy_sha256=plan.value["external_iam_policy_sha256"],
        now_unix=current,
    )
    receipt.require_fresh(
        current,
        minimum_remaining_seconds=minimum_remaining_seconds,
    )
    if expected_source_approval_sha256 is not None and (
        receipt.value.get("source_approval_sha256")
        != _digest(
            expected_source_approval_sha256,
            "external IAM source approval digest",
        )
    ):
        raise ValueError("external IAM receipt is bound to another approval")
    return receipt


class RetryableExternalIAMRefreshRequired(RuntimeError):
    pass


class RetryableOwnerApprovalRefreshRequired(RuntimeError):
    pass


def _require_lifecycle_owner_approval(
    receipt: OwnerApprovalReceipt,
    *,
    scope: str,
    plan_sha256: str,
) -> None:
    try:
        receipt.require(
            scope=scope,
            plan_sha256=plan_sha256,
            now_unix=int(time.time()),
        )
    except PermissionError as exc:
        raise RetryableOwnerApprovalRefreshRequired(
            "fresh owner approval is required before service mutation"
        ) from exc


def _load_lifecycle_external_iam(
    plan: NativeObservationPlan,
    *,
    path_value: str | os.PathLike[str],
    minimum_remaining_seconds: int,
    expected_source_approval_sha256: str | None = None,
) -> ExternalIAMReceipt:
    try:
        return load_external_iam_receipt(
            plan,
            path_value=path_value,
            minimum_remaining_seconds=minimum_remaining_seconds,
            expected_source_approval_sha256=expected_source_approval_sha256,
        )
    except ValueError as exc:
        raise RetryableExternalIAMRefreshRequired(
            "fresh external IAM evidence is required before service mutation"
        ) from exc


@dataclass(frozen=True)
class Command:
    argv: tuple[str, ...]
    timeout_seconds: int = _COMMAND_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        _argv(list(self.argv), "activation command argv")
        if (
            type(self.timeout_seconds) is not int
            or not 1 <= self.timeout_seconds <= 600
        ):
            raise ValueError("activation command timeout is invalid")

    @property
    def environment(self) -> dict[str, str]:
        return {
            "HOME": "/nonexistent",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/sbin:/usr/bin:/sbin:/bin",
            "TZ": "UTC",
        }


Runner = Callable[[Command], subprocess.CompletedProcess[bytes]]


def _runner(command: Command) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        list(command.argv),
        env=command.environment,
        cwd="/",
        check=False,
        shell=False,
        capture_output=True,
        timeout=command.timeout_seconds,
    )


def _run(
    command: Command,
    *,
    runner: Runner,
    label: str,
    accepted: frozenset[int] = frozenset({0}),
) -> subprocess.CompletedProcess[bytes]:
    result = runner(command)
    if not isinstance(result, subprocess.CompletedProcess):
        raise TypeError("activation runner returned an invalid result")
    stdout = result.stdout if isinstance(result.stdout, bytes) else b""
    stderr = result.stderr if isinstance(result.stderr, bytes) else b""
    if (
        len(stdout) > _MAX_COMMAND_OUTPUT_BYTES
        or len(stderr) > _MAX_COMMAND_OUTPUT_BYTES
    ):
        raise RuntimeError(f"{label} output exceeds its bound")
    if result.returncode not in accepted:
        raise RuntimeError(
            f"{label} failed: rc={result.returncode} "
            f"stdout_sha256={_sha256_bytes(stdout)} "
            f"stderr_sha256={_sha256_bytes(stderr)}"
        )
    return result


def _systemd_show(unit: str, *, runner: Runner) -> Mapping[str, str]:
    result = _run(
        Command(
            (
                SYSTEMCTL,
                "show",
                unit,
                (
                    "--property=LoadState,ActiveState,SubState,MainPID,"
                    "UnitFileState,FragmentPath,DropInPaths,NeedDaemonReload"
                ),
                "--no-pager",
            ),
            timeout_seconds=30,
        ),
        runner=runner,
        label=f"systemd state for {unit}",
    )
    try:
        lines = result.stdout.decode("utf-8", errors="strict").splitlines()
    except UnicodeDecodeError as exc:
        raise RuntimeError("systemd state output is not UTF-8") from exc
    state: dict[str, str] = {}
    for line in lines:
        key, separator, value = line.partition("=")
        if separator != "=" or key in state:
            raise RuntimeError("systemd state output is ambiguous")
        state[key] = value
    if (
        set(state)
        != {
            "LoadState",
            "ActiveState",
            "SubState",
            "MainPID",
            "UnitFileState",
            "FragmentPath",
            "DropInPaths",
            "NeedDaemonReload",
        }
        or not state["MainPID"].isdigit()
    ):
        raise RuntimeError("systemd state fields are not exact")
    return state


def _off_state_is_exact(
    unit: str,
    state: Mapping[str, str],
    *,
    absent: bool,
) -> bool:
    expected_load = "not-found" if absent else "loaded"
    expected_file = "" if absent else "disabled"
    unit_paths = {
        WRITER_UNIT: str(DEFAULT_WRITER_UNIT_PATH),
        GATEWAY_UNIT: str(DEFAULT_GATEWAY_UNIT_PATH),
        EXPORTER_UNIT: str(DEFAULT_EXPORTER_UNIT_PATH),
    }
    expected_fragment = "" if absent else unit_paths.get(unit, "")
    return bool(
        state["LoadState"] == expected_load
        and state["ActiveState"] == "inactive"
        and state["SubState"] == "dead"
        and state["MainPID"] == "0"
        and state["UnitFileState"] == expected_file
        and state["FragmentPath"] == expected_fragment
        and state["DropInPaths"] == ""
        and state["NeedDaemonReload"] == "no"
    )


def _require_off_disabled(unit: str, *, runner: Runner, absent: bool = False) -> None:
    state = _systemd_show(unit, runner=runner)
    if not _off_state_is_exact(unit, state, absent=absent):
        raise RuntimeError(f"{unit} is not exact stopped/disabled state")


def _require_off_or_absent(unit: str, *, runner: Runner) -> None:
    state = _systemd_show(unit, runner=runner)
    if not (
        _off_state_is_exact(unit, state, absent=False)
        or _off_state_is_exact(unit, state, absent=True)
    ):
        raise RuntimeError(f"{unit} is not exact stopped/disabled or absent state")


def _require_active(unit: str, *, runner: Runner) -> int:
    state = _systemd_show(unit, runner=runner)
    if (
        state["LoadState"] != "loaded"
        or state["ActiveState"] != "active"
        or state["SubState"] != "running"
        or state["UnitFileState"] != "disabled"
        or state["MainPID"] == "0"
        or state["FragmentPath"]
        != {
            WRITER_UNIT: str(DEFAULT_WRITER_UNIT_PATH),
            GATEWAY_UNIT: str(DEFAULT_GATEWAY_UNIT_PATH),
        }.get(unit, "")
        or state["DropInPaths"] != ""
        or state["NeedDaemonReload"] != "no"
    ):
        raise RuntimeError(f"{unit} did not reach exact active state")
    return int(state["MainPID"])


def _systemd_invocation_id(unit: str, *, runner: Runner) -> str:
    result = _run(
        Command(
            (
                SYSTEMCTL,
                "show",
                unit,
                "--property=InvocationID",
                "--value",
                "--no-pager",
            ),
            timeout_seconds=30,
        ),
        runner=runner,
        label=f"systemd invocation identity for {unit}",
    )
    try:
        value = result.stdout.decode("ascii", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise RuntimeError("systemd invocation identity is not ASCII") from exc
    if re.fullmatch(r"[0-9a-f]{32}", value) is None:
        raise RuntimeError("systemd invocation identity is invalid")
    return value


def _exporter_stdout_receipt(
    invocation_id: str,
    *,
    runner: Runner,
) -> Mapping[str, Any]:
    if re.fullmatch(r"[0-9a-f]{32}", invocation_id) is None:
        raise ValueError("exporter invocation identity is invalid")
    _run(
        Command((JOURNALCTL, "--sync"), timeout_seconds=30),
        runner=runner,
        label="journal synchronization",
    )
    result = _run(
        Command(
            (
                JOURNALCTL,
                "--no-pager",
                "--output=cat",
                f"_SYSTEMD_INVOCATION_ID={invocation_id}",
                "PRIORITY=6",
            ),
            timeout_seconds=30,
        ),
        runner=runner,
        label="projection exporter stdout collection",
    )
    lines = [line for line in result.stdout.splitlines() if line]
    if len(lines) != 1:
        raise RuntimeError("projection exporter emitted ambiguous stdout")
    value = _decode_strict_json(lines[0], label="projection exporter stdout")
    if (
        set(value) != {"event_count", "success"}
        or value.get("success") is not True
        or type(value.get("event_count")) is not int
        or not 0 <= value["event_count"] <= 1_000_000
    ):
        raise RuntimeError("projection exporter stdout receipt is invalid")
    return value


def _lookup_group(name: str) -> grp.struct_group | None:
    try:
        return grp.getgrnam(name)
    except KeyError:
        return None


def _lookup_gid(gid: int) -> grp.struct_group | None:
    try:
        return grp.getgrgid(gid)
    except KeyError:
        return None


def _lookup_user(name: str) -> pwd.struct_passwd:
    try:
        return pwd.getpwnam(name)
    except KeyError as exc:
        raise RuntimeError(f"required host user is absent: {name}") from exc


def _supplementary_gids(name: str, primary_gid: int) -> tuple[int, ...]:
    values = os.getgrouplist(name, primary_gid)
    if any(type(value) is not int or value < 0 for value in values):
        raise RuntimeError("host group membership is invalid")
    return tuple(sorted(set(values)))


def _host_identity_snapshot() -> dict[str, Any]:
    gateway = _lookup_user(GATEWAY_USER)
    writer = _lookup_user(WRITER_USER)
    projector = _lookup_user(PROJECTOR_USER)
    groups: dict[str, Mapping[str, Any]] = {}
    for name in (
        GATEWAY_GROUP,
        WRITER_GROUP,
        SOCKET_CLIENT_GROUP,
        PROJECTOR_GROUP,
    ):
        group = _lookup_group(name)
        if group is None:
            continue
        groups[name] = {
            "gid": group.gr_gid,
            "members": sorted(set(group.gr_mem)),
        }
    return {
        "gateway": {
            "name": gateway.pw_name,
            "uid": gateway.pw_uid,
            "gid": gateway.pw_gid,
            "home": gateway.pw_dir,
            "shell": gateway.pw_shell,
            "groups": list(_supplementary_gids(GATEWAY_USER, gateway.pw_gid)),
        },
        "writer": {
            "name": writer.pw_name,
            "uid": writer.pw_uid,
            "gid": writer.pw_gid,
            "home": writer.pw_dir,
            "shell": writer.pw_shell,
            "groups": list(_supplementary_gids(WRITER_USER, writer.pw_gid)),
        },
        "projector": {
            "name": projector.pw_name,
            "uid": projector.pw_uid,
            "gid": projector.pw_gid,
            "home": projector.pw_dir,
            "shell": projector.pw_shell,
            "groups": list(_supplementary_gids(PROJECTOR_USER, projector.pw_gid)),
        },
        "groups": groups,
        "effective_gid_members": _effective_gid_members((
            CANARY_SOCKET_CLIENT_GID,
            CANARY_PROJECTOR_GID,
            CANARY_GATEWAY_GID,
            CANARY_WRITER_GID,
        )),
    }


def _effective_gid_members(gids: Sequence[int]) -> dict[str, list[str]]:
    targets = set(gids)
    if len(targets) != len(tuple(gids)):
        raise ValueError("effective group targets are not unique")
    accounts = tuple(pwd.getpwall())
    groups = tuple(grp.getgrall())
    if len({item.pw_name for item in accounts}) != len(accounts) or len({
        item.pw_uid for item in accounts
    }) != len(accounts):
        raise RuntimeError("NSS passwd identities are ambiguous")
    for name in (
        GATEWAY_GROUP,
        WRITER_GROUP,
        SOCKET_CLIENT_GROUP,
        PROJECTOR_GROUP,
    ):
        if sum(item.gr_name == name for item in groups) > 1:
            raise RuntimeError("NSS pinned group name is ambiguous")
    for gid in targets:
        names = [item.gr_name for item in groups if item.gr_gid == gid]
        if len(names) > 1:
            raise RuntimeError("NSS group GID identity is ambiguous")
    result: dict[str, list[str]] = {}
    for gid in sorted(targets):
        members = {item.pw_name for item in accounts if item.pw_gid == gid}
        for group in groups:
            if group.gr_gid == gid:
                members.update(group.gr_mem)
        unknown = members - {item.pw_name for item in accounts}
        if unknown:
            raise RuntimeError("NSS group contains an unknown account")
        result[str(gid)] = sorted(members)
    return result


def _host_identities_are_exact(snapshot: Mapping[str, Any]) -> bool:
    gateway = snapshot.get("gateway")
    writer = snapshot.get("writer")
    projector = snapshot.get("projector")
    groups = snapshot.get("groups")
    effective = snapshot.get("effective_gid_members")
    return bool(
        isinstance(gateway, Mapping)
        and isinstance(writer, Mapping)
        and isinstance(projector, Mapping)
        and isinstance(groups, Mapping)
        and isinstance(effective, Mapping)
        and gateway
        == {
            "name": GATEWAY_USER,
            "uid": CANARY_GATEWAY_UID,
            "gid": CANARY_GATEWAY_GID,
            "home": GATEWAY_HOME,
            "shell": NOLOGIN_SHELL,
            "groups": [CANARY_SOCKET_CLIENT_GID, CANARY_GATEWAY_GID],
        }
        and writer.get("name") == WRITER_USER
        and writer.get("uid") == CANARY_WRITER_UID
        and writer.get("gid") == CANARY_WRITER_GID
        and writer.get("home") == "/nonexistent"
        and writer.get("shell") == NOLOGIN_SHELL
        and writer.get("groups") == [CANARY_SOCKET_CLIENT_GID + 1, CANARY_WRITER_GID]
        and projector
        == {
            "name": PROJECTOR_USER,
            "uid": CANARY_PROJECTOR_UID,
            "gid": CANARY_PROJECTOR_GID,
            "home": "/nonexistent",
            "shell": NOLOGIN_SHELL,
            "groups": [CANARY_PROJECTOR_GID],
        }
        and groups.get(GATEWAY_GROUP, {}).get("gid") == CANARY_GATEWAY_GID
        and groups.get(GATEWAY_GROUP, {}).get("members") == []
        and groups.get(WRITER_GROUP, {}).get("gid") == CANARY_WRITER_GID
        and groups.get(WRITER_GROUP, {}).get("members") == []
        and groups.get(SOCKET_CLIENT_GROUP)
        == {"gid": CANARY_SOCKET_CLIENT_GID, "members": [GATEWAY_USER]}
        and groups.get(PROJECTOR_GROUP, {}).get("gid") == CANARY_PROJECTOR_GID
        and groups.get(PROJECTOR_GROUP, {}).get("members") == [WRITER_USER]
        and effective
        == {
            str(CANARY_SOCKET_CLIENT_GID): [GATEWAY_USER],
            str(CANARY_PROJECTOR_GID): [PROJECTOR_USER, WRITER_USER],
            str(CANARY_GATEWAY_GID): [GATEWAY_USER],
            str(CANARY_WRITER_GID): [WRITER_USER],
        }
    )


def prepare_canary_host_identities(
    plan: NativeObservationPlan,
    *,
    approved_plan_sha256: str,
    owner_approval_receipt: OwnerApprovalReceipt,
    runner: Runner = _runner,
) -> Mapping[str, Any]:
    """Reconcile only the pre-existing canary users and one missing group."""

    if not isinstance(plan, NativeObservationPlan):
        raise TypeError("native observation plan is required")
    _digest(approved_plan_sha256, "approved native observation plan sha256")
    if plan.sha256 != approved_plan_sha256:
        raise PermissionError("native host preparation approval does not match plan")
    if (
        not isinstance(owner_approval_receipt, OwnerApprovalReceipt)
        or owner_approval_receipt.value.get("scope") != "native_observation"
        or owner_approval_receipt.value.get("plan_sha256") != plan.sha256
    ):
        raise PermissionError("native host preparation owner receipt is not exact")
    owner_approval_receipt.require(
        scope="native_observation",
        plan_sha256=plan.sha256,
        now_unix=int(time.time()),
    )
    identities = plan.value["identities"]
    expected = {
        "gateway_uid": CANARY_GATEWAY_UID,
        "gateway_gid": CANARY_GATEWAY_GID,
        "gateway_home": GATEWAY_HOME,
        "gateway_supplementary_gids": [
            CANARY_SOCKET_CLIENT_GID,
            CANARY_GATEWAY_GID,
        ],
        "writer_uid": CANARY_WRITER_UID,
        "writer_gid": CANARY_WRITER_GID,
        "writer_home": "/nonexistent",
        "writer_supplementary_gids": [
            CANARY_PROJECTOR_GID,
            CANARY_WRITER_GID,
        ],
        "socket_group_gid": CANARY_SOCKET_CLIENT_GID,
        "projector_uid": CANARY_PROJECTOR_UID,
        "projector_gid": CANARY_PROJECTOR_GID,
        "projector_home": "/nonexistent",
    }
    if identities != expected:
        raise ValueError("native host preparation identities are not canary-pinned")
    _require_root_linux()
    before = _host_identity_snapshot()
    if _host_identities_are_exact(before):
        return {"changed": False, "before": before, "after": before}
    gateway = _lookup_user(GATEWAY_USER)
    writer = _lookup_user(WRITER_USER)
    projector = _lookup_user(PROJECTOR_USER)
    try:
        gateway_by_uid = pwd.getpwuid(CANARY_GATEWAY_UID)
        writer_by_uid = pwd.getpwuid(CANARY_WRITER_UID)
        projector_by_uid = pwd.getpwuid(CANARY_PROJECTOR_UID)
    except KeyError as exc:
        raise RuntimeError("pre-existing canary numeric UID is absent") from exc
    socket_by_name = _lookup_group(SOCKET_CLIENT_GROUP)
    socket_by_gid = _lookup_gid(CANARY_SOCKET_CLIENT_GID)
    gateway_group = _lookup_group(GATEWAY_GROUP)
    writer_group = _lookup_group(WRITER_GROUP)
    projector_group = _lookup_group(PROJECTOR_GROUP)
    if (
        gateway.pw_uid != CANARY_GATEWAY_UID
        or gateway.pw_gid != CANARY_GATEWAY_GID
        or writer.pw_uid != CANARY_WRITER_UID
        or writer.pw_gid != CANARY_WRITER_GID
        or projector.pw_uid != CANARY_PROJECTOR_UID
        or projector.pw_gid != CANARY_PROJECTOR_GID
        or projector.pw_dir != "/nonexistent"
        or projector.pw_shell != NOLOGIN_SHELL
        or gateway_by_uid.pw_name != GATEWAY_USER
        or writer_by_uid.pw_name != WRITER_USER
        or projector_by_uid.pw_name != PROJECTOR_USER
        or gateway_group is None
        or gateway_group.gr_gid != CANARY_GATEWAY_GID
        or _lookup_gid(CANARY_GATEWAY_GID).gr_name != GATEWAY_GROUP
        or writer_group is None
        or writer_group.gr_gid != CANARY_WRITER_GID
        or _lookup_gid(CANARY_WRITER_GID).gr_name != WRITER_GROUP
        or projector_group is None
        or projector_group.gr_gid != CANARY_PROJECTOR_GID
        or _lookup_gid(CANARY_PROJECTOR_GID).gr_name != PROJECTOR_GROUP
    ):
        raise RuntimeError("pre-existing canary user/group identities collided")
    if socket_by_name is None:
        if socket_by_gid is not None:
            raise RuntimeError("writer-client GID is occupied by another group")
        _run(
            Command(
                (
                    GROUPADD,
                    "--gid",
                    str(CANARY_SOCKET_CLIENT_GID),
                    SOCKET_CLIENT_GROUP,
                ),
                timeout_seconds=30,
            ),
            runner=runner,
            label="create exact writer-client group",
        )
    elif (
        socket_by_name.gr_gid != CANARY_SOCKET_CLIENT_GID
        or socket_by_gid is None
        or socket_by_gid.gr_name != SOCKET_CLIENT_GROUP
    ):
        raise RuntimeError("writer-client group name or GID collided")
    _run(
        Command(
            (
                USERMOD,
                "--home",
                GATEWAY_HOME,
                "--shell",
                NOLOGIN_SHELL,
                "--gid",
                str(CANARY_GATEWAY_GID),
                "--groups",
                str(CANARY_SOCKET_CLIENT_GID),
                GATEWAY_USER,
            ),
            timeout_seconds=30,
        ),
        runner=runner,
        label="reconcile exact gateway identity",
    )
    _run(
        Command(
            (
                USERMOD,
                "--home",
                "/nonexistent",
                "--shell",
                NOLOGIN_SHELL,
                "--gid",
                str(CANARY_WRITER_GID),
                "--groups",
                str(CANARY_PROJECTOR_GID),
                WRITER_USER,
            ),
            timeout_seconds=30,
        ),
        runner=runner,
        label="reconcile exact writer identity",
    )
    after = _host_identity_snapshot()
    if not _host_identities_are_exact(after):
        raise RuntimeError("canary host identity reconciliation did not converge")
    return {"changed": True, "before": before, "after": after}


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _install_exact_bytes(
    path: Path,
    payload: bytes,
    *,
    uid: int,
    gid: int,
    mode: int,
) -> bool:
    """Install absent content atomically; existing content is idempotent only."""

    _validate_root_parent_chain(path.parent)
    digest = _sha256_bytes(payload)
    try:
        existing = _read_trusted_file(
            path,
            expected_uid=uid,
            expected_gid=gid,
            allowed_modes=frozenset({mode}),
            maximum=max(len(payload), 1),
        )
    except ValueError as exc:
        if os.path.lexists(path):
            raise RuntimeError(
                "activation target exists with invalid identity"
            ) from exc
    except PermissionError:
        raise
    else:
        if existing != payload or _sha256_bytes(existing) != digest:
            raise RuntimeError("activation target exists with different content")
        return False

    temporary = path.with_name(f".{path.name}.activation.{os.getpid()}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    target_created = False
    temporary_identity: tuple[int, int] | None = None
    try:
        os.fchown(descriptor, uid, gid)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("activation file write made no progress")
            offset += written
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
        opened = os.fstat(descriptor)
        temporary_identity = (opened.st_dev, opened.st_ino)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_uid != uid
            or opened.st_gid != gid
            or stat.S_IMODE(opened.st_mode) != mode
            or opened.st_size != len(payload)
        ):
            raise RuntimeError("activation temporary file identity is invalid")
        os.close(descriptor)
        descriptor = -1
        os.link(temporary, path, follow_symlinks=False)
        target_created = True
        temporary.unlink()
        _fsync_directory(path.parent)
    except BaseException:
        if target_created and temporary_identity is not None:
            try:
                target = os.lstat(path)
                if (target.st_dev, target.st_ino) == temporary_identity:
                    path.unlink()
                    _fsync_directory(path.parent)
            except FileNotFoundError:
                pass
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    try:
        installed = _read_trusted_file(
            path,
            expected_uid=uid,
            expected_gid=gid,
            allowed_modes=frozenset({mode}),
            maximum=max(len(payload), 1),
        )
        if installed != payload or _sha256_bytes(installed) != digest:
            raise RuntimeError("activation installed file readback failed")
    except BaseException:
        if target_created and temporary_identity is not None:
            try:
                target = os.lstat(path)
                if (target.st_dev, target.st_ino) == temporary_identity:
                    path.unlink()
                    _fsync_directory(path.parent)
            except FileNotFoundError:
                pass
        raise
    return True


def _unlink_exact(
    path: Path,
    *,
    uid: int,
    gid: int,
    mode: int,
    sha256: str | None = None,
) -> None:
    if not os.path.lexists(path):
        return
    raw = _read_trusted_file(
        path,
        expected_uid=uid,
        expected_gid=gid,
        allowed_modes=frozenset({mode}),
        maximum=_MAX_MANIFEST_BYTES,
    )
    if sha256 is not None and _sha256_bytes(raw) != sha256:
        raise RuntimeError("activation refuses to remove drifted evidence")
    path.unlink()
    _fsync_directory(path.parent)


def _external_iam_archive_path(receipt: ExternalIAMReceipt) -> Path:
    if not isinstance(receipt, ExternalIAMReceipt):
        raise TypeError("external IAM receipt is required")
    return (
        DEFAULT_EXTERNAL_IAM_RECEIPT_ROOT
        / receipt.policy_sha256
        / f"{receipt.sha256}.json"
    )


def _archive_external_iam_at(
    receipt: ExternalIAMReceipt,
    target: Path,
) -> Mapping[str, Any]:
    payload = _canonical_bytes(receipt.to_mapping())
    if _sha256_bytes(payload) != receipt.sha256:
        raise RuntimeError("external IAM receipt canonical digest drifted")
    _ensure_root_directory(target.parent)
    _install_exact_bytes(target, payload, uid=0, gid=0, mode=0o400)
    return {
        "path": str(target),
        "sha256": receipt.sha256,
        "policy_sha256": receipt.policy_sha256,
        "mode": "0400",
        "owner_uid": 0,
        "group_gid": 0,
    }


def _replace_live_external_iam(receipt: ExternalIAMReceipt) -> bool:
    """Atomically renew the fixed live receipt after sealing both generations."""

    live = DEFAULT_EXTERNAL_IAM_LIVE_PATH
    _ensure_root_directory(live.parent)
    payload = _canonical_bytes(receipt.to_mapping())
    if os.path.lexists(live):
        previous_raw = _read_trusted_file(
            live,
            expected_uid=0,
            expected_gid=0,
            allowed_modes=frozenset({0o400}),
            maximum=64 * 1024,
        )
        previous = ExternalIAMReceipt.from_mapping(
            _decode_strict_json(previous_raw, label="existing external IAM receipt")
        )
        _archive_external_iam_at(previous, _external_iam_archive_path(previous))
        if previous.sha256 == receipt.sha256:
            return False
    temporary = live.with_name(
        f".{live.name}.{receipt.sha256}.activation.{os.getpid()}"
    )
    _install_exact_bytes(temporary, payload, uid=0, gid=0, mode=0o400)
    try:
        os.replace(temporary, live)
        _fsync_directory(live.parent)
    except BaseException:
        _unlink_exact(
            temporary,
            uid=0,
            gid=0,
            mode=0o400,
            sha256=receipt.sha256,
        )
        raise
    installed = _read_trusted_file(
        live,
        expected_uid=0,
        expected_gid=0,
        allowed_modes=frozenset({0o400}),
        maximum=64 * 1024,
    )
    if installed != payload or _sha256_bytes(installed) != receipt.sha256:
        raise RuntimeError("external IAM live receipt readback failed")
    return True


def install_staged_external_iam_receipt(
    path_value: str | os.PathLike[str],
    *,
    authorized_plan: NativeObservationPlan | ActivationPlan,
    owner_approval_receipt: OwnerApprovalReceipt,
) -> tuple[ExternalIAMReceipt, Mapping[str, Any], bool]:
    """Seal only IAM evidence cross-bound to this exact owner-approved plan."""

    _require_root_linux()
    if isinstance(authorized_plan, NativeObservationPlan):
        authorized_plan = NativeObservationPlan.from_mapping(
            authorized_plan.to_mapping()
        )
        scope = "native_observation"
        approved_plan_sha256 = authorized_plan.sha256
        expected_policy_sha256 = authorized_plan.value[
            "external_iam_policy_sha256"
        ]
    elif isinstance(authorized_plan, ActivationPlan):
        authorized_plan = ActivationPlan.from_mapping(authorized_plan.to_mapping())
        scope = "activation"
        approved_plan_sha256 = authorized_plan.sha256
        expected_policy_sha256 = (
            authorized_plan.digests.external_iam_policy_sha256
        )
    else:
        raise TypeError("external IAM requires a validated native or final plan")
    _digest(expected_policy_sha256, "external IAM expected policy digest")
    if not isinstance(owner_approval_receipt, OwnerApprovalReceipt):
        raise PermissionError("external IAM owner approval receipt is required")
    owner_approval_receipt.require(
        scope=scope,
        plan_sha256=approved_plan_sha256,
        now_unix=int(time.time()),
    )
    source = _absolute_path(os.fspath(path_value), "staged external IAM path")
    if source != DEFAULT_STAGED_EXTERNAL_IAM_PATH:
        raise ValueError("staged external IAM path is not production-pinned")
    raw = _read_trusted_file(
        source,
        expected_uid=0,
        expected_gid=0,
        allowed_modes=frozenset({0o400}),
        maximum=64 * 1024,
    )
    receipt = ExternalIAMReceipt.from_mapping(
        _decode_strict_json(raw, label="staged external IAM receipt")
    )
    receipt.require_fresh(int(time.time()))
    if receipt.policy_sha256 != expected_policy_sha256:
        raise ValueError("staged external IAM policy does not match approval")
    if (
        receipt.value.get("source_approval_sha256")
        != owner_approval_receipt.sha256
    ):
        raise PermissionError(
            "staged external IAM evidence is not bound to owner approval"
        )
    canonical = _canonical_bytes(receipt.to_mapping())
    if raw not in {canonical, canonical + b"\n"}:
        raise ValueError("staged external IAM receipt changed during validation")
    archive = _archive_external_iam_at(
        receipt,
        _external_iam_archive_path(receipt),
    )
    replaced = _replace_live_external_iam(receipt)
    return receipt, archive, replaced


def _artifact_payload(plan: ActivationPlan, name: str) -> bytes:
    if name == "manifest":
        return _canonical_bytes(plan.deployment_manifest)
    if name == "writer_unit":
        return plan.unit_bundle.writer_service.encode("utf-8")
    if name == "gateway_unit":
        return plan.unit_bundle.gateway_service.encode("utf-8")
    if name == "tmpfiles":
        return plan.unit_bundle.tmpfiles.encode("utf-8")
    artifact = plan.install_artifacts[name]
    if artifact.source_path is None:
        raise RuntimeError("activation artifact has no trusted content source")
    raw = _read_trusted_file(
        artifact.source_path,
        expected_uid=0,
        expected_gid=0,
        allowed_modes=frozenset({0o400}),
        maximum=artifact.maximum_bytes,
    )
    if _sha256_bytes(raw) != artifact.sha256:
        raise RuntimeError(f"activation {name} staging digest drifted")
    return raw


def _install_plan_artifacts(plan: ActivationPlan) -> tuple[Path, ...]:
    created: list[Path] = []
    try:
        for name in (
            "manifest",
            "writer_config",
            "gateway_config",
            "writer_unit",
            "gateway_unit",
            "tmpfiles",
        ):
            artifact = plan.install_artifacts[name]
            payload = _artifact_payload(plan, name)
            if (
                len(payload) > artifact.maximum_bytes
                or _sha256_bytes(payload) != artifact.sha256
            ):
                raise RuntimeError(f"activation {name} payload digest is invalid")
            if _install_exact_bytes(
                artifact.target_path,
                payload,
                uid=artifact.uid,
                gid=artifact.gid,
                mode=artifact.mode,
            ):
                created.append(artifact.target_path)
        return tuple(created)
    except Exception as install_error:
        rollback_errors: list[BaseException] = []
        by_target = {item.target_path: item for item in plan.install_artifacts.values()}
        for path in reversed(created):
            artifact = by_target[path]
            try:
                _unlink_exact(
                    path,
                    uid=artifact.uid,
                    gid=artifact.gid,
                    mode=artifact.mode,
                    sha256=artifact.sha256,
                )
            except Exception as exc:
                rollback_errors.append(exc)
        if rollback_errors:
            raise ExceptionGroup(
                "activation install and rollback failed",
                [install_error, *rollback_errors],
            ) from None
        raise


def _native_artifact_contract(
    plan: NativeObservationPlan,
) -> Mapping[str, InstallArtifact]:
    value = plan.value
    writer_unit = value["writer_unit"]
    gateway_unit = value["gateway_unit"]
    writer_config = value["writer_config"]
    gateway_config = value["gateway_config"]
    exact = (
        writer_unit["name"] == WRITER_UNIT
        and writer_unit["path"] == str(DEFAULT_WRITER_UNIT_PATH)
        and gateway_unit["name"] == GATEWAY_UNIT
        and gateway_unit["path"] == str(DEFAULT_GATEWAY_UNIT_PATH)
        and writer_config["path"] == str(DEFAULT_WRITER_CONFIG_PATH)
        and gateway_config["path"] == str(DEFAULT_GATEWAY_CONFIG_PATH)
    )
    if not exact:
        raise ValueError("native observation install paths are not production-pinned")
    return {
        "writer_unit": InstallArtifact(
            source_path=DEFAULT_STAGED_WRITER_UNIT_PATH,
            target_path=DEFAULT_WRITER_UNIT_PATH,
            sha256=writer_unit["sha256"],
            mode=0o644,
            uid=0,
            gid=0,
            maximum_bytes=256 * 1024,
        ),
        "gateway_unit": InstallArtifact(
            source_path=DEFAULT_STAGED_GATEWAY_UNIT_PATH,
            target_path=DEFAULT_GATEWAY_UNIT_PATH,
            sha256=gateway_unit["sha256"],
            mode=0o644,
            uid=0,
            gid=0,
            maximum_bytes=256 * 1024,
        ),
        "writer_config": InstallArtifact(
            source_path=DEFAULT_WRITER_CONFIG_SOURCE_PATH,
            target_path=DEFAULT_WRITER_CONFIG_PATH,
            sha256=writer_config["sha256"],
            mode=0o440,
            uid=0,
            gid=CANARY_WRITER_GID,
            maximum_bytes=_MAX_CONFIG_BYTES,
        ),
        "gateway_config": InstallArtifact(
            source_path=DEFAULT_GATEWAY_CONFIG_SOURCE_PATH,
            target_path=DEFAULT_GATEWAY_CONFIG_PATH,
            sha256=gateway_config["sha256"],
            mode=0o444,
            uid=0,
            gid=0,
            maximum_bytes=_MAX_CONFIG_BYTES,
        ),
    }


def _install_native_observation_artifacts(
    plan: NativeObservationPlan,
) -> tuple[Path, ...]:
    artifacts = _native_artifact_contract(plan)
    created: list[Path] = []
    try:
        for name in (
            "writer_config",
            "gateway_config",
            "writer_unit",
            "gateway_unit",
        ):
            artifact = artifacts[name]
            if artifact.source_path is None:
                raise RuntimeError("native observation staging source is absent")
            payload = _read_trusted_file(
                artifact.source_path,
                expected_uid=0,
                expected_gid=0,
                allowed_modes=frozenset({0o400}),
                maximum=artifact.maximum_bytes,
            )
            if _sha256_bytes(payload) != artifact.sha256:
                raise RuntimeError(f"native observation {name} digest drifted")
            if _install_exact_bytes(
                artifact.target_path,
                payload,
                uid=artifact.uid,
                gid=artifact.gid,
                mode=artifact.mode,
            ):
                created.append(artifact.target_path)
        return tuple(created)
    except Exception as install_error:
        rollback: list[BaseException] = []
        by_path = {item.target_path: item for item in artifacts.values()}
        for path in reversed(created):
            artifact = by_path[path]
            try:
                _unlink_exact(
                    path,
                    uid=artifact.uid,
                    gid=artifact.gid,
                    mode=artifact.mode,
                    sha256=artifact.sha256,
                )
            except BaseException as exc:
                rollback.append(exc)
        if rollback:
            raise ExceptionGroup(
                "native observation install and rollback failed",
                [install_error, *rollback],
            ) from None
        raise


def _ensure_runtime_directory(
    path: Path,
    *,
    uid: int,
    gid: int,
    mode: int,
) -> None:
    _validate_root_parent_chain(path.parent)
    try:
        os.mkdir(path, mode)
    except FileExistsError:
        pass
    else:
        os.chown(path, uid, gid)
        os.chmod(path, mode)
        _fsync_directory(path.parent)
    item = os.lstat(path)
    if (
        stat.S_ISLNK(item.st_mode)
        or not stat.S_ISDIR(item.st_mode)
        or item.st_uid != uid
        or item.st_gid != gid
        or stat.S_IMODE(item.st_mode) != mode
        or _list_xattrs(path)
    ):
        raise PermissionError("native observation runtime directory drifted")


def _prepare_native_runtime_directories() -> None:
    _ensure_root_directory(
        DEFAULT_PROJECTION_DIRECTORY.parent,
        mode=0o755,
    )
    for path, uid, gid, mode in (
        (
            DEFAULT_WRITER_RUNTIME,
            CANARY_WRITER_UID,
            CANARY_SOCKET_CLIENT_GID,
            0o2750,
        ),
        (
            DEFAULT_PROJECTION_DIRECTORY,
            CANARY_WRITER_UID,
            CANARY_PROJECTOR_GID,
            0o750,
        ),
        (
            DEFAULT_GATEWAY_RUNTIME,
            CANARY_GATEWAY_UID,
            CANARY_GATEWAY_GID,
            0o700,
        ),
        (
            DEFAULT_GATEWAY_HOME,
            CANARY_GATEWAY_UID,
            CANARY_GATEWAY_GID,
            0o700,
        ),
        (
            DEFAULT_GATEWAY_LOGS,
            CANARY_GATEWAY_UID,
            CANARY_GATEWAY_GID,
            0o700,
        ),
    ):
        _ensure_runtime_directory(path, uid=uid, gid=gid, mode=mode)


def _verify_native_release(plan: NativeObservationPlan) -> None:
    view = SimpleNamespace(
        revision=plan.value["revision"],
        deployment_manifest={
            "snapshot_template": {
                "writer_deployment": {
                    "policy": {
                        "artifact_root": plan.value["artifact_root"],
                        "artifact_digest_sha256": plan.value["artifact_sha256"],
                    }
                }
            }
        },
    )
    _verify_release_tree(view)
    raw = _read_trusted_file(
        Path(plan.value["artifact_root"]) / "release-manifest.json",
        expected_uid=0,
        expected_gid=0,
        allowed_modes=frozenset({0o400}),
        maximum=_MAX_MANIFEST_BYTES,
    )
    _decode_strict_json(raw, label="native release manifest")
    if plan.value["release_manifest_file_sha256"] != _sha256_bytes(raw):
        raise RuntimeError("native release manifest digest drifted")


def _load_bound_config_collector_receipt(
    plan: NativeObservationPlan,
    *,
    require_fresh: bool,
) -> ConfigCollectorReceipt:
    database = plan.value["database"]
    sql_private_ip = str(
        ipaddress.ip_network(database["ip_network"], strict=True).network_address
    )
    receipt = load_config_collector_receipt(
        revision=plan.value["revision"],
        receipt_sha256=plan.value["config_collector_receipt_sha256"],
        require_fresh=require_fresh,
    )
    receipt.require_bindings(
        revision=plan.value["revision"],
        release_artifact_sha256=plan.value["artifact_sha256"],
        release_manifest_file_sha256=(
            plan.value["release_manifest_file_sha256"]
        ),
        writer_config_sha256=plan.value["writer_config"]["sha256"],
        gateway_config_sha256=plan.value["gateway_config"]["sha256"],
        database_ca_sha256=database["ca_sha256"],
        sql_private_ip=sql_private_ip,
        sql_tls_server_name=database["tls_server_name"],
    )
    return receipt


def _verify_native_preflight_inputs(
    plan: NativeObservationPlan,
    *,
    runner: Runner,
    require_installed: bool,
    require_original_boot: bool,
) -> None:
    """Perform all bounded host reads before the native lifecycle mutates state."""

    if (
        require_original_boot
        and _current_boot_id_sha256() != plan.value["boot_id_sha256"]
    ):
        raise RuntimeError("native observation boot identity drifted")
    if current_host_identity_sha256() != plan.value["host_identity_sha256"]:
        raise RuntimeError("native observation host identity drifted")
    collector_receipt = None
    if not require_installed:
        collector_receipt = _load_bound_config_collector_receipt(
            plan,
            require_fresh=True,
        )
    _verify_native_release(plan)
    artifacts = _native_artifact_contract(plan)
    for name, artifact in artifacts.items():
        if artifact.source_path is None:
            raise RuntimeError(f"native {name} staging source is absent")
        source = _read_trusted_file(
            artifact.source_path,
            expected_uid=0,
            expected_gid=0,
            allowed_modes=frozenset({0o400}),
            maximum=artifact.maximum_bytes,
        )
        if _sha256_bytes(source) != artifact.sha256:
            raise RuntimeError(f"native {name} staging digest drifted")
        if os.path.lexists(artifact.target_path):
            installed = _read_trusted_file(
                artifact.target_path,
                expected_uid=artifact.uid,
                expected_gid=artifact.gid,
                allowed_modes=frozenset({artifact.mode}),
                maximum=artifact.maximum_bytes,
            )
            if installed != source or _sha256_bytes(installed) != artifact.sha256:
                raise RuntimeError(f"native {name} install collision drifted")
        elif require_installed:
            raise RuntimeError(f"native {name} installed artifact is absent")
        else:
            _validate_root_parent_chain(artifact.target_path.parent)
    database = plan.value["database"]
    ca_path = Path(database["ca_path"])
    ca = _read_trusted_file(
        ca_path,
        expected_uid=0,
        expected_gid=CANARY_WRITER_GID,
        allowed_modes=frozenset({0o400, 0o440, 0o444}),
        maximum=_MAX_CONFIG_BYTES,
    )
    if _sha256_bytes(ca) != database["ca_sha256"]:
        raise RuntimeError("native database CA digest drifted")
    _verify_database_read_only(
        config_path=DEFAULT_WRITER_CONFIG_SOURCE_PATH,
        expected_database=database,
    )
    if require_installed:
        collector_receipt = _load_bound_config_collector_receipt(
            plan,
            require_fresh=False,
        )
    for unit in (WRITER_UNIT, GATEWAY_UNIT):
        if require_installed:
            _require_off_disabled(unit, runner=runner)
        else:
            _require_off_or_absent(unit, runner=runner)
    _require_off_disabled(EXPORTER_UNIT, runner=runner, absent=True)
    _require_off_disabled(DISCORD_UNIT, runner=runner, absent=True)
    discord = plan.value["discord"]
    for name in ("config_path", "token_path", "socket_path"):
        if os.path.lexists(discord[name]):
            raise RuntimeError(f"native Discord {name} must remain absent")
    helper = Path(plan.value["legacy_helper_path"])
    if os.path.lexists(helper) or os.path.lexists(helper.parent):
        raise RuntimeError("native legacy helper authority must remain absent")
    final_collector_receipt = _load_bound_config_collector_receipt(
        plan,
        require_fresh=not require_installed,
    )
    if (
        collector_receipt is None
        or final_collector_receipt.to_mapping()
        != collector_receipt.to_mapping()
    ):
        raise RuntimeError("config collector receipt rotated during preflight")


def _verify_database_read_only(
    *,
    config_path: Path,
    expected_database: Mapping[str, Any],
) -> None:
    """Attest exact credential provenance, TLS coordinates, and PG privileges."""

    from gateway.canonical_writer_bootstrap import load_service_config
    from gateway.canonical_writer_db import CanonicalWriterDB
    from gateway.canonical_writer_postgres_backend import (
        PRODUCTION_STATEMENT_CATALOG,
    )

    config = load_service_config(config_path)
    if config.discord_edge_authority.enabled:
        raise RuntimeError("writer database preflight enables Discord authority")
    database = config.database
    if (
        expected_database.get("ip_network") != f"{database.host}/32"
        or expected_database.get("tls_server_name") != database.tls_server_name
        or expected_database.get("ca_path") != str(database.ca_file)
    ):
        raise RuntimeError("writer database preflight coordinates drifted")
    credential = database.credential
    if credential.path is None:
        raise RuntimeError("writer database credential is not file-backed")
    item = os.lstat(credential.path)
    if (
        stat.S_ISLNK(item.st_mode)
        or not stat.S_ISREG(item.st_mode)
        or item.st_nlink != 1
        or item.st_uid != credential.expected_uid
        or (
            credential.expected_gid is not None
            and item.st_gid != credential.expected_gid
        )
        or stat.S_IMODE(item.st_mode) not in credential.allowed_modes
        or _list_xattrs(credential.path)
    ):
        raise RuntimeError("writer database credential provenance drifted")
    _validate_root_parent_chain(credential.path.parent)
    CanonicalWriterDB(
        config=database,
        privilege_policy=config.privileges,
        statements=PRODUCTION_STATEMENT_CATALOG,
    ).startup_attest()


def _load_durable_native_authority_chain(
    plan: NativeObservationPlan,
    receipt: NativeObservationReceipt,
) -> tuple[OwnerApprovalReceipt, ExternalIAMReceipt, Mapping[str, Any]]:
    """Re-read the exact historical approval and IAM evidence for a receipt."""

    if not isinstance(plan, NativeObservationPlan) or not isinstance(
        receipt,
        NativeObservationReceipt,
    ):
        raise TypeError("durable native plan and receipt are required")
    owner_sha256 = _digest(
        receipt.value.get("owner_approval_receipt_sha256"),
        "durable native owner approval receipt",
    )
    owner_path = (
        DEFAULT_OWNER_APPROVAL_ROOT
        / "native_observation"
        / plan.sha256
        / f"{owner_sha256}.json"
    )
    owner_raw = _read_trusted_file(
        owner_path,
        expected_uid=0,
        expected_gid=0,
        allowed_modes=frozenset({0o400}),
        maximum=64 * 1024,
    )
    owner = OwnerApprovalReceipt.from_mapping(
        _decode_strict_json(owner_raw, label="durable native owner approval")
    )
    owner.require(
        scope="native_observation",
        plan_sha256=plan.sha256,
        now_unix=owner.value["approved_at_unix"],
    )
    if owner.sha256 != owner_sha256 or owner_approval_receipt_path(owner) != owner_path:
        raise RuntimeError("durable native owner approval chain drifted")

    iam_sha256 = _digest(
        receipt.value.get("external_iam_receipt_sha256"),
        "durable native external IAM receipt",
    )
    iam_path = (
        DEFAULT_NATIVE_OBSERVATION_EVIDENCE_ROOT
        / plan.value["revision"]
        / plan.sha256
        / "external-iam"
        / f"{iam_sha256}.json"
    )
    iam_raw = _read_trusted_file(
        iam_path,
        expected_uid=0,
        expected_gid=0,
        allowed_modes=frozenset({0o400}),
        maximum=64 * 1024,
    )
    iam = ExternalIAMReceipt.from_mapping(
        _decode_strict_json(iam_raw, label="durable native archived IAM receipt")
    )
    if (
        iam.sha256 != iam_sha256
        or iam.policy_sha256 != plan.value["external_iam_policy_sha256"]
        or iam.value.get("source_approval_sha256") != owner.sha256
    ):
        raise RuntimeError("durable native IAM approval chain drifted")
    evidence = {
        "path": str(iam_path),
        "sha256": iam.sha256,
        "policy_sha256": iam.policy_sha256,
        "mode": "0400",
        "owner_uid": 0,
        "group_gid": 0,
        "live_path": str(DEFAULT_EXTERNAL_IAM_LIVE_PATH),
    }
    return owner, iam, evidence


def _load_existing_native_receipt(
    plan: NativeObservationPlan,
    *,
    runner: Runner,
) -> NativeObservationReceipt | None:
    receipt_path = _native_receipt_path(plan)
    stage_path = _native_stage_path(plan)
    if not os.path.lexists(receipt_path):
        if os.path.lexists(stage_path):
            raise RuntimeError(
                "incomplete native observation stage requires quarantine"
            )
        return None
    raw = _read_trusted_file(
        receipt_path,
        expected_uid=0,
        expected_gid=0,
        allowed_modes=frozenset({0o400}),
        maximum=_MAX_MANIFEST_BYTES,
    )
    receipt = NativeObservationReceipt.from_mapping(
        _decode_strict_json(raw, label="existing native observation receipt"),
        expected_plan_sha256=plan.sha256,
    )
    if not os.path.lexists(stage_path):
        raise RuntimeError("native receipt has no append-only observation stage")
    stage_raw = _read_trusted_file(
        stage_path,
        expected_uid=0,
        expected_gid=0,
        allowed_modes=frozenset({0o400}),
        maximum=_MAX_MANIFEST_BYTES,
    )
    stage = _strict_mapping(
        _decode_strict_json(stage_raw, label="existing native observation stage"),
        fields=frozenset({
            "schema",
            "native_observation_plan_sha256",
            "owner_approval_receipt_sha256",
            "host_preparation_receipt_sha256",
            "external_iam_receipt_sha256",
            "plan",
            "observation",
        }),
        label="existing native observation stage",
    )
    if (
        stage["schema"] != NATIVE_OBSERVATION_STAGE_SCHEMA
        or stage["native_observation_plan_sha256"] != plan.sha256
        or stage["plan"] != receipt.value["plan"]
        or stage["observation"] != receipt.value["observation"]
        or stage["owner_approval_receipt_sha256"]
        != receipt.value["owner_approval_receipt_sha256"]
        or stage["host_preparation_receipt_sha256"]
        != receipt.value["host_preparation_receipt_sha256"]
        or stage["external_iam_receipt_sha256"]
        != receipt.value["external_iam_receipt_sha256"]
    ):
        raise RuntimeError("native receipt and append-only stage drifted")
    _load_durable_native_authority_chain(plan, receipt)
    host_preparation = _load_host_preparation_receipt(plan)
    if (
        host_preparation.get("owner_approval_receipt_sha256")
        != stage["owner_approval_receipt_sha256"]
    ):
        raise RuntimeError("native host preparation approval chain drifted")
    if (
        host_preparation.get("receipt_sha256")
        != stage["host_preparation_receipt_sha256"]
    ):
        raise RuntimeError("native host preparation receipt chain drifted")
    if current_host_identity_sha256() != plan.value["host_identity_sha256"]:
        raise RuntimeError("native receipt was replayed on another host")
    _verify_native_preflight_inputs(
        plan,
        runner=runner,
        require_installed=True,
        require_original_boot=False,
    )
    if not _host_identities_are_exact(_host_identity_snapshot()):
        raise RuntimeError("native receipt host identities drifted")
    return receipt


def load_durable_native_observation_receipt(
    plan: NativeObservationPlan,
) -> NativeObservationReceipt:
    """Load and fully re-attest one already-installed native receipt."""

    if not isinstance(plan, NativeObservationPlan):
        raise TypeError("native observation plan is required")
    receipt = _load_existing_native_receipt(plan, runner=_runner)
    if receipt is None:
        raise FileNotFoundError(_native_receipt_path(plan))
    return receipt


def _verify_release_tree(plan: ActivationPlan) -> None:
    policy = plan.deployment_manifest["snapshot_template"]["writer_deployment"][
        "policy"
    ]
    root = _absolute_path(policy["artifact_root"], "release artifact root")
    if root != Path("/opt/muncho-canary-releases") / plan.revision:
        raise RuntimeError("release artifact path is not revision-addressed")
    _validate_root_parent_chain(root.parent)
    root_stat = os.lstat(root)
    if (
        stat.S_ISLNK(root_stat.st_mode)
        or not stat.S_ISDIR(root_stat.st_mode)
        or root_stat.st_uid != 0
        or root_stat.st_gid != 0
        or stat.S_IMODE(root_stat.st_mode) != 0o555
    ):
        raise RuntimeError("release artifact root identity is invalid")
    raw = _read_trusted_file(
        root / "release-manifest.json",
        expected_uid=0,
        expected_gid=0,
        allowed_modes=frozenset({0o400}),
        maximum=_MAX_MANIFEST_BYTES,
    )
    manifest = _decode_strict_json(raw, label="release manifest")
    fields = frozenset({
        "schema",
        "revision",
        "artifact_root",
        "python_version",
        "interpreter",
        "writer_module",
        "writer_module_origin",
        "gateway_module",
        "gateway_module_origin",
        "entries",
        "artifact_sha256",
    })
    _strict_mapping(manifest, fields=fields, label="release manifest")
    unsigned = {
        key: copy.deepcopy(value)
        for key, value in manifest.items()
        if key != "artifact_sha256"
    }
    if (
        manifest["schema"] != RELEASE_SCHEMA
        or manifest["revision"] != plan.revision
        or manifest["artifact_root"] != str(root)
        or manifest["artifact_sha256"] != policy["artifact_digest_sha256"]
        or _sha256_json(unsigned) != manifest["artifact_sha256"]
    ):
        raise RuntimeError("release manifest identity is invalid")
    entries = manifest["entries"]
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("release manifest entries are invalid")
    declared: list[str] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise RuntimeError("release manifest entry is invalid")
        kind = entry.get("kind")
        keys = {
            "file": {"path", "kind", "mode", "size", "sha256"},
            "directory": {"path", "kind", "mode"},
            "symlink": {"path", "kind", "mode", "target"},
        }.get(kind)
        if keys is None or set(entry) != keys:
            raise RuntimeError("release manifest entry fields are invalid")
        relative = entry["path"]
        if (
            not isinstance(relative, str)
            or not relative
            or relative.startswith("/")
            or ".." in Path(relative).parts
            or Path(relative).as_posix() != relative
            or _CONTROL_RE.search(relative) is not None
        ):
            raise RuntimeError("release manifest path is invalid")
        declared.append(relative)
        path = root / relative
        item = os.lstat(path)
        if (
            item.st_uid != 0
            or item.st_gid != 0
            or entry["mode"] != f"{stat.S_IMODE(item.st_mode):04o}"
            or _list_xattrs(path)
        ):
            raise RuntimeError("release entry ownership or mode drifted")
        if kind == "directory":
            if stat.S_ISLNK(item.st_mode) or not stat.S_ISDIR(item.st_mode):
                raise RuntimeError("release directory identity drifted")
        elif kind == "file":
            if (
                stat.S_ISLNK(item.st_mode)
                or not stat.S_ISREG(item.st_mode)
                or item.st_nlink != 1
                or type(entry["size"]) is not int
                or not 0 <= entry["size"] <= _MAX_RELEASE_FILE_BYTES
                or item.st_size != entry["size"]
            ):
                raise RuntimeError("release file identity drifted")
            content = _read_trusted_file(
                path,
                expected_uid=0,
                expected_gid=0,
                allowed_modes=frozenset({stat.S_IMODE(item.st_mode)}),
                maximum=max(entry["size"], 1),
                trusted_parents=False,
                allow_empty=True,
            )
            if _sha256_bytes(content) != _digest(
                entry["sha256"], "release file digest"
            ):
                raise RuntimeError("release file digest drifted")
        else:
            target = entry["target"]
            if (
                not stat.S_ISLNK(item.st_mode)
                or not isinstance(target, str)
                or os.readlink(path) != target
            ):
                raise RuntimeError("release symlink identity drifted")
            resolved = path.resolve(strict=True)
            if resolved != root and root not in resolved.parents:
                raise RuntimeError("release symlink escapes artifact")
    live: list[str] = []
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        directories.sort()
        files.sort()
        for name in [*directories, *files]:
            relative = (Path(current) / name).relative_to(root).as_posix()
            if relative != "release-manifest.json":
                live.append(relative)
    if declared != sorted(set(declared)) or declared != sorted(live):
        raise RuntimeError("release manifest paths do not match live artifact")


def _validate_projection(
    path: Path, identities: NumericIdentities
) -> Mapping[str, Any]:
    raw = _read_trusted_file(
        path,
        expected_uid=identities.writer_uid,
        expected_gid=identities.projector_gid,
        allowed_modes=frozenset({0o640}),
        maximum=_MAX_EXPORT_BYTES,
        trusted_parents=False,
    )
    value = _decode_strict_json(raw, label="projection export")
    if set(value) != {"events"} or not isinstance(value["events"], list):
        raise RuntimeError("projection export fields are not exact")
    return {
        "event_count": len(value["events"]),
        "sha256": _sha256_bytes(raw),
        "size": len(raw),
        "owner_uid": identities.writer_uid,
        "group_gid": identities.projector_gid,
        "mode": "0640",
    }


def _write_root_receipt(path: Path, value: Mapping[str, Any]) -> None:
    payload = _canonical_bytes(dict(value))
    _install_exact_bytes(path, payload, uid=0, gid=0, mode=0o400)


def _ensure_root_directory(path: Path, *, mode: int = 0o700) -> None:
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("activation evidence directory is invalid")
    missing: list[Path] = []
    current = path
    while not os.path.lexists(current):
        missing.append(current)
        current = current.parent
    _validate_root_parent_chain(current)
    for item in reversed(missing):
        os.mkdir(item, mode)
        os.chown(item, 0, 0)
        os.chmod(item, mode)
        _fsync_directory(item.parent)
    observed = os.lstat(path)
    if (
        stat.S_ISLNK(observed.st_mode)
        or not stat.S_ISDIR(observed.st_mode)
        or observed.st_uid != 0
        or observed.st_gid != 0
        or stat.S_IMODE(observed.st_mode) != mode
        or _list_xattrs(path)
    ):
        raise PermissionError("activation evidence directory is not exact")
    _validate_root_parent_chain(path.parent)


def _plan_evidence_directory(plan: ActivationPlan) -> Path:
    return plan.paths.evidence_root / "plans" / plan.revision / plan.sha256


def _archive_plan_external_iam(
    plan: ActivationPlan | NativeObservationPlan,
    receipt: ExternalIAMReceipt,
    *,
    live_path: str | os.PathLike[str],
) -> Mapping[str, Any]:
    path = _absolute_path(os.fspath(live_path), "external IAM live receipt path")
    if path != DEFAULT_EXTERNAL_IAM_LIVE_PATH:
        raise ValueError("external IAM live receipt path is not pinned")
    raw = _read_trusted_file(
        path,
        expected_uid=0,
        expected_gid=0,
        allowed_modes=frozenset({0o400}),
        maximum=64 * 1024,
    )
    observed = ExternalIAMReceipt.from_mapping(
        _decode_strict_json(raw, label="external IAM live receipt")
    )
    if (
        observed.sha256 != receipt.sha256
        or observed.to_mapping() != receipt.to_mapping()
    ):
        raise RuntimeError("external IAM live receipt changed before archival")
    if isinstance(plan, ActivationPlan):
        directory = _plan_evidence_directory(plan)
    elif isinstance(plan, NativeObservationPlan):
        directory = (
            DEFAULT_NATIVE_OBSERVATION_EVIDENCE_ROOT
            / plan.value["revision"]
            / plan.sha256
        )
    else:
        raise TypeError("activation or native plan is required")
    target = directory / "external-iam" / f"{receipt.sha256}.json"
    result = dict(_archive_external_iam_at(receipt, target))
    result["live_path"] = str(path)
    return result


def _validate_plan_external_iam_evidence(
    plan: ActivationPlan,
    value: Any,
) -> ExternalIAMReceipt:
    evidence = _strict_mapping(
        value,
        fields=frozenset({
            "path",
            "sha256",
            "policy_sha256",
            "mode",
            "owner_uid",
            "group_gid",
            "live_path",
        }),
        label="activation external IAM evidence",
    )
    digest = _digest(evidence["sha256"], "archived external IAM receipt")
    expected = _plan_evidence_directory(plan) / "external-iam" / f"{digest}.json"
    if (
        evidence["path"] != str(expected)
        or evidence["policy_sha256"] != plan.digests.external_iam_policy_sha256
        or evidence["mode"] != "0400"
        or evidence["owner_uid"] != 0
        or evidence["group_gid"] != 0
        or evidence["live_path"] != str(DEFAULT_EXTERNAL_IAM_LIVE_PATH)
    ):
        raise RuntimeError("activation external IAM archive binding drifted")
    raw = _read_trusted_file(
        expected,
        expected_uid=0,
        expected_gid=0,
        allowed_modes=frozenset({0o400}),
        maximum=64 * 1024,
    )
    receipt = ExternalIAMReceipt.from_mapping(
        _decode_strict_json(raw, label="archived external IAM receipt")
    )
    if receipt.sha256 != digest or receipt.policy_sha256 != evidence["policy_sha256"]:
        raise RuntimeError("archived external IAM receipt drifted")
    return receipt


def _archive_root_receipt(
    plan: ActivationPlan,
    *,
    expected_sha256: str,
) -> Mapping[str, Any]:
    source = _read_trusted_file(
        plan.paths.root_receipt_path,
        expected_uid=0,
        expected_gid=0,
        allowed_modes=frozenset({0o400}),
        maximum=_MAX_MANIFEST_BYTES,
    )
    value = _decode_strict_json(source, label="validated root preflight receipt")
    canonical = _canonical_bytes(value)
    observed = _sha256_bytes(canonical)
    if observed != expected_sha256:
        raise RuntimeError("validated root receipt digest does not match collector")
    bundle_sha256 = _digest(
        value.get("evidence_bundle_sha256"),
        "root preflight evidence bundle sha256",
    )
    bundle_path = _absolute_path(
        value.get("evidence_bundle_path"),
        "root preflight evidence bundle path",
    )
    expected_bundle_path = (
        DEFAULT_ROOT_EVIDENCE_ROOT
        / plan.revision
        / plan.sha256
        / f"{bundle_sha256}.json"
    )
    if bundle_path != expected_bundle_path:
        raise RuntimeError("root preflight evidence bundle path drifted")
    bundle_raw = _read_trusted_file(
        bundle_path,
        expected_uid=0,
        expected_gid=0,
        allowed_modes=frozenset({0o400}),
        maximum=16 * 1024 * 1024,
    )
    bundle = _decode_strict_json(bundle_raw, label="root preflight evidence bundle")
    if _sha256_bytes(_canonical_bytes(bundle)) != bundle_sha256:
        raise RuntimeError("root preflight evidence bundle digest drifted")
    directory = _plan_evidence_directory(plan) / "root-preflight"
    _ensure_root_directory(directory)
    target = directory / "root-preflight.json"
    _install_exact_bytes(target, canonical, uid=0, gid=0, mode=0o400)
    return {
        "path": str(target),
        "sha256": observed,
        "external_iam_receipt_sha256": _digest(
            value.get("external_iam_receipt_sha256"),
            "root preflight external IAM receipt sha256",
        ),
        "host_preparation_receipt_sha256": _digest(
            value.get("host_preparation_receipt_sha256"),
            "root preflight host preparation receipt sha256",
        ),
        "evidence_bundle": {
            "path": str(bundle_path),
            "sha256": bundle_sha256,
            "mode": "0400",
            "owner_uid": 0,
            "group_gid": 0,
        },
        "mode": "0400",
        "owner_uid": 0,
        "group_gid": 0,
    }


def _validate_archived_root_preflight(
    plan: ActivationPlan,
    value: Any,
) -> Mapping[str, Any]:
    archive = _strict_mapping(
        value,
        fields=frozenset({
            "path",
            "sha256",
            "external_iam_receipt_sha256",
            "host_preparation_receipt_sha256",
            "evidence_bundle",
            "mode",
            "owner_uid",
            "group_gid",
        }),
        label="archived root preflight evidence",
    )
    expected_path = (
        _plan_evidence_directory(plan) / "root-preflight/root-preflight.json"
    )
    if (
        archive["path"] != str(expected_path)
        or archive["mode"] != "0400"
        or archive["owner_uid"] != 0
        or archive["group_gid"] != 0
    ):
        raise RuntimeError("archived root preflight binding drifted")
    raw = _read_trusted_file(
        expected_path,
        expected_uid=0,
        expected_gid=0,
        allowed_modes=frozenset({0o400}),
        maximum=_MAX_MANIFEST_BYTES,
    )
    receipt = _decode_strict_json(raw, label="archived root preflight receipt")
    if (
        _sha256_bytes(_canonical_bytes(receipt)) != archive["sha256"]
        or receipt.get("external_iam_receipt_sha256")
        != archive["external_iam_receipt_sha256"]
        or receipt.get("host_preparation_receipt_sha256")
        != archive["host_preparation_receipt_sha256"]
    ):
        raise RuntimeError("archived root preflight receipt drifted")
    bundle = _strict_mapping(
        archive["evidence_bundle"],
        fields=frozenset({"path", "sha256", "mode", "owner_uid", "group_gid"}),
        label="archived root evidence bundle",
    )
    bundle_sha = _digest(bundle["sha256"], "archived root evidence bundle")
    expected_bundle = (
        DEFAULT_ROOT_EVIDENCE_ROOT / plan.revision / plan.sha256 / f"{bundle_sha}.json"
    )
    if (
        bundle["path"] != str(expected_bundle)
        or receipt.get("evidence_bundle_path") != str(expected_bundle)
        or receipt.get("evidence_bundle_sha256") != bundle_sha
        or bundle["mode"] != "0400"
        or bundle["owner_uid"] != 0
        or bundle["group_gid"] != 0
    ):
        raise RuntimeError("archived root evidence bundle binding drifted")
    bundle_raw = _read_trusted_file(
        expected_bundle,
        expected_uid=0,
        expected_gid=0,
        allowed_modes=frozenset({0o400}),
        maximum=16 * 1024 * 1024,
    )
    if _sha256_bytes(bundle_raw) != bundle_sha:
        raise RuntimeError("archived root evidence bundle file drifted")
    return receipt


def _new_failure_receipt_path(plan: ActivationPlan) -> Path:
    directory = _plan_evidence_directory(plan) / "failures"
    _ensure_root_directory(directory)
    attempt = f"{time.monotonic_ns()}-{os.getpid()}"
    if re.fullmatch(r"[0-9]+-[0-9]+", attempt) is None:
        raise RuntimeError("activation failure attempt identity is invalid")
    return directory / f"failure-{attempt}.json"


def _success_receipt_path(
    plan: ActivationPlan,
    *,
    create_parent: bool = True,
) -> Path:
    directory = _plan_evidence_directory(plan) / "success"
    if create_parent:
        _ensure_root_directory(directory)
    return directory / "activation.json"


def _load_existing_success_receipt(
    plan: ActivationPlan,
    *,
    runner: Runner,
) -> Mapping[str, Any] | None:
    path = _success_receipt_path(plan, create_parent=False)
    if not os.path.lexists(path):
        return None
    raw = _read_trusted_file(
        path,
        expected_uid=0,
        expected_gid=0,
        allowed_modes=frozenset({0o400}),
        maximum=_MAX_MANIFEST_BYTES,
    )
    value = _decode_strict_json(raw, label="activation success receipt")
    fields = frozenset({
        "schema",
        "revision",
        "activation_plan_sha256",
        "approved_plan_sha256",
        "native_observation_plan_sha256",
        "native_observation_receipt_sha256",
        "owner_approval_receipt_sha256",
        "owner_approval_receipt",
        "external_iam_evidence",
        "read_only_preflight",
        "projection_export",
        "live_preflight",
        "services_stopped",
        "discord_started",
        "completed_at_unix",
        "activation_receipt_path",
        "receipt_sha256",
    })
    _strict_mapping(value, fields=fields, label="activation success receipt")
    unsigned = {
        name: copy.deepcopy(item)
        for name, item in value.items()
        if name != "receipt_sha256"
    }
    if (
        value["schema"] != ACTIVATION_RECEIPT_SCHEMA
        or value["revision"] != plan.revision
        or value["activation_plan_sha256"] != plan.sha256
        or value["approved_plan_sha256"] != plan.sha256
        or value["native_observation_plan_sha256"]
        != plan.digests.native_observation_plan_sha256
        or value["native_observation_receipt_sha256"]
        != plan.digests.native_observation_receipt_sha256
        or value["services_stopped"] is not True
        or value["discord_started"] is not False
        or value["activation_receipt_path"] != str(path)
        or value["receipt_sha256"] != _sha256_json(unsigned)
    ):
        raise RuntimeError("existing activation success receipt conflicts with plan")
    approval = OwnerApprovalReceipt.from_mapping(value["owner_approval_receipt"])
    approval.require(
        scope="activation",
        plan_sha256=plan.sha256,
        now_unix=value["completed_at_unix"],
    )
    if approval.sha256 != value["owner_approval_receipt_sha256"]:
        raise RuntimeError("existing activation owner approval digest drifted")
    external = _validate_plan_external_iam_evidence(
        plan,
        value["external_iam_evidence"],
    )
    if external.value.get("source_approval_sha256") != approval.sha256:
        raise RuntimeError("existing activation IAM approval chain drifted")
    preflight = value["read_only_preflight"]
    if (
        not isinstance(preflight, Mapping)
        or preflight.get("ok") is not True
        or preflight.get("activation_plan_sha256") != plan.sha256
        or not isinstance(preflight.get("evidence"), Mapping)
        or preflight["evidence"].get("report_sha256") != preflight.get("report_sha256")
    ):
        raise RuntimeError("existing activation preflight evidence drifted")
    preflight_digest = _digest(
        preflight["report_sha256"],
        "existing activation preflight report",
    )
    preflight_path = (
        _plan_evidence_directory(plan) / "preflights" / f"{preflight_digest}.json"
    )
    if preflight["evidence"].get("path") != str(preflight_path):
        raise RuntimeError("existing activation preflight path drifted")
    archived_raw = _read_trusted_file(
        preflight_path,
        expected_uid=0,
        expected_gid=0,
        allowed_modes=frozenset({0o400}),
        maximum=_MAX_MANIFEST_BYTES,
    )
    if _sha256_bytes(archived_raw) != preflight["evidence"].get("file_sha256"):
        raise RuntimeError("archived activation preflight file digest drifted")
    archived_preflight = _decode_strict_json(
        archived_raw,
        label="archived activation preflight",
    )
    expected_preflight = {
        name: copy.deepcopy(item)
        for name, item in preflight.items()
        if name != "evidence"
    }
    if archived_preflight != expected_preflight:
        raise RuntimeError("archived activation preflight content drifted")
    live = value["live_preflight"]
    archived_root = None
    if isinstance(live, Mapping):
        archived_root = live.get("archive")
    if (
        not isinstance(live, Mapping)
        or not isinstance(archived_root, Mapping)
        or archived_root.get("external_iam_receipt_sha256") != external.sha256
        or archived_root.get("host_preparation_receipt_sha256")
        != plan.native_observation_receipt["host_preparation_receipt_sha256"]
    ):
        raise RuntimeError("root preflight did not bind archived external IAM receipt")
    _validate_archived_root_preflight(plan, archived_root)
    _require_off_disabled(GATEWAY_UNIT, runner=runner)
    _require_off_disabled(WRITER_UNIT, runner=runner)
    _require_off_disabled(EXPORTER_UNIT, runner=runner, absent=True)
    _require_off_disabled(DISCORD_UNIT, runner=runner, absent=True)
    return value


def _failure_value(
    plan: ActivationPlan,
    *,
    stage: str,
    error: BaseException,
    owner_approval_receipt: OwnerApprovalReceipt,
    external_iam_evidence: Mapping[str, Any],
    read_only_preflight: Mapping[str, Any],
) -> dict[str, Any]:
    message = f"{type(error).__name__}:{error}"
    return {
        "schema": ACTIVATION_FAILURE_SCHEMA,
        "revision": plan.revision,
        "activation_plan_sha256": plan.sha256,
        "approved_plan_sha256": plan.sha256,
        "owner_approval_receipt_sha256": owner_approval_receipt.sha256,
        "owner_approval_receipt": owner_approval_receipt.to_mapping(),
        "external_iam_evidence": copy.deepcopy(dict(external_iam_evidence)),
        "read_only_preflight": copy.deepcopy(dict(read_only_preflight)),
        "stage": stage,
        "error_type": type(error).__name__,
        "error_sha256": _sha256_bytes(message.encode("utf-8", errors="replace")),
        "failed_at_unix": int(time.time()),
        "quarantined": True,
    }


def _native_failure_value(
    plan: NativeObservationPlan,
    *,
    stage: str,
    error: BaseException,
    owner_approval_receipt: OwnerApprovalReceipt,
    external_iam_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    message = f"{type(error).__name__}:{error}"
    return {
        "schema": ACTIVATION_FAILURE_SCHEMA,
        "revision": plan.value["revision"],
        "native_observation_plan_sha256": plan.sha256,
        "owner_approval_receipt_sha256": owner_approval_receipt.sha256,
        "owner_approval_receipt": owner_approval_receipt.to_mapping(),
        "external_iam_evidence": copy.deepcopy(dict(external_iam_evidence)),
        "stage": stage,
        "error_type": type(error).__name__,
        "error_sha256": _sha256_bytes(message.encode("utf-8", errors="replace")),
        "failed_at_unix": int(time.time()),
        "quarantined": True,
    }


def _seal_activation_failure(
    plan: ActivationPlan,
    *,
    stage: str,
    error: BaseException,
    owner_approval_receipt: OwnerApprovalReceipt,
    external_iam_evidence: Mapping[str, Any],
    read_only_preflight: Mapping[str, Any],
) -> None:
    failure = _failure_value(
        plan,
        stage=stage,
        error=error,
        owner_approval_receipt=owner_approval_receipt,
        external_iam_evidence=external_iam_evidence,
        read_only_preflight=read_only_preflight,
    )
    receipt_errors: list[BaseException] = []
    failure_path = _new_failure_receipt_path(plan)
    failure = {**failure, "failure_receipt_path": str(failure_path)}
    for path in (failure_path, plan.paths.quarantine_path):
        try:
            _write_root_receipt(path, failure)
        except Exception as exc:
            receipt_errors.append(exc)
    if receipt_errors:
        raise ExceptionGroup(
            "activation failed and quarantine evidence could not be sealed",
            [error, *receipt_errors],
        ) from None
    raise RuntimeError(
        "activation failed closed; root failure receipt and quarantine sealed"
    ) from error


def _native_failure_path(plan: NativeObservationPlan) -> Path:
    directory = (
        DEFAULT_EVIDENCE_ROOT
        / "native-plans"
        / plan.value["revision"]
        / plan.sha256
        / "failures"
    )
    _ensure_root_directory(directory)
    attempt = f"{time.monotonic_ns()}-{os.getpid()}"
    return directory / f"failure-{attempt}.json"


def _native_stage_path(plan: NativeObservationPlan) -> Path:
    return (
        DEFAULT_NATIVE_OBSERVATION_STAGE_ROOT
        / plan.value["revision"]
        / plan.sha256
        / "native-observation-stage.json"
    )


def _native_receipt_path(plan: NativeObservationPlan) -> Path:
    return (
        DEFAULT_NATIVE_OBSERVATION_EVIDENCE_ROOT
        / plan.value["revision"]
        / plan.sha256
        / "native-observation.json"
    )


def _host_preparation_path(plan: NativeObservationPlan) -> Path:
    return (
        DEFAULT_NATIVE_OBSERVATION_EVIDENCE_ROOT
        / plan.value["revision"]
        / plan.sha256
        / "host-preparation.json"
    )


def _record_host_preparation(
    plan: NativeObservationPlan,
    host_state: Mapping[str, Any],
    *,
    owner_approval_receipt: OwnerApprovalReceipt,
) -> Mapping[str, Any]:
    path = _host_preparation_path(plan)
    value = {
        "schema": "muncho-writer-host-preparation.v1",
        "revision": plan.value["revision"],
        "native_observation_plan_sha256": plan.sha256,
        "owner_approval_receipt_sha256": owner_approval_receipt.sha256,
        "changed": host_state.get("changed"),
        "before": copy.deepcopy(host_state.get("before")),
        "after": copy.deepcopy(host_state.get("after")),
        "prepared_at_unix": int(time.time()),
        "receipt_path": str(path),
    }
    value["receipt_sha256"] = _sha256_json(value)
    if os.path.lexists(path):
        raw = _read_trusted_file(
            path,
            expected_uid=0,
            expected_gid=0,
            allowed_modes=frozenset({0o400}),
            maximum=_MAX_MANIFEST_BYTES,
        )
        existing = _decode_strict_json(raw, label="host preparation receipt")
        if (
            existing.get("schema") != value["schema"]
            or existing.get("revision") != value["revision"]
            or existing.get("native_observation_plan_sha256") != plan.sha256
            or existing.get("owner_approval_receipt_sha256")
            != owner_approval_receipt.sha256
            or existing.get("after") != host_state.get("after")
        ):
            raise RuntimeError("existing host preparation receipt conflicts")
        unsigned = {
            name: copy.deepcopy(item)
            for name, item in existing.items()
            if name != "receipt_sha256"
        }
        if existing.get("receipt_sha256") != _sha256_json(unsigned):
            raise RuntimeError("existing host preparation receipt digest drifted")
        return existing
    _ensure_root_directory(path.parent)
    _write_root_receipt(path, value)
    return value


def _load_host_preparation_receipt(
    plan: NativeObservationPlan,
) -> Mapping[str, Any]:
    path = _host_preparation_path(plan)
    raw = _read_trusted_file(
        path,
        expected_uid=0,
        expected_gid=0,
        allowed_modes=frozenset({0o400}),
        maximum=_MAX_MANIFEST_BYTES,
    )
    value = _decode_strict_json(raw, label="host preparation receipt")
    unsigned = {
        name: copy.deepcopy(item)
        for name, item in value.items()
        if name != "receipt_sha256"
    }
    if (
        value.get("schema") != "muncho-writer-host-preparation.v1"
        or value.get("revision") != plan.value["revision"]
        or value.get("native_observation_plan_sha256") != plan.sha256
        or value.get("receipt_path") != str(path)
        or value.get("receipt_sha256") != _sha256_json(unsigned)
    ):
        raise RuntimeError("existing host preparation receipt drifted")
    return value


def _record_host_preparation_failure(
    plan: NativeObservationPlan,
    host_state: Mapping[str, Any],
    *,
    owner_approval_receipt: OwnerApprovalReceipt,
    error: BaseException,
) -> Mapping[str, Any]:
    directory = _host_preparation_path(plan).parent / "host-preparation-failures"
    _ensure_root_directory(directory)
    attempt = f"{time.monotonic_ns()}-{os.getpid()}"
    path = directory / f"failure-{attempt}.json"
    message = f"{type(error).__name__}:{error}"
    value = {
        "schema": "muncho-writer-host-preparation-failure.v1",
        "revision": plan.value["revision"],
        "native_observation_plan_sha256": plan.sha256,
        "owner_approval_receipt_sha256": owner_approval_receipt.sha256,
        "changed": host_state.get("changed"),
        "before": copy.deepcopy(host_state.get("before")),
        "after": copy.deepcopy(host_state.get("after")),
        "error_type": type(error).__name__,
        "error_sha256": _sha256_bytes(message.encode("utf-8", errors="replace")),
        "failed_at_unix": int(time.time()),
        "receipt_path": str(path),
    }
    value["receipt_sha256"] = _sha256_json(value)
    _write_root_receipt(path, value)
    return value


class NativeObservationExecutor:
    """Run the separately approved first-start observation and always stop."""

    def __init__(self, plan: NativeObservationPlan, *, runner: Runner = _runner):
        if not isinstance(plan, NativeObservationPlan):
            raise TypeError("native observation plan is required")
        self.plan = plan
        self.runner = runner
        self.stage = "loaded"
        self.host_preparation_receipt: Mapping[str, Any] = {}
        self.external_iam_evidence: Mapping[str, Any] = {}
        self.idempotent = False

    def _command(
        self,
        argv: tuple[str, ...],
        label: str,
        *,
        timeout: int = 60,
    ) -> subprocess.CompletedProcess[bytes]:
        return _run(
            Command(argv, timeout_seconds=timeout),
            runner=self.runner,
            label=label,
        )

    def _stop(self, unit: str) -> None:
        self._command((SYSTEMCTL, "stop", unit), f"stop {unit}", timeout=60)

    def observe(
        self,
        *,
        approved_plan_sha256: str,
        owner_approval_receipt: OwnerApprovalReceipt,
        external_iam_receipt_path: str | os.PathLike[str],
    ) -> NativeObservationReceipt:
        _digest(approved_plan_sha256, "approved native observation plan sha256")
        if approved_plan_sha256 != self.plan.sha256:
            raise PermissionError(
                "native observation approval digest does not match plan"
            )
        if not isinstance(owner_approval_receipt, OwnerApprovalReceipt):
            raise PermissionError("native owner approval receipt is required")
        owner_approval_receipt.require(
            scope="native_observation",
            plan_sha256=self.plan.sha256,
            now_unix=int(time.time()),
        )
        self.idempotent = False
        with _host_activation_lock():
            return self._observe_locked(
                approved_plan_sha256=approved_plan_sha256,
                owner_approval_receipt=owner_approval_receipt,
                external_iam_receipt_path=external_iam_receipt_path,
            )

    def _observe_locked(
        self,
        *,
        approved_plan_sha256: str,
        owner_approval_receipt: OwnerApprovalReceipt,
        external_iam_receipt_path: str | os.PathLike[str],
    ) -> NativeObservationReceipt:
        _digest(approved_plan_sha256, "approved native observation plan sha256")
        if approved_plan_sha256 != self.plan.sha256:
            raise PermissionError(
                "native observation approval digest does not match plan"
            )
        if (
            not isinstance(owner_approval_receipt, OwnerApprovalReceipt)
            or owner_approval_receipt.value.get("scope") != "native_observation"
            or owner_approval_receipt.value.get("plan_sha256") != self.plan.sha256
        ):
            raise PermissionError("native owner approval receipt is not exact")
        owner_approval_receipt.require(
            scope="native_observation",
            plan_sha256=self.plan.sha256,
            now_unix=int(time.time()),
        )
        _require_root_linux()
        if os.path.lexists(DEFAULT_QUARANTINE_PATH):
            raise PermissionError("native observation is blocked by quarantine")

        permanent_units_installed = False
        daemon_reloaded = False
        lifecycle_start_attempted = False
        host_mutation_attempted = False
        stage_written = False
        host_receipt: Mapping[str, Any] = {}
        receipt: NativeObservationReceipt | None = None
        idempotent = False
        primary: BaseException | None = None
        failed_stage = ""
        try:
            self.stage = "external_iam"
            iam_receipt = _load_lifecycle_external_iam(
                self.plan,
                path_value=external_iam_receipt_path,
                minimum_remaining_seconds=720,
                expected_source_approval_sha256=owner_approval_receipt.sha256,
            )
            self.stage = "read_only_preflight"
            if os.path.lexists(_native_receipt_path(self.plan)):
                existing_receipt = _load_existing_native_receipt(
                    self.plan,
                    runner=self.runner,
                )
            else:
                _verify_native_preflight_inputs(
                    self.plan,
                    runner=self.runner,
                    require_installed=False,
                    require_original_boot=True,
                )
                if os.path.lexists(_native_stage_path(self.plan)):
                    raise RuntimeError(
                        "incomplete native observation stage requires quarantine"
                    )
                existing_receipt = None
            self.external_iam_evidence = _archive_plan_external_iam(
                self.plan,
                iam_receipt,
                live_path=external_iam_receipt_path,
            )
            if existing_receipt is not None:
                self.host_preparation_receipt = _load_host_preparation_receipt(
                    self.plan
                )
                _owner, _historical_iam, historical_evidence = (
                    _load_durable_native_authority_chain(
                        self.plan,
                        existing_receipt,
                    )
                )
                self.external_iam_evidence = historical_evidence
                receipt = existing_receipt
                idempotent = True
                self.idempotent = True
            else:
                self.stage = "prepare_host_identities"
                _require_lifecycle_owner_approval(
                    owner_approval_receipt,
                    scope="native_observation",
                    plan_sha256=self.plan.sha256,
                )
                host_mutation_attempted = True
                host_before = _host_identity_snapshot()
                try:
                    host_receipt = prepare_canary_host_identities(
                        self.plan,
                        approved_plan_sha256=approved_plan_sha256,
                        owner_approval_receipt=owner_approval_receipt,
                        runner=self.runner,
                    )
                except BaseException as host_error:
                    host_after: Mapping[str, Any]
                    try:
                        host_after = _host_identity_snapshot()
                    except BaseException as snapshot_error:
                        host_after = {
                            "snapshot_error_sha256": _sha256_bytes(
                                f"{type(snapshot_error).__name__}:{snapshot_error}".encode(
                                    "utf-8", errors="replace"
                                )
                            )
                        }
                    host_receipt = {
                        "changed": host_before != host_after,
                        "before": host_before,
                        "after": host_after,
                        "failed": True,
                    }
                    self.host_preparation_receipt = _record_host_preparation_failure(
                        self.plan,
                        host_receipt,
                        owner_approval_receipt=owner_approval_receipt,
                        error=host_error,
                    )
                    raise
                self.host_preparation_receipt = _record_host_preparation(
                    self.plan,
                    host_receipt,
                    owner_approval_receipt=owner_approval_receipt,
                )
                self.stage = "install"
                _require_lifecycle_owner_approval(
                    owner_approval_receipt,
                    scope="native_observation",
                    plan_sha256=self.plan.sha256,
                )
                _install_native_observation_artifacts(self.plan)
                permanent_units_installed = True
                _prepare_native_runtime_directories()
                self._command(
                    (
                        SYSTEMD_ANALYZE,
                        "verify",
                        str(DEFAULT_WRITER_UNIT_PATH),
                        str(DEFAULT_GATEWAY_UNIT_PATH),
                    ),
                    "native systemd unit verification",
                )
                self._command((SYSTEMCTL, "daemon-reload"), "native daemon reload")
                daemon_reloaded = True
                _require_off_disabled(WRITER_UNIT, runner=self.runner)
                _require_off_disabled(GATEWAY_UNIT, runner=self.runner)
                _require_off_disabled(EXPORTER_UNIT, runner=self.runner, absent=True)
                _require_off_disabled(DISCORD_UNIT, runner=self.runner, absent=True)
                self.stage = "refresh_external_iam"
                iam_receipt = _load_lifecycle_external_iam(
                    self.plan,
                    path_value=external_iam_receipt_path,
                    minimum_remaining_seconds=360,
                    expected_source_approval_sha256=(
                        owner_approval_receipt.sha256
                    ),
                )
                self.external_iam_evidence = _archive_plan_external_iam(
                    self.plan,
                    iam_receipt,
                    live_path=external_iam_receipt_path,
                )
                self.stage = "start_writer"
                _require_lifecycle_owner_approval(
                    owner_approval_receipt,
                    scope="native_observation",
                    plan_sha256=self.plan.sha256,
                )
                lifecycle_start_attempted = True
                self._command(
                    (SYSTEMCTL, "start", WRITER_UNIT), "native start writer", timeout=90
                )
                _require_active(WRITER_UNIT, runner=self.runner)
                self.stage = "start_gateway"
                _require_lifecycle_owner_approval(
                    owner_approval_receipt,
                    scope="native_observation",
                    plan_sha256=self.plan.sha256,
                )
                self._command(
                    (SYSTEMCTL, "start", GATEWAY_UNIT),
                    "native start gateway",
                    timeout=90,
                )
                _require_active(GATEWAY_UNIT, runner=self.runner)
                self.stage = "collect_native"
                stage_path = _native_stage_path(self.plan)
                _ensure_root_directory(stage_path.parent)
                write_native_observation_stage(
                    stage_path,
                    plan=self.plan,
                    approved_plan_sha256=approved_plan_sha256,
                    owner_approval_receipt=owner_approval_receipt.to_mapping(),
                    host_preparation_receipt_sha256=self.host_preparation_receipt[
                        "receipt_sha256"
                    ],
                    external_iam_receipt_sha256=iam_receipt.sha256,
                )
                stage_written = True
        except BaseException as exc:
            primary = exc
            failed_stage = self.stage
        finally:
            self.stage = "stop_services"
            cleanup: list[BaseException] = []
            if lifecycle_start_attempted:
                for unit in (GATEWAY_UNIT, WRITER_UNIT):
                    try:
                        self._stop(unit)
                    except BaseException as exc:
                        cleanup.append(exc)
            try:
                for unit in (GATEWAY_UNIT, WRITER_UNIT):
                    if permanent_units_installed and daemon_reloaded:
                        _require_off_disabled(unit, runner=self.runner)
                    else:
                        _require_off_or_absent(unit, runner=self.runner)
                _require_off_disabled(EXPORTER_UNIT, runner=self.runner, absent=True)
                _require_off_disabled(DISCORD_UNIT, runner=self.runner, absent=True)
            except BaseException as exc:
                cleanup.append(exc)
            if cleanup:
                if not failed_stage:
                    failed_stage = self.stage
                primary = ExceptionGroup(
                    "native observation cleanup failed",
                    ([primary] if primary is not None else []) + cleanup,
                )
        if primary is None and stage_written:
            try:
                self.stage = "finalize_native"
                stage_path = _native_stage_path(self.plan)
                receipt_path = _native_receipt_path(self.plan)
                _ensure_root_directory(receipt_path.parent)
                receipt = finalize_native_observation_stage(
                    stage_path,
                    receipt_path,
                    approved_plan_sha256=approved_plan_sha256,
                    owner_approval_receipt=owner_approval_receipt.to_mapping(),
                    host_preparation_receipt_sha256=self.host_preparation_receipt[
                        "receipt_sha256"
                    ],
                    external_iam_receipt_sha256=iam_receipt.sha256,
                )
            except BaseException as exc:
                primary = exc
                failed_stage = self.stage
        if (
            isinstance(primary, RetryableExternalIAMRefreshRequired)
            and not lifecycle_start_attempted
        ):
            raise RuntimeError(
                "native observation safely stopped before start; renew IAM and retry"
            ) from primary
        if (
            isinstance(primary, RetryableOwnerApprovalRefreshRequired)
            and not host_mutation_attempted
            and not permanent_units_installed
            and not lifecycle_start_attempted
        ):
            raise RuntimeError(
                "native observation safely stopped before mutation; renew owner "
                "approval and retry"
            ) from primary
        if primary is not None or receipt is None:
            error = primary or RuntimeError("native observation produced no receipt")
            failure_path = _native_failure_path(self.plan)
            failure = _native_failure_value(
                self.plan,
                stage=failed_stage or self.stage,
                error=error,
                owner_approval_receipt=owner_approval_receipt,
                external_iam_evidence=self.external_iam_evidence,
            )
            failure = {
                **failure,
                "failure_receipt_path": str(failure_path),
                "host_preparation_sha256": _sha256_json(host_receipt),
                "host_preparation_evidence": copy.deepcopy(
                    dict(self.host_preparation_receipt)
                ),
                "stage_preserved": stage_written,
            }
            receipt_errors: list[BaseException] = []
            for path in (failure_path, DEFAULT_QUARANTINE_PATH):
                try:
                    _ensure_root_directory(path.parent)
                    _write_root_receipt(path, failure)
                except BaseException as exc:
                    receipt_errors.append(exc)
            if receipt_errors:
                raise ExceptionGroup(
                    "native observation failed and evidence could not be sealed",
                    [error, *receipt_errors],
                ) from None
            raise RuntimeError(
                "native observation failed closed; services stopped and quarantined"
            ) from error
        if idempotent:
            NativeObservationReceipt.from_mapping(
                receipt.to_mapping(),
                expected_plan_sha256=approved_plan_sha256,
            )
            if (
                current_host_identity_sha256()
                != self.plan.value["host_identity_sha256"]
            ):
                raise RuntimeError("native receipt was replayed on another host")
        else:
            NativeObservationReceipt.from_mapping(
                receipt.to_mapping(),
                expected_plan_sha256=approved_plan_sha256,
                current_boot_id_sha256=_current_boot_id_sha256(),
                current_boottime_ns=_current_boottime_ns(),
            )
        return receipt


class ActivationReadOnlyPreflightError(RuntimeError):
    def __init__(self, report: Mapping[str, Any]):
        super().__init__("activation read-only preflight failed")
        self.report = copy.deepcopy(dict(report))


def _run_checked_preflight(
    plan: ActivationPlan,
    checks: Sequence[tuple[str, Callable[[], None]]],
) -> Mapping[str, Any]:
    values: list[dict[str, Any]] = []
    failed: list[str] = []
    for name, action in checks:
        try:
            action()
        except Exception as exc:
            message = f"{type(exc).__name__}:{exc}"
            values.append({
                "name": name,
                "passed": False,
                "error_type": type(exc).__name__,
                "error_sha256": _sha256_bytes(
                    message.encode("utf-8", errors="replace")
                ),
            })
            failed.append(name)
        else:
            values.append({"name": name, "passed": True})
    report: dict[str, Any] = {
        "schema": "muncho-writer-activation-read-only-preflight.v1",
        "ok": not failed,
        "revision": plan.revision,
        "activation_plan_sha256": plan.sha256,
        "checks": values,
        "failed_checks": failed,
        "checked_at_unix": int(time.time()),
    }
    report["report_sha256"] = _sha256_json(report)
    if failed:
        raise ActivationReadOnlyPreflightError(report)
    return report


def _seal_activation_preflight_report(
    plan: ActivationPlan,
    report: Mapping[str, Any],
) -> Mapping[str, Any]:
    digest = _digest(report.get("report_sha256"), "activation preflight report")
    unsigned = {
        name: copy.deepcopy(item)
        for name, item in report.items()
        if name != "report_sha256"
    }
    if _sha256_json(unsigned) != digest:
        raise RuntimeError("activation preflight report digest drifted")
    path = _plan_evidence_directory(plan) / "preflights" / f"{digest}.json"
    _ensure_root_directory(path.parent)
    payload = _canonical_bytes(report)
    _install_exact_bytes(
        path,
        payload,
        uid=0,
        gid=0,
        mode=0o400,
    )
    return {
        "path": str(path),
        "report_sha256": digest,
        "file_sha256": _sha256_bytes(payload),
        "mode": "0400",
        "owner_uid": 0,
        "group_gid": 0,
    }


def _rehash_native_receipt_external_mappings(
    receipt: NativeObservationReceipt,
) -> None:
    """Re-attest every stopped-receipt library before final host mutation."""

    if not isinstance(receipt, NativeObservationReceipt):
        raise TypeError("native observation receipt is required")
    plan = NativeObservationPlan.from_mapping(receipt.value["plan"])
    policy = plan.value["native_discovery_policy"]
    roots = tuple(Path(value) for value in policy["allowed_roots"])
    artifact_root = Path(plan.value["artifact_root"])
    observed = receipt.value["observation"]
    cache: dict[Path, str] = {}
    for label in ("gateway", "writer"):
        mappings = observed[f"{label}_service"]["external_native_mappings"]
        if len(mappings) > policy["maximum_mappings"]:
            raise RuntimeError("native receipt mapping count exceeds policy")
        paths = [Path(item["path"]) for item in mappings]
        if paths != sorted(set(paths)):
            raise RuntimeError("native receipt mapping paths are not exact sorted set")
        for item, path in zip(mappings, paths, strict=True):
            if (
                path == artifact_root
                or artifact_root in path.parents
                or not any(path == root or root in path.parents for root in roots)
            ):
                raise RuntimeError("native receipt mapping escaped discovery policy")
            before = os.lstat(path)
            mode = stat.S_IMODE(before.st_mode)
            if (
                stat.S_ISLNK(before.st_mode)
                or not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or before.st_uid != policy["required_owner_uid"]
                or before.st_gid != policy["required_owner_gid"]
                or mode & 0o222
                or _list_xattrs(path)
                or before.st_size < 1
                or before.st_size > _MAX_RELEASE_FILE_BYTES * 8
            ):
                raise RuntimeError("native receipt mapping protection drifted")
            digest = cache.get(path)
            if digest is None:
                raw = _read_trusted_file(
                    path,
                    expected_uid=policy["required_owner_uid"],
                    expected_gid=policy["required_owner_gid"],
                    allowed_modes=frozenset({mode}),
                    maximum=_MAX_RELEASE_FILE_BYTES * 8,
                )
                digest = _sha256_bytes(raw)
                cache[path] = digest
            if digest != item["sha256"]:
                raise RuntimeError("native receipt mapping digest drifted")


def _verify_final_native_host(plan: ActivationPlan, *, runner: Runner) -> None:
    receipt = NativeObservationReceipt.from_mapping(
        plan.native_observation_receipt,
        expected_plan_sha256=plan.digests.native_observation_plan_sha256,
    )
    native_plan = NativeObservationPlan.from_mapping(receipt.value["plan"])
    if current_host_identity_sha256() != native_plan.value["host_identity_sha256"]:
        raise RuntimeError("final activation native receipt belongs to another host")
    disk = _load_existing_native_receipt(native_plan, runner=runner)
    if (
        disk is None
        or disk.sha256 != plan.digests.native_observation_receipt_sha256
        or disk.to_mapping() != receipt.to_mapping()
    ):
        raise RuntimeError("durable native observation receipt drifted")
    _rehash_native_receipt_external_mappings(receipt)


def _verify_final_artifacts(plan: ActivationPlan) -> None:
    _verify_release_tree(plan)
    required_installed = {
        "writer_unit",
        "gateway_unit",
        "writer_config",
        "gateway_config",
    }
    for name, artifact in plan.install_artifacts.items():
        payload = _artifact_payload(plan, name)
        if (
            len(payload) > artifact.maximum_bytes
            or _sha256_bytes(payload) != artifact.sha256
        ):
            raise RuntimeError(f"final activation {name} source drifted")
        if os.path.lexists(artifact.target_path):
            installed = _read_trusted_file(
                artifact.target_path,
                expected_uid=artifact.uid,
                expected_gid=artifact.gid,
                allowed_modes=frozenset({artifact.mode}),
                maximum=artifact.maximum_bytes,
            )
            if installed != payload or _sha256_bytes(installed) != artifact.sha256:
                raise RuntimeError(f"final activation {name} install collision drifted")
        elif name in required_installed:
            raise RuntimeError(f"final activation {name} native install is absent")
        else:
            _validate_root_parent_chain(artifact.target_path.parent)
    ca = _read_trusted_file(
        plan.paths.database_ca_path,
        expected_uid=0,
        expected_gid=CANARY_WRITER_GID,
        allowed_modes=frozenset({0o400, 0o440, 0o444}),
        maximum=_MAX_CONFIG_BYTES,
    )
    if _sha256_bytes(ca) != plan.digests.database_ca_sha256:
        raise RuntimeError("final activation database CA drifted")
    connection = plan.deployment_manifest["snapshot_template"]["database"]["connection"]
    _verify_database_read_only(
        config_path=plan.paths.writer_config_path,
        expected_database={
            "ip_network": f"{connection['host']}/32",
            "tls_server_name": connection["tls_server_name"],
            "ca_path": str(plan.paths.database_ca_path),
        },
    )


def _verify_final_stopped_boundary(plan: ActivationPlan, *, runner: Runner) -> None:
    if not _host_identities_are_exact(_host_identity_snapshot()):
        raise RuntimeError("final activation host identities drifted")
    _require_off_disabled(WRITER_UNIT, runner=runner)
    _require_off_disabled(GATEWAY_UNIT, runner=runner)
    _require_off_disabled(EXPORTER_UNIT, runner=runner, absent=True)
    _require_off_disabled(DISCORD_UNIT, runner=runner, absent=True)
    discord = plan.deployment_manifest["snapshot_template"]["discord_edge"]
    for name in ("config_path", "token_path", "socket_path"):
        if os.path.lexists(discord[name]):
            raise RuntimeError(f"final Discord {name} must remain absent")
    helper = Path(LEGACY_CLOUD_SQL_HELPER_PATH)
    if os.path.lexists(helper) or os.path.lexists(helper.parent):
        raise RuntimeError("final legacy helper authority must remain absent")
    if os.path.lexists(plan.paths.root_receipt_path):
        raise RuntimeError(
            "live root preflight receipt must be absent before activation"
        )


def activation_read_only_preflight(
    plan: ActivationPlan,
    *,
    owner_approval_receipt: OwnerApprovalReceipt,
    runner: Runner = _runner,
) -> tuple[Mapping[str, Any], Mapping[str, Any] | None, ExternalIAMReceipt | None]:
    """Validate the complete current host without mutating it."""

    if not isinstance(owner_approval_receipt, OwnerApprovalReceipt):
        raise PermissionError("activation preflight owner approval is required")
    owner_approval_receipt.require(
        scope="activation",
        plan_sha256=plan.sha256,
        now_unix=int(time.time()),
    )
    _require_root_linux()
    existing: Mapping[str, Any] | None = None
    iam_receipt: ExternalIAMReceipt | None = None

    def quarantine_absent() -> None:
        if os.path.lexists(plan.paths.quarantine_path):
            raise PermissionError("activation is blocked by root quarantine")

    def success_or_fresh_iam() -> None:
        nonlocal existing, iam_receipt
        existing = _load_existing_success_receipt(plan, runner=runner)
        if existing is None:
            native_plan = NativeObservationPlan.from_mapping(
                plan.native_observation_receipt["plan"]
            )
            iam_receipt = _load_lifecycle_external_iam(
                native_plan,
                path_value=plan.paths.external_iam_receipt_path,
                minimum_remaining_seconds=720,
                expected_source_approval_sha256=owner_approval_receipt.sha256,
            )

    try:
        report = _run_checked_preflight(
            plan,
            (
                ("quarantine.absent", quarantine_absent),
                (
                    "native_receipt.same_host_exact",
                    lambda: _verify_final_native_host(plan, runner=runner),
                ),
                ("release_config_ca_db.exact", lambda: _verify_final_artifacts(plan)),
                (
                    "services_discord_authority.stopped_exact",
                    lambda: _verify_final_stopped_boundary(plan, runner=runner),
                ),
                ("success_or_fresh_iam.exact", success_or_fresh_iam),
            ),
        )
    except ActivationReadOnlyPreflightError as exc:
        evidence = _seal_activation_preflight_report(plan, exc.report)
        raise ActivationReadOnlyPreflightError({
            **exc.report,
            "evidence": evidence,
        }) from None
    evidence = _seal_activation_preflight_report(plan, report)
    return {**report, "evidence": evidence}, existing, iam_receipt


class ActivationExecutor:
    def __init__(self, plan: ActivationPlan, *, runner: Runner = _runner):
        self.plan = plan
        self.runner = runner
        self.stage = "loaded"

    def _command(
        self, argv: tuple[str, ...], label: str, *, timeout: int = 60
    ) -> subprocess.CompletedProcess[bytes]:
        return _run(
            Command(argv, timeout_seconds=timeout),
            runner=self.runner,
            label=label,
        )

    def _daemon_reload(self) -> None:
        self._command((SYSTEMCTL, "daemon-reload"), "systemd daemon reload")

    def _stop(self, unit: str) -> None:
        self._command((SYSTEMCTL, "stop", unit), f"stop {unit}", timeout=60)

    def _verify_installed(self) -> None:
        self._command(
            (
                SYSTEMD_ANALYZE,
                "verify",
                str(self.plan.paths.writer_unit_path),
                str(self.plan.paths.gateway_unit_path),
            ),
            "systemd permanent unit verification",
        )
        self._daemon_reload()
        _require_off_disabled(WRITER_UNIT, runner=self.runner)
        _require_off_disabled(GATEWAY_UNIT, runner=self.runner)
        _require_off_disabled(EXPORTER_UNIT, runner=self.runner, absent=True)
        self._command(
            (SYSTEMD_TMPFILES, "--create", str(self.plan.paths.tmpfiles_path)),
            "writer tmpfiles creation",
        )

    def _run_projection_export(self) -> Mapping[str, Any]:
        exporter = self.plan.unit_bundle.exporter_service.encode("utf-8")
        created = False
        primary: BaseException | None = None
        try:
            created = _install_exact_bytes(
                self.plan.paths.exporter_unit_path,
                exporter,
                uid=0,
                gid=0,
                mode=0o644,
            )
            if not created:
                raise RuntimeError("temporary exporter unit unexpectedly pre-existed")
            self._command(
                (SYSTEMD_ANALYZE, "verify", str(self.plan.paths.exporter_unit_path)),
                "systemd exporter unit verification",
            )
            self._daemon_reload()
            state = _systemd_show(EXPORTER_UNIT, runner=self.runner)
            if (
                state["LoadState"] != "loaded"
                or state["ActiveState"] != "inactive"
                or state["MainPID"] != "0"
                or state["UnitFileState"] not in {"static", "disabled"}
                or state["FragmentPath"] != str(self.plan.paths.exporter_unit_path)
                or state["DropInPaths"] != ""
                or state["NeedDaemonReload"] != "no"
            ):
                raise RuntimeError("temporary exporter initial state is invalid")
            self._command(
                (SYSTEMCTL, "start", EXPORTER_UNIT),
                "temporary projection export",
                timeout=360,
            )
            completed = _systemd_show(EXPORTER_UNIT, runner=self.runner)
            if (
                completed["LoadState"] != "loaded"
                or completed["ActiveState"] != "inactive"
                or completed["SubState"] != "dead"
                or completed["MainPID"] != "0"
                or completed["FragmentPath"] != str(self.plan.paths.exporter_unit_path)
                or completed["DropInPaths"] != ""
                or completed["NeedDaemonReload"] != "no"
            ):
                raise RuntimeError("temporary exporter did not finish cleanly")
            stdout_receipt = _exporter_stdout_receipt(
                _systemd_invocation_id(EXPORTER_UNIT, runner=self.runner),
                runner=self.runner,
            )
            projection = _validate_projection(
                self.plan.paths.projection_export_path,
                self.plan.identities,
            )
            if stdout_receipt["event_count"] != projection["event_count"]:
                raise RuntimeError("projection exporter count does not match export")
            return {**projection, "stdout_receipt": dict(stdout_receipt)}
        except BaseException as exc:
            primary = exc
            raise
        finally:
            cleanup: list[BaseException] = []
            try:
                self._stop(EXPORTER_UNIT)
            except BaseException as exc:
                cleanup.append(exc)
            if created:
                try:
                    _unlink_exact(
                        self.plan.paths.exporter_unit_path,
                        uid=0,
                        gid=0,
                        mode=0o644,
                        sha256=self.plan.digests.exporter_unit_sha256,
                    )
                except BaseException as exc:
                    cleanup.append(exc)
            try:
                self._daemon_reload()
                _require_off_disabled(EXPORTER_UNIT, runner=self.runner, absent=True)
            except BaseException as exc:
                cleanup.append(exc)
            if cleanup:
                if primary is None:
                    raise ExceptionGroup("projection export cleanup failed", cleanup)
                raise ExceptionGroup(
                    "projection export and cleanup failed",
                    [primary, *cleanup],
                ) from None

    def _invalidate_root_receipt(self) -> None:
        path = self.plan.paths.root_receipt_path
        if not os.path.lexists(path):
            return
        _unlink_exact(path, uid=0, gid=0, mode=0o400)

    def _run_live_collector(self) -> Mapping[str, Any]:
        collect = self._command(
            (
                *self.plan.collector_argv,
                "--approved-plan-sha256",
                self.plan.sha256,
            ),
            "root collector",
            timeout=120,
        )
        validate = self._command(
            (
                *self.plan.validator_argv,
                "--approved-plan-sha256",
                self.plan.sha256,
            ),
            "root receipt validator",
            timeout=120,
        )
        result: dict[str, Any] = {}
        for label, raw in (("collect", collect.stdout), ("validate", validate.stdout)):
            value = _decode_strict_json(raw.strip(), label=f"{label} command receipt")
            if value.get("ok") is not True or not isinstance(
                value.get("receipt_sha256"), str
            ):
                raise RuntimeError(f"{label} command did not return success receipt")
            result[f"{label}_receipt_sha256"] = _digest(
                value["receipt_sha256"], f"{label} receipt sha256"
            )
        if result["collect_receipt_sha256"] != result["validate_receipt_sha256"]:
            raise RuntimeError("collector and validator receipt identity differ")
        result["archive"] = _archive_root_receipt(
            self.plan,
            expected_sha256=result["validate_receipt_sha256"],
        )
        return result

    def apply(
        self,
        *,
        approved_plan_sha256: str,
        owner_approval_receipt: OwnerApprovalReceipt,
    ) -> Mapping[str, Any]:
        _digest(approved_plan_sha256, "approved activation plan sha256")
        if approved_plan_sha256 != self.plan.sha256:
            raise PermissionError(
                "activation owner approval digest does not match plan"
            )
        if not isinstance(owner_approval_receipt, OwnerApprovalReceipt):
            raise PermissionError("activation owner approval receipt is required")
        owner_approval_receipt.require(
            scope="activation",
            plan_sha256=self.plan.sha256,
            now_unix=int(time.time()),
        )
        with _host_activation_lock():
            return self._apply_locked(
                approved_plan_sha256=approved_plan_sha256,
                owner_approval_receipt=owner_approval_receipt,
            )

    def _apply_locked(
        self,
        *,
        approved_plan_sha256: str,
        owner_approval_receipt: OwnerApprovalReceipt,
    ) -> Mapping[str, Any]:
        _digest(approved_plan_sha256, "approved activation plan sha256")
        if approved_plan_sha256 != self.plan.sha256:
            raise PermissionError(
                "activation owner approval digest does not match plan"
            )
        if (
            not isinstance(owner_approval_receipt, OwnerApprovalReceipt)
            or owner_approval_receipt.value.get("scope") != "activation"
            or owner_approval_receipt.value.get("plan_sha256") != self.plan.sha256
        ):
            raise PermissionError("activation owner approval receipt is not exact")
        owner_approval_receipt.require(
            scope="activation",
            plan_sha256=self.plan.sha256,
            now_unix=int(time.time()),
        )
        _require_root_linux()
        self.stage = "read_only_preflight"
        _preflight, existing, iam_receipt = activation_read_only_preflight(
            self.plan,
            owner_approval_receipt=owner_approval_receipt,
            runner=self.runner,
        )
        if existing is not None:
            return existing
        if iam_receipt is None:
            raise RuntimeError("activation preflight produced no fresh IAM receipt")
        if (
            iam_receipt.value.get("source_approval_sha256")
            != owner_approval_receipt.sha256
        ):
            raise RetryableExternalIAMRefreshRequired(
                "external IAM evidence must be refreshed for this approval"
            )
        native_plan = NativeObservationPlan.from_mapping(
            self.plan.native_observation_receipt["plan"]
        )
        external_iam_evidence = _archive_plan_external_iam(
            self.plan,
            iam_receipt,
            live_path=self.plan.paths.external_iam_receipt_path,
        )
        projection: Mapping[str, Any] = {}
        live: Mapping[str, Any] = {}
        permanent_units_installed = False
        activation_mutation_attempted = False
        service_start_attempted = False
        primary: BaseException | None = None
        failed_stage = ""
        try:
            self.stage = "install"
            _verify_release_tree(self.plan)
            _require_lifecycle_owner_approval(
                owner_approval_receipt,
                scope="activation",
                plan_sha256=self.plan.sha256,
            )
            activation_mutation_attempted = True
            _install_plan_artifacts(self.plan)
            permanent_units_installed = True
            self._verify_installed()
            self.stage = "projection_export"
            _require_lifecycle_owner_approval(
                owner_approval_receipt,
                scope="activation",
                plan_sha256=self.plan.sha256,
            )
            projection = self._run_projection_export()
            self.stage = "invalidate_old_receipt"
            self._invalidate_root_receipt()
            self.stage = "refresh_external_iam"
            iam_receipt = _load_lifecycle_external_iam(
                native_plan,
                path_value=self.plan.paths.external_iam_receipt_path,
                minimum_remaining_seconds=480,
                expected_source_approval_sha256=owner_approval_receipt.sha256,
            )
            external_iam_evidence = _archive_plan_external_iam(
                self.plan,
                iam_receipt,
                live_path=self.plan.paths.external_iam_receipt_path,
            )
            self.stage = "start_writer"
            _require_lifecycle_owner_approval(
                owner_approval_receipt,
                scope="activation",
                plan_sha256=self.plan.sha256,
            )
            service_start_attempted = True
            self._command((SYSTEMCTL, "start", WRITER_UNIT), "start writer", timeout=90)
            writer_pid = _require_active(WRITER_UNIT, runner=self.runner)
            self.stage = "start_gateway"
            _require_lifecycle_owner_approval(
                owner_approval_receipt,
                scope="activation",
                plan_sha256=self.plan.sha256,
            )
            self._command(
                (SYSTEMCTL, "start", GATEWAY_UNIT), "start gateway", timeout=90
            )
            gateway_pid = _require_active(GATEWAY_UNIT, runner=self.runner)
            self.stage = "collect_validate"
            live = {
                **self._run_live_collector(),
                "writer_main_pid": writer_pid,
                "gateway_main_pid": gateway_pid,
            }
            if (
                not isinstance(live.get("archive"), Mapping)
                or live["archive"].get("external_iam_receipt_sha256")
                != iam_receipt.sha256
                or live["archive"].get("host_preparation_receipt_sha256")
                != self.plan.native_observation_receipt[
                    "host_preparation_receipt_sha256"
                ]
            ):
                raise RuntimeError(
                    "root preflight external IAM receipt does not match archive"
                )
        except BaseException as exc:
            primary = exc
            failed_stage = self.stage
        finally:
            self.stage = "stop_services"
            cleanup: list[BaseException] = []
            if permanent_units_installed:
                try:
                    self._stop(GATEWAY_UNIT)
                except BaseException as exc:
                    cleanup.append(exc)
                try:
                    self._stop(WRITER_UNIT)
                except BaseException as exc:
                    cleanup.append(exc)
            if not (
                isinstance(primary, RetryableOwnerApprovalRefreshRequired)
                and not activation_mutation_attempted
            ):
                try:
                    self._invalidate_root_receipt()
                except BaseException as exc:
                    cleanup.append(exc)
            try:
                _require_off_disabled(GATEWAY_UNIT, runner=self.runner)
                _require_off_disabled(WRITER_UNIT, runner=self.runner)
                _require_off_disabled(EXPORTER_UNIT, runner=self.runner, absent=True)
            except BaseException as exc:
                cleanup.append(exc)
            if cleanup:
                if not failed_stage:
                    failed_stage = self.stage
                primary = ExceptionGroup(
                    "activation lifecycle cleanup failed",
                    ([primary] if primary is not None else []) + cleanup,
                )
        if (
            isinstance(primary, RetryableExternalIAMRefreshRequired)
            and not service_start_attempted
        ):
            raise RuntimeError(
                "activation safely stopped before service start; renew IAM and retry"
            ) from primary
        if (
            isinstance(primary, RetryableOwnerApprovalRefreshRequired)
            and not activation_mutation_attempted
            and not service_start_attempted
        ):
            raise RuntimeError(
                "activation safely stopped before mutation; renew owner approval "
                "and retry"
            ) from primary
        if primary is not None:
            _seal_activation_failure(
                self.plan,
                stage=failed_stage or self.stage,
                error=primary,
                owner_approval_receipt=owner_approval_receipt,
                external_iam_evidence=external_iam_evidence,
                read_only_preflight=_preflight,
            )
        try:
            receipt = {
                "schema": ACTIVATION_RECEIPT_SCHEMA,
                "revision": self.plan.revision,
                "activation_plan_sha256": self.plan.sha256,
                "approved_plan_sha256": self.plan.sha256,
                "native_observation_plan_sha256": (
                    self.plan.digests.native_observation_plan_sha256
                ),
                "native_observation_receipt_sha256": (
                    self.plan.digests.native_observation_receipt_sha256
                ),
                "owner_approval_receipt_sha256": owner_approval_receipt.sha256,
                "owner_approval_receipt": owner_approval_receipt.to_mapping(),
                "external_iam_evidence": dict(external_iam_evidence),
                "read_only_preflight": copy.deepcopy(dict(_preflight)),
                "projection_export": dict(projection),
                "live_preflight": dict(live),
                "services_stopped": True,
                "discord_started": False,
                "completed_at_unix": int(time.time()),
            }
            receipt["receipt_sha256"] = _sha256_json(receipt)
            self.stage = "success_receipt"
            success_path = _success_receipt_path(self.plan)
            receipt = {**receipt, "activation_receipt_path": str(success_path)}
            receipt.pop("receipt_sha256", None)
            receipt["receipt_sha256"] = _sha256_json(receipt)
            _write_root_receipt(success_path, receipt)
            return receipt
        except BaseException as success_error:
            _seal_activation_failure(
                self.plan,
                stage=self.stage,
                error=success_error,
                owner_approval_receipt=owner_approval_receipt,
                external_iam_evidence=external_iam_evidence,
                read_only_preflight=_preflight,
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Apply one owner-approved isolated writer-only activation",
    )
    parser.add_argument(
        "action",
        choices=(
            "install-approval",
            "install-external-iam",
            "install-native-plan",
            "observe-native",
            "install-plan",
            "apply",
            "validate-plan",
        ),
    )
    parser.add_argument("--plan")
    parser.add_argument("--staged-receipt")
    parser.add_argument("--approved-plan-sha256")
    parser.add_argument("--owner-approval-receipt")
    parser.add_argument("--external-iam-receipt")
    parser.add_argument("--external-iam-policy-sha256")
    arguments = parser.parse_args(argv)

    def required(value: str | None, label: str) -> str:
        if not value:
            parser.error(f"{label} is required for {arguments.action}")
        return value

    if arguments.action == "install-approval":
        approval, target = install_staged_owner_approval(
            required(arguments.staged_receipt, "--staged-receipt")
        )
        result: Mapping[str, Any] = {
            "ok": True,
            "schema": approval.value["schema"],
            "scope": approval.value["scope"],
            "plan_sha256": approval.value["plan_sha256"],
            "owner_approval_receipt_sha256": approval.sha256,
            "installed_path": str(target),
        }
    elif arguments.action == "install-external-iam":
        plan_path = _absolute_path(
            required(arguments.plan, "--plan"),
            "external IAM approval plan path",
        )
        if plan_path == DEFAULT_NATIVE_PLAN_PATH:
            authorized_plan = load_native_observation_plan(plan_path)
            approval_scope = "native_observation"
            authorized_plan_sha256 = authorized_plan.sha256
            authorized_policy_sha256 = authorized_plan.value[
                "external_iam_policy_sha256"
            ]
        elif plan_path == DEFAULT_PLAN_PATH:
            authorized_plan = load_activation_plan(plan_path)
            approval_scope = "activation"
            authorized_plan_sha256 = authorized_plan.sha256
            authorized_policy_sha256 = (
                authorized_plan.digests.external_iam_policy_sha256
            )
        else:
            raise ValueError("external IAM approval plan path is not pinned")
        if arguments.approved_plan_sha256 != authorized_plan_sha256:
            raise PermissionError(
                "external IAM requires the exact owner-approved plan digest"
            )
        requested_policy_sha256 = required(
            arguments.external_iam_policy_sha256,
            "--external-iam-policy-sha256",
        )
        if requested_policy_sha256 != authorized_policy_sha256:
            raise PermissionError(
                "external IAM policy does not match the approved plan"
            )
        owner_approval = load_owner_approval_receipt(
            required(
                arguments.owner_approval_receipt,
                "--owner-approval-receipt",
            ),
            scope=approval_scope,
            plan_sha256=authorized_plan_sha256,
        )
        iam, archive, replaced = install_staged_external_iam_receipt(
            required(arguments.staged_receipt, "--staged-receipt"),
            authorized_plan=authorized_plan,
            owner_approval_receipt=owner_approval,
        )
        result = {
            "ok": True,
            "schema": iam.value["schema"],
            "scope": approval_scope,
            "plan_sha256": authorized_plan_sha256,
            "owner_approval_receipt_sha256": owner_approval.sha256,
            "external_iam_receipt_sha256": iam.sha256,
            "external_iam_policy_sha256": iam.policy_sha256,
            "archive": dict(archive),
            "live_path": str(DEFAULT_EXTERNAL_IAM_LIVE_PATH),
            "live_replaced": replaced,
        }
    elif arguments.action == "install-native-plan":
        native_plan = install_staged_native_observation_plan(
            required(arguments.plan, "--plan")
        )
        result: Mapping[str, Any] = {
            "ok": True,
            "schema": native_plan.value["schema"],
            "revision": native_plan.value["revision"],
            "native_observation_plan_sha256": native_plan.sha256,
            "installed_path": str(DEFAULT_NATIVE_PLAN_PATH),
        }
    elif arguments.action == "observe-native":
        native_plan = load_native_observation_plan(required(arguments.plan, "--plan"))
        if arguments.approved_plan_sha256 != native_plan.sha256:
            raise PermissionError(
                "native observation requires exact external owner-approved digest"
            )
        if not arguments.owner_approval_receipt:
            raise PermissionError("native owner approval receipt is required")
        owner_approval = load_owner_approval_receipt(
            arguments.owner_approval_receipt,
            scope="native_observation",
            plan_sha256=native_plan.sha256,
        )
        if not arguments.external_iam_receipt:
            raise PermissionError("fresh external IAM receipt is required")
        native_executor = NativeObservationExecutor(native_plan)
        native_receipt = native_executor.observe(
            approved_plan_sha256=arguments.approved_plan_sha256,
            owner_approval_receipt=owner_approval,
            external_iam_receipt_path=arguments.external_iam_receipt,
        )
        result = {
            "ok": True,
            "schema": native_receipt.value["schema"],
            "revision": native_plan.value["revision"],
            "native_observation_plan_sha256": native_plan.sha256,
            "native_observation_receipt_sha256": native_receipt.sha256,
            "idempotent": native_executor.idempotent,
            "host_preparation_receipt_path": native_executor.host_preparation_receipt[
                "receipt_path"
            ],
            "host_preparation_receipt_sha256": native_executor.host_preparation_receipt[
                "receipt_sha256"
            ],
            "external_iam_evidence": dict(native_executor.external_iam_evidence),
        }
    elif arguments.action == "install-plan":
        plan = install_staged_activation_plan(required(arguments.plan, "--plan"))
        result: Mapping[str, Any] = {
            "ok": True,
            "schema": plan.schema,
            "revision": plan.revision,
            "activation_plan_sha256": plan.sha256,
            "installed_path": str(DEFAULT_PLAN_PATH),
        }
    elif arguments.action in {"apply", "validate-plan"}:
        plan = load_activation_plan(required(arguments.plan, "--plan"))
    if arguments.action == "validate-plan":
        if arguments.approved_plan_sha256 != plan.sha256:
            raise PermissionError(
                "activation preflight requires exact owner-approved plan digest"
            )
        if not arguments.owner_approval_receipt:
            raise PermissionError(
                "activation preflight owner approval receipt is required"
            )
        owner_approval = load_owner_approval_receipt(
            arguments.owner_approval_receipt,
            scope="activation",
            plan_sha256=plan.sha256,
        )
        try:
            preflight, _existing, _iam = activation_read_only_preflight(
                plan,
                owner_approval_receipt=owner_approval,
            )
        except ActivationReadOnlyPreflightError as exc:
            print(
                json.dumps(
                    dict(exc.report),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            return 2
        result = dict(preflight)
    elif arguments.action == "apply":
        if arguments.approved_plan_sha256 != plan.sha256:
            raise PermissionError(
                "activation requires exact external owner-approved plan digest"
            )
        if not arguments.owner_approval_receipt:
            raise PermissionError("activation owner approval receipt is required")
        owner_approval = load_owner_approval_receipt(
            arguments.owner_approval_receipt,
            scope="activation",
            plan_sha256=plan.sha256,
        )
        result = ActivationExecutor(plan).apply(
            approved_plan_sha256=arguments.approved_plan_sha256,
            owner_approval_receipt=owner_approval,
        )
    print(
        json.dumps(
            dict(result),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


__all__ = [
    "ACTIVATION_FAILURE_SCHEMA",
    "ACTIVATION_PLAN_SCHEMA",
    "ACTIVATION_RECEIPT_SCHEMA",
    "ActivationDigests",
    "ActivationExecutor",
    "ActivationPaths",
    "ActivationPlan",
    "ActivationReadOnlyPreflightError",
    "Command",
    "EXPORTER_UNIT",
    "GATEWAY_UNIT",
    "InstallArtifact",
    "NATIVE_OBSERVATION_SCHEMA",
    "NativeObservationExecutor",
    "NumericIdentities",
    "OWNER_APPROVAL_SCHEMA",
    "OwnerApprovalReceipt",
    "SystemdBundle",
    "WRITER_UNIT",
    "load_activation_plan",
    "load_durable_native_observation_receipt",
    "load_external_iam_receipt",
    "load_native_observation_plan",
    "load_owner_approval_receipt",
    "activation_read_only_preflight",
    "install_staged_activation_plan",
    "install_staged_external_iam_receipt",
    "install_staged_owner_approval",
    "main",
    "prepare_canary_host_identities",
]


if __name__ == "__main__":
    raise SystemExit(main())

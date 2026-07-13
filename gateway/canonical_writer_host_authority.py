"""Root-only host authority and native canary observation contracts.

This module is deliberately mechanical.  It does not decide what a request
means and it never routes work.  It records the local privilege surface of the
two fixed writer-only services and seals the observation that a canary was
started from one approved artifact and was stopped again afterwards.

All command execution uses fixed absolute binaries, a fixed C locale, bounded
output and no shell.  Missing or ambiguous evidence fails closed.
"""

from __future__ import annotations

import argparse
import grp
import hashlib
import ipaddress
import json
import os
import pwd
import re
import socket
import stat
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


NATIVE_OBSERVATION_PLAN_SCHEMA = "muncho-writer-native-observation-plan.v2"
NATIVE_OBSERVATION_RECEIPT_SCHEMA = "muncho-writer-native-observation.v1"
NATIVE_OBSERVATION_STAGE_SCHEMA = "muncho-writer-native-observation-stage.v1"
OWNER_APPROVAL_RECEIPT_SCHEMA = "muncho-writer-owner-approval.v1"
EXTERNAL_IAM_RECEIPT_SCHEMA = "muncho-writer-external-iam-evidence.v1"
# Observation remains live through two independently bounded 60-second service
# stops, state re-attestation, durable receipt installation, and fsync margin.
NATIVE_OBSERVATION_TTL_SECONDS = 300
EXTERNAL_IAM_TTL_SECONDS = 1200

LEGACY_CLOUD_SQL_HELPER_PATH = Path(
    "/opt/adventico-ai-platform/canonical-brain/bin/"
    "cloud_sql_synthetic_write_gate.py"
)
LEGACY_CLOUD_SQL_HELPER_PARENT = LEGACY_CLOUD_SQL_HELPER_PATH.parent
DEFAULT_NATIVE_OBSERVATION_PLAN_PATH = Path(
    "/etc/muncho/writer-activation/native-observation-plan.json"
)
DEFAULT_NATIVE_OBSERVATION_STAGE_ROOT = Path("/run/muncho-canonical-preflight")
DEFAULT_NATIVE_OBSERVATION_EVIDENCE_ROOT = Path(
    "/var/lib/muncho-writer-canary-evidence"
)
DEFAULT_OWNER_APPROVAL_ROOT = Path(
    "/etc/muncho/writer-activation/approvals"
)
DEFAULT_EXTERNAL_IAM_RECEIPT_ROOT = Path(
    "/etc/muncho/writer-activation/external-iam"
)
DEFAULT_EXTERNAL_IAM_LIVE_PATH = Path(
    "/run/muncho-canonical-preflight/external-iam-receipt.json"
)

_SYSTEMCTL = "/usr/bin/systemctl"
_SUDO = "/usr/bin/sudo"
_PKCHECK = "/usr/bin/pkcheck"
_CRONTAB = "/usr/bin/crontab"
_DPKG_QUERY = "/usr/bin/dpkg-query"
_ATQ_CANDIDATES = ("/usr/bin/atq", "/bin/atq")
_DOAS_CANDIDATES = ("/usr/bin/doas", "/bin/doas", "/usr/local/bin/doas")
_FIXED_ENV = {"LC_ALL": "C", "LANG": "C", "PATH": "/usr/bin:/bin"}
_MAX_COMMAND_OUTPUT = 1024 * 1024
_MAX_PROCESS_FILE = 1024 * 1024
_MAX_CRON_FILE = 256 * 1024
_MAX_INVENTORY_ITEMS = 4096
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_UNIT_RE = re.compile(
    r"^[A-Za-z0-9_.@\\x2d-]+\."
    r"(?:service|timer|socket|path|target|automount|mount|device|swap|slice|scope|busname)$"
)
_SYSTEMD_ACTIVATOR_SUFFIXES = (
    ".timer",
    ".socket",
    ".path",
    ".target",
    ".automount",
)

_DANGEROUS_GROUP_NAMES = (
    "adm",
    "crontab",
    "disk",
    "docker",
    "incus",
    "kmem",
    "kvm",
    "libvirt",
    "lxd",
    "operator",
    "root",
    "shadow",
    "sudo",
    "systemd-journal",
    "wheel",
)

_SYSTEMD_POLKIT_ACTIONS = (
    "org.freedesktop.systemd1.manage-unit-files",
    "org.freedesktop.systemd1.manage-units",
    "org.freedesktop.systemd1.reload-daemon",
)

_CAPABILITY_NAMES = (
    "CAP_CHOWN",
    "CAP_DAC_OVERRIDE",
    "CAP_DAC_READ_SEARCH",
    "CAP_FOWNER",
    "CAP_FSETID",
    "CAP_KILL",
    "CAP_SETGID",
    "CAP_SETUID",
    "CAP_SETPCAP",
    "CAP_LINUX_IMMUTABLE",
    "CAP_NET_BIND_SERVICE",
    "CAP_NET_BROADCAST",
    "CAP_NET_ADMIN",
    "CAP_NET_RAW",
    "CAP_IPC_LOCK",
    "CAP_IPC_OWNER",
    "CAP_SYS_MODULE",
    "CAP_SYS_RAWIO",
    "CAP_SYS_CHROOT",
    "CAP_SYS_PTRACE",
    "CAP_SYS_PACCT",
    "CAP_SYS_ADMIN",
    "CAP_SYS_BOOT",
    "CAP_SYS_NICE",
    "CAP_SYS_RESOURCE",
    "CAP_SYS_TIME",
    "CAP_SYS_TTY_CONFIG",
    "CAP_MKNOD",
    "CAP_LEASE",
    "CAP_AUDIT_WRITE",
    "CAP_AUDIT_CONTROL",
    "CAP_SETFCAP",
    "CAP_MAC_OVERRIDE",
    "CAP_MAC_ADMIN",
    "CAP_SYSLOG",
    "CAP_WAKE_ALARM",
    "CAP_BLOCK_SUSPEND",
    "CAP_AUDIT_READ",
    "CAP_PERFMON",
    "CAP_BPF",
    "CAP_CHECKPOINT_RESTORE",
)

_SYSTEMD_UNIT_DIRECTORIES = (
    Path("/etc/systemd/system"),
    Path("/run/systemd/system"),
    Path("/usr/local/lib/systemd/system"),
    Path("/usr/lib/systemd/system"),
    Path("/lib/systemd/system"),
)
_CRON_PATHS = (
    Path("/etc/crontab"),
    Path("/etc/cron.allow"),
    Path("/etc/cron.deny"),
    Path("/etc/cron.d"),
    Path("/etc/cron.hourly"),
    Path("/etc/cron.daily"),
    Path("/etc/cron.weekly"),
    Path("/etc/cron.monthly"),
    Path("/var/spool/cron"),
    Path("/var/spool/cron/crontabs"),
)

_CANARY_PROJECT = "adventico-ai-platform"
_CANARY_ZONE = "europe-west3-a"
_CANARY_INSTANCE = "muncho-canary-v2-01"
_CANARY_SERVICE_ACCOUNT = (
    "muncho-canary-v2-runtime@adventico-ai-platform.iam.gserviceaccount.com"
)
_CANARY_SCOPES = (
    "https://www.googleapis.com/auth/logging.write",
    "https://www.googleapis.com/auth/monitoring.write",
)
_CANARY_ROLES = (
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
)
_CANARY_PERMISSIONS = (
    "logging.logEntries.create",
    "logging.logEntries.route",
    "monitoring.metricDescriptors.create",
    "monitoring.metricDescriptors.get",
    "monitoring.metricDescriptors.list",
    "monitoring.monitoredResourceDescriptors.get",
    "monitoring.monitoredResourceDescriptors.list",
    "monitoring.timeSeries.create",
)


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
        raise ValueError("authority value is not canonical JSON") from exc
    return encoded.encode("utf-8", errors="strict")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _revision(value: Any) -> str:
    if not isinstance(value, str) or _REVISION_RE.fullmatch(value) is None:
        raise ValueError("native observation revision is invalid")
    return value


def _positive_integer(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_integer(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _absolute_path(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an absolute path")
    path = Path(value)
    if (
        not path.is_absolute()
        or str(path) != value
        or ".." in path.parts
        or any(character in value for character in "\x00\r\n")
    ):
        raise ValueError(f"{label} must be an absolute normalized path")
    return value


def _exact_keys(value: Any, keys: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{label} fields are not exact")
    return value


def _strict_strings(value: Any, label: str, *, nonempty: bool = False) -> tuple[str, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or (nonempty and not value)
        or len(value) > _MAX_INVENTORY_ITEMS
        or any(
            not isinstance(item, str)
            or not item
            or len(item.encode("utf-8")) > 4096
            or any(ord(character) < 32 for character in item)
            for item in value
        )
    ):
        raise ValueError(f"{label} is not a bounded string sequence")
    result = tuple(value)
    if result != tuple(sorted(set(result))):
        raise ValueError(f"{label} must be sorted and unique")
    return result


def _strict_integers(value: Any, label: str, *, nonempty: bool = False) -> tuple[int, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or (nonempty and not value)
        or len(value) > _MAX_INVENTORY_ITEMS
        or any(type(item) is not int or item < 0 for item in value)
    ):
        raise ValueError(f"{label} is not a bounded integer sequence")
    result = tuple(value)
    if result != tuple(sorted(set(result))):
        raise ValueError(f"{label} must be sorted and unique")
    return result


def _native_mappings(value: Any, label: str) -> tuple[dict[str, str], ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or not value
        or len(value) > _MAX_INVENTORY_ITEMS
    ):
        raise ValueError(f"{label} native mapping set is invalid")
    result: list[dict[str, str]] = []
    for raw in value:
        item = _exact_keys(raw, frozenset({"path", "sha256"}), label)
        result.append(
            {
                "path": _absolute_path(item.get("path"), f"{label} native path"),
                "sha256": _digest(item.get("sha256"), f"{label} native digest"),
            }
        )
    paths = [item["path"] for item in result]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError(f"{label} native mapping paths are not exact")
    return tuple(result)


_UNIT_BINDING_KEYS = frozenset({"name", "path", "sha256"})
_CONFIG_BINDING_KEYS = frozenset({"path", "sha256"})
_PLAN_IDENTITY_KEYS = frozenset(
    {
        "gateway_uid",
        "gateway_gid",
        "gateway_supplementary_gids",
        "writer_uid",
        "writer_gid",
        "writer_supplementary_gids",
        "socket_group_gid",
        "projector_uid",
        "projector_gid",
        "gateway_home",
        "writer_home",
        "projector_home",
    }
)
_PLAN_DATABASE_KEYS = frozenset(
    {"ip_network", "tls_server_name", "ca_path", "ca_sha256"}
)
_PLAN_DISCORD_KEYS = frozenset(
    {"unit_name", "config_path", "token_path", "socket_path", "required_absent"}
)
_NATIVE_DISCOVERY_POLICY_KEYS = frozenset(
    {
        "allowed_roots",
        "allowed_kernel_executable_mappings",
        "maximum_mappings",
        "required_owner_uid",
        "required_owner_gid",
        "require_regular",
        "require_single_link",
        "forbid_symlink",
        "forbid_acl",
        "forbid_xattrs",
        "forbid_writable",
        "forbid_deleted",
        "exclude_artifact_root",
        "digest_algorithm",
    }
)
_NATIVE_PLAN_KEYS = frozenset(
    {
        "schema",
        "boot_id_sha256",
        "host_identity_sha256",
        "observation_id",
        "revision",
        "artifact_root",
        "artifact_sha256",
        "release_manifest_file_sha256",
        "config_collector_receipt_sha256",
        "gateway_unit",
        "writer_unit",
        "gateway_argv",
        "writer_argv",
        "gateway_config",
        "writer_config",
        "identities",
        "database",
        "discord",
        "native_discovery_policy",
        "legacy_helper_path",
        "external_iam_policy_sha256",
    }
)


@dataclass(frozen=True)
class NativeObservationPlan:
    """Exact independently digestible plan for one native canary observation."""

    value: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "NativeObservationPlan":
        value = _exact_keys(raw, _NATIVE_PLAN_KEYS, "native observation plan")
        if value.get("schema") != NATIVE_OBSERVATION_PLAN_SCHEMA:
            raise ValueError("native observation plan schema is invalid")
        _digest(value.get("boot_id_sha256"), "native plan boot identity")
        _digest(value.get("host_identity_sha256"), "native plan host identity")
        try:
            observation_id = uuid.UUID(str(value.get("observation_id")))
        except ValueError as exc:
            raise ValueError("native observation ID is invalid") from exc
        if (
            observation_id.version != 4
            or str(observation_id) != value.get("observation_id")
        ):
            raise ValueError("native observation ID is invalid")
        _revision(value.get("revision"))
        artifact_root = _absolute_path(
            value.get("artifact_root"), "native plan artifact root"
        )
        if not artifact_root.endswith("/" + str(value.get("revision"))):
            raise ValueError("native plan artifact root is not revision-addressed")
        for name in (
            "artifact_sha256",
            "release_manifest_file_sha256",
            "config_collector_receipt_sha256",
            "external_iam_policy_sha256",
        ):
            _digest(value.get(name), f"native plan {name}")
        for name in ("gateway_unit", "writer_unit"):
            binding = _exact_keys(value.get(name), _UNIT_BINDING_KEYS, name)
            unit_name = binding.get("name")
            if not isinstance(unit_name, str) or _UNIT_RE.fullmatch(unit_name) is None:
                raise ValueError(f"{name} name is invalid")
            _absolute_path(binding.get("path"), f"{name} path")
            _digest(binding.get("sha256"), f"{name} digest")
        for name in ("gateway_config", "writer_config"):
            binding = _exact_keys(value.get(name), _CONFIG_BINDING_KEYS, name)
            _absolute_path(binding.get("path"), f"{name} path")
            _digest(binding.get("sha256"), f"{name} digest")
        for name in ("gateway_argv", "writer_argv"):
            argv = value.get(name)
            if (
                not isinstance(argv, Sequence)
                or isinstance(argv, (str, bytes, bytearray))
                or not argv
                or len(argv) > 128
                or any(
                    not isinstance(item, str)
                    or not item
                    or len(item.encode("utf-8")) > 4096
                    or any(character in item for character in "\x00\r\n")
                    for item in argv
                )
            ):
                raise ValueError(f"native plan {name} is invalid")
            if argv[0] != str(Path(artifact_root) / "venv/bin/python"):
                raise ValueError(f"native plan {name} interpreter is not sealed")
        identities = _exact_keys(value.get("identities"), _PLAN_IDENTITY_KEYS, "native identities")
        for name in _PLAN_IDENTITY_KEYS - {
            "gateway_supplementary_gids",
            "writer_supplementary_gids",
            "gateway_home",
            "writer_home",
            "projector_home",
        }:
            _positive_integer(identities.get(name), f"native identity {name}")
        gateway_groups = _strict_integers(
            identities.get("gateway_supplementary_gids"),
            "gateway supplementary groups",
            nonempty=True,
        )
        writer_groups = _strict_integers(
            identities.get("writer_supplementary_gids"),
            "writer supplementary groups",
            nonempty=True,
        )
        if gateway_groups != tuple(
            sorted((identities["gateway_gid"], identities["socket_group_gid"]))
        ) or writer_groups != tuple(
            sorted((identities["writer_gid"], identities["projector_gid"]))
        ):
            raise ValueError("native identity group sets are not exact")
        if identities["gateway_uid"] == identities["writer_uid"]:
            raise ValueError("native service UIDs must be distinct")
        if (
            identities.get("projector_uid") != 992
            or identities.get("projector_gid") != 991
            or identities.get("gateway_home") != "/var/lib/hermes-gateway"
            or identities.get("writer_home") != "/nonexistent"
            or identities.get("projector_home") != "/nonexistent"
        ):
            raise ValueError("native dormant identity homes are not pinned")
        database = _exact_keys(value.get("database"), _PLAN_DATABASE_KEYS, "native database")
        network = database.get("ip_network")
        try:
            parsed_network = ipaddress.ip_network(network, strict=True)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "native database network is not exact private IPv4 /32"
            ) from exc
        if (
            parsed_network.version != 4
            or parsed_network.prefixlen != 32
            or str(parsed_network) != network
            or not parsed_network.network_address.is_private
            or parsed_network.network_address.is_loopback
            or parsed_network.network_address.is_link_local
            or parsed_network.network_address.is_multicast
            or parsed_network.network_address.is_reserved
            or parsed_network.network_address.is_unspecified
        ):
            raise ValueError("native database network is not exact private IPv4 /32")
        tls_name = database.get("tls_server_name")
        if (
            not isinstance(tls_name, str)
            or not 1 <= len(tls_name) <= 253
            or tls_name != tls_name.lower()
            or any(character in tls_name for character in "\x00\r\n/ ")
        ):
            raise ValueError("native database TLS server name is invalid")
        _absolute_path(database.get("ca_path"), "native database CA path")
        _digest(database.get("ca_sha256"), "native database CA digest")
        discord = _exact_keys(value.get("discord"), _PLAN_DISCORD_KEYS, "native Discord policy")
        if discord.get("required_absent") is not True:
            raise ValueError("native Discord policy must require absence")
        if not isinstance(discord.get("unit_name"), str) or _UNIT_RE.fullmatch(discord["unit_name"]) is None:
            raise ValueError("native Discord unit is invalid")
        for name in ("config_path", "token_path", "socket_path"):
            _absolute_path(discord.get(name), f"native Discord {name}")
        discovery = _exact_keys(
            value.get("native_discovery_policy"),
            _NATIVE_DISCOVERY_POLICY_KEYS,
            "native discovery policy",
        )
        roots = tuple(discovery.get("allowed_roots") or ())
        if roots != ("/usr/lib",):
            raise ValueError("native discovery roots are not production-pinned")
        for root in roots:
            _absolute_path(root, "native discovery root")
        if tuple(discovery.get("allowed_kernel_executable_mappings") or ()) != (
            "[vdso]",
            "[vsyscall]",
        ):
            raise ValueError("native kernel executable mapping policy is not exact")
        if discovery.get("maximum_mappings") != 256:
            raise ValueError("native discovery mapping bound is invalid")
        if discovery.get("required_owner_uid") != 0 or discovery.get("required_owner_gid") != 0:
            raise ValueError("native discovery ownership is not root-pinned")
        if any(
            discovery.get(name) is not True
            for name in (
                "require_regular",
                "require_single_link",
                "forbid_symlink",
                "forbid_acl",
                "forbid_xattrs",
                "forbid_writable",
                "forbid_deleted",
                "exclude_artifact_root",
            )
        ) or discovery.get("digest_algorithm") != "sha256":
            raise ValueError("native discovery constraints are not exact")
        if _absolute_path(value.get("legacy_helper_path"), "legacy helper path") != str(
            LEGACY_CLOUD_SQL_HELPER_PATH
        ):
            raise ValueError("legacy helper path is not pinned")
        return cls(json.loads(_canonical_bytes(dict(value)).decode("utf-8")))

    def to_mapping(self) -> dict[str, Any]:
        return json.loads(_canonical_bytes(dict(self.value)).decode("utf-8"))

    @property
    def sha256(self) -> str:
        return _sha256_json(self.to_mapping())


_EXTERNAL_IAM_KEYS = frozenset(
    {
        "schema",
        "project",
        "zone",
        "instance",
        "service_account",
        "scopes",
        "roles",
        "permissions",
        "foundation_plan_sha256",
        "host_plan_sha256",
        "foundation_report_sha256",
        "host_report_sha256",
        "source_approval_sha256",
        "collected_at_unix",
        "expires_at_unix",
    }
)


@dataclass(frozen=True)
class ExternalIAMReceipt:
    """Canonical projection of the reviewed foundation and host preflights."""

    value: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ExternalIAMReceipt":
        value = _exact_keys(raw, _EXTERNAL_IAM_KEYS, "external IAM receipt")
        if value.get("schema") != EXTERNAL_IAM_RECEIPT_SCHEMA:
            raise ValueError("external IAM receipt schema is invalid")
        if (
            value.get("project") != _CANARY_PROJECT
            or value.get("zone") != _CANARY_ZONE
            or value.get("instance") != _CANARY_INSTANCE
            or value.get("service_account") != _CANARY_SERVICE_ACCOUNT
            or tuple(value.get("scopes") or ()) != _CANARY_SCOPES
            or tuple(value.get("roles") or ()) != _CANARY_ROLES
            or tuple(value.get("permissions") or ()) != _CANARY_PERMISSIONS
        ):
            raise ValueError("external IAM receipt authority set is not exact")
        for name in (
            "foundation_plan_sha256",
            "host_plan_sha256",
            "foundation_report_sha256",
            "host_report_sha256",
            "source_approval_sha256",
        ):
            _digest(value.get(name), f"external IAM {name}")
        collected = _nonnegative_integer(value.get("collected_at_unix"), "external IAM collected time")
        expires = _positive_integer(value.get("expires_at_unix"), "external IAM expiry")
        if expires - collected != EXTERNAL_IAM_TTL_SECONDS:
            raise ValueError("external IAM receipt validity window is invalid")
        return cls(json.loads(_canonical_bytes(dict(value)).decode("utf-8")))

    def require_fresh(
        self,
        now_unix: int,
        *,
        minimum_remaining_seconds: int = 0,
    ) -> None:
        if (
            type(now_unix) is not int
            or type(minimum_remaining_seconds) is not int
            or minimum_remaining_seconds < 0
            or minimum_remaining_seconds > EXTERNAL_IAM_TTL_SECONDS
            or not self.value["collected_at_unix"] <= now_unix
            or now_unix + minimum_remaining_seconds
            > self.value["expires_at_unix"]
        ):
            raise ValueError("external IAM receipt is stale or from the future")

    def to_mapping(self) -> dict[str, Any]:
        return json.loads(_canonical_bytes(dict(self.value)).decode("utf-8"))

    @property
    def sha256(self) -> str:
        return _sha256_json(self.to_mapping())

    @property
    def policy_sha256(self) -> str:
        return _sha256_json(
            {
                name: self.value[name]
                for name in (
                    "project",
                    "zone",
                    "instance",
                    "service_account",
                    "scopes",
                    "roles",
                    "permissions",
                    "foundation_plan_sha256",
                    "host_plan_sha256",
                )
            }
        )

    def evaluator_projection(self) -> dict[str, Any]:
        return {
            "complete": True,
            "roles": list(self.value["roles"]),
            "permissions": list(self.value["permissions"]),
        }


_FOUNDATION_REQUIRED_CHECKS = frozenset(
    {
        "identity.active_account",
        "project.exact",
        "apis.foundation_enabled",
        "resource.network_absent_or_exact",
        "resource.subnet_absent_or_exact",
        "resource.private_service_range_absent_or_exact",
        "resource.service_networking_absent_or_exact",
        "resource.network_routes_exact",
        "resource.foundation_dependency_order_exact",
        "resource.service_account_absent_or_exact",
        "resource.service_account_roles_allowed",
        "resource.sql_absent_or_exact_ready",
        "resource.database_absent_or_exact",
        "resource.canary_secret_names_absent",
    }
)
_FOUNDATION_REQUIRED_STEPS = (
    "create_isolated_vpc",
    "create_isolated_subnet",
    "reserve_private_service_range",
    "connect_private_service_networking",
    "create_runtime_service_account",
    "grant_logging_writer",
    "grant_monitoring_writer",
    "create_isolated_postgres",
    "create_canonical_database",
)
_HOST_REQUIRED_CHECKS = frozenset(
    {
        "network.complete_exact",
        "network.preflight_fresh",
        "image.exact_ready",
        "resource.vm_absent_or_exact_running",
    }
)


def _successful_check_names(
    report: Mapping[str, Any],
    *,
    expected: frozenset[str],
    label: str,
) -> None:
    raw = report.get("checks")
    if not isinstance(raw, list) or len(raw) != len(expected):
        raise ValueError(f"{label} checks are incomplete")
    names: set[str] = set()
    for item in raw:
        if (
            not isinstance(item, Mapping)
            or set(item) != {"name", "passed", "detail"}
            or not isinstance(item.get("name"), str)
            or item.get("passed") is not True
            or not isinstance(item.get("detail"), str)
            or not item["detail"]
            or item["name"] in names
        ):
            raise ValueError(f"{label} check evidence is invalid")
        names.add(item["name"])
    if names != set(expected):
        raise ValueError(f"{label} check set is not exact")


def build_external_iam_receipt(
    foundation_report: Mapping[str, Any],
    host_report: Mapping[str, Any],
    *,
    source_approval_sha256: str,
    now_unix: int | None = None,
) -> ExternalIAMReceipt:
    """Project only already-reviewed exact preflight reports; never run gcloud."""

    _digest(source_approval_sha256, "external IAM source approval")
    if not isinstance(foundation_report, Mapping) or not isinstance(host_report, Mapping):
        raise TypeError("foundation and host reports must be mappings")
    current = int(time.time()) if now_unix is None else now_unix
    _nonnegative_integer(current, "external IAM projection time")
    if (
        foundation_report.get("schema")
        != "muncho-isolated-canary-foundation-preflight.v2"
        or foundation_report.get("ok") is not True
        or host_report.get("schema") != "muncho-isolated-canary-host-preflight.v1"
        or host_report.get("ok") is not True
    ):
        raise ValueError("external IAM source preflight did not pass")
    _successful_check_names(
        foundation_report,
        expected=_FOUNDATION_REQUIRED_CHECKS,
        label="foundation preflight",
    )
    _successful_check_names(
        host_report,
        expected=_HOST_REQUIRED_CHECKS,
        label="host preflight",
    )
    if tuple(foundation_report.get("satisfied_steps") or ()) != _FOUNDATION_REQUIRED_STEPS:
        raise ValueError("foundation preflight does not attest the complete resource set")
    if host_report.get("satisfied_steps") != ["create_isolated_canary_vm"]:
        raise ValueError("host preflight does not attest the exact running VM")
    spec = foundation_report.get("spec")
    if not isinstance(spec, Mapping) or (
        spec.get("project") != _CANARY_PROJECT
        or spec.get("zone") != _CANARY_ZONE
        or spec.get("service_account_name") != "muncho-canary-v2-runtime"
    ):
        raise ValueError("foundation preflight target identity is not exact")
    foundation_collected = foundation_report.get("collected_at_unix")
    host_collected = host_report.get("collected_at_unix")
    if (
        type(foundation_collected) is not int
        or type(host_collected) is not int
        or not 0 <= current - foundation_collected <= 300
        or not 0 <= current - host_collected <= 300
    ):
        raise ValueError("external IAM source preflight is stale or future-dated")
    foundation_plan = _digest(
        foundation_report.get("plan_sha256"), "foundation plan digest"
    )
    host_plan = _digest(host_report.get("plan_sha256"), "host plan digest")
    return ExternalIAMReceipt.from_mapping(
        {
            "schema": EXTERNAL_IAM_RECEIPT_SCHEMA,
            "project": _CANARY_PROJECT,
            "zone": _CANARY_ZONE,
            "instance": _CANARY_INSTANCE,
            "service_account": _CANARY_SERVICE_ACCOUNT,
            "scopes": list(_CANARY_SCOPES),
            "roles": list(_CANARY_ROLES),
            "permissions": list(_CANARY_PERMISSIONS),
            "foundation_plan_sha256": foundation_plan,
            "host_plan_sha256": host_plan,
            "foundation_report_sha256": _sha256_json(dict(foundation_report)),
            "host_report_sha256": _sha256_json(dict(host_report)),
            "source_approval_sha256": source_approval_sha256,
            "collected_at_unix": current,
            "expires_at_unix": current + EXTERNAL_IAM_TTL_SECONDS,
        }
    )


_LIVE_SERVICE_KEYS = frozenset(
    {
        "unit_name",
        "active_state",
        "sub_state",
        "unit_file_state",
        "main_pid",
        "start_time_ticks",
        "argv",
        "external_native_mappings",
        "kernel_executable_mappings",
        "process_authority",
    }
)
_STOPPED_SERVICE_KEYS = frozenset(
    {
        "unit_name",
        "load_state",
        "active_state",
        "sub_state",
        "unit_file_state",
        "main_pid",
    }
)
_DISCORD_ABSENCE_KEYS = frozenset(
    {
        "unit_name",
        "unit_exists",
        "unit_enabled",
        "unit_active",
        "main_pid",
        "config_exists",
        "token_exists",
        "socket_exists",
        "process_pids",
    }
)
_HELPER_ABSENCE_KEYS = frozenset(
    {
        "path",
        "file_exists",
        "file_symlink",
        "parent_exists",
        "parent_symlink",
        "gateway_access",
    }
)
_LIVE_OBSERVATION_KEYS = frozenset(
    {
        "boot_id_sha256",
        "host_identity_sha256",
        "observed_at_unix",
        "observed_at_boottime_ns",
        "expires_at_boottime_ns",
        "gateway_service",
        "writer_service",
        "discord_absence",
        "legacy_helper_absence",
    }
)
_FINAL_STATE_KEYS = frozenset(
    {
        "boot_id_sha256",
        "host_identity_sha256",
        "finalized_at_unix",
        "finalized_at_boottime_ns",
        "gateway_service",
        "writer_service",
        "discord_absence",
    }
)
_NATIVE_RECEIPT_KEYS = frozenset(
    {
        "schema",
        "native_observation_plan_sha256",
        "owner_approval_receipt_sha256",
        "host_preparation_receipt_sha256",
        "external_iam_receipt_sha256",
        "plan",
        "observation",
        "final_state",
    }
)
_OWNER_APPROVAL_KEYS = frozenset(
    {
        "schema",
        "scope",
        "plan_sha256",
        "authority_kind",
        "cryptographic_owner_proof",
        "owner_subject_sha256",
        "approval_source_sha256",
        "nonce_sha256",
        "approved_at_unix",
        "expires_at_unix",
    }
)


@dataclass(frozen=True)
class OwnerApprovalReceipt:
    """Bound out-of-band bootstrap approval, explicitly not passkey proof."""

    value: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "OwnerApprovalReceipt":
        value = _exact_keys(raw, _OWNER_APPROVAL_KEYS, "owner approval receipt")
        if value.get("schema") != OWNER_APPROVAL_RECEIPT_SCHEMA:
            raise ValueError("owner approval receipt schema is invalid")
        if (
            value.get("authority_kind")
            != "trusted_root_bootstrap_out_of_band_owner"
            or value.get("cryptographic_owner_proof") is not False
        ):
            raise ValueError("owner approval trust semantics are not exact")
        if value.get("scope") not in {"native_observation", "activation"}:
            raise ValueError("owner approval receipt scope is invalid")
        for name in (
            "plan_sha256",
            "owner_subject_sha256",
            "approval_source_sha256",
            "nonce_sha256",
        ):
            _digest(value.get(name), f"owner approval {name}")
        approved = _nonnegative_integer(
            value.get("approved_at_unix"), "owner approval time"
        )
        expires = _positive_integer(
            value.get("expires_at_unix"), "owner approval expiry"
        )
        if not 1 <= expires - approved <= 900:
            raise ValueError("owner approval validity window is invalid")
        return cls(json.loads(_canonical_bytes(dict(value)).decode("utf-8")))

    def require(
        self,
        *,
        scope: str,
        plan_sha256: str,
        now_unix: int,
    ) -> None:
        if (
            self.value["scope"] != scope
            or self.value["plan_sha256"] != _digest(plan_sha256, "approved plan digest")
            or type(now_unix) is not int
            or not self.value["approved_at_unix"] <= now_unix <= self.value["expires_at_unix"]
        ):
            raise PermissionError("owner approval does not authorize this exact action")

    def to_mapping(self) -> dict[str, Any]:
        return json.loads(_canonical_bytes(dict(self.value)).decode("utf-8"))

    @property
    def sha256(self) -> str:
        return _sha256_json(self.to_mapping())


def validate_native_observation_approval(
    value: Mapping[str, Any],
    *,
    expected_plan_sha256: str,
    now_unix: int,
) -> str:
    approval = OwnerApprovalReceipt.from_mapping(value)
    approval.require(
        scope="native_observation",
        plan_sha256=expected_plan_sha256,
        now_unix=now_unix,
    )
    return approval.sha256


def _validate_discord_absence(value: Any, expected: Mapping[str, Any]) -> None:
    evidence = _exact_keys(value, _DISCORD_ABSENCE_KEYS, "Discord absence")
    if (
        evidence.get("unit_name") != expected.get("unit_name")
        or evidence.get("unit_exists") is not False
        or evidence.get("unit_enabled") is not False
        or evidence.get("unit_active") is not False
        or evidence.get("main_pid") != 0
        or evidence.get("config_exists") is not False
        or evidence.get("token_exists") is not False
        or evidence.get("socket_exists") is not False
        or evidence.get("process_pids") != []
    ):
        raise ValueError("Discord absence evidence is not exact")


def _validate_live_service(
    value: Any,
    *,
    unit: Mapping[str, Any],
    expected_argv: Sequence[str],
    expected_identity: Mapping[str, Any],
    discovery_policy: Mapping[str, Any],
    artifact_root: str,
) -> None:
    service = _exact_keys(value, _LIVE_SERVICE_KEYS, "live service")
    if (
        service.get("unit_name") != unit.get("name")
        or service.get("active_state") != "active"
        or service.get("sub_state") != "running"
        or service.get("unit_file_state") != "disabled"
    ):
        raise ValueError("native live service state is not exact")
    _positive_integer(service.get("main_pid"), "native live MainPID")
    _positive_integer(service.get("start_time_ticks"), "native live start time")
    argv = service.get("argv")
    if (
        not isinstance(argv, Sequence)
        or isinstance(argv, (str, bytes, bytearray))
        or not argv
        or len(argv) > 128
        or any(not isinstance(item, str) or not item or len(item) > 4096 for item in argv)
    ):
        raise ValueError("native live argv is invalid")
    if list(argv) != list(expected_argv):
        raise ValueError("native live argv does not match the approved plan")
    authority = _exact_keys(
        service.get("process_authority"),
        frozenset(
            {
                "pid",
                "process_start_time_ticks",
                "effective_uid",
                "effective_gid",
                "supplementary_gids",
                "no_new_privileges",
                "effective_capabilities",
                "executable",
            }
        ),
        "native process authority",
    )
    if (
        authority.get("pid") != service.get("main_pid")
        or authority.get("process_start_time_ticks")
        != service.get("start_time_ticks")
        or authority.get("effective_uid") != expected_identity.get("uid")
        or authority.get("effective_gid") != expected_identity.get("gid")
        or authority.get("supplementary_gids") != expected_identity.get("groups")
        or authority.get("no_new_privileges") is not True
        or authority.get("effective_capabilities") != []
        or authority.get("executable") != expected_argv[0]
    ):
        raise ValueError("native process authority is not exact")
    observed = _native_mappings(service.get("external_native_mappings"), "observed")
    kernel = _strict_strings(
        service.get("kernel_executable_mappings"),
        "observed kernel executable mappings",
        nonempty=True,
    )
    if not set(kernel) <= set(
        discovery_policy["allowed_kernel_executable_mappings"]
    ):
        raise ValueError("kernel executable mapping escapes discovery policy")
    if len(observed) > discovery_policy["maximum_mappings"]:
        raise ValueError("native path/hash observation exceeds the approved bound")
    roots = tuple(Path(root) for root in discovery_policy["allowed_roots"])
    release = Path(artifact_root)
    for item in observed:
        path = Path(item["path"])
        if (
            path == release
            or release in path.parents
            or not any(path == root or root in path.parents for root in roots)
        ):
            raise ValueError("native path/hash observation escapes discovery policy")


def _planned_process_identity(
    plan: NativeObservationPlan,
    label: str,
) -> dict[str, Any]:
    identities = plan.value["identities"]
    return {
        "uid": identities[f"{label}_uid"],
        "gid": identities[f"{label}_gid"],
        "groups": identities[f"{label}_supplementary_gids"],
    }


def _validate_stopped_service(value: Any, *, unit: Mapping[str, Any]) -> None:
    service = _exact_keys(value, _STOPPED_SERVICE_KEYS, "stopped service")
    if (
        service.get("unit_name") != unit.get("name")
        or service.get("load_state") != "loaded"
        or service.get("active_state") != "inactive"
        or service.get("sub_state") != "dead"
        or service.get("unit_file_state") != "disabled"
        or service.get("main_pid") != 0
    ):
        raise ValueError("native receipt service was not finalized stopped/off")


@dataclass(frozen=True)
class NativeObservationReceipt:
    """Final two-phase receipt: an approved live observation, then stopped/off."""

    value: Mapping[str, Any]

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        expected_plan_sha256: str | None = None,
        current_boot_id_sha256: str | None = None,
        current_boottime_ns: int | None = None,
    ) -> "NativeObservationReceipt":
        value = _exact_keys(raw, _NATIVE_RECEIPT_KEYS, "native observation receipt")
        if value.get("schema") != NATIVE_OBSERVATION_RECEIPT_SCHEMA:
            raise ValueError("native observation receipt schema is invalid")
        plan = NativeObservationPlan.from_mapping(
            _exact_keys(value.get("plan"), _NATIVE_PLAN_KEYS, "native receipt plan")
        )
        plan_digest = _digest(
            value.get("native_observation_plan_sha256"),
            "native observation plan digest",
        )
        if plan.sha256 != plan_digest or (
            expected_plan_sha256 is not None and plan_digest != expected_plan_sha256
        ):
            raise ValueError("native observation plan digest does not match approval")
        _digest(
            value.get("owner_approval_receipt_sha256"),
            "native owner approval receipt digest",
        )
        _digest(
            value.get("host_preparation_receipt_sha256"),
            "native host preparation receipt digest",
        )
        _digest(
            value.get("external_iam_receipt_sha256"),
            "native external IAM receipt digest",
        )
        observation = _exact_keys(value.get("observation"), _LIVE_OBSERVATION_KEYS, "native observation")
        final = _exact_keys(value.get("final_state"), _FINAL_STATE_KEYS, "native final state")
        boot = plan.value["boot_id_sha256"]
        host = plan.value["host_identity_sha256"]
        if (
            observation.get("boot_id_sha256") != boot
            or final.get("boot_id_sha256") != boot
            or observation.get("host_identity_sha256") != host
            or final.get("host_identity_sha256") != host
            or (current_boot_id_sha256 is not None and current_boot_id_sha256 != boot)
        ):
            raise ValueError("native observation receipt was replayed on another host or boot")
        observed_unix = _nonnegative_integer(observation.get("observed_at_unix"), "native observed time")
        observed_boot = _positive_integer(
            observation.get("observed_at_boottime_ns"), "native observed boottime"
        )
        expires_boot = _positive_integer(
            observation.get("expires_at_boottime_ns"), "native observation expiry"
        )
        finalized_unix = _nonnegative_integer(final.get("finalized_at_unix"), "native finalized time")
        finalized_boot = _positive_integer(
            final.get("finalized_at_boottime_ns"), "native finalized boottime"
        )
        if (
            expires_boot - observed_boot != NATIVE_OBSERVATION_TTL_SECONDS * 1_000_000_000
            or not observed_boot <= finalized_boot <= expires_boot
            or finalized_unix < observed_unix
            or (
                current_boottime_ns is not None
                and not finalized_boot <= current_boottime_ns <= expires_boot
            )
        ):
            raise ValueError("native observation receipt validity window is invalid")
        discovery_policy = _exact_keys(
            plan.value["native_discovery_policy"],
            _NATIVE_DISCOVERY_POLICY_KEYS,
            "native discovery policy",
        )
        artifact_root = plan.value["artifact_root"]
        _validate_live_service(
            observation.get("gateway_service"),
            unit=plan.value["gateway_unit"],
            expected_argv=plan.value["gateway_argv"],
            expected_identity=_planned_process_identity(plan, "gateway"),
            discovery_policy=discovery_policy,
            artifact_root=artifact_root,
        )
        _validate_live_service(
            observation.get("writer_service"),
            unit=plan.value["writer_unit"],
            expected_argv=plan.value["writer_argv"],
            expected_identity=_planned_process_identity(plan, "writer"),
            discovery_policy=discovery_policy,
            artifact_root=artifact_root,
        )
        _validate_stopped_service(final.get("gateway_service"), unit=plan.value["gateway_unit"])
        _validate_stopped_service(final.get("writer_service"), unit=plan.value["writer_unit"])
        _validate_discord_absence(observation.get("discord_absence"), plan.value["discord"])
        _validate_discord_absence(final.get("discord_absence"), plan.value["discord"])
        helper = _exact_keys(
            observation.get("legacy_helper_absence"),
            _HELPER_ABSENCE_KEYS,
            "legacy helper absence",
        )
        if (
            helper.get("path") != str(LEGACY_CLOUD_SQL_HELPER_PATH)
            or helper.get("file_exists") is not False
            or helper.get("file_symlink") is not False
            or helper.get("parent_exists") is not False
            or helper.get("parent_symlink") is not False
            or helper.get("gateway_access")
            != {"read": False, "write": False, "execute": False}
        ):
            raise ValueError("legacy helper absence evidence is not exact")
        return cls(json.loads(_canonical_bytes(dict(value)).decode("utf-8")))

    @classmethod
    def finalize(
        cls,
        *,
        plan: NativeObservationPlan,
        expected_plan_sha256: str,
        owner_approval_receipt_sha256: str,
        host_preparation_receipt_sha256: str,
        external_iam_receipt_sha256: str,
        observation: Mapping[str, Any],
        final_state: Mapping[str, Any],
        current_boot_id_sha256: str,
        current_boottime_ns: int,
    ) -> "NativeObservationReceipt":
        if plan.sha256 != _digest(expected_plan_sha256, "expected native plan digest"):
            raise ValueError("native observation invocation is not approved")
        _digest(owner_approval_receipt_sha256, "owner approval receipt digest")
        _digest(
            host_preparation_receipt_sha256,
            "host preparation receipt digest",
        )
        _digest(external_iam_receipt_sha256, "external IAM receipt digest")
        return cls.from_mapping(
            {
                "schema": NATIVE_OBSERVATION_RECEIPT_SCHEMA,
                "native_observation_plan_sha256": plan.sha256,
                "owner_approval_receipt_sha256": owner_approval_receipt_sha256,
                "host_preparation_receipt_sha256": (
                    host_preparation_receipt_sha256
                ),
                "external_iam_receipt_sha256": external_iam_receipt_sha256,
                "plan": plan.to_mapping(),
                "observation": dict(observation),
                "final_state": dict(final_state),
            },
            expected_plan_sha256=expected_plan_sha256,
            current_boot_id_sha256=current_boot_id_sha256,
            current_boottime_ns=current_boottime_ns,
        )

    def to_mapping(self) -> dict[str, Any]:
        return json.loads(_canonical_bytes(dict(self.value)).decode("utf-8"))

    @property
    def sha256(self) -> str:
        return _sha256_json(self.to_mapping())


def _require_root_linux() -> None:
    getter = getattr(os, "geteuid", None)
    if not callable(getter) or int(getter()) != 0:
        raise PermissionError("canonical_writer_host_authority_requires_uid_0")
    if sys.platform != "linux":
        raise RuntimeError("canonical_writer_host_authority_requires_linux")


def _validate_root_parent_chain(path: Path) -> None:
    current = path
    while True:
        item = os.lstat(current)
        if (
            stat.S_ISLNK(item.st_mode)
            or not stat.S_ISDIR(item.st_mode)
            or item.st_uid != 0
            or item.st_gid != 0
            or stat.S_IMODE(item.st_mode) & 0o022
        ):
            raise PermissionError("native receipt parent is not root-controlled")
        if current == current.parent:
            return
        current = current.parent


def write_native_observation_receipt(
    path_value: str | os.PathLike[str],
    receipt: NativeObservationReceipt,
) -> None:
    """Atomically install one root-owned, no-follow, immutable receipt file."""

    _require_root_linux()
    if not isinstance(receipt, NativeObservationReceipt):
        raise TypeError("native observation receipt is required")
    path = Path(_absolute_path(os.fspath(path_value), "native receipt path"))
    receipt_plan = NativeObservationPlan.from_mapping(receipt.value["plan"])
    if path != _expected_receipt_path(receipt_plan):
        raise ValueError("native observation receipt path is not plan-addressed")
    _validate_root_parent_chain(path.parent.parent)
    parent = os.lstat(path.parent)
    if (
        stat.S_ISLNK(parent.st_mode)
        or not stat.S_ISDIR(parent.st_mode)
        or parent.st_uid != 0
        or parent.st_gid != 0
        or stat.S_IMODE(parent.st_mode) != 0o700
    ):
        raise PermissionError("native receipt directory is not root-only")
    raw = _canonical_bytes(receipt.to_mapping()) + b"\n"
    if len(raw) > _MAX_COMMAND_OUTPUT:
        raise ValueError("native receipt exceeds its size bound")
    temporary = path.parent / f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = -1
    try:
        descriptor = os.open(temporary, flags, 0o600)
        os.fchown(descriptor, 0, 0)
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise OSError("native receipt write made no progress")
            offset += written
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as exc:
            raise FileExistsError(
                "native observation receipt is append-only and already exists"
            ) from exc
        temporary.unlink()
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    observed = os.lstat(path)
    if (
        not stat.S_ISREG(observed.st_mode)
        or observed.st_nlink != 1
        or observed.st_uid != 0
        or observed.st_gid != 0
        or stat.S_IMODE(observed.st_mode) != 0o400
    ):
        raise RuntimeError("native receipt ownership verification failed")


@dataclass(frozen=True)
class ProcessAuthorityEvidence:
    pid: int
    start_time_ticks: int
    effective_uid: int
    effective_gid: int
    supplementary_gids: tuple[int, ...]
    no_new_privileges: bool
    effective_capabilities: tuple[str, ...]
    executable: str
    argv: tuple[str, ...]

    def evaluator_mapping(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "process_start_time_ticks": self.start_time_ticks,
            "effective_uid": self.effective_uid,
            "effective_gid": self.effective_gid,
            "supplementary_gids": list(self.supplementary_gids),
            "no_new_privileges": self.no_new_privileges,
            "effective_capabilities": list(self.effective_capabilities),
            "executable": self.executable,
        }


def _run_fixed(
    argv: Sequence[str],
    *,
    timeout: int = 5,
    allowed_returncodes: frozenset[int] = frozenset({0}),
) -> subprocess.CompletedProcess[str]:
    if (
        not isinstance(argv, Sequence)
        or isinstance(argv, (str, bytes, bytearray))
        or not argv
        or not isinstance(argv[0], str)
        or not Path(argv[0]).is_absolute()
        or len(argv) > _MAX_INVENTORY_ITEMS
        or any(
            not isinstance(item, str)
            or not item
            or len(item.encode("utf-8")) > 4096
            or any(character in item for character in "\x00\r\n")
            for item in argv
        )
    ):
        raise ValueError("authority command argv is not fixed and bounded")
    try:
        completed = subprocess.run(
            list(argv),
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            timeout=timeout,
            env=dict(_FIXED_ENV),
        )
    except (OSError, subprocess.SubprocessError, UnicodeError) as exc:
        raise RuntimeError("bounded authority command failed") from exc
    if (
        completed.returncode not in allowed_returncodes
        or len(completed.stdout.encode("utf-8")) > _MAX_COMMAND_OUTPUT
        or len(completed.stderr.encode("utf-8")) > _MAX_COMMAND_OUTPUT
    ):
        raise RuntimeError("bounded authority command returned ambiguous evidence")
    return completed


def _read_bounded(path: Path, *, maximum: int, binary: bool = False) -> str | bytes:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise RuntimeError("process or authority evidence is unavailable") from exc
    if len(data) > maximum:
        raise RuntimeError("process or authority evidence exceeds its bound")
    if binary:
        return data
    try:
        return data.decode("ascii", errors="strict")
    except UnicodeError as exc:
        raise RuntimeError("process or authority evidence is not strict ASCII") from exc


def _process_start_time(proc_directory: Path) -> int:
    raw = _read_bounded(proc_directory / "stat", maximum=64 * 1024)
    assert isinstance(raw, str)
    suffix = raw.rsplit(")", 1)
    if len(suffix) != 2:
        raise RuntimeError("process stat identity is invalid")
    try:
        value = int(suffix[1].strip().split()[19])
    except (IndexError, ValueError) as exc:
        raise RuntimeError("process start time is invalid") from exc
    if value <= 0:
        raise RuntimeError("process start time is invalid")
    return value


def _decode_cap_eff(value: str) -> tuple[str, ...]:
    if re.fullmatch(r"[0-9a-fA-F]{16}", value) is None:
        raise RuntimeError("process CapEff is invalid")
    bits = int(value, 16)
    if bits >> len(_CAPABILITY_NAMES):
        raise RuntimeError("process CapEff contains an unknown capability")
    return tuple(
        name for index, name in enumerate(_CAPABILITY_NAMES) if bits & (1 << index)
    )


def _observe_process(
    pid: int,
    *,
    proc_root: Path = Path("/proc"),
) -> ProcessAuthorityEvidence:
    if type(pid) is not int or pid <= 1:
        raise ValueError("authority process PID is invalid")
    directory = proc_root / str(pid)
    start_before = _process_start_time(directory)
    status_raw = _read_bounded(directory / "status", maximum=_MAX_PROCESS_FILE)
    cmdline_raw = _read_bounded(
        directory / "cmdline", maximum=64 * 1024, binary=True
    )
    assert isinstance(status_raw, str)
    assert isinstance(cmdline_raw, bytes)
    fields: dict[str, str] = {}
    for line in status_raw.splitlines():
        name, separator, raw_value = line.partition(":")
        if separator:
            if name in fields:
                raise RuntimeError("process status contains duplicate fields")
            fields[name] = raw_value.strip()
    try:
        uid_values = fields["Uid"].split()
        gid_values = fields["Gid"].split()
        if len(uid_values) != 4 or len(gid_values) != 4:
            raise ValueError
        effective_uid = int(uid_values[1])
        effective_gid = int(gid_values[1])
        supplementary = tuple(sorted(int(item) for item in fields["Groups"].split()))
        if supplementary != tuple(sorted(set(supplementary))):
            raise ValueError
        no_new_privileges_raw = fields["NoNewPrivs"]
        if no_new_privileges_raw not in {"0", "1"}:
            raise ValueError
        capabilities = _decode_cap_eff(fields["CapEff"])
    except (KeyError, ValueError) as exc:
        raise RuntimeError("process credential authority is invalid") from exc
    try:
        argv = tuple(
            item.decode("utf-8", errors="strict")
            for item in cmdline_raw.rstrip(b"\x00").split(b"\x00")
            if item
        )
    except UnicodeError as exc:
        raise RuntimeError("process argv is not strict UTF-8") from exc
    if (
        not argv
        or len(argv) > 128
        or any(not item or len(item.encode("utf-8")) > 4096 for item in argv)
    ):
        raise RuntimeError("process argv is invalid")
    try:
        executable = os.readlink(directory / "exe")
    except OSError as exc:
        raise RuntimeError("process executable identity is unavailable") from exc
    if executable.endswith(" (deleted)"):
        raise RuntimeError("process executable was deleted")
    _absolute_path(executable, "process executable")
    if _process_start_time(directory) != start_before:
        raise RuntimeError("process identity changed during authority collection")
    return ProcessAuthorityEvidence(
        pid=pid,
        start_time_ticks=start_before,
        effective_uid=effective_uid,
        effective_gid=effective_gid,
        supplementary_gids=supplementary,
        no_new_privileges=no_new_privileges_raw == "1",
        effective_capabilities=capabilities,
        executable=executable,
        argv=argv,
    )


def _pids_for_uid(uid: int, *, proc_root: Path = Path("/proc")) -> tuple[int, ...]:
    if type(uid) is not int or uid <= 0:
        raise ValueError("authority UID is invalid")
    result: list[int] = []
    try:
        items = tuple(proc_root.iterdir())
    except OSError as exc:
        raise RuntimeError("process inventory is unavailable") from exc
    if len(items) > 1_000_000:
        raise RuntimeError("process inventory exceeds its bound")
    for item in items:
        if not item.name.isdigit():
            continue
        try:
            status = (item / "status").read_text(encoding="ascii")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if len(status) > _MAX_PROCESS_FILE:
            raise RuntimeError("process inventory evidence exceeds its bound")
        uid_lines = [line for line in status.splitlines() if line.startswith("Uid:")]
        if len(uid_lines) != 1:
            raise RuntimeError("process inventory UID evidence is ambiguous")
        values = uid_lines[0].partition(":")[2].split()
        if len(values) != 4 or not all(value.isdigit() for value in values):
            raise RuntimeError("process inventory UID evidence is invalid")
        if int(values[1]) == uid:
            result.append(int(item.name))
    return tuple(sorted(result))


def _descendant_pids(root_pid: int, *, proc_root: Path = Path("/proc")) -> tuple[int, ...]:
    parents: dict[int, int] = {}
    for item in proc_root.iterdir():
        if not item.name.isdigit():
            continue
        try:
            raw = (item / "stat").read_text(encoding="ascii")
            suffix = raw.rsplit(")", 1)
            parent = int(suffix[1].strip().split()[1])
        except (FileNotFoundError, IndexError, PermissionError, ProcessLookupError, ValueError):
            continue
        parents[int(item.name)] = parent
    descendants: set[int] = set()
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if pid not in descendants and (parent == root_pid or parent in descendants):
                descendants.add(pid)
                changed = True
    return tuple(sorted(descendants))


def _validate_privileged_binary(path: Path, *, require_setid: bool = False) -> os.stat_result:
    try:
        item = os.lstat(path)
    except OSError as exc:
        raise RuntimeError(f"required authority executable is unavailable: {path}") from exc
    xattrs = set(os.listxattr(path, follow_symlinks=False))
    if (
        stat.S_ISLNK(item.st_mode)
        or not stat.S_ISREG(item.st_mode)
        or item.st_nlink != 1
        or item.st_uid != 0
        or stat.S_IMODE(item.st_mode) & 0o022
        or xattrs & {"system.posix_acl_access", "system.posix_acl_default"}
        or (require_setid and not item.st_mode & (stat.S_ISUID | stat.S_ISGID))
    ):
        raise RuntimeError(f"authority executable provenance is invalid: {path}")
    return item


def _sudo_policy(user_name: str) -> tuple[str, ...]:
    _validate_privileged_binary(Path(_SUDO), require_setid=True)
    completed = _run_fixed(
        (_SUDO, "-n", "-l", "-U", user_name),
        allowed_returncodes=frozenset({0, 1}),
    )
    combined = "\n".join(
        line.strip()
        for line in (completed.stdout + "\n" + completed.stderr).splitlines()
        if line.strip()
    )
    denied = re.fullmatch(
        rf"User {re.escape(user_name)} is not allowed to run sudo on [^\n]+\.",
        combined,
    )
    if completed.returncode == 1 and denied is not None:
        return ()
    if completed.returncode != 0 or "may run the following commands" not in combined:
        raise RuntimeError("sudo authority evidence is ambiguous")
    marker = combined.index("may run the following commands")
    suffix = combined[marker:].splitlines()[1:]
    commands = tuple(sorted(set(line.strip() for line in suffix if line.strip())))
    if not commands:
        raise RuntimeError("sudo allowed-policy evidence is incomplete")
    return commands


def _package_absent(package: str) -> bool:
    _validate_privileged_binary(Path(_DPKG_QUERY))
    completed = _run_fixed(
        (_DPKG_QUERY, "-W", "-f=${db:Status-Abbrev}", "--", package),
        allowed_returncodes=frozenset({0, 1}),
    )
    if (
        completed.returncode == 1
        and completed.stdout == ""
        and completed.stderr.strip()
        == f"dpkg-query: no packages found matching {package}"
    ):
        return True
    if (
        completed.returncode == 0
        and completed.stdout.strip() == "ii"
        and completed.stderr == ""
    ):
        return False
    raise RuntimeError(f"package authority evidence is ambiguous: {package}")


def _doas_policy() -> tuple[str, ...]:
    present = [path for path in _DOAS_CANDIDATES if os.path.lexists(path)]
    config_present = os.path.lexists("/etc/doas.conf")
    absent_package = _package_absent("doas")
    if not present and not config_present and absent_package:
        return ()
    if len(present) > 1:
        raise RuntimeError("doas executable identity is ambiguous")
    # A present doas policy is conservatively surfaced as authority.  Parsing
    # vendor-specific doas grammars would itself become an authorization engine.
    return ("doas:present-unreviewed",)


def _polkit_actions(process: ProcessAuthorityEvidence) -> tuple[str, ...]:
    _validate_privileged_binary(Path(_PKCHECK))
    allowed: list[str] = []
    process_identity = (
        f"{process.pid},{process.start_time_ticks},{process.effective_uid}"
    )
    for action in _SYSTEMD_POLKIT_ACTIONS:
        completed = _run_fixed(
            (
                _PKCHECK,
                "--action-id",
                action,
                "--process",
                process_identity,
            ),
            allowed_returncodes=frozenset({0, 1}),
        )
        if completed.returncode == 0:
            allowed.append(action)
        elif completed.stdout.strip() or completed.stderr.strip():
            # pkcheck 122/systemd 252 normally returns 1 silently for denial.
            raise RuntimeError("polkit denial evidence is ambiguous")
        if _process_start_time(Path("/proc") / str(process.pid)) != process.start_time_ticks:
            raise RuntimeError("process changed during polkit authority probe")
    return tuple(sorted(allowed))


def _mode_bits(path: Path, *, uid: int, gids: tuple[int, ...]) -> tuple[int, os.stat_result]:
    item = os.lstat(path)
    if stat.S_ISLNK(item.st_mode):
        raise RuntimeError("authority access path is a symlink")
    xattrs = set(os.listxattr(path, follow_symlinks=False))
    if xattrs & {"system.posix_acl_access", "system.posix_acl_default"}:
        raise RuntimeError("authority access path has an unreviewed ACL")
    mode = stat.S_IMODE(item.st_mode)
    if item.st_uid == uid:
        return (mode >> 6) & 0o7, item
    if item.st_gid in set(gids):
        return (mode >> 3) & 0o7, item
    return mode & 0o7, item


def _can_mutate_path(path: Path, *, uid: int, gids: tuple[int, ...]) -> bool:
    current = path
    exists = False
    while True:
        try:
            os.lstat(current)
        except FileNotFoundError:
            if current == current.parent:
                raise RuntimeError("authority path has no existing ancestor")
            current = current.parent
            continue
        exists = current == path
        break
    bits, _item = _mode_bits(current, uid=uid, gids=gids)
    if exists and bits & 0o2:
        return True
    parent = path.parent if exists else current
    parent_bits, parent_item = _mode_bits(parent, uid=uid, gids=gids)
    if not stat.S_ISDIR(parent_item.st_mode):
        raise RuntimeError("authority path parent is not a directory")
    return bool(parent_bits & 0o3 == 0o3)


def _writable_paths(
    paths: Sequence[Path],
    *,
    identities: Sequence[ProcessAuthorityEvidence],
) -> tuple[str, ...]:
    writable: set[str] = set()
    for process in identities:
        gids = tuple(sorted({process.effective_gid, *process.supplementary_gids}))
        for path in paths:
            if _can_mutate_path(path, uid=process.effective_uid, gids=gids):
                writable.add(str(path))
    return tuple(sorted(writable))


def _systemd_unit_names() -> tuple[str, ...]:
    names: set[str] = set()
    for command in (
        (
            _SYSTEMCTL,
            "list-unit-files",
            "--type=service",
            "--type=timer",
            "--type=socket",
            "--type=path",
            "--type=target",
            "--type=automount",
            "--type=mount",
            "--type=swap",
            "--type=slice",
            "--type=scope",
            "--type=device",
            "--no-legend",
            "--no-pager",
            "--plain",
        ),
        (
            _SYSTEMCTL,
            "list-units",
            "--all",
            "--no-legend",
            "--no-pager",
            "--plain",
        ),
    ):
        completed = _run_fixed(command, timeout=10)
        for line in completed.stdout.splitlines():
            parts = line.split()
            if not parts:
                continue
            name = parts[0]
            if _UNIT_RE.fullmatch(name) is None:
                raise RuntimeError("systemd inventory contains an invalid unit")
            if "@." in name:
                # An uninstantiated template is not a runnable unit identity and
                # systemd 252 rejects it in `show`.  Concrete instances are
                # independently returned by list-units/list-unit-files; the
                # ability to instantiate a template is covered by the exact
                # polkit/transient-unit and user-manager boundaries.
                continue
            names.add(name)
    if not names or len(names) > _MAX_INVENTORY_ITEMS:
        raise RuntimeError("systemd inventory is empty or oversized")
    return tuple(sorted(names))


_INVENTORY_SYSTEMD_PROPERTIES = (
    "Id",
    "LoadState",
    "FragmentPath",
    "Transient",
    "User",
    "ExecStart",
    "Triggers",
    "TriggeredBy",
    "WantedBy",
    "RequiredBy",
    "BoundBy",
    "UpheldBy",
    "RequisiteOf",
    "OnSuccessOf",
    "OnFailureOf",
    "BindsTo",
    "Upholds",
    "Requisite",
    "OnSuccess",
    "OnFailure",
)


def _systemd_inventory(
    writer_uid: int,
    gateway_uid: int,
) -> tuple[dict[str, str], ...]:
    names = _systemd_unit_names()
    command = [
        _SYSTEMCTL,
        "show",
        *(f"--property={name}" for name in _INVENTORY_SYSTEMD_PROPERTIES),
        "--",
        *names,
    ]
    completed = _run_fixed(command, timeout=20)
    blocks = [block for block in completed.stdout.split("\n\n") if block.strip()]
    if not blocks or len(blocks) != len(names):
        raise RuntimeError("systemd unit inventory blocks are invalid")
    result: list[dict[str, str]] = []
    canonical: dict[str, dict[str, str]] = {}
    for requested_name, block in zip(names, blocks, strict=True):
        fields: dict[str, str] = {}
        for line in block.splitlines():
            name, separator, value = line.partition("=")
            if separator != "=" or name not in _INVENTORY_SYSTEMD_PROPERTIES or name in fields:
                raise RuntimeError("systemd unit inventory fields are invalid")
            fields[name] = value
        if set(fields) != set(_INVENTORY_SYSTEMD_PROPERTIES):
            raise RuntimeError("systemd unit inventory is incomplete")
        identifier = fields["Id"]
        if _UNIT_RE.fullmatch(identifier) is None:
            raise RuntimeError("systemd unit inventory identity is ambiguous")
        if requested_name != identifier:
            _verify_systemd_alias(
                requested_name,
                canonical_name=identifier,
                fragment_path=fields["FragmentPath"],
            )
        previous = canonical.get(identifier)
        if previous is not None:
            if previous != fields:
                raise RuntimeError("systemd alias evidence is inconsistent")
            continue
        canonical[identifier] = fields
        user_value = fields["User"]
        if user_value:
            try:
                int(user_value) if user_value.isdigit() else pwd.getpwnam(user_value).pw_uid
            except (KeyError, ValueError) as exc:
                raise RuntimeError("systemd unit user identity is ambiguous") from exc
        result.append(fields)
    return tuple(sorted(result, key=lambda item: item["Id"]))


_REVERSE_ACTIVATION_KEYS = (
    "TriggeredBy",
    "WantedBy",
    "RequiredBy",
    "BoundBy",
    "UpheldBy",
    "RequisiteOf",
    "OnSuccessOf",
    "OnFailureOf",
)


def _reverse_activation_evidence(
    unit_name: str,
    systemd_inventory: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    """Project every mechanical systemd path that can activate *unit_name*."""

    if _UNIT_RE.fullmatch(unit_name) is None or not unit_name.endswith(".service"):
        raise RuntimeError("reverse activation target is invalid")
    indexed = {item.get("Id"): item for item in systemd_inventory}
    if len(indexed) != len(systemd_inventory) or unit_name not in indexed:
        raise RuntimeError("reverse activation target is missing or ambiguous")
    target = indexed[unit_name]
    direct: dict[str, tuple[str, ...]] = {}
    for property_name in _REVERSE_ACTIVATION_KEYS:
        names = tuple(sorted(set(target.get(property_name, "").split())))
        if any(_UNIT_RE.fullmatch(name) is None for name in names):
            raise RuntimeError("reverse activation property is invalid")
        direct[property_name] = names

    reverse_references: set[str] = set()
    relevant: set[str] = set().union(*direct.values())
    for item in systemd_inventory:
        identifier = str(item.get("Id") or "")
        if identifier == unit_name:
            continue
        for property_name in (
            "Triggers",
            "TriggeredBy",
            "WantedBy",
            "RequiredBy",
            "BoundBy",
            "UpheldBy",
            "RequisiteOf",
            "OnSuccessOf",
            "OnFailureOf",
            "BindsTo",
            "Upholds",
            "Requisite",
            "OnSuccess",
            "OnFailure",
        ):
            references = tuple(item.get(property_name, "").split())
            if any(_UNIT_RE.fullmatch(name) is None for name in references):
                raise RuntimeError("systemd relation inventory is invalid")
            if unit_name in references:
                reverse_references.add(identifier)
                relevant.add(identifier)

    def by_suffix(suffix: str) -> list[str]:
        return sorted(name for name in relevant if name.endswith(suffix))

    transient = sorted(
        name
        for name in relevant
        if indexed.get(name, {}).get("Transient") in {"yes", "true"}
    )
    classified = {
        name
        for name in relevant
        if name.endswith(
            (
                ".service",
                ".timer",
                ".socket",
                ".path",
                ".target",
                ".automount",
            )
        )
    }
    return {
        "unit_name": unit_name,
        "triggered_by": list(direct["TriggeredBy"]),
        "wanted_by": list(direct["WantedBy"]),
        "required_by": list(direct["RequiredBy"]),
        "bound_by": list(direct["BoundBy"]),
        "upheld_by": list(direct["UpheldBy"]),
        "requisite_of": list(direct["RequisiteOf"]),
        "on_success_of": list(direct["OnSuccessOf"]),
        "on_failure_of": list(direct["OnFailureOf"]),
        "reverse_references": sorted(reverse_references),
        "service_units": by_suffix(".service"),
        "timer_units": by_suffix(".timer"),
        "socket_units": by_suffix(".socket"),
        "path_units": by_suffix(".path"),
        "target_units": by_suffix(".target"),
        "automount_units": by_suffix(".automount"),
        "other_units": sorted(relevant - classified),
        "transient_units": transient,
    }


def _verify_systemd_alias(
    alias: str,
    *,
    canonical_name: str,
    fragment_path: str,
) -> None:
    if _UNIT_RE.fullmatch(alias) is None or _UNIT_RE.fullmatch(canonical_name) is None:
        raise RuntimeError("systemd alias name is invalid")
    fragment = Path(_absolute_path(fragment_path, "systemd alias fragment"))
    candidates: dict[tuple[int, int], Path] = {}
    for directory in _SYSTEMD_UNIT_DIRECTORIES:
        candidate = directory / alias
        if not os.path.lexists(candidate):
            continue
        item = os.lstat(candidate)
        candidates[(item.st_dev, item.st_ino)] = candidate
    if not candidates:
        raise RuntimeError("systemd alias has no root-owned unit provenance")
    valid = False
    for candidate in candidates.values():
        item = os.lstat(candidate)
        if (
            not stat.S_ISLNK(item.st_mode)
            or item.st_uid != 0
            or item.st_gid != 0
        ):
            raise RuntimeError("systemd alias provenance is invalid")
        try:
            resolved = candidate.resolve(strict=True)
            expected = fragment.resolve(strict=True)
        except OSError as exc:
            raise RuntimeError("systemd alias target is unavailable") from exc
        if resolved == expected:
            valid = True
    if not valid:
        raise RuntimeError("systemd alias target does not match canonical fragment")


def _cron_entries(writer_user: str) -> tuple[str, ...]:
    crontab_stat = _validate_privileged_binary(Path(_CRONTAB), require_setid=True)
    try:
        crontab_group = grp.getgrnam("crontab")
    except KeyError as exc:
        raise RuntimeError("crontab group identity is unavailable") from exc
    if not crontab_stat.st_mode & stat.S_ISGID or crontab_stat.st_gid != crontab_group.gr_gid:
        raise RuntimeError("crontab setgid boundary is invalid")
    completed = _run_fixed(
        (_CRONTAB, "-l", "-u", writer_user),
        allowed_returncodes=frozenset({0, 1}),
    )
    entries: set[str] = set()
    if completed.returncode == 0:
        for line in completed.stdout.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                entries.add("user:" + stripped)
    elif not re.fullmatch(
        rf"no crontab for {re.escape(writer_user)}\n?",
        completed.stderr,
    ):
        raise RuntimeError("writer crontab absence is ambiguous")
    for path in (Path("/etc/crontab"), Path("/etc/cron.d")):
        candidates: tuple[Path, ...]
        if path.is_dir():
            candidates = tuple(sorted(path.iterdir()))
        elif path.exists():
            candidates = (path,)
        else:
            continue
        if len(candidates) > _MAX_INVENTORY_ITEMS:
            raise RuntimeError("system cron inventory is oversized")
        for candidate in candidates:
            item = os.lstat(candidate)
            if stat.S_ISLNK(item.st_mode):
                raise RuntimeError("system cron inventory contains a symlink")
            if not stat.S_ISREG(item.st_mode):
                continue
            raw = candidate.read_bytes()
            if len(raw) > _MAX_CRON_FILE:
                raise RuntimeError("system cron entry exceeds its bound")
            try:
                text = raw.decode("utf-8", errors="strict")
            except UnicodeError as exc:
                raise RuntimeError("system cron entry is not strict UTF-8") from exc
            for number, line in enumerate(text.splitlines(), start=1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or "=" in stripped.split()[0]:
                    continue
                fields = stripped.split()
                is_macro = fields[0].startswith("@")
                user_index = 1 if is_macro else 5
                if (
                    (is_macro and fields[0] not in {
                        "@reboot",
                        "@yearly",
                        "@annually",
                        "@monthly",
                        "@weekly",
                        "@daily",
                        "@midnight",
                        "@hourly",
                    })
                    or len(fields) <= user_index + 1
                ):
                    raise RuntimeError("system cron entry syntax is ambiguous")
                if fields[user_index] == writer_user:
                    entries.add(f"{candidate}:{number}:{stripped}")
    return tuple(sorted(entries))


def _at_jobs(writer_user: str) -> tuple[str, ...]:
    present = [path for path in _ATQ_CANDIDATES if os.path.lexists(path)]
    spool_present = any(
        os.path.lexists(path)
        for path in ("/var/spool/cron/atjobs", "/var/spool/atjobs")
    )
    absent_package = _package_absent("at")
    if not present and not spool_present and absent_package:
        return ()
    if len(present) != 1:
        raise RuntimeError("at authority evidence is ambiguous")
    executable = Path(present[0])
    _validate_privileged_binary(executable)
    completed = _run_fixed((str(executable),))
    jobs: list[str] = []
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) < 6:
            raise RuntimeError("at job inventory is invalid")
        if writer_user in fields:
            jobs.append(line.strip())
    return tuple(sorted(set(jobs)))


def _identity_group_name(gid: int) -> str | None:
    try:
        by_gid = grp.getgrgid(gid)
        by_name = grp.getgrnam(by_gid.gr_name)
    except KeyError:
        return None
    if by_name.gr_gid != gid:
        raise RuntimeError("group NSS identity is ambiguous")
    return by_gid.gr_name


_USER_SYSTEMD_ACTIVATION_SUFFIXES = (
    ".socket",
    ".path",
    ".target",
    ".automount",
)


def _protected_user_systemd_item(
    path: Path,
    *,
    trusted_roots: Sequence[Path],
) -> bool:
    """Validate one global user-systemd tree item; return False for a mask."""

    item = os.lstat(path)
    xattrs = set(os.listxattr(path, follow_symlinks=False))
    if item.st_uid != 0 or item.st_gid != 0 or xattrs & {
        "system.posix_acl_access",
        "system.posix_acl_default",
    }:
        raise RuntimeError("global user-systemd entry provenance is invalid")
    if stat.S_ISLNK(item.st_mode):
        try:
            resolved = path.resolve(strict=True)
        except OSError as exc:
            raise RuntimeError("global user-systemd symlink target is unavailable") from exc
        if resolved == Path("/dev/null"):
            # A root-owned mask is not an effective activation unit.
            return False
        resolved_roots = tuple(root.resolve(strict=True) for root in trusted_roots)
        if not any(
            resolved == root or root in resolved.parents for root in resolved_roots
        ):
            raise RuntimeError("global user-systemd symlink escapes protected roots")
        target = os.lstat(resolved)
        target_xattrs = set(os.listxattr(resolved, follow_symlinks=False))
        if (
            not stat.S_ISREG(target.st_mode)
            or target.st_uid != 0
            or target.st_gid != 0
            or stat.S_IMODE(target.st_mode) & 0o022
            or target_xattrs
            & {"system.posix_acl_access", "system.posix_acl_default"}
        ):
            raise RuntimeError("global user-systemd symlink target is not protected")
        return True
    if (
        not (stat.S_ISDIR(item.st_mode) or stat.S_ISREG(item.st_mode))
        or stat.S_IMODE(item.st_mode) & 0o022
    ):
        raise RuntimeError("global user-systemd entry is not protected")
    return True


def _protected_user_systemd_tree(
    directory: Path,
    *,
    trusted_roots: Sequence[Path],
) -> tuple[Path, ...]:
    if not _protected_user_systemd_item(directory, trusted_roots=trusted_roots):
        raise RuntimeError("global user-systemd root cannot be masked")
    entries = tuple(sorted(directory.rglob("*")))
    if len(entries) > _MAX_INVENTORY_ITEMS:
        raise RuntimeError("global user-systemd inventory is oversized")
    effective: list[Path] = []
    for entry in entries:
        if _protected_user_systemd_item(entry, trusted_roots=trusted_roots):
            effective.append(entry)
    return tuple(effective)


def _user_systemd_evidence(
    *,
    user_name: str,
    uid: int,
    home: str,
    service_read_write_paths: Sequence[str],
) -> dict[str, Any]:
    linger_path = Path("/var/lib/systemd/linger") / user_name
    runtime = Path("/run/user") / str(uid)
    private_socket = runtime / "systemd/private"
    home_directory = Path(home)
    home_unit_candidates = (
        home_directory / ".config/systemd/user",
        home_directory / ".local/share/systemd/user",
        home_directory / ".config/systemd/user-generators",
        home_directory / ".local/share/systemd/user-generators",
        home_directory / ".config/systemd/user-environment-generators",
        home_directory / ".local/share/systemd/user-environment-generators",
    )
    home_units = home_unit_candidates[0]
    manager = _native_systemd_state(f"user@{uid}.service")
    service_units: list[str] = []
    timer_units: list[str] = []
    runtime_service_units: list[str] = []
    runtime_timer_units: list[str] = []
    activation_units: list[str] = []
    runtime_activation_units: list[str] = []
    transient_units: list[str] = []
    global_service_units: set[str] = set()
    global_timer_units: set[str] = set()
    global_activation_units: set[str] = set()
    global_generators: set[str] = set()
    existing_home_unit_paths: list[str] = []
    for candidate in home_unit_candidates:
        if not os.path.lexists(candidate):
            continue
        existing_home_unit_paths.append(str(candidate))
        item = os.lstat(candidate)
        if stat.S_ISLNK(item.st_mode) or not stat.S_ISDIR(item.st_mode):
            raise RuntimeError("per-user systemd directory identity is invalid")
        entries = tuple(sorted(candidate.rglob("*")))
        if len(entries) > _MAX_INVENTORY_ITEMS:
            raise RuntimeError("per-user systemd inventory is oversized")
        for entry in entries:
            entry_item = os.lstat(entry)
            if not (stat.S_ISREG(entry_item.st_mode) or stat.S_ISLNK(entry_item.st_mode)):
                continue
            if entry.name.endswith(".service"):
                service_units.append(str(entry))
            elif entry.name.endswith(".timer"):
                timer_units.append(str(entry))
            elif entry.name.endswith(_USER_SYSTEMD_ACTIVATION_SUFFIXES):
                activation_units.append(str(entry))
    mandatory_global_roots = (
        Path("/etc/systemd/user"),
        Path("/usr/lib/systemd/user"),
    )
    optional_global_roots = (
        Path("/etc/xdg/systemd/user"),
        Path("/usr/local/lib/systemd/user"),
        Path("/usr/local/share/systemd/user"),
        Path("/usr/share/systemd/user"),
        Path("/run/systemd/user"),
    )
    global_roots = mandatory_global_roots + tuple(
        path for path in optional_global_roots if os.path.lexists(path)
    )
    for directory in mandatory_global_roots:
        if not os.path.lexists(directory):
            raise RuntimeError("required global user-systemd root is unavailable")
    for directory in global_roots:
        entries = _protected_user_systemd_tree(
            directory,
            trusted_roots=global_roots,
        )
        for entry in entries:
            if entry.name.endswith(".service"):
                global_service_units.add(str(entry))
            elif entry.name.endswith(".timer"):
                global_timer_units.add(str(entry))
            elif entry.name.endswith(_USER_SYSTEMD_ACTIVATION_SUFFIXES):
                global_activation_units.add(str(entry))
    generator_candidates = (
        Path("/etc/systemd/user-generators"),
        Path("/etc/xdg/systemd/user-generators"),
        Path("/usr/local/lib/systemd/user-generators"),
        Path("/usr/lib/systemd/user-generators"),
        Path("/run/systemd/user-generators"),
        Path("/etc/systemd/user-environment-generators"),
        Path("/usr/local/lib/systemd/user-environment-generators"),
        Path("/usr/lib/systemd/user-environment-generators"),
        Path("/run/systemd/user-environment-generators"),
    )
    generator_roots = tuple(
        path for path in generator_candidates if os.path.lexists(path)
    )
    for generator_directory in generator_roots:
        entries = _protected_user_systemd_tree(
            generator_directory,
            trusted_roots=generator_roots,
        )
        if len(entries) > _MAX_INVENTORY_ITEMS:
            raise RuntimeError("user-systemd generator inventory is oversized")
        for entry in entries:
            entry_stat = os.lstat(entry)
            if stat.S_ISREG(entry_stat.st_mode) and stat.S_IMODE(entry_stat.st_mode) & 0o111:
                global_generators.add(str(entry))
    runtime_systemd = runtime / "systemd"
    if os.path.lexists(runtime_systemd):
        runtime_entries = tuple(sorted(runtime_systemd.rglob("*")))
        if len(runtime_entries) > _MAX_INVENTORY_ITEMS:
            raise RuntimeError("runtime user-systemd inventory is oversized")
        for entry in runtime_entries:
            if entry.name.endswith(".service"):
                runtime_service_units.append(str(entry))
            elif entry.name.endswith(".timer"):
                runtime_timer_units.append(str(entry))
            elif entry.name.endswith(_USER_SYSTEMD_ACTIVATION_SUFFIXES):
                runtime_activation_units.append(str(entry))
            if "transient" in entry.parts and (
                entry.name.endswith(".service") or entry.name.endswith(".timer")
            ):
                transient_units.append(str(entry))
    home_units_text = str(home_units)
    writable_home_unit_paths = sorted(
        str(candidate)
        for candidate in home_unit_candidates
        if any(
            str(candidate) == path
            or str(candidate).startswith(path.rstrip("/") + "/")
            for path in service_read_write_paths
        )
    )
    mount_writable = home_units_text in writable_home_unit_paths
    return {
        "uid": uid,
        "linger_path": str(linger_path),
        "linger_enabled": os.path.lexists(linger_path),
        "user_manager_unit": f"user@{uid}.service",
        "load_state": manager["LoadState"],
        "active_state": manager["ActiveState"],
        "sub_state": manager["SubState"],
        "main_pid": manager["MainPID"],
        "runtime_directory_exists": os.path.lexists(runtime),
        "private_socket_exists": os.path.lexists(private_socket),
        "home_user_unit_path": home_units_text,
        "home_user_unit_path_exists": os.path.lexists(home_units),
        "home_user_unit_path_service_writable": mount_writable,
        "home_directory_exists": os.path.lexists(home_directory),
        "evaluated_home_user_unit_paths": [
            str(path) for path in home_unit_candidates
        ],
        "existing_home_user_unit_paths": sorted(existing_home_unit_paths),
        "service_writable_home_user_unit_paths": writable_home_unit_paths,
        "service_units": sorted(service_units),
        "timer_units": sorted(timer_units),
        "activation_units": sorted(activation_units),
        "transient_units": sorted(transient_units),
        "runtime_service_units": sorted(runtime_service_units),
        "runtime_timer_units": sorted(runtime_timer_units),
        "runtime_activation_units": sorted(runtime_activation_units),
        "global_service_units": sorted(global_service_units),
        "global_timer_units": sorted(global_timer_units),
        "global_activation_units": sorted(global_activation_units),
        "global_generators": sorted(global_generators),
        "evaluated_global_unit_roots": [
            str(path) for path in mandatory_global_roots + optional_global_roots
        ],
        "existing_global_unit_roots": sorted(str(path) for path in global_roots),
        "evaluated_global_generator_roots": [
            str(path) for path in generator_candidates
        ],
        "existing_global_generator_roots": sorted(
            str(path) for path in generator_roots
        ),
        "global_directories_protected": True,
    }


def _group_policy(
    *,
    gateway: ProcessAuthorityEvidence,
    children: Sequence[ProcessAuthorityEvidence],
    writer: ProcessAuthorityEvidence,
    allowed_gids: frozenset[int],
    gateway_user_name: str,
    writer_user_name: str,
    gateway_gid: int,
    writer_gid: int,
    socket_group_gid: int,
    projector_gid: int,
) -> dict[str, Any]:
    def memberships(processes: Sequence[ProcessAuthorityEvidence]) -> tuple[str, ...]:
        result: set[str] = set()
        for process in processes:
            for gid in {process.effective_gid, *process.supplementary_gids}:
                name = _identity_group_name(gid)
                if name in _DANGEROUS_GROUP_NAMES:
                    result.add(name)
        return tuple(sorted(result))

    unknown: set[int] = set()
    for process in (gateway, *children, writer):
        for gid in {process.effective_gid, *process.supplementary_gids}:
            name = _identity_group_name(gid)
            if name is None or (gid not in allowed_gids and name not in _DANGEROUS_GROUP_NAMES):
                unknown.add(gid)
    gateway_account_gids = tuple(
        sorted(set(os.getgrouplist(gateway_user_name, gateway_gid)))
    )
    writer_account_gids = tuple(
        sorted(set(os.getgrouplist(writer_user_name, writer_gid)))
    )
    protected_expected = {
        gateway_gid: ("muncho-gateway", [gateway_user_name]),
        writer_gid: ("muncho-canonical-writer", [writer_user_name]),
        socket_group_gid: ("muncho-writer-client", [gateway_user_name]),
        projector_gid: (
            "muncho-projector",
            sorted(["muncho-projector", writer_user_name]),
        ),
    }
    protected_memberships: dict[str, list[str]] = {}
    for gid, (expected_name, expected_members) in protected_expected.items():
        try:
            identity = grp.getgrgid(gid)
        except KeyError as exc:
            raise RuntimeError("protected group identity is unavailable") from exc
        primary_members = {
            account.pw_name
            for account in pwd.getpwall()
            if account.pw_gid == gid
        }
        members = sorted(primary_members | set(identity.gr_mem))
        if identity.gr_name != expected_name or members != expected_members:
            raise RuntimeError("protected group membership drifted")
        protected_memberships[str(gid)] = members
    try:
        projector_account = pwd.getpwnam("muncho-projector")
    except KeyError as exc:
        raise RuntimeError("projector account identity is unavailable") from exc
    projector_groups = tuple(
        sorted(set(os.getgrouplist(projector_account.pw_name, projector_account.pw_gid)))
    )
    if (
        projector_account.pw_uid != 992
        or projector_account.pw_gid != projector_gid
        or projector_account.pw_dir != "/nonexistent"
        or projector_account.pw_shell != "/usr/sbin/nologin"
        or projector_groups != (projector_gid,)
        or _pids_for_uid(projector_account.pw_uid)
    ):
        raise RuntimeError("projector dormant account boundary drifted")
    return {
        "complete": True,
        "evaluated_dangerous_group_names": list(_DANGEROUS_GROUP_NAMES),
        "gateway_dangerous_memberships": list(memberships((gateway,))),
        "gateway_child_dangerous_memberships": list(memberships(children)),
        "writer_dangerous_memberships": list(memberships((writer,))),
        "unknown_privileged_gids": sorted(unknown),
        "gateway_account_gids": list(gateway_account_gids),
        "writer_account_gids": list(writer_account_gids),
        "protected_group_memberships": protected_memberships,
        "projector_identity": {
            "uid": projector_account.pw_uid,
            "gid": projector_account.pw_gid,
            "home": projector_account.pw_dir,
            "shell": projector_account.pw_shell,
            "gids": list(projector_groups),
            "process_pids": [],
        },
    }


def _authority_mapping(
    *,
    processes: Sequence[ProcessAuthorityEvidence],
    sudo_commands: tuple[str, ...],
    doas_commands: tuple[str, ...],
    polkit_actions: tuple[str, ...],
    writable_unit_paths: tuple[str, ...],
    writable_cron_paths: tuple[str, ...],
) -> dict[str, Any]:
    capabilities = tuple(
        sorted({capability for process in processes for capability in process.effective_capabilities})
    )
    elevated = bool(sudo_commands or doas_commands or capabilities)
    systemd_authority = bool(polkit_actions or writable_unit_paths or elevated)
    cron_executable_elevation = any(not process.no_new_privileges for process in processes)
    cron_authority = bool(writable_cron_paths or elevated or cron_executable_elevation)
    return {
        "can_manage_writer_units": systemd_authority,
        "can_create_transient_units": systemd_authority,
        "can_manage_timers": systemd_authority,
        "can_manage_cron": cron_authority,
        "can_switch_to_writer_uid": bool(
            sudo_commands
            or doas_commands
            or {"CAP_SETUID", "CAP_SETGID"} & set(capabilities)
        ),
        "can_invoke_polkit_actions": bool(polkit_actions),
        "can_write_systemd_unit_paths": bool(writable_unit_paths),
        "can_write_cron_paths": bool(writable_cron_paths),
        "authorized_systemd_dbus_actions": list(polkit_actions),
        "authorized_polkit_actions": list(polkit_actions),
        "authorized_sudo_commands": list(sudo_commands),
        "authorized_doas_commands": list(doas_commands),
        "effective_capabilities": list(capabilities),
        "writable_systemd_unit_paths": list(writable_unit_paths),
        "writable_cron_paths": list(writable_cron_paths),
    }


def collect_writer_authority_surface(
    snapshot: Mapping[str, Any],
    *,
    gateway_pid: int,
    writer_pid: int,
    observed_at_unix: int,
) -> dict[str, Any]:
    """Collect the complete evaluator authority surface from the live host."""

    _require_root_linux()
    if not isinstance(snapshot, Mapping):
        raise TypeError("authority snapshot must be a mapping")
    gateway_uid = _positive_integer(snapshot.get("gateway_uid"), "gateway UID")
    gateway_gid = _positive_integer(snapshot.get("gateway_gid"), "gateway GID")
    writer_uid = _positive_integer(snapshot.get("writer_uid"), "writer UID")
    writer_gid = _positive_integer(snapshot.get("writer_gid"), "writer GID")
    projector_gid = _positive_integer(snapshot.get("projector_gid"), "projector GID")
    socket_group_gid = _positive_integer(
        (snapshot.get("socket") or {}).get("expected_group_gid"), "socket group GID"
    )
    gateway_expected_groups = tuple(sorted((gateway_gid, socket_group_gid)))
    writer_expected_groups = tuple(sorted((writer_gid, projector_gid)))
    if tuple(snapshot.get("gateway_supplementary_gids") or ()) != gateway_expected_groups:
        raise RuntimeError("gateway approved group set is not exact")
    if tuple(snapshot.get("writer_supplementary_gids") or ()) != writer_expected_groups:
        raise RuntimeError("writer approved group set is not exact")
    writer_policy = (snapshot.get("writer_deployment") or {}).get("policy") or {}
    gateway_policy = (snapshot.get("gateway_deployment") or {}).get("policy") or {}
    gateway = _observe_process(gateway_pid)
    writer = _observe_process(writer_pid)
    child_pids = _descendant_pids(gateway_pid)
    children = tuple(_observe_process(pid) for pid in child_pids)
    for process, expected_uid, expected_gid, expected_groups, policy in (
        (gateway, gateway_uid, gateway_gid, gateway_expected_groups, gateway_policy),
        (writer, writer_uid, writer_gid, writer_expected_groups, writer_policy),
    ):
        if (
            process.effective_uid != expected_uid
            or process.effective_gid != expected_gid
            or process.supplementary_gids != expected_groups
            or process.executable != policy.get("interpreter")
            or list(process.argv) != list(policy.get("exec_start") or ())
        ):
            raise RuntimeError("service process authority identity drifted")
    for child in children:
        if (
            child.effective_uid != gateway_uid
            or child.effective_gid != gateway_gid
            or child.supplementary_gids != gateway_expected_groups
            or not child.no_new_privileges
            or child.effective_capabilities
        ):
            raise RuntimeError("gateway child authority is not isolated")
    gateway_uid_pids = _pids_for_uid(gateway_uid)
    writer_uid_pids = _pids_for_uid(writer_uid)
    if gateway_uid_pids != tuple(sorted((gateway_pid, *child_pids))):
        raise RuntimeError("gateway UID process inventory is unattributed")
    writer_processes = tuple(_observe_process(pid) for pid in writer_uid_pids)
    if writer_uid_pids != (writer_pid,):
        unattributed = [
            f"pid:{process.pid}:{process.executable}"
            for process in writer_processes
            if process.pid != writer_pid
        ]
    else:
        unattributed = []
    try:
        gateway_user = pwd.getpwuid(gateway_uid)
        writer_user = pwd.getpwuid(writer_uid)
    except KeyError as exc:
        raise RuntimeError("service passwd identity is unavailable") from exc
    if gateway_user.pw_uid != gateway_uid or writer_user.pw_uid != writer_uid:
        raise RuntimeError("service passwd identity is ambiguous")
    sudo_commands = _sudo_policy(gateway_user.pw_name)
    doas_commands = _doas_policy()
    gateway_polkit = _polkit_actions(gateway)
    child_polkit = tuple(
        sorted({action for child in children for action in _polkit_actions(child)})
    )
    unit_names = (
        "muncho-canonical-writer.service",
        "muncho-canonical-writer-export.service",
        "muncho-canonical-writer-export.timer",
    )
    unit_paths = tuple(
        directory / name for directory in _SYSTEMD_UNIT_DIRECTORIES for name in unit_names
    )
    gateway_writable_units = _writable_paths(unit_paths, identities=(gateway,))
    child_writable_units = _writable_paths(unit_paths, identities=children) if children else ()
    gateway_writable_cron = _writable_paths(_CRON_PATHS, identities=(gateway,))
    child_writable_cron = _writable_paths(_CRON_PATHS, identities=children) if children else ()
    gateway_authority = _authority_mapping(
        processes=(gateway,),
        sudo_commands=sudo_commands,
        doas_commands=doas_commands,
        polkit_actions=gateway_polkit,
        writable_unit_paths=gateway_writable_units,
        writable_cron_paths=gateway_writable_cron,
    )
    # The future same-UID child surface retains the parent's no-new-privileges,
    # sudo/doas and filesystem boundaries.  Any observed child is additionally
    # probed by exact PID/start-time above.
    child_authority = _authority_mapping(
        processes=children or (gateway,),
        sudo_commands=sudo_commands,
        doas_commands=doas_commands,
        polkit_actions=tuple(sorted(set(gateway_polkit) | set(child_polkit))),
        writable_unit_paths=tuple(sorted(set(gateway_writable_units) | set(child_writable_units))),
        writable_cron_paths=tuple(sorted(set(gateway_writable_cron) | set(child_writable_cron))),
    )
    writer_sudo_commands = _sudo_policy(writer_user.pw_name)
    writer_polkit = _polkit_actions(writer)
    writer_writable_units = _writable_paths(unit_paths, identities=(writer,))
    writer_writable_cron = _writable_paths(_CRON_PATHS, identities=(writer,))
    writer_authority = _authority_mapping(
        processes=(writer,),
        sudo_commands=writer_sudo_commands,
        doas_commands=doas_commands,
        polkit_actions=writer_polkit,
        writable_unit_paths=writer_writable_units,
        writable_cron_paths=writer_writable_cron,
    )
    systemd_inventory = _systemd_inventory(writer_uid, gateway_uid)
    writer_services = tuple(
        sorted(
            item["Id"]
            for item in systemd_inventory
            if item["Id"].endswith(".service")
            and item["User"]
            and (
                int(item["User"])
                if item["User"].isdigit()
                else pwd.getpwnam(item["User"]).pw_uid
            )
            == writer_uid
        )
    )
    writer_timers = tuple(
        sorted(
            item["Id"]
            for item in systemd_inventory
            if item["Id"].endswith(".timer")
            and set(item["Triggers"].split()) & set(writer_services)
        )
    )
    writer_transient = tuple(
        sorted(
            item["Id"]
            for item in systemd_inventory
            if item["Transient"] in {"yes", "true"}
            and (
                item["Id"] in writer_services or item["Id"] in writer_timers
            )
        )
    )
    gateway_services = tuple(
        sorted(
            item["Id"]
            for item in systemd_inventory
            if item["Id"].endswith(".service")
            and item["User"]
            and (
                int(item["User"])
                if item["User"].isdigit()
                else pwd.getpwnam(item["User"]).pw_uid
            )
            == gateway_uid
        )
    )
    gateway_timers = tuple(
        sorted(
            item["Id"]
            for item in systemd_inventory
            if item["Id"].endswith(".timer")
            and set(item["Triggers"].split()) & set(gateway_services)
        )
    )
    gateway_transient = tuple(
        sorted(
            item["Id"]
            for item in systemd_inventory
            if item["Transient"] in {"yes", "true"}
            and (
                item["Id"] in gateway_services
                or item["Id"] in gateway_timers
            )
        )
    )
    writer_reverse_activation = _reverse_activation_evidence(
        "muncho-canonical-writer.service",
        systemd_inventory,
    )
    gateway_reverse_activation = _reverse_activation_evidence(
        "hermes-cloud-gateway.service",
        systemd_inventory,
    )
    authority_template = snapshot.get("writer_authority_surface") or {}
    exporter = authority_template.get("projection_exporter") or {}
    exporter_policy = exporter.get("policy") or {}
    if exporter_policy.get("enabled") is not False:
        raise RuntimeError("live authority collector currently requires exporter disabled")
    inventory = {
        "complete": True,
        "writer_uid_service_units": list(writer_services),
        "writer_uid_timer_units": list(writer_timers),
        "writer_uid_transient_units": list(writer_transient),
        "writer_uid_cron_entries": list(_cron_entries(writer_user.pw_name)),
        "writer_uid_at_jobs": list(_at_jobs(writer_user.pw_name)),
        "writer_uid_process_executables": sorted(
            {process.executable for process in writer_processes}
        ),
        "writer_uid_unattributed_processes": sorted(unattributed),
        "writer_unit_reverse_activation": writer_reverse_activation,
        "gateway_uid_service_units": list(gateway_services),
        "gateway_uid_timer_units": list(gateway_timers),
        "gateway_uid_transient_units": list(gateway_transient),
        "gateway_uid_cron_entries": list(_cron_entries(gateway_user.pw_name)),
        "gateway_uid_at_jobs": list(_at_jobs(gateway_user.pw_name)),
        "gateway_uid_process_executables": sorted(
            {gateway.executable, *(child.executable for child in children)}
        ),
        "gateway_uid_unattributed_processes": [],
        "gateway_unit_reverse_activation": gateway_reverse_activation,
        "gateway_writable_writer_unit_files": list(gateway_writable_units),
        "gateway_child_writable_writer_unit_files": list(child_writable_units),
    }
    surface = {
        "complete": True,
        "collected_by_uid": 0,
        "observed_at_unix": _nonnegative_integer(observed_at_unix, "authority observed time"),
        "identities": {
            "gateway": gateway.evaluator_mapping(),
            "gateway_children": {
                "complete": True,
                "processes": [child.evaluator_mapping() for child in children],
            },
            "writer": writer.evaluator_mapping(),
        },
        "group_policy": _group_policy(
            gateway=gateway,
            children=children,
            writer=writer,
            allowed_gids=frozenset(
                {gateway_gid, socket_group_gid, writer_gid, projector_gid}
            ),
            gateway_user_name=gateway_user.pw_name,
            writer_user_name=writer_user.pw_name,
            gateway_gid=gateway_gid,
            writer_gid=writer_gid,
            socket_group_gid=socket_group_gid,
            projector_gid=projector_gid,
        ),
        "gateway_authority": gateway_authority,
        "gateway_child_authority": child_authority,
        "writer_authority": writer_authority,
        "user_systemd": {
            "complete": True,
            "gateway": _user_systemd_evidence(
                user_name=gateway_user.pw_name,
                uid=gateway_uid,
                home=gateway_user.pw_dir,
                service_read_write_paths=tuple(
                    gateway_policy.get("read_write_paths") or ()
                ),
            ),
            "writer": _user_systemd_evidence(
                user_name=writer_user.pw_name,
                uid=writer_uid,
                home=writer_user.pw_dir,
                service_read_write_paths=tuple(
                    writer_policy.get("read_write_paths") or ()
                ),
            ),
        },
        "privileged_execution_inventory": inventory,
        "projection_exporter": {
            "policy": json.loads(_canonical_bytes(exporter_policy).decode("utf-8")),
            "attestation": {
                "complete": True,
                "enabled": False,
                "unit": {},
                "timer": {},
            },
        },
    }
    # Final PID/start/UID-set fence closes process exit/reuse races during the
    # fixed command probes and filesystem inventory.
    for before in (gateway, *children, writer):
        after = _observe_process(before.pid)
        if after != before:
            raise RuntimeError("process authority evidence changed during collection")
    if _pids_for_uid(gateway_uid) != gateway_uid_pids or _pids_for_uid(writer_uid) != writer_uid_pids:
        raise RuntimeError("process UID inventory changed during collection")
    return surface


def collect_legacy_helper_absence(
    *,
    gateway_uid: int,
    gateway_gids: Sequence[int],
) -> dict[str, Any]:
    """Collect exact retired-helper absence, including its retired parent."""

    _require_root_linux()
    file_exists = os.path.lexists(LEGACY_CLOUD_SQL_HELPER_PATH)
    parent_exists = os.path.lexists(LEGACY_CLOUD_SQL_HELPER_PARENT)
    file_symlink = file_exists and stat.S_ISLNK(os.lstat(LEGACY_CLOUD_SQL_HELPER_PATH).st_mode)
    parent_symlink = parent_exists and stat.S_ISLNK(os.lstat(LEGACY_CLOUD_SQL_HELPER_PARENT).st_mode)
    if file_exists or parent_exists:
        raise RuntimeError("retired Cloud SQL helper path is not fully absent")
    _positive_integer(gateway_uid, "gateway UID")
    _strict_integers(tuple(gateway_gids), "gateway helper group set", nonempty=True)
    return {
        "path": str(LEGACY_CLOUD_SQL_HELPER_PATH),
        "file_exists": False,
        "file_symlink": bool(file_symlink),
        "parent_exists": False,
        "parent_symlink": bool(parent_symlink),
        "gateway_access": {"read": False, "write": False, "execute": False},
    }


def _boot_id_sha256() -> str:
    raw = _read_bounded(
        Path("/proc/sys/kernel/random/boot_id"), maximum=128
    )
    assert isinstance(raw, str)
    value = raw.strip()
    try:
        parsed = uuid.UUID(value)
    except ValueError as exc:
        raise RuntimeError("native observation boot identity is invalid") from exc
    if parsed.int == 0 or str(parsed) != value:
        raise RuntimeError("native observation boot identity is invalid")
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _host_identity_sha256() -> str:
    raw = _read_bounded(Path("/etc/machine-id"), maximum=128)
    assert isinstance(raw, str)
    machine_id = raw.strip()
    if re.fullmatch(r"[0-9a-f]{32}", machine_id) is None or set(machine_id) == {"0"}:
        raise RuntimeError("native observation machine identity is invalid")
    try:
        hostname = socket.gethostname()
    except OSError as exc:
        raise RuntimeError("native observation hostname is unavailable") from exc
    if (
        not hostname
        or len(hostname) > 253
        or hostname != hostname.strip()
        or any(character in hostname for character in "\x00\r\n")
    ):
        raise RuntimeError("native observation hostname is invalid")
    return _sha256_json({"machine_id": machine_id, "hostname": hostname})


def current_host_identity_sha256() -> str:
    _require_root_linux()
    return _host_identity_sha256()


def _boottime_ns() -> int:
    clock = getattr(time, "CLOCK_BOOTTIME", None)
    getter = getattr(time, "clock_gettime_ns", None)
    if clock is None or not callable(getter):
        raise RuntimeError("native observation boottime clock is unavailable")
    value = int(getter(clock))
    if value <= 0:
        raise RuntimeError("native observation boottime is invalid")
    return value


def _trusted_file_sha256(path: Path, *, maximum: int = 1024 * 1024 * 1024) -> str:
    try:
        before = os.lstat(path)
    except OSError as exc:
        raise RuntimeError(f"trusted observation file is unavailable: {path}") from exc
    xattrs = set(os.listxattr(path, follow_symlinks=False))
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_uid != 0
        or stat.S_IMODE(before.st_mode) & 0o022
        or before.st_size < 0
        or before.st_size > maximum
        or xattrs & {"system.posix_acl_access", "system.posix_acl_default"}
    ):
        raise RuntimeError(f"trusted observation file identity is invalid: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise RuntimeError("trusted observation file changed during open")
        digest = hashlib.sha256()
        total = 0
        while total <= maximum:
            chunk = os.read(descriptor, min(1024 * 1024, maximum + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            digest.update(chunk)
        if total != before.st_size or total > maximum:
            raise RuntimeError("trusted observation file size changed")
    finally:
        os.close(descriptor)
    after = os.lstat(path)
    if (
        (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    ):
        raise RuntimeError("trusted observation file changed while hashing")
    return digest.hexdigest()


_NATIVE_SYSTEMD_PROPERTIES = (
    "LoadState",
    "ActiveState",
    "SubState",
    "MainPID",
    "FragmentPath",
    "UnitFileState",
)


def _native_systemd_state(unit_name: str) -> dict[str, Any]:
    if not isinstance(unit_name, str) or _UNIT_RE.fullmatch(unit_name) is None:
        raise ValueError("native observation unit name is invalid")
    completed = _run_fixed(
        (
            _SYSTEMCTL,
            "show",
            *(f"--property={name}" for name in _NATIVE_SYSTEMD_PROPERTIES),
            "--",
            unit_name,
        )
    )
    fields: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        name, separator, value = line.partition("=")
        if separator != "=" or name not in _NATIVE_SYSTEMD_PROPERTIES or name in fields:
            raise RuntimeError("native systemd evidence fields are invalid")
        fields[name] = value
    if set(fields) != set(_NATIVE_SYSTEMD_PROPERTIES):
        raise RuntimeError("native systemd evidence is incomplete")
    try:
        main_pid = int(fields["MainPID"] or "0")
    except ValueError as exc:
        raise RuntimeError("native systemd MainPID is invalid") from exc
    if main_pid < 0:
        raise RuntimeError("native systemd MainPID is invalid")
    return {**fields, "MainPID": main_pid}


def _discord_process_pids() -> list[int]:
    result: set[int] = set()
    try:
        discord_uid = pwd.getpwnam("muncho-discord-edge").pw_uid
    except KeyError:
        discord_uid = None
    for item in Path("/proc").iterdir():
        if not item.name.isdigit():
            continue
        try:
            status = (item / "status").read_text(encoding="ascii")
            raw = (item / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if len(status) > _MAX_PROCESS_FILE or len(raw) > 64 * 1024:
            raise RuntimeError("Discord process evidence exceeds its bound")
        uid_lines = [line for line in status.splitlines() if line.startswith("Uid:")]
        if len(uid_lines) != 1:
            raise RuntimeError("Discord process UID evidence is ambiguous")
        values = uid_lines[0].partition(":")[2].split()
        if len(values) != 4 or not all(value.isdigit() for value in values):
            raise RuntimeError("Discord process UID evidence is invalid")
        try:
            argv = tuple(
                value.decode("utf-8", errors="strict")
                for value in raw.rstrip(b"\x00").split(b"\x00")
                if value
            )
        except UnicodeError as exc:
            raise RuntimeError("Discord process argv is invalid") from exc
        if (
            (discord_uid is not None and int(values[1]) == discord_uid)
            or "gateway.discord_edge_bootstrap" in argv
            or "scripts.discord_edge_bootstrap" in argv
            or any(
                value.endswith("/gateway/discord_edge_bootstrap.py")
                or value.endswith("/scripts/discord_edge_bootstrap.py")
                for value in argv
            )
        ):
            result.add(int(item.name))
    return sorted(result)


def _collect_discord_absence(plan: NativeObservationPlan) -> dict[str, Any]:
    policy = plan.value["discord"]
    systemd = _native_systemd_state(policy["unit_name"])
    evidence = {
        "unit_name": policy["unit_name"],
        "unit_exists": systemd["LoadState"] != "not-found",
        "unit_enabled": systemd["UnitFileState"] not in {"", "disabled"},
        "unit_active": systemd["ActiveState"] == "active",
        "main_pid": systemd["MainPID"],
        "config_exists": os.path.lexists(policy["config_path"]),
        "token_exists": os.path.lexists(policy["token_path"]),
        "socket_exists": os.path.lexists(policy["socket_path"]),
        "process_pids": _discord_process_pids(),
    }
    _validate_discord_absence(evidence, policy)
    return evidence


def _hash_native_mapping(path: Path, policy: Mapping[str, Any]) -> dict[str, str]:
    try:
        before = os.lstat(path)
    except OSError as exc:
        raise RuntimeError("native executable mapping is unavailable") from exc
    xattrs = set(os.listxattr(path, follow_symlinks=False))
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != policy["require_single_link"]
        or before.st_uid != policy["required_owner_uid"]
        or before.st_gid != policy["required_owner_gid"]
        or stat.S_IMODE(before.st_mode) & 0o222
        or xattrs
        or before.st_size < 1
        or before.st_size > 1024 * 1024 * 1024
    ):
        raise RuntimeError("native executable mapping protection is invalid")
    return {"path": str(path), "sha256": _trusted_file_sha256(path)}


def _discover_native_mappings(
    process: ProcessAuthorityEvidence,
    plan: NativeObservationPlan,
) -> tuple[list[dict[str, str]], list[str]]:
    maps_path = Path("/proc") / str(process.pid) / "maps"
    start_before = _process_start_time(maps_path.parent)
    raw = _read_bounded(maps_path, maximum=16 * 1024 * 1024)
    assert isinstance(raw, str)
    artifact_root = Path(plan.value["artifact_root"])
    policy = plan.value["native_discovery_policy"]
    allowed_roots = tuple(Path(root) for root in policy["allowed_roots"])
    paths: set[Path] = set()
    kernel_mappings: set[str] = set()

    def executable_lines(text: str) -> tuple[str, ...]:
        projected: list[str] = []
        for candidate in text.splitlines():
            candidate_fields = candidate.split(None, 5)
            if len(candidate_fields) < 5:
                raise RuntimeError("native executable map line is invalid")
            candidate_permissions = candidate_fields[1]
            if len(candidate_permissions) != 4 or any(
                character not in "rwxps-" for character in candidate_permissions
            ):
                raise RuntimeError("native executable map permissions are invalid")
            if "x" in candidate_permissions:
                projected.append(candidate)
        return tuple(projected)

    projected_before = executable_lines(raw)
    for line in projected_before:
        fields = line.split(None, 5)
        permissions = fields[1]
        assert "x" in permissions
        if len(fields) == 5:
            raise RuntimeError("anonymous executable mapping is forbidden")
        raw_path = fields[5]
        if raw_path.startswith("["):
            if raw_path not in policy["allowed_kernel_executable_mappings"]:
                raise RuntimeError("unapproved kernel executable mapping is present")
            kernel_mappings.add(raw_path)
            continue
        if raw_path.endswith(" (deleted)"):
            raise RuntimeError("native executable mapping was deleted")
        path = Path(_absolute_path(raw_path, "native executable mapping"))
        if path == artifact_root or artifact_root in path.parents:
            continue
        if not any(path == root or root in path.parents for root in allowed_roots):
            raise RuntimeError("native executable mapping escapes discovery roots")
        paths.add(path)
    if (
        not paths
        or not kernel_mappings
        or len(paths) > policy["maximum_mappings"]
    ):
        raise RuntimeError("native executable mapping set is empty or oversized")
    result = [_hash_native_mapping(path, policy) for path in sorted(paths)]
    raw_after = _read_bounded(maps_path, maximum=16 * 1024 * 1024)
    assert isinstance(raw_after, str)
    if (
        executable_lines(raw_after) != projected_before
        or _process_start_time(maps_path.parent) != start_before
        or start_before != process.start_time_ticks
    ):
        raise RuntimeError("process changed during native mapping discovery")
    return result, sorted(kernel_mappings)


def collect_native_observation(
    plan: NativeObservationPlan,
    *,
    approved_plan_sha256: str,
) -> dict[str, Any]:
    """Collect live native mappings only after exact plan-digest approval."""

    _require_root_linux()
    if not isinstance(plan, NativeObservationPlan):
        raise TypeError("native observation plan is required")
    if plan.sha256 != _digest(approved_plan_sha256, "approved native observation plan"):
        raise PermissionError("native observation plan digest is not approved")
    boot = _boot_id_sha256()
    host = _host_identity_sha256()
    if boot != plan.value["boot_id_sha256"] or host != plan.value["host_identity_sha256"]:
        raise RuntimeError("native observation host or boot identity drifted")
    for name in ("gateway_unit", "writer_unit", "gateway_config", "writer_config"):
        binding = plan.value[name]
        if _trusted_file_sha256(Path(binding["path"])) != binding["sha256"]:
            raise RuntimeError(f"native observation {name} digest drifted")
    release_manifest = Path(plan.value["artifact_root"]) / "release-manifest.json"
    if _trusted_file_sha256(release_manifest, maximum=64 * 1024 * 1024) != plan.value[
        "release_manifest_file_sha256"
    ]:
        raise RuntimeError("native release manifest file digest drifted")
    database = plan.value["database"]
    if _trusted_file_sha256(Path(database["ca_path"]), maximum=16 * 1024 * 1024) != database[
        "ca_sha256"
    ]:
        raise RuntimeError("native database CA digest drifted")
    services: dict[str, dict[str, Any]] = {}
    for label in ("gateway", "writer"):
        unit = plan.value[f"{label}_unit"]
        state = _native_systemd_state(unit["name"])
        if (
            state["LoadState"] != "loaded"
            or state["ActiveState"] != "active"
            or state["SubState"] != "running"
            or state["UnitFileState"] != "disabled"
            or state["FragmentPath"] != unit["path"]
            or state["MainPID"] <= 1
        ):
            raise RuntimeError(f"native {label} service is not exactly active")
        process = _observe_process(state["MainPID"])
        if list(process.argv) != plan.value[f"{label}_argv"]:
            raise RuntimeError(f"native {label} service argv drifted")
        external_mappings, kernel_mappings = _discover_native_mappings(
            process, plan
        )
        services[label] = {
            "unit_name": unit["name"],
            "active_state": "active",
            "sub_state": "running",
            "unit_file_state": state["UnitFileState"],
            "main_pid": process.pid,
            "start_time_ticks": process.start_time_ticks,
            "argv": list(process.argv),
            "external_native_mappings": external_mappings,
            "kernel_executable_mappings": kernel_mappings,
            "process_authority": process.evaluator_mapping(),
        }
    observed_boot = _boottime_ns()
    observation = {
        "boot_id_sha256": boot,
        "host_identity_sha256": host,
        "observed_at_unix": int(time.time()),
        "observed_at_boottime_ns": observed_boot,
        "expires_at_boottime_ns": (
            observed_boot + NATIVE_OBSERVATION_TTL_SECONDS * 1_000_000_000
        ),
        "gateway_service": services["gateway"],
        "writer_service": services["writer"],
        "discord_absence": _collect_discord_absence(plan),
        "legacy_helper_absence": collect_legacy_helper_absence(
            gateway_uid=plan.value["identities"]["gateway_uid"],
            gateway_gids=plan.value["identities"]["gateway_supplementary_gids"],
        ),
    }
    if _boot_id_sha256() != boot or _host_identity_sha256() != host:
        raise RuntimeError("native observation host changed during collection")
    # Parse the stage through the final receipt validator's live-service rules
    # before any caller can persist it.
    for label in ("gateway", "writer"):
        _validate_live_service(
            observation[f"{label}_service"],
            unit=plan.value[f"{label}_unit"],
            expected_argv=plan.value[f"{label}_argv"],
            expected_identity=_planned_process_identity(plan, label),
            discovery_policy=plan.value["native_discovery_policy"],
            artifact_root=plan.value["artifact_root"],
        )
    return observation


_STAGE_KEYS = frozenset(
    {
        "schema",
        "native_observation_plan_sha256",
        "owner_approval_receipt_sha256",
        "host_preparation_receipt_sha256",
        "external_iam_receipt_sha256",
        "plan",
        "observation",
    }
)


def _expected_stage_path(plan: NativeObservationPlan) -> Path:
    return (
        DEFAULT_NATIVE_OBSERVATION_STAGE_ROOT
        / plan.value["revision"]
        / plan.sha256
        / "native-observation-stage.json"
    )


def _expected_receipt_path(plan: NativeObservationPlan) -> Path:
    return (
        DEFAULT_NATIVE_OBSERVATION_EVIDENCE_ROOT
        / plan.value["revision"]
        / plan.sha256
        / "native-observation.json"
    )


def _root_mapping_bytes(value: Mapping[str, Any]) -> bytes:
    raw = _canonical_bytes(dict(value)) + b"\n"
    if len(raw) > _MAX_COMMAND_OUTPUT:
        raise ValueError("root authority JSON exceeds its bound")
    return raw


def _write_append_only_root_mapping(
    path: Path,
    value: Mapping[str, Any],
) -> None:
    _require_root_linux()
    _absolute_path(str(path), "root authority JSON path")
    _validate_root_parent_chain(path.parent.parent)
    parent = os.lstat(path.parent)
    if (
        stat.S_ISLNK(parent.st_mode)
        or not stat.S_ISDIR(parent.st_mode)
        or parent.st_uid != 0
        or parent.st_gid != 0
        or stat.S_IMODE(parent.st_mode) != 0o700
    ):
        raise PermissionError("root authority JSON directory is not root-only")
    raw = _root_mapping_bytes(value)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        os.fchown(descriptor, 0, 0)
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise OSError("root authority JSON write made no progress")
            offset += written
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        descriptor = -1
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _read_root_mapping(path: Path) -> Mapping[str, Any]:
    _validate_root_parent_chain(path.parent)
    before = os.lstat(path)
    xattrs = set(os.listxattr(path, follow_symlinks=False))
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_uid != 0
        or before.st_gid != 0
        or stat.S_IMODE(before.st_mode) != 0o400
        or xattrs
    ):
        raise ValueError("root authority JSON ownership or mode is invalid")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError("root authority JSON changed during open")
        raw = b""
        while len(raw) <= _MAX_COMMAND_OUTPUT:
            chunk = os.read(descriptor, min(4096, _MAX_COMMAND_OUTPUT + 1 - len(raw)))
            if not chunk:
                break
            raw += chunk
    finally:
        os.close(descriptor)
    if not raw or len(raw) > _MAX_COMMAND_OUTPUT:
        raise ValueError("root authority JSON size is invalid")
    after = os.lstat(path)
    if (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) != (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ):
        raise ValueError("root authority JSON changed while reading")

    def reject_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, item in pairs:
            if name in result:
                raise ValueError("root authority JSON contains duplicate keys")
            result[name] = item
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"root authority JSON contains non-JSON constant: {value}")

    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("root authority JSON is invalid") from exc
    if not isinstance(value, Mapping):
        raise ValueError("root authority JSON must be an object")
    canonical = _canonical_bytes(dict(value))
    if raw not in {canonical, canonical + b"\n"}:
        raise ValueError("root authority JSON is not canonical")
    return value


def load_owner_approval_receipt(
    path_value: str | os.PathLike[str],
    *,
    scope: str,
    plan_sha256: str,
    now_unix: int | None = None,
) -> OwnerApprovalReceipt:
    _require_root_linux()
    _digest(plan_sha256, "owner-approved plan digest")
    path = Path(_absolute_path(os.fspath(path_value), "owner approval path"))
    receipt = OwnerApprovalReceipt.from_mapping(_read_root_mapping(path))
    receipt.require(
        scope=scope,
        plan_sha256=plan_sha256,
        now_unix=int(time.time()) if now_unix is None else now_unix,
    )
    if path != owner_approval_receipt_path(receipt):
        raise ValueError("owner approval path is not receipt-addressed")
    return receipt


def owner_approval_receipt_path(receipt: OwnerApprovalReceipt) -> Path:
    if not isinstance(receipt, OwnerApprovalReceipt):
        raise TypeError("owner approval receipt is required")
    return (
        DEFAULT_OWNER_APPROVAL_ROOT
        / str(receipt.value["scope"])
        / str(receipt.value["plan_sha256"])
        / f"{receipt.sha256}.json"
    )


def load_external_iam_receipt(
    path_value: str | os.PathLike[str],
    *,
    expected_policy_sha256: str,
    now_unix: int | None = None,
) -> ExternalIAMReceipt:
    _require_root_linux()
    path = Path(_absolute_path(os.fspath(path_value), "external IAM receipt path"))
    if path != DEFAULT_EXTERNAL_IAM_LIVE_PATH:
        raise ValueError("external IAM live receipt path is not pinned")
    receipt = ExternalIAMReceipt.from_mapping(_read_root_mapping(path))
    if receipt.policy_sha256 != _digest(
        expected_policy_sha256, "external IAM policy digest"
    ):
        raise ValueError("external IAM receipt policy does not match the plan")
    receipt.require_fresh(int(time.time()) if now_unix is None else now_unix)
    return receipt


def write_native_observation_stage(
    path: str | os.PathLike[str],
    *,
    plan: NativeObservationPlan,
    approved_plan_sha256: str,
    owner_approval_receipt: Mapping[str, Any],
    host_preparation_receipt_sha256: str,
    external_iam_receipt_sha256: str,
) -> Mapping[str, Any]:
    target = Path(path)
    if target != _expected_stage_path(plan):
        raise ValueError("native observation stage path is not plan-addressed")
    approval_digest = validate_native_observation_approval(
        owner_approval_receipt,
        expected_plan_sha256=approved_plan_sha256,
        now_unix=int(time.time()),
    )
    host_preparation_digest = _digest(
        host_preparation_receipt_sha256,
        "native stage host preparation receipt",
    )
    external_iam_digest = _digest(
        external_iam_receipt_sha256,
        "native stage external IAM receipt",
    )
    observation = collect_native_observation(
        plan, approved_plan_sha256=approved_plan_sha256
    )
    stage = {
        "schema": NATIVE_OBSERVATION_STAGE_SCHEMA,
        "native_observation_plan_sha256": plan.sha256,
        "owner_approval_receipt_sha256": approval_digest,
        "host_preparation_receipt_sha256": host_preparation_digest,
        "external_iam_receipt_sha256": external_iam_digest,
        "plan": plan.to_mapping(),
        "observation": observation,
    }
    _write_append_only_root_mapping(target, stage)
    return stage


def _validate_stage(
    value: Mapping[str, Any],
    *,
    expected_plan_sha256: str,
) -> tuple[NativeObservationPlan, Mapping[str, Any]]:
    stage = _exact_keys(value, _STAGE_KEYS, "native observation stage")
    if stage.get("schema") != NATIVE_OBSERVATION_STAGE_SCHEMA:
        raise ValueError("native observation stage schema is invalid")
    plan = NativeObservationPlan.from_mapping(stage["plan"])
    if (
        plan.sha256 != stage.get("native_observation_plan_sha256")
        or plan.sha256 != expected_plan_sha256
    ):
        raise ValueError("native observation stage plan digest drifted")
    _digest(
        stage.get("owner_approval_receipt_sha256"),
        "native stage owner approval receipt",
    )
    _digest(
        stage.get("host_preparation_receipt_sha256"),
        "native stage host preparation receipt",
    )
    _digest(
        stage.get("external_iam_receipt_sha256"),
        "native stage external IAM receipt",
    )
    observation = _exact_keys(
        stage.get("observation"), _LIVE_OBSERVATION_KEYS, "native stage observation"
    )
    for label in ("gateway", "writer"):
        _validate_live_service(
            observation[f"{label}_service"],
            unit=plan.value[f"{label}_unit"],
            expected_argv=plan.value[f"{label}_argv"],
            expected_identity=_planned_process_identity(plan, label),
            discovery_policy=plan.value["native_discovery_policy"],
            artifact_root=plan.value["artifact_root"],
        )
    _validate_discord_absence(observation["discord_absence"], plan.value["discord"])
    return plan, observation


def finalize_native_observation_stage(
    stage_path: str | os.PathLike[str],
    receipt_path: str | os.PathLike[str],
    *,
    approved_plan_sha256: str,
    owner_approval_receipt: Mapping[str, Any],
    host_preparation_receipt_sha256: str,
    external_iam_receipt_sha256: str,
) -> NativeObservationReceipt:
    """Finalize only after both canary services are loaded, stopped and dead."""

    _require_root_linux()
    _digest(approved_plan_sha256, "approved native observation plan")
    stage_value = _read_root_mapping(Path(stage_path))
    plan, observation = _validate_stage(
        stage_value,
        expected_plan_sha256=approved_plan_sha256,
    )
    if Path(stage_path) != _expected_stage_path(plan):
        raise ValueError("native observation stage path is not plan-addressed")
    if Path(receipt_path) != _expected_receipt_path(plan):
        raise ValueError("native observation receipt path is not plan-addressed")
    boot = _boot_id_sha256()
    host = _host_identity_sha256()
    if boot != plan.value["boot_id_sha256"] or host != plan.value["host_identity_sha256"]:
        raise RuntimeError("native finalization host or boot identity drifted")
    services: dict[str, Mapping[str, Any]] = {}
    for label in ("gateway", "writer"):
        unit = plan.value[f"{label}_unit"]
        state = _native_systemd_state(unit["name"])
        services[label] = {
            "unit_name": unit["name"],
            "load_state": state["LoadState"],
            "active_state": state["ActiveState"],
            "sub_state": state["SubState"],
            "unit_file_state": state["UnitFileState"],
            "main_pid": state["MainPID"],
        }
        _validate_stopped_service(services[label], unit=unit)
    owner_digest = validate_native_observation_approval(
        owner_approval_receipt,
        expected_plan_sha256=approved_plan_sha256,
        now_unix=int(time.time()),
    )
    if stage_value.get("owner_approval_receipt_sha256") != owner_digest:
        raise PermissionError("native observation approval changed after live collection")
    host_preparation_digest = _digest(
        host_preparation_receipt_sha256,
        "native final host preparation receipt",
    )
    external_iam_digest = _digest(
        external_iam_receipt_sha256,
        "native final external IAM receipt",
    )
    if (
        stage_value.get("host_preparation_receipt_sha256")
        != host_preparation_digest
        or stage_value.get("external_iam_receipt_sha256")
        != external_iam_digest
    ):
        raise PermissionError("native prerequisite evidence changed after live collection")
    final_state = {
        "boot_id_sha256": boot,
        "host_identity_sha256": host,
        "finalized_at_unix": int(time.time()),
        "finalized_at_boottime_ns": _boottime_ns(),
        "gateway_service": services["gateway"],
        "writer_service": services["writer"],
        "discord_absence": _collect_discord_absence(plan),
    }
    receipt = NativeObservationReceipt.finalize(
        plan=plan,
        expected_plan_sha256=approved_plan_sha256,
        owner_approval_receipt_sha256=owner_digest,
        host_preparation_receipt_sha256=host_preparation_digest,
        external_iam_receipt_sha256=external_iam_digest,
        observation=observation,
        final_state=final_state,
        current_boot_id_sha256=boot,
        current_boottime_ns=final_state["finalized_at_boottime_ns"],
    )
    write_native_observation_receipt(receipt_path, receipt)
    return receipt


def load_native_observation_plan(
    path: str | os.PathLike[str] = DEFAULT_NATIVE_OBSERVATION_PLAN_PATH,
) -> NativeObservationPlan:
    _require_root_linux()
    return NativeObservationPlan.from_mapping(_read_root_mapping(Path(path)))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Internal authority library; native mutations are available only "
            "through gateway.canonical_writer_activation"
        )
    )
    parser.parse_args(argv)
    parser.error(
        "no public mutation actions; use gateway.canonical_writer_activation"
    )


__all__ = [
    "DEFAULT_NATIVE_OBSERVATION_EVIDENCE_ROOT",
    "DEFAULT_NATIVE_OBSERVATION_PLAN_PATH",
    "DEFAULT_NATIVE_OBSERVATION_STAGE_ROOT",
    "DEFAULT_EXTERNAL_IAM_LIVE_PATH",
    "DEFAULT_EXTERNAL_IAM_RECEIPT_ROOT",
    "DEFAULT_OWNER_APPROVAL_ROOT",
    "EXTERNAL_IAM_TTL_SECONDS",
    "EXTERNAL_IAM_RECEIPT_SCHEMA",
    "ExternalIAMReceipt",
    "LEGACY_CLOUD_SQL_HELPER_PATH",
    "NATIVE_OBSERVATION_PLAN_SCHEMA",
    "NATIVE_OBSERVATION_RECEIPT_SCHEMA",
    "NATIVE_OBSERVATION_STAGE_SCHEMA",
    "NATIVE_OBSERVATION_TTL_SECONDS",
    "NativeObservationPlan",
    "NativeObservationReceipt",
    "OWNER_APPROVAL_RECEIPT_SCHEMA",
    "OwnerApprovalReceipt",
    "ProcessAuthorityEvidence",
    "build_external_iam_receipt",
    "collect_legacy_helper_absence",
    "collect_native_observation",
    "collect_writer_authority_surface",
    "current_host_identity_sha256",
    "load_native_observation_plan",
    "load_external_iam_receipt",
    "load_owner_approval_receipt",
    "owner_approval_receipt_path",
    "validate_native_observation_approval",
]


if __name__ == "__main__":
    raise SystemExit(main())

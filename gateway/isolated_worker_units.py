"""Exact production artifacts for the socket-activated isolated worker.

This module is a mechanical renderer only.  It does not install units, create
principals, select commands, or make task decisions.  Every host identity is
an explicit input, while all security-sensitive paths and service names are
fixed by this contract.

The worker configuration is canonical JSON with an embedded digest of the
unsigned policy.  ``gateway.isolated_worker_service`` independently verifies
that digest and the executable identities before accepting the systemd socket.
The writable lease tree is an exact, service-private tmpfs so aggregate blocks
and inodes are kernel-bounded on the production systemd 252 hosts.  Lease data
is deliberately ephemeral across service restarts.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from gateway.isolated_worker import PROTOCOL
from gateway.isolated_worker_service import CONFIG_SCHEMA


PRODUCTION_RELEASES = Path("/opt/adventico-ai-platform/hermes-agent-releases")
ISOLATED_WORKER_CONFIG = Path("/etc/muncho/isolated-worker.json")
ISOLATED_WORKER_SOCKET = Path("/run/muncho-isolated-worker/worker.sock")
ISOLATED_WORKER_LEASE_BASE = Path("/var/lib/muncho-isolated-worker")

ISOLATED_WORKER_SOCKET_UNIT = "muncho-isolated-worker.socket"
ISOLATED_WORKER_SERVICE_UNIT = "muncho-isolated-worker.service"
ISOLATED_WORKER_FD_NAME = "isolated-worker"
ISOLATED_WORKER_USER = "muncho-worker"
ISOLATED_WORKER_GROUP = "muncho-worker"
ISOLATED_WORKER_CLIENT_GROUP = "muncho-worker-clients"

BWRAP_PATH = Path("/usr/bin/bwrap")
SHELL_PATH = Path("/bin/bash")
ROOT_UID = 0
CONFIG_MODE = 0o440

SOCKET_BACKLOG = 128
SERVICE_TASKS_MAX = 512
SERVICE_MEMORY_MAX_BYTES = 2_684_354_560
SERVICE_MEMORY_SWAP_MAX_BYTES = 1_073_741_824
SERVICE_FILE_SIZE_LIMIT_BYTES = 4_294_967_296
SERVICE_GLOBAL_QUOTA_BYTES = SERVICE_FILE_SIZE_LIMIT_BYTES
SERVICE_GLOBAL_QUOTA_ENTRIES = 200_000
SERVICE_TMPFS_INODE_LIMIT = SERVICE_GLOBAL_QUOTA_ENTRIES + 1
LEASE_TMPFS_MINIMUM_SYSTEMD_VERSION = 252
LEASE_TMPFS_PREFLIGHT_CONTRACT = "private-tmpfs-capacity-inodes-identity-flags"

MAXIMUM_TIMEOUT_SECONDS = 300
MAXIMUM_OUTPUT_BYTES = 1_048_576
MAXIMUM_ACTIVE_LEASES = 128
MAXIMUM_ACTIVE_JOBS_PER_LEASE = 8
LEASE_TTL_SECONDS = 900
LEASE_QUOTA_BYTES = SERVICE_FILE_SIZE_LIMIT_BYTES
LEASE_QUOTA_ENTRIES = 100_000

GATEWAY_READY_PROBE_CONTRACT = "authenticated-af-unix-exec-round-trip"
UNIT_BUNDLE_SCHEMA = "muncho-isolated-worker-unit-bundle.v2"

_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTITY = re.compile(r"^[a-z_][a-z0-9_-]{0,30}$")
_SECRET_NAME = re.compile(
    r"(?:TOKEN|PASSWORD|PASSKEY|SECRET|PRIVATE_KEY|CREDENTIAL)",
    re.IGNORECASE,
)


class IsolatedWorkerUnitError(ValueError):
    """Stable fail-closed validation error for worker artifacts."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii", errors="strict")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise IsolatedWorkerUnitError("worker artifact is not canonical JSON") from exc


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _validate_revision(value: Any) -> str:
    if not isinstance(value, str) or _REVISION.fullmatch(value) is None:
        raise IsolatedWorkerUnitError("isolated worker release revision is invalid")
    return value


def _validate_digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise IsolatedWorkerUnitError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _validate_exact_identity(value: Any, expected: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or _IDENTITY.fullmatch(value) is None
        or value != expected
    ):
        raise IsolatedWorkerUnitError(f"{label} is not the exact production identity")
    return value


def _validate_positive_id(value: Any, label: str) -> int:
    if type(value) is not int or not 1 <= value < (1 << 31):
        raise IsolatedWorkerUnitError(f"{label} is invalid")
    return value


def _validate_root_uid(value: Any) -> int:
    if type(value) is not int or value != ROOT_UID:
        raise IsolatedWorkerUnitError("socket owner uid must be root")
    return value


def _unit_bytes(lines: list[str]) -> bytes:
    if any("\x00" in line or "\r" in line or "\n" in line for line in lines):
        raise IsolatedWorkerUnitError("systemd unit line is invalid")
    return ("\n".join(lines) + "\n").encode("ascii", errors="strict")


def _lease_tmpfs_directive(*, worker_uid: int, worker_gid: int) -> str:
    # ``exec`` is deliberate: compiling and executing a workspace artifact is
    # core worker behavior. ``noexec`` would not prevent interpreted code, but
    # would break that legitimate path. nosuid + nodev remain exact, alongside
    # NoNewPrivileges and the bwrap capability drop.
    return (
        f"TemporaryFileSystem={ISOLATED_WORKER_LEASE_BASE}:"
        f"size={SERVICE_GLOBAL_QUOTA_BYTES},"
        f"nr_inodes={SERVICE_TMPFS_INODE_LIMIT},"
        f"mode=0700,uid={worker_uid},gid={worker_gid},nodev,nosuid,exec"
    )


def _require_safe_service(
    unit: bytes,
    *,
    revision: str,
    worker_uid: int,
    worker_gid: int,
) -> None:
    try:
        text = unit.decode("ascii", errors="strict")
    except UnicodeError as exc:  # pragma: no cover - renderer emits ASCII literals
        raise IsolatedWorkerUnitError("isolated worker service is not ASCII") from exc
    required = (
        f"# ReleaseRevision={revision}\n",
        (
            "# GatewayReadyIntegration="
            f"required:{GATEWAY_READY_PROBE_CONTRACT}\n"
        ),
        (
            "# LeaseFilesystemPreflight="
            f"required:{LEASE_TMPFS_PREFLIGHT_CONTRACT}\n"
        ),
        f"User={ISOLATED_WORKER_USER}\n",
        f"Group={ISOLATED_WORKER_GROUP}\n",
        f"AssertPathIsDirectory={ISOLATED_WORKER_LEASE_BASE}\n",
        _lease_tmpfs_directive(worker_uid=worker_uid, worker_gid=worker_gid) + "\n",
        "KillMode=control-group\n",
        "PrivateNetwork=yes\n",
        "NoNewPrivileges=yes\n",
        "CapabilityBoundingSet=\n",
        "AmbientCapabilities=\n",
        "ProtectHome=yes\n",
        "ProtectSystem=strict\n",
        "RestrictAddressFamilies=AF_UNIX\n",
        "IPAddressDeny=any\n",
        f"TasksMax={SERVICE_TASKS_MAX}\n",
        f"MemoryMax={SERVICE_MEMORY_MAX_BYTES}\n",
        f"LimitFSIZE={SERVICE_FILE_SIZE_LIMIT_BYTES}\n",
    )
    forbidden = (
        "docker",
        "Docker",
        "EnvironmentFile=",
        "LoadCredential=",
        "PassEnvironment=",
        "SupplementaryGroups=",
        "NetworkNamespacePath=",
        "PrivateNetwork=no",
        "IPAddressAllow=",
        "ProtectHome=no",
        "ProtectSystem=false",
        "CapabilityBoundingSet=CAP_",
        "AmbientCapabilities=CAP_",
        "ExecStart=/usr/bin/env",
        "ExecStart=python",
        "ExecStart=python3",
        "KillMode=mixed",
        "KillMode=process",
        "KillMode=none",
        "StateDirectory=",
        "StateDirectoryQuota=",
        "StateDirectoryAccounting=",
        f"ReadWritePaths={ISOLATED_WORKER_LEASE_BASE}",
    )
    if any(item not in text for item in required) or any(
        item in text for item in forbidden
    ):
        raise IsolatedWorkerUnitError("isolated worker service safety contract drifted")
    for line in text.splitlines():
        if not line.startswith("Environment="):
            continue
        name = line.removeprefix("Environment=").split("=", 1)[0]
        if _SECRET_NAME.search(name):
            raise IsolatedWorkerUnitError(
                "isolated worker service embeds a secret environment"
            )


def _render_config(
    *,
    gateway_uid: int,
    gateway_primary_gid: int,
    socket_root_uid: int,
    socket_client_gid: int,
    worker_uid: int,
    worker_gid: int,
    bwrap_sha256: str,
    shell_sha256: str,
) -> tuple[bytes, str]:
    unsigned = {
        "schema": CONFIG_SCHEMA,
        "protocol": PROTOCOL,
        "listener_path": str(ISOLATED_WORKER_SOCKET),
        "expected_peer_uid": gateway_uid,
        "expected_peer_gid": gateway_primary_gid,
        "socket_uid": socket_root_uid,
        "socket_gid": socket_client_gid,
        "lease_base": str(ISOLATED_WORKER_LEASE_BASE),
        "lease_uid": worker_uid,
        "lease_gid": worker_gid,
        "network_isolated": True,
        "bwrap": {
            "path": str(BWRAP_PATH),
            "sha256": bwrap_sha256,
            "uid": ROOT_UID,
        },
        "shell": {
            "path": str(SHELL_PATH),
            "sha256": shell_sha256,
            "uid": ROOT_UID,
        },
        "limits": {
            "maximum_timeout_seconds": MAXIMUM_TIMEOUT_SECONDS,
            "maximum_output_bytes": MAXIMUM_OUTPUT_BYTES,
            "maximum_active_leases": MAXIMUM_ACTIVE_LEASES,
            "maximum_active_jobs_per_lease": MAXIMUM_ACTIVE_JOBS_PER_LEASE,
            "lease_ttl_seconds": LEASE_TTL_SECONDS,
            "lease_quota_bytes": LEASE_QUOTA_BYTES,
            "lease_quota_entries": LEASE_QUOTA_ENTRIES,
            "global_quota_bytes": SERVICE_GLOBAL_QUOTA_BYTES,
            "global_quota_entries": SERVICE_GLOBAL_QUOTA_ENTRIES,
        },
        # Optional operator-sealed shared trees must be a separate, reviewed
        # revision.  Production starts with no host tree projected into leases.
        "read_only_binds": [],
    }
    policy_sha256 = _sha256(_canonical_bytes(unsigned))
    return _canonical_bytes({**unsigned, "config_sha256": policy_sha256}), policy_sha256


def _render_socket_unit(*, socket_client_group: str) -> bytes:
    return _unit_bytes(
        [
            "# Exact production socket for the unprivileged isolated worker.",
            "[Unit]",
            "Description=Muncho isolated execution worker socket",
            "Before=hermes-cloud-gateway.service",
            "StartLimitIntervalSec=300s",
            "StartLimitBurst=5",
            "",
            "[Socket]",
            f"ListenStream={ISOLATED_WORKER_SOCKET}",
            f"Service={ISOLATED_WORKER_SERVICE_UNIT}",
            "SocketUser=root",
            f"SocketGroup={socket_client_group}",
            "SocketMode=0660",
            # systemd owns auto-created socket directories as root:root.  0711
            # permits the named client group to traverse to the 0660 socket
            # without making the directory enumerable.
            "DirectoryMode=0711",
            "Accept=no",
            f"FileDescriptorName={ISOLATED_WORKER_FD_NAME}",
            f"Backlog={SOCKET_BACKLOG}",
            "RemoveOnStop=yes",
            "",
            "[Install]",
            "WantedBy=sockets.target",
        ]
    )


def _render_service_unit(
    *,
    revision: str,
    release_root: Path,
    interpreter: Path,
    worker_user: str,
    worker_group: str,
    worker_uid: int,
    worker_gid: int,
) -> bytes:
    unit = _unit_bytes(
        [
            "# Exact production unprivileged isolated execution worker.",
            f"# ReleaseRevision={revision}",
            f"# PrincipalUID={worker_uid}",
            f"# PrincipalGID={worker_gid}",
            "# NetworkAvailable=false",
            "# CredentialMaterialAvailable=false",
            (
                "# GatewayReadyIntegration="
                f"required:{GATEWAY_READY_PROBE_CONTRACT}"
            ),
            (
                "# LeaseFilesystemPreflight="
                f"required:{LEASE_TMPFS_PREFLIGHT_CONTRACT}"
            ),
            "[Unit]",
            "Description=Muncho isolated execution worker",
            f"Requires={ISOLATED_WORKER_SOCKET_UNIT}",
            f"After={ISOLATED_WORKER_SOCKET_UNIT}",
            "Before=hermes-cloud-gateway.service",
            "StartLimitIntervalSec=300s",
            "StartLimitBurst=5",
            f"AssertPathExists={interpreter}",
            f"AssertPathExists={release_root / '.codex-source-commit'}",
            f"AssertPathExists={ISOLATED_WORKER_CONFIG}",
            f"AssertPathIsDirectory={ISOLATED_WORKER_LEASE_BASE}",
            f"AssertPathExists={BWRAP_PATH}",
            f"AssertPathExists={SHELL_PATH}",
            "",
            "[Service]",
            "Type=simple",
            f"User={worker_user}",
            f"Group={worker_group}",
            _lease_tmpfs_directive(worker_uid=worker_uid, worker_gid=worker_gid),
            f"WorkingDirectory={release_root}",
            (
                f"ExecStart={interpreter} -B -P -s -m "
                "gateway.isolated_worker_service "
                f"--config {ISOLATED_WORKER_CONFIG}"
            ),
            "Restart=on-failure",
            "RestartSec=5s",
            "TimeoutStartSec=30s",
            "TimeoutStopSec=30s",
            "KillMode=control-group",
            "OOMPolicy=stop",
            "LimitCORE=0",
            "LimitNOFILE=4096",
            f"LimitNPROC={SERVICE_TASKS_MAX}",
            f"LimitFSIZE={SERVICE_FILE_SIZE_LIMIT_BYTES}",
            f"TasksMax={SERVICE_TASKS_MAX}",
            f"MemoryMax={SERVICE_MEMORY_MAX_BYTES}",
            f"MemorySwapMax={SERVICE_MEMORY_SWAP_MAX_BYTES}",
            "Environment=HOME=/var/empty",
            "Environment=LANG=C.UTF-8",
            "Environment=LC_ALL=C.UTF-8",
            f"Environment=LOGNAME={worker_user}",
            "Environment=PATH=/run/muncho-no-path-fallback",
            "Environment=PYTHONNOUSERSITE=1",
            "Environment=SHELL=/usr/sbin/nologin",
            "Environment=TZ=UTC",
            f"Environment=USER={worker_user}",
            "UnsetEnvironment=PYTHONPATH PYTHONHOME BASH_ENV ENV CDPATH GLOBIGNORE",
            "NoNewPrivileges=yes",
            "CapabilityBoundingSet=",
            "AmbientCapabilities=",
            "LockPersonality=yes",
            "PrivateDevices=yes",
            "PrivateNetwork=yes",
            "PrivateTmp=yes",
            "ProtectClock=yes",
            "ProtectControlGroups=yes",
            "ProtectHome=yes",
            "ProtectHostname=yes",
            "ProtectKernelLogs=yes",
            "ProtectKernelModules=yes",
            "ProtectKernelTunables=yes",
            "ProtectProc=invisible",
            "ProtectSystem=strict",
            "ProcSubset=pid",
            "RemoveIPC=yes",
            "RestrictAddressFamilies=AF_UNIX",
            "RestrictRealtime=yes",
            "RestrictSUIDSGID=yes",
            "SystemCallArchitectures=native",
            "UMask=0077",
            "IPAddressDeny=any",
            f"BindReadOnlyPaths={release_root}",
            f"ReadOnlyPaths={ISOLATED_WORKER_CONFIG}",
            "InaccessiblePaths=-/home -/root -/run/user -/run/credentials",
            (
                "InaccessiblePaths=-/etc/muncho/credentials "
                "-/etc/muncho/keys -/etc/muncho-production-cutover"
            ),
            "InaccessiblePaths=-/opt/adventico-ai-platform/hermes-home",
            "StandardInput=null",
            "StandardOutput=journal",
            "StandardError=journal",
        ]
    )
    _require_safe_service(
        unit,
        revision=revision,
        worker_uid=worker_uid,
        worker_gid=worker_gid,
    )
    return unit


@dataclass(frozen=True)
class IsolatedWorkerUnitBundle:
    """Immutable exact artifacts and content identities for one release."""

    revision: str
    release_root: Path
    interpreter: Path
    config_path: Path
    config_owner_uid: int
    config_owner_gid: int
    config_mode: int
    socket_path: Path
    lease_base: Path
    lease_owner_uid: int
    lease_owner_gid: int
    lease_mode: int
    config: bytes
    config_sha256: str
    config_policy_sha256: str
    socket_unit: bytes
    socket_unit_sha256: str
    service_unit: bytes
    service_unit_sha256: str
    bwrap_sha256: str
    shell_sha256: str
    bundle_sha256: str

    def artifacts(self) -> Mapping[str, bytes]:
        return MappingProxyType(
            {
                str(self.config_path): self.config,
                ISOLATED_WORKER_SOCKET_UNIT: self.socket_unit,
                ISOLATED_WORKER_SERVICE_UNIT: self.service_unit,
            }
        )

    def artifact_sha256(self) -> Mapping[str, str]:
        return MappingProxyType(
            {
                str(self.config_path): self.config_sha256,
                ISOLATED_WORKER_SOCKET_UNIT: self.socket_unit_sha256,
                ISOLATED_WORKER_SERVICE_UNIT: self.service_unit_sha256,
            }
        )

    def manifest(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "schema": UNIT_BUNDLE_SCHEMA,
                "release_revision": self.revision,
                "release_root": str(self.release_root),
                "interpreter": str(self.interpreter),
                "config_path": str(self.config_path),
                "config_install": {
                    "owner_uid": self.config_owner_uid,
                    "owner_gid": self.config_owner_gid,
                    "mode": f"{self.config_mode:04o}",
                },
                "socket_path": str(self.socket_path),
                "lease_base": str(self.lease_base),
                "lease_mountpoint_install": {
                    "owner_uid": ROOT_UID,
                    "owner_gid": ROOT_UID,
                    "mode": "0700",
                },
                "lease_runtime_filesystem": {
                    "type": "tmpfs",
                    "ephemeral_across_service_restart": True,
                    "bytes": SERVICE_GLOBAL_QUOTA_BYTES,
                    "inodes": SERVICE_TMPFS_INODE_LIMIT,
                    "runtime_entry_limit": SERVICE_GLOBAL_QUOTA_ENTRIES,
                    "owner_uid": self.lease_owner_uid,
                    "owner_gid": self.lease_owner_gid,
                    "mode": f"{self.lease_mode:04o}",
                    "mount_flags": ["nodev", "nosuid", "exec"],
                    "kernel_enforced": True,
                    "host_preflight_required": True,
                    "host_preflight_contract": LEASE_TMPFS_PREFLIGHT_CONTRACT,
                    "minimum_systemd_version": (
                        LEASE_TMPFS_MINIMUM_SYSTEMD_VERSION
                    ),
                },
                "artifacts": dict(self.artifact_sha256()),
                "config_policy_sha256": self.config_policy_sha256,
                "bwrap_sha256": self.bwrap_sha256,
                "shell_sha256": self.shell_sha256,
                "secret_material_recorded": False,
                "secret_digest_recorded": False,
                "gateway_ready_integration": {
                    "required": True,
                    "unit_ordering_sufficient": False,
                    "probe_contract": GATEWAY_READY_PROBE_CONTRACT,
                    "required_host_preflight_contract": (
                        LEASE_TMPFS_PREFLIGHT_CONTRACT
                    ),
                },
                "bundle_sha256": self.bundle_sha256,
            }
        )


def render_isolated_worker_units(
    *,
    revision: str,
    gateway_uid: int,
    gateway_primary_gid: int,
    socket_root_uid: int,
    socket_client_group: str,
    socket_client_gid: int,
    worker_user: str,
    worker_group: str,
    worker_uid: int,
    worker_gid: int,
    bwrap_sha256: str,
    shell_sha256: str,
) -> IsolatedWorkerUnitBundle:
    """Render the exact config, socket and service without mutating a host."""

    revision = _validate_revision(revision)
    socket_client_group = _validate_exact_identity(
        socket_client_group,
        ISOLATED_WORKER_CLIENT_GROUP,
        "isolated worker socket client group",
    )
    worker_user = _validate_exact_identity(
        worker_user, ISOLATED_WORKER_USER, "isolated worker user"
    )
    worker_group = _validate_exact_identity(
        worker_group, ISOLATED_WORKER_GROUP, "isolated worker group"
    )
    gateway_uid = _validate_positive_id(gateway_uid, "gateway uid")
    gateway_primary_gid = _validate_positive_id(
        gateway_primary_gid, "gateway primary gid"
    )
    socket_root_uid = _validate_root_uid(socket_root_uid)
    socket_client_gid = _validate_positive_id(
        socket_client_gid, "socket client group gid"
    )
    worker_uid = _validate_positive_id(worker_uid, "isolated worker uid")
    worker_gid = _validate_positive_id(worker_gid, "isolated worker gid")
    if gateway_uid == worker_uid:
        raise IsolatedWorkerUnitError(
            "gateway and isolated worker UIDs must be distinct"
        )
    if len({gateway_primary_gid, socket_client_gid, worker_gid}) != 3:
        raise IsolatedWorkerUnitError(
            "gateway, socket client, and worker GIDs must be distinct"
        )
    bwrap_sha256 = _validate_digest(bwrap_sha256, "bwrap digest")
    shell_sha256 = _validate_digest(shell_sha256, "shell digest")
    if bwrap_sha256 == shell_sha256:
        raise IsolatedWorkerUnitError(
            "bwrap and shell executable digests must be distinct"
        )

    release_root = PRODUCTION_RELEASES / f"hermes-agent-{revision[:12]}"
    if (
        release_root.parent != PRODUCTION_RELEASES
        or release_root.name != f"hermes-agent-{revision[:12]}"
    ):
        raise IsolatedWorkerUnitError("isolated worker release path is not exact")
    interpreter = release_root / ".venv/bin/python"
    config, config_policy_sha256 = _render_config(
        gateway_uid=gateway_uid,
        gateway_primary_gid=gateway_primary_gid,
        socket_root_uid=socket_root_uid,
        socket_client_gid=socket_client_gid,
        worker_uid=worker_uid,
        worker_gid=worker_gid,
        bwrap_sha256=bwrap_sha256,
        shell_sha256=shell_sha256,
    )
    socket_unit = _render_socket_unit(socket_client_group=socket_client_group)
    service_unit = _render_service_unit(
        revision=revision,
        release_root=release_root,
        interpreter=interpreter,
        worker_user=worker_user,
        worker_group=worker_group,
        worker_uid=worker_uid,
        worker_gid=worker_gid,
    )
    artifact_hashes = {
        str(ISOLATED_WORKER_CONFIG): _sha256(config),
        ISOLATED_WORKER_SOCKET_UNIT: _sha256(socket_unit),
        ISOLATED_WORKER_SERVICE_UNIT: _sha256(service_unit),
    }
    unsigned_manifest = {
        "schema": UNIT_BUNDLE_SCHEMA,
        "release_revision": revision,
        "release_root": str(release_root),
        "interpreter": str(interpreter),
        "config_path": str(ISOLATED_WORKER_CONFIG),
        "config_install": {
            "owner_uid": ROOT_UID,
            "owner_gid": worker_gid,
            "mode": f"{CONFIG_MODE:04o}",
        },
        "socket_path": str(ISOLATED_WORKER_SOCKET),
        "lease_base": str(ISOLATED_WORKER_LEASE_BASE),
        "lease_mountpoint_install": {
            "owner_uid": ROOT_UID,
            "owner_gid": ROOT_UID,
            "mode": "0700",
        },
        "lease_runtime_filesystem": {
            "type": "tmpfs",
            "ephemeral_across_service_restart": True,
            "bytes": SERVICE_GLOBAL_QUOTA_BYTES,
            "inodes": SERVICE_TMPFS_INODE_LIMIT,
            "runtime_entry_limit": SERVICE_GLOBAL_QUOTA_ENTRIES,
            "owner_uid": worker_uid,
            "owner_gid": worker_gid,
            "mode": "0700",
            "mount_flags": ["nodev", "nosuid", "exec"],
            "kernel_enforced": True,
            "host_preflight_required": True,
            "host_preflight_contract": LEASE_TMPFS_PREFLIGHT_CONTRACT,
            "minimum_systemd_version": (
                LEASE_TMPFS_MINIMUM_SYSTEMD_VERSION
            ),
        },
        "artifacts": artifact_hashes,
        "config_policy_sha256": config_policy_sha256,
        "bwrap_sha256": bwrap_sha256,
        "shell_sha256": shell_sha256,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
        "gateway_ready_integration": {
            "required": True,
            "unit_ordering_sufficient": False,
            "probe_contract": GATEWAY_READY_PROBE_CONTRACT,
            "required_host_preflight_contract": LEASE_TMPFS_PREFLIGHT_CONTRACT,
        },
    }
    return IsolatedWorkerUnitBundle(
        revision=revision,
        release_root=release_root,
        interpreter=interpreter,
        config_path=ISOLATED_WORKER_CONFIG,
        config_owner_uid=ROOT_UID,
        config_owner_gid=worker_gid,
        config_mode=CONFIG_MODE,
        socket_path=ISOLATED_WORKER_SOCKET,
        lease_base=ISOLATED_WORKER_LEASE_BASE,
        lease_owner_uid=worker_uid,
        lease_owner_gid=worker_gid,
        lease_mode=0o700,
        config=config,
        config_sha256=artifact_hashes[str(ISOLATED_WORKER_CONFIG)],
        config_policy_sha256=config_policy_sha256,
        socket_unit=socket_unit,
        socket_unit_sha256=artifact_hashes[ISOLATED_WORKER_SOCKET_UNIT],
        service_unit=service_unit,
        service_unit_sha256=artifact_hashes[ISOLATED_WORKER_SERVICE_UNIT],
        bwrap_sha256=bwrap_sha256,
        shell_sha256=shell_sha256,
        bundle_sha256=_sha256(_canonical_bytes(unsigned_manifest)),
    )


__all__ = [
    "BWRAP_PATH",
    "CONFIG_MODE",
    "GATEWAY_READY_PROBE_CONTRACT",
    "ISOLATED_WORKER_CLIENT_GROUP",
    "ISOLATED_WORKER_CONFIG",
    "ISOLATED_WORKER_FD_NAME",
    "ISOLATED_WORKER_GROUP",
    "ISOLATED_WORKER_LEASE_BASE",
    "ISOLATED_WORKER_SERVICE_UNIT",
    "ISOLATED_WORKER_SOCKET",
    "ISOLATED_WORKER_SOCKET_UNIT",
    "ISOLATED_WORKER_USER",
    "IsolatedWorkerUnitBundle",
    "IsolatedWorkerUnitError",
    "LEASE_TMPFS_MINIMUM_SYSTEMD_VERSION",
    "LEASE_TMPFS_PREFLIGHT_CONTRACT",
    "SHELL_PATH",
    "UNIT_BUNDLE_SCHEMA",
    "render_isolated_worker_units",
]

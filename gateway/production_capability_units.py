"""Mechanical production systemd artifacts for capability prerequisites.

The functions in this module only render exact, release-addressed service and
configuration bytes.  They do not install, enable, start, stop, or otherwise
mutate a host, and they contain no task classification or routing policy.

Secret values are deliberately not accepted by any public function.  Secret
source files are copied into a service-private systemd credential directory by
``LoadCredential=``; the service configuration refers only to that runtime
copy.  The rendered artifacts therefore contain credential paths and names,
never credential values or credential digests.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from gateway.browser_controller import CONFIG_SCHEMA as BROWSER_CONTROLLER_CONFIG_SCHEMA
from gateway.mac_ops_edge_service import (
    CONFIG_SCHEMA as MAC_OPS_CONFIG_SCHEMA,
    DEFAULT_PROJECT_ID as MAC_OPS_PROJECT_ID,
)
from gateway.production_capability_prerequisites import (
    BROWSER_CONFIG_PATH,
    BROWSER_SOCKET_PATH,
    BROWSER_STATE_PATH,
    BROWSER_UNIT,
    CODEX_AUTH_PATH,
    GATEWAY_UNIT,
    MAC_OPS_CONFIG_PATH,
    MAC_OPS_CREDENTIAL_PATH,
    MAC_OPS_JOURNAL_PATH,
    MAC_OPS_SOCKET_PATH,
    MAC_OPS_UNIT,
    PHASE_B_RECEIPT_PATH,
    PHASE_B_UNIT,
    PRODUCTION_HOME,
    PUBLIC_CONNECTOR_CREDENTIAL_PATH,
    ROUTEBACK_EDGE_CONFIG_PATH,
    ROUTEBACK_EDGE_CREDENTIAL_PATH,
    ROUTEBACK_EDGE_SOCKET_PATH,
    ROUTEBACK_EDGE_UNIT,
    RUNTIME_DEPENDENCY_MANIFEST_RELATIVE,
    production_release_root,
)
from gateway.production_runtime_dependencies import (
    AGENT_BROWSER_CONFIG,
    AGENT_BROWSER_NATIVE,
    AGENT_BROWSER_WRAPPER,
    CHROME_EXECUTABLE,
    NODE_EXECUTABLE,
)


PRODUCTION_RELEASES = Path("/opt/adventico-ai-platform/hermes-agent-releases")
PRODUCTION_EVIDENCE_ROOT = Path("/var/lib/muncho-production-legacy-cutover")
STAGED_CUTOVER_PLAN_PATH = PRODUCTION_EVIDENCE_ROOT / "staged" / "cutover-plan.json"
PRODUCTION_CUTOVER_CREDENTIAL_ROOT = Path("/etc/muncho-production-cutover")
PRODUCTION_CUTOVER_CA_PATH = (
    PRODUCTION_CUTOVER_CREDENTIAL_ROOT / "cloudsql-server-ca.pem"
)
PRODUCTION_CUTOVER_PGPASS_PATH = PRODUCTION_CUTOVER_CREDENTIAL_ROOT / "pgpass"

ROUTEBACK_EDGE_STATE_PATH = Path("/var/lib/muncho-discord-egress")
ROUTEBACK_EDGE_JOURNAL_PATH = ROUTEBACK_EDGE_STATE_PATH / "discord-edge-journal.sqlite3"
ROUTEBACK_WRITER_PUBLIC_KEY_PATH = Path("/etc/muncho/keys/writer-capability-public.pem")
ROUTEBACK_EDGE_PRIVATE_KEY_PATH = Path(
    "/etc/muncho/keys/discord-edge-receipt-private.pem"
)
MAC_OPS_STATE_PATH = MAC_OPS_JOURNAL_PATH.parent
BROWSER_RUNTIME_PATH = BROWSER_SOCKET_PATH.parent
BROWSER_RESOLV_CONF_PATH = Path("/run/systemd/resolve/stub-resolv.conf")
BROWSER_CONFIG_UID = 0
BROWSER_CONFIG_MODE = 0o440

BROWSER_COMMAND_TIMEOUT_SECONDS = 120
BROWSER_IDLE_TIMEOUT_SECONDS = 900
BROWSER_MAX_CONNECTIONS = 8
BROWSER_MAX_SESSIONS = 4
BROWSER_SESSION_QUOTA_BYTES = 256 * 1024 * 1024
BROWSER_SESSION_QUOTA_ENTRIES = 4096

# ``IPAddressAllow`` is evaluated before ``IPAddressDeny`` by systemd's cgroup
# BPF policy.  The local resolved stub is therefore the sole loopback
# exception, while unmatched public destinations retain the default allow.
BROWSER_DNS_ALLOW = "127.0.0.53/32"
BROWSER_NETWORK_DENY_RANGES = (
    # IPv4: unspecified/current host, private, shared, loopback, link-local,
    # metadata, documentation, benchmarking, multicast, and reserved space.
    "0.0.0.0/8",
    "10.0.0.0/8",
    "100.64.0.0/10",
    "127.0.0.0/8",
    "169.254.0.0/16",
    "172.16.0.0/12",
    "192.0.0.0/24",
    "192.0.2.0/24",
    "192.88.99.0/24",
    "192.168.0.0/16",
    "198.18.0.0/15",
    "198.51.100.0/24",
    "203.0.113.0/24",
    "224.0.0.0/4",
    "240.0.0.0/4",
    # IPv6: unspecified, loopback, mapped/translation, discard, protocol and
    # documentation allocations, 6to4, ULA, link-local, and multicast.
    "::/128",
    "::1/128",
    "::ffff:0:0/96",
    "64:ff9b::/96",
    "64:ff9b:1::/48",
    "100::/64",
    "2001::/23",
    "2001:db8::/32",
    "2002::/16",
    "3fff::/20",
    "fc00::/7",
    "fe80::/10",
    "fec0::/10",
    "ff00::/8",
)

PHASE_B_PGPASS_CREDENTIAL_NAME = "postgresql-pgpass"
ROUTEBACK_TOKEN_CREDENTIAL_NAME = "discord-bot-token"
ROUTEBACK_PRIVATE_KEY_CREDENTIAL_NAME = "discord-edge-receipt-private-key"
MAC_OPS_CREDENTIAL_NAME = "mac-ops-gitlab-env"

_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTITY = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]{0,63}$")
_SECRET_ENVIRONMENT_NAME = re.compile(
    r"(?:TOKEN|PASSWORD|PASSKEY|SECRET|PRIVATE_KEY|CREDENTIAL)", re.IGNORECASE
)


class ProductionCapabilityUnitError(ValueError):
    """Stable validation failure for production capability artifacts."""


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
        raise ProductionCapabilityUnitError(
            "production capability config is not canonical JSON"
        ) from exc


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _revision(value: Any) -> str:
    if not isinstance(value, str) or _REVISION.fullmatch(value) is None:
        raise ProductionCapabilityUnitError(
            "production capability release revision is invalid"
        )
    return value


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ProductionCapabilityUnitError(f"{label} is not lowercase SHA-256")
    return value


def _identity(value: Any, label: str) -> str:
    if not isinstance(value, str) or _IDENTITY.fullmatch(value) is None:
        raise ProductionCapabilityUnitError(f"{label} is invalid")
    return value


def _positive_id(value: Any, label: str) -> int:
    if type(value) is not int or not 1 <= value < (1 << 31):
        raise ProductionCapabilityUnitError(f"{label} is invalid")
    return value


def _bounded_integer(value: Any, label: str, *, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ProductionCapabilityUnitError(f"{label} is invalid")
    return value


def _bounded_number(value: Any, label: str, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProductionCapabilityUnitError(f"{label} is invalid")
    result = float(value)
    if not minimum <= result <= maximum:
        raise ProductionCapabilityUnitError(f"{label} is invalid")
    return result


def _database_ip_allow(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ProductionCapabilityUnitError("production SQL host is invalid")
    raw = value[:-3] if value.endswith("/32") else value
    try:
        address = ipaddress.ip_address(raw)
    except ValueError as exc:
        raise ProductionCapabilityUnitError(
            "production SQL host must be one exact IPv4 address"
        ) from exc
    if (
        not isinstance(address, ipaddress.IPv4Address)
        or address.is_unspecified
        or address.is_loopback
        or address.is_multicast
        or address.is_link_local
    ):
        raise ProductionCapabilityUnitError(
            "production SQL host must be one exact IPv4 address"
        )
    return f"{address}/32"


def _release_artifact_path(
    value: Any,
    *,
    expected: Path,
    label: str,
) -> Path:
    if (
        not isinstance(value, str)
        or not value
        or "\x00" in value
        or value != str(expected)
    ):
        raise ProductionCapabilityUnitError(
            f"{label} is not the exact release-local path"
        )
    return expected


def _credential_runtime_path(unit: str, name: str) -> Path:
    return Path("/run/credentials") / unit / name


def _fixed_environment(*, user: str, home: Path, release: Path) -> list[str]:
    return [
        f"Environment=HOME={home}",
        "Environment=LANG=C.UTF-8",
        "Environment=LC_ALL=C.UTF-8",
        f"Environment=LOGNAME={user}",
        "Environment=PATH=/usr/bin:/bin",
        "Environment=PYTHONDONTWRITEBYTECODE=1",
        "Environment=PYTHONNOUSERSITE=1",
        f"Environment=PYTHONPATH={release}",
        "Environment=SHELL=/usr/sbin/nologin",
        "Environment=TZ=UTC",
        f"Environment=USER={user}",
    ]


def _common_hardening(
    *,
    restrict_namespaces: bool = True,
    capability_bounding_set: str = "",
) -> list[str]:
    result = [
        "NoNewPrivileges=yes",
        f"CapabilityBoundingSet={capability_bounding_set}",
        "AmbientCapabilities=",
        "LockPersonality=yes",
        "PrivateDevices=yes",
        "PrivateTmp=yes",
        "ProtectClock=yes",
        "ProtectControlGroups=yes",
        "ProtectHome=yes",
        "ProtectHostname=yes",
        "ProtectKernelLogs=yes",
        "ProtectKernelModules=yes",
        "ProtectKernelTunables=yes",
        "ProtectSystem=strict",
        "RemoveIPC=yes",
        "RestrictRealtime=yes",
        "RestrictSUIDSGID=yes",
        "SystemCallArchitectures=native",
        "UMask=0077",
    ]
    if restrict_namespaces:
        result.append("RestrictNamespaces=yes")
    return result


def _unit_bytes(lines: list[str]) -> bytes:
    return ("\n".join(lines) + "\n").encode("utf-8", errors="strict")


def _require_safe_unit(
    value: bytes,
    *,
    revision: str,
    persistent: bool,
) -> None:
    try:
        text = value.decode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise ProductionCapabilityUnitError(
            "production capability unit is not UTF-8"
        ) from exc
    expected_release = str(production_release_root(revision))
    if (
        text.count(expected_release) < 1
        or re.search(r"(?im)^Description=.*canary", text) is not None
        or "/opt/muncho-canary" in text
        or "/etc/muncho/capability-canary" in text
        or "RuntimeMaxSec=" in text
        or "EnvironmentFile=" in text
        or "PassEnvironment=" in text
        or "--no-sandbox" in text
        or "Restart=always" in text
        or (persistent and text.count("Restart=on-failure\n") != 1)
        or (persistent and text.count("RestartSec=5s\n") != 1)
        or (persistent and text.count("StartLimitIntervalSec=300s\n") != 1)
        or (persistent and text.count("StartLimitBurst=5\n") != 1)
        or (persistent and "Restart=no" in text)
    ):
        raise ProductionCapabilityUnitError(
            "production capability unit safety contract is invalid"
        )
    for line in text.splitlines():
        if not line.startswith("Environment="):
            continue
        name = line.removeprefix("Environment=").split("=", 1)[0]
        if _SECRET_ENVIRONMENT_NAME.search(name):
            raise ProductionCapabilityUnitError(
                "production capability unit embeds a secret environment"
            )


@dataclass(frozen=True)
class ProductionCapabilityUnitBundle:
    """Exact production units, browser config, and their content identities."""

    revision: str
    release_root: Path
    interpreter: Path
    phase_b_unit: bytes
    phase_b_sha256: str
    routeback_unit: bytes
    routeback_sha256: str
    mac_ops_unit: bytes
    mac_ops_sha256: str
    browser_unit: bytes
    browser_sha256: str
    browser_config_path: Path
    browser_config: bytes
    browser_config_sha256: str
    browser_config_uid: int
    browser_config_gid: int
    browser_config_mode: int
    browser_service_uid: int
    browser_service_gid: int
    browser_allowed_client_uid: int
    bundle_sha256: str

    def units(self) -> Mapping[str, bytes]:
        return {
            PHASE_B_UNIT: self.phase_b_unit,
            ROUTEBACK_EDGE_UNIT: self.routeback_unit,
            MAC_OPS_UNIT: self.mac_ops_unit,
            BROWSER_UNIT: self.browser_unit,
        }

    def unit_sha256(self) -> Mapping[str, str]:
        return {
            PHASE_B_UNIT: self.phase_b_sha256,
            ROUTEBACK_EDGE_UNIT: self.routeback_sha256,
            MAC_OPS_UNIT: self.mac_ops_sha256,
            BROWSER_UNIT: self.browser_sha256,
        }

    def configs(self) -> Mapping[str, bytes]:
        return {str(self.browser_config_path): self.browser_config}

    def manifest(self) -> Mapping[str, Any]:
        return {
            "schema": "muncho-production-capability-unit-bundle.v1",
            "release_revision": self.revision,
            "release_root": str(self.release_root),
            "interpreter": str(self.interpreter),
            "units": dict(self.unit_sha256()),
            "configs": {
                str(self.browser_config_path): {
                    "gid": self.browser_config_gid,
                    "mode": f"{self.browser_config_mode:04o}",
                    "schema": BROWSER_CONTROLLER_CONFIG_SCHEMA,
                    "sha256": self.browser_config_sha256,
                    "uid": self.browser_config_uid,
                }
            },
            "browser_controller": {
                "allowed_client_uid": self.browser_allowed_client_uid,
                "service_gid": self.browser_service_gid,
                "service_uid": self.browser_service_uid,
                "socket_path": str(BROWSER_SOCKET_PATH),
                "state_path": str(BROWSER_STATE_PATH),
            },
            "secret_material_recorded": False,
            "secret_digest_recorded": False,
            "bundle_sha256": self.bundle_sha256,
        }


def _render_phase_b_unit(
    *, revision: str, release: Path, interpreter: Path, database_ip_allow: str
) -> bytes:
    pgpass_runtime = _credential_runtime_path(
        PHASE_B_UNIT, PHASE_B_PGPASS_CREDENTIAL_NAME
    )
    lines = [
        "# Exact production Canonical Writer Phase-B read-only preflight.",
        f"# ReleaseRevision={revision}",
        "[Unit]",
        "Description=Muncho production Canonical Writer Phase-B readiness",
        "After=network-online.target",
        "Wants=network-online.target",
        "Before=muncho-canonical-writer.service",
        f"AssertPathExists={interpreter}",
        f"AssertPathExists={release / '.codex-source-commit'}",
        f"AssertPathExists={STAGED_CUTOVER_PLAN_PATH}",
        f"AssertPathExists={PRODUCTION_CUTOVER_CA_PATH}",
        f"AssertPathExists={PRODUCTION_CUTOVER_PGPASS_PATH}",
        "",
        "[Service]",
        "Type=oneshot",
        "User=root",
        "Group=root",
        f"LoadCredential={PHASE_B_PGPASS_CREDENTIAL_NAME}:{PRODUCTION_CUTOVER_PGPASS_PATH}",
        f"WorkingDirectory={release}",
        (
            f"ExecStart={interpreter} -B -P -s -m "
            "gateway.canonical_writer_production_cutover phase-b-preflight"
        ),
        "RemainAfterExit=yes",
        "TimeoutStartSec=900s",
        "TimeoutStopSec=15s",
        "KillMode=mixed",
        "LimitCORE=0",
        *_fixed_environment(user="root", home=Path("/root"), release=release),
        "UnsetEnvironment=PGPASSWORD PGSERVICEFILE",
        *_common_hardening(capability_bounding_set="CAP_DAC_READ_SEARCH"),
        "RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6",
        "IPAddressDeny=any",
        f"IPAddressAllow={database_ip_allow}",
        f"BindReadOnlyPaths={release}",
        f"ReadOnlyPaths={PRODUCTION_EVIDENCE_ROOT}",
        f"ReadOnlyPaths={PRODUCTION_CUTOVER_CREDENTIAL_ROOT}",
        (f"BindReadOnlyPaths={pgpass_runtime}:{PRODUCTION_CUTOVER_PGPASS_PATH}"),
        f"ReadWritePaths={PRODUCTION_EVIDENCE_ROOT / 'plans'}",
        f"ReadWritePaths={PHASE_B_RECEIPT_PATH.parent}",
        "StandardOutput=journal",
        "StandardError=journal",
        "",
        "[Install]",
        "WantedBy=multi-user.target",
    ]
    result = _unit_bytes(lines)
    _require_safe_unit(result, revision=revision, persistent=False)
    return result


def _render_routeback_unit(
    *,
    revision: str,
    release: Path,
    interpreter: Path,
    routeback_user: str,
    routeback_group: str,
    routeback_uid: int,
    routeback_gid: int,
) -> bytes:
    lines = [
        "# Exact production Discord canonical route-back edge.",
        f"# ReleaseRevision={revision}",
        f"# PrincipalUID={routeback_uid}",
        f"# PrincipalGID={routeback_gid}",
        "# DiscordDirectMessageAllowed=false",
        "[Unit]",
        "Description=Muncho production privileged Discord route-back edge",
        "After=network-online.target",
        "Wants=network-online.target",
        f"Before={GATEWAY_UNIT}",
        "StartLimitIntervalSec=300s",
        "StartLimitBurst=5",
        f"AssertPathExists={interpreter}",
        f"AssertPathExists={release / '.codex-source-commit'}",
        f"AssertPathExists={ROUTEBACK_EDGE_CONFIG_PATH}",
        f"AssertPathExists={ROUTEBACK_EDGE_CREDENTIAL_PATH}",
        f"AssertPathExists={ROUTEBACK_EDGE_PRIVATE_KEY_PATH}",
        f"AssertPathExists={ROUTEBACK_WRITER_PUBLIC_KEY_PATH}",
        "",
        "[Service]",
        "Type=notify",
        "NotifyAccess=main",
        f"User={routeback_user}",
        f"Group={routeback_group}",
        (
            f"LoadCredential={ROUTEBACK_TOKEN_CREDENTIAL_NAME}:"
            f"{ROUTEBACK_EDGE_CREDENTIAL_PATH}"
        ),
        (
            f"LoadCredential={ROUTEBACK_PRIVATE_KEY_CREDENTIAL_NAME}:"
            f"{ROUTEBACK_EDGE_PRIVATE_KEY_PATH}"
        ),
        "RuntimeDirectory=muncho-discord-egress",
        "RuntimeDirectoryMode=0750",
        "RuntimeDirectoryPreserve=no",
        "StateDirectory=muncho-discord-egress",
        "StateDirectoryMode=0700",
        f"WorkingDirectory={release}",
        (
            f"ExecStart={interpreter} -B -P -s -m "
            "gateway.production_discord_edge_bootstrap "
            f"--config {ROUTEBACK_EDGE_CONFIG_PATH}"
        ),
        "Restart=on-failure",
        "RestartSec=5s",
        "TimeoutStartSec=60s",
        "TimeoutStopSec=30s",
        "KillMode=mixed",
        "LimitCORE=0",
        *_fixed_environment(
            user=routeback_user, home=ROUTEBACK_EDGE_STATE_PATH, release=release
        ),
        (
            "UnsetEnvironment=DISCORD_BOT_TOKEN DISCORD_TOKEN "
            "MUNCHO_DISCORD_EDGE_PRIVATE_KEY"
        ),
        *_common_hardening(),
        "RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6",
        "IPAddressDeny=169.254.169.254/32",
        f"BindReadOnlyPaths={release}",
        f"ReadOnlyPaths={ROUTEBACK_EDGE_CONFIG_PATH}",
        f"ReadOnlyPaths={ROUTEBACK_WRITER_PUBLIC_KEY_PATH}",
        f"InaccessiblePaths={ROUTEBACK_EDGE_CREDENTIAL_PATH}",
        f"InaccessiblePaths={ROUTEBACK_EDGE_PRIVATE_KEY_PATH}",
        f"ReadWritePaths={ROUTEBACK_EDGE_SOCKET_PATH.parent}",
        f"ReadWritePaths={ROUTEBACK_EDGE_STATE_PATH}",
        "",
        "[Install]",
        "WantedBy=multi-user.target",
    ]
    result = _unit_bytes(lines)
    _require_safe_unit(result, revision=revision, persistent=True)
    return result


def _render_mac_ops_unit(
    *,
    revision: str,
    release: Path,
    interpreter: Path,
    mac_ops_user: str,
    mac_ops_group: str,
    mac_ops_uid: int,
    mac_ops_gid: int,
    socket_client_group: str,
) -> bytes:
    lines = [
        "# Exact production privileged Mac operations edge.",
        f"# ReleaseRevision={revision}",
        f"# PrincipalUID={mac_ops_uid}",
        f"# PrincipalGID={mac_ops_gid}",
        "[Unit]",
        "Description=Muncho production privileged Mac operations edge",
        "After=network-online.target",
        "Wants=network-online.target",
        f"Before={GATEWAY_UNIT}",
        "StartLimitIntervalSec=300s",
        "StartLimitBurst=5",
        f"AssertPathExists={interpreter}",
        f"AssertPathExists={release / '.codex-source-commit'}",
        f"AssertPathExists={MAC_OPS_CONFIG_PATH}",
        f"AssertPathExists={MAC_OPS_CREDENTIAL_PATH}",
        "",
        "[Service]",
        "Type=simple",
        f"User={mac_ops_user}",
        f"Group={mac_ops_group}",
        (f"LoadCredential={MAC_OPS_CREDENTIAL_NAME}:{MAC_OPS_CREDENTIAL_PATH}"),
        "RuntimeDirectory=muncho-mac-ops",
        "RuntimeDirectoryMode=0750",
        "RuntimeDirectoryPreserve=no",
        "StateDirectory=muncho-mac-ops",
        "StateDirectoryMode=0700",
        f"WorkingDirectory={release}",
        (
            f"ExecStart={interpreter} -B -P -s -m "
            f"gateway.mac_ops_edge_service --config {MAC_OPS_CONFIG_PATH}"
        ),
        "Restart=on-failure",
        "RestartSec=5s",
        "TimeoutStartSec=30s",
        "TimeoutStopSec=30s",
        "KillMode=mixed",
        "LimitCORE=0",
        *_fixed_environment(
            user=mac_ops_user, home=MAC_OPS_STATE_PATH, release=release
        ),
        "UnsetEnvironment=GITLAB_TOKEN PRIVATE_TOKEN GITLAB_PASSWORD",
        *_common_hardening(),
        "RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6",
        "IPAddressDeny=169.254.169.254/32",
        f"BindReadOnlyPaths={release}",
        f"ReadOnlyPaths={MAC_OPS_CONFIG_PATH}",
        f"InaccessiblePaths={MAC_OPS_CREDENTIAL_PATH}",
        f"ReadWritePaths={MAC_OPS_SOCKET_PATH.parent}",
        f"ReadWritePaths={MAC_OPS_STATE_PATH}",
        "",
        "[Install]",
        "WantedBy=multi-user.target",
    ]
    result = _unit_bytes(lines)
    _require_safe_unit(result, revision=revision, persistent=True)
    return result


def _render_browser_controller_config(
    *,
    release: Path,
    gateway_uid: int,
    browser_gid: int,
    node_path: Path,
    node_sha256: str,
    wrapper_path: Path,
    wrapper_sha256: str,
    native_path: Path,
    native_sha256: str,
    chrome_path: Path,
    chrome_sha256: str,
    agent_browser_config_path: Path,
    agent_browser_config_sha256: str,
) -> bytes:
    value = {
        "schema": BROWSER_CONTROLLER_CONFIG_SCHEMA,
        "socket_path": str(BROWSER_SOCKET_PATH),
        "socket_runtime_root": str(BROWSER_RUNTIME_PATH),
        "socket_gid": browser_gid,
        "allowed_client_uid": gateway_uid,
        "session_root": str(BROWSER_STATE_PATH),
        "release_root": str(release),
        "node_path": str(node_path),
        "node_sha256": node_sha256,
        "wrapper_path": str(wrapper_path),
        "wrapper_sha256": wrapper_sha256,
        "native_path": str(native_path),
        "native_sha256": native_sha256,
        "chrome_path": str(chrome_path),
        "chrome_sha256": chrome_sha256,
        "agent_browser_config_path": str(agent_browser_config_path),
        "agent_browser_config_sha256": agent_browser_config_sha256,
        "command_timeout_seconds": BROWSER_COMMAND_TIMEOUT_SECONDS,
        "idle_timeout_seconds": BROWSER_IDLE_TIMEOUT_SECONDS,
        "max_connections": BROWSER_MAX_CONNECTIONS,
        "max_sessions": BROWSER_MAX_SESSIONS,
        "session_quota_bytes": BROWSER_SESSION_QUOTA_BYTES,
        "session_quota_entries": BROWSER_SESSION_QUOTA_ENTRIES,
    }
    return _canonical_bytes(value)


def _render_browser_unit(
    *,
    revision: str,
    release: Path,
    interpreter: Path,
    browser_user: str,
    browser_group: str,
    browser_uid: int,
    browser_gid: int,
    browser_config_sha256: str,
    node_path: Path,
    wrapper_path: Path,
    native_path: Path,
    chrome_path: Path,
    agent_browser_config_path: Path,
) -> bytes:
    lines = [
        "# Dedicated no-secret browser controller for Cloud Muncho production.",
        f"# ReleaseRevision={revision}",
        f"# PrincipalUID={browser_uid}",
        f"# PrincipalGID={browser_gid}",
        f"# ControllerConfigSHA256={browser_config_sha256}",
        "[Unit]",
        "Description=Muncho production isolated browser controller",
        "After=network-online.target",
        "Wants=network-online.target",
        f"Before={GATEWAY_UNIT}",
        "StartLimitIntervalSec=300s",
        "StartLimitBurst=5",
        f"AssertPathExists={interpreter}",
        f"AssertPathExists={release / '.codex-source-commit'}",
        f"AssertPathExists={release / RUNTIME_DEPENDENCY_MANIFEST_RELATIVE}",
        f"AssertPathExists={BROWSER_CONFIG_PATH}",
        f"AssertPathExists={node_path}",
        f"AssertPathExists={wrapper_path}",
        f"AssertPathExists={native_path}",
        f"AssertPathExists={chrome_path}",
        f"AssertPathExists={agent_browser_config_path}",
        f"AssertPathExists={BROWSER_RESOLV_CONF_PATH}",
        "",
        "[Service]",
        "Type=notify",
        "NotifyAccess=main",
        f"User={browser_user}",
        f"Group={browser_group}",
        "RuntimeDirectory=muncho-browser-controller",
        "RuntimeDirectoryMode=0750",
        "RuntimeDirectoryPreserve=no",
        "StateDirectory=muncho-browser-controller",
        "StateDirectoryMode=0700",
        f"WorkingDirectory={release}",
        (
            f"ExecStart={interpreter} -B -P -s -m "
            "gateway.browser_controller "
            f"--config {BROWSER_CONFIG_PATH}"
        ),
        "Restart=on-failure",
        "RestartSec=5s",
        "TimeoutStartSec=30s",
        "TimeoutStopSec=30s",
        "KillMode=control-group",
        "OOMPolicy=stop",
        "TasksMax=512",
        "MemoryMax=2G",
        "MemorySwapMax=512M",
        "LimitNOFILE=8192",
        "LimitCORE=0",
        *_fixed_environment(
            user=browser_user, home=BROWSER_STATE_PATH, release=release
        ),
        f"Environment=XDG_RUNTIME_DIR={BROWSER_RUNTIME_PATH}",
        "UnsetEnvironment=ALL_PROXY HTTP_PROXY HTTPS_PROXY NO_PROXY",
        # Chrome needs its own namespace sandbox.  Do not add
        # RestrictNamespaces=yes or --no-sandbox to this unit.
        *_common_hardening(restrict_namespaces=False),
        "RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6",
        f"IPAddressAllow={BROWSER_DNS_ALLOW}",
        *(f"IPAddressDeny={value}" for value in BROWSER_NETWORK_DENY_RANGES),
        (f"BindReadOnlyPaths={BROWSER_RESOLV_CONF_PATH}:/etc/resolv.conf"),
        f"ReadOnlyPaths={release}",
        f"ReadOnlyPaths={BROWSER_CONFIG_PATH}",
        f"ReadOnlyPaths={BROWSER_RESOLV_CONF_PATH}",
        f"ReadWritePaths={BROWSER_RUNTIME_PATH}",
        f"ReadWritePaths={BROWSER_STATE_PATH}",
        f"InaccessiblePaths={PRODUCTION_HOME}",
        f"InaccessiblePaths={CODEX_AUTH_PATH}",
        "InaccessiblePaths=/run/credentials",
        f"InaccessiblePaths={PUBLIC_CONNECTOR_CREDENTIAL_PATH.parent}",
        f"InaccessiblePaths={ROUTEBACK_EDGE_CREDENTIAL_PATH.parent}",
        f"InaccessiblePaths={MAC_OPS_CREDENTIAL_PATH.parent}",
        "InaccessiblePaths=/etc/muncho/keys",
        f"InaccessiblePaths={PRODUCTION_CUTOVER_CREDENTIAL_ROOT}",
        "",
        "[Install]",
        "WantedBy=multi-user.target",
    ]
    result = _unit_bytes(lines)
    _require_safe_unit(result, revision=revision, persistent=True)
    return result


def render_production_capability_units(
    *,
    revision: str,
    database_ip: str,
    gateway_user: str,
    gateway_group: str,
    gateway_uid: int,
    gateway_gid: int,
    routeback_user: str,
    routeback_group: str,
    routeback_uid: int,
    routeback_gid: int,
    mac_ops_user: str,
    mac_ops_group: str,
    mac_ops_uid: int,
    mac_ops_gid: int,
    browser_user: str,
    browser_group: str,
    browser_uid: int,
    browser_gid: int,
    socket_client_group: str,
    browser_node_path: str,
    browser_node_sha256: str,
    browser_wrapper_path: str,
    browser_wrapper_sha256: str,
    browser_native_path: str,
    browser_native_sha256: str,
    browser_chrome_path: str,
    browser_chrome_sha256: str,
    agent_browser_config_path: str,
    agent_browser_config_sha256: str,
) -> ProductionCapabilityUnitBundle:
    """Render the exact four-unit production prerequisite bundle.

    Every principal name and the SQL endpoint are explicit inputs.  This
    renderer therefore cannot infer an identity or choose a network target.
    """

    revision = _revision(revision)
    gateway_user = _identity(gateway_user, "gateway user")
    gateway_group = _identity(gateway_group, "gateway group")
    gateway_uid = _positive_id(gateway_uid, "gateway uid")
    gateway_gid = _positive_id(gateway_gid, "gateway gid")
    routeback_user = _identity(routeback_user, "route-back user")
    routeback_group = _identity(routeback_group, "route-back group")
    routeback_uid = _positive_id(routeback_uid, "route-back uid")
    routeback_gid = _positive_id(routeback_gid, "route-back gid")
    mac_ops_user = _identity(mac_ops_user, "Mac operations user")
    mac_ops_group = _identity(mac_ops_group, "Mac operations group")
    mac_ops_uid = _positive_id(mac_ops_uid, "Mac operations uid")
    mac_ops_gid = _positive_id(mac_ops_gid, "Mac operations gid")
    browser_user = _identity(browser_user, "browser user")
    browser_group = _identity(browser_group, "browser group")
    browser_uid = _positive_id(browser_uid, "browser uid")
    browser_gid = _positive_id(browser_gid, "browser gid")
    socket_client_group = _identity(
        socket_client_group, "Mac operations socket client group"
    )
    if (
        socket_client_group != mac_ops_group
        or len({gateway_user, routeback_user, mac_ops_user, browser_user}) != 4
        or len({gateway_group, routeback_group, mac_ops_group, browser_group}) != 4
        or len({gateway_uid, routeback_uid, mac_ops_uid, browser_uid}) != 4
        or len({gateway_gid, routeback_gid, mac_ops_gid, browser_gid}) != 4
    ):
        raise ProductionCapabilityUnitError(
            "production capability service identities must be pairwise distinct"
        )
    database_ip_allow = _database_ip_allow(database_ip)
    release = production_release_root(revision)
    if (
        release.parent != PRODUCTION_RELEASES
        or release.name != f"hermes-agent-{revision[:12]}"
    ):
        raise ProductionCapabilityUnitError(
            "production capability release path is not exact"
        )
    interpreter = release / ".venv/bin/python"
    node_path = _release_artifact_path(
        browser_node_path,
        expected=release / NODE_EXECUTABLE,
        label="browser Node path",
    )
    wrapper_path = _release_artifact_path(
        browser_wrapper_path,
        expected=release / AGENT_BROWSER_WRAPPER,
        label="agent-browser wrapper path",
    )
    native_path = _release_artifact_path(
        browser_native_path,
        expected=release / AGENT_BROWSER_NATIVE,
        label="agent-browser native path",
    )
    chrome_path = _release_artifact_path(
        browser_chrome_path,
        expected=release / CHROME_EXECUTABLE,
        label="browser Chrome path",
    )
    trusted_agent_browser_config_path = _release_artifact_path(
        agent_browser_config_path,
        expected=release / AGENT_BROWSER_CONFIG,
        label="agent-browser config path",
    )
    node_sha256 = _digest(browser_node_sha256, "browser Node digest")
    wrapper_sha256 = _digest(browser_wrapper_sha256, "agent-browser wrapper digest")
    native_sha256 = _digest(browser_native_sha256, "agent-browser native digest")
    chrome_sha256 = _digest(browser_chrome_sha256, "browser Chrome digest")
    trusted_agent_browser_config_sha256 = _digest(
        agent_browser_config_sha256, "agent-browser config digest"
    )

    phase_b = _render_phase_b_unit(
        revision=revision,
        release=release,
        interpreter=interpreter,
        database_ip_allow=database_ip_allow,
    )
    routeback = _render_routeback_unit(
        revision=revision,
        release=release,
        interpreter=interpreter,
        routeback_user=routeback_user,
        routeback_group=routeback_group,
        routeback_uid=routeback_uid,
        routeback_gid=routeback_gid,
    )
    mac_ops = _render_mac_ops_unit(
        revision=revision,
        release=release,
        interpreter=interpreter,
        mac_ops_user=mac_ops_user,
        mac_ops_group=mac_ops_group,
        mac_ops_uid=mac_ops_uid,
        mac_ops_gid=mac_ops_gid,
        socket_client_group=socket_client_group,
    )
    browser_config = _render_browser_controller_config(
        release=release,
        gateway_uid=gateway_uid,
        browser_gid=browser_gid,
        node_path=node_path,
        node_sha256=node_sha256,
        wrapper_path=wrapper_path,
        wrapper_sha256=wrapper_sha256,
        native_path=native_path,
        native_sha256=native_sha256,
        chrome_path=chrome_path,
        chrome_sha256=chrome_sha256,
        agent_browser_config_path=trusted_agent_browser_config_path,
        agent_browser_config_sha256=trusted_agent_browser_config_sha256,
    )
    browser_config_sha256 = _sha256(browser_config)
    browser = _render_browser_unit(
        revision=revision,
        release=release,
        interpreter=interpreter,
        browser_user=browser_user,
        browser_group=browser_group,
        browser_uid=browser_uid,
        browser_gid=browser_gid,
        browser_config_sha256=browser_config_sha256,
        node_path=node_path,
        wrapper_path=wrapper_path,
        native_path=native_path,
        chrome_path=chrome_path,
        agent_browser_config_path=trusted_agent_browser_config_path,
    )
    hashes = {
        PHASE_B_UNIT: _sha256(phase_b),
        ROUTEBACK_EDGE_UNIT: _sha256(routeback),
        MAC_OPS_UNIT: _sha256(mac_ops),
        BROWSER_UNIT: _sha256(browser),
    }
    unsigned_manifest = {
        "schema": "muncho-production-capability-unit-bundle.v1",
        "release_revision": revision,
        "release_root": str(release),
        "interpreter": str(interpreter),
        "units": hashes,
        "configs": {
            str(BROWSER_CONFIG_PATH): {
                "gid": browser_gid,
                "mode": f"{BROWSER_CONFIG_MODE:04o}",
                "schema": BROWSER_CONTROLLER_CONFIG_SCHEMA,
                "sha256": browser_config_sha256,
                "uid": BROWSER_CONFIG_UID,
            }
        },
        "browser_controller": {
            "allowed_client_uid": gateway_uid,
            "service_gid": browser_gid,
            "service_uid": browser_uid,
            "socket_path": str(BROWSER_SOCKET_PATH),
            "state_path": str(BROWSER_STATE_PATH),
        },
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    return ProductionCapabilityUnitBundle(
        revision=revision,
        release_root=release,
        interpreter=interpreter,
        phase_b_unit=phase_b,
        phase_b_sha256=hashes[PHASE_B_UNIT],
        routeback_unit=routeback,
        routeback_sha256=hashes[ROUTEBACK_EDGE_UNIT],
        mac_ops_unit=mac_ops,
        mac_ops_sha256=hashes[MAC_OPS_UNIT],
        browser_unit=browser,
        browser_sha256=hashes[BROWSER_UNIT],
        browser_config_path=BROWSER_CONFIG_PATH,
        browser_config=browser_config,
        browser_config_sha256=browser_config_sha256,
        browser_config_uid=BROWSER_CONFIG_UID,
        browser_config_gid=browser_gid,
        browser_config_mode=BROWSER_CONFIG_MODE,
        browser_service_uid=browser_uid,
        browser_service_gid=browser_gid,
        browser_allowed_client_uid=gateway_uid,
        bundle_sha256=_sha256(_canonical_bytes(unsigned_manifest)),
    )


def render_production_routeback_config(
    *,
    gateway_uid: int,
    routeback_uid: int,
    routeback_gid: int,
    writer_capability_public_key_id: str,
    edge_receipt_public_key_id: str,
    connection_timeout_seconds: float,
    max_connections: int,
    api_timeout_seconds: float,
    journal_busy_timeout_ms: int,
    max_proof_age_ms: int,
) -> bytes:
    """Render strict route-back config from explicit mechanical identities."""

    gateway_uid = _positive_id(gateway_uid, "gateway uid")
    routeback_uid = _positive_id(routeback_uid, "route-back uid")
    routeback_gid = _positive_id(routeback_gid, "route-back gid")
    if gateway_uid == routeback_uid:
        raise ProductionCapabilityUnitError(
            "gateway and route-back UIDs must be distinct"
        )
    writer_key_id = _digest(
        writer_capability_public_key_id, "writer capability public key id"
    )
    edge_key_id = _digest(edge_receipt_public_key_id, "edge receipt public key id")
    if writer_key_id == edge_key_id:
        raise ProductionCapabilityUnitError(
            "writer and route-back signing identities must be distinct"
        )
    connection_timeout = _bounded_number(
        connection_timeout_seconds,
        "route-back connection timeout",
        minimum=1,
        maximum=300,
    )
    connections = _bounded_integer(
        max_connections, "route-back max connections", minimum=1, maximum=64
    )
    api_timeout = _bounded_number(
        api_timeout_seconds,
        "route-back API timeout",
        minimum=0.1,
        maximum=15,
    )
    journal_timeout = _bounded_integer(
        journal_busy_timeout_ms,
        "route-back journal timeout",
        minimum=1,
        maximum=30_000,
    )
    proof_age = _bounded_integer(
        max_proof_age_ms,
        "route-back proof age",
        minimum=1,
        maximum=30_000,
    )
    credential_root = _credential_runtime_path(
        ROUTEBACK_EDGE_UNIT, ROUTEBACK_TOKEN_CREDENTIAL_NAME
    ).parent
    value = {
        "service": {
            "socket_path": str(ROUTEBACK_EDGE_SOCKET_PATH),
            "gateway_unit": GATEWAY_UNIT,
            "edge_unit": ROUTEBACK_EDGE_UNIT,
            "gateway_uid": gateway_uid,
            "edge_uid": routeback_uid,
            "edge_gid": routeback_gid,
            "connection_timeout_seconds": connection_timeout,
            "max_connections": connections,
        },
        "keys": {
            "writer_capability_public_key_file": str(ROUTEBACK_WRITER_PUBLIC_KEY_PATH),
            "writer_capability_public_key_id": writer_key_id,
            "edge_receipt_private_key_file": str(
                credential_root / ROUTEBACK_PRIVATE_KEY_CREDENTIAL_NAME
            ),
            "edge_receipt_public_key_id": edge_key_id,
        },
        "discord": {
            "token_file": str(credential_root / ROUTEBACK_TOKEN_CREDENTIAL_NAME),
            "credentials_directory": str(credential_root),
            "api_timeout_seconds": api_timeout,
            "target_policy": "guild_acl",
        },
        "journal": {
            "path": str(ROUTEBACK_EDGE_JOURNAL_PATH),
            "busy_timeout_ms": journal_timeout,
        },
        "runtime": {"max_proof_age_ms": proof_age},
    }
    result = _canonical_bytes(value)
    if (
        str(ROUTEBACK_EDGE_CREDENTIAL_PATH).encode("ascii") in result
        or str(ROUTEBACK_EDGE_PRIVATE_KEY_PATH).encode("ascii") in result
    ):
        raise ProductionCapabilityUnitError(
            "route-back config bypasses its systemd credential boundary"
        )
    return result


def render_production_mac_ops_config(
    *,
    gateway_uid: int,
    socket_gid: int,
    service_identity_sha256: str,
    max_connections: int,
    project_id: str,
    timeout_seconds: float,
    journal_busy_timeout_ms: int,
) -> bytes:
    """Render strict Mac-edge config from explicit mechanical identities."""

    gateway_uid = _positive_id(gateway_uid, "gateway uid")
    socket_gid = _positive_id(socket_gid, "Mac operations socket gid")
    service_identity = _digest(
        service_identity_sha256, "Mac operations service identity"
    )
    connections = _bounded_integer(
        max_connections, "Mac operations max connections", minimum=1, maximum=32
    )
    if not isinstance(project_id, str) or project_id != MAC_OPS_PROJECT_ID:
        raise ProductionCapabilityUnitError(
            "Mac operations project identity is not runtime-pinned"
        )
    timeout = _bounded_number(
        timeout_seconds,
        "Mac operations GitLab timeout",
        minimum=1,
        maximum=30,
    )
    journal_timeout = _bounded_integer(
        journal_busy_timeout_ms,
        "Mac operations journal timeout",
        minimum=100,
        maximum=30_000,
    )
    value = {
        "schema": MAC_OPS_CONFIG_SCHEMA,
        "service": {
            "socket_path": str(MAC_OPS_SOCKET_PATH),
            "gateway_uid": gateway_uid,
            "socket_gid": socket_gid,
            "service_identity_sha256": service_identity,
            "max_connections": connections,
        },
        "gitlab": {
            "env_file": str(
                _credential_runtime_path(MAC_OPS_UNIT, MAC_OPS_CREDENTIAL_NAME)
            ),
            "project_id": project_id,
            "timeout_seconds": timeout,
        },
        "journal": {
            "path": str(MAC_OPS_JOURNAL_PATH),
            "busy_timeout_ms": journal_timeout,
        },
    }
    result = _canonical_bytes(value)
    if str(MAC_OPS_CREDENTIAL_PATH).encode("ascii") in result:
        raise ProductionCapabilityUnitError(
            "Mac operations config bypasses its systemd credential boundary"
        )
    return result


__all__ = [
    "BROWSER_CONFIG_PATH",
    "BROWSER_NETWORK_DENY_RANGES",
    "BROWSER_SOCKET_PATH",
    "BROWSER_STATE_PATH",
    "MAC_OPS_CREDENTIAL_NAME",
    "PHASE_B_PGPASS_CREDENTIAL_NAME",
    "ProductionCapabilityUnitBundle",
    "ProductionCapabilityUnitError",
    "ROUTEBACK_PRIVATE_KEY_CREDENTIAL_NAME",
    "ROUTEBACK_TOKEN_CREDENTIAL_NAME",
    "render_production_capability_units",
    "render_production_mac_ops_config",
    "render_production_routeback_config",
]

"""Mechanical production capability topology and prerequisite receipts.

This module contains no request classification, tool selection, task routing,
or completion policy.  It validates only fixed service identities, mounts,
credential leases, runtime readiness facts, and their exact owner-approved
receipt.  Secret values and secret digests are forbidden from every receipt.
"""

from __future__ import annotations

import hashlib
import argparse
import grp
import json
import os
import pwd
import re
import secrets
import shlex
import socket
import stat
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

from gateway.api_verifier_credentials import (
    parse_api_approval_scrypt_verifier,
    parse_api_bearer_verifier,
)
from gateway.isolated_worker_units import (
    BWRAP_PATH,
    ISOLATED_WORKER_CONFIG,
    ISOLATED_WORKER_SERVICE_UNIT,
    ISOLATED_WORKER_SOCKET,
    ISOLATED_WORKER_SOCKET_UNIT,
    SHELL_PATH,
)
from gateway.production_execution_readiness import (
    BROWSER_RECEIPT_SCHEMA,
    WORKER_RECEIPT_SCHEMA,
)


TOPOLOGY_SCHEMA = "muncho-production-capability-topology.v3"
PREREQUISITE_SCHEMA = "muncho-production-capability-prerequisite.v3"
ISOLATED_CANARY_GOAL_TERMINAL_SCHEMA = (
    "muncho-production-capability-goal-continuation-terminal.v2"
)
PREREQUISITE_LIFECYCLE_STAGED = "staged"
PREREQUISITE_LIFECYCLE_COMMITTED = "committed"
PREREQUISITE_LIFECYCLE_PHASES = frozenset({
    PREREQUISITE_LIFECYCLE_STAGED,
    PREREQUISITE_LIFECYCLE_COMMITTED,
})
PREREQUISITE_PATH = Path(
    "/var/lib/muncho-production-capability/prerequisite-receipt.json"
)
BOOT_ID_PATH = Path("/proc/sys/kernel/random/boot_id")
MAX_PREREQUISITE_BYTES = 2 * 1024 * 1024
MAX_PREREQUISITE_AGE_SECONDS = 900

API_SERVER_HOST = "127.0.0.1"
API_SERVER_PORT = 8642
API_SERVER_CREDENTIAL_NAME = "api-server-bearer-sha256"
API_SERVER_CREDENTIAL_PATH = Path("/etc/muncho/keys/api-server-bearer-sha256.json")
API_APPROVAL_CREDENTIAL_NAME = "api-approval-passkey-scrypt"
API_APPROVAL_CREDENTIAL_PATH = Path("/etc/muncho/keys/api-approval-passkey-scrypt.json")
PRODUCTION_CONFIG_PATH = Path("/opt/adventico-ai-platform/hermes-home/config.yaml")
PRODUCTION_HOME = PRODUCTION_CONFIG_PATH.parent
PRODUCTION_RELEASES = Path("/opt/adventico-ai-platform/hermes-agent-releases")
RUNTIME_DEPENDENCY_MANIFEST_RELATIVE = Path(
    "ops/muncho/runtime/dependencies/manifest.json"
)
RUNTIME_DEPENDENCY_MANIFEST_SCHEMA = "muncho-production-runtime-dependencies.v1"
AGENT_BROWSER_VERSION = "0.26.0"
DDGS_VERSION = "9.14.4"
CHROME_VERSION = "150.0.7871.114"
NODE_VERSION = "v24.18.0"
GATEWAY_STATE_DIRECTORY = Path("/var/lib/hermes-cloud-gateway")
CODEX_AUTH_PATH = Path("/opt/adventico-ai-platform/hermes-home/auth.json")

PHASE_B_UNIT = "muncho-canonical-writer-phase-b-readiness.service"
ROUTEBACK_EDGE_UNIT = "muncho-discord-egress.service"
PUBLIC_CONNECTOR_UNIT = "muncho-discord-connector.service"
PUBLIC_CONNECTOR_CONFIG_PATH = Path("/etc/muncho/discord-public-connector.json")
PUBLIC_CONNECTOR_SOCKET_PATH = Path("/run/muncho-discord-connector/connector.sock")
PUBLIC_CONNECTOR_CREDENTIAL_PATH = Path(
    "/etc/muncho/discord-connector-credentials/bot-token"
)
PUBLIC_CONNECTOR_READINESS_PATH = Path("/run/muncho-discord-connector/readiness.json")
MAC_OPS_UNIT = "muncho-mac-ops-edge.service"
BROWSER_UNIT = "muncho-capability-browser.service"
WRITER_UNIT = "muncho-canonical-writer.service"
GATEWAY_UNIT = "hermes-cloud-gateway.service"

PHASE_B_RECEIPT_PATH = Path(
    "/var/lib/muncho/canonical-writer-phase-b/runtime-receipt.json"
)
ROUTEBACK_EDGE_CONFIG_PATH = Path("/etc/muncho/discord-edge.json")
ROUTEBACK_EDGE_SOCKET_PATH = Path("/run/muncho-discord-egress/edge.sock")
ROUTEBACK_EDGE_CREDENTIAL_PATH = Path("/etc/muncho/discord-edge-credentials/bot-token")
ROUTEBACK_EDGE_READINESS_PATH = Path(
    "/run/muncho-discord-egress/runtime-attestation.json"
)
MAC_OPS_CONFIG_PATH = Path("/etc/muncho/mac-ops-edge/config.json")
MAC_OPS_SOCKET_PATH = Path("/run/muncho-mac-ops/edge.sock")
MAC_OPS_CREDENTIAL_PATH = Path("/etc/muncho/mac-ops-edge-credentials/gitlab.env")
MAC_OPS_JOURNAL_PATH = Path("/var/lib/muncho-mac-ops/journal.db")
BROWSER_CONFIG_PATH = Path("/etc/muncho/browser-controller.json")
BROWSER_SOCKET_PATH = Path("/run/muncho-browser-controller/controller.sock")
BROWSER_STATE_PATH = Path("/var/lib/muncho-browser-controller")
BROWSER_ARTIFACT_PATH = Path("/run/hermes-cloud-gateway/browser-artifacts")

FIRST_WAVE_TOOLSETS = (
    "browser",
    "canonical_brain",
    "clarify",
    "delegation",
    "discord_guild_read",
    "file",
    "mac_ops",
    "memory",
    "session_search",
    "skills",
    "terminal",
    "todo",
    "web",
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40}")
_ISOLATED_WORKER_TOPOLOGY_FIELDS = frozenset({
    "socket_unit",
    "socket_fragment_sha256",
    "service_unit",
    "service_fragment_sha256",
    "config_path",
    "config_sha256",
    "socket_path",
    "socket_uid",
    "socket_gid",
    "server_uid",
    "server_gid",
    "gateway_uid",
    "gateway_gid",
    "bwrap_path",
    "bwrap_sha256",
    "shell_path",
    "shell_sha256",
})
_BROWSER_TOPOLOGY_FIELDS = frozenset({
    "unit",
    "fragment_sha256",
    "config_path",
    "config_sha256",
    "socket_path",
    "service_uid",
    "service_gid",
    "node_path",
    "node_sha256",
    "wrapper_path",
    "wrapper_sha256",
    "native_path",
    "native_sha256",
    "executable",
    "executable_sha256",
    "agent_browser_config_path",
    "agent_browser_config_sha256",
})
_ISOLATED_WORKER_RECEIPT_FIELDS = frozenset({
    "socket_unit",
    "socket_fragment_path",
    "socket_fragment_sha256",
    "socket_unit_file_state",
    "socket_active_state",
    "socket_sub_state",
    "socket_drop_in_paths",
    "socket_need_daemon_reload",
    "service_unit",
    "service_main_pid",
    "config_path",
    "config_sha256",
    "config_uid",
    "config_gid",
    "config_mode",
    "socket_path",
    "socket_uid",
    "socket_gid",
    "socket_device",
    "socket_inode",
    "socket_mode",
    "bwrap_path",
    "bwrap_sha256",
    "shell_path",
    "shell_sha256",
    "ready",
})
_BROWSER_RECEIPT_FIELDS = frozenset(
    set(_BROWSER_TOPOLOGY_FIELDS)
    | {
        "config_uid",
        "config_gid",
        "config_mode",
        "socket_uid",
        "socket_gid",
        "socket_device",
        "socket_inode",
        "socket_mode",
        "service_main_pid",
        "ready",
    }
)
_TOPOLOGY_FIELDS = frozenset({
    "schema",
    "prerequisite_receipt_path",
    "collector_contract_sha256",
    "isolated_worker",
    "browser",
    "mac_ops",
    "routeback_edge",
    "public_connector",
    "phase_b",
    "codex_auth_file",
    "api_control_credential_file",
    "api_approval_credential_file",
    "gateway_identity",
})
_RECEIPT_FIELDS = frozenset({
    "schema",
    "release_revision",
    "lifecycle_phase",
    "topology_identity_sha256",
    "boot_id_sha256",
    "observed_at_unix",
    "services",
    "sockets",
    "isolated_worker",
    "browser",
    "runtime_dependencies",
    "gateway_state",
    "capability_proofs",
    "credentials",
    "secret_material_recorded",
    "secret_digest_recorded",
    "receipt_sha256",
})
_SERVICE_FIELDS = frozenset({
    "unit",
    "fragment_path",
    "fragment_sha256",
    "unit_file_state",
    "active_state",
    "sub_state",
    "service_type",
    "main_pid",
    "drop_in_paths",
    "need_daemon_reload",
    "effective_user",
    "effective_uid",
    "effective_group",
    "effective_gid",
    "effective_supplementary_groups",
    "unit_executable",
    "unit_cmdline_sha256",
    "unit_service_contract_sha256",
    "main_pid_executable",
    "main_pid_uid",
    "main_pid_gid",
    "main_pid_groups",
    "main_pid_cmdline_sha256",
    "main_pid_cgroup",
    "main_pid_mount_namespace_inode",
    "main_pid_network_namespace_inode",
    "process_identity_matches_unit",
    "readiness_receipt_sha256",
    "ready",
})
_SOCKET_FIELDS = frozenset({
    "path",
    "device",
    "inode",
    "owner_uid",
    "group_gid",
    "mode",
    "main_pid",
    "ready",
})
_LEASE_FIELDS = frozenset({
    "path",
    "owner_uid",
    "group_gid",
    "mode",
    "size",
    "regular_one_link",
    "usable",
    "refresh_capable",
    "secret_material_recorded",
    "secret_digest_recorded",
})


class ProductionCapabilityPrerequisiteError(RuntimeError):
    """Stable, non-secret prerequisite failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _exact_mapping(value: Any, fields: frozenset[str], code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ProductionCapabilityPrerequisiteError(code)
    return value


def _require_lifecycle_phase(value: Any) -> str:
    if type(value) is not str or value not in PREREQUISITE_LIFECYCLE_PHASES:
        raise ProductionCapabilityPrerequisiteError(
            "production_prerequisite_lifecycle_phase_invalid"
        )
    return value


def _digest(value: Any, code: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ProductionCapabilityPrerequisiteError(code)
    return value


def _absolute_path(value: Any, expected: Path, code: str) -> str:
    if (
        not isinstance(value, str)
        or Path(value) != expected
        or not expected.is_absolute()
        or os.path.normpath(value) != value
    ):
        raise ProductionCapabilityPrerequisiteError(code)
    return value


def production_release_root(revision: str) -> Path:
    if not isinstance(revision, str) or _REVISION.fullmatch(revision) is None:
        raise ProductionCapabilityPrerequisiteError("production_revision_invalid")
    return PRODUCTION_RELEASES / f"hermes-agent-{revision[:12]}"


def production_browser_executable(revision: str) -> Path:
    return (
        production_release_root(revision)
        / "ops/muncho/runtime/dependencies/chrome-linux64/chrome"
    )


def production_browser_node(revision: str) -> Path:
    return (
        production_release_root(revision)
        / "ops/muncho/runtime/dependencies/node-linux-x64/bin/node"
    )


def production_browser_wrapper(revision: str) -> Path:
    return (
        production_release_root(revision)
        / "node_modules/agent-browser/bin/agent-browser.js"
    )


def production_browser_native(revision: str) -> Path:
    return (
        production_release_root(revision)
        / "node_modules/agent-browser/bin/agent-browser-linux-x64"
    )


def production_agent_browser_config(revision: str) -> Path:
    return (
        production_release_root(revision)
        / "ops/muncho/runtime/dependencies/agent-browser.json"
    )


def _exact_release_path(value: Any, expected: Path, code: str) -> str:
    if (
        not isinstance(value, str)
        or Path(value) != expected
        or not expected.is_absolute()
        or Path(os.path.normpath(value)) != expected
    ):
        raise ProductionCapabilityPrerequisiteError(code)
    return value


def _release_artifact_path(value: Any, relative: Path, code: str) -> tuple[str, Path]:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ProductionCapabilityPrerequisiteError(code)
    path = Path(value)
    if not path.is_absolute() or os.path.normpath(value) != value:
        raise ProductionCapabilityPrerequisiteError(code)
    try:
        parts = path.relative_to(PRODUCTION_RELEASES).parts
    except ValueError as exc:
        raise ProductionCapabilityPrerequisiteError(code) from exc
    if (
        len(parts) != len(relative.parts) + 1
        or re.fullmatch(r"hermes-agent-[0-9a-f]{12}", parts[0]) is None
        or Path(*parts[1:]) != relative
    ):
        raise ProductionCapabilityPrerequisiteError(code)
    return value, PRODUCTION_RELEASES / parts[0]


def _require_browser_revision_binding(
    browser: Mapping[str, Any], *, revision: str
) -> None:
    expected = {
        "node_path": production_browser_node(revision),
        "wrapper_path": production_browser_wrapper(revision),
        "native_path": production_browser_native(revision),
        "executable": production_browser_executable(revision),
        "agent_browser_config_path": production_agent_browser_config(revision),
    }
    if any(browser.get(field) != str(path) for field, path in expected.items()):
        raise ProductionCapabilityPrerequisiteError(
            "production_browser_release_binding_invalid"
        )


def validate_production_capability_topology(
    value: Any,
) -> Mapping[str, Any]:
    """Validate the exact config-bound production capability topology."""

    raw = _exact_mapping(value, _TOPOLOGY_FIELDS, "production_topology_fields_invalid")
    if raw["schema"] != TOPOLOGY_SCHEMA:
        raise ProductionCapabilityPrerequisiteError(
            "production_topology_schema_invalid"
        )
    _absolute_path(
        raw["prerequisite_receipt_path"],
        PREREQUISITE_PATH,
        "production_prerequisite_path_invalid",
    )
    if raw["collector_contract_sha256"] != packaged_prerequisite_contract_sha256():
        raise ProductionCapabilityPrerequisiteError(
            "production_collector_contract_digest_invalid"
        )

    worker = _exact_mapping(
        raw["isolated_worker"],
        _ISOLATED_WORKER_TOPOLOGY_FIELDS,
        "production_isolated_worker_topology_invalid",
    )
    if (
        worker["socket_unit"] != ISOLATED_WORKER_SOCKET_UNIT
        or worker["service_unit"] != ISOLATED_WORKER_SERVICE_UNIT
        or any(
            _SHA256.fullmatch(str(worker[field])) is None
            for field in (
                "socket_fragment_sha256",
                "service_fragment_sha256",
                "config_sha256",
                "bwrap_sha256",
                "shell_sha256",
            )
        )
        or _absolute_path(
            worker["config_path"],
            ISOLATED_WORKER_CONFIG,
            "production_isolated_worker_topology_invalid",
        )
        != str(ISOLATED_WORKER_CONFIG)
        or _absolute_path(
            worker["socket_path"],
            ISOLATED_WORKER_SOCKET,
            "production_isolated_worker_topology_invalid",
        )
        != str(ISOLATED_WORKER_SOCKET)
        or _absolute_path(
            worker["bwrap_path"],
            BWRAP_PATH,
            "production_isolated_worker_topology_invalid",
        )
        != str(BWRAP_PATH)
        or _absolute_path(
            worker["shell_path"],
            SHELL_PATH,
            "production_isolated_worker_topology_invalid",
        )
        != str(SHELL_PATH)
        or type(worker["socket_uid"]) is not int
        or worker["socket_uid"] != 0
        or any(
            type(worker[field]) is not int or worker[field] <= 0
            for field in (
                "socket_gid",
                "server_uid",
                "server_gid",
                "gateway_uid",
                "gateway_gid",
            )
        )
        or worker["server_uid"] == worker["gateway_uid"]
        or worker["server_gid"] == worker["gateway_gid"]
        or len({
            worker["socket_gid"],
            worker["server_gid"],
            worker["gateway_gid"],
        })
        != 3
    ):
        raise ProductionCapabilityPrerequisiteError(
            "production_isolated_worker_topology_invalid"
        )

    browser = _exact_mapping(
        raw["browser"],
        _BROWSER_TOPOLOGY_FIELDS,
        "production_browser_topology_invalid",
    )
    release_roots: set[Path] = set()
    for field, relative in (
        (
            "node_path",
            Path("ops/muncho/runtime/dependencies/node-linux-x64/bin/node"),
        ),
        ("wrapper_path", Path("node_modules/agent-browser/bin/agent-browser.js")),
        (
            "native_path",
            Path("node_modules/agent-browser/bin/agent-browser-linux-x64"),
        ),
        (
            "executable",
            Path("ops/muncho/runtime/dependencies/chrome-linux64/chrome"),
        ),
        (
            "agent_browser_config_path",
            Path("ops/muncho/runtime/dependencies/agent-browser.json"),
        ),
    ):
        _value, release_root = _release_artifact_path(
            browser[field], relative, "production_browser_topology_invalid"
        )
        release_roots.add(release_root)
    if (
        browser["unit"] != BROWSER_UNIT
        or any(
            _SHA256.fullmatch(str(browser[field])) is None
            for field in (
                "fragment_sha256",
                "config_sha256",
                "node_sha256",
                "wrapper_sha256",
                "native_sha256",
                "executable_sha256",
                "agent_browser_config_sha256",
            )
        )
        or _absolute_path(
            browser["config_path"],
            BROWSER_CONFIG_PATH,
            "production_browser_topology_invalid",
        )
        != str(BROWSER_CONFIG_PATH)
        or _absolute_path(
            browser["socket_path"],
            BROWSER_SOCKET_PATH,
            "production_browser_topology_invalid",
        )
        != str(BROWSER_SOCKET_PATH)
        or type(browser["service_uid"]) is not int
        or browser["service_uid"] <= 0
        or type(browser["service_gid"]) is not int
        or browser["service_gid"] <= 0
        or len(release_roots) != 1
    ):
        raise ProductionCapabilityPrerequisiteError(
            "production_browser_topology_invalid"
        )

    mac_ops = _exact_mapping(
        raw["mac_ops"],
        frozenset({
            "unit",
            "fragment_sha256",
            "config_sha256",
            "config_path",
            "socket_path",
            "credential_path",
            "journal_path",
        }),
        "production_mac_ops_topology_invalid",
    )
    if (
        mac_ops["unit"] != MAC_OPS_UNIT
        or _digest(
            mac_ops["fragment_sha256"],
            "production_mac_ops_topology_invalid",
        )
        != mac_ops["fragment_sha256"]
        or _digest(
            mac_ops["config_sha256"],
            "production_mac_ops_topology_invalid",
        )
        != mac_ops["config_sha256"]
    ):
        raise ProductionCapabilityPrerequisiteError(
            "production_mac_ops_topology_invalid"
        )
    for key, expected in (
        ("config_path", MAC_OPS_CONFIG_PATH),
        ("socket_path", MAC_OPS_SOCKET_PATH),
        ("credential_path", MAC_OPS_CREDENTIAL_PATH),
        ("journal_path", MAC_OPS_JOURNAL_PATH),
    ):
        _absolute_path(mac_ops[key], expected, "production_mac_ops_topology_invalid")

    routeback = _exact_mapping(
        raw["routeback_edge"],
        frozenset({
            "unit",
            "fragment_sha256",
            "config_sha256",
            "config_path",
            "socket_path",
            "credential_path",
            "readiness_path",
        }),
        "production_routeback_topology_invalid",
    )
    if (
        routeback["unit"] != ROUTEBACK_EDGE_UNIT
        or _digest(
            routeback["fragment_sha256"],
            "production_routeback_topology_invalid",
        )
        != routeback["fragment_sha256"]
        or _digest(
            routeback["config_sha256"],
            "production_routeback_topology_invalid",
        )
        != routeback["config_sha256"]
    ):
        raise ProductionCapabilityPrerequisiteError(
            "production_routeback_topology_invalid"
        )
    for key, expected in (
        ("config_path", ROUTEBACK_EDGE_CONFIG_PATH),
        ("socket_path", ROUTEBACK_EDGE_SOCKET_PATH),
        ("credential_path", ROUTEBACK_EDGE_CREDENTIAL_PATH),
        ("readiness_path", ROUTEBACK_EDGE_READINESS_PATH),
    ):
        _absolute_path(
            routeback[key], expected, "production_routeback_topology_invalid"
        )

    connector = _exact_mapping(
        raw["public_connector"],
        frozenset({
            "unit",
            "fragment_sha256",
            "config_path",
            "socket_path",
            "credential_path",
            "readiness_path",
        }),
        "production_connector_topology_invalid",
    )
    if (
        connector["unit"] != PUBLIC_CONNECTOR_UNIT
        or _digest(
            connector["fragment_sha256"],
            "production_connector_topology_invalid",
        )
        != connector["fragment_sha256"]
    ):
        raise ProductionCapabilityPrerequisiteError(
            "production_connector_topology_invalid"
        )
    for key, expected in (
        ("config_path", PUBLIC_CONNECTOR_CONFIG_PATH),
        ("socket_path", PUBLIC_CONNECTOR_SOCKET_PATH),
        ("credential_path", PUBLIC_CONNECTOR_CREDENTIAL_PATH),
        ("readiness_path", PUBLIC_CONNECTOR_READINESS_PATH),
    ):
        _absolute_path(
            connector[key], expected, "production_connector_topology_invalid"
        )

    phase_b = _exact_mapping(
        raw["phase_b"],
        frozenset({"unit", "fragment_sha256", "readiness_path"}),
        "production_phase_b_topology_invalid",
    )
    if (
        phase_b["unit"] != PHASE_B_UNIT
        or _digest(
            phase_b["fragment_sha256"],
            "production_phase_b_topology_invalid",
        )
        != phase_b["fragment_sha256"]
    ):
        raise ProductionCapabilityPrerequisiteError(
            "production_phase_b_topology_invalid"
        )
    _absolute_path(
        phase_b["readiness_path"],
        PHASE_B_RECEIPT_PATH,
        "production_phase_b_topology_invalid",
    )

    _absolute_path(
        raw["codex_auth_file"],
        CODEX_AUTH_PATH,
        "production_codex_auth_path_invalid",
    )
    _absolute_path(
        raw["api_control_credential_file"],
        API_SERVER_CREDENTIAL_PATH,
        "production_api_credential_path_invalid",
    )
    _absolute_path(
        raw["api_approval_credential_file"],
        API_APPROVAL_CREDENTIAL_PATH,
        "production_api_approval_credential_path_invalid",
    )
    gateway_identity = _exact_mapping(
        raw["gateway_identity"],
        frozenset({"uid", "gid"}),
        "production_gateway_identity_invalid",
    )
    if (
        type(gateway_identity["uid"]) is not int
        or gateway_identity["uid"] <= 0
        or type(gateway_identity["gid"]) is not int
        or gateway_identity["gid"] <= 0
    ):
        raise ProductionCapabilityPrerequisiteError(
            "production_gateway_identity_invalid"
        )
    if (
        worker["gateway_uid"] != gateway_identity["uid"]
        or worker["gateway_gid"] != gateway_identity["gid"]
        or browser["service_uid"] in {gateway_identity["uid"], worker["server_uid"]}
        or browser["service_gid"]
        in {
            gateway_identity["gid"],
            worker["server_gid"],
            worker["socket_gid"],
        }
    ):
        raise ProductionCapabilityPrerequisiteError(
            "production_gateway_identity_invalid"
        )
    return raw


def production_capability_topology_identity_sha256(value: Any) -> str:
    raw = dict(validate_production_capability_topology(value))
    return _sha256(_canonical_bytes(raw))


def _validate_runtime_dependency_proof(
    value: Any, *, revision: str, topology: Mapping[str, Any]
) -> Mapping[str, Any]:
    raw = _exact_mapping(
        value,
        frozenset({
            "manifest_path",
            "manifest_sha256",
            "agent_browser",
            "chrome",
            "ddgs",
            "ready",
        }),
        "production_runtime_dependency_proof_invalid",
    )
    release = production_release_root(revision)
    if (
        raw["manifest_path"] != str(release / RUNTIME_DEPENDENCY_MANIFEST_RELATIVE)
        or _SHA256.fullmatch(str(raw["manifest_sha256"])) is None
        or raw["ready"] is not True
    ):
        raise ProductionCapabilityPrerequisiteError(
            "production_runtime_dependency_proof_invalid"
        )
    agent_browser = _exact_mapping(
        raw["agent_browser"],
        frozenset({
            "version",
            "config_path",
            "config_sha256",
            "wrapper_path",
            "wrapper_sha256",
            "native_path",
            "native_sha256",
            "node_path",
            "node_version",
            "node_sha256",
        }),
        "production_agent_browser_proof_invalid",
    )
    if (
        agent_browser["version"] != AGENT_BROWSER_VERSION
        or agent_browser["config_path"]
        != str(production_agent_browser_config(revision))
        or agent_browser["config_path"]
        != topology["browser"]["agent_browser_config_path"]
        or agent_browser["config_sha256"]
        != topology["browser"]["agent_browser_config_sha256"]
        or agent_browser["wrapper_path"] != str(production_browser_wrapper(revision))
        or agent_browser["wrapper_path"] != topology["browser"]["wrapper_path"]
        or agent_browser["wrapper_sha256"] != topology["browser"]["wrapper_sha256"]
        or agent_browser["native_path"] != str(production_browser_native(revision))
        or agent_browser["native_path"] != topology["browser"]["native_path"]
        or agent_browser["native_sha256"] != topology["browser"]["native_sha256"]
        or agent_browser["node_path"] != str(production_browser_node(revision))
        or agent_browser["node_path"] != topology["browser"]["node_path"]
        or agent_browser["node_sha256"] != topology["browser"]["node_sha256"]
        or agent_browser["node_version"] != NODE_VERSION
        or any(
            _SHA256.fullmatch(str(agent_browser[field])) is None
            for field in (
                "config_sha256",
                "wrapper_sha256",
                "native_sha256",
                "node_sha256",
            )
        )
    ):
        raise ProductionCapabilityPrerequisiteError(
            "production_agent_browser_proof_invalid"
        )
    chrome = _exact_mapping(
        raw["chrome"],
        frozenset({
            "version",
            "executable_path",
            "executable_sha256",
        }),
        "production_chrome_dependency_proof_invalid",
    )
    if (
        chrome["version"] != CHROME_VERSION
        or chrome["executable_path"] != str(production_browser_executable(revision))
        or chrome["executable_path"] != topology["browser"]["executable"]
        or chrome["executable_sha256"] != topology["browser"]["executable_sha256"]
    ):
        raise ProductionCapabilityPrerequisiteError(
            "production_chrome_dependency_proof_invalid"
        )
    ddgs = _exact_mapping(
        raw["ddgs"],
        frozenset({"version", "files_sha256", "gateway_uid_import_smoke"}),
        "production_ddgs_dependency_proof_invalid",
    )
    if (
        ddgs["version"] != DDGS_VERSION
        or _SHA256.fullmatch(str(ddgs["files_sha256"])) is None
        or ddgs["gateway_uid_import_smoke"] is not True
    ):
        raise ProductionCapabilityPrerequisiteError(
            "production_ddgs_dependency_proof_invalid"
        )
    return raw


def _validate_gateway_state_proof(
    value: Any, *, topology: Mapping[str, Any], revision: str
) -> Mapping[str, Any]:
    raw = _exact_mapping(
        value,
        frozenset({
            "schema",
            "gateway_uid",
            "gateway_gid",
            "hermes_home",
            "config",
            "memory",
            "skills",
            "session_db",
            "state_directory",
            "secret_material_recorded",
            "secret_digest_recorded",
            "proof_sha256",
        }),
        "production_gateway_state_proof_invalid",
    )
    unsigned = {key: item for key, item in raw.items() if key != "proof_sha256"}
    if (
        raw["schema"] != "muncho-production-gateway-state-proof.v1"
        or raw["gateway_uid"] != topology["gateway_identity"]["uid"]
        or raw["gateway_gid"] != topology["gateway_identity"]["gid"]
        or raw["hermes_home"] != str(PRODUCTION_HOME)
        or raw["secret_material_recorded"] is not False
        or raw["secret_digest_recorded"] is not False
        or raw["proof_sha256"] != _sha256(_canonical_bytes(unsigned))
    ):
        raise ProductionCapabilityPrerequisiteError(
            "production_gateway_state_proof_invalid"
        )
    config = _exact_mapping(
        raw["config"],
        frozenset({"path", "memory_enabled", "user_profile_enabled", "readable"}),
        "production_gateway_config_proof_invalid",
    )
    if (
        config["path"] != str(PRODUCTION_CONFIG_PATH)
        or config["memory_enabled"] is not True
        or config["user_profile_enabled"] is not True
        or config["readable"] is not True
    ):
        raise ProductionCapabilityPrerequisiteError(
            "production_gateway_config_proof_invalid"
        )
    memory = _exact_mapping(
        raw["memory"],
        frozenset({
            "home",
            "memory",
            "user",
            "built_in_load",
            "built_in_atomic_create_rewrite",
        }),
        "production_gateway_memory_proof_invalid",
    )
    if (
        memory["home"] != str(PRODUCTION_HOME / "memories")
        or memory["built_in_load"] is not True
        or memory["built_in_atomic_create_rewrite"] is not True
    ):
        raise ProductionCapabilityPrerequisiteError(
            "production_gateway_memory_proof_invalid"
        )
    memory_identity_fields = frozenset({
        "path",
        "exists",
        "size",
        "owner_uid",
        "group_gid",
        "mode",
        "readable",
    })
    for name, filename in (("memory", "MEMORY.md"), ("user", "USER.md")):
        item = _exact_mapping(
            memory[name],
            memory_identity_fields,
            "production_gateway_memory_proof_invalid",
        )
        if (
            item["path"] != str(PRODUCTION_HOME / "memories" / filename)
            or type(item["exists"]) is not bool
            or type(item["size"]) is not int
            or not 0 <= item["size"] <= 8 * 1024 * 1024
            or item["owner_uid"] != topology["gateway_identity"]["uid"]
            or item["group_gid"] != topology["gateway_identity"]["gid"]
            or item["readable"] is not True
            or (item["exists"] and item["mode"] not in {"0600", "0640", "0644"})
            or (not item["exists"] and item["mode"] is not None)
        ):
            raise ProductionCapabilityPrerequisiteError(
                "production_gateway_memory_proof_invalid"
            )
    skills = _exact_mapping(
        raw["skills"],
        frozenset({
            "bundled_path",
            "bundled_count",
            "bundled_index_sha256",
            "user_path",
            "user_atomic_roundtrip",
        }),
        "production_gateway_skills_proof_invalid",
    )
    if (
        skills["bundled_path"] != str(production_release_root(revision) / "skills")
        or type(skills["bundled_count"]) is not int
        or not 1 <= skills["bundled_count"] <= 4096
        or _SHA256.fullmatch(str(skills["bundled_index_sha256"])) is None
        or skills["user_path"] != str(PRODUCTION_HOME / "skills")
        or skills["user_atomic_roundtrip"] is not True
    ):
        raise ProductionCapabilityPrerequisiteError(
            "production_gateway_skills_proof_invalid"
        )
    session = _exact_mapping(
        raw["session_db"],
        frozenset({
            "path",
            "journal_mode",
            "fts5_enabled",
            "real_fts_query",
            "owner_uid",
            "group_gid",
            "mode",
        }),
        "production_gateway_session_db_proof_invalid",
    )
    if (
        session["path"] != str(PRODUCTION_HOME / "state.db")
        or session["journal_mode"] != "wal"
        or session["fts5_enabled"] is not True
        or session["real_fts_query"] is not True
        or session["owner_uid"] != topology["gateway_identity"]["uid"]
        or session["group_gid"] != topology["gateway_identity"]["gid"]
        or session["mode"] not in {"0600", "0640", "0644"}
    ):
        raise ProductionCapabilityPrerequisiteError(
            "production_gateway_session_db_proof_invalid"
        )
    state_directory = _exact_mapping(
        raw["state_directory"],
        frozenset({"path", "owner_uid", "group_gid", "mode", "atomic_roundtrip"}),
        "production_gateway_state_directory_proof_invalid",
    )
    if (
        state_directory["path"] != str(GATEWAY_STATE_DIRECTORY)
        or state_directory["owner_uid"] != topology["gateway_identity"]["uid"]
        or state_directory["group_gid"] != topology["gateway_identity"]["gid"]
        or state_directory["mode"] != "0700"
        or state_directory["atomic_roundtrip"] is not True
    ):
        raise ProductionCapabilityPrerequisiteError(
            "production_gateway_state_directory_proof_invalid"
        )
    return raw


def _validate_capability_proofs(
    value: Any, *, topology: Mapping[str, Any], services: Mapping[str, Any]
) -> Mapping[str, Any]:
    raw = _exact_mapping(
        value,
        frozenset({
            "mac_ops_ping",
            "isolated_worker_exec",
            "browser_controller_command",
        }),
        "production_capability_proofs_invalid",
    )
    mac = _exact_mapping(
        raw["mac_ops_ping"],
        frozenset({
            "main_pid",
            "service_identity_sha256",
            "receipt_sha256",
            "peer_main_pid_validated",
            "external_io",
            "ready",
        }),
        "production_mac_ops_ping_proof_invalid",
    )
    if (
        mac["main_pid"] != services["mac_ops"]["main_pid"]
        or _SHA256.fullmatch(str(mac["service_identity_sha256"])) is None
        or _SHA256.fullmatch(str(mac["receipt_sha256"])) is None
        or mac["peer_main_pid_validated"] is not True
        or mac["external_io"] is not False
        or mac["ready"] is not True
    ):
        raise ProductionCapabilityPrerequisiteError(
            "production_mac_ops_ping_proof_invalid"
        )
    worker = _exact_mapping(
        raw["isolated_worker_exec"],
        frozenset({
            "schema",
            "lease_identity_sha256",
            "socket_path",
            "server_uid",
            "server_gid",
            "socket_uid",
            "socket_gid",
            "execution_round_trip",
            "output_sha256",
            "secret_material_recorded",
        }),
        "production_isolated_worker_exec_proof_invalid",
    )
    if (
        worker["schema"] != WORKER_RECEIPT_SCHEMA
        or _SHA256.fullmatch(str(worker["lease_identity_sha256"])) is None
        or worker["socket_path"] != topology["isolated_worker"]["socket_path"]
        or worker["server_uid"] != topology["isolated_worker"]["server_uid"]
        or worker["server_gid"] != topology["isolated_worker"]["server_gid"]
        or worker["socket_uid"] != topology["isolated_worker"]["socket_uid"]
        or worker["socket_gid"] != topology["isolated_worker"]["socket_gid"]
        or worker["execution_round_trip"] is not True
        or worker["output_sha256"] != _sha256(b"MUNCHO_ISOLATED_WORKER_READY\n")
        or worker["secret_material_recorded"] is not False
    ):
        raise ProductionCapabilityPrerequisiteError(
            "production_isolated_worker_exec_proof_invalid"
        )
    browser = _exact_mapping(
        raw["browser_controller_command"],
        frozenset({
            "schema",
            "session_identity_sha256",
            "socket_path",
            "server_uid",
            "command_round_trip",
            "secret_material_recorded",
        }),
        "production_browser_controller_proof_invalid",
    )
    if (
        browser["schema"] != BROWSER_RECEIPT_SCHEMA
        or _SHA256.fullmatch(str(browser["session_identity_sha256"])) is None
        or browser["socket_path"] != topology["browser"]["socket_path"]
        or browser["server_uid"] != topology["browser"]["service_uid"]
        or browser["command_round_trip"] is not True
        or browser["secret_material_recorded"] is not False
        or services["isolated_worker"]["effective_uid"]
        != topology["isolated_worker"]["server_uid"]
        or services["isolated_worker"]["effective_gid"]
        != topology["isolated_worker"]["server_gid"]
        or services["browser"]["effective_uid"] != topology["browser"]["service_uid"]
        or services["browser"]["effective_gid"] != topology["browser"]["service_gid"]
    ):
        raise ProductionCapabilityPrerequisiteError(
            "production_browser_controller_proof_invalid"
        )
    return raw


def _recorded_service_contract_sha256(item: Mapping[str, Any]) -> str:
    return _sha256(
        _canonical_bytes({
            "effective_user": item["effective_user"],
            "effective_uid": item["effective_uid"],
            "effective_group": item["effective_group"],
            "effective_gid": item["effective_gid"],
            "effective_supplementary_groups": item["effective_supplementary_groups"],
            "unit_executable": item["unit_executable"],
            "unit_cmdline_sha256": item["unit_cmdline_sha256"],
        })
    )


def validate_production_capability_prerequisite_receipt(
    value: Any,
    *,
    revision: str,
    topology: Mapping[str, Any],
    lifecycle_phase: str,
    now_unix: int | None = None,
) -> Mapping[str, Any]:
    """Validate a secret-free, topology-bound production prerequisite receipt."""

    lifecycle_phase = _require_lifecycle_phase(lifecycle_phase)
    topology = validate_production_capability_topology(topology)
    _require_browser_revision_binding(topology["browser"], revision=revision)
    raw = _exact_mapping(
        value, _RECEIPT_FIELDS, "production_prerequisite_fields_invalid"
    )
    if raw["lifecycle_phase"] != lifecycle_phase:
        raise ProductionCapabilityPrerequisiteError(
            "production_prerequisite_lifecycle_phase_invalid"
        )
    unsigned = {key: item for key, item in raw.items() if key != "receipt_sha256"}
    if (
        raw["schema"] != PREREQUISITE_SCHEMA
        or not isinstance(revision, str)
        or _REVISION.fullmatch(revision) is None
        or raw["release_revision"] != revision
        or raw["topology_identity_sha256"]
        != production_capability_topology_identity_sha256(topology)
        or raw["receipt_sha256"] != _sha256(_canonical_bytes(unsigned))
        or raw["secret_material_recorded"] is not False
        or raw["secret_digest_recorded"] is not False
        or _SHA256.fullmatch(str(raw["boot_id_sha256"])) is None
        or type(raw["observed_at_unix"]) is not int
        or raw["observed_at_unix"] <= 0
    ):
        raise ProductionCapabilityPrerequisiteError(
            "production_prerequisite_identity_invalid"
        )
    current = int(time.time()) if now_unix is None else now_unix
    if (
        type(current) is not int
        or raw["observed_at_unix"] > current + 30
        or current - raw["observed_at_unix"] > MAX_PREREQUISITE_AGE_SECONDS
    ):
        raise ProductionCapabilityPrerequisiteError(
            "production_prerequisite_clock_invalid"
        )

    services = _exact_mapping(
        raw["services"],
        frozenset({
            "phase_b",
            "routeback_edge",
            "public_connector",
            "mac_ops",
            "isolated_worker",
            "browser",
        }),
        "production_prerequisite_services_invalid",
    )
    boot_unit_file_state = (
        "disabled" if lifecycle_phase == PREREQUISITE_LIFECYCLE_STAGED else "enabled"
    )
    expected_services = {
        "phase_b": (
            PHASE_B_UNIT,
            "oneshot",
            "exited",
            False,
            boot_unit_file_state,
        ),
        "routeback_edge": (
            ROUTEBACK_EDGE_UNIT,
            "notify",
            "running",
            True,
            boot_unit_file_state,
        ),
        "public_connector": (
            PUBLIC_CONNECTOR_UNIT,
            "notify",
            "running",
            True,
            boot_unit_file_state,
        ),
        "mac_ops": (
            MAC_OPS_UNIT,
            "simple",
            "running",
            True,
            boot_unit_file_state,
        ),
        "isolated_worker": (
            ISOLATED_WORKER_SERVICE_UNIT,
            "simple",
            "running",
            True,
            "static",
        ),
        "browser": (
            BROWSER_UNIT,
            "notify",
            "running",
            True,
            boot_unit_file_state,
        ),
    }
    topology_names = {
        "phase_b": "phase_b",
        "routeback_edge": "routeback_edge",
        "public_connector": "public_connector",
        "mac_ops": "mac_ops",
        "isolated_worker": "isolated_worker",
        "browser": "browser",
    }
    fragment_fields = {
        "isolated_worker": "service_fragment_sha256",
    }
    for name, (
        unit,
        service_type,
        sub_state,
        needs_pid,
        unit_file_state,
    ) in expected_services.items():
        item = _exact_mapping(
            services[name], _SERVICE_FIELDS, "production_prerequisite_service_invalid"
        )
        main_pid = item["main_pid"]
        readiness = item["readiness_receipt_sha256"]
        effective_uid = item["effective_uid"]
        effective_gid = item["effective_gid"]
        expected_groups = item["effective_supplementary_groups"]
        expected_executable = item["unit_executable"]
        actual_process_fields = (
            "main_pid_executable",
            "main_pid_uid",
            "main_pid_gid",
            "main_pid_groups",
            "main_pid_cmdline_sha256",
            "main_pid_cgroup",
            "main_pid_mount_namespace_inode",
            "main_pid_network_namespace_inode",
        )
        persistent_process_invalid = needs_pid and (
            item["main_pid_executable"] != expected_executable
            or item["main_pid_uid"] != effective_uid
            or item["main_pid_gid"] != effective_gid
            or item["main_pid_groups"] != expected_groups
            or item["main_pid_cmdline_sha256"] != item["unit_cmdline_sha256"]
            or item["main_pid_cgroup"] != f"/system.slice/{unit}"
            or type(item["main_pid_mount_namespace_inode"]) is not int
            or item["main_pid_mount_namespace_inode"] <= 0
            or type(item["main_pid_network_namespace_inode"]) is not int
            or item["main_pid_network_namespace_inode"] <= 0
        )
        if (
            item["unit"] != unit
            or item["fragment_path"] != f"/etc/systemd/system/{unit}"
            or item["fragment_sha256"]
            != topology[topology_names[name]][
                fragment_fields.get(name, "fragment_sha256")
            ]
            or item["unit_file_state"] != unit_file_state
            or item["active_state"] != "active"
            or item["sub_state"] != sub_state
            or item["service_type"] != service_type
            or type(main_pid) is not int
            or main_pid < 0
            or (needs_pid and main_pid <= 0)
            or (not needs_pid and main_pid != 0)
            or item["drop_in_paths"] != []
            or item["need_daemon_reload"] is not False
            or not isinstance(item["effective_user"], str)
            or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]{0,63}", item["effective_user"])
            is None
            or type(effective_uid) is not int
            or effective_uid < 0
            or not isinstance(item["effective_group"], str)
            or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]{0,63}", item["effective_group"])
            is None
            or type(effective_gid) is not int
            or effective_gid < 0
            or not isinstance(expected_groups, list)
            or not expected_groups
            or any(
                type(group_id) is not int or group_id < 0
                for group_id in expected_groups
            )
            or expected_groups != sorted(set(expected_groups))
            or expected_groups != [effective_gid]
            or (needs_pid and (effective_uid <= 0 or effective_gid <= 0))
            or (
                not needs_pid
                and (
                    item["effective_user"] != "root"
                    or effective_uid != 0
                    or item["effective_group"] != "root"
                    or effective_gid != 0
                )
            )
            or not isinstance(expected_executable, str)
            or not expected_executable.startswith("/")
            or _SHA256.fullmatch(str(item["unit_cmdline_sha256"])) is None
            or item["unit_service_contract_sha256"]
            != _recorded_service_contract_sha256(item)
            or item["process_identity_matches_unit"] is not True
            or persistent_process_invalid
            or (
                not needs_pid
                and any(item[field] is not None for field in actual_process_fields)
            )
            or item["ready"] is not True
            or (
                name in {"phase_b", "routeback_edge", "public_connector"}
                and _SHA256.fullmatch(str(readiness)) is None
            )
            or (
                name in {"mac_ops", "isolated_worker", "browser"}
                and readiness is not None
            )
        ):
            raise ProductionCapabilityPrerequisiteError(
                "production_prerequisite_service_invalid"
            )

    sockets = _exact_mapping(
        raw["sockets"],
        frozenset({
            "routeback_edge",
            "public_connector",
            "mac_ops",
            "isolated_worker",
            "browser",
        }),
        "production_prerequisite_sockets_invalid",
    )
    socket_contracts = (
        ("routeback_edge", ROUTEBACK_EDGE_SOCKET_PATH, "routeback_edge", None, None),
        (
            "public_connector",
            PUBLIC_CONNECTOR_SOCKET_PATH,
            "public_connector",
            None,
            None,
        ),
        ("mac_ops", MAC_OPS_SOCKET_PATH, "mac_ops", None, None),
        (
            "isolated_worker",
            ISOLATED_WORKER_SOCKET,
            "isolated_worker",
            topology["isolated_worker"]["socket_uid"],
            topology["isolated_worker"]["socket_gid"],
        ),
        (
            "browser",
            BROWSER_SOCKET_PATH,
            "browser",
            topology["browser"]["service_uid"],
            topology["browser"]["service_gid"],
        ),
    )
    for (
        name,
        expected_path,
        service_name,
        expected_uid,
        expected_gid,
    ) in socket_contracts:
        item = _exact_mapping(
            sockets[name], _SOCKET_FIELDS, "production_prerequisite_socket_invalid"
        )
        if (
            item["path"] != str(expected_path)
            or type(item["device"]) is not int
            or item["device"] <= 0
            or type(item["inode"]) is not int
            or item["inode"] <= 0
            or type(item["owner_uid"]) is not int
            or item["owner_uid"] < 0
            or (expected_uid is not None and item["owner_uid"] != expected_uid)
            or type(item["group_gid"]) is not int
            or item["group_gid"] <= 0
            or (expected_gid is not None and item["group_gid"] != expected_gid)
            or item["mode"] != "0660"
            or item["main_pid"] != services[service_name]["main_pid"]
            or item["ready"] is not True
        ):
            raise ProductionCapabilityPrerequisiteError(
                "production_prerequisite_socket_invalid"
            )

    worker = _exact_mapping(
        raw["isolated_worker"],
        _ISOLATED_WORKER_RECEIPT_FIELDS,
        "production_prerequisite_isolated_worker_invalid",
    )
    expected_worker = topology["isolated_worker"]
    worker_socket = sockets["isolated_worker"]
    if (
        worker["socket_unit"] != expected_worker["socket_unit"]
        or worker["socket_fragment_path"]
        != f"/etc/systemd/system/{ISOLATED_WORKER_SOCKET_UNIT}"
        or worker["socket_fragment_sha256"] != expected_worker["socket_fragment_sha256"]
        or worker["socket_unit_file_state"] != boot_unit_file_state
        or worker["socket_active_state"] != "active"
        or worker["socket_sub_state"] != "listening"
        or worker["socket_drop_in_paths"] != []
        or worker["socket_need_daemon_reload"] is not False
        or worker["service_unit"] != expected_worker["service_unit"]
        or worker["service_main_pid"] != services["isolated_worker"]["main_pid"]
        or worker["config_path"] != expected_worker["config_path"]
        or worker["config_sha256"] != expected_worker["config_sha256"]
        or worker["config_uid"] != 0
        or worker["config_gid"] != expected_worker["server_gid"]
        or worker["config_mode"] != "0440"
        or worker["socket_path"] != expected_worker["socket_path"]
        or worker["socket_uid"] != expected_worker["socket_uid"]
        or worker["socket_gid"] != expected_worker["socket_gid"]
        or worker["socket_device"] != worker_socket["device"]
        or worker["socket_inode"] != worker_socket["inode"]
        or worker["socket_mode"] != worker_socket["mode"]
        or worker["bwrap_path"] != expected_worker["bwrap_path"]
        or worker["bwrap_sha256"] != expected_worker["bwrap_sha256"]
        or worker["shell_path"] != expected_worker["shell_path"]
        or worker["shell_sha256"] != expected_worker["shell_sha256"]
        or worker["ready"] is not True
    ):
        raise ProductionCapabilityPrerequisiteError(
            "production_prerequisite_isolated_worker_invalid"
        )

    browser = _exact_mapping(
        raw["browser"],
        _BROWSER_RECEIPT_FIELDS,
        "production_prerequisite_browser_invalid",
    )
    expected_browser = topology["browser"]
    browser_socket = sockets["browser"]
    if (
        any(browser[field] != expected_browser[field] for field in expected_browser)
        or browser["config_uid"] != 0
        or browser["config_gid"] != expected_browser["service_gid"]
        or browser["config_mode"] != "0440"
        or browser["socket_uid"] != expected_browser["service_uid"]
        or browser["socket_gid"] != expected_browser["service_gid"]
        or browser["socket_device"] != browser_socket["device"]
        or browser["socket_inode"] != browser_socket["inode"]
        or browser["socket_mode"] != browser_socket["mode"]
        or browser["service_main_pid"] != services["browser"]["main_pid"]
        or browser["ready"] is not True
    ):
        raise ProductionCapabilityPrerequisiteError(
            "production_prerequisite_browser_invalid"
        )

    _validate_runtime_dependency_proof(
        raw["runtime_dependencies"],
        revision=revision,
        topology=topology,
    )
    _validate_gateway_state_proof(
        raw["gateway_state"],
        topology=topology,
        revision=revision,
    )
    _validate_capability_proofs(
        raw["capability_proofs"],
        topology=topology,
        services=services,
    )

    credentials = _exact_mapping(
        raw["credentials"],
        frozenset({"api_control", "api_approval", "openai_codex"}),
        "production_prerequisite_credentials_invalid",
    )
    for name, path, owner_uid, owner_gid, refresh_capable in (
        ("api_control", API_SERVER_CREDENTIAL_PATH, 0, 0, False),
        ("api_approval", API_APPROVAL_CREDENTIAL_PATH, 0, 0, False),
        (
            "openai_codex",
            CODEX_AUTH_PATH,
            topology["gateway_identity"]["uid"],
            topology["gateway_identity"]["gid"],
            True,
        ),
    ):
        item = _exact_mapping(
            credentials[name], _LEASE_FIELDS, "production_prerequisite_lease_invalid"
        )
        allowed_modes = {"0400"} if owner_uid == 0 else {"0400", "0600"}
        minimum_size = 32 if name == "api_approval" else 1
        maximum_size = 4098 if name == "api_approval" else 1024 * 1024
        if (
            item["path"] != str(path)
            or type(item["owner_uid"]) is not int
            or item["owner_uid"] != owner_uid
            or type(item["group_gid"]) is not int
            or item["group_gid"] != owner_gid
            or item["mode"] not in allowed_modes
            or type(item["size"]) is not int
            or not minimum_size <= item["size"] <= maximum_size
            or item["regular_one_link"] is not True
            or item["usable"] is not True
            or item["refresh_capable"] is not refresh_capable
            or item["secret_material_recorded"] is not False
            or item["secret_digest_recorded"] is not False
        ):
            raise ProductionCapabilityPrerequisiteError(
                "production_prerequisite_lease_invalid"
            )
    return raw


def _read_stable_file(path: Path) -> tuple[bytes, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ProductionCapabilityPrerequisiteError(
            "production_prerequisite_unavailable"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != 0
            or before.st_gid != 0
            or stat.S_IMODE(before.st_mode) != 0o444
            or not 0 < before.st_size <= MAX_PREREQUISITE_BYTES
        ):
            raise ProductionCapabilityPrerequisiteError(
                "production_prerequisite_file_identity_invalid"
            )
        raw = os.read(descriptor, MAX_PREREQUISITE_BYTES + 1)
        after = os.fstat(descriptor)
    finally:
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
    if len(raw) != before.st_size or identity(before) != identity(after):
        raise ProductionCapabilityPrerequisiteError(
            "production_prerequisite_file_changed"
        )
    return raw, before


def _read_boot_id() -> str:
    try:
        value = BOOT_ID_PATH.read_text(encoding="ascii").strip()
    except OSError as exc:
        raise ProductionCapabilityPrerequisiteError(
            "production_boot_identity_unavailable"
        ) from exc
    if not value:
        raise ProductionCapabilityPrerequisiteError(
            "production_boot_identity_unavailable"
        )
    return value


def _read_bounded_regular(path: Path, *, maximum: int) -> tuple[bytes, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ProductionCapabilityPrerequisiteError(
            "production_live_file_unavailable"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size < 0
            or before.st_size > maximum
        ):
            raise ProductionCapabilityPrerequisiteError(
                "production_live_file_identity_invalid"
            )
        raw = os.read(descriptor, maximum + 1)
        after = os.fstat(descriptor)
    finally:
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
    if len(raw) != before.st_size or identity(before) != identity(after):
        raise ProductionCapabilityPrerequisiteError("production_live_file_changed")
    return raw, before


_SYSTEMD_SERVICE_PROPERTIES = (
    "ActiveState,SubState,Type,MainPID,FragmentPath,UnitFileState,"
    "DropInPaths,NeedDaemonReload,User,Group,SupplementaryGroups,ControlGroup"
)
_SYSTEMD_SOCKET_PROPERTIES = (
    "ActiveState,SubState,FragmentPath,UnitFileState,DropInPaths,NeedDaemonReload"
)


def _systemd_show_service(unit: str) -> dict[str, str]:
    try:
        result = subprocess.run(
            (
                "/usr/bin/systemctl",
                "show",
                f"--property={_SYSTEMD_SERVICE_PROPERTIES}",
                "--",
                unit,
            ),
            check=False,
            capture_output=True,
            timeout=5,
            env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ProductionCapabilityPrerequisiteError(
            "production_live_service_unavailable"
        ) from exc
    if result.returncode != 0 or result.stderr or len(result.stdout) > 64 * 1024:
        raise ProductionCapabilityPrerequisiteError(
            "production_live_service_unavailable"
        )
    values: dict[str, str] = {}
    try:
        for line in result.stdout.decode("ascii", errors="strict").splitlines():
            key, separator, value = line.partition("=")
            if not separator or key in values:
                raise ValueError
            values[key] = value
        int(values["MainPID"])
        for required in _SYSTEMD_SERVICE_PROPERTIES.split(","):
            if required not in values:
                raise ValueError
    except (KeyError, UnicodeError, ValueError) as exc:
        raise ProductionCapabilityPrerequisiteError(
            "production_live_service_invalid"
        ) from exc
    return values


def _systemd_show_socket_unit(unit: str) -> dict[str, str]:
    try:
        result = subprocess.run(
            (
                "/usr/bin/systemctl",
                "show",
                f"--property={_SYSTEMD_SOCKET_PROPERTIES}",
                "--",
                unit,
            ),
            check=False,
            capture_output=True,
            timeout=5,
            env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ProductionCapabilityPrerequisiteError(
            "production_live_socket_unit_unavailable"
        ) from exc
    if result.returncode != 0 or result.stderr or len(result.stdout) > 64 * 1024:
        raise ProductionCapabilityPrerequisiteError(
            "production_live_socket_unit_unavailable"
        )
    values: dict[str, str] = {}
    try:
        for line in result.stdout.decode("ascii", errors="strict").splitlines():
            key, separator, item = line.partition("=")
            if not separator or key in values:
                raise ValueError
            values[key] = item
        for required in _SYSTEMD_SOCKET_PROPERTIES.split(","):
            if required not in values:
                raise ValueError
    except (UnicodeError, ValueError) as exc:
        raise ProductionCapabilityPrerequisiteError(
            "production_live_socket_unit_invalid"
        ) from exc
    return values


def _systemd_socket_unit_observation(
    unit: str,
    *,
    lifecycle_phase: str,
) -> Mapping[str, Any]:
    lifecycle_phase = _require_lifecycle_phase(lifecycle_phase)
    expected_unit_file_state = (
        "disabled" if lifecycle_phase == PREREQUISITE_LIFECYCLE_STAGED else "enabled"
    )
    values = _systemd_show_socket_unit(unit)
    fragment_path = Path(f"/etc/systemd/system/{unit}")
    if values["FragmentPath"] != str(fragment_path):
        raise ProductionCapabilityPrerequisiteError(
            "production_live_socket_unit_invalid"
        )
    fragment, _ = _read_bounded_regular(fragment_path, maximum=1024 * 1024)
    try:
        drop_in_paths = shlex.split(values["DropInPaths"], posix=True)
    except ValueError as exc:
        raise ProductionCapabilityPrerequisiteError(
            "production_live_socket_unit_invalid"
        ) from exc
    if (
        drop_in_paths
        or values["NeedDaemonReload"] != "no"
        or values["UnitFileState"] != expected_unit_file_state
        or values["ActiveState"] != "active"
        or values["SubState"] != "listening"
        or _systemd_show_socket_unit(unit) != values
    ):
        raise ProductionCapabilityPrerequisiteError(
            "production_live_socket_unit_invalid"
        )
    return {
        "socket_unit": unit,
        "socket_fragment_path": str(fragment_path),
        "socket_fragment_sha256": _sha256(fragment),
        "socket_unit_file_state": values["UnitFileState"],
        "socket_active_state": values["ActiveState"],
        "socket_sub_state": values["SubState"],
        "socket_drop_in_paths": drop_in_paths,
        "socket_need_daemon_reload": False,
    }


def _parse_exact_service_contract(
    fragment: bytes,
) -> tuple[str, str, tuple[str, ...], tuple[str, ...]]:
    """Read the small generated [Service] identity without systemd expansion."""

    try:
        text = fragment.decode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise ProductionCapabilityPrerequisiteError(
            "production_live_service_contract_invalid"
        ) from exc
    section = ""
    user_values: list[str] = []
    group_values: list[str] = []
    exec_values: list[str] = []
    supplementary: list[str] = []
    try:
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith(("#", ";")):
                continue
            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1]
                continue
            if section != "Service":
                continue
            key, separator, value = line.partition("=")
            if not separator:
                raise ValueError
            if key == "User":
                user_values.append(value)
            elif key == "Group":
                group_values.append(value)
            elif key == "ExecStart":
                exec_values.append(value)
            elif key == "SupplementaryGroups":
                if not value:
                    supplementary.clear()
                else:
                    supplementary.extend(shlex.split(value, posix=True))
        if (
            len(user_values) != 1
            or len(group_values) != 1
            or len(exec_values) != 1
            or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]{0,63}", user_values[0])
            or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]{0,63}", group_values[0])
        ):
            raise ValueError
        argv = tuple(shlex.split(exec_values[0], posix=True))
        if (
            not argv
            or not argv[0].startswith("/")
            or argv[0].startswith(("/-", "/+", "/!", "/@", "/:"))
            or any("\x00" in item for item in argv)
            or any(
                re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]{0,63}", item) is None
                for item in supplementary
            )
        ):
            raise ValueError
    except (ValueError, OSError) as exc:
        raise ProductionCapabilityPrerequisiteError(
            "production_live_service_contract_invalid"
        ) from exc
    return user_values[0], group_values[0], tuple(supplementary), argv


def _service_identity_from_contract(
    *,
    user: str,
    group: str,
    supplementary_groups: tuple[str, ...],
    argv: tuple[str, ...],
    required_supplementary_groups: tuple[str, ...] = (),
) -> Mapping[str, Any]:
    try:
        uid = pwd.getpwnam(user).pw_uid
        gid = grp.getgrnam(group).gr_gid
        required_names = tuple(sorted(required_supplementary_groups))
        if tuple(sorted(supplementary_groups)) != required_names:
            raise OSError
        required_gids = {gid}
        required_gids.update(grp.getgrnam(item).gr_gid for item in required_names)
        supplementary_gids = set(os.getgrouplist(user, gid))
        if supplementary_gids != required_gids:
            raise OSError
        executable = os.path.realpath(argv[0])
        if not executable.startswith("/") or not Path(executable).is_file():
            raise OSError
        cmdline_sha256 = _sha256(
            b"".join(item.encode("utf-8", errors="strict") + b"\x00" for item in argv)
        )
    except (KeyError, OSError, UnicodeError, OverflowError) as exc:
        raise ProductionCapabilityPrerequisiteError(
            "production_live_service_contract_invalid"
        ) from exc
    expected = {
        "effective_user": user,
        "effective_uid": uid,
        "effective_group": group,
        "effective_gid": gid,
        "effective_supplementary_groups": sorted(supplementary_gids),
        "unit_executable": executable,
        "unit_cmdline_sha256": cmdline_sha256,
    }
    return {
        **expected,
        "unit_service_contract_sha256": _sha256(_canonical_bytes(expected)),
    }


def _read_bounded_proc_file(path: Path, *, maximum: int) -> bytes:
    try:
        value = path.read_bytes()
    except OSError as exc:
        raise ProductionCapabilityPrerequisiteError(
            "production_live_service_process_unavailable"
        ) from exc
    if not value or len(value) > maximum:
        raise ProductionCapabilityPrerequisiteError(
            "production_live_service_process_invalid"
        )
    return value


def _service_process_identity(
    *,
    unit: str,
    main_pid: int,
    control_group: str,
    expected: Mapping[str, Any],
) -> Mapping[str, Any]:
    if main_pid == 0:
        return {
            "main_pid_executable": None,
            "main_pid_uid": None,
            "main_pid_gid": None,
            "main_pid_groups": None,
            "main_pid_cmdline_sha256": None,
            "main_pid_cgroup": None,
            "main_pid_mount_namespace_inode": None,
            "main_pid_network_namespace_inode": None,
            "process_identity_matches_unit": True,
        }
    proc = Path("/proc") / str(main_pid)
    status = _read_bounded_proc_file(proc / "status", maximum=128 * 1024)
    cmdline = _read_bounded_proc_file(proc / "cmdline", maximum=128 * 1024)
    cgroup = _read_bounded_proc_file(proc / "cgroup", maximum=128 * 1024)
    try:
        status_fields: dict[str, str] = {}
        for line in status.decode("ascii", errors="strict").splitlines():
            key, separator, value = line.partition(":")
            if separator and key in {"Uid", "Gid", "Groups"}:
                if key in status_fields:
                    raise ValueError
                status_fields[key] = value.strip()
        uid_values = tuple(int(item) for item in status_fields["Uid"].split())
        gid_values = tuple(int(item) for item in status_fields["Gid"].split())
        groups = sorted({int(item) for item in status_fields["Groups"].split()})
        if (
            len(uid_values) != 4
            or len(gid_values) != 4
            or not cmdline.endswith(b"\x00")
        ):
            raise ValueError
        executable = os.path.realpath(os.readlink(proc / "exe"))
        cgroup_lines = cgroup.decode("ascii", errors="strict").splitlines()
        unified = [
            line.removeprefix("0::") for line in cgroup_lines if line.startswith("0::")
        ]
        if len(unified) != 1 or not control_group.startswith("/"):
            raise ValueError
        mount_namespace = os.stat(proc / "ns/mnt", follow_symlinks=True).st_ino
        network_namespace = os.stat(proc / "ns/net", follow_symlinks=True).st_ino
    except (KeyError, OSError, UnicodeError, ValueError) as exc:
        raise ProductionCapabilityPrerequisiteError(
            "production_live_service_process_invalid"
        ) from exc
    uid = uid_values[1]
    gid = gid_values[1]
    cmdline_sha256 = _sha256(cmdline)
    process_matches = (
        len(set(uid_values)) == 1
        and len(set(gid_values)) == 1
        and uid == expected["effective_uid"]
        and gid == expected["effective_gid"]
        and groups == expected["effective_supplementary_groups"]
        and executable == expected["unit_executable"]
        and cmdline_sha256 == expected["unit_cmdline_sha256"]
        and unified[0] == control_group
        and control_group == f"/system.slice/{unit}"
        and mount_namespace > 0
        and network_namespace > 0
    )
    return {
        "main_pid_executable": executable,
        "main_pid_uid": uid,
        "main_pid_gid": gid,
        "main_pid_groups": groups,
        "main_pid_cmdline_sha256": cmdline_sha256,
        "main_pid_cgroup": unified[0],
        "main_pid_mount_namespace_inode": mount_namespace,
        "main_pid_network_namespace_inode": network_namespace,
        "process_identity_matches_unit": process_matches,
    }


def _systemd_service_observation(
    name: str,
    unit: str,
    *,
    readiness_path: Path | None,
    required_supplementary_groups: tuple[str, ...] = (),
) -> Mapping[str, Any]:
    values = _systemd_show_service(unit)
    main_pid = int(values["MainPID"])
    expected_fragment = Path(f"/etc/systemd/system/{unit}")
    if values.get("FragmentPath") != str(expected_fragment):
        raise ProductionCapabilityPrerequisiteError("production_live_service_invalid")
    fragment, _ = _read_bounded_regular(expected_fragment, maximum=1024 * 1024)
    user, group, supplementary, argv = _parse_exact_service_contract(fragment)
    expected_identity = _service_identity_from_contract(
        user=user,
        group=group,
        supplementary_groups=supplementary,
        argv=argv,
        required_supplementary_groups=required_supplementary_groups,
    )
    try:
        drop_in_paths = shlex.split(values["DropInPaths"], posix=True)
        shown_supplementary = tuple(
            shlex.split(values["SupplementaryGroups"], posix=True)
        )
    except ValueError as exc:
        raise ProductionCapabilityPrerequisiteError(
            "production_live_service_invalid"
        ) from exc
    if (
        values["User"] != user
        or values["Group"] != group
        or shown_supplementary != supplementary
        or values["NeedDaemonReload"] not in {"yes", "no"}
    ):
        raise ProductionCapabilityPrerequisiteError(
            "production_live_service_contract_drifted"
        )
    if drop_in_paths or values["NeedDaemonReload"] != "no":
        raise ProductionCapabilityPrerequisiteError(
            "production_live_service_effective_config_invalid"
        )
    process_identity = _service_process_identity(
        unit=unit,
        main_pid=main_pid,
        control_group=values["ControlGroup"],
        expected=expected_identity,
    )
    if process_identity["process_identity_matches_unit"] is not True:
        raise ProductionCapabilityPrerequisiteError(
            "production_live_service_process_identity_invalid"
        )
    readiness_sha256: str | None = None
    if readiness_path is not None:
        readiness, _ = _read_bounded_regular(readiness_path, maximum=2 * 1024 * 1024)
        readiness_sha256 = _sha256(readiness)
    if _systemd_show_service(unit) != values:
        raise ProductionCapabilityPrerequisiteError("production_live_service_changed")
    return {
        "unit": unit,
        "fragment_path": str(expected_fragment),
        "fragment_sha256": _sha256(fragment),
        "unit_file_state": values.get("UnitFileState"),
        "active_state": values.get("ActiveState"),
        "sub_state": values.get("SubState"),
        "service_type": values.get("Type"),
        "main_pid": main_pid,
        "drop_in_paths": drop_in_paths,
        "need_daemon_reload": values["NeedDaemonReload"] == "yes",
        **expected_identity,
        **process_identity,
        "readiness_receipt_sha256": readiness_sha256,
        "ready": True,
    }


def attest_live_production_gateway_service_identity(
    *,
    expected_unit: bytes,
) -> Mapping[str, Any]:
    """Bind the calling MainPID to the exact production gateway unit.

    Dependency receipts are collected by ``ExecStartPre`` and therefore
    cannot attest the not-yet-started gateway process.  The gateway calls this
    in-process after adapters are connected but before READY.  At that point a
    ``Type=notify`` unit must still be ``activating/start`` and every effective
    identity field must match the immutable generated fragment.
    """

    if not isinstance(expected_unit, bytes) or not expected_unit:
        raise ProductionCapabilityPrerequisiteError(
            "production_gateway_unit_contract_invalid"
        )
    expected_user, expected_group, expected_supplementary, expected_argv = (
        _parse_exact_service_contract(expected_unit)
    )
    observation = _systemd_service_observation(
        "gateway",
        GATEWAY_UNIT,
        readiness_path=None,
        required_supplementary_groups=expected_supplementary,
    )
    try:
        process_groups = sorted(set(os.getgroups()) | {os.getegid()})  # windows-footgun: ok — Linux production/canary boundary
    except OSError as exc:
        raise ProductionCapabilityPrerequisiteError(
            "production_live_gateway_identity_unavailable"
        ) from exc
    if (
        observation["fragment_sha256"] != _sha256(expected_unit)
        or observation["unit_file_state"] != "enabled"
        or observation["active_state"] != "activating"
        or observation["sub_state"] != "start"
        or observation["service_type"] != "notify"
        or observation["main_pid"] != os.getpid()
        or observation["effective_user"] != expected_user
        or observation["effective_group"] != expected_group
        or observation["effective_uid"] != os.geteuid()  # windows-footgun: ok — Linux production/canary boundary
        or observation["effective_gid"] != os.getegid()  # windows-footgun: ok — Linux production/canary boundary
        or observation["effective_supplementary_groups"] != process_groups
        or observation["unit_executable"] != os.path.realpath(expected_argv[0])
        or observation["main_pid_executable"] != observation["unit_executable"]
        or observation["process_identity_matches_unit"] is not True
        or observation["drop_in_paths"] != []
        or observation["need_daemon_reload"] is not False
        or observation["readiness_receipt_sha256"] is not None
        or observation["ready"] is not True
    ):
        raise ProductionCapabilityPrerequisiteError(
            "production_live_gateway_service_identity_invalid"
        )
    return observation


def _socket_observation(
    path: Path,
    *,
    main_pid: int,
) -> Mapping[str, Any]:
    try:
        item = os.lstat(path)
    except OSError as exc:
        raise ProductionCapabilityPrerequisiteError(
            "production_live_socket_unavailable"
        ) from exc
    if not stat.S_ISSOCK(item.st_mode):
        raise ProductionCapabilityPrerequisiteError("production_live_socket_invalid")
    return {
        "path": str(path),
        "device": item.st_dev,
        "inode": item.st_ino,
        "owner_uid": item.st_uid,
        "group_gid": item.st_gid,
        "mode": f"{stat.S_IMODE(item.st_mode):04o}",
        "main_pid": main_pid,
        "ready": True,
    }


def _credential_observation(
    logical_path: Path,
    actual_path: Path,
    *,
    refresh_capable: bool,
    verifier_kind: str | None = None,
) -> Mapping[str, Any]:
    if verifier_kind is not None:
        try:
            payload, item = _read_bounded_regular(
                actual_path,
                maximum=8_192,
            )
            if verifier_kind == "bearer":
                parse_api_bearer_verifier(payload)
            elif verifier_kind == "approval":
                parse_api_approval_scrypt_verifier(payload)
            else:  # private programming contract, never config-driven.
                raise ValueError("unsupported verifier kind")
        except (
            OSError,
            TypeError,
            ValueError,
            ProductionCapabilityPrerequisiteError,
        ) as exc:
            raise ProductionCapabilityPrerequisiteError(
                "production_live_api_verifier_invalid"
            ) from exc
    else:
        try:
            item = os.lstat(actual_path)
        except OSError as exc:
            raise ProductionCapabilityPrerequisiteError(
                "production_live_credential_unavailable"
            ) from exc
    usable = os.access(actual_path, os.R_OK) and (
        not refresh_capable or os.access(actual_path, os.W_OK)
    )
    return {
        "path": str(logical_path),
        "owner_uid": item.st_uid,
        "group_gid": item.st_gid,
        "mode": f"{stat.S_IMODE(item.st_mode):04o}",
        "size": item.st_size,
        "regular_one_link": stat.S_ISREG(item.st_mode) and item.st_nlink == 1,
        "usable": usable,
        "refresh_capable": refresh_capable,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }


def _gateway_identity_kwargs(
    topology: Mapping[str, Any],
    *,
    extra_groups: tuple[int, ...] = (),
) -> dict[str, Any]:
    """Return subprocess identity arguments for the exact gateway principal."""

    uid = topology["gateway_identity"]["uid"]
    gid = topology["gateway_identity"]["gid"]
    groups = tuple(sorted(set(extra_groups)))
    if os.geteuid() == 0:  # windows-footgun: ok — Linux production/canary boundary
        return {"user": uid, "group": gid, "extra_groups": groups}
    if (
        os.geteuid() != uid  # windows-footgun: ok — Linux production/canary boundary
        or os.getegid() != gid  # windows-footgun: ok — Linux production/canary boundary
        or any(group not in os.getgroups() for group in groups)
    ):
        raise ProductionCapabilityPrerequisiteError(
            "production_gateway_probe_identity_unavailable"
        )
    return {}


def _run_gateway_process(
    arguments: tuple[str, ...],
    *,
    topology: Mapping[str, Any],
    cwd: Path,
    env: Mapping[str, str],
    timeout: int,
    extra_groups: tuple[int, ...] = (),
) -> subprocess.CompletedProcess[bytes]:
    try:
        result = subprocess.run(
            arguments,
            cwd=cwd,
            env=dict(env),
            check=False,
            capture_output=True,
            timeout=timeout,
            close_fds=True,
            **_gateway_identity_kwargs(topology, extra_groups=extra_groups),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ProductionCapabilityPrerequisiteError(
            "production_gateway_probe_execution_failed"
        ) from exc
    if (
        result.returncode != 0
        or len(result.stdout) > 2 * 1024 * 1024
        or len(result.stderr) > 2 * 1024 * 1024
    ):
        raise ProductionCapabilityPrerequisiteError(
            "production_gateway_probe_execution_failed"
        )
    return result


def _runtime_dependency_manifest(
    *, revision: str, topology: Mapping[str, Any]
) -> tuple[Mapping[str, Any], bytes]:
    release = production_release_root(revision)
    manifest_path = release / RUNTIME_DEPENDENCY_MANIFEST_RELATIVE
    raw, _ = _read_bounded_regular(manifest_path, maximum=2 * 1024 * 1024)
    try:
        manifest = json.loads(raw.decode("ascii", errors="strict"))
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise ProductionCapabilityPrerequisiteError(
            "production_runtime_dependency_manifest_invalid"
        ) from exc
    if not isinstance(manifest, Mapping) or raw != _canonical_bytes(manifest) + b"\n":
        raise ProductionCapabilityPrerequisiteError(
            "production_runtime_dependency_manifest_invalid"
        )
    unsigned = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    if (
        manifest.get("schema") != RUNTIME_DEPENDENCY_MANIFEST_SCHEMA
        or manifest.get("release_revision") != revision
        or manifest.get("secret_material_recorded") is not False
        or manifest.get("manifest_sha256") != _sha256(_canonical_bytes(unsigned))
    ):
        raise ProductionCapabilityPrerequisiteError(
            "production_runtime_dependency_manifest_invalid"
        )
    try:
        agent_browser = manifest["agent_browser"]
        chrome = manifest["chrome"]
        ddgs = manifest["python"]["distributions"]["ddgs"]
    except (KeyError, TypeError) as exc:
        raise ProductionCapabilityPrerequisiteError(
            "production_runtime_dependency_manifest_invalid"
        ) from exc
    if (
        not isinstance(agent_browser, Mapping)
        or not isinstance(chrome, Mapping)
        or not isinstance(ddgs, Mapping)
        or agent_browser.get("version") != AGENT_BROWSER_VERSION
        or agent_browser.get("config_path")
        != topology["browser"]["agent_browser_config_path"]
        or agent_browser.get("config_sha256")
        != topology["browser"]["agent_browser_config_sha256"]
        or agent_browser.get("wrapper_path") != topology["browser"]["wrapper_path"]
        or agent_browser.get("wrapper_sha256") != topology["browser"]["wrapper_sha256"]
        or agent_browser.get("native_path") != topology["browser"]["native_path"]
        or agent_browser.get("native_sha256") != topology["browser"]["native_sha256"]
        or agent_browser.get("node_path") != topology["browser"]["node_path"]
        or agent_browser.get("node_sha256") != topology["browser"]["node_sha256"]
        or agent_browser.get("node_version") != NODE_VERSION
        or chrome.get("version") != CHROME_VERSION
        or chrome.get("executable_path") != topology["browser"]["executable"]
        or chrome.get("executable_sha256") != topology["browser"]["executable_sha256"]
        or ddgs.get("version") != DDGS_VERSION
    ):
        raise ProductionCapabilityPrerequisiteError(
            "production_runtime_dependency_manifest_invalid"
        )
    return manifest, raw


def _gateway_runtime_environment(release: Path) -> dict[str, str]:
    node_bin = release / "ops/muncho/runtime/dependencies/node-linux-x64/bin"
    return {
        "HOME": str(GATEWAY_STATE_DIRECTORY),
        "HERMES_HOME": str(PRODUCTION_HOME),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "NO_PROXY": "127.0.0.1,localhost",
        "PATH": f"{node_bin}:/usr/bin:/bin",
        "PYTHONNOUSERSITE": "1",
    }


def _collect_runtime_dependency_proof(
    *,
    revision: str,
    topology: Mapping[str, Any],
) -> Mapping[str, Any]:
    release = production_release_root(revision)
    interpreter = release / ".venv/bin/python"
    verifier = release / "scripts/canary/package_production_runtime_dependencies.py"
    try:
        verified = subprocess.run(
            (
                str(interpreter),
                "-I",
                str(verifier),
                "verify",
                "--release-root",
                str(release),
                "--revision",
                revision,
            ),
            cwd=release,
            env={
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "PATH": "/usr/bin:/bin",
                "PYTHONNOUSERSITE": "1",
            },
            capture_output=True,
            check=False,
            timeout=180,
            close_fds=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ProductionCapabilityPrerequisiteError(
            "production_runtime_dependency_verification_failed"
        ) from exc
    if verified.returncode != 0 or verified.stderr or len(verified.stdout) > 64 * 1024:
        raise ProductionCapabilityPrerequisiteError(
            "production_runtime_dependency_verification_failed"
        )
    manifest, manifest_raw = _runtime_dependency_manifest(
        revision=revision,
        topology=topology,
    )
    release_environment = _gateway_runtime_environment(release)
    ddgs_result = _run_gateway_process(
        (
            str(interpreter),
            "-I",
            "-c",
            (
                "import importlib.metadata,json;import ddgs;"
                "print(json.dumps({'version':importlib.metadata.version('ddgs'),"
                "'imported':True},sort_keys=True,separators=(',',':')))"
            ),
        ),
        topology=topology,
        cwd=release,
        env=release_environment,
        timeout=15,
    )
    try:
        ddgs_smoke = json.loads(ddgs_result.stdout.decode("ascii", errors="strict"))
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise ProductionCapabilityPrerequisiteError(
            "production_ddgs_gateway_smoke_failed"
        ) from exc
    if ddgs_smoke != {"imported": True, "version": DDGS_VERSION}:
        raise ProductionCapabilityPrerequisiteError(
            "production_ddgs_gateway_smoke_failed"
        )
    agent_browser = manifest["agent_browser"]
    chrome = manifest["chrome"]
    ddgs = manifest["python"]["distributions"]["ddgs"]
    return {
        "manifest_path": str(release / RUNTIME_DEPENDENCY_MANIFEST_RELATIVE),
        "manifest_sha256": _sha256(manifest_raw),
        "agent_browser": {
            "version": AGENT_BROWSER_VERSION,
            "config_path": agent_browser["config_path"],
            "config_sha256": agent_browser["config_sha256"],
            "wrapper_path": agent_browser["wrapper_path"],
            "wrapper_sha256": agent_browser["wrapper_sha256"],
            "native_path": agent_browser["native_path"],
            "native_sha256": agent_browser["native_sha256"],
            "node_path": agent_browser["node_path"],
            "node_version": agent_browser["node_version"],
            "node_sha256": agent_browser["node_sha256"],
        },
        "chrome": {
            "version": CHROME_VERSION,
            "executable_path": chrome["executable_path"],
            "executable_sha256": chrome["executable_sha256"],
        },
        "ddgs": {
            "version": DDGS_VERSION,
            "files_sha256": ddgs["files_sha256"],
            "gateway_uid_import_smoke": True,
        },
        "ready": True,
    }


def _collect_gateway_state_proof(
    *, revision: str, topology: Mapping[str, Any]
) -> Mapping[str, Any]:
    release = production_release_root(revision)
    interpreter = release / ".venv/bin/python"
    code = (
        "import sys;sys.path.insert(0,sys.argv[1]);"
        "from gateway.production_capability_gateway_probe import main;"
        "raise SystemExit(main([sys.argv[1]]))"
    )
    result = _run_gateway_process(
        (str(interpreter), "-I", "-c", code, str(release)),
        topology=topology,
        cwd=release,
        env=_gateway_runtime_environment(release),
        timeout=60,
    )
    try:
        proof = json.loads(result.stdout.decode("ascii", errors="strict"))
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise ProductionCapabilityPrerequisiteError(
            "production_gateway_state_probe_invalid"
        ) from exc
    if (
        not isinstance(proof, Mapping)
        or result.stdout != _canonical_bytes(proof) + b"\n"
    ):
        raise ProductionCapabilityPrerequisiteError(
            "production_gateway_state_probe_invalid"
        )
    return _validate_gateway_state_proof(
        proof,
        topology=topology,
        revision=revision,
    )


def _collect_mac_ops_ping_proof(
    *,
    config: Mapping[str, Any],
    services: Mapping[str, Any],
) -> Mapping[str, Any]:
    try:
        from gateway.mac_ops_edge_client import (
            MacOpsEdgeClient,
            MacOpsEdgeClientConfig,
        )

        client_config = MacOpsEdgeClientConfig.from_mapping(config)
        nonce = secrets.token_hex(32)
        response = MacOpsEdgeClient(client_config).ping(nonce=nonce)
        result = response["result"]
        receipt = response["receipt"]
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ProductionCapabilityPrerequisiteError(
            "production_mac_ops_ping_failed"
        ) from exc
    if (
        result != {"nonce": nonce, "external_io": False}
        or receipt.get("service_identity_sha256")
        != client_config.service_identity_sha256
        or _SHA256.fullmatch(str(receipt.get("sha256"))) is None
    ):
        raise ProductionCapabilityPrerequisiteError("production_mac_ops_ping_failed")
    return {
        "main_pid": services["mac_ops"]["main_pid"],
        "service_identity_sha256": client_config.service_identity_sha256,
        "receipt_sha256": receipt["sha256"],
        "peer_main_pid_validated": True,
        "external_io": False,
        "ready": True,
    }


def _collect_execution_readiness_proofs(
    *, revision: str, topology: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Run both execution-edge probes as the exact gateway principal."""

    release = production_release_root(revision)
    interpreter = release / ".venv/bin/python"
    worker = topology["isolated_worker"]
    browser = topology["browser"]
    _require_browser_revision_binding(browser, revision=revision)
    code = (
        "import json,sys;from pathlib import Path;"
        "sys.path.insert(0,sys.argv[1]);"
        "from gateway.production_execution_readiness import "
        "attest_browser_controller_execution,attest_isolated_worker_execution;"
        "from tools.browser_controller_client import BrowserControllerClientConfig;"
        "worker=attest_isolated_worker_execution("
        "socket_path=Path(sys.argv[2]),server_uid=int(sys.argv[3]),"
        "server_gid=int(sys.argv[4]),socket_uid=int(sys.argv[5]),"
        "socket_gid=int(sys.argv[6]),revision=sys.argv[7],"
        "config_sha256=sys.argv[8],timeout_seconds=10);"
        "browser=attest_browser_controller_execution("
        "client_config=BrowserControllerClientConfig("
        "socket_path=Path(sys.argv[9]),server_uid=int(sys.argv[10]),"
        "artifact_root=Path(sys.argv[11]),connect_timeout_seconds=10,"
        "request_timeout_seconds=120),revision=sys.argv[7],"
        "config_sha256=sys.argv[12]);"
        "value={'browser_controller_command':browser,"
        "'isolated_worker_exec':worker};"
        "print(json.dumps(value,ensure_ascii=True,sort_keys=True,"
        "separators=(',',':'),allow_nan=False))"
    )
    result = _run_gateway_process(
        (
            str(interpreter),
            "-I",
            "-c",
            code,
            str(release),
            worker["socket_path"],
            str(worker["server_uid"]),
            str(worker["server_gid"]),
            str(worker["socket_uid"]),
            str(worker["socket_gid"]),
            revision,
            worker["config_sha256"],
            browser["socket_path"],
            str(browser["service_uid"]),
            str(BROWSER_ARTIFACT_PATH),
            browser["config_sha256"],
        ),
        topology=topology,
        cwd=release,
        env=_gateway_runtime_environment(release),
        timeout=180,
        extra_groups=(worker["socket_gid"], browser["service_gid"]),
    )
    try:
        proofs = json.loads(result.stdout.decode("ascii", errors="strict"))
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise ProductionCapabilityPrerequisiteError(
            "production_execution_readiness_probe_invalid"
        ) from exc
    if (
        not isinstance(proofs, Mapping)
        or set(proofs) != {"isolated_worker_exec", "browser_controller_command"}
        or result.stdout != _canonical_bytes(proofs) + b"\n"
    ):
        raise ProductionCapabilityPrerequisiteError(
            "production_execution_readiness_probe_invalid"
        )
    return proofs


def collect_current_production_capability_prerequisite_receipt(
    *,
    revision: str,
    topology: Mapping[str, Any],
    lifecycle_phase: str,
    mac_ops_edge_config: Mapping[str, Any] | None = None,
    now_unix: int | None = None,
) -> Mapping[str, Any]:
    """Collect current non-secret prerequisite identities from the live host."""

    lifecycle_phase = _require_lifecycle_phase(lifecycle_phase)
    topology = validate_production_capability_topology(topology)
    _require_browser_revision_binding(topology["browser"], revision=revision)
    current = int(time.time()) if now_unix is None else now_unix
    worker_topology = topology["isolated_worker"]
    browser_topology = topology["browser"]

    worker_socket_unit = _systemd_socket_unit_observation(
        ISOLATED_WORKER_SOCKET_UNIT,
        lifecycle_phase=lifecycle_phase,
    )
    worker_service_fragment, _ = _read_bounded_regular(
        Path(f"/etc/systemd/system/{ISOLATED_WORKER_SERVICE_UNIT}"),
        maximum=1024 * 1024,
    )
    worker_config, worker_config_state = _read_bounded_regular(
        ISOLATED_WORKER_CONFIG,
        maximum=2 * 1024 * 1024,
    )
    bwrap, bwrap_state = _read_bounded_regular(
        BWRAP_PATH,
        maximum=64 * 1024 * 1024,
    )
    shell, shell_state = _read_bounded_regular(
        SHELL_PATH,
        maximum=16 * 1024 * 1024,
    )
    browser_config, browser_config_state = _read_bounded_regular(
        BROWSER_CONFIG_PATH,
        maximum=2 * 1024 * 1024,
    )
    if (
        worker_socket_unit["socket_fragment_sha256"]
        != worker_topology["socket_fragment_sha256"]
        or _sha256(worker_service_fragment)
        != worker_topology["service_fragment_sha256"]
        or _sha256(worker_config) != worker_topology["config_sha256"]
        or worker_config_state.st_uid != 0
        or worker_config_state.st_gid != worker_topology["server_gid"]
        or stat.S_IMODE(worker_config_state.st_mode) != 0o440
        or _sha256(bwrap) != worker_topology["bwrap_sha256"]
        or bwrap_state.st_uid != 0
        or stat.S_IMODE(bwrap_state.st_mode) & 0o022
        or not stat.S_IMODE(bwrap_state.st_mode) & 0o111
        or _sha256(shell) != worker_topology["shell_sha256"]
        or shell_state.st_uid != 0
        or stat.S_IMODE(shell_state.st_mode) & 0o022
        or not stat.S_IMODE(shell_state.st_mode) & 0o111
        or _sha256(browser_config) != browser_topology["config_sha256"]
        or browser_config_state.st_uid != 0
        or browser_config_state.st_gid != browser_topology["service_gid"]
        or stat.S_IMODE(browser_config_state.st_mode) != 0o440
    ):
        raise ProductionCapabilityPrerequisiteError(
            "production_execution_boundary_identity_invalid"
        )

    browser_service = _systemd_service_observation(
        "browser", BROWSER_UNIT, readiness_path=None
    )
    if (
        browser_service["fragment_sha256"] != browser_topology["fragment_sha256"]
        or browser_service["effective_uid"] != browser_topology["service_uid"]
        or browser_service["effective_gid"] != browser_topology["service_gid"]
    ):
        raise ProductionCapabilityPrerequisiteError(
            "production_live_browser_controller_identity_invalid"
        )
    browser_socket_before = _socket_observation(
        BROWSER_SOCKET_PATH,
        main_pid=browser_service["main_pid"],
    )
    worker_socket_before = _socket_observation(
        ISOLATED_WORKER_SOCKET,
        main_pid=0,
    )
    if (
        browser_socket_before["owner_uid"] != browser_topology["service_uid"]
        or browser_socket_before["group_gid"] != browser_topology["service_gid"]
        or browser_socket_before["mode"] != "0660"
        or worker_socket_before["owner_uid"] != worker_topology["socket_uid"]
        or worker_socket_before["group_gid"] != worker_topology["socket_gid"]
        or worker_socket_before["mode"] != "0660"
    ):
        raise ProductionCapabilityPrerequisiteError(
            "production_execution_boundary_socket_invalid"
        )

    execution_proofs = _collect_execution_readiness_proofs(
        revision=revision,
        topology=topology,
    )
    isolated_worker_service = _systemd_service_observation(
        "isolated_worker",
        ISOLATED_WORKER_SERVICE_UNIT,
        readiness_path=None,
    )
    if (
        isolated_worker_service["fragment_sha256"]
        != worker_topology["service_fragment_sha256"]
        or isolated_worker_service["effective_uid"] != worker_topology["server_uid"]
        or isolated_worker_service["effective_gid"] != worker_topology["server_gid"]
    ):
        raise ProductionCapabilityPrerequisiteError(
            "production_live_isolated_worker_identity_invalid"
        )
    browser_service_after = _systemd_service_observation(
        "browser", BROWSER_UNIT, readiness_path=None
    )
    if browser_service_after != browser_service:
        raise ProductionCapabilityPrerequisiteError(
            "production_live_browser_controller_changed"
        )
    services = {
        "phase_b": _systemd_service_observation(
            "phase_b",
            PHASE_B_UNIT,
            readiness_path=PHASE_B_RECEIPT_PATH,
        ),
        "routeback_edge": _systemd_service_observation(
            "routeback_edge",
            ROUTEBACK_EDGE_UNIT,
            readiness_path=ROUTEBACK_EDGE_READINESS_PATH,
        ),
        "public_connector": _systemd_service_observation(
            "public_connector",
            PUBLIC_CONNECTOR_UNIT,
            readiness_path=PUBLIC_CONNECTOR_READINESS_PATH,
        ),
        "mac_ops": _systemd_service_observation(
            "mac_ops", MAC_OPS_UNIT, readiness_path=None
        ),
        "isolated_worker": isolated_worker_service,
        "browser": browser_service,
    }
    credential_directory = os.environ.get("CREDENTIALS_DIRECTORY", "")
    api_actual = API_SERVER_CREDENTIAL_PATH
    approval_actual = API_APPROVAL_CREDENTIAL_PATH
    if os.geteuid() != 0:  # windows-footgun: ok — Linux production/canary boundary
        if not credential_directory:
            raise ProductionCapabilityPrerequisiteError(
                "production_live_api_credential_unavailable"
            )
        api_actual = Path(credential_directory) / API_SERVER_CREDENTIAL_NAME
        approval_actual = Path(credential_directory) / API_APPROVAL_CREDENTIAL_NAME
    if mac_ops_edge_config is None:
        config_raw, _ = _read_bounded_regular(
            PRODUCTION_CONFIG_PATH,
            maximum=2 * 1024 * 1024,
        )
        try:
            from gateway.production_model_sovereignty_runtime import (
                load_strict_production_config,
            )

            production_config = load_strict_production_config(config_raw)
            mac_ops_edge_config = production_config["mac_ops_edge"]
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            raise ProductionCapabilityPrerequisiteError(
                "production_mac_ops_config_invalid"
            ) from exc
    runtime_dependencies = _collect_runtime_dependency_proof(
        revision=revision,
        topology=topology,
    )
    gateway_state = _collect_gateway_state_proof(
        revision=revision,
        topology=topology,
    )
    capability_proofs = {
        "mac_ops_ping": _collect_mac_ops_ping_proof(
            config=mac_ops_edge_config,
            services=services,
        ),
        **execution_proofs,
    }
    socket_observations = {
        "routeback_edge": _socket_observation(
            ROUTEBACK_EDGE_SOCKET_PATH,
            main_pid=services["routeback_edge"]["main_pid"],
        ),
        "public_connector": _socket_observation(
            PUBLIC_CONNECTOR_SOCKET_PATH,
            main_pid=services["public_connector"]["main_pid"],
        ),
        "mac_ops": _socket_observation(
            MAC_OPS_SOCKET_PATH,
            main_pid=services["mac_ops"]["main_pid"],
        ),
        "isolated_worker": _socket_observation(
            ISOLATED_WORKER_SOCKET,
            main_pid=services["isolated_worker"]["main_pid"],
        ),
        "browser": _socket_observation(
            BROWSER_SOCKET_PATH,
            main_pid=services["browser"]["main_pid"],
        ),
    }
    worker_socket = socket_observations["isolated_worker"]
    browser_socket = socket_observations["browser"]
    for before, after in (
        (worker_socket_before, worker_socket),
        (browser_socket_before, browser_socket),
    ):
        for field in (
            "path",
            "device",
            "inode",
            "owner_uid",
            "group_gid",
            "mode",
            "ready",
        ):
            if before[field] != after[field]:
                raise ProductionCapabilityPrerequisiteError(
                    "production_execution_boundary_socket_changed"
                )
    unsigned = {
        "schema": PREREQUISITE_SCHEMA,
        "release_revision": revision,
        "lifecycle_phase": lifecycle_phase,
        "topology_identity_sha256": production_capability_topology_identity_sha256(
            topology
        ),
        "boot_id_sha256": _sha256(_read_boot_id().encode("ascii")),
        "observed_at_unix": current,
        "services": services,
        "sockets": socket_observations,
        "isolated_worker": {
            **worker_socket_unit,
            "service_unit": worker_topology["service_unit"],
            "service_main_pid": services["isolated_worker"]["main_pid"],
            "config_path": worker_topology["config_path"],
            "config_sha256": _sha256(worker_config),
            "config_uid": worker_config_state.st_uid,
            "config_gid": worker_config_state.st_gid,
            "config_mode": f"{stat.S_IMODE(worker_config_state.st_mode):04o}",
            "socket_path": worker_socket["path"],
            "socket_uid": worker_socket["owner_uid"],
            "socket_gid": worker_socket["group_gid"],
            "socket_device": worker_socket["device"],
            "socket_inode": worker_socket["inode"],
            "socket_mode": worker_socket["mode"],
            "bwrap_path": worker_topology["bwrap_path"],
            "bwrap_sha256": _sha256(bwrap),
            "shell_path": worker_topology["shell_path"],
            "shell_sha256": _sha256(shell),
            "ready": True,
        },
        "browser": {
            **browser_topology,
            "config_uid": browser_config_state.st_uid,
            "config_gid": browser_config_state.st_gid,
            "config_mode": f"{stat.S_IMODE(browser_config_state.st_mode):04o}",
            "socket_uid": browser_socket["owner_uid"],
            "socket_gid": browser_socket["group_gid"],
            "socket_device": browser_socket["device"],
            "socket_inode": browser_socket["inode"],
            "socket_mode": browser_socket["mode"],
            "service_main_pid": services["browser"]["main_pid"],
            "ready": True,
        },
        "runtime_dependencies": runtime_dependencies,
        "gateway_state": gateway_state,
        "capability_proofs": capability_proofs,
        "credentials": {
            "api_control": _credential_observation(
                API_SERVER_CREDENTIAL_PATH,
                api_actual,
                refresh_capable=False,
                verifier_kind="bearer",
            ),
            "api_approval": _credential_observation(
                API_APPROVAL_CREDENTIAL_PATH,
                approval_actual,
                refresh_capable=False,
                verifier_kind="approval",
            ),
            "openai_codex": _credential_observation(
                CODEX_AUTH_PATH,
                CODEX_AUTH_PATH,
                refresh_capable=True,
            ),
        },
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    receipt = {**unsigned, "receipt_sha256": _sha256(_canonical_bytes(unsigned))}
    return validate_production_capability_prerequisite_receipt(
        receipt,
        revision=revision,
        topology=topology,
        lifecycle_phase=lifecycle_phase,
        now_unix=current,
    )


def validate_live_production_capability_prerequisites(
    current: Mapping[str, Any],
    *,
    signed: Mapping[str, Any],
    topology: Mapping[str, Any],
    lifecycle_phase: str,
) -> None:
    """Compare live identities to the signed receipt without pinning token bytes."""

    lifecycle_phase = _require_lifecycle_phase(lifecycle_phase)
    if (
        current.get("lifecycle_phase") != lifecycle_phase
        or signed.get("lifecycle_phase") != lifecycle_phase
    ):
        raise ProductionCapabilityPrerequisiteError(
            "production_live_lifecycle_phase_drifted"
        )
    for field in (
        "services",
        "sockets",
        "isolated_worker",
        "browser",
        "runtime_dependencies",
        "gateway_state",
    ):
        if current[field] != signed[field]:
            raise ProductionCapabilityPrerequisiteError(
                f"production_live_{field}_drifted"
            )
    current_capabilities = current["capability_proofs"]
    signed_capabilities = signed["capability_proofs"]
    for name in ("isolated_worker_exec", "browser_controller_command"):
        if current_capabilities[name] != signed_capabilities[name]:
            raise ProductionCapabilityPrerequisiteError(
                "production_live_capability_proofs_drifted"
            )
    current_mac = current_capabilities["mac_ops_ping"]
    signed_mac = signed_capabilities["mac_ops_ping"]
    for field in (
        "main_pid",
        "service_identity_sha256",
        "peer_main_pid_validated",
        "external_io",
        "ready",
    ):
        if current_mac[field] != signed_mac[field]:
            raise ProductionCapabilityPrerequisiteError(
                "production_live_capability_proofs_drifted"
            )
    current_credentials = current["credentials"]
    signed_credentials = signed["credentials"]
    for name in ("api_control", "api_approval", "openai_codex"):
        item = current_credentials[name]
        expected = signed_credentials[name]
        if (
            item["path"] != expected["path"]
            or item["regular_one_link"] is not True
            or item["usable"] is not True
            or item["refresh_capable"] is not expected["refresh_capable"]
            or item["secret_material_recorded"] is not False
            or item["secret_digest_recorded"] is not False
            or not 0 < item["size"] <= 1024 * 1024
        ):
            raise ProductionCapabilityPrerequisiteError(
                "production_live_credential_drifted"
            )
        if name == "openai_codex" and (
            item["owner_uid"] != topology["gateway_identity"]["uid"]
            or item["group_gid"] != topology["gateway_identity"]["gid"]
            or item["mode"] not in {"0400", "0600"}
        ):
            raise ProductionCapabilityPrerequisiteError(
                "production_live_codex_credential_owner_drifted"
            )
    if os.geteuid() != 0 and (  # windows-footgun: ok — Linux production/canary boundary
        os.geteuid() != topology["gateway_identity"]["uid"]  # windows-footgun: ok — Linux production/canary boundary
        or os.getegid() != topology["gateway_identity"]["gid"]  # windows-footgun: ok — Linux production/canary boundary
    ):
        raise ProductionCapabilityPrerequisiteError(
            "production_live_gateway_identity_drifted"
        )


def load_production_capability_prerequisite_receipt(
    *,
    revision: str,
    topology: Mapping[str, Any],
    lifecycle_phase: str,
    now_unix: int | None = None,
    reobserve_live: bool = True,
) -> Mapping[str, Any]:
    """Load the exact config-bound root collector receipt without mutation."""

    lifecycle_phase = _require_lifecycle_phase(lifecycle_phase)
    topology = validate_production_capability_topology(topology)
    raw, _identity = _read_stable_file(PREREQUISITE_PATH)
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise ProductionCapabilityPrerequisiteError(
            "production_prerequisite_json_invalid"
        ) from exc
    if not isinstance(value, Mapping) or raw != _canonical_bytes(value):
        raise ProductionCapabilityPrerequisiteError(
            "production_prerequisite_json_not_canonical"
        )
    receipt = validate_production_capability_prerequisite_receipt(
        value,
        revision=revision,
        topology=topology,
        lifecycle_phase=lifecycle_phase,
        now_unix=now_unix,
    )
    boot_id = _read_boot_id()
    if not boot_id or _sha256(boot_id.encode("ascii")) != receipt["boot_id_sha256"]:
        raise ProductionCapabilityPrerequisiteError(
            "production_prerequisite_boot_identity_drifted"
        )
    if reobserve_live:
        current = collect_current_production_capability_prerequisite_receipt(
            revision=revision,
            topology=topology,
            lifecycle_phase=lifecycle_phase,
            now_unix=now_unix,
        )
        validate_live_production_capability_prerequisites(
            current,
            signed=receipt,
            topology=topology,
            lifecycle_phase=lifecycle_phase,
        )
    return receipt


def packaged_prerequisite_contract() -> Mapping[str, Any]:
    """Return the non-secret contract embedded into cutover artifacts."""

    return {
        "schema": PREREQUISITE_SCHEMA,
        "topology_schema": TOPOLOGY_SCHEMA,
        "path": str(PREREQUISITE_PATH),
        "boot_id_path": str(BOOT_ID_PATH),
        "uid": 0,
        "gid": 0,
        "mode": 0o444,
        "maximum_bytes": MAX_PREREQUISITE_BYTES,
        "maximum_age_seconds": MAX_PREREQUISITE_AGE_SECONDS,
        "lifecycle_phases": sorted(PREREQUISITE_LIFECYCLE_PHASES),
        "fields": sorted(_RECEIPT_FIELDS),
        "topology_fields": sorted(_TOPOLOGY_FIELDS),
        "isolated_worker_topology_fields": sorted(_ISOLATED_WORKER_TOPOLOGY_FIELDS),
        "browser_topology_fields": sorted(_BROWSER_TOPOLOGY_FIELDS),
        "isolated_worker_receipt_fields": sorted(_ISOLATED_WORKER_RECEIPT_FIELDS),
        "browser_receipt_fields": sorted(_BROWSER_RECEIPT_FIELDS),
        "collector_entrypoint": ("gateway.production_capability_prerequisites:collect"),
        "cutover_acceptance_schema": (
            "muncho-production-capability-prerequisite-acceptance.v3"
        ),
        "isolated_canary_goal_terminal_schema": (
            ISOLATED_CANARY_GOAL_TERMINAL_SCHEMA
        ),
        "isolated_canary_signed_workspace_required": True,
        "isolation_equivalence_projection_required": True,
        "canary_zero_production_mutation_observation_required": True,
        "pre_db_zero_canonical_database_mutation_observation_required": True,
        "post_staging_service_mutation_observation_required": True,
        "exact_run_release_fixture_evidence_binding_required": True,
        "production_owner_approval_binding_required": True,
        "dynamic_receipt": True,
        "owner_approval_binds_collector_criteria": True,
        "live_reobservation_required": True,
        "atomic_root_install": True,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }


def packaged_prerequisite_contract_sha256() -> str:
    return _sha256(_canonical_bytes(packaged_prerequisite_contract()))


def _atomic_install_collected_receipt(receipt: Mapping[str, Any]) -> None:
    if not sys_platform_is_linux_root():
        raise ProductionCapabilityPrerequisiteError(
            "production_collector_requires_linux_root"
        )
    parent = PREREQUISITE_PATH.parent
    try:
        parent.mkdir(parents=True, mode=0o755, exist_ok=True)
        parent_state = os.lstat(parent)
    except OSError as exc:
        raise ProductionCapabilityPrerequisiteError(
            "production_collector_directory_unavailable"
        ) from exc
    if (
        not stat.S_ISDIR(parent_state.st_mode)
        or stat.S_ISLNK(parent_state.st_mode)
        or parent_state.st_uid != 0
        or parent_state.st_gid != 0
        or stat.S_IMODE(parent_state.st_mode) != 0o755
    ):
        raise ProductionCapabilityPrerequisiteError(
            "production_collector_directory_identity_invalid"
        )
    payload = _canonical_bytes(receipt)
    temporary = parent / f".{PREREQUISITE_PATH.name}.{secrets.token_hex(8)}"
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = -1
    try:
        descriptor = os.open(temporary, flags, 0o444)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise ProductionCapabilityPrerequisiteError(
                    "production_collector_write_failed"
                )
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary, PREREQUISITE_PATH)
        directory = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except OSError as exc:
        raise ProductionCapabilityPrerequisiteError(
            "production_collector_write_failed"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def sys_platform_is_linux_root() -> bool:
    import sys

    return (
        sys.platform.startswith("linux")
        and hasattr(os, "geteuid")
        and os.geteuid() == 0  # windows-footgun: ok — Linux production/canary boundary
    )


def collect_and_install_from_production_config(
    *,
    revision: str,
    config_sha256: str,
    lifecycle_phase: str,
    now_unix: int | None = None,
) -> Mapping[str, Any]:
    """Root collector entry point used by the SHA-bound gateway unit."""

    lifecycle_phase = _require_lifecycle_phase(lifecycle_phase)
    if (
        _REVISION.fullmatch(revision or "") is None
        or _SHA256.fullmatch(config_sha256 or "") is None
    ):
        raise ProductionCapabilityPrerequisiteError(
            "production_collector_identity_invalid"
        )
    raw, _ = _read_bounded_regular(PRODUCTION_CONFIG_PATH, maximum=2 * 1024 * 1024)
    if _sha256(raw) != config_sha256:
        raise ProductionCapabilityPrerequisiteError(
            "production_collector_config_digest_mismatch"
        )
    try:
        from gateway.production_model_sovereignty_runtime import (
            load_strict_production_config,
            validate_production_gateway_config,
        )

        config = load_strict_production_config(raw)
        validate_production_gateway_config(config)
        topology = config["production_capabilities"]
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise ProductionCapabilityPrerequisiteError(
            "production_collector_config_invalid"
        ) from exc
    receipt = collect_current_production_capability_prerequisite_receipt(
        revision=revision,
        topology=topology,
        lifecycle_phase=lifecycle_phase,
        mac_ops_edge_config=config["mac_ops_edge"],
        now_unix=now_unix,
    )
    _atomic_install_collected_receipt(receipt)
    return {
        "schema": receipt["schema"],
        "release_revision": revision,
        "lifecycle_phase": receipt["lifecycle_phase"],
        "receipt_sha256": receipt["receipt_sha256"],
        "topology_identity_sha256": receipt["topology_identity_sha256"],
        "observed_at_unix": receipt["observed_at_unix"],
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Collect exact Cloud Muncho production prerequisites"
    )
    parser.add_argument("command", choices=("collect",))
    parser.add_argument("--revision", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument(
        "--lifecycle-phase",
        required=True,
        choices=sorted(PREREQUISITE_LIFECYCLE_PHASES),
    )
    args = parser.parse_args(argv)
    try:
        result = collect_and_install_from_production_config(
            revision=args.revision,
            config_sha256=args.config_sha256,
            lifecycle_phase=args.lifecycle_phase,
        )
    except (OSError, ProductionCapabilityPrerequisiteError):
        return 2
    print(_canonical_bytes(result).decode("ascii"))
    return 0


__all__ = [
    "API_APPROVAL_CREDENTIAL_NAME",
    "API_APPROVAL_CREDENTIAL_PATH",
    "API_SERVER_CREDENTIAL_NAME",
    "API_SERVER_CREDENTIAL_PATH",
    "API_SERVER_HOST",
    "API_SERVER_PORT",
    "BROWSER_ARTIFACT_PATH",
    "BROWSER_CONFIG_PATH",
    "BROWSER_SOCKET_PATH",
    "BROWSER_STATE_PATH",
    "BROWSER_UNIT",
    "CHROME_VERSION",
    "CODEX_AUTH_PATH",
    "PRODUCTION_CONFIG_PATH",
    "FIRST_WAVE_TOOLSETS",
    "GATEWAY_UNIT",
    "ISOLATED_CANARY_GOAL_TERMINAL_SCHEMA",
    "MAC_OPS_UNIT",
    "MAX_PREREQUISITE_BYTES",
    "MAX_PREREQUISITE_AGE_SECONDS",
    "PHASE_B_UNIT",
    "PREREQUISITE_PATH",
    "PREREQUISITE_SCHEMA",
    "PREREQUISITE_LIFECYCLE_COMMITTED",
    "PREREQUISITE_LIFECYCLE_PHASES",
    "PREREQUISITE_LIFECYCLE_STAGED",
    "PUBLIC_CONNECTOR_UNIT",
    "PUBLIC_CONNECTOR_CONFIG_PATH",
    "PUBLIC_CONNECTOR_SOCKET_PATH",
    "PUBLIC_CONNECTOR_CREDENTIAL_PATH",
    "PUBLIC_CONNECTOR_READINESS_PATH",
    "ProductionCapabilityPrerequisiteError",
    "ROUTEBACK_EDGE_UNIT",
    "TOPOLOGY_SCHEMA",
    "WRITER_UNIT",
    "attest_live_production_gateway_service_identity",
    "load_production_capability_prerequisite_receipt",
    "collect_and_install_from_production_config",
    "collect_current_production_capability_prerequisite_receipt",
    "packaged_prerequisite_contract",
    "packaged_prerequisite_contract_sha256",
    "production_agent_browser_config",
    "production_browser_executable",
    "production_browser_native",
    "production_browser_node",
    "production_browser_wrapper",
    "production_capability_topology_identity_sha256",
    "production_release_root",
    "validate_production_capability_prerequisite_receipt",
    "validate_production_capability_topology",
    "validate_live_production_capability_prerequisites",
]


if __name__ == "__main__":
    raise SystemExit(_main())

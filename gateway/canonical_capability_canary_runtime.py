#!/usr/bin/env python3
"""Production-shaped Muncho capability-canary runtime boundary.

This module is intentionally mechanical.  It does not interpret a task,
select a tool, choose a route, or manufacture an approval.  It packages one
reviewed gateway surface around the already sealed full-canary release and
provides bounded stdin-only leases for the six credentials needed by that
surface:

* an access-token-only OpenAI Codex lease for the unprivileged gateway; and
* a GitLab environment lease owned only by the privileged Mac operations edge;
* the generated API control key;
* the root-owned Bitrix operational-edge webhook URL;
* the canonical Discord route-back token; and
* the isolated public-session Discord connector token.

The normal :mod:`gateway.run` loop remains the semantic authority.  Terminal
work crosses one authenticated AF_UNIX boundary into a credential-free,
network-free bubblewrap worker.  Browser work crosses a separate AF_UNIX
boundary into a dedicated controller that alone can see the release-local
Node/agent-browser/Chrome executables.  No Docker socket, container-root
authority, CDP endpoint, or browser executable is exposed to the gateway.  The
gateway receives neither the Discord token nor the Mac-edge GitLab credential.

Cloud/IAM creation, deployment, service enablement, and evidence semantics are
deliberately outside this module.  The owner launcher may stream already
authorized opaque credentials over stdin, and a separately approved lifecycle
may install/start the rendered units.  Receipts never contain or hash secret
values.
"""

from __future__ import annotations

import argparse
import base64
import copy
import fcntl
import grp
import hashlib
import io
import json
import os
import pwd
import re
import stat
import struct
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO, Callable, Mapping, Sequence

import yaml

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from gateway.browser_controller import (
    CONFIG_SCHEMA as BROWSER_CONTROLLER_CONFIG_SCHEMA,
    BrowserControllerConfig,
)
from gateway.discord_edge_protocol import ed25519_public_key_id
from gateway.operational_edge_assets import (
    OperationalEdgeAssetError,
    verify_packaged_operational_assets,
)
from gateway.operational_edge_units import (
    _service_config as _render_bitrix_service_config,
    _service_unit as _render_bitrix_service_unit,
)
from gateway.canonical_capability_canary_producers import (
    ENDPOINT_ROLES as CAPABILITY_PRODUCER_ROLES,
    PRODUCER_SERVICE_UNITS as CAPABILITY_PRODUCER_SERVICE_UNITS,
)
from gateway.support_ops_team_registry import (
    SKYVISION_LOCKED_NONPUBLIC_CHANNEL_IDS,
)

from gateway.canonical_full_canary_runtime import (
    API_SERVER_CREDENTIAL_NAME,
    DEFAULT_API_SERVER_CONTROL_KEY,
    DEFAULT_EDGE_RECEIPT_PRIVATE_KEY,
    DEFAULT_EDGE_TOKEN_DIRECTORY,
    DEFAULT_EDGE_TOKEN_PATH,
    DEFAULT_EDGE_UNIT_PATH,
    DEFAULT_DISABLED_MANAGED_SCOPE,
    DEFAULT_E2E_FIXTURE,
    DEFAULT_GATEWAY_CA_BUNDLE,
    DEFAULT_GATEWAY_READINESS_PATH,
    DEFAULT_OBSERVER_CONFIG,
    DEFAULT_PHASE_B_READINESS_UNIT_PATH,
    DEFAULT_WRITER_CAPABILITY_PRIVATE_KEY,
    DEFAULT_WRITER_RUNTIME_ATTESTATION_PATH,
    DEFAULT_WRITER_UNIT_PATH,
    EDGE_UNIT_NAME,
    GATEWAY_UNIT_NAME,
    PHASE_B_READINESS_UNIT_NAME,
    SYSTEMCTL,
    SYSTEMD_ANALYZE,
    SYSTEMD_TMPFILES,
    DEFAULT_TMPFILES_PATH,
    WRITER_UNIT_NAME,
    Command,
    FullCanaryLifecycle,
    FullCanaryOwnerApproval,
    FullCanaryPlan,
    _atomic_install_payload,
    _await_plugin_readiness,
    _api_loopback_listener_identity,
    _canonical_bytes,
    _common_hardening,
    _decode_json,
    _install_plan_artifacts,
    _lifecycle_lock,
    _readiness_receipt,
    _require_root_linux,
    _service_identity_receipts,
    _ensure_root_directory,
    _write_exclusive_bytes,
    _validate_edge_collector_gate,
    _validate_secret_source_metadata,
    _validate_artifact_source,
    _validate_inert_gateway_paths,
    _validate_writer_config,
    _fixed_environment,
    _GATEWAY_SYSTEMD_UNSET_ENVIRONMENT_NAMES,
    _read_stable_file,
    _run_checked,
    _runner,
    _sha256_bytes,
    _sha256_json,
    _validate_release_manifest,
    collect_full_canary_preflight,
    collect_service_state,
    evaluate_service_states,
    load_collector_readiness,
    load_full_canary_approval,
    load_full_canary_plan,
    materialize_observer_config,
    edge_start_command,
    phase_b_readiness_start_command,
    post_collector_start_commands,
    _await_collector_readiness,
    readiness_receipt_sha256,
    validate_dedicated_canary_host,
)
from gateway.mac_ops_edge_client import (
    DEFAULT_SERVICE_UNIT as MAC_OPS_UNIT_NAME,
    MacOpsEdgeClientConfig,
)
from gateway.discord_connector_bootstrap import (
    DEFAULT_CONFIG_PATH as DEFAULT_CONNECTOR_CONFIG,
    DEFAULT_READINESS_PATH as DEFAULT_CONNECTOR_READINESS,
    READINESS_SCHEMA as CONNECTOR_READINESS_SCHEMA,
    load_config as load_connector_config,
    load_readiness_receipt as load_connector_readiness,
)
from gateway.discord_connector_service import (
    DEFAULT_DISCORD_CONNECTOR_JOURNAL,
    DEFAULT_DISCORD_CONNECTOR_SOCKET,
    DEFAULT_DISCORD_CONNECTOR_UNIT,
    DurableDiscordConnectorJournal,
)
from gateway.discord_history_authority import (
    CANARY_HISTORY_READER_SERVICE_UNIT,
    CANARY_HISTORY_READER_SERVICE_USER,
    CANARY_REQUESTER_USER_ID,
)
from gateway.discord_rest_edge import DiscordRestEdgeAdapter
from gateway.mac_ops_edge_service import (
    CONFIG_SCHEMA as MAC_OPS_CONFIG_SCHEMA,
    DEFAULT_CONFIG_PATH as DEFAULT_MAC_OPS_CONFIG,
    DEFAULT_PROJECT_ID as MAC_OPS_PROJECT_ID,
    DEFAULT_SOCKET_PATH as DEFAULT_MAC_OPS_SOCKET,
    _parse_secret_env,
)
from gateway.production_capability_prerequisites import FIRST_WAVE_TOOLSETS
from gateway.production_capability_prerequisites import (
    BROWSER_ARTIFACT_PATH,
    BROWSER_CONFIG_PATH,
    BROWSER_SOCKET_PATH,
    BROWSER_STATE_PATH,
)
from gateway.production_capability_units import (
    BROWSER_COMMAND_TIMEOUT_SECONDS,
    BROWSER_DNS_ALLOW,
    BROWSER_IDLE_TIMEOUT_SECONDS,
    BROWSER_MAX_CONNECTIONS,
    BROWSER_MAX_SESSIONS,
    BROWSER_NETWORK_DENY_RANGES,
    BROWSER_RESOLV_CONF_PATH,
    BROWSER_SESSION_QUOTA_BYTES,
    BROWSER_SESSION_QUOTA_ENTRIES,
)
from gateway.production_execution_readiness import (
    BROWSER_RECEIPT_SCHEMA,
    WORKER_RECEIPT_SCHEMA,
    attest_browser_controller_execution,
    attest_isolated_worker_execution,
)
from gateway.production_runtime_dependencies import (
    AGENT_BROWSER_CONFIG,
    AGENT_BROWSER_CONFIG_BYTES,
    AGENT_BROWSER_NATIVE,
    AGENT_BROWSER_WRAPPER,
    AGENT_BROWSER_VERSION as RELEASE_AGENT_BROWSER_VERSION,
    CHROME_EXECUTABLE,
    CHROME_VERSION as RELEASE_CHROME_VERSION,
    DDGS_LOCKED_DISTRIBUTIONS as RELEASE_DDGS_DISTRIBUTIONS,
    MANIFEST_RELATIVE_PATH as RUNTIME_DEPENDENCY_MANIFEST_RELATIVE,
    MANIFEST_SCHEMA as RUNTIME_DEPENDENCY_MANIFEST_SCHEMA,
    NODE_EXECUTABLE,
    NODE_ROOT as RELEASE_NODE_ROOT,
    verify_manifest as verify_release_runtime_dependency_manifest,
)
from gateway.isolated_worker_units import (
    BWRAP_PATH,
    CONFIG_MODE as ISOLATED_WORKER_CONFIG_MODE,
    GATEWAY_READY_PROBE_CONTRACT,
    ISOLATED_WORKER_CLIENT_GROUP,
    ISOLATED_WORKER_CONFIG,
    ISOLATED_WORKER_FD_NAME,
    ISOLATED_WORKER_GROUP,
    ISOLATED_WORKER_LEASE_BASE,
    ISOLATED_WORKER_SERVICE_UNIT,
    ISOLATED_WORKER_SOCKET,
    ISOLATED_WORKER_SOCKET_UNIT,
    ISOLATED_WORKER_USER,
    LEASE_TMPFS_MINIMUM_SYSTEMD_VERSION,
    LEASE_TMPFS_PREFLIGHT_CONTRACT,
    LEASE_QUOTA_ENTRIES,
    SERVICE_GLOBAL_QUOTA_BYTES,
    SERVICE_GLOBAL_QUOTA_ENTRIES,
    SERVICE_TMPFS_INODE_LIMIT,
    SHELL_PATH,
    _render_config as _render_isolated_worker_config,
    _render_service_unit as _render_isolated_worker_service_unit,
    _render_socket_unit as _render_isolated_worker_socket_unit,
)
from tools.browser_controller_client import (
    CLIENT_CONFIG_SCHEMA as BROWSER_CONTROLLER_CLIENT_SCHEMA,
    BrowserControllerClientConfig,
)


CAPABILITY_PLAN_SCHEMA = "muncho-production-capability-runtime-plan.v4"
CAPABILITY_CONTRACT_SCHEMA = "muncho-production-capability-runtime-contract.v2"
CAPABILITY_PREFLIGHT_SCHEMA = "muncho-production-capability-runtime-preflight.v2"
CAPABILITY_LEASE_FRAME_SCHEMA = "muncho-production-capability-secret-lease-frame.v1"
CAPABILITY_LEASE_INSTALL_INTENT_SCHEMA = (
    "muncho-production-capability-secret-install-intent.v1"
)
CAPABILITY_LEASE_RECEIPT_SCHEMA = (
    "muncho-production-capability-secret-install-receipt.v2"
)
CAPABILITY_RETIREMENT_INTENT_SCHEMA = (
    "muncho-production-capability-secret-retirement-intent.v1"
)
CAPABILITY_RETIREMENT_RECEIPT_SCHEMA = (
    "muncho-production-capability-secret-retirement-completion.v2"
)
CAPABILITY_SERVICE_STOP_PROOF_SCHEMA = (
    "muncho-production-capability-service-stop-proof.v1"
)
CAPABILITY_CREDENTIAL_CONSUMER_STOP_PROOF_SCHEMA = (
    "muncho-production-capability-credential-consumer-stop-proof.v1"
)
CAPABILITY_CLEANUP_FACTS_SCHEMA = (
    "muncho-production-capability-cleanup-facts.v1"
)
CAPABILITY_OBSERVER_STOP_RECEIPT_SCHEMA = (
    "muncho-production-capability-observer-stop-receipt.v1"
)
CAPABILITY_CLEANUP_FINALIZATION_SCHEMA = (
    "muncho-production-capability-cleanup-finalization.v1"
)
CAPABILITY_PRODUCTION_OBSERVATION_SCHEMA = (
    "muncho-production-capability-production-observation.v1"
)
CAPABILITY_PRODUCTION_DIFF_SCHEMA = (
    "muncho-production-capability-production-diff.v1"
)
CAPABILITY_PRODUCTION_OBSERVATION_ENVELOPE_SCHEMA = (
    "muncho-production-capability-owner-signed-production-observation.v1"
)
CAPABILITY_PRODUCTION_OBSERVATION_MARKER_SCHEMA = (
    "muncho-production-capability-production-observation-marker.v1"
)
CAPABILITY_PRODUCTION_OBSERVATION_WAIT_REQUEST_SCHEMA = (
    "muncho-production-capability-production-observation-wait-request.v1"
)
CAPABILITY_PRODUCTION_OBSERVATION_STAGE_RECEIPT_SCHEMA = (
    "muncho-production-capability-production-observation-stage.v1"
)
CAPABILITY_PRODUCTION_OBSERVATION_SSHSIG_NAMESPACE = (
    "muncho-production-capability-production-observation-v1"
)
CAPABILITY_LIFECYCLE_RECEIPT_SCHEMA = "muncho-production-capability-runtime-lifecycle.v1"
CAPABILITY_APPROVAL_SCHEMA = "muncho-production-capability-owner-approval.v1"
CAPABILITY_APPROVAL_INSTALL_RECEIPT_SCHEMA = (
    "muncho-production-capability-owner-approval-install.v1"
)
CAPABILITY_BROWSER_HOST_IDENTITY_SCHEMA = (
    "muncho-production-capability-browser-host-identity.v1"
)
CAPABILITY_BROWSER_IDENTITY_FOUNDATION_SCHEMA = (
    "muncho-production-capability-browser-identity-foundation.v1"
)
CAPABILITY_EXECUTION_HOST_IDENTITY_SCHEMA = (
    "muncho-production-capability-execution-host-identity.v1"
)
CAPABILITY_EXECUTION_IDENTITY_FOUNDATION_SCHEMA = (
    "muncho-production-capability-execution-identity-foundation.v1"
)
CAPABILITY_EXECUTION_READINESS_SCHEMA = (
    "muncho-production-capability-execution-readiness.v1"
)
CAPABILITY_PLAN_PUBLICATION_AUTHORITY_SCHEMA = (
    "muncho-production-capability-plan-publication-authority.v1"
)
CAPABILITY_PLAN_PUBLICATION_RECEIPT_SCHEMA = (
    "muncho-production-capability-plan-publication-receipt.v1"
)
CAPABILITY_ROUTEBACK_BOT_IDENTITY_SCHEMA = (
    "muncho-production-capability-routeback-bot-identity.v1"
)
CAPABILITY_BITRIX_FOUNDATION_AUTHORITY_SCHEMA = (
    "muncho-production-capability-bitrix-foundation-authority.v1"
)
CAPABILITY_BITRIX_FOUNDATION_SCOPE = (
    "production_capability_canary_bitrix_foundation"
)
CAPABILITY_BITRIX_IDENTITY_BOOTSTRAP_SCHEMA = (
    "muncho-production-capability-bitrix-identity-bootstrap.v1"
)
CAPABILITY_BITRIX_KEY_BOOTSTRAP_SCHEMA = (
    "muncho-production-capability-bitrix-key-bootstrap.v1"
)
CAPABILITY_BITRIX_FOUNDATION_RECEIPT_SCHEMA = (
    "muncho-production-capability-bitrix-foundation.v1"
)
CAPABILITY_BITRIX_INTERNAL_KEY_RETIREMENT_INTENT_SCHEMA = (
    "muncho-production-capability-internal-key-retirement-intent.v1"
)
CAPABILITY_BITRIX_INTERNAL_KEY_RETIREMENT_SCHEMA = (
    "muncho-production-capability-internal-key-retirement.v1"
)
CAPABILITY_EXPIRY_WATCHDOG_AUTHORITY_SCHEMA = (
    "muncho-production-capability-expiry-watchdog-authority.v1"
)
CAPABILITY_EXPIRY_WATCHDOG_COMPLETION_SCHEMA = (
    "muncho-production-capability-expiry-watchdog-completion.v1"
)
CAPABILITY_PLAN_PUBLICATION_SCOPE = (
    "production_capability_canary_plan_publication"
)
PRODUCTION_CANARY_PUBLIC_GUILD_ID = "1282725267068157972"
PRODUCTION_CANARY_PUBLIC_CHANNEL_ID = "1526858760100909066"
PRODUCTION_OWNER_USER_ID = "1279454038731264061"
LOCKED_NONPUBLIC_CHANNEL_IDS = SKYVISION_LOCKED_NONPUBLIC_CHANNEL_IDS

CAPABILITY_OBSERVER_PLUGIN = "muncho_canary_evidence"
CAPABILITY_OBSERVER_HOOKS = (
    "pre_api_request",
    "post_api_request",
    "post_tool_call",
    "on_session_start",
    "on_session_end",
)
_CAPABILITY_GATEWAY_CONFIG_KEYS = frozenset(
    {
        "agent",
        "browser",
        "canonical_brain",
        "cron",
        "curator",
        "gateway",
        "hooks",
        "kanban",
        "mac_ops_edge",
        "memory",
        "model",
        "platform_toolsets",
        "platforms",
        "plugins",
        "terminal",
    }
)

CODEX_FRAME_MAGIC = b"MCO1"
MAC_OPS_FRAME_MAGIC = b"MCG1"
CONNECTOR_FRAME_MAGIC = b"MCD1"
API_CONTROL_FRAME_MAGIC = b"MCK1"
ROUTEBACK_FRAME_MAGIC = b"MDR1"
BITRIX_FRAME_MAGIC = b"MBX1"
APPROVAL_FRAME_MAGIC = b"MCA1"

_SECRET_LEASE_MAGIC_BY_KIND = {
    "api_server_control_key": API_CONTROL_FRAME_MAGIC,
    "discord_routeback_token": ROUTEBACK_FRAME_MAGIC,
    "bitrix_operational_edge_webhook": BITRIX_FRAME_MAGIC,
    "discord_connector_token": CONNECTOR_FRAME_MAGIC,
    "mac_ops_gitlab_env": MAC_OPS_FRAME_MAGIC,
    "codex_access_token": CODEX_FRAME_MAGIC,
}
_CREDENTIAL_BINDING_BY_KIND = {
    "api_server_control_key": "api_control",
    "discord_routeback_token": "discord_canonical_routeback_bot_token",
    "bitrix_operational_edge_webhook": "bitrix_operational_edge_webhook",
    "discord_connector_token": "discord_public_session_bot_token",
    "mac_ops_gitlab_env": "mac_ops_gitlab",
    "codex_access_token": "openai_codex",
}
_MAX_LEASE_ARTIFACTS = 64

DEFAULT_PLAN_PATH = Path("/etc/muncho/capability-canary/runtime-plan.json")
DEFAULT_APPROVAL_PATH = Path("/etc/muncho/capability-canary/owner-approval.json")
DEFAULT_GATEWAY_CONFIG = Path("/etc/muncho/capability-canary/gateway.yaml")
DEFAULT_GATEWAY_HOME = Path("/var/lib/muncho-capability-canary")
DEFAULT_GATEWAY_PROFILE_HOME = DEFAULT_GATEWAY_HOME / ".hermes"
DEFAULT_GATEWAY_AUTH_STORE = DEFAULT_GATEWAY_PROFILE_HOME / "auth.json"
DEFAULT_GATEWAY_WORK_ROOT = DEFAULT_GATEWAY_HOME / "work"
DEFAULT_GATEWAY_LOG_ROOT = DEFAULT_GATEWAY_HOME / "logs"
DEFAULT_GATEWAY_RUNTIME = Path("/run/hermes-cloud-gateway")
DEFAULT_CONTROL_ROOT = Path("/var/lib/muncho-capability-canary-control")
DEFAULT_PLAN_PUBLICATION_RECEIPT_ROOT = (
    DEFAULT_CONTROL_ROOT / "plan-publications"
)
DEFAULT_APPROVAL_RECEIPT_ROOT = DEFAULT_CONTROL_ROOT / "approval-installs"
DEFAULT_CODEX_LEASE_JOURNAL = DEFAULT_CONTROL_ROOT / "codex-leases"
DEFAULT_MAC_OPS_LEASE_JOURNAL = DEFAULT_CONTROL_ROOT / "mac-ops-leases"
DEFAULT_API_CONTROL_LEASE_JOURNAL = DEFAULT_CONTROL_ROOT / "api-control-leases"
DEFAULT_ROUTEBACK_LEASE_JOURNAL = DEFAULT_CONTROL_ROOT / "routeback-leases"
DEFAULT_BITRIX_LEASE_JOURNAL = DEFAULT_CONTROL_ROOT / "bitrix-leases"
DEFAULT_BITRIX_WEBHOOK_PATH = Path(
    "/opt/adventico-ai-platform/hermes-home/secrets/"
    "bitrix_skyvision_crm_webhook.url"
)
BITRIX_OPERATIONAL_EDGE_UNIT = "muncho-operational-edge-bitrix.service"
DEFAULT_BITRIX_UNIT_PATH = Path("/etc/systemd/system") / BITRIX_OPERATIONAL_EDGE_UNIT
CAPABILITY_PRODUCER_UNIT_PATHS = {
    role: Path("/etc/systemd/system") / CAPABILITY_PRODUCER_SERVICE_UNITS[role]
    for role in CAPABILITY_PRODUCER_ROLES
}
DEFAULT_BITRIX_CONFIG_PATH = Path("/etc/muncho/operational-edge/bitrix.json")
DEFAULT_BITRIX_SOCKET_PATH = Path(
    "/run/muncho-operational-edge/bitrix/edge.sock"
)
DEFAULT_BITRIX_TRUST_PATH = Path(
    "/etc/muncho/operational-edge/trust/bitrix-receipt-public.pem"
)
DEFAULT_BITRIX_RECEIPT_PRIVATE_KEY_PATH = Path(
    "/etc/muncho/keys/operational-edge-bitrix-receipt-private.pem"
)
DEFAULT_BITRIX_WRITER_PUBLIC_KEY_PATH = Path(
    "/etc/muncho/keys/writer-capability-public.pem"
)
_BITRIX_CREDENTIAL_RUNTIME_ROOT = (
    Path("/run/credentials") / BITRIX_OPERATIONAL_EDGE_UNIT
)
DEFAULT_BITRIX_WEBHOOK_PROJECTION_PATH = (
    _BITRIX_CREDENTIAL_RUNTIME_ROOT / "bitrix-webhook-url"
)
DEFAULT_BITRIX_RECEIPT_PRIVATE_KEY_PROJECTION_PATH = (
    _BITRIX_CREDENTIAL_RUNTIME_ROOT / "receipt-private-key"
)
DEFAULT_BITRIX_WRITER_PUBLIC_KEY_PROJECTION_PATH = (
    _BITRIX_CREDENTIAL_RUNTIME_ROOT / "writer-public-key"
)
DEFAULT_BITRIX_IDENTITY_BOOTSTRAP_RECEIPT = (
    DEFAULT_CONTROL_ROOT / "bitrix-identity-bootstrap.json"
)
DEFAULT_BITRIX_FOUNDATION_ROOT = DEFAULT_CONTROL_ROOT / "bitrix-foundations"
DEFAULT_BITRIX_KEY_BOOTSTRAP_ROOT = (
    DEFAULT_CONTROL_ROOT / "bitrix-key-bootstraps"
)
DEFAULT_BITRIX_KEY_RETIREMENT_ROOT = (
    DEFAULT_CONTROL_ROOT / "bitrix-key-retirements"
)
DEFAULT_EXPIRY_WATCHDOG_ROOT = DEFAULT_CONTROL_ROOT / "expiry-watchdogs"
EXPIRY_WATCHDOG_UNIT_PREFIX = "muncho-capability-canary-expiry-"
BITRIX_OPERATIONAL_EDGE_ASSET_NAMES = (
    "bitrix_skyvision_crm.py",
    "bitrix_voucher_ops.py",
    "muncho_step_up_verify",
    "dangerous_action_guard",
)
BITRIX_OPERATIONAL_EDGE_ASSET_IDS = {
    "bitrix_skyvision_crm.py": "bitrix_skyvision_crm.py",
    "bitrix_voucher_ops.py": "bitrix_voucher_ops.py",
    "muncho_step_up_verify": "muncho_step_up_verify",
    "dangerous_action_guard": "muncho_dangerous_action_guard",
}
BITRIX_OPERATIONAL_EDGE_ASSET_MANIFEST_RELATIVE = Path(
    "ops/muncho/runtime/operational-assets/manifest.json"
)
DEFAULT_MAC_OPS_CREDENTIAL_DIR = Path("/etc/muncho/mac-ops-edge-credentials")
DEFAULT_MAC_OPS_CREDENTIAL = DEFAULT_MAC_OPS_CREDENTIAL_DIR / "gitlab.env"
DEFAULT_MAC_OPS_RUNTIME = DEFAULT_MAC_OPS_SOCKET.parent
DEFAULT_MAC_OPS_STATE = Path("/var/lib/muncho-mac-ops")
DEFAULT_MAC_OPS_JOURNAL = DEFAULT_MAC_OPS_STATE / "journal.db"
DEFAULT_MAC_OPS_UNIT_PATH = Path("/etc/systemd/system") / MAC_OPS_UNIT_NAME
DEFAULT_GATEWAY_UNIT_PATH = Path("/etc/systemd/system") / GATEWAY_UNIT_NAME
DEFAULT_BROWSER_UNIT_NAME = "muncho-capability-browser.service"
DEFAULT_BROWSER_UNIT_PATH = Path("/etc/systemd/system") / DEFAULT_BROWSER_UNIT_NAME
DEFAULT_BROWSER_CONFIG = BROWSER_CONFIG_PATH
DEFAULT_BROWSER_SOCKET = BROWSER_SOCKET_PATH
DEFAULT_BROWSER_RUNTIME = DEFAULT_BROWSER_SOCKET.parent
DEFAULT_BROWSER_STATE = BROWSER_STATE_PATH
DEFAULT_BROWSER_ARTIFACT_ROOT = BROWSER_ARTIFACT_PATH
DEFAULT_BROWSER_USER = "muncho-capability-browser"
DEFAULT_BROWSER_GROUP = "muncho-capability-browser"
DEFAULT_BROWSER_HOME = "/nonexistent"
DEFAULT_BROWSER_SHELL = "/usr/sbin/nologin"
DEFAULT_WORKER_CONFIG = ISOLATED_WORKER_CONFIG
DEFAULT_WORKER_SOCKET = ISOLATED_WORKER_SOCKET
DEFAULT_WORKER_LEASE_BASE = ISOLATED_WORKER_LEASE_BASE
DEFAULT_WORKER_SOCKET_UNIT_NAME = ISOLATED_WORKER_SOCKET_UNIT
DEFAULT_WORKER_SERVICE_UNIT_NAME = ISOLATED_WORKER_SERVICE_UNIT
DEFAULT_WORKER_SOCKET_UNIT_PATH = (
    Path("/etc/systemd/system") / DEFAULT_WORKER_SOCKET_UNIT_NAME
)
DEFAULT_WORKER_SERVICE_UNIT_PATH = (
    Path("/etc/systemd/system") / DEFAULT_WORKER_SERVICE_UNIT_NAME
)
DEFAULT_WORKER_USER = ISOLATED_WORKER_USER
DEFAULT_WORKER_GROUP = ISOLATED_WORKER_GROUP
DEFAULT_WORKER_CLIENT_GROUP = ISOLATED_WORKER_CLIENT_GROUP
DEFAULT_WORKER_HOME = "/nonexistent"
DEFAULT_WORKER_SHELL = "/usr/sbin/nologin"
DEFAULT_PROJECTOR_USER = "muncho-projector"
DEFAULT_PROJECTOR_GROUP = "muncho-projector"
_LOOPBACK_DENY_DROP_IN_NAME = "50-muncho-capability-loopback-deny.conf"
DEFAULT_EDGE_LOOPBACK_DENY_DROP_IN = (
    DEFAULT_EDGE_UNIT_PATH.parent
    / f"{EDGE_UNIT_NAME}.d"
    / _LOOPBACK_DENY_DROP_IN_NAME
)
DEFAULT_WRITER_LOOPBACK_DENY_DROP_IN = (
    DEFAULT_WRITER_UNIT_PATH.parent
    / f"{WRITER_UNIT_NAME}.d"
    / _LOOPBACK_DENY_DROP_IN_NAME
)
DEFAULT_PHASE_B_LOOPBACK_DENY_DROP_IN = (
    DEFAULT_PHASE_B_READINESS_UNIT_PATH.parent
    / f"{PHASE_B_READINESS_UNIT_NAME}.d"
    / _LOOPBACK_DENY_DROP_IN_NAME
)
DEFAULT_CONNECTOR_CREDENTIAL_DIR = Path(
    "/etc/muncho/discord-connector-credentials"
)
DEFAULT_CONNECTOR_TOKEN = DEFAULT_CONNECTOR_CREDENTIAL_DIR / "bot-token"
DEFAULT_CONNECTOR_UNIT_PATH = (
    Path("/etc/systemd/system") / DEFAULT_DISCORD_CONNECTOR_UNIT
)
DEFAULT_CONNECTOR_STATE = DEFAULT_DISCORD_CONNECTOR_JOURNAL.parent
DEFAULT_CONNECTOR_LEASE_JOURNAL = DEFAULT_CONTROL_ROOT / "connector-leases"
DEFAULT_LIFECYCLE_RECEIPT_ROOT = DEFAULT_CONTROL_ROOT / "lifecycle"
RUNUSER = "/usr/sbin/runuser"
GROUPADD = "/usr/sbin/groupadd"
USERADD = "/usr/sbin/useradd"
SYSTEMD = "/usr/bin/systemd"
_BROWSER_USERNS_SYSCTLS = {
    "unprivileged_userns_clone": Path(
        "/proc/sys/kernel/unprivileged_userns_clone"
    ),
    "max_user_namespaces": Path("/proc/sys/user/max_user_namespaces"),
}

CAPABILITY_START_ORDER = (
    PHASE_B_READINESS_UNIT_NAME,
    EDGE_UNIT_NAME,
    DEFAULT_DISCORD_CONNECTOR_UNIT,
    MAC_OPS_UNIT_NAME,
    DEFAULT_WORKER_SOCKET_UNIT_NAME,
    DEFAULT_WORKER_SERVICE_UNIT_NAME,
    DEFAULT_BROWSER_UNIT_NAME,
    WRITER_UNIT_NAME,
    BITRIX_OPERATIONAL_EDGE_UNIT,
    *(CAPABILITY_PRODUCER_SERVICE_UNITS[role] for role in CAPABILITY_PRODUCER_ROLES),
    GATEWAY_UNIT_NAME,
)
CAPABILITY_OBSERVER_ROLE = "gateway_observer"
CAPABILITY_OBSERVER_UNIT = CAPABILITY_PRODUCER_SERVICE_UNITS[
    CAPABILITY_OBSERVER_ROLE
]
CAPABILITY_PRE_CLEANUP_STOP_ORDER = (
    GATEWAY_UNIT_NAME,
    *(
        CAPABILITY_PRODUCER_SERVICE_UNITS[role]
        for role in reversed(CAPABILITY_PRODUCER_ROLES)
        if role != CAPABILITY_OBSERVER_ROLE
    ),
    BITRIX_OPERATIONAL_EDGE_UNIT,
    WRITER_UNIT_NAME,
    DEFAULT_BROWSER_UNIT_NAME,
    DEFAULT_WORKER_SERVICE_UNIT_NAME,
    DEFAULT_WORKER_SOCKET_UNIT_NAME,
    MAC_OPS_UNIT_NAME,
    DEFAULT_DISCORD_CONNECTOR_UNIT,
    EDGE_UNIT_NAME,
    PHASE_B_READINESS_UNIT_NAME,
)
CAPABILITY_STOP_ORDER = (
    *CAPABILITY_PRE_CLEANUP_STOP_ORDER,
    CAPABILITY_OBSERVER_UNIT,
)

CAPABILITY_CREDENTIAL_BINDINGS = (
    "api_control",
    "bitrix_operational_edge_webhook",
    "discord_canonical_routeback_bot_token",
    "discord_public_session_bot_token",
    "mac_ops_gitlab",
    "openai_codex",
)

_DISCORD_CONNECTOR_OPERATION_CLASS = (
    "ordinary_public_ingress_and_session_replies"
)
# This is a public Discord snowflake, never a credential or credential digest.
# The capability plan must bind two separate clean-canary applications and
# prove mechanically that neither can silently reuse the production bot.
PRODUCTION_DISCORD_BOT_USER_ID = "1501976597455044801"
_PINNED_RELAY_URL = f"unix://{DEFAULT_DISCORD_CONNECTOR_SOCKET}"
_ROUTEBACK_BOT_IDENTITY_TIMEOUT_SECONDS = 5.0
_MAX_ROUTEBACK_CREDENTIAL_BYTES = 512


def capability_browser_executable(release_root: Path) -> Path:
    """Return the same release-local Chrome-for-Testing layout as production."""

    return (
        release_root
        / "ops/muncho/runtime/dependencies/chrome-linux64/chrome"
    )


def _bitrix_operational_edge_identity(
    *,
    revision: str,
    release_artifact_sha256: str,
    asset_manifest_sha256: str,
    rendered_unit_sha256: str,
    rendered_config_sha256: str,
    rendered_trust_sha256: str,
    identity_bootstrap_receipt_sha256: str,
    receipt_public_key_id: str,
    key_bootstrap_receipt_sha256: str,
    service_uid: int,
    service_gid: int,
    client_gid: int,
) -> str:
    return _sha256_json(
        {
            "schema": "muncho-capability-bitrix-operational-edge-identity.v1",
            "revision": revision,
            "release_artifact_sha256": release_artifact_sha256,
            "service_unit": BITRIX_OPERATIONAL_EDGE_UNIT,
            "service_user": "muncho-edge-bitrix",
            "service_group": "muncho-edge-bitrix",
            "service_uid": service_uid,
            "service_gid": service_gid,
            "client_group": "muncho-edge-bitrix-c",
            "client_gid": client_gid,
            "asset_manifest_sha256": asset_manifest_sha256,
            "asset_names": list(BITRIX_OPERATIONAL_EDGE_ASSET_NAMES),
            "rendered_unit_sha256": rendered_unit_sha256,
            "rendered_config_sha256": rendered_config_sha256,
            "rendered_trust_sha256": rendered_trust_sha256,
            "identity_bootstrap_receipt_sha256": (
                identity_bootstrap_receipt_sha256
            ),
            "receipt_public_key_id": receipt_public_key_id,
            "key_bootstrap_receipt_sha256": key_bootstrap_receipt_sha256,
            "credential_binding": "bitrix_operational_edge_webhook",
        }
    )


def _credential_bindings_mapping() -> dict[str, dict[str, Any]]:
    """Return the six fixed leases consumed by one capability run.

    This is identity/ownership metadata only.  It never contains a credential,
    a credential digest, or semantic dispatch information.  The two foundation
    leases are all installed and retired by this capability-isolated runtime.
    """

    return {
        "api_control": {
            "kind": "api_server_control_key",
            "target_path": str(DEFAULT_API_SERVER_CONTROL_KEY),
            "owner_unit": GATEWAY_UNIT_NAME,
            "managed_by": "capability_canary_runtime",
        },
        "bitrix_operational_edge_webhook": {
            "kind": "bitrix_operational_edge_webhook",
            "target_path": str(DEFAULT_BITRIX_WEBHOOK_PATH),
            "owner_unit": BITRIX_OPERATIONAL_EDGE_UNIT,
            "managed_by": "capability_canary_runtime",
        },
        "discord_canonical_routeback_bot_token": {
            "kind": "discord_routeback_token",
            "target_path": str(DEFAULT_EDGE_TOKEN_PATH),
            "owner_unit": EDGE_UNIT_NAME,
            "managed_by": "capability_canary_runtime",
        },
        "discord_public_session_bot_token": {
            "kind": "discord_connector_token",
            "target_path": str(DEFAULT_CONNECTOR_TOKEN),
            "owner_unit": DEFAULT_DISCORD_CONNECTOR_UNIT,
            "managed_by": "capability_canary_runtime",
        },
        "mac_ops_gitlab": {
            "kind": "mac_ops_gitlab_env",
            "target_path": str(DEFAULT_MAC_OPS_CREDENTIAL),
            "owner_unit": MAC_OPS_UNIT_NAME,
            "managed_by": "capability_canary_runtime",
        },
        "openai_codex": {
            "kind": "codex_access_token",
            "target_path": str(DEFAULT_GATEWAY_AUTH_STORE),
            "owner_unit": GATEWAY_UNIT_NAME,
            "managed_by": "capability_canary_runtime",
        },
    }

_SERVICE_PROPERTIES = (
    "LoadState",
    "ActiveState",
    "SubState",
    "UnitFileState",
    "MainPID",
    "FragmentPath",
    "DropInPaths",
    "Type",
    "NotifyAccess",
    "StatusText",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_IDENTITY_RE = re.compile(r"^[a-z_][a-z0-9_-]{0,63}$")
_LEASE_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_MAX_PLAN_BYTES = 2 * 1024 * 1024
_MAX_SECRET_BYTES = 64 * 1024
_MAX_AUTH_STORE_BYTES = 256 * 1024
_MAX_LEASE_SECONDS = 1_200

_CAPABILITY_GATEWAY_UNSET_ENVIRONMENT_NAMES = tuple(
    sorted(
        set(_GATEWAY_SYSTEMD_UNSET_ENVIRONMENT_NAMES)
        | {
            "AGENT_BROWSER_CONFIG",
            "AGENT_BROWSER_EXECUTABLE_PATH",
            "CAMOFOX_API_KEY",
            "CAMOFOX_URL",
            "DISCORD_TOKEN",
            "GITLAB_BASE_URL",
            "GITLAB_TOKEN",
        }
    )
)
_CAPABILITY_GATEWAY_UNSET_ENVIRONMENT_DIRECTIVE = (
    "UnsetEnvironment="
    + " ".join(_CAPABILITY_GATEWAY_UNSET_ENVIRONMENT_NAMES)
)


def _strict_mapping(value: Any, fields: set[str] | frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        raise ValueError(f"{label} fields are not exact")
    return value


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be lowercase SHA-256")
    return value


def _positive_id(value: Any, label: str) -> int:
    if type(value) is not int or not 0 < value < (1 << 31):
        raise ValueError(f"{label} must be a positive numeric identity")
    return value


def _snowflake_id(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.isdigit()
        or value.startswith("0")
        or len(value) > 25
    ):
        raise ValueError(f"{label} must be a Discord snowflake")
    return value


def _identity(value: Any, label: str) -> str:
    if not isinstance(value, str) or _IDENTITY_RE.fullmatch(value) is None:
        raise ValueError(f"{label} is invalid")
    return value


def _absolute(value: Any, label: str) -> Path:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an absolute path")
    path = Path(value)
    if not path.is_absolute() or path != Path(os.path.normpath(value)) or ".." in path.parts:
        raise ValueError(f"{label} must be an absolute normalized path")
    return path


@dataclass(frozen=True)
class RuntimeIdentities:
    gateway_user: str
    gateway_group: str
    gateway_uid: int
    gateway_gid: int
    socket_client_group: str
    socket_client_gid: int
    edge_group: str
    mac_ops_user: str
    mac_ops_group: str
    mac_ops_uid: int
    mac_ops_gid: int
    connector_user: str
    connector_group: str
    connector_uid: int
    connector_gid: int
    bitrix_operational_edge_user: str
    bitrix_operational_edge_group: str
    bitrix_operational_edge_uid: int
    bitrix_operational_edge_gid: int
    bitrix_operational_edge_client_group: str
    bitrix_operational_edge_client_gid: int
    browser_user: str
    browser_group: str
    browser_uid: int
    browser_gid: int
    worker_user: str
    worker_group: str
    worker_uid: int
    worker_gid: int
    worker_client_group: str
    worker_client_gid: int

    @classmethod
    def from_mapping(cls, value: Any) -> "RuntimeIdentities":
        raw = _strict_mapping(value, set(cls.__dataclass_fields__), "capability identities")
        result = cls(
            **{
                key: (_positive_id(raw[key], key) if key.endswith(("_uid", "_gid")) else _identity(raw[key], key))
                for key in cls.__dataclass_fields__
            }
        )
        if (
            result.mac_ops_user != "muncho-mac-ops-edge"
            or result.mac_ops_group != "muncho-mac-ops-edge"
            or result.connector_user != "muncho-discord-connector"
            or result.connector_group != "muncho-discord-connector"
            or result.bitrix_operational_edge_user != "muncho-edge-bitrix"
            or result.bitrix_operational_edge_group != "muncho-edge-bitrix"
            or result.bitrix_operational_edge_client_group
            != "muncho-edge-bitrix-c"
            or result.browser_user != DEFAULT_BROWSER_USER
            or result.browser_group != DEFAULT_BROWSER_GROUP
            or result.worker_user != DEFAULT_WORKER_USER
            or result.worker_group != DEFAULT_WORKER_GROUP
            or result.worker_client_group != DEFAULT_WORKER_CLIENT_GROUP
        ):
            raise ValueError("capability service identities are not pinned")
        if len(
            {
                result.gateway_user,
                result.mac_ops_user,
                result.connector_user,
                result.bitrix_operational_edge_user,
                result.browser_user,
                result.worker_user,
            }
        ) != 6 or len(
            {
                result.gateway_group,
                result.mac_ops_group,
                result.connector_group,
                result.bitrix_operational_edge_group,
                result.bitrix_operational_edge_client_group,
                result.browser_group,
                result.worker_group,
                result.worker_client_group,
            }
        ) != 8:
            raise ValueError("capability service identity names are not isolated")
        if len(
            {
                result.gateway_uid,
                result.mac_ops_uid,
                result.connector_uid,
                result.bitrix_operational_edge_uid,
                result.browser_uid,
                result.worker_uid,
            }
        ) != 6:
            raise ValueError("capability service identities are not isolated")
        if len(
            {
                result.gateway_gid,
                result.socket_client_gid,
                result.mac_ops_gid,
                result.connector_gid,
                result.bitrix_operational_edge_gid,
                result.bitrix_operational_edge_client_gid,
                result.browser_gid,
                result.worker_gid,
                result.worker_client_gid,
            }
        ) != 9:
            raise ValueError("capability execution group identities are not isolated")
        return result

    def to_mapping(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class CapabilityCanaryPlan:
    revision: str
    full_canary_plan_sha256: str
    release_artifact_sha256: str
    release_root: Path
    interpreter: Path
    identities: RuntimeIdentities
    browser_socket_path: Path
    browser_artifact_root: Path
    browser_node: Path
    browser_node_sha256: str
    browser_wrapper: Path
    browser_wrapper_sha256: str
    browser_native: Path
    browser_native_sha256: str
    browser_executable: Path
    browser_executable_sha256: str
    agent_browser_config: Path
    agent_browser_config_sha256: str
    runtime_dependency_manifest_sha256: str
    worker_bwrap_sha256: str
    worker_shell_sha256: str
    connector_bot_user_id: str
    routeback_bot_user_id: str
    connector_allowed_guild_ids: tuple[str, ...]
    connector_allowed_channel_ids: tuple[str, ...]
    connector_allowed_user_ids: tuple[str, ...]
    mac_ops_service_identity_sha256: str
    bitrix_operational_edge_revision: str
    bitrix_operational_edge_service_unit: str
    bitrix_operational_edge_asset_manifest_path: Path
    bitrix_operational_edge_rendered_unit_path: Path
    bitrix_operational_edge_rendered_config_path: Path
    bitrix_operational_edge_rendered_trust_path: Path
    bitrix_operational_edge_service_user: str
    bitrix_operational_edge_service_group: str
    bitrix_operational_edge_service_uid: int
    bitrix_operational_edge_service_gid: int
    bitrix_operational_edge_socket_client_group: str
    bitrix_operational_edge_socket_client_gid: int
    bitrix_operational_edge_service_identity_sha256: str
    bitrix_operational_edge_asset_manifest_sha256: str
    bitrix_operational_edge_asset_names: tuple[str, ...]
    bitrix_operational_edge_rendered_unit_sha256: str
    bitrix_operational_edge_rendered_config_sha256: str
    bitrix_operational_edge_rendered_trust_sha256: str
    bitrix_operational_edge_identity_bootstrap_receipt_sha256: str
    bitrix_operational_edge_receipt_public_key_id: str
    bitrix_operational_edge_key_bootstrap_receipt_sha256: str
    bitrix_operational_edge_credential_binding: str
    gateway_config_sha256: str
    gateway_unit_sha256: str
    mac_ops_config_sha256: str
    mac_ops_unit_sha256: str
    worker_config_sha256: str
    worker_socket_unit_sha256: str
    worker_service_unit_sha256: str
    browser_config_sha256: str
    browser_unit_sha256: str
    connector_config_sha256: str
    connector_unit_sha256: str
    loopback_deny_drop_in_sha256: str
    sha256: str
    schema: str = CAPABILITY_PLAN_SCHEMA

    @classmethod
    def from_mapping(cls, value: Any) -> "CapabilityCanaryPlan":
        fields = {
            "schema", "revision", "full_canary_plan_sha256", "release",
            "identities", "isolated_worker", "browser", "execution_workspace",
            "toolsets", "api_loopback", "mac_ops", "bitrix_operational_edge",
            "discord_connector", "artifacts",
            "credential_bindings",
            "capability_plan_sha256",
        }
        raw = _strict_mapping(value, fields, "capability plan")
        revision = raw["revision"]
        if raw["schema"] != CAPABILITY_PLAN_SCHEMA or not isinstance(revision, str) or _REVISION_RE.fullmatch(revision) is None:
            raise ValueError("capability plan identity is invalid")
        release = _strict_mapping(raw["release"], {"artifact_root", "artifact_sha256", "interpreter"}, "capability release")
        root = _absolute(release["artifact_root"], "release root")
        interpreter = _absolute(release["interpreter"], "release interpreter")
        if root != Path("/opt/muncho-canary-releases") / revision or interpreter != root / "venv/bin/python":
            raise ValueError("capability release is not revision-bound")
        identities = RuntimeIdentities.from_mapping(raw["identities"])
        browser = _strict_mapping(
            raw["browser"],
            {
                "kind",
                "service_unit",
                "config_path",
                "socket_path",
                "artifact_root",
                "node",
                "wrapper",
                "native",
                "executable",
                "agent_browser_config",
                "runtime_dependency_manifest_path",
                "runtime_dependency_manifest_sha256",
            },
            "browser runtime",
        )
        worker = _strict_mapping(
            raw["isolated_worker"],
            {
                "kind",
                "socket_unit",
                "service_unit",
                "config_path",
                "socket_path",
                "lease_base",
                "socket_uid",
                "socket_gid",
                "server_uid",
                "server_gid",
                "bwrap_path",
                "bwrap_sha256",
                "shell_path",
                "shell_sha256",
                "tmpfs_contract",
                "gateway_ready_probe_contract",
            },
            "isolated worker runtime",
        )
        if (
            worker["kind"] != "authenticated_af_unix_bwrap"
            or worker["socket_unit"] != DEFAULT_WORKER_SOCKET_UNIT_NAME
            or worker["service_unit"] != DEFAULT_WORKER_SERVICE_UNIT_NAME
            or worker["config_path"] != str(DEFAULT_WORKER_CONFIG)
            or worker["socket_path"] != str(DEFAULT_WORKER_SOCKET)
            or worker["lease_base"] != str(DEFAULT_WORKER_LEASE_BASE)
            or worker["socket_uid"] != 0
            or worker["socket_gid"] != identities.worker_client_gid
            or worker["server_uid"] != identities.worker_uid
            or worker["server_gid"] != identities.worker_gid
            or worker["bwrap_path"] != str(BWRAP_PATH)
            or worker["shell_path"] != str(SHELL_PATH)
            or worker["tmpfs_contract"] != LEASE_TMPFS_PREFLIGHT_CONTRACT
            or worker["gateway_ready_probe_contract"]
            != GATEWAY_READY_PROBE_CONTRACT
        ):
            raise ValueError("isolated worker runtime is not exact")
        expected_node = root / NODE_EXECUTABLE
        expected_wrapper = root / AGENT_BROWSER_WRAPPER
        expected_native = root / AGENT_BROWSER_NATIVE
        expected_browser = capability_browser_executable(root)
        expected_agent_browser_config = root / AGENT_BROWSER_CONFIG
        expected_dependency_manifest = root / RUNTIME_DEPENDENCY_MANIFEST_RELATIVE
        def browser_artifact(
            name: str,
            expected: Path,
        ) -> tuple[Path, str]:
            item = _strict_mapping(
                browser[name], {"path", "sha256"}, f"browser {name}"
            )
            path = _absolute(item["path"], f"browser {name} path")
            if path != expected:
                raise ValueError(f"browser {name} path is not release-local")
            return path, _digest(item["sha256"], f"browser {name}")

        browser_node, browser_node_sha256 = browser_artifact("node", expected_node)
        browser_wrapper, browser_wrapper_sha256 = browser_artifact(
            "wrapper", expected_wrapper
        )
        browser_native, browser_native_sha256 = browser_artifact(
            "native", expected_native
        )
        browser_executable, browser_executable_sha256 = browser_artifact(
            "executable", expected_browser
        )
        agent_browser_config, agent_browser_config_sha256 = browser_artifact(
            "agent_browser_config", expected_agent_browser_config
        )
        if (
            browser["kind"] != "af_unix_controller"
            or browser["service_unit"] != DEFAULT_BROWSER_UNIT_NAME
            or browser["config_path"] != str(DEFAULT_BROWSER_CONFIG)
            or browser["socket_path"] != str(DEFAULT_BROWSER_SOCKET)
            or browser["artifact_root"] != str(DEFAULT_BROWSER_ARTIFACT_ROOT)
            or _absolute(
                browser["runtime_dependency_manifest_path"],
                "runtime dependency manifest",
            )
            != expected_dependency_manifest
        ):
            raise ValueError("browser runtime is not the pinned AF_UNIX controller")
        if raw["execution_workspace"] != {
            "path": "/workspace",
            "host_projection_enabled": False,
            "read_only_binds": [],
            "ephemeral_across_worker_restart": True,
            "lease_quota_bytes": SERVICE_GLOBAL_QUOTA_BYTES,
            "lease_quota_entries": LEASE_QUOTA_ENTRIES,
        }:
            raise ValueError("capability execution workspace is not isolated")
        if raw["toolsets"] != list(FIRST_WAVE_TOOLSETS):
            raise ValueError("capability toolsets are not the reviewed first wave")
        if raw["api_loopback"] != {"host": "127.0.0.1", "port": 8642, "key_credential": API_SERVER_CREDENTIAL_NAME}:
            raise ValueError("capability API boundary is not exact")
        mac = _strict_mapping(raw["mac_ops"], {"service_unit", "socket_path", "credential_path", "journal_path", "service_identity_sha256"}, "Mac operations edge")
        if mac["service_unit"] != MAC_OPS_UNIT_NAME or mac["socket_path"] != str(DEFAULT_MAC_OPS_SOCKET) or mac["credential_path"] != str(DEFAULT_MAC_OPS_CREDENTIAL) or mac["journal_path"] != str(DEFAULT_MAC_OPS_JOURNAL):
            raise ValueError("Mac operations edge paths are not pinned")
        bitrix = _strict_mapping(
            raw["bitrix_operational_edge"],
            {
                "revision",
                "service_unit",
                "service_identity_sha256",
                "asset_manifest_sha256",
                "asset_names",
                "asset_manifest_path",
                "rendered_unit_sha256",
                "rendered_unit_path",
                "rendered_config_sha256",
                "rendered_config_path",
                "rendered_trust_sha256",
                "rendered_trust_path",
                "identity_bootstrap",
                "credential_projection",
                "receipt_key_contract",
                "expected_active_service_state",
                "expected_cleanup_service_state",
                "credential_binding",
                "staging_protocol",
                "secret_material_recorded",
                "secret_digest_recorded",
            },
            "Bitrix operational edge",
        )
        bitrix_digests = {
            field: _digest(bitrix[field], f"Bitrix operational edge {field}")
            for field in (
                "service_identity_sha256",
                "asset_manifest_sha256",
                "rendered_unit_sha256",
                "rendered_config_sha256",
                "rendered_trust_sha256",
            )
        }
        bitrix_identity = _strict_mapping(
            bitrix["identity_bootstrap"],
            {
                "service_user",
                "service_group",
                "service_uid",
                "service_gid",
                "socket_client_group",
                "socket_client_gid",
                "receipt_sha256",
            },
            "Bitrix operational edge identity bootstrap",
        )
        bitrix_identity_receipt_sha256 = _digest(
            bitrix_identity["receipt_sha256"],
            "Bitrix operational edge identity bootstrap receipt",
        )
        bitrix_projection = _strict_mapping(
            bitrix["credential_projection"],
            {
                "name",
                "source_path",
                "projected_path",
                "bind_target_path",
                "source_owner_uid",
                "source_owner_gid",
                "source_mode",
                "service_reads_projection",
                "original_source_inaccessible",
                "value_or_digest_recorded",
            },
            "Bitrix operational edge credential projection",
        )
        bitrix_key = _strict_mapping(
            bitrix["receipt_key_contract"],
            {
                "private_credential_name",
                "private_source_path",
                "private_projection_path",
                "private_owner_uid",
                "private_owner_gid",
                "private_mode",
                "public_path",
                "public_key_id",
                "public_trust_sha256",
                "writer_public_key_credential_name",
                "writer_public_key_source_path",
                "writer_public_key_projection_path",
                "key_bootstrap_receipt_sha256",
                "create_only",
                "retire_private_on_stop",
                "retire_public_on_stop",
                "private_content_or_digest_recorded",
            },
            "Bitrix operational edge receipt key contract",
        )
        bitrix_public_key_id = _digest(
            bitrix_key["public_key_id"],
            "Bitrix operational edge receipt public key ID",
        )
        bitrix_key_bootstrap_sha256 = _digest(
            bitrix_key["key_bootstrap_receipt_sha256"],
            "Bitrix operational edge key bootstrap receipt",
        )
        expected_bitrix_identity = _bitrix_operational_edge_identity(
            revision=revision,
            release_artifact_sha256=_digest(
                release["artifact_sha256"], "release artifact"
            ),
            asset_manifest_sha256=bitrix_digests["asset_manifest_sha256"],
            rendered_unit_sha256=bitrix_digests["rendered_unit_sha256"],
            rendered_config_sha256=bitrix_digests["rendered_config_sha256"],
            rendered_trust_sha256=bitrix_digests["rendered_trust_sha256"],
            identity_bootstrap_receipt_sha256=(
                bitrix_identity_receipt_sha256
            ),
            receipt_public_key_id=bitrix_public_key_id,
            key_bootstrap_receipt_sha256=bitrix_key_bootstrap_sha256,
            service_uid=identities.bitrix_operational_edge_uid,
            service_gid=identities.bitrix_operational_edge_gid,
            client_gid=identities.bitrix_operational_edge_client_gid,
        )
        if (
            bitrix["revision"] != revision
            or bitrix["service_unit"] != BITRIX_OPERATIONAL_EDGE_UNIT
            or bitrix["asset_names"] != list(BITRIX_OPERATIONAL_EDGE_ASSET_NAMES)
            or bitrix["asset_manifest_path"]
            != str(root / BITRIX_OPERATIONAL_EDGE_ASSET_MANIFEST_RELATIVE)
            or bitrix["rendered_unit_path"] != str(DEFAULT_BITRIX_UNIT_PATH)
            or bitrix["rendered_config_path"] != str(DEFAULT_BITRIX_CONFIG_PATH)
            or bitrix["rendered_trust_path"] != str(DEFAULT_BITRIX_TRUST_PATH)
            or bitrix["credential_binding"] != "bitrix_operational_edge_webhook"
            or bitrix["staging_protocol"]
            != "sealed_nonsecret_assets_before_service_activation.v1"
            or bitrix["secret_material_recorded"] is not False
            or bitrix["secret_digest_recorded"] is not False
            or bitrix_projection
            != {
                "name": "bitrix-webhook-url",
                "source_path": str(DEFAULT_BITRIX_WEBHOOK_PATH),
                "projected_path": str(DEFAULT_BITRIX_WEBHOOK_PROJECTION_PATH),
                "bind_target_path": str(DEFAULT_BITRIX_WEBHOOK_PATH),
                "source_owner_uid": 0,
                "source_owner_gid": 0,
                "source_mode": "0400",
                "service_reads_projection": True,
                "original_source_inaccessible": True,
                "value_or_digest_recorded": False,
            }
            or bitrix_key
            != {
                "private_credential_name": "receipt-private-key",
                "private_source_path": str(
                    DEFAULT_BITRIX_RECEIPT_PRIVATE_KEY_PATH
                ),
                "private_projection_path": str(
                    DEFAULT_BITRIX_RECEIPT_PRIVATE_KEY_PROJECTION_PATH
                ),
                "private_owner_uid": 0,
                "private_owner_gid": 0,
                "private_mode": "0400",
                "public_path": str(DEFAULT_BITRIX_TRUST_PATH),
                "public_key_id": bitrix_public_key_id,
                "public_trust_sha256": bitrix_digests[
                    "rendered_trust_sha256"
                ],
                "writer_public_key_credential_name": "writer-public-key",
                "writer_public_key_source_path": str(
                    DEFAULT_BITRIX_WRITER_PUBLIC_KEY_PATH
                ),
                "writer_public_key_projection_path": str(
                    DEFAULT_BITRIX_WRITER_PUBLIC_KEY_PROJECTION_PATH
                ),
                "key_bootstrap_receipt_sha256": bitrix_key_bootstrap_sha256,
                "create_only": True,
                "retire_private_on_stop": True,
                "retire_public_on_stop": True,
                "private_content_or_digest_recorded": False,
            }
            or bitrix_identity
            != {
                "service_user": identities.bitrix_operational_edge_user,
                "service_group": identities.bitrix_operational_edge_group,
                "service_uid": identities.bitrix_operational_edge_uid,
                "service_gid": identities.bitrix_operational_edge_gid,
                "socket_client_group": (
                    identities.bitrix_operational_edge_client_group
                ),
                "socket_client_gid": (
                    identities.bitrix_operational_edge_client_gid
                ),
                "receipt_sha256": bitrix_identity_receipt_sha256,
            }
            or bitrix["expected_active_service_state"]
            != {
                "load_state": "loaded",
                "active_state": "active",
                "sub_state": "running",
                "unit_file_state": "disabled",
            }
            or bitrix["expected_cleanup_service_state"]
            != {
                "active_state": "inactive",
                "sub_state": "dead",
                "overlay_retired_or_prior_restored": True,
            }
            or bitrix_digests["service_identity_sha256"]
            != expected_bitrix_identity
        ):
            raise ValueError("Bitrix operational edge identity is not exact")
        connector = _strict_mapping(
            raw["discord_connector"],
            {
                "service_unit",
                "socket_path",
                "token_path",
                "journal_path",
                "connector_bot_user_id",
                "routeback_bot_user_id",
                "production_bot_user_id",
                "allowed_guild_ids",
                "allowed_channel_ids",
                "allowed_user_ids",
                "operation_class",
            },
            "Discord connector",
        )
        if (
            connector["service_unit"] != DEFAULT_DISCORD_CONNECTOR_UNIT
            or connector["socket_path"] != str(DEFAULT_DISCORD_CONNECTOR_SOCKET)
            or connector["token_path"] != str(DEFAULT_CONNECTOR_TOKEN)
            or connector["journal_path"] != str(DEFAULT_DISCORD_CONNECTOR_JOURNAL)
            or connector["operation_class"]
            != _DISCORD_CONNECTOR_OPERATION_CLASS
        ):
            raise ValueError("Discord connector paths/class are not pinned")

        def snowflakes(value: Any, label: str) -> tuple[str, ...]:
            if (
                not isinstance(value, list)
                or not value
                or value != sorted(set(value))
                or any(
                    not isinstance(item, str)
                    or not item.isdigit()
                    or item.startswith("0")
                    for item in value
                )
            ):
                raise ValueError(f"{label} is invalid")
            return tuple(value)

        allowed_guilds = snowflakes(
            connector["allowed_guild_ids"], "connector guild allowlist"
        )
        allowed_channels = snowflakes(
            connector["allowed_channel_ids"], "connector channel allowlist"
        )
        allowed_users = snowflakes(
            connector["allowed_user_ids"], "connector user allowlist"
        )
        connector_bot_user_id = _snowflake_id(
            connector["connector_bot_user_id"], "connector bot identity"
        )
        routeback_bot_user_id = _snowflake_id(
            connector["routeback_bot_user_id"], "route-back bot identity"
        )
        if (
            connector["production_bot_user_id"]
            != PRODUCTION_DISCORD_BOT_USER_ID
            or len(
                {
                    connector_bot_user_id,
                    routeback_bot_user_id,
                    PRODUCTION_DISCORD_BOT_USER_ID,
                }
            )
            != 3
        ):
            raise ValueError("capability Discord bot identities are not isolated")
        if raw["credential_bindings"] != _credential_bindings_mapping():
            raise ValueError("capability credential bindings are not exact")
        artifacts = _strict_mapping(
            raw["artifacts"],
            {
                "gateway_config_sha256",
                "gateway_unit_sha256",
                "mac_ops_config_sha256",
                "mac_ops_unit_sha256",
                "worker_config_sha256",
                "worker_socket_unit_sha256",
                "worker_service_unit_sha256",
                "browser_config_sha256",
                "browser_unit_sha256",
                "connector_config_sha256",
                "connector_unit_sha256",
                "loopback_deny_drop_in_sha256",
            },
            "capability artifacts",
        )
        unsigned = {key: copy.deepcopy(item) for key, item in raw.items() if key != "capability_plan_sha256"}
        digest = _digest(raw["capability_plan_sha256"], "capability plan")
        if _sha256_json(unsigned) != digest:
            raise ValueError("capability plan self-digest drifted")
        result = cls(
            revision=revision,
            full_canary_plan_sha256=_digest(raw["full_canary_plan_sha256"], "full canary plan"),
            release_artifact_sha256=_digest(release["artifact_sha256"], "release artifact"),
            release_root=root,
            interpreter=interpreter,
            identities=identities,
            browser_socket_path=DEFAULT_BROWSER_SOCKET,
            browser_artifact_root=DEFAULT_BROWSER_ARTIFACT_ROOT,
            browser_node=browser_node,
            browser_node_sha256=browser_node_sha256,
            browser_wrapper=browser_wrapper,
            browser_wrapper_sha256=browser_wrapper_sha256,
            browser_native=browser_native,
            browser_native_sha256=browser_native_sha256,
            browser_executable=browser_executable,
            browser_executable_sha256=browser_executable_sha256,
            agent_browser_config=agent_browser_config,
            agent_browser_config_sha256=agent_browser_config_sha256,
            runtime_dependency_manifest_sha256=_digest(
                browser["runtime_dependency_manifest_sha256"],
                "runtime dependency manifest",
            ),
            worker_bwrap_sha256=_digest(
                worker["bwrap_sha256"], "worker bwrap"
            ),
            worker_shell_sha256=_digest(
                worker["shell_sha256"], "worker shell"
            ),
            connector_bot_user_id=connector_bot_user_id,
            routeback_bot_user_id=routeback_bot_user_id,
            connector_allowed_guild_ids=allowed_guilds,
            connector_allowed_channel_ids=allowed_channels,
            connector_allowed_user_ids=allowed_users,
            mac_ops_service_identity_sha256=_digest(mac["service_identity_sha256"], "Mac operations service identity"),
            bitrix_operational_edge_revision=bitrix["revision"],
            bitrix_operational_edge_service_unit=bitrix["service_unit"],
            bitrix_operational_edge_asset_manifest_path=_absolute(
                bitrix["asset_manifest_path"],
                "Bitrix operational edge asset manifest path",
            ),
            bitrix_operational_edge_rendered_unit_path=_absolute(
                bitrix["rendered_unit_path"],
                "Bitrix operational edge rendered unit path",
            ),
            bitrix_operational_edge_rendered_config_path=_absolute(
                bitrix["rendered_config_path"],
                "Bitrix operational edge rendered config path",
            ),
            bitrix_operational_edge_rendered_trust_path=_absolute(
                bitrix["rendered_trust_path"],
                "Bitrix operational edge rendered trust path",
            ),
            bitrix_operational_edge_service_user=bitrix_identity["service_user"],
            bitrix_operational_edge_service_group=bitrix_identity["service_group"],
            bitrix_operational_edge_service_uid=bitrix_identity["service_uid"],
            bitrix_operational_edge_service_gid=bitrix_identity["service_gid"],
            bitrix_operational_edge_socket_client_group=bitrix_identity[
                "socket_client_group"
            ],
            bitrix_operational_edge_socket_client_gid=bitrix_identity[
                "socket_client_gid"
            ],
            bitrix_operational_edge_service_identity_sha256=bitrix_digests[
                "service_identity_sha256"
            ],
            bitrix_operational_edge_asset_manifest_sha256=bitrix_digests[
                "asset_manifest_sha256"
            ],
            bitrix_operational_edge_asset_names=tuple(bitrix["asset_names"]),
            bitrix_operational_edge_rendered_unit_sha256=bitrix_digests[
                "rendered_unit_sha256"
            ],
            bitrix_operational_edge_rendered_config_sha256=bitrix_digests[
                "rendered_config_sha256"
            ],
            bitrix_operational_edge_rendered_trust_sha256=bitrix_digests[
                "rendered_trust_sha256"
            ],
            bitrix_operational_edge_identity_bootstrap_receipt_sha256=(
                bitrix_identity_receipt_sha256
            ),
            bitrix_operational_edge_receipt_public_key_id=bitrix_public_key_id,
            bitrix_operational_edge_key_bootstrap_receipt_sha256=(
                bitrix_key_bootstrap_sha256
            ),
            bitrix_operational_edge_credential_binding=bitrix[
                "credential_binding"
            ],
            gateway_config_sha256=_digest(artifacts["gateway_config_sha256"], "gateway config"),
            gateway_unit_sha256=_digest(artifacts["gateway_unit_sha256"], "gateway unit"),
            mac_ops_config_sha256=_digest(artifacts["mac_ops_config_sha256"], "Mac operations config"),
            mac_ops_unit_sha256=_digest(artifacts["mac_ops_unit_sha256"], "Mac operations unit"),
            worker_config_sha256=_digest(
                artifacts["worker_config_sha256"], "isolated worker config"
            ),
            worker_socket_unit_sha256=_digest(
                artifacts["worker_socket_unit_sha256"],
                "isolated worker socket unit",
            ),
            worker_service_unit_sha256=_digest(
                artifacts["worker_service_unit_sha256"],
                "isolated worker service unit",
            ),
            browser_config_sha256=_digest(
                artifacts["browser_config_sha256"], "browser controller config"
            ),
            browser_unit_sha256=_digest(
                artifacts["browser_unit_sha256"], "browser unit"
            ),
            connector_config_sha256=_digest(
                artifacts["connector_config_sha256"], "connector config"
            ),
            connector_unit_sha256=_digest(
                artifacts["connector_unit_sha256"], "connector unit"
            ),
            loopback_deny_drop_in_sha256=_digest(
                artifacts["loopback_deny_drop_in_sha256"],
                "loopback deny drop-in",
            ),
            sha256=digest,
        )
        result.validate_derived_artifacts()
        return result

    def unsigned_mapping(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "revision": self.revision,
            "full_canary_plan_sha256": self.full_canary_plan_sha256,
            "release": {"artifact_root": str(self.release_root), "artifact_sha256": self.release_artifact_sha256, "interpreter": str(self.interpreter)},
            "identities": self.identities.to_mapping(),
            "isolated_worker": {
                "kind": "authenticated_af_unix_bwrap",
                "socket_unit": DEFAULT_WORKER_SOCKET_UNIT_NAME,
                "service_unit": DEFAULT_WORKER_SERVICE_UNIT_NAME,
                "config_path": str(DEFAULT_WORKER_CONFIG),
                "socket_path": str(DEFAULT_WORKER_SOCKET),
                "lease_base": str(DEFAULT_WORKER_LEASE_BASE),
                "socket_uid": 0,
                "socket_gid": self.identities.worker_client_gid,
                "server_uid": self.identities.worker_uid,
                "server_gid": self.identities.worker_gid,
                "bwrap_path": str(BWRAP_PATH),
                "bwrap_sha256": self.worker_bwrap_sha256,
                "shell_path": str(SHELL_PATH),
                "shell_sha256": self.worker_shell_sha256,
                "tmpfs_contract": LEASE_TMPFS_PREFLIGHT_CONTRACT,
                "gateway_ready_probe_contract": GATEWAY_READY_PROBE_CONTRACT,
            },
            "browser": {
                "kind": "af_unix_controller",
                "service_unit": DEFAULT_BROWSER_UNIT_NAME,
                "config_path": str(DEFAULT_BROWSER_CONFIG),
                "socket_path": str(self.browser_socket_path),
                "artifact_root": str(self.browser_artifact_root),
                "node": {
                    "path": str(self.browser_node),
                    "sha256": self.browser_node_sha256,
                },
                "wrapper": {
                    "path": str(self.browser_wrapper),
                    "sha256": self.browser_wrapper_sha256,
                },
                "native": {
                    "path": str(self.browser_native),
                    "sha256": self.browser_native_sha256,
                },
                "executable": {
                    "path": str(self.browser_executable),
                    "sha256": self.browser_executable_sha256,
                },
                "agent_browser_config": {
                    "path": str(self.agent_browser_config),
                    "sha256": self.agent_browser_config_sha256,
                },
                "runtime_dependency_manifest_path": str(
                    self.release_root / RUNTIME_DEPENDENCY_MANIFEST_RELATIVE
                ),
                "runtime_dependency_manifest_sha256": (
                    self.runtime_dependency_manifest_sha256
                ),
            },
            "execution_workspace": {
                "path": "/workspace",
                "host_projection_enabled": False,
                "read_only_binds": [],
                "ephemeral_across_worker_restart": True,
                "lease_quota_bytes": SERVICE_GLOBAL_QUOTA_BYTES,
                "lease_quota_entries": LEASE_QUOTA_ENTRIES,
            },
            "toolsets": list(FIRST_WAVE_TOOLSETS),
            "api_loopback": {"host": "127.0.0.1", "port": 8642, "key_credential": API_SERVER_CREDENTIAL_NAME},
            "mac_ops": {"service_unit": MAC_OPS_UNIT_NAME, "socket_path": str(DEFAULT_MAC_OPS_SOCKET), "credential_path": str(DEFAULT_MAC_OPS_CREDENTIAL), "journal_path": str(DEFAULT_MAC_OPS_JOURNAL), "service_identity_sha256": self.mac_ops_service_identity_sha256},
            "bitrix_operational_edge": {
                "revision": self.bitrix_operational_edge_revision,
                "service_unit": self.bitrix_operational_edge_service_unit,
                "service_identity_sha256": (
                    self.bitrix_operational_edge_service_identity_sha256
                ),
                "asset_manifest_sha256": (
                    self.bitrix_operational_edge_asset_manifest_sha256
                ),
                "asset_names": list(self.bitrix_operational_edge_asset_names),
                "asset_manifest_path": str(
                    self.bitrix_operational_edge_asset_manifest_path
                ),
                "rendered_unit_sha256": (
                    self.bitrix_operational_edge_rendered_unit_sha256
                ),
                "rendered_unit_path": str(
                    self.bitrix_operational_edge_rendered_unit_path
                ),
                "rendered_config_sha256": (
                    self.bitrix_operational_edge_rendered_config_sha256
                ),
                "rendered_config_path": str(
                    self.bitrix_operational_edge_rendered_config_path
                ),
                "rendered_trust_sha256": (
                    self.bitrix_operational_edge_rendered_trust_sha256
                ),
                "rendered_trust_path": str(
                    self.bitrix_operational_edge_rendered_trust_path
                ),
                "identity_bootstrap": {
                    "service_user": self.bitrix_operational_edge_service_user,
                    "service_group": self.bitrix_operational_edge_service_group,
                    "service_uid": self.bitrix_operational_edge_service_uid,
                    "service_gid": self.bitrix_operational_edge_service_gid,
                    "socket_client_group": (
                        self.bitrix_operational_edge_socket_client_group
                    ),
                    "socket_client_gid": (
                        self.bitrix_operational_edge_socket_client_gid
                    ),
                    "receipt_sha256": (
                        self.bitrix_operational_edge_identity_bootstrap_receipt_sha256
                    ),
                },
                "credential_projection": {
                    "name": "bitrix-webhook-url",
                    "source_path": str(DEFAULT_BITRIX_WEBHOOK_PATH),
                    "projected_path": str(
                        DEFAULT_BITRIX_WEBHOOK_PROJECTION_PATH
                    ),
                    "bind_target_path": str(DEFAULT_BITRIX_WEBHOOK_PATH),
                    "source_owner_uid": 0,
                    "source_owner_gid": 0,
                    "source_mode": "0400",
                    "service_reads_projection": True,
                    "original_source_inaccessible": True,
                    "value_or_digest_recorded": False,
                },
                "receipt_key_contract": {
                    "private_credential_name": "receipt-private-key",
                    "private_source_path": str(
                        DEFAULT_BITRIX_RECEIPT_PRIVATE_KEY_PATH
                    ),
                    "private_projection_path": str(
                        DEFAULT_BITRIX_RECEIPT_PRIVATE_KEY_PROJECTION_PATH
                    ),
                    "private_owner_uid": 0,
                    "private_owner_gid": 0,
                    "private_mode": "0400",
                    "public_path": str(DEFAULT_BITRIX_TRUST_PATH),
                    "public_key_id": (
                        self.bitrix_operational_edge_receipt_public_key_id
                    ),
                    "public_trust_sha256": (
                        self.bitrix_operational_edge_rendered_trust_sha256
                    ),
                    "writer_public_key_credential_name": "writer-public-key",
                    "writer_public_key_source_path": str(
                        DEFAULT_BITRIX_WRITER_PUBLIC_KEY_PATH
                    ),
                    "writer_public_key_projection_path": str(
                        DEFAULT_BITRIX_WRITER_PUBLIC_KEY_PROJECTION_PATH
                    ),
                    "key_bootstrap_receipt_sha256": (
                        self.bitrix_operational_edge_key_bootstrap_receipt_sha256
                    ),
                    "create_only": True,
                    "retire_private_on_stop": True,
                    "retire_public_on_stop": True,
                    "private_content_or_digest_recorded": False,
                },
                "expected_active_service_state": {
                    "load_state": "loaded",
                    "active_state": "active",
                    "sub_state": "running",
                    "unit_file_state": "disabled",
                },
                "expected_cleanup_service_state": {
                    "active_state": "inactive",
                    "sub_state": "dead",
                    "overlay_retired_or_prior_restored": True,
                },
                "credential_binding": (
                    self.bitrix_operational_edge_credential_binding
                ),
                "staging_protocol": (
                    "sealed_nonsecret_assets_before_service_activation.v1"
                ),
                "secret_material_recorded": False,
                "secret_digest_recorded": False,
            },
            "discord_connector": {
                "service_unit": DEFAULT_DISCORD_CONNECTOR_UNIT,
                "socket_path": str(DEFAULT_DISCORD_CONNECTOR_SOCKET),
                "token_path": str(DEFAULT_CONNECTOR_TOKEN),
                "journal_path": str(DEFAULT_DISCORD_CONNECTOR_JOURNAL),
                "connector_bot_user_id": self.connector_bot_user_id,
                "routeback_bot_user_id": self.routeback_bot_user_id,
                "production_bot_user_id": PRODUCTION_DISCORD_BOT_USER_ID,
                "allowed_guild_ids": list(self.connector_allowed_guild_ids),
                "allowed_channel_ids": list(self.connector_allowed_channel_ids),
                "allowed_user_ids": list(self.connector_allowed_user_ids),
                "operation_class": _DISCORD_CONNECTOR_OPERATION_CLASS,
            },
            "credential_bindings": _credential_bindings_mapping(),
            "artifacts": {
                "gateway_config_sha256": self.gateway_config_sha256,
                "gateway_unit_sha256": self.gateway_unit_sha256,
                "mac_ops_config_sha256": self.mac_ops_config_sha256,
                "mac_ops_unit_sha256": self.mac_ops_unit_sha256,
                "worker_config_sha256": self.worker_config_sha256,
                "worker_socket_unit_sha256": self.worker_socket_unit_sha256,
                "worker_service_unit_sha256": self.worker_service_unit_sha256,
                "browser_config_sha256": self.browser_config_sha256,
                "browser_unit_sha256": self.browser_unit_sha256,
                "connector_config_sha256": self.connector_config_sha256,
                "connector_unit_sha256": self.connector_unit_sha256,
                "loopback_deny_drop_in_sha256": (
                    self.loopback_deny_drop_in_sha256
                ),
            },
        }

    def to_mapping(self) -> dict[str, Any]:
        return {**self.unsigned_mapping(), "capability_plan_sha256": self.sha256}

    def validate_derived_artifacts(self) -> None:
        if (
            self.bitrix_operational_edge_revision != self.revision
            or self.bitrix_operational_edge_service_unit
            != BITRIX_OPERATIONAL_EDGE_UNIT
            or self.bitrix_operational_edge_asset_manifest_path
            != self.release_root / BITRIX_OPERATIONAL_EDGE_ASSET_MANIFEST_RELATIVE
            or self.bitrix_operational_edge_rendered_unit_path
            != DEFAULT_BITRIX_UNIT_PATH
            or self.bitrix_operational_edge_rendered_config_path
            != DEFAULT_BITRIX_CONFIG_PATH
            or self.bitrix_operational_edge_rendered_trust_path
            != DEFAULT_BITRIX_TRUST_PATH
            or self.bitrix_operational_edge_service_user
            != self.identities.bitrix_operational_edge_user
            or self.bitrix_operational_edge_service_group
            != self.identities.bitrix_operational_edge_group
            or self.bitrix_operational_edge_service_uid
            != self.identities.bitrix_operational_edge_uid
            or self.bitrix_operational_edge_service_gid
            != self.identities.bitrix_operational_edge_gid
            or self.bitrix_operational_edge_socket_client_group
            != self.identities.bitrix_operational_edge_client_group
            or self.bitrix_operational_edge_socket_client_gid
            != self.identities.bitrix_operational_edge_client_gid
            or self.bitrix_operational_edge_asset_names
            != BITRIX_OPERATIONAL_EDGE_ASSET_NAMES
            or self.bitrix_operational_edge_credential_binding
            != "bitrix_operational_edge_webhook"
            or self.bitrix_operational_edge_service_identity_sha256
            != _bitrix_operational_edge_identity(
                revision=self.revision,
                release_artifact_sha256=self.release_artifact_sha256,
                asset_manifest_sha256=(
                    self.bitrix_operational_edge_asset_manifest_sha256
                ),
                rendered_unit_sha256=(
                    self.bitrix_operational_edge_rendered_unit_sha256
                ),
                rendered_config_sha256=(
                    self.bitrix_operational_edge_rendered_config_sha256
                ),
                rendered_trust_sha256=(
                    self.bitrix_operational_edge_rendered_trust_sha256
                ),
                identity_bootstrap_receipt_sha256=(
                    self.bitrix_operational_edge_identity_bootstrap_receipt_sha256
                ),
                receipt_public_key_id=(
                    self.bitrix_operational_edge_receipt_public_key_id
                ),
                key_bootstrap_receipt_sha256=(
                    self.bitrix_operational_edge_key_bootstrap_receipt_sha256
                ),
                service_uid=self.identities.bitrix_operational_edge_uid,
                service_gid=self.identities.bitrix_operational_edge_gid,
                client_gid=self.identities.bitrix_operational_edge_client_gid,
            )
        ):
            raise ValueError("Bitrix operational edge identity drifted")
        mac_unit = render_mac_ops_unit(self)
        mac_identity = _sha256_bytes(mac_unit.encode("utf-8"))
        if mac_identity != self.mac_ops_service_identity_sha256 or mac_identity != self.mac_ops_unit_sha256:
            raise ValueError("Mac operations unit identity drifted")
        if _sha256_bytes(render_gateway_config(self)) != self.gateway_config_sha256 or _sha256_bytes(render_gateway_unit(self).encode("utf-8")) != self.gateway_unit_sha256 or _sha256_bytes(render_mac_ops_config(self)) != self.mac_ops_config_sha256:
            raise ValueError("capability derived artifacts drifted")
        if (
            _sha256_bytes(render_worker_config(self))
            != self.worker_config_sha256
            or _sha256_bytes(render_worker_socket_unit(self).encode("ascii"))
            != self.worker_socket_unit_sha256
            or _sha256_bytes(render_worker_service_unit(self).encode("ascii"))
            != self.worker_service_unit_sha256
        ):
            raise ValueError("capability isolated worker artifacts drifted")
        if (
            _sha256_bytes(render_browser_config(self))
            != self.browser_config_sha256
        ):
            raise ValueError("capability browser controller config drifted")
        if (
            _sha256_bytes(render_browser_unit(self).encode("utf-8"))
            != self.browser_unit_sha256
        ):
            raise ValueError("capability browser unit drifted")
        if (
            _sha256_bytes(render_connector_config(self))
            != self.connector_config_sha256
            or _sha256_bytes(render_connector_unit(self).encode("utf-8"))
            != self.connector_unit_sha256
        ):
            raise ValueError("capability connector artifacts drifted")
        if (
            _sha256_bytes(render_loopback_deny_drop_in())
            != self.loopback_deny_drop_in_sha256
        ):
            raise ValueError("capability loopback deny drop-in drifted")


@dataclass(frozen=True)
class CapabilityCanaryOwnerApproval:
    """Short-lived, exact-plan authority for the capability transition.

    The trusted owner launcher installs this root-owned file through the same
    reviewed out-of-band boundary as the full-canary approval.  It authorizes
    only the mechanical transition from the already sealed full canary to the
    exact capability plan; it carries no task or routing semantics.
    """

    value: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, value: Any) -> "CapabilityCanaryOwnerApproval":
        raw = _strict_mapping(
            value,
            {
                "schema",
                "scope",
                "plan_sha256",
                "full_canary_plan_sha256",
                "authority_kind",
                "cryptographic_owner_proof",
                "owner_subject_sha256",
                "approval_source_sha256",
                "stopped_preflight_state_sha256",
                "nonce_sha256",
                "approved_at_unix",
                "expires_at_unix",
            },
            "capability-canary owner approval",
        )
        if (
            raw["schema"] != CAPABILITY_APPROVAL_SCHEMA
            or raw["scope"] != "production_capability_canary_runtime_start"
            or raw["authority_kind"]
            != "trusted_root_bootstrap_out_of_band_owner"
            or raw["cryptographic_owner_proof"] is not False
        ):
            raise ValueError("capability-canary owner approval is invalid")
        for field in (
            "plan_sha256",
            "full_canary_plan_sha256",
            "owner_subject_sha256",
            "approval_source_sha256",
            "stopped_preflight_state_sha256",
            "nonce_sha256",
        ):
            _digest(raw[field], f"capability approval {field}")
        approved = raw["approved_at_unix"]
        expires = raw["expires_at_unix"]
        if (
            type(approved) is not int
            or type(expires) is not int
            or approved < 0
            or not 1 <= expires - approved <= 900
        ):
            raise ValueError("capability-canary approval window is invalid")
        return cls(copy.deepcopy(dict(raw)))

    def require(
        self,
        *,
        plan_sha256: str,
        full_canary_plan_sha256: str,
        now_unix: int,
    ) -> None:
        if (
            self.value["plan_sha256"] != plan_sha256
            or self.value["full_canary_plan_sha256"]
            != full_canary_plan_sha256
            or type(now_unix) is not int
            or not self.value["approved_at_unix"]
            <= now_unix
            <= self.value["expires_at_unix"]
        ):
            raise PermissionError(
                "owner approval does not authorize this capability canary"
            )

    @property
    def sha256(self) -> str:
        return _sha256_json(self.value)


def _approval_install_receipt_path(
    plan: CapabilityCanaryPlan,
    approval: CapabilityCanaryOwnerApproval,
) -> Path:
    return (
        DEFAULT_APPROVAL_RECEIPT_ROOT
        / plan.revision
        / plan.sha256
        / f"{approval.value['nonce_sha256']}.json"
    )


def install_capability_approval(
    plan: CapabilityCanaryPlan,
    full_plan: FullCanaryPlan,
    approval: CapabilityCanaryOwnerApproval,
    *,
    runner: Callable[[Command], subprocess.CompletedProcess[bytes]] = _runner,
    metadata_reader: Callable[[str], bytes | str] | None = None,
    local_identity_reader: Callable[[str], bytes | str] | None = None,
) -> Mapping[str, Any]:
    """Install one fresh, exact approval without overwrite or replay."""

    _require_root_linux()
    if not isinstance(approval, CapabilityCanaryOwnerApproval):
        raise TypeError("capability owner approval is required")
    validate_plan_against_full(plan, full_plan)
    approval.require(
        plan_sha256=plan.sha256,
        full_canary_plan_sha256=full_plan.sha256,
        now_unix=int(time.time()),
    )
    validate_dedicated_canary_host(
        full_plan,
        metadata_reader=metadata_reader,
        local_identity_reader=local_identity_reader,
    )
    _validate_release_manifest(full_plan)
    with _lifecycle_lock():
        preflight = collect_capability_preflight(
            plan,
            full_plan,
            phase="stopped",
            runner=runner,
            metadata_reader=metadata_reader,
            local_identity_reader=local_identity_reader,
        )
        approval.require(
            plan_sha256=plan.sha256,
            full_canary_plan_sha256=full_plan.sha256,
            now_unix=int(time.time()),
        )
        if approval.value["stopped_preflight_state_sha256"] != preflight.get(
            "state_sha256"
        ):
            raise PermissionError(
                "owner approval does not bind the current stopped preflight"
            )
        receipt_path = _approval_install_receipt_path(plan, approval)
        if os.path.lexists(DEFAULT_APPROVAL_PATH):
            raise FileExistsError("a capability owner approval is already installed")
        if os.path.lexists(receipt_path):
            raise PermissionError("capability owner approval nonce was already consumed")
        _ensure_root_directory(DEFAULT_APPROVAL_PATH.parent)
        _ensure_root_directory(receipt_path.parent)
        payload = _canonical_bytes(approval.value)
        _write_exclusive_bytes(DEFAULT_APPROVAL_PATH, payload, mode=0o400)
        installed_payload, target = _read_stable_file(
            DEFAULT_APPROVAL_PATH,
            maximum=64 * 1024,
            expected_uid=0,
            expected_gid=0,
            allowed_modes=frozenset({0o400}),
        )
        if installed_payload != payload:
            raise RuntimeError("capability owner approval install readback failed")
        unsigned = {
            "schema": CAPABILITY_APPROVAL_INSTALL_RECEIPT_SCHEMA,
            "operation": "install_capability_owner_approval",
            "revision": plan.revision,
            "plan_sha256": plan.sha256,
            "full_canary_plan_sha256": full_plan.sha256,
            "approval_sha256": approval.sha256,
            "owner_subject_sha256": approval.value["owner_subject_sha256"],
            "approval_source_sha256": approval.value["approval_source_sha256"],
            "stopped_preflight_state_sha256": approval.value[
                "stopped_preflight_state_sha256"
            ],
            "nonce_sha256": approval.value["nonce_sha256"],
            "approved_at_unix": approval.value["approved_at_unix"],
            "expires_at_unix": approval.value["expires_at_unix"],
            "target_path": str(DEFAULT_APPROVAL_PATH),
            "target_device": target.st_dev,
            "target_inode": target.st_ino,
            "target_uid": target.st_uid,
            "target_gid": target.st_gid,
            "target_mode": f"{stat.S_IMODE(target.st_mode):04o}",
            "stopped_preflight_report_sha256": preflight["report_sha256"],
            "installed_at_unix": int(time.time()),
            "secret_material_recorded": False,
            "secret_digest_recorded": False,
        }
        receipt = {**unsigned, "receipt_sha256": _sha256_json(unsigned)}
        try:
            _write_exclusive_bytes(
                receipt_path, _canonical_bytes(receipt), mode=0o400
            )
        except BaseException:
            current = os.lstat(DEFAULT_APPROVAL_PATH)
            if (current.st_dev, current.st_ino) == (target.st_dev, target.st_ino):
                os.unlink(DEFAULT_APPROVAL_PATH)
                _fsync_directory(DEFAULT_APPROVAL_PATH.parent)
            raise
        return {**receipt, "receipt_path": str(receipt_path)}


def _remove_installed_capability_approval(
    plan: CapabilityCanaryPlan,
    full_plan: FullCanaryPlan,
) -> Mapping[str, Any]:
    """Remove only the approval installed by its append-only nonce receipt."""

    try:
        raw, target = _read_stable_file(
            DEFAULT_APPROVAL_PATH,
            maximum=64 * 1024,
            expected_uid=0,
            expected_gid=0,
            allowed_modes=frozenset({0o400}),
        )
    except FileNotFoundError:
        return {"path": str(DEFAULT_APPROVAL_PATH), "removed": False, "absent": True}
    if raw != _canonical_bytes(_decode_json(raw, label="installed capability approval")):
        raise RuntimeError("installed capability approval is not canonical")
    approval = CapabilityCanaryOwnerApproval.from_mapping(
        _decode_json(raw, label="installed capability approval")
    )
    if (
        approval.value["plan_sha256"] != plan.sha256
        or approval.value["full_canary_plan_sha256"] != full_plan.sha256
    ):
        raise RuntimeError("installed capability approval binding drifted")
    receipt_path = _approval_install_receipt_path(plan, approval)
    receipt_raw, _ = _read_stable_file(
        receipt_path,
        maximum=64 * 1024,
        expected_uid=0,
        expected_gid=0,
        allowed_modes=frozenset({0o400}),
    )
    receipt = _decode_json(receipt_raw, label="capability approval install receipt")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if (
        receipt_raw != _canonical_bytes(receipt)
        or receipt.get("schema") != CAPABILITY_APPROVAL_INSTALL_RECEIPT_SCHEMA
        or receipt.get("receipt_sha256") != _sha256_json(unsigned)
        or receipt.get("approval_sha256") != approval.sha256
        or receipt.get("plan_sha256") != plan.sha256
        or receipt.get("full_canary_plan_sha256") != full_plan.sha256
        or receipt.get("nonce_sha256") != approval.value["nonce_sha256"]
        or receipt.get("stopped_preflight_state_sha256")
        != approval.value["stopped_preflight_state_sha256"]
        or receipt.get("target_path") != str(DEFAULT_APPROVAL_PATH)
        or receipt.get("target_device") != target.st_dev
        or receipt.get("target_inode") != target.st_ino
        or receipt.get("target_uid") != target.st_uid
        or receipt.get("target_gid") != target.st_gid
        or receipt.get("target_mode") != f"{stat.S_IMODE(target.st_mode):04o}"
    ):
        raise RuntimeError("capability approval install receipt drifted")
    reachable = os.lstat(DEFAULT_APPROVAL_PATH)
    if (reachable.st_dev, reachable.st_ino) != (target.st_dev, target.st_ino):
        raise RuntimeError("capability approval changed before retirement")
    os.unlink(DEFAULT_APPROVAL_PATH)
    _fsync_directory(DEFAULT_APPROVAL_PATH.parent)
    return {
        "path": str(DEFAULT_APPROVAL_PATH),
        "approval_sha256": approval.sha256,
        "install_receipt_sha256": receipt["receipt_sha256"],
        "removed": True,
        "absent": not os.path.lexists(DEFAULT_APPROVAL_PATH),
    }


def capability_browser_controller_client_mapping(
    plan: CapabilityCanaryPlan,
) -> dict[str, Any]:
    """Return the exact credential-free gateway-side controller binding."""

    value = {
        "schema": BROWSER_CONTROLLER_CLIENT_SCHEMA,
        "socket_path": str(DEFAULT_BROWSER_SOCKET),
        "server_uid": plan.identities.browser_uid,
        "artifact_root": str(DEFAULT_BROWSER_ARTIFACT_ROOT),
        "connect_timeout_seconds": 5,
        "request_timeout_seconds": BROWSER_COMMAND_TIMEOUT_SECONDS,
    }
    BrowserControllerClientConfig.from_mapping(value)
    return value


def capability_browser_controller_client_config(
    plan: CapabilityCanaryPlan,
) -> BrowserControllerClientConfig:
    return BrowserControllerClientConfig.from_mapping(
        capability_browser_controller_client_mapping(plan)
    )


def render_gateway_config(plan: CapabilityCanaryPlan) -> bytes:
    value = {
        "canonical_brain": {"writer_boundary": {"enabled": True}, "discord_edge": {"enabled": True}, "tools_enabled": True},
        "model": {"default": "gpt-5.6-sol", "provider": "openai-codex"},
        "agent": {"reasoning_effort": "high", "max_turns": 90, "adaptive_reasoning": {"enabled": True, "max_effort": "max"}},
        "memory": {"memory_enabled": True, "user_profile_enabled": True},
        "cron": {"enabled": False},
        "kanban": {"auxiliary_planning_enabled": False, "auto_decompose": False, "dispatch_in_gateway": False},
        "curator": {"enabled": False, "prune_builtins": False},
        "plugins": {"enabled": [CAPABILITY_OBSERVER_PLUGIN]},
        # Shell hooks and gateway event hooks can block, rewrite, or inject
        # work.  The capability runtime admits no such extension surface.
        "hooks": {},
        "platform_toolsets": {
            "api_server": list(FIRST_WAVE_TOOLSETS),
            "relay": list(FIRST_WAVE_TOOLSETS),
        },
        "terminal": {
            "backend": "isolated_worker",
            "cwd": "/workspace",
            "timeout": 180,
            "home_mode": "profile", "lifetime_seconds": 900,
            "isolated_worker_socket": str(DEFAULT_WORKER_SOCKET),
            "isolated_worker_server_uid": plan.identities.worker_uid,
            "isolated_worker_server_gid": plan.identities.worker_gid,
            "isolated_worker_socket_uid": 0,
            "isolated_worker_socket_gid": plan.identities.worker_client_gid,
        },
        "browser": {
            "controller": capability_browser_controller_client_mapping(plan),
        },
        "mac_ops_edge": {
            "enabled": True,
            "socket_path": str(DEFAULT_MAC_OPS_SOCKET),
            "service_unit": MAC_OPS_UNIT_NAME,
            "service_uid": plan.identities.mac_ops_uid,
            "socket_gid": plan.identities.socket_client_gid,
            "service_identity_sha256": plan.mac_ops_service_identity_sha256,
            "connect_timeout_seconds": 2.0,
            "request_timeout_seconds": 30.0,
        },
        "gateway": {
            "api_server": {"max_concurrent_runs": 1},
            "isolated_runtime": False,
            "platforms": {
                "api_server": {
                    "enabled": True,
                    "extra": {
                        "host": "127.0.0.1",
                        "port": 8642,
                        "key_credential": API_SERVER_CREDENTIAL_NAME,
                    },
                },
                "relay": {
                    "enabled": True,
                    "extra": {"relay_url": _PINNED_RELAY_URL},
                },
            },
        },
        "platforms": {
            "api_server": {
                "enabled": True,
                "extra": {
                    "host": "127.0.0.1",
                    "port": 8642,
                    "key_credential": API_SERVER_CREDENTIAL_NAME,
                },
            },
            "relay": {
                "enabled": True,
                "extra": {"relay_url": _PINNED_RELAY_URL},
            },
        },
    }
    MacOpsEdgeClientConfig.from_mapping(value["mac_ops_edge"])
    validate_capability_gateway_config(value)
    return yaml.safe_dump(value, sort_keys=True, allow_unicode=True).encode("utf-8")


def validate_capability_gateway_config(raw: Mapping[str, Any]) -> None:
    """Fail closed unless a loaded config preserves the reviewed boundary.

    The model remains the semantic authority inside the normal gateway loop;
    this validator checks only mechanical identities, transports, feature
    boundaries, and the reviewed first-wave tool surface.
    """

    if not isinstance(raw, Mapping):
        raise ValueError("capability gateway config is not a mapping")
    if set(raw) != _CAPABILITY_GATEWAY_CONFIG_KEYS:
        raise ValueError("capability gateway config fields are not exact")
    if raw.get("model") != {
        "default": "gpt-5.6-sol",
        "provider": "openai-codex",
    }:
        raise ValueError("capability gateway model is not exact")
    agent = raw.get("agent")
    if agent != {
        "reasoning_effort": "high",
        "max_turns": 90,
        "adaptive_reasoning": {"enabled": True, "max_effort": "max"},
    }:
        raise ValueError("capability adaptive reasoning is not exact")
    if raw.get("kanban") != {
        "auxiliary_planning_enabled": False,
        "auto_decompose": False,
        "dispatch_in_gateway": False,
    }:
        raise ValueError("capability Kanban boundary is not exact")
    if raw.get("cron") != {"enabled": False}:
        raise ValueError("capability cron boundary is not exact")
    if raw.get("curator") != {"enabled": False, "prune_builtins": False}:
        raise ValueError("capability curator boundary is not exact")
    if raw.get("memory") != {
        "memory_enabled": True,
        "user_profile_enabled": True,
    }:
        raise ValueError("capability memory boundary is not exact")
    if raw.get("plugins") != {"enabled": [CAPABILITY_OBSERVER_PLUGIN]}:
        raise ValueError("capability plugin allowlist is not exact")
    if raw.get("hooks") != {}:
        raise ValueError("capability hook surface must be empty")
    terminal = raw.get("terminal")
    if not isinstance(terminal, Mapping) or terminal.get("backend") != "isolated_worker":
        raise ValueError("capability isolated worker boundary is not exact")
    expected_terminal = {
        "backend": "isolated_worker",
        "cwd": "/workspace",
        "timeout": 180,
        "home_mode": "profile",
        "lifetime_seconds": 900,
        "isolated_worker_socket": str(DEFAULT_WORKER_SOCKET),
        "isolated_worker_server_uid": terminal.get("isolated_worker_server_uid"),
        "isolated_worker_server_gid": terminal.get("isolated_worker_server_gid"),
        "isolated_worker_socket_uid": 0,
        "isolated_worker_socket_gid": terminal.get("isolated_worker_socket_gid"),
    }
    if terminal != expected_terminal or any(
        type(terminal[name]) is not int or terminal[name] < 0
        for name in (
            "isolated_worker_server_uid",
            "isolated_worker_server_gid",
            "isolated_worker_socket_uid",
            "isolated_worker_socket_gid",
        )
    ):
        raise ValueError("capability isolated worker boundary is not exact")
    try:
        browser_raw = raw.get("browser")
        if not isinstance(browser_raw, Mapping) or set(browser_raw) != {"controller"}:
            raise ValueError
        browser_client = BrowserControllerClientConfig.from_mapping(
            browser_raw.get("controller")
        )
        if (
            browser_client.socket_path != DEFAULT_BROWSER_SOCKET
            or browser_client.artifact_root != DEFAULT_BROWSER_ARTIFACT_ROOT
            or browser_client.connect_timeout_seconds != 5
            or browser_client.request_timeout_seconds
            != BROWSER_COMMAND_TIMEOUT_SECONDS
        ):
            raise ValueError
    except Exception as exc:
        raise ValueError("capability browser controller boundary is not exact") from exc
    platform_toolsets = raw.get("platform_toolsets")
    if platform_toolsets != {
        "api_server": list(FIRST_WAVE_TOOLSETS),
        "relay": list(FIRST_WAVE_TOOLSETS),
    }:
        raise ValueError("capability toolsets are not exact")
    gateway = raw.get("gateway")
    if not isinstance(gateway, Mapping) or gateway.get("isolated_runtime") is not False:
        raise ValueError("capability gateway must use the normal loop")
    expected_platforms = {
        "api_server": {
            "enabled": True,
            "extra": {
                "host": "127.0.0.1",
                "port": 8642,
                "key_credential": API_SERVER_CREDENTIAL_NAME,
            },
        },
        "relay": {
            "enabled": True,
            "extra": {"relay_url": _PINNED_RELAY_URL},
        },
    }
    if raw.get("platforms") != expected_platforms or gateway.get("platforms") != expected_platforms:
        raise ValueError("capability gateway platforms are not exact")
    if "discord" in raw.get("platforms", {}) or "discord" in gateway.get("platforms", {}):
        raise ValueError("direct Discord is forbidden in the capability gateway")
    canonical = raw.get("canonical_brain")
    if canonical != {
        "writer_boundary": {"enabled": True},
        "discord_edge": {"enabled": True},
        "tools_enabled": True,
    }:
        raise ValueError("capability Canonical Writer boundary is not exact")


def validate_capability_extension_surface(
    plugin_manager: Any,
    gateway_hooks: Any,
    *,
    plan: CapabilityCanaryPlan | None = None,
) -> None:
    """Attest the exact non-semantic in-process extension surface.

    The normal agent loop remains active, but plugin discovery must have used
    the clean bundled allowlist independently of clean-room task-loop mode.
    Only the sealed evidence observer's five lifecycle callbacks are admitted;
    behavior-changing middleware, tools, commands, skills, auxiliary tasks,
    context engines, platform plugins, shell hooks, and gateway event hooks all
    fail closed.
    """

    expected_allowlist = frozenset({CAPABILITY_OBSERVER_PLUGIN})
    if (
        getattr(plugin_manager, "_discovered", None) is not True
        or getattr(plugin_manager, "_isolated_allowlist", None)
        != expected_allowlist
        or getattr(plugin_manager, "_isolated_discovery_failure", None) is not None
    ):
        raise RuntimeError("capability plugin discovery is not isolated")

    plugins = getattr(plugin_manager, "_plugins", None)
    if not isinstance(plugins, Mapping) or list(plugins) != [
        CAPABILITY_OBSERVER_PLUGIN
    ]:
        raise RuntimeError("capability loaded plugin set is not exact")
    loaded = plugins[CAPABILITY_OBSERVER_PLUGIN]
    manifest = getattr(loaded, "manifest", None)
    module = getattr(loaded, "module", None)
    if (
        manifest is None
        or getattr(manifest, "name", None) != CAPABILITY_OBSERVER_PLUGIN
        or getattr(manifest, "key", None) not in {"", CAPABILITY_OBSERVER_PLUGIN}
        or getattr(manifest, "kind", None) != "standalone"
        or getattr(manifest, "source", None) != "bundled"
        or list(getattr(manifest, "provides_tools", ())) != []
        or list(getattr(manifest, "provides_hooks", ()))
        != list(CAPABILITY_OBSERVER_HOOKS)
        or getattr(loaded, "enabled", None) is not True
        or getattr(loaded, "error", None) is not None
        or getattr(loaded, "deferred", None) is not False
        or list(getattr(loaded, "tools_registered", ())) != []
        or list(getattr(loaded, "hooks_registered", ()))
        != list(CAPABILITY_OBSERVER_HOOKS)
        or list(getattr(loaded, "middleware_registered", ())) != []
        or list(getattr(loaded, "commands_registered", ())) != []
        or module is None
        or getattr(module, "__name__", None)
        != "hermes_plugins.muncho_canary_evidence"
    ):
        raise RuntimeError("capability observer plugin identity is not exact")

    if plan is not None:
        expected_directory = plan.release_root / "plugins" / CAPABILITY_OBSERVER_PLUGIN
        expected_module = expected_directory / "__init__.py"
        if (
            Path(str(getattr(manifest, "path", ""))) != expected_directory
            or Path(str(getattr(module, "__file__", ""))) != expected_module
        ):
            raise RuntimeError("capability observer plugin is outside sealed release")

    plugin_instance = getattr(module, "_PLUGIN", None)
    hooks = getattr(plugin_manager, "_hooks", None)
    if (
        plugin_instance is None
        or not isinstance(hooks, Mapping)
        or list(hooks) != list(CAPABILITY_OBSERVER_HOOKS)
    ):
        raise RuntimeError("capability observer hook set is not exact")
    for hook_name in CAPABILITY_OBSERVER_HOOKS:
        callbacks = hooks.get(hook_name)
        if (
            not isinstance(callbacks, list)
            or len(callbacks) != 1
            or not callable(callbacks[0])
            or getattr(callbacks[0], "__self__", None) is not plugin_instance
            or getattr(callbacks[0], "__name__", None) != hook_name
            or getattr(callbacks[0], "__module__", None)
            != "hermes_plugins.muncho_canary_evidence"
        ):
            raise RuntimeError("capability observer callback identity is not exact")

    empty_surfaces = {
        "_middleware": {},
        "_plugin_tool_names": set(),
        "_plugin_platform_names": set(),
        "_cli_commands": {},
        "_plugin_commands": {},
        "_plugin_skills": {},
        "_aux_tasks": {},
        "_slack_action_handlers": [],
    }
    if any(
        getattr(plugin_manager, name, None) != expected
        for name, expected in empty_surfaces.items()
    ) or getattr(plugin_manager, "_context_engine", None) is not None:
        raise RuntimeError("capability behavior-changing plugin surface is not empty")
    if getattr(plugin_manager, "_cli_ref", None) is not None:
        raise RuntimeError("capability plugin manager is attached to a CLI")
    if (
        getattr(gateway_hooks, "_handlers", None) != {}
        or getattr(gateway_hooks, "_loaded_hooks", None) != []
    ):
        raise RuntimeError("capability gateway event hook surface is not empty")


def capability_gateway_effective_environment_is_sealed(
    env: Mapping[str, str],
    raw_config: Mapping[str, Any],
) -> bool:
    """Validate the complete environment visible to the gateway MainPID.

    Static values are exact.  The small systemd-generated projection is
    allowlisted and shape-checked.  Unknown names, secret-bearing names, and
    config/environment drift all fail closed.
    """

    if not isinstance(env, Mapping) or any(
        not isinstance(name, str) or not isinstance(value, str)
        for name, value in env.items()
    ):
        return False
    try:
        validate_capability_gateway_config(raw_config)
    except (KeyError, TypeError, ValueError):
        return False
    gateway_user = env.get("USER")
    if (
        not isinstance(gateway_user, str)
        or _IDENTITY_RE.fullmatch(gateway_user) is None
        or env.get("LOGNAME") != gateway_user
    ):
        return False
    static = {
        "HOME": str(DEFAULT_GATEWAY_HOME),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "LOGNAME": gateway_user,
        "PATH": "/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "SHELL": "/usr/sbin/nologin",
        "TZ": "UTC",
        "USER": gateway_user,
        "HERMES_CONFIG": str(DEFAULT_GATEWAY_CONFIG),
        "HERMES_HOME": str(DEFAULT_GATEWAY_PROFILE_HOME),
        "HERMES_EXEC_ASK": "1",
        "HERMES_MANAGED_DIR": str(DEFAULT_DISABLED_MANAGED_SCOPE),
        "HERMES_MAX_ITERATIONS": "90",
        "HERMES_QUIET": "1",
        "SSL_CERT_FILE": str(DEFAULT_GATEWAY_CA_BUNDLE),
        "GATEWAY_RELAY_URL": _PINNED_RELAY_URL,
        "GATEWAY_RELAY_PLATFORMS": "discord",
        "TERMINAL_ENV": "isolated_worker",
        "TERMINAL_CWD": "/workspace",
        "TERMINAL_TIMEOUT": "180",
        "TERMINAL_HOME_MODE": "profile",
        "TERMINAL_LIFETIME_SECONDS": "900",
        "TERMINAL_ISOLATED_WORKER_SOCKET": str(DEFAULT_WORKER_SOCKET),
        "TERMINAL_ISOLATED_WORKER_SERVER_UID": str(
            raw_config["terminal"]["isolated_worker_server_uid"]
        ),
        "TERMINAL_ISOLATED_WORKER_SERVER_GID": str(
            raw_config["terminal"]["isolated_worker_server_gid"]
        ),
        "TERMINAL_ISOLATED_WORKER_SOCKET_UID": "0",
        "TERMINAL_ISOLATED_WORKER_SOCKET_GID": str(
            raw_config["terminal"]["isolated_worker_socket_gid"]
        ),
        "CREDENTIALS_DIRECTORY": f"/run/credentials/{GATEWAY_UNIT_NAME}",
        "RUNTIME_DIRECTORY": str(DEFAULT_GATEWAY_RUNTIME),
        "STATE_DIRECTORY": str(DEFAULT_GATEWAY_HOME),
        "_HERMES_GATEWAY": "1",
    }
    if any(env.get(name) != value for name, value in static.items()):
        return False
    optional = set(env) - set(static)
    if not optional.issubset(
        {
            "INVOCATION_ID",
            "JOURNAL_STREAM",
            "NOTIFY_SOCKET",
            "SYSTEMD_EXEC_PID",
        }
    ):
        return False
    if "NOTIFY_SOCKET" not in env:
        return False
    notify = env["NOTIFY_SOCKET"]
    if (
        not notify
        or len(notify) > 256
        or any(ord(character) < 32 for character in notify)
        or not (notify.startswith("/") or notify.startswith("@"))
    ):
        return False
    if "INVOCATION_ID" in env and re.fullmatch(r"[0-9a-f]{32}", env["INVOCATION_ID"]) is None:
        return False
    if "JOURNAL_STREAM" in env and re.fullmatch(r"[0-9]+:[0-9]+", env["JOURNAL_STREAM"]) is None:
        return False
    if "SYSTEMD_EXEC_PID" in env and env["SYSTEMD_EXEC_PID"] != str(os.getpid()):
        return False
    return True


def render_connector_config(plan: CapabilityCanaryPlan) -> bytes:
    """Render the token-owning public Discord connector's exact config."""

    value = {
        "service": {
            "socket_path": str(DEFAULT_DISCORD_CONNECTOR_SOCKET),
            "gateway_unit": GATEWAY_UNIT_NAME,
            "connector_unit": DEFAULT_DISCORD_CONNECTOR_UNIT,
            "gateway_uid": plan.identities.gateway_uid,
            "connector_uid": plan.identities.connector_uid,
            "connector_gid": plan.identities.connector_gid,
            "canary_history_reader": {
                "service_unit": CANARY_HISTORY_READER_SERVICE_UNIT,
                "service_user": CANARY_HISTORY_READER_SERVICE_USER,
                "requester_user_id": CANARY_REQUESTER_USER_ID,
            },
            "connection_timeout_seconds": 10.0,
        },
        "discord": {
            "token_file": str(DEFAULT_CONNECTOR_TOKEN),
            "credentials_directory": str(DEFAULT_CONNECTOR_CREDENTIAL_DIR),
            "allowed_guild_ids": list(plan.connector_allowed_guild_ids),
            "allowed_channel_ids": list(plan.connector_allowed_channel_ids),
            "allowed_user_ids": list(plan.connector_allowed_user_ids),
            "reviewed_cron_history_targets": {},
            "allow_bot_authors": False,
            "ready_timeout_seconds": 30.0,
            "request_timeout_seconds": 15.0,
        },
        "journal": {
            "path": str(DEFAULT_DISCORD_CONNECTOR_JOURNAL),
            "busy_timeout_ms": 5_000,
        },
    }
    return _canonical_bytes(value)


def render_loopback_deny_drop_in() -> bytes:
    """Deny the gateway API loopback addresses for unrelated services."""

    return (
        b"# Capability canary: only the gateway owns the API loopback.\n"
        b"[Service]\n"
        b"IPAddressDeny=127.0.0.1/32\n"
        b"IPAddressDeny=::1/128\n"
    )


def render_connector_unit(plan: CapabilityCanaryPlan) -> str:
    """Render one disabled, bounded, readiness-notifying connector unit."""

    lines = [
        "# Digest-bound public Discord connector for the capability canary.",
        f"# ArtifactSHA256={plan.release_artifact_sha256}",
        "# OperationClass=" + _DISCORD_CONNECTOR_OPERATION_CLASS,
        "[Unit]",
        "Description=Muncho public Discord connector (capability canary)",
        "After=network-online.target",
        "Wants=network-online.target",
        f"Before={GATEWAY_UNIT_NAME}",
        f"AssertPathExists={DEFAULT_CONNECTOR_CONFIG}",
        f"AssertPathExists={DEFAULT_CONNECTOR_TOKEN}",
        f"AssertPathExists={DEFAULT_DISCORD_CONNECTOR_JOURNAL}",
        "",
        "[Service]",
        "Type=notify",
        "NotifyAccess=main",
        f"User={plan.identities.connector_user}",
        f"Group={plan.identities.connector_group}",
        "RuntimeDirectory=muncho-discord-connector",
        "RuntimeDirectoryMode=0750",
        "RuntimeDirectoryPreserve=no",
        "StateDirectory=muncho-discord-connector",
        "StateDirectoryMode=0700",
        f"WorkingDirectory={plan.release_root}",
        f"ExecStart={plan.interpreter} -B -I -m gateway.discord_connector_bootstrap --config {DEFAULT_CONNECTOR_CONFIG}",
        "Restart=no",
        "RuntimeMaxSec=900s",
        "TimeoutStartSec=90s",
        "TimeoutStopSec=30s",
        "KillMode=mixed",
        "LimitCORE=0",
        "UMask=0077",
        *_fixed_environment(
            user=plan.identities.connector_user,
            home=DEFAULT_CONNECTOR_STATE,
        ),
        _CAPABILITY_GATEWAY_UNSET_ENVIRONMENT_DIRECTIVE,
        *_common_hardening(),
        "RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6",
        "IPAddressDeny=169.254.169.254/32",
        "IPAddressDeny=127.0.0.1/32",
        "IPAddressDeny=::1/128",
        f"BindReadOnlyPaths={plan.release_root}",
        f"ReadOnlyPaths={DEFAULT_CONNECTOR_CONFIG}",
        f"ReadOnlyPaths={DEFAULT_CONNECTOR_TOKEN}",
        f"InaccessiblePaths={DEFAULT_EDGE_TOKEN_DIRECTORY}",
        f"InaccessiblePaths={DEFAULT_GATEWAY_AUTH_STORE}",
        f"InaccessiblePaths={DEFAULT_MAC_OPS_CREDENTIAL_DIR}",
        f"InaccessiblePaths={DEFAULT_EDGE_RECEIPT_PRIVATE_KEY}",
        f"InaccessiblePaths={DEFAULT_WRITER_CAPABILITY_PRIVATE_KEY}",
        f"ReadWritePaths={DEFAULT_DISCORD_CONNECTOR_SOCKET.parent}",
        f"ReadWritePaths={DEFAULT_CONNECTOR_STATE}",
        "",
        "[Install]",
        "WantedBy=multi-user.target",
    ]
    result = "\n".join(lines) + "\n"
    if any(
        marker in result
        for marker in (
            "DISCORD_BOT_TOKEN=",
            "DISCORD_TOKEN=",
            "EnvironmentFile=",
            "PassEnvironment=",
            "Restart=on-failure",
        )
    ):
        raise ValueError("capability connector unit crosses a credential boundary")
    return result


def render_mac_ops_unit(plan: CapabilityCanaryPlan) -> str:
    lines = [
        "# Digest-bound production-shaped canary Mac operations edge.",
        f"# ArtifactSHA256={plan.release_artifact_sha256}",
        "[Unit]", "Description=Muncho privileged Mac operations edge (capability canary)",
        "After=network-online.target", "Wants=network-online.target",
        f"Before={GATEWAY_UNIT_NAME}", f"AssertPathExists={DEFAULT_MAC_OPS_CONFIG}",
        f"AssertPathExists={DEFAULT_MAC_OPS_CREDENTIAL}", "", "[Service]",
        "Type=simple", f"User={plan.identities.mac_ops_user}", f"Group={plan.identities.mac_ops_group}",
        f"SupplementaryGroups={plan.identities.socket_client_group}",
        "RuntimeDirectory=muncho-mac-ops", "RuntimeDirectoryMode=0750",
        "RuntimeDirectoryPreserve=no",
        "StateDirectory=muncho-mac-ops", "StateDirectoryMode=0700",
        f"WorkingDirectory={plan.release_root}",
        f"ExecStart={plan.interpreter} -B -I -m gateway.mac_ops_edge_service --config {DEFAULT_MAC_OPS_CONFIG}",
        "Restart=no", "RuntimeMaxSec=900s", "TimeoutStartSec=30s", "TimeoutStopSec=30s",
        "KillMode=mixed", "LimitCORE=0",
        *_fixed_environment(user=plan.identities.mac_ops_user, home=DEFAULT_MAC_OPS_STATE),
        _CAPABILITY_GATEWAY_UNSET_ENVIRONMENT_DIRECTIVE,
        *_common_hardening(),
        "RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6", "IPAddressDeny=169.254.169.254/32",
        "IPAddressDeny=127.0.0.1/32", "IPAddressDeny=::1/128",
        f"BindReadOnlyPaths={plan.release_root}", f"ReadOnlyPaths={DEFAULT_MAC_OPS_CONFIG}",
        f"ReadOnlyPaths={DEFAULT_MAC_OPS_CREDENTIAL}",
        f"InaccessiblePaths={DEFAULT_GATEWAY_AUTH_STORE}", f"InaccessiblePaths={DEFAULT_EDGE_TOKEN_DIRECTORY}",
        f"ReadWritePaths={DEFAULT_MAC_OPS_RUNTIME}", f"ReadWritePaths={DEFAULT_MAC_OPS_STATE}",
        "", "[Install]", "WantedBy=multi-user.target",
    ]
    return "\n".join(lines) + "\n"


def render_mac_ops_config(plan: CapabilityCanaryPlan) -> bytes:
    value = {
        "schema": MAC_OPS_CONFIG_SCHEMA,
        "service": {
            "socket_path": str(DEFAULT_MAC_OPS_SOCKET),
            "gateway_uid": plan.identities.gateway_uid,
            "socket_gid": plan.identities.socket_client_gid,
            "service_identity_sha256": plan.mac_ops_service_identity_sha256,
            "max_connections": 4,
        },
        "gitlab": {"env_file": str(DEFAULT_MAC_OPS_CREDENTIAL), "project_id": MAC_OPS_PROJECT_ID, "timeout_seconds": 20.0},
        "journal": {"path": str(DEFAULT_MAC_OPS_JOURNAL), "busy_timeout_ms": 5_000},
    }
    return _canonical_bytes(value)


def render_worker_config(plan: CapabilityCanaryPlan) -> bytes:
    config, _policy_sha256 = _render_isolated_worker_config(
        gateway_uid=plan.identities.gateway_uid,
        gateway_primary_gid=plan.identities.gateway_gid,
        socket_root_uid=0,
        socket_client_gid=plan.identities.worker_client_gid,
        worker_uid=plan.identities.worker_uid,
        worker_gid=plan.identities.worker_gid,
        bwrap_sha256=plan.worker_bwrap_sha256,
        shell_sha256=plan.worker_shell_sha256,
    )
    return config


def render_worker_socket_unit(plan: CapabilityCanaryPlan) -> str:
    del plan
    return _render_isolated_worker_socket_unit(
        socket_client_group=DEFAULT_WORKER_CLIENT_GROUP
    ).decode("ascii", errors="strict")


def render_worker_service_unit(plan: CapabilityCanaryPlan) -> str:
    return _render_isolated_worker_service_unit(
        revision=plan.revision,
        release_root=plan.release_root,
        interpreter=plan.interpreter,
        worker_user=plan.identities.worker_user,
        worker_group=plan.identities.worker_group,
        worker_uid=plan.identities.worker_uid,
        worker_gid=plan.identities.worker_gid,
    ).decode("ascii", errors="strict")


def render_browser_config(plan: CapabilityCanaryPlan) -> bytes:
    value = {
        "schema": BROWSER_CONTROLLER_CONFIG_SCHEMA,
        "socket_path": str(DEFAULT_BROWSER_SOCKET),
        "socket_runtime_root": str(DEFAULT_BROWSER_RUNTIME),
        "socket_gid": plan.identities.browser_gid,
        "allowed_client_uid": plan.identities.gateway_uid,
        "session_root": str(DEFAULT_BROWSER_STATE),
        "release_root": str(plan.release_root),
        "node_path": str(plan.browser_node),
        "node_sha256": plan.browser_node_sha256,
        "wrapper_path": str(plan.browser_wrapper),
        "wrapper_sha256": plan.browser_wrapper_sha256,
        "native_path": str(plan.browser_native),
        "native_sha256": plan.browser_native_sha256,
        "chrome_path": str(plan.browser_executable),
        "chrome_sha256": plan.browser_executable_sha256,
        "agent_browser_config_path": str(plan.agent_browser_config),
        "agent_browser_config_sha256": plan.agent_browser_config_sha256,
        "command_timeout_seconds": BROWSER_COMMAND_TIMEOUT_SECONDS,
        "idle_timeout_seconds": BROWSER_IDLE_TIMEOUT_SECONDS,
        "max_connections": BROWSER_MAX_CONNECTIONS,
        "max_sessions": BROWSER_MAX_SESSIONS,
        "session_quota_bytes": BROWSER_SESSION_QUOTA_BYTES,
        "session_quota_entries": BROWSER_SESSION_QUOTA_ENTRIES,
    }
    BrowserControllerConfig.from_mapping(value)
    return _canonical_bytes(value)


def render_browser_unit(plan: CapabilityCanaryPlan) -> str:
    """Render one credential-free AF_UNIX browser-controller service."""

    lines = [
        "# Digest-bound browser controller for the capability canary.",
        f"# PrincipalUID={plan.identities.browser_uid}",
        f"# PrincipalGID={plan.identities.browser_gid}",
        f"# ControllerConfigSHA256={plan.browser_config_sha256}",
        "[Unit]",
        "Description=Muncho isolated browser controller (capability canary)",
        "After=network-online.target",
        "Wants=network-online.target",
        f"Before={GATEWAY_UNIT_NAME}",
        f"AssertPathExists={plan.interpreter}",
        f"AssertPathExists={plan.release_root / '.codex-source-commit'}",
        f"AssertPathExists={DEFAULT_BROWSER_CONFIG}",
        f"AssertPathExists={plan.browser_node}",
        f"AssertPathExists={plan.browser_wrapper}",
        f"AssertPathExists={plan.browser_native}",
        f"AssertPathExists={plan.browser_executable}",
        f"AssertPathExists={plan.agent_browser_config}",
        f"AssertPathExists={plan.release_root / RUNTIME_DEPENDENCY_MANIFEST_RELATIVE}",
        f"AssertPathExists={BROWSER_RESOLV_CONF_PATH}",
        "",
        "[Service]",
        "Type=notify",
        "NotifyAccess=main",
        f"User={plan.identities.browser_user}",
        f"Group={plan.identities.browser_group}",
        "RuntimeDirectory=muncho-browser-controller",
        "RuntimeDirectoryMode=0750",
        "RuntimeDirectoryPreserve=no",
        "UMask=0077",
        "StateDirectory=muncho-browser-controller",
        "StateDirectoryMode=0700",
        f"WorkingDirectory={plan.release_root}",
        (
            f"ExecStart={plan.interpreter} -B -P -s -m "
            "gateway.browser_controller "
            f"--config {DEFAULT_BROWSER_CONFIG}"
        ),
        "Restart=no",
        "RuntimeMaxSec=900s",
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
            user=plan.identities.browser_user,
            home=DEFAULT_BROWSER_STATE,
        ),
        f"Environment=XDG_RUNTIME_DIR={DEFAULT_BROWSER_RUNTIME}",
        _CAPABILITY_GATEWAY_UNSET_ENVIRONMENT_DIRECTIVE,
        "UnsetEnvironment=ALL_PROXY HTTP_PROXY HTTPS_PROXY NO_PROXY",
        # Chrome must retain its own unprivileged namespace sandbox.  The
        # stopped preflight attests both host sysctls and this exact unit; do
        # not add RestrictNamespaces=yes or --no-sandbox here.
        "NoNewPrivileges=yes",
        "CapabilityBoundingSet=",
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
        "RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6",
        "RestrictRealtime=yes",
        "RestrictSUIDSGID=yes",
        "SystemCallArchitectures=native",
        f"IPAddressAllow={BROWSER_DNS_ALLOW}",
        *(f"IPAddressDeny={value}" for value in BROWSER_NETWORK_DENY_RANGES),
        f"BindReadOnlyPaths={BROWSER_RESOLV_CONF_PATH}:/etc/resolv.conf",
        f"ReadOnlyPaths={plan.release_root}",
        f"ReadOnlyPaths={DEFAULT_BROWSER_CONFIG}",
        f"ReadOnlyPaths={BROWSER_RESOLV_CONF_PATH}",
        f"ReadWritePaths={DEFAULT_BROWSER_RUNTIME}",
        f"ReadWritePaths={DEFAULT_BROWSER_STATE}",
        f"InaccessiblePaths={DEFAULT_GATEWAY_AUTH_STORE}",
        f"InaccessiblePaths={DEFAULT_MAC_OPS_CREDENTIAL_DIR}",
        f"InaccessiblePaths={DEFAULT_EDGE_TOKEN_DIRECTORY}",
        "InaccessiblePaths=/run/credentials",
        "",
        "[Install]",
        "WantedBy=multi-user.target",
    ]
    result = "\n".join(lines) + "\n"
    if any(
        marker in result
        for marker in (
            "--no-sandbox",
            "remote-debugging",
            "127.0.0.1:9222",
            "EnvironmentFile=",
            "LoadCredential=",
            "PassEnvironment=",
        )
    ):
        raise ValueError("capability browser controller boundary is unsafe")
    return result


def render_gateway_unit(plan: CapabilityCanaryPlan) -> str:
    runtime_dependency_manifest = (
        plan.release_root / RUNTIME_DEPENDENCY_MANIFEST_RELATIVE
    )
    terminal_env = {
        "TERMINAL_ENV": "isolated_worker",
        "TERMINAL_CWD": "/workspace",
        "TERMINAL_TIMEOUT": "180", "TERMINAL_HOME_MODE": "profile",
        "TERMINAL_LIFETIME_SECONDS": "900",
        "TERMINAL_ISOLATED_WORKER_SOCKET": str(DEFAULT_WORKER_SOCKET),
        "TERMINAL_ISOLATED_WORKER_SERVER_UID": str(plan.identities.worker_uid),
        "TERMINAL_ISOLATED_WORKER_SERVER_GID": str(plan.identities.worker_gid),
        "TERMINAL_ISOLATED_WORKER_SOCKET_UID": "0",
        "TERMINAL_ISOLATED_WORKER_SOCKET_GID": str(
            plan.identities.worker_client_gid
        ),
    }
    dependencies = " ".join(
        (
            WRITER_UNIT_NAME,
            EDGE_UNIT_NAME,
            DEFAULT_DISCORD_CONNECTOR_UNIT,
            MAC_OPS_UNIT_NAME,
            DEFAULT_WORKER_SOCKET_UNIT_NAME,
            DEFAULT_WORKER_SERVICE_UNIT_NAME,
            DEFAULT_BROWSER_UNIT_NAME,
        )
    )
    lines = [
        "# Digest-bound production-shaped capability canary; do not edit.",
        f"# ArtifactSHA256={plan.release_artifact_sha256}", "# DiscordCredentialInGateway=false",
        "[Unit]", "Description=Muncho production-shaped model gateway (capability canary)",
        f"Requires={dependencies}",
        f"BindsTo={dependencies}",
        f"After={dependencies}",
        f"AssertPathExists={DEFAULT_GATEWAY_CONFIG}",
        f"AssertPathExists={DEFAULT_PLAN_PATH}",
        f"AssertPathExists={DEFAULT_GATEWAY_AUTH_STORE}",
        f"AssertPathExists={DEFAULT_GATEWAY_CA_BUNDLE}",
        f"AssertPathExists={runtime_dependency_manifest}",
        f"AssertPathExists={DEFAULT_WORKER_CONFIG}",
        f"AssertPathExists={DEFAULT_WORKER_SOCKET}",
        f"AssertPathExists={DEFAULT_BROWSER_CONFIG}",
        f"AssertPathExists={DEFAULT_BROWSER_SOCKET}",
        "", "[Service]", "Type=notify", "NotifyAccess=main",
        f"User={plan.identities.gateway_user}", f"Group={plan.identities.gateway_group}",
        (
            f"SupplementaryGroups={plan.identities.socket_client_group} "
            f"{plan.identities.edge_group} {plan.identities.connector_group} "
            f"{plan.identities.worker_client_group} {plan.identities.browser_group}"
        ),
        "RuntimeDirectory=hermes-cloud-gateway", "RuntimeDirectoryMode=0700",
        "RuntimeDirectoryPreserve=no",
        "StateDirectory=muncho-capability-canary", "StateDirectoryMode=0700",
        f"WorkingDirectory={plan.release_root}",
        f"ExecStart={plan.interpreter} -B -I -m gateway.run --config {DEFAULT_GATEWAY_CONFIG} --require-capability-canary",
        "Restart=no", "RuntimeMaxSec=900s", "TimeoutStartSec=180s", "TimeoutStopSec=90s",
        "KillMode=mixed", "LimitCORE=0",
        f"LoadCredential={API_SERVER_CREDENTIAL_NAME}:{DEFAULT_API_SERVER_CONTROL_KEY}",
        *_fixed_environment(user=plan.identities.gateway_user, home=DEFAULT_GATEWAY_HOME),
        "Environment=PATH=/usr/bin:/bin",
        f"Environment=HERMES_CONFIG={DEFAULT_GATEWAY_CONFIG}",
        f"Environment=HERMES_HOME={DEFAULT_GATEWAY_PROFILE_HOME}",
        f"Environment=HERMES_MANAGED_DIR={DEFAULT_DISABLED_MANAGED_SCOPE}",
        f"Environment=SSL_CERT_FILE={DEFAULT_GATEWAY_CA_BUNDLE}",
        f"Environment=GATEWAY_RELAY_URL={_PINNED_RELAY_URL}",
        "Environment=GATEWAY_RELAY_PLATFORMS=discord",
        *(f"Environment={key}={value}" for key, value in sorted(terminal_env.items())),
        _CAPABILITY_GATEWAY_UNSET_ENVIRONMENT_DIRECTIVE,
        *_common_hardening(), "RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6",
        "IPAddressDeny=169.254.169.254/32", f"BindReadOnlyPaths={plan.release_root}",
        f"BindReadOnlyPaths={DEFAULT_GATEWAY_CONFIG}", f"ReadOnlyPaths={DEFAULT_OBSERVER_CONFIG}",
        f"ReadOnlyPaths={DEFAULT_PLAN_PATH}",
        f"BindReadOnlyPaths={DEFAULT_GATEWAY_AUTH_STORE}",
        f"ReadOnlyPaths={DEFAULT_E2E_FIXTURE}", f"ReadOnlyPaths={DEFAULT_GATEWAY_CA_BUNDLE}",
        f"ReadOnlyPaths={DEFAULT_MAC_OPS_RUNTIME}",
        f"ReadOnlyPaths={DEFAULT_DISCORD_CONNECTOR_SOCKET.parent}",
        f"ReadOnlyPaths={DEFAULT_WORKER_CONFIG}",
        f"ReadOnlyPaths={DEFAULT_WORKER_SOCKET.parent}",
        f"ReadOnlyPaths={DEFAULT_BROWSER_CONFIG}",
        f"ReadOnlyPaths={DEFAULT_BROWSER_SOCKET.parent}",
        f"InaccessiblePaths={DEFAULT_MAC_OPS_CREDENTIAL_DIR}",
        f"InaccessiblePaths={DEFAULT_CONNECTOR_CREDENTIAL_DIR}",
        f"InaccessiblePaths={DEFAULT_EDGE_TOKEN_DIRECTORY}",
        f"InaccessiblePaths={DEFAULT_EDGE_RECEIPT_PRIVATE_KEY}",
        f"InaccessiblePaths={DEFAULT_WRITER_CAPABILITY_PRIVATE_KEY}",
        f"InaccessiblePaths={DEFAULT_DISABLED_MANAGED_SCOPE}",
        f"InaccessiblePaths={DEFAULT_API_SERVER_CONTROL_KEY}",
        f"InaccessiblePaths={DEFAULT_WORKER_LEASE_BASE}",
        f"InaccessiblePaths={DEFAULT_BROWSER_STATE}",
        f"InaccessiblePaths={plan.browser_node}",
        f"InaccessiblePaths={plan.browser_wrapper}",
        f"InaccessiblePaths={plan.browser_native}",
        f"InaccessiblePaths={plan.browser_executable}",
        f"InaccessiblePaths={plan.agent_browser_config}",
        "InaccessiblePaths=-/etc/hermes", "InaccessiblePaths=-/root/.codex",
        f"ReadWritePaths={DEFAULT_GATEWAY_RUNTIME}", f"ReadWritePaths={DEFAULT_GATEWAY_HOME}",
        f"ReadWritePaths={DEFAULT_GATEWAY_LOG_ROOT}", f"ReadWritePaths={DEFAULT_GATEWAY_WORK_ROOT}",
        "", "[Install]", "WantedBy=multi-user.target",
    ]
    result = "\n".join(lines) + "\n"
    if any(marker in result for marker in (
        "DISCORD_BOT_TOKEN=", "GITLAB_TOKEN=", "EnvironmentFile=",
        "PassEnvironment=", "Environment=TERMINAL_DOCKER_", "docker.sock",
        "remote-debugging", "127.0.0.1:9222", "BROWSER_CDP_URL=",
    )):
        raise ValueError("capability gateway unit crosses a credential boundary")
    return result


def build_capability_plan(
    *, full_plan: FullCanaryPlan, mac_ops_uid: int, mac_ops_gid: int,
    connector_uid: int, connector_gid: int,
    bitrix_operational_edge_uid: int,
    bitrix_operational_edge_gid: int,
    bitrix_operational_edge_client_gid: int,
    browser_uid: int, browser_gid: int,
    worker_uid: int, worker_gid: int, worker_client_gid: int,
    connector_bot_user_id: str,
    routeback_bot_user_id: str,
    connector_allowed_guild_ids: Sequence[str],
    connector_allowed_channel_ids: Sequence[str],
    connector_allowed_user_ids: Sequence[str],
    browser_node_sha256: str,
    browser_wrapper_sha256: str,
    browser_native_sha256: str,
    browser_executable_sha256: str,
    agent_browser_config_sha256: str,
    worker_bwrap_sha256: str,
    worker_shell_sha256: str,
    runtime_dependency_manifest_sha256: str,
    bitrix_operational_edge_asset_manifest_sha256: str,
    bitrix_operational_edge_rendered_unit_sha256: str,
    bitrix_operational_edge_rendered_config_sha256: str,
    bitrix_operational_edge_rendered_trust_sha256: str,
    bitrix_operational_edge_identity_bootstrap_receipt_sha256: str,
    bitrix_operational_edge_receipt_public_key_id: str,
    bitrix_operational_edge_key_bootstrap_receipt_sha256: str,
) -> CapabilityCanaryPlan:
    if not isinstance(full_plan, FullCanaryPlan):
        raise TypeError("sealed full-canary plan is required")
    identities = RuntimeIdentities(
        gateway_user=full_plan.identities.gateway_user,
        gateway_group=full_plan.identities.gateway_group,
        gateway_uid=full_plan.identities.gateway_uid,
        gateway_gid=full_plan.identities.gateway_gid,
        socket_client_group=full_plan.identities.socket_client_group,
        socket_client_gid=full_plan.identities.socket_client_gid,
        edge_group=full_plan.identities.edge_group,
        mac_ops_user="muncho-mac-ops-edge", mac_ops_group="muncho-mac-ops-edge",
        mac_ops_uid=_positive_id(mac_ops_uid, "mac_ops_uid"), mac_ops_gid=_positive_id(mac_ops_gid, "mac_ops_gid"),
        connector_user="muncho-discord-connector",
        connector_group="muncho-discord-connector",
        connector_uid=_positive_id(connector_uid, "connector_uid"),
        connector_gid=_positive_id(connector_gid, "connector_gid"),
        bitrix_operational_edge_user="muncho-edge-bitrix",
        bitrix_operational_edge_group="muncho-edge-bitrix",
        bitrix_operational_edge_uid=_positive_id(
            bitrix_operational_edge_uid,
            "bitrix_operational_edge_uid",
        ),
        bitrix_operational_edge_gid=_positive_id(
            bitrix_operational_edge_gid,
            "bitrix_operational_edge_gid",
        ),
        bitrix_operational_edge_client_group="muncho-edge-bitrix-c",
        bitrix_operational_edge_client_gid=_positive_id(
            bitrix_operational_edge_client_gid,
            "bitrix_operational_edge_client_gid",
        ),
        browser_user=DEFAULT_BROWSER_USER,
        browser_group=DEFAULT_BROWSER_GROUP,
        browser_uid=_positive_id(browser_uid, "browser_uid"),
        browser_gid=_positive_id(browser_gid, "browser_gid"),
        worker_user=DEFAULT_WORKER_USER,
        worker_group=DEFAULT_WORKER_GROUP,
        worker_uid=_positive_id(worker_uid, "worker_uid"),
        worker_gid=_positive_id(worker_gid, "worker_gid"),
        worker_client_group=DEFAULT_WORKER_CLIENT_GROUP,
        worker_client_gid=_positive_id(
            worker_client_gid, "worker_client_gid"
        ),
    )
    seed = object.__new__(CapabilityCanaryPlan)
    values = dict(
        revision=full_plan.revision, full_canary_plan_sha256=full_plan.sha256,
        release_artifact_sha256=full_plan.release["artifact_sha256"],
        release_root=Path(full_plan.release["artifact_root"]), interpreter=Path(full_plan.release["interpreter"]),
        identities=identities,
        browser_socket_path=DEFAULT_BROWSER_SOCKET,
        browser_artifact_root=DEFAULT_BROWSER_ARTIFACT_ROOT,
        browser_node=Path(full_plan.release["artifact_root"]) / NODE_EXECUTABLE,
        browser_node_sha256=_digest(browser_node_sha256, "browser node"),
        browser_wrapper=(
            Path(full_plan.release["artifact_root"]) / AGENT_BROWSER_WRAPPER
        ),
        browser_wrapper_sha256=_digest(
            browser_wrapper_sha256, "browser wrapper"
        ),
        browser_native=(
            Path(full_plan.release["artifact_root"]) / AGENT_BROWSER_NATIVE
        ),
        browser_native_sha256=_digest(browser_native_sha256, "browser native"),
        browser_executable=capability_browser_executable(
            Path(full_plan.release["artifact_root"])
        ),
        browser_executable_sha256=_digest(
            browser_executable_sha256, "browser executable"
        ),
        agent_browser_config=(
            Path(full_plan.release["artifact_root"]) / AGENT_BROWSER_CONFIG
        ),
        agent_browser_config_sha256=_digest(
            agent_browser_config_sha256, "agent-browser config"
        ),
        runtime_dependency_manifest_sha256=_digest(
            runtime_dependency_manifest_sha256,
            "runtime dependency manifest",
        ),
        worker_bwrap_sha256=_digest(worker_bwrap_sha256, "worker bwrap"),
        worker_shell_sha256=_digest(worker_shell_sha256, "worker shell"),
        connector_bot_user_id=_snowflake_id(
            connector_bot_user_id, "connector bot identity"
        ),
        routeback_bot_user_id=_snowflake_id(
            routeback_bot_user_id, "route-back bot identity"
        ),
        connector_allowed_guild_ids=tuple(sorted(set(connector_allowed_guild_ids))),
        connector_allowed_channel_ids=tuple(
            sorted(set(connector_allowed_channel_ids))
        ),
        connector_allowed_user_ids=tuple(sorted(set(connector_allowed_user_ids))),
        bitrix_operational_edge_revision=full_plan.revision,
        bitrix_operational_edge_service_unit=BITRIX_OPERATIONAL_EDGE_UNIT,
        bitrix_operational_edge_asset_manifest_path=(
            Path(full_plan.release["artifact_root"])
            / BITRIX_OPERATIONAL_EDGE_ASSET_MANIFEST_RELATIVE
        ),
        bitrix_operational_edge_rendered_unit_path=DEFAULT_BITRIX_UNIT_PATH,
        bitrix_operational_edge_rendered_config_path=DEFAULT_BITRIX_CONFIG_PATH,
        bitrix_operational_edge_rendered_trust_path=DEFAULT_BITRIX_TRUST_PATH,
        bitrix_operational_edge_service_user=(
            identities.bitrix_operational_edge_user
        ),
        bitrix_operational_edge_service_group=(
            identities.bitrix_operational_edge_group
        ),
        bitrix_operational_edge_service_uid=(
            identities.bitrix_operational_edge_uid
        ),
        bitrix_operational_edge_service_gid=(
            identities.bitrix_operational_edge_gid
        ),
        bitrix_operational_edge_socket_client_group=(
            identities.bitrix_operational_edge_client_group
        ),
        bitrix_operational_edge_socket_client_gid=(
            identities.bitrix_operational_edge_client_gid
        ),
        bitrix_operational_edge_asset_manifest_sha256=_digest(
            bitrix_operational_edge_asset_manifest_sha256,
            "Bitrix operational edge asset manifest",
        ),
        bitrix_operational_edge_asset_names=BITRIX_OPERATIONAL_EDGE_ASSET_NAMES,
        bitrix_operational_edge_rendered_unit_sha256=_digest(
            bitrix_operational_edge_rendered_unit_sha256,
            "Bitrix operational edge rendered unit",
        ),
        bitrix_operational_edge_rendered_config_sha256=_digest(
            bitrix_operational_edge_rendered_config_sha256,
            "Bitrix operational edge rendered config",
        ),
        bitrix_operational_edge_rendered_trust_sha256=_digest(
            bitrix_operational_edge_rendered_trust_sha256,
            "Bitrix operational edge rendered trust",
        ),
        bitrix_operational_edge_identity_bootstrap_receipt_sha256=_digest(
            bitrix_operational_edge_identity_bootstrap_receipt_sha256,
            "Bitrix operational edge identity bootstrap receipt",
        ),
        bitrix_operational_edge_receipt_public_key_id=_digest(
            bitrix_operational_edge_receipt_public_key_id,
            "Bitrix operational edge receipt public key ID",
        ),
        bitrix_operational_edge_key_bootstrap_receipt_sha256=_digest(
            bitrix_operational_edge_key_bootstrap_receipt_sha256,
            "Bitrix operational edge key bootstrap receipt",
        ),
        bitrix_operational_edge_credential_binding=(
            "bitrix_operational_edge_webhook"
        ),
    )
    for key, value in values.items():
        object.__setattr__(seed, key, value)
    object.__setattr__(
        seed,
        "bitrix_operational_edge_service_identity_sha256",
        _bitrix_operational_edge_identity(
            revision=seed.revision,
            release_artifact_sha256=seed.release_artifact_sha256,
            asset_manifest_sha256=(
                seed.bitrix_operational_edge_asset_manifest_sha256
            ),
            rendered_unit_sha256=(
                seed.bitrix_operational_edge_rendered_unit_sha256
            ),
            rendered_config_sha256=(
                seed.bitrix_operational_edge_rendered_config_sha256
            ),
            rendered_trust_sha256=(
                seed.bitrix_operational_edge_rendered_trust_sha256
            ),
            identity_bootstrap_receipt_sha256=(
                seed.bitrix_operational_edge_identity_bootstrap_receipt_sha256
            ),
            receipt_public_key_id=(
                seed.bitrix_operational_edge_receipt_public_key_id
            ),
            key_bootstrap_receipt_sha256=(
                seed.bitrix_operational_edge_key_bootstrap_receipt_sha256
            ),
            service_uid=seed.identities.bitrix_operational_edge_uid,
            service_gid=seed.identities.bitrix_operational_edge_gid,
            client_gid=seed.identities.bitrix_operational_edge_client_gid,
        ),
    )
    mac_unit = render_mac_ops_unit(seed)
    mac_identity = _sha256_bytes(mac_unit.encode("utf-8"))
    object.__setattr__(seed, "mac_ops_service_identity_sha256", mac_identity)
    object.__setattr__(seed, "mac_ops_unit_sha256", mac_identity)
    object.__setattr__(seed, "mac_ops_config_sha256", _sha256_bytes(render_mac_ops_config(seed)))
    object.__setattr__(
        seed, "worker_config_sha256", _sha256_bytes(render_worker_config(seed))
    )
    object.__setattr__(
        seed,
        "worker_socket_unit_sha256",
        _sha256_bytes(render_worker_socket_unit(seed).encode("ascii")),
    )
    object.__setattr__(
        seed,
        "worker_service_unit_sha256",
        _sha256_bytes(render_worker_service_unit(seed).encode("ascii")),
    )
    object.__setattr__(
        seed,
        "browser_config_sha256",
        _sha256_bytes(render_browser_config(seed)),
    )
    object.__setattr__(seed, "gateway_config_sha256", _sha256_bytes(render_gateway_config(seed)))
    object.__setattr__(seed, "gateway_unit_sha256", _sha256_bytes(render_gateway_unit(seed).encode("utf-8")))
    object.__setattr__(
        seed,
        "browser_unit_sha256",
        _sha256_bytes(render_browser_unit(seed).encode("utf-8")),
    )
    object.__setattr__(
        seed,
        "connector_config_sha256",
        _sha256_bytes(render_connector_config(seed)),
    )
    object.__setattr__(
        seed,
        "connector_unit_sha256",
        _sha256_bytes(render_connector_unit(seed).encode("utf-8")),
    )
    object.__setattr__(
        seed,
        "loopback_deny_drop_in_sha256",
        _sha256_bytes(render_loopback_deny_drop_in()),
    )
    object.__setattr__(seed, "schema", CAPABILITY_PLAN_SCHEMA)
    unsigned = seed.unsigned_mapping()
    object.__setattr__(seed, "sha256", _sha256_json(unsigned))
    result = CapabilityCanaryPlan.from_mapping(seed.to_mapping())
    validate_plan_against_full(result, full_plan)
    return result


def runtime_dependency_manifest_preflight(
    plan: CapabilityCanaryPlan,
) -> Mapping[str, Any]:
    path = plan.release_root / RUNTIME_DEPENDENCY_MANIFEST_RELATIVE
    raw, _ = _read_stable_file(
        path,
        maximum=2 * 1024 * 1024,
        expected_uid=0,
        allowed_modes=frozenset({0o444, 0o644}),
    )
    if _sha256_bytes(raw) != plan.runtime_dependency_manifest_sha256:
        raise RuntimeError("runtime dependency manifest file drifted")
    if not raw.endswith(b"\n"):
        raise RuntimeError("runtime dependency manifest is not canonical")
    value = _decode_json(raw[:-1], label="runtime dependency manifest")
    if raw != _canonical_bytes(value) + b"\n":
        raise RuntimeError("runtime dependency manifest is not canonical")
    unsigned = {
        name: item for name, item in value.items() if name != "manifest_sha256"
    }
    try:
        chrome = value["chrome"]
        agent_browser = value["agent_browser"]
        distributions = value["python"]["distributions"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError("runtime dependency manifest shape is invalid") from exc
    if (
        value.get("schema") != RUNTIME_DEPENDENCY_MANIFEST_SCHEMA
        or value.get("release_revision") != plan.revision
        or value.get("secret_material_recorded") is not False
        or value.get("manifest_sha256") != _sha256_json(unsigned)
        or not isinstance(chrome, Mapping)
        or chrome.get("version") != RELEASE_CHROME_VERSION
        or chrome.get("executable_path") != str(plan.browser_executable)
        or chrome.get("executable_sha256") != plan.browser_executable_sha256
        or not isinstance(agent_browser, Mapping)
        or agent_browser.get("version") != RELEASE_AGENT_BROWSER_VERSION
        or agent_browser.get("node_path") != str(plan.browser_node)
        or agent_browser.get("node_sha256") != plan.browser_node_sha256
        or agent_browser.get("wrapper_path") != str(plan.browser_wrapper)
        or agent_browser.get("wrapper_sha256") != plan.browser_wrapper_sha256
        or agent_browser.get("native_path") != str(plan.browser_native)
        or agent_browser.get("native_sha256") != plan.browser_native_sha256
        or agent_browser.get("config_path") != str(plan.agent_browser_config)
        or agent_browser.get("config_sha256")
        != plan.agent_browser_config_sha256
        or not isinstance(distributions, Mapping)
        or {
            name: item.get("version")
            for name, item in distributions.items()
            if isinstance(item, Mapping)
        }
        != RELEASE_DDGS_DISTRIBUTIONS
    ):
        raise RuntimeError("runtime dependency manifest identity is invalid")
    observed = verify_release_runtime_dependency_manifest(
        plan.release_root,
        plan.revision,
    )
    if observed != value:
        raise RuntimeError("runtime dependency installation drifted")
    return {
        "path": str(path),
        "file_sha256": plan.runtime_dependency_manifest_sha256,
        "identity_sha256": value["manifest_sha256"],
        "chrome_version": RELEASE_CHROME_VERSION,
        "agent_browser_version": RELEASE_AGENT_BROWSER_VERSION,
        "node_sha256": plan.browser_node_sha256,
        "wrapper_sha256": plan.browser_wrapper_sha256,
        "native_sha256": plan.browser_native_sha256,
        "agent_browser_config_sha256": plan.agent_browser_config_sha256,
        "ddgs_version": RELEASE_DDGS_DISTRIBUTIONS["ddgs"],
        "ready": True,
    }


def browser_executable_preflight(plan: CapabilityCanaryPlan) -> Mapping[str, Any]:
    before = os.lstat(plan.browser_executable)
    mode = stat.S_IMODE(before.st_mode)
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_uid != 0
        or mode & 0o022
        or not mode & 0o111
        or before.st_size <= 0
        or before.st_size > 512 * 1024 * 1024
    ):
        raise RuntimeError("pinned Chrome-for-Testing executable identity is invalid")
    descriptor = os.open(
        plan.browser_executable,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    digest = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    reachable = os.lstat(plan.browser_executable)
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_nlink,
        item.st_uid,
        item.st_gid,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if (
        identity(before) != identity(opened)
        or identity(before) != identity(after)
        or identity(before) != identity(reachable)
        or digest.hexdigest() != plan.browser_executable_sha256
    ):
        raise RuntimeError("pinned Chrome-for-Testing executable drifted")
    return {
        "path": str(plan.browser_executable),
        "sha256": plan.browser_executable_sha256,
        "uid": before.st_uid,
        "mode": f"{mode:04o}",
        "ready": True,
    }


def _root_executable_preflight(path: Path, expected_sha256: str) -> Mapping[str, Any]:
    before = os.lstat(path)
    mode = stat.S_IMODE(before.st_mode)
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_uid != 0
        or mode & 0o022
        or not mode & 0o111
        or not 0 < before.st_size <= 64 * 1024 * 1024
    ):
        raise RuntimeError("isolated worker executable identity is invalid")
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    digest = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    reachable = os.lstat(path)
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_nlink,
        item.st_uid,
        item.st_gid,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    if (
        identity(before) != identity(opened)
        or identity(before) != identity(after)
        or identity(before) != identity(reachable)
        or digest.hexdigest() != expected_sha256
    ):
        raise RuntimeError("isolated worker executable drifted")
    return {
        "path": str(path),
        "sha256": expected_sha256,
        "uid": before.st_uid,
        "mode": f"{mode:04o}",
    }


def worker_executables_preflight(plan: CapabilityCanaryPlan) -> Mapping[str, Any]:
    return {
        "bwrap": _root_executable_preflight(BWRAP_PATH, plan.worker_bwrap_sha256),
        "shell": _root_executable_preflight(SHELL_PATH, plan.worker_shell_sha256),
        "ready": True,
    }


def _optional_passwd_by_name(name: str) -> Any | None:
    try:
        return pwd.getpwnam(name)
    except KeyError:
        return None


def _optional_passwd_by_uid(uid: int) -> Any | None:
    try:
        return pwd.getpwuid(uid)
    except KeyError:
        return None


def _optional_group_by_name(name: str) -> Any | None:
    try:
        return grp.getgrnam(name)
    except KeyError:
        return None


def _optional_group_by_gid(gid: int) -> Any | None:
    try:
        return grp.getgrgid(gid)
    except KeyError:
        return None


def browser_host_identity_receipt(
    plan: CapabilityCanaryPlan,
    full_plan: FullCanaryPlan,
    *,
    allow_create_only_absence: bool,
) -> Mapping[str, Any]:
    """Attest the browser principal or its collision-free create-only slot.

    The receipt is deliberately read-only.  It binds the exact plan identities,
    observes the fixed projector identity when present, and rejects any UID/GID
    alias before the owner can approve a stopped transition.
    """

    validate_plan_against_full(plan, full_plan)
    identities = plan.identities
    browser_user = _optional_passwd_by_name(identities.browser_user)
    browser_group = _optional_group_by_name(identities.browser_group)
    uid_owner = _optional_passwd_by_uid(identities.browser_uid)
    gid_owner = _optional_group_by_gid(identities.browser_gid)

    if browser_user is not None and browser_group is None:
        raise RuntimeError("capability browser user exists without its group")
    if browser_user is None and uid_owner is not None:
        raise RuntimeError("capability browser UID is owned by another user")
    if browser_group is None and gid_owner is not None:
        raise RuntimeError("capability browser GID is owned by another group")
    if browser_group is not None and (
        browser_group.gr_gid != identities.browser_gid
        or gid_owner is None
        or gid_owner.gr_name != identities.browser_group
        or list(browser_group.gr_mem) != []
    ):
        raise RuntimeError("capability browser group identity is not exact")

    supplementary_group_ids: list[int] | None = None
    if browser_user is not None:
        if (
            browser_user.pw_uid != identities.browser_uid
            or browser_user.pw_gid != identities.browser_gid
            or browser_user.pw_dir != DEFAULT_BROWSER_HOME
            or browser_user.pw_shell != DEFAULT_BROWSER_SHELL
            or uid_owner is None
            or uid_owner.pw_name != identities.browser_user
        ):
            raise RuntimeError("capability browser user identity is not exact")
        supplementary_group_ids = sorted(
            set(os.getgrouplist(identities.browser_user, identities.browser_gid))
        )
        if supplementary_group_ids != [identities.browser_gid]:
            raise RuntimeError("capability browser has supplementary authority")

    if browser_user is None and browser_group is None:
        state = "absent_create_only_slot"
    elif browser_user is None:
        state = "group_present_user_absent_create_only_slot"
    else:
        state = "present_exact"
    if state != "present_exact" and not allow_create_only_absence:
        raise RuntimeError("capability browser principal is absent")

    projector_user = _optional_passwd_by_name(DEFAULT_PROJECTOR_USER)
    projector_group = _optional_group_by_name(DEFAULT_PROJECTOR_GROUP)
    if (projector_user is None) != (projector_group is None):
        raise RuntimeError("capability projector identity is incomplete")
    projector: Mapping[str, Any]
    if projector_user is None or projector_group is None:
        projector = {"state": "absent"}
    else:
        if projector_user.pw_gid != projector_group.gr_gid:
            raise RuntimeError("capability projector primary group drifted")
        projector = {
            "state": "present",
            "user": projector_user.pw_name,
            "group": projector_group.gr_name,
            "uid": projector_user.pw_uid,
            "gid": projector_group.gr_gid,
        }
        if (
            projector_user.pw_uid == identities.browser_uid
            or projector_group.gr_gid == identities.browser_gid
        ):
            raise RuntimeError("capability browser aliases the projector")

    isolated_roles = {
        "gateway": {
            "user": full_plan.identities.gateway_user,
            "group": full_plan.identities.gateway_group,
            "uid": full_plan.identities.gateway_uid,
            "gid": full_plan.identities.gateway_gid,
        },
        "writer": {
            "user": full_plan.identities.writer_user,
            "group": full_plan.identities.writer_group,
            "uid": full_plan.identities.writer_uid,
            "gid": full_plan.identities.writer_gid,
        },
        "routeback": {
            "user": full_plan.identities.edge_user,
            "group": full_plan.identities.edge_group,
            "uid": full_plan.identities.edge_uid,
            "gid": full_plan.identities.edge_gid,
        },
        "connector": {
            "user": identities.connector_user,
            "group": identities.connector_group,
            "uid": identities.connector_uid,
            "gid": identities.connector_gid,
        },
        "mac_ops": {
            "user": identities.mac_ops_user,
            "group": identities.mac_ops_group,
            "uid": identities.mac_ops_uid,
            "gid": identities.mac_ops_gid,
        },
    }
    if any(
        identity["user"] == identities.browser_user
        or identity["group"] == identities.browser_group
        or identity["uid"] == identities.browser_uid
        or identity["gid"] == identities.browser_gid
        for identity in isolated_roles.values()
    ):
        raise RuntimeError("capability browser identity aliases another service")

    unsigned = {
        "schema": CAPABILITY_BROWSER_HOST_IDENTITY_SCHEMA,
        "plan_sha256": plan.sha256,
        "browser": {
            "state": state,
            "user": identities.browser_user,
            "group": identities.browser_group,
            "uid": identities.browser_uid,
            "gid": identities.browser_gid,
            "home": DEFAULT_BROWSER_HOME,
            "shell": DEFAULT_BROWSER_SHELL,
            "group_members": [] if browser_group is not None else None,
            "supplementary_group_ids": supplementary_group_ids,
        },
        "projector": projector,
        "isolated_roles": isolated_roles,
        "create_only_eligible": True,
        "secret_material_recorded": False,
    }
    return {**unsigned, "receipt_sha256": _sha256_json(unsigned)}


def ensure_browser_identity_create_only(
    plan: CapabilityCanaryPlan,
    full_plan: FullCanaryPlan,
    *,
    runner: Callable[[Command], subprocess.CompletedProcess[bytes]] = _runner,
    observer: Callable[..., Mapping[str, Any]] = browser_host_identity_receipt,
) -> Mapping[str, Any]:
    """Create only the exact browser principal; never alter an existing one."""

    _require_root_linux()
    before = observer(
        plan,
        full_plan,
        allow_create_only_absence=True,
    )
    state = before["browser"]["state"]
    created_group = False
    created_user = False
    if state == "absent_create_only_slot":
        _run_checked(
            Command(
                (
                    GROUPADD,
                    "--system",
                    "--gid",
                    str(plan.identities.browser_gid),
                    "--",
                    plan.identities.browser_group,
                )
            ),
            runner=runner,
            label="create capability browser group",
        )
        created_group = True
    if state != "present_exact":
        _run_checked(
            Command(
                (
                    USERADD,
                    "--system",
                    "--uid",
                    str(plan.identities.browser_uid),
                    "--gid",
                    plan.identities.browser_group,
                    "--home-dir",
                    DEFAULT_BROWSER_HOME,
                    "--no-create-home",
                    "--shell",
                    DEFAULT_BROWSER_SHELL,
                    "--",
                    plan.identities.browser_user,
                )
            ),
            runner=runner,
            label="create capability browser user",
        )
        created_user = True
    after = observer(
        plan,
        full_plan,
        allow_create_only_absence=False,
    )
    unsigned = {
        "schema": CAPABILITY_BROWSER_IDENTITY_FOUNDATION_SCHEMA,
        "plan_sha256": plan.sha256,
        "before_receipt_sha256": before["receipt_sha256"],
        "after_receipt_sha256": after["receipt_sha256"],
        "created_group": created_group,
        "created_user": created_user,
        "retained_dormant_on_rollback": True,
        "secret_material_recorded": False,
    }
    return {
        **unsigned,
        "receipt_sha256": _sha256_json(unsigned),
        "host_identity": copy.deepcopy(dict(after)),
    }


def execution_host_identity_receipt(
    plan: CapabilityCanaryPlan,
    full_plan: FullCanaryPlan,
    *,
    allow_create_only_absence: bool,
) -> Mapping[str, Any]:
    """Attest collision-free worker/server and socket-client identities."""

    validate_plan_against_full(plan, full_plan)
    identities = plan.identities
    worker_user = _optional_passwd_by_name(identities.worker_user)
    worker_group = _optional_group_by_name(identities.worker_group)
    client_group = _optional_group_by_name(identities.worker_client_group)
    uid_owner = _optional_passwd_by_uid(identities.worker_uid)
    worker_gid_owner = _optional_group_by_gid(identities.worker_gid)
    client_gid_owner = _optional_group_by_gid(identities.worker_client_gid)

    if worker_user is None and uid_owner is not None:
        raise RuntimeError("capability worker UID is owned by another user")
    for group, owner, name, gid, label in (
        (
            worker_group,
            worker_gid_owner,
            identities.worker_group,
            identities.worker_gid,
            "worker",
        ),
        (
            client_group,
            client_gid_owner,
            identities.worker_client_group,
            identities.worker_client_gid,
            "worker client",
        ),
    ):
        if group is None and owner is not None:
            raise RuntimeError(f"capability {label} GID is owned by another group")
        if group is not None and (
            group.gr_gid != gid
            or owner is None
            or owner.gr_name != name
            or list(group.gr_mem) != []
        ):
            raise RuntimeError(f"capability {label} group identity is not exact")

    supplementary: list[int] | None = None
    if worker_user is not None:
        if (
            worker_group is None
            or worker_user.pw_uid != identities.worker_uid
            or worker_user.pw_gid != identities.worker_gid
            or worker_user.pw_dir != DEFAULT_WORKER_HOME
            or worker_user.pw_shell != DEFAULT_WORKER_SHELL
            or uid_owner is None
            or uid_owner.pw_name != identities.worker_user
        ):
            raise RuntimeError("capability worker user identity is not exact")
        supplementary = sorted(
            set(os.getgrouplist(identities.worker_user, identities.worker_gid))
        )
        if supplementary != [identities.worker_gid]:
            raise RuntimeError("capability worker has supplementary authority")

    if worker_user is not None:
        worker_state = "present_exact"
    elif worker_group is not None:
        worker_state = "group_present_user_absent_create_only_slot"
    else:
        worker_state = "absent_create_only_slot"
    client_state = (
        "present_exact" if client_group is not None else "absent_create_only_slot"
    )
    if not allow_create_only_absence and (
        worker_state != "present_exact" or client_state != "present_exact"
    ):
        raise RuntimeError("capability execution principal is absent")

    unsigned = {
        "schema": CAPABILITY_EXECUTION_HOST_IDENTITY_SCHEMA,
        "plan_sha256": plan.sha256,
        "worker": {
            "state": worker_state,
            "user": identities.worker_user,
            "group": identities.worker_group,
            "uid": identities.worker_uid,
            "gid": identities.worker_gid,
            "home": DEFAULT_WORKER_HOME,
            "shell": DEFAULT_WORKER_SHELL,
            "group_members": [] if worker_group is not None else None,
            "supplementary_group_ids": supplementary,
        },
        "socket_client_group": {
            "state": client_state,
            "group": identities.worker_client_group,
            "gid": identities.worker_client_gid,
            "group_members": [] if client_group is not None else None,
        },
        "create_only_eligible": True,
        "secret_material_recorded": False,
    }
    return {**unsigned, "receipt_sha256": _sha256_json(unsigned)}


def ensure_execution_identities_create_only(
    plan: CapabilityCanaryPlan,
    full_plan: FullCanaryPlan,
    *,
    runner: Callable[[Command], subprocess.CompletedProcess[bytes]] = _runner,
    observer: Callable[..., Mapping[str, Any]] = execution_host_identity_receipt,
) -> Mapping[str, Any]:
    """Create only absent worker identities; never rewrite an existing one."""

    _require_root_linux()
    before = observer(plan, full_plan, allow_create_only_absence=True)
    created: list[str] = []
    if before["worker"]["state"] == "absent_create_only_slot":
        _run_checked(
            Command(
                (
                    GROUPADD,
                    "--system",
                    "--gid",
                    str(plan.identities.worker_gid),
                    "--",
                    plan.identities.worker_group,
                )
            ),
            runner=runner,
            label="create capability worker group",
        )
        created.append("worker_group")
    if before["socket_client_group"]["state"] == "absent_create_only_slot":
        _run_checked(
            Command(
                (
                    GROUPADD,
                    "--system",
                    "--gid",
                    str(plan.identities.worker_client_gid),
                    "--",
                    plan.identities.worker_client_group,
                )
            ),
            runner=runner,
            label="create capability worker client group",
        )
        created.append("worker_client_group")
    if before["worker"]["state"] != "present_exact":
        _run_checked(
            Command(
                (
                    USERADD,
                    "--system",
                    "--uid",
                    str(plan.identities.worker_uid),
                    "--gid",
                    plan.identities.worker_group,
                    "--home-dir",
                    DEFAULT_WORKER_HOME,
                    "--no-create-home",
                    "--shell",
                    DEFAULT_WORKER_SHELL,
                    "--",
                    plan.identities.worker_user,
                )
            ),
            runner=runner,
            label="create capability worker user",
        )
        created.append("worker_user")
    after = observer(plan, full_plan, allow_create_only_absence=False)
    unsigned = {
        "schema": CAPABILITY_EXECUTION_IDENTITY_FOUNDATION_SCHEMA,
        "plan_sha256": plan.sha256,
        "before_receipt_sha256": before["receipt_sha256"],
        "after_receipt_sha256": after["receipt_sha256"],
        "created": created,
        "retained_dormant_on_rollback": True,
        "secret_material_recorded": False,
    }
    return {
        **unsigned,
        "receipt_sha256": _sha256_json(unsigned),
        "host_identity": copy.deepcopy(dict(after)),
    }


def _read_browser_userns_sysctl(path: Path) -> int:
    if path not in _BROWSER_USERNS_SYSCTLS.values():
        raise RuntimeError("browser userns sysctl path is not allowlisted")
    before = os.lstat(path)
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or before.st_uid != 0
        or stat.S_IMODE(before.st_mode) & 0o022
    ):
        raise RuntimeError("browser userns sysctl source is unsafe")
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened = os.fstat(descriptor)
        raw = os.read(descriptor, 65)
        extra = os.read(descriptor, 1)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    reachable = os.lstat(path)
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_uid,
        item.st_gid,
    )
    if (
        extra
        or identity(before) != identity(opened)
        or identity(before) != identity(after)
        or identity(before) != identity(reachable)
    ):
        raise RuntimeError("browser userns sysctl changed during observation")
    try:
        text = raw.decode("ascii", errors="strict")
    except UnicodeError as exc:
        raise RuntimeError("browser userns sysctl is not ASCII") from exc
    if re.fullmatch(r"[0-9]+\n?", text) is None:
        raise RuntimeError("browser userns sysctl value is invalid")
    return int(text)


def browser_userns_preflight(
    *, reader: Callable[[Path], int] = _read_browser_userns_sysctl
) -> Mapping[str, Any]:
    values = {
        name: reader(path) for name, path in _BROWSER_USERNS_SYSCTLS.items()
    }
    if (
        values["unprivileged_userns_clone"] != 1
        or values["max_user_namespaces"] <= 0
    ):
        raise RuntimeError("capability browser user namespace sandbox is disabled")
    return {
        "unprivileged_userns_clone": values["unprivileged_userns_clone"],
        "max_user_namespaces": values["max_user_namespaces"],
        "sandbox_required": True,
        "no_sandbox_flag_allowed": False,
        "ready": True,
    }


def worker_systemd252_preflight(
    plan: CapabilityCanaryPlan,
    *,
    allow_create_only_mountpoint_absence: bool = False,
    runner: Callable[[Command], subprocess.CompletedProcess[bytes]] = _runner,
) -> Mapping[str, Any]:
    completed = _run_checked(
        Command((SYSTEMD, "--version"), timeout_seconds=10),
        runner=runner,
        label="systemd isolated-worker tmpfs contract",
    )
    first = completed.stdout.splitlines()[:1]
    match = re.fullmatch(rb"systemd ([0-9]+)(?: .*)?", first[0] if first else b"")
    if (
        match is None
        or int(match.group(1)) < LEASE_TMPFS_MINIMUM_SYSTEMD_VERSION
        or completed.stderr
    ):
        raise RuntimeError("isolated worker requires the systemd252 tmpfs contract")
    try:
        mountpoint = os.lstat(DEFAULT_WORKER_LEASE_BASE)
    except FileNotFoundError:
        if not allow_create_only_mountpoint_absence:
            raise RuntimeError(
                "isolated worker tmpfs mountpoint is absent"
            ) from None
        mountpoint_state = "absent_create_only_slot"
    else:
        if (
            not stat.S_ISDIR(mountpoint.st_mode)
            or stat.S_ISLNK(mountpoint.st_mode)
            or mountpoint.st_uid != 0
            or mountpoint.st_gid != 0
            or stat.S_IMODE(mountpoint.st_mode) != 0o700
        ):
            raise RuntimeError("isolated worker tmpfs mountpoint identity is invalid")
        mountpoint_state = "present_exact"
    unit = render_worker_service_unit(plan)
    directive = (
        f"TemporaryFileSystem={DEFAULT_WORKER_LEASE_BASE}:"
        f"size={SERVICE_GLOBAL_QUOTA_BYTES},"
        f"nr_inodes={SERVICE_TMPFS_INODE_LIMIT},"
        f"mode=0700,uid={plan.identities.worker_uid},"
        f"gid={plan.identities.worker_gid},nodev,nosuid,exec"
    )
    if unit.count(directive) != 1:
        raise RuntimeError("isolated worker tmpfs unit contract drifted")
    return {
        "systemd_major": int(match.group(1)),
        "minimum_systemd_major": LEASE_TMPFS_MINIMUM_SYSTEMD_VERSION,
        "mountpoint": str(DEFAULT_WORKER_LEASE_BASE),
        "mountpoint_state": mountpoint_state,
        "unit_directive_sha256": _sha256_bytes(directive.encode("ascii")),
        "contract": LEASE_TMPFS_PREFLIGHT_CONTRACT,
        "ready": True,
    }


def worker_tmpfs_runtime_preflight(
    plan: CapabilityCanaryPlan,
    state: Mapping[str, Any],
    *,
    mountinfo_reader: Callable[[Path], bytes] = Path.read_bytes,
    path_lstat: Callable[[Path], os.stat_result] = os.lstat,
    path_statvfs: Callable[[Path], os.statvfs_result] = os.statvfs,
) -> Mapping[str, Any]:
    if not _service_live(
        state,
        path=DEFAULT_WORKER_SERVICE_UNIT_PATH,
        service_type="simple",
    ):
        raise RuntimeError("isolated worker service is not live")
    main_pid = state.get("MainPID")
    if type(main_pid) is not int or main_pid <= 1:
        raise RuntimeError("isolated worker MainPID is invalid")
    mountinfo_path = Path(f"/proc/{main_pid}/mountinfo")
    raw = mountinfo_reader(mountinfo_path)
    if not isinstance(raw, bytes) or not 0 < len(raw) <= 2 * 1024 * 1024:
        raise RuntimeError("isolated worker mountinfo is invalid")
    matches: list[tuple[set[str], set[str]]] = []
    for line in raw.decode("ascii", errors="strict").splitlines():
        fields = line.split()
        if "-" not in fields:
            raise RuntimeError("isolated worker mountinfo is malformed")
        separator = fields.index("-")
        if separator < 6 or len(fields) < separator + 4:
            raise RuntimeError("isolated worker mountinfo is malformed")
        if fields[4] == str(DEFAULT_WORKER_LEASE_BASE):
            if fields[separator + 1] != "tmpfs":
                raise RuntimeError("isolated worker lease filesystem is not tmpfs")
            matches.append(
                (
                    set(fields[5].split(",")),
                    set(fields[separator + 3].split(",")),
                )
            )
    if len(matches) != 1:
        raise RuntimeError("isolated worker tmpfs mount is not unique")
    mount_options, super_options = matches[0]
    if not {"rw", "nosuid", "nodev"}.issubset(mount_options) or "noexec" in (
        mount_options | super_options
    ):
        raise RuntimeError("isolated worker tmpfs mount flags drifted")
    process_path = Path(f"/proc/{main_pid}/root") / str(
        DEFAULT_WORKER_LEASE_BASE
    ).lstrip("/")
    item = path_lstat(process_path)
    capacity = path_statvfs(process_path)
    if (
        not stat.S_ISDIR(item.st_mode)
        or stat.S_ISLNK(item.st_mode)
        or item.st_uid != plan.identities.worker_uid
        or item.st_gid != plan.identities.worker_gid
        or stat.S_IMODE(item.st_mode) != 0o700
        or capacity.f_blocks * capacity.f_frsize != SERVICE_GLOBAL_QUOTA_BYTES
        or capacity.f_files != SERVICE_TMPFS_INODE_LIMIT
    ):
        raise RuntimeError("isolated worker tmpfs capacity/identity drifted")
    return {
        "unit": DEFAULT_WORKER_SERVICE_UNIT_NAME,
        "main_pid": main_pid,
        "mountpoint": str(DEFAULT_WORKER_LEASE_BASE),
        "filesystem": "tmpfs",
        "capacity_bytes": SERVICE_GLOBAL_QUOTA_BYTES,
        "inode_limit": SERVICE_TMPFS_INODE_LIMIT,
        "runtime_entry_limit": SERVICE_GLOBAL_QUOTA_ENTRIES,
        "mount_flags": ["nodev", "nosuid", "exec"],
        "contract": LEASE_TMPFS_PREFLIGHT_CONTRACT,
        "ready": True,
    }


def browser_principal_version_smoke(
    plan: CapabilityCanaryPlan,
    *,
    runner: Callable[[Command], subprocess.CompletedProcess[bytes]] = _runner,
) -> Mapping[str, Any]:
    completed = _run_checked(
        Command(
            (
                RUNUSER,
                "--user",
                plan.identities.browser_user,
                "--group",
                plan.identities.browser_group,
                "--",
                str(plan.browser_executable),
                "--version",
            ),
            timeout_seconds=20,
        ),
        runner=runner,
        label="capability browser principal version smoke",
    )
    expected = f"Google Chrome for Testing {RELEASE_CHROME_VERSION}\n".encode(
        "ascii"
    )
    if completed.stdout != expected or completed.stderr:
        raise RuntimeError("capability browser principal version is not exact")
    return {
        "user": plan.identities.browser_user,
        "group": plan.identities.browser_group,
        "uid": plan.identities.browser_uid,
        "gid": plan.identities.browser_gid,
        "version": RELEASE_CHROME_VERSION,
        "sandbox_required": True,
        "ready": True,
    }


def browser_service_runtime_preflight(
    plan: CapabilityCanaryPlan,
    state: Mapping[str, Any],
    *,
    proc_stat: Callable[[str], os.stat_result] = os.stat,
    socket_lstat: Callable[[Path], os.stat_result] = os.lstat,
    listener_paths: Callable[[int], set[str]] | None = None,
) -> Mapping[str, Any]:
    """Bind controller socket readiness to the exact service principal."""

    if not _service_live(
        state,
        path=DEFAULT_BROWSER_UNIT_PATH,
        service_type="notify",
    ):
        raise RuntimeError("capability browser controller service is not live")
    main_pid = state.get("MainPID")
    if type(main_pid) is not int or main_pid <= 1:
        raise RuntimeError("capability browser controller MainPID is invalid")
    process = proc_stat(f"/proc/{main_pid}")
    socket_state = socket_lstat(DEFAULT_BROWSER_SOCKET)
    if (
        process.st_uid != plan.identities.browser_uid
        or process.st_gid != plan.identities.browser_gid
        or not stat.S_ISSOCK(socket_state.st_mode)
        or socket_state.st_uid != plan.identities.browser_uid
        or socket_state.st_gid != plan.identities.browser_gid
        or stat.S_IMODE(socket_state.st_mode) != 0o660
    ):
        raise RuntimeError("capability browser controller identity drifted")
    if listener_paths is None:
        from gateway.canonical_writer_root_collector import (
            _unix_listener_paths_for_pid,
        )

        listener_paths = _unix_listener_paths_for_pid
    if str(DEFAULT_BROWSER_SOCKET) not in listener_paths(main_pid):
        raise RuntimeError("capability browser controller does not own its socket")
    return {
        "unit": DEFAULT_BROWSER_UNIT_NAME,
        "main_pid": main_pid,
        "process_uid": process.st_uid,
        "process_gid": process.st_gid,
        "socket_path": str(DEFAULT_BROWSER_SOCKET),
        "socket_device": socket_state.st_dev,
        "socket_inode": socket_state.st_ino,
        "transport": "authenticated_af_unix",
        "ready": True,
    }


def attest_capability_execution_readiness(
    plan: CapabilityCanaryPlan,
) -> Mapping[str, Any]:
    """Run both real gateway-side execution probes before systemd READY."""

    if os.geteuid() != plan.identities.gateway_uid or os.getegid() != plan.identities.gateway_gid:  # windows-footgun: ok — Linux production/canary boundary
        raise RuntimeError("capability execution readiness must run as the gateway")
    worker = attest_isolated_worker_execution(
        socket_path=DEFAULT_WORKER_SOCKET,
        server_uid=plan.identities.worker_uid,
        server_gid=plan.identities.worker_gid,
        socket_uid=0,
        socket_gid=plan.identities.worker_client_gid,
        revision=plan.revision,
        config_sha256=plan.worker_config_sha256,
        timeout_seconds=10,
    )
    browser = attest_browser_controller_execution(
        client_config=capability_browser_controller_client_config(plan),
        revision=plan.revision,
        config_sha256=plan.browser_config_sha256,
    )
    worker_fields = {
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
    }
    browser_fields = {
        "schema",
        "session_identity_sha256",
        "socket_path",
        "server_uid",
        "command_round_trip",
        "secret_material_recorded",
    }
    expected_worker_identity = hashlib.sha256(
        b"muncho-worker-readiness-v1\x00"
        + plan.revision.encode("ascii")
        + b"\x00"
        + plan.worker_config_sha256.encode("ascii")
    ).hexdigest()
    expected_browser_identity = hashlib.sha256(
        b"muncho-browser-readiness-v1\x00"
        + plan.revision.encode("ascii")
        + b"\x00"
        + plan.browser_config_sha256.encode("ascii")
    ).hexdigest()
    if (
        not isinstance(worker, Mapping)
        or set(worker) != worker_fields
        or worker.get("schema") != WORKER_RECEIPT_SCHEMA
        or worker.get("lease_identity_sha256") != expected_worker_identity
        or worker.get("socket_path") != str(DEFAULT_WORKER_SOCKET)
        or worker.get("server_uid") != plan.identities.worker_uid
        or worker.get("server_gid") != plan.identities.worker_gid
        or worker.get("socket_uid") != 0
        or worker.get("socket_gid") != plan.identities.worker_client_gid
        or worker.get("execution_round_trip") is not True
        or worker.get("output_sha256")
        != hashlib.sha256(b"MUNCHO_ISOLATED_WORKER_READY\n").hexdigest()
        or worker.get("secret_material_recorded") is not False
        or not isinstance(browser, Mapping)
        or set(browser) != browser_fields
        or browser.get("schema") != BROWSER_RECEIPT_SCHEMA
        or browser.get("session_identity_sha256") != expected_browser_identity
        or browser.get("socket_path") != str(DEFAULT_BROWSER_SOCKET)
        or browser.get("server_uid") != plan.identities.browser_uid
        or browser.get("command_round_trip") is not True
        or browser.get("secret_material_recorded") is not False
    ):
        raise RuntimeError("capability execution readiness receipt is invalid")
    unsigned = {
        "schema": CAPABILITY_EXECUTION_READINESS_SCHEMA,
        "plan_sha256": plan.sha256,
        "isolated_worker": copy.deepcopy(dict(worker)),
        "browser_controller": copy.deepcopy(dict(browser)),
        "secret_material_recorded": False,
    }
    return {**unsigned, "receipt_sha256": _sha256_json(unsigned)}


def _execution_readiness_as_gateway(
    plan: CapabilityCanaryPlan,
    *,
    runner: Callable[[Command], subprocess.CompletedProcess[bytes]] = _runner,
) -> Mapping[str, Any]:
    """Run the same in-process probes under the exact gateway credentials."""

    encoded_plan = base64.urlsafe_b64encode(_canonical_bytes(plan.to_mapping()))
    code = (
        "import base64,json,sys;"
        "sys.path.insert(0,sys.argv[1]);"
        "from gateway.canonical_capability_canary_runtime import "
        "CapabilityCanaryPlan,attest_capability_execution_readiness;"
        "plan=CapabilityCanaryPlan.from_mapping(json.loads("
        "base64.urlsafe_b64decode(sys.argv[2])));"
        "value=attest_capability_execution_readiness(plan);"
        "print(json.dumps(value,ensure_ascii=True,sort_keys=True,"
        "separators=(',',':'),allow_nan=False))"
    )
    completed = _run_checked(
        Command(
            (
                RUNUSER,
                "--user",
                plan.identities.gateway_user,
                "--group",
                plan.identities.gateway_group,
                "--supp-group",
                plan.identities.worker_client_group,
                "--supp-group",
                plan.identities.browser_group,
                "--",
                str(plan.interpreter),
                "-I",
                "-c",
                code,
                str(plan.release_root),
                encoded_plan.decode("ascii"),
            ),
            timeout_seconds=180,
        ),
        runner=runner,
        label="capability execution readiness as gateway",
    )
    if completed.stderr or not completed.stdout.endswith(b"\n"):
        raise RuntimeError("capability execution readiness output is invalid")
    value = _decode_json(
        completed.stdout[:-1], label="capability execution readiness"
    )
    if (
        completed.stdout != _canonical_bytes(value) + b"\n"
        or set(value)
        != {
            "schema",
            "plan_sha256",
            "isolated_worker",
            "browser_controller",
            "secret_material_recorded",
            "receipt_sha256",
        }
        or value.get("schema") != CAPABILITY_EXECUTION_READINESS_SCHEMA
        or value.get("plan_sha256") != plan.sha256
        or value.get("secret_material_recorded") is not False
        or value.get("receipt_sha256")
        != _sha256_json(
            {key: item for key, item in value.items() if key != "receipt_sha256"}
        )
    ):
        raise RuntimeError("capability execution readiness receipt drifted")
    return value


class CapabilityCanaryPreflightError(RuntimeError):
    def __init__(self, report: Mapping[str, Any]) -> None:
        self.report = copy.deepcopy(dict(report))
        super().__init__("capability-canary preflight blocked")


def collect_capability_service_state(
    unit: str,
    *,
    runner: Callable[[Command], subprocess.CompletedProcess[bytes]] = _runner,
) -> Mapping[str, Any]:
    """Collect exact systemd state for the fixed capability service set."""

    allowed = {
        EDGE_UNIT_NAME,
        WRITER_UNIT_NAME,
        GATEWAY_UNIT_NAME,
        PHASE_B_READINESS_UNIT_NAME,
        MAC_OPS_UNIT_NAME,
        DEFAULT_BROWSER_UNIT_NAME,
        DEFAULT_DISCORD_CONNECTOR_UNIT,
        DEFAULT_WORKER_SOCKET_UNIT_NAME,
        DEFAULT_WORKER_SERVICE_UNIT_NAME,
        BITRIX_OPERATIONAL_EDGE_UNIT,
        *CAPABILITY_PRODUCER_SERVICE_UNITS.values(),
    }
    if unit not in allowed:
        raise ValueError("capability-canary unit is not allowlisted")
    completed = _run_checked(
        Command(
            (
                SYSTEMCTL,
                "show",
                *(f"--property={name}" for name in _SERVICE_PROPERTIES),
                "--",
                unit,
            )
        ),
        runner=runner,
        label=f"collect {unit}",
    )
    try:
        text = completed.stdout.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise RuntimeError("capability service state is not UTF-8") from exc
    values: dict[str, str] = {}
    for line in text.splitlines():
        if "=" not in line:
            raise RuntimeError("capability service state is malformed")
        name, item = line.split("=", 1)
        if name in values:
            raise RuntimeError("capability service state is ambiguous")
        values[name] = item
    if set(values) != set(_SERVICE_PROPERTIES):
        raise RuntimeError("capability service state fields are not exact")
    try:
        main_pid = int(values.pop("MainPID"))
    except ValueError as exc:
        raise RuntimeError("capability service MainPID is invalid") from exc
    return {**values, "MainPID": main_pid}


def _service_stopped(state: Mapping[str, Any]) -> bool:
    return (
        state.get("LoadState") in {"loaded", "not-found"}
        and state.get("ActiveState") in {"inactive", "failed"}
        and state.get("MainPID") == 0
        and state.get("UnitFileState") in {"disabled", ""}
        and state.get("DropInPaths") in {"", "[]"}
    )


def build_capability_stop_proof(
    plan: CapabilityCanaryPlan,
    services: Mapping[str, Mapping[str, Any]],
    *,
    stop_order: Sequence[str] = CAPABILITY_STOP_ORDER,
    observed_at_unix: int | None = None,
) -> Mapping[str, Any]:
    """Seal an exact, secret-free proof that every capability unit is stopped."""

    if tuple(stop_order) != CAPABILITY_STOP_ORDER:
        raise ValueError("capability stop proof order is not exact")
    if set(services) != set(CAPABILITY_STOP_ORDER):
        raise ValueError("capability stop proof service inventory is not exact")
    if not all(_service_stopped(services[unit]) for unit in CAPABILITY_STOP_ORDER):
        raise RuntimeError("capability stop proof contains a live service")
    observed = int(time.time()) if observed_at_unix is None else observed_at_unix
    if type(observed) is not int or observed < 0:
        raise ValueError("capability stop proof time is invalid")
    service_state = {
        unit: copy.deepcopy(dict(services[unit]))
        for unit in CAPABILITY_STOP_ORDER
    }
    unsigned = {
        "schema": CAPABILITY_SERVICE_STOP_PROOF_SCHEMA,
        "plan_sha256": plan.sha256,
        "stop_order": list(CAPABILITY_STOP_ORDER),
        "services_state_sha256": _sha256_json(service_state),
        "all_services_stopped": True,
        "observed_at_unix": observed,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    return {
        **unsigned,
        "stop_proof_sha256": _sha256_json(unsigned),
    }


def _producer_credential_inaccessibility_contract() -> Mapping[str, Any]:
    from gateway.canonical_capability_canary_producer_units import (
        PRODUCER_INACCESSIBLE_CREDENTIAL_PATHS,
    )

    return {
        "paths": list(PRODUCER_INACCESSIBLE_CREDENTIAL_PATHS),
        "applies_to_roles": list(CAPABILITY_PRODUCER_ROLES),
        "unit_hash_bound": True,
        "cleanup_observer_has_no_credential_read_access": True,
    }


def build_credential_consumer_stop_proof(
    plan: CapabilityCanaryPlan,
    services: Mapping[str, Mapping[str, Any]],
    *,
    producer_foundation: Mapping[str, Any],
    observed_at_unix: int | None = None,
) -> Mapping[str, Any]:
    """Prove every credential reader stopped while the isolated signer lives."""

    if set(services) != set(CAPABILITY_STOP_ORDER):
        raise ValueError("credential-consumer proof service inventory is not exact")
    if not all(
        _service_stopped(services[unit])
        for unit in CAPABILITY_PRE_CLEANUP_STOP_ORDER
    ):
        raise RuntimeError("credential-consumer proof contains a live consumer")
    observer_state = services[CAPABILITY_OBSERVER_UNIT]
    observer_path = Path("/etc/systemd/system") / CAPABILITY_OBSERVER_UNIT
    if not _service_live(observer_state, path=observer_path, service_type="simple"):
        raise RuntimeError("cleanup observer is not an exact live signer")
    if (
        not isinstance(producer_foundation, Mapping)
        or producer_foundation.get("ready") is not True
        or producer_foundation.get("mutation_performed") is not False
        or producer_foundation.get("revision") != plan.revision
    ):
        raise RuntimeError("credential-consumer proof lacks producer foundation")
    foundation_sha256 = _digest(
        producer_foundation.get("foundation_sha256"),
        "producer foundation",
    )
    manifest_sha256 = _digest(
        producer_foundation.get("unit_bundle_manifest_sha256"),
        "producer unit manifest",
    )
    observed = int(time.time()) if observed_at_unix is None else observed_at_unix
    if type(observed) is not int or observed < 0:
        raise ValueError("credential-consumer proof time is invalid")
    non_observer_state = {
        unit: copy.deepcopy(dict(services[unit]))
        for unit in CAPABILITY_PRE_CLEANUP_STOP_ORDER
    }
    contract_sha256 = _sha256_json(
        _producer_credential_inaccessibility_contract()
    )
    unsigned = {
        "schema": CAPABILITY_CREDENTIAL_CONSUMER_STOP_PROOF_SCHEMA,
        "plan_sha256": plan.sha256,
        "non_observer_stop_order": list(CAPABILITY_PRE_CLEANUP_STOP_ORDER),
        "non_observer_services_state_sha256": _sha256_json(
            non_observer_state
        ),
        "all_credential_consumers_stopped": True,
        "observer_service_unit": CAPABILITY_OBSERVER_UNIT,
        "observer_state_sha256": _sha256_json(dict(observer_state)),
        "observer_live_signing_only": True,
        "observer_credential_read_access": False,
        "producer_foundation_sha256": foundation_sha256,
        "unit_bundle_manifest_sha256": manifest_sha256,
        "credential_inaccessibility_contract_sha256": contract_sha256,
        "observed_at_unix": observed,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    return {**unsigned, "stop_proof_sha256": _sha256_json(unsigned)}


def _validate_capability_stop_proof(
    plan: CapabilityCanaryPlan | None,
    proof: Mapping[str, Any],
    *,
    installed_at_unix: int,
    now_unix: int,
) -> Mapping[str, Any]:
    if proof.get("schema") == CAPABILITY_CREDENTIAL_CONSUMER_STOP_PROOF_SCHEMA:
        expected = {
            "schema",
            "plan_sha256",
            "non_observer_stop_order",
            "non_observer_services_state_sha256",
            "all_credential_consumers_stopped",
            "observer_service_unit",
            "observer_state_sha256",
            "observer_live_signing_only",
            "observer_credential_read_access",
            "producer_foundation_sha256",
            "unit_bundle_manifest_sha256",
            "credential_inaccessibility_contract_sha256",
            "observed_at_unix",
            "secret_material_recorded",
            "secret_digest_recorded",
            "stop_proof_sha256",
        }
        value = _strict_mapping(
            proof, expected, "capability credential-consumer stop proof"
        )
        unsigned = {
            key: copy.deepcopy(item)
            for key, item in value.items()
            if key != "stop_proof_sha256"
        }
        if (
            value["plan_sha256"]
            != (plan.sha256 if plan is not None else value["plan_sha256"])
            or value["non_observer_stop_order"]
            != list(CAPABILITY_PRE_CLEANUP_STOP_ORDER)
            or value["all_credential_consumers_stopped"] is not True
            or value["observer_service_unit"] != CAPABILITY_OBSERVER_UNIT
            or value["observer_live_signing_only"] is not True
            or value["observer_credential_read_access"] is not False
            or value["secret_material_recorded"] is not False
            or value["secret_digest_recorded"] is not False
            or value["stop_proof_sha256"] != _sha256_json(unsigned)
            or type(value["observed_at_unix"]) is not int
            or not installed_at_unix <= value["observed_at_unix"] <= now_unix
        ):
            raise PermissionError(
                "credential retirement lacks an exact post-install consumer stop proof"
            )
        for field in (
            "plan_sha256",
            "non_observer_services_state_sha256",
            "observer_state_sha256",
            "producer_foundation_sha256",
            "unit_bundle_manifest_sha256",
            "credential_inaccessibility_contract_sha256",
        ):
            _digest(value[field], f"credential-consumer proof {field}")
        return copy.deepcopy(dict(value))

    expected = {
        "schema",
        "plan_sha256",
        "stop_order",
        "services_state_sha256",
        "all_services_stopped",
        "observed_at_unix",
        "secret_material_recorded",
        "secret_digest_recorded",
        "stop_proof_sha256",
    }
    value = _strict_mapping(proof, expected, "capability service stop proof")
    unsigned = {
        key: copy.deepcopy(item)
        for key, item in value.items()
        if key != "stop_proof_sha256"
    }
    if (
        value["schema"] != CAPABILITY_SERVICE_STOP_PROOF_SCHEMA
        or value["plan_sha256"]
        != (plan.sha256 if plan is not None else value["plan_sha256"])
        or value["stop_order"] != list(CAPABILITY_STOP_ORDER)
        or value["all_services_stopped"] is not True
        or value["secret_material_recorded"] is not False
        or value["secret_digest_recorded"] is not False
        or value["stop_proof_sha256"] != _sha256_json(unsigned)
        or type(value["observed_at_unix"]) is not int
        or not installed_at_unix <= value["observed_at_unix"] <= now_unix
    ):
        raise PermissionError("credential retirement lacks an exact post-install stop proof")
    _digest(value["plan_sha256"], "capability stop proof plan")
    _digest(value["services_state_sha256"], "capability stop proof service state")
    return copy.deepcopy(dict(value))


def _service_live(
    state: Mapping[str, Any],
    *,
    path: Path,
    service_type: str,
) -> bool:
    return (
        state.get("LoadState") == "loaded"
        and state.get("ActiveState") == "active"
        and state.get("SubState") == "running"
        and type(state.get("MainPID")) is int
        and state["MainPID"] > 1
        and state.get("FragmentPath") == str(path)
        and state.get("UnitFileState") in {"disabled", ""}
        and state.get("DropInPaths") in {"", "[]"}
        and state.get("Type") == service_type
        and (
            state.get("NotifyAccess") == "main"
            if service_type == "notify"
            else state.get("NotifyAccess") in {"", "none"}
        )
    )


def _oneshot_live(state: Mapping[str, Any], *, path: Path) -> bool:
    return (
        state.get("LoadState") == "loaded"
        and state.get("ActiveState") == "active"
        and state.get("SubState") == "exited"
        and state.get("MainPID") == 0
        and state.get("FragmentPath") == str(path)
        and state.get("UnitFileState") in {"disabled", ""}
        and state.get("DropInPaths") in {"", "[]"}
        and state.get("Type") == "oneshot"
    )


def _socket_live(state: Mapping[str, Any], *, path: Path) -> bool:
    return (
        state.get("LoadState") == "loaded"
        and state.get("ActiveState") == "active"
        and state.get("SubState") == "listening"
        and state.get("MainPID") == 0
        and state.get("FragmentPath") == str(path)
        and state.get("UnitFileState") in {"disabled", ""}
        and state.get("DropInPaths") in {"", "[]"}
    )


def _active_lease_receipt(
    plan: CapabilityCanaryPlan,
    *,
    kind: str,
    target: Path,
    journal: Path,
    now_unix: int | None = None,
) -> Mapping[str, Any]:
    state = _active_install_state(
        kind=kind,
        target=target,
        journal=journal,
        plan_sha256=plan.sha256,
    )
    prior = state["install_receipt"]
    if (
        prior.get("schema") != CAPABILITY_LEASE_RECEIPT_SCHEMA
        or prior.get("operation") != "install"
        or prior.get("state") != "provisioned"
        or prior.get("kind") != kind
        or prior.get("credential_binding") != _CREDENTIAL_BINDING_BY_KIND[kind]
        or prior.get("plan_sha256") != plan.sha256
        or prior.get("full_canary_plan_sha256") != plan.full_canary_plan_sha256
        or prior.get("target_path") != str(target)
        or type(prior.get("expires_at_unix")) is not int
        or prior["expires_at_unix"] < (int(time.time()) if now_unix is None else now_unix) + 30
    ):
        raise RuntimeError("required capability credential lease is not active")
    item = os.lstat(target)
    if (
        not stat.S_ISREG(item.st_mode)
        or stat.S_ISLNK(item.st_mode)
        or item.st_nlink != 1
        or not _receipt_matches_target(prior, item)
        or item.st_size <= 0
    ):
        raise RuntimeError("capability credential lease identity drifted")
    return {
        "kind": kind,
        "lease_id": prior["lease_id"],
        "expires_at_unix": prior["expires_at_unix"],
        "target_path": str(target),
        "target_device": item.st_dev,
        "target_inode": item.st_ino,
        "install_receipt_path": prior["receipt_path"],
        "install_receipt_sha256": prior["receipt_sha256"],
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
        "ready": True,
    }


def _mac_ops_runtime_preflight(
    plan: CapabilityCanaryPlan,
    state: Mapping[str, Any],
) -> Mapping[str, Any]:
    if not _service_live(
        state,
        path=DEFAULT_MAC_OPS_UNIT_PATH,
        service_type="simple",
    ):
        raise RuntimeError("Mac operations edge service is not live")
    item = os.lstat(DEFAULT_MAC_OPS_SOCKET)
    if (
        not stat.S_ISSOCK(item.st_mode)
        or item.st_uid != plan.identities.mac_ops_uid
        or item.st_gid != plan.identities.socket_client_gid
        or stat.S_IMODE(item.st_mode) != 0o660
    ):
        raise RuntimeError("Mac operations edge socket identity drifted")
    from gateway.canonical_writer_root_collector import _unix_listener_paths_for_pid

    if str(DEFAULT_MAC_OPS_SOCKET) not in _unix_listener_paths_for_pid(
        int(state["MainPID"])
    ):
        raise RuntimeError("Mac operations edge MainPID does not own its socket")
    return {
        "unit": MAC_OPS_UNIT_NAME,
        "main_pid": state["MainPID"],
        "socket_path": str(DEFAULT_MAC_OPS_SOCKET),
        "socket_device": item.st_dev,
        "socket_inode": item.st_ino,
        "service_identity_sha256": plan.mac_ops_service_identity_sha256,
        "ready": True,
    }


def _bitrix_runtime_preflight(
    plan: CapabilityCanaryPlan,
    full_plan: FullCanaryPlan,
    state: Mapping[str, Any],
) -> Mapping[str, Any]:
    if not _service_live(
        state,
        path=DEFAULT_BITRIX_UNIT_PATH,
        service_type="simple",
    ):
        raise RuntimeError("Bitrix operational edge service is not live")
    config_raw, _ = _read_exact_file(
        DEFAULT_BITRIX_CONFIG_PATH,
        maximum=256 * 1024,
        uid=0,
        gid=0,
        mode=0o400,
    )
    config = _decode_json(config_raw.rstrip(b"\n"), label="Bitrix service config")
    expected_read_peers = sorted(
        [plan.identities.mac_ops_uid, full_plan.identities.writer_uid]
    )
    if (
        _sha256_bytes(config_raw)
        != plan.bitrix_operational_edge_rendered_config_sha256
        or config.get("allowed_read_peer_uids") != expected_read_peers
        or config.get("mutation_peer_uid") != full_plan.identities.writer_uid
        or config.get("service_uid")
        != plan.identities.bitrix_operational_edge_uid
        or config.get("service_gid")
        != plan.identities.bitrix_operational_edge_gid
        or config.get("socket_gid")
        != plan.identities.bitrix_operational_edge_client_gid
    ):
        raise RuntimeError("Bitrix operational edge config topology drifted")
    item = os.lstat(DEFAULT_BITRIX_SOCKET_PATH)
    if (
        not stat.S_ISSOCK(item.st_mode)
        or item.st_uid != plan.identities.bitrix_operational_edge_uid
        or item.st_gid != plan.identities.bitrix_operational_edge_client_gid
        or stat.S_IMODE(item.st_mode) != 0o660
    ):
        raise RuntimeError("Bitrix operational edge socket identity drifted")
    from gateway.canonical_writer_root_collector import _unix_listener_paths_for_pid

    if str(DEFAULT_BITRIX_SOCKET_PATH) not in _unix_listener_paths_for_pid(
        int(state["MainPID"])
    ):
        raise RuntimeError("Bitrix operational edge MainPID does not own its socket")
    return {
        "unit": BITRIX_OPERATIONAL_EDGE_UNIT,
        "main_pid": state["MainPID"],
        "socket_path": str(DEFAULT_BITRIX_SOCKET_PATH),
        "socket_device": item.st_dev,
        "socket_inode": item.st_ino,
        "service_identity_sha256": (
            plan.bitrix_operational_edge_service_identity_sha256
        ),
        "allowed_read_peer_uids": sorted(
            expected_read_peers
        ),
        "mutation_peer_uid": full_plan.identities.writer_uid,
        "ready": True,
    }


def _routeback_credential_file_metadata(
    full_plan: FullCanaryPlan,
) -> Mapping[str, int]:
    """Return secret-free identity metadata for the sealed credential file."""

    if not isinstance(full_plan, FullCanaryPlan):
        raise TypeError("sealed full-canary plan is required")
    if DEFAULT_EDGE_TOKEN_PATH.parent != DEFAULT_EDGE_TOKEN_DIRECTORY:
        raise RuntimeError("Discord route-back credential path drifted")
    try:
        resolved_directory = DEFAULT_EDGE_TOKEN_DIRECTORY.resolve(strict=True)
        directory = os.lstat(DEFAULT_EDGE_TOKEN_DIRECTORY)
        credential = os.lstat(DEFAULT_EDGE_TOKEN_PATH)
    except OSError:
        raise RuntimeError(
            "Discord route-back credential boundary is unavailable"
        ) from None
    if (
        resolved_directory != DEFAULT_EDGE_TOKEN_DIRECTORY
        or stat.S_ISLNK(directory.st_mode)
        or not stat.S_ISDIR(directory.st_mode)
        or directory.st_uid != 0
        or directory.st_mode & 0o022
        or full_plan.identities.edge_uid == 0
    ):
        raise RuntimeError(
            "Discord route-back credential directory is writable by the edge"
        )
    if (
        stat.S_ISLNK(credential.st_mode)
        or not stat.S_ISREG(credential.st_mode)
        or credential.st_nlink != 1
        or credential.st_uid != full_plan.identities.edge_uid
        or credential.st_gid != full_plan.identities.edge_gid
        or stat.S_IMODE(credential.st_mode) != 0o400
        or not 0 < credential.st_size <= _MAX_ROUTEBACK_CREDENTIAL_BYTES
    ):
        raise RuntimeError("Discord route-back credential identity is invalid")
    return {
        "device": credential.st_dev,
        "inode": credential.st_ino,
        "uid": credential.st_uid,
        "gid": credential.st_gid,
        "mode": stat.S_IMODE(credential.st_mode),
        "size": credential.st_size,
        "mtime_ns": credential.st_mtime_ns,
        "ctime_ns": credential.st_ctime_ns,
    }


def _attest_live_routeback_bot_identity(
    plan: CapabilityCanaryPlan,
    full_plan: FullCanaryPlan,
    *,
    adapter_factory: Callable[..., Any] = (
        DiscordRestEdgeAdapter.from_credential_file
    ),
    now_unix: int | None = None,
) -> Mapping[str, Any]:
    """Bind the sealed route-back credential to its live Discord bot ID.

    This is a fixed mechanical identity check.  It performs only Discord's
    read-only ``GET /users/@me`` through the existing strict credential
    loader.  Neither the credential nor any digest derived from its content is
    returned or recorded; only a digest of stable filesystem identity metadata
    is retained for the start/readiness TOCTOU checks.
    """

    if not isinstance(plan, CapabilityCanaryPlan) or not isinstance(
        full_plan, FullCanaryPlan
    ):
        raise TypeError("sealed capability and full-canary plans are required")
    validate_plan_against_full(plan, full_plan)
    if not callable(adapter_factory):
        raise TypeError("Discord route-back adapter factory must be callable")

    metadata_before = _routeback_credential_file_metadata(full_plan)
    adapter = adapter_factory(
        DEFAULT_EDGE_TOKEN_PATH,
        credentials_directory=DEFAULT_EDGE_TOKEN_DIRECTORY,
        expected_owner_uid=full_plan.identities.edge_uid,
        timeout_seconds=_ROUTEBACK_BOT_IDENTITY_TIMEOUT_SECONDS,
    )
    try:
        # The adapter deliberately keeps the fixed REST client private.  This
        # gate reaches only its already bounded current-user endpoint and does
        # not expose a generic request surface to the capability runtime.
        current_user = adapter._api.current_user(  # noqa: SLF001
            timeout_seconds=_ROUTEBACK_BOT_IDENTITY_TIMEOUT_SECONDS,
        )
    finally:
        adapter.close()

    metadata_after = _routeback_credential_file_metadata(full_plan)
    if metadata_after != metadata_before:
        raise RuntimeError(
            "Discord route-back credential changed during identity readback"
        )

    if not isinstance(current_user, Mapping):
        raise RuntimeError("Discord route-back identity response is invalid")
    observed_bot_user_id = _snowflake_id(
        current_user.get("id"), "live route-back bot identity"
    )
    if current_user.get("bot") is not True:
        raise RuntimeError("Discord route-back credential is not a bot identity")
    if observed_bot_user_id != plan.routeback_bot_user_id:
        raise RuntimeError(
            "live Discord route-back bot identity does not match the sealed plan"
        )
    if len(
        {
            observed_bot_user_id,
            plan.connector_bot_user_id,
            PRODUCTION_DISCORD_BOT_USER_ID,
        }
    ) != 3:
        raise RuntimeError(
            "live Discord route-back bot identity is not isolated"
        )

    unsigned = {
        "schema": CAPABILITY_ROUTEBACK_BOT_IDENTITY_SCHEMA,
        "plan_sha256": plan.sha256,
        "full_canary_plan_sha256": full_plan.sha256,
        "live_bot_user_id": observed_bot_user_id,
        "planned_routeback_bot_user_id": plan.routeback_bot_user_id,
        "connector_bot_user_id": plan.connector_bot_user_id,
        "production_bot_user_id": PRODUCTION_DISCORD_BOT_USER_ID,
        "provenance": {
            "source": "discord_rest_api_v10_current_user",
            "http_method": "GET",
            "resource": "/users/@me",
            "credential_boundary": "sealed_routeback_credential_file",
        },
        "pairwise_distinct": True,
        "credential_file_metadata_sha256": _sha256_json(metadata_before),
        "observed_at_unix": int(time.time()) if now_unix is None else now_unix,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    return {**unsigned, "attestation_sha256": _sha256_json(unsigned)}


def _require_routeback_credential_binding(
    plan: CapabilityCanaryPlan,
    full_plan: FullCanaryPlan,
    identity: Mapping[str, Any],
) -> Mapping[str, int]:
    """Require the same secret-free credential object bound by readback."""

    if not isinstance(identity, Mapping):
        raise RuntimeError("Discord route-back identity attestation is invalid")
    unsigned = {
        key: copy.deepcopy(value)
        for key, value in identity.items()
        if key != "attestation_sha256"
    }
    if (
        identity.get("schema") != CAPABILITY_ROUTEBACK_BOT_IDENTITY_SCHEMA
        or identity.get("plan_sha256") != plan.sha256
        or identity.get("full_canary_plan_sha256") != full_plan.sha256
        or identity.get("live_bot_user_id") != plan.routeback_bot_user_id
        or identity.get("planned_routeback_bot_user_id")
        != plan.routeback_bot_user_id
        or identity.get("connector_bot_user_id") != plan.connector_bot_user_id
        or identity.get("production_bot_user_id")
        != PRODUCTION_DISCORD_BOT_USER_ID
        or identity.get("pairwise_distinct") is not True
        or identity.get("secret_material_recorded") is not False
        or identity.get("secret_digest_recorded") is not False
        or identity.get("attestation_sha256") != _sha256_json(unsigned)
    ):
        raise RuntimeError("Discord route-back identity attestation drifted")
    expected_metadata_sha256 = _digest(
        identity.get("credential_file_metadata_sha256"),
        "Discord route-back credential file metadata",
    )
    current = _routeback_credential_file_metadata(full_plan)
    if _sha256_json(current) != expected_metadata_sha256:
        raise RuntimeError(
            "Discord route-back credential object changed after identity readback"
        )
    return current


def _connector_runtime_preflight(
    plan: CapabilityCanaryPlan,
    state: Mapping[str, Any],
) -> Mapping[str, Any]:
    if not _service_live(
        state,
        path=DEFAULT_CONNECTOR_UNIT_PATH,
        service_type="notify",
    ):
        raise RuntimeError("Discord connector service is not live")
    config = load_connector_config(DEFAULT_CONNECTOR_CONFIG)
    receipt = load_connector_readiness(config, DEFAULT_CONNECTOR_READINESS)
    history_reader = config.canary_history_reader
    if (
        receipt.get("schema") != CONNECTOR_READINESS_SCHEMA
        or receipt.get("main_pid") != state.get("MainPID")
        or receipt.get("operation_class") != _DISCORD_CONNECTOR_OPERATION_CLASS
        or receipt.get("config_sha256") != plan.connector_config_sha256
        or receipt.get("allowed_guild_ids")
        != list(plan.connector_allowed_guild_ids)
        or receipt.get("allowed_channel_ids")
        != list(plan.connector_allowed_channel_ids)
        or receipt.get("allowed_user_ids")
        != list(plan.connector_allowed_user_ids)
        or receipt.get("discord", {}).get(
            "reviewed_cron_history_targets_sha256"
        )
        != _sha256_json({})
        or history_reader is None
        or receipt.get("canary_history_reader")
        != history_reader.readiness_mapping()
        or receipt.get("discord", {}).get("bot_user_id")
        != plan.connector_bot_user_id
        or plan.connector_bot_user_id == plan.routeback_bot_user_id
        or plan.connector_bot_user_id == PRODUCTION_DISCORD_BOT_USER_ID
        or plan.routeback_bot_user_id == PRODUCTION_DISCORD_BOT_USER_ID
        or state.get("StatusText")
        != f"{CONNECTOR_READINESS_SCHEMA}:{receipt['receipt_sha256']}"
    ):
        raise RuntimeError("Discord connector readiness drifted")
    from gateway.canonical_writer_root_collector import _unix_listener_paths_for_pid

    if str(DEFAULT_DISCORD_CONNECTOR_SOCKET) not in _unix_listener_paths_for_pid(
        int(state["MainPID"])
    ):
        raise RuntimeError("Discord connector MainPID does not own its socket")
    target_proofs = receipt["discord"].get("public_target_proofs")
    if (
        not isinstance(target_proofs, list)
        or len(target_proofs) != len(plan.connector_allowed_channel_ids)
    ):
        raise RuntimeError("Discord connector public target proofs drifted")
    return {
        "unit": DEFAULT_DISCORD_CONNECTOR_UNIT,
        "main_pid": state["MainPID"],
        "receipt_sha256": receipt["receipt_sha256"],
        "discord_gateway_ready": receipt["discord"]["discord_gateway_ready"],
        "dm_messages": receipt["discord"]["dm_messages"],
        "reviewed_cron_history_targets_sha256": receipt["discord"][
            "reviewed_cron_history_targets_sha256"
        ],
        "canary_history_reader_sha256": _sha256_json(
            receipt["canary_history_reader"]
        ),
        "public_target_proof_count": len(target_proofs),
        "public_target_proofs_sha256": _sha256_json(target_proofs),
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
        "ready": True,
    }


def _connector_cleanup_snapshot_is_safe(value: Mapping[str, Any]) -> bool:
    events = value.get("event_state_counts")
    sends = value.get("send_state_counts")
    if not isinstance(events, Mapping) or not isinstance(sends, Mapping):
        return False
    if any(type(count) is not int or count < 0 for count in (*events.values(), *sends.values())):
        return False
    unacked = sum(events.get(state, 0) for state in ("pending", "delivering"))
    unresolved = sum(
        sends.get(state, 0) for state in ("prepared", "dispatching", "uncertain")
    )
    return (
        value.get("schema") == "discord-public-connector-cleanup-snapshot.v1"
        and value.get("unacked_event_count") == unacked == 0
        and value.get("unresolved_dispatch_count") == unresolved == 0
        and value.get("safe_to_retire") is True
    )


def _capability_artifact_bindings(
    plan: CapabilityCanaryPlan,
    full_plan: FullCanaryPlan,
) -> Mapping[str, tuple[Path, bytes, int, int, int, frozenset[str]]]:
    full_gateway = full_plan.unit_bundle.gateway_service.encode("utf-8")
    bitrix = validate_bitrix_foundation_for_plan(plan, full_plan)
    bitrix_unit, _ = _read_exact_file(
        Path(bitrix["rendered_unit_stage_path"]),
        maximum=256 * 1024,
        uid=0,
        gid=0,
        mode=0o444,
    )
    bitrix_config, _ = _read_exact_file(
        Path(bitrix["rendered_config_stage_path"]),
        maximum=256 * 1024,
        uid=0,
        gid=0,
        mode=0o400,
    )
    return {
        "bitrix_operational_edge_unit": (
            DEFAULT_BITRIX_UNIT_PATH,
            bitrix_unit,
            0o644,
            0,
            0,
            frozenset(),
        ),
        "bitrix_operational_edge_config": (
            DEFAULT_BITRIX_CONFIG_PATH,
            bitrix_config,
            0o400,
            0,
            0,
            frozenset(),
        ),
        "gateway_config": (
            DEFAULT_GATEWAY_CONFIG,
            render_gateway_config(plan),
            0o440,
            0,
            plan.identities.gateway_gid,
            frozenset(),
        ),
        "mac_ops_config": (
            DEFAULT_MAC_OPS_CONFIG,
            render_mac_ops_config(plan),
            0o440,
            0,
            plan.identities.mac_ops_gid,
            frozenset(),
        ),
        "connector_config": (
            DEFAULT_CONNECTOR_CONFIG,
            render_connector_config(plan),
            0o440,
            0,
            plan.identities.connector_gid,
            frozenset(),
        ),
        "worker_config": (
            DEFAULT_WORKER_CONFIG,
            render_worker_config(plan),
            ISOLATED_WORKER_CONFIG_MODE,
            0,
            plan.identities.worker_gid,
            frozenset(),
        ),
        "browser_config": (
            DEFAULT_BROWSER_CONFIG,
            render_browser_config(plan),
            0o440,
            0,
            plan.identities.browser_gid,
            frozenset(),
        ),
        "worker_socket_unit": (
            DEFAULT_WORKER_SOCKET_UNIT_PATH,
            render_worker_socket_unit(plan).encode("ascii"),
            0o644,
            0,
            0,
            frozenset(),
        ),
        "worker_service_unit": (
            DEFAULT_WORKER_SERVICE_UNIT_PATH,
            render_worker_service_unit(plan).encode("ascii"),
            0o644,
            0,
            0,
            frozenset(),
        ),
        "browser_unit": (
            DEFAULT_BROWSER_UNIT_PATH,
            render_browser_unit(plan).encode("utf-8"),
            0o644,
            0,
            0,
            frozenset(),
        ),
        "mac_ops_unit": (
            DEFAULT_MAC_OPS_UNIT_PATH,
            render_mac_ops_unit(plan).encode("utf-8"),
            0o644,
            0,
            0,
            frozenset(),
        ),
        "connector_unit": (
            DEFAULT_CONNECTOR_UNIT_PATH,
            render_connector_unit(plan).encode("utf-8"),
            0o644,
            0,
            0,
            frozenset(),
        ),
        "gateway_unit": (
            DEFAULT_GATEWAY_UNIT_PATH,
            render_gateway_unit(plan).encode("utf-8"),
            0o644,
            0,
            0,
            frozenset({_sha256_bytes(full_gateway)}),
        ),
    }


def _install_capability_artifacts(
    plan: CapabilityCanaryPlan,
    full_plan: FullCanaryPlan,
) -> Mapping[str, Any]:
    installed: dict[str, Any] = {}
    for name, (path, payload, mode, uid, gid, previous) in (
        _capability_artifact_bindings(plan, full_plan).items()
    ):
        installed[name] = _atomic_install_payload(
            full_plan,
            name=f"capability_{name}",
            path=path,
            payload=payload,
            mode=mode,
            uid=uid,
            gid=gid,
            allowed_previous=previous,
        )
    return installed


def _prepare_connector_state(plan: CapabilityCanaryPlan) -> Mapping[str, Any]:
    """Create/validate the connector state root and explicit journal."""

    try:
        state = os.lstat(DEFAULT_CONNECTOR_STATE)
    except FileNotFoundError:
        DEFAULT_CONNECTOR_STATE.mkdir(parents=True, mode=0o700)
        os.chown(
            DEFAULT_CONNECTOR_STATE,
            plan.identities.connector_uid,
            plan.identities.connector_gid,
        )
        os.chmod(DEFAULT_CONNECTOR_STATE, 0o700)
        _fsync_directory(DEFAULT_CONNECTOR_STATE.parent)
        state = os.lstat(DEFAULT_CONNECTOR_STATE)
    if (
        stat.S_ISLNK(state.st_mode)
        or not stat.S_ISDIR(state.st_mode)
        or state.st_uid != plan.identities.connector_uid
        or state.st_gid != plan.identities.connector_gid
        or stat.S_IMODE(state.st_mode) != 0o700
    ):
        raise RuntimeError("Discord connector state identity is unsafe")
    if not os.path.lexists(DEFAULT_DISCORD_CONNECTOR_JOURNAL):
        _run_checked(
            Command(
                (
                    RUNUSER,
                    "--user",
                    plan.identities.connector_user,
                    "--group",
                    plan.identities.connector_group,
                    "--",
                    str(plan.interpreter),
                    "-B",
                    "-I",
                    "-m",
                    "gateway.discord_connector_bootstrap",
                    "--config",
                    str(DEFAULT_CONNECTOR_CONFIG),
                    "--bootstrap-journal",
                ),
                timeout_seconds=30,
            ),
            label="bootstrap Discord connector journal",
        )
    journal = os.lstat(DEFAULT_DISCORD_CONNECTOR_JOURNAL)
    if (
        stat.S_ISLNK(journal.st_mode)
        or not stat.S_ISREG(journal.st_mode)
        or journal.st_nlink != 1
        or journal.st_uid != plan.identities.connector_uid
        or journal.st_gid != plan.identities.connector_gid
        or stat.S_IMODE(journal.st_mode) != 0o600
    ):
        raise RuntimeError("Discord connector journal identity is unsafe")
    snapshot = DurableDiscordConnectorJournal(
        DEFAULT_DISCORD_CONNECTOR_JOURNAL
    ).cleanup_snapshot()
    if not _connector_cleanup_snapshot_is_safe(snapshot):
        raise RuntimeError("Discord connector journal has unresolved dispatch state")
    return {
        "journal_path": str(DEFAULT_DISCORD_CONNECTOR_JOURNAL),
        "journal_device": journal.st_dev,
        "journal_inode": journal.st_ino,
        "cleanup_snapshot": snapshot,
        "ready": True,
    }


def _remove_exact_overlay_artifacts(
    plan: CapabilityCanaryPlan,
    full_plan: FullCanaryPlan,
) -> Mapping[str, Any]:
    """Remove only bytes installed by this exact plan; reject substitution."""

    removed: dict[str, Any] = {}
    bindings = _capability_artifact_bindings(plan, full_plan)
    for name, (path, payload, mode, uid, gid, _previous) in bindings.items():
        if name == "gateway_unit":
            continue
        try:
            raw, item = _read_stable_file(
                path,
                maximum=max(len(payload), 1),
                expected_uid=uid,
                expected_gid=gid,
                allowed_modes=frozenset({mode}),
            )
        except FileNotFoundError:
            removed[name] = {"path": str(path), "removed": False, "absent": True}
            continue
        except RuntimeError as exc:
            raise RuntimeError(
                f"capability overlay substitution detected: {name}"
            ) from exc
        if raw != payload:
            raise RuntimeError(f"capability overlay substitution detected: {name}")
        before = (
            item.st_dev,
            item.st_ino,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
        )
        current = os.lstat(path)
        after = (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
            current.st_ctime_ns,
        )
        if before != after:
            raise RuntimeError(f"capability overlay changed before removal: {name}")
        os.unlink(path)
        _fsync_directory(path.parent)
        removed[name] = {
            "path": str(path),
            "sha256": _sha256_bytes(payload),
            "removed": True,
            "absent": not os.path.lexists(path),
        }
    return removed


def _overlay_targets_are_absent(
    plan: CapabilityCanaryPlan,
    full_plan: FullCanaryPlan,
) -> bool:
    for name, (path, _payload, _mode, _uid, _gid, _previous) in (
        _capability_artifact_bindings(plan, full_plan).items()
    ):
        if name != "gateway_unit" and os.path.lexists(path):
            return False
    return True


def _restore_full_gateway_unit(
    plan: CapabilityCanaryPlan,
    full_plan: FullCanaryPlan,
) -> Mapping[str, Any]:
    payload = full_plan.unit_bundle.gateway_service.encode("utf-8")
    return _atomic_install_payload(
        full_plan,
        name="capability_gateway_unit_restore",
        path=DEFAULT_GATEWAY_UNIT_PATH,
        payload=payload,
        mode=0o644,
        uid=0,
        gid=0,
        allowed_previous=frozenset({plan.gateway_unit_sha256}),
    )


def _write_lifecycle_receipt(
    plan: CapabilityCanaryPlan,
    *,
    stage: str,
    value: Mapping[str, Any],
) -> Mapping[str, Any]:
    if stage not in {"started", "stopped", "failure"}:
        raise ValueError("capability lifecycle receipt stage is invalid")
    directory = (
        DEFAULT_LIFECYCLE_RECEIPT_ROOT / plan.revision / plan.sha256 / stage
    )
    _ensure_root_directory(directory)
    path = directory / f"{time.time_ns()}-{os.getpid()}-{uuid.uuid4().hex}.json"
    unsigned = {
        **copy.deepcopy(dict(value)),
        "schema": CAPABILITY_LIFECYCLE_RECEIPT_SCHEMA,
        "stage": stage,
        "revision": plan.revision,
        "plan_sha256": plan.sha256,
        "full_canary_plan_sha256": plan.full_canary_plan_sha256,
        "receipt_path": str(path),
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    receipt = {**unsigned, "receipt_sha256": _sha256_json(unsigned)}
    _write_exclusive_bytes(path, _canonical_bytes(receipt), mode=0o400)
    return receipt


def _jwt_exp(token: bytes) -> int:
    try:
        pieces = token.decode("ascii", errors="strict").split(".")
        if len(pieces) != 3:
            raise ValueError
        payload = pieces[1] + "=" * (-len(pieces[1]) % 4)
        value = json.loads(base64.urlsafe_b64decode(payload.encode("ascii")))
        expiry = value.get("exp")
    except Exception as exc:
        raise ValueError("Codex access token is not a bounded JWT") from exc
    if type(expiry) is not int or expiry <= 0:
        raise ValueError("Codex access token expiry is invalid")
    return expiry


def build_secret_lease_frame(
    *, kind: str, secret: bytes | bytearray, plan_sha256: str,
    owner_subject_sha256: str, now_unix: int | None = None,
    ttl_seconds: int = 900, lease_id: str | None = None,
) -> bytearray:
    if kind not in _SECRET_LEASE_MAGIC_BY_KIND:
        raise ValueError("secret lease kind is invalid")
    payload = bytes(secret)
    if not payload or len(payload) > _MAX_SECRET_BYTES:
        raise ValueError("secret lease payload size is invalid")
    issued = int(time.time()) if now_unix is None else now_unix
    if type(issued) is not int or type(ttl_seconds) is not int or not 60 <= ttl_seconds <= _MAX_LEASE_SECONDS:
        raise ValueError("secret lease window is invalid")
    lease = uuid.uuid4().hex if lease_id is None else lease_id
    if _LEASE_ID_RE.fullmatch(lease) is None:
        raise ValueError("secret lease id is invalid")
    token_expiry = _jwt_exp(payload) if kind == "codex_access_token" else None
    if token_expiry is not None and token_expiry < issued + ttl_seconds + 120:
        raise ValueError("Codex access token expires inside the canary window")
    metadata = {
        "schema": CAPABILITY_LEASE_FRAME_SCHEMA, "kind": kind,
        "plan_sha256": _digest(plan_sha256, "capability plan"),
        "owner_subject_sha256": _digest(owner_subject_sha256, "owner subject"),
        "lease_id": lease, "issued_at_unix": issued,
        "expires_at_unix": issued + ttl_seconds, "secret_bytes": len(payload),
        "token_expires_at_unix": token_expiry,
    }
    encoded = _canonical_bytes(metadata)
    magic = _SECRET_LEASE_MAGIC_BY_KIND[kind]
    return bytearray(magic + struct.pack(">II", len(encoded), len(payload)) + encoded + payload)


def read_secret_lease_frame(
    stream: BinaryIO, *, expected_kind: str, now_unix: int | None = None,
) -> tuple[Mapping[str, Any], bytearray]:
    header = stream.read(12)
    if len(header) != 12:
        raise ValueError("secret lease frame header is invalid")
    magic, metadata_size, secret_size = header[:4], *struct.unpack(">II", header[4:])
    try:
        expected_magic = _SECRET_LEASE_MAGIC_BY_KIND[expected_kind]
    except KeyError as exc:
        raise ValueError("secret lease kind is invalid") from exc
    if magic != expected_magic or not 0 < metadata_size <= 64 * 1024 or not 0 < secret_size <= _MAX_SECRET_BYTES:
        raise ValueError("secret lease frame bounds are invalid")
    metadata_raw = stream.read(metadata_size)
    secret = bytearray(stream.read(secret_size))
    if len(metadata_raw) != metadata_size or len(secret) != secret_size or stream.read(1):
        raise ValueError("secret lease frame length is invalid")
    metadata = _decode_json(metadata_raw, label="secret lease metadata")
    fields = {"schema", "kind", "plan_sha256", "owner_subject_sha256", "lease_id", "issued_at_unix", "expires_at_unix", "secret_bytes", "token_expires_at_unix"}
    _strict_mapping(metadata, fields, "secret lease metadata")
    now = int(time.time()) if now_unix is None else now_unix
    if metadata_raw != _canonical_bytes(metadata) or metadata["schema"] != CAPABILITY_LEASE_FRAME_SCHEMA or metadata["kind"] != expected_kind or metadata["secret_bytes"] != secret_size or _LEASE_ID_RE.fullmatch(str(metadata["lease_id"])) is None or type(metadata["issued_at_unix"]) is not int or type(metadata["expires_at_unix"]) is not int or not metadata["issued_at_unix"] <= now <= metadata["expires_at_unix"] or not 60 <= metadata["expires_at_unix"] - metadata["issued_at_unix"] <= _MAX_LEASE_SECONDS:
        raise ValueError("secret lease metadata is invalid")
    _digest(metadata["plan_sha256"], "capability plan")
    _digest(metadata["owner_subject_sha256"], "owner subject")
    if expected_kind == "codex_access_token":
        expiry = _jwt_exp(bytes(secret))
        if metadata["token_expires_at_unix"] != expiry or expiry < metadata["expires_at_unix"] + 120:
            raise ValueError("Codex access token lease is invalid")
    elif metadata["token_expires_at_unix"] is not None:
        raise ValueError("opaque capability lease carries token metadata")
    return copy.deepcopy(dict(metadata)), secret


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    offset = 0
    while offset < len(view):
        written = os.write(descriptor, view[offset:])
        if written <= 0:
            raise OSError("credential write made no progress")
        offset += written


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class _SecretLeaseTarget:
    kind: str
    credential_binding: str
    path: Path
    journal: Path
    uid: int
    gid: int
    mode: int
    parent_uid: int
    parent_gid: int
    parent_mode: int
    maximum_bytes: int


@dataclass(frozen=True)
class _LeaseArtifactPaths:
    root: Path
    install_intent: Path
    install_receipt: Path
    retirement_intent: Path
    retirement_completion: Path


def _lease_target(
    plan: CapabilityCanaryPlan,
    *,
    kind: str,
    full_plan: FullCanaryPlan | None = None,
    auth_path: Path = DEFAULT_GATEWAY_AUTH_STORE,
    mac_path: Path = DEFAULT_MAC_OPS_CREDENTIAL,
    connector_path: Path = DEFAULT_CONNECTOR_TOKEN,
    api_control_path: Path = DEFAULT_API_SERVER_CONTROL_KEY,
    routeback_path: Path = DEFAULT_EDGE_TOKEN_PATH,
    bitrix_path: Path = DEFAULT_BITRIX_WEBHOOK_PATH,
    journal_path: Path | None = None,
) -> _SecretLeaseTarget:
    if kind not in _SECRET_LEASE_MAGIC_BY_KIND:
        raise ValueError("secret lease kind is invalid")
    administrative_uid = os.geteuid()  # windows-footgun: ok — Linux production/canary boundary
    administrative_gid = os.getegid()  # windows-footgun: ok — Linux production/canary boundary
    if kind == "codex_access_token":
        values = (
            auth_path,
            DEFAULT_CODEX_LEASE_JOURNAL,
            plan.identities.gateway_uid,
            plan.identities.gateway_gid,
            0o600,
            plan.identities.gateway_uid,
            plan.identities.gateway_gid,
            0o700,
            128 * 1024,
        )
    elif kind == "mac_ops_gitlab_env":
        values = (
            mac_path,
            DEFAULT_MAC_OPS_LEASE_JOURNAL,
            plan.identities.mac_ops_uid,
            plan.identities.mac_ops_gid,
            0o400,
            plan.identities.mac_ops_uid,
            plan.identities.mac_ops_gid,
            0o700,
            _MAX_SECRET_BYTES,
        )
    elif kind == "discord_connector_token":
        values = (
            connector_path,
            DEFAULT_CONNECTOR_LEASE_JOURNAL,
            plan.identities.connector_uid,
            plan.identities.connector_gid,
            0o400,
            plan.identities.connector_uid,
            plan.identities.connector_gid,
            0o700,
            512,
        )
    elif kind == "api_server_control_key":
        values = (
            api_control_path,
            DEFAULT_API_CONTROL_LEASE_JOURNAL,
            administrative_uid,
            administrative_gid,
            0o400,
            administrative_uid,
            administrative_gid,
            0o711,
            8 * 1024,
        )
    elif kind == "bitrix_operational_edge_webhook":
        values = (
            bitrix_path,
            DEFAULT_BITRIX_LEASE_JOURNAL,
            administrative_uid,
            administrative_gid,
            0o400,
            administrative_uid,
            administrative_gid,
            0o700,
            8 * 1024,
        )
    else:
        if full_plan is None:
            raise ValueError("full-canary plan is required for the route-back lease")
        validate_plan_against_full(plan, full_plan)
        values = (
            routeback_path,
            DEFAULT_ROUTEBACK_LEASE_JOURNAL,
            full_plan.identities.edge_uid,
            full_plan.identities.edge_gid,
            0o400,
            administrative_uid,
            full_plan.identities.edge_gid,
            0o750,
            512,
        )
    path, default_journal, uid, gid, mode, parent_uid, parent_gid, parent_mode, maximum = values
    if not path.is_absolute() or not (journal_path or default_journal).is_absolute():
        raise ValueError("credential lease paths must be absolute")
    return _SecretLeaseTarget(
        kind=kind,
        credential_binding=_CREDENTIAL_BINDING_BY_KIND[kind],
        path=path,
        journal=journal_path or default_journal,
        uid=uid,
        gid=gid,
        mode=mode,
        parent_uid=parent_uid,
        parent_gid=parent_gid,
        parent_mode=parent_mode,
        maximum_bytes=maximum,
    )


def _prepare_secret_parent(
    path: Path,
    *,
    uid: int,
    gid: int,
    mode: int,
) -> None:
    created = False
    try:
        item = os.lstat(path)
    except FileNotFoundError:
        try:
            path.mkdir(parents=True, mode=mode, exist_ok=False)
            created = True
        except FileExistsError:
            pass
        if created:
            os.chown(path, uid, gid)
            os.chmod(path, mode)
        item = os.lstat(path)
    if (
        stat.S_ISLNK(item.st_mode)
        or not stat.S_ISDIR(item.st_mode)
        or item.st_uid != uid
        or item.st_gid != gid
        or stat.S_IMODE(item.st_mode) != mode
    ):
        raise PermissionError("credential lease parent identity is unsafe")
    if created:
        _fsync_directory(path.parent)


def _prepare_journal_directory(path: Path) -> None:
    _prepare_secret_parent(
        path,
        uid=os.geteuid(),  # windows-footgun: ok — Linux production/canary boundary
        gid=os.getegid(),  # windows-footgun: ok — Linux production/canary boundary
        mode=0o700,
    )


def _validate_journal_directory(path: Path) -> None:
    item = os.lstat(path)
    if (
        stat.S_ISLNK(item.st_mode)
        or not stat.S_ISDIR(item.st_mode)
        or item.st_uid != os.geteuid()  # windows-footgun: ok — Linux production/canary boundary
        or item.st_gid != os.getegid()  # windows-footgun: ok — Linux production/canary boundary
        or stat.S_IMODE(item.st_mode) != 0o700
    ):
        raise RuntimeError("credential lease journal directory is unsafe")


@contextmanager
def _lease_journal_lock(journal: Path):
    _prepare_journal_directory(journal)
    lock_path = journal / ".lock"
    descriptor = os.open(
        lock_path,
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        os.fchmod(descriptor, 0o600)
        os.fchown(descriptor, os.geteuid(), os.getegid())  # windows-footgun: ok — Linux production/canary boundary
        item = os.fstat(descriptor)
        if (
            not stat.S_ISREG(item.st_mode)
            or item.st_nlink != 1
            or item.st_uid != os.geteuid()  # windows-footgun: ok — Linux production/canary boundary
            or item.st_gid != os.getegid()  # windows-footgun: ok — Linux production/canary boundary
            or stat.S_IMODE(item.st_mode) != 0o600
        ):
            raise RuntimeError("credential lease journal lock is unsafe")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _rename_no_replace(source: Path, target: Path) -> None:
    """Atomically publish one file without ever replacing the target."""

    if sys.platform.startswith("linux"):
        import ctypes
        import errno

        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise RuntimeError("renameat2 is required for credential publication")
        renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
        if renameat2(
            -100,
            os.fsencode(source),
            -100,
            os.fsencode(target),
            1,
        ) != 0:
            number = ctypes.get_errno()
            if number == errno.EEXIST:
                raise FileExistsError(str(target))
            raise OSError(number, os.strerror(number), str(target))
        return
    os.link(source, target, follow_symlinks=False)
    os.unlink(source)


def _read_exact_file(
    path: Path,
    *,
    maximum: int,
    uid: int,
    gid: int,
    mode: int,
) -> tuple[bytes, os.stat_result]:
    return _read_stable_file(
        path,
        maximum=maximum,
        expected_uid=uid,
        expected_gid=gid,
        allowed_modes=frozenset({mode}),
    )


def _atomic_no_replace_file(
    path: Path,
    payload: bytes,
    *,
    uid: int,
    gid: int,
    mode: int,
    temporary_name: str,
    maximum: int,
) -> os.stat_result:
    temporary = path.parent / temporary_name
    if os.path.lexists(temporary):
        temporary_payload, _ = _read_exact_file(
            temporary,
            maximum=maximum,
            uid=uid,
            gid=gid,
            mode=mode,
        )
        if temporary_payload != payload:
            raise RuntimeError("credential publication half-state is inconsistent")
    else:
        descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            mode,
        )
        try:
            _write_all(descriptor, payload)
            os.fchmod(descriptor, mode)
            os.fchown(descriptor, uid, gid)
            os.fsync(descriptor)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
        finally:
            os.close(descriptor)
        _fsync_directory(path.parent)
    try:
        _rename_no_replace(temporary, path)
    except FileExistsError:
        existing, item = _read_exact_file(
            path,
            maximum=maximum,
            uid=uid,
            gid=gid,
            mode=mode,
        )
        if existing != payload:
            raise RuntimeError("credential publication collided with different bytes")
        if os.path.lexists(temporary):
            # The deterministic temporary was already stable-read above and the
            # per-kind lock excludes a competing publisher.  It is uncommitted
            # state regardless of whether the fallback hard-link shares the
            # winning inode.
            os.unlink(temporary)
            _fsync_directory(path.parent)
        return item
    _fsync_directory(path.parent)
    installed, item = _read_exact_file(
        path,
        maximum=maximum,
        uid=uid,
        gid=gid,
        mode=mode,
    )
    if installed != payload:
        raise RuntimeError("credential publication readback failed")
    return item


def _lease_artifact_paths(journal: Path, lease_id: str) -> _LeaseArtifactPaths:
    if _LEASE_ID_RE.fullmatch(lease_id) is None:
        raise ValueError("secret lease id is invalid")
    root = journal / lease_id
    return _LeaseArtifactPaths(
        root=root,
        install_intent=root / "install-intent.json",
        install_receipt=root / "install-receipt.json",
        retirement_intent=root / "retirement-intent.json",
        retirement_completion=root / "retirement-completion.json",
    )


def _load_lease_artifact(path: Path, *, schema: str) -> Mapping[str, Any]:
    raw, _ = _read_exact_file(
        path,
        maximum=64 * 1024,
        uid=os.geteuid(),  # windows-footgun: ok — Linux production/canary boundary
        gid=os.getegid(),  # windows-footgun: ok — Linux production/canary boundary
        mode=0o400,
    )
    value = _decode_json(raw, label="credential lease artifact")
    unsigned = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if (
        raw != _canonical_bytes(value)
        or value.get("schema") != schema
        or value.get("receipt_path") != str(path)
        or value.get("receipt_sha256") != _sha256_json(unsigned)
        or value.get("secret_material_recorded") is not False
        or value.get("secret_digest_recorded") is not False
    ):
        raise RuntimeError("credential lease artifact is invalid")
    return copy.deepcopy(dict(value))


def _append_lease_artifact(
    path: Path,
    *,
    schema: str,
    value: Mapping[str, Any],
) -> Mapping[str, Any]:
    _prepare_journal_directory(path.parent)
    unsigned = {
        **copy.deepcopy(dict(value)),
        "schema": schema,
        "receipt_path": str(path),
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    receipt = {**unsigned, "receipt_sha256": _sha256_json(unsigned)}
    payload = _canonical_bytes(receipt)
    _atomic_no_replace_file(
        path,
        payload,
        uid=os.geteuid(),  # windows-footgun: ok — Linux production/canary boundary
        gid=os.getegid(),  # windows-footgun: ok — Linux production/canary boundary
        mode=0o400,
        temporary_name=f".{path.name}.tmp",
        maximum=64 * 1024,
    )
    return _load_lease_artifact(path, schema=schema)


def _journal_states(journal: Path) -> list[Mapping[str, Any]]:
    _validate_journal_directory(journal)
    names = sorted(os.listdir(journal))
    lease_names = [name for name in names if name != ".lock"]
    if len(lease_names) > _MAX_LEASE_ARTIFACTS or any(
        _LEASE_ID_RE.fullmatch(name) is None for name in lease_names
    ):
        raise RuntimeError("credential lease journal inventory is invalid")
    states: list[Mapping[str, Any]] = []
    for lease_id in lease_names:
        paths = _lease_artifact_paths(journal, lease_id)
        directory = os.lstat(paths.root)
        if (
            stat.S_ISLNK(directory.st_mode)
            or not stat.S_ISDIR(directory.st_mode)
            or directory.st_uid != os.geteuid()  # windows-footgun: ok — Linux production/canary boundary
            or directory.st_gid != os.getegid()  # windows-footgun: ok — Linux production/canary boundary
            or stat.S_IMODE(directory.st_mode) != 0o700
        ):
            raise RuntimeError("credential lease artifact directory is unsafe")
        allowed = {
            paths.install_intent.name,
            paths.install_receipt.name,
            paths.retirement_intent.name,
            paths.retirement_completion.name,
            f".{paths.install_intent.name}.tmp",
            f".{paths.install_receipt.name}.tmp",
            f".{paths.retirement_intent.name}.tmp",
            f".{paths.retirement_completion.name}.tmp",
        }
        if not set(os.listdir(paths.root)).issubset(allowed):
            raise RuntimeError("credential lease artifact inventory is invalid")
        artifacts: dict[str, Mapping[str, Any] | None] = {}
        for field, path, schema in (
            ("install_intent", paths.install_intent, CAPABILITY_LEASE_INSTALL_INTENT_SCHEMA),
            ("install_receipt", paths.install_receipt, CAPABILITY_LEASE_RECEIPT_SCHEMA),
            ("retirement_intent", paths.retirement_intent, CAPABILITY_RETIREMENT_INTENT_SCHEMA),
            ("retirement_completion", paths.retirement_completion, CAPABILITY_RETIREMENT_RECEIPT_SCHEMA),
        ):
            artifacts[field] = (
                _load_lease_artifact(path, schema=schema)
                if os.path.lexists(path)
                else None
            )
        if artifacts["install_intent"] is None:
            raise RuntimeError("credential lease journal has an orphan artifact")
        if artifacts["install_receipt"] is None and any(
            artifacts[name] is not None
            for name in ("retirement_intent", "retirement_completion")
        ):
            raise RuntimeError("credential lease retirement lacks an install receipt")
        if artifacts["retirement_intent"] is None and artifacts["retirement_completion"] is not None:
            raise RuntimeError("credential lease completion lacks retirement intent")
        for artifact in (item for item in artifacts.values() if item is not None):
            if artifact.get("lease_id") != lease_id:
                raise RuntimeError("credential lease artifact lease binding drifted")
        states.append({"lease_id": lease_id, "paths": paths, **artifacts})
    return states


def _target_metadata(item: os.stat_result) -> Mapping[str, Any]:
    return {
        "target_device": item.st_dev,
        "target_inode": item.st_ino,
        "target_uid": item.st_uid,
        "target_gid": item.st_gid,
        "target_mode": f"{stat.S_IMODE(item.st_mode):04o}",
        "target_size": item.st_size,
        "target_mtime_ns": item.st_mtime_ns,
        "target_ctime_ns": item.st_ctime_ns,
    }


def _receipt_matches_target(receipt: Mapping[str, Any], item: os.stat_result) -> bool:
    return all(receipt.get(key) == value for key, value in _target_metadata(item).items())


def _validate_lease_metadata(
    plan: CapabilityCanaryPlan,
    metadata: Mapping[str, Any],
    secret: bytearray,
) -> str:
    kind = metadata.get("kind")
    now = int(time.time())
    if (
        metadata.get("schema") != CAPABILITY_LEASE_FRAME_SCHEMA
        or metadata.get("plan_sha256") != plan.sha256
        or kind not in _SECRET_LEASE_MAGIC_BY_KIND
        or _LEASE_ID_RE.fullmatch(str(metadata.get("lease_id"))) is None
        or type(metadata.get("issued_at_unix")) is not int
        or type(metadata.get("expires_at_unix")) is not int
        or not metadata["issued_at_unix"] <= now <= metadata["expires_at_unix"]
        or not 60 <= metadata["expires_at_unix"] - metadata["issued_at_unix"] <= _MAX_LEASE_SECONDS
        or metadata.get("secret_bytes") != len(secret)
    ):
        raise PermissionError("secret lease is not bound to this capability plan")
    _digest(metadata.get("owner_subject_sha256"), "owner subject")
    return kind


def _secret_payload(
    *,
    kind: str,
    lease_id: str,
    secret: bytearray,
) -> bytes:
    if kind == "codex_access_token":
        token = bytes(secret).decode("ascii", errors="strict")
        _jwt_exp(token.encode("ascii"))
        return _canonical_bytes(
            {
                "version": 1,
                "providers": {},
                "credential_pool": {
                    "openai-codex": [
                        {
                            "id": lease_id[:12],
                            "label": "capability-canary-lease",
                            "auth_type": "api_key",
                            "priority": 0,
                            "source": "manual:capability_canary_lease",
                            "access_token": token,
                            "request_count": 0,
                            "last_status": None,
                            "last_status_at": None,
                            "last_error_code": None,
                            "last_error_reason": None,
                            "last_error_message": None,
                            "last_error_reset_at": None,
                        }
                    ]
                },
            }
        )
    if kind == "mac_ops_gitlab_env":
        _parse_secret_env(bytes(secret))
        return bytes(secret)
    try:
        token = bytes(secret).decode("ascii", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("opaque capability token encoding is invalid") from exc
    maximum = 8 * 1024 if kind == "bitrix_operational_edge_webhook" else 512
    if (
        not token
        or len(secret) > maximum
        or token != token.strip()
        or any(character.isspace() or ord(character) < 0x21 or ord(character) == 0x7F for character in token)
    ):
        raise ValueError("opaque capability token is invalid")
    return bytes(secret)


def provision_secret_lease(
    plan: CapabilityCanaryPlan,
    metadata: Mapping[str, Any],
    secret: bytearray,
    *,
    full_plan: FullCanaryPlan | None = None,
    auth_path: Path = DEFAULT_GATEWAY_AUTH_STORE,
    mac_path: Path = DEFAULT_MAC_OPS_CREDENTIAL,
    connector_path: Path = DEFAULT_CONNECTOR_TOKEN,
    api_control_path: Path = DEFAULT_API_SERVER_CONTROL_KEY,
    routeback_path: Path = DEFAULT_EDGE_TOKEN_PATH,
    bitrix_path: Path = DEFAULT_BITRIX_WEBHOOK_PATH,
    journal_path: Path | None = None,
) -> Mapping[str, Any]:
    kind = _validate_lease_metadata(plan, metadata, secret)
    spec = _lease_target(
        plan,
        kind=kind,
        full_plan=full_plan,
        auth_path=auth_path,
        mac_path=mac_path,
        connector_path=connector_path,
        api_control_path=api_control_path,
        routeback_path=routeback_path,
        bitrix_path=bitrix_path,
        journal_path=journal_path,
    )
    payload: bytes | None = None
    try:
        payload = _secret_payload(
            kind=kind,
            lease_id=metadata["lease_id"],
            secret=secret,
        )
        production_targets = (
            journal_path is None
            and auth_path == DEFAULT_GATEWAY_AUTH_STORE
            and mac_path == DEFAULT_MAC_OPS_CREDENTIAL
            and connector_path == DEFAULT_CONNECTOR_TOKEN
            and api_control_path == DEFAULT_API_SERVER_CONTROL_KEY
            and routeback_path == DEFAULT_EDGE_TOKEN_PATH
            and bitrix_path == DEFAULT_BITRIX_WEBHOOK_PATH
        )
        expiry_watchdog: Mapping[str, Any] | None = None
        if production_targets:
            if full_plan is None:
                raise RuntimeError(
                    "production credential lease lacks sealed full-canary plan"
                )
            expiry_watchdog = arm_capability_expiry_watchdog(
                kind="credential_lease",
                revision=plan.revision,
                full_canary_plan_sha256=plan.full_canary_plan_sha256,
                release_artifact_sha256=plan.release_artifact_sha256,
                interpreter=plan.interpreter,
                expires_at_unix=metadata["expires_at_unix"],
                authority_sha256=_sha256_json(metadata),
                plan_sha256=plan.sha256,
                credential_binding=spec.credential_binding,
            )
        with _lease_journal_lock(spec.journal):
            states = _journal_states(spec.journal)
            current = next(
                (item for item in states if item["lease_id"] == metadata["lease_id"]),
                None,
            )
            incomplete = [
                item
                for item in states
                if item["retirement_completion"] is None
                and item["lease_id"] != metadata["lease_id"]
            ]
            if incomplete:
                raise RuntimeError("another secret lease remains active or incomplete")
            if current is not None and current["retirement_completion"] is not None:
                raise RuntimeError("secret lease id was already retired")
            paths = _lease_artifact_paths(spec.journal, metadata["lease_id"])
            _prepare_journal_directory(paths.root)
            intent = _append_lease_artifact(
                paths.install_intent,
                schema=CAPABILITY_LEASE_INSTALL_INTENT_SCHEMA,
                value={
                    "operation": "install_intent",
                    "kind": kind,
                    "credential_binding": spec.credential_binding,
                    "revision": plan.revision,
                    "plan_sha256": plan.sha256,
                    "full_canary_plan_sha256": plan.full_canary_plan_sha256,
                    "owner_subject_sha256": metadata["owner_subject_sha256"],
                    "lease_id": metadata["lease_id"],
                    "issued_at_unix": metadata["issued_at_unix"],
                    "expires_at_unix": metadata["expires_at_unix"],
                    "target_path": str(spec.path),
                    "target_uid": spec.uid,
                    "target_gid": spec.gid,
                    "target_mode": f"{spec.mode:04o}",
                    "target_parent_uid": spec.parent_uid,
                    "target_parent_gid": spec.parent_gid,
                    "target_parent_mode": f"{spec.parent_mode:04o}",
                    "intent_at_unix": metadata["issued_at_unix"],
                    "expiry_watchdog": copy.deepcopy(
                        dict(expiry_watchdog or {})
                    ),
                },
            )
            _prepare_secret_parent(
                spec.path.parent,
                uid=spec.parent_uid,
                gid=spec.parent_gid,
                mode=spec.parent_mode,
            )
            if os.path.lexists(spec.path):
                existing_item = os.lstat(spec.path)
                if (
                    stat.S_ISLNK(existing_item.st_mode)
                    or not stat.S_ISREG(existing_item.st_mode)
                ):
                    raise FileExistsError(
                        "active credential lease already exists with unsafe type"
                    )
                installed_payload, target_item = _read_exact_file(
                    spec.path,
                    maximum=spec.maximum_bytes,
                    uid=spec.uid,
                    gid=spec.gid,
                    mode=spec.mode,
                )
                if installed_payload != payload:
                    raise RuntimeError("credential lease retry carries different secret bytes")
            else:
                target_item = _atomic_no_replace_file(
                    spec.path,
                    payload,
                    uid=spec.uid,
                    gid=spec.gid,
                    mode=spec.mode,
                    temporary_name=f".{spec.path.name}.{metadata['lease_id']}.installing",
                    maximum=spec.maximum_bytes,
                )
            receipt = _append_lease_artifact(
                paths.install_receipt,
                schema=CAPABILITY_LEASE_RECEIPT_SCHEMA,
                value={
                    "operation": "install",
                    "state": "provisioned",
                    "kind": kind,
                    "credential_binding": spec.credential_binding,
                    "revision": plan.revision,
                    "plan_sha256": plan.sha256,
                    "full_canary_plan_sha256": plan.full_canary_plan_sha256,
                    "owner_subject_sha256": metadata["owner_subject_sha256"],
                    "lease_id": metadata["lease_id"],
                    "issued_at_unix": metadata["issued_at_unix"],
                    "expires_at_unix": metadata["expires_at_unix"],
                    "target_path": str(spec.path),
                    **_target_metadata(target_item),
                    "install_intent_path": intent["receipt_path"],
                    "install_intent_sha256": intent["receipt_sha256"],
                    "installed_at_unix": target_item.st_ctime_ns // 1_000_000_000,
                    "expiry_watchdog": copy.deepcopy(
                        dict(expiry_watchdog or {})
                    ),
                },
            )
            if not _receipt_matches_target(receipt, os.lstat(spec.path)):
                raise RuntimeError("credential install receipt target binding drifted")
            return receipt
    finally:
        for index in range(len(secret)):
            secret[index] = 0
        payload = None


def _active_install_state(
    *,
    kind: str,
    target: Path,
    journal: Path,
    plan_sha256: str | None = None,
) -> Mapping[str, Any]:
    matches = []
    for state in _journal_states(journal):
        receipt = state["install_receipt"]
        if (
            receipt is not None
            and state["retirement_completion"] is None
            and receipt.get("kind") == kind
            and receipt.get("credential_binding") == _CREDENTIAL_BINDING_BY_KIND[kind]
            and receipt.get("target_path") == str(target)
            and (plan_sha256 is None or receipt.get("plan_sha256") == plan_sha256)
        ):
            matches.append(state)
    if len(matches) != 1:
        raise RuntimeError("credential lease journal does not identify one active install")
    return matches[0]


def retire_secret_lease(
    *,
    kind: str,
    target: Path,
    journal: Path,
    stop_proof: Mapping[str, Any],
    plan: CapabilityCanaryPlan | None = None,
    now_unix: int | None = None,
) -> Mapping[str, Any]:
    if kind not in _SECRET_LEASE_MAGIC_BY_KIND:
        raise ValueError("secret lease kind is invalid")
    with _lease_journal_lock(journal):
        states = _journal_states(journal)
        completed = [
            state
            for state in states
            if state["retirement_completion"] is not None
            and state["retirement_completion"].get("kind") == kind
            and state["retirement_completion"].get("target_path") == str(target)
            and (
                plan is None
                or state["retirement_completion"].get("plan_sha256") == plan.sha256
            )
        ]
        active = [
            state
            for state in states
            if state["install_receipt"] is not None
            and state["retirement_completion"] is None
            and state["install_receipt"].get("kind") == kind
            and state["install_receipt"].get("target_path") == str(target)
            and (plan is None or state["install_receipt"].get("plan_sha256") == plan.sha256)
        ]
        if not active:
            if len(completed) == 1 and not os.path.lexists(target):
                return completed[0]["retirement_completion"]
            raise RuntimeError("credential retirement lacks one active install receipt")
        if len(active) != 1:
            raise RuntimeError("credential retirement active lease is ambiguous")
        state = active[0]
        install = state["install_receipt"]
        paths = state["paths"]
        if (
            install.get("credential_binding") != _CREDENTIAL_BINDING_BY_KIND[kind]
            or (plan is not None and install.get("full_canary_plan_sha256") != plan.full_canary_plan_sha256)
        ):
            raise RuntimeError("credential retirement install binding drifted")
        current_time = int(time.time()) if now_unix is None else now_unix
        if type(current_time) is not int or current_time < 0:
            raise ValueError("credential retirement time is invalid")
        validated_stop_proof = _validate_capability_stop_proof(
            plan,
            stop_proof,
            installed_at_unix=install["installed_at_unix"],
            now_unix=current_time,
        )
        intent = state["retirement_intent"]
        if intent is None:
            try:
                item = os.lstat(target)
            except FileNotFoundError as exc:
                raise RuntimeError("credential disappeared before retirement intent") from exc
            if not _receipt_matches_target(install, item):
                raise RuntimeError("credential lease target identity is unsafe")
            intent = _append_lease_artifact(
                paths.retirement_intent,
                schema=CAPABILITY_RETIREMENT_INTENT_SCHEMA,
                value={
                    "operation": "retirement_intent",
                    "kind": kind,
                    "credential_binding": install["credential_binding"],
                    "revision": install["revision"],
                    "plan_sha256": install["plan_sha256"],
                    "full_canary_plan_sha256": install["full_canary_plan_sha256"],
                    "lease_id": install["lease_id"],
                    "target_path": str(target),
                    **{
                        key: install[key]
                        for key in _target_metadata(item)
                    },
                    "install_receipt_path": install["receipt_path"],
                    "install_receipt_sha256": install["receipt_sha256"],
                    "service_stop_proof": validated_stop_proof,
                    "service_stop_proof_sha256": validated_stop_proof[
                        "stop_proof_sha256"
                    ],
                    "requested_at_unix": current_time,
                    "services_stopped_required": True,
                },
            )
        else:
            intent_stop_proof = _validate_capability_stop_proof(
                plan,
                intent.get("service_stop_proof", {}),
                installed_at_unix=install["installed_at_unix"],
                now_unix=current_time,
            )
            if (
                intent.get("install_receipt_sha256") != install["receipt_sha256"]
                or intent.get("install_receipt_path") != install["receipt_path"]
                or intent.get("service_stop_proof_sha256")
                != intent_stop_proof["stop_proof_sha256"]
                or type(intent.get("requested_at_unix")) is not int
                or intent["requested_at_unix"]
                < intent_stop_proof["observed_at_unix"]
            ):
                raise RuntimeError("credential retirement intent binding drifted")
        if os.path.lexists(target):
            item = os.lstat(target)
            if not _receipt_matches_target(install, item):
                raise RuntimeError("credential lease target identity is unsafe")
            os.unlink(target)
            _fsync_directory(target.parent)
        if os.path.lexists(target):
            raise RuntimeError("credential target remains after retirement")
        retired_at_unix = int(time.time()) if now_unix is None else now_unix
        stop_observed_at_unix = intent["service_stop_proof"][
            "observed_at_unix"
        ]
        if retired_at_unix < max(intent["requested_at_unix"], stop_observed_at_unix):
            raise RuntimeError("credential retirement completion predates stop proof")
        completion = _append_lease_artifact(
            paths.retirement_completion,
            schema=CAPABILITY_RETIREMENT_RECEIPT_SCHEMA,
            value={
                "operation": "retirement_completion",
                "state": "retired",
                "kind": kind,
                "credential_binding": install["credential_binding"],
                "revision": install["revision"],
                "plan_sha256": install["plan_sha256"],
                "full_canary_plan_sha256": install["full_canary_plan_sha256"],
                "lease_id": install["lease_id"],
                "target_path": str(target),
                **{
                    key: install[key]
                    for key in (
                        "target_device",
                        "target_inode",
                        "target_uid",
                        "target_gid",
                        "target_mode",
                        "target_size",
                        "target_mtime_ns",
                        "target_ctime_ns",
                    )
                },
                "install_receipt_path": install["receipt_path"],
                "install_receipt_sha256": install["receipt_sha256"],
                "retirement_intent_path": intent["receipt_path"],
                "retirement_intent_sha256": intent["receipt_sha256"],
                "service_stop_proof_sha256": intent[
                    "service_stop_proof_sha256"
                ],
                "service_stop_observed_at_unix": stop_observed_at_unix,
                "removed": True,
                "absent": True,
                "absent_after_stop": retired_at_unix >= stop_observed_at_unix,
                "retired_at_unix": retired_at_unix,
            },
        )
        if os.path.lexists(target):
            raise RuntimeError("credential target reappeared after retirement")
        return completion


def _default_secret_lease_targets(
    plan: CapabilityCanaryPlan,
    full_plan: FullCanaryPlan,
) -> tuple[_SecretLeaseTarget, ...]:
    """Return all six fixed slots in their stable public binding order."""

    validate_plan_against_full(plan, full_plan)
    return tuple(
        _lease_target(plan, kind=kind, full_plan=full_plan)
        for kind in (
            "api_server_control_key",
            "bitrix_operational_edge_webhook",
            "discord_routeback_token",
            "discord_connector_token",
            "mac_ops_gitlab_env",
            "codex_access_token",
        )
    )


def _same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_uid",
        "st_gid",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    return all(getattr(left, field) == getattr(right, field) for field in fields)


def _remove_incomplete_lease_file(
    path: Path,
    *,
    spec: _SecretLeaseTarget,
) -> Mapping[str, Any]:
    """Remove one install-intent-bound half-state without reading its bytes."""

    before = os.lstat(path)
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_uid != spec.uid
        or before.st_gid != spec.gid
        or stat.S_IMODE(before.st_mode) != spec.mode
        or not 0 < before.st_size <= spec.maximum_bytes
    ):
        raise RuntimeError("incomplete credential object identity is unsafe")
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    reachable = os.lstat(path)
    if not _same_file_identity(before, opened) or not _same_file_identity(
        before, reachable
    ):
        raise RuntimeError("incomplete credential object changed before cleanup")
    os.unlink(path)
    _fsync_directory(path.parent)
    if os.path.lexists(path):
        raise RuntimeError("incomplete credential object remains after cleanup")
    return {
        "path": str(path),
        "device": before.st_dev,
        "inode": before.st_ino,
        "uid": before.st_uid,
        "gid": before.st_gid,
        "mode": f"{stat.S_IMODE(before.st_mode):04o}",
        "size": before.st_size,
        "removed": True,
        "absent": True,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }


def _retire_secret_slot_best_effort(
    plan: CapabilityCanaryPlan,
    spec: _SecretLeaseTarget,
    *,
    stop_proof: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Retire one active/incomplete slot or attest never-installed absence."""

    if not os.path.lexists(spec.journal):
        if os.path.lexists(spec.path):
            raise RuntimeError("credential exists without an append-only lease journal")
        unsigned = {
            "kind": spec.kind,
            "credential_binding": spec.credential_binding,
            "target_path": str(spec.path),
            "state": "never_installed_absent",
            "install_bound_retirement": False,
            "absent": True,
            "secret_material_recorded": False,
            "secret_digest_recorded": False,
        }
        return {**unsigned, "observation_sha256": _sha256_json(unsigned)}

    incomplete: Mapping[str, Any] | None = None
    completed_history: list[Mapping[str, Any]] = []
    has_install_bound_state = False
    with _lease_journal_lock(spec.journal):
        states = _journal_states(spec.journal)
        matching = [
            state
            for state in states
            if state["install_intent"].get("plan_sha256") == plan.sha256
            and state["install_intent"].get("kind") == spec.kind
            and state["install_intent"].get("target_path") == str(spec.path)
        ]
        unfinished = [
            state
            for state in matching
            if state["retirement_completion"] is None
        ]
        if len(unfinished) > 1:
            raise RuntimeError("credential slot has multiple unfinished leases")
        if unfinished:
            current = unfinished[0]
            if current["install_receipt"] is not None:
                has_install_bound_state = True
            else:
                incomplete = current
        else:
            completed_history = [
                state
                for state in matching
                if state["retirement_completion"] is not None
            ]
            has_install_bound_state = bool(completed_history)

        if incomplete is not None:
            intent = incomplete["install_intent"]
            if (
                intent.get("credential_binding") != spec.credential_binding
                or intent.get("revision") != plan.revision
                or intent.get("full_canary_plan_sha256")
                != plan.full_canary_plan_sha256
                or intent.get("lease_id") != incomplete["lease_id"]
                or intent.get("target_uid") != spec.uid
                or intent.get("target_gid") != spec.gid
                or intent.get("target_mode") != f"{spec.mode:04o}"
                or intent.get("target_parent_uid") != spec.parent_uid
                or intent.get("target_parent_gid") != spec.parent_gid
                or intent.get("target_parent_mode") != f"{spec.parent_mode:04o}"
            ):
                raise RuntimeError("incomplete credential install intent drifted")
            removed: list[Mapping[str, Any]] = []
            temporary = spec.path.parent / (
                f".{spec.path.name}.{incomplete['lease_id']}.installing"
            )
            for candidate in (spec.path, temporary):
                if os.path.lexists(candidate):
                    removed.append(
                        _remove_incomplete_lease_file(candidate, spec=spec)
                    )
            if os.path.lexists(spec.path) or os.path.lexists(temporary):
                raise RuntimeError("incomplete credential lease cleanup is incomplete")
            unsigned = {
                "kind": spec.kind,
                "credential_binding": spec.credential_binding,
                "target_path": str(spec.path),
                "state": "incomplete_install_retired",
                "lease_id": incomplete["lease_id"],
                "install_intent_path": intent["receipt_path"],
                "install_intent_sha256": intent["receipt_sha256"],
                "removed_objects": removed,
                "install_bound_retirement": False,
                "absent": True,
                "secret_material_recorded": False,
                "secret_digest_recorded": False,
            }
            return {**unsigned, "observation_sha256": _sha256_json(unsigned)}

    if completed_history:
        if os.path.lexists(spec.path):
            raise RuntimeError("retired credential target reappeared")
        completions = [
            state["retirement_completion"] for state in completed_history
        ]
        return {
            "kind": spec.kind,
            "credential_binding": spec.credential_binding,
            "target_path": str(spec.path),
            "state": "install_bound_retired",
            "install_bound_retirement": True,
            "retirement_completion": completions[-1],
            "retirement_receipt_sha256": completions[-1]["receipt_sha256"],
            "retirement_history_receipt_sha256s": [
                completion["receipt_sha256"] for completion in completions
            ],
            "absent": True,
            "secret_material_recorded": False,
            "secret_digest_recorded": False,
        }
    if has_install_bound_state:
        completion = retire_secret_lease(
            kind=spec.kind,
            target=spec.path,
            journal=spec.journal,
            stop_proof=stop_proof,
            plan=plan,
        )
        return {
            "kind": spec.kind,
            "credential_binding": spec.credential_binding,
            "target_path": str(spec.path),
            "state": "install_bound_retired",
            "install_bound_retirement": True,
            "retirement_completion": completion,
            "retirement_receipt_sha256": completion["receipt_sha256"],
            "absent": not os.path.lexists(spec.path),
            "secret_material_recorded": False,
            "secret_digest_recorded": False,
        }
    if os.path.lexists(spec.path):
        raise RuntimeError("credential exists without a matching lease intent")
    unsigned = {
        "kind": spec.kind,
        "credential_binding": spec.credential_binding,
        "target_path": str(spec.path),
        "state": "never_installed_absent",
        "install_bound_retirement": False,
        "absent": True,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    return {**unsigned, "observation_sha256": _sha256_json(unsigned)}


def retire_secret_leases_best_effort(
    plan: CapabilityCanaryPlan,
    full_plan: FullCanaryPlan,
    *,
    targets: Sequence[_SecretLeaseTarget] | None = None,
    stop_proof: Mapping[str, Any] | None = None,
    runner: Callable[[Command], subprocess.CompletedProcess[bytes]] = _runner,
) -> Mapping[str, Any]:
    """Attempt all slots and persist exact final absence even on partial setup."""

    validate_plan_against_full(plan, full_plan)
    exact = tuple(targets or _default_secret_lease_targets(plan, full_plan))
    if (
        len(exact) != len(CAPABILITY_CREDENTIAL_BINDINGS)
        or tuple(spec.credential_binding for spec in exact)
        != CAPABILITY_CREDENTIAL_BINDINGS
    ):
        raise ValueError("partial credential cleanup target order is not exact")
    exact_stop_proof = stop_proof
    if exact_stop_proof is None:
        services = _capability_services(runner=runner)
        exact_stop_proof = build_capability_stop_proof(plan, services)
    slots: dict[str, Mapping[str, Any]] = {}
    errors: dict[str, str] = {}
    for spec in exact:
        try:
            slots[spec.credential_binding] = _retire_secret_slot_best_effort(
                plan,
                spec,
                stop_proof=exact_stop_proof,
            )
        except BaseException as exc:
            errors[spec.credential_binding] = _sha256_bytes(
                f"{type(exc).__name__}:{exc}".encode("utf-8", errors="replace")
            )
        finally:
            current = dict(slots.get(spec.credential_binding, {}))
            current.update(
                {
                    "kind": spec.kind,
                    "credential_binding": spec.credential_binding,
                    "target_path": str(spec.path),
                    "absent": not os.path.lexists(spec.path),
                }
            )
            slots[spec.credential_binding] = current
    all_absent = all(slot["absent"] for slot in slots.values())
    result = _write_lifecycle_receipt(
        plan,
        stage="stopped" if not errors and all_absent else "failure",
        value={
            "operation": "partial_or_prestart_secret_retirement",
            "service_stop_proof": copy.deepcopy(dict(exact_stop_proof)),
            "service_stop_proof_sha256": exact_stop_proof[
                "stop_proof_sha256"
            ],
            "slots": slots,
            "error_sha256s": errors,
            "all_six_credentials_absent_readback": all_absent,
            "all_six_install_bound_retirement_completions": all(
                slot.get("install_bound_retirement") is True
                for slot in slots.values()
            ),
            "ok": not errors and all_absent,
            "completed_at_unix": int(time.time()),
        },
    )
    return result


def validate_plan_against_full(plan: CapabilityCanaryPlan, full_plan: FullCanaryPlan) -> None:
    if plan.revision != full_plan.revision or plan.full_canary_plan_sha256 != full_plan.sha256 or plan.release_artifact_sha256 != full_plan.release["artifact_sha256"] or plan.release_root != Path(full_plan.release["artifact_root"]) or plan.interpreter != Path(full_plan.release["interpreter"]):
        raise RuntimeError("capability plan is not bound to the sealed full canary")
    if (plan.identities.gateway_user, plan.identities.gateway_group, plan.identities.gateway_uid, plan.identities.gateway_gid, plan.identities.socket_client_group, plan.identities.socket_client_gid, plan.identities.edge_group) != (full_plan.identities.gateway_user, full_plan.identities.gateway_group, full_plan.identities.gateway_uid, full_plan.identities.gateway_gid, full_plan.identities.socket_client_group, full_plan.identities.socket_client_gid, full_plan.identities.edge_group):
        raise RuntimeError("capability identities differ from the full canary")
    if (
        plan.identities.browser_user
        in {
            full_plan.identities.gateway_user,
            full_plan.identities.writer_user,
            full_plan.identities.edge_user,
            plan.identities.connector_user,
            plan.identities.mac_ops_user,
            plan.identities.worker_user,
            DEFAULT_PROJECTOR_USER,
        }
        or plan.identities.browser_group
        in {
            full_plan.identities.gateway_group,
            full_plan.identities.writer_group,
            full_plan.identities.edge_group,
            plan.identities.connector_group,
            plan.identities.mac_ops_group,
            plan.identities.worker_group,
            plan.identities.worker_client_group,
            DEFAULT_PROJECTOR_GROUP,
        }
        or plan.identities.browser_uid
        in {
            full_plan.identities.gateway_uid,
            full_plan.identities.writer_uid,
            full_plan.identities.edge_uid,
            plan.identities.connector_uid,
            plan.identities.mac_ops_uid,
            plan.identities.worker_uid,
        }
        or plan.identities.browser_gid
        in {
            full_plan.identities.gateway_gid,
            full_plan.identities.writer_gid,
            full_plan.identities.edge_gid,
            plan.identities.connector_gid,
            plan.identities.mac_ops_gid,
            plan.identities.socket_client_gid,
            plan.identities.worker_gid,
            plan.identities.worker_client_gid,
        }
    ):
        raise RuntimeError("capability browser identity is not isolated")
    if (
        plan.identities.worker_user
        in {
            full_plan.identities.gateway_user,
            full_plan.identities.writer_user,
            full_plan.identities.edge_user,
            plan.identities.connector_user,
            plan.identities.mac_ops_user,
            plan.identities.browser_user,
            DEFAULT_PROJECTOR_USER,
        }
        or plan.identities.worker_group
        in {
            full_plan.identities.gateway_group,
            full_plan.identities.writer_group,
            full_plan.identities.edge_group,
            plan.identities.connector_group,
            plan.identities.mac_ops_group,
            plan.identities.browser_group,
            plan.identities.worker_client_group,
            DEFAULT_PROJECTOR_GROUP,
        }
        or plan.identities.worker_uid
        in {
            full_plan.identities.gateway_uid,
            full_plan.identities.writer_uid,
            full_plan.identities.edge_uid,
            plan.identities.connector_uid,
            plan.identities.mac_ops_uid,
            plan.identities.browser_uid,
        }
        or plan.identities.worker_gid
        in {
            full_plan.identities.gateway_gid,
            full_plan.identities.writer_gid,
            full_plan.identities.edge_gid,
            plan.identities.connector_gid,
            plan.identities.mac_ops_gid,
            plan.identities.browser_gid,
            plan.identities.worker_client_gid,
            plan.identities.socket_client_gid,
        }
        or plan.identities.worker_client_gid
        in {
            full_plan.identities.gateway_gid,
            full_plan.identities.writer_gid,
            full_plan.identities.edge_gid,
            plan.identities.connector_gid,
            plan.identities.mac_ops_gid,
            plan.identities.browser_gid,
            plan.identities.socket_client_gid,
        }
    ):
        raise RuntimeError("capability isolated-worker identity is not isolated")


def _prepare_gateway_directories(plan: CapabilityCanaryPlan) -> None:
    for path in (
        DEFAULT_GATEWAY_HOME,
        DEFAULT_GATEWAY_PROFILE_HOME,
        DEFAULT_GATEWAY_WORK_ROOT,
        DEFAULT_GATEWAY_LOG_ROOT,
    ):
        try:
            item = os.lstat(path)
        except FileNotFoundError:
            path.mkdir(mode=0o700)
            os.chown(
                path,
                plan.identities.gateway_uid,
                plan.identities.gateway_gid,
            )
            os.chmod(path, 0o700)
            item = os.lstat(path)
            _fsync_directory(path.parent)
        if (
            not stat.S_ISDIR(item.st_mode)
            or stat.S_ISLNK(item.st_mode)
            or item.st_uid != plan.identities.gateway_uid
            or item.st_gid != plan.identities.gateway_gid
            or stat.S_IMODE(item.st_mode) != 0o700
        ):
            raise RuntimeError("capability gateway directory identity is unsafe")


def _prepare_worker_mountpoint() -> Mapping[str, Any]:
    try:
        item = os.lstat(DEFAULT_WORKER_LEASE_BASE)
    except FileNotFoundError:
        DEFAULT_WORKER_LEASE_BASE.mkdir(parents=True, mode=0o700)
        os.chown(DEFAULT_WORKER_LEASE_BASE, 0, 0)
        os.chmod(DEFAULT_WORKER_LEASE_BASE, 0o700)
        _fsync_directory(DEFAULT_WORKER_LEASE_BASE.parent)
        item = os.lstat(DEFAULT_WORKER_LEASE_BASE)
    if (
        not stat.S_ISDIR(item.st_mode)
        or stat.S_ISLNK(item.st_mode)
        or item.st_uid != 0
        or item.st_gid != 0
        or stat.S_IMODE(item.st_mode) != 0o700
    ):
        raise RuntimeError("isolated worker mountpoint identity is unsafe")
    return {
        "path": str(DEFAULT_WORKER_LEASE_BASE),
        "uid": 0,
        "gid": 0,
        "mode": "0700",
        "ephemeral_runtime_tmpfs": True,
    }


def _execution_cleanup_snapshot(plan: CapabilityCanaryPlan) -> Mapping[str, Any]:
    worker_has_entries = False
    if os.path.lexists(DEFAULT_WORKER_LEASE_BASE):
        worker = os.lstat(DEFAULT_WORKER_LEASE_BASE)
        with os.scandir(DEFAULT_WORKER_LEASE_BASE) as entries:
            worker_has_entries = next(entries, None) is not None
        if (
            not stat.S_ISDIR(worker.st_mode)
            or stat.S_ISLNK(worker.st_mode)
            or worker.st_uid != 0
            or worker.st_gid != 0
            or stat.S_IMODE(worker.st_mode) != 0o700
            or worker_has_entries
        ):
            raise RuntimeError("isolated worker lease state was not retired")
    browser_entries: list[str] = []
    if os.path.lexists(DEFAULT_BROWSER_STATE):
        browser = os.lstat(DEFAULT_BROWSER_STATE)
        if (
            not stat.S_ISDIR(browser.st_mode)
            or stat.S_ISLNK(browser.st_mode)
            or browser.st_uid != plan.identities.browser_uid
            or browser.st_gid != plan.identities.browser_gid
            or stat.S_IMODE(browser.st_mode) != 0o700
        ):
            raise RuntimeError("browser controller state identity drifted")
        with os.scandir(DEFAULT_BROWSER_STATE) as entries:
            browser_entries = sorted(entry.name for entry in entries)
        if browser_entries:
            raise RuntimeError("browser controller session state was not retired")
    return {
        "isolated_worker_lease_empty": not worker_has_entries,
        "browser_session_root_empty": not browser_entries,
        "secret_material_recorded": False,
    }


def _capability_services(
    *,
    runner: Callable[[Command], subprocess.CompletedProcess[bytes]],
) -> Mapping[str, Mapping[str, Any]]:
    return {
        unit: collect_capability_service_state(unit, runner=runner)
        for unit in (
            EDGE_UNIT_NAME,
            WRITER_UNIT_NAME,
            GATEWAY_UNIT_NAME,
            PHASE_B_READINESS_UNIT_NAME,
            MAC_OPS_UNIT_NAME,
            DEFAULT_BROWSER_UNIT_NAME,
            DEFAULT_DISCORD_CONNECTOR_UNIT,
            DEFAULT_WORKER_SOCKET_UNIT_NAME,
            DEFAULT_WORKER_SERVICE_UNIT_NAME,
            BITRIX_OPERATIONAL_EDGE_UNIT,
            *(
                CAPABILITY_PRODUCER_SERVICE_UNITS[role]
                for role in CAPABILITY_PRODUCER_ROLES
            ),
        )
    }


def _producer_foundation_preflight(
    plan: CapabilityCanaryPlan,
    full_plan: FullCanaryPlan,
) -> Mapping[str, Any]:
    from gateway.canonical_capability_canary_producer_units import (
        validate_installed_producer_foundation,
    )

    value = validate_installed_producer_foundation(
        plan=plan,
        full_plan=full_plan,
    )
    if (
        not isinstance(value, Mapping)
        or value.get("revision") != plan.revision
        or value.get("ready") is not True
        or value.get("mutation_performed") is not False
    ):
        raise RuntimeError("producer foundation preflight is invalid")
    return copy.deepcopy(dict(value))


def _cleanup_observer_identity() -> Mapping[str, int]:
    from gateway.canonical_capability_canary_producers import (
        load_installed_producer_foundation,
    )

    installed = load_installed_producer_foundation()
    endpoint = installed.value["endpoints"][CAPABILITY_OBSERVER_ROLE]
    uid = endpoint.get("uid")
    gid = endpoint.get("gid")
    if type(uid) is not int or type(gid) is not int or uid <= 0 or gid <= 0:
        raise RuntimeError("cleanup observer identity is invalid")
    return {"uid": uid, "gid": gid}


def build_capability_cleanup_facts(
    plan: CapabilityCanaryPlan,
    *,
    services: Mapping[str, Mapping[str, Any]],
    credential_consumer_stop_proof: Mapping[str, Any],
    producer_foundation: Mapping[str, Any],
    retirements: Mapping[str, Mapping[str, Any]],
    retirement_receipt_sha256s: Mapping[str, str],
    credential_absence: Mapping[str, Mapping[str, Any]],
    bitrix_receipt_key_retirement: Mapping[str, Any],
    bitrix_receipt_key_absence: Mapping[str, Any],
    execution_cleanup: Mapping[str, Any],
    observed_at_unix_ms: int | None = None,
) -> Mapping[str, Any]:
    """Build only mechanical root facts; the observer authors semantics."""

    if set(services) != set(CAPABILITY_STOP_ORDER):
        raise ValueError("cleanup facts service inventory is not exact")
    non_observer_states = {
        unit: copy.deepcopy(dict(services[unit]))
        for unit in CAPABILITY_PRE_CLEANUP_STOP_ORDER
    }
    if not all(_service_stopped(state) for state in non_observer_states.values()):
        raise RuntimeError("cleanup facts contain a live credential consumer")
    proof = _validate_capability_stop_proof(
        plan,
        credential_consumer_stop_proof,
        installed_at_unix=0,
        now_unix=int(time.time()) + 30,
    )
    if proof["non_observer_services_state_sha256"] != _sha256_json(
        non_observer_states
    ):
        raise RuntimeError("cleanup facts do not bind the stopped service state")
    if set(retirements) != set(CAPABILITY_CREDENTIAL_BINDINGS) or set(
        retirement_receipt_sha256s
    ) != set(CAPABILITY_CREDENTIAL_BINDINGS):
        raise RuntimeError("cleanup facts require six exact retirements")
    if set(credential_absence) != set(CAPABILITY_CREDENTIAL_BINDINGS):
        raise RuntimeError("cleanup facts require six exact absence readbacks")
    for binding in CAPABILITY_CREDENTIAL_BINDINGS:
        receipt = retirements[binding]
        if (
            receipt.get("receipt_sha256")
            != retirement_receipt_sha256s[binding]
            or receipt.get("service_stop_proof_sha256")
            != proof["stop_proof_sha256"]
            or credential_absence[binding].get("absent") is not True
        ):
            raise RuntimeError("cleanup facts credential retirement drifted")
    if (
        bitrix_receipt_key_retirement.get("service_stop_proof_sha256")
        != proof["stop_proof_sha256"]
        or bitrix_receipt_key_absence.get("both_pair_members_absent") is not True
        or execution_cleanup.get("isolated_worker_lease_empty") is not True
        or execution_cleanup.get("browser_session_root_empty") is not True
        or execution_cleanup.get("secret_material_recorded") is not False
    ):
        raise RuntimeError("cleanup facts retirement state is incomplete")
    observer_state = services[CAPABILITY_OBSERVER_UNIT]
    observed = (
        int(time.time() * 1000)
        if observed_at_unix_ms is None
        else observed_at_unix_ms
    )
    if type(observed) is not int or observed < proof["observed_at_unix"] * 1000:
        raise ValueError("cleanup facts observation time is invalid")
    observer = {
        "role": CAPABILITY_OBSERVER_ROLE,
        "service_unit": CAPABILITY_OBSERVER_UNIT,
        "live": True,
        "signing_only": True,
        "credential_read_access": False,
        "service_state_sha256": _sha256_json(dict(observer_state)),
        "producer_foundation_sha256": proof["producer_foundation_sha256"],
        "unit_bundle_manifest_sha256": proof["unit_bundle_manifest_sha256"],
        "credential_inaccessibility_contract_sha256": proof[
            "credential_inaccessibility_contract_sha256"
        ],
    }
    if observer["service_state_sha256"] != proof["observer_state_sha256"]:
        raise RuntimeError("cleanup facts observer state drifted")
    unsigned = {
        "schema": CAPABILITY_CLEANUP_FACTS_SCHEMA,
        "revision": plan.revision,
        "capability_plan_sha256": plan.sha256,
        "full_canary_plan_sha256": plan.full_canary_plan_sha256,
        "non_observer_stop_order": list(CAPABILITY_PRE_CLEANUP_STOP_ORDER),
        "non_observer_service_states": non_observer_states,
        "credential_consumer_stop_proof": copy.deepcopy(dict(proof)),
        "observer_signer_identity": observer,
        "retirements": copy.deepcopy(dict(retirements)),
        "retirement_receipt_sha256s": copy.deepcopy(
            dict(retirement_receipt_sha256s)
        ),
        "credential_absence": copy.deepcopy(dict(credential_absence)),
        "bitrix_receipt_key_retirement": copy.deepcopy(
            dict(bitrix_receipt_key_retirement)
        ),
        "bitrix_receipt_key_absence": copy.deepcopy(
            dict(bitrix_receipt_key_absence)
        ),
        "browser_session_retirement": {
            "path": str(DEFAULT_BROWSER_STATE),
            "empty": True,
            "retired": True,
            "secret_material_recorded": False,
        },
        "isolated_worker_lease_cleanup": {
            "path": str(DEFAULT_WORKER_LEASE_BASE),
            "empty": True,
            "retired": True,
            "secret_material_recorded": False,
        },
        "observed_at_unix_ms": observed,
    }
    return {**unsigned, "facts_sha256": _sha256_json(unsigned)}


def publish_capability_cleanup_facts(
    facts: Mapping[str, Any],
    *,
    run_id: str,
    observer_gid: int,
) -> Mapping[str, Any]:
    """Publish immutable root facts readable by only the observer group."""

    from gateway.canonical_capability_canary_producers import DEFAULT_RECEIPT_ROOT

    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id or "") is None:
        raise ValueError("cleanup facts run id is invalid")
    if type(observer_gid) is not int or observer_gid <= 0:
        raise ValueError("cleanup facts observer group is invalid")
    if (
        facts.get("schema") != CAPABILITY_CLEANUP_FACTS_SCHEMA
        or facts.get("facts_sha256")
        != _sha256_json(
            {key: item for key, item in facts.items() if key != "facts_sha256"}
        )
    ):
        raise ValueError("cleanup facts are invalid")
    directory = DEFAULT_RECEIPT_ROOT / run_id
    item = os.lstat(directory)
    if (
        stat.S_ISLNK(item.st_mode)
        or not stat.S_ISDIR(item.st_mode)
        or item.st_uid != 0
        or stat.S_IMODE(item.st_mode) != 0o3770
    ):
        raise RuntimeError("cleanup facts run directory is unsafe")
    path = directory / "cleanup-facts.json"
    raw = _canonical_bytes(facts)
    installed = _atomic_no_replace_file(
        path,
        raw,
        uid=0,
        gid=observer_gid,
        mode=0o440,
        temporary_name=".cleanup-facts.installing",
        maximum=2 * 1024 * 1024,
    )
    return {
        "facts": copy.deepcopy(dict(facts)),
        "facts_path": str(path),
        "facts_file_sha256": _sha256_bytes(raw),
        "facts_uid": installed.st_uid,
        "facts_gid": installed.st_gid,
        "facts_mode": f"{stat.S_IMODE(installed.st_mode):04o}",
    }


def _production_observation_marker_path(
    *, run_id: str, phase: str
) -> Path:
    from gateway.canonical_capability_canary_producers import (
        DEFAULT_RECEIPT_ROOT,
    )

    if (
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id or "")
        is None
        or phase not in {"before", "after"}
    ):
        raise ValueError("production observation marker binding is invalid")
    return DEFAULT_RECEIPT_ROOT / run_id / f"awaiting-production-{phase}.json"


def read_production_observation_wait_request(
    stream: BinaryIO,
    *,
    plan: CapabilityCanaryPlan,
) -> Mapping[str, Any]:
    """Read one canonical, non-secret marker-wait request from stdin."""

    raw = stream.read(64 * 1024 + 1)
    if not raw or len(raw) > 64 * 1024 or stream.read(1):
        raise ValueError("production observation wait request is invalid")
    value = _decode_json(raw, label="production observation wait request")
    if raw != _canonical_bytes(value):
        raise ValueError("production observation wait request is not canonical")
    fields = {
        "schema",
        "phase",
        "canary_revision",
        "capability_plan_sha256",
        "full_canary_plan_sha256",
        "fixture_sha256",
        "run_id",
        "owner_subject_sha256",
        "timeout_seconds",
        "secret_material_recorded",
        "secret_digest_recorded",
    }
    request = _strict_mapping(
        value, fields, "production observation wait request"
    )
    if (
        request["schema"]
        != CAPABILITY_PRODUCTION_OBSERVATION_WAIT_REQUEST_SCHEMA
        or request["phase"] not in {"before", "after"}
        or request["canary_revision"] != plan.revision
        or request["capability_plan_sha256"] != plan.sha256
        or request["full_canary_plan_sha256"]
        != plan.full_canary_plan_sha256
        or re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}",
            str(request["run_id"] or ""),
        )
        is None
        or type(request["timeout_seconds"]) is not int
        or not 1 <= request["timeout_seconds"] <= 300
        or request["secret_material_recorded"] is not False
        or request["secret_digest_recorded"] is not False
    ):
        raise PermissionError("production observation wait request is invalid")
    _digest(request["fixture_sha256"], "fixture")
    _digest(request["owner_subject_sha256"], "owner subject")
    return copy.deepcopy(dict(request))


def wait_for_capability_production_observation_marker(
    plan: CapabilityCanaryPlan,
    request: Mapping[str, Any],
    *,
    observer_gid: int,
    poll_seconds: float = 0.05,
) -> Mapping[str, Any]:
    """Boundedly wait for one exact immutable live-driver marker."""

    if not isinstance(plan, CapabilityCanaryPlan):
        raise TypeError("sealed capability plan is required")
    if type(observer_gid) is not int or observer_gid <= 0:
        raise ValueError("production observation observer group is invalid")
    if not 0.01 <= poll_seconds <= 1.0:
        raise ValueError("production observation poll interval is invalid")
    # Reuse the stdin validator's exact semantic contract without accepting a
    # second, looser mapping shape inside the privileged runtime.
    canonical = _canonical_bytes(request)
    validated = read_production_observation_wait_request(
        io.BytesIO(canonical),
        plan=plan,
    )
    deadline = time.monotonic() + validated["timeout_seconds"]
    marker_path = _production_observation_marker_path(
        run_id=validated["run_id"], phase=validated["phase"]
    )
    while not os.path.lexists(marker_path):
        if time.monotonic() >= deadline:
            raise TimeoutError("production observation marker wait expired")
        time.sleep(poll_seconds)
    marker = load_capability_production_observation_marker(
        plan,
        phase=validated["phase"],
        fixture_sha256=validated["fixture_sha256"],
        run_id=validated["run_id"],
        owner_subject_sha256=validated["owner_subject_sha256"],
        observer_gid=observer_gid,
        require_current_observer=validated["phase"] == "after",
    )
    unsigned = {
        "schema": "muncho-production-capability-production-observation-marker-wait.v1",
        "phase": validated["phase"],
        "run_id": validated["run_id"],
        "fixture_sha256": validated["fixture_sha256"],
        "marker_sha256": marker["marker_sha256"],
        "observer_live_verified": validated["phase"] == "after",
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    return {**unsigned, "receipt_sha256": _sha256_json(unsigned)}


def read_owner_signed_production_observation(
    stream: BinaryIO,
) -> Mapping[str, Any]:
    raw = stream.read(2 * 1024 * 1024 + 1)
    if not raw or len(raw) > 2 * 1024 * 1024 or stream.read(1):
        raise ValueError("production observation envelope input is invalid")
    value = _decode_json(raw, label="production observation envelope input")
    if raw != _canonical_bytes(value) or not isinstance(value, Mapping):
        raise ValueError("production observation envelope input is not canonical")
    return copy.deepcopy(dict(value))


def stage_and_publish_owner_signed_production_observation(
    envelope: Mapping[str, Any],
    *,
    plan: CapabilityCanaryPlan,
    observer_gid: int,
) -> Mapping[str, Any]:
    """Stage one exact owner envelope and publish the after no-change diff."""

    if not isinstance(envelope, Mapping):
        raise TypeError("production observation envelope is required")
    phase = envelope.get("phase")
    fixture_sha256 = envelope.get("fixture_sha256")
    run_id = envelope.get("run_id")
    owner_subject_sha256 = envelope.get("owner_subject_sha256")
    if phase not in {"before", "after"} or not isinstance(run_id, str):
        raise ValueError("production observation envelope binding is invalid")
    _digest(fixture_sha256, "fixture")
    _digest(owner_subject_sha256, "owner subject")
    staged = stage_owner_signed_production_observation(
        envelope,
        plan=plan,
        phase=phase,
        fixture_sha256=fixture_sha256,
        run_id=run_id,
        owner_subject_sha256=owner_subject_sha256,
        observer_gid=observer_gid,
    )
    diff_publication: Mapping[str, Any] | None = None
    if phase == "after":
        from gateway.canonical_capability_canary_producers import (
            DEFAULT_RECEIPT_ROOT,
        )

        before_path = (
            DEFAULT_RECEIPT_ROOT
            / run_id
            / "production-observation-before.json"
        )
        before_raw, _before_item = _read_exact_file(
            before_path,
            maximum=2 * 1024 * 1024,
            uid=0,
            gid=observer_gid,
            mode=0o440,
        )
        before_envelope = _decode_json(
            before_raw, label="staged production before observation"
        )
        before_signed_at = (
            before_envelope.get("signed_at_unix_ms")
            if isinstance(before_envelope, Mapping)
            else None
        )
        if (
            before_raw != _canonical_bytes(before_envelope)
            or type(before_signed_at) is not int
            or before_signed_at <= 0
        ):
            raise ValueError(
                "staged production before observation is invalid"
            )
        # The before marker is intentionally short lived.  Revalidate the
        # immutable staged envelope against the time it was owner-signed, not
        # the later end of a legitimately long canary run.
        before = load_staged_owner_signed_production_observation(
            plan=plan,
            phase="before",
            fixture_sha256=fixture_sha256,
            run_id=run_id,
            owner_subject_sha256=owner_subject_sha256,
            observer_gid=observer_gid,
            now_unix_ms=before_signed_at,
        )
        after = load_staged_owner_signed_production_observation(
            plan=plan,
            phase="after",
            fixture_sha256=fixture_sha256,
            run_id=run_id,
            owner_subject_sha256=owner_subject_sha256,
            observer_gid=observer_gid,
        )
        diff = build_capability_production_diff(
            before,
            after,
            plan=plan,
            fixture_sha256=fixture_sha256,
            run_id=run_id,
            owner_subject_sha256=owner_subject_sha256,
        )
        diff_publication = publish_capability_production_diff(
            diff,
            run_id=run_id,
            observer_gid=observer_gid,
        )
    unsigned = {
        "schema": CAPABILITY_PRODUCTION_OBSERVATION_STAGE_RECEIPT_SCHEMA,
        "phase": phase,
        "run_id": run_id,
        "fixture_sha256": fixture_sha256,
        "staged_envelope_sha256": staged["envelope_sha256"],
        "observation_sha256": envelope["observation_sha256"],
        "marker_sha256": staged["marker_sha256"],
        "production_diff_sha256": (
            diff_publication["diff_sha256"]
            if diff_publication is not None
            else None
        ),
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    return {**unsigned, "receipt_sha256": _sha256_json(unsigned)}


def _require_live_cleanup_observer(
    *,
    state_reader: Callable[[str], Mapping[str, Any]] | None = None,
) -> Mapping[str, Any]:
    state = (
        collect_capability_service_state(
            CAPABILITY_OBSERVER_UNIT,
            runner=_runner,
        )
        if state_reader is None
        else state_reader(CAPABILITY_OBSERVER_UNIT)
    )
    if not isinstance(state, Mapping) or not _service_live(
        state,
        path=Path("/etc/systemd/system") / CAPABILITY_OBSERVER_UNIT,
        service_type="notify",
    ):
        raise RuntimeError("production observation cleanup observer is not live")
    return copy.deepcopy(dict(state))


def publish_capability_production_observation_marker(
    plan: CapabilityCanaryPlan,
    *,
    phase: str,
    fixture_sha256: str,
    run_id: str,
    owner_subject_sha256: str,
    observer_gid: int,
    now_unix_ms: int | None = None,
    observer_state_reader: Callable[[str], Mapping[str, Any]] | None = None,
) -> Mapping[str, Any]:
    """Publish one immutable owner/fixture-bound handshake marker.

    The ``before`` marker is published before any canary service starts.  The
    ``after`` marker is publishable only while the credential-blind cleanup
    observer is the exact live systemd MainPID.  Neither marker contains
    production data or task semantics.
    """

    if not isinstance(plan, CapabilityCanaryPlan):
        raise TypeError("sealed capability plan is required")
    if type(observer_gid) is not int or observer_gid <= 0:
        raise ValueError("production observation observer group is invalid")
    fixture = _digest(fixture_sha256, "fixture")
    owner = _digest(owner_subject_sha256, "owner subject")
    path = _production_observation_marker_path(run_id=run_id, phase=phase)
    directory = path.parent
    parent = os.lstat(directory)
    if (
        stat.S_ISLNK(parent.st_mode)
        or not stat.S_ISDIR(parent.st_mode)
        or parent.st_uid != 0
        or stat.S_IMODE(parent.st_mode) != 0o3770
    ):
        raise RuntimeError("production observation run directory is unsafe")
    created = (
        int(time.time() * 1000) if now_unix_ms is None else now_unix_ms
    )
    if type(created) is not int or created <= 0:
        raise ValueError("production observation marker time is invalid")
    observer_state = (
        _require_live_cleanup_observer(state_reader=observer_state_reader)
        if phase == "after"
        else None
    )
    unsigned = {
        "schema": CAPABILITY_PRODUCTION_OBSERVATION_MARKER_SCHEMA,
        "phase": phase,
        "run_id": run_id,
        "canary_revision": plan.revision,
        "capability_plan_sha256": plan.sha256,
        "full_canary_plan_sha256": plan.full_canary_plan_sha256,
        "fixture_sha256": fixture,
        "owner_subject_sha256": owner,
        "observer_live_required": phase == "after",
        "observer_service_unit": CAPABILITY_OBSERVER_UNIT,
        "observer_state_sha256": (
            _sha256_json(observer_state) if observer_state is not None else None
        ),
        "created_at_unix_ms": created,
        "expires_at_unix_ms": created + 300_000,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    marker = {**unsigned, "marker_sha256": _sha256_json(unsigned)}
    payload = _canonical_bytes(marker)
    installed = _atomic_no_replace_file(
        path,
        payload,
        uid=0,
        gid=observer_gid,
        mode=0o440,
        temporary_name=f".awaiting-production-{phase}.installing",
        maximum=64 * 1024,
    )
    return {
        "marker": marker,
        "path": str(path),
        "file_sha256": _sha256_bytes(payload),
        "uid": installed.st_uid,
        "gid": installed.st_gid,
        "mode": f"{stat.S_IMODE(installed.st_mode):04o}",
    }


def load_capability_production_observation_marker(
    plan: CapabilityCanaryPlan,
    *,
    phase: str,
    fixture_sha256: str,
    run_id: str,
    owner_subject_sha256: str,
    observer_gid: int,
    now_unix_ms: int | None = None,
    require_current_observer: bool = False,
    observer_state_reader: Callable[[str], Mapping[str, Any]] | None = None,
) -> Mapping[str, Any]:
    """Stable-read and exact-validate one immutable handshake marker."""

    if not isinstance(plan, CapabilityCanaryPlan):
        raise TypeError("sealed capability plan is required")
    if type(observer_gid) is not int or observer_gid <= 0:
        raise ValueError("production observation observer group is invalid")
    path = _production_observation_marker_path(run_id=run_id, phase=phase)
    raw, _item = _read_exact_file(
        path,
        maximum=64 * 1024,
        uid=0,
        gid=observer_gid,
        mode=0o440,
    )
    marker = _decode_json(raw, label="production observation marker")
    if raw != _canonical_bytes(marker):
        raise ValueError("production observation marker is not canonical")
    fields = {
        "schema",
        "phase",
        "run_id",
        "canary_revision",
        "capability_plan_sha256",
        "full_canary_plan_sha256",
        "fixture_sha256",
        "owner_subject_sha256",
        "observer_live_required",
        "observer_service_unit",
        "observer_state_sha256",
        "created_at_unix_ms",
        "expires_at_unix_ms",
        "secret_material_recorded",
        "secret_digest_recorded",
        "marker_sha256",
    }
    value = _strict_mapping(marker, fields, "production observation marker")
    unsigned = {
        key: copy.deepcopy(item)
        for key, item in value.items()
        if key != "marker_sha256"
    }
    now = int(time.time() * 1000) if now_unix_ms is None else now_unix_ms
    expected_observer = phase == "after"
    if (
        value["schema"] != CAPABILITY_PRODUCTION_OBSERVATION_MARKER_SCHEMA
        or value["phase"] != phase
        or value["run_id"] != run_id
        or value["canary_revision"] != plan.revision
        or value["capability_plan_sha256"] != plan.sha256
        or value["full_canary_plan_sha256"]
        != plan.full_canary_plan_sha256
        or value["fixture_sha256"] != _digest(fixture_sha256, "fixture")
        or value["owner_subject_sha256"]
        != _digest(owner_subject_sha256, "owner subject")
        or value["observer_live_required"] is not expected_observer
        or value["observer_service_unit"] != CAPABILITY_OBSERVER_UNIT
        or type(value["created_at_unix_ms"]) is not int
        or type(value["expires_at_unix_ms"]) is not int
        or type(now) is not int
        or value["expires_at_unix_ms"]
        != value["created_at_unix_ms"] + 300_000
        or not value["created_at_unix_ms"] <= now <= value["expires_at_unix_ms"]
        or value["secret_material_recorded"] is not False
        or value["secret_digest_recorded"] is not False
        or value["marker_sha256"] != _sha256_json(unsigned)
    ):
        raise PermissionError("production observation marker is invalid")
    if phase == "before":
        if value["observer_state_sha256"] is not None:
            raise PermissionError("production before marker contains observer state")
    else:
        _digest(value["observer_state_sha256"], "production observer state")
        if require_current_observer:
            current = _require_live_cleanup_observer(
                state_reader=observer_state_reader
            )
            if _sha256_json(current) != value["observer_state_sha256"]:
                raise RuntimeError("production cleanup observer identity changed")
    return copy.deepcopy(dict(value))


def validate_owner_signed_production_observation(
    value: Any,
    *,
    plan: CapabilityCanaryPlan,
    phase: str,
    fixture_sha256: str,
    run_id: str,
    owner_subject_sha256: str,
    now_unix_ms: int | None = None,
) -> Mapping[str, Any]:
    """Verify owner SSHSIG over a fixed observation from the real prod VM."""

    from gateway.canonical_capability_canary_producers import (
        load_installed_producer_foundation,
        project_pinned_owner_public_key_source,
    )
    from gateway.canonical_writer_foundation_phase_b import verify_phase_b_sshsig

    fields = {
        "schema",
        "phase",
        "canary_revision",
        "capability_plan_sha256",
        "full_canary_plan_sha256",
        "fixture_sha256",
        "run_id",
        "observation",
        "observation_sha256",
        "transport_authority",
        "owner_subject_sha256",
        "owner_public_authority",
        "signed_at_unix_ms",
        "secret_material_recorded",
        "secret_digest_recorded",
        "owner_signature",
        "envelope_sha256",
    }
    raw = _strict_mapping(value, fields, "production observation envelope")
    signed = {
        key: copy.deepcopy(item)
        for key, item in raw.items()
        if key != "envelope_sha256"
    }
    unsigned = {
        key: copy.deepcopy(item)
        for key, item in signed.items()
        if key != "owner_signature"
    }
    validated_now = (
        int(time.time() * 1000) if now_unix_ms is None else now_unix_ms
    )
    if (
        raw["schema"] != CAPABILITY_PRODUCTION_OBSERVATION_ENVELOPE_SCHEMA
        or raw["phase"] != phase
        or raw["canary_revision"] != plan.revision
        or raw["capability_plan_sha256"] != plan.sha256
        or raw["full_canary_plan_sha256"]
        != plan.full_canary_plan_sha256
        or raw["fixture_sha256"] != _digest(fixture_sha256, "fixture")
        or raw["run_id"] != run_id
        or raw["owner_subject_sha256"]
        != _digest(owner_subject_sha256, "owner subject")
        or raw["secret_material_recorded"] is not False
        or raw["secret_digest_recorded"] is not False
        or raw["envelope_sha256"] != _sha256_json(signed)
        or type(raw["signed_at_unix_ms"]) is not int
        or type(validated_now) is not int
        or not 0 <= validated_now - raw["signed_at_unix_ms"] <= 300_000
    ):
        raise PermissionError("production observation envelope is invalid")
    authority = _strict_mapping(
        raw["transport_authority"],
        {
            "kind",
            "project",
            "zone",
            "vm",
            "instance_id",
            "known_hosts_file_sha256",
            "observer_source_sha256",
            "instance_authorization_sha256",
            "project_authorization_sha256",
            "oslogin_authorization_sha256",
        },
        "production observation transport authority",
    )
    if (
        authority["kind"] != "pinned_owner_gcloud_iap_ssh_read_only"
        or authority["project"] != "adventico-ai-platform"
        or authority["zone"] != "europe-west3-a"
        or authority["vm"] != "ai-platform-runtime-01"
        or authority["instance_id"] != "1094477181810932795"
    ):
        raise PermissionError("production observation transport is not pinned")
    for field in (
        "known_hosts_file_sha256",
        "observer_source_sha256",
        "instance_authorization_sha256",
        "project_authorization_sha256",
        "oslogin_authorization_sha256",
    ):
        _digest(authority[field], f"production transport {field}")
    try:
        from scripts.canary.production_capability_observer import (
            ProductionObservationError,
            validate_production_observation,
        )
    except ImportError as exc:
        raise PermissionError("production observation is invalid") from exc
    try:
        observation = validate_production_observation(
            raw["observation"],
            phase=phase,
            canary_revision=plan.revision,
            capability_plan_sha256=plan.sha256,
            full_canary_plan_sha256=plan.full_canary_plan_sha256,
            fixture_sha256=fixture_sha256,
            run_id=run_id,
            now_unix_ms=raw["signed_at_unix_ms"],
        )
    except ProductionObservationError as exc:
        raise PermissionError("production observation is invalid") from exc
    if raw["observation_sha256"] != observation["observation_sha256"]:
        raise PermissionError("production observation digest is invalid")
    installed = load_installed_producer_foundation()
    projected_owner = project_pinned_owner_public_key_source(
        raw["owner_public_authority"],
        expected_comment="skyvision-mac-ops-emil-20260710",
    )
    if (
        projected_owner["public_key_ed25519_hex"]
        != installed.pinned_owner_public_key_ed25519_hex
        or projected_owner["public_key_source_sha256"]
        != installed.pinned_owner_public_key_source_sha256
    ):
        raise PermissionError("production observation owner key is not pinned")
    verify_phase_b_sshsig(
        raw["owner_signature"],
        message=_canonical_bytes(unsigned),
        public_key_ed25519_hex=projected_owner["public_key_ed25519_hex"],
        namespace=CAPABILITY_PRODUCTION_OBSERVATION_SSHSIG_NAMESPACE,
    )
    return copy.deepcopy(dict(raw))


def stage_owner_signed_production_observation(
    envelope: Mapping[str, Any],
    *,
    plan: CapabilityCanaryPlan,
    phase: str,
    fixture_sha256: str,
    run_id: str,
    owner_subject_sha256: str,
    observer_gid: int,
    now_unix_ms: int | None = None,
    observer_state_reader: Callable[[str], Mapping[str, Any]] | None = None,
) -> Mapping[str, Any]:
    from gateway.canonical_capability_canary_producers import DEFAULT_RECEIPT_ROOT

    now = int(time.time() * 1000) if now_unix_ms is None else now_unix_ms
    marker = load_capability_production_observation_marker(
        plan,
        phase=phase,
        fixture_sha256=fixture_sha256,
        run_id=run_id,
        owner_subject_sha256=owner_subject_sha256,
        observer_gid=observer_gid,
        now_unix_ms=now,
        require_current_observer=phase == "after",
        observer_state_reader=observer_state_reader,
    )
    validated = validate_owner_signed_production_observation(
        envelope,
        plan=plan,
        phase=phase,
        fixture_sha256=fixture_sha256,
        run_id=run_id,
        owner_subject_sha256=owner_subject_sha256,
        now_unix_ms=now,
    )
    if type(observer_gid) is not int or observer_gid <= 0:
        raise ValueError("production observation observer group is invalid")
    directory = DEFAULT_RECEIPT_ROOT / run_id
    item = os.lstat(directory)
    if (
        stat.S_ISLNK(item.st_mode)
        or not stat.S_ISDIR(item.st_mode)
        or item.st_uid != 0
        or stat.S_IMODE(item.st_mode) != 0o3770
    ):
        raise RuntimeError("production observation run directory is unsafe")
    path = directory / f"production-observation-{phase}.json"
    observation_at = validated["observation"]["observed_at_unix_ms"]
    if (
        phase == "after"
        and observation_at < marker["created_at_unix_ms"]
    ):
        raise PermissionError("production after observation predates its marker")
    raw = _canonical_bytes(validated)
    installed = _atomic_no_replace_file(
        path,
        raw,
        uid=0,
        gid=observer_gid,
        mode=0o440,
        temporary_name=f".production-observation-{phase}.installing",
        maximum=2 * 1024 * 1024,
    )
    return {
        "phase": phase,
        "path": str(path),
        "file_sha256": _sha256_bytes(raw),
        "envelope_sha256": validated["envelope_sha256"],
        "marker_sha256": marker["marker_sha256"],
        "uid": installed.st_uid,
        "gid": installed.st_gid,
        "mode": f"{stat.S_IMODE(installed.st_mode):04o}",
    }


def load_staged_owner_signed_production_observation(
    *,
    plan: CapabilityCanaryPlan,
    phase: str,
    fixture_sha256: str,
    run_id: str,
    owner_subject_sha256: str,
    observer_gid: int,
    now_unix_ms: int | None = None,
    observer_state_reader: Callable[[str], Mapping[str, Any]] | None = None,
) -> Mapping[str, Any]:
    """Stable-read one staged envelope and revalidate its live marker."""

    from gateway.canonical_capability_canary_producers import DEFAULT_RECEIPT_ROOT

    now = int(time.time() * 1000) if now_unix_ms is None else now_unix_ms
    marker = load_capability_production_observation_marker(
        plan,
        phase=phase,
        fixture_sha256=fixture_sha256,
        run_id=run_id,
        owner_subject_sha256=owner_subject_sha256,
        observer_gid=observer_gid,
        now_unix_ms=now,
        require_current_observer=phase == "after",
        observer_state_reader=observer_state_reader,
    )
    path = DEFAULT_RECEIPT_ROOT / run_id / f"production-observation-{phase}.json"
    raw, _item = _read_exact_file(
        path,
        maximum=2 * 1024 * 1024,
        uid=0,
        gid=observer_gid,
        mode=0o440,
    )
    envelope = _decode_json(raw, label="staged production observation")
    if raw != _canonical_bytes(envelope):
        raise ValueError("staged production observation is not canonical")
    validated = validate_owner_signed_production_observation(
        envelope,
        plan=plan,
        phase=phase,
        fixture_sha256=fixture_sha256,
        run_id=run_id,
        owner_subject_sha256=owner_subject_sha256,
        now_unix_ms=now,
    )
    if (
        phase == "after"
        and validated["observation"]["observed_at_unix_ms"]
        < marker["created_at_unix_ms"]
    ):
        raise PermissionError("production after observation predates its marker")
    return copy.deepcopy(dict(validated))


def build_capability_production_diff(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    plan: CapabilityCanaryPlan,
    fixture_sha256: str,
    run_id: str,
    owner_subject_sha256: str,
    observed_at_unix_ms: int | None = None,
) -> Mapping[str, Any]:
    first_signed_at = (
        before.get("signed_at_unix_ms") if isinstance(before, Mapping) else None
    )
    second_signed_at = (
        after.get("signed_at_unix_ms") if isinstance(after, Mapping) else None
    )
    if type(first_signed_at) is not int or type(second_signed_at) is not int:
        raise ValueError("production observation signing time is invalid")
    first = validate_owner_signed_production_observation(
        before,
        plan=plan,
        phase="before",
        fixture_sha256=fixture_sha256,
        run_id=run_id,
        owner_subject_sha256=owner_subject_sha256,
        now_unix_ms=first_signed_at,
    )
    observed = (
        int(time.time() * 1000)
        if observed_at_unix_ms is None
        else observed_at_unix_ms
    )
    second = validate_owner_signed_production_observation(
        after,
        plan=plan,
        phase="after",
        fixture_sha256=fixture_sha256,
        run_id=run_id,
        owner_subject_sha256=owner_subject_sha256,
        now_unix_ms=observed,
    )
    if (
        first["transport_authority"] != second["transport_authority"]
        or first["owner_subject_sha256"] != second["owner_subject_sha256"]
        or first["owner_public_authority"] != second["owner_public_authority"]
    ):
        raise RuntimeError("production observation authority changed")
    before_at = first["observation"]["observed_at_unix_ms"]
    after_at = second["observation"]["observed_at_unix_ms"]
    if (
        type(observed) is not int
        or not before_at < after_at <= second["signed_at_unix_ms"] <= observed
        or first["signed_at_unix_ms"] > second["signed_at_unix_ms"]
    ):
        raise ValueError("production diff ordering is invalid")
    first_observation = first["observation"]
    second_observation = second["observation"]
    first_host = first_observation["host_identity"]
    second_host = second_observation["host_identity"]
    if (
        first_observation["target"] != second_observation["target"]
        or first_host["machine_id_sha256"]
        != second_host["machine_id_sha256"]
        or first_host["hostname_sha256"] != second_host["hostname_sha256"]
    ):
        raise RuntimeError("production observation host identity changed")
    surface_names = (
        "code",
        "config",
        "identities_permissions",
        "jobs",
        "migration_assets",
    )

    def _surface_values(observation: Mapping[str, Any]) -> Mapping[str, Any]:
        surfaces = observation["surfaces"]
        return {
            "code": {
                "active_release": observation["active_release"],
                "gateway_service": observation["gateway_service"],
                "projection": surfaces["code"],
            },
            "config": surfaces["config"],
            "identities_permissions": surfaces["identities_permissions"],
            "jobs": surfaces["jobs"],
            "migration_assets": surfaces["migration_assets"],
        }

    first_surfaces = _surface_values(first_observation)
    second_surfaces = _surface_values(second_observation)
    surface_diffs: dict[str, Mapping[str, Any]] = {}
    changed_surfaces: list[str] = []
    for name in surface_names:
        first_digest = _sha256_json(first_surfaces[name])
        second_digest = _sha256_json(second_surfaces[name])
        changed = first_digest != second_digest
        surface_diffs[name] = {
            "before_sha256": first_digest,
            "after_sha256": second_digest,
            "changed": changed,
        }
        if changed:
            changed_surfaces.append(name)
    static_first = {
        "target": first_observation["target"],
        "machine_id_sha256": first_host["machine_id_sha256"],
        "hostname_sha256": first_host["hostname_sha256"],
        "surfaces": {
            name: surface_diffs[name]["before_sha256"]
            for name in surface_names
        },
    }
    static_second = {
        "target": second_observation["target"],
        "machine_id_sha256": second_host["machine_id_sha256"],
        "hostname_sha256": second_host["hostname_sha256"],
        "surfaces": {
            name: surface_diffs[name]["after_sha256"]
            for name in surface_names
        },
    }
    expected_change_contract = {
        "schema": "muncho-production-capability-production-no-change-contract.v1",
        "run_id": run_id,
        "canary_revision": plan.revision,
        "capability_plan_sha256": plan.sha256,
        "full_canary_plan_sha256": plan.full_canary_plan_sha256,
        "fixture_sha256": fixture_sha256,
        "target": first_observation["target"],
        "expected_changed_surfaces": [],
        "boot_identity_change_allowed": True,
    }
    unsigned = {
        "schema": CAPABILITY_PRODUCTION_DIFF_SCHEMA,
        "run_id": run_id,
        "canary_revision": plan.revision,
        "capability_plan_sha256": plan.sha256,
        "full_canary_plan_sha256": plan.full_canary_plan_sha256,
        "fixture_sha256": fixture_sha256,
        "target": copy.deepcopy(
            dict(first["observation"]["target"])
        ),
        "before_envelope_sha256": first["envelope_sha256"],
        "after_envelope_sha256": second["envelope_sha256"],
        "before_observation_sha256": first["observation_sha256"],
        "after_observation_sha256": second["observation_sha256"],
        "before_observed_at_unix_ms": before_at,
        "after_observed_at_unix_ms": after_at,
        "static_before_sha256": _sha256_json(static_first),
        "static_after_sha256": _sha256_json(static_second),
        "changed_surfaces": changed_surfaces,
        "surface_diffs": surface_diffs,
        "expected_change_contract_sha256": _sha256_json(
            expected_change_contract
        ),
        "unexpected_change_count": len(changed_surfaces),
        "production_mutation_observed": bool(changed_surfaces),
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
        "semantic_job_content_recorded": False,
    }
    return {**unsigned, "diff_sha256": _sha256_json(unsigned)}


def publish_capability_production_diff(
    value: Mapping[str, Any],
    *,
    run_id: str,
    observer_gid: int,
) -> Mapping[str, Any]:
    """Publish the one immutable native production diff for the observer."""

    from gateway.canonical_capability_canary_producers import DEFAULT_RECEIPT_ROOT

    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id or "") is None:
        raise ValueError("production diff run id is invalid")
    if type(observer_gid) is not int or observer_gid <= 0:
        raise ValueError("production diff observer group is invalid")
    fields = {
        "schema",
        "run_id",
        "canary_revision",
        "capability_plan_sha256",
        "full_canary_plan_sha256",
        "fixture_sha256",
        "target",
        "before_envelope_sha256",
        "after_envelope_sha256",
        "before_observation_sha256",
        "after_observation_sha256",
        "before_observed_at_unix_ms",
        "after_observed_at_unix_ms",
        "static_before_sha256",
        "static_after_sha256",
        "changed_surfaces",
        "surface_diffs",
        "expected_change_contract_sha256",
        "unexpected_change_count",
        "production_mutation_observed",
        "secret_material_recorded",
        "secret_digest_recorded",
        "semantic_job_content_recorded",
        "diff_sha256",
    }
    raw = _strict_mapping(value, fields, "production diff")
    unsigned = {key: item for key, item in raw.items() if key != "diff_sha256"}
    surface_names = (
        "code",
        "config",
        "identities_permissions",
        "jobs",
        "migration_assets",
    )
    surface_diffs = _strict_mapping(
        raw["surface_diffs"], set(surface_names), "production surface diffs"
    )
    derived_changed: list[str] = []
    for name in surface_names:
        item = _strict_mapping(
            surface_diffs[name],
            {"before_sha256", "after_sha256", "changed"},
            "production surface diff",
        )
        _digest(item["before_sha256"], "production surface before")
        _digest(item["after_sha256"], "production surface after")
        if type(item["changed"]) is not bool or item["changed"] is not (
            item["before_sha256"] != item["after_sha256"]
        ):
            raise ValueError("production surface diff is invalid")
        if item["changed"]:
            derived_changed.append(name)
    expected_contract = {
        "schema": "muncho-production-capability-production-no-change-contract.v1",
        "run_id": raw["run_id"],
        "canary_revision": raw["canary_revision"],
        "capability_plan_sha256": raw["capability_plan_sha256"],
        "full_canary_plan_sha256": raw["full_canary_plan_sha256"],
        "fixture_sha256": raw["fixture_sha256"],
        "target": raw["target"],
        "expected_changed_surfaces": [],
        "boot_identity_change_allowed": True,
    }
    if (
        raw["schema"] != CAPABILITY_PRODUCTION_DIFF_SCHEMA
        or raw["run_id"] != run_id
        or re.fullmatch(r"[0-9a-f]{40}", str(raw["canary_revision"] or ""))
        is None
        or any(
            re.fullmatch(r"[0-9a-f]{64}", str(raw[field] or "")) is None
            for field in (
                "capability_plan_sha256",
                "full_canary_plan_sha256",
                "fixture_sha256",
                "before_envelope_sha256",
                "after_envelope_sha256",
                "before_observation_sha256",
                "after_observation_sha256",
                "static_before_sha256",
                "static_after_sha256",
                "expected_change_contract_sha256",
            )
        )
        or raw["target"]
        != {
            "project": "adventico-ai-platform",
            "zone": "europe-west3-a",
            "vm": "ai-platform-runtime-01",
            "instance_id": "1094477181810932795",
        }
        or type(raw["before_observed_at_unix_ms"]) is not int
        or type(raw["after_observed_at_unix_ms"]) is not int
        or raw["before_observed_at_unix_ms"]
        >= raw["after_observed_at_unix_ms"]
        or raw["changed_surfaces"] != derived_changed
        or type(raw["unexpected_change_count"]) is not int
        or raw["unexpected_change_count"] != len(derived_changed)
        or type(raw["production_mutation_observed"]) is not bool
        or raw["production_mutation_observed"] is not bool(derived_changed)
        or (
            not derived_changed
            and raw["static_before_sha256"] != raw["static_after_sha256"]
        )
        or raw["expected_change_contract_sha256"]
        != _sha256_json(expected_contract)
        or raw["diff_sha256"] != _sha256_json(unsigned)
        or raw["secret_material_recorded"] is not False
        or raw["secret_digest_recorded"] is not False
        or raw["semantic_job_content_recorded"] is not False
    ):
        raise ValueError("production diff is invalid")
    directory = DEFAULT_RECEIPT_ROOT / run_id
    parent = os.lstat(directory)
    if (
        stat.S_ISLNK(parent.st_mode)
        or not stat.S_ISDIR(parent.st_mode)
        or parent.st_uid != 0
        or stat.S_IMODE(parent.st_mode) != 0o3770
    ):
        raise RuntimeError("production diff run directory is unsafe")
    path = directory / "production-diff.json"
    payload = _canonical_bytes(raw)
    installed = _atomic_no_replace_file(
        path,
        payload,
        uid=0,
        gid=observer_gid,
        mode=0o440,
        temporary_name=".production-diff.installing",
        maximum=2 * 1024 * 1024,
    )
    return {
        "path": str(path),
        "file_sha256": _sha256_bytes(payload),
        "diff_sha256": raw["diff_sha256"],
        "uid": installed.st_uid,
        "gid": installed.st_gid,
        "mode": f"{stat.S_IMODE(installed.st_mode):04o}",
    }


def load_published_capability_production_diff(
    *,
    plan: CapabilityCanaryPlan,
    fixture_sha256: str,
    run_id: str,
    owner_subject_sha256: str,
    observer_gid: int,
    now_unix_ms: int | None = None,
    observer_state_reader: Callable[[str], Mapping[str, Any]] | None = None,
) -> Mapping[str, Any]:
    """Load the immutable no-change diff while the observer is still live."""

    from gateway.canonical_capability_canary_producers import DEFAULT_RECEIPT_ROOT

    now = int(time.time() * 1000) if now_unix_ms is None else now_unix_ms
    marker = load_capability_production_observation_marker(
        plan,
        phase="after",
        fixture_sha256=fixture_sha256,
        run_id=run_id,
        owner_subject_sha256=owner_subject_sha256,
        observer_gid=observer_gid,
        now_unix_ms=now,
        require_current_observer=True,
        observer_state_reader=observer_state_reader,
    )
    path = DEFAULT_RECEIPT_ROOT / run_id / "production-diff.json"
    raw, _item = _read_exact_file(
        path,
        maximum=2 * 1024 * 1024,
        uid=0,
        gid=observer_gid,
        mode=0o440,
    )
    value = _decode_json(raw, label="published production diff")
    if raw != _canonical_bytes(value):
        raise ValueError("published production diff is not canonical")
    # Reuse the exact immutable publisher validator.  Existing identical
    # bytes are accepted without replacement by _atomic_no_replace_file.
    publish_capability_production_diff(
        value,
        run_id=run_id,
        observer_gid=observer_gid,
    )
    if (
        value.get("canary_revision") != plan.revision
        or value.get("capability_plan_sha256") != plan.sha256
        or value.get("full_canary_plan_sha256")
        != plan.full_canary_plan_sha256
        or value.get("fixture_sha256") != _digest(fixture_sha256, "fixture")
        or value.get("changed_surfaces") != []
        or value.get("unexpected_change_count") != 0
        or value.get("production_mutation_observed") is not False
        or value.get("static_before_sha256")
        != value.get("static_after_sha256")
        or type(value.get("after_observed_at_unix_ms")) is not int
        or value["after_observed_at_unix_ms"]
        < marker["created_at_unix_ms"]
    ):
        raise RuntimeError("production no-change diff is invalid")
    return copy.deepcopy(dict(value))


def build_capability_observer_stop_receipt(
    plan: CapabilityCanaryPlan,
    observer_state: Mapping[str, Any],
    *,
    stopped_at_unix_ms: int | None = None,
) -> Mapping[str, Any]:
    if not _service_stopped(observer_state):
        raise RuntimeError("cleanup observer stop receipt contains a live service")
    stopped_at = (
        int(time.time() * 1000)
        if stopped_at_unix_ms is None
        else stopped_at_unix_ms
    )
    if type(stopped_at) is not int or stopped_at < 0:
        raise ValueError("cleanup observer stop time is invalid")
    unsigned = {
        "schema": CAPABILITY_OBSERVER_STOP_RECEIPT_SCHEMA,
        "plan_sha256": plan.sha256,
        "service_unit": CAPABILITY_OBSERVER_UNIT,
        "service_state_sha256": _sha256_json(dict(observer_state)),
        "stopped": True,
        "stopped_at_unix_ms": stopped_at,
        "secret_material_recorded": False,
    }
    return {**unsigned, "receipt_sha256": _sha256_json(unsigned)}


def build_capability_cleanup_finalization(
    plan: CapabilityCanaryPlan,
    *,
    cleanup_receipt: Mapping[str, Any],
    observer_stop_receipt: Mapping[str, Any],
    service_stop_proof: Mapping[str, Any],
    producer_fleet_retirement: Mapping[str, Any],
    producer_activation_absent: bool,
    credentials_absent: bool,
    bitrix_receipt_key_pair_absent: bool,
    full_canary_stopped_preflight_sha256: str,
    finalized_at_unix_ms: int | None = None,
) -> Mapping[str, Any]:
    """Root-authored final truth after the observer and activation are gone."""

    stop = _validate_capability_stop_proof(
        plan,
        service_stop_proof,
        installed_at_unix=0,
        now_unix=int(time.time()) + 30,
    )
    if stop["schema"] != CAPABILITY_SERVICE_STOP_PROOF_SCHEMA:
        raise ValueError("cleanup finalization requires the all-unit stop proof")
    observer = _strict_mapping(
        observer_stop_receipt,
        {
            "schema",
            "plan_sha256",
            "service_unit",
            "service_state_sha256",
            "stopped",
            "stopped_at_unix_ms",
            "secret_material_recorded",
            "receipt_sha256",
        },
        "cleanup observer stop receipt",
    )
    observer_unsigned = {
        key: item for key, item in observer.items() if key != "receipt_sha256"
    }
    if (
        observer["schema"] != CAPABILITY_OBSERVER_STOP_RECEIPT_SCHEMA
        or observer["plan_sha256"] != plan.sha256
        or observer["service_unit"] != CAPABILITY_OBSERVER_UNIT
        or observer["stopped"] is not True
        or observer["secret_material_recorded"] is not False
        or type(observer["stopped_at_unix_ms"]) is not int
        or observer["receipt_sha256"] != _sha256_json(observer_unsigned)
    ):
        raise ValueError("cleanup observer stop receipt is invalid")
    fleet_fields = {
        "schema",
        "readiness_sha256",
        "foundation_sha256",
        "release_sha",
        "capability_plan_sha256",
        "full_canary_plan_sha256",
        "fixture_sha256",
        "run_id",
        "path",
        "retired",
        "absence_verified",
        "retired_at_unix_ms",
        "receipt_sha256",
    }
    fleet = _strict_mapping(
        producer_fleet_retirement,
        fleet_fields,
        "producer fleet retirement",
    )
    fleet_unsigned = {
        key: item for key, item in fleet.items() if key != "receipt_sha256"
    }
    cleanup = _strict_mapping(
        cleanup_receipt,
        {
            "schema",
            "authority_role",
            "key_id",
            "signature_algorithm",
            "payload",
            "native_evidence",
            "signature",
        },
        "observer cleanup signed receipt",
    )
    cleanup_payload = cleanup["payload"]
    if (
        fleet["schema"] != "muncho-production-capability-fleet-retirement.v1"
        or fleet["release_sha"] != plan.revision
        or fleet["capability_plan_sha256"] != plan.sha256
        or fleet["full_canary_plan_sha256"] != plan.full_canary_plan_sha256
        or fleet["retired"] is not True
        or fleet["absence_verified"] is not True
        or fleet["receipt_sha256"] != _sha256_json(fleet_unsigned)
        or cleanup["schema"]
        != "muncho-production-capability-canary-signed-receipt.v1"
        or cleanup["authority_role"] != CAPABILITY_OBSERVER_ROLE
        or cleanup["signature_algorithm"] != "ed25519"
        or re.fullmatch(r"[0-9a-f]{64}", str(cleanup["key_id"])) is None
        or re.fullmatch(r"[0-9a-f]{128}", str(cleanup["signature"])) is None
        or not isinstance(cleanup_payload, Mapping)
        or cleanup_payload.get("run_id") != fleet["run_id"]
        or cleanup_payload.get("fixture_sha256") != fleet["fixture_sha256"]
    ):
        raise ValueError("cleanup finalization fleet binding is invalid")
    for field in (
        "readiness_sha256",
        "foundation_sha256",
        "fixture_sha256",
    ):
        _digest(fleet[field], f"cleanup finalization {field}")
    _digest(
        full_canary_stopped_preflight_sha256,
        "full canary stopped preflight",
    )
    finalized = (
        int(time.time() * 1000)
        if finalized_at_unix_ms is None
        else finalized_at_unix_ms
    )
    signed_at = cleanup_payload.get("observed_at_unix_ms")
    if (
        type(finalized) is not int
        or type(signed_at) is not int
        or finalized
        < max(
            signed_at,
            observer["stopped_at_unix_ms"],
            fleet["retired_at_unix_ms"],
            stop["observed_at_unix"] * 1000,
        )
        or producer_activation_absent is not True
        or credentials_absent is not True
        or bitrix_receipt_key_pair_absent is not True
    ):
        raise ValueError("cleanup finalization is not terminal truth")
    unsigned = {
        "schema": CAPABILITY_CLEANUP_FINALIZATION_SCHEMA,
        "release_sha": plan.revision,
        "capability_plan_sha256": plan.sha256,
        "full_canary_plan_sha256": plan.full_canary_plan_sha256,
        "fixture_sha256": fleet["fixture_sha256"],
        "run_id": fleet["run_id"],
        "cleanup_receipt_sha256": _sha256_json(dict(cleanup)),
        "observer_stop_receipt": copy.deepcopy(dict(observer)),
        "service_stop_proof": copy.deepcopy(dict(stop)),
        "producer_fleet_retirement": copy.deepcopy(dict(fleet)),
        "producer_activation_absent": True,
        "credentials_absent": True,
        "bitrix_receipt_key_pair_absent": True,
        "full_canary_stopped_preflight_sha256": (
            full_canary_stopped_preflight_sha256
        ),
        "finalized_at_unix_ms": finalized,
    }
    return {**unsigned, "finalization_sha256": _sha256_json(unsigned)}


def collect_capability_preflight(
    plan: CapabilityCanaryPlan,
    full_plan: FullCanaryPlan,
    *,
    phase: str,
    runner: Callable[[Command], subprocess.CompletedProcess[bytes]] = _runner,
    metadata_reader: Callable[[str], bytes | str] | None = None,
    local_identity_reader: Callable[[str], bytes | str] | None = None,
) -> Mapping[str, Any]:
    """Observe the exact stopped/live capability runtime without mutation."""

    if phase not in {"stopped", "live"}:
        raise ValueError("capability preflight phase is invalid")
    validate_plan_against_full(plan, full_plan)
    try:
        host = validate_dedicated_canary_host(
            full_plan,
            metadata_reader=metadata_reader,
            local_identity_reader=local_identity_reader,
        )
    except Exception:
        state = {
            "schema": CAPABILITY_PREFLIGHT_SCHEMA,
            "phase": phase,
            "revision": plan.revision,
            "plan_sha256": plan.sha256,
            "full_canary_plan_sha256": full_plan.sha256,
            "checks": {"host.dedicated_canary_exact": False},
            "blockers": ["host.dedicated_canary_exact"],
        }
        report = {
            **state,
            "state_sha256": _sha256_json(state),
            "observed_at_unix": int(time.time()),
        }
        report["report_sha256"] = _sha256_json(report)
        raise CapabilityCanaryPreflightError(report) from None

    checks: dict[str, bool] = {"host.dedicated_canary_exact": True}
    evidence: dict[str, Any] = {
        "host": host,
        "credential_bindings": _credential_bindings_mapping(),
    }

    def observe(name: str, operation: Callable[[], Any]) -> None:
        try:
            evidence[name] = operation()
            checks[name] = True
        except Exception as exc:
            checks[name] = False
            evidence[f"{name}_error_sha256"] = _sha256_bytes(
                f"{type(exc).__name__}:{exc}".encode("utf-8", errors="replace")
            )

    observe("release.exact", lambda: _validate_release_manifest(full_plan))
    observe(
        "producer.foundation",
        lambda: _producer_foundation_preflight(plan, full_plan),
    )
    if phase == "stopped":
        observe(
            "overlay.targets_absent",
            lambda: _overlay_targets_are_absent(plan, full_plan)
            or (_ for _ in ()).throw(
                RuntimeError("capability overlay targets are not absent")
            ),
        )
    observe(
        "runtime.dependencies",
        lambda: runtime_dependency_manifest_preflight(plan),
    )
    observe("browser.executable", lambda: browser_executable_preflight(plan))
    observe("worker.executables", lambda: worker_executables_preflight(plan))
    observe(
        "worker.systemd252_tmpfs_contract",
        lambda: worker_systemd252_preflight(
            plan,
            allow_create_only_mountpoint_absence=phase == "stopped",
            runner=runner,
        ),
    )
    observe(
        "execution.host_identity",
        lambda: execution_host_identity_receipt(
            plan,
            full_plan,
            allow_create_only_absence=phase == "stopped",
        ),
    )
    browser_identity: Mapping[str, Any] | None = None

    def observe_browser_identity() -> Mapping[str, Any]:
        nonlocal browser_identity
        browser_identity = browser_host_identity_receipt(
            plan,
            full_plan,
            allow_create_only_absence=phase == "stopped",
        )
        return browser_identity

    observe("browser.host_identity", observe_browser_identity)
    observe("browser.userns_sandbox", browser_userns_preflight)
    if phase == "stopped":
        def stopped_browser_principal_smoke() -> Mapping[str, Any]:
            if browser_identity is None:
                raise RuntimeError("capability browser host identity is unavailable")
            if browser_identity["browser"]["state"] != "present_exact":
                return {
                    "state": "deferred_until_create_only_foundation",
                    "browser_host_identity_receipt_sha256": browser_identity[
                        "receipt_sha256"
                    ],
                    "ready": True,
                }
            return browser_principal_version_smoke(plan, runner=runner)

        observe("browser.principal_smoke", stopped_browser_principal_smoke)
    observe(
        "lease.codex",
        lambda: _active_lease_receipt(
            plan,
            kind="codex_access_token",
            target=DEFAULT_GATEWAY_AUTH_STORE,
            journal=DEFAULT_CODEX_LEASE_JOURNAL,
        ),
    )
    observe(
        "lease.mac_ops",
        lambda: _active_lease_receipt(
            plan,
            kind="mac_ops_gitlab_env",
            target=DEFAULT_MAC_OPS_CREDENTIAL,
            journal=DEFAULT_MAC_OPS_LEASE_JOURNAL,
        ),
    )
    observe(
        "lease.discord_connector",
        lambda: _active_lease_receipt(
            plan,
            kind="discord_connector_token",
            target=DEFAULT_CONNECTOR_TOKEN,
            journal=DEFAULT_CONNECTOR_LEASE_JOURNAL,
        ),
    )
    observe(
        "lease.api_control",
        lambda: _active_lease_receipt(
            plan,
            kind="api_server_control_key",
            target=DEFAULT_API_SERVER_CONTROL_KEY,
            journal=DEFAULT_API_CONTROL_LEASE_JOURNAL,
        ),
    )
    observe(
        "lease.bitrix_operational_edge",
        lambda: _active_lease_receipt(
            plan,
            kind="bitrix_operational_edge_webhook",
            target=DEFAULT_BITRIX_WEBHOOK_PATH,
            journal=DEFAULT_BITRIX_LEASE_JOURNAL,
        ),
    )
    observe(
        "lease.discord_routeback",
        lambda: _active_lease_receipt(
            plan,
            kind="discord_routeback_token",
            target=DEFAULT_EDGE_TOKEN_PATH,
            journal=DEFAULT_ROUTEBACK_LEASE_JOURNAL,
        ),
    )
    services: Mapping[str, Mapping[str, Any]] = {}

    def service_observation() -> Mapping[str, Mapping[str, Any]]:
        nonlocal services
        services = _capability_services(runner=runner)
        return services

    observe("services.observed", service_observation)
    if services:
        if phase == "stopped":
            for unit, state in services.items():
                checks[f"service.{unit}.stopped"] = _service_stopped(state)
        else:
            expected = {
                EDGE_UNIT_NAME: (DEFAULT_EDGE_UNIT_PATH, "notify"),
                WRITER_UNIT_NAME: (DEFAULT_WRITER_UNIT_PATH, "notify"),
                GATEWAY_UNIT_NAME: (DEFAULT_GATEWAY_UNIT_PATH, "notify"),
                MAC_OPS_UNIT_NAME: (DEFAULT_MAC_OPS_UNIT_PATH, "simple"),
                DEFAULT_BROWSER_UNIT_NAME: (DEFAULT_BROWSER_UNIT_PATH, "notify"),
                DEFAULT_WORKER_SERVICE_UNIT_NAME: (
                    DEFAULT_WORKER_SERVICE_UNIT_PATH,
                    "simple",
                ),
                DEFAULT_DISCORD_CONNECTOR_UNIT: (
                    DEFAULT_CONNECTOR_UNIT_PATH,
                    "notify",
                ),
                BITRIX_OPERATIONAL_EDGE_UNIT: (
                    DEFAULT_BITRIX_UNIT_PATH,
                    "simple",
                ),
                **{
                    CAPABILITY_PRODUCER_SERVICE_UNITS[role]: (
                        CAPABILITY_PRODUCER_UNIT_PATHS[role],
                        "simple",
                    )
                    for role in CAPABILITY_PRODUCER_ROLES
                },
            }
            for unit, (path, service_type) in expected.items():
                checks[f"service.{unit}.live"] = _service_live(
                    services[unit], path=path, service_type=service_type
                )
            checks[f"service.{PHASE_B_READINESS_UNIT_NAME}.live"] = _oneshot_live(
                services[PHASE_B_READINESS_UNIT_NAME],
                path=DEFAULT_PHASE_B_READINESS_UNIT_PATH,
            )
            checks[f"service.{DEFAULT_WORKER_SOCKET_UNIT_NAME}.live"] = _socket_live(
                services[DEFAULT_WORKER_SOCKET_UNIT_NAME],
                path=DEFAULT_WORKER_SOCKET_UNIT_PATH,
            )
            observe(
                "browser.runtime",
                lambda: browser_service_runtime_preflight(
                    plan, services[DEFAULT_BROWSER_UNIT_NAME]
                ),
            )
            observe(
                "worker.tmpfs_runtime",
                lambda: worker_tmpfs_runtime_preflight(
                    plan, services[DEFAULT_WORKER_SERVICE_UNIT_NAME]
                ),
            )
            observe(
                "execution.readiness",
                lambda: _execution_readiness_as_gateway(plan, runner=runner),
            )
            observe(
                "mac_ops.runtime",
                lambda: _mac_ops_runtime_preflight(plan, services[MAC_OPS_UNIT_NAME]),
            )
            observe(
                "discord_connector.runtime",
                lambda: _connector_runtime_preflight(
                    plan, services[DEFAULT_DISCORD_CONNECTOR_UNIT]
                ),
            )
            observe(
                "bitrix_operational_edge.runtime",
                lambda: _bitrix_runtime_preflight(
                    plan,
                    full_plan,
                    services[BITRIX_OPERATIONAL_EDGE_UNIT],
                ),
            )

            def gateway_readiness() -> Mapping[str, Any]:
                from gateway.canonical_writer_readiness import (
                    READINESS_RECEIPT_VERSION,
                )

                receipt = _readiness_receipt(
                    DEFAULT_GATEWAY_READINESS_PATH,
                    uid=plan.identities.gateway_uid,
                    gid=plan.identities.gateway_gid,
                )
                state = services[GATEWAY_UNIT_NAME]
                digest = readiness_receipt_sha256(receipt)
                if (
                    receipt.get("version") != READINESS_RECEIPT_VERSION
                    or receipt.get("gateway_pid") != state.get("MainPID")
                    or state.get("StatusText")
                    != f"{READINESS_RECEIPT_VERSION}:{digest}"
                ):
                    raise RuntimeError("capability gateway readiness drifted")
                listener = _api_loopback_listener_identity(int(state["MainPID"]))
                return {
                    "receipt_sha256": digest,
                    "gateway_pid": state["MainPID"],
                    "api_loopback_listener": listener,
                    "ready": True,
                }

            observe("gateway.readiness", gateway_readiness)

            def plugin_readiness() -> Mapping[str, Any]:
                edge_state = services[EDGE_UNIT_NAME]
                edge = _validate_edge_collector_gate(full_plan, edge_state)
                edge_identity = readiness_receipt_sha256(edge)
                collector = load_collector_readiness(
                    full_plan,
                    edge_pid=int(edge_state["MainPID"]),
                    edge_service_identity_sha256=edge_identity,
                )
                plugin = _await_plugin_readiness(
                    full_plan,
                    collector=collector,
                    gateway_pid=int(services[GATEWAY_UNIT_NAME]["MainPID"]),
                    edge_pid=int(edge_state["MainPID"]),
                    edge_service_identity_sha256=edge_identity,
                )
                return {
                    "collector_readiness_file_sha256": collector.file_sha256,
                    "plugin_readiness_file_sha256": plugin.file_sha256,
                    "plugin_ready_frame_sha256": plugin.frame_sha256,
                    "ready": True,
                }

            observe("evidence.plugin", plugin_readiness)

    blockers = sorted(name for name, passed in checks.items() if not passed)
    state = {
        "schema": CAPABILITY_PREFLIGHT_SCHEMA,
        "phase": phase,
        "revision": plan.revision,
        "plan_sha256": plan.sha256,
        "full_canary_plan_sha256": full_plan.sha256,
        "checks": checks,
        "blockers": blockers,
        "evidence": evidence,
        "ok": not blockers,
    }
    report = {
        **state,
        "state_sha256": _sha256_json(state),
        "observed_at_unix": int(time.time()),
    }
    report["report_sha256"] = _sha256_json(report)
    if blockers:
        raise CapabilityCanaryPreflightError(report)
    return report


def _await_runtime_ready(
    operation: Callable[[], Mapping[str, Any]],
    *,
    label: str,
    timeout_seconds: float = 30.0,
) -> Mapping[str, Any]:
    if not 0 < timeout_seconds <= 60:
        raise ValueError("capability readiness timeout is invalid")
    deadline = time.monotonic() + timeout_seconds
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        try:
            return operation()
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            last_error = exc
        time.sleep(0.1)
    raise RuntimeError(f"{label} did not become ready") from last_error


def _attempt_capability_stop_order(
    stop: Callable[[str], None],
    *,
    stop_order: Sequence[str] = CAPABILITY_STOP_ORDER,
) -> tuple[list[str], list[BaseException]]:
    """Attempt every requested fixed-order stop despite individual failures."""

    exact = tuple(stop_order)
    if exact not in {
        CAPABILITY_STOP_ORDER,
        CAPABILITY_PRE_CLEANUP_STOP_ORDER,
        (CAPABILITY_OBSERVER_UNIT,),
    }:
        raise ValueError("capability stop attempt order is not exact")
    stopped: list[str] = []
    errors: list[BaseException] = []
    for unit in exact:
        try:
            stop(unit)
            stopped.append(unit)
        except BaseException as exc:
            errors.append(exc)
    return stopped, errors


class CapabilityCanaryLifecycle:
    """Mechanical overlay lifecycle on the sealed full-canary foundation."""

    def __init__(
        self,
        plan: CapabilityCanaryPlan,
        full_plan: FullCanaryPlan,
        *,
        runner: Callable[[Command], subprocess.CompletedProcess[bytes]] = _runner,
        metadata_reader: Callable[[str], bytes | str] | None = None,
        local_identity_reader: Callable[[str], bytes | str] | None = None,
    ) -> None:
        if not isinstance(plan, CapabilityCanaryPlan) or not isinstance(
            full_plan, FullCanaryPlan
        ):
            raise TypeError("sealed capability and full-canary plans are required")
        validate_plan_against_full(plan, full_plan)
        self.plan = plan
        self.full_plan = full_plan
        self.runner = runner
        self.metadata_reader = metadata_reader
        self.local_identity_reader = local_identity_reader

    def _require_host(self) -> Mapping[str, Any]:
        return validate_dedicated_canary_host(
            self.full_plan,
            metadata_reader=self.metadata_reader,
            local_identity_reader=self.local_identity_reader,
        )

    def _stop_command(self, unit: str) -> None:
        try:
            _run_checked(
                Command((SYSTEMCTL, "stop", unit), timeout_seconds=120),
                runner=self.runner,
                label=f"stop {unit}",
            )
        except BaseException:
            state = collect_capability_service_state(unit, runner=self.runner)
            if not _service_stopped(state):
                raise

    def _start_routeback_edge(self) -> Mapping[str, Any]:
        identity = _attest_live_routeback_bot_identity(self.plan, self.full_plan)
        _require_routeback_credential_binding(self.plan, self.full_plan, identity)
        started = False
        try:
            _run_checked(
                edge_start_command(),
                runner=self.runner,
                label=f"start {EDGE_UNIT_NAME}",
            )
            started = True
            _require_routeback_credential_binding(
                self.plan, self.full_plan, identity
            )
        except BaseException as error:
            if not started:
                raise
            try:
                self._stop_command(EDGE_UNIT_NAME)
            except BaseException as stop_error:
                raise BaseExceptionGroup(
                    "Discord route-back credential drift and edge stop failed",
                    [error, stop_error],
                ) from None
            raise
        return identity

    def _cleanup_locked(
        self,
        *,
        cleanup_producer: Callable[[Mapping[str, Any]], Mapping[str, Any]]
        | None = None,
        cleanup_run_id: str | None = None,
        producer_activation_retirer: Callable[[], Mapping[str, Any]]
        | None = None,
    ) -> Mapping[str, Any]:
        stopped, errors = _attempt_capability_stop_order(
            self._stop_command,
            stop_order=CAPABILITY_PRE_CLEANUP_STOP_ORDER,
        )
        execution_cleanup: Mapping[str, Any] | None = None
        try:
            execution_cleanup = _execution_cleanup_snapshot(self.plan)
        except BaseException as exc:
            errors.append(exc)

        connector_cleanup: Mapping[str, Any] | None = None
        try:
            if os.path.lexists(DEFAULT_DISCORD_CONNECTOR_JOURNAL):
                journal = os.lstat(DEFAULT_DISCORD_CONNECTOR_JOURNAL)
                if (
                    stat.S_ISLNK(journal.st_mode)
                    or not stat.S_ISREG(journal.st_mode)
                    or journal.st_uid != self.plan.identities.connector_uid
                    or journal.st_gid != self.plan.identities.connector_gid
                    or stat.S_IMODE(journal.st_mode) != 0o600
                ):
                    raise RuntimeError("Discord connector journal identity drifted")
                connector_cleanup = DurableDiscordConnectorJournal(
                    DEFAULT_DISCORD_CONNECTOR_JOURNAL
                ).cleanup_snapshot()
                if not _connector_cleanup_snapshot_is_safe(connector_cleanup):
                    raise RuntimeError(
                        "Discord connector has unresolved dispatch state"
                    )
            else:
                connector_cleanup = {
                    "journal_absent": True,
                    "safe_to_retire": True,
                }
        except BaseException as exc:
            errors.append(exc)

        if errors:
            raise BaseExceptionGroup(
                "capability stop or safety proof failed closed before restore",
                errors,
            )

        restored: Mapping[str, Any] | None = None
        removed_artifacts: Mapping[str, Any] = {}
        try:
            restored = _restore_full_gateway_unit(self.plan, self.full_plan)
            removed_artifacts = _remove_exact_overlay_artifacts(
                self.plan, self.full_plan
            )
            _run_checked(
                Command((SYSTEMCTL, "daemon-reload")),
                runner=self.runner,
                label="reload restored full-canary unit",
            )
        except BaseException as exc:
            errors.append(exc)

        if errors:
            raise BaseExceptionGroup(
                "capability restore failed closed before credential retirement",
                errors,
            )

        full_stopped: Mapping[str, Any] | None = None
        try:
            full_stopped = collect_full_canary_preflight(
                self.full_plan,
                phase="stopped",
                runner=self.runner,
                metadata_reader=self.metadata_reader,
                local_identity_reader=self.local_identity_reader,
            )
        except BaseException as exc:
            errors.append(exc)

        if errors:
            raise BaseExceptionGroup(
                "full-canary stopped preflight failed before credential retirement",
                errors,
            )

        approval_retirement: Mapping[str, Any] | None = None
        try:
            approval_retirement = _remove_installed_capability_approval(
                self.plan, self.full_plan
            )
        except BaseException as exc:
            errors.append(exc)

        services: Mapping[str, Mapping[str, Any]] = {}
        producer_foundation: Mapping[str, Any] | None = None
        credential_consumer_stop_proof: Mapping[str, Any] | None = None
        try:
            producer_foundation = _producer_foundation_preflight(
                self.plan, self.full_plan
            )
            services = _capability_services(runner=self.runner)
            credential_consumer_stop_proof = (
                build_credential_consumer_stop_proof(
                    self.plan,
                    services,
                    producer_foundation=producer_foundation,
                )
            )
            if stopped != list(CAPABILITY_PRE_CLEANUP_STOP_ORDER):
                raise RuntimeError("non-observer stop order was incomplete")
        except BaseException as exc:
            errors.append(exc)

        if errors:
            # No credential may be removed unless every reader is stopped and
            # the remaining signer is proven credential-blind.
            _observer_stopped, observer_errors = _attempt_capability_stop_order(
                self._stop_command,
                stop_order=(CAPABILITY_OBSERVER_UNIT,),
            )
            errors.extend(observer_errors)
            raise BaseExceptionGroup(
                "credential-consumer stop proof failed before retirement",
                errors,
            )

        bitrix_key_retirement: Mapping[str, Any] | None = None
        try:
            bitrix_key_bootstrap = load_bitrix_key_bootstrap_receipt(
                public_key_id=(
                    self.plan.bitrix_operational_edge_receipt_public_key_id
                ),
                receipt_sha256=(
                    self.plan.bitrix_operational_edge_key_bootstrap_receipt_sha256
                ),
            )
            bitrix_key_retirement = retire_bitrix_foundation_key_pair(
                bitrix_key_bootstrap,
                reason="service_stop",
                plan=self.plan,
                stop_proof=credential_consumer_stop_proof,
            )
        except BaseException as exc:
            errors.append(exc)

        retirements: dict[str, Mapping[str, Any]] = {}
        targets = (
            (
                "api_server_control_key",
                DEFAULT_API_SERVER_CONTROL_KEY,
                DEFAULT_API_CONTROL_LEASE_JOURNAL,
            ),
            (
                "bitrix_operational_edge_webhook",
                DEFAULT_BITRIX_WEBHOOK_PATH,
                DEFAULT_BITRIX_LEASE_JOURNAL,
            ),
            (
                "discord_routeback_token",
                DEFAULT_EDGE_TOKEN_PATH,
                DEFAULT_ROUTEBACK_LEASE_JOURNAL,
            ),
            (
                "discord_connector_token",
                DEFAULT_CONNECTOR_TOKEN,
                DEFAULT_CONNECTOR_LEASE_JOURNAL,
            ),
            (
                "codex_access_token",
                DEFAULT_GATEWAY_AUTH_STORE,
                DEFAULT_CODEX_LEASE_JOURNAL,
            ),
            (
                "mac_ops_gitlab_env",
                DEFAULT_MAC_OPS_CREDENTIAL,
                DEFAULT_MAC_OPS_LEASE_JOURNAL,
            ),
        )
        for kind, target, journal in targets:
            binding = _CREDENTIAL_BINDING_BY_KIND[kind]
            try:
                retirements[binding] = retire_secret_lease(
                    kind=kind,
                    target=target,
                    journal=journal,
                    stop_proof=credential_consumer_stop_proof,
                    plan=self.plan,
                )
            except BaseException as exc:
                errors.append(exc)
        credential_absence = {
            _CREDENTIAL_BINDING_BY_KIND[kind]: {
                "path": str(target),
                "absent": not os.path.lexists(target),
            }
            for kind, target, _journal in targets
        }
        retirement_receipt_sha256s = {
            binding: receipt["receipt_sha256"]
            for binding, receipt in retirements.items()
        }
        bitrix_key_absence = {
            "private_path": str(DEFAULT_BITRIX_RECEIPT_PRIVATE_KEY_PATH),
            "private_absent": not os.path.lexists(
                DEFAULT_BITRIX_RECEIPT_PRIVATE_KEY_PATH
            ),
            "public_path": str(DEFAULT_BITRIX_TRUST_PATH),
            "public_absent": not os.path.lexists(DEFAULT_BITRIX_TRUST_PATH),
            "both_pair_members_absent": (
                not os.path.lexists(DEFAULT_BITRIX_RECEIPT_PRIVATE_KEY_PATH)
                and not os.path.lexists(DEFAULT_BITRIX_TRUST_PATH)
            ),
        }
        credentials_absent = all(
            evidence["absent"] for evidence in credential_absence.values()
        )
        if (
            set(retirements) != set(CAPABILITY_CREDENTIAL_BINDINGS)
            or set(retirement_receipt_sha256s) != set(CAPABILITY_CREDENTIAL_BINDINGS)
        ):
            errors.append(
                RuntimeError("six exact credential retirement receipts are required")
            )
        if not credentials_absent:
            errors.append(RuntimeError("capability credential retirement is incomplete"))
        if bitrix_key_absence["both_pair_members_absent"] is not True:
            errors.append(RuntimeError("Bitrix receipt key retirement is incomplete"))

        cleanup_facts: Mapping[str, Any] | None = None
        cleanup_facts_publication: Mapping[str, Any] | None = None
        cleanup_receipt: Mapping[str, Any] | None = None
        if not errors:
            try:
                cleanup_facts = build_capability_cleanup_facts(
                    self.plan,
                    services=services,
                    credential_consumer_stop_proof=(
                        credential_consumer_stop_proof
                    ),
                    producer_foundation=producer_foundation or {},
                    retirements=retirements,
                    retirement_receipt_sha256s=(
                        retirement_receipt_sha256s
                    ),
                    credential_absence=credential_absence,
                    bitrix_receipt_key_retirement=(
                        bitrix_key_retirement or {}
                    ),
                    bitrix_receipt_key_absence=bitrix_key_absence,
                    execution_cleanup=execution_cleanup or {},
                )
                if cleanup_run_id is not None:
                    observer_identity = _cleanup_observer_identity()
                    cleanup_facts_publication = publish_capability_cleanup_facts(
                        cleanup_facts,
                        run_id=cleanup_run_id,
                        observer_gid=observer_identity["gid"],
                    )
                    if cleanup_producer is None:
                        raise RuntimeError("live cleanup lacks the observer producer")
                    cleanup_receipt = cleanup_producer(
                        cleanup_facts_publication
                    )
                    if not isinstance(cleanup_receipt, Mapping):
                        raise RuntimeError("observer cleanup receipt is invalid")
                elif cleanup_producer is not None:
                    raise RuntimeError("cleanup producer lacks a fixed run id")
            except BaseException as exc:
                errors.append(exc)

        # The observer is the last unit to stop, even when signing failed.  It
        # must never remain live after credential/key retirement.
        observer_stopped, observer_errors = _attempt_capability_stop_order(
            self._stop_command,
            stop_order=(CAPABILITY_OBSERVER_UNIT,),
        )
        errors.extend(observer_errors)
        final_services: Mapping[str, Mapping[str, Any]] = {}
        service_stop_proof: Mapping[str, Any] | None = None
        observer_stop_receipt: Mapping[str, Any] | None = None
        try:
            if observer_stopped != [CAPABILITY_OBSERVER_UNIT]:
                raise RuntimeError("cleanup observer did not stop last")
            final_services = _capability_services(runner=self.runner)
            service_stop_proof = build_capability_stop_proof(
                self.plan,
                final_services,
                stop_order=CAPABILITY_STOP_ORDER,
            )
            observer_stop_receipt = build_capability_observer_stop_receipt(
                self.plan,
                final_services[CAPABILITY_OBSERVER_UNIT],
            )
        except BaseException as exc:
            errors.append(exc)

        producer_fleet_retirement: Mapping[str, Any] | None = None
        if cleanup_run_id is not None:
            try:
                if producer_activation_retirer is None:
                    raise RuntimeError("live cleanup lacks activation retirement")
                producer_fleet_retirement = producer_activation_retirer()
                if (
                    not isinstance(producer_fleet_retirement, Mapping)
                    or producer_fleet_retirement.get("run_id") != cleanup_run_id
                    or producer_fleet_retirement.get("retired") is not True
                    or producer_fleet_retirement.get("absence_verified") is not True
                ):
                    raise RuntimeError("producer activation retirement is invalid")
            except BaseException as exc:
                errors.append(exc)

        try:
            from gateway.canonical_capability_canary_producers import (
                DEFAULT_READINESS_PATH,
            )

            producer_activation_absent = not os.path.lexists(
                DEFAULT_READINESS_PATH
            )
            if not producer_activation_absent:
                raise RuntimeError("producer activation remains after retirement")
        except BaseException as exc:
            producer_activation_absent = False
            errors.append(exc)

        expiry_watchdog_retirement: Mapping[str, Any] | None = None
        if (
            credentials_absent
            and bitrix_key_absence["both_pair_members_absent"] is True
            and service_stop_proof is not None
            and producer_activation_absent
        ):
            try:
                expiry_watchdog_retirement = (
                    disarm_all_capability_expiry_watchdogs(runner=self.runner)
                )
                if (
                    expiry_watchdog_retirement.get("all_timers_disabled")
                    is not True
                    or expiry_watchdog_retirement.get("all_unit_files_absent")
                    is not True
                ):
                    raise RuntimeError(
                        "capability expiry watchdog retirement failed"
                    )
            except BaseException as exc:
                errors.append(exc)

        cleanup_finalization: Mapping[str, Any] | None = None
        if (
            cleanup_receipt is not None
            and observer_stop_receipt is not None
            and service_stop_proof is not None
            and producer_fleet_retirement is not None
            and full_stopped is not None
            and not errors
        ):
            try:
                cleanup_finalization = build_capability_cleanup_finalization(
                    self.plan,
                    cleanup_receipt=cleanup_receipt,
                    observer_stop_receipt=observer_stop_receipt,
                    service_stop_proof=service_stop_proof,
                    producer_fleet_retirement=producer_fleet_retirement,
                    producer_activation_absent=producer_activation_absent,
                    credentials_absent=credentials_absent,
                    bitrix_receipt_key_pair_absent=(
                        bitrix_key_absence["both_pair_members_absent"]
                    ),
                    full_canary_stopped_preflight_sha256=full_stopped[
                        "report_sha256"
                    ],
                )
            except BaseException as exc:
                errors.append(exc)

        result = {
            "stop_order": [*stopped, *observer_stopped],
            "credential_consumer_stop_proof": copy.deepcopy(
                dict(credential_consumer_stop_proof or {})
            ),
            "service_stop_proof": copy.deepcopy(
                dict(service_stop_proof or {})
            ),
            "cleanup_facts": copy.deepcopy(dict(cleanup_facts or {})),
            "cleanup_facts_publication": copy.deepcopy(
                dict(cleanup_facts_publication or {})
            ),
            "cleanup_receipt": copy.deepcopy(dict(cleanup_receipt or {})),
            "observer_stop_receipt": copy.deepcopy(
                dict(observer_stop_receipt or {})
            ),
            "producer_fleet_retirement": copy.deepcopy(
                dict(producer_fleet_retirement or {})
            ),
            "cleanup_finalization": copy.deepcopy(
                dict(cleanup_finalization or {})
            ),
            "bitrix_receipt_key_pair_retirement": copy.deepcopy(
                dict(bitrix_key_retirement or {})
            ),
            "bitrix_receipt_key_pair_absence": bitrix_key_absence,
            "retirements": retirements,
            "retirement_receipt_sha256s": retirement_receipt_sha256s,
            "credential_absence": credential_absence,
            "approval_retirement": copy.deepcopy(
                dict(approval_retirement or {})
            ),
            "connector_cleanup": copy.deepcopy(dict(connector_cleanup or {})),
            "execution_cleanup": copy.deepcopy(dict(execution_cleanup or {})),
            "full_gateway_unit_restore": copy.deepcopy(dict(restored or {})),
            "removed_overlay_artifacts": copy.deepcopy(dict(removed_artifacts)),
            "full_canary_stopped_preflight_sha256": (
                full_stopped.get("report_sha256") if full_stopped else None
            ),
            "services_stopped": bool(final_services)
            and all(_service_stopped(state) for state in final_services.values()),
            "credentials_absent": credentials_absent,
            "producer_activation_absent": producer_activation_absent,
            "expiry_watchdog_retirement": copy.deepcopy(
                dict(expiry_watchdog_retirement or {})
            ),
            "units_enabled": False,
            "completed_at_unix": int(time.time()),
        }
        if errors:
            raise BaseExceptionGroup("capability cleanup failed closed", errors)
        if result["services_stopped"] is not True:
            raise RuntimeError("capability services did not stop exactly")
        if (
            cleanup_run_id is not None
            and not result["cleanup_finalization"]
        ):
            raise RuntimeError("live cleanup finalization is missing")
        return result

    def _cleanup(
        self,
        *,
        cleanup_producer: Callable[[Mapping[str, Any]], Mapping[str, Any]]
        | None = None,
        cleanup_run_id: str | None = None,
        producer_activation_retirer: Callable[[], Mapping[str, Any]]
        | None = None,
    ) -> Mapping[str, Any]:
        _require_root_linux()
        self._require_host()
        with _lifecycle_lock():
            return self._cleanup_locked(
                cleanup_producer=cleanup_producer,
                cleanup_run_id=cleanup_run_id,
                producer_activation_retirer=producer_activation_retirer,
            )

    def start(
        self,
        approval: CapabilityCanaryOwnerApproval,
        full_approval: FullCanaryOwnerApproval,
    ) -> Mapping[str, Any]:
        _require_root_linux()
        if not isinstance(approval, CapabilityCanaryOwnerApproval) or not isinstance(
            full_approval, FullCanaryOwnerApproval
        ):
            raise PermissionError("both exact owner approvals are required")

        def require_approvals() -> None:
            now = int(time.time())
            approval.require(
                plan_sha256=self.plan.sha256,
                full_canary_plan_sha256=self.full_plan.sha256,
                now_unix=now,
            )
            full_approval.require(
                plan_sha256=self.full_plan.sha256,
                now_unix=now,
            )

        require_approvals()
        self._require_host()
        _validate_release_manifest(self.full_plan)
        writer_config_raw = _validate_artifact_source(
            self.full_plan.artifacts["writer_config"], label="writer_config"
        )
        _validate_writer_config(writer_config_raw, self.full_plan.identities)
        preflight: Mapping[str, Any] | None = None
        full_preflight: Mapping[str, Any] | None = None
        installed: Mapping[str, Any] = {}
        started: list[str] = []
        phase_b_current: Mapping[str, Any] | None = None
        installed_phase_b_anchor: Mapping[str, Any] | None = None
        connector_state: Mapping[str, Any] | None = None
        observer: Mapping[str, Any] | None = None
        browser_identity_foundation: Mapping[str, Any] | None = None
        browser_principal_smoke: Mapping[str, Any] | None = None
        execution_identity_foundation: Mapping[str, Any] | None = None
        worker_mountpoint: Mapping[str, Any] | None = None
        execution_readiness: Mapping[str, Any] | None = None
        routeback_bot_identity: Mapping[str, Any] | None = None
        producer_foundation: Mapping[str, Any] | None = None
        try:
            with _lifecycle_lock():
                full_preflight = collect_full_canary_preflight(
                    self.full_plan,
                    phase="stopped",
                    runner=self.runner,
                    metadata_reader=self.metadata_reader,
                    local_identity_reader=self.local_identity_reader,
                )
                preflight = collect_capability_preflight(
                    self.plan,
                    self.full_plan,
                    phase="stopped",
                    runner=self.runner,
                    metadata_reader=self.metadata_reader,
                    local_identity_reader=self.local_identity_reader,
                )
                if approval.value["stopped_preflight_state_sha256"] != preflight.get(
                    "state_sha256"
                ):
                    raise PermissionError(
                        "owner approval stopped-preflight state changed before start"
                    )
                require_approvals()
                self._require_host()
                browser_identity_foundation = ensure_browser_identity_create_only(
                    self.plan,
                    self.full_plan,
                    runner=self.runner,
                )
                browser_principal_smoke = browser_principal_version_smoke(
                    self.plan,
                    runner=self.runner,
                )
                browser_userns_preflight()
                execution_identity_foundation = ensure_execution_identities_create_only(
                    self.plan,
                    self.full_plan,
                    runner=self.runner,
                )
                worker_mountpoint = _prepare_worker_mountpoint()
                worker_systemd252_preflight(self.plan, runner=self.runner)
                _prepare_gateway_directories(self.plan)
                installed = {
                    "full_canary_foundation": copy.deepcopy(
                        dict(_install_plan_artifacts(self.full_plan))
                    ),
                    "capability_overlay": copy.deepcopy(
                        dict(_install_capability_artifacts(self.plan, self.full_plan))
                    ),
                }
                connector_state = _prepare_connector_state(self.plan)
                producer_foundation = _producer_foundation_preflight(
                    self.plan,
                    self.full_plan,
                )
                _run_checked(
                    Command(
                        (
                            SYSTEMD_ANALYZE,
                            "verify",
                            str(DEFAULT_EDGE_UNIT_PATH),
                            str(DEFAULT_WRITER_UNIT_PATH),
                            str(DEFAULT_GATEWAY_UNIT_PATH),
                            str(DEFAULT_PHASE_B_READINESS_UNIT_PATH),
                            str(DEFAULT_CONNECTOR_UNIT_PATH),
                            str(DEFAULT_MAC_OPS_UNIT_PATH),
                            str(DEFAULT_BROWSER_UNIT_PATH),
                            str(DEFAULT_WORKER_SOCKET_UNIT_PATH),
                            str(DEFAULT_WORKER_SERVICE_UNIT_PATH),
                            str(DEFAULT_BITRIX_UNIT_PATH),
                            *(
                                str(CAPABILITY_PRODUCER_UNIT_PATHS[role])
                                for role in CAPABILITY_PRODUCER_ROLES
                            ),
                        )
                    ),
                    runner=self.runner,
                    label="verify capability-canary units",
                )
                _run_checked(
                    Command((SYSTEMCTL, "daemon-reload")),
                    runner=self.runner,
                    label="reload capability-canary units",
                )
                _run_checked(
                    Command((SYSTEMD_TMPFILES, "--create", str(DEFAULT_TMPFILES_PATH))),
                    runner=self.runner,
                    label="create full-canary runtime directories",
                )
                from gateway.canonical_writer_phase_b_runtime import (
                    install_fixed_phase_b_full_canary_anchor,
                    validate_fixed_phase_b_readiness_descendant,
                )

                _run_checked(
                    phase_b_readiness_start_command(),
                    runner=self.runner,
                    label=f"start {PHASE_B_READINESS_UNIT_NAME}",
                )
                started.append(PHASE_B_READINESS_UNIT_NAME)
                _validate_inert_gateway_paths(
                    gateway_uid=self.full_plan.identities.gateway_uid,
                    gateway_gid=self.full_plan.identities.gateway_gid,
                )
                phase_b_current = validate_fixed_phase_b_readiness_descendant(
                    self.full_plan.phase_b_readiness_anchor
                )
                installed_phase_b_anchor = install_fixed_phase_b_full_canary_anchor(
                    self.full_plan.phase_b_readiness_anchor
                )

                require_approvals()
                routeback_bot_identity = self._start_routeback_edge()
                started.append(EDGE_UNIT_NAME)
                edge_state = collect_service_state(EDGE_UNIT_NAME, runner=self.runner)
                edge_readiness = _validate_edge_collector_gate(
                    self.full_plan, edge_state
                )
                edge_identity_sha256 = readiness_receipt_sha256(edge_readiness)
                collector = _await_collector_readiness(
                    self.full_plan,
                    edge_pid=int(edge_state["MainPID"]),
                    edge_service_identity_sha256=edge_identity_sha256,
                )
                _require_routeback_credential_binding(
                    self.plan,
                    self.full_plan,
                    routeback_bot_identity,
                )
                observer = materialize_observer_config(
                    self.full_plan,
                    collector=collector,
                    edge_pid=int(edge_state["MainPID"]),
                    edge_service_identity_sha256=edge_identity_sha256,
                )

                for unit in (
                    DEFAULT_DISCORD_CONNECTOR_UNIT,
                    MAC_OPS_UNIT_NAME,
                    DEFAULT_WORKER_SOCKET_UNIT_NAME,
                    DEFAULT_WORKER_SERVICE_UNIT_NAME,
                    DEFAULT_BROWSER_UNIT_NAME,
                ):
                    require_approvals()
                    _run_checked(
                        Command((SYSTEMCTL, "start", unit), timeout_seconds=180),
                        runner=self.runner,
                        label=f"start {unit}",
                    )
                    started.append(unit)
                    if unit == DEFAULT_DISCORD_CONNECTOR_UNIT:
                        _await_runtime_ready(
                            lambda: _connector_runtime_preflight(
                                self.plan,
                                collect_capability_service_state(
                                    unit, runner=self.runner
                                ),
                            ),
                            label="public Discord connector",
                            timeout_seconds=60.0,
                        )
                    elif unit == MAC_OPS_UNIT_NAME:
                        _await_runtime_ready(
                            lambda: _mac_ops_runtime_preflight(
                                self.plan,
                                collect_capability_service_state(
                                    unit, runner=self.runner
                                ),
                            ),
                            label="Mac operations edge",
                        )
                    elif unit == DEFAULT_WORKER_SOCKET_UNIT_NAME:
                        def worker_socket_ready() -> Mapping[str, Any]:
                            state = collect_capability_service_state(
                                unit, runner=self.runner
                            )
                            if not _socket_live(
                                state, path=DEFAULT_WORKER_SOCKET_UNIT_PATH
                            ):
                                raise RuntimeError(
                                    "isolated worker socket is not ready"
                                )
                            return {"ready": True}

                        _await_runtime_ready(
                            worker_socket_ready,
                            label="isolated worker socket",
                        )
                    elif unit == DEFAULT_WORKER_SERVICE_UNIT_NAME:
                        _await_runtime_ready(
                            lambda: worker_tmpfs_runtime_preflight(
                                self.plan,
                                collect_capability_service_state(
                                    unit, runner=self.runner
                                ),
                            ),
                            label="isolated worker tmpfs",
                        )
                    else:
                        _await_runtime_ready(
                            lambda: browser_service_runtime_preflight(
                                self.plan,
                                collect_capability_service_state(
                                    unit, runner=self.runner
                                ),
                            ),
                            label="AF_UNIX browser controller",
                        )

                execution_readiness = _execution_readiness_as_gateway(
                    self.plan, runner=self.runner
                )

                writer_command, gateway_command = post_collector_start_commands()
                require_approvals()
                _run_checked(
                    writer_command,
                    runner=self.runner,
                    label=f"start {WRITER_UNIT_NAME}",
                )
                started.append(WRITER_UNIT_NAME)
                writer_readiness = _readiness_receipt(
                    DEFAULT_WRITER_RUNTIME_ATTESTATION_PATH,
                    uid=self.full_plan.identities.writer_uid,
                    gid=self.full_plan.identities.writer_gid,
                )
                require_approvals()
                _run_checked(
                    Command(
                        (SYSTEMCTL, "start", BITRIX_OPERATIONAL_EDGE_UNIT),
                        timeout_seconds=180,
                    ),
                    runner=self.runner,
                    label=f"start {BITRIX_OPERATIONAL_EDGE_UNIT}",
                )
                started.append(BITRIX_OPERATIONAL_EDGE_UNIT)
                _await_runtime_ready(
                    lambda: _bitrix_runtime_preflight(
                        self.plan,
                        self.full_plan,
                        collect_capability_service_state(
                            BITRIX_OPERATIONAL_EDGE_UNIT,
                            runner=self.runner,
                        ),
                    ),
                    label="Bitrix operational edge",
                )
                for role in CAPABILITY_PRODUCER_ROLES:
                    unit = CAPABILITY_PRODUCER_SERVICE_UNITS[role]
                    require_approvals()
                    _run_checked(
                        Command((SYSTEMCTL, "start", unit), timeout_seconds=180),
                        runner=self.runner,
                        label=f"start {unit}",
                    )
                    started.append(unit)

                    def producer_ready(
                        producer_unit: str = unit,
                        producer_role: str = role,
                    ) -> Mapping[str, Any]:
                        state = collect_capability_service_state(
                            producer_unit,
                            runner=self.runner,
                        )
                        if not _service_live(
                            state,
                            path=CAPABILITY_PRODUCER_UNIT_PATHS[producer_role],
                            service_type="simple",
                        ):
                            raise RuntimeError(
                                f"capability producer {producer_role} is not live"
                            )
                        return {
                            "role": producer_role,
                            "unit": producer_unit,
                            "main_pid": state["MainPID"],
                            "ready": True,
                        }

                    _await_runtime_ready(
                        producer_ready,
                        label=f"capability producer {role}",
                    )
                require_approvals()
                _run_checked(
                    gateway_command,
                    runner=self.runner,
                    label=f"start {GATEWAY_UNIT_NAME}",
                )
                started.append(GATEWAY_UNIT_NAME)
                if tuple(started) != CAPABILITY_START_ORDER:
                    raise RuntimeError("capability start order drifted")
                live = collect_capability_preflight(
                    self.plan,
                    self.full_plan,
                    phase="live",
                    runner=self.runner,
                    metadata_reader=self.metadata_reader,
                    local_identity_reader=self.local_identity_reader,
                )
                return _write_lifecycle_receipt(
                    self.plan,
                    stage="started",
                    value={
                        "operation": "start",
                        "owner_approval_sha256": approval.sha256,
                        "full_canary_owner_approval_sha256": full_approval.sha256,
                        "full_canary_stopped_preflight_sha256": full_preflight[
                            "report_sha256"
                        ],
                        "stopped_preflight_sha256": preflight["report_sha256"],
                        "live_preflight_sha256": live["report_sha256"],
                        "installed_artifacts": copy.deepcopy(dict(installed)),
                        "connector_state": copy.deepcopy(dict(connector_state or {})),
                        "phase_b_current_readiness": copy.deepcopy(
                            dict(phase_b_current or {})
                        ),
                        "phase_b_full_canary_anchor": copy.deepcopy(
                            dict(installed_phase_b_anchor or {})
                        ),
                        "writer_runtime_readiness": copy.deepcopy(
                            dict(writer_readiness)
                        ),
                        "observer_config": copy.deepcopy(dict(observer or {})),
                        "browser_identity_foundation": copy.deepcopy(
                            dict(browser_identity_foundation or {})
                        ),
                        "browser_principal_smoke": copy.deepcopy(
                            dict(browser_principal_smoke or {})
                        ),
                        "execution_identity_foundation": copy.deepcopy(
                            dict(execution_identity_foundation or {})
                        ),
                        "worker_mountpoint": copy.deepcopy(
                            dict(worker_mountpoint or {})
                        ),
                        "execution_readiness": copy.deepcopy(
                            dict(execution_readiness or {})
                        ),
                        "routeback_bot_identity": copy.deepcopy(
                            dict(routeback_bot_identity or {})
                        ),
                        "producer_foundation": copy.deepcopy(
                            dict(producer_foundation or {})
                        ),
                        "credential_bindings": _credential_bindings_mapping(),
                        "start_order": started,
                        "units_enabled": False,
                        "runtime_max_seconds": 900,
                        "started_at_unix": int(time.time()),
                    },
                )
        except BaseException as error:
            cleanup_error: BaseException | None = None
            try:
                self._cleanup()
            except BaseException as exc:
                cleanup_error = exc
            failure: Mapping[str, Any] | None = None
            try:
                failure = _write_lifecycle_receipt(
                    self.plan,
                    stage="failure",
                    value={
                        "operation": "start",
                        "owner_approval_sha256": approval.sha256,
                        "full_canary_owner_approval_sha256": full_approval.sha256,
                        "full_canary_stopped_preflight_sha256": (
                            full_preflight.get("report_sha256")
                            if full_preflight
                            else None
                        ),
                        "stopped_preflight_sha256": (
                            preflight.get("report_sha256") if preflight else None
                        ),
                        "started_before_failure": started,
                        "installed_artifacts": copy.deepcopy(dict(installed)),
                        "browser_identity_foundation": copy.deepcopy(
                            dict(browser_identity_foundation or {})
                        ),
                        "browser_principal_smoke": copy.deepcopy(
                            dict(browser_principal_smoke or {})
                        ),
                        "execution_identity_foundation": copy.deepcopy(
                            dict(execution_identity_foundation or {})
                        ),
                        "worker_mountpoint": copy.deepcopy(
                            dict(worker_mountpoint or {})
                        ),
                        "execution_readiness": copy.deepcopy(
                            dict(execution_readiness or {})
                        ),
                        "routeback_bot_identity": copy.deepcopy(
                            dict(routeback_bot_identity or {})
                        ),
                        "error_type": type(error).__name__,
                        "error_sha256": _sha256_bytes(
                            f"{type(error).__name__}:{error}".encode(
                                "utf-8", errors="replace"
                            )
                        ),
                        "cleanup_complete": cleanup_error is None,
                        "failed_at_unix": int(time.time()),
                    },
                )
            except BaseException as receipt_error:
                cleanup_error = BaseExceptionGroup(
                    "capability failure receipt/cleanup failed",
                    [
                        *([cleanup_error] if cleanup_error is not None else []),
                        receipt_error,
                    ],
                )
            if cleanup_error is not None:
                raise BaseExceptionGroup(
                    "capability start and cleanup failed", [error, cleanup_error]
                ) from None
            raise RuntimeError(
                "capability start failed closed"
                + (f"; receipt={failure['receipt_path']}" if failure else "")
            ) from error

    def stop(
        self,
        *,
        cleanup_producer: Callable[[Mapping[str, Any]], Mapping[str, Any]]
        | None = None,
        cleanup_run_id: str | None = None,
        producer_activation_retirer: Callable[[], Mapping[str, Any]]
        | None = None,
    ) -> Mapping[str, Any]:
        try:
            cleanup = self._cleanup(
                cleanup_producer=cleanup_producer,
                cleanup_run_id=cleanup_run_id,
                producer_activation_retirer=producer_activation_retirer,
            )
            return _write_lifecycle_receipt(
                self.plan,
                stage="stopped",
                value={"operation": "stop", **dict(cleanup)},
            )
        except BaseException as error:
            try:
                _write_lifecycle_receipt(
                    self.plan,
                    stage="failure",
                    value={
                        "operation": "stop",
                        "error_type": type(error).__name__,
                        "error_sha256": _sha256_bytes(
                            f"{type(error).__name__}:{error}".encode(
                                "utf-8", errors="replace"
                            )
                        ),
                        "failed_at_unix": int(time.time()),
                    },
                )
            except BaseException:
                pass
            raise


_BITRIX_FOUNDATION_AUTHORITY_FIELDS = frozenset(
    {
        "schema",
        "scope",
        "revision",
        "full_canary_plan_sha256",
        "release_artifact_sha256",
        "owner_subject_sha256",
        "authority_kind",
        "cryptographic_owner_proof",
        "issued_at_unix",
        "expires_at_unix",
        "identities",
        "asset_manifest_sha256",
        "secret_material_recorded",
        "secret_digest_recorded",
        "semantic_content_recorded",
        "authority_sha256",
    }
)
_BITRIX_FOUNDATION_IDENTITY_FIELDS = frozenset(
    {
        "service_uid",
        "service_gid",
        "socket_client_gid",
        "business_edge_uid",
    }
)
_MAX_BITRIX_FOUNDATION_AUTHORITY_BYTES = 128 * 1024
_BITRIX_FOUNDATION_MAX_SECONDS = 1_200


def validate_bitrix_foundation_authority(
    value: Any,
    *,
    now_unix: int | None = None,
) -> Mapping[str, Any]:
    """Validate the exact pre-plan owner authority; a plan digest is forbidden."""

    raw = _strict_mapping(
        value,
        _BITRIX_FOUNDATION_AUTHORITY_FIELDS,
        "Bitrix foundation authority",
    )
    identities = _strict_mapping(
        raw["identities"],
        _BITRIX_FOUNDATION_IDENTITY_FIELDS,
        "Bitrix foundation identities",
    )
    now = int(time.time()) if now_unix is None else now_unix
    if (
        raw["schema"] != CAPABILITY_BITRIX_FOUNDATION_AUTHORITY_SCHEMA
        or raw["scope"] != CAPABILITY_BITRIX_FOUNDATION_SCOPE
        or raw["authority_kind"]
        != "trusted_gcloud_owner_explicit_foundation_digest"
        or raw["cryptographic_owner_proof"] is not False
        or raw["secret_material_recorded"] is not False
        or raw["secret_digest_recorded"] is not False
        or raw["semantic_content_recorded"] is not False
        or not isinstance(raw["revision"], str)
        or _REVISION_RE.fullmatch(raw["revision"]) is None
        or type(raw["issued_at_unix"]) is not int
        or type(raw["expires_at_unix"]) is not int
        or type(now) is not int
        or not raw["issued_at_unix"] <= now <= raw["expires_at_unix"]
        or not 60
        <= raw["expires_at_unix"] - raw["issued_at_unix"]
        <= _BITRIX_FOUNDATION_MAX_SECONDS
        or any(
            type(identities[field]) is not int
            or not 0 < identities[field] < (1 << 31)
            for field in _BITRIX_FOUNDATION_IDENTITY_FIELDS
        )
        or len(set(identities.values())) != len(identities)
    ):
        raise ValueError("Bitrix foundation authority is invalid")
    for field in (
        "full_canary_plan_sha256",
        "release_artifact_sha256",
        "owner_subject_sha256",
        "asset_manifest_sha256",
    ):
        _digest(raw[field], f"Bitrix foundation authority {field}")
    unsigned = {
        key: copy.deepcopy(item)
        for key, item in raw.items()
        if key != "authority_sha256"
    }
    if raw["authority_sha256"] != _sha256_json(unsigned):
        raise ValueError("Bitrix foundation authority self-digest drifted")
    if "plan_sha256" in raw or "capability_plan_sha256" in raw:
        raise ValueError("Bitrix foundation authority cannot self-reference a plan")
    return copy.deepcopy(dict(raw))


def read_bitrix_foundation_authority(stream: BinaryIO) -> Mapping[str, Any]:
    raw = stream.read(_MAX_BITRIX_FOUNDATION_AUTHORITY_BYTES + 1)
    if (
        not raw
        or len(raw) > _MAX_BITRIX_FOUNDATION_AUTHORITY_BYTES
        or stream.read(1)
    ):
        raise ValueError("Bitrix foundation authority input is invalid")
    value = _decode_json(raw, label="Bitrix foundation authority")
    if raw != _canonical_bytes(value):
        raise ValueError("Bitrix foundation authority is not canonical")
    return validate_bitrix_foundation_authority(value)


def _expiry_watchdog_paths(
    watchdog_id: str,
    *,
    state_root: Path = DEFAULT_EXPIRY_WATCHDOG_ROOT,
    systemd_root: Path = Path("/etc/systemd/system"),
) -> Mapping[str, Path | str]:
    if _LEASE_ID_RE.fullmatch(watchdog_id) is None:
        raise ValueError("capability expiry watchdog ID is invalid")
    base = f"{EXPIRY_WATCHDOG_UNIT_PREFIX}{watchdog_id}"
    timer_name = f"{base}.timer"
    return {
        "state_root": state_root / watchdog_id,
        "authority": state_root / watchdog_id / "authority.json",
        "completion": state_root / watchdog_id / "completion.json",
        "service_name": f"{base}.service",
        "timer_name": timer_name,
        "service_path": systemd_root / f"{base}.service",
        "timer_path": systemd_root / timer_name,
        "timer_wants_path": systemd_root / "timers.target.wants" / timer_name,
    }


def render_expiry_watchdog_units(
    *,
    watchdog_id: str,
    interpreter: Path,
    cleanup_at_unix: int,
) -> tuple[bytes, bytes]:
    paths = _expiry_watchdog_paths(watchdog_id)
    if (
        not interpreter.is_absolute()
        or ".." in interpreter.parts
        or type(cleanup_at_unix) is not int
        or cleanup_at_unix < 0
    ):
        raise ValueError("capability expiry watchdog input is invalid")
    service_name = str(paths["service_name"])
    service = (
        "# Persistent owner-disconnect/reboot cleanup for one bounded canary lease.\n"
        "[Unit]\n"
        "Description=Muncho capability canary expiry cleanup\n"
        "After=local-fs.target\n"
        "StartLimitIntervalSec=0\n"
        "\n"
        "[Service]\n"
        "Type=oneshot\n"
        f"ExecStart={interpreter} -B -I -m gateway.canonical_capability_canary_runtime expiry-cleanup --watchdog-id {watchdog_id}\n"
        "Restart=on-failure\n"
        "RestartSec=30s\n"
        "User=root\n"
        "Group=root\n"
        "UMask=0077\n"
        "NoNewPrivileges=yes\n"
        "PrivateTmp=yes\n"
        "ProtectHome=yes\n"
        "ProtectSystem=full\n"
        "ReadWritePaths=/etc/muncho /etc/systemd/system -/opt/adventico-ai-platform/hermes-home/secrets -/var/lib/muncho-capability-canary /var/lib/muncho-capability-canary-control\n"
        "StandardInput=null\n"
        "StandardOutput=journal\n"
        "StandardError=journal\n"
    ).encode("ascii")
    timer = (
        "# Persistent absolute expiry; missed runs fire after reboot.\n"
        "[Unit]\n"
        "Description=Muncho capability canary persistent expiry timer\n"
        "\n"
        "[Timer]\n"
        f"OnCalendar=@{cleanup_at_unix}\n"
        "Persistent=true\n"
        "AccuracySec=1s\n"
        f"Unit={service_name}\n"
        "\n"
        "[Install]\n"
        "WantedBy=timers.target\n"
    ).encode("ascii")
    if (
        service.count(b"ExecStart=") != 1
        or b"--watchdog-id " + watchdog_id.encode("ascii") not in service
        or timer.count(b"Persistent=true\n") != 1
        or timer.count(b"OnCalendar=@") != 1
    ):
        raise RuntimeError("capability expiry watchdog rendering drifted")
    return service, timer


def _install_exact_timer_wants_link(
    path: Path,
    *,
    timer_name: str,
    uid: int,
    gid: int,
) -> None:
    """Install the fixed boot-persistence link without an enable mutation."""

    if (
        not path.is_absolute()
        or ".." in path.parts
        or path.name != timer_name
        or not timer_name.endswith(".timer")
    ):
        raise ValueError("capability expiry watchdog wants link is invalid")
    _prepare_secret_parent(path.parent, uid=uid, gid=gid, mode=0o755)
    target = f"../{timer_name}"
    try:
        os.symlink(target, path)
        _fsync_directory(path.parent)
    except FileExistsError:
        item = os.lstat(path)
        if not stat.S_ISLNK(item.st_mode) or os.readlink(path) != target:
            raise RuntimeError("capability expiry watchdog wants link drifted")


def arm_capability_expiry_watchdog(
    *,
    kind: str,
    revision: str,
    full_canary_plan_sha256: str,
    release_artifact_sha256: str,
    interpreter: Path,
    expires_at_unix: int,
    authority_sha256: str,
    plan_sha256: str | None,
    credential_binding: str | None,
    runner: Callable[[Command], subprocess.CompletedProcess[bytes]] = _runner,
    state_root: Path = DEFAULT_EXPIRY_WATCHDOG_ROOT,
    systemd_root: Path = Path("/etc/systemd/system"),
    require_root: bool = True,
    now_unix: int | None = None,
) -> Mapping[str, Any]:
    """Install and enable a persistent absolute timer before secret commit."""

    if kind not in {"bitrix_foundation", "credential_lease"}:
        raise ValueError("capability expiry watchdog kind is invalid")
    now = int(time.time()) if now_unix is None else now_unix
    if (
        _REVISION_RE.fullmatch(revision or "") is None
        or type(expires_at_unix) is not int
        or type(now) is not int
        or expires_at_unix < now
    ):
        raise ValueError("capability expiry watchdog window is invalid")
    for value, label in (
        (full_canary_plan_sha256, "full-canary plan"),
        (release_artifact_sha256, "release artifact"),
        (authority_sha256, "watchdog authority source"),
    ):
        _digest(value, label)
    if kind == "credential_lease":
        _digest(plan_sha256, "capability watchdog plan")
        if credential_binding not in CAPABILITY_CREDENTIAL_BINDINGS:
            raise ValueError("capability watchdog credential binding is invalid")
    elif plan_sha256 is not None or credential_binding is not None:
        raise ValueError("foundation watchdog cannot bind a capability plan")
    watchdog_id = _sha256_json(
        {
            "kind": kind,
            "authority_sha256": authority_sha256,
            "expires_at_unix": expires_at_unix,
            "credential_binding": credential_binding,
        }
    )[:32]
    paths = _expiry_watchdog_paths(
        watchdog_id,
        state_root=state_root,
        systemd_root=systemd_root,
    )
    if require_root:
        _require_root_linux()
        if (
            state_root != DEFAULT_EXPIRY_WATCHDOG_ROOT
            or systemd_root != Path("/etc/systemd/system")
        ):
            raise ValueError("capability expiry watchdog production paths are fixed")
    else:
        _prepare_bitrix_foundation_directory(state_root, require_root=False)
        _prepare_bitrix_foundation_directory(systemd_root, require_root=False)
    cleanup_at = expires_at_unix
    authority = _append_lease_artifact(
        paths["authority"],
        schema=CAPABILITY_EXPIRY_WATCHDOG_AUTHORITY_SCHEMA,
        value={
            "operation": "arm_persistent_expiry_watchdog",
            "watchdog_id": watchdog_id,
            "kind": kind,
            "revision": revision,
            "full_canary_plan_sha256": full_canary_plan_sha256,
            "release_artifact_sha256": release_artifact_sha256,
            "plan_sha256": plan_sha256,
            "credential_binding": credential_binding,
            "authority_source_sha256": authority_sha256,
            "expires_at_unix": expires_at_unix,
            "cleanup_at_unix": cleanup_at,
            "interpreter": str(interpreter),
            "service_name": paths["service_name"],
            "timer_name": paths["timer_name"],
            "service_path": str(paths["service_path"]),
            "timer_path": str(paths["timer_path"]),
            "timer_wants_path": str(paths["timer_wants_path"]),
            "timer_wants_target": f"../{paths['timer_name']}",
            "persistent_across_reboot": True,
            "earliest_expiry_not_extended": True,
        },
    )
    service, timer = render_expiry_watchdog_units(
        watchdog_id=watchdog_id,
        interpreter=interpreter,
        cleanup_at_unix=cleanup_at,
    )
    owner = 0 if require_root else os.geteuid()  # windows-footgun: ok — Linux production/canary boundary
    group = 0 if require_root else os.getegid()  # windows-footgun: ok — Linux production/canary boundary
    for path, payload in (
        (Path(paths["service_path"]), service),
        (Path(paths["timer_path"]), timer),
    ):
        _atomic_no_replace_file(
            path,
            payload,
            uid=owner,
            gid=group,
            mode=0o644,
            temporary_name=f".{path.name}.arming",
            maximum=64 * 1024,
        )
    _install_exact_timer_wants_link(
        Path(paths["timer_wants_path"]),
        timer_name=str(paths["timer_name"]),
        uid=owner,
        gid=group,
    )
    _run_checked(
        Command((SYSTEMCTL, "daemon-reload")),
        runner=runner,
        label="reload capability expiry watchdog",
    )
    _run_checked(
        Command((SYSTEMCTL, "start", str(paths["timer_name"]))),
        runner=runner,
        label="arm capability expiry watchdog",
    )
    return {
        "watchdog_id": watchdog_id,
        "authority_receipt_path": authority["receipt_path"],
        "authority_receipt_sha256": authority["receipt_sha256"],
        "timer_name": paths["timer_name"],
        "cleanup_at_unix": cleanup_at,
        "expires_at_unix": expires_at_unix,
        "persistent_across_reboot": True,
        "armed_before_secret_commit": True,
    }


def _prepare_bitrix_foundation_directory(path: Path, *, require_root: bool) -> None:
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("Bitrix foundation directory is invalid")
    if require_root:
        _ensure_root_directory(path)
        return
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chown(path, os.geteuid(), os.getegid())  # windows-footgun: ok — Linux production/canary boundary
    os.chmod(path, 0o700)


def _observe_bitrix_foundation_identity(
    *,
    service_uid: int,
    service_gid: int,
    socket_client_gid: int,
    allow_absence: bool,
) -> Mapping[str, Any]:
    user = _optional_passwd_by_name("muncho-edge-bitrix")
    uid_owner = _optional_passwd_by_uid(service_uid)
    group = _optional_group_by_name("muncho-edge-bitrix")
    gid_owner = _optional_group_by_gid(service_gid)
    client = _optional_group_by_name("muncho-edge-bitrix-c")
    client_gid_owner = _optional_group_by_gid(socket_client_gid)
    if user is None and uid_owner is not None:
        raise RuntimeError("Bitrix foundation UID is already owned")
    if group is None and gid_owner is not None:
        raise RuntimeError("Bitrix foundation service GID is already owned")
    if client is None and client_gid_owner is not None:
        raise RuntimeError("Bitrix foundation client GID is already owned")
    if group is not None and (
        group.gr_name != "muncho-edge-bitrix"
        or group.gr_gid != service_gid
        or gid_owner is None
        or gid_owner.gr_name != group.gr_name
        or list(group.gr_mem) != []
    ):
        raise RuntimeError("Bitrix foundation service group drifted")
    if client is not None and (
        client.gr_name != "muncho-edge-bitrix-c"
        or client.gr_gid != socket_client_gid
        or client_gid_owner is None
        or client_gid_owner.gr_name != client.gr_name
        or list(client.gr_mem) != []
    ):
        raise RuntimeError("Bitrix foundation client group drifted")
    if user is not None and (
        user.pw_name != "muncho-edge-bitrix"
        or user.pw_uid != service_uid
        or user.pw_gid != service_gid
        or user.pw_dir != "/nonexistent"
        or user.pw_shell != "/usr/sbin/nologin"
        or uid_owner is None
        or uid_owner.pw_name != user.pw_name
        or sorted(set(os.getgrouplist(user.pw_name, service_gid)))
        != [service_gid]
    ):
        raise RuntimeError("Bitrix foundation service user drifted")
    state = (
        "present_exact"
        if user is not None and group is not None and client is not None
        else "absent_create_only_slot"
    )
    if state != "present_exact" and not allow_absence:
        raise RuntimeError("Bitrix foundation identity is incomplete")
    return {
        "service_user": "muncho-edge-bitrix",
        "service_group": "muncho-edge-bitrix",
        "service_uid": service_uid,
        "service_gid": service_gid,
        "socket_client_group": "muncho-edge-bitrix-c",
        "socket_client_gid": socket_client_gid,
        "state": state,
    }


def _ensure_bitrix_foundation_identity(
    *,
    service_uid: int,
    service_gid: int,
    socket_client_gid: int,
    runner: Callable[[Command], subprocess.CompletedProcess[bytes]],
    observer: Callable[..., Mapping[str, Any]] = _observe_bitrix_foundation_identity,
) -> Mapping[str, Any]:
    before = observer(
        service_uid=service_uid,
        service_gid=service_gid,
        socket_client_gid=socket_client_gid,
        allow_absence=True,
    )
    if before["state"] != "present_exact":
        for name, gid in (
            ("muncho-edge-bitrix", service_gid),
            ("muncho-edge-bitrix-c", socket_client_gid),
        ):
            if _optional_group_by_name(name) is None:
                _run_checked(
                    Command((GROUPADD, "--system", "--gid", str(gid), "--", name)),
                    runner=runner,
                    label=f"create {name} group",
                )
        if _optional_passwd_by_name("muncho-edge-bitrix") is None:
            _run_checked(
                Command(
                    (
                        USERADD,
                        "--system",
                        "--uid",
                        str(service_uid),
                        "--gid",
                        "muncho-edge-bitrix",
                        "--home-dir",
                        "/nonexistent",
                        "--no-create-home",
                        "--shell",
                        "/usr/sbin/nologin",
                        "--",
                        "muncho-edge-bitrix",
                    )
                ),
                runner=runner,
                label="create Bitrix operational-edge user",
            )
    return observer(
        service_uid=service_uid,
        service_gid=service_gid,
        socket_client_gid=socket_client_gid,
        allow_absence=False,
    )


def _ed25519_public_pem(key: Ed25519PublicKey) -> bytes:
    return key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def _ed25519_private_pem(key: Ed25519PrivateKey) -> bytes:
    return key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


def _load_exact_ed25519_public(raw: bytes) -> Ed25519PublicKey:
    try:
        key = serialization.load_pem_public_key(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Bitrix foundation public key is invalid") from exc
    if not isinstance(key, Ed25519PublicKey) or _ed25519_public_pem(key) != raw:
        raise RuntimeError("Bitrix foundation public key is invalid")
    return key


def _load_exact_ed25519_private(raw: bytes) -> Ed25519PrivateKey:
    try:
        key = serialization.load_pem_private_key(raw, password=None)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Bitrix foundation private key is invalid") from exc
    if not isinstance(key, Ed25519PrivateKey) or _ed25519_private_pem(key) != raw:
        raise RuntimeError("Bitrix foundation private key is invalid")
    return key


def _stage_bitrix_receipt_key_pair(
    *,
    private_path: Path,
    public_path: Path,
    require_root: bool,
) -> Mapping[str, Any]:
    owner = 0 if require_root else os.geteuid()  # windows-footgun: ok — Linux production/canary boundary
    group = 0 if require_root else os.getegid()  # windows-footgun: ok — Linux production/canary boundary
    for parent in (private_path.parent, public_path.parent):
        _prepare_bitrix_foundation_directory(parent, require_root=require_root)
    private_exists = os.path.lexists(private_path)
    public_exists = os.path.lexists(public_path)
    if private_exists != public_exists:
        raise RuntimeError("Bitrix foundation key pair is partial")
    if not private_exists:
        private = Ed25519PrivateKey.generate()
        private_raw = _ed25519_private_pem(private)
        public_raw = _ed25519_public_pem(private.public_key())
        _atomic_no_replace_file(
            private_path,
            private_raw,
            uid=owner,
            gid=group,
            mode=0o400,
            temporary_name=f".{private_path.name}.creating",
            maximum=16 * 1024,
        )
        try:
            _atomic_no_replace_file(
                public_path,
                public_raw,
                uid=owner,
                gid=group,
                mode=0o444,
                temporary_name=f".{public_path.name}.creating",
                maximum=16 * 1024,
            )
        except BaseException:
            os.unlink(private_path)
            _fsync_directory(private_path.parent)
            raise
    private_raw, private_item = _read_exact_file(
        private_path,
        maximum=16 * 1024,
        uid=owner,
        gid=group,
        mode=0o400,
    )
    public_raw, public_item = _read_exact_file(
        public_path,
        maximum=16 * 1024,
        uid=owner,
        gid=group,
        mode=0o444,
    )
    private = _load_exact_ed25519_private(private_raw)
    public = _load_exact_ed25519_public(public_raw)
    if _ed25519_public_pem(private.public_key()) != public_raw:
        raise RuntimeError("Bitrix foundation key pair does not match")
    return {
        "private_path": str(private_path),
        "private_device": private_item.st_dev,
        "private_inode": private_item.st_ino,
        "private_uid": private_item.st_uid,
        "private_gid": private_item.st_gid,
        "private_mode": "0400",
        "public_path": str(public_path),
        "public_device": public_item.st_dev,
        "public_inode": public_item.st_ino,
        "public_uid": public_item.st_uid,
        "public_gid": public_item.st_gid,
        "public_mode": "0444",
        "public_key_id": ed25519_public_key_id(public),
        "public_sha256": _sha256_bytes(public_raw),
        "private_content_or_digest_recorded": False,
    }


def _render_bitrix_canary_artifacts(
    *,
    full_plan: FullCanaryPlan,
    identities: Mapping[str, Any],
    receipt_public_key_id: str,
    writer_public_key_id: str,
) -> tuple[bytes, bytes]:
    read_peers = tuple(
        sorted(
            {
                identities["business_edge_uid"],
                full_plan.identities.writer_uid,
            }
        )
    )
    if read_peers != tuple(
        sorted((identities["business_edge_uid"], full_plan.identities.writer_uid))
    ):
        raise ValueError("Bitrix canary read-peer topology is not exact")
    config = _render_bitrix_service_config(
        revision=full_plan.revision,
        release=Path(full_plan.release["artifact_root"]),
        domain="bitrix",
        release_owner_uid=0,
        release_owner_gid=0,
        service_uid=identities["service_uid"],
        service_gid=identities["service_gid"],
        socket_gid=identities["socket_client_gid"],
        read_peer_uids=read_peers,
        mutation_peer_uid=full_plan.identities.writer_uid,
        receipt_public_key_id=receipt_public_key_id,
        writer_key_id=writer_public_key_id,
    )
    unit = _render_bitrix_service_unit(
        revision=full_plan.revision,
        release=Path(full_plan.release["artifact_root"]),
        interpreter=Path(full_plan.release["interpreter"]),
        domain="bitrix",
        service_user="muncho-edge-bitrix",
        service_group="muncho-edge-bitrix",
        release_owner_uid=0,
        release_owner_gid=0,
        service_uid=identities["service_uid"],
        service_gid=identities["service_gid"],
        socket_group="muncho-edge-bitrix-c",
        socket_gid=identities["socket_client_gid"],
    )
    marker = b"Restart=on-failure\nRestartSec=5s\n"
    replacement = b"Restart=no\nRuntimeMaxSec=900s\n"
    if unit.count(marker) != 1:
        raise RuntimeError("Bitrix canary unit restart contract drifted")
    unit = unit.replace(marker, replacement)
    if (
        b"Restart=on-failure" in unit
        or unit.count(b"Restart=no\n") != 1
        or unit.count(b"RuntimeMaxSec=900s\n") != 1
    ):
        raise RuntimeError("Bitrix canary unit bound is invalid")
    return unit, config


def bootstrap_bitrix_foundation(
    authority_value: Any,
    *,
    full_plan: FullCanaryPlan | None = None,
    runner: Callable[[Command], subprocess.CompletedProcess[bytes]] = _runner,
    identity_observer: Callable[..., Mapping[str, Any]] = _observe_bitrix_foundation_identity,
    asset_verifier: Callable[..., Mapping[str, Any]] = verify_packaged_operational_assets,
    private_key_path: Path = DEFAULT_BITRIX_RECEIPT_PRIVATE_KEY_PATH,
    public_key_path: Path = DEFAULT_BITRIX_TRUST_PATH,
    writer_public_key_path: Path = DEFAULT_BITRIX_WRITER_PUBLIC_KEY_PATH,
    identity_receipt_path: Path = DEFAULT_BITRIX_IDENTITY_BOOTSTRAP_RECEIPT,
    foundation_root: Path = DEFAULT_BITRIX_FOUNDATION_ROOT,
    key_bootstrap_root: Path = DEFAULT_BITRIX_KEY_BOOTSTRAP_ROOT,
    require_root: bool = True,
) -> Mapping[str, Any]:
    """Create the pre-plan Bitrix identity/key/artifact precursor exactly once."""

    if require_root:
        _require_root_linux()
        if (
            private_key_path != DEFAULT_BITRIX_RECEIPT_PRIVATE_KEY_PATH
            or public_key_path != DEFAULT_BITRIX_TRUST_PATH
            or writer_public_key_path != DEFAULT_BITRIX_WRITER_PUBLIC_KEY_PATH
            or identity_receipt_path != DEFAULT_BITRIX_IDENTITY_BOOTSTRAP_RECEIPT
            or foundation_root != DEFAULT_BITRIX_FOUNDATION_ROOT
            or key_bootstrap_root != DEFAULT_BITRIX_KEY_BOOTSTRAP_ROOT
        ):
            raise ValueError("Bitrix foundation production paths are fixed")
    else:
        _prepare_bitrix_foundation_directory(
            foundation_root.parent,
            require_root=False,
        )
    authority = validate_bitrix_foundation_authority(authority_value)
    full = load_full_canary_plan() if full_plan is None else full_plan
    if (
        authority["revision"] != full.revision
        or authority["full_canary_plan_sha256"] != full.sha256
        or authority["release_artifact_sha256"]
        != full.release["artifact_sha256"]
    ):
        raise PermissionError("Bitrix foundation authority differs from full canary")
    identities = authority["identities"]
    watchdog = arm_capability_expiry_watchdog(
        kind="bitrix_foundation",
        revision=full.revision,
        full_canary_plan_sha256=full.sha256,
        release_artifact_sha256=full.release["artifact_sha256"],
        interpreter=Path(full.release["interpreter"]),
        expires_at_unix=authority["expires_at_unix"],
        authority_sha256=authority["authority_sha256"],
        plan_sha256=None,
        credential_binding=None,
        runner=runner,
        state_root=(
            DEFAULT_EXPIRY_WATCHDOG_ROOT
            if require_root
            else foundation_root.parent / "expiry-watchdogs"
        ),
        systemd_root=(
            Path("/etc/systemd/system")
            if require_root
            else foundation_root.parent / "systemd"
        ),
        require_root=require_root,
    )
    assets = asset_verifier(
        release_root=Path(full.release["artifact_root"]),
        revision=full.revision,
        expected_uid=0,
        expected_gid=0,
        expected_manifest_sha256=authority["asset_manifest_sha256"],
    )
    rows = {row["asset_id"]: row for row in assets["files"]}
    if any(asset_id not in rows for asset_id in BITRIX_OPERATIONAL_EDGE_ASSET_IDS.values()):
        raise RuntimeError("bitrix_operational_edge_assets_not_packaged")
    identity = _ensure_bitrix_foundation_identity(
        service_uid=identities["service_uid"],
        service_gid=identities["service_gid"],
        socket_client_gid=identities["socket_client_gid"],
        runner=runner,
        observer=identity_observer,
    )
    identity_unsigned = {
        "operation": "create_or_attest_identity",
        "identity": identity,
        "create_only": True,
        "retained_dormant_on_rollback": True,
    }
    identity_receipt = _append_lease_artifact(
        identity_receipt_path,
        schema=CAPABILITY_BITRIX_IDENTITY_BOOTSTRAP_SCHEMA,
        value=identity_unsigned,
    )
    key = _stage_bitrix_receipt_key_pair(
        private_path=private_key_path,
        public_path=public_key_path,
        require_root=require_root,
    )
    owner = 0 if require_root else os.geteuid()  # windows-footgun: ok — Linux production/canary boundary
    group = 0 if require_root else os.getegid()  # windows-footgun: ok — Linux production/canary boundary
    writer_raw, _writer_item = _read_exact_file(
        writer_public_key_path,
        maximum=16 * 1024,
        uid=owner,
        gid=group,
        mode=0o444,
    )
    writer_public_key_id = ed25519_public_key_id(
        _load_exact_ed25519_public(writer_raw)
    )
    key_receipt_path = key_bootstrap_root / key["public_key_id"] / "bootstrap.json"
    key_receipt = _append_lease_artifact(
        key_receipt_path,
        schema=CAPABILITY_BITRIX_KEY_BOOTSTRAP_SCHEMA,
        value={
            "operation": "create_or_attest_receipt_key_pair",
            "revision": full.revision,
            "full_canary_plan_sha256": full.sha256,
            "authority_sha256": authority["authority_sha256"],
            "expires_at_unix": authority["expires_at_unix"],
            "private_path": str(private_key_path),
            "private_device": key["private_device"],
            "private_inode": key["private_inode"],
            "private_uid": key["private_uid"],
            "private_gid": key["private_gid"],
            "private_mode": "0400",
            "public_path": str(public_key_path),
            "public_device": key["public_device"],
            "public_inode": key["public_inode"],
            "public_uid": key["public_uid"],
            "public_gid": key["public_gid"],
            "public_mode": "0444",
            "public_key_id": key["public_key_id"],
            "public_sha256": key["public_sha256"],
            "writer_public_key_path": str(writer_public_key_path),
            "writer_public_key_id": writer_public_key_id,
            "create_only": True,
            "retire_private_on_stop": True,
            "retire_public_on_stop": True,
            "private_content_or_digest_recorded": False,
        },
    )
    unit, config = _render_bitrix_canary_artifacts(
        full_plan=full,
        identities=identities,
        receipt_public_key_id=key["public_key_id"],
        writer_public_key_id=writer_public_key_id,
    )
    stage_root = foundation_root / authority["authority_sha256"] / key["public_key_id"]
    _prepare_bitrix_foundation_directory(stage_root, require_root=require_root)
    unit_stage = stage_root / "muncho-operational-edge-bitrix.service"
    config_stage = stage_root / "bitrix.json"
    _atomic_no_replace_file(
        unit_stage,
        unit,
        uid=owner,
        gid=group,
        mode=0o444,
        temporary_name=f".{unit_stage.name}.staging",
        maximum=256 * 1024,
    )
    _atomic_no_replace_file(
        config_stage,
        config,
        uid=owner,
        gid=group,
        mode=0o400,
        temporary_name=f".{config_stage.name}.staging",
        maximum=256 * 1024,
    )
    foundation_receipt = _append_lease_artifact(
        stage_root / "foundation.json",
        schema=CAPABILITY_BITRIX_FOUNDATION_RECEIPT_SCHEMA,
        value={
            "operation": "bootstrap_bitrix_foundation",
            "revision": full.revision,
            "full_canary_plan_sha256": full.sha256,
            "release_artifact_sha256": full.release["artifact_sha256"],
            "authority_sha256": authority["authority_sha256"],
            "owner_subject_sha256": authority["owner_subject_sha256"],
            "expires_at_unix": authority["expires_at_unix"],
            "expiry_watchdog": watchdog,
            "identity_bootstrap_receipt_path": identity_receipt["receipt_path"],
            "identity_bootstrap_receipt_sha256": identity_receipt["receipt_sha256"],
            "key_bootstrap_receipt_path": key_receipt["receipt_path"],
            "key_bootstrap_receipt_sha256": key_receipt["receipt_sha256"],
            "receipt_public_key_id": key["public_key_id"],
            "asset_manifest_sha256": assets["manifest_sha256"],
            "asset_verification_sha256": assets["verification_sha256"],
            "asset_name_to_id": dict(BITRIX_OPERATIONAL_EDGE_ASSET_IDS),
            "rendered_unit_stage_path": str(unit_stage),
            "rendered_unit_sha256": _sha256_bytes(unit),
            "rendered_config_stage_path": str(config_stage),
            "rendered_config_sha256": _sha256_bytes(config),
            "rendered_trust_path": str(public_key_path),
            "rendered_trust_sha256": key["public_sha256"],
            "read_peer_uids": sorted(
                [identities["business_edge_uid"], full.identities.writer_uid]
            ),
            "mutation_peer_uid": full.identities.writer_uid,
            "private_content_or_digest_recorded": False,
        },
    )
    return copy.deepcopy(dict(foundation_receipt))


def load_bitrix_key_bootstrap_receipt(
    *,
    public_key_id: str,
    receipt_sha256: str,
    root: Path = DEFAULT_BITRIX_KEY_BOOTSTRAP_ROOT,
) -> Mapping[str, Any]:
    _digest(public_key_id, "Bitrix receipt public key ID")
    _digest(receipt_sha256, "Bitrix key bootstrap receipt")
    path = root / public_key_id / "bootstrap.json"
    value = _load_lease_artifact(
        path,
        schema=CAPABILITY_BITRIX_KEY_BOOTSTRAP_SCHEMA,
    )
    if (
        value.get("receipt_sha256") != receipt_sha256
        or value.get("public_key_id") != public_key_id
        or value.get("private_path")
        != str(DEFAULT_BITRIX_RECEIPT_PRIVATE_KEY_PATH)
        or value.get("public_path") != str(DEFAULT_BITRIX_TRUST_PATH)
        or value.get("retire_private_on_stop") is not True
        or value.get("retire_public_on_stop") is not True
        or value.get("private_content_or_digest_recorded") is not False
    ):
        raise RuntimeError("Bitrix key bootstrap receipt drifted")
    return value


def retire_bitrix_foundation_key_pair(
    key_bootstrap_receipt: Mapping[str, Any],
    *,
    reason: str,
    plan: CapabilityCanaryPlan | None = None,
    stop_proof: Mapping[str, Any] | None = None,
    now_unix: int | None = None,
    retirement_root: Path = DEFAULT_BITRIX_KEY_RETIREMENT_ROOT,
    require_root: bool = True,
) -> Mapping[str, Any]:
    """Retire both halves of the ephemeral pair; never leave orphan trust."""

    if reason not in {"service_stop", "foundation_expired"}:
        raise ValueError("Bitrix key retirement reason is invalid")
    current = int(time.time()) if now_unix is None else now_unix
    if type(current) is not int or current < 0:
        raise ValueError("Bitrix key retirement time is invalid")
    public_key_id = _digest(
        key_bootstrap_receipt.get("public_key_id"),
        "Bitrix key bootstrap public key ID",
    )
    receipt_path = Path(str(key_bootstrap_receipt.get("receipt_path", "")))
    receipt = _load_lease_artifact(
        receipt_path,
        schema=CAPABILITY_BITRIX_KEY_BOOTSTRAP_SCHEMA,
    )
    if receipt != dict(key_bootstrap_receipt):
        raise RuntimeError("Bitrix key bootstrap receipt changed")
    private_path = Path(receipt["private_path"])
    public_path = Path(receipt["public_path"])
    if require_root and (
        private_path != DEFAULT_BITRIX_RECEIPT_PRIVATE_KEY_PATH
        or public_path != DEFAULT_BITRIX_TRUST_PATH
        or retirement_root != DEFAULT_BITRIX_KEY_RETIREMENT_ROOT
    ):
        raise ValueError("Bitrix key retirement production paths are fixed")
    proof: Mapping[str, Any] | None = None
    if reason == "service_stop":
        if plan is None or stop_proof is None:
            raise PermissionError("Bitrix service-stop retirement lacks stop proof")
        proof = _validate_capability_stop_proof(
            plan,
            stop_proof,
            installed_at_unix=0,
            now_unix=current,
        )
        if (
            receipt.get("revision") != plan.revision
            or receipt.get("full_canary_plan_sha256")
            != plan.full_canary_plan_sha256
            or receipt.get("receipt_sha256")
            != plan.bitrix_operational_edge_key_bootstrap_receipt_sha256
            or public_key_id
            != plan.bitrix_operational_edge_receipt_public_key_id
        ):
            raise PermissionError("Bitrix key retirement plan binding drifted")
    elif current < receipt.get("expires_at_unix", current + 1):
        raise PermissionError("Bitrix foundation has not expired")

    retirement_dir = retirement_root / public_key_id
    _prepare_bitrix_foundation_directory(
        retirement_dir,
        require_root=require_root,
    )
    intent_path = retirement_dir / f"{reason}-intent.json"
    completion_path = retirement_dir / f"{reason}-completion.json"
    if os.path.lexists(completion_path):
        completed = _load_lease_artifact(
            completion_path,
            schema=CAPABILITY_BITRIX_INTERNAL_KEY_RETIREMENT_SCHEMA,
        )
        if (
            completed.get("key_bootstrap_receipt_sha256")
            != receipt["receipt_sha256"]
            or completed.get("reason") != reason
            or completed.get("both_pair_members_absent") is not True
            or os.path.lexists(private_path)
            or os.path.lexists(public_path)
        ):
            raise RuntimeError("Bitrix key retirement completion drifted")
        return completed
    intent_value = {
            "operation": "retire_bitrix_receipt_key_pair_intent",
            "reason": reason,
            "revision": receipt["revision"],
            "full_canary_plan_sha256": receipt["full_canary_plan_sha256"],
            "key_bootstrap_receipt_path": receipt["receipt_path"],
            "key_bootstrap_receipt_sha256": receipt["receipt_sha256"],
            "public_key_id": public_key_id,
            "private_path": str(private_path),
            "private_device": receipt["private_device"],
            "private_inode": receipt["private_inode"],
            "public_path": str(public_path),
            "public_device": receipt["public_device"],
            "public_inode": receipt["public_inode"],
            "public_sha256": receipt["public_sha256"],
            "service_stop_proof_sha256": (
                proof["stop_proof_sha256"] if proof is not None else None
            ),
            "foundation_expires_at_unix": receipt["expires_at_unix"],
            "requested_at_unix": current,
            "private_content_or_digest_recorded": False,
        }
    if os.path.lexists(intent_path):
        intent = _load_lease_artifact(
            intent_path,
            schema=CAPABILITY_BITRIX_INTERNAL_KEY_RETIREMENT_INTENT_SCHEMA,
        )
        stable_fields = {
            key: value
            for key, value in intent_value.items()
            if key != "requested_at_unix"
        }
        if any(intent.get(key) != value for key, value in stable_fields.items()):
            raise RuntimeError("Bitrix key retirement intent drifted")
    else:
        intent = _append_lease_artifact(
            intent_path,
            schema=CAPABILITY_BITRIX_INTERNAL_KEY_RETIREMENT_INTENT_SCHEMA,
            value=intent_value,
        )

    owner = 0 if require_root else os.geteuid()  # windows-footgun: ok — Linux production/canary boundary
    group = 0 if require_root else os.getegid()  # windows-footgun: ok — Linux production/canary boundary
    if os.path.lexists(private_path):
        private_raw, private_item = _read_exact_file(
            private_path,
            maximum=16 * 1024,
            uid=owner,
            gid=group,
            mode=0o400,
        )
        private = _load_exact_ed25519_private(private_raw)
        if (
            private_item.st_dev != receipt["private_device"]
            or private_item.st_ino != receipt["private_inode"]
            or ed25519_public_key_id(private.public_key()) != public_key_id
        ):
            raise RuntimeError("Bitrix private key substitution detected")
    if os.path.lexists(public_path):
        public_raw, public_item = _read_exact_file(
            public_path,
            maximum=16 * 1024,
            uid=owner,
            gid=group,
            mode=0o444,
        )
        if (
            public_item.st_dev != receipt["public_device"]
            or public_item.st_ino != receipt["public_inode"]
            or _sha256_bytes(public_raw) != receipt["public_sha256"]
            or ed25519_public_key_id(_load_exact_ed25519_public(public_raw))
            != public_key_id
        ):
            raise RuntimeError("Bitrix public trust substitution detected")
    for path in (private_path, public_path):
        if os.path.lexists(path):
            os.unlink(path)
            _fsync_directory(path.parent)
    private_absent = not os.path.lexists(private_path)
    public_absent = not os.path.lexists(public_path)
    if not private_absent or not public_absent:
        raise RuntimeError("Bitrix key-pair retirement is incomplete")
    retired_at = int(time.time()) if now_unix is None else now_unix
    return _append_lease_artifact(
        completion_path,
        schema=CAPABILITY_BITRIX_INTERNAL_KEY_RETIREMENT_SCHEMA,
        value={
            "operation": "retire_bitrix_receipt_key_pair",
            "reason": reason,
            "revision": receipt["revision"],
            "full_canary_plan_sha256": receipt["full_canary_plan_sha256"],
            "key_bootstrap_receipt_path": receipt["receipt_path"],
            "key_bootstrap_receipt_sha256": receipt["receipt_sha256"],
            "retirement_intent_path": intent["receipt_path"],
            "retirement_intent_sha256": intent["receipt_sha256"],
            "public_key_id": public_key_id,
            "private_path": str(private_path),
            "public_path": str(public_path),
            "private_absent": private_absent,
            "public_absent": public_absent,
            "both_pair_members_absent": private_absent and public_absent,
            "service_stop_proof_sha256": (
                proof["stop_proof_sha256"] if proof is not None else None
            ),
            "retired_at_unix": retired_at,
            "private_content_or_digest_recorded": False,
        },
    )


def validate_bitrix_foundation_for_plan(
    plan: CapabilityCanaryPlan,
    full_plan: FullCanaryPlan,
    *,
    foundation_root: Path = DEFAULT_BITRIX_FOUNDATION_ROOT,
    key_bootstrap_root: Path = DEFAULT_BITRIX_KEY_BOOTSTRAP_ROOT,
    identity_receipt_path: Path = DEFAULT_BITRIX_IDENTITY_BOOTSTRAP_RECEIPT,
    now_unix: int | None = None,
) -> Mapping[str, Any]:
    """Bind a published plan only to one current, pre-plan Bitrix precursor."""

    validate_plan_against_full(plan, full_plan)
    now = int(time.time()) if now_unix is None else now_unix
    identity = _load_lease_artifact(
        identity_receipt_path,
        schema=CAPABILITY_BITRIX_IDENTITY_BOOTSTRAP_SCHEMA,
    )
    if (
        identity.get("receipt_sha256")
        != plan.bitrix_operational_edge_identity_bootstrap_receipt_sha256
        or identity.get("identity")
        != {
            "service_user": plan.identities.bitrix_operational_edge_user,
            "service_group": plan.identities.bitrix_operational_edge_group,
            "service_uid": plan.identities.bitrix_operational_edge_uid,
            "service_gid": plan.identities.bitrix_operational_edge_gid,
            "socket_client_group": (
                plan.identities.bitrix_operational_edge_client_group
            ),
            "socket_client_gid": (
                plan.identities.bitrix_operational_edge_client_gid
            ),
            "state": "present_exact",
        }
    ):
        raise RuntimeError("bitrix_foundation_bootstrap_unavailable")
    key = load_bitrix_key_bootstrap_receipt(
        public_key_id=plan.bitrix_operational_edge_receipt_public_key_id,
        receipt_sha256=(
            plan.bitrix_operational_edge_key_bootstrap_receipt_sha256
        ),
        root=key_bootstrap_root,
    )
    if (
        key.get("revision") != plan.revision
        or key.get("full_canary_plan_sha256") != plan.full_canary_plan_sha256
        or key.get("expires_at_unix", -1) < now
        or not os.path.lexists(DEFAULT_BITRIX_RECEIPT_PRIVATE_KEY_PATH)
        or not os.path.lexists(DEFAULT_BITRIX_TRUST_PATH)
    ):
        raise RuntimeError("bitrix_foundation_bootstrap_unavailable")
    candidates: list[Mapping[str, Any]] = []
    try:
        authority_names = sorted(os.listdir(foundation_root))
    except FileNotFoundError:
        authority_names = []
    if len(authority_names) > 16 or any(
        _SHA256_RE.fullmatch(name) is None for name in authority_names
    ):
        raise RuntimeError("bitrix_foundation_bootstrap_unavailable")
    for authority_name in authority_names:
        receipt_path = (
            foundation_root
            / authority_name
            / plan.bitrix_operational_edge_receipt_public_key_id
            / "foundation.json"
        )
        if not os.path.lexists(receipt_path):
            continue
        receipt = _load_lease_artifact(
            receipt_path,
            schema=CAPABILITY_BITRIX_FOUNDATION_RECEIPT_SCHEMA,
        )
        if (
            receipt.get("revision") == plan.revision
            and receipt.get("full_canary_plan_sha256")
            == plan.full_canary_plan_sha256
            and receipt.get("receipt_public_key_id")
            == plan.bitrix_operational_edge_receipt_public_key_id
            and receipt.get("asset_manifest_sha256")
            == plan.bitrix_operational_edge_asset_manifest_sha256
            and receipt.get("identity_bootstrap_receipt_sha256")
            == plan.bitrix_operational_edge_identity_bootstrap_receipt_sha256
            and receipt.get("key_bootstrap_receipt_sha256")
            == plan.bitrix_operational_edge_key_bootstrap_receipt_sha256
            and receipt.get("rendered_unit_sha256")
            == plan.bitrix_operational_edge_rendered_unit_sha256
            and receipt.get("rendered_config_sha256")
            == plan.bitrix_operational_edge_rendered_config_sha256
            and receipt.get("rendered_trust_sha256")
            == plan.bitrix_operational_edge_rendered_trust_sha256
            and receipt.get("read_peer_uids")
            == sorted(
                [plan.identities.mac_ops_uid, full_plan.identities.writer_uid]
            )
            and receipt.get("mutation_peer_uid")
            == full_plan.identities.writer_uid
            and receipt.get("expires_at_unix", -1) >= now
        ):
            candidates.append(receipt)
    if len(candidates) != 1:
        raise RuntimeError("bitrix_foundation_bootstrap_unavailable")
    foundation = candidates[0]
    owner = 0
    unit_raw, _ = _read_exact_file(
        Path(foundation["rendered_unit_stage_path"]),
        maximum=256 * 1024,
        uid=owner,
        gid=owner,
        mode=0o444,
    )
    config_raw, _ = _read_exact_file(
        Path(foundation["rendered_config_stage_path"]),
        maximum=256 * 1024,
        uid=owner,
        gid=owner,
        mode=0o400,
    )
    trust_raw, _ = _read_exact_file(
        DEFAULT_BITRIX_TRUST_PATH,
        maximum=16 * 1024,
        uid=owner,
        gid=owner,
        mode=0o444,
    )
    if (
        _sha256_bytes(unit_raw)
        != plan.bitrix_operational_edge_rendered_unit_sha256
        or _sha256_bytes(config_raw)
        != plan.bitrix_operational_edge_rendered_config_sha256
        or _sha256_bytes(trust_raw)
        != plan.bitrix_operational_edge_rendered_trust_sha256
        or b"Restart=no\n" not in unit_raw
        or b"RuntimeMaxSec=900s\n" not in unit_raw
        or b"Restart=on-failure" in unit_raw
    ):
        raise RuntimeError("bitrix_foundation_bootstrap_unavailable")
    precursor_bytes = b"".join(
        _canonical_bytes(value) for value in (identity, key, foundation)
    )
    if b'"plan_sha256"' in precursor_bytes or b'"capability_plan_sha256"' in precursor_bytes:
        raise RuntimeError("bitrix_foundation_bootstrap_unavailable")
    return {
        "foundation_receipt_path": foundation["receipt_path"],
        "foundation_receipt_sha256": foundation["receipt_sha256"],
        "identity_bootstrap_receipt_sha256": identity["receipt_sha256"],
        "key_bootstrap_receipt_sha256": key["receipt_sha256"],
        "receipt_public_key_id": key["public_key_id"],
        "asset_verification_sha256": foundation["asset_verification_sha256"],
        "rendered_unit_stage_path": foundation["rendered_unit_stage_path"],
        "rendered_config_stage_path": foundation["rendered_config_stage_path"],
        "expires_at_unix": foundation["expires_at_unix"],
        "ready": True,
    }


def _load_expiry_watchdog_authority(
    watchdog_id: str,
    *,
    state_root: Path = DEFAULT_EXPIRY_WATCHDOG_ROOT,
) -> Mapping[str, Any]:
    paths = _expiry_watchdog_paths(watchdog_id, state_root=state_root)
    value = _load_lease_artifact(
        Path(paths["authority"]),
        schema=CAPABILITY_EXPIRY_WATCHDOG_AUTHORITY_SCHEMA,
    )
    if (
        value.get("watchdog_id") != watchdog_id
        or value.get("persistent_across_reboot") is not True
        or value.get("earliest_expiry_not_extended") is not True
        or value.get("cleanup_at_unix") != value.get("expires_at_unix")
    ):
        raise RuntimeError("capability expiry watchdog authority drifted")
    return value


def _bitrix_key_receipts_for_authority(
    authority_sha256: str,
    *,
    root: Path = DEFAULT_BITRIX_KEY_BOOTSTRAP_ROOT,
) -> list[Mapping[str, Any]]:
    _digest(authority_sha256, "Bitrix foundation authority")
    try:
        names = sorted(os.listdir(root))
    except FileNotFoundError:
        return []
    if len(names) > 16 or any(_SHA256_RE.fullmatch(name) is None for name in names):
        raise RuntimeError("Bitrix key bootstrap inventory is invalid")
    matches = []
    for name in names:
        path = root / name / "bootstrap.json"
        if not os.path.lexists(path):
            continue
        receipt = _load_lease_artifact(
            path,
            schema=CAPABILITY_BITRIX_KEY_BOOTSTRAP_SCHEMA,
        )
        if receipt.get("authority_sha256") == authority_sha256:
            matches.append(receipt)
    return matches


def _retire_unjournaled_bitrix_pair_after_expiry(
    authority: Mapping[str, Any],
    *,
    now_unix: int,
    private_path: Path = DEFAULT_BITRIX_RECEIPT_PRIVATE_KEY_PATH,
    public_path: Path = DEFAULT_BITRIX_TRUST_PATH,
    retirement_root: Path = DEFAULT_BITRIX_KEY_RETIREMENT_ROOT,
    require_root: bool = True,
) -> Mapping[str, Any]:
    """Close the watchdog-before-key-receipt crash window without key digests."""

    if (
        type(now_unix) is not int
        or now_unix < authority.get("expires_at_unix", now_unix + 1)
    ):
        raise PermissionError("Bitrix orphan key pair has not expired")
    authority_sha256 = _digest(
        authority.get("authority_source_sha256"),
        "Bitrix foundation authority",
    )
    if require_root and (
        private_path != DEFAULT_BITRIX_RECEIPT_PRIVATE_KEY_PATH
        or public_path != DEFAULT_BITRIX_TRUST_PATH
        or retirement_root != DEFAULT_BITRIX_KEY_RETIREMENT_ROOT
    ):
        raise ValueError("Bitrix orphan-key production paths are fixed")
    if not os.path.lexists(private_path) and not os.path.lexists(public_path):
        return {
            "state": "never_created_absent",
            "private_path": str(private_path),
            "public_path": str(public_path),
            "private_absent": True,
            "public_absent": True,
            "both_pair_members_absent": True,
            "private_content_or_digest_recorded": False,
        }

    owner = 0 if require_root else os.geteuid()  # windows-footgun: ok — Linux production/canary boundary
    group = 0 if require_root else os.getegid()  # windows-footgun: ok — Linux production/canary boundary
    retirement_dir = retirement_root / authority_sha256
    _prepare_bitrix_foundation_directory(
        retirement_dir,
        require_root=require_root,
    )
    intent_path = retirement_dir / "foundation-expired-orphan-intent.json"
    completion_path = retirement_dir / "foundation-expired-orphan-completion.json"
    if os.path.lexists(completion_path):
        completed = _load_lease_artifact(
            completion_path,
            schema=CAPABILITY_BITRIX_INTERNAL_KEY_RETIREMENT_SCHEMA,
        )
        if (
            completed.get("authority_source_sha256") != authority_sha256
            or completed.get("both_pair_members_absent") is not True
            or os.path.lexists(private_path)
            or os.path.lexists(public_path)
        ):
            raise RuntimeError("Bitrix orphan key retirement completion drifted")
        return completed

    def identity(path: Path, mode: int) -> Mapping[str, Any] | None:
        if not os.path.lexists(path):
            return None
        _raw, item = _read_exact_file(
            path,
            maximum=16 * 1024,
            uid=owner,
            gid=group,
            mode=mode,
        )
        return {
            "device": item.st_dev,
            "inode": item.st_ino,
            "uid": item.st_uid,
            "gid": item.st_gid,
            "mode": f"{stat.S_IMODE(item.st_mode):04o}",
            "size": item.st_size,
        }

    current_private = identity(private_path, 0o400)
    current_public = identity(public_path, 0o444)
    if os.path.lexists(intent_path):
        intent = _load_lease_artifact(
            intent_path,
            schema=CAPABILITY_BITRIX_INTERNAL_KEY_RETIREMENT_INTENT_SCHEMA,
        )
        if (
            intent.get("operation")
            != "retire_unjournaled_bitrix_receipt_key_pair_intent"
            or intent.get("authority_source_sha256") != authority_sha256
            or intent.get("private_path") != str(private_path)
            or intent.get("public_path") != str(public_path)
            or intent.get("private_content_or_digest_recorded") is not False
        ):
            raise RuntimeError("Bitrix orphan key retirement intent drifted")
        for label, observed in (
            ("private", current_private),
            ("public", current_public),
        ):
            recorded = intent.get(f"{label}_identity")
            if observed is not None and recorded != observed:
                raise RuntimeError("Bitrix orphan key substitution detected")
            if observed is not None and intent.get(f"{label}_was_present") is not True:
                raise RuntimeError("Bitrix orphan key appeared after intent")
    else:
        intent = _append_lease_artifact(
            intent_path,
            schema=CAPABILITY_BITRIX_INTERNAL_KEY_RETIREMENT_INTENT_SCHEMA,
            value={
                "operation": "retire_unjournaled_bitrix_receipt_key_pair_intent",
                "reason": "foundation_expired",
                "revision": authority["revision"],
                "full_canary_plan_sha256": authority[
                    "full_canary_plan_sha256"
                ],
                "authority_source_sha256": authority_sha256,
                "private_path": str(private_path),
                "private_was_present": current_private is not None,
                "private_identity": current_private,
                "public_path": str(public_path),
                "public_was_present": current_public is not None,
                "public_identity": current_public,
                "foundation_expires_at_unix": authority["expires_at_unix"],
                "requested_at_unix": now_unix,
                "private_content_or_digest_recorded": False,
            },
        )
    for path in (private_path, public_path):
        if os.path.lexists(path):
            os.unlink(path)
            _fsync_directory(path.parent)
    private_absent = not os.path.lexists(private_path)
    public_absent = not os.path.lexists(public_path)
    if not private_absent or not public_absent:
        raise RuntimeError("Bitrix orphan key-pair retirement is incomplete")
    return _append_lease_artifact(
        completion_path,
        schema=CAPABILITY_BITRIX_INTERNAL_KEY_RETIREMENT_SCHEMA,
        value={
            "operation": "retire_unjournaled_bitrix_receipt_key_pair",
            "reason": "foundation_expired",
            "revision": authority["revision"],
            "full_canary_plan_sha256": authority["full_canary_plan_sha256"],
            "authority_source_sha256": authority_sha256,
            "retirement_intent_path": intent["receipt_path"],
            "retirement_intent_sha256": intent["receipt_sha256"],
            "private_path": str(private_path),
            "public_path": str(public_path),
            "private_absent": private_absent,
            "public_absent": public_absent,
            "both_pair_members_absent": private_absent and public_absent,
            "retired_at_unix": now_unix,
            "private_content_or_digest_recorded": False,
        },
    )


def disarm_all_capability_expiry_watchdogs(
    *,
    runner: Callable[[Command], subprocess.CompletedProcess[bytes]] = _runner,
    state_root: Path = DEFAULT_EXPIRY_WATCHDOG_ROOT,
    systemd_root: Path = Path("/etc/systemd/system"),
    require_root: bool = True,
) -> Mapping[str, Any]:
    """Disable every exact timer only after complete credential/key absence."""

    if require_root:
        _require_root_linux()
    try:
        names = sorted(os.listdir(state_root))
    except FileNotFoundError:
        names = []
    if len(names) > 64 or any(_LEASE_ID_RE.fullmatch(name) is None for name in names):
        raise RuntimeError("capability expiry watchdog inventory is invalid")
    retired: list[Mapping[str, Any]] = []
    for watchdog_id in names:
        authority = _load_expiry_watchdog_authority(
            watchdog_id,
            state_root=state_root,
        )
        paths = _expiry_watchdog_paths(
            watchdog_id,
            state_root=state_root,
            systemd_root=systemd_root,
        )
        if any(
            os.path.lexists(Path(paths[field]))
            for field in ("service_path", "timer_path", "timer_wants_path")
        ):
            _run_checked(
                Command((SYSTEMCTL, "stop", str(paths["timer_name"]))),
                runner=runner,
                label=f"stop {paths['timer_name']}",
            )
        expected_service, expected_timer = render_expiry_watchdog_units(
            watchdog_id=watchdog_id,
            interpreter=Path(authority["interpreter"]),
            cleanup_at_unix=authority["cleanup_at_unix"],
        )
        wants_path = Path(paths["timer_wants_path"])
        if os.path.lexists(wants_path):
            wants_item = os.lstat(wants_path)
            if (
                not stat.S_ISLNK(wants_item.st_mode)
                or os.readlink(wants_path) != f"../{paths['timer_name']}"
            ):
                raise RuntimeError("capability expiry watchdog wants link drifted")
            os.unlink(wants_path)
            _fsync_directory(wants_path.parent)
        removals: dict[str, bool] = {
            "wants": not os.path.lexists(wants_path)
        }
        for label, path, expected in (
            ("service", Path(paths["service_path"]), expected_service),
            ("timer", Path(paths["timer_path"]), expected_timer),
        ):
            if os.path.lexists(path):
                raw, _ = _read_exact_file(
                    path,
                    maximum=64 * 1024,
                    uid=0 if require_root else os.geteuid(),  # windows-footgun: ok — Linux production/canary boundary
                    gid=0 if require_root else os.getegid(),  # windows-footgun: ok — Linux production/canary boundary
                    mode=0o644,
                )
                if raw != expected:
                    raise RuntimeError("capability expiry watchdog unit substitution")
                os.unlink(path)
                _fsync_directory(path.parent)
            removals[label] = not os.path.lexists(path)
        retired.append(
            {
                "watchdog_id": watchdog_id,
                "authority_receipt_sha256": authority["receipt_sha256"],
                "timer_disabled": True,
                "timer_wants_absent": removals["wants"],
                "service_absent": removals["service"],
                "timer_absent": removals["timer"],
            }
        )
    if names:
        _run_checked(
            Command((SYSTEMCTL, "daemon-reload")),
            runner=runner,
            label="reload retired capability expiry watchdogs",
        )
    return {
        "watchdog_count": len(names),
        "retired": retired,
        "all_timers_disabled": all(item["timer_disabled"] for item in retired),
        "all_unit_files_absent": all(
            item["timer_wants_absent"]
            and item["service_absent"]
            and item["timer_absent"]
            for item in retired
        ),
    }


def run_capability_expiry_cleanup(
    watchdog_id: str,
    *,
    runner: Callable[[Command], subprocess.CompletedProcess[bytes]] = _runner,
    now_unix: int | None = None,
    state_root: Path = DEFAULT_EXPIRY_WATCHDOG_ROOT,
    systemd_root: Path = Path("/etc/systemd/system"),
    credential_paths: Mapping[str, Path] | None = None,
    private_key_path: Path = DEFAULT_BITRIX_RECEIPT_PRIVATE_KEY_PATH,
    public_key_path: Path = DEFAULT_BITRIX_TRUST_PATH,
    key_bootstrap_root: Path = DEFAULT_BITRIX_KEY_BOOTSTRAP_ROOT,
    key_retirement_root: Path = DEFAULT_BITRIX_KEY_RETIREMENT_ROOT,
    service_observer: Callable[..., Mapping[str, Mapping[str, Any]]] = (
        _capability_services
    ),
    require_root: bool = True,
) -> Mapping[str, Any]:
    """Execute persistent stop→six-lease→Bitrix-pair cleanup after expiry."""

    fixed_credential_paths = {
        binding: Path(value["target_path"])
        for binding, value in _credential_bindings_mapping().items()
    }
    observed_credential_paths = (
        fixed_credential_paths
        if credential_paths is None
        else dict(credential_paths)
    )
    if (
        set(observed_credential_paths) != set(CAPABILITY_CREDENTIAL_BINDINGS)
        or any(
            not isinstance(path, Path)
            or not path.is_absolute()
            or ".." in path.parts
            for path in observed_credential_paths.values()
        )
    ):
        raise ValueError("capability expiry credential paths are invalid")
    if require_root:
        _require_root_linux()
        if (
            state_root != DEFAULT_EXPIRY_WATCHDOG_ROOT
            or systemd_root != Path("/etc/systemd/system")
            or observed_credential_paths != fixed_credential_paths
            or private_key_path != DEFAULT_BITRIX_RECEIPT_PRIVATE_KEY_PATH
            or public_key_path != DEFAULT_BITRIX_TRUST_PATH
            or key_bootstrap_root != DEFAULT_BITRIX_KEY_BOOTSTRAP_ROOT
            or key_retirement_root != DEFAULT_BITRIX_KEY_RETIREMENT_ROOT
            or service_observer is not _capability_services
        ):
            raise ValueError("capability expiry production boundaries are fixed")
    now = int(time.time()) if now_unix is None else now_unix
    authority = _load_expiry_watchdog_authority(
        watchdog_id,
        state_root=state_root,
    )
    if type(now) is not int or now < authority["cleanup_at_unix"]:
        raise PermissionError("capability expiry watchdog fired before its bound time")
    paths = _expiry_watchdog_paths(
        watchdog_id,
        state_root=state_root,
        systemd_root=systemd_root,
    )
    completion_path = Path(paths["completion"])
    if os.path.lexists(completion_path):
        completed = _load_lease_artifact(
            completion_path,
            schema=CAPABILITY_EXPIRY_WATCHDOG_COMPLETION_SCHEMA,
        )
        if (
            completed.get("watchdog_authority_sha256")
            != authority["receipt_sha256"]
            or completed.get("ok") is not True
        ):
            raise RuntimeError("capability expiry watchdog completion drifted")
        disarm_all_capability_expiry_watchdogs(
            runner=runner,
            state_root=state_root,
            systemd_root=systemd_root,
            require_root=require_root,
        )
        return completed

    stopped, stop_errors = _attempt_capability_stop_order(
        lambda unit: _run_checked(
            Command((SYSTEMCTL, "stop", unit), timeout_seconds=120),
            runner=runner,
            label=f"expiry stop {unit}",
        )
    )
    services: Mapping[str, Mapping[str, Any]] = {}
    service_error_sha256s = [
        _sha256_bytes(
            f"{type(exc).__name__}:{exc}".encode("utf-8", errors="replace")
        )
        for exc in stop_errors
    ]
    try:
        services = service_observer(runner=runner)
        if set(services) != set(CAPABILITY_STOP_ORDER):
            raise RuntimeError("capability expiry service inventory drifted")
    except BaseException as exc:
        service_error_sha256s.append(
            _sha256_bytes(
                f"{type(exc).__name__}:{exc}".encode("utf-8", errors="replace")
            )
        )
    all_services_stopped = bool(services) and all(
        _service_stopped(state) for state in services.values()
    )
    plan: CapabilityCanaryPlan | None = None
    full: FullCanaryPlan | None = None
    external: Mapping[str, Any] = {}
    bitrix_pair: Mapping[str, Any] = {}
    errors: dict[str, str] = {}
    if os.path.lexists(DEFAULT_PLAN_PATH):
        try:
            plan = load_capability_plan()
            full = load_full_canary_plan()
            if authority.get("plan_sha256") not in {None, plan.sha256}:
                raise PermissionError("watchdog plan binding drifted")
            if not all_services_stopped:
                raise RuntimeError("services remain live at watchdog expiry")
            stop_proof = build_capability_stop_proof(
                plan,
                services,
                stop_order=CAPABILITY_STOP_ORDER,
                observed_at_unix=now,
            )
            external = retire_secret_leases_best_effort(
                plan,
                full,
                stop_proof=stop_proof,
                runner=runner,
            )
            key = load_bitrix_key_bootstrap_receipt(
                public_key_id=plan.bitrix_operational_edge_receipt_public_key_id,
                receipt_sha256=(
                    plan.bitrix_operational_edge_key_bootstrap_receipt_sha256
                ),
                root=key_bootstrap_root,
            )
            bitrix_pair = retire_bitrix_foundation_key_pair(
                key,
                reason="service_stop",
                plan=plan,
                stop_proof=stop_proof,
                now_unix=now,
                retirement_root=key_retirement_root,
                require_root=require_root,
            )
        except BaseException as exc:
            errors["planned_cleanup"] = _sha256_bytes(
                f"{type(exc).__name__}:{exc}".encode("utf-8", errors="replace")
            )
    else:
        if not all_services_stopped:
            errors["preplan_services"] = _sha256_bytes(
                b"services_remain_live_before_foundation_expiry_cleanup"
            )
        absence = {
            binding: not os.path.lexists(path)
            for binding, path in observed_credential_paths.items()
        }
        external = {
            "state": "preplan_absence_observation",
            "credential_absence": absence,
            "all_six_credentials_absent_readback": all(absence.values()),
        }
        if not all(absence.values()):
            errors["preplan_external_credential"] = _sha256_bytes(
                b"credential_exists_before_plan_publication"
            )
        try:
            if not all_services_stopped:
                raise RuntimeError(
                    "services remain live at foundation watchdog expiry"
                )
            keys = _bitrix_key_receipts_for_authority(
                authority["authority_source_sha256"],
                root=key_bootstrap_root,
            )
            if len(keys) > 1:
                raise RuntimeError("multiple Bitrix keys bind one foundation authority")
            if keys:
                bitrix_pair = retire_bitrix_foundation_key_pair(
                    keys[0],
                    reason="foundation_expired",
                    now_unix=now,
                    retirement_root=key_retirement_root,
                    require_root=require_root,
                )
            else:
                bitrix_pair = _retire_unjournaled_bitrix_pair_after_expiry(
                    authority,
                    now_unix=now,
                    private_path=private_key_path,
                    public_path=public_key_path,
                    retirement_root=key_retirement_root,
                    require_root=require_root,
                )
        except BaseException as exc:
            errors["bitrix_pair_cleanup"] = _sha256_bytes(
                f"{type(exc).__name__}:{exc}".encode("utf-8", errors="replace")
            )
    all_external_absent = all(
        not os.path.lexists(path) for path in observed_credential_paths.values()
    )
    pair_absent = (
        not os.path.lexists(private_key_path)
        and not os.path.lexists(public_key_path)
    )
    ok = all_services_stopped and all_external_absent and pair_absent and not errors
    if not ok:
        failures: list[BaseException] = [
            RuntimeError("capability persistent expiry cleanup remains incomplete")
        ]
        failures.extend(
            RuntimeError(f"{label}:{digest}")
            for label, digest in sorted(errors.items())
        )
        raise BaseExceptionGroup(
            "capability persistent expiry cleanup will retry",
            failures,
        )
    completion = _append_lease_artifact(
        completion_path,
        schema=CAPABILITY_EXPIRY_WATCHDOG_COMPLETION_SCHEMA,
        value={
            "operation": "persistent_expiry_cleanup",
            "watchdog_id": watchdog_id,
            "watchdog_authority_sha256": authority["receipt_sha256"],
            "stop_order_attempted": list(CAPABILITY_STOP_ORDER),
            "stop_order_completed": stopped,
            "service_error_sha256s": service_error_sha256s,
            "all_services_stopped": all_services_stopped,
            "external_cleanup": copy.deepcopy(dict(external)),
            "bitrix_key_pair_cleanup": copy.deepcopy(dict(bitrix_pair)),
            "error_sha256s": errors,
            "all_six_credentials_absent_readback": all_external_absent,
            "bitrix_private_absent": not os.path.lexists(
                private_key_path
            ),
            "bitrix_public_absent": not os.path.lexists(public_key_path),
            "bitrix_pair_absent": pair_absent,
            "completed_at_unix": now,
            "ok": ok,
        },
    )
    disarm_all_capability_expiry_watchdogs(
        runner=runner,
        state_root=state_root,
        systemd_root=systemd_root,
        require_root=require_root,
    )
    return completion


_PLAN_PUBLICATION_IDENTITY_FIELDS = frozenset(
    {
        "mac_ops_uid",
        "mac_ops_gid",
        "connector_uid",
        "connector_gid",
        "bitrix_operational_edge_uid",
        "bitrix_operational_edge_gid",
        "bitrix_operational_edge_client_gid",
        "browser_uid",
        "browser_gid",
        "worker_uid",
        "worker_gid",
        "worker_client_gid",
    }
)
_PLAN_PUBLICATION_DISCORD_FIELDS = frozenset(
    {
        "connector_bot_user_id",
        "routeback_bot_user_id",
        "allowed_guild_ids",
        "allowed_channel_ids",
        "allowed_user_ids",
    }
)
_PLAN_PUBLICATION_ARTIFACT_FIELDS = frozenset(
    {
        "browser_node_sha256",
        "browser_wrapper_sha256",
        "browser_native_sha256",
        "browser_executable_sha256",
        "agent_browser_config_sha256",
        "worker_bwrap_sha256",
        "worker_shell_sha256",
        "runtime_dependency_manifest_sha256",
        "bitrix_operational_edge_asset_manifest_sha256",
        "bitrix_operational_edge_rendered_unit_sha256",
        "bitrix_operational_edge_rendered_config_sha256",
        "bitrix_operational_edge_rendered_trust_sha256",
        "bitrix_operational_edge_identity_bootstrap_receipt_sha256",
        "bitrix_operational_edge_receipt_public_key_id",
        "bitrix_operational_edge_key_bootstrap_receipt_sha256",
    }
)
_PLAN_PUBLICATION_AUTHORITY_FIELDS = frozenset(
    {
        "schema",
        "scope",
        "revision",
        "full_canary_plan_sha256",
        "plan_sha256",
        "owner_subject_sha256",
        "authority_kind",
        "cryptographic_owner_proof",
        "inputs",
        "secret_material_recorded",
        "secret_digest_recorded",
        "semantic_content_recorded",
        "authority_sha256",
    }
)
_PLAN_PUBLICATION_RECEIPT_FIELDS = frozenset(
    {
        "schema",
        "operation",
        "revision",
        "plan_sha256",
        "full_canary_plan_sha256",
        "plan_path",
        "plan_file_sha256",
        "authority_sha256",
        "owner_subject_sha256",
        "connector_bot_user_id",
        "routeback_bot_user_id",
        "production_bot_user_id",
        "stopped_service_state_sha256",
        "prerequisite_evidence_sha256",
        "receipt_path",
        "published_at_unix",
        "secret_material_recorded",
        "secret_digest_recorded",
        "semantic_content_recorded",
        "receipt_sha256",
    }
)
_MAX_PLAN_PUBLICATION_AUTHORITY_BYTES = 512 * 1024


def _canonical_snowflake_list(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or len(value) > 64:
        raise ValueError(f"{label} is not an exact allowlist")
    result = tuple(_snowflake_id(item, label) for item in value)
    if list(result) != sorted(set(result)):
        raise ValueError(f"{label} is not an exact allowlist")
    return result


def validate_plan_publication_authority(value: Any) -> Mapping[str, Any]:
    """Validate one secret-free owner authority for an exact plan digest."""

    raw = _strict_mapping(
        value,
        _PLAN_PUBLICATION_AUTHORITY_FIELDS,
        "capability plan publication authority",
    )
    if (
        raw["schema"] != CAPABILITY_PLAN_PUBLICATION_AUTHORITY_SCHEMA
        or raw["scope"] != CAPABILITY_PLAN_PUBLICATION_SCOPE
        or raw["authority_kind"]
        != "trusted_gcloud_owner_explicit_plan_digest"
        or raw["cryptographic_owner_proof"] is not False
        or raw["secret_material_recorded"] is not False
        or raw["secret_digest_recorded"] is not False
        or raw["semantic_content_recorded"] is not False
        or not isinstance(raw["revision"], str)
        or _REVISION_RE.fullmatch(raw["revision"]) is None
    ):
        raise ValueError("capability plan publication authority is invalid")
    for field in (
        "full_canary_plan_sha256",
        "plan_sha256",
        "owner_subject_sha256",
    ):
        _digest(raw[field], f"capability plan authority {field}")
    inputs = _strict_mapping(
        raw["inputs"],
        {"identities", "discord", "artifacts"},
        "capability plan publication inputs",
    )
    identities = _strict_mapping(
        inputs["identities"],
        _PLAN_PUBLICATION_IDENTITY_FIELDS,
        "capability plan publication identities",
    )
    for field in _PLAN_PUBLICATION_IDENTITY_FIELDS:
        _positive_id(identities[field], f"capability plan {field}")
    discord = _strict_mapping(
        inputs["discord"],
        _PLAN_PUBLICATION_DISCORD_FIELDS,
        "capability plan publication Discord input",
    )
    connector_bot = _snowflake_id(
        discord["connector_bot_user_id"], "connector bot identity"
    )
    routeback_bot = _snowflake_id(
        discord["routeback_bot_user_id"], "route-back bot identity"
    )
    if len({connector_bot, routeback_bot, PRODUCTION_DISCORD_BOT_USER_ID}) != 3:
        raise ValueError("capability Discord bot identities are not isolated")
    for field in (
        "allowed_guild_ids",
        "allowed_channel_ids",
        "allowed_user_ids",
    ):
        _canonical_snowflake_list(discord[field], f"Discord {field}")
    if (
        discord["allowed_guild_ids"] != [PRODUCTION_CANARY_PUBLIC_GUILD_ID]
        or discord["allowed_channel_ids"]
        != [PRODUCTION_CANARY_PUBLIC_CHANNEL_ID]
        or discord["allowed_user_ids"] != [PRODUCTION_OWNER_USER_ID]
        or LOCKED_NONPUBLIC_CHANNEL_IDS.intersection(
            discord["allowed_channel_ids"]
        )
    ):
        raise ValueError("capability canary public Discord target is invalid")
    artifacts = _strict_mapping(
        inputs["artifacts"],
        _PLAN_PUBLICATION_ARTIFACT_FIELDS,
        "capability plan publication artifact hashes",
    )
    for field in _PLAN_PUBLICATION_ARTIFACT_FIELDS:
        _digest(artifacts[field], f"capability plan {field}")
    unsigned = {
        key: copy.deepcopy(item)
        for key, item in raw.items()
        if key != "authority_sha256"
    }
    if raw["authority_sha256"] != _sha256_json(unsigned):
        raise ValueError("capability plan publication authority self-digest drifted")
    return copy.deepcopy(dict(raw))


def read_plan_publication_authority(stream: BinaryIO) -> Mapping[str, Any]:
    raw = stream.read(_MAX_PLAN_PUBLICATION_AUTHORITY_BYTES + 1)
    if not raw or len(raw) > _MAX_PLAN_PUBLICATION_AUTHORITY_BYTES or stream.read(1):
        raise ValueError("capability plan publication authority input is invalid")
    value = _decode_json(raw, label="capability plan publication authority")
    if raw != _canonical_bytes(value):
        raise ValueError("capability plan publication authority is not canonical")
    return validate_plan_publication_authority(value)


def build_plan_from_publication_authority(
    authority_value: Any,
    full_plan: FullCanaryPlan,
) -> CapabilityCanaryPlan:
    authority = validate_plan_publication_authority(authority_value)
    if (
        authority["revision"] != full_plan.revision
        or authority["full_canary_plan_sha256"] != full_plan.sha256
    ):
        raise PermissionError(
            "capability plan authority does not bind the sealed full canary"
        )
    inputs = authority["inputs"]
    identities = inputs["identities"]
    discord = inputs["discord"]
    artifacts = inputs["artifacts"]
    plan = build_capability_plan(
        full_plan=full_plan,
        **identities,
        connector_bot_user_id=discord["connector_bot_user_id"],
        routeback_bot_user_id=discord["routeback_bot_user_id"],
        connector_allowed_guild_ids=discord["allowed_guild_ids"],
        connector_allowed_channel_ids=discord["allowed_channel_ids"],
        connector_allowed_user_ids=discord["allowed_user_ids"],
        **artifacts,
    )
    if plan.sha256 != authority["plan_sha256"]:
        raise PermissionError(
            "owner-approved capability plan digest does not match rendered plan"
        )
    return plan


def _read_published_plan_file(path: Path, *, maximum: int) -> bytes:
    raw, _item = _read_stable_file(
        path,
        maximum=maximum,
        expected_uid=0,
        expected_gid=0,
        allowed_modes=frozenset({0o400}),
    )
    return raw


def _atomic_publish_root_file(path: Path, payload: bytes) -> None:
    """Atomically publish one complete root-only file without replacement."""

    _require_root_linux()
    _ensure_root_directory(path.parent)
    temporary = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(
        temporary,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o400,
    )
    linked = False
    try:
        _write_all(descriptor, payload)
        os.fchmod(descriptor, 0o400)
        os.fchown(descriptor, 0, 0)
        os.fsync(descriptor)
        item = os.fstat(descriptor)
        if (
            not stat.S_ISREG(item.st_mode)
            or item.st_uid != 0
            or item.st_gid != 0
            or stat.S_IMODE(item.st_mode) != 0o400
            or item.st_size != len(payload)
        ):
            raise RuntimeError("capability publication temporary file is unsafe")
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, path, follow_symlinks=False)
        linked = True
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    if not linked:
        raise RuntimeError("capability publication target already exists")
    _fsync_directory(path.parent)
    observed = _read_published_plan_file(path, maximum=max(len(payload), 1))
    if observed != payload:
        raise RuntimeError("capability publication readback drifted")


def _plan_publication_receipt_path(plan: CapabilityCanaryPlan) -> Path:
    return (
        DEFAULT_PLAN_PUBLICATION_RECEIPT_ROOT
        / plan.revision
        / plan.sha256
        / "publication.json"
    )


def _validate_plan_publication_receipt(
    value: Any,
    *,
    authority: Mapping[str, Any],
    plan: CapabilityCanaryPlan,
    plan_payload: bytes,
    receipt_path: Path,
    stopped_service_state_sha256: str,
    prerequisite_evidence_sha256: str,
) -> Mapping[str, Any]:
    raw = _strict_mapping(
        value,
        _PLAN_PUBLICATION_RECEIPT_FIELDS,
        "capability plan publication receipt",
    )
    unsigned = {
        key: copy.deepcopy(item)
        for key, item in raw.items()
        if key != "receipt_sha256"
    }
    if (
        raw["schema"] != CAPABILITY_PLAN_PUBLICATION_RECEIPT_SCHEMA
        or raw["operation"] != "publish_plan"
        or raw["revision"] != plan.revision
        or raw["plan_sha256"] != plan.sha256
        or raw["full_canary_plan_sha256"] != plan.full_canary_plan_sha256
        or raw["plan_path"] != str(DEFAULT_PLAN_PATH)
        or raw["plan_file_sha256"] != _sha256_bytes(plan_payload)
        or raw["authority_sha256"] != authority["authority_sha256"]
        or raw["owner_subject_sha256"] != authority["owner_subject_sha256"]
        or raw["connector_bot_user_id"] != plan.connector_bot_user_id
        or raw["routeback_bot_user_id"] != plan.routeback_bot_user_id
        or raw["production_bot_user_id"] != PRODUCTION_DISCORD_BOT_USER_ID
        or raw["stopped_service_state_sha256"]
        != stopped_service_state_sha256
        or raw["prerequisite_evidence_sha256"]
        != prerequisite_evidence_sha256
        or raw["receipt_path"] != str(receipt_path)
        or type(raw["published_at_unix"]) is not int
        or raw["published_at_unix"] < 0
        or raw["secret_material_recorded"] is not False
        or raw["secret_digest_recorded"] is not False
        or raw["semantic_content_recorded"] is not False
        or raw["receipt_sha256"] != _sha256_json(unsigned)
    ):
        raise RuntimeError("capability plan publication receipt drifted")
    _digest(
        raw["stopped_service_state_sha256"],
        "capability publication stopped state",
    )
    _digest(
        raw["prerequisite_evidence_sha256"],
        "capability publication prerequisite evidence",
    )
    return copy.deepcopy(dict(raw))


def publish_capability_plan(authority_value: Any) -> Mapping[str, Any]:
    """Publish the one exact stopped-host capability plan and append-only receipt."""

    _require_root_linux()
    authority = validate_plan_publication_authority(authority_value)
    full_plan = load_full_canary_plan()
    validate_dedicated_canary_host(full_plan)
    release_evidence = _validate_release_manifest(full_plan)
    plan = build_plan_from_publication_authority(authority, full_plan)
    dependency_evidence = runtime_dependency_manifest_preflight(plan)
    browser_evidence = browser_executable_preflight(plan)
    worker_evidence = worker_executables_preflight(plan)
    try:
        bitrix_foundation_evidence = validate_bitrix_foundation_for_plan(
            plan,
            full_plan,
        )
    except Exception as exc:
        raise RuntimeError("bitrix_foundation_bootstrap_unavailable") from exc
    states = {
        unit: collect_service_state(unit)
        for unit in (EDGE_UNIT_NAME, WRITER_UNIT_NAME, GATEWAY_UNIT_NAME)
    }
    stopped_checks = evaluate_service_states(states, phase="stopped")
    if not stopped_checks or not all(stopped_checks.values()):
        raise RuntimeError("capability plan publication requires stopped services")
    stopped_state_sha256 = _sha256_json(
        {"states": states, "checks": dict(sorted(stopped_checks.items()))}
    )
    prerequisite_evidence_sha256 = _sha256_json(
        {
            "release": release_evidence,
            "runtime_dependencies": dependency_evidence,
            "browser": browser_evidence,
            "worker": worker_evidence,
            "bitrix_foundation": bitrix_foundation_evidence,
        }
    )
    plan_payload = _canonical_bytes(plan.to_mapping())
    receipt_path = _plan_publication_receipt_path(plan)
    with _lifecycle_lock():
        plan_exists = os.path.lexists(DEFAULT_PLAN_PATH)
        receipt_exists = os.path.lexists(receipt_path)
        if plan_exists != receipt_exists:
            raise RuntimeError("capability plan publication is incomplete")
        if plan_exists:
            existing_plan = _read_published_plan_file(
                DEFAULT_PLAN_PATH,
                maximum=_MAX_PLAN_BYTES,
            )
            existing_receipt = _read_published_plan_file(
                receipt_path,
                maximum=256 * 1024,
            )
            if existing_plan != plan_payload:
                raise RuntimeError("capability plan publication conflicts")
            receipt_value = _decode_json(
                existing_receipt,
                label="capability plan publication receipt",
            )
            if existing_receipt != _canonical_bytes(receipt_value):
                raise RuntimeError("capability plan publication receipt is not canonical")
            return _validate_plan_publication_receipt(
                receipt_value,
                authority=authority,
                plan=plan,
                plan_payload=plan_payload,
                receipt_path=receipt_path,
                stopped_service_state_sha256=stopped_state_sha256,
                prerequisite_evidence_sha256=prerequisite_evidence_sha256,
            )
        published_at = int(time.time())
        receipt_unsigned = {
            "schema": CAPABILITY_PLAN_PUBLICATION_RECEIPT_SCHEMA,
            "operation": "publish_plan",
            "revision": plan.revision,
            "plan_sha256": plan.sha256,
            "full_canary_plan_sha256": plan.full_canary_plan_sha256,
            "plan_path": str(DEFAULT_PLAN_PATH),
            "plan_file_sha256": _sha256_bytes(plan_payload),
            "authority_sha256": authority["authority_sha256"],
            "owner_subject_sha256": authority["owner_subject_sha256"],
            "connector_bot_user_id": plan.connector_bot_user_id,
            "routeback_bot_user_id": plan.routeback_bot_user_id,
            "production_bot_user_id": PRODUCTION_DISCORD_BOT_USER_ID,
            "stopped_service_state_sha256": stopped_state_sha256,
            "prerequisite_evidence_sha256": prerequisite_evidence_sha256,
            "receipt_path": str(receipt_path),
            "published_at_unix": published_at,
            "secret_material_recorded": False,
            "secret_digest_recorded": False,
            "semantic_content_recorded": False,
        }
        receipt = {
            **receipt_unsigned,
            "receipt_sha256": _sha256_json(receipt_unsigned),
        }
        _atomic_publish_root_file(DEFAULT_PLAN_PATH, plan_payload)
        _atomic_publish_root_file(receipt_path, _canonical_bytes(receipt))
        return _validate_plan_publication_receipt(
            receipt,
            authority=authority,
            plan=plan,
            plan_payload=plan_payload,
            receipt_path=receipt_path,
            stopped_service_state_sha256=stopped_state_sha256,
            prerequisite_evidence_sha256=prerequisite_evidence_sha256,
        )


def runtime_contract() -> Mapping[str, Any]:
    unsigned = {
        "schema": CAPABILITY_CONTRACT_SCHEMA,
        "normal_gateway_loop": True,
        "model_semantic_authority": True,
        "model": "gpt-5.6-sol", "provider": "openai-codex",
        "toolsets": list(FIRST_WAVE_TOOLSETS),
        "kanban_auxiliary_planning_enabled": False,
        "kanban_auto_decompose": False,
        "kanban_dispatch_in_gateway": False,
        "cron_enabled": False,
        "goal_judge_enabled": False,
        "goal_continuations_enabled": False,
        "mcp_auto_discovery_enabled": False,
        "gateway_event_hooks_enabled": False,
        "shell_hooks_enabled": False,
        "plugin_allowlist": [CAPABILITY_OBSERVER_PLUGIN],
        "plugin_observer_hooks": list(CAPABILITY_OBSERVER_HOOKS),
        "plugin_middleware_enabled": False,
        "api_loopback": "127.0.0.1:8642",
        "public_discord_transport": "credential_free_local_connector_relay",
        "direct_discord_in_gateway": False,
        "discord_dm_enabled": False,
        "start_order": list(CAPABILITY_START_ORDER),
        "stop_order": list(CAPABILITY_STOP_ORDER),
        "credential_bindings": list(CAPABILITY_CREDENTIAL_BINDINGS),
        "workspace_policy": "ephemeral_isolated_worker_lease_no_host_projection",
        "browser_identity": "dedicated_create_only_principal",
        "browser_gateway_access": "authenticated_af_unix_controller_only",
        "browser_sandbox": "unprivileged_user_namespace_required",
        "browser_controller_readiness": "real_agent_browser_command_round_trip",
        "terminal_gateway_access": "authenticated_af_unix_isolated_worker_only",
        "terminal_network_access": False,
        "terminal_credentials_available": False,
        "terminal_ready_probe": GATEWAY_READY_PROBE_CONTRACT,
        "terminal_tmpfs_contract": LEASE_TMPFS_PREFLIGHT_CONTRACT,
        "codex_refresh_token_leased": False,
        "discord_credential_in_gateway": False,
        "mac_ops_credential_in_gateway": False,
    }
    return {**unsigned, "contract_sha256": _sha256_json(unsigned)}


def load_capability_plan(path: Path = DEFAULT_PLAN_PATH) -> CapabilityCanaryPlan:
    if path != DEFAULT_PLAN_PATH:
        raise ValueError("capability plan path is fixed")
    raw, _ = _read_stable_file(path, maximum=_MAX_PLAN_BYTES, expected_uid=0, expected_gid=0, allowed_modes=frozenset({0o400}))
    value = _decode_json(raw, label="capability runtime plan")
    if raw != _canonical_bytes(value):
        raise RuntimeError("capability plan bytes are not canonical")
    return CapabilityCanaryPlan.from_mapping(value)


def load_capability_approval(
    path: Path = DEFAULT_APPROVAL_PATH,
) -> CapabilityCanaryOwnerApproval:
    if path != DEFAULT_APPROVAL_PATH:
        raise ValueError("capability approval path is fixed")
    raw, _ = _read_stable_file(
        path,
        maximum=64 * 1024,
        expected_uid=0,
        expected_gid=0,
        allowed_modes=frozenset({0o400}),
    )
    value = _decode_json(raw, label="capability owner approval")
    if raw != _canonical_bytes(value):
        raise RuntimeError("capability owner approval bytes are not canonical")
    return CapabilityCanaryOwnerApproval.from_mapping(value)


def read_capability_approval(
    stream: BinaryIO,
) -> CapabilityCanaryOwnerApproval:
    raw = stream.read(64 * 1024 + 1)
    if not raw or len(raw) > 64 * 1024 or stream.read(1):
        raise ValueError("capability owner approval input is invalid")
    value = _decode_json(raw, label="capability owner approval input")
    if raw != _canonical_bytes(value):
        raise ValueError("capability owner approval input is not canonical")
    return CapabilityCanaryOwnerApproval.from_mapping(value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Production-shaped Muncho capability-canary runtime")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("contract")
    sub.add_parser("bootstrap-bitrix-foundation")
    expiry_cleanup = sub.add_parser("expiry-cleanup")
    expiry_cleanup.add_argument("--watchdog-id", required=True)
    sub.add_parser("publish-plan")
    sub.add_parser("preflight-stopped")
    sub.add_parser("preflight-live")
    sub.add_parser("provision-codex")
    sub.add_parser("provision-mac-ops")
    sub.add_parser("provision-discord-connector")
    sub.add_parser("provision-api-control")
    sub.add_parser("provision-discord-routeback")
    sub.add_parser("provision-bitrix-operational-edge")
    sub.add_parser("install-approval")
    sub.add_parser("wait-production-observation-marker")
    sub.add_parser("stage-production-observation")
    sub.add_parser("start")
    sub.add_parser("stop")
    sub.add_parser("retire-secrets")
    return parser


def _emit(value: Mapping[str, Any]) -> None:
    sys.stdout.buffer.write(_canonical_bytes(value) + b"\n")
    sys.stdout.buffer.flush()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "contract":
        _emit(runtime_contract())
        return 0
    try:
        if args.command == "publish-plan":
            _emit(publish_capability_plan(read_plan_publication_authority(sys.stdin.buffer)))
            return 0
        if args.command == "bootstrap-bitrix-foundation":
            _emit(
                bootstrap_bitrix_foundation(
                    read_bitrix_foundation_authority(sys.stdin.buffer)
                )
            )
            return 0
        if args.command == "expiry-cleanup":
            _emit(run_capability_expiry_cleanup(args.watchdog_id))
            return 0
        plan = load_capability_plan()
        full = load_full_canary_plan()
        validate_plan_against_full(plan, full)
        validate_dedicated_canary_host(full)
        _validate_release_manifest(full)
        if args.command in {
            "provision-codex",
            "provision-mac-ops",
            "provision-discord-connector",
            "provision-api-control",
            "provision-discord-routeback",
            "provision-bitrix-operational-edge",
        }:
            kind = {
                "provision-codex": "codex_access_token",
                "provision-mac-ops": "mac_ops_gitlab_env",
                "provision-discord-connector": "discord_connector_token",
                "provision-api-control": "api_server_control_key",
                "provision-discord-routeback": "discord_routeback_token",
                "provision-bitrix-operational-edge": (
                    "bitrix_operational_edge_webhook"
                ),
            }[args.command]
            metadata, secret = read_secret_lease_frame(sys.stdin.buffer, expected_kind=kind)
            _require_root_linux()
            result = provision_secret_lease(
                plan,
                metadata,
                secret,
                full_plan=full,
            )
        elif args.command in {"preflight-stopped", "preflight-live"}:
            result = collect_capability_preflight(
                plan,
                full,
                phase=(
                    "stopped"
                    if args.command == "preflight-stopped"
                    else "live"
                ),
            )
        elif args.command == "install-approval":
            result = install_capability_approval(
                plan,
                full,
                read_capability_approval(sys.stdin.buffer),
            )
        elif args.command == "wait-production-observation-marker":
            _require_root_linux()
            request = read_production_observation_wait_request(
                sys.stdin.buffer,
                plan=plan,
            )
            result = wait_for_capability_production_observation_marker(
                plan,
                request,
                observer_gid=_cleanup_observer_identity()["gid"],
            )
        elif args.command == "stage-production-observation":
            _require_root_linux()
            result = stage_and_publish_owner_signed_production_observation(
                read_owner_signed_production_observation(sys.stdin.buffer),
                plan=plan,
                observer_gid=_cleanup_observer_identity()["gid"],
            )
        elif args.command == "start":
            result = CapabilityCanaryLifecycle(plan, full).start(
                load_capability_approval(), load_full_canary_approval()
            )
        elif args.command == "stop":
            result = CapabilityCanaryLifecycle(plan, full).stop()
        else:
            _require_root_linux()
            states = _capability_services(runner=_runner)
            if not all(_service_stopped(state) for state in states.values()):
                raise RuntimeError("credential retirement requires all services stopped")
            collect_full_canary_preflight(full, phase="stopped")
            result = retire_secret_leases_best_effort(
                plan,
                full,
                stop_proof=build_capability_stop_proof(plan, states),
            )
            _emit(result)
            return 0 if result.get("ok") is True else 1
        _emit(result)
        return 0
    except Exception as exc:
        failure = {"schema": "muncho-production-capability-runtime-failure.v1", "ok": False, "error_type": type(exc).__name__, "error_sha256": _sha256_bytes(f"{type(exc).__name__}:{exc}".encode("utf-8", errors="replace"))}
        _emit({**failure, "receipt_sha256": _sha256_json(failure)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "APPROVAL_FRAME_MAGIC", "CAPABILITY_APPROVAL_INSTALL_RECEIPT_SCHEMA",
    "CAPABILITY_PLAN_PUBLICATION_AUTHORITY_SCHEMA",
    "CAPABILITY_PLAN_PUBLICATION_RECEIPT_SCHEMA",
    "CAPABILITY_ROUTEBACK_BOT_IDENTITY_SCHEMA",
    "CAPABILITY_PLAN_PUBLICATION_SCOPE",
    "CAPABILITY_APPROVAL_SCHEMA", "CAPABILITY_BROWSER_HOST_IDENTITY_SCHEMA",
    "CAPABILITY_BROWSER_IDENTITY_FOUNDATION_SCHEMA",
    "CAPABILITY_EXECUTION_HOST_IDENTITY_SCHEMA",
    "CAPABILITY_EXECUTION_IDENTITY_FOUNDATION_SCHEMA",
    "CAPABILITY_EXECUTION_READINESS_SCHEMA",
    "CAPABILITY_OBSERVER_HOOKS",
    "CAPABILITY_OBSERVER_PLUGIN", "CAPABILITY_PLAN_SCHEMA", "CODEX_FRAME_MAGIC",
    "CAPABILITY_PRODUCTION_OBSERVATION_WAIT_REQUEST_SCHEMA",
    "CAPABILITY_PRODUCTION_OBSERVATION_STAGE_RECEIPT_SCHEMA",
    "CONNECTOR_FRAME_MAGIC", "CapabilityCanaryLifecycle",
    "CapabilityCanaryOwnerApproval", "CapabilityCanaryPlan",
    "FIRST_WAVE_TOOLSETS", "MAC_OPS_FRAME_MAGIC",
    "RuntimeIdentities", "attest_capability_execution_readiness",
    "browser_host_identity_receipt",
    "browser_principal_version_smoke",
    "browser_service_runtime_preflight", "browser_userns_preflight",
    "build_capability_plan", "build_secret_lease_frame",
    "build_capability_stop_proof",
    "build_credential_consumer_stop_proof",
    "build_capability_cleanup_facts",
    "publish_capability_cleanup_facts",
    "build_capability_observer_stop_receipt",
    "build_capability_cleanup_finalization",
    "capability_browser_controller_client_config",
    "capability_browser_controller_client_mapping",
    "capability_gateway_effective_environment_is_sealed",
    "collect_capability_preflight", "load_capability_approval",
    "load_capability_plan", "install_capability_approval",
    "provision_secret_lease", "read_capability_approval", "read_secret_lease_frame",
    "render_browser_config", "render_browser_unit",
    "render_connector_config", "render_connector_unit", "render_gateway_config",
    "render_gateway_unit", "render_mac_ops_config", "render_mac_ops_unit",
    "render_worker_config", "render_worker_service_unit",
    "render_worker_socket_unit",
    "retire_secret_lease", "runtime_contract",
    "ensure_browser_identity_create_only",
    "ensure_execution_identities_create_only",
    "build_plan_from_publication_authority",
    "publish_capability_plan",
    "read_owner_signed_production_observation",
    "read_production_observation_wait_request",
    "read_plan_publication_authority",
    "validate_plan_publication_authority",
    "execution_host_identity_receipt",
    "stage_and_publish_owner_signed_production_observation",
    "wait_for_capability_production_observation_marker",
    "validate_capability_extension_surface", "validate_capability_gateway_config",
    "worker_executables_preflight", "worker_systemd252_preflight",
    "worker_tmpfs_runtime_preflight",
]

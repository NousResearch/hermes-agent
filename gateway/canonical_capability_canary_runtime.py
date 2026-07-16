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
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
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
from gateway.posix_identity import effective_gid, effective_uid
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


CAPABILITY_PLAN_SCHEMA = "muncho-production-capability-runtime-plan.v5"
CAPABILITY_CONTRACT_SCHEMA = "muncho-production-capability-runtime-contract.v2"
CAPABILITY_PREFLIGHT_SCHEMA = "muncho-production-capability-runtime-preflight.v3"
CAPABILITY_LEASE_FRAME_SCHEMA = "muncho-production-capability-secret-lease-frame.v1"
CAPABILITY_LEASE_INSTALL_INTENT_SCHEMA = (
    "muncho-production-capability-secret-install-intent.v1"
)
CAPABILITY_LEASE_INSTALL_ABORT_SCHEMA = (
    "muncho-production-capability-secret-install-abort.v1"
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
CAPABILITY_CLEANUP_FACTS_SCHEMA = "muncho-production-capability-cleanup-facts.v1"
CAPABILITY_OBSERVER_STOP_RECEIPT_SCHEMA = (
    "muncho-production-capability-observer-stop-receipt.v1"
)
CAPABILITY_CLEANUP_FINALIZATION_SCHEMA = (
    "muncho-production-capability-cleanup-finalization.v1"
)
CAPABILITY_CLEANUP_TRANSACTION_SCHEMA = (
    "muncho-production-capability-cleanup-transaction-checkpoint.v1"
)
CAPABILITY_PRODUCTION_OBSERVATION_SCHEMA = (
    "muncho-production-capability-production-observation.v1"
)
CAPABILITY_PRODUCTION_DIFF_SCHEMA = "muncho-production-capability-production-diff.v1"
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
CAPABILITY_LIFECYCLE_RECEIPT_SCHEMA = (
    "muncho-production-capability-runtime-lifecycle.v2"
)
CAPABILITY_GATEWAY_CORE_READY_STAGE = "gateway-core-ready"
CAPABILITY_API_INPUTS_PREPARED_STAGE = "api-inputs-prepared"
CAPABILITY_RUNTIME_PENDING_ACK_STAGE = (
    "runtime-live-pending-gateway-commit-ack"
)
CAPABILITY_GATEWAY_ACK_PRE_MODEL_STAGE = (
    "gateway-commit-acknowledged-pre-model"
)
CAPABILITY_LIFECYCLE_STAGES = frozenset({
    CAPABILITY_GATEWAY_CORE_READY_STAGE,
    CAPABILITY_API_INPUTS_PREPARED_STAGE,
    CAPABILITY_RUNTIME_PENDING_ACK_STAGE,
    CAPABILITY_GATEWAY_ACK_PRE_MODEL_STAGE,
    "core-started",  # read-only compatibility for pre-remediation receipts
    "started",  # non-admission lifecycle compatibility
    "stopped",
    "failure",
})
CAPABILITY_APPROVAL_SCHEMA = "muncho-production-capability-owner-approval.v2"
CAPABILITY_APPROVAL_INSTALL_RECEIPT_SCHEMA = (
    "muncho-production-capability-owner-approval-install.v2"
)
CAPABILITY_APPROVAL_RETIREMENT_RECEIPT_SCHEMA = (
    "muncho-production-capability-owner-approval-retirement.v1"
)
CAPABILITY_APPROVAL_RETIREMENT_INTENT_SCHEMA = (
    "muncho-production-capability-owner-approval-retirement-intent.v1"
)
CAPABILITY_BROWSER_HOST_IDENTITY_SCHEMA = (
    "muncho-production-capability-browser-host-identity.v2"
)
CAPABILITY_BROWSER_IDENTITY_FOUNDATION_SCHEMA = (
    "muncho-production-capability-browser-identity-foundation.v1"
)
CAPABILITY_EXECUTION_HOST_IDENTITY_SCHEMA = (
    "muncho-production-capability-execution-host-identity.v2"
)
CAPABILITY_SERVICE_HOST_IDENTITY_SCHEMA = (
    "muncho-production-capability-service-host-identity.v2"
)
CAPABILITY_SERVICE_IDENTITY_FOUNDATION_SCHEMA = (
    "muncho-production-capability-service-identity-foundation.v1"
)
CAPABILITY_EXECUTION_IDENTITY_FOUNDATION_SCHEMA = (
    "muncho-production-capability-execution-identity-foundation.v1"
)
CAPABILITY_EXECUTION_READINESS_SCHEMA = (
    "muncho-production-capability-execution-readiness.v1"
)
CAPABILITY_PLAN_PUBLICATION_AUTHORITY_SCHEMA = (
    "muncho-production-capability-plan-publication-authority.v2"
)
CAPABILITY_PLAN_PUBLICATION_RECEIPT_SCHEMA = (
    "muncho-production-capability-plan-publication-receipt.v2"
)
CAPABILITY_PLAN_INPUTS_SCHEMA = (
    "muncho-production-capability-plan-publication-inputs.v2"
)
CAPABILITY_FOUNDATION_AUTHORING_REQUEST_SCHEMA = (
    "muncho-production-capability-foundation-authoring-request.v1"
)
CAPABILITY_FOUNDATION_AUTHORING_CONTEXT_SCHEMA = (
    "muncho-production-capability-foundation-authoring-context.v2"
)
CAPABILITY_PLAN_AUTHORING_REQUEST_SCHEMA = (
    "muncho-production-capability-plan-authoring-request.v2"
)
CAPABILITY_PLAN_AUTHORING_CONTEXT_SCHEMA = (
    "muncho-production-capability-plan-authoring-context.v2"
)
FULL_CANARY_TERMINAL_RECEIPT_SCHEMA = (
    "muncho-full-canary-session-bound-owner-receipt.v1"
)
CAPABILITY_ROUTEBACK_BOT_IDENTITY_SCHEMA = (
    "muncho-production-capability-routeback-bot-identity.v1"
)
CAPABILITY_BITRIX_FOUNDATION_AUTHORITY_SCHEMA = (
    "muncho-production-capability-bitrix-foundation-authority.v2"
)
CAPABILITY_BITRIX_FOUNDATION_SCOPE = "production_capability_canary_bitrix_foundation"
CAPABILITY_BITRIX_IDENTITY_BOOTSTRAP_SCHEMA = (
    "muncho-production-capability-bitrix-identity-bootstrap.v1"
)
CAPABILITY_BITRIX_KEY_BOOTSTRAP_SCHEMA = (
    "muncho-production-capability-bitrix-key-bootstrap.v1"
)
CAPABILITY_BITRIX_KEY_AUTHORITY_INDEX_SCHEMA = (
    "muncho-production-capability-bitrix-key-authority-index.v1"
)
CAPABILITY_BITRIX_FOUNDATION_RECEIPT_SCHEMA = (
    "muncho-production-capability-bitrix-foundation.v2"
)
CAPABILITY_BITRIX_FOUNDATION_ABORT_SCHEMA = (
    "muncho-production-capability-bitrix-foundation-abort.v1"
)
CAPABILITY_BITRIX_FOUNDATION_KEY_STAGE_INTENT_SCHEMA = (
    "muncho-production-capability-bitrix-foundation-key-stage-intent.v1"
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
    "muncho-production-capability-expiry-watchdog-completion.v2"
)
CAPABILITY_EXPIRY_WATCHDOG_DISARM_INTENT_SCHEMA = (
    "muncho-production-capability-expiry-watchdog-disarm-intent.v1"
)
CAPABILITY_EXPIRY_WATCHDOG_DISARM_COMPLETION_SCHEMA = (
    "muncho-production-capability-expiry-watchdog-disarm-completion.v1"
)
CAPABILITY_EXPIRY_RECONCILIATION_SCHEMA = (
    "muncho-production-capability-expiry-cleanup-reconciliation.v2"
)
CAPABILITY_EXPIRY_ACTIVE_RUN_RETIREMENT_SCHEMA = (
    "muncho-production-capability-active-api-admission-retirement.v1"
)
CAPABILITY_PLAN_PUBLICATION_SCOPE = "production_capability_canary_plan_publication"
PRODUCTION_CANARY_PUBLIC_GUILD_ID = "1282725267068157972"
PRODUCTION_CANARY_PUBLIC_CHANNEL_ID = "1526858760100909066"
PRODUCTION_OWNER_USER_ID = "1279454038731264061"
DEFAULT_GOAL_OBSERVER_CONFIG = Path(
    "/etc/muncho/capability-canary/goal-observer.json"
)
DEFAULT_GOAL_COLLECTOR_SOCKET = Path(
    "/run/muncho-capability-goal/collector.sock"
)
LOCKED_NONPUBLIC_CHANNEL_IDS = SKYVISION_LOCKED_NONPUBLIC_CHANNEL_IDS

CAPABILITY_OBSERVER_PLUGIN = "muncho_canary_evidence"
CAPABILITY_OBSERVER_HOOKS = (
    "pre_api_request",
    "post_api_request",
    "post_tool_call",
    "on_session_start",
    "on_session_end",
)
_CAPABILITY_GATEWAY_CONFIG_KEYS = frozenset({
    "agent",
    "auxiliary",
    "browser",
    "canonical_brain",
    "compression",
    "context",
    "cron",
    "curator",
    "gateway",
    "goals",
    "hooks",
    "kanban",
    "mac_ops_edge",
    "memory",
    "model",
    "platform_toolsets",
    "platforms",
    "plugins",
    "terminal",
    "tool_loop_guardrails",
    "tools",
})

_CAPABILITY_MODEL_ROUTE = {
    "default": "gpt-5.6-sol",
    "provider": "openai-codex",
    "base_url": "https://chatgpt.com/backend-api/codex",
}
_CAPABILITY_AUXILIARY_ROUTE = {
    "provider": "openai-codex",
    "model": "gpt-5.6-sol",
    "base_url": "",
    "api_key": "",
    "fallback_chain": [],
    "extra_body": {},
}
_CAPABILITY_AUXILIARY_TASKS = (
    "vision",
    "web_extract",
    "compression",
    "skills_hub",
    "mcp",
    "title_generation",
    "tts_audio_tags",
    "profile_describer",
    "curator",
    "monitor",
    "background_review",
    "moa_reference",
    "moa_aggregator",
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
_MAX_LEASE_ARTIFACTS = 4096
_MAX_EXPIRY_WATCHDOGS = 4096

DEFAULT_PLAN_PATH = Path("/etc/muncho/capability-canary/runtime-plan.json")
DEFAULT_STAGED_FULL_CANARY_PLAN_PATH = Path(
    "/etc/muncho/full-canary/staged/runtime-plan.json"
)
DEFAULT_APPROVAL_PATH = Path("/etc/muncho/capability-canary/owner-approval.json")
DEFAULT_GATEWAY_CONFIG = Path("/etc/muncho/capability-canary/gateway.yaml")
DEFAULT_GATEWAY_HOME = Path("/var/lib/muncho-capability-canary")
DEFAULT_GATEWAY_PROFILE_HOME = DEFAULT_GATEWAY_HOME / ".hermes"
DEFAULT_GATEWAY_AUTH_STORE = DEFAULT_GATEWAY_PROFILE_HOME / "auth.json"
DEFAULT_GATEWAY_WORK_ROOT = DEFAULT_GATEWAY_HOME / "work"
DEFAULT_GATEWAY_LOG_ROOT = DEFAULT_GATEWAY_HOME / "logs"
DEFAULT_GATEWAY_RUNTIME = Path("/run/hermes-cloud-gateway")
DEFAULT_CONTROL_ROOT = Path("/var/lib/muncho-capability-canary-control")
DEFAULT_PLAN_PUBLICATION_RECEIPT_ROOT = DEFAULT_CONTROL_ROOT / "plan-publications"
DEFAULT_SERVICE_IDENTITY_FOUNDATION_ROOT = (
    DEFAULT_CONTROL_ROOT / "service-identity-foundations"
)
DEFAULT_APPROVAL_RECEIPT_ROOT = DEFAULT_CONTROL_ROOT / "approval-installs"
DEFAULT_CODEX_LEASE_JOURNAL = DEFAULT_CONTROL_ROOT / "codex-leases"
DEFAULT_MAC_OPS_LEASE_JOURNAL = DEFAULT_CONTROL_ROOT / "mac-ops-leases"
DEFAULT_API_CONTROL_LEASE_JOURNAL = DEFAULT_CONTROL_ROOT / "api-control-leases"
DEFAULT_ROUTEBACK_LEASE_JOURNAL = DEFAULT_CONTROL_ROOT / "routeback-leases"
DEFAULT_BITRIX_LEASE_JOURNAL = DEFAULT_CONTROL_ROOT / "bitrix-leases"
DEFAULT_BITRIX_WEBHOOK_PATH = Path(
    "/opt/adventico-ai-platform/hermes-home/secrets/bitrix_skyvision_crm_webhook.url"
)
BITRIX_OPERATIONAL_EDGE_UNIT = "muncho-operational-edge-bitrix.service"
DEFAULT_BITRIX_UNIT_PATH = Path("/etc/systemd/system") / BITRIX_OPERATIONAL_EDGE_UNIT
CAPABILITY_PRODUCER_UNIT_PATHS = {
    role: Path("/etc/systemd/system") / CAPABILITY_PRODUCER_SERVICE_UNITS[role]
    for role in CAPABILITY_PRODUCER_ROLES
}
DEFAULT_BITRIX_CONFIG_PATH = Path("/etc/muncho/operational-edge/bitrix.json")
DEFAULT_BITRIX_SOCKET_PATH = Path("/run/muncho-operational-edge/bitrix/edge.sock")
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
DEFAULT_BITRIX_KEY_BOOTSTRAP_ROOT = DEFAULT_CONTROL_ROOT / "bitrix-key-bootstraps"
DEFAULT_BITRIX_KEY_RETIREMENT_ROOT = DEFAULT_CONTROL_ROOT / "bitrix-key-retirements"
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
CAPABILITY_PLANNED_IDENTITIES = {
    "mac_ops_uid": 2104,
    "mac_ops_gid": 2205,
    "connector_uid": 2105,
    "connector_gid": 2206,
    "browser_uid": 2106,
    "browser_gid": 2207,
    "worker_uid": 2107,
    "worker_gid": 2208,
    "worker_client_gid": 2209,
    "bitrix_operational_edge_uid": 2108,
    "bitrix_operational_edge_gid": 2210,
    "bitrix_operational_edge_client_gid": 2211,
    "producer_business_edge_uid": 2109,
    "producer_business_edge_gid": 2212,
    "producer_canonical_writer_uid": 2110,
    "producer_canonical_writer_gid": 2213,
    "producer_discord_edge_uid": 2111,
    "producer_discord_edge_gid": 2214,
    "producer_gateway_observer_uid": 2112,
    "producer_gateway_observer_gid": 2215,
    "producer_receipt_writer_gid": 2216,
}
_LOOPBACK_DENY_DROP_IN_NAME = "50-muncho-capability-loopback-deny.conf"
DEFAULT_EDGE_LOOPBACK_DENY_DROP_IN = (
    DEFAULT_EDGE_UNIT_PATH.parent / f"{EDGE_UNIT_NAME}.d" / _LOOPBACK_DENY_DROP_IN_NAME
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
DEFAULT_CONNECTOR_CREDENTIAL_DIR = Path("/etc/muncho/discord-connector-credentials")
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
    "unprivileged_userns_clone": Path("/proc/sys/kernel/unprivileged_userns_clone"),
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
CAPABILITY_DEFERRED_CORE_START_ORDER = (
    PHASE_B_READINESS_UNIT_NAME,
    EDGE_UNIT_NAME,
    DEFAULT_DISCORD_CONNECTOR_UNIT,
    MAC_OPS_UNIT_NAME,
    DEFAULT_WORKER_SOCKET_UNIT_NAME,
    DEFAULT_WORKER_SERVICE_UNIT_NAME,
    DEFAULT_BROWSER_UNIT_NAME,
    WRITER_UNIT_NAME,
    BITRIX_OPERATIONAL_EDGE_UNIT,
    GATEWAY_UNIT_NAME,
)
CAPABILITY_ADMITTED_PRODUCER_START_ORDER = tuple(
    CAPABILITY_PRODUCER_SERVICE_UNITS[role] for role in CAPABILITY_PRODUCER_ROLES
)
CAPABILITY_DEFERRED_START_ORDER = (
    *CAPABILITY_DEFERRED_CORE_START_ORDER,
    *CAPABILITY_ADMITTED_PRODUCER_START_ORDER,
)
CAPABILITY_OBSERVER_ROLE = "gateway_observer"
CAPABILITY_OBSERVER_UNIT = CAPABILITY_PRODUCER_SERVICE_UNITS[CAPABILITY_OBSERVER_ROLE]
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

CAPABILITY_CLEANUP_TRANSACTION_STAGES = (
    (1, "facts_collected", "cleanup-transaction-01-facts-collected.json"),
    (2, "facts_published", "cleanup-transaction-02-facts-published.json"),
    (
        3,
        "signed_receipt_verified",
        "cleanup-transaction-03-signed-receipt-verified.json",
    ),
    (4, "observer_stopped", "cleanup-transaction-04-observer-stopped.json"),
    (5, "runtime_retired", "cleanup-transaction-05-runtime-retired.json"),
    (6, "watchdogs_disarmed", "cleanup-transaction-06-watchdogs-disarmed.json"),
    (7, "finalized", "cleanup-transaction-07-finalized.json"),
)

CAPABILITY_CREDENTIAL_BINDINGS = (
    "api_control",
    "bitrix_operational_edge_webhook",
    "discord_canonical_routeback_bot_token",
    "discord_public_session_bot_token",
    "mac_ops_gitlab",
    "openai_codex",
)

_DISCORD_CONNECTOR_OPERATION_CLASS = "ordinary_public_ingress_and_session_replies"
# This is a public Discord snowflake, never a credential or credential digest.
# The capability plan must bind two separate clean-canary applications and
# prove mechanically that neither can silently reuse the production bot.
PRODUCTION_DISCORD_BOT_USER_ID = "1501976597455044801"
_PINNED_RELAY_URL = f"unix://{DEFAULT_DISCORD_CONNECTOR_SOCKET}"
_ROUTEBACK_BOT_IDENTITY_TIMEOUT_SECONDS = 5.0
_MAX_ROUTEBACK_CREDENTIAL_BYTES = 512


def capability_browser_executable(release_root: Path) -> Path:
    """Return the same release-local Chrome-for-Testing layout as production."""

    return release_root / "ops/muncho/runtime/dependencies/chrome-linux64/chrome"


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
    return _sha256_json({
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
        "identity_bootstrap_receipt_sha256": (identity_bootstrap_receipt_sha256),
        "receipt_public_key_id": receipt_public_key_id,
        "key_bootstrap_receipt_sha256": key_bootstrap_receipt_sha256,
        "credential_binding": "bitrix_operational_edge_webhook",
    })


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
_PROCESSLESS_UNIT_PROPERTY_DEFAULTS = {
    DEFAULT_WORKER_SOCKET_UNIT_NAME: {
        "MainPID": "0",
        "Type": "",
        "NotifyAccess": "",
        "StatusText": "",
    },
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_IDENTITY_RE = re.compile(r"^[a-z_][a-z0-9_-]{0,63}$")
_LEASE_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_MAX_PLAN_BYTES = 2 * 1024 * 1024
_MAX_SECRET_BYTES = 64 * 1024
_MAX_AUTH_STORE_BYTES = 256 * 1024
_MAX_LEASE_SECONDS = 1_200
CAPABILITY_MUTATION_MIN_RESERVE_SECONDS = 60
CAPABILITY_ACTIVE_USE_MIN_RESERVE_SECONDS = 30
_MAX_EXPIRY_RECONCILIATIONS = 256


class CapabilityLeaseReserveError(RuntimeError):
    """The lease cannot be safely committed for long enough to be used."""


class _OperationClock:
    """Injectable, non-regressing wall clock for a multi-step mutation."""

    def __init__(self, clock: Callable[[], int] | None = None) -> None:
        self._clock = clock or (lambda: int(time.time()))
        self._last: int | None = None

    def sample(self, label: str) -> int:
        value = self._clock()
        if type(value) is not int or value < 0:
            raise RuntimeError(f"{label} clock value is invalid")
        if self._last is not None and value < self._last:
            raise RuntimeError(f"{label} clock regressed")
        self._last = value
        return value


def _require_remaining_reserve(
    *,
    expires_at_unix: int,
    now_unix: int,
    minimum_seconds: int = CAPABILITY_MUTATION_MIN_RESERVE_SECONDS,
) -> None:
    if (
        type(expires_at_unix) is not int
        or type(now_unix) is not int
        or type(minimum_seconds) is not int
        or minimum_seconds < 1
        or not now_unix < expires_at_unix
        or expires_at_unix - now_unix < minimum_seconds
    ):
        raise CapabilityLeaseReserveError(
            "capability mutation lacks the minimum expiry reserve"
        )


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
_CAPABILITY_GATEWAY_UNSET_ENVIRONMENT_DIRECTIVE = "UnsetEnvironment=" + " ".join(
    _CAPABILITY_GATEWAY_UNSET_ENVIRONMENT_NAMES
)


def _strict_mapping(
    value: Any, fields: set[str] | frozenset[str], label: str
) -> Mapping[str, Any]:
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
    if (
        not path.is_absolute()
        or path != Path(os.path.normpath(value))
        or ".." in path.parts
    ):
        raise ValueError(f"{label} must be an absolute normalized path")
    return path


_FULL_CANARY_TERMINAL_RECEIPT_FIELDS = frozenset({
    "schema",
    "ok",
    "state",
    "release_sha",
    "coordinator_input_sha256",
    "full_canary_plan_sha256",
    "owner_approval_sha256",
    "phase_b_readiness_anchor_sha256",
    "api_session_key_sha256",
    "fixture_sha256",
    "discord_token_install_receipt_sha256",
    "coordinator_receipt_sha256",
    "live_driver_receipt_sha256",
    "services_stopped",
    "discord_token_retired",
    "temporary_admin_created",
    "bootstrap_credential_created",
    "completed_at_unix",
    "receipt_sha256",
})


def validate_full_canary_terminal_receipt(
    value: Any,
    *,
    revision: str | None = None,
    full_canary_plan_sha256: str | None = None,
) -> Mapping[str, Any]:
    """Validate the canonical, terminal stopped truth inherited by capability.

    This is deliberately the complete non-secret owner receipt, not a caller
    projection.  Every downstream capability authority and receipt carries the
    same mapping and its self-digest.
    """

    raw = _strict_mapping(
        value,
        _FULL_CANARY_TERMINAL_RECEIPT_FIELDS,
        "full-canary terminal receipt",
    )
    unsigned = {
        name: copy.deepcopy(item)
        for name, item in raw.items()
        if name != "receipt_sha256"
    }
    if (
        raw["schema"] != FULL_CANARY_TERMINAL_RECEIPT_SCHEMA
        or raw["ok"] is not True
        or raw["state"] != "verified_stopped_and_credentials_retired"
        or not isinstance(raw["release_sha"], str)
        or _REVISION_RE.fullmatch(raw["release_sha"]) is None
        or revision is not None
        and raw["release_sha"] != revision
        or full_canary_plan_sha256 is not None
        and raw["full_canary_plan_sha256"] != full_canary_plan_sha256
        or raw["services_stopped"] is not True
        or raw["discord_token_retired"] is not True
        or raw["temporary_admin_created"] is not False
        or raw["bootstrap_credential_created"] is not False
        or type(raw["completed_at_unix"]) is not int
        or raw["completed_at_unix"] < 0
        or raw["receipt_sha256"] != _sha256_json(unsigned)
    ):
        raise ValueError("full-canary terminal receipt is invalid")
    for name in _FULL_CANARY_TERMINAL_RECEIPT_FIELDS - {
        "schema",
        "ok",
        "state",
        "release_sha",
        "services_stopped",
        "discord_token_retired",
        "temporary_admin_created",
        "bootstrap_credential_created",
        "completed_at_unix",
    }:
        _digest(raw[name], f"full-canary terminal receipt {name}")
    return copy.deepcopy(dict(raw))


def _terminal_receipt_binding(
    value: Any,
    sha256: Any,
    *,
    revision: str | None = None,
    full_canary_plan_sha256: str | None = None,
) -> tuple[Mapping[str, Any], str]:
    terminal = validate_full_canary_terminal_receipt(
        value,
        revision=revision,
        full_canary_plan_sha256=full_canary_plan_sha256,
    )
    digest = _digest(sha256, "full-canary terminal receipt")
    if digest != terminal["receipt_sha256"] or digest != _sha256_json({
        name: copy.deepcopy(item)
        for name, item in terminal.items()
        if name != "receipt_sha256"
    }):
        raise ValueError("full-canary terminal receipt digest drifted")
    return terminal, digest


def _read_staged_full_canary_plan(
    path: Path = DEFAULT_STAGED_FULL_CANARY_PLAN_PATH,
) -> tuple[FullCanaryPlan, bytes, Mapping[str, Any]]:
    if path != DEFAULT_STAGED_FULL_CANARY_PLAN_PATH:
        raise ValueError("staged full-canary plan path is fixed")
    raw, item = _read_stable_file(
        path,
        maximum=_MAX_PLAN_BYTES,
        expected_uid=0,
        expected_gid=0,
        allowed_modes=frozenset({0o400}),
    )
    value = _decode_json(raw, label="staged full-canary plan")
    if raw != _canonical_bytes(value):
        raise RuntimeError("staged full-canary plan is not canonical")
    plan = FullCanaryPlan.from_mapping(value)
    identity = {
        "device": item.st_dev,
        "inode": item.st_ino,
        "uid": item.st_uid,
        "gid": item.st_gid,
        "mode": f"{stat.S_IMODE(item.st_mode):04o}",
        "size": item.st_size,
        "mtime_ns": item.st_mtime_ns,
    }
    return plan, raw, identity


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
        raw = _strict_mapping(
            value, set(cls.__dataclass_fields__), "capability identities"
        )
        result = cls(**{
            key: (
                _positive_id(raw[key], key)
                if key.endswith(("_uid", "_gid"))
                else _identity(raw[key], key)
            )
            for key in cls.__dataclass_fields__
        })
        if (
            result.mac_ops_user != "muncho-mac-ops-edge"
            or result.mac_ops_group != "muncho-mac-ops-edge"
            or result.connector_user != "muncho-discord-connector"
            or result.connector_group != "muncho-discord-connector"
            or result.bitrix_operational_edge_user != "muncho-edge-bitrix"
            or result.bitrix_operational_edge_group != "muncho-edge-bitrix"
            or result.bitrix_operational_edge_client_group != "muncho-edge-bitrix-c"
            or result.browser_user != DEFAULT_BROWSER_USER
            or result.browser_group != DEFAULT_BROWSER_GROUP
            or result.worker_user != DEFAULT_WORKER_USER
            or result.worker_group != DEFAULT_WORKER_GROUP
            or result.worker_client_group != DEFAULT_WORKER_CLIENT_GROUP
        ):
            raise ValueError("capability service identities are not pinned")
        if (
            len({
                result.gateway_user,
                result.mac_ops_user,
                result.connector_user,
                result.bitrix_operational_edge_user,
                result.browser_user,
                result.worker_user,
            })
            != 6
            or len({
                result.gateway_group,
                result.mac_ops_group,
                result.connector_group,
                result.bitrix_operational_edge_group,
                result.bitrix_operational_edge_client_group,
                result.browser_group,
                result.worker_group,
                result.worker_client_group,
            })
            != 8
        ):
            raise ValueError("capability service identity names are not isolated")
        if (
            len({
                result.gateway_uid,
                result.mac_ops_uid,
                result.connector_uid,
                result.bitrix_operational_edge_uid,
                result.browser_uid,
                result.worker_uid,
            })
            != 6
        ):
            raise ValueError("capability service identities are not isolated")
        if (
            len({
                result.gateway_gid,
                result.socket_client_gid,
                result.mac_ops_gid,
                result.connector_gid,
                result.bitrix_operational_edge_gid,
                result.bitrix_operational_edge_client_gid,
                result.browser_gid,
                result.worker_gid,
                result.worker_client_gid,
            })
            != 9
        ):
            raise ValueError("capability execution group identities are not isolated")
        return result

    def to_mapping(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class CapabilityCanaryPlan:
    revision: str
    full_canary_plan_sha256: str
    full_canary_terminal_receipt: Mapping[str, Any]
    full_canary_terminal_receipt_sha256: str
    original_full_canary_owner_approval_sha256: str
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
            "schema",
            "revision",
            "full_canary_plan_sha256",
            "full_canary_terminal_receipt",
            "full_canary_terminal_receipt_sha256",
            "original_full_canary_owner_approval_sha256",
            "release",
            "identities",
            "isolated_worker",
            "browser",
            "execution_workspace",
            "toolsets",
            "api_loopback",
            "mac_ops",
            "bitrix_operational_edge",
            "discord_connector",
            "artifacts",
            "credential_bindings",
            "capability_plan_sha256",
        }
        raw = _strict_mapping(value, fields, "capability plan")
        revision = raw["revision"]
        if (
            raw["schema"] != CAPABILITY_PLAN_SCHEMA
            or not isinstance(revision, str)
            or _REVISION_RE.fullmatch(revision) is None
        ):
            raise ValueError("capability plan identity is invalid")
        terminal, terminal_sha256 = _terminal_receipt_binding(
            raw["full_canary_terminal_receipt"],
            raw["full_canary_terminal_receipt_sha256"],
            revision=revision,
            full_canary_plan_sha256=raw["full_canary_plan_sha256"],
        )
        original_full_approval_sha256 = _digest(
            raw["original_full_canary_owner_approval_sha256"],
            "original full-canary owner approval",
        )
        if original_full_approval_sha256 != terminal["owner_approval_sha256"]:
            raise ValueError("original full-canary owner approval binding drifted")
        release = _strict_mapping(
            raw["release"],
            {"artifact_root", "artifact_sha256", "interpreter"},
            "capability release",
        )
        root = _absolute(release["artifact_root"], "release root")
        interpreter = _absolute(release["interpreter"], "release interpreter")
        if (
            root != Path("/opt/muncho-canary-releases") / revision
            or interpreter != root / "venv/bin/python"
        ):
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
            or worker["gateway_ready_probe_contract"] != GATEWAY_READY_PROBE_CONTRACT
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
            item = _strict_mapping(browser[name], {"path", "sha256"}, f"browser {name}")
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
        if raw["api_loopback"] != {
            "host": "127.0.0.1",
            "port": 8642,
            "key_credential": API_SERVER_CREDENTIAL_NAME,
        }:
            raise ValueError("capability API boundary is not exact")
        mac = _strict_mapping(
            raw["mac_ops"],
            {
                "service_unit",
                "socket_path",
                "credential_path",
                "journal_path",
                "service_identity_sha256",
            },
            "Mac operations edge",
        )
        if (
            mac["service_unit"] != MAC_OPS_UNIT_NAME
            or mac["socket_path"] != str(DEFAULT_MAC_OPS_SOCKET)
            or mac["credential_path"] != str(DEFAULT_MAC_OPS_CREDENTIAL)
            or mac["journal_path"] != str(DEFAULT_MAC_OPS_JOURNAL)
        ):
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
            identity_bootstrap_receipt_sha256=(bitrix_identity_receipt_sha256),
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
                "private_source_path": str(DEFAULT_BITRIX_RECEIPT_PRIVATE_KEY_PATH),
                "private_projection_path": str(
                    DEFAULT_BITRIX_RECEIPT_PRIVATE_KEY_PROJECTION_PATH
                ),
                "private_owner_uid": 0,
                "private_owner_gid": 0,
                "private_mode": "0400",
                "public_path": str(DEFAULT_BITRIX_TRUST_PATH),
                "public_key_id": bitrix_public_key_id,
                "public_trust_sha256": bitrix_digests["rendered_trust_sha256"],
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
                "socket_client_gid": (identities.bitrix_operational_edge_client_gid),
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
            or bitrix_digests["service_identity_sha256"] != expected_bitrix_identity
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
            or connector["operation_class"] != _DISCORD_CONNECTOR_OPERATION_CLASS
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
            connector["production_bot_user_id"] != PRODUCTION_DISCORD_BOT_USER_ID
            or len({
                connector_bot_user_id,
                routeback_bot_user_id,
                PRODUCTION_DISCORD_BOT_USER_ID,
            })
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
        unsigned = {
            key: copy.deepcopy(item)
            for key, item in raw.items()
            if key != "capability_plan_sha256"
        }
        digest = _digest(raw["capability_plan_sha256"], "capability plan")
        if _sha256_json(unsigned) != digest:
            raise ValueError("capability plan self-digest drifted")
        result = cls(
            revision=revision,
            full_canary_plan_sha256=_digest(
                raw["full_canary_plan_sha256"], "full canary plan"
            ),
            full_canary_terminal_receipt=terminal,
            full_canary_terminal_receipt_sha256=terminal_sha256,
            original_full_canary_owner_approval_sha256=(original_full_approval_sha256),
            release_artifact_sha256=_digest(
                release["artifact_sha256"], "release artifact"
            ),
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
            worker_bwrap_sha256=_digest(worker["bwrap_sha256"], "worker bwrap"),
            worker_shell_sha256=_digest(worker["shell_sha256"], "worker shell"),
            connector_bot_user_id=connector_bot_user_id,
            routeback_bot_user_id=routeback_bot_user_id,
            connector_allowed_guild_ids=allowed_guilds,
            connector_allowed_channel_ids=allowed_channels,
            connector_allowed_user_ids=allowed_users,
            mac_ops_service_identity_sha256=_digest(
                mac["service_identity_sha256"], "Mac operations service identity"
            ),
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
            bitrix_operational_edge_credential_binding=bitrix["credential_binding"],
            gateway_config_sha256=_digest(
                artifacts["gateway_config_sha256"], "gateway config"
            ),
            gateway_unit_sha256=_digest(
                artifacts["gateway_unit_sha256"], "gateway unit"
            ),
            mac_ops_config_sha256=_digest(
                artifacts["mac_ops_config_sha256"], "Mac operations config"
            ),
            mac_ops_unit_sha256=_digest(
                artifacts["mac_ops_unit_sha256"], "Mac operations unit"
            ),
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
            "full_canary_terminal_receipt": copy.deepcopy(
                dict(self.full_canary_terminal_receipt)
            ),
            "full_canary_terminal_receipt_sha256": (
                self.full_canary_terminal_receipt_sha256
            ),
            "original_full_canary_owner_approval_sha256": (
                self.original_full_canary_owner_approval_sha256
            ),
            "release": {
                "artifact_root": str(self.release_root),
                "artifact_sha256": self.release_artifact_sha256,
                "interpreter": str(self.interpreter),
            },
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
            "api_loopback": {
                "host": "127.0.0.1",
                "port": 8642,
                "key_credential": API_SERVER_CREDENTIAL_NAME,
            },
            "mac_ops": {
                "service_unit": MAC_OPS_UNIT_NAME,
                "socket_path": str(DEFAULT_MAC_OPS_SOCKET),
                "credential_path": str(DEFAULT_MAC_OPS_CREDENTIAL),
                "journal_path": str(DEFAULT_MAC_OPS_JOURNAL),
                "service_identity_sha256": self.mac_ops_service_identity_sha256,
            },
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
                    "projected_path": str(DEFAULT_BITRIX_WEBHOOK_PROJECTION_PATH),
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
                    "private_source_path": str(DEFAULT_BITRIX_RECEIPT_PRIVATE_KEY_PATH),
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
                "credential_binding": (self.bitrix_operational_edge_credential_binding),
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
                "loopback_deny_drop_in_sha256": (self.loopback_deny_drop_in_sha256),
            },
        }

    def to_mapping(self) -> dict[str, Any]:
        return {**self.unsigned_mapping(), "capability_plan_sha256": self.sha256}

    def validate_derived_artifacts(self) -> None:
        if (
            self.bitrix_operational_edge_revision != self.revision
            or self.bitrix_operational_edge_service_unit != BITRIX_OPERATIONAL_EDGE_UNIT
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
        if (
            mac_identity != self.mac_ops_service_identity_sha256
            or mac_identity != self.mac_ops_unit_sha256
        ):
            raise ValueError("Mac operations unit identity drifted")
        if (
            _sha256_bytes(render_gateway_config(self)) != self.gateway_config_sha256
            or _sha256_bytes(render_gateway_unit(self).encode("utf-8"))
            != self.gateway_unit_sha256
            or _sha256_bytes(render_mac_ops_config(self)) != self.mac_ops_config_sha256
        ):
            raise ValueError("capability derived artifacts drifted")
        if (
            _sha256_bytes(render_worker_config(self)) != self.worker_config_sha256
            or _sha256_bytes(render_worker_socket_unit(self).encode("ascii"))
            != self.worker_socket_unit_sha256
            or _sha256_bytes(render_worker_service_unit(self).encode("ascii"))
            != self.worker_service_unit_sha256
        ):
            raise ValueError("capability isolated worker artifacts drifted")
        if _sha256_bytes(render_browser_config(self)) != self.browser_config_sha256:
            raise ValueError("capability browser controller config drifted")
        if (
            _sha256_bytes(render_browser_unit(self).encode("utf-8"))
            != self.browser_unit_sha256
        ):
            raise ValueError("capability browser unit drifted")
        if (
            _sha256_bytes(render_connector_config(self)) != self.connector_config_sha256
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
                "full_canary_terminal_receipt",
                "full_canary_terminal_receipt_sha256",
                "original_full_canary_owner_approval_sha256",
                "plan_publication_receipt_sha256",
                "authority_kind",
                "cryptographic_owner_proof",
                "owner_subject_sha256",
                "approval_source_sha256",
                "stopped_preflight_state_sha256",
                "stopped_preflight_report_sha256",
                "stopped_preflight_observed_at_unix",
                "fixture_sha256",
                "fixture_publication_receipt_sha256",
                "lease_install_receipt_sha256_by_binding",
                "bitrix_expiry_watchdog_authority_sha256",
                "approval_not_after_unix",
                "nonce_sha256",
                "approved_at_unix",
                "expires_at_unix",
            },
            "capability-canary owner approval",
        )
        if (
            raw["schema"] != CAPABILITY_APPROVAL_SCHEMA
            or raw["scope"] != "production_capability_canary_runtime_start"
            or raw["authority_kind"] != "trusted_root_bootstrap_out_of_band_owner"
            or raw["cryptographic_owner_proof"] is not False
        ):
            raise ValueError("capability-canary owner approval is invalid")
        for field in (
            "plan_sha256",
            "full_canary_plan_sha256",
            "full_canary_terminal_receipt_sha256",
            "original_full_canary_owner_approval_sha256",
            "plan_publication_receipt_sha256",
            "owner_subject_sha256",
            "approval_source_sha256",
            "stopped_preflight_state_sha256",
            "stopped_preflight_report_sha256",
            "fixture_sha256",
            "fixture_publication_receipt_sha256",
            "bitrix_expiry_watchdog_authority_sha256",
            "nonce_sha256",
        ):
            _digest(raw[field], f"capability approval {field}")
        terminal, terminal_sha256 = _terminal_receipt_binding(
            raw["full_canary_terminal_receipt"],
            raw["full_canary_terminal_receipt_sha256"],
            full_canary_plan_sha256=raw["full_canary_plan_sha256"],
        )
        if (
            raw["original_full_canary_owner_approval_sha256"]
            != terminal["owner_approval_sha256"]
            or terminal_sha256 != raw["full_canary_terminal_receipt_sha256"]
        ):
            raise ValueError("capability approval terminal binding drifted")
        lease_receipts = _strict_mapping(
            raw["lease_install_receipt_sha256_by_binding"],
            set(CAPABILITY_CREDENTIAL_BINDINGS),
            "capability approval lease receipts",
        )
        for binding, digest in lease_receipts.items():
            _digest(digest, f"capability approval {binding} lease receipt")
        approved = raw["approved_at_unix"]
        expires = raw["expires_at_unix"]
        observed = raw["stopped_preflight_observed_at_unix"]
        not_after = raw["approval_not_after_unix"]
        if (
            type(approved) is not int
            or type(expires) is not int
            or type(observed) is not int
            or type(not_after) is not int
            or approved < 0
            or observed > approved
            or approved > observed + 30
            or expires != not_after
            or not 30 <= expires - approved <= 900
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
            or self.value["full_canary_plan_sha256"] != full_canary_plan_sha256
            or type(now_unix) is not int
            or not self.value["approved_at_unix"]
            <= now_unix
            < self.value["expires_at_unix"]
        ):
            raise PermissionError(
                "owner approval does not authorize this capability canary"
            )

    @property
    def sha256(self) -> str:
        return _sha256_json(self.value)


def _require_capability_approval_preflight_binding(
    plan: CapabilityCanaryPlan,
    full_plan: FullCanaryPlan,
    approval: CapabilityCanaryOwnerApproval,
    preflight: Mapping[str, Any],
) -> None:
    """Require every fresh stopped-preflight authority input exactly."""

    if (
        preflight.get("schema") != CAPABILITY_PREFLIGHT_SCHEMA
        or preflight.get("phase") != "stopped"
        or preflight.get("ok") is not True
        or approval.value["plan_sha256"] != plan.sha256
        or approval.value["full_canary_plan_sha256"] != full_plan.sha256
        or approval.value["full_canary_terminal_receipt"]
        != plan.full_canary_terminal_receipt
        or approval.value["full_canary_terminal_receipt_sha256"]
        != plan.full_canary_terminal_receipt_sha256
        or approval.value["original_full_canary_owner_approval_sha256"]
        != plan.original_full_canary_owner_approval_sha256
        or approval.value["plan_publication_receipt_sha256"]
        != preflight.get("plan_publication_receipt_sha256")
        or approval.value["stopped_preflight_state_sha256"]
        != preflight.get("state_sha256")
        or approval.value["stopped_preflight_report_sha256"]
        != preflight.get("report_sha256")
        or approval.value["stopped_preflight_observed_at_unix"]
        != preflight.get("observed_at_unix")
        or approval.value["fixture_sha256"] != preflight.get("fixture_sha256")
        or approval.value["fixture_publication_receipt_sha256"]
        != preflight.get("fixture_publication_receipt_sha256")
        or approval.value["lease_install_receipt_sha256_by_binding"]
        != preflight.get("lease_install_receipt_sha256_by_binding")
        or approval.value["bitrix_expiry_watchdog_authority_sha256"]
        != preflight.get("bitrix_expiry_watchdog_authority_sha256")
        or approval.value["approval_not_after_unix"]
        != preflight.get("approval_not_after_unix")
        or approval.value["expires_at_unix"] != preflight.get("approval_not_after_unix")
    ):
        raise PermissionError(
            "owner approval does not bind the complete fresh capability preflight"
        )


def _capability_approval_chain_fields(
    approval: CapabilityCanaryOwnerApproval,
) -> Mapping[str, Any]:
    return {
        "capability_owner_approval_sha256": approval.sha256,
        "plan_publication_receipt_sha256": approval.value[
            "plan_publication_receipt_sha256"
        ],
        "stopped_preflight_state_sha256": approval.value[
            "stopped_preflight_state_sha256"
        ],
        "stopped_preflight_report_sha256": approval.value[
            "stopped_preflight_report_sha256"
        ],
        "stopped_preflight_observed_at_unix": approval.value[
            "stopped_preflight_observed_at_unix"
        ],
        "fixture_sha256": approval.value["fixture_sha256"],
        "fixture_publication_receipt_sha256": approval.value[
            "fixture_publication_receipt_sha256"
        ],
        "lease_install_receipt_sha256_by_binding": copy.deepcopy(
            dict(approval.value["lease_install_receipt_sha256_by_binding"])
        ),
        "bitrix_expiry_watchdog_authority_sha256": approval.value[
            "bitrix_expiry_watchdog_authority_sha256"
        ],
        "approval_not_after_unix": approval.value["approval_not_after_unix"],
    }


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


def _approval_retirement_receipt_path(
    plan: CapabilityCanaryPlan,
    approval: CapabilityCanaryOwnerApproval,
) -> Path:
    return (
        DEFAULT_APPROVAL_RECEIPT_ROOT
        / plan.revision
        / plan.sha256
        / f"{approval.value['nonce_sha256']}.retirement.json"
    )


def _approval_retirement_intent_path(
    plan: CapabilityCanaryPlan,
    approval: CapabilityCanaryOwnerApproval,
) -> Path:
    return (
        DEFAULT_APPROVAL_RECEIPT_ROOT
        / plan.revision
        / plan.sha256
        / f"{approval.value['nonce_sha256']}.retirement-intent.json"
    )


def _require_approval_reserve(
    approval: CapabilityCanaryOwnerApproval,
    plan: CapabilityCanaryPlan,
    full_plan: FullCanaryPlan,
    *,
    now_unix: int,
    minimum_reserve_seconds: int = CAPABILITY_ACTIVE_USE_MIN_RESERVE_SECONDS,
) -> None:
    approval.require(
        plan_sha256=plan.sha256,
        full_canary_plan_sha256=full_plan.sha256,
        now_unix=now_unix,
    )
    if approval.value["expires_at_unix"] - now_unix < minimum_reserve_seconds:
        raise PermissionError(
            "owner approval lacks the minimum reserve for a capability mutation"
        )


def _read_exact_installed_approval(
    plan: CapabilityCanaryPlan,
    full_plan: FullCanaryPlan,
    *,
    expected: CapabilityCanaryOwnerApproval | None = None,
) -> tuple[CapabilityCanaryOwnerApproval, os.stat_result, bytes]:
    raw, target = _read_stable_file(
        DEFAULT_APPROVAL_PATH,
        maximum=64 * 1024,
        expected_uid=0,
        expected_gid=0,
        allowed_modes=frozenset({0o400}),
    )
    value = _decode_json(raw, label="installed capability approval")
    if raw != _canonical_bytes(value):
        raise RuntimeError("installed capability approval is not canonical")
    approval = CapabilityCanaryOwnerApproval.from_mapping(value)
    if (
        approval.value["plan_sha256"] != plan.sha256
        or approval.value["full_canary_plan_sha256"] != full_plan.sha256
        or (expected is not None and approval.value != expected.value)
    ):
        raise RuntimeError("installed capability approval binding drifted")
    return approval, target, raw


def _build_approval_install_receipt(
    plan: CapabilityCanaryPlan,
    full_plan: FullCanaryPlan,
    approval: CapabilityCanaryOwnerApproval,
    *,
    target: os.stat_result,
    receipt_path: Path,
) -> Mapping[str, Any]:
    unsigned = {
        "schema": CAPABILITY_APPROVAL_INSTALL_RECEIPT_SCHEMA,
        "operation": "install_capability_owner_approval",
        "revision": plan.revision,
        "plan_sha256": plan.sha256,
        "full_canary_plan_sha256": full_plan.sha256,
        "full_canary_terminal_receipt": copy.deepcopy(
            dict(plan.full_canary_terminal_receipt)
        ),
        "full_canary_terminal_receipt_sha256": (
            plan.full_canary_terminal_receipt_sha256
        ),
        "original_full_canary_owner_approval_sha256": (
            plan.original_full_canary_owner_approval_sha256
        ),
        "plan_publication_receipt_sha256": approval.value[
            "plan_publication_receipt_sha256"
        ],
        "approval_sha256": approval.sha256,
        "owner_subject_sha256": approval.value["owner_subject_sha256"],
        "approval_source_sha256": approval.value["approval_source_sha256"],
        "stopped_preflight_state_sha256": approval.value[
            "stopped_preflight_state_sha256"
        ],
        "stopped_preflight_report_sha256": approval.value[
            "stopped_preflight_report_sha256"
        ],
        "stopped_preflight_observed_at_unix": approval.value[
            "stopped_preflight_observed_at_unix"
        ],
        "fixture_sha256": approval.value["fixture_sha256"],
        "fixture_publication_receipt_sha256": approval.value[
            "fixture_publication_receipt_sha256"
        ],
        "lease_install_receipt_sha256_by_binding": copy.deepcopy(
            dict(approval.value["lease_install_receipt_sha256_by_binding"])
        ),
        "bitrix_expiry_watchdog_authority_sha256": approval.value[
            "bitrix_expiry_watchdog_authority_sha256"
        ],
        "approval_not_after_unix": approval.value["approval_not_after_unix"],
        "nonce_sha256": approval.value["nonce_sha256"],
        "approved_at_unix": approval.value["approved_at_unix"],
        "expires_at_unix": approval.value["expires_at_unix"],
        "target_path": str(DEFAULT_APPROVAL_PATH),
        "target_device": target.st_dev,
        "target_inode": target.st_ino,
        "target_uid": target.st_uid,
        "target_gid": target.st_gid,
        "target_mode": f"{stat.S_IMODE(target.st_mode):04o}",
        "installed_at_unix": target.st_mtime_ns // 1_000_000_000,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    return {**unsigned, "receipt_sha256": _sha256_json(unsigned)}


def _validate_approval_install_receipt(
    value: Any,
    *,
    plan: CapabilityCanaryPlan,
    full_plan: FullCanaryPlan,
    approval: CapabilityCanaryOwnerApproval,
    target: os.stat_result,
    receipt_path: Path,
) -> Mapping[str, Any]:
    receipt = _strict_mapping(
        value,
        set(
            _build_approval_install_receipt(
                plan,
                full_plan,
                approval,
                target=target,
                receipt_path=receipt_path,
            )
        ),
        "capability approval install receipt",
    )
    unsigned = {
        key: copy.deepcopy(item)
        for key, item in receipt.items()
        if key != "receipt_sha256"
    }
    expected = _build_approval_install_receipt(
        plan,
        full_plan,
        approval,
        target=target,
        receipt_path=receipt_path,
    )
    # The original publication time remains authoritative for an existing
    # receipt; all binding, identity, and digest fields must otherwise match.
    expected = dict(expected)
    expected["installed_at_unix"] = receipt.get("installed_at_unix")
    expected_unsigned = {
        key: copy.deepcopy(item)
        for key, item in expected.items()
        if key != "receipt_sha256"
    }
    expected["receipt_sha256"] = _sha256_json(expected_unsigned)
    if (
        type(receipt.get("installed_at_unix")) is not int
        or receipt["installed_at_unix"] < 0
        or not approval.value["approved_at_unix"]
        <= receipt["installed_at_unix"]
        < approval.value["expires_at_unix"]
        or receipt != expected
        or receipt["receipt_sha256"] != _sha256_json(unsigned)
    ):
        raise RuntimeError("capability approval install receipt drifted")
    return copy.deepcopy(dict(receipt))


def install_capability_approval(
    plan: CapabilityCanaryPlan,
    full_plan: FullCanaryPlan,
    approval: CapabilityCanaryOwnerApproval,
    *,
    runner: Callable[[Command], subprocess.CompletedProcess[bytes]] = _runner,
    metadata_reader: Callable[[str], bytes | str] | None = None,
    local_identity_reader: Callable[[str], bytes | str] | None = None,
    clock: Callable[[], int] | None = None,
) -> Mapping[str, Any]:
    """Install one fresh, exact approval without overwrite or replay."""

    _require_root_linux()
    if not isinstance(approval, CapabilityCanaryOwnerApproval):
        raise TypeError("capability owner approval is required")
    validate_plan_against_full(plan, full_plan)
    operation_clock = _OperationClock(clock)
    validate_dedicated_canary_host(
        full_plan,
        metadata_reader=metadata_reader,
        local_identity_reader=local_identity_reader,
    )
    _validate_release_manifest(full_plan)
    with _lifecycle_lock():
        # A process death can leave a durable retirement intent after the
        # exact approval inode was already unlinked but before its completion
        # receipt became reachable.  Reconcile that bounded half-state before
        # admitting a later nonce.  When the target still exists the same
        # helper either retires the exact intent-bound inode or fails closed on
        # substitution; an unrelated approval can never step over it.
        retirement_histories = _approval_retirement_histories(plan, full_plan)
        if any(history[3] is None for history in retirement_histories):
            _remove_installed_capability_approval(plan, full_plan)
        receipt_path = _approval_install_receipt_path(plan, approval)
        payload = _canonical_bytes(approval.value)
        approval_exists = os.path.lexists(DEFAULT_APPROVAL_PATH)
        receipt_exists = os.path.lexists(receipt_path)
        if receipt_exists and not approval_exists:
            raise PermissionError(
                "capability owner approval nonce was already consumed"
            )
        if approval_exists:
            # The former direct-to-final O_EXCL publisher could be killed
            # after making a strict canonical prefix reachable.  Reconcile
            # only that exact, provenance-valid prefix before decoding it.
            _write_exclusive_bytes(DEFAULT_APPROVAL_PATH, payload, mode=0o400)
            installed, target, installed_payload = _read_exact_installed_approval(
                plan,
                full_plan,
                expected=approval,
            )
            if receipt_exists:
                expected_receipt = _build_approval_install_receipt(
                    plan,
                    full_plan,
                    installed,
                    target=target,
                    receipt_path=receipt_path,
                )
                _write_exclusive_bytes(
                    receipt_path,
                    _canonical_bytes(expected_receipt),
                    mode=0o400,
                )
                receipt_raw, _ = _read_stable_file(
                    receipt_path,
                    maximum=64 * 1024,
                    expected_uid=0,
                    expected_gid=0,
                    allowed_modes=frozenset({0o400}),
                )
                receipt_value = _decode_json(
                    receipt_raw,
                    label="capability approval install receipt",
                )
                if receipt_raw != _canonical_bytes(receipt_value):
                    raise RuntimeError(
                        "capability approval install receipt is not canonical"
                    )
                _validate_approval_install_receipt(
                    receipt_value,
                    plan=plan,
                    full_plan=full_plan,
                    approval=installed,
                    target=target,
                    receipt_path=receipt_path,
                )
                raise PermissionError(
                    "capability owner approval nonce was already consumed"
                )
        else:
            target = None
            installed_payload = None

        try:
            _require_approval_reserve(
                approval,
                plan,
                full_plan,
                now_unix=operation_clock.sample("approval install admission"),
            )
        except PermissionError:
            if target is not None:
                # The exact approval-only SIGKILL orphan is never left as a
                # permanent blocker.  Complete its immutable install receipt
                # from the stable inode, then synchronously retire it while
                # the lifecycle lock is still held.
                _require_same_file_identity(DEFAULT_APPROVAL_PATH, target)
                orphan_receipt = _build_approval_install_receipt(
                    plan,
                    full_plan,
                    approval,
                    target=target,
                    receipt_path=receipt_path,
                )
                _write_exclusive_bytes(
                    receipt_path,
                    _canonical_bytes(orphan_receipt),
                    mode=0o400,
                )
                _remove_installed_capability_approval(plan, full_plan)
            raise
        preflight = collect_capability_preflight(
            plan,
            full_plan,
            phase="stopped",
            runner=runner,
            metadata_reader=metadata_reader,
            local_identity_reader=local_identity_reader,
            approval_window_started_at_unix=approval.value[
                "stopped_preflight_observed_at_unix"
            ],
        )
        _require_approval_reserve(
            approval,
            plan,
            full_plan,
            now_unix=operation_clock.sample("approval preflight recheck"),
        )
        _require_capability_approval_preflight_binding(
            plan, full_plan, approval, preflight
        )
        _ensure_root_directory(DEFAULT_APPROVAL_PATH.parent)
        _ensure_root_directory(receipt_path.parent)
        created_here = target is None
        if created_here:
            _require_approval_reserve(
                approval,
                plan,
                full_plan,
                now_unix=operation_clock.sample("approval publish commit"),
            )
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
        try:
            _require_approval_reserve(
                approval,
                plan,
                full_plan,
                now_unix=operation_clock.sample("approval receipt commit"),
            )
            _require_same_file_identity(DEFAULT_APPROVAL_PATH, target)
            receipt = _build_approval_install_receipt(
                plan,
                full_plan,
                approval,
                target=target,
                receipt_path=receipt_path,
            )
            _write_exclusive_bytes(receipt_path, _canonical_bytes(receipt), mode=0o400)
        except BaseException:
            if created_here:
                current = os.lstat(DEFAULT_APPROVAL_PATH)
                if (current.st_dev, current.st_ino) == (
                    target.st_dev,
                    target.st_ino,
                ):
                    os.unlink(DEFAULT_APPROVAL_PATH)
                    _fsync_directory(DEFAULT_APPROVAL_PATH.parent)
            raise
        validated = _validate_approval_install_receipt(
            receipt,
            plan=plan,
            full_plan=full_plan,
            approval=approval,
            target=target,
            receipt_path=receipt_path,
        )
        return {**validated, "receipt_path": str(receipt_path)}


def _validate_approval_retirement_intent(
    value: Any,
    *,
    plan: CapabilityCanaryPlan,
    full_plan: FullCanaryPlan,
    path: Path,
) -> tuple[Mapping[str, Any], CapabilityCanaryOwnerApproval, Mapping[str, Any]]:
    fields = {
        "schema",
        "operation",
        "revision",
        "plan_sha256",
        "full_canary_plan_sha256",
        "approval",
        "approval_sha256",
        "nonce_sha256",
        "install_receipt_path",
        "install_receipt_sha256",
        "target_path",
        "target_device",
        "target_inode",
        "target_uid",
        "target_gid",
        "target_mode",
        "target_mtime_ns",
        "requested_at_unix",
        "receipt_path",
        "secret_material_recorded",
        "secret_digest_recorded",
        "receipt_sha256",
    }
    intent = _strict_mapping(value, fields, "capability approval retirement intent")
    approval = CapabilityCanaryOwnerApproval.from_mapping(intent["approval"])
    install_path = _approval_install_receipt_path(plan, approval)
    if (
        intent["schema"] != CAPABILITY_APPROVAL_RETIREMENT_INTENT_SCHEMA
        or intent["operation"] != "retire_capability_owner_approval_intent"
        or intent["revision"] != plan.revision
        or intent["plan_sha256"] != plan.sha256
        or intent["full_canary_plan_sha256"] != full_plan.sha256
        or approval.value["plan_sha256"] != plan.sha256
        or approval.value["full_canary_plan_sha256"] != full_plan.sha256
        or intent["approval_sha256"] != approval.sha256
        or intent["nonce_sha256"] != approval.value["nonce_sha256"]
        or intent["install_receipt_path"] != str(install_path)
        or intent["target_path"] != str(DEFAULT_APPROVAL_PATH)
        or intent["target_uid"] != 0
        or intent["target_gid"] != 0
        or intent["target_mode"] != "0400"
        or type(intent["target_device"]) is not int
        or type(intent["target_inode"]) is not int
        or type(intent["target_mtime_ns"]) is not int
        or type(intent["requested_at_unix"]) is not int
        or intent["requested_at_unix"] < 0
        or intent["receipt_path"] != str(path)
    ):
        raise RuntimeError("capability approval retirement intent drifted")
    receipt_raw, _ = _read_stable_file(
        install_path,
        maximum=64 * 1024,
        expected_uid=0,
        expected_gid=0,
        allowed_modes=frozenset({0o400}),
    )
    receipt_value = _decode_json(
        receipt_raw,
        label="capability approval install receipt",
    )
    if receipt_raw != _canonical_bytes(receipt_value):
        raise RuntimeError("capability approval install receipt is not canonical")
    target = SimpleNamespace(
        st_dev=intent["target_device"],
        st_ino=intent["target_inode"],
        st_uid=intent["target_uid"],
        st_gid=intent["target_gid"],
        st_mode=int(intent["target_mode"], 8),
        st_mtime_ns=intent["target_mtime_ns"],
    )
    receipt = _validate_approval_install_receipt(
        receipt_value,
        plan=plan,
        full_plan=full_plan,
        approval=approval,
        target=target,
        receipt_path=install_path,
    )
    if receipt["receipt_sha256"] != intent["install_receipt_sha256"]:
        raise RuntimeError("capability approval retirement intent drifted")
    return copy.deepcopy(dict(intent)), approval, receipt


def _approval_retirement_histories(
    plan: CapabilityCanaryPlan,
    full_plan: FullCanaryPlan,
) -> list[
    tuple[
        Mapping[str, Any],
        CapabilityCanaryOwnerApproval,
        Mapping[str, Any],
        Mapping[str, Any] | None,
    ]
]:
    root = DEFAULT_APPROVAL_RECEIPT_ROOT / plan.revision / plan.sha256
    try:
        names = sorted(os.listdir(root))
    except FileNotFoundError:
        return []
    nonce_pattern = re.compile(
        r"(?:\.)?([0-9a-f]{64})\.retirement-intent\.json(?:\.tmp)?"
    )
    nonces = sorted({
        match.group(1)
        for name in names
        if (match := nonce_pattern.fullmatch(name)) is not None
    })
    histories = []
    for nonce in nonces:
        intent_path = root / f"{nonce}.retirement-intent.json"
        _reconcile_lease_artifact_temporary(
            intent_path,
            schema=CAPABILITY_APPROVAL_RETIREMENT_INTENT_SCHEMA,
        )
        intent = _load_lease_artifact(
            intent_path,
            schema=CAPABILITY_APPROVAL_RETIREMENT_INTENT_SCHEMA,
        )
        intent, approval, install = _validate_approval_retirement_intent(
            intent,
            plan=plan,
            full_plan=full_plan,
            path=intent_path,
        )
        completion_path = _approval_retirement_receipt_path(plan, approval)
        _reconcile_lease_artifact_temporary(
            completion_path,
            schema=CAPABILITY_APPROVAL_RETIREMENT_RECEIPT_SCHEMA,
        )
        completion = (
            _load_lease_artifact(
                completion_path,
                schema=CAPABILITY_APPROVAL_RETIREMENT_RECEIPT_SCHEMA,
            )
            if os.path.lexists(completion_path)
            else None
        )
        if completion is not None and (
            completion.get("operation") != "retire_capability_owner_approval"
            or completion.get("revision") != plan.revision
            or completion.get("plan_sha256") != plan.sha256
            or completion.get("full_canary_plan_sha256") != full_plan.sha256
            or completion.get("approval_sha256") != approval.sha256
            or completion.get("nonce_sha256") != nonce
            or completion.get("install_receipt_sha256") != install["receipt_sha256"]
            or completion.get("retirement_intent_path") != str(intent_path)
            or completion.get("retirement_intent_sha256") != intent["receipt_sha256"]
            or completion.get("target_path") != str(DEFAULT_APPROVAL_PATH)
            or completion.get("target_absent") is not True
            or completion.get("removed") is not True
            or type(completion.get("removed_at_unix")) is not int
            or completion["removed_at_unix"] < intent["requested_at_unix"]
        ):
            raise RuntimeError("capability approval retirement receipt drifted")
        histories.append((intent, approval, install, completion))
    return histories


def _approval_retirement_result(
    intent: Mapping[str, Any],
    approval: CapabilityCanaryOwnerApproval,
    install: Mapping[str, Any],
    completion: Mapping[str, Any],
) -> Mapping[str, Any]:
    return {
        "path": str(DEFAULT_APPROVAL_PATH),
        "approval_sha256": approval.sha256,
        "install_receipt_sha256": install["receipt_sha256"],
        "retirement_intent_path": intent["receipt_path"],
        "retirement_intent_sha256": intent["receipt_sha256"],
        "retirement_receipt_path": completion["receipt_path"],
        "retirement_receipt_sha256": completion["receipt_sha256"],
        **_capability_approval_chain_fields(approval),
        "removed": True,
        "absent": not os.path.lexists(DEFAULT_APPROVAL_PATH),
    }


def _remove_installed_capability_approval(
    plan: CapabilityCanaryPlan,
    full_plan: FullCanaryPlan,
) -> Mapping[str, Any]:
    """Crash-safely retire only the exact installed approval inode."""

    histories = _approval_retirement_histories(plan, full_plan)
    unfinished = [history for history in histories if history[3] is None]
    if len(unfinished) > 1:
        raise RuntimeError("capability approval retirement history is ambiguous")
    if os.path.lexists(DEFAULT_APPROVAL_PATH):
        approval, target, _raw = _read_exact_installed_approval(plan, full_plan)
        receipt_path = _approval_install_receipt_path(plan, approval)
        receipt_raw, _ = _read_stable_file(
            receipt_path,
            maximum=64 * 1024,
            expected_uid=0,
            expected_gid=0,
            allowed_modes=frozenset({0o400}),
        )
        receipt_value = _decode_json(
            receipt_raw,
            label="capability approval install receipt",
        )
        if receipt_raw != _canonical_bytes(receipt_value):
            raise RuntimeError("capability approval install receipt is not canonical")
        receipt = _validate_approval_install_receipt(
            receipt_value,
            plan=plan,
            full_plan=full_plan,
            approval=approval,
            target=target,
            receipt_path=receipt_path,
        )
        matching = [
            history for history in unfinished if history[1].sha256 == approval.sha256
        ]
        if unfinished and len(matching) != 1:
            raise RuntimeError(
                "capability approval retirement intent mismatches target"
            )
        if matching:
            intent, _approval, _install, _completion = matching[0]
        else:
            intent_path = _approval_retirement_intent_path(plan, approval)
            _ensure_root_directory(intent_path.parent)
            intent = _append_lease_artifact(
                intent_path,
                schema=CAPABILITY_APPROVAL_RETIREMENT_INTENT_SCHEMA,
                value={
                    "operation": "retire_capability_owner_approval_intent",
                    "revision": plan.revision,
                    "plan_sha256": plan.sha256,
                    "full_canary_plan_sha256": full_plan.sha256,
                    "approval": copy.deepcopy(dict(approval.value)),
                    "approval_sha256": approval.sha256,
                    "nonce_sha256": approval.value["nonce_sha256"],
                    "install_receipt_path": str(receipt_path),
                    "install_receipt_sha256": receipt["receipt_sha256"],
                    "target_path": str(DEFAULT_APPROVAL_PATH),
                    "target_device": target.st_dev,
                    "target_inode": target.st_ino,
                    "target_uid": target.st_uid,
                    "target_gid": target.st_gid,
                    "target_mode": f"{stat.S_IMODE(target.st_mode):04o}",
                    "target_mtime_ns": target.st_mtime_ns,
                    "requested_at_unix": int(time.time()),
                },
            )
        _require_same_file_identity(DEFAULT_APPROVAL_PATH, target)
        os.unlink(DEFAULT_APPROVAL_PATH)
        _fsync_directory(DEFAULT_APPROVAL_PATH.parent)
        install = receipt
    elif unfinished:
        intent, approval, install, _completion = unfinished[0]
    else:
        completed = [history for history in histories if history[3] is not None]
        if not completed:
            return {
                "path": str(DEFAULT_APPROVAL_PATH),
                "removed": False,
                "absent": True,
            }
        latest = max(
            completed,
            key=lambda history: (
                history[3]["removed_at_unix"],
                history[1].value["nonce_sha256"],
            ),
        )
        return _approval_retirement_result(latest[0], latest[1], latest[2], latest[3])
    if os.path.lexists(DEFAULT_APPROVAL_PATH):
        raise RuntimeError("capability approval remains after retirement")
    retirement_path = _approval_retirement_receipt_path(plan, approval)
    completion = _append_lease_artifact(
        retirement_path,
        schema=CAPABILITY_APPROVAL_RETIREMENT_RECEIPT_SCHEMA,
        value={
            "operation": "retire_capability_owner_approval",
            "revision": plan.revision,
            "plan_sha256": plan.sha256,
            "full_canary_plan_sha256": full_plan.sha256,
            "approval_sha256": approval.sha256,
            "nonce_sha256": approval.value["nonce_sha256"],
            "install_receipt_sha256": install["receipt_sha256"],
            "retirement_intent_path": intent["receipt_path"],
            "retirement_intent_sha256": intent["receipt_sha256"],
            "target_path": str(DEFAULT_APPROVAL_PATH),
            "removed_at_unix": max(
                int(time.time()),
                intent["requested_at_unix"],
            ),
            "removed": True,
            "target_absent": True,
        },
    )
    return _approval_retirement_result(intent, approval, install, completion)


def _require_installed_capability_approval(
    plan: CapabilityCanaryPlan,
    full_plan: FullCanaryPlan,
    expected: CapabilityCanaryOwnerApproval,
) -> Mapping[str, Any]:
    """Fail closed unless the exact installed approval and nonce receipt remain."""

    raw, target = _read_stable_file(
        DEFAULT_APPROVAL_PATH,
        maximum=256 * 1024,
        expected_uid=0,
        expected_gid=0,
        allowed_modes=frozenset({0o400}),
    )
    value = _decode_json(raw, label="installed capability approval")
    if raw != _canonical_bytes(value):
        raise RuntimeError("installed capability approval is not canonical")
    approval = CapabilityCanaryOwnerApproval.from_mapping(value)
    if approval.sha256 != expected.sha256 or approval.value != expected.value:
        raise PermissionError("installed capability approval differs from start input")
    receipt_path = _approval_install_receipt_path(plan, approval)
    receipt_raw, _ = _read_stable_file(
        receipt_path,
        maximum=256 * 1024,
        expected_uid=0,
        expected_gid=0,
        allowed_modes=frozenset({0o400}),
    )
    receipt = _decode_json(receipt_raw, label="capability approval install receipt")
    unsigned = {
        key: copy.deepcopy(item)
        for key, item in receipt.items()
        if key != "receipt_sha256"
    }
    if (
        receipt_raw != _canonical_bytes(receipt)
        or receipt.get("schema") != CAPABILITY_APPROVAL_INSTALL_RECEIPT_SCHEMA
        or receipt.get("receipt_sha256") != _sha256_json(unsigned)
        or receipt.get("approval_sha256") != approval.sha256
        or receipt.get("plan_sha256") != plan.sha256
        or receipt.get("full_canary_plan_sha256") != full_plan.sha256
        or receipt.get("nonce_sha256") != approval.value["nonce_sha256"]
        or receipt.get("target_path") != str(DEFAULT_APPROVAL_PATH)
        or receipt.get("target_device") != target.st_dev
        or receipt.get("target_inode") != target.st_ino
        or receipt.get("target_uid") != target.st_uid
        or receipt.get("target_gid") != target.st_gid
        or receipt.get("target_mode") != f"{stat.S_IMODE(target.st_mode):04o}"
    ):
        raise RuntimeError("capability approval install receipt drifted")
    return copy.deepcopy(dict(receipt))


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
        "canonical_brain": {
            "writer_boundary": {"enabled": True},
            "discord_edge": {"enabled": True},
            "tools_enabled": True,
        },
        "model": copy.deepcopy(_CAPABILITY_MODEL_ROUTE),
        "agent": {
            "reasoning_effort": "high",
            "max_turns": 90,
            "adaptive_reasoning": {"enabled": True, "max_effort": "max"},
            "tool_use_enforcement": True,
            "task_completion_guidance": True,
            "parallel_tool_call_guidance": True,
            "background_review_enabled": False,
            "verification_ledger_enabled": False,
            "verify_on_stop": False,
        },
        "compression": {
            "enabled": True,
            "abort_on_summary_failure": True,
        },
        "context": {"engine": "compressor"},
        "auxiliary": {
            task: copy.deepcopy(_CAPABILITY_AUXILIARY_ROUTE)
            for task in _CAPABILITY_AUXILIARY_TASKS
        },
        "memory": {
            "provider": "",
            "memory_enabled": True,
            "user_profile_enabled": True,
        },
        "cron": {"enabled": False},
        "goals": {"max_turns": 0},
        "kanban": {
            "auxiliary_planning_enabled": False,
            "auto_decompose": False,
            "dispatch_in_gateway": False,
        },
        "curator": {"enabled": False, "prune_builtins": False},
        "tool_loop_guardrails": {
            "warnings_enabled": True,
            "hard_stop_enabled": False,
        },
        "tools": {"tool_search": {"enabled": "off"}},
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
            "home_mode": "profile",
            "lifetime_seconds": 900,
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
    if raw.get("model") != _CAPABILITY_MODEL_ROUTE:
        raise ValueError("capability gateway model is not exact")
    agent = raw.get("agent")
    if agent != {
        "reasoning_effort": "high",
        "max_turns": 90,
        "adaptive_reasoning": {"enabled": True, "max_effort": "max"},
        "tool_use_enforcement": True,
        "task_completion_guidance": True,
        "parallel_tool_call_guidance": True,
        "background_review_enabled": False,
        "verification_ledger_enabled": False,
        "verify_on_stop": False,
    }:
        raise ValueError("capability adaptive reasoning is not exact")
    if raw.get("compression") != {
        "enabled": True,
        "abort_on_summary_failure": True,
    } or raw.get("context") != {"engine": "compressor"}:
        raise ValueError("capability compression boundary is not exact")
    auxiliary = raw.get("auxiliary")
    if (
        not isinstance(auxiliary, Mapping)
        or set(auxiliary) != set(_CAPABILITY_AUXILIARY_TASKS)
        or any(
            auxiliary.get(task) != _CAPABILITY_AUXILIARY_ROUTE
            for task in _CAPABILITY_AUXILIARY_TASKS
        )
    ):
        raise ValueError("capability auxiliary routes are not exact")
    if raw.get("kanban") != {
        "auxiliary_planning_enabled": False,
        "auto_decompose": False,
        "dispatch_in_gateway": False,
    }:
        raise ValueError("capability Kanban boundary is not exact")
    if raw.get("cron") != {"enabled": False}:
        raise ValueError("capability cron boundary is not exact")
    if raw.get("goals") != {"max_turns": 0}:
        raise ValueError("capability goal continuation budget is not exact")
    if raw.get("curator") != {"enabled": False, "prune_builtins": False}:
        raise ValueError("capability curator boundary is not exact")
    if raw.get("memory") != {
        "provider": "",
        "memory_enabled": True,
        "user_profile_enabled": True,
    }:
        raise ValueError("capability memory boundary is not exact")
    if raw.get("tool_loop_guardrails") != {
        "warnings_enabled": True,
        "hard_stop_enabled": False,
    }:
        raise ValueError("capability tool-loop boundary is not exact")
    if raw.get("tools") != {"tool_search": {"enabled": "off"}}:
        raise ValueError("capability tool search must remain disabled")
    if raw.get("plugins") != {"enabled": [CAPABILITY_OBSERVER_PLUGIN]}:
        raise ValueError("capability plugin allowlist is not exact")
    if raw.get("hooks") != {}:
        raise ValueError("capability hook surface must be empty")
    terminal = raw.get("terminal")
    if (
        not isinstance(terminal, Mapping)
        or terminal.get("backend") != "isolated_worker"
    ):
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
            or browser_client.request_timeout_seconds != BROWSER_COMMAND_TIMEOUT_SECONDS
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
    if (
        raw.get("platforms") != expected_platforms
        or gateway.get("platforms") != expected_platforms
    ):
        raise ValueError("capability gateway platforms are not exact")
    if "discord" in raw.get("platforms", {}) or "discord" in gateway.get(
        "platforms", {}
    ):
        raise ValueError("direct Discord is forbidden in the capability gateway")
    canonical = raw.get("canonical_brain")
    if canonical != {
        "writer_boundary": {"enabled": True},
        "discord_edge": {"enabled": True},
        "tools_enabled": True,
    }:
        raise ValueError("capability Canonical Writer boundary is not exact")


def validate_capability_provider_registry(provider_registry: Any) -> None:
    """Attest the irreversible single-provider canary registry."""

    registry = getattr(provider_registry, "_REGISTRY", None)
    aliases = getattr(provider_registry, "_ALIASES", None)
    if (
        getattr(provider_registry, "_discovered", None) is not True
        or getattr(provider_registry, "_discovery_error", None) is not None
        or getattr(provider_registry, "_isolated_provider_allowlist", None)
        != frozenset({"openai-codex"})
        or getattr(provider_registry, "_isolated_discovery_validated", None)
        is not True
        or not isinstance(registry, Mapping)
        or set(registry) != {"openai-codex"}
        or aliases
        != {"codex": "openai-codex", "openai_codex": "openai-codex"}
    ):
        raise RuntimeError("capability provider registry is not exact")
    profile = registry["openai-codex"]
    if (
        getattr(profile, "name", None) != "openai-codex"
        or tuple(getattr(profile, "aliases", ())) != ("codex", "openai_codex")
        or getattr(profile, "api_mode", None) != "codex_responses"
        or getattr(profile, "base_url", None)
        != _CAPABILITY_MODEL_ROUTE["base_url"]
        or getattr(profile, "auth_type", None) != "oauth_external"
        or tuple(getattr(profile, "env_vars", ())) != ()
    ):
        raise RuntimeError("capability provider profile is not exact")


def validate_capability_model_runtime_route(
    model: Any,
    runtime_kwargs: Mapping[str, Any],
) -> None:
    """Attest the fixed GPT-5.6 Codex transport before agent construction.

    This boundary is deliberately semantic-free: it never examines a prompt,
    task, channel, or user.  It only prevents a sealed capability process from
    silently inheriting a session/request route, subprocess transport, or
    fallback provider after its reviewed config was admitted.
    """

    if (
        model != _CAPABILITY_MODEL_ROUTE["default"]
        or not isinstance(runtime_kwargs, Mapping)
        or runtime_kwargs.get("provider") != _CAPABILITY_MODEL_ROUTE["provider"]
        or runtime_kwargs.get("api_mode") != "codex_responses"
        or runtime_kwargs.get("base_url") != _CAPABILITY_MODEL_ROUTE["base_url"]
        or runtime_kwargs.get("command") not in (None, "")
        or list(runtime_kwargs.get("args") or []) != []
    ):
        raise RuntimeError("capability canary model route is not exact")


def validate_capability_agent_policy(agent: Any) -> None:
    """Attest the constructed agent immediately before its first model call."""

    if (
        getattr(agent, "model", None) != _CAPABILITY_MODEL_ROUTE["default"]
        or getattr(agent, "provider", None) != _CAPABILITY_MODEL_ROUTE["provider"]
        or getattr(agent, "api_mode", None) != "codex_responses"
        or getattr(agent, "base_url", None) != _CAPABILITY_MODEL_ROUTE["base_url"]
        or getattr(agent, "reasoning_config", None)
        != {"enabled": True, "effort": "high"}
        or getattr(agent, "_adaptive_reasoning_policy", None)
        != {"enabled": True, "max_effort": "max"}
        or getattr(agent, "_tool_use_enforcement", None) is not True
        or getattr(agent, "_task_completion_guidance", None) is not True
        or getattr(agent, "_parallel_tool_call_guidance", None) is not True
        or getattr(agent, "_background_review_enabled", None) is not False
        or list(getattr(agent, "_fallback_chain", ()) or ()) != []
        or getattr(agent, "_fallback_model", None) is not None
        or dict(getattr(agent, "request_overrides", {}) or {}) != {}
        or getattr(agent, "service_tier", None) is not None
    ):
        raise RuntimeError("capability canary agent policy is not exact")


def validate_capability_extension_surface(
    plugin_manager: Any,
    gateway_hooks: Any,
    provider_registry: Any,
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
        or getattr(plugin_manager, "_isolated_allowlist", None) != expected_allowlist
        or getattr(plugin_manager, "_isolated_discovery_failure", None) is not None
    ):
        raise RuntimeError("capability plugin discovery is not isolated")
    validate_capability_provider_registry(provider_registry)

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
        or getattr(module, "__name__", None) != "hermes_plugins.muncho_canary_evidence"
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
    if (
        any(
            getattr(plugin_manager, name, None) != expected
            for name, expected in empty_surfaces.items()
        )
        or getattr(plugin_manager, "_context_engine", None) is not None
    ):
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
    if not optional.issubset({
        "INVOCATION_ID",
        "JOURNAL_STREAM",
        "NOTIFY_SOCKET",
        "SYSTEMD_EXEC_PID",
    }):
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
    if (
        "INVOCATION_ID" in env
        and re.fullmatch(r"[0-9a-f]{32}", env["INVOCATION_ID"]) is None
    ):
        return False
    if (
        "JOURNAL_STREAM" in env
        and re.fullmatch(r"[0-9]+:[0-9]+", env["JOURNAL_STREAM"]) is None
    ):
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
        "[Unit]",
        "Description=Muncho privileged Mac operations edge (capability canary)",
        "After=network-online.target",
        "Wants=network-online.target",
        f"Before={GATEWAY_UNIT_NAME}",
        f"AssertPathExists={DEFAULT_MAC_OPS_CONFIG}",
        f"AssertPathExists={DEFAULT_MAC_OPS_CREDENTIAL}",
        "",
        "[Service]",
        "Type=simple",
        f"User={plan.identities.mac_ops_user}",
        f"Group={plan.identities.mac_ops_group}",
        f"SupplementaryGroups={plan.identities.socket_client_group}",
        "RuntimeDirectory=muncho-mac-ops",
        "RuntimeDirectoryMode=0750",
        "RuntimeDirectoryPreserve=no",
        "StateDirectory=muncho-mac-ops",
        "StateDirectoryMode=0700",
        f"WorkingDirectory={plan.release_root}",
        f"ExecStart={plan.interpreter} -B -I -m gateway.mac_ops_edge_service --config {DEFAULT_MAC_OPS_CONFIG}",
        "Restart=no",
        "RuntimeMaxSec=900s",
        "TimeoutStartSec=30s",
        "TimeoutStopSec=30s",
        "KillMode=mixed",
        "LimitCORE=0",
        *_fixed_environment(
            user=plan.identities.mac_ops_user, home=DEFAULT_MAC_OPS_STATE
        ),
        _CAPABILITY_GATEWAY_UNSET_ENVIRONMENT_DIRECTIVE,
        *_common_hardening(),
        "RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6",
        "IPAddressDeny=169.254.169.254/32",
        "IPAddressDeny=127.0.0.1/32",
        "IPAddressDeny=::1/128",
        f"BindReadOnlyPaths={plan.release_root}",
        f"ReadOnlyPaths={DEFAULT_MAC_OPS_CONFIG}",
        f"ReadOnlyPaths={DEFAULT_MAC_OPS_CREDENTIAL}",
        f"InaccessiblePaths={DEFAULT_GATEWAY_AUTH_STORE}",
        f"InaccessiblePaths={DEFAULT_EDGE_TOKEN_DIRECTORY}",
        f"ReadWritePaths={DEFAULT_MAC_OPS_RUNTIME}",
        f"ReadWritePaths={DEFAULT_MAC_OPS_STATE}",
        "",
        "[Install]",
        "WantedBy=multi-user.target",
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
        "gitlab": {
            "env_file": str(DEFAULT_MAC_OPS_CREDENTIAL),
            "project_id": MAC_OPS_PROJECT_ID,
            "timeout_seconds": 20.0,
        },
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
        "TERMINAL_TIMEOUT": "180",
        "TERMINAL_HOME_MODE": "profile",
        "TERMINAL_LIFETIME_SECONDS": "900",
        "TERMINAL_ISOLATED_WORKER_SOCKET": str(DEFAULT_WORKER_SOCKET),
        "TERMINAL_ISOLATED_WORKER_SERVER_UID": str(plan.identities.worker_uid),
        "TERMINAL_ISOLATED_WORKER_SERVER_GID": str(plan.identities.worker_gid),
        "TERMINAL_ISOLATED_WORKER_SOCKET_UID": "0",
        "TERMINAL_ISOLATED_WORKER_SOCKET_GID": str(plan.identities.worker_client_gid),
    }
    dependencies = " ".join((
        WRITER_UNIT_NAME,
        EDGE_UNIT_NAME,
        DEFAULT_DISCORD_CONNECTOR_UNIT,
        MAC_OPS_UNIT_NAME,
        DEFAULT_WORKER_SOCKET_UNIT_NAME,
        DEFAULT_WORKER_SERVICE_UNIT_NAME,
        DEFAULT_BROWSER_UNIT_NAME,
    ))
    lines = [
        "# Digest-bound production-shaped capability canary; do not edit.",
        f"# ArtifactSHA256={plan.release_artifact_sha256}",
        "# DiscordCredentialInGateway=false",
        "[Unit]",
        "Description=Muncho production-shaped model gateway (capability canary)",
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
        f"AssertPathExists={DEFAULT_GOAL_OBSERVER_CONFIG}",
        f"AssertPathExists={DEFAULT_GOAL_COLLECTOR_SOCKET}",
        "",
        "[Service]",
        "Type=notify",
        "NotifyAccess=main",
        f"User={plan.identities.gateway_user}",
        f"Group={plan.identities.gateway_group}",
        (
            f"SupplementaryGroups={plan.identities.socket_client_group} "
            f"{plan.identities.edge_group} {plan.identities.connector_group} "
            f"{plan.identities.worker_client_group} {plan.identities.browser_group}"
        ),
        "RuntimeDirectory=hermes-cloud-gateway",
        "RuntimeDirectoryMode=0700",
        "RuntimeDirectoryPreserve=no",
        "StateDirectory=muncho-capability-canary",
        "StateDirectoryMode=0700",
        f"WorkingDirectory={plan.release_root}",
        f"ExecStart={plan.interpreter} -B -I -m gateway.run --config {DEFAULT_GATEWAY_CONFIG} --require-capability-canary",
        "Restart=no",
        "RuntimeMaxSec=900s",
        "TimeoutStartSec=180s",
        "TimeoutStopSec=90s",
        "KillMode=mixed",
        "LimitCORE=0",
        f"LoadCredential={API_SERVER_CREDENTIAL_NAME}:{DEFAULT_API_SERVER_CONTROL_KEY}",
        *_fixed_environment(
            user=plan.identities.gateway_user, home=DEFAULT_GATEWAY_HOME
        ),
        "Environment=PATH=/usr/bin:/bin",
        f"Environment=HERMES_CONFIG={DEFAULT_GATEWAY_CONFIG}",
        f"Environment=HERMES_HOME={DEFAULT_GATEWAY_PROFILE_HOME}",
        f"Environment=HERMES_MANAGED_DIR={DEFAULT_DISABLED_MANAGED_SCOPE}",
        f"Environment=SSL_CERT_FILE={DEFAULT_GATEWAY_CA_BUNDLE}",
        f"Environment=GATEWAY_RELAY_URL={_PINNED_RELAY_URL}",
        "Environment=GATEWAY_RELAY_PLATFORMS=discord",
        *(f"Environment={key}={value}" for key, value in sorted(terminal_env.items())),
        _CAPABILITY_GATEWAY_UNSET_ENVIRONMENT_DIRECTIVE,
        *_common_hardening(),
        "RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6",
        "IPAddressDeny=169.254.169.254/32",
        f"BindReadOnlyPaths={plan.release_root}",
        f"BindReadOnlyPaths={DEFAULT_GATEWAY_CONFIG}",
        f"ReadOnlyPaths={DEFAULT_OBSERVER_CONFIG}",
        f"ReadOnlyPaths={DEFAULT_PLAN_PATH}",
        f"BindReadOnlyPaths={DEFAULT_GATEWAY_AUTH_STORE}",
        f"ReadOnlyPaths={DEFAULT_E2E_FIXTURE}",
        f"ReadOnlyPaths={DEFAULT_GATEWAY_CA_BUNDLE}",
        f"ReadOnlyPaths={DEFAULT_MAC_OPS_RUNTIME}",
        f"ReadOnlyPaths={DEFAULT_DISCORD_CONNECTOR_SOCKET.parent}",
        f"ReadOnlyPaths={DEFAULT_WORKER_CONFIG}",
        f"ReadOnlyPaths={DEFAULT_WORKER_SOCKET.parent}",
        f"ReadOnlyPaths={DEFAULT_BROWSER_CONFIG}",
        f"ReadOnlyPaths={DEFAULT_BROWSER_SOCKET.parent}",
        f"ReadOnlyPaths={DEFAULT_GOAL_OBSERVER_CONFIG}",
        f"ReadOnlyPaths={DEFAULT_GOAL_COLLECTOR_SOCKET.parent}",
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
        "InaccessiblePaths=-/etc/hermes",
        "InaccessiblePaths=-/root/.codex",
        f"ReadWritePaths={DEFAULT_GATEWAY_RUNTIME}",
        f"ReadWritePaths={DEFAULT_GATEWAY_HOME}",
        f"ReadWritePaths={DEFAULT_GATEWAY_LOG_ROOT}",
        f"ReadWritePaths={DEFAULT_GATEWAY_WORK_ROOT}",
        "",
        "[Install]",
        "WantedBy=multi-user.target",
    ]
    result = "\n".join(lines) + "\n"
    if any(
        marker in result
        for marker in (
            "DISCORD_BOT_TOKEN=",
            "GITLAB_TOKEN=",
            "EnvironmentFile=",
            "PassEnvironment=",
            "Environment=TERMINAL_DOCKER_",
            "docker.sock",
            "remote-debugging",
            "127.0.0.1:9222",
            "BROWSER_CDP_URL=",
        )
    ):
        raise ValueError("capability gateway unit crosses a credential boundary")
    return result


def build_capability_plan(
    *,
    full_plan: FullCanaryPlan,
    full_canary_terminal_receipt: Mapping[str, Any],
    full_canary_terminal_receipt_sha256: str,
    mac_ops_uid: int,
    mac_ops_gid: int,
    connector_uid: int,
    connector_gid: int,
    bitrix_operational_edge_uid: int,
    bitrix_operational_edge_gid: int,
    bitrix_operational_edge_client_gid: int,
    browser_uid: int,
    browser_gid: int,
    worker_uid: int,
    worker_gid: int,
    worker_client_gid: int,
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
    terminal, terminal_sha256 = _terminal_receipt_binding(
        full_canary_terminal_receipt,
        full_canary_terminal_receipt_sha256,
        revision=full_plan.revision,
        full_canary_plan_sha256=full_plan.sha256,
    )
    identities = RuntimeIdentities(
        gateway_user=full_plan.identities.gateway_user,
        gateway_group=full_plan.identities.gateway_group,
        gateway_uid=full_plan.identities.gateway_uid,
        gateway_gid=full_plan.identities.gateway_gid,
        socket_client_group=full_plan.identities.socket_client_group,
        socket_client_gid=full_plan.identities.socket_client_gid,
        edge_group=full_plan.identities.edge_group,
        mac_ops_user="muncho-mac-ops-edge",
        mac_ops_group="muncho-mac-ops-edge",
        mac_ops_uid=_positive_id(mac_ops_uid, "mac_ops_uid"),
        mac_ops_gid=_positive_id(mac_ops_gid, "mac_ops_gid"),
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
        worker_client_gid=_positive_id(worker_client_gid, "worker_client_gid"),
    )
    seed = object.__new__(CapabilityCanaryPlan)
    values = dict(
        revision=full_plan.revision,
        full_canary_plan_sha256=full_plan.sha256,
        full_canary_terminal_receipt=terminal,
        full_canary_terminal_receipt_sha256=terminal_sha256,
        original_full_canary_owner_approval_sha256=terminal["owner_approval_sha256"],
        release_artifact_sha256=full_plan.release["artifact_sha256"],
        release_root=Path(full_plan.release["artifact_root"]),
        interpreter=Path(full_plan.release["interpreter"]),
        identities=identities,
        browser_socket_path=DEFAULT_BROWSER_SOCKET,
        browser_artifact_root=DEFAULT_BROWSER_ARTIFACT_ROOT,
        browser_node=Path(full_plan.release["artifact_root"]) / NODE_EXECUTABLE,
        browser_node_sha256=_digest(browser_node_sha256, "browser node"),
        browser_wrapper=(
            Path(full_plan.release["artifact_root"]) / AGENT_BROWSER_WRAPPER
        ),
        browser_wrapper_sha256=_digest(browser_wrapper_sha256, "browser wrapper"),
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
        connector_allowed_channel_ids=tuple(sorted(set(connector_allowed_channel_ids))),
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
        bitrix_operational_edge_service_user=(identities.bitrix_operational_edge_user),
        bitrix_operational_edge_service_group=(
            identities.bitrix_operational_edge_group
        ),
        bitrix_operational_edge_service_uid=(identities.bitrix_operational_edge_uid),
        bitrix_operational_edge_service_gid=(identities.bitrix_operational_edge_gid),
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
        bitrix_operational_edge_credential_binding=("bitrix_operational_edge_webhook"),
    )
    for key, value in values.items():
        object.__setattr__(seed, key, value)
    object.__setattr__(
        seed,
        "bitrix_operational_edge_service_identity_sha256",
        _bitrix_operational_edge_identity(
            revision=seed.revision,
            release_artifact_sha256=seed.release_artifact_sha256,
            asset_manifest_sha256=(seed.bitrix_operational_edge_asset_manifest_sha256),
            rendered_unit_sha256=(seed.bitrix_operational_edge_rendered_unit_sha256),
            rendered_config_sha256=(
                seed.bitrix_operational_edge_rendered_config_sha256
            ),
            rendered_trust_sha256=(seed.bitrix_operational_edge_rendered_trust_sha256),
            identity_bootstrap_receipt_sha256=(
                seed.bitrix_operational_edge_identity_bootstrap_receipt_sha256
            ),
            receipt_public_key_id=(seed.bitrix_operational_edge_receipt_public_key_id),
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
    object.__setattr__(
        seed, "mac_ops_config_sha256", _sha256_bytes(render_mac_ops_config(seed))
    )
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
    object.__setattr__(
        seed, "gateway_config_sha256", _sha256_bytes(render_gateway_config(seed))
    )
    object.__setattr__(
        seed,
        "gateway_unit_sha256",
        _sha256_bytes(render_gateway_unit(seed).encode("utf-8")),
    )
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
    unsigned = {name: item for name, item in value.items() if name != "manifest_sha256"}
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
        or agent_browser.get("config_sha256") != plan.agent_browser_config_sha256
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
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
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


def _capability_passwd_slot_inventory(
    user_name: str,
    uid: int,
    gid: int,
) -> tuple[list[tuple[str, int, int, str, str]], list[str]]:
    """Return raw NSS user-slot collisions and primary-GID members.

    ``grp.gr_mem`` does not normally list primary-group membership.  Raw lists
    deliberately retain duplicate NSS rows so aliases cannot disappear behind
    name-based lookup or set de-duplication.
    """

    try:
        entries = pwd.getpwall()
    except (KeyError, OSError) as exc:
        raise RuntimeError("capability NSS passwd inventory is unavailable") from exc
    slot_rows = sorted(
        (
            entry.pw_name,
            entry.pw_uid,
            entry.pw_gid,
            entry.pw_dir,
            entry.pw_shell,
        )
        for entry in entries
        if entry.pw_name == user_name or entry.pw_uid == uid
    )
    primary_names = sorted(entry.pw_name for entry in entries if entry.pw_gid == gid)
    if any(not isinstance(name, str) or not name for name in primary_names):
        raise RuntimeError("capability NSS passwd inventory is invalid")
    return slot_rows, primary_names


def _capability_primary_group_user_names(gid: int) -> list[str]:
    """Return raw, non-deduplicated primary users for a group-only slot."""

    try:
        entries = pwd.getpwall()
    except (KeyError, OSError) as exc:
        raise RuntimeError("capability NSS passwd inventory is unavailable") from exc
    names = sorted(entry.pw_name for entry in entries if entry.pw_gid == gid)
    if any(not isinstance(name, str) or not name for name in names):
        raise RuntimeError("capability NSS passwd inventory is invalid")
    return names


def _capability_group_slot_inventory(
    group_name: str,
    gid: int,
) -> list[tuple[str, int, tuple[str, ...]]]:
    """Return every NSS group row colliding by fixed name or numeric GID."""

    try:
        entries = grp.getgrall()
    except (KeyError, OSError) as exc:
        raise RuntimeError("capability NSS group inventory is unavailable") from exc
    return sorted(
        (
            entry.gr_name,
            entry.gr_gid,
            tuple(sorted(entry.gr_mem)),
        )
        for entry in entries
        if entry.gr_name == group_name or entry.gr_gid == gid
    )


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
    passwd_slot_rows, primary_group_users = _capability_passwd_slot_inventory(
        identities.browser_user,
        identities.browser_uid,
        identities.browser_gid,
    )
    group_slot_rows = _capability_group_slot_inventory(
        identities.browser_group,
        identities.browser_gid,
    )
    expected_passwd_row = (
        identities.browser_user,
        identities.browser_uid,
        identities.browser_gid,
        DEFAULT_BROWSER_HOME,
        DEFAULT_BROWSER_SHELL,
    )
    expected_group_row = (
        identities.browser_group,
        identities.browser_gid,
        (),
    )
    supplementary_group_ids: list[int] | None = None
    all_absent = all(
        item is None for item in (browser_user, browser_group, uid_owner, gid_owner)
    ) and (
        passwd_slot_rows == [] and primary_group_users == [] and group_slot_rows == []
    )
    group_present_user_absent = (
        browser_user is None
        and uid_owner is None
        and browser_group is not None
        and gid_owner is not None
        and browser_group.gr_name == identities.browser_group
        and browser_group.gr_gid == identities.browser_gid
        and gid_owner.gr_name == identities.browser_group
        and list(browser_group.gr_mem) == []
        and passwd_slot_rows == []
        and primary_group_users == []
        and group_slot_rows == [expected_group_row]
    )
    present_exact = (
        browser_user is not None
        and uid_owner is not None
        and browser_group is not None
        and gid_owner is not None
        and browser_user.pw_name == identities.browser_user
        and browser_user.pw_uid == identities.browser_uid
        and browser_user.pw_gid == identities.browser_gid
        and browser_user.pw_dir == DEFAULT_BROWSER_HOME
        and browser_user.pw_shell == DEFAULT_BROWSER_SHELL
        and uid_owner.pw_name == identities.browser_user
        and browser_group.gr_name == identities.browser_group
        and browser_group.gr_gid == identities.browser_gid
        and gid_owner.gr_name == identities.browser_group
        and list(browser_group.gr_mem) == []
        and passwd_slot_rows == [expected_passwd_row]
        and primary_group_users == [identities.browser_user]
        and group_slot_rows == [expected_group_row]
    )
    if present_exact:
        supplementary_group_ids = sorted(
            set(os.getgrouplist(identities.browser_user, identities.browser_gid))
        )
        if supplementary_group_ids != [identities.browser_gid]:
            raise RuntimeError("capability browser has supplementary authority")
        state = "present_exact"
    elif group_present_user_absent:
        state = "group_present_user_absent_create_only_slot"
    elif all_absent:
        state = "absent_create_only_slot"
    else:
        raise RuntimeError("capability browser identity slot collides or drifted")
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
            Command((
                GROUPADD,
                "--system",
                "--gid",
                str(plan.identities.browser_gid),
                "--",
                plan.identities.browser_group,
            )),
            runner=runner,
            label="create capability browser group",
        )
        created_group = True
    if state != "present_exact":
        _run_checked(
            Command((
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
            )),
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
    worker_passwd_rows, worker_primary_users = _capability_passwd_slot_inventory(
        identities.worker_user,
        identities.worker_uid,
        identities.worker_gid,
    )
    worker_group_rows = _capability_group_slot_inventory(
        identities.worker_group,
        identities.worker_gid,
    )
    client_primary_users = _capability_primary_group_user_names(
        identities.worker_client_gid
    )
    client_group_rows = _capability_group_slot_inventory(
        identities.worker_client_group,
        identities.worker_client_gid,
    )
    expected_worker_passwd = (
        identities.worker_user,
        identities.worker_uid,
        identities.worker_gid,
        DEFAULT_WORKER_HOME,
        DEFAULT_WORKER_SHELL,
    )
    expected_worker_group = (
        identities.worker_group,
        identities.worker_gid,
        (),
    )
    expected_client_group = (
        identities.worker_client_group,
        identities.worker_client_gid,
        (),
    )

    worker_all_absent = all(
        item is None
        for item in (worker_user, worker_group, uid_owner, worker_gid_owner)
    ) and (
        worker_passwd_rows == []
        and worker_primary_users == []
        and worker_group_rows == []
    )
    worker_group_present = (
        worker_user is None
        and uid_owner is None
        and worker_group is not None
        and worker_gid_owner is not None
        and worker_group.gr_name == identities.worker_group
        and worker_group.gr_gid == identities.worker_gid
        and worker_gid_owner.gr_name == identities.worker_group
        and list(worker_group.gr_mem) == []
        and worker_passwd_rows == []
        and worker_primary_users == []
        and worker_group_rows == [expected_worker_group]
    )
    worker_present = (
        worker_user is not None
        and uid_owner is not None
        and worker_group is not None
        and worker_gid_owner is not None
        and worker_user.pw_name == identities.worker_user
        and worker_user.pw_uid == identities.worker_uid
        and worker_user.pw_gid == identities.worker_gid
        and worker_user.pw_dir == DEFAULT_WORKER_HOME
        and worker_user.pw_shell == DEFAULT_WORKER_SHELL
        and uid_owner.pw_name == identities.worker_user
        and worker_group.gr_name == identities.worker_group
        and worker_group.gr_gid == identities.worker_gid
        and worker_gid_owner.gr_name == identities.worker_group
        and list(worker_group.gr_mem) == []
        and worker_passwd_rows == [expected_worker_passwd]
        and worker_primary_users == [identities.worker_user]
        and worker_group_rows == [expected_worker_group]
    )
    supplementary: list[int] | None = None
    if worker_present:
        supplementary = sorted(
            set(os.getgrouplist(identities.worker_user, identities.worker_gid))
        )
        if supplementary != [identities.worker_gid]:
            raise RuntimeError("capability worker has supplementary authority")
        worker_state = "present_exact"
    elif worker_group_present:
        worker_state = "group_present_user_absent_create_only_slot"
    elif worker_all_absent:
        worker_state = "absent_create_only_slot"
    else:
        raise RuntimeError("capability worker identity slot collides or drifted")

    client_absent = (
        client_group is None
        and client_gid_owner is None
        and client_primary_users == []
        and client_group_rows == []
    )
    client_present = (
        client_group is not None
        and client_gid_owner is not None
        and client_group.gr_name == identities.worker_client_group
        and client_group.gr_gid == identities.worker_client_gid
        and client_gid_owner.gr_name == identities.worker_client_group
        and list(client_group.gr_mem) == []
        and client_primary_users == []
        and client_group_rows == [expected_client_group]
    )
    if client_present:
        client_state = "present_exact"
    elif client_absent:
        client_state = "absent_create_only_slot"
    else:
        raise RuntimeError("capability worker client group collides or drifted")
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


def service_host_identity_receipt(
    plan: CapabilityCanaryPlan,
    full_plan: FullCanaryPlan,
    *,
    role: str,
    allow_create_only_absence: bool,
    expected_group_members: Sequence[str] = (),
) -> Mapping[str, Any]:
    """Attest an exact or collision-free mac-ops/connector identity slot."""

    validate_plan_against_full(plan, full_plan)
    if role == "mac_ops":
        user_name = plan.identities.mac_ops_user
        group_name = plan.identities.mac_ops_group
        uid = plan.identities.mac_ops_uid
        gid = plan.identities.mac_ops_gid
    elif role == "connector":
        user_name = plan.identities.connector_user
        group_name = plan.identities.connector_group
        uid = plan.identities.connector_uid
        gid = plan.identities.connector_gid
    else:
        raise ValueError("capability service identity role is invalid")

    user = _optional_passwd_by_name(user_name)
    group = _optional_group_by_name(group_name)
    uid_owner = _optional_passwd_by_uid(uid)
    gid_owner = _optional_group_by_gid(gid)
    expected_members = sorted(set(expected_group_members))
    if len(expected_members) != len(tuple(expected_group_members)) or any(
        not isinstance(member, str) or not member for member in expected_members
    ):
        raise ValueError("capability service expected group members are invalid")
    passwd_slot_rows, primary_group_users = _capability_passwd_slot_inventory(
        user_name,
        uid,
        gid,
    )
    group_slot_rows = _capability_group_slot_inventory(group_name, gid)
    expected_passwd_row = (
        user_name,
        uid,
        gid,
        "/nonexistent",
        "/usr/sbin/nologin",
    )
    expected_group_row = (group_name, gid, tuple(expected_members))
    all_absent = all(item is None for item in (user, group, uid_owner, gid_owner)) and (
        passwd_slot_rows == [] and primary_group_users == [] and group_slot_rows == []
    )
    group_present_user_absent = (
        user is None
        and uid_owner is None
        and group is not None
        and gid_owner is not None
        and group.gr_name == group_name
        and group.gr_gid == gid
        and gid_owner.gr_name == group_name
        and sorted(group.gr_mem) == expected_members
        and passwd_slot_rows == []
        and primary_group_users == []
        and group_slot_rows == [expected_group_row]
    )
    present_exact = (
        user is not None
        and uid_owner is not None
        and group is not None
        and gid_owner is not None
        and user.pw_name == user_name
        and user.pw_uid == uid
        and user.pw_gid == gid
        and user.pw_dir == "/nonexistent"
        and user.pw_shell == "/usr/sbin/nologin"
        and uid_owner.pw_name == user_name
        and group.gr_name == group_name
        and group.gr_gid == gid
        and gid_owner.gr_name == group_name
        and sorted(group.gr_mem) == expected_members
        and passwd_slot_rows == [expected_passwd_row]
        and primary_group_users == [user_name]
        and group_slot_rows == [expected_group_row]
    )
    supplementary: list[int] | None = None
    if present_exact:
        supplementary = sorted(set(os.getgrouplist(user_name, gid)))
        if supplementary != [gid]:
            raise RuntimeError(f"capability {role} has supplementary authority")
        state = "present_exact"
    elif group_present_user_absent:
        state = "group_present_user_absent_create_only_slot"
    elif all_absent:
        state = "absent_create_only_slot"
    else:
        raise RuntimeError(f"capability {role} identity slot collides or drifted")
    if state != "present_exact" and not allow_create_only_absence:
        raise RuntimeError(f"capability {role} principal is absent")
    unsigned = {
        "schema": CAPABILITY_SERVICE_HOST_IDENTITY_SCHEMA,
        "plan_sha256": plan.sha256,
        "role": role,
        "state": state,
        "user": user_name,
        "group": group_name,
        "uid": uid,
        "gid": gid,
        "home": "/nonexistent",
        "shell": "/usr/sbin/nologin",
        "group_members": expected_members if group is not None else None,
        "supplementary_group_ids": supplementary,
        "create_only_eligible": True,
        "secret_material_recorded": False,
    }
    return {**unsigned, "receipt_sha256": _sha256_json(unsigned)}


_SERVICE_IDENTITY_FOUNDATION_FIELDS = frozenset({
    "schema",
    "operation",
    "revision",
    "capability_plan_sha256",
    "full_canary_plan_sha256",
    "full_canary_terminal_receipt_sha256",
    "original_full_canary_owner_approval_sha256",
    "plan_publication_receipt_sha256",
    "receipt_path",
    "before",
    "after",
    "created",
    "create_only",
    "existing_identities_mutated",
    "retained_dormant_on_rollback",
    "mutation_performed",
    "secret_material_recorded",
    "secret_digest_recorded",
    "receipt_sha256",
})


def _service_identity_foundation_path(
    plan: CapabilityCanaryPlan,
) -> Path:
    return (
        DEFAULT_SERVICE_IDENTITY_FOUNDATION_ROOT
        / plan.revision
        / plan.sha256
        / "foundation.json"
    )


def _service_identity_values(
    plan: CapabilityCanaryPlan,
    role: str,
) -> tuple[str, str, int, int]:
    if role == "mac_ops":
        return (
            plan.identities.mac_ops_user,
            plan.identities.mac_ops_group,
            plan.identities.mac_ops_uid,
            plan.identities.mac_ops_gid,
        )
    if role == "connector":
        return (
            plan.identities.connector_user,
            plan.identities.connector_group,
            plan.identities.connector_uid,
            plan.identities.connector_gid,
        )
    raise ValueError("capability service identity role is invalid")


def _validate_service_host_identity_observation(
    value: Any,
    *,
    plan: CapabilityCanaryPlan,
    role: str,
    require_present: bool,
) -> Mapping[str, Any]:
    raw = _strict_mapping(
        value,
        {
            "schema",
            "plan_sha256",
            "role",
            "state",
            "user",
            "group",
            "uid",
            "gid",
            "home",
            "shell",
            "group_members",
            "supplementary_group_ids",
            "create_only_eligible",
            "secret_material_recorded",
            "receipt_sha256",
        },
        f"capability {role} service identity observation",
    )
    user, group, uid, gid = _service_identity_values(plan, role)
    state = raw["state"]
    allowed_states = {
        "present_exact",
        "group_present_user_absent_create_only_slot",
        "absent_create_only_slot",
    }
    if (
        raw["schema"] != CAPABILITY_SERVICE_HOST_IDENTITY_SCHEMA
        or raw["plan_sha256"] != plan.sha256
        or raw["role"] != role
        or state not in allowed_states
        or (require_present and state != "present_exact")
        or raw["user"] != user
        or raw["group"] != group
        or raw["uid"] != uid
        or raw["gid"] != gid
        or raw["home"] != "/nonexistent"
        or raw["shell"] != "/usr/sbin/nologin"
        or raw["create_only_eligible"] is not True
        or raw["secret_material_recorded"] is not False
        or (
            state == "absent_create_only_slot"
            and (
                raw["group_members"] is not None
                or raw["supplementary_group_ids"] is not None
            )
        )
        or (
            state == "group_present_user_absent_create_only_slot"
            and (
                raw["group_members"] != [] or raw["supplementary_group_ids"] is not None
            )
        )
        or (
            state == "present_exact"
            and (raw["group_members"] != [] or raw["supplementary_group_ids"] != [gid])
        )
    ):
        raise RuntimeError(f"capability {role} service identity observation is invalid")
    _validate_self_digest(
        raw,
        "receipt_sha256",
        f"capability {role} service identity observation",
    )
    return copy.deepcopy(dict(raw))


def _validate_service_identity_foundation_receipt(
    value: Any,
    *,
    plan: CapabilityCanaryPlan,
    full_plan: FullCanaryPlan,
    plan_publication_receipt: Mapping[str, Any],
    receipt_path: Path,
) -> Mapping[str, Any]:
    validate_plan_against_full(plan, full_plan)
    raw = _strict_mapping(
        value,
        _SERVICE_IDENTITY_FOUNDATION_FIELDS,
        "capability service identity foundation receipt",
    )
    before = _strict_mapping(
        raw["before"],
        {"mac_ops", "connector"},
        "capability service identity foundation before observations",
    )
    after = _strict_mapping(
        raw["after"],
        {"mac_ops", "connector"},
        "capability service identity foundation after observations",
    )
    expected_created: list[str] = []
    for role in ("mac_ops", "connector"):
        before_item = _validate_service_host_identity_observation(
            before[role], plan=plan, role=role, require_present=False
        )
        _validate_service_host_identity_observation(
            after[role], plan=plan, role=role, require_present=True
        )
        if before_item["state"] == "absent_create_only_slot":
            expected_created.append(f"{role}_group")
        if before_item["state"] != "present_exact":
            expected_created.append(f"{role}_user")
    unsigned = {
        key: copy.deepcopy(item) for key, item in raw.items() if key != "receipt_sha256"
    }
    if (
        raw["schema"] != CAPABILITY_SERVICE_IDENTITY_FOUNDATION_SCHEMA
        or raw["operation"] != "create_only_service_principals"
        or raw["revision"] != plan.revision
        or raw["capability_plan_sha256"] != plan.sha256
        or raw["full_canary_plan_sha256"] != full_plan.sha256
        or raw["full_canary_terminal_receipt_sha256"]
        != plan.full_canary_terminal_receipt_sha256
        or raw["original_full_canary_owner_approval_sha256"]
        != plan.original_full_canary_owner_approval_sha256
        or raw["plan_publication_receipt_sha256"]
        != plan_publication_receipt.get("receipt_sha256")
        or raw["receipt_path"] != str(receipt_path)
        or raw["created"] != expected_created
        or raw["create_only"] is not True
        or raw["existing_identities_mutated"] is not False
        or raw["retained_dormant_on_rollback"] is not True
        or raw["mutation_performed"] is not bool(expected_created)
        or raw["secret_material_recorded"] is not False
        or raw["secret_digest_recorded"] is not False
        or raw["receipt_sha256"] != _sha256_json(unsigned)
    ):
        raise RuntimeError("capability service identity foundation receipt drifted")
    return copy.deepcopy(dict(raw))


def load_service_identity_foundation_receipt(
    plan: CapabilityCanaryPlan,
    full_plan: FullCanaryPlan,
    *,
    plan_publication_loader: Callable[[CapabilityCanaryPlan], Mapping[str, Any]]
    | None = None,
) -> Mapping[str, Any]:
    """Load the immutable create-only principal receipt and bind it to plan."""

    loader = (
        load_bound_plan_publication_receipt
        if plan_publication_loader is None
        else plan_publication_loader
    )
    publication = loader(plan)
    receipt_path = _service_identity_foundation_path(plan)
    raw, _ = _read_stable_file(
        receipt_path,
        maximum=256 * 1024,
        expected_uid=0,
        expected_gid=0,
        allowed_modes=frozenset({0o400}),
    )
    value = _decode_json(raw, label="capability service identity foundation")
    if raw != _canonical_bytes(value):
        raise RuntimeError("capability service identity foundation is not canonical")
    return _validate_service_identity_foundation_receipt(
        value,
        plan=plan,
        full_plan=full_plan,
        plan_publication_receipt=publication,
        receipt_path=receipt_path,
    )


def ensure_service_identities_create_only(
    plan: CapabilityCanaryPlan,
    full_plan: FullCanaryPlan,
    *,
    runner: Callable[[Command], subprocess.CompletedProcess[bytes]] = _runner,
    observer: Callable[..., Mapping[str, Any]] = service_host_identity_receipt,
    plan_publication_loader: Callable[[CapabilityCanaryPlan], Mapping[str, Any]]
    | None = None,
    publisher: Callable[[Path, bytes], None] | None = None,
) -> Mapping[str, Any]:
    """Create only the two absent service principals after plan publication."""

    _require_root_linux()
    validate_plan_against_full(plan, full_plan)
    loader = (
        load_bound_plan_publication_receipt
        if plan_publication_loader is None
        else plan_publication_loader
    )
    publication = loader(plan)
    receipt_path = _service_identity_foundation_path(plan)
    publish = _atomic_publish_root_file if publisher is None else publisher
    with _lifecycle_lock():
        if os.path.lexists(receipt_path):
            return load_service_identity_foundation_receipt(
                plan,
                full_plan,
                plan_publication_loader=lambda _plan: publication,
            )
        before = {
            role: _validate_service_host_identity_observation(
                observer(
                    plan,
                    full_plan,
                    role=role,
                    allow_create_only_absence=True,
                ),
                plan=plan,
                role=role,
                require_present=False,
            )
            for role in ("mac_ops", "connector")
        }
        created: list[str] = []
        for role in ("mac_ops", "connector"):
            user, group, uid, gid = _service_identity_values(plan, role)
            state = before[role]["state"]
            if state == "absent_create_only_slot":
                _run_checked(
                    Command((
                        GROUPADD,
                        "--system",
                        "--gid",
                        str(gid),
                        "--",
                        group,
                    )),
                    runner=runner,
                    label=f"create capability {role} group",
                )
                created.append(f"{role}_group")
            if state != "present_exact":
                _run_checked(
                    Command((
                        USERADD,
                        "--system",
                        "--uid",
                        str(uid),
                        "--gid",
                        group,
                        "--home-dir",
                        "/nonexistent",
                        "--no-create-home",
                        "--shell",
                        "/usr/sbin/nologin",
                        "--",
                        user,
                    )),
                    runner=runner,
                    label=f"create capability {role} user",
                )
                created.append(f"{role}_user")
        after = {
            role: _validate_service_host_identity_observation(
                observer(
                    plan,
                    full_plan,
                    role=role,
                    allow_create_only_absence=False,
                ),
                plan=plan,
                role=role,
                require_present=True,
            )
            for role in ("mac_ops", "connector")
        }
        unsigned = {
            "schema": CAPABILITY_SERVICE_IDENTITY_FOUNDATION_SCHEMA,
            "operation": "create_only_service_principals",
            "revision": plan.revision,
            "capability_plan_sha256": plan.sha256,
            "full_canary_plan_sha256": full_plan.sha256,
            "full_canary_terminal_receipt_sha256": (
                plan.full_canary_terminal_receipt_sha256
            ),
            "original_full_canary_owner_approval_sha256": (
                plan.original_full_canary_owner_approval_sha256
            ),
            "plan_publication_receipt_sha256": publication["receipt_sha256"],
            "receipt_path": str(receipt_path),
            "before": before,
            "after": after,
            "created": created,
            "create_only": True,
            "existing_identities_mutated": False,
            "retained_dormant_on_rollback": True,
            "mutation_performed": bool(created),
            "secret_material_recorded": False,
            "secret_digest_recorded": False,
        }
        receipt = _validate_service_identity_foundation_receipt(
            {**unsigned, "receipt_sha256": _sha256_json(unsigned)},
            plan=plan,
            full_plan=full_plan,
            plan_publication_receipt=publication,
            receipt_path=receipt_path,
        )
        publish(receipt_path, _canonical_bytes(receipt))
        return copy.deepcopy(dict(receipt))


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
            Command((
                GROUPADD,
                "--system",
                "--gid",
                str(plan.identities.worker_gid),
                "--",
                plan.identities.worker_group,
            )),
            runner=runner,
            label="create capability worker group",
        )
        created.append("worker_group")
    if before["socket_client_group"]["state"] == "absent_create_only_slot":
        _run_checked(
            Command((
                GROUPADD,
                "--system",
                "--gid",
                str(plan.identities.worker_client_gid),
                "--",
                plan.identities.worker_client_group,
            )),
            runner=runner,
            label="create capability worker client group",
        )
        created.append("worker_client_group")
    if before["worker"]["state"] != "present_exact":
        _run_checked(
            Command((
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
            )),
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
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
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
    values = {name: reader(path) for name, path in _BROWSER_USERNS_SYSCTLS.items()}
    if values["unprivileged_userns_clone"] != 1 or values["max_user_namespaces"] <= 0:
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
            raise RuntimeError("isolated worker tmpfs mountpoint is absent") from None
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
            matches.append((
                set(fields[5].split(",")),
                set(fields[separator + 3].split(",")),
            ))
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
    expected = f"Google Chrome for Testing {RELEASE_CHROME_VERSION}\n".encode("ascii")
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

    if (
        effective_uid() != plan.identities.gateway_uid
        or effective_gid() != plan.identities.gateway_gid
    ):
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
    value = _decode_json(completed.stdout[:-1], label="capability execution readiness")
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
        != _sha256_json({
            key: item for key, item in value.items() if key != "receipt_sha256"
        })
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
        Command((
            SYSTEMCTL,
            "show",
            *(f"--property={name}" for name in _SERVICE_PROPERTIES),
            "--",
            unit,
        )),
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
    missing = set(_SERVICE_PROPERTIES) - set(values)
    unexpected = set(values) - set(_SERVICE_PROPERTIES)
    defaults = _PROCESSLESS_UNIT_PROPERTY_DEFAULTS.get(unit, {})
    if unexpected or not missing <= set(defaults):
        raise RuntimeError("capability service state fields are not exact")
    for name in missing:
        values[name] = defaults[name]
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
        unit: copy.deepcopy(dict(services[unit])) for unit in CAPABILITY_STOP_ORDER
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
        _service_stopped(services[unit]) for unit in CAPABILITY_PRE_CLEANUP_STOP_ORDER
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
    contract_sha256 = _sha256_json(_producer_credential_inaccessibility_contract())
    unsigned = {
        "schema": CAPABILITY_CREDENTIAL_CONSUMER_STOP_PROOF_SCHEMA,
        "plan_sha256": plan.sha256,
        "non_observer_stop_order": list(CAPABILITY_PRE_CLEANUP_STOP_ORDER),
        "non_observer_services_state_sha256": _sha256_json(non_observer_state),
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
        raise PermissionError(
            "credential retirement lacks an exact post-install stop proof"
        )
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
        or prior["expires_at_unix"]
        < (int(time.time()) if now_unix is None else now_unix)
        + CAPABILITY_ACTIVE_USE_MIN_RESERVE_SECONDS
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
    expected_read_peers = sorted([
        plan.identities.mac_ops_uid,
        full_plan.identities.writer_uid,
    ])
    if (
        _sha256_bytes(config_raw) != plan.bitrix_operational_edge_rendered_config_sha256
        or config.get("allowed_read_peer_uids") != expected_read_peers
        or config.get("mutation_peer_uid") != full_plan.identities.writer_uid
        or config.get("service_uid") != plan.identities.bitrix_operational_edge_uid
        or config.get("service_gid") != plan.identities.bitrix_operational_edge_gid
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
        "allowed_read_peer_uids": sorted(expected_read_peers),
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
    adapter_factory: Callable[..., Any] = (DiscordRestEdgeAdapter.from_credential_file),
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
    if (
        len({
            observed_bot_user_id,
            plan.connector_bot_user_id,
            PRODUCTION_DISCORD_BOT_USER_ID,
        })
        != 3
    ):
        raise RuntimeError("live Discord route-back bot identity is not isolated")

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
        or identity.get("planned_routeback_bot_user_id") != plan.routeback_bot_user_id
        or identity.get("connector_bot_user_id") != plan.connector_bot_user_id
        or identity.get("production_bot_user_id") != PRODUCTION_DISCORD_BOT_USER_ID
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
        or receipt.get("allowed_guild_ids") != list(plan.connector_allowed_guild_ids)
        or receipt.get("allowed_channel_ids")
        != list(plan.connector_allowed_channel_ids)
        or receipt.get("allowed_user_ids") != list(plan.connector_allowed_user_ids)
        or receipt.get("discord", {}).get("reviewed_cron_history_targets_sha256")
        != _sha256_json({})
        or history_reader is None
        or receipt.get("canary_history_reader") != history_reader.readiness_mapping()
        or receipt.get("discord", {}).get("bot_user_id") != plan.connector_bot_user_id
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
    if not isinstance(target_proofs, list) or len(target_proofs) != len(
        plan.connector_allowed_channel_ids
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
        "canary_history_reader_sha256": _sha256_json(receipt["canary_history_reader"]),
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
    if any(
        type(count) is not int or count < 0
        for count in (*events.values(), *sends.values())
    ):
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
    for name, (
        path,
        payload,
        mode,
        uid,
        gid,
        previous,
    ) in _capability_artifact_bindings(plan, full_plan).items():
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
    for name, (
        path,
        _payload,
        _mode,
        _uid,
        _gid,
        _previous,
    ) in _capability_artifact_bindings(plan, full_plan).items():
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
    if stage not in CAPABILITY_LIFECYCLE_STAGES:
        raise ValueError("capability lifecycle receipt stage is invalid")
    try:
        plan_publication_receipt_sha256: str | None = (
            load_bound_plan_publication_receipt(plan)["receipt_sha256"]
        )
        service_identity_foundation_receipt_sha256: str | None = (
            load_service_identity_foundation_receipt(plan, load_full_canary_plan())[
                "receipt_sha256"
            ]
        )
    except FileNotFoundError:
        if value.get("operation") != "partial_or_prestart_secret_retirement":
            raise
        # This cleanup receipt is also used to prove safe retirement after a
        # failed pre-publication attempt, where no plan publication can exist.
        plan_publication_receipt_sha256 = None
        service_identity_foundation_receipt_sha256 = None
    directory = DEFAULT_LIFECYCLE_RECEIPT_ROOT / plan.revision / plan.sha256 / stage
    _ensure_root_directory(directory)
    path = directory / f"{time.time_ns()}-{os.getpid()}-{uuid.uuid4().hex}.json"
    unsigned = {
        **copy.deepcopy(dict(value)),
        "schema": CAPABILITY_LIFECYCLE_RECEIPT_SCHEMA,
        "stage": stage,
        "revision": plan.revision,
        "plan_sha256": plan.sha256,
        "full_canary_plan_sha256": plan.full_canary_plan_sha256,
        "full_canary_terminal_receipt": copy.deepcopy(
            dict(plan.full_canary_terminal_receipt)
        ),
        "full_canary_terminal_receipt_sha256": (
            plan.full_canary_terminal_receipt_sha256
        ),
        "original_full_canary_owner_approval_sha256": (
            plan.original_full_canary_owner_approval_sha256
        ),
        "plan_publication_receipt_sha256": plan_publication_receipt_sha256,
        "service_identity_foundation_receipt_sha256": (
            service_identity_foundation_receipt_sha256
        ),
        "receipt_path": str(path),
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    receipt = {**unsigned, "receipt_sha256": _sha256_json(unsigned)}
    _write_exclusive_bytes(path, _canonical_bytes(receipt), mode=0o400)
    return receipt


_MAX_LIFECYCLE_STAGE_RECEIPTS = 4_096
_LIFECYCLE_RECEIPT_NAME_RE = re.compile(r"^[0-9]+-[0-9]+-[0-9a-f]{32}\.json$")


def _load_lifecycle_stage_receipts(
    plan: CapabilityCanaryPlan,
    *,
    stage: str,
) -> tuple[Mapping[str, Any], ...]:
    """Load an immutable lifecycle stage inventory as durable authority."""

    if stage not in CAPABILITY_LIFECYCLE_STAGES:
        raise ValueError("capability lifecycle receipt stage is invalid")
    directory = DEFAULT_LIFECYCLE_RECEIPT_ROOT / plan.revision / plan.sha256 / stage
    try:
        names = sorted(os.listdir(directory))
    except FileNotFoundError:
        return ()
    if len(names) > _MAX_LIFECYCLE_STAGE_RECEIPTS or any(
        _LIFECYCLE_RECEIPT_NAME_RE.fullmatch(name) is None for name in names
    ):
        raise RuntimeError("capability lifecycle receipt inventory is invalid")
    result: list[Mapping[str, Any]] = []
    for name in names:
        path = directory / name
        raw, _item = _read_stable_file(
            path,
            maximum=2 * 1024 * 1024,
            expected_uid=effective_uid(),
            expected_gid=effective_gid(),
            allowed_modes=frozenset({0o400}),
        )
        value = _decode_json(raw, label=f"capability lifecycle {stage} receipt")
        if not isinstance(value, Mapping) or raw != _canonical_bytes(value):
            raise RuntimeError("capability lifecycle receipt is not canonical")
        unsigned = {
            key: copy.deepcopy(item)
            for key, item in value.items()
            if key != "receipt_sha256"
        }
        if (
            value.get("schema") != CAPABILITY_LIFECYCLE_RECEIPT_SCHEMA
            or value.get("stage") != stage
            or value.get("revision") != plan.revision
            or value.get("plan_sha256") != plan.sha256
            or value.get("full_canary_plan_sha256")
            != plan.full_canary_plan_sha256
            or value.get("full_canary_terminal_receipt")
            != plan.full_canary_terminal_receipt
            or value.get("full_canary_terminal_receipt_sha256")
            != plan.full_canary_terminal_receipt_sha256
            or value.get("original_full_canary_owner_approval_sha256")
            != plan.original_full_canary_owner_approval_sha256
            or value.get("receipt_path") != str(path)
            or value.get("secret_material_recorded") is not False
            or value.get("secret_digest_recorded") is not False
            or value.get("receipt_sha256") != _sha256_json(unsigned)
        ):
            raise RuntimeError("capability lifecycle receipt drifted")
        result.append(copy.deepcopy(dict(value)))
    return tuple(result)


def _deferred_core_receipt_to_pending(
    plan: CapabilityCanaryPlan,
    approval: CapabilityCanaryOwnerApproval,
    core_receipt: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Rebuild phase-two state solely from the append-only core receipt."""

    mapping_fields = {
        "installed_artifacts": "installed",
        "connector_state": "connector_state",
        "phase_b_current_readiness": "phase_b_current",
        "phase_b_full_canary_anchor": "installed_phase_b_anchor",
        "writer_runtime_readiness": "writer_readiness",
        "gateway_runtime_readiness": "gateway_readiness",
        "observer_config": "observer",
        "browser_identity_foundation": "browser_identity_foundation",
        "browser_principal_smoke": "browser_principal_smoke",
        "execution_identity_foundation": "execution_identity_foundation",
        "worker_mountpoint": "worker_mountpoint",
        "execution_readiness": "execution_readiness",
        "routeback_bot_identity": "routeback_bot_identity",
        "producer_foundation": "producer_foundation",
    }
    if (
        core_receipt.get("stage") != CAPABILITY_GATEWAY_CORE_READY_STAGE
        or core_receipt.get("operation") != "start_core_before_api_admission"
        or core_receipt.get("plan_sha256") != plan.sha256
        or core_receipt.get("owner_approval_sha256") != approval.sha256
        or core_receipt.get("producer_units_started") is not False
        or core_receipt.get("api_admission_pending") is not True
        or core_receipt.get("core_start_order")
        != list(CAPABILITY_DEFERRED_CORE_START_ORDER)
        or any(
            not isinstance(core_receipt.get(source), Mapping)
            for source in mapping_fields
        )
        or _SHA256_RE.fullmatch(
            str(core_receipt.get("full_canary_stopped_preflight_sha256", ""))
        )
        is None
        or _SHA256_RE.fullmatch(
            str(core_receipt.get("stopped_preflight_sha256", ""))
        )
        is None
    ):
        raise RuntimeError("deferred core lifecycle receipt drifted")
    pending: dict[str, Any] = {
        "approval_sha256": approval.sha256,
        "core_receipt": copy.deepcopy(dict(core_receipt)),
        "full_preflight": {
            "report_sha256": core_receipt["full_canary_stopped_preflight_sha256"]
        },
        "preflight": {"report_sha256": core_receipt["stopped_preflight_sha256"]},
        "started": list(CAPABILITY_DEFERRED_CORE_START_ORDER),
    }
    for source, destination in mapping_fields.items():
        pending[destination] = copy.deepcopy(dict(core_receipt[source]))
    return pending


def _load_deferred_lifecycle_state(
    plan: CapabilityCanaryPlan,
    approval: CapabilityCanaryOwnerApproval,
) -> tuple[Mapping[str, Any] | None, Mapping[str, Any] | None]:
    """Return the single active core generation or its terminal start receipt."""

    cores = [
        receipt
        for receipt in _load_lifecycle_stage_receipts(
            plan, stage=CAPABILITY_GATEWAY_CORE_READY_STAGE
        )
        if receipt.get("operation") == "start_core_before_api_admission"
        and receipt.get("owner_approval_sha256") == approval.sha256
    ]
    started = [
        receipt
        for receipt in _load_lifecycle_stage_receipts(
            plan, stage=CAPABILITY_RUNTIME_PENDING_ACK_STAGE
        )
        if receipt.get("operation")
        == "runtime_live_pending_gateway_commit_ack"
        and receipt.get("owner_approval_sha256") == approval.sha256
    ]
    failed = [
        receipt
        for receipt in _load_lifecycle_stage_receipts(plan, stage="failure")
        if receipt.get("operation") == "complete_start_after_api_admission"
        and receipt.get("owner_approval_sha256") == approval.sha256
    ]
    started_by_core: dict[str, Mapping[str, Any]] = {}
    for receipt in started:
        core_sha256 = str(receipt.get("core_start_receipt_sha256", ""))
        if _SHA256_RE.fullmatch(core_sha256) is None:
            raise RuntimeError("terminal deferred start lacks its core binding")
        if core_sha256 in started_by_core:
            raise RuntimeError("duplicate terminal deferred lifecycle truth")
        started_by_core[core_sha256] = receipt
    failed_core_sha256s: set[str] = set()
    for receipt in failed:
        core_sha256 = str(receipt.get("core_start_receipt_sha256", ""))
        if _SHA256_RE.fullmatch(core_sha256) is None:
            raise RuntimeError("deferred start failure lacks its core binding")
        if receipt.get("cleanup_complete") is not True:
            raise RuntimeError(
                "deferred lifecycle failure still requires exact reconciliation"
            )
        if core_sha256 in failed_core_sha256s:
            raise RuntimeError("duplicate deferred lifecycle failure truth")
        failed_core_sha256s.add(core_sha256)
    if set(started_by_core) & failed_core_sha256s:
        raise RuntimeError("deferred lifecycle has contradictory terminal truth")
    active = [
        receipt
        for receipt in cores
        if receipt["receipt_sha256"] not in started_by_core
        and receipt["receipt_sha256"] not in failed_core_sha256s
    ]
    if len(active) > 1:
        raise RuntimeError("multiple deferred core lifecycle generations are active")
    if active:
        _deferred_core_receipt_to_pending(plan, approval, active[0])
        return active[0], None
    completed = [
        (core, started_by_core[core["receipt_sha256"]])
        for core in cores
        if core["receipt_sha256"] in started_by_core
    ]
    if not completed:
        return None, None
    core, terminal = completed[-1]
    _deferred_core_receipt_to_pending(plan, approval, core)
    return core, terminal


def _load_bound_deferred_stage(
    plan: CapabilityCanaryPlan,
    approval: CapabilityCanaryOwnerApproval,
    *,
    stage: str,
    operation: str,
    core_receipt_sha256: str,
) -> Mapping[str, Any] | None:
    """Load at most one immutable admission stage for one core generation."""

    _digest(core_receipt_sha256, "deferred lifecycle core receipt")
    matches = [
        receipt
        for receipt in _load_lifecycle_stage_receipts(plan, stage=stage)
        if receipt.get("operation") == operation
        and receipt.get("owner_approval_sha256") == approval.sha256
        and receipt.get("core_start_receipt_sha256") == core_receipt_sha256
    ]
    if len(matches) > 1:
        raise RuntimeError(f"duplicate deferred lifecycle {stage} truth")
    return matches[0] if matches else None


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
    *,
    kind: str,
    secret: bytes | bytearray,
    plan_sha256: str,
    owner_subject_sha256: str,
    now_unix: int | None = None,
    ttl_seconds: int = 900,
    lease_id: str | None = None,
) -> bytearray:
    if kind not in _SECRET_LEASE_MAGIC_BY_KIND:
        raise ValueError("secret lease kind is invalid")
    payload = bytes(secret)
    if not payload or len(payload) > _MAX_SECRET_BYTES:
        raise ValueError("secret lease payload size is invalid")
    issued = int(time.time()) if now_unix is None else now_unix
    if (
        type(issued) is not int
        or type(ttl_seconds) is not int
        or not 60 <= ttl_seconds <= _MAX_LEASE_SECONDS
    ):
        raise ValueError("secret lease window is invalid")
    lease = uuid.uuid4().hex if lease_id is None else lease_id
    if _LEASE_ID_RE.fullmatch(lease) is None:
        raise ValueError("secret lease id is invalid")
    token_expiry = _jwt_exp(payload) if kind == "codex_access_token" else None
    if token_expiry is not None and token_expiry < issued + ttl_seconds + 120:
        raise ValueError("Codex access token expires inside the canary window")
    metadata = {
        "schema": CAPABILITY_LEASE_FRAME_SCHEMA,
        "kind": kind,
        "plan_sha256": _digest(plan_sha256, "capability plan"),
        "owner_subject_sha256": _digest(owner_subject_sha256, "owner subject"),
        "lease_id": lease,
        "issued_at_unix": issued,
        "expires_at_unix": issued + ttl_seconds,
        "secret_bytes": len(payload),
        "token_expires_at_unix": token_expiry,
    }
    encoded = _canonical_bytes(metadata)
    magic = _SECRET_LEASE_MAGIC_BY_KIND[kind]
    return bytearray(
        magic + struct.pack(">II", len(encoded), len(payload)) + encoded + payload
    )


def read_secret_lease_frame(
    stream: BinaryIO,
    *,
    expected_kind: str,
    now_unix: int | None = None,
) -> tuple[Mapping[str, Any], bytearray]:
    header = stream.read(12)
    if len(header) != 12:
        raise ValueError("secret lease frame header is invalid")
    magic, metadata_size, secret_size = header[:4], *struct.unpack(">II", header[4:])
    try:
        expected_magic = _SECRET_LEASE_MAGIC_BY_KIND[expected_kind]
    except KeyError as exc:
        raise ValueError("secret lease kind is invalid") from exc
    if (
        magic != expected_magic
        or not 0 < metadata_size <= 64 * 1024
        or not 0 < secret_size <= _MAX_SECRET_BYTES
    ):
        raise ValueError("secret lease frame bounds are invalid")
    metadata_raw = stream.read(metadata_size)
    secret = bytearray(stream.read(secret_size))
    if (
        len(metadata_raw) != metadata_size
        or len(secret) != secret_size
        or stream.read(1)
    ):
        raise ValueError("secret lease frame length is invalid")
    metadata = _decode_json(metadata_raw, label="secret lease metadata")
    fields = {
        "schema",
        "kind",
        "plan_sha256",
        "owner_subject_sha256",
        "lease_id",
        "issued_at_unix",
        "expires_at_unix",
        "secret_bytes",
        "token_expires_at_unix",
    }
    _strict_mapping(metadata, fields, "secret lease metadata")
    now = int(time.time()) if now_unix is None else now_unix
    if (
        metadata_raw != _canonical_bytes(metadata)
        or metadata["schema"] != CAPABILITY_LEASE_FRAME_SCHEMA
        or metadata["kind"] != expected_kind
        or metadata["secret_bytes"] != secret_size
        or _LEASE_ID_RE.fullmatch(str(metadata["lease_id"])) is None
        or type(metadata["issued_at_unix"]) is not int
        or type(metadata["expires_at_unix"]) is not int
        or not metadata["issued_at_unix"] <= now < metadata["expires_at_unix"]
        or not 60
        <= metadata["expires_at_unix"] - metadata["issued_at_unix"]
        <= _MAX_LEASE_SECONDS
    ):
        raise ValueError("secret lease metadata is invalid")
    _digest(metadata["plan_sha256"], "capability plan")
    _digest(metadata["owner_subject_sha256"], "owner subject")
    if expected_kind == "codex_access_token":
        expiry = _jwt_exp(bytes(secret))
        if (
            metadata["token_expires_at_unix"] != expiry
            or expiry < metadata["expires_at_unix"] + 120
        ):
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
    install_abort: Path
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
    administrative_uid = effective_uid()
    administrative_gid = effective_gid()
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
    (
        path,
        default_journal,
        uid,
        gid,
        mode,
        parent_uid,
        parent_gid,
        parent_mode,
        maximum,
    ) = values
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
        uid=effective_uid(),
        gid=effective_gid(),
        mode=0o700,
    )


def _validate_journal_directory(path: Path) -> None:
    item = os.lstat(path)
    if (
        stat.S_ISLNK(item.st_mode)
        or not stat.S_ISDIR(item.st_mode)
        or item.st_uid
        != effective_uid()
        or item.st_gid
        != effective_gid()
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
        os.fchown(
            descriptor, effective_uid(), effective_gid()
        )
        item = os.fstat(descriptor)
        if (
            not stat.S_ISREG(item.st_mode)
            or item.st_nlink != 1
            or item.st_uid
            != effective_uid()
            or item.st_gid
            != effective_gid()
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
        if (
            renameat2(
                -100,
                os.fsencode(source),
                -100,
                os.fsencode(target),
                1,
            )
            != 0
        ):
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
        install_abort=root / "install-abort.json",
        retirement_intent=root / "retirement-intent.json",
        retirement_completion=root / "retirement-completion.json",
    )


def _load_lease_artifact(path: Path, *, schema: str) -> Mapping[str, Any]:
    raw, _ = _read_exact_file(
        path,
        maximum=64 * 1024,
        uid=effective_uid(),
        gid=effective_gid(),
        mode=0o400,
    )
    return _validate_lease_artifact_payload(
        raw,
        receipt_path=path,
        schema=schema,
    )


def _validate_lease_artifact_payload(
    raw: bytes,
    *,
    receipt_path: Path,
    schema: str,
) -> Mapping[str, Any]:
    value = _decode_json(raw, label="credential lease artifact")
    unsigned = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if (
        raw != _canonical_bytes(value)
        or value.get("schema") != schema
        or value.get("receipt_path") != str(receipt_path)
        or value.get("receipt_sha256") != _sha256_json(unsigned)
        or value.get("secret_material_recorded") is not False
        or value.get("secret_digest_recorded") is not False
    ):
        raise RuntimeError("credential lease artifact is invalid")
    return copy.deepcopy(dict(value))


def _reconcile_lease_artifact_temporary(
    path: Path,
    *,
    schema: str,
) -> None:
    """Finish an exact fsynced no-replace publication after process death."""

    temporary = path.parent / f".{path.name}.tmp"
    if not os.path.lexists(temporary):
        return
    temporary_raw, _ = _read_exact_file(
        temporary,
        maximum=64 * 1024,
        uid=effective_uid(),
        gid=effective_gid(),
        mode=0o400,
    )
    _validate_lease_artifact_payload(
        temporary_raw,
        receipt_path=path,
        schema=schema,
    )
    if not os.path.lexists(path):
        try:
            _rename_no_replace(temporary, path)
        except FileExistsError:
            pass
        else:
            _fsync_directory(path.parent)
    installed_raw, _ = _read_exact_file(
        path,
        maximum=64 * 1024,
        uid=effective_uid(),
        gid=effective_gid(),
        mode=0o400,
    )
    if installed_raw != temporary_raw:
        raise RuntimeError("credential publication half-state is inconsistent")
    _validate_lease_artifact_payload(
        installed_raw,
        receipt_path=path,
        schema=schema,
    )
    if os.path.lexists(temporary):
        os.unlink(temporary)
        _fsync_directory(path.parent)


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
        uid=effective_uid(),
        gid=effective_gid(),
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
            or directory.st_uid
            != effective_uid()
            or directory.st_gid
            != effective_gid()
            or stat.S_IMODE(directory.st_mode) != 0o700
        ):
            raise RuntimeError("credential lease artifact directory is unsafe")
        allowed = {
            paths.install_intent.name,
            paths.install_receipt.name,
            paths.install_abort.name,
            paths.retirement_intent.name,
            paths.retirement_completion.name,
            f".{paths.install_intent.name}.tmp",
            f".{paths.install_receipt.name}.tmp",
            f".{paths.install_abort.name}.tmp",
            f".{paths.retirement_intent.name}.tmp",
            f".{paths.retirement_completion.name}.tmp",
        }
        if not set(os.listdir(paths.root)).issubset(allowed):
            raise RuntimeError("credential lease artifact inventory is invalid")
        artifacts: dict[str, Mapping[str, Any] | None] = {}
        for field, path, schema in (
            (
                "install_intent",
                paths.install_intent,
                CAPABILITY_LEASE_INSTALL_INTENT_SCHEMA,
            ),
            ("install_receipt", paths.install_receipt, CAPABILITY_LEASE_RECEIPT_SCHEMA),
            (
                "install_abort",
                paths.install_abort,
                CAPABILITY_LEASE_INSTALL_ABORT_SCHEMA,
            ),
            (
                "retirement_intent",
                paths.retirement_intent,
                CAPABILITY_RETIREMENT_INTENT_SCHEMA,
            ),
            (
                "retirement_completion",
                paths.retirement_completion,
                CAPABILITY_RETIREMENT_RECEIPT_SCHEMA,
            ),
        ):
            _reconcile_lease_artifact_temporary(path, schema=schema)
            artifacts[field] = (
                _load_lease_artifact(path, schema=schema)
                if os.path.lexists(path)
                else None
            )
        if artifacts["install_intent"] is None:
            raise RuntimeError("credential lease journal has an orphan artifact")
        if (
            artifacts["install_receipt"] is not None
            and artifacts["install_abort"] is not None
        ):
            raise RuntimeError("credential lease has both install and abort terminals")
        if artifacts["install_abort"] is not None and any(
            artifacts[name] is not None
            for name in ("retirement_intent", "retirement_completion")
        ):
            raise RuntimeError("aborted credential lease has retirement artifacts")
        if artifacts["install_receipt"] is None and any(
            artifacts[name] is not None
            for name in ("retirement_intent", "retirement_completion")
        ):
            raise RuntimeError("credential lease retirement lacks an install receipt")
        if (
            artifacts["retirement_intent"] is None
            and artifacts["retirement_completion"] is not None
        ):
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
    return all(
        receipt.get(key) == value for key, value in _target_metadata(item).items()
    )


def _validate_lease_metadata(
    plan: CapabilityCanaryPlan,
    metadata: Mapping[str, Any],
    secret: bytearray,
    *,
    now_unix: int | None = None,
    minimum_reserve_seconds: int = 0,
) -> str:
    kind = metadata.get("kind")
    now = int(time.time()) if now_unix is None else now_unix
    if (
        metadata.get("schema") != CAPABILITY_LEASE_FRAME_SCHEMA
        or metadata.get("plan_sha256") != plan.sha256
        or kind not in _SECRET_LEASE_MAGIC_BY_KIND
        or _LEASE_ID_RE.fullmatch(str(metadata.get("lease_id"))) is None
        or type(metadata.get("issued_at_unix")) is not int
        or type(metadata.get("expires_at_unix")) is not int
        or not metadata["issued_at_unix"] <= now < metadata["expires_at_unix"]
        or not 60
        <= metadata["expires_at_unix"] - metadata["issued_at_unix"]
        <= _MAX_LEASE_SECONDS
        or metadata.get("secret_bytes") != len(secret)
    ):
        raise PermissionError("secret lease is not bound to this capability plan")
    if minimum_reserve_seconds:
        _require_remaining_reserve(
            expires_at_unix=metadata["expires_at_unix"],
            now_unix=now,
            minimum_seconds=minimum_reserve_seconds,
        )
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
        return _canonical_bytes({
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
        })
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
        or any(
            character.isspace() or ord(character) < 0x21 or ord(character) == 0x7F
            for character in token
        )
    ):
        raise ValueError("opaque capability token is invalid")
    return bytes(secret)


def _require_secret_provision_consumers_stopped() -> Mapping[str, Mapping[str, Any]]:
    """Freshly prove the closed capability consumer set is fully stopped.

    The production lifecycle lock serializes Hermes-controlled starts and
    stops.  A fresh systemd observation is still required before any lease
    mutation, and again immediately before publication, so an independently
    started or compromised consumer cannot retain a consumer-owned secret
    inode outside the append-only lease journal.
    """

    services = _capability_services(runner=_runner)
    if set(services) != set(CAPABILITY_STOP_ORDER) or not all(
        _service_stopped(services[unit]) for unit in CAPABILITY_STOP_ORDER
    ):
        raise RuntimeError(
            "secret lease provisioning requires all capability consumers stopped"
        )
    return {
        unit: copy.deepcopy(dict(services[unit]))
        for unit in CAPABILITY_STOP_ORDER
    }


def _provision_secret_lease_locked(
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
    operation_clock: _OperationClock,
) -> Mapping[str, Any]:
    kind = _validate_lease_metadata(
        plan,
        metadata,
        secret,
        now_unix=operation_clock.sample("credential lease admission"),
        minimum_reserve_seconds=CAPABILITY_MUTATION_MIN_RESERVE_SECONDS,
    )
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
        if production_targets:
            # This is deliberately before watchdog, journal, directory, or
            # temporary-inode creation.  The caller holds _lifecycle_lock().
            _require_secret_provision_consumers_stopped()
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
                now_unix=operation_clock.sample("credential watchdog arm"),
                minimum_reserve_seconds=(CAPABILITY_MUTATION_MIN_RESERVE_SECONDS),
            )
        with _lease_journal_lock(spec.journal):
            _validate_lease_metadata(
                plan,
                metadata,
                secret,
                now_unix=operation_clock.sample("credential journal admission"),
                minimum_reserve_seconds=CAPABILITY_MUTATION_MIN_RESERVE_SECONDS,
            )
            states = _journal_states(spec.journal)
            current = next(
                (item for item in states if item["lease_id"] == metadata["lease_id"]),
                None,
            )
            if current is None and len(states) >= _MAX_LEASE_ARTIFACTS:
                raise RuntimeError("credential lease journal admission is full")
            incomplete = [
                item
                for item in states
                if item["retirement_completion"] is None
                and item["install_abort"] is None
                and item["lease_id"] != metadata["lease_id"]
            ]
            if incomplete:
                raise RuntimeError("another secret lease remains active or incomplete")
            if current is not None and (
                current["retirement_completion"] is not None
                or current["install_abort"] is not None
            ):
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
                    "expiry_watchdog": copy.deepcopy(dict(expiry_watchdog or {})),
                },
            )
            _prepare_secret_parent(
                spec.path.parent,
                uid=spec.parent_uid,
                gid=spec.parent_gid,
                mode=spec.parent_mode,
            )
            _require_remaining_reserve(
                expires_at_unix=metadata["expires_at_unix"],
                now_unix=operation_clock.sample("credential publication"),
            )
            if production_targets:
                # Re-observe after journal/watchdog preparation and directly
                # before any existing target is accepted or a secret inode is
                # atomically published.
                _require_secret_provision_consumers_stopped()
            if os.path.lexists(spec.path):
                existing_item = os.lstat(spec.path)
                if stat.S_ISLNK(existing_item.st_mode) or not stat.S_ISREG(
                    existing_item.st_mode
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
                    raise RuntimeError(
                        "credential lease retry carries different secret bytes"
                    )
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
                    "expiry_watchdog": copy.deepcopy(dict(expiry_watchdog or {})),
                },
            )
            if not _receipt_matches_target(receipt, os.lstat(spec.path)):
                raise RuntimeError("credential install receipt target binding drifted")
            _require_remaining_reserve(
                expires_at_unix=metadata["expires_at_unix"],
                now_unix=operation_clock.sample("credential commit"),
            )
            return receipt
    finally:
        for index in range(len(secret)):
            secret[index] = 0
        payload = None


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
    clock: Callable[[], int] | None = None,
) -> Mapping[str, Any]:
    """Provision one lease while serialized with start, stop and expiry cleanup."""

    production_targets = (
        journal_path is None
        and auth_path == DEFAULT_GATEWAY_AUTH_STORE
        and mac_path == DEFAULT_MAC_OPS_CREDENTIAL
        and connector_path == DEFAULT_CONNECTOR_TOKEN
        and api_control_path == DEFAULT_API_SERVER_CONTROL_KEY
        and routeback_path == DEFAULT_EDGE_TOKEN_PATH
        and bitrix_path == DEFAULT_BITRIX_WEBHOOK_PATH
    )
    operation_clock = _OperationClock(clock)
    lock = _lifecycle_lock() if production_targets else nullcontext()
    try:
        with lock:
            try:
                return _provision_secret_lease_locked(
                    plan,
                    metadata,
                    secret,
                    full_plan=full_plan,
                    auth_path=auth_path,
                    mac_path=mac_path,
                    connector_path=connector_path,
                    api_control_path=api_control_path,
                    routeback_path=routeback_path,
                    bitrix_path=bitrix_path,
                    journal_path=journal_path,
                    operation_clock=operation_clock,
                )
            except BaseException as error:
                if production_targets and full_plan is not None:
                    try:
                        _compensate_failed_secret_provision_locked(
                            plan,
                            full_plan,
                            metadata=metadata,
                            now_unix=operation_clock.sample(
                                "credential provision compensation"
                            ),
                        )
                    except BaseException as cleanup_error:
                        raise BaseExceptionGroup(
                            "credential provision and synchronous retirement failed",
                            [error, cleanup_error],
                        ) from None
                raise
    finally:
        for index in range(len(secret)):
            secret[index] = 0


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
        raise RuntimeError(
            "credential lease journal does not identify one active install"
        )
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
            and (
                plan is None
                or state["install_receipt"].get("plan_sha256") == plan.sha256
            )
        ]
        if not active:
            if completed and not os.path.lexists(target):
                latest = max(
                    completed,
                    key=lambda item: (
                        item["retirement_completion"]["retired_at_unix"],
                        item["install_receipt"]["installed_at_unix"],
                        item["lease_id"],
                    ),
                )
                return latest["retirement_completion"]
            raise RuntimeError("credential retirement lacks one active install receipt")
        if len(active) != 1:
            raise RuntimeError("credential retirement active lease is ambiguous")
        state = active[0]
        install = state["install_receipt"]
        paths = state["paths"]
        if install.get("credential_binding") != _CREDENTIAL_BINDING_BY_KIND[kind] or (
            plan is not None
            and install.get("full_canary_plan_sha256") != plan.full_canary_plan_sha256
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
                raise RuntimeError(
                    "credential disappeared before retirement intent"
                ) from exc
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
                    **{key: install[key] for key in _target_metadata(item)},
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
                or intent["requested_at_unix"] < intent_stop_proof["observed_at_unix"]
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
        stop_observed_at_unix = intent["service_stop_proof"]["observed_at_unix"]
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
                "service_stop_proof_sha256": intent["service_stop_proof_sha256"],
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
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
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
    now_unix: int | None = None,
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
    active_install_bound = False
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
            if state["retirement_completion"] is None and state["install_abort"] is None
        ]
        if len(unfinished) > 1:
            raise RuntimeError("credential slot has multiple unfinished leases")
        if unfinished:
            current = unfinished[0]
            if current["install_receipt"] is not None:
                active_install_bound = True
            else:
                incomplete = current
        else:
            completed_history = [
                state
                for state in matching
                if state["retirement_completion"] is not None
            ]

        aborted_history = [
            state for state in matching if state["install_abort"] is not None
        ]

        if incomplete is not None:
            intent = incomplete["install_intent"]
            if (
                intent.get("credential_binding") != spec.credential_binding
                or intent.get("revision") != plan.revision
                or intent.get("full_canary_plan_sha256") != plan.full_canary_plan_sha256
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
                    removed.append(_remove_incomplete_lease_file(candidate, spec=spec))
            if os.path.lexists(spec.path) or os.path.lexists(temporary):
                raise RuntimeError("incomplete credential lease cleanup is incomplete")
            abort = _append_lease_artifact(
                incomplete["paths"].install_abort,
                schema=CAPABILITY_LEASE_INSTALL_ABORT_SCHEMA,
                value={
                    "operation": "install_abort",
                    "state": "never_committed",
                    "reason": "cleanup_of_incomplete_install",
                    "kind": spec.kind,
                    "credential_binding": spec.credential_binding,
                    "revision": plan.revision,
                    "plan_sha256": plan.sha256,
                    "full_canary_plan_sha256": plan.full_canary_plan_sha256,
                    "lease_id": incomplete["lease_id"],
                    "issued_at_unix": intent["issued_at_unix"],
                    "expires_at_unix": intent["expires_at_unix"],
                    "target_path": str(spec.path),
                    "temporary_path": str(temporary),
                    "target_absent": not os.path.lexists(spec.path),
                    "temporary_absent": not os.path.lexists(temporary),
                    "install_intent_path": intent["receipt_path"],
                    "install_intent_sha256": intent["receipt_sha256"],
                    "expiry_watchdog": copy.deepcopy(
                        dict(intent.get("expiry_watchdog", {}))
                    ),
                    "aborted_at_unix": (
                        int(time.time()) if now_unix is None else now_unix
                    ),
                },
            )
            unsigned = {
                "kind": spec.kind,
                "credential_binding": spec.credential_binding,
                "target_path": str(spec.path),
                "state": "incomplete_install_retired",
                "lease_id": incomplete["lease_id"],
                "install_intent_path": intent["receipt_path"],
                "install_intent_sha256": intent["receipt_sha256"],
                "removed_objects": removed,
                "install_abort": abort,
                "install_abort_receipt_sha256": abort["receipt_sha256"],
                "install_bound_retirement": False,
                "absent": True,
                "secret_material_recorded": False,
                "secret_digest_recorded": False,
            }
            return {**unsigned, "observation_sha256": _sha256_json(unsigned)}

    if active_install_bound:
        completion = retire_secret_lease(
            kind=spec.kind,
            target=spec.path,
            journal=spec.journal,
            stop_proof=stop_proof,
            plan=plan,
            now_unix=now_unix,
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

    if completed_history:
        if os.path.lexists(spec.path):
            raise RuntimeError("retired credential target reappeared")
        completed_history = sorted(
            completed_history,
            key=lambda item: (
                item["retirement_completion"]["retired_at_unix"],
                item["install_receipt"]["installed_at_unix"],
                item["lease_id"],
            ),
        )
        completions = [state["retirement_completion"] for state in completed_history]
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
    if aborted_history:
        if os.path.lexists(spec.path):
            raise RuntimeError("aborted credential target reappeared")
        aborted_history = sorted(aborted_history, key=lambda item: item["lease_id"])
        aborts = [state["install_abort"] for state in aborted_history]
        return {
            "kind": spec.kind,
            "credential_binding": spec.credential_binding,
            "target_path": str(spec.path),
            "state": "install_aborted",
            "install_bound_retirement": False,
            "install_abort": aborts[-1],
            "install_abort_receipt_sha256": aborts[-1]["receipt_sha256"],
            "install_abort_history_receipt_sha256s": [
                abort["receipt_sha256"] for abort in aborts
            ],
            "absent": True,
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
    now_unix: int | None = None,
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
                now_unix=now_unix,
            )
        except BaseException as exc:
            errors[spec.credential_binding] = _sha256_bytes(
                f"{type(exc).__name__}:{exc}".encode("utf-8", errors="replace")
            )
        finally:
            current = dict(slots.get(spec.credential_binding, {}))
            current.update({
                "kind": spec.kind,
                "credential_binding": spec.credential_binding,
                "target_path": str(spec.path),
                "absent": not os.path.lexists(spec.path),
            })
            slots[spec.credential_binding] = current
    all_absent = all(slot["absent"] for slot in slots.values())
    result = _write_lifecycle_receipt(
        plan,
        stage="stopped" if not errors and all_absent else "failure",
        value={
            "operation": "partial_or_prestart_secret_retirement",
            "service_stop_proof": copy.deepcopy(dict(exact_stop_proof)),
            "service_stop_proof_sha256": exact_stop_proof["stop_proof_sha256"],
            "slots": slots,
            "error_sha256s": errors,
            "all_six_credentials_absent_readback": all_absent,
            "all_six_install_bound_retirement_completions": all(
                slot.get("install_bound_retirement") is True for slot in slots.values()
            ),
            "ok": not errors and all_absent,
            "completed_at_unix": (int(time.time()) if now_unix is None else now_unix),
        },
    )
    return result


def _compensate_failed_secret_provision_locked(
    plan: CapabilityCanaryPlan,
    full_plan: FullCanaryPlan,
    *,
    metadata: Mapping[str, Any],
    now_unix: int,
) -> Mapping[str, Any] | None:
    """Synchronously close only the exact lease touched by a failed commit."""

    kind = metadata.get("kind")
    lease_id = metadata.get("lease_id")
    if kind not in _SECRET_LEASE_MAGIC_BY_KIND or not isinstance(lease_id, str):
        return None
    spec = _lease_target(plan, kind=kind, full_plan=full_plan)
    if not os.path.lexists(spec.journal):
        if os.path.lexists(spec.path):
            raise RuntimeError("failed lease published without its journal")
        return None
    with _lease_journal_lock(spec.journal):
        states = _journal_states(spec.journal)
        current = next(
            (state for state in states if state["lease_id"] == lease_id),
            None,
        )
    if current is None:
        if os.path.lexists(spec.path):
            raise RuntimeError("failed lease target lacks its exact intent")
        return None
    if (
        current["install_abort"] is not None
        or current["retirement_completion"] is not None
    ):
        if os.path.lexists(spec.path):
            raise RuntimeError("terminal failed lease target reappeared")
        return current["install_abort"] or current["retirement_completion"]
    services = _capability_services()
    if set(services) != set(CAPABILITY_STOP_ORDER) or not all(
        _service_stopped(state) for state in services.values()
    ):
        raise RuntimeError("failed lease compensation found a live consumer")
    stop_proof = build_capability_stop_proof(
        plan,
        services,
        stop_order=CAPABILITY_STOP_ORDER,
        observed_at_unix=now_unix,
    )
    return _retire_secret_slot_best_effort(
        plan,
        spec,
        stop_proof=stop_proof,
        now_unix=now_unix,
    )


def validate_plan_against_full(
    plan: CapabilityCanaryPlan, full_plan: FullCanaryPlan
) -> None:
    if (
        plan.revision != full_plan.revision
        or plan.full_canary_plan_sha256 != full_plan.sha256
        or plan.release_artifact_sha256 != full_plan.release["artifact_sha256"]
        or plan.release_root != Path(full_plan.release["artifact_root"])
        or plan.interpreter != Path(full_plan.release["interpreter"])
    ):
        raise RuntimeError("capability plan is not bound to the sealed full canary")
    if (
        plan.identities.gateway_user,
        plan.identities.gateway_group,
        plan.identities.gateway_uid,
        plan.identities.gateway_gid,
        plan.identities.socket_client_group,
        plan.identities.socket_client_gid,
        plan.identities.edge_group,
    ) != (
        full_plan.identities.gateway_user,
        full_plan.identities.gateway_group,
        full_plan.identities.gateway_uid,
        full_plan.identities.gateway_gid,
        full_plan.identities.socket_client_group,
        full_plan.identities.socket_client_gid,
        full_plan.identities.edge_group,
    ):
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
    runner: Callable[[Command], subprocess.CompletedProcess[bytes]] = _runner,
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
    if proof["non_observer_services_state_sha256"] != _sha256_json(non_observer_states):
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
            receipt.get("receipt_sha256") != retirement_receipt_sha256s[binding]
            or receipt.get("service_stop_proof_sha256") != proof["stop_proof_sha256"]
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
        int(time.time() * 1000) if observed_at_unix_ms is None else observed_at_unix_ms
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
        "retirement_receipt_sha256s": copy.deepcopy(dict(retirement_receipt_sha256s)),
        "credential_absence": copy.deepcopy(dict(credential_absence)),
        "bitrix_receipt_key_retirement": copy.deepcopy(
            dict(bitrix_receipt_key_retirement)
        ),
        "bitrix_receipt_key_absence": copy.deepcopy(dict(bitrix_receipt_key_absence)),
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
    if facts.get("schema") != CAPABILITY_CLEANUP_FACTS_SCHEMA or facts.get(
        "facts_sha256"
    ) != _sha256_json({
        key: item for key, item in facts.items() if key != "facts_sha256"
    }):
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


def _cleanup_transaction_paths(run_id: str) -> tuple[tuple[int, str, Path], ...]:
    from gateway.canonical_capability_canary_producers import DEFAULT_RECEIPT_ROOT

    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id or "") is None:
        raise ValueError("cleanup transaction run id is invalid")
    root = DEFAULT_RECEIPT_ROOT / run_id
    return tuple(
        (ordinal, stage, root / filename)
        for ordinal, stage, filename in CAPABILITY_CLEANUP_TRANSACTION_STAGES
    )


def _cleanup_transaction_run_gid(run_id: str) -> int:
    """Return the installed, owner-authored receipt-directory group."""

    from gateway.canonical_capability_canary_producers import (
        load_installed_producer_foundation,
    )

    installed = load_installed_producer_foundation()
    expected_gid = installed.value["receipt_contract"]["run_directory_gid"]
    if type(expected_gid) is not int or expected_gid <= 0:
        raise RuntimeError("cleanup transaction receipt identity is invalid")
    directory = _cleanup_transaction_paths(run_id)[0][2].parent
    item = os.lstat(directory)
    if (
        stat.S_ISLNK(item.st_mode)
        or not stat.S_ISDIR(item.st_mode)
        or item.st_uid != 0
        or item.st_gid != expected_gid
        or stat.S_IMODE(item.st_mode) != 0o3770
        or item.st_nlink < 2
    ):
        raise RuntimeError("cleanup transaction run directory is unsafe")
    return expected_gid


def _validate_cleanup_transaction_checkpoint(
    value: Any,
    *,
    plan: CapabilityCanaryPlan,
    fixture_sha256: str,
    run_id: str,
    ordinal: int,
    stage: str,
    path: Path,
    prior_checkpoint_sha256: str | None,
) -> Mapping[str, Any]:
    expected_fields = {
        "schema",
        "ordinal",
        "stage",
        "revision",
        "capability_plan_sha256",
        "full_canary_plan_sha256",
        "fixture_sha256",
        "run_id",
        "prior_checkpoint_sha256",
        "payload",
        "payload_sha256",
        "recorded_at_unix_ms",
        "checkpoint_path",
        "secret_material_recorded",
        "secret_digest_recorded",
        "checkpoint_sha256",
    }
    raw = _strict_mapping(
        value,
        expected_fields,
        "cleanup transaction checkpoint",
    )
    payload = raw["payload"]
    unsigned = {
        key: copy.deepcopy(item)
        for key, item in raw.items()
        if key != "checkpoint_sha256"
    }
    if (
        raw["schema"] != CAPABILITY_CLEANUP_TRANSACTION_SCHEMA
        or raw["ordinal"] != ordinal
        or raw["stage"] != stage
        or raw["revision"] != plan.revision
        or raw["capability_plan_sha256"] != plan.sha256
        or raw["full_canary_plan_sha256"] != plan.full_canary_plan_sha256
        or raw["fixture_sha256"] != fixture_sha256
        or raw["run_id"] != run_id
        or raw["prior_checkpoint_sha256"] != prior_checkpoint_sha256
        or not isinstance(payload, Mapping)
        or raw["payload_sha256"] != _sha256_json(dict(payload))
        or type(raw["recorded_at_unix_ms"]) is not int
        or raw["recorded_at_unix_ms"] <= 0
        or raw["checkpoint_path"] != str(path)
        or raw["secret_material_recorded"] is not False
        or raw["secret_digest_recorded"] is not False
        or raw["checkpoint_sha256"] != _sha256_json(unsigned)
    ):
        raise RuntimeError("cleanup transaction checkpoint is invalid")
    for field in (
        "capability_plan_sha256",
        "full_canary_plan_sha256",
        "fixture_sha256",
        "payload_sha256",
        "checkpoint_sha256",
    ):
        _digest(raw[field], f"cleanup transaction {field}")
    if prior_checkpoint_sha256 is not None:
        _digest(prior_checkpoint_sha256, "cleanup transaction prior checkpoint")
    return copy.deepcopy(dict(raw))


def load_capability_cleanup_transaction(
    plan: CapabilityCanaryPlan,
    *,
    fixture_sha256: str,
    run_id: str,
) -> Mapping[str, Mapping[str, Any]]:
    """Load the contiguous fixed checkpoint chain, rejecting gaps or drift."""

    _digest(fixture_sha256, "cleanup transaction fixture")
    _cleanup_transaction_run_gid(run_id)
    checkpoints: dict[str, Mapping[str, Any]] = {}
    prior: str | None = None
    gap = False
    for ordinal, stage, path in _cleanup_transaction_paths(run_id):
        if not os.path.lexists(path):
            gap = True
            continue
        if gap:
            raise RuntimeError("cleanup transaction checkpoint chain has a gap")
        raw, item = _read_exact_file(
            path,
            maximum=4 * 1024 * 1024,
            uid=0,
            gid=0,
            mode=0o400,
        )
        if item.st_nlink != 1:
            raise RuntimeError("cleanup transaction checkpoint identity is unsafe")
        value = _decode_json(raw, label="cleanup transaction checkpoint")
        if raw != _canonical_bytes(value):
            raise RuntimeError("cleanup transaction checkpoint is not canonical")
        checkpoint = _validate_cleanup_transaction_checkpoint(
            value,
            plan=plan,
            fixture_sha256=fixture_sha256,
            run_id=run_id,
            ordinal=ordinal,
            stage=stage,
            path=path,
            prior_checkpoint_sha256=prior,
        )
        checkpoints[stage] = checkpoint
        prior = checkpoint["checkpoint_sha256"]
    return copy.deepcopy(checkpoints)


def publish_capability_cleanup_transaction_checkpoint(
    plan: CapabilityCanaryPlan,
    *,
    fixture_sha256: str,
    run_id: str,
    stage: str,
    payload: Mapping[str, Any],
    existing: Mapping[str, Mapping[str, Any]] | None = None,
    recorded_at_unix_ms: int | None = None,
) -> Mapping[str, Any]:
    """Append one exact checkpoint and stable-read it before returning."""

    if not isinstance(payload, Mapping):
        raise TypeError("cleanup transaction payload is invalid")
    _digest(fixture_sha256, "cleanup transaction fixture")
    _cleanup_transaction_run_gid(run_id)
    paths = _cleanup_transaction_paths(run_id)
    selected = next((item for item in paths if item[1] == stage), None)
    if selected is None:
        raise ValueError("cleanup transaction stage is invalid")
    ordinal, _stage, path = selected
    loaded = (
        load_capability_cleanup_transaction(
            plan,
            fixture_sha256=fixture_sha256,
            run_id=run_id,
        )
        if existing is None
        else copy.deepcopy(dict(existing))
    )
    prior_stage = paths[ordinal - 2][1] if ordinal > 1 else None
    if stage in loaded:
        checkpoint = loaded[stage]
        if checkpoint.get("payload") != payload:
            raise RuntimeError("cleanup transaction replay payload drifted")
        return checkpoint
    if len(loaded) != ordinal - 1 or (
        prior_stage is not None and prior_stage not in loaded
    ):
        raise RuntimeError("cleanup transaction stage is out of order")
    prior = (
        loaded[prior_stage]["checkpoint_sha256"]
        if prior_stage is not None
        else None
    )
    observed = (
        int(time.time() * 1000)
        if recorded_at_unix_ms is None
        else recorded_at_unix_ms
    )
    if type(observed) is not int or observed <= 0:
        raise ValueError("cleanup transaction checkpoint time is invalid")
    value = {
        "schema": CAPABILITY_CLEANUP_TRANSACTION_SCHEMA,
        "ordinal": ordinal,
        "stage": stage,
        "revision": plan.revision,
        "capability_plan_sha256": plan.sha256,
        "full_canary_plan_sha256": plan.full_canary_plan_sha256,
        "fixture_sha256": fixture_sha256,
        "run_id": run_id,
        "prior_checkpoint_sha256": prior,
        "payload": copy.deepcopy(dict(payload)),
        "payload_sha256": _sha256_json(dict(payload)),
        "recorded_at_unix_ms": observed,
        "checkpoint_path": str(path),
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    checkpoint = {**value, "checkpoint_sha256": _sha256_json(value)}
    _atomic_no_replace_file(
        path,
        _canonical_bytes(checkpoint),
        uid=0,
        gid=0,
        mode=0o400,
        temporary_name=f".{path.name}.installing",
        maximum=4 * 1024 * 1024,
    )
    reloaded = load_capability_cleanup_transaction(
        plan,
        fixture_sha256=fixture_sha256,
        run_id=run_id,
    )
    if reloaded.get(stage) != checkpoint:
        raise RuntimeError("cleanup transaction checkpoint readback drifted")
    return checkpoint


def _production_observation_marker_path(*, run_id: str, phase: str) -> Path:
    from gateway.canonical_capability_canary_producers import (
        DEFAULT_RECEIPT_ROOT,
    )

    if re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id or ""
    ) is None or phase not in {"before", "after"}:
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
    request = _strict_mapping(value, fields, "production observation wait request")
    if (
        request["schema"] != CAPABILITY_PRODUCTION_OBSERVATION_WAIT_REQUEST_SCHEMA
        or request["phase"] not in {"before", "after"}
        or request["canary_revision"] != plan.revision
        or request["capability_plan_sha256"] != plan.sha256
        or request["full_canary_plan_sha256"] != plan.full_canary_plan_sha256
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
            DEFAULT_RECEIPT_ROOT / run_id / "production-observation-before.json"
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
            raise ValueError("staged production before observation is invalid")
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
            diff_publication["diff_sha256"] if diff_publication is not None else None
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
    created = int(time.time() * 1000) if now_unix_ms is None else now_unix_ms
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
        or value["full_canary_plan_sha256"] != plan.full_canary_plan_sha256
        or value["fixture_sha256"] != _digest(fixture_sha256, "fixture")
        or value["owner_subject_sha256"]
        != _digest(owner_subject_sha256, "owner subject")
        or value["observer_live_required"] is not expected_observer
        or value["observer_service_unit"] != CAPABILITY_OBSERVER_UNIT
        or type(value["created_at_unix_ms"]) is not int
        or type(value["expires_at_unix_ms"]) is not int
        or type(now) is not int
        or value["expires_at_unix_ms"] != value["created_at_unix_ms"] + 300_000
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
            current = _require_live_cleanup_observer(state_reader=observer_state_reader)
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
    validated_now = int(time.time() * 1000) if now_unix_ms is None else now_unix_ms
    if (
        raw["schema"] != CAPABILITY_PRODUCTION_OBSERVATION_ENVELOPE_SCHEMA
        or raw["phase"] != phase
        or raw["canary_revision"] != plan.revision
        or raw["capability_plan_sha256"] != plan.sha256
        or raw["full_canary_plan_sha256"] != plan.full_canary_plan_sha256
        or raw["fixture_sha256"] != _digest(fixture_sha256, "fixture")
        or raw["run_id"] != run_id
        or raw["owner_subject_sha256"] != _digest(owner_subject_sha256, "owner subject")
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
    if phase == "after" and observation_at < marker["created_at_unix_ms"]:
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
        int(time.time() * 1000) if observed_at_unix_ms is None else observed_at_unix_ms
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
        or first_host["machine_id_sha256"] != second_host["machine_id_sha256"]
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
            name: surface_diffs[name]["before_sha256"] for name in surface_names
        },
    }
    static_second = {
        "target": second_observation["target"],
        "machine_id_sha256": second_host["machine_id_sha256"],
        "hostname_sha256": second_host["hostname_sha256"],
        "surfaces": {
            name: surface_diffs[name]["after_sha256"] for name in surface_names
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
        "target": copy.deepcopy(dict(first["observation"]["target"])),
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
        "expected_change_contract_sha256": _sha256_json(expected_change_contract),
        "unexpected_change_count": len(changed_surfaces),
        "production_mutation_observed": bool(changed_surfaces),
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
        "semantic_job_content_recorded": False,
    }
    return {**unsigned, "diff_sha256": _sha256_json(unsigned)}


def validate_capability_production_diff(
    value: Mapping[str, Any],
    *,
    run_id: str,
    revision: str,
    capability_plan_sha256: str,
    full_canary_plan_sha256: str,
    fixture_sha256: str,
) -> Mapping[str, Any]:
    """Purely validate the exact native no-change production diff.

    This is the read-only promotion boundary used after the root publisher has
    installed ``production-diff.json``.  It accepts no inferred identity: the
    caller supplies the exact run, release, plans, and fixture selected by the
    owner, and every one must match the canonical self-digested artifact.
    """

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
    if (
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id or "")
        is None
        or _REVISION_RE.fullmatch(revision or "") is None
        or any(
            _SHA256_RE.fullmatch(item or "") is None
            for item in (
                capability_plan_sha256,
                full_canary_plan_sha256,
                fixture_sha256,
            )
        )
    ):
        raise ValueError("production diff expected identity is invalid")
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
        "run_id": run_id,
        "canary_revision": revision,
        "capability_plan_sha256": capability_plan_sha256,
        "full_canary_plan_sha256": full_canary_plan_sha256,
        "fixture_sha256": fixture_sha256,
        "target": raw["target"],
        "expected_changed_surfaces": [],
        "boot_identity_change_allowed": True,
    }
    if (
        raw["schema"] != CAPABILITY_PRODUCTION_DIFF_SCHEMA
        or raw["run_id"] != run_id
        or raw["canary_revision"] != revision
        or raw["capability_plan_sha256"] != capability_plan_sha256
        or raw["full_canary_plan_sha256"] != full_canary_plan_sha256
        or raw["fixture_sha256"] != fixture_sha256
        or any(
            _SHA256_RE.fullmatch(str(raw[field] or "")) is None
            for field in (
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
        or raw["changed_surfaces"] != []
        or derived_changed != []
        or raw["unexpected_change_count"] != 0
        or raw["production_mutation_observed"] is not False
        or raw["static_before_sha256"] != raw["static_after_sha256"]
        or raw["expected_change_contract_sha256"]
        != _sha256_json(expected_contract)
        or raw["diff_sha256"] != _sha256_json(unsigned)
        or raw["secret_material_recorded"] is not False
        or raw["secret_digest_recorded"] is not False
        or raw["semantic_job_content_recorded"] is not False
    ):
        raise ValueError("production no-change diff is invalid")
    return copy.deepcopy(dict(raw))


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
        or re.fullmatch(r"[0-9a-f]{40}", str(raw["canary_revision"] or "")) is None
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
        or raw["before_observed_at_unix_ms"] >= raw["after_observed_at_unix_ms"]
        or raw["changed_surfaces"] != derived_changed
        or type(raw["unexpected_change_count"]) is not int
        or raw["unexpected_change_count"] != len(derived_changed)
        or type(raw["production_mutation_observed"]) is not bool
        or raw["production_mutation_observed"] is not bool(derived_changed)
        or (
            not derived_changed
            and raw["static_before_sha256"] != raw["static_after_sha256"]
        )
        or raw["expected_change_contract_sha256"] != _sha256_json(expected_contract)
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
        or value.get("full_canary_plan_sha256") != plan.full_canary_plan_sha256
        or value.get("fixture_sha256") != _digest(fixture_sha256, "fixture")
        or value.get("changed_surfaces") != []
        or value.get("unexpected_change_count") != 0
        or value.get("production_mutation_observed") is not False
        or value.get("static_before_sha256") != value.get("static_after_sha256")
        or type(value.get("after_observed_at_unix_ms")) is not int
        or value["after_observed_at_unix_ms"] < marker["created_at_unix_ms"]
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
        int(time.time() * 1000) if stopped_at_unix_ms is None else stopped_at_unix_ms
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
    admission_input_retirement: Mapping[str, Any],
    expiry_watchdog_retirement: Mapping[str, Any],
    expected_expiry_watchdog_authority_sha256s: Sequence[str],
    producer_activation_absent: bool,
    admission_inputs_absent: bool,
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
        "retirement_intent_sha256",
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
    admission_fields = {
        "schema",
        "run_id",
        "fixture_sha256",
        "session_id",
        "capability_epoch_sha256",
        "challenge_sha256",
        "owner_authority_path",
        "owner_authority_sha256",
        "install_publication_sha256",
        "intent_sha256",
        "catalog_sha256",
        "owner_grant_sha256",
        "catalog_absent",
        "owner_grant_absent",
        "retired_at_unix_ms",
        "receipt_sha256",
    }
    admission = _strict_mapping(
        admission_input_retirement,
        admission_fields,
        "API admission input retirement",
    )
    admission_unsigned = {
        key: item for key, item in admission.items() if key != "receipt_sha256"
    }
    watchdogs = _strict_mapping(
        expiry_watchdog_retirement,
        {
            "watchdog_count",
            "authority_receipt_sha256s",
            "authority_set_sha256",
            "retired",
            "all_timers_disabled",
            "all_unit_files_absent",
        },
        "expiry watchdog retirement",
    )
    retired_watchdogs = watchdogs["retired"]
    expected_watchdog_authorities = tuple(
        sorted(
            _digest(value, "expected cleanup expiry watchdog authority")
            for value in expected_expiry_watchdog_authority_sha256s
        )
    )
    reported_watchdog_authorities = watchdogs["authority_receipt_sha256s"]
    expected_watchdog_authority_set_sha256 = _sha256_json({
        "authority_receipt_sha256s": list(expected_watchdog_authorities),
    })
    if (
        not expected_watchdog_authorities
        or len(set(expected_watchdog_authorities))
        != len(expected_watchdog_authorities)
        or type(watchdogs["watchdog_count"]) is not int
        or watchdogs["watchdog_count"] != len(expected_watchdog_authorities)
        or not isinstance(reported_watchdog_authorities, list)
        or tuple(reported_watchdog_authorities) != expected_watchdog_authorities
        or watchdogs["authority_set_sha256"]
        != expected_watchdog_authority_set_sha256
        or not isinstance(retired_watchdogs, list)
        or len(retired_watchdogs) != watchdogs["watchdog_count"]
        or watchdogs["all_timers_disabled"] is not True
        or watchdogs["all_unit_files_absent"] is not True
    ):
        raise ValueError("expiry watchdog retirement is invalid")
    watchdog_completed_at: list[int] = []
    seen_watchdog_ids: set[str] = set()
    for item in retired_watchdogs:
        completion = _strict_mapping(
            item,
            {
                "operation",
                "watchdog_id",
                "watchdog_authority_sha256",
                "disarm_intent_path",
                "disarm_intent_sha256",
                "timer_name",
                "timer_disabled",
                "timer_wants_absent",
                "service_absent",
                "timer_absent",
                "completed_at_unix",
                "ok",
                "schema",
                "receipt_path",
                "secret_material_recorded",
                "secret_digest_recorded",
                "receipt_sha256",
            },
            "expiry watchdog disarm completion",
        )
        completion_unsigned = {
            key: value
            for key, value in completion.items()
            if key != "receipt_sha256"
        }
        completion_paths = _expiry_watchdog_paths(completion["watchdog_id"])
        if (
            completion["schema"]
            != CAPABILITY_EXPIRY_WATCHDOG_DISARM_COMPLETION_SCHEMA
            or _LEASE_ID_RE.fullmatch(str(completion["watchdog_id"])) is None
            or completion["watchdog_id"] in seen_watchdog_ids
            or completion["operation"] != "normal_lifecycle_disarm"
            or completion["disarm_intent_path"]
            != str(completion_paths["disarm_intent"])
            or completion["timer_name"] != completion_paths["timer_name"]
            or completion["receipt_path"]
            != str(completion_paths["disarm_completion"])
            or completion["timer_disabled"] is not True
            or completion["timer_wants_absent"] is not True
            or completion["service_absent"] is not True
            or completion["timer_absent"] is not True
            or completion["ok"] is not True
            or type(completion["completed_at_unix"]) is not int
            or completion["secret_material_recorded"] is not False
            or completion["secret_digest_recorded"] is not False
            or completion["receipt_sha256"] != _sha256_json(completion_unsigned)
        ):
            raise ValueError("expiry watchdog disarm completion is invalid")
        for field in (
            "watchdog_authority_sha256",
            "disarm_intent_sha256",
        ):
            _digest(completion[field], f"cleanup finalization watchdog {field}")
        seen_watchdog_ids.add(completion["watchdog_id"])
        watchdog_completed_at.append(completion["completed_at_unix"])
    if {
        item["watchdog_authority_sha256"] for item in retired_watchdogs
    } != set(expected_watchdog_authorities):
        raise ValueError("expiry watchdog authority binding is invalid")
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
        or admission["schema"]
        != "muncho-production-capability-api-admission-retirement.v1"
        or admission["run_id"] != fleet["run_id"]
        or admission["fixture_sha256"] != fleet["fixture_sha256"]
        or admission["catalog_absent"] is not True
        or admission["owner_grant_absent"] is not True
        or admission["receipt_sha256"] != _sha256_json(admission_unsigned)
        or cleanup["schema"] != "muncho-production-capability-canary-signed-receipt.v1"
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
    for field in (
        "retirement_intent_sha256",
        "owner_authority_sha256",
        "install_publication_sha256",
        "intent_sha256",
        "catalog_sha256",
        "owner_grant_sha256",
        "capability_epoch_sha256",
        "challenge_sha256",
    ):
        _digest(
            fleet[field] if field == "retirement_intent_sha256" else admission[field],
            f"cleanup finalization {field}",
        )
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
            admission["retired_at_unix_ms"],
            stop["observed_at_unix"] * 1000,
            *(value * 1000 for value in watchdog_completed_at),
        )
        or producer_activation_absent is not True
        or admission_inputs_absent is not True
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
        "admission_input_retirement": copy.deepcopy(dict(admission)),
        "expiry_watchdog_retirement": copy.deepcopy(dict(watchdogs)),
        "producer_activation_absent": True,
        "admission_inputs_absent": True,
        "credentials_absent": True,
        "bitrix_receipt_key_pair_absent": True,
        "full_canary_stopped_preflight_sha256": (full_canary_stopped_preflight_sha256),
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
    approval_window_started_at_unix: int | None = None,
) -> Mapping[str, Any]:
    """Observe the exact stopped/live capability runtime without mutation."""

    if phase not in {"stopped", "live"}:
        raise ValueError("capability preflight phase is invalid")
    observed_at_unix = (
        int(time.time())
        if approval_window_started_at_unix is None
        else approval_window_started_at_unix
    )
    if type(observed_at_unix) is not int or observed_at_unix < 0:
        raise ValueError("capability preflight approval anchor is invalid")
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
            "full_canary_terminal_receipt": copy.deepcopy(
                dict(plan.full_canary_terminal_receipt)
            ),
            "full_canary_terminal_receipt_sha256": (
                plan.full_canary_terminal_receipt_sha256
            ),
            "original_full_canary_owner_approval_sha256": (
                plan.original_full_canary_owner_approval_sha256
            ),
            "checks": {"host.dedicated_canary_exact": False},
            "blockers": ["host.dedicated_canary_exact"],
        }
        report = {
            **state,
            "state_sha256": _sha256_json(state),
            "observed_at_unix": observed_at_unix,
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
    plan_publication: Mapping[str, Any] | None = None

    def observe_plan_publication() -> Mapping[str, Any]:
        nonlocal plan_publication
        plan_publication = load_bound_plan_publication_receipt(plan)
        return plan_publication

    observe("plan.publication", observe_plan_publication)
    fixture_publication: Mapping[str, Any] | None = None

    def observe_fixture_publication() -> Mapping[str, Any]:
        nonlocal fixture_publication
        fixture_publication = load_bound_reviewed_fixture_publication(plan, full_plan)
        return fixture_publication

    observe("fixture.publication", observe_fixture_publication)
    bitrix_foundation: Mapping[str, Any] | None = None

    def observe_bitrix_foundation() -> Mapping[str, Any]:
        nonlocal bitrix_foundation
        bitrix_foundation = validate_bitrix_foundation_for_plan(
            plan,
            full_plan,
            now_unix=observed_at_unix,
        )
        return bitrix_foundation

    observe("bitrix.foundation", observe_bitrix_foundation)
    observe(
        "service_identities.foundation",
        lambda: load_service_identity_foundation_receipt(plan, full_plan),
    )
    observe(
        "service_identity.mac_ops",
        lambda: service_host_identity_receipt(
            plan,
            full_plan,
            role="mac_ops",
            allow_create_only_absence=False,
        ),
    )

    def observe_connector_identity() -> Mapping[str, Any]:
        return service_host_identity_receipt(
            plan,
            full_plan,
            role="connector",
            allow_create_only_absence=False,
        )

    observe("service_identity.connector", observe_connector_identity)
    observe(
        "producer.foundation",
        lambda: _producer_foundation_preflight(plan, full_plan),
    )
    from gateway.canonical_capability_canary_producer_units import (
        producer_host_identity_receipt,
    )

    observe(
        "producer.host_identity",
        lambda: producer_host_identity_receipt(
            plan.sha256,
            allow_create_only_absence=False,
        ),
    )
    if phase == "stopped":
        observe(
            "overlay.targets_absent",
            lambda: (
                _overlay_targets_are_absent(plan, full_plan)
                or (_ for _ in ()).throw(
                    RuntimeError("capability overlay targets are not absent")
                )
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

    lease_evidence_names = {
        "api_control": "lease.api_control",
        "bitrix_operational_edge_webhook": "lease.bitrix_operational_edge",
        "discord_canonical_routeback_bot_token": "lease.discord_routeback",
        "discord_public_session_bot_token": "lease.discord_connector",
        "mac_ops_gitlab": "lease.mac_ops",
        "openai_codex": "lease.codex",
    }
    lease_receipt_sha256_by_binding: dict[str, str] = {}
    lease_expires_at_unix_by_binding: dict[str, int] = {}
    for binding, evidence_name in lease_evidence_names.items():
        value = evidence.get(evidence_name)
        if isinstance(value, Mapping):
            receipt_sha256 = value.get("install_receipt_sha256")
            expires_at = value.get("expires_at_unix")
            if (
                isinstance(receipt_sha256, str)
                and _SHA256_RE.fullmatch(receipt_sha256) is not None
                and type(expires_at) is int
            ):
                lease_receipt_sha256_by_binding[binding] = receipt_sha256
                lease_expires_at_unix_by_binding[binding] = expires_at
    fixture_sha256 = (
        fixture_publication.get("fixture_sha256")
        if isinstance(fixture_publication, Mapping)
        else None
    )
    fixture_publication_receipt_sha256 = (
        fixture_publication.get("publication_receipt_sha256")
        if isinstance(fixture_publication, Mapping)
        else None
    )
    fixture_valid_until_unix_ms = (
        fixture_publication.get("fixture_valid_until_unix_ms")
        if isinstance(fixture_publication, Mapping)
        else None
    )
    plan_publication_receipt_sha256 = (
        plan_publication.get("receipt_sha256")
        if isinstance(plan_publication, Mapping)
        else None
    )
    service_identity_foundation = evidence.get("service_identities.foundation")
    service_identity_foundation_receipt_sha256 = (
        service_identity_foundation.get("receipt_sha256")
        if isinstance(service_identity_foundation, Mapping)
        else None
    )
    producer_foundation = evidence.get("producer.foundation")
    producer_identity_foundation_receipt_sha256 = (
        producer_foundation.get("producer_identity_foundation_receipt_sha256")
        if isinstance(producer_foundation, Mapping)
        else None
    )
    bitrix_foundation_receipt_sha256 = (
        bitrix_foundation.get("foundation_receipt_sha256")
        if isinstance(bitrix_foundation, Mapping)
        else None
    )
    bitrix_watchdog_sha256 = (
        bitrix_foundation.get("expiry_watchdog_authority_sha256")
        if isinstance(bitrix_foundation, Mapping)
        else None
    )
    bitrix_watchdog_expires_at = (
        bitrix_foundation.get("expiry_watchdog_expires_at_unix")
        if isinstance(bitrix_foundation, Mapping)
        else None
    )
    fixture_valid_until_unix = (
        fixture_valid_until_unix_ms // 1000
        if type(fixture_valid_until_unix_ms) is int
        else None
    )
    minimum_lease_expires_at_unix = (
        min(lease_expires_at_unix_by_binding.values())
        if lease_expires_at_unix_by_binding
        else None
    )
    bitrix_foundation_expires_at_unix = (
        bitrix_foundation.get("expires_at_unix")
        if isinstance(bitrix_foundation, Mapping)
        else None
    )
    approval_not_after_unix: int | None = None
    chain_complete = (
        set(lease_receipt_sha256_by_binding) == set(CAPABILITY_CREDENTIAL_BINDINGS)
        and set(lease_expires_at_unix_by_binding) == set(CAPABILITY_CREDENTIAL_BINDINGS)
        and isinstance(fixture_sha256, str)
        and _SHA256_RE.fullmatch(fixture_sha256) is not None
        and isinstance(fixture_publication_receipt_sha256, str)
        and _SHA256_RE.fullmatch(fixture_publication_receipt_sha256) is not None
        and type(fixture_valid_until_unix) is int
        and isinstance(plan_publication_receipt_sha256, str)
        and _SHA256_RE.fullmatch(plan_publication_receipt_sha256) is not None
        and isinstance(service_identity_foundation_receipt_sha256, str)
        and _SHA256_RE.fullmatch(service_identity_foundation_receipt_sha256) is not None
        and isinstance(producer_identity_foundation_receipt_sha256, str)
        and _SHA256_RE.fullmatch(producer_identity_foundation_receipt_sha256)
        is not None
        and isinstance(bitrix_foundation_receipt_sha256, str)
        and _SHA256_RE.fullmatch(bitrix_foundation_receipt_sha256) is not None
        and isinstance(bitrix_watchdog_sha256, str)
        and _SHA256_RE.fullmatch(bitrix_watchdog_sha256) is not None
        and type(bitrix_watchdog_expires_at) is int
        and type(bitrix_foundation_expires_at_unix) is int
        and bitrix_foundation_expires_at_unix == bitrix_watchdog_expires_at
    )
    if chain_complete:
        approval_not_after_unix = (
            min(
                observed_at_unix + 900,
                fixture_valid_until_unix,
                minimum_lease_expires_at_unix,
                bitrix_foundation_expires_at_unix,
            )
            - 5
        )
        chain_complete = approval_not_after_unix > observed_at_unix + 30
    checks["approval.fresh_evidence_chain"] = chain_complete
    blockers = sorted(name for name, passed in checks.items() if not passed)
    state = {
        "schema": CAPABILITY_PREFLIGHT_SCHEMA,
        "phase": phase,
        "revision": plan.revision,
        "plan_sha256": plan.sha256,
        "full_canary_plan_sha256": full_plan.sha256,
        "full_canary_terminal_receipt": copy.deepcopy(
            dict(plan.full_canary_terminal_receipt)
        ),
        "full_canary_terminal_receipt_sha256": (
            plan.full_canary_terminal_receipt_sha256
        ),
        "original_full_canary_owner_approval_sha256": (
            plan.original_full_canary_owner_approval_sha256
        ),
        "plan_publication_receipt_sha256": plan_publication_receipt_sha256,
        "service_identity_foundation_receipt_sha256": (
            service_identity_foundation_receipt_sha256
        ),
        "producer_identity_foundation_receipt_sha256": (
            producer_identity_foundation_receipt_sha256
        ),
        "fixture_sha256": fixture_sha256,
        "fixture_publication_receipt_sha256": (fixture_publication_receipt_sha256),
        "fixture_valid_until_unix": fixture_valid_until_unix,
        "lease_install_receipt_sha256_by_binding": dict(
            sorted(lease_receipt_sha256_by_binding.items())
        ),
        "lease_expires_at_unix_by_binding": dict(
            sorted(lease_expires_at_unix_by_binding.items())
        ),
        "minimum_lease_expires_at_unix": minimum_lease_expires_at_unix,
        "bitrix_foundation_receipt_sha256": (bitrix_foundation_receipt_sha256),
        "bitrix_expiry_watchdog_authority_sha256": bitrix_watchdog_sha256,
        "bitrix_foundation_expires_at_unix": (bitrix_foundation_expires_at_unix),
        "approval_window_started_at_unix": observed_at_unix,
        "approval_not_after_unix": approval_not_after_unix,
        "checks": checks,
        "blockers": blockers,
        "evidence": evidence,
        "ok": not blockers,
    }
    report = {
        **state,
        "state_sha256": _sha256_json(state),
        "observed_at_unix": observed_at_unix,
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
        self._pending_deferred_start: Mapping[str, Any] | None = None

    def _require_host(self) -> Mapping[str, Any]:
        return validate_dedicated_canary_host(
            self.full_plan,
            metadata_reader=self.metadata_reader,
            local_identity_reader=self.local_identity_reader,
        )

    def _validate_deferred_core_runtime(
        self,
        approval: CapabilityCanaryOwnerApproval,
        core_receipt: Mapping[str, Any],
        *,
        require_producers_stopped: bool,
    ) -> Mapping[str, Any]:
        """Revalidate the durable phase-one receipt against the live gateway."""

        from gateway.canonical_writer_readiness import READINESS_RECEIPT_VERSION

        pending = _deferred_core_receipt_to_pending(
            self.plan,
            approval,
            core_receipt,
        )
        expected = pending["gateway_readiness"]
        receipt = _readiness_receipt(
            DEFAULT_GATEWAY_READINESS_PATH,
            uid=self.plan.identities.gateway_uid,
            gid=self.plan.identities.gateway_gid,
        )
        state = collect_capability_service_state(
            GATEWAY_UNIT_NAME,
            runner=self.runner,
        )
        digest = readiness_receipt_sha256(receipt)
        pid = state.get("MainPID")
        if (
            receipt.get("version") != READINESS_RECEIPT_VERSION
            or type(pid) is not int
            or pid <= 1
            or not _service_live(
                state,
                path=DEFAULT_GATEWAY_UNIT_PATH,
                service_type="notify",
            )
            or receipt.get("gateway_pid") != pid
            or state.get("StatusText")
            != f"{READINESS_RECEIPT_VERSION}:{digest}"
        ):
            raise RuntimeError("deferred capability gateway readiness drifted")
        listener = _api_loopback_listener_identity(pid)
        observed = {
            "receipt_sha256": digest,
            "gateway_pid": pid,
            "gateway_uid": self.plan.identities.gateway_uid,
            "gateway_gid": self.plan.identities.gateway_gid,
            "api_loopback_listener": listener,
            "ready": True,
        }
        if dict(expected) != observed:
            raise RuntimeError("deferred capability gateway identity changed")
        if require_producers_stopped:
            for role in CAPABILITY_PRODUCER_ROLES:
                state = collect_capability_service_state(
                    CAPABILITY_PRODUCER_SERVICE_UNITS[role],
                    runner=self.runner,
                )
                if not _service_stopped(state):
                    raise RuntimeError("producer became live before owner admission")
        return pending

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

    def _start_routeback_edge(
        self,
        *,
        require_approval: Callable[[], None],
    ) -> Mapping[str, Any]:
        identity = _attest_live_routeback_bot_identity(self.plan, self.full_plan)
        _require_routeback_credential_binding(self.plan, self.full_plan, identity)
        started = False
        try:
            require_approval()
            _run_checked(
                edge_start_command(),
                runner=self.runner,
                label=f"start {EDGE_UNIT_NAME}",
            )
            started = True
            _require_routeback_credential_binding(self.plan, self.full_plan, identity)
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

    def _complete_live_cleanup_transaction_locked(
        self,
        *,
        transaction: Mapping[str, Mapping[str, Any]],
        cleanup_run_id: str,
        cleanup_fixture_sha256: str,
        producer_activation_retirer: Callable[[], Mapping[str, Any]] | None,
        admission_input_retirer: Callable[[], Mapping[str, Any]] | None,
    ) -> Mapping[str, Any]:
        """Resume only from the verified signed-receipt cleanup checkpoint."""

        verified = transaction.get("signed_receipt_verified")
        facts_checkpoint = transaction.get("facts_collected")
        publication_checkpoint = transaction.get("facts_published")
        if not all(
            isinstance(value, Mapping)
            for value in (verified, facts_checkpoint, publication_checkpoint)
        ):
            raise RuntimeError("verified cleanup transaction prefix is incomplete")
        snapshot = facts_checkpoint["payload"]
        required_snapshot = {
            "stopped",
            "execution_cleanup",
            "connector_cleanup",
            "full_gateway_unit_restore",
            "removed_overlay_artifacts",
            "full_canary_stopped_preflight",
            "approval_retirement",
            "producer_foundation",
            "credential_consumer_stop_proof",
            "bitrix_receipt_key_pair_retirement",
            "bitrix_receipt_key_pair_absence",
            "retirements",
            "retirement_receipt_sha256s",
            "credential_absence",
            "credentials_absent",
            "cleanup_facts",
        }
        if set(snapshot) != required_snapshot:
            raise RuntimeError("cleanup transaction facts snapshot is invalid")
        cleanup_facts = snapshot["cleanup_facts"]
        publication = publication_checkpoint["payload"].get(
            "cleanup_facts_publication"
        )
        cleanup_receipt = verified["payload"].get("cleanup_receipt")
        if (
            not isinstance(cleanup_facts, Mapping)
            or not isinstance(publication, Mapping)
            or not isinstance(cleanup_receipt, Mapping)
            or publication.get("facts") != cleanup_facts
            or verified["payload"].get("cleanup_facts_sha256")
            != cleanup_facts.get("facts_sha256")
            or verified["payload"].get("cleanup_facts_file_sha256")
            != publication.get("facts_file_sha256")
            or verified["payload"].get("cleanup_receipt_file_sha256")
            != _sha256_bytes(_canonical_bytes(cleanup_receipt))
            or cleanup_receipt.get("authority_role") != CAPABILITY_OBSERVER_ROLE
            or cleanup_receipt.get("payload", {}).get("run_id")
            != cleanup_run_id
            or cleanup_receipt.get("payload", {}).get("fixture_sha256")
            != cleanup_fixture_sha256
        ):
            raise RuntimeError("verified cleanup transaction binding is invalid")

        current = dict(transaction)
        observer_stopped, observer_errors = _attempt_capability_stop_order(
            self._stop_command,
            stop_order=(CAPABILITY_OBSERVER_UNIT,),
        )
        if observer_errors or observer_stopped != [CAPABILITY_OBSERVER_UNIT]:
            raise BaseExceptionGroup(
                "cleanup observer stop failed",
                [
                    *observer_errors,
                    *(
                        [RuntimeError("cleanup observer did not stop last")]
                        if observer_stopped != [CAPABILITY_OBSERVER_UNIT]
                        else []
                    ),
                ],
            )
        final_services = _capability_services(runner=self.runner)
        existing_observer = current.get("observer_stopped")
        if existing_observer is None:
            service_stop_proof = build_capability_stop_proof(
                self.plan,
                final_services,
                stop_order=CAPABILITY_STOP_ORDER,
            )
            observer_stop_receipt = build_capability_observer_stop_receipt(
                self.plan,
                final_services[CAPABILITY_OBSERVER_UNIT],
            )
            observer_checkpoint = publish_capability_cleanup_transaction_checkpoint(
                self.plan,
                fixture_sha256=cleanup_fixture_sha256,
                run_id=cleanup_run_id,
                stage="observer_stopped",
                payload={
                    "service_stop_proof": service_stop_proof,
                    "observer_stop_receipt": observer_stop_receipt,
                    "final_services_sha256": _sha256_json(dict(final_services)),
                },
                existing=current,
            )
            current["observer_stopped"] = observer_checkpoint
        else:
            observer_payload = existing_observer["payload"]
            service_stop_proof = observer_payload.get("service_stop_proof")
            observer_stop_receipt = observer_payload.get("observer_stop_receipt")
            if (
                not isinstance(service_stop_proof, Mapping)
                or not isinstance(observer_stop_receipt, Mapping)
                or observer_payload.get("final_services_sha256")
                != _sha256_json(dict(final_services))
                or build_capability_stop_proof(
                    self.plan,
                    final_services,
                    stop_order=CAPABILITY_STOP_ORDER,
                    observed_at_unix=service_stop_proof.get("observed_at_unix"),
                )
                != service_stop_proof
                or build_capability_observer_stop_receipt(
                    self.plan,
                    final_services[CAPABILITY_OBSERVER_UNIT],
                    stopped_at_unix_ms=observer_stop_receipt.get(
                        "stopped_at_unix_ms"
                    ),
                )
                != observer_stop_receipt
            ):
                raise RuntimeError("observer stop checkpoint no longer matches host")

        runtime_retired = current.get("runtime_retired")
        if runtime_retired is None:
            if producer_activation_retirer is None:
                raise RuntimeError("live cleanup lacks activation retirement")
            if admission_input_retirer is None:
                raise RuntimeError("live cleanup lacks admission input retirement")
            admission_input_retirement = admission_input_retirer()
            producer_fleet_retirement = producer_activation_retirer()
            admission_unsigned = {
                key: item
                for key, item in dict(admission_input_retirement).items()
                if key != "receipt_sha256"
            } if isinstance(admission_input_retirement, Mapping) else {}
            if (
                not isinstance(admission_input_retirement, Mapping)
                or admission_input_retirement.get("schema")
                != "muncho-production-capability-api-admission-retirement.v1"
                or admission_input_retirement.get("run_id") != cleanup_run_id
                or admission_input_retirement.get("fixture_sha256")
                != cleanup_fixture_sha256
                or admission_input_retirement.get("catalog_absent") is not True
                or admission_input_retirement.get("owner_grant_absent") is not True
                or admission_input_retirement.get("receipt_sha256")
                != _sha256_json(admission_unsigned)
                or not isinstance(producer_fleet_retirement, Mapping)
                or producer_fleet_retirement.get("run_id") != cleanup_run_id
                or producer_fleet_retirement.get("fixture_sha256")
                != cleanup_fixture_sha256
                or producer_fleet_retirement.get("retired") is not True
                or producer_fleet_retirement.get("absence_verified") is not True
            ):
                raise RuntimeError("live cleanup runtime retirement is invalid")
            from gateway.canonical_capability_canary_producers import (
                DEFAULT_OWNER_GRANT_PATH,
                DEFAULT_PROBE_CATALOG_PATH,
                DEFAULT_READINESS_PATH,
            )

            producer_activation_absent = not os.path.lexists(DEFAULT_READINESS_PATH)
            admission_inputs_absent = not os.path.lexists(
                DEFAULT_PROBE_CATALOG_PATH
            ) and not os.path.lexists(DEFAULT_OWNER_GRANT_PATH)
            if not producer_activation_absent or not admission_inputs_absent:
                raise RuntimeError("live cleanup runtime artifacts remain")
            retirement_checkpoint = publish_capability_cleanup_transaction_checkpoint(
                self.plan,
                fixture_sha256=cleanup_fixture_sha256,
                run_id=cleanup_run_id,
                stage="runtime_retired",
                payload={
                    "producer_fleet_retirement": copy.deepcopy(
                        dict(producer_fleet_retirement)
                    ),
                    "admission_input_retirement": copy.deepcopy(
                        dict(admission_input_retirement)
                    ),
                    "producer_activation_absent": True,
                    "admission_inputs_absent": True,
                },
                existing=current,
            )
            current["runtime_retired"] = retirement_checkpoint
        else:
            retirement_payload = runtime_retired["payload"]
            producer_fleet_retirement = retirement_payload.get(
                "producer_fleet_retirement"
            )
            admission_input_retirement = retirement_payload.get(
                "admission_input_retirement"
            )
            from gateway.canonical_capability_canary_producers import (
                DEFAULT_OWNER_GRANT_PATH,
                DEFAULT_PROBE_CATALOG_PATH,
                DEFAULT_READINESS_PATH,
            )

            producer_activation_absent = not os.path.lexists(DEFAULT_READINESS_PATH)
            admission_inputs_absent = not os.path.lexists(
                DEFAULT_PROBE_CATALOG_PATH
            ) and not os.path.lexists(DEFAULT_OWNER_GRANT_PATH)
            if (
                not isinstance(producer_fleet_retirement, Mapping)
                or not isinstance(admission_input_retirement, Mapping)
                or producer_fleet_retirement.get("run_id") != cleanup_run_id
                or producer_fleet_retirement.get("fixture_sha256")
                != cleanup_fixture_sha256
                or producer_fleet_retirement.get("retired") is not True
                or producer_fleet_retirement.get("absence_verified") is not True
                or admission_input_retirement.get("run_id") != cleanup_run_id
                or admission_input_retirement.get("fixture_sha256")
                != cleanup_fixture_sha256
                or admission_input_retirement.get("catalog_absent") is not True
                or admission_input_retirement.get("owner_grant_absent") is not True
                or producer_activation_absent is not True
                or admission_inputs_absent is not True
            ):
                raise RuntimeError("runtime retirement checkpoint no longer holds")

        expected_watchdog_authorities = _expected_cleanup_expiry_watchdog_authorities(
            self.plan,
            approval_retirement=snapshot["approval_retirement"],
            lease_retirements=snapshot["retirements"],
        )
        expiry_watchdog_retirement = disarm_all_capability_expiry_watchdogs(
            runner=self.runner,
            expected_authority_receipt_sha256s=expected_watchdog_authorities,
        )
        if (
            expiry_watchdog_retirement.get("all_timers_disabled") is not True
            or expiry_watchdog_retirement.get("all_unit_files_absent") is not True
            or tuple(
                expiry_watchdog_retirement.get("authority_receipt_sha256s", ())
            )
            != expected_watchdog_authorities
        ):
            raise RuntimeError("capability expiry watchdog retirement failed")
        existing_watchdogs = current.get("watchdogs_disarmed")
        if existing_watchdogs is not None:
            if existing_watchdogs["payload"].get("expiry_watchdog_retirement") != (
                expiry_watchdog_retirement
            ):
                raise RuntimeError("watchdog checkpoint no longer matches host")
        else:
            watchdog_checkpoint = publish_capability_cleanup_transaction_checkpoint(
                self.plan,
                fixture_sha256=cleanup_fixture_sha256,
                run_id=cleanup_run_id,
                stage="watchdogs_disarmed",
                payload={
                    "expected_authority_receipt_sha256s": list(
                        expected_watchdog_authorities
                    ),
                    "expiry_watchdog_retirement": copy.deepcopy(
                        dict(expiry_watchdog_retirement)
                    ),
                },
                existing=current,
            )
            current["watchdogs_disarmed"] = watchdog_checkpoint

        full_stopped = snapshot["full_canary_stopped_preflight"]
        cleanup_finalization = build_capability_cleanup_finalization(
            self.plan,
            cleanup_receipt=cleanup_receipt,
            observer_stop_receipt=observer_stop_receipt,
            service_stop_proof=service_stop_proof,
            producer_fleet_retirement=producer_fleet_retirement,
            admission_input_retirement=admission_input_retirement,
            expiry_watchdog_retirement=expiry_watchdog_retirement,
            expected_expiry_watchdog_authority_sha256s=(
                expected_watchdog_authorities
            ),
            producer_activation_absent=True,
            admission_inputs_absent=True,
            credentials_absent=snapshot["credentials_absent"],
            bitrix_receipt_key_pair_absent=snapshot[
                "bitrix_receipt_key_pair_absence"
            ]["both_pair_members_absent"],
            full_canary_stopped_preflight_sha256=full_stopped["report_sha256"],
            finalized_at_unix_ms=(
                current.get("finalized", {})
                .get("payload", {})
                .get("cleanup_finalization", {})
                .get("finalized_at_unix_ms")
            ),
        )
        existing_finalized = current.get("finalized")
        if existing_finalized is not None:
            if existing_finalized["payload"].get("cleanup_finalization") != (
                cleanup_finalization
            ):
                raise RuntimeError("cleanup finalization checkpoint drifted")
        else:
            final_checkpoint = publish_capability_cleanup_transaction_checkpoint(
                self.plan,
                fixture_sha256=cleanup_fixture_sha256,
                run_id=cleanup_run_id,
                stage="finalized",
                payload={"cleanup_finalization": cleanup_finalization},
                existing=current,
            )
            current["finalized"] = final_checkpoint

        result = {
            "stop_order": [*snapshot["stopped"], CAPABILITY_OBSERVER_UNIT],
            "credential_consumer_stop_proof": copy.deepcopy(
                dict(snapshot["credential_consumer_stop_proof"])
            ),
            "service_stop_proof": copy.deepcopy(dict(service_stop_proof)),
            "cleanup_facts": copy.deepcopy(dict(cleanup_facts)),
            "cleanup_facts_publication": copy.deepcopy(dict(publication)),
            "cleanup_receipt": copy.deepcopy(dict(cleanup_receipt)),
            "observer_stop_receipt": copy.deepcopy(dict(observer_stop_receipt)),
            "producer_fleet_retirement": copy.deepcopy(
                dict(producer_fleet_retirement)
            ),
            "cleanup_finalization": copy.deepcopy(dict(cleanup_finalization)),
            "bitrix_receipt_key_pair_retirement": copy.deepcopy(
                dict(snapshot["bitrix_receipt_key_pair_retirement"])
            ),
            "bitrix_receipt_key_pair_absence": copy.deepcopy(
                dict(snapshot["bitrix_receipt_key_pair_absence"])
            ),
            "retirements": copy.deepcopy(dict(snapshot["retirements"])),
            "retirement_receipt_sha256s": copy.deepcopy(
                dict(snapshot["retirement_receipt_sha256s"])
            ),
            "credential_absence": copy.deepcopy(
                dict(snapshot["credential_absence"])
            ),
            "approval_retirement": copy.deepcopy(
                dict(snapshot["approval_retirement"])
            ),
            "connector_cleanup": copy.deepcopy(dict(snapshot["connector_cleanup"])),
            "execution_cleanup": copy.deepcopy(dict(snapshot["execution_cleanup"])),
            "full_gateway_unit_restore": copy.deepcopy(
                dict(snapshot["full_gateway_unit_restore"])
            ),
            "removed_overlay_artifacts": copy.deepcopy(
                dict(snapshot["removed_overlay_artifacts"])
            ),
            "full_canary_stopped_preflight_sha256": full_stopped["report_sha256"],
            "services_stopped": all(
                _service_stopped(state) for state in final_services.values()
            ),
            "credentials_absent": snapshot["credentials_absent"],
            "producer_activation_absent": True,
            "admission_input_retirement": copy.deepcopy(
                dict(admission_input_retirement)
            ),
            "admission_inputs_absent": True,
            "expiry_watchdog_retirement": copy.deepcopy(
                dict(expiry_watchdog_retirement)
            ),
            "cleanup_transaction": copy.deepcopy(dict(current)),
            "units_enabled": False,
            "completed_at_unix": int(time.time()),
        }
        if result["services_stopped"] is not True:
            raise RuntimeError("capability services did not stop exactly")
        self._pending_deferred_start = None
        return result

    def _cleanup_locked(
        self,
        *,
        cleanup_producer: Callable[[Mapping[str, Any]], Mapping[str, Any]]
        | None = None,
        cleanup_receipt_loader: Callable[[], Mapping[str, Any] | None]
        | None = None,
        cleanup_receipt_verifier: Callable[[Mapping[str, Any]], Any] | None = None,
        cleanup_run_id: str | None = None,
        cleanup_fixture_sha256: str | None = None,
        producer_activation_retirer: Callable[[], Mapping[str, Any]] | None = None,
        admission_input_retirer: Callable[[], Mapping[str, Any]] | None = None,
    ) -> Mapping[str, Any]:
        cleanup_transaction: dict[str, Mapping[str, Any]] = {}
        if cleanup_run_id is not None:
            if (
                cleanup_fixture_sha256 is None
                or _SHA256_RE.fullmatch(cleanup_fixture_sha256) is None
                or cleanup_receipt_loader is None
                or cleanup_receipt_verifier is None
            ):
                raise RuntimeError("live cleanup transaction boundary is incomplete")
            cleanup_transaction = dict(load_capability_cleanup_transaction(
                self.plan,
                fixture_sha256=cleanup_fixture_sha256,
                run_id=cleanup_run_id,
            ))
            verified = cleanup_transaction.get("signed_receipt_verified")
            if verified is not None:
                cleanup_receipt = verified["payload"].get("cleanup_receipt")
                if not isinstance(cleanup_receipt, Mapping):
                    raise RuntimeError("verified cleanup receipt checkpoint is invalid")
                cleanup_receipt_verifier(cleanup_receipt)
                readback = cleanup_receipt_loader()
                if readback != cleanup_receipt:
                    raise RuntimeError("verified cleanup receipt readback drifted")
                return self._complete_live_cleanup_transaction_locked(
                    transaction=cleanup_transaction,
                    cleanup_run_id=cleanup_run_id,
                    cleanup_fixture_sha256=cleanup_fixture_sha256,
                    producer_activation_retirer=producer_activation_retirer,
                    admission_input_retirer=admission_input_retirer,
                )
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
            credential_consumer_stop_proof = build_credential_consumer_stop_proof(
                self.plan,
                services,
                producer_foundation=producer_foundation,
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
                public_key_id=(self.plan.bitrix_operational_edge_receipt_public_key_id),
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
        if set(retirements) != set(CAPABILITY_CREDENTIAL_BINDINGS) or set(
            retirement_receipt_sha256s
        ) != set(CAPABILITY_CREDENTIAL_BINDINGS):
            errors.append(
                RuntimeError("six exact credential retirement receipts are required")
            )
        if not credentials_absent:
            errors.append(
                RuntimeError("capability credential retirement is incomplete")
            )
        if bitrix_key_absence["both_pair_members_absent"] is not True:
            errors.append(RuntimeError("Bitrix receipt key retirement is incomplete"))

        cleanup_facts: Mapping[str, Any] | None = None
        cleanup_facts_publication: Mapping[str, Any] | None = None
        cleanup_receipt: Mapping[str, Any] | None = None
        if not errors:
            try:
                facts_checkpoint = cleanup_transaction.get("facts_collected")
                if facts_checkpoint is None:
                    cleanup_facts = build_capability_cleanup_facts(
                        self.plan,
                        services=services,
                        credential_consumer_stop_proof=(credential_consumer_stop_proof),
                        producer_foundation=producer_foundation or {},
                        retirements=retirements,
                        retirement_receipt_sha256s=(retirement_receipt_sha256s),
                        credential_absence=credential_absence,
                        bitrix_receipt_key_retirement=(bitrix_key_retirement or {}),
                        bitrix_receipt_key_absence=bitrix_key_absence,
                        execution_cleanup=execution_cleanup or {},
                    )
                else:
                    cleanup_facts = facts_checkpoint["payload"].get("cleanup_facts")
                    if not isinstance(cleanup_facts, Mapping):
                        raise RuntimeError("cleanup facts checkpoint is invalid")
                if cleanup_run_id is not None:
                    assert cleanup_fixture_sha256 is not None
                    observer_identity = _cleanup_observer_identity()
                    if facts_checkpoint is None:
                        facts_checkpoint = (
                            publish_capability_cleanup_transaction_checkpoint(
                                self.plan,
                                fixture_sha256=cleanup_fixture_sha256,
                                run_id=cleanup_run_id,
                                stage="facts_collected",
                                payload={
                                    "stopped": list(stopped),
                                    "execution_cleanup": copy.deepcopy(
                                        dict(execution_cleanup or {})
                                    ),
                                    "connector_cleanup": copy.deepcopy(
                                        dict(connector_cleanup or {})
                                    ),
                                    "full_gateway_unit_restore": copy.deepcopy(
                                        dict(restored or {})
                                    ),
                                    "removed_overlay_artifacts": copy.deepcopy(
                                        dict(removed_artifacts)
                                    ),
                                    "full_canary_stopped_preflight": copy.deepcopy(
                                        dict(full_stopped or {})
                                    ),
                                    "approval_retirement": copy.deepcopy(
                                        dict(approval_retirement or {})
                                    ),
                                    "producer_foundation": copy.deepcopy(
                                        dict(producer_foundation or {})
                                    ),
                                    "credential_consumer_stop_proof": copy.deepcopy(
                                        dict(credential_consumer_stop_proof or {})
                                    ),
                                    "bitrix_receipt_key_pair_retirement": copy.deepcopy(
                                        dict(bitrix_key_retirement or {})
                                    ),
                                    "bitrix_receipt_key_pair_absence": copy.deepcopy(
                                        dict(bitrix_key_absence)
                                    ),
                                    "retirements": copy.deepcopy(dict(retirements)),
                                    "retirement_receipt_sha256s": copy.deepcopy(
                                        dict(retirement_receipt_sha256s)
                                    ),
                                    "credential_absence": copy.deepcopy(
                                        dict(credential_absence)
                                    ),
                                    "credentials_absent": credentials_absent,
                                    "cleanup_facts": copy.deepcopy(
                                        dict(cleanup_facts)
                                    ),
                                },
                                existing=cleanup_transaction,
                            )
                        )
                        cleanup_transaction["facts_collected"] = facts_checkpoint
                    publication_checkpoint = cleanup_transaction.get(
                        "facts_published"
                    )
                    if publication_checkpoint is None:
                        cleanup_facts_publication = publish_capability_cleanup_facts(
                            cleanup_facts,
                            run_id=cleanup_run_id,
                            observer_gid=observer_identity["gid"],
                        )
                        facts_raw, facts_item = _read_exact_file(
                            Path(cleanup_facts_publication["facts_path"]),
                            maximum=2 * 1024 * 1024,
                            uid=0,
                            gid=observer_identity["gid"],
                            mode=0o440,
                        )
                        if (
                            facts_item.st_nlink != 1
                            or facts_raw != _canonical_bytes(cleanup_facts)
                            or _sha256_bytes(facts_raw)
                            != cleanup_facts_publication["facts_file_sha256"]
                        ):
                            raise RuntimeError("cleanup facts publication drifted")
                        publication_checkpoint = (
                            publish_capability_cleanup_transaction_checkpoint(
                                self.plan,
                                fixture_sha256=cleanup_fixture_sha256,
                                run_id=cleanup_run_id,
                                stage="facts_published",
                                payload={
                                    "cleanup_facts_publication": copy.deepcopy(
                                        dict(cleanup_facts_publication)
                                    )
                                },
                                existing=cleanup_transaction,
                            )
                        )
                        cleanup_transaction["facts_published"] = (
                            publication_checkpoint
                        )
                    else:
                        cleanup_facts_publication = publication_checkpoint[
                            "payload"
                        ].get("cleanup_facts_publication")
                        if (
                            not isinstance(cleanup_facts_publication, Mapping)
                            or cleanup_facts_publication.get("facts")
                            != cleanup_facts
                        ):
                            raise RuntimeError(
                                "cleanup facts publication checkpoint is invalid"
                            )
                    assert cleanup_receipt_loader is not None
                    assert cleanup_receipt_verifier is not None
                    cleanup_receipt = cleanup_receipt_loader()
                    if cleanup_receipt is None:
                        if cleanup_producer is None:
                            raise RuntimeError(
                                "live cleanup lacks the observer producer"
                            )
                        cleanup_receipt = cleanup_producer(
                            cleanup_facts_publication
                        )
                    if not isinstance(cleanup_receipt, Mapping):
                        raise RuntimeError("observer cleanup receipt is invalid")
                    cleanup_receipt_readback = cleanup_receipt_loader()
                    if cleanup_receipt_readback != cleanup_receipt:
                        raise RuntimeError("observer cleanup receipt readback drifted")
                    cleanup_receipt_verifier(cleanup_receipt)
                    if (
                        cleanup_receipt.get("authority_role")
                        != CAPABILITY_OBSERVER_ROLE
                        or cleanup_receipt.get("payload", {}).get("run_id")
                        != cleanup_run_id
                        or cleanup_receipt.get("payload", {}).get(
                            "fixture_sha256"
                        )
                        != cleanup_fixture_sha256
                    ):
                        raise RuntimeError("observer cleanup receipt binding drifted")
                    receipt_checkpoint = (
                        publish_capability_cleanup_transaction_checkpoint(
                            self.plan,
                            fixture_sha256=cleanup_fixture_sha256,
                            run_id=cleanup_run_id,
                            stage="signed_receipt_verified",
                            payload={
                                "cleanup_facts_sha256": cleanup_facts[
                                    "facts_sha256"
                                ],
                                "cleanup_facts_file_sha256": (
                                    cleanup_facts_publication[
                                        "facts_file_sha256"
                                    ]
                                ),
                                "cleanup_receipt": copy.deepcopy(
                                    dict(cleanup_receipt)
                                ),
                                "cleanup_receipt_file_sha256": _sha256_bytes(
                                    _canonical_bytes(cleanup_receipt)
                                ),
                                "receipt_readback_verified": True,
                                "signature_and_native_evidence_verified": True,
                            },
                            existing=cleanup_transaction,
                        )
                    )
                    cleanup_transaction["signed_receipt_verified"] = (
                        receipt_checkpoint
                    )
                    return self._complete_live_cleanup_transaction_locked(
                        transaction=cleanup_transaction,
                        cleanup_run_id=cleanup_run_id,
                        cleanup_fixture_sha256=cleanup_fixture_sha256,
                        producer_activation_retirer=producer_activation_retirer,
                        admission_input_retirer=admission_input_retirer,
                    )
                elif cleanup_producer is not None or cleanup_receipt_loader is not None:
                    raise RuntimeError("cleanup receipt boundary lacks a fixed run id")
            except BaseException as exc:
                errors.append(exc)

        if cleanup_run_id is not None and errors:
            # The credential-blind observer remains available for exact retry;
            # stopping it before the verified signed-receipt checkpoint would
            # make the append-only transaction unrecoverable.
            raise BaseExceptionGroup(
                "live cleanup failed before observer-stop checkpoint",
                errors,
            )

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

        admission_input_retirement: Mapping[str, Any] | None = None
        if admission_input_retirer is not None and service_stop_proof is not None:
            try:
                admission_input_retirement = admission_input_retirer()
                admission_unsigned = {
                    key: item
                    for key, item in dict(admission_input_retirement).items()
                    if key != "receipt_sha256"
                } if isinstance(admission_input_retirement, Mapping) else {}
                if (
                    not isinstance(admission_input_retirement, Mapping)
                    or admission_input_retirement.get("schema")
                    != "muncho-production-capability-api-admission-retirement.v1"
                    or (
                        cleanup_run_id is not None
                        and admission_input_retirement.get("run_id")
                        != cleanup_run_id
                    )
                    or admission_input_retirement.get("catalog_absent") is not True
                    or admission_input_retirement.get("owner_grant_absent") is not True
                    or any(
                        _SHA256_RE.fullmatch(
                            str(admission_input_retirement.get(field, ""))
                        )
                        is None
                        for field in (
                            "fixture_sha256",
                            "capability_epoch_sha256",
                            "challenge_sha256",
                            "owner_authority_sha256",
                            "install_publication_sha256",
                            "intent_sha256",
                            "catalog_sha256",
                            "owner_grant_sha256",
                        )
                    )
                    or admission_input_retirement.get("receipt_sha256")
                    != _sha256_json(admission_unsigned)
                ):
                    raise RuntimeError("API admission input retirement is invalid")
            except BaseException as exc:
                errors.append(exc)
        elif cleanup_run_id is not None:
            errors.append(RuntimeError("live cleanup lacks admission input retirement"))

        producer_fleet_retirement: Mapping[str, Any] | None = None
        if producer_activation_retirer is not None:
            try:
                producer_fleet_retirement = producer_activation_retirer()
                if (
                    not isinstance(producer_fleet_retirement, Mapping)
                    or producer_fleet_retirement.get("retired") is not True
                    or producer_fleet_retirement.get("absence_verified") is not True
                    or (
                        cleanup_run_id is not None
                        and producer_fleet_retirement.get("run_id") != cleanup_run_id
                    )
                ):
                    raise RuntimeError("producer activation retirement is invalid")
            except BaseException as exc:
                errors.append(exc)
        elif cleanup_run_id is not None:
            errors.append(RuntimeError("live cleanup lacks activation retirement"))

        try:
            from gateway.canonical_capability_canary_producers import (
                DEFAULT_OWNER_GRANT_PATH,
                DEFAULT_PROBE_CATALOG_PATH,
                DEFAULT_READINESS_PATH,
            )

            producer_activation_absent = not os.path.lexists(DEFAULT_READINESS_PATH)
            if not producer_activation_absent:
                raise RuntimeError("producer activation remains after retirement")
            admission_inputs_absent = not os.path.lexists(
                DEFAULT_PROBE_CATALOG_PATH
            ) and not os.path.lexists(DEFAULT_OWNER_GRANT_PATH)
            if not admission_inputs_absent:
                raise RuntimeError("API admission inputs remain after retirement")
        except BaseException as exc:
            producer_activation_absent = False
            admission_inputs_absent = False
            errors.append(exc)

        expiry_watchdog_retirement: Mapping[str, Any] | None = None
        expected_watchdog_authorities: tuple[str, ...] = ()
        if (
            credentials_absent
            and bitrix_key_absence["both_pair_members_absent"] is True
            and service_stop_proof is not None
            and producer_activation_absent
            and admission_inputs_absent
            and producer_fleet_retirement is not None
            and admission_input_retirement is not None
            and not errors
        ):
            try:
                expected_watchdog_authorities = (
                    _expected_cleanup_expiry_watchdog_authorities(
                        self.plan,
                        approval_retirement=approval_retirement or {},
                        lease_retirements=retirements,
                    )
                )
                expiry_watchdog_retirement = disarm_all_capability_expiry_watchdogs(
                    runner=self.runner,
                    expected_authority_receipt_sha256s=(
                        expected_watchdog_authorities
                    ),
                )
                if (
                    expiry_watchdog_retirement.get("all_timers_disabled") is not True
                    or expiry_watchdog_retirement.get("all_unit_files_absent")
                    is not True
                    or tuple(
                        expiry_watchdog_retirement.get(
                            "authority_receipt_sha256s", ()
                        )
                    )
                    != expected_watchdog_authorities
                ):
                    raise RuntimeError("capability expiry watchdog retirement failed")
            except BaseException as exc:
                errors.append(exc)

        cleanup_finalization: Mapping[str, Any] | None = None
        if (
            cleanup_receipt is not None
            and observer_stop_receipt is not None
            and service_stop_proof is not None
            and producer_fleet_retirement is not None
            and admission_input_retirement is not None
            and expiry_watchdog_retirement is not None
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
                    admission_input_retirement=admission_input_retirement,
                    expiry_watchdog_retirement=expiry_watchdog_retirement,
                    expected_expiry_watchdog_authority_sha256s=(
                        expected_watchdog_authorities
                    ),
                    producer_activation_absent=producer_activation_absent,
                    admission_inputs_absent=admission_inputs_absent,
                    credentials_absent=credentials_absent,
                    bitrix_receipt_key_pair_absent=(
                        bitrix_key_absence["both_pair_members_absent"]
                    ),
                    full_canary_stopped_preflight_sha256=full_stopped["report_sha256"],
                )
            except BaseException as exc:
                errors.append(exc)

        result = {
            "stop_order": [*stopped, *observer_stopped],
            "credential_consumer_stop_proof": copy.deepcopy(
                dict(credential_consumer_stop_proof or {})
            ),
            "service_stop_proof": copy.deepcopy(dict(service_stop_proof or {})),
            "cleanup_facts": copy.deepcopy(dict(cleanup_facts or {})),
            "cleanup_facts_publication": copy.deepcopy(
                dict(cleanup_facts_publication or {})
            ),
            "cleanup_receipt": copy.deepcopy(dict(cleanup_receipt or {})),
            "observer_stop_receipt": copy.deepcopy(dict(observer_stop_receipt or {})),
            "producer_fleet_retirement": copy.deepcopy(
                dict(producer_fleet_retirement or {})
            ),
            "cleanup_finalization": copy.deepcopy(dict(cleanup_finalization or {})),
            "bitrix_receipt_key_pair_retirement": copy.deepcopy(
                dict(bitrix_key_retirement or {})
            ),
            "bitrix_receipt_key_pair_absence": bitrix_key_absence,
            "retirements": retirements,
            "retirement_receipt_sha256s": retirement_receipt_sha256s,
            "credential_absence": credential_absence,
            "approval_retirement": copy.deepcopy(dict(approval_retirement or {})),
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
            "admission_input_retirement": copy.deepcopy(
                dict(admission_input_retirement or {})
            ),
            "admission_inputs_absent": admission_inputs_absent,
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
        if cleanup_run_id is not None and not result["cleanup_finalization"]:
            raise RuntimeError("live cleanup finalization is missing")
        self._pending_deferred_start = None
        return result

    def _cleanup(
        self,
        *,
        cleanup_producer: Callable[[Mapping[str, Any]], Mapping[str, Any]]
        | None = None,
        cleanup_receipt_loader: Callable[[], Mapping[str, Any] | None]
        | None = None,
        cleanup_receipt_verifier: Callable[[Mapping[str, Any]], Any] | None = None,
        cleanup_run_id: str | None = None,
        cleanup_fixture_sha256: str | None = None,
        producer_activation_retirer: Callable[[], Mapping[str, Any]] | None = None,
        admission_input_retirer: Callable[[], Mapping[str, Any]] | None = None,
    ) -> Mapping[str, Any]:
        _require_root_linux()
        self._require_host()
        with _lifecycle_lock():
            return self._cleanup_locked(
                cleanup_producer=cleanup_producer,
                cleanup_receipt_loader=cleanup_receipt_loader,
                cleanup_receipt_verifier=cleanup_receipt_verifier,
                cleanup_run_id=cleanup_run_id,
                cleanup_fixture_sha256=cleanup_fixture_sha256,
                producer_activation_retirer=producer_activation_retirer,
                admission_input_retirer=admission_input_retirer,
            )

    def start(
        self,
        approval: CapabilityCanaryOwnerApproval,
        *,
        defer_producers_until_api_admission: bool = False,
    ) -> Mapping[str, Any]:
        _require_root_linux()
        if not isinstance(approval, CapabilityCanaryOwnerApproval):
            raise PermissionError("fresh exact capability owner approval is required")
        if type(defer_producers_until_api_admission) is not bool:
            raise TypeError("deferred producer admission flag must be boolean")

        def require_approval() -> None:
            now = int(time.time())
            approval.require(
                plan_sha256=self.plan.sha256,
                full_canary_plan_sha256=self.full_plan.sha256,
                now_unix=now,
            )
            _require_installed_capability_approval(self.plan, self.full_plan, approval)

        require_approval()
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
        writer_readiness: Mapping[str, Any] | None = None
        gateway_readiness: Mapping[str, Any] | None = None
        try:
            with _lifecycle_lock():
                if defer_producers_until_api_admission:
                    recovered_core, recovered_terminal = (
                        _load_deferred_lifecycle_state(self.plan, approval)
                    )
                    if recovered_terminal is not None:
                        if recovered_core is None:
                            raise RuntimeError(
                                "terminal deferred lifecycle lacks its core receipt"
                            )
                        self._validate_deferred_core_runtime(
                            approval,
                            recovered_core,
                            require_producers_stopped=False,
                        )
                        for role in CAPABILITY_PRODUCER_ROLES:
                            state = collect_capability_service_state(
                                CAPABILITY_PRODUCER_SERVICE_UNITS[role],
                                runner=self.runner,
                            )
                            if not _service_live(
                                state,
                                path=CAPABILITY_PRODUCER_UNIT_PATHS[role],
                                service_type="simple",
                            ):
                                raise PermissionError(
                                    "terminal deferred generation is no longer live"
                                )
                        return recovered_terminal
                    if recovered_core is not None:
                        self._pending_deferred_start = (
                            self._validate_deferred_core_runtime(
                                approval,
                                recovered_core,
                                require_producers_stopped=True,
                            )
                        )
                        return recovered_core
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
                    approval_window_started_at_unix=approval.value[
                        "stopped_preflight_observed_at_unix"
                    ],
                )
                _require_capability_approval_preflight_binding(
                    self.plan, self.full_plan, approval, preflight
                )
                require_approval()
                self._require_host()
                require_approval()
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
                require_approval()
                execution_identity_foundation = ensure_execution_identities_create_only(
                    self.plan,
                    self.full_plan,
                    runner=self.runner,
                )
                require_approval()
                worker_mountpoint = _prepare_worker_mountpoint()
                worker_systemd252_preflight(self.plan, runner=self.runner)
                require_approval()
                _prepare_gateway_directories(self.plan)
                require_approval()
                full_canary_install = _install_plan_artifacts(self.full_plan)
                require_approval()
                capability_overlay_install = _install_capability_artifacts(
                    self.plan, self.full_plan
                )
                installed = {
                    "full_canary_foundation": copy.deepcopy(dict(full_canary_install)),
                    "capability_overlay": copy.deepcopy(
                        dict(capability_overlay_install)
                    ),
                }
                require_approval()
                connector_state = _prepare_connector_state(self.plan)
                producer_foundation = _producer_foundation_preflight(
                    self.plan,
                    self.full_plan,
                )
                _run_checked(
                    Command((
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
                    )),
                    runner=self.runner,
                    label="verify capability-canary units",
                )
                require_approval()
                _run_checked(
                    Command((SYSTEMCTL, "daemon-reload")),
                    runner=self.runner,
                    label="reload capability-canary units",
                )
                require_approval()
                _run_checked(
                    Command((SYSTEMD_TMPFILES, "--create", str(DEFAULT_TMPFILES_PATH))),
                    runner=self.runner,
                    label="create full-canary runtime directories",
                )
                from gateway.canonical_writer_phase_b_runtime import (
                    install_fixed_phase_b_full_canary_anchor,
                    validate_fixed_phase_b_readiness_descendant,
                )

                require_approval()
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
                require_approval()
                installed_phase_b_anchor = install_fixed_phase_b_full_canary_anchor(
                    self.full_plan.phase_b_readiness_anchor
                )

                require_approval()
                routeback_bot_identity = self._start_routeback_edge(
                    require_approval=require_approval
                )
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
                    require_approval()
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
                require_approval()
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
                require_approval()
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
                if defer_producers_until_api_admission:
                    require_approval()
                    _run_checked(
                        gateway_command,
                        runner=self.runner,
                        label=f"start {GATEWAY_UNIT_NAME}",
                    )
                    started.append(GATEWAY_UNIT_NAME)

                    def deferred_gateway_ready() -> Mapping[str, Any]:
                        receipt = _readiness_receipt(
                            DEFAULT_GATEWAY_READINESS_PATH,
                            uid=self.plan.identities.gateway_uid,
                            gid=self.plan.identities.gateway_gid,
                        )
                        state = collect_capability_service_state(
                            GATEWAY_UNIT_NAME,
                            runner=self.runner,
                        )
                        digest = readiness_receipt_sha256(receipt)
                        if (
                            not _service_live(
                                state,
                                path=DEFAULT_GATEWAY_UNIT_PATH,
                                service_type="notify",
                            )
                            or receipt.get("gateway_pid") != state.get("MainPID")
                            or state.get("StatusText")
                            != f"{receipt.get('version')}:{digest}"
                        ):
                            raise RuntimeError(
                                "deferred capability gateway readiness drifted"
                            )
                        listener = _api_loopback_listener_identity(
                            int(state["MainPID"])
                        )
                        return {
                            "receipt_sha256": digest,
                            "gateway_pid": state["MainPID"],
                            "gateway_uid": self.plan.identities.gateway_uid,
                            "gateway_gid": self.plan.identities.gateway_gid,
                            "api_loopback_listener": listener,
                            "ready": True,
                        }

                    gateway_readiness = _await_runtime_ready(
                        deferred_gateway_ready,
                        label="deferred capability gateway",
                        timeout_seconds=60.0,
                    )
                    if tuple(started) != CAPABILITY_DEFERRED_CORE_START_ORDER:
                        raise RuntimeError("deferred core start order drifted")
                    core_receipt = _write_lifecycle_receipt(
                        self.plan,
                        stage=CAPABILITY_GATEWAY_CORE_READY_STAGE,
                        value={
                            **_capability_approval_chain_fields(approval),
                            "operation": "start_core_before_api_admission",
                            "owner_approval_sha256": approval.sha256,
                            "full_canary_stopped_preflight_sha256": full_preflight[
                                "report_sha256"
                            ],
                            "stopped_preflight_sha256": preflight["report_sha256"],
                            "installed_artifacts": copy.deepcopy(dict(installed)),
                            "connector_state": copy.deepcopy(
                                dict(connector_state or {})
                            ),
                            "phase_b_current_readiness": copy.deepcopy(
                                dict(phase_b_current or {})
                            ),
                            "phase_b_full_canary_anchor": copy.deepcopy(
                                dict(installed_phase_b_anchor or {})
                            ),
                            "writer_runtime_readiness": copy.deepcopy(
                                dict(writer_readiness or {})
                            ),
                            "gateway_runtime_readiness": copy.deepcopy(
                                dict(gateway_readiness)
                            ),
                            "producer_foundation": copy.deepcopy(
                                dict(producer_foundation or {})
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
                            "core_start_order": list(started),
                            "producer_units_started": False,
                            "api_admission_pending": True,
                            "units_enabled": False,
                            "runtime_max_seconds": 900,
                            "started_at_unix": int(time.time()),
                        },
                    )
                    self._pending_deferred_start = {
                        "approval_sha256": approval.sha256,
                        "core_receipt": copy.deepcopy(dict(core_receipt)),
                        "full_preflight": copy.deepcopy(dict(full_preflight)),
                        "preflight": copy.deepcopy(dict(preflight)),
                        "installed": copy.deepcopy(dict(installed)),
                        "connector_state": copy.deepcopy(dict(connector_state or {})),
                        "phase_b_current": copy.deepcopy(dict(phase_b_current or {})),
                        "installed_phase_b_anchor": copy.deepcopy(
                            dict(installed_phase_b_anchor or {})
                        ),
                        "writer_readiness": copy.deepcopy(dict(writer_readiness or {})),
                        "gateway_readiness": copy.deepcopy(dict(gateway_readiness)),
                        "observer": copy.deepcopy(dict(observer or {})),
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
                        "started": list(started),
                    }
                    return core_receipt
                for role in CAPABILITY_PRODUCER_ROLES:
                    unit = CAPABILITY_PRODUCER_SERVICE_UNITS[role]
                    require_approval()
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
                require_approval()
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
                    approval_window_started_at_unix=approval.value[
                        "stopped_preflight_observed_at_unix"
                    ],
                )
                return _write_lifecycle_receipt(
                    self.plan,
                    stage="started",
                    value={
                        **_capability_approval_chain_fields(approval),
                        "operation": "start",
                        "owner_approval_sha256": approval.sha256,
                        "original_full_canary_owner_approval_sha256": (
                            self.plan.original_full_canary_owner_approval_sha256
                        ),
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
                        **_capability_approval_chain_fields(approval),
                        "operation": "start",
                        "owner_approval_sha256": approval.sha256,
                        "original_full_canary_owner_approval_sha256": (
                            self.plan.original_full_canary_owner_approval_sha256
                        ),
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

    def prepare_api_admission_inputs(
        self,
        approval: CapabilityCanaryOwnerApproval,
        *,
        expected_gateway_pid: int,
        expected_run_id: str,
        expected_session_id: str,
        expected_capability_epoch_sha256: str,
        expected_catalog_sha256: str,
        expected_owner_authority_sha256: str,
        admission_publisher: Callable[[], Mapping[str, Any]],
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        """Publish exact owner inputs without starting a producer or model."""

        _require_root_linux()
        if not isinstance(approval, CapabilityCanaryOwnerApproval):
            raise PermissionError("fresh exact capability owner approval is required")
        if (
            type(expected_gateway_pid) is not int
            or expected_gateway_pid <= 1
            or not isinstance(expected_run_id, str)
            or re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9._:-]{0,239}", expected_run_id
            )
            is None
            or not isinstance(expected_session_id, str)
            or not expected_session_id
            or _SHA256_RE.fullmatch(expected_capability_epoch_sha256 or "") is None
            or _SHA256_RE.fullmatch(expected_catalog_sha256 or "") is None
            or _SHA256_RE.fullmatch(expected_owner_authority_sha256 or "") is None
            or not callable(admission_publisher)
        ):
            raise ValueError("deferred API input preparation binding is invalid")

        approval.require(
            plan_sha256=self.plan.sha256,
            full_canary_plan_sha256=self.full_plan.sha256,
            now_unix=int(time.time()),
        )
        _require_installed_capability_approval(self.plan, self.full_plan, approval)
        self._require_host()
        with _lifecycle_lock():
            core_receipt, _runtime_pending = _load_deferred_lifecycle_state(
                self.plan, approval
            )
            if core_receipt is None:
                raise RuntimeError("deferred core lifecycle receipt is unavailable")
            self._validate_deferred_core_runtime(
                approval,
                core_receipt,
                require_producers_stopped=_runtime_pending is None,
            )
            if (
                core_receipt.get("gateway_runtime_readiness", {}).get("gateway_pid")
                != expected_gateway_pid
            ):
                raise RuntimeError("deferred gateway identity changed")
            prepared = _load_bound_deferred_stage(
                self.plan,
                approval,
                stage=CAPABILITY_API_INPUTS_PREPARED_STAGE,
                operation="prepare_api_admission_inputs",
                core_receipt_sha256=core_receipt["receipt_sha256"],
            )
            publication = admission_publisher()
            readback_verified = publication.get("readback_verified") is True or all(
                publication.get(field) is True
                for field in (
                    "catalog_readback_verified",
                    "grant_readback_verified",
                    "owner_receipt_readback_verified",
                    "authority_readback_verified",
                )
            )
            if (
                not isinstance(publication, Mapping)
                or publication.get("run_id") != expected_run_id
                or publication.get("session_id") != expected_session_id
                or publication.get("capability_epoch_sha256")
                != expected_capability_epoch_sha256
                or publication.get("catalog_sha256") != expected_catalog_sha256
                or publication.get("authority_sha256")
                != expected_owner_authority_sha256
                or readback_verified is not True
                or _SHA256_RE.fullmatch(
                    str(publication.get("receipt_sha256", ""))
                )
                is None
            ):
                raise RuntimeError("API admission publication is invalid")
            if prepared is not None:
                if (
                    prepared.get("api_admission_publication") != publication
                    or prepared.get("runtime_started") is not False
                    or prepared.get("gateway_commit_acknowledged") is not False
                    or prepared.get("model_release_allowed") is not False
                ):
                    raise RuntimeError("prepared API admission inputs drifted")
                return prepared, copy.deepcopy(dict(publication))
            receipt = _write_lifecycle_receipt(
                self.plan,
                stage=CAPABILITY_API_INPUTS_PREPARED_STAGE,
                value={
                    **_capability_approval_chain_fields(approval),
                    "operation": "prepare_api_admission_inputs",
                    "owner_approval_sha256": approval.sha256,
                    "core_start_receipt_sha256": core_receipt["receipt_sha256"],
                    "run_id": expected_run_id,
                    "session_id": expected_session_id,
                    "capability_epoch_sha256": (
                        expected_capability_epoch_sha256
                    ),
                    "catalog_sha256": expected_catalog_sha256,
                    "owner_authority_sha256": (
                        expected_owner_authority_sha256
                    ),
                    "api_admission_publication": copy.deepcopy(dict(publication)),
                    "inputs_readback_verified": True,
                    "runtime_started": False,
                    "gateway_commit_acknowledged": False,
                    "model_release_allowed": False,
                    "prepared_at_unix_ms": int(time.time() * 1000),
                },
            )
            return receipt, copy.deepcopy(dict(publication))

    def start_admitted_producers(
        self,
        approval: CapabilityCanaryOwnerApproval,
        *,
        expected_gateway_pid: int,
        expected_run_id: str,
        expected_session_id: str,
        expected_capability_epoch_sha256: str,
        expected_catalog_sha256: str,
        expected_owner_authority_sha256: str,
        api_admission_ready_ack: Mapping[str, Any],
        admission_publisher: Callable[[], Mapping[str, Any]],
        producer_fleet_activator: Callable[[], Any],
        producer_activation_retirer: Callable[[Any], Mapping[str, Any]],
        admission_input_retirer: Callable[[], Mapping[str, Any]],
    ) -> tuple[Mapping[str, Any], Any, Mapping[str, Any]]:
        """Finish a deferred start only after the gateway-owned epoch is signed.

        Owner signing/waiting happens before this method.  Every mutation from
        catalog publication through producer activation is serialized by the
        lifecycle lock, while the first model call remains blocked in the API
        adapter's admission callback.
        """

        _require_root_linux()
        if not isinstance(approval, CapabilityCanaryOwnerApproval):
            raise PermissionError("fresh exact capability owner approval is required")
        if (
            type(expected_gateway_pid) is not int
            or expected_gateway_pid <= 1
            or not isinstance(expected_run_id, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,239}", expected_run_id)
            is None
            or not isinstance(expected_session_id, str)
            or not expected_session_id
            or len(expected_session_id.encode("utf-8", errors="strict")) > 256
            or _SHA256_RE.fullmatch(expected_capability_epoch_sha256 or "") is None
            or _SHA256_RE.fullmatch(expected_catalog_sha256 or "") is None
            or _SHA256_RE.fullmatch(expected_owner_authority_sha256 or "") is None
            or not isinstance(api_admission_ready_ack, Mapping)
            or not callable(admission_publisher)
            or not callable(producer_fleet_activator)
            or not callable(producer_activation_retirer)
            or not callable(admission_input_retirer)
        ):
            raise ValueError("deferred producer admission binding is invalid")
        ready_ack_unsigned = {
            key: item
            for key, item in api_admission_ready_ack.items()
            if key != "receipt_sha256"
        }
        if (
            set(api_admission_ready_ack)
            != {
                "schema",
                "session_id",
                "capability_epoch_sha256",
                "challenge_sha256",
                "ready_receipt_sha256",
                "acknowledged",
                "receipt_sha256",
            }
            or api_admission_ready_ack.get("schema")
            != "hermes.api.run-admission-ready-ack.v1"
            or api_admission_ready_ack.get("session_id") != expected_session_id
            or api_admission_ready_ack.get("capability_epoch_sha256")
            != expected_capability_epoch_sha256
            or api_admission_ready_ack.get("acknowledged") is not True
            or any(
                _SHA256_RE.fullmatch(
                    str(api_admission_ready_ack.get(field, ""))
                )
                is None
                for field in (
                    "challenge_sha256",
                    "ready_receipt_sha256",
                    "receipt_sha256",
                )
            )
            or api_admission_ready_ack.get("receipt_sha256")
            != _sha256_json(ready_ack_unsigned)
        ):
            raise ValueError("deferred API ready acknowledgement is invalid")

        def require_approval() -> None:
            now = int(time.time())
            approval.require(
                plan_sha256=self.plan.sha256,
                full_canary_plan_sha256=self.full_plan.sha256,
                now_unix=now,
            )
            _require_installed_capability_approval(
                self.plan,
                self.full_plan,
                approval,
            )

        require_approval()
        self._require_host()
        activation: Any | None = None
        publication: Mapping[str, Any] | None = None
        producer_started: list[str] = []
        core_receipt: Mapping[str, Any] | None = None
        terminal_generation_observed = False
        try:
            with _lifecycle_lock():
                durable_core, durable_terminal = _load_deferred_lifecycle_state(
                    self.plan,
                    approval,
                )
                if durable_core is None:
                    raise RuntimeError("deferred core lifecycle receipt is unavailable")
                prepared_receipt = _load_bound_deferred_stage(
                    self.plan,
                    approval,
                    stage=CAPABILITY_API_INPUTS_PREPARED_STAGE,
                    operation="prepare_api_admission_inputs",
                    core_receipt_sha256=durable_core["receipt_sha256"],
                )
                if prepared_receipt is None:
                    raise RuntimeError("deferred API inputs are not durably prepared")
                if durable_terminal is not None:
                    terminal_generation_observed = True
                    terminal_publication = durable_terminal.get(
                        "api_admission_publication"
                    )
                    terminal_readiness = durable_terminal.get("producer_readiness")
                    if (
                        durable_core is None
                        or durable_terminal.get("core_start_receipt_sha256")
                        != durable_core.get("receipt_sha256")
                        or not isinstance(terminal_publication, Mapping)
                        or not isinstance(terminal_readiness, Mapping)
                        or terminal_publication.get("run_id") != expected_run_id
                        or terminal_publication.get("session_id")
                        != expected_session_id
                        or terminal_publication.get("capability_epoch_sha256")
                        != expected_capability_epoch_sha256
                        or terminal_publication.get("catalog_sha256")
                        != expected_catalog_sha256
                        or terminal_publication.get("authority_sha256")
                        != expected_owner_authority_sha256
                        or terminal_readiness.get("catalog_sha256")
                        != expected_catalog_sha256
                        or terminal_readiness.get("owner_authority_sha256")
                        != expected_owner_authority_sha256
                        or durable_terminal.get("api_admission_ready_ack")
                        != api_admission_ready_ack
                        or durable_terminal.get("prepared_inputs_receipt_sha256")
                        != prepared_receipt.get("receipt_sha256")
                        or durable_terminal.get("runtime_started") is not True
                        or durable_terminal.get("gateway_commit_acknowledged")
                        is not False
                        or durable_terminal.get("model_release_allowed") is not False
                    ):
                        raise PermissionError(
                            "deferred terminal generation does not match contender"
                        )
                    self._validate_deferred_core_runtime(
                        approval,
                        durable_core,
                        require_producers_stopped=False,
                    )
                    for role in CAPABILITY_PRODUCER_ROLES:
                        state = collect_capability_service_state(
                            CAPABILITY_PRODUCER_SERVICE_UNITS[role],
                            runner=self.runner,
                        )
                        if not _service_live(
                            state,
                            path=CAPABILITY_PRODUCER_UNIT_PATHS[role],
                            service_type="simple",
                        ):
                            raise PermissionError(
                                "terminal deferred generation is no longer live"
                            )
                    publication = admission_publisher()
                    activation = producer_fleet_activator()
                    readiness = getattr(activation, "readiness", None)
                    if (
                        not isinstance(publication, Mapping)
                        or publication.get("receipt_sha256")
                        != terminal_publication.get("receipt_sha256")
                        or not isinstance(readiness, Mapping)
                        or readiness.get("readiness_sha256")
                        != terminal_readiness.get("readiness_sha256")
                    ):
                        raise RuntimeError(
                            "deferred terminal generation recovery drifted"
                        )
                    return durable_terminal, activation, publication
                core_receipt = dict(durable_core)
                pending = self._validate_deferred_core_runtime(
                    approval,
                    durable_core,
                    # The prepared owner/admission receipt is already exact at
                    # this point.  A process death may have occurred after one
                    # or all producer units started but before terminal
                    # lifecycle truth was appended.  ``systemctl start`` and
                    # fleet activation are idempotent, so resume this exact
                    # generation instead of misclassifying it as a pre-owner
                    # producer escape and falling into generic cleanup.
                    require_producers_stopped=False,
                )
                core_path = Path(str(core_receipt.get("receipt_path", "")))
                core_raw, _core_item = _read_stable_file(
                    core_path,
                    maximum=2 * 1024 * 1024,
                    expected_uid=0,
                    expected_gid=0,
                    allowed_modes=frozenset({0o400}),
                )
                if (
                    core_raw != _canonical_bytes(core_receipt)
                    or core_receipt.get("stage")
                    != CAPABILITY_GATEWAY_CORE_READY_STAGE
                    or core_receipt.get("operation")
                    != "start_core_before_api_admission"
                    or core_receipt.get("plan_sha256") != self.plan.sha256
                    or core_receipt.get("owner_approval_sha256") != approval.sha256
                    or core_receipt.get("producer_units_started") is not False
                    or core_receipt.get("api_admission_pending") is not True
                    or core_receipt.get("receipt_sha256")
                    != _sha256_json({
                        key: item
                        for key, item in core_receipt.items()
                        if key != "receipt_sha256"
                    })
                ):
                    raise RuntimeError("deferred core lifecycle receipt drifted")
                if pending["gateway_readiness"].get(
                    "gateway_pid"
                ) != expected_gateway_pid:
                    raise RuntimeError("deferred gateway identity changed")
                require_approval()
                publication = admission_publisher()
                if (
                    not isinstance(publication, Mapping)
                    or publication.get("run_id") != expected_run_id
                    or publication.get("session_id") != expected_session_id
                    or publication.get("capability_epoch_sha256")
                    != expected_capability_epoch_sha256
                    or publication.get("catalog_sha256") != expected_catalog_sha256
                    or publication.get("authority_sha256")
                    != expected_owner_authority_sha256
                    or publication.get("readback_verified") is not True
                    or _SHA256_RE.fullmatch(str(publication.get("receipt_sha256", "")))
                    is None
                    or publication
                    != prepared_receipt.get("api_admission_publication")
                ):
                    raise RuntimeError("API admission publication is invalid")
                for role in CAPABILITY_PRODUCER_ROLES:
                    require_approval()
                    unit = CAPABILITY_PRODUCER_SERVICE_UNITS[role]
                    _run_checked(
                        Command((SYSTEMCTL, "start", unit), timeout_seconds=180),
                        runner=self.runner,
                        label=f"start admitted {unit}",
                    )
                    producer_started.append(unit)

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
                                f"admitted producer {producer_role} is not live"
                            )
                        return {
                            "role": producer_role,
                            "unit": producer_unit,
                            "main_pid": state["MainPID"],
                            "ready": True,
                        }

                    _await_runtime_ready(
                        producer_ready,
                        label=f"admitted capability producer {role}",
                    )
                if tuple(producer_started) != CAPABILITY_ADMITTED_PRODUCER_START_ORDER:
                    raise RuntimeError("admitted producer start order drifted")
                require_approval()
                activation = producer_fleet_activator()
                readiness = getattr(activation, "readiness", None)
                if (
                    not isinstance(readiness, Mapping)
                    or readiness.get("catalog_sha256") != expected_catalog_sha256
                    or readiness.get("owner_authority_sha256")
                    != expected_owner_authority_sha256
                    or _SHA256_RE.fullmatch(str(readiness.get("readiness_sha256", "")))
                    is None
                ):
                    raise RuntimeError("admitted producer fleet readiness is invalid")
                live = collect_capability_preflight(
                    self.plan,
                    self.full_plan,
                    phase="live",
                    runner=self.runner,
                    metadata_reader=self.metadata_reader,
                    local_identity_reader=self.local_identity_reader,
                    approval_window_started_at_unix=approval.value[
                        "stopped_preflight_observed_at_unix"
                    ],
                )
                final_receipt = _write_lifecycle_receipt(
                    self.plan,
                    stage=CAPABILITY_RUNTIME_PENDING_ACK_STAGE,
                    value={
                        **_capability_approval_chain_fields(approval),
                        "operation": "runtime_live_pending_gateway_commit_ack",
                        "owner_approval_sha256": approval.sha256,
                        "core_start_receipt_sha256": core_receipt["receipt_sha256"],
                        "prepared_inputs_receipt_sha256": prepared_receipt[
                            "receipt_sha256"
                        ],
                        "api_admission_ready_ack": copy.deepcopy(
                            dict(api_admission_ready_ack)
                        ),
                        "full_canary_stopped_preflight_sha256": pending[
                            "full_preflight"
                        ]["report_sha256"],
                        "stopped_preflight_sha256": pending["preflight"][
                            "report_sha256"
                        ],
                        "live_preflight_sha256": live["report_sha256"],
                        "installed_artifacts": copy.deepcopy(
                            dict(pending["installed"])
                        ),
                        "connector_state": copy.deepcopy(
                            dict(pending["connector_state"])
                        ),
                        "phase_b_current_readiness": copy.deepcopy(
                            dict(pending["phase_b_current"])
                        ),
                        "phase_b_full_canary_anchor": copy.deepcopy(
                            dict(pending["installed_phase_b_anchor"])
                        ),
                        "writer_runtime_readiness": copy.deepcopy(
                            dict(pending["writer_readiness"])
                        ),
                        "gateway_runtime_readiness": copy.deepcopy(
                            dict(pending["gateway_readiness"])
                        ),
                        "observer_config": copy.deepcopy(dict(pending["observer"])),
                        "browser_identity_foundation": copy.deepcopy(
                            dict(pending["browser_identity_foundation"])
                        ),
                        "browser_principal_smoke": copy.deepcopy(
                            dict(pending["browser_principal_smoke"])
                        ),
                        "execution_identity_foundation": copy.deepcopy(
                            dict(pending["execution_identity_foundation"])
                        ),
                        "worker_mountpoint": copy.deepcopy(
                            dict(pending["worker_mountpoint"])
                        ),
                        "execution_readiness": copy.deepcopy(
                            dict(pending["execution_readiness"])
                        ),
                        "routeback_bot_identity": copy.deepcopy(
                            dict(pending["routeback_bot_identity"])
                        ),
                        "producer_foundation": copy.deepcopy(
                            dict(pending["producer_foundation"])
                        ),
                        "api_admission_publication": copy.deepcopy(dict(publication)),
                        "producer_readiness": copy.deepcopy(dict(readiness)),
                        "credential_bindings": _credential_bindings_mapping(),
                        "start_order": list(CAPABILITY_DEFERRED_START_ORDER),
                        "runtime_started": True,
                        "gateway_commit_acknowledged": False,
                        "model_release_allowed": False,
                        "model_callback_released": False,
                        "units_enabled": False,
                        "runtime_max_seconds": 900,
                        "started_at_unix": int(time.time()),
                    },
                )
                self._pending_deferred_start = None
                return final_receipt, activation, publication
        except BaseException as error:
            if terminal_generation_observed:
                raise
            cleanup_error: BaseException | None = None
            try:
                self._cleanup(
                    producer_activation_retirer=(
                        (lambda: producer_activation_retirer(activation))
                        if activation is not None
                        else None
                    ),
                    admission_input_retirer=admission_input_retirer,
                )
            except BaseException as exc:
                cleanup_error = exc
            try:
                _write_lifecycle_receipt(
                    self.plan,
                    stage="failure",
                    value={
                        **_capability_approval_chain_fields(approval),
                        "operation": "complete_start_after_api_admission",
                        "owner_approval_sha256": approval.sha256,
                        "core_start_receipt_sha256": (
                            core_receipt.get("receipt_sha256")
                            if isinstance(core_receipt, Mapping)
                            else None
                        ),
                        "run_id": expected_run_id,
                        "session_id": expected_session_id,
                        "capability_epoch_sha256": (expected_capability_epoch_sha256),
                        "catalog_sha256": expected_catalog_sha256,
                        "owner_authority_sha256": (expected_owner_authority_sha256),
                        "producer_units_started_before_failure": producer_started,
                        "admission_publication_sha256": (
                            publication.get("receipt_sha256")
                            if isinstance(publication, Mapping)
                            else None
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
                    "deferred producer failure receipt/cleanup failed",
                    [
                        *([cleanup_error] if cleanup_error is not None else []),
                        receipt_error,
                    ],
                )
            if cleanup_error is not None:
                raise BaseExceptionGroup(
                    "deferred producer start and cleanup failed",
                    [error, cleanup_error],
                ) from None
            raise RuntimeError("deferred producer start failed closed") from error

    def finalize_api_admission_gateway_commit(
        self,
        approval: CapabilityCanaryOwnerApproval,
        *,
        runtime_pending_receipt: Mapping[str, Any],
        api_admission_commit_ack: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Record gateway acknowledgement without claiming model execution."""

        _require_root_linux()
        if not isinstance(approval, CapabilityCanaryOwnerApproval):
            raise PermissionError("fresh exact capability owner approval is required")
        if not isinstance(runtime_pending_receipt, Mapping) or not isinstance(
            api_admission_commit_ack, Mapping
        ):
            raise ValueError("gateway admission commit binding is invalid")
        commit_ack_unsigned = {
            key: item
            for key, item in api_admission_commit_ack.items()
            if key != "receipt_sha256"
        }
        if (
            set(api_admission_commit_ack)
            != {
                "schema",
                "session_id",
                "capability_epoch_sha256",
                "challenge_sha256",
                "commit_receipt_sha256",
                "acknowledged",
                "receipt_sha256",
            }
            or api_admission_commit_ack.get("schema")
            != "hermes.api.run-admission-commit-ack.v1"
            or api_admission_commit_ack.get("acknowledged") is not True
            or any(
                _SHA256_RE.fullmatch(
                    str(api_admission_commit_ack.get(field, ""))
                )
                is None
                for field in (
                    "capability_epoch_sha256",
                    "challenge_sha256",
                    "commit_receipt_sha256",
                    "receipt_sha256",
                )
            )
            or api_admission_commit_ack.get("receipt_sha256")
            != _sha256_json(commit_ack_unsigned)
        ):
            raise ValueError("gateway admission commit acknowledgement is invalid")
        approval.require(
            plan_sha256=self.plan.sha256,
            full_canary_plan_sha256=self.full_plan.sha256,
            now_unix=int(time.time()),
        )
        _require_installed_capability_approval(self.plan, self.full_plan, approval)
        self._require_host()
        with _lifecycle_lock():
            core_receipt, durable_pending = _load_deferred_lifecycle_state(
                self.plan, approval
            )
            if core_receipt is None or durable_pending is None:
                raise RuntimeError("pending admission runtime is unavailable")
            if durable_pending != runtime_pending_receipt:
                raise RuntimeError("pending admission runtime receipt drifted")
            publication = durable_pending.get("api_admission_publication")
            readiness = durable_pending.get("producer_readiness")
            if (
                durable_pending.get("stage")
                != CAPABILITY_RUNTIME_PENDING_ACK_STAGE
                or durable_pending.get("runtime_started") is not True
                or durable_pending.get("gateway_commit_acknowledged") is not False
                or durable_pending.get("model_release_allowed") is not False
                or not isinstance(publication, Mapping)
                or not isinstance(readiness, Mapping)
                or api_admission_commit_ack.get("session_id")
                != publication.get("session_id")
                or api_admission_commit_ack.get("capability_epoch_sha256")
                != publication.get("capability_epoch_sha256")
                or api_admission_commit_ack.get("challenge_sha256")
                != publication.get("challenge_sha256")
            ):
                raise RuntimeError("pending admission truth is invalid")
            existing = _load_bound_deferred_stage(
                self.plan,
                approval,
                stage=CAPABILITY_GATEWAY_ACK_PRE_MODEL_STAGE,
                operation="gateway_commit_acknowledged_pre_model",
                core_receipt_sha256=core_receipt["receipt_sha256"],
            )
            if existing is not None:
                if (
                    existing.get("runtime_pending_receipt_sha256")
                    != durable_pending.get("receipt_sha256")
                    or existing.get("api_admission_commit_ack")
                    != api_admission_commit_ack
                    or existing.get("gateway_commit_acknowledged") is not True
                    or existing.get("model_release_allowed") is not True
                    or existing.get("model_callback_released") is not False
                ):
                    raise RuntimeError("gateway acknowledgement truth drifted")
                return existing
            return _write_lifecycle_receipt(
                self.plan,
                stage=CAPABILITY_GATEWAY_ACK_PRE_MODEL_STAGE,
                value={
                    **_capability_approval_chain_fields(approval),
                    "operation": "gateway_commit_acknowledged_pre_model",
                    "owner_approval_sha256": approval.sha256,
                    "core_start_receipt_sha256": core_receipt["receipt_sha256"],
                    "runtime_pending_receipt_sha256": durable_pending[
                        "receipt_sha256"
                    ],
                    "run_id": publication["run_id"],
                    "session_id": publication["session_id"],
                    "capability_epoch_sha256": publication[
                        "capability_epoch_sha256"
                    ],
                    "api_admission_publication_sha256": publication[
                        "receipt_sha256"
                    ],
                    "producer_readiness_sha256": readiness["readiness_sha256"],
                    "api_admission_commit_ack": copy.deepcopy(
                        dict(api_admission_commit_ack)
                    ),
                    "runtime_started": True,
                    "gateway_commit_acknowledged": True,
                    "model_release_allowed": True,
                    "model_callback_released": False,
                    "acknowledged_at_unix_ms": int(time.time() * 1000),
                },
            )

    def stop(
        self,
        *,
        cleanup_producer: Callable[[Mapping[str, Any]], Mapping[str, Any]]
        | None = None,
        cleanup_receipt_loader: Callable[[], Mapping[str, Any] | None]
        | None = None,
        cleanup_receipt_verifier: Callable[[Mapping[str, Any]], Any] | None = None,
        cleanup_run_id: str | None = None,
        cleanup_fixture_sha256: str | None = None,
        producer_activation_retirer: Callable[[], Mapping[str, Any]] | None = None,
        admission_input_retirer: Callable[[], Mapping[str, Any]] | None = None,
    ) -> Mapping[str, Any]:
        try:
            cleanup = self._cleanup(
                cleanup_producer=cleanup_producer,
                cleanup_receipt_loader=cleanup_receipt_loader,
                cleanup_receipt_verifier=cleanup_receipt_verifier,
                cleanup_run_id=cleanup_run_id,
                cleanup_fixture_sha256=cleanup_fixture_sha256,
                producer_activation_retirer=producer_activation_retirer,
                admission_input_retirer=admission_input_retirer,
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


_BITRIX_FOUNDATION_AUTHORITY_FIELDS = frozenset({
    "schema",
    "scope",
    "revision",
    "full_canary_plan_sha256",
    "full_canary_terminal_receipt",
    "full_canary_terminal_receipt_sha256",
    "original_full_canary_owner_approval_sha256",
    "foundation_authoring_context_receipt_sha256",
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
})
_BITRIX_FOUNDATION_IDENTITY_FIELDS = frozenset({
    "service_uid",
    "service_gid",
    "socket_client_gid",
    "business_edge_uid",
})
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
        or raw["authority_kind"] != "trusted_gcloud_owner_explicit_foundation_digest"
        or raw["cryptographic_owner_proof"] is not False
        or raw["secret_material_recorded"] is not False
        or raw["secret_digest_recorded"] is not False
        or raw["semantic_content_recorded"] is not False
        or not isinstance(raw["revision"], str)
        or _REVISION_RE.fullmatch(raw["revision"]) is None
        or type(raw["issued_at_unix"]) is not int
        or type(raw["expires_at_unix"]) is not int
        or type(now) is not int
        or not raw["issued_at_unix"] <= now < raw["expires_at_unix"]
        or not 60
        <= raw["expires_at_unix"] - raw["issued_at_unix"]
        <= _BITRIX_FOUNDATION_MAX_SECONDS
        or any(
            type(identities[field]) is not int or not 0 < identities[field] < (1 << 31)
            for field in _BITRIX_FOUNDATION_IDENTITY_FIELDS
        )
        or len(set(identities.values())) != len(identities)
    ):
        raise ValueError("Bitrix foundation authority is invalid")
    for field in (
        "full_canary_plan_sha256",
        "full_canary_terminal_receipt_sha256",
        "original_full_canary_owner_approval_sha256",
        "foundation_authoring_context_receipt_sha256",
        "release_artifact_sha256",
        "owner_subject_sha256",
        "asset_manifest_sha256",
    ):
        _digest(raw[field], f"Bitrix foundation authority {field}")
    terminal, terminal_sha256 = _terminal_receipt_binding(
        raw["full_canary_terminal_receipt"],
        raw["full_canary_terminal_receipt_sha256"],
        revision=raw["revision"],
        full_canary_plan_sha256=raw["full_canary_plan_sha256"],
    )
    if (
        terminal_sha256 != raw["full_canary_terminal_receipt_sha256"]
        or raw["original_full_canary_owner_approval_sha256"]
        != terminal["owner_approval_sha256"]
    ):
        raise ValueError("Bitrix foundation terminal binding is invalid")
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
    if not raw or len(raw) > _MAX_BITRIX_FOUNDATION_AUTHORITY_BYTES or stream.read(1):
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
        "disarm_intent": state_root / watchdog_id / "disarm-intent.json",
        "disarm_completion": state_root / watchdog_id / "disarm-completion.json",
        "reconciliations": state_root / watchdog_id / "reconciliations",
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
    minimum_reserve_seconds: int = 0,
) -> Mapping[str, Any]:
    """Install and enable a persistent absolute timer before secret commit."""

    if kind not in {"bitrix_foundation", "credential_lease"}:
        raise ValueError("capability expiry watchdog kind is invalid")
    now = int(time.time()) if now_unix is None else now_unix
    if (
        _REVISION_RE.fullmatch(revision or "") is None
        or type(expires_at_unix) is not int
        or type(now) is not int
        or expires_at_unix <= now
    ):
        raise ValueError("capability expiry watchdog window is invalid")
    if minimum_reserve_seconds:
        _require_remaining_reserve(
            expires_at_unix=expires_at_unix,
            now_unix=now,
            minimum_seconds=minimum_reserve_seconds,
        )
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
    watchdog_id = _sha256_json({
        "kind": kind,
        "authority_sha256": authority_sha256,
        "expires_at_unix": expires_at_unix,
        "credential_binding": credential_binding,
    })[:32]
    paths = _expiry_watchdog_paths(
        watchdog_id,
        state_root=state_root,
        systemd_root=systemd_root,
    )
    if require_root:
        _require_root_linux()
        if state_root != DEFAULT_EXPIRY_WATCHDOG_ROOT or systemd_root != Path(
            "/etc/systemd/system"
        ):
            raise ValueError("capability expiry watchdog production paths are fixed")
    else:
        _prepare_bitrix_foundation_directory(state_root, require_root=False)
        _prepare_bitrix_foundation_directory(systemd_root, require_root=False)
    try:
        watchdog_names = sorted(os.listdir(state_root))
    except FileNotFoundError:
        watchdog_names = []
    if any(_LEASE_ID_RE.fullmatch(name) is None for name in watchdog_names):
        raise RuntimeError("capability expiry watchdog inventory is invalid")
    if (
        watchdog_id not in watchdog_names
        and len(watchdog_names) >= _MAX_EXPIRY_WATCHDOGS
    ):
        raise RuntimeError("capability expiry watchdog admission is full")
    disarm_intent_path = Path(paths["disarm_intent"])
    disarm_completion_path = Path(paths["disarm_completion"])
    if os.path.lexists(disarm_intent_path) or os.path.lexists(
        disarm_completion_path
    ):
        # A normal lifecycle stop is itself durable authority to retire this
        # deterministic watchdog generation.  Finish a crash-interrupted
        # disarm, then reject any attempt to recreate the same timer.
        _disarm_capability_expiry_watchdog(
            watchdog_id,
            runner=runner,
            state_root=state_root,
            systemd_root=systemd_root,
            require_root=require_root,
        )
        raise PermissionError("normally disarmed expiry watchdog cannot be rearmed")
    completion_path = Path(paths["completion"])
    if os.path.lexists(completion_path):
        completion = _load_lease_artifact(
            completion_path,
            schema=CAPABILITY_EXPIRY_WATCHDOG_COMPLETION_SCHEMA,
        )
        prior_authority = _load_expiry_watchdog_authority(
            watchdog_id,
            state_root=state_root,
            systemd_root=systemd_root,
        )
        if (
            completion.get("watchdog_id") != watchdog_id
            or completion.get("watchdog_authority_sha256")
            != prior_authority.get("receipt_sha256")
            or completion.get("ok") is not True
            or prior_authority.get("kind") != kind
            or prior_authority.get("revision") != revision
            or prior_authority.get("full_canary_plan_sha256")
            != full_canary_plan_sha256
            or prior_authority.get("release_artifact_sha256")
            != release_artifact_sha256
            or prior_authority.get("plan_sha256") != plan_sha256
            or prior_authority.get("credential_binding") != credential_binding
            or prior_authority.get("authority_source_sha256") != authority_sha256
            or prior_authority.get("expires_at_unix") != expires_at_unix
            or prior_authority.get("interpreter") != str(interpreter)
        ):
            raise RuntimeError("capability expiry watchdog completion drifted")
        bound_prior_active_run = _validate_expiry_active_run_retirement(
            completion.get("active_run_retirement"),
            authority=prior_authority,
            require_current_absence=False,
        )
        if (
            bound_prior_active_run["receipt_sha256"]
            != completion.get("active_run_retirement_sha256")
        ):
            raise RuntimeError("capability expiry watchdog active-run binding drifted")
        raise PermissionError("completed capability expiry watchdog cannot be rearmed")
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
    owner = (
        0 if require_root else effective_uid()
    )
    group = (
        0 if require_root else effective_gid()
    )
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
    os.chown(path, effective_uid(), effective_gid())
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
    passwd_slot_rows, service_primary_users = _capability_passwd_slot_inventory(
        "muncho-edge-bitrix",
        service_uid,
        service_gid,
    )
    service_group_rows = _capability_group_slot_inventory(
        "muncho-edge-bitrix",
        service_gid,
    )
    client_primary_users = _capability_primary_group_user_names(socket_client_gid)
    client_group_rows = _capability_group_slot_inventory(
        "muncho-edge-bitrix-c",
        socket_client_gid,
    )
    expected_passwd = (
        "muncho-edge-bitrix",
        service_uid,
        service_gid,
        "/nonexistent",
        "/usr/sbin/nologin",
    )
    expected_service_group = ("muncho-edge-bitrix", service_gid, ())
    expected_client_group = ("muncho-edge-bitrix-c", socket_client_gid, ())
    service_absent = all(
        item is None for item in (user, uid_owner, group, gid_owner)
    ) and (
        passwd_slot_rows == []
        and service_primary_users == []
        and service_group_rows == []
    )
    service_group_only = (
        user is None
        and uid_owner is None
        and group is not None
        and gid_owner is not None
        and group.gr_name == "muncho-edge-bitrix"
        and group.gr_gid == service_gid
        and gid_owner.gr_name == "muncho-edge-bitrix"
        and list(group.gr_mem) == []
        and passwd_slot_rows == []
        and service_primary_users == []
        and service_group_rows == [expected_service_group]
    )
    service_present = (
        user is not None
        and uid_owner is not None
        and group is not None
        and gid_owner is not None
        and user.pw_name == "muncho-edge-bitrix"
        and user.pw_uid == service_uid
        and user.pw_gid == service_gid
        and user.pw_dir == "/nonexistent"
        and user.pw_shell == "/usr/sbin/nologin"
        and uid_owner.pw_name == "muncho-edge-bitrix"
        and group.gr_name == "muncho-edge-bitrix"
        and group.gr_gid == service_gid
        and gid_owner.gr_name == "muncho-edge-bitrix"
        and list(group.gr_mem) == []
        and passwd_slot_rows == [expected_passwd]
        and service_primary_users == ["muncho-edge-bitrix"]
        and service_group_rows == [expected_service_group]
    )
    if service_present and sorted(
        set(os.getgrouplist("muncho-edge-bitrix", service_gid))
    ) != [service_gid]:
        raise RuntimeError("Bitrix foundation service user has extra authority")
    client_absent = (
        client is None
        and client_gid_owner is None
        and client_primary_users == []
        and client_group_rows == []
    )
    client_present = (
        client is not None
        and client_gid_owner is not None
        and client.gr_name == "muncho-edge-bitrix-c"
        and client.gr_gid == socket_client_gid
        and client_gid_owner.gr_name == "muncho-edge-bitrix-c"
        and list(client.gr_mem) == []
        and client_primary_users == []
        and client_group_rows == [expected_client_group]
    )
    if service_present and client_present:
        state = "present_exact"
    elif service_group_only and client_present:
        state = "groups_present_user_absent_create_only_slot"
    elif service_group_only and client_absent:
        state = "service_group_present_create_only_slot"
    elif service_absent and client_absent:
        state = "absent_create_only_slot"
    else:
        raise RuntimeError("Bitrix foundation identity slot collides or drifted")
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
                Command((
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
                )),
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
    owner = (
        0 if require_root else effective_uid()
    )
    group = (
        0 if require_root else effective_gid()
    )
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
        sorted({
            identities["business_edge_uid"],
            full_plan.identities.writer_uid,
        })
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


def _bootstrap_bitrix_foundation_locked(
    authority_value: Any,
    *,
    full_plan: FullCanaryPlan | None = None,
    runner: Callable[[Command], subprocess.CompletedProcess[bytes]] = _runner,
    identity_observer: Callable[
        ..., Mapping[str, Any]
    ] = _observe_bitrix_foundation_identity,
    asset_verifier: Callable[
        ..., Mapping[str, Any]
    ] = verify_packaged_operational_assets,
    private_key_path: Path = DEFAULT_BITRIX_RECEIPT_PRIVATE_KEY_PATH,
    public_key_path: Path = DEFAULT_BITRIX_TRUST_PATH,
    writer_public_key_path: Path = DEFAULT_BITRIX_WRITER_PUBLIC_KEY_PATH,
    identity_receipt_path: Path = DEFAULT_BITRIX_IDENTITY_BOOTSTRAP_RECEIPT,
    foundation_root: Path = DEFAULT_BITRIX_FOUNDATION_ROOT,
    key_bootstrap_root: Path = DEFAULT_BITRIX_KEY_BOOTSTRAP_ROOT,
    require_root: bool = True,
    operation_clock: _OperationClock,
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
    authority = validate_bitrix_foundation_authority(
        authority_value,
        now_unix=operation_clock.sample("Bitrix foundation admission"),
    )
    abort_path = foundation_root / authority["authority_sha256"] / "abort.json"
    if os.path.lexists(abort_path):
        abort = _load_lease_artifact(
            abort_path,
            schema=CAPABILITY_BITRIX_FOUNDATION_ABORT_SCHEMA,
        )
        if (
            abort.get("authority_sha256") != authority["authority_sha256"]
            or abort.get("private_absent") is not True
            or abort.get("public_absent") is not True
        ):
            raise RuntimeError("Bitrix foundation abort receipt drifted")
        raise PermissionError("aborted Bitrix foundation authority cannot be reused")
    _require_remaining_reserve(
        expires_at_unix=authority["expires_at_unix"],
        now_unix=operation_clock.sample("Bitrix foundation reserve"),
    )
    full = load_full_canary_plan() if full_plan is None else full_plan
    if (
        authority["revision"] != full.revision
        or authority["full_canary_plan_sha256"] != full.sha256
        or authority["full_canary_terminal_receipt"]["release_sha"] != full.revision
        or authority["full_canary_terminal_receipt"]["full_canary_plan_sha256"]
        != full.sha256
        or authority["release_artifact_sha256"] != full.release["artifact_sha256"]
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
        now_unix=operation_clock.sample("Bitrix foundation watchdog arm"),
        minimum_reserve_seconds=CAPABILITY_MUTATION_MIN_RESERVE_SECONDS,
    )
    assets = asset_verifier(
        release_root=Path(full.release["artifact_root"]),
        revision=full.revision,
        expected_uid=0,
        expected_gid=0,
        expected_manifest_sha256=authority["asset_manifest_sha256"],
    )
    rows = {row["asset_id"]: row for row in assets["files"]}
    if any(
        asset_id not in rows for asset_id in BITRIX_OPERATIONAL_EDGE_ASSET_IDS.values()
    ):
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
    _require_remaining_reserve(
        expires_at_unix=authority["expires_at_unix"],
        now_unix=operation_clock.sample("Bitrix key-pair publication"),
    )
    _append_lease_artifact(
        foundation_root / authority["authority_sha256"] / "key-stage-intent.json",
        schema=CAPABILITY_BITRIX_FOUNDATION_KEY_STAGE_INTENT_SCHEMA,
        value={
            "operation": "stage_bitrix_foundation_key_pair_intent",
            "revision": authority["revision"],
            "full_canary_plan_sha256": authority["full_canary_plan_sha256"],
            "authority_sha256": authority["authority_sha256"],
            "private_path": str(private_key_path),
            "public_path": str(public_key_path),
            "expires_at_unix": authority["expires_at_unix"],
            "private_content_or_digest_recorded": False,
        },
    )
    key = _stage_bitrix_receipt_key_pair(
        private_path=private_key_path,
        public_path=public_key_path,
        require_root=require_root,
    )
    owner = (
        0 if require_root else effective_uid()
    )
    group = (
        0 if require_root else effective_gid()
    )
    writer_raw, _writer_item = _read_exact_file(
        writer_public_key_path,
        maximum=16 * 1024,
        uid=owner,
        gid=group,
        mode=0o444,
    )
    writer_public_key_id = ed25519_public_key_id(_load_exact_ed25519_public(writer_raw))
    key_receipt_path = key_bootstrap_root / key["public_key_id"] / "bootstrap.json"
    key_receipt = _append_lease_artifact(
        key_receipt_path,
        schema=CAPABILITY_BITRIX_KEY_BOOTSTRAP_SCHEMA,
        value={
            "operation": "create_or_attest_receipt_key_pair",
            "revision": full.revision,
            "full_canary_plan_sha256": full.sha256,
            "full_canary_terminal_receipt": copy.deepcopy(
                dict(authority["full_canary_terminal_receipt"])
            ),
            "full_canary_terminal_receipt_sha256": authority[
                "full_canary_terminal_receipt_sha256"
            ],
            "original_full_canary_owner_approval_sha256": authority[
                "original_full_canary_owner_approval_sha256"
            ],
            "foundation_authoring_context_receipt_sha256": authority[
                "foundation_authoring_context_receipt_sha256"
            ],
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
    _append_lease_artifact(
        key_bootstrap_root
        / ".authority-index"
        / authority["authority_sha256"]
        / "key-bootstrap.json",
        schema=CAPABILITY_BITRIX_KEY_AUTHORITY_INDEX_SCHEMA,
        value={
            "operation": "index_bitrix_key_bootstrap_by_foundation_authority",
            "revision": full.revision,
            "full_canary_plan_sha256": full.sha256,
            "authority_sha256": authority["authority_sha256"],
            "public_key_id": key["public_key_id"],
            "key_bootstrap_receipt_path": key_receipt["receipt_path"],
            "key_bootstrap_receipt_sha256": key_receipt["receipt_sha256"],
            "expires_at_unix": authority["expires_at_unix"],
            "append_only_history": True,
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
            "full_canary_terminal_receipt": copy.deepcopy(
                dict(authority["full_canary_terminal_receipt"])
            ),
            "full_canary_terminal_receipt_sha256": authority[
                "full_canary_terminal_receipt_sha256"
            ],
            "original_full_canary_owner_approval_sha256": authority[
                "original_full_canary_owner_approval_sha256"
            ],
            "foundation_authoring_context_receipt_sha256": authority[
                "foundation_authoring_context_receipt_sha256"
            ],
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
            "read_peer_uids": sorted([
                identities["business_edge_uid"],
                full.identities.writer_uid,
            ]),
            "mutation_peer_uid": full.identities.writer_uid,
            "private_content_or_digest_recorded": False,
        },
    )
    _require_remaining_reserve(
        expires_at_unix=authority["expires_at_unix"],
        now_unix=operation_clock.sample("Bitrix foundation commit"),
    )
    return copy.deepcopy(dict(foundation_receipt))


def _retire_bitrix_bootstrap_pair_after_abort(
    authority: Mapping[str, Any],
    *,
    private_path: Path,
    public_path: Path,
    abort_root: Path,
    now_unix: int,
    require_root: bool,
) -> Mapping[str, Any]:
    """Retire an in-flight key pair without waiting for authority expiry."""

    authority_sha256 = _digest(
        authority.get("authority_sha256"), "Bitrix foundation authority"
    )
    directory = abort_root / authority_sha256
    _prepare_bitrix_foundation_directory(directory, require_root=require_root)
    intent_path = directory / "key-retirement-intent.json"
    completion_path = directory / "key-retirement-completion.json"
    if os.path.lexists(completion_path):
        completion = _load_lease_artifact(
            completion_path,
            schema=CAPABILITY_BITRIX_INTERNAL_KEY_RETIREMENT_SCHEMA,
        )
        if (
            completion.get("reason") != "foundation_bootstrap_abort"
            or completion.get("authority_source_sha256") != authority_sha256
            or completion.get("both_pair_members_absent") is not True
            or os.path.lexists(private_path)
            or os.path.lexists(public_path)
        ):
            raise RuntimeError("Bitrix bootstrap-abort retirement drifted")
        return completion
    owner = 0 if require_root else effective_uid()
    group = 0 if require_root else effective_gid()

    def identity(path: Path, mode: int) -> Mapping[str, Any] | None:
        if not os.path.lexists(path):
            return None
        raw, item = _read_exact_file(
            path,
            maximum=16 * 1024,
            uid=owner,
            gid=group,
            mode=mode,
        )
        if mode == 0o400:
            _load_exact_ed25519_private(raw)
        else:
            _load_exact_ed25519_public(raw)
        return {
            "device": item.st_dev,
            "inode": item.st_ino,
            "uid": item.st_uid,
            "gid": item.st_gid,
            "mode": f"{stat.S_IMODE(item.st_mode):04o}",
            "size": item.st_size,
        }

    private_identity = identity(private_path, 0o400)
    public_identity = identity(public_path, 0o444)
    intent = _append_lease_artifact(
        intent_path,
        schema=CAPABILITY_BITRIX_INTERNAL_KEY_RETIREMENT_INTENT_SCHEMA,
        value={
            "operation": "retire_bitrix_bootstrap_pair_intent",
            "reason": "foundation_bootstrap_abort",
            "revision": authority["revision"],
            "full_canary_plan_sha256": authority["full_canary_plan_sha256"],
            "authority_source_sha256": authority_sha256,
            "private_path": str(private_path),
            "private_identity": private_identity,
            "public_path": str(public_path),
            "public_identity": public_identity,
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
        raise RuntimeError("Bitrix bootstrap-abort key retirement is incomplete")
    return _append_lease_artifact(
        completion_path,
        schema=CAPABILITY_BITRIX_INTERNAL_KEY_RETIREMENT_SCHEMA,
        value={
            "operation": "retire_bitrix_bootstrap_pair",
            "reason": "foundation_bootstrap_abort",
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


def bootstrap_bitrix_foundation(
    authority_value: Any,
    *,
    full_plan: FullCanaryPlan | None = None,
    runner: Callable[[Command], subprocess.CompletedProcess[bytes]] = _runner,
    identity_observer: Callable[
        ..., Mapping[str, Any]
    ] = _observe_bitrix_foundation_identity,
    asset_verifier: Callable[
        ..., Mapping[str, Any]
    ] = verify_packaged_operational_assets,
    private_key_path: Path = DEFAULT_BITRIX_RECEIPT_PRIVATE_KEY_PATH,
    public_key_path: Path = DEFAULT_BITRIX_TRUST_PATH,
    writer_public_key_path: Path = DEFAULT_BITRIX_WRITER_PUBLIC_KEY_PATH,
    identity_receipt_path: Path = DEFAULT_BITRIX_IDENTITY_BOOTSTRAP_RECEIPT,
    foundation_root: Path = DEFAULT_BITRIX_FOUNDATION_ROOT,
    key_bootstrap_root: Path = DEFAULT_BITRIX_KEY_BOOTSTRAP_ROOT,
    require_root: bool = True,
    clock: Callable[[], int] | None = None,
) -> Mapping[str, Any]:
    operation_clock = _OperationClock(clock)
    lock = _lifecycle_lock() if require_root else nullcontext()
    with lock:
        try:
            return _bootstrap_bitrix_foundation_locked(
                authority_value,
                full_plan=full_plan,
                runner=runner,
                identity_observer=identity_observer,
                asset_verifier=asset_verifier,
                private_key_path=private_key_path,
                public_key_path=public_key_path,
                writer_public_key_path=writer_public_key_path,
                identity_receipt_path=identity_receipt_path,
                foundation_root=foundation_root,
                key_bootstrap_root=key_bootstrap_root,
                require_root=require_root,
                operation_clock=operation_clock,
            )
        except BaseException as error:
            if not isinstance(authority_value, Mapping):
                raise
            try:
                candidate = validate_bitrix_foundation_authority(
                    authority_value,
                    now_unix=authority_value.get("issued_at_unix"),
                )
                candidate_full = (
                    load_full_canary_plan() if full_plan is None else full_plan
                )
            except BaseException:
                raise
            authority_sha256 = candidate["authority_sha256"]
            if (
                candidate["revision"] != candidate_full.revision
                or candidate["full_canary_plan_sha256"] != candidate_full.sha256
                or candidate["release_artifact_sha256"]
                != candidate_full.release["artifact_sha256"]
            ):
                raise
            key_stage_intent_path = (
                foundation_root / authority_sha256 / "key-stage-intent.json"
            )
            if not os.path.lexists(key_stage_intent_path):
                # No key mutation for this exact authority was admitted, so
                # an unrelated fixed-path pair must never be touched.
                raise
            key_stage_intent = _load_lease_artifact(
                key_stage_intent_path,
                schema=CAPABILITY_BITRIX_FOUNDATION_KEY_STAGE_INTENT_SCHEMA,
            )
            if (
                key_stage_intent.get("authority_sha256") != authority_sha256
                or key_stage_intent.get("revision") != candidate["revision"]
                or key_stage_intent.get("full_canary_plan_sha256")
                != candidate["full_canary_plan_sha256"]
                or key_stage_intent.get("private_path") != str(private_key_path)
                or key_stage_intent.get("public_path") != str(public_key_path)
                or key_stage_intent.get("expires_at_unix")
                != candidate["expires_at_unix"]
            ):
                raise RuntimeError(
                    "Bitrix foundation key-stage intent drifted"
                ) from error
            abort_path = foundation_root / authority_sha256 / "abort.json"
            if os.path.lexists(abort_path):
                raise
            now = operation_clock.sample("Bitrix foundation compensation")
            retirement = _retire_bitrix_bootstrap_pair_after_abort(
                candidate,
                private_path=private_key_path,
                public_path=public_key_path,
                abort_root=foundation_root,
                now_unix=now,
                require_root=require_root,
            )
            abort = _append_lease_artifact(
                abort_path,
                schema=CAPABILITY_BITRIX_FOUNDATION_ABORT_SCHEMA,
                value={
                    "operation": "bootstrap_bitrix_foundation_abort",
                    "state": "retired_before_return",
                    "reason": type(error).__name__,
                    "revision": candidate["revision"],
                    "full_canary_plan_sha256": candidate["full_canary_plan_sha256"],
                    "authority_sha256": authority_sha256,
                    "expires_at_unix": candidate["expires_at_unix"],
                    "key_pair_retirement": copy.deepcopy(dict(retirement)),
                    "key_pair_retirement_sha256": retirement.get("receipt_sha256"),
                    "private_absent": not os.path.lexists(private_key_path),
                    "public_absent": not os.path.lexists(public_key_path),
                    "dormant_identity_retained": True,
                    "aborted_at_unix": now,
                },
            )
            raise RuntimeError(
                f"Bitrix foundation bootstrap failed and was retired; "
                f"abort={abort['receipt_sha256']}"
            ) from error


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
        or value.get("private_path") != str(DEFAULT_BITRIX_RECEIPT_PRIVATE_KEY_PATH)
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
            or receipt.get("full_canary_plan_sha256") != plan.full_canary_plan_sha256
            or receipt.get("receipt_sha256")
            != plan.bitrix_operational_edge_key_bootstrap_receipt_sha256
            or public_key_id != plan.bitrix_operational_edge_receipt_public_key_id
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
            completed.get("key_bootstrap_receipt_sha256") != receipt["receipt_sha256"]
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

    owner = (
        0 if require_root else effective_uid()
    )
    group = (
        0 if require_root else effective_gid()
    )
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
    if identity.get(
        "receipt_sha256"
    ) != plan.bitrix_operational_edge_identity_bootstrap_receipt_sha256 or identity.get(
        "identity"
    ) != {
        "service_user": plan.identities.bitrix_operational_edge_user,
        "service_group": plan.identities.bitrix_operational_edge_group,
        "service_uid": plan.identities.bitrix_operational_edge_uid,
        "service_gid": plan.identities.bitrix_operational_edge_gid,
        "socket_client_group": (plan.identities.bitrix_operational_edge_client_group),
        "socket_client_gid": (plan.identities.bitrix_operational_edge_client_gid),
        "state": "present_exact",
    }:
        raise RuntimeError("bitrix_foundation_bootstrap_unavailable")
    key = load_bitrix_key_bootstrap_receipt(
        public_key_id=plan.bitrix_operational_edge_receipt_public_key_id,
        receipt_sha256=(plan.bitrix_operational_edge_key_bootstrap_receipt_sha256),
        root=key_bootstrap_root,
    )
    if (
        key.get("revision") != plan.revision
        or key.get("full_canary_plan_sha256") != plan.full_canary_plan_sha256
        or key.get("expires_at_unix", -1) <= now
        or not os.path.lexists(DEFAULT_BITRIX_RECEIPT_PRIVATE_KEY_PATH)
        or not os.path.lexists(DEFAULT_BITRIX_TRUST_PATH)
    ):
        raise RuntimeError("bitrix_foundation_bootstrap_unavailable")
    authority_sha256 = _digest(
        key.get("authority_sha256"),
        "Bitrix foundation authority",
    )
    receipt_path = (
        foundation_root
        / authority_sha256
        / plan.bitrix_operational_edge_receipt_public_key_id
        / "foundation.json"
    )
    if not os.path.lexists(receipt_path):
        raise RuntimeError("bitrix_foundation_bootstrap_unavailable")
    foundation = _load_lease_artifact(
        receipt_path,
        schema=CAPABILITY_BITRIX_FOUNDATION_RECEIPT_SCHEMA,
    )
    if not (
        foundation.get("authority_sha256") == authority_sha256
        and foundation.get("revision") == plan.revision
        and foundation.get("full_canary_plan_sha256") == plan.full_canary_plan_sha256
        and foundation.get("full_canary_terminal_receipt")
        == plan.full_canary_terminal_receipt
        and foundation.get("full_canary_terminal_receipt_sha256")
        == plan.full_canary_terminal_receipt_sha256
        and foundation.get("original_full_canary_owner_approval_sha256")
        == plan.original_full_canary_owner_approval_sha256
        and foundation.get("receipt_public_key_id")
        == plan.bitrix_operational_edge_receipt_public_key_id
        and foundation.get("asset_manifest_sha256")
        == plan.bitrix_operational_edge_asset_manifest_sha256
        and foundation.get("identity_bootstrap_receipt_sha256")
        == plan.bitrix_operational_edge_identity_bootstrap_receipt_sha256
        and foundation.get("key_bootstrap_receipt_sha256")
        == plan.bitrix_operational_edge_key_bootstrap_receipt_sha256
        and foundation.get("rendered_unit_sha256")
        == plan.bitrix_operational_edge_rendered_unit_sha256
        and foundation.get("rendered_config_sha256")
        == plan.bitrix_operational_edge_rendered_config_sha256
        and foundation.get("rendered_trust_sha256")
        == plan.bitrix_operational_edge_rendered_trust_sha256
        and foundation.get("read_peer_uids")
        == sorted([plan.identities.mac_ops_uid, full_plan.identities.writer_uid])
        and foundation.get("mutation_peer_uid") == full_plan.identities.writer_uid
        and foundation.get("expires_at_unix", -1) > now
    ):
        raise RuntimeError("bitrix_foundation_bootstrap_unavailable")
    watchdog = foundation.get("expiry_watchdog")
    if (
        not isinstance(watchdog, Mapping)
        or watchdog.get("expires_at_unix") != foundation["expires_at_unix"]
        or not isinstance(watchdog.get("authority_receipt_sha256"), str)
        or _SHA256_RE.fullmatch(watchdog["authority_receipt_sha256"]) is None
    ):
        raise RuntimeError("bitrix_foundation_bootstrap_unavailable")
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
        _sha256_bytes(unit_raw) != plan.bitrix_operational_edge_rendered_unit_sha256
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
    if (
        b'"plan_sha256"' in precursor_bytes
        or b'"capability_plan_sha256"' in precursor_bytes
    ):
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
        "expiry_watchdog_authority_sha256": watchdog["authority_receipt_sha256"],
        "expiry_watchdog_expires_at_unix": watchdog["expires_at_unix"],
        "full_canary_terminal_receipt": copy.deepcopy(
            dict(plan.full_canary_terminal_receipt)
        ),
        "full_canary_terminal_receipt_sha256": (
            plan.full_canary_terminal_receipt_sha256
        ),
        "original_full_canary_owner_approval_sha256": (
            plan.original_full_canary_owner_approval_sha256
        ),
        "ready": True,
    }


def _load_expiry_watchdog_authority(
    watchdog_id: str,
    *,
    state_root: Path = DEFAULT_EXPIRY_WATCHDOG_ROOT,
    systemd_root: Path = Path("/etc/systemd/system"),
) -> Mapping[str, Any]:
    paths = _expiry_watchdog_paths(
        watchdog_id,
        state_root=state_root,
        systemd_root=systemd_root,
    )
    value = _load_lease_artifact(
        Path(paths["authority"]),
        schema=CAPABILITY_EXPIRY_WATCHDOG_AUTHORITY_SCHEMA,
    )
    kind = value.get("kind")
    binding = value.get("credential_binding")
    expires_at = value.get("expires_at_unix")
    authority_source_sha256 = value.get("authority_source_sha256")
    expected_id = _sha256_json({
        "kind": kind,
        "authority_sha256": authority_source_sha256,
        "expires_at_unix": expires_at,
        "credential_binding": binding,
    })[:32]
    if (
        value.get("operation") != "arm_persistent_expiry_watchdog"
        or value.get("watchdog_id") != watchdog_id
        or expected_id != watchdog_id
        or kind not in {"bitrix_foundation", "credential_lease"}
        or _REVISION_RE.fullmatch(str(value.get("revision", ""))) is None
        or type(expires_at) is not int
        or expires_at <= 0
        or not isinstance(value.get("interpreter"), str)
        or not Path(value["interpreter"]).is_absolute()
        or ".." in Path(value["interpreter"]).parts
        or value.get("service_name") != paths["service_name"]
        or value.get("timer_name") != paths["timer_name"]
        or value.get("service_path") != str(paths["service_path"])
        or value.get("timer_path") != str(paths["timer_path"])
        or value.get("timer_wants_path") != str(paths["timer_wants_path"])
        or value.get("timer_wants_target") != f"../{paths['timer_name']}"
        or value.get("receipt_path") != str(paths["authority"])
        or value.get("persistent_across_reboot") is not True
        or value.get("earliest_expiry_not_extended") is not True
        or value.get("cleanup_at_unix") != value.get("expires_at_unix")
    ):
        raise RuntimeError("capability expiry watchdog authority drifted")
    for field in (
        "full_canary_plan_sha256",
        "release_artifact_sha256",
        "authority_source_sha256",
    ):
        _digest(value.get(field), f"capability expiry watchdog {field}")
    if kind == "credential_lease":
        _digest(value.get("plan_sha256"), "capability expiry watchdog plan")
        if binding not in CAPABILITY_CREDENTIAL_BINDINGS:
            raise RuntimeError("capability expiry watchdog credential binding drifted")
    elif value.get("plan_sha256") is not None or binding is not None:
        raise RuntimeError("capability foundation watchdog binding drifted")
    return value


def _bitrix_key_receipts_for_authority(
    authority_sha256: str,
    *,
    root: Path = DEFAULT_BITRIX_KEY_BOOTSTRAP_ROOT,
) -> list[Mapping[str, Any]]:
    authority_sha256 = _digest(authority_sha256, "Bitrix foundation authority")
    index_path = root / ".authority-index" / authority_sha256 / "key-bootstrap.json"
    if not os.path.lexists(index_path):
        # A crash after staging the fixed key pair but before publishing the
        # authority index is handled by the digest-free orphan retirement
        # path.  Never scan or globally cap valid append-only key history.
        return []
    index = _load_lease_artifact(
        index_path,
        schema=CAPABILITY_BITRIX_KEY_AUTHORITY_INDEX_SCHEMA,
    )
    public_key_id = _digest(
        index.get("public_key_id"),
        "Bitrix key authority index public key",
    )
    receipt_path = root / public_key_id / "bootstrap.json"
    if (
        index.get("operation") != "index_bitrix_key_bootstrap_by_foundation_authority"
        or index.get("authority_sha256") != authority_sha256
        or index.get("key_bootstrap_receipt_path") != str(receipt_path)
        or index.get("append_only_history") is not True
    ):
        raise RuntimeError("Bitrix key authority index drifted")
    receipt = _load_lease_artifact(
        receipt_path,
        schema=CAPABILITY_BITRIX_KEY_BOOTSTRAP_SCHEMA,
    )
    if (
        receipt.get("authority_sha256") != authority_sha256
        or receipt.get("public_key_id") != public_key_id
        or receipt.get("receipt_sha256") != index.get("key_bootstrap_receipt_sha256")
        or receipt.get("revision") != index.get("revision")
        or receipt.get("full_canary_plan_sha256")
        != index.get("full_canary_plan_sha256")
        or receipt.get("expires_at_unix") != index.get("expires_at_unix")
    ):
        raise RuntimeError("Bitrix key authority index binding drifted")
    return [receipt]


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

    if type(now_unix) is not int or now_unix < authority.get(
        "expires_at_unix", now_unix + 1
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

    owner = (
        0 if require_root else effective_uid()
    )
    group = (
        0 if require_root else effective_gid()
    )
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
                "full_canary_plan_sha256": authority["full_canary_plan_sha256"],
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


def _disarm_capability_expiry_watchdog(
    watchdog_id: str,
    *,
    runner: Callable[[Command], subprocess.CompletedProcess[bytes]] = _runner,
    state_root: Path = DEFAULT_EXPIRY_WATCHDOG_ROOT,
    systemd_root: Path = Path("/etc/systemd/system"),
    require_root: bool = True,
) -> Mapping[str, Any]:
    """Durably disable one exact watchdog generation.

    The intent is append-only and precedes every systemd mutation.  A process
    death after stopping/removing a timer is therefore recoverable, and the
    same deterministic watchdog generation can never be armed again after a
    normal lifecycle stop.
    """

    if require_root:
        _require_root_linux()
    authority = _load_expiry_watchdog_authority(
        watchdog_id,
        state_root=state_root,
        systemd_root=systemd_root,
    )
    paths = _expiry_watchdog_paths(
        watchdog_id,
        state_root=state_root,
        systemd_root=systemd_root,
    )
    intent_path = Path(paths["disarm_intent"])
    completion_path = Path(paths["disarm_completion"])
    if os.path.lexists(intent_path):
        intent = _load_lease_artifact(
            intent_path,
            schema=CAPABILITY_EXPIRY_WATCHDOG_DISARM_INTENT_SCHEMA,
        )
    else:
        intent = _append_lease_artifact(
            intent_path,
            schema=CAPABILITY_EXPIRY_WATCHDOG_DISARM_INTENT_SCHEMA,
            value={
                "operation": "normal_lifecycle_disarm_intent",
                "watchdog_id": watchdog_id,
                "watchdog_authority_sha256": authority["receipt_sha256"],
                "service_name": paths["service_name"],
                "timer_name": paths["timer_name"],
                "service_path": str(paths["service_path"]),
                "timer_path": str(paths["timer_path"]),
                "timer_wants_path": str(paths["timer_wants_path"]),
                "requested_at_unix": int(time.time()),
            },
        )
    if (
        intent.get("operation") != "normal_lifecycle_disarm_intent"
        or intent.get("watchdog_id") != watchdog_id
        or intent.get("watchdog_authority_sha256") != authority["receipt_sha256"]
        or intent.get("service_name") != paths["service_name"]
        or intent.get("timer_name") != paths["timer_name"]
        or intent.get("service_path") != str(paths["service_path"])
        or intent.get("timer_path") != str(paths["timer_path"])
        or intent.get("timer_wants_path") != str(paths["timer_wants_path"])
        or type(intent.get("requested_at_unix")) is not int
    ):
        raise RuntimeError("capability expiry watchdog disarm intent drifted")
    if os.path.lexists(completion_path):
        completion = _load_lease_artifact(
            completion_path,
            schema=CAPABILITY_EXPIRY_WATCHDOG_DISARM_COMPLETION_SCHEMA,
        )
        if (
            completion.get("operation") != "normal_lifecycle_disarm"
            or completion.get("watchdog_id") != watchdog_id
            or completion.get("watchdog_authority_sha256")
            != authority["receipt_sha256"]
            or completion.get("disarm_intent_path") != str(intent_path)
            or completion.get("disarm_intent_sha256") != intent["receipt_sha256"]
            or completion.get("timer_name") != paths["timer_name"]
            or completion.get("receipt_path") != str(completion_path)
            or completion.get("timer_disabled") is not True
            or completion.get("timer_wants_absent") is not True
            or completion.get("service_absent") is not True
            or completion.get("timer_absent") is not True
            or completion.get("ok") is not True
            or any(
                os.path.lexists(Path(paths[field]))
                for field in ("service_path", "timer_path", "timer_wants_path")
            )
        ):
            raise RuntimeError("capability expiry watchdog disarm completion drifted")
        return completion
    changed = False
    if any(
        os.path.lexists(Path(paths[field]))
        for field in ("service_path", "timer_path", "timer_wants_path")
    ):
        _run_checked(
            Command((SYSTEMCTL, "stop", str(paths["timer_name"]))),
            runner=runner,
            label=f"stop {paths['timer_name']}",
        )
        changed = True
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
        changed = True
    removals: dict[str, bool] = {"wants": not os.path.lexists(wants_path)}
    for label, path, expected in (
        ("service", Path(paths["service_path"]), expected_service),
        ("timer", Path(paths["timer_path"]), expected_timer),
    ):
        if os.path.lexists(path):
            raw, _ = _read_exact_file(
                path,
                maximum=64 * 1024,
                uid=0 if require_root else effective_uid(),
                gid=0 if require_root else effective_gid(),
                mode=0o644,
            )
            if raw != expected:
                raise RuntimeError("capability expiry watchdog unit substitution")
            os.unlink(path)
            _fsync_directory(path.parent)
            changed = True
        removals[label] = not os.path.lexists(path)
    if not all(removals.values()):
        raise RuntimeError("capability expiry watchdog unit retirement failed")
    if changed:
        _run_checked(
            Command((SYSTEMCTL, "daemon-reload")),
            runner=runner,
            label="reload retired capability expiry watchdog",
        )
    return _append_lease_artifact(
        completion_path,
        schema=CAPABILITY_EXPIRY_WATCHDOG_DISARM_COMPLETION_SCHEMA,
        value={
            "operation": "normal_lifecycle_disarm",
            "watchdog_id": watchdog_id,
            "watchdog_authority_sha256": authority["receipt_sha256"],
            "disarm_intent_path": intent["receipt_path"],
            "disarm_intent_sha256": intent["receipt_sha256"],
            "timer_name": paths["timer_name"],
            "timer_disabled": True,
            "timer_wants_absent": removals["wants"],
            "service_absent": removals["service"],
            "timer_absent": removals["timer"],
            "completed_at_unix": int(time.time()),
            "ok": True,
        },
    )


def _expiry_watchdog_retirement_summary(
    retired: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    values = [copy.deepcopy(dict(item)) for item in retired]
    authorities = sorted(
        _digest(
            item.get("watchdog_authority_sha256"),
            "retired expiry watchdog authority",
        )
        for item in values
    )
    if len(set(authorities)) != len(authorities):
        raise RuntimeError("retired expiry watchdog authorities are not unique")
    return {
        "watchdog_count": len(values),
        "authority_receipt_sha256s": authorities,
        "authority_set_sha256": _sha256_json({
            "authority_receipt_sha256s": authorities,
        }),
        "retired": values,
        "all_timers_disabled": all(
            item.get("timer_disabled") is True for item in values
        ),
        "all_unit_files_absent": all(
            item.get("timer_wants_absent") is True
            and item.get("service_absent") is True
            and item.get("timer_absent") is True
            for item in values
        ),
    }


def _expected_cleanup_expiry_watchdog_authorities(
    plan: CapabilityCanaryPlan,
    *,
    approval_retirement: Mapping[str, Any],
    lease_retirements: Mapping[str, Mapping[str, Any]],
    state_root: Path = DEFAULT_EXPIRY_WATCHDOG_ROOT,
    systemd_root: Path = Path("/etc/systemd/system"),
) -> tuple[str, ...]:
    """Recover the seven exact watchdog authorities from sealed run truth."""

    if set(lease_retirements) != set(CAPABILITY_CREDENTIAL_BINDINGS):
        raise RuntimeError("cleanup lease retirements are not exact")
    approved_installs = approval_retirement.get(
        "lease_install_receipt_sha256_by_binding"
    )
    if not isinstance(approved_installs, Mapping) or set(approved_installs) != set(
        CAPABILITY_CREDENTIAL_BINDINGS
    ):
        raise RuntimeError("cleanup approval lacks exact lease watchdog bindings")
    try:
        watchdog_ids = sorted(os.listdir(state_root))
    except FileNotFoundError as exc:
        raise RuntimeError("cleanup expiry watchdog inventory is absent") from exc
    if (
        not watchdog_ids
        or len(watchdog_ids) > _MAX_EXPIRY_WATCHDOGS
        or any(_LEASE_ID_RE.fullmatch(value) is None for value in watchdog_ids)
    ):
        raise RuntimeError("cleanup expiry watchdog inventory is invalid")
    authority_by_sha256: dict[str, Mapping[str, Any]] = {}
    for watchdog_id in watchdog_ids:
        authority = _load_expiry_watchdog_authority(
            watchdog_id,
            state_root=state_root,
            systemd_root=systemd_root,
        )
        authority_sha256 = authority["receipt_sha256"]
        if authority_sha256 in authority_by_sha256:
            raise RuntimeError("cleanup expiry watchdog authority is duplicated")
        authority_by_sha256[authority_sha256] = authority
    bitrix_authority_sha256 = _digest(
        approval_retirement.get("bitrix_expiry_watchdog_authority_sha256"),
        "cleanup Bitrix expiry watchdog authority",
    )
    bitrix_authority = authority_by_sha256.get(bitrix_authority_sha256)
    if (
        bitrix_authority is None
        or bitrix_authority.get("kind") != "bitrix_foundation"
        or bitrix_authority.get("revision") != plan.revision
        or bitrix_authority.get("full_canary_plan_sha256")
        != plan.full_canary_plan_sha256
        or bitrix_authority.get("release_artifact_sha256")
        != plan.release_artifact_sha256
        or bitrix_authority.get("interpreter") != str(plan.interpreter)
        or bitrix_authority.get("plan_sha256") is not None
        or bitrix_authority.get("credential_binding") is not None
    ):
        raise RuntimeError("cleanup Bitrix watchdog authority binding drifted")
    authorities = [bitrix_authority_sha256]
    for binding in CAPABILITY_CREDENTIAL_BINDINGS:
        retirement = lease_retirements[binding]
        install_path_raw = retirement.get("install_receipt_path")
        install_sha256 = _digest(
            retirement.get("install_receipt_sha256"),
            f"cleanup {binding} install receipt",
        )
        if (
            not isinstance(install_path_raw, str)
            or not Path(install_path_raw).is_absolute()
            or ".." in Path(install_path_raw).parts
            or approved_installs.get(binding) != install_sha256
        ):
            raise RuntimeError("cleanup lease install binding drifted")
        install = _load_lease_artifact(
            Path(install_path_raw),
            schema=CAPABILITY_LEASE_RECEIPT_SCHEMA,
        )
        watchdog = install.get("expiry_watchdog")
        if (
            install.get("receipt_sha256") != install_sha256
            or install.get("plan_sha256") != plan.sha256
            or install.get("full_canary_plan_sha256")
            != plan.full_canary_plan_sha256
            or install.get("credential_binding") != binding
            or not isinstance(watchdog, Mapping)
            or watchdog.get("armed_before_secret_commit") is not True
            or watchdog.get("persistent_across_reboot") is not True
        ):
            raise RuntimeError("cleanup credential watchdog binding drifted")
        watchdog_id = watchdog.get("watchdog_id")
        authority_sha256 = _digest(
            watchdog.get("authority_receipt_sha256"),
            f"cleanup {binding} expiry watchdog authority",
        )
        if not isinstance(watchdog_id, str):
            raise RuntimeError("cleanup credential watchdog ID is invalid")
        expected_paths = _expiry_watchdog_paths(
            watchdog_id,
            state_root=state_root,
            systemd_root=systemd_root,
        )
        authority = authority_by_sha256.get(authority_sha256)
        if (
            authority is None
            or authority.get("watchdog_id") != watchdog_id
            or authority.get("kind") != "credential_lease"
            or authority.get("revision") != plan.revision
            or authority.get("plan_sha256") != plan.sha256
            or authority.get("full_canary_plan_sha256")
            != plan.full_canary_plan_sha256
            or authority.get("release_artifact_sha256")
            != plan.release_artifact_sha256
            or authority.get("credential_binding") != binding
            or authority.get("interpreter") != str(plan.interpreter)
            or authority.get("expires_at_unix") != install.get("expires_at_unix")
            or watchdog.get("authority_receipt_path")
            != str(expected_paths["authority"])
            or watchdog.get("timer_name") != expected_paths["timer_name"]
            or watchdog.get("cleanup_at_unix") != authority.get("cleanup_at_unix")
            or watchdog.get("expires_at_unix") != authority.get("expires_at_unix")
        ):
            raise RuntimeError("cleanup credential watchdog authority drifted")
        authorities.append(authority_sha256)
    if len(authorities) != 1 + len(CAPABILITY_CREDENTIAL_BINDINGS) or len(
        set(authorities)
    ) != len(authorities):
        raise RuntimeError("cleanup expiry watchdog authority set is not exact")
    return tuple(sorted(authorities))


def disarm_all_capability_expiry_watchdogs(
    *,
    runner: Callable[[Command], subprocess.CompletedProcess[bytes]] = _runner,
    state_root: Path = DEFAULT_EXPIRY_WATCHDOG_ROOT,
    systemd_root: Path = Path("/etc/systemd/system"),
    require_root: bool = True,
    expected_authority_receipt_sha256s: Sequence[str] | None = None,
) -> Mapping[str, Any]:
    """Durably disarm all, or one explicitly bound set, of watchdogs."""

    if require_root:
        _require_root_linux()
    try:
        names = sorted(os.listdir(state_root))
    except FileNotFoundError:
        names = []
    if len(names) > _MAX_EXPIRY_WATCHDOGS or any(
        _LEASE_ID_RE.fullmatch(name) is None for name in names
    ):
        raise RuntimeError("capability expiry watchdog inventory is invalid")
    authority_by_id: dict[str, str] = {}
    for watchdog_id in names:
        authority = _load_expiry_watchdog_authority(
            watchdog_id,
            state_root=state_root,
            systemd_root=systemd_root,
        )
        authority_sha256 = _digest(
            authority.get("receipt_sha256"),
            "capability expiry watchdog authority receipt",
        )
        if authority_sha256 in authority_by_id.values():
            raise RuntimeError("capability expiry watchdog authority is duplicated")
        authority_by_id[watchdog_id] = authority_sha256
    selected_names = names
    if expected_authority_receipt_sha256s is not None:
        expected = {
            _digest(value, "expected expiry watchdog authority")
            for value in expected_authority_receipt_sha256s
        }
        if not expected or len(expected) != len(expected_authority_receipt_sha256s):
            raise ValueError(
                "expected expiry watchdog authorities are empty or not unique"
            )
        observed = set(authority_by_id.values())
        if not expected <= observed:
            raise RuntimeError("expected expiry watchdog authority is absent")
        selected_names = [
            watchdog_id
            for watchdog_id in names
            if authority_by_id[watchdog_id] in expected
        ]
    retired = [
        _disarm_capability_expiry_watchdog(
            watchdog_id,
            runner=runner,
            state_root=state_root,
            systemd_root=systemd_root,
            require_root=require_root,
        )
        for watchdog_id in selected_names
    ]
    return _expiry_watchdog_retirement_summary(retired)


def _reconcile_expired_active_run_artifacts(
    *,
    retired_at_unix_ms: int,
) -> Mapping[str, Any]:
    """Delegate fixed-path discovery/retirement to the producer boundary."""

    from gateway.canonical_capability_canary_producers import (
        recover_and_retire_active_api_admission,
    )

    return recover_and_retire_active_api_admission(
        retired_at_unix_ms=retired_at_unix_ms,
    )


def _validate_expiry_active_run_retirement(
    value: Any,
    *,
    authority: Mapping[str, Any],
    require_current_absence: bool = True,
) -> Mapping[str, Any]:
    """Require durable exact absence before watchdog terminal truth."""

    if not isinstance(value, Mapping):
        raise RuntimeError("capability expiry active-run retirement is invalid")
    from gateway.canonical_capability_canary_producers import (
        ACTIVE_API_ADMISSION_RETIREMENT_SCHEMA,
        DEFAULT_OWNER_GRANT_PATH,
        DEFAULT_PROBE_CATALOG_PATH,
        DEFAULT_READINESS_PATH,
        validate_active_api_admission_retirement,
    )

    if (
        ACTIVE_API_ADMISSION_RETIREMENT_SCHEMA
        != CAPABILITY_EXPIRY_ACTIVE_RUN_RETIREMENT_SCHEMA
    ):
        raise RuntimeError("capability expiry active-run schema drifted")
    raw = validate_active_api_admission_retirement(value)
    expected_plan_sha256 = authority.get("plan_sha256")
    if (
        raw.get("schema") != CAPABILITY_EXPIRY_ACTIVE_RUN_RETIREMENT_SCHEMA
        or raw.get("catalog_absent") is not True
        or raw.get("owner_grant_absent") is not True
        or raw.get("producer_activation_absent") is not True
    ):
        raise RuntimeError("capability expiry active-run retirement is invalid")
    if raw.get("outcome") in {
        "retired_active_run",
        "retired_partial_install",
        "reconciled_published_run_without_admission",
    }:
        if (
            raw.get("full_canary_plan_sha256")
            != authority.get("full_canary_plan_sha256")
            or (
                expected_plan_sha256 is not None
                and raw.get("capability_plan_sha256") != expected_plan_sha256
            )
        ):
            raise RuntimeError("capability expiry active-run binding drifted")
    elif raw.get("outcome") != "confirmed_no_active_run":
        raise RuntimeError("capability expiry active-run outcome is invalid")

    if require_current_absence and any(
        os.path.lexists(path)
        for path in (
            DEFAULT_PROBE_CATALOG_PATH,
            DEFAULT_OWNER_GRANT_PATH,
            DEFAULT_READINESS_PATH,
        )
    ):
        raise RuntimeError("capability expiry fixed run artifacts remain live")
    return raw


def _append_expiry_reconciliation(
    *,
    root: Path,
    authority: Mapping[str, Any],
    completion: Mapping[str, Any],
    services: Mapping[str, Mapping[str, Any]],
    external: Mapping[str, Any],
    bitrix_pair: Mapping[str, Any],
    approval_retirement: Mapping[str, Any],
    credential_absence: Mapping[str, bool],
    approval_absent: bool,
    pair_absent: bool,
    active_run_retirement: Mapping[str, Any],
    disarm: Mapping[str, Any],
    remediation_performed: bool,
    observed_at_unix: int,
) -> Mapping[str, Any]:
    _prepare_journal_directory(root)
    inventory = sorted(name for name in os.listdir(root) if name != ".lock")
    canonical = [name for name in inventory if re.fullmatch(r"[0-9]{16}\.json", name)]
    temporaries = [
        name for name in inventory if re.fullmatch(r"\.[0-9]{16}\.json\.tmp", name)
    ]
    if len(canonical) + len(temporaries) != len(inventory) or len(temporaries) > 1:
        raise RuntimeError("capability expiry reconciliation inventory is invalid")
    previous: Mapping[str, Any] | None = None
    if canonical:
        expected = [f"{index:016d}.json" for index in range(1, len(canonical) + 1)]
        if canonical != expected:
            raise RuntimeError(
                "capability expiry reconciliation chain is not contiguous"
            )
        previous = _load_lease_artifact(
            root / canonical[-1],
            schema=CAPABILITY_EXPIRY_RECONCILIATION_SCHEMA,
        )
    if temporaries:
        temporary_sequence = int(temporaries[0][1:17])
        allowed_sequences = {len(canonical), len(canonical) + 1}
        if temporary_sequence not in allowed_sequences or temporary_sequence < 1:
            raise RuntimeError(
                "capability expiry reconciliation temporary is out of sequence"
            )
        _reconcile_lease_artifact_temporary(
            root / f"{temporary_sequence:016d}.json",
            schema=CAPABILITY_EXPIRY_RECONCILIATION_SCHEMA,
        )
        canonical = sorted(
            name for name in os.listdir(root) if re.fullmatch(r"[0-9]{16}\.json", name)
        )
        expected = [f"{index:016d}.json" for index in range(1, len(canonical) + 1)]
        if canonical != expected:
            raise RuntimeError(
                "capability expiry reconciliation chain is not contiguous"
            )
        previous = _load_lease_artifact(
            root / canonical[-1],
            schema=CAPABILITY_EXPIRY_RECONCILIATION_SCHEMA,
        )
    if len(canonical) >= _MAX_EXPIRY_RECONCILIATIONS:
        raise RuntimeError("capability expiry reconciliation inventory is full")
    sequence = len(canonical) + 1
    terminal_states: dict[str, Mapping[str, Any]] = {}
    slots = external.get("slots") if isinstance(external, Mapping) else None
    for binding in CAPABILITY_CREDENTIAL_BINDINGS:
        slot = slots.get(binding, {}) if isinstance(slots, Mapping) else {}
        state = slot.get("state", "never_installed_absent")
        terminal_digest = (
            slot.get("retirement_receipt_sha256")
            or slot.get("install_abort_receipt_sha256")
            or slot.get("observation_sha256")
        )
        terminal_states[binding] = {
            "state": state,
            "terminal_receipt_sha256": terminal_digest,
            "target_absent": credential_absence[binding],
        }
    return _append_lease_artifact(
        root / f"{sequence:016d}.json",
        schema=CAPABILITY_EXPIRY_RECONCILIATION_SCHEMA,
        value={
            "operation": "reconcile_persistent_expiry_cleanup",
            "sequence": sequence,
            "previous_reconciliation_sha256": (
                previous["receipt_sha256"] if previous is not None else None
            ),
            "watchdog_id": authority["watchdog_id"],
            "watchdog_authority_sha256": authority["receipt_sha256"],
            "cleanup_completion_sha256": completion["receipt_sha256"],
            "revision": authority["revision"],
            "full_canary_plan_sha256": authority["full_canary_plan_sha256"],
            "plan_sha256": authority["plan_sha256"],
            "credential_binding": authority["credential_binding"],
            "service_state_sha256": _sha256_json(services),
            "all_services_stopped": all(
                _service_stopped(state) for state in services.values()
            ),
            "lease_terminal_state_by_binding": terminal_states,
            "credential_absence": dict(credential_absence),
            "all_six_credentials_absent_readback": all(credential_absence.values()),
            "bitrix_key_pair_cleanup_sha256": (
                bitrix_pair.get("receipt_sha256")
                if isinstance(bitrix_pair, Mapping)
                else None
            ),
            "approval_retirement": copy.deepcopy(dict(approval_retirement)),
            "approval_retirement_sha256": _sha256_json(approval_retirement),
            "approval_absent": approval_absent,
            "bitrix_pair_absent": pair_absent,
            "active_run_retirement": copy.deepcopy(dict(active_run_retirement)),
            "active_run_retirement_sha256": active_run_retirement[
                "receipt_sha256"
            ],
            "catalog_absent": active_run_retirement["catalog_absent"],
            "owner_grant_absent": active_run_retirement["owner_grant_absent"],
            "producer_activation_absent": active_run_retirement[
                "producer_activation_absent"
            ],
            "remediation_performed": remediation_performed,
            "watchdog_disarm": copy.deepcopy(dict(disarm)),
            "final_state": "reconciled_stopped_and_retired",
            "observed_at_unix": observed_at_unix,
        },
    )


def _run_capability_expiry_cleanup_locked(
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
    active_run_retirer: Callable[..., Mapping[str, Any]] | None = None,
    require_root: bool = True,
    operation_clock: _OperationClock,
) -> Mapping[str, Any]:
    """Execute persistent stop→six-lease→Bitrix-pair cleanup after expiry."""

    fixed_credential_paths = {
        binding: Path(value["target_path"])
        for binding, value in _credential_bindings_mapping().items()
    }
    observed_credential_paths = (
        fixed_credential_paths if credential_paths is None else dict(credential_paths)
    )
    if set(observed_credential_paths) != set(CAPABILITY_CREDENTIAL_BINDINGS) or any(
        not isinstance(path, Path) or not path.is_absolute() or ".." in path.parts
        for path in observed_credential_paths.values()
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
            or active_run_retirer is not None
        ):
            raise ValueError("capability expiry production boundaries are fixed")
    now = (
        operation_clock.sample("capability expiry cleanup admission")
        if now_unix is None
        else now_unix
    )
    authority = _load_expiry_watchdog_authority(
        watchdog_id,
        state_root=state_root,
        systemd_root=systemd_root,
    )
    if type(now) is not int or now < authority["cleanup_at_unix"]:
        raise PermissionError("capability expiry watchdog fired before its bound time")
    paths = _expiry_watchdog_paths(
        watchdog_id,
        state_root=state_root,
        systemd_root=systemd_root,
    )
    completion_path = Path(paths["completion"])
    prior_completion: Mapping[str, Any] | None = None
    if os.path.lexists(completion_path):
        prior_completion = _load_lease_artifact(
            completion_path,
            schema=CAPABILITY_EXPIRY_WATCHDOG_COMPLETION_SCHEMA,
        )
        if (
            prior_completion.get("watchdog_authority_sha256")
            != authority["receipt_sha256"]
            or prior_completion.get("ok") is not True
            or prior_completion.get("all_services_stopped") is not True
            or prior_completion.get("all_six_credentials_absent_readback") is not True
            or prior_completion.get("bitrix_pair_absent") is not True
            or prior_completion.get("approval_absent") is not True
            or not isinstance(prior_completion.get("approval_retirement"), Mapping)
            or prior_completion.get("approval_retirement_sha256")
            != _sha256_json(prior_completion["approval_retirement"])
            or not isinstance(
                prior_completion.get("active_run_retirement"), Mapping
            )
            or prior_completion.get("active_run_retirement_sha256")
            != prior_completion["active_run_retirement"].get("receipt_sha256")
            or prior_completion.get("catalog_absent") is not True
            or prior_completion.get("owner_grant_absent") is not True
            or prior_completion.get("producer_activation_absent") is not True
        ):
            raise RuntimeError("capability expiry watchdog completion drifted")

    from gateway.canonical_capability_canary_producers import (
        DEFAULT_OWNER_GRANT_PATH,
        DEFAULT_PROBE_CATALOG_PATH,
        DEFAULT_READINESS_PATH,
    )

    dirty_before_reconciliation = (
        any(os.path.lexists(path) for path in observed_credential_paths.values())
        or os.path.lexists(private_key_path)
        or os.path.lexists(public_key_path)
        or os.path.lexists(DEFAULT_APPROVAL_PATH)
        or os.path.lexists(DEFAULT_PROBE_CATALOG_PATH)
        or os.path.lexists(DEFAULT_OWNER_GRANT_PATH)
        or os.path.lexists(DEFAULT_READINESS_PATH)
    )

    stopped, stop_errors = _attempt_capability_stop_order(
        lambda unit: _run_checked(
            Command((SYSTEMCTL, "stop", unit), timeout_seconds=120),
            runner=runner,
            label=f"expiry stop {unit}",
        )
    )
    services: Mapping[str, Mapping[str, Any]] = {}
    service_error_sha256s = [
        _sha256_bytes(f"{type(exc).__name__}:{exc}".encode("utf-8", errors="replace"))
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
    active_run_retirement: Mapping[str, Any] = {}
    active_run_error: BaseException | None = None
    if all_services_stopped:
        try:
            retire_active_run = (
                _reconcile_expired_active_run_artifacts
                if active_run_retirer is None
                else active_run_retirer
            )
            active_run_retirement = _validate_expiry_active_run_retirement(
                retire_active_run(
                    retired_at_unix_ms=now * 1000,
                ),
                authority=authority,
            )
        except BaseException as exc:
            active_run_error = exc
    else:
        active_run_error = RuntimeError(
            "services remain live before active-run retirement"
        )
    plan: CapabilityCanaryPlan | None = None
    full: FullCanaryPlan | None = None
    external: Mapping[str, Any] = {}
    bitrix_pair: Mapping[str, Any] = {}
    approval_retirement: Mapping[str, Any] = {
        "path": str(DEFAULT_APPROVAL_PATH),
        "removed": False,
        "absent": not os.path.lexists(DEFAULT_APPROVAL_PATH),
    }
    errors: dict[str, str] = {}
    if active_run_error is not None:
        errors["active_run_retirement"] = _sha256_bytes(
            f"{type(active_run_error).__name__}:{active_run_error}".encode(
                "utf-8", errors="replace"
            )
        )
    if os.path.lexists(DEFAULT_PLAN_PATH):
        try:
            plan = load_capability_plan()
            full = load_full_canary_plan()
            validate_plan_against_full(plan, full)
            if (
                authority.get("revision") != plan.revision
                or authority.get("full_canary_plan_sha256")
                != plan.full_canary_plan_sha256
                or authority.get("release_artifact_sha256")
                != plan.release_artifact_sha256
                or authority.get("interpreter") != str(plan.interpreter)
                or (
                    authority.get("kind") == "credential_lease"
                    and authority.get("plan_sha256") != plan.sha256
                )
                or (
                    authority.get("kind") == "bitrix_foundation"
                    and authority.get("plan_sha256") is not None
                )
            ):
                raise PermissionError("watchdog plan binding drifted")
            if not all_services_stopped:
                raise RuntimeError("services remain live at watchdog expiry")
            approval_retirement = _remove_installed_capability_approval(plan, full)
            if approval_retirement.get("absent") is not True or os.path.lexists(
                DEFAULT_APPROVAL_PATH
            ):
                raise RuntimeError("capability approval retirement is incomplete")
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
                now_unix=now,
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
        if (
            authority.get("kind") != "bitrix_foundation"
            or authority.get("plan_sha256") is not None
        ):
            errors["preplan_watchdog_authority"] = _sha256_bytes(
                b"credential_watchdog_requires_published_plan"
            )
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
        if os.path.lexists(DEFAULT_APPROVAL_PATH):
            errors["preplan_owner_approval"] = _sha256_bytes(
                b"owner_approval_exists_before_plan_publication"
            )
        try:
            if not all_services_stopped:
                raise RuntimeError("services remain live at foundation watchdog expiry")
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
    pair_absent = not os.path.lexists(private_key_path) and not os.path.lexists(
        public_key_path
    )
    approval_absent = not os.path.lexists(DEFAULT_APPROVAL_PATH)
    ok = (
        all_services_stopped
        and all_external_absent
        and pair_absent
        and approval_absent
        and bool(active_run_retirement)
        and active_run_retirement.get("catalog_absent") is True
        and active_run_retirement.get("owner_grant_absent") is True
        and active_run_retirement.get("producer_activation_absent") is True
        and not errors
    )
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
    completion = prior_completion
    if completion is None:
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
                "approval_retirement": copy.deepcopy(dict(approval_retirement)),
                "approval_retirement_sha256": _sha256_json(approval_retirement),
                "active_run_retirement": copy.deepcopy(
                    dict(active_run_retirement)
                ),
                "active_run_retirement_sha256": active_run_retirement[
                    "receipt_sha256"
                ],
                "catalog_absent": active_run_retirement["catalog_absent"],
                "owner_grant_absent": active_run_retirement[
                    "owner_grant_absent"
                ],
                "producer_activation_absent": active_run_retirement[
                    "producer_activation_absent"
                ],
                "error_sha256s": errors,
                "all_six_credentials_absent_readback": all_external_absent,
                "bitrix_private_absent": not os.path.lexists(private_key_path),
                "bitrix_public_absent": not os.path.lexists(public_key_path),
                "bitrix_pair_absent": pair_absent,
                "approval_absent": approval_absent,
                "completed_at_unix": now,
                "ok": ok,
            },
        )
    disarm = _expiry_watchdog_retirement_summary((
        _disarm_capability_expiry_watchdog(
            watchdog_id,
            runner=runner,
            state_root=state_root,
            systemd_root=systemd_root,
            require_root=require_root,
        ),
    ))
    final_now = (
        operation_clock.sample("capability expiry reconciliation")
        if now_unix is None
        else now_unix
    )
    final_services = service_observer(runner=runner)
    credential_absence = {
        binding: not os.path.lexists(path)
        for binding, path in observed_credential_paths.items()
    }
    final_pair_absent = not os.path.lexists(private_key_path) and not os.path.lexists(
        public_key_path
    )
    final_approval_absent = not os.path.lexists(DEFAULT_APPROVAL_PATH)
    final_active_run_retirement = _validate_expiry_active_run_retirement(
        active_run_retirement,
        authority=authority,
    )
    if (
        set(final_services) != set(CAPABILITY_STOP_ORDER)
        or not all(_service_stopped(state) for state in final_services.values())
        or not all(credential_absence.values())
        or not final_pair_absent
        or not final_approval_absent
        or disarm.get("all_timers_disabled") is not True
        or disarm.get("all_unit_files_absent") is not True
    ):
        raise RuntimeError("capability expiry reconciliation is incomplete")
    _append_expiry_reconciliation(
        root=Path(paths["reconciliations"]),
        authority=authority,
        completion=completion,
        services=final_services,
        external=external,
        bitrix_pair=bitrix_pair,
        approval_retirement=approval_retirement,
        credential_absence=credential_absence,
        approval_absent=final_approval_absent,
        pair_absent=final_pair_absent,
        active_run_retirement=final_active_run_retirement,
        disarm=disarm,
        remediation_performed=(
            prior_completion is not None and dirty_before_reconciliation
        ),
        observed_at_unix=final_now,
    )
    return completion


def run_capability_expiry_cleanup(
    watchdog_id: str,
    *,
    runner: Callable[[Command], subprocess.CompletedProcess[bytes]] = _runner,
    now_unix: int | None = None,
    clock: Callable[[], int] | None = None,
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
    active_run_retirer: Callable[..., Mapping[str, Any]] | None = None,
    require_root: bool = True,
) -> Mapping[str, Any]:
    if now_unix is not None and clock is not None:
        raise ValueError("capability expiry cleanup clock is ambiguous")
    operation_clock = _OperationClock(
        (lambda: now_unix) if now_unix is not None else clock
    )
    lock = _lifecycle_lock() if require_root else nullcontext()
    with lock:
        return _run_capability_expiry_cleanup_locked(
            watchdog_id,
            runner=runner,
            now_unix=now_unix,
            state_root=state_root,
            systemd_root=systemd_root,
            credential_paths=credential_paths,
            private_key_path=private_key_path,
            public_key_path=public_key_path,
            key_bootstrap_root=key_bootstrap_root,
            key_retirement_root=key_retirement_root,
            service_observer=service_observer,
            active_run_retirer=active_run_retirer,
            require_root=require_root,
            operation_clock=operation_clock,
        )


_FOUNDATION_AUTHORING_REQUEST_FIELDS = frozenset({
    "schema",
    "revision",
    "full_canary_terminal_receipt",
    "full_canary_terminal_receipt_sha256",
    "secret_material_recorded",
    "secret_digest_recorded",
    "semantic_content_recorded",
    "request_sha256",
})
_FOUNDATION_AUTHORING_CONTEXT_FIELDS = frozenset({
    "schema",
    "revision",
    "staged_plan_path",
    "staged_plan_file_sha256",
    "staged_plan_identity",
    "full_canary_plan_sha256",
    "release_artifact_sha256",
    "full_canary_terminal_receipt",
    "full_canary_terminal_receipt_sha256",
    "original_full_canary_owner_approval_sha256",
    "identities",
    "identity_observation",
    "asset_manifest_sha256",
    "mutation_performed",
    "secret_material_recorded",
    "secret_digest_recorded",
    "semantic_content_recorded",
    "receipt_sha256",
})
_FOUNDATION_AUTHORING_IDENTITY_FIELDS = frozenset({
    "service_uid",
    "service_gid",
    "socket_client_gid",
    "business_edge_uid",
})


def _validate_self_digest(value: Mapping[str, Any], field: str, label: str) -> None:
    _digest(value[field], f"{label} {field}")
    unsigned = {
        name: copy.deepcopy(item) for name, item in value.items() if name != field
    }
    if value[field] != _sha256_json(unsigned):
        raise ValueError(f"{label} self-digest drifted")


def validate_foundation_authoring_request(value: Any) -> Mapping[str, Any]:
    raw = _strict_mapping(
        value,
        _FOUNDATION_AUTHORING_REQUEST_FIELDS,
        "capability foundation authoring request",
    )
    if (
        raw["schema"] != CAPABILITY_FOUNDATION_AUTHORING_REQUEST_SCHEMA
        or not isinstance(raw["revision"], str)
        or _REVISION_RE.fullmatch(raw["revision"]) is None
        or raw["secret_material_recorded"] is not False
        or raw["secret_digest_recorded"] is not False
        or raw["semantic_content_recorded"] is not False
    ):
        raise ValueError("capability foundation authoring request is invalid")
    _terminal_receipt_binding(
        raw["full_canary_terminal_receipt"],
        raw["full_canary_terminal_receipt_sha256"],
        revision=raw["revision"],
    )
    _validate_self_digest(
        raw, "request_sha256", "capability foundation authoring request"
    )
    return copy.deepcopy(dict(raw))


def read_foundation_authoring_request(stream: BinaryIO) -> Mapping[str, Any]:
    raw = stream.read(2 * 1024 * 1024 + 1)
    if not raw or len(raw) > 2 * 1024 * 1024 or stream.read(1):
        raise ValueError("capability foundation authoring request input is invalid")
    value = _decode_json(raw, label="capability foundation authoring request")
    if raw != _canonical_bytes(value):
        raise ValueError("capability foundation authoring request is not canonical")
    return validate_foundation_authoring_request(value)


def validate_foundation_authoring_context(value: Any) -> Mapping[str, Any]:
    raw = _strict_mapping(
        value,
        _FOUNDATION_AUTHORING_CONTEXT_FIELDS,
        "capability foundation authoring context",
    )
    identities = _strict_mapping(
        raw["identities"],
        _FOUNDATION_AUTHORING_IDENTITY_FIELDS,
        "capability foundation authoring identities",
    )
    identity_observation = _strict_mapping(
        raw["identity_observation"],
        {
            "service_user",
            "service_group",
            "service_uid",
            "service_gid",
            "socket_client_group",
            "socket_client_gid",
            "state",
        },
        "capability foundation identity observation",
    )
    if (
        raw["schema"] != CAPABILITY_FOUNDATION_AUTHORING_CONTEXT_SCHEMA
        or not isinstance(raw["revision"], str)
        or _REVISION_RE.fullmatch(raw["revision"]) is None
        or raw["staged_plan_path"] != str(DEFAULT_STAGED_FULL_CANARY_PLAN_PATH)
        or raw["mutation_performed"] is not False
        or raw["secret_material_recorded"] is not False
        or raw["secret_digest_recorded"] is not False
        or raw["semantic_content_recorded"] is not False
        or any(
            _positive_id(identities[name], f"foundation {name}") != identities[name]
            for name in _FOUNDATION_AUTHORING_IDENTITY_FIELDS
        )
        or identities
        != {
            "service_uid": CAPABILITY_PLANNED_IDENTITIES["bitrix_operational_edge_uid"],
            "service_gid": CAPABILITY_PLANNED_IDENTITIES["bitrix_operational_edge_gid"],
            "socket_client_gid": CAPABILITY_PLANNED_IDENTITIES[
                "bitrix_operational_edge_client_gid"
            ],
            "business_edge_uid": CAPABILITY_PLANNED_IDENTITIES["mac_ops_uid"],
        }
        or identity_observation
        != {
            "service_user": "muncho-edge-bitrix",
            "service_group": "muncho-edge-bitrix",
            "service_uid": CAPABILITY_PLANNED_IDENTITIES["bitrix_operational_edge_uid"],
            "service_gid": CAPABILITY_PLANNED_IDENTITIES["bitrix_operational_edge_gid"],
            "socket_client_group": "muncho-edge-bitrix-c",
            "socket_client_gid": CAPABILITY_PLANNED_IDENTITIES[
                "bitrix_operational_edge_client_gid"
            ],
            "state": identity_observation.get("state"),
        }
        or identity_observation["state"]
        not in {
            "present_exact",
            "groups_present_user_absent_create_only_slot",
            "service_group_present_create_only_slot",
            "absent_create_only_slot",
        }
    ):
        raise ValueError("capability foundation authoring context is invalid")
    terminal, terminal_sha256 = _terminal_receipt_binding(
        raw["full_canary_terminal_receipt"],
        raw["full_canary_terminal_receipt_sha256"],
        revision=raw["revision"],
        full_canary_plan_sha256=raw["full_canary_plan_sha256"],
    )
    if (
        raw["original_full_canary_owner_approval_sha256"]
        != terminal["owner_approval_sha256"]
    ):
        raise ValueError("foundation original owner approval binding drifted")
    for name in (
        "staged_plan_file_sha256",
        "full_canary_plan_sha256",
        "release_artifact_sha256",
        "asset_manifest_sha256",
        "original_full_canary_owner_approval_sha256",
    ):
        _digest(raw[name], f"foundation authoring context {name}")
    if terminal_sha256 != raw["full_canary_terminal_receipt_sha256"]:
        raise ValueError("foundation terminal receipt binding drifted")
    _validate_self_digest(
        raw, "receipt_sha256", "capability foundation authoring context"
    )
    return copy.deepcopy(dict(raw))


def collect_foundation_authoring_context(request_value: Any) -> Mapping[str, Any]:
    """Collect pre-plan foundation facts from fixed sealed paths without mutation."""

    _require_root_linux()
    request = validate_foundation_authoring_request(request_value)
    full_plan, plan_raw, identity = _read_staged_full_canary_plan()
    terminal, terminal_sha256 = _terminal_receipt_binding(
        request["full_canary_terminal_receipt"],
        request["full_canary_terminal_receipt_sha256"],
        revision=full_plan.revision,
        full_canary_plan_sha256=full_plan.sha256,
    )
    if request["revision"] != full_plan.revision:
        raise PermissionError("foundation authoring request revision drifted")
    assets = verify_packaged_operational_assets(
        release_root=Path(full_plan.release["artifact_root"]),
        revision=full_plan.revision,
        expected_uid=0,
        expected_gid=0,
    )
    identity_observation = _observe_bitrix_foundation_identity(
        service_uid=CAPABILITY_PLANNED_IDENTITIES["bitrix_operational_edge_uid"],
        service_gid=CAPABILITY_PLANNED_IDENTITIES["bitrix_operational_edge_gid"],
        socket_client_gid=CAPABILITY_PLANNED_IDENTITIES[
            "bitrix_operational_edge_client_gid"
        ],
        allow_absence=True,
    )
    unsigned = {
        "schema": CAPABILITY_FOUNDATION_AUTHORING_CONTEXT_SCHEMA,
        "revision": full_plan.revision,
        "staged_plan_path": str(DEFAULT_STAGED_FULL_CANARY_PLAN_PATH),
        "staged_plan_file_sha256": _sha256_bytes(plan_raw),
        "staged_plan_identity": copy.deepcopy(dict(identity)),
        "full_canary_plan_sha256": full_plan.sha256,
        "release_artifact_sha256": full_plan.release["artifact_sha256"],
        "full_canary_terminal_receipt": terminal,
        "full_canary_terminal_receipt_sha256": terminal_sha256,
        "original_full_canary_owner_approval_sha256": terminal["owner_approval_sha256"],
        "identities": {
            "service_uid": CAPABILITY_PLANNED_IDENTITIES["bitrix_operational_edge_uid"],
            "service_gid": CAPABILITY_PLANNED_IDENTITIES["bitrix_operational_edge_gid"],
            "socket_client_gid": CAPABILITY_PLANNED_IDENTITIES[
                "bitrix_operational_edge_client_gid"
            ],
            "business_edge_uid": CAPABILITY_PLANNED_IDENTITIES["mac_ops_uid"],
        },
        "identity_observation": copy.deepcopy(dict(identity_observation)),
        "asset_manifest_sha256": assets["manifest_sha256"],
        "mutation_performed": False,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
        "semantic_content_recorded": False,
    }
    return validate_foundation_authoring_context({
        **unsigned,
        "receipt_sha256": _sha256_json(unsigned),
    })


_PLAN_PUBLICATION_IDENTITY_FIELDS = frozenset({
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
    "producer_business_edge_uid",
    "producer_business_edge_gid",
    "producer_canonical_writer_uid",
    "producer_canonical_writer_gid",
    "producer_discord_edge_uid",
    "producer_discord_edge_gid",
    "producer_gateway_observer_uid",
    "producer_gateway_observer_gid",
    "producer_receipt_writer_gid",
})
_RUNTIME_PLAN_IDENTITY_FIELDS = frozenset({
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
})


def _runtime_plan_identity_inputs(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return {field: value[field] for field in _RUNTIME_PLAN_IDENTITY_FIELDS}


_PLAN_PUBLICATION_DISCORD_FIELDS = frozenset({
    "connector_bot_user_id",
    "routeback_bot_user_id",
    "allowed_guild_ids",
    "allowed_channel_ids",
    "allowed_user_ids",
})
_PLAN_PUBLICATION_ARTIFACT_FIELDS = frozenset({
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
})
_PLAN_INPUT_FIELDS = frozenset({
    "schema",
    "full_canary_terminal_receipt",
    "full_canary_terminal_receipt_sha256",
    "identities",
    "discord",
    "artifacts",
    "inputs_sha256",
})
_PLAN_AUTHORING_REQUEST_FIELDS = frozenset({
    "schema",
    "revision",
    "full_canary_terminal_receipt",
    "full_canary_terminal_receipt_sha256",
    "foundation_authoring_context",
    "foundation_authoring_context_receipt_sha256",
    "bitrix_foundation_receipt",
    "bitrix_foundation_receipt_sha256",
    "connector_bot_user_id",
    "routeback_bot_user_id",
    "secret_material_recorded",
    "secret_digest_recorded",
    "semantic_content_recorded",
    "request_sha256",
})
_PLAN_AUTHORING_CONTEXT_FIELDS = frozenset({
    "schema",
    "revision",
    "staged_plan_path",
    "staged_plan_file_sha256",
    "staged_plan_identity",
    "full_canary_plan_sha256",
    "full_canary_terminal_receipt",
    "full_canary_terminal_receipt_sha256",
    "original_full_canary_owner_approval_sha256",
    "foundation_authoring_context",
    "foundation_authoring_context_receipt_sha256",
    "bitrix_foundation_receipt",
    "bitrix_foundation_receipt_sha256",
    "plan_inputs",
    "host_identity_observations",
    "capability_inputs_sha256",
    "capability_plan_sha256",
    "mutation_performed",
    "secret_material_recorded",
    "secret_digest_recorded",
    "semantic_content_recorded",
    "receipt_sha256",
})


def validate_plan_publication_inputs(value: Any) -> Mapping[str, Any]:
    raw = _strict_mapping(value, _PLAN_INPUT_FIELDS, "capability plan inputs")
    if raw["schema"] != CAPABILITY_PLAN_INPUTS_SCHEMA:
        raise ValueError("capability plan inputs schema is invalid")
    terminal, terminal_sha256 = _terminal_receipt_binding(
        raw["full_canary_terminal_receipt"],
        raw["full_canary_terminal_receipt_sha256"],
    )
    identities = _strict_mapping(
        raw["identities"],
        _PLAN_PUBLICATION_IDENTITY_FIELDS,
        "capability plan identities",
    )
    if identities != CAPABILITY_PLANNED_IDENTITIES:
        raise ValueError("capability plan identities are not the fixed inventory")
    discord = _strict_mapping(
        raw["discord"],
        _PLAN_PUBLICATION_DISCORD_FIELDS,
        "capability plan Discord inputs",
    )
    connector = _snowflake_id(
        discord["connector_bot_user_id"], "connector bot identity"
    )
    routeback = _snowflake_id(
        discord["routeback_bot_user_id"], "route-back bot identity"
    )
    if len({connector, routeback, PRODUCTION_DISCORD_BOT_USER_ID}) != 3:
        raise ValueError("capability Discord bot identities are not isolated")
    if (
        discord["allowed_guild_ids"] != [PRODUCTION_CANARY_PUBLIC_GUILD_ID]
        or discord["allowed_channel_ids"] != [PRODUCTION_CANARY_PUBLIC_CHANNEL_ID]
        or discord["allowed_user_ids"] != [PRODUCTION_OWNER_USER_ID]
        or LOCKED_NONPUBLIC_CHANNEL_IDS.intersection(discord["allowed_channel_ids"])
    ):
        raise ValueError("capability canary public Discord target is invalid")
    for name in (
        "allowed_guild_ids",
        "allowed_channel_ids",
        "allowed_user_ids",
    ):
        _canonical_snowflake_list(discord[name], f"Discord {name}")
    artifacts = _strict_mapping(
        raw["artifacts"],
        _PLAN_PUBLICATION_ARTIFACT_FIELDS,
        "capability plan artifacts",
    )
    for name in _PLAN_PUBLICATION_ARTIFACT_FIELDS:
        _digest(artifacts[name], f"capability plan {name}")
    _validate_self_digest(raw, "inputs_sha256", "capability plan inputs")
    if terminal_sha256 != terminal["receipt_sha256"]:
        raise ValueError("capability plan terminal receipt drifted")
    return copy.deepcopy(dict(raw))


def _validate_supplied_bitrix_foundation_receipt(
    value: Any,
    *,
    receipt_sha256: Any,
    terminal: Mapping[str, Any],
    context: Mapping[str, Any],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("Bitrix foundation receipt is invalid")
    raw = copy.deepcopy(dict(value))
    _validate_self_digest(raw, "receipt_sha256", "Bitrix foundation receipt")
    digest = _digest(receipt_sha256, "Bitrix foundation receipt")
    if (
        raw.get("schema") != CAPABILITY_BITRIX_FOUNDATION_RECEIPT_SCHEMA
        or digest != raw["receipt_sha256"]
        or raw.get("revision") != terminal["release_sha"]
        or raw.get("full_canary_plan_sha256") != terminal["full_canary_plan_sha256"]
        or raw.get("full_canary_terminal_receipt") != terminal
        or raw.get("full_canary_terminal_receipt_sha256") != terminal["receipt_sha256"]
        or raw.get("original_full_canary_owner_approval_sha256")
        != terminal["owner_approval_sha256"]
        or raw.get("foundation_authoring_context_receipt_sha256")
        != context["receipt_sha256"]
        or raw.get("identity_bootstrap_receipt_sha256") is None
        or raw.get("key_bootstrap_receipt_sha256") is None
        or raw.get("receipt_public_key_id") is None
        or raw.get("secret_material_recorded") is not False
        or raw.get("secret_digest_recorded") is not False
    ):
        raise ValueError("Bitrix foundation receipt binding is invalid")
    return raw


def validate_plan_authoring_request(value: Any) -> Mapping[str, Any]:
    raw = _strict_mapping(
        value, _PLAN_AUTHORING_REQUEST_FIELDS, "capability plan authoring request"
    )
    if (
        raw["schema"] != CAPABILITY_PLAN_AUTHORING_REQUEST_SCHEMA
        or not isinstance(raw["revision"], str)
        or _REVISION_RE.fullmatch(raw["revision"]) is None
        or raw["secret_material_recorded"] is not False
        or raw["secret_digest_recorded"] is not False
        or raw["semantic_content_recorded"] is not False
    ):
        raise ValueError("capability plan authoring request is invalid")
    terminal, terminal_sha256 = _terminal_receipt_binding(
        raw["full_canary_terminal_receipt"],
        raw["full_canary_terminal_receipt_sha256"],
        revision=raw["revision"],
    )
    context = validate_foundation_authoring_context(raw["foundation_authoring_context"])
    if (
        raw["foundation_authoring_context_receipt_sha256"] != context["receipt_sha256"]
        or context["revision"] != raw["revision"]
        or context["full_canary_terminal_receipt"] != terminal
        or context["full_canary_terminal_receipt_sha256"] != terminal_sha256
    ):
        raise ValueError("capability foundation authoring binding drifted")
    _validate_supplied_bitrix_foundation_receipt(
        raw["bitrix_foundation_receipt"],
        receipt_sha256=raw["bitrix_foundation_receipt_sha256"],
        terminal=terminal,
        context=context,
    )
    connector = _snowflake_id(raw["connector_bot_user_id"], "connector bot identity")
    routeback = _snowflake_id(raw["routeback_bot_user_id"], "route-back bot identity")
    if len({connector, routeback, PRODUCTION_DISCORD_BOT_USER_ID}) != 3:
        raise ValueError("capability Discord bot identities are not isolated")
    _validate_self_digest(raw, "request_sha256", "capability plan authoring request")
    return copy.deepcopy(dict(raw))


def read_plan_authoring_request(stream: BinaryIO) -> Mapping[str, Any]:
    raw = stream.read(4 * 1024 * 1024 + 1)
    if not raw or len(raw) > 4 * 1024 * 1024 or stream.read(1):
        raise ValueError("capability plan authoring request input is invalid")
    value = _decode_json(raw, label="capability plan authoring request")
    if raw != _canonical_bytes(value):
        raise ValueError("capability plan authoring request is not canonical")
    return validate_plan_authoring_request(value)


def _stable_executable_sha256(path: Path, label: str) -> str:
    raw, _ = _read_stable_file(
        path,
        maximum=256 * 1024 * 1024,
        expected_uid=0,
        allowed_modes=frozenset({0o555, 0o755}),
    )
    if not raw:
        raise RuntimeError(f"{label} is empty")
    return _sha256_bytes(raw)


def collect_plan_authoring_context(request_value: Any) -> Mapping[str, Any]:
    """Build complete public plan inputs only from sealed remote truth."""

    _require_root_linux()
    request = validate_plan_authoring_request(request_value)
    full_plan, staged_raw, staged_identity = _read_staged_full_canary_plan()
    terminal, terminal_sha256 = _terminal_receipt_binding(
        request["full_canary_terminal_receipt"],
        request["full_canary_terminal_receipt_sha256"],
        revision=full_plan.revision,
        full_canary_plan_sha256=full_plan.sha256,
    )
    if request["revision"] != full_plan.revision:
        raise PermissionError("capability plan authoring revision drifted")
    foundation_context = validate_foundation_authoring_context(
        request["foundation_authoring_context"]
    )
    bitrix = _validate_supplied_bitrix_foundation_receipt(
        request["bitrix_foundation_receipt"],
        receipt_sha256=request["bitrix_foundation_receipt_sha256"],
        terminal=terminal,
        context=foundation_context,
    )
    release_root = Path(full_plan.release["artifact_root"])
    runtime_manifest = verify_release_runtime_dependency_manifest(
        release_root, full_plan.revision
    )
    manifest_raw, _ = _read_stable_file(
        release_root / RUNTIME_DEPENDENCY_MANIFEST_RELATIVE,
        maximum=2 * 1024 * 1024,
        expected_uid=0,
        allowed_modes=frozenset({0o444, 0o644}),
    )
    agent_browser = runtime_manifest["agent_browser"]
    chrome = runtime_manifest["chrome"]
    identities = copy.deepcopy(dict(CAPABILITY_PLANNED_IDENTITIES))
    artifacts = {
        "browser_node_sha256": agent_browser["node_sha256"],
        "browser_wrapper_sha256": agent_browser["wrapper_sha256"],
        "browser_native_sha256": agent_browser["native_sha256"],
        "browser_executable_sha256": chrome["executable_sha256"],
        "agent_browser_config_sha256": agent_browser["config_sha256"],
        "worker_bwrap_sha256": _stable_executable_sha256(
            BWRAP_PATH, "isolated worker bwrap"
        ),
        "worker_shell_sha256": _stable_executable_sha256(
            SHELL_PATH, "isolated worker shell"
        ),
        "runtime_dependency_manifest_sha256": _sha256_bytes(manifest_raw),
        "bitrix_operational_edge_asset_manifest_sha256": bitrix[
            "asset_manifest_sha256"
        ],
        "bitrix_operational_edge_rendered_unit_sha256": bitrix["rendered_unit_sha256"],
        "bitrix_operational_edge_rendered_config_sha256": bitrix[
            "rendered_config_sha256"
        ],
        "bitrix_operational_edge_rendered_trust_sha256": bitrix[
            "rendered_trust_sha256"
        ],
        "bitrix_operational_edge_identity_bootstrap_receipt_sha256": bitrix[
            "identity_bootstrap_receipt_sha256"
        ],
        "bitrix_operational_edge_receipt_public_key_id": bitrix[
            "receipt_public_key_id"
        ],
        "bitrix_operational_edge_key_bootstrap_receipt_sha256": bitrix[
            "key_bootstrap_receipt_sha256"
        ],
    }
    inputs_unsigned = {
        "schema": CAPABILITY_PLAN_INPUTS_SCHEMA,
        "full_canary_terminal_receipt": terminal,
        "full_canary_terminal_receipt_sha256": terminal_sha256,
        "identities": identities,
        "discord": {
            "connector_bot_user_id": request["connector_bot_user_id"],
            "routeback_bot_user_id": request["routeback_bot_user_id"],
            "allowed_guild_ids": [PRODUCTION_CANARY_PUBLIC_GUILD_ID],
            "allowed_channel_ids": [PRODUCTION_CANARY_PUBLIC_CHANNEL_ID],
            "allowed_user_ids": [PRODUCTION_OWNER_USER_ID],
        },
        "artifacts": artifacts,
    }
    plan_inputs = validate_plan_publication_inputs({
        **inputs_unsigned,
        "inputs_sha256": _sha256_json(inputs_unsigned),
    })
    plan = build_capability_plan(
        full_plan=full_plan,
        full_canary_terminal_receipt=terminal,
        full_canary_terminal_receipt_sha256=terminal_sha256,
        **_runtime_plan_identity_inputs(identities),
        connector_bot_user_id=request["connector_bot_user_id"],
        routeback_bot_user_id=request["routeback_bot_user_id"],
        connector_allowed_guild_ids=[PRODUCTION_CANARY_PUBLIC_GUILD_ID],
        connector_allowed_channel_ids=[PRODUCTION_CANARY_PUBLIC_CHANNEL_ID],
        connector_allowed_user_ids=[PRODUCTION_OWNER_USER_ID],
        **artifacts,
    )
    from gateway.canonical_capability_canary_producer_units import (
        producer_host_identity_receipt,
    )

    host_identity_observations = {
        "browser": browser_host_identity_receipt(
            plan,
            full_plan,
            allow_create_only_absence=True,
        ),
        "execution": execution_host_identity_receipt(
            plan,
            full_plan,
            allow_create_only_absence=True,
        ),
        "mac_ops": service_host_identity_receipt(
            plan,
            full_plan,
            role="mac_ops",
            allow_create_only_absence=True,
        ),
        "connector": service_host_identity_receipt(
            plan,
            full_plan,
            role="connector",
            allow_create_only_absence=True,
        ),
        "producer": producer_host_identity_receipt(
            plan.sha256,
            allow_create_only_absence=True,
        ),
    }
    unsigned = {
        "schema": CAPABILITY_PLAN_AUTHORING_CONTEXT_SCHEMA,
        "revision": full_plan.revision,
        "staged_plan_path": str(DEFAULT_STAGED_FULL_CANARY_PLAN_PATH),
        "staged_plan_file_sha256": _sha256_bytes(staged_raw),
        "staged_plan_identity": copy.deepcopy(dict(staged_identity)),
        "full_canary_plan_sha256": full_plan.sha256,
        "full_canary_terminal_receipt": terminal,
        "full_canary_terminal_receipt_sha256": terminal_sha256,
        "original_full_canary_owner_approval_sha256": terminal["owner_approval_sha256"],
        "foundation_authoring_context": foundation_context,
        "foundation_authoring_context_receipt_sha256": foundation_context[
            "receipt_sha256"
        ],
        "bitrix_foundation_receipt": bitrix,
        "bitrix_foundation_receipt_sha256": bitrix["receipt_sha256"],
        "plan_inputs": plan_inputs,
        "host_identity_observations": host_identity_observations,
        "capability_inputs_sha256": plan_inputs["inputs_sha256"],
        "capability_plan_sha256": plan.sha256,
        "mutation_performed": False,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
        "semantic_content_recorded": False,
    }
    return validate_plan_authoring_context({
        **unsigned,
        "receipt_sha256": _sha256_json(unsigned),
    })


def validate_plan_authoring_context(value: Any) -> Mapping[str, Any]:
    raw = _strict_mapping(
        value, _PLAN_AUTHORING_CONTEXT_FIELDS, "capability plan authoring context"
    )
    if (
        raw["schema"] != CAPABILITY_PLAN_AUTHORING_CONTEXT_SCHEMA
        or not isinstance(raw["revision"], str)
        or _REVISION_RE.fullmatch(raw["revision"]) is None
        or raw["staged_plan_path"] != str(DEFAULT_STAGED_FULL_CANARY_PLAN_PATH)
        or raw["mutation_performed"] is not False
        or raw["secret_material_recorded"] is not False
        or raw["secret_digest_recorded"] is not False
        or raw["semantic_content_recorded"] is not False
    ):
        raise ValueError("capability plan authoring context is invalid")
    terminal, terminal_sha256 = _terminal_receipt_binding(
        raw["full_canary_terminal_receipt"],
        raw["full_canary_terminal_receipt_sha256"],
        revision=raw["revision"],
        full_canary_plan_sha256=raw["full_canary_plan_sha256"],
    )
    foundation = validate_foundation_authoring_context(
        raw["foundation_authoring_context"]
    )
    if (
        raw["original_full_canary_owner_approval_sha256"]
        != terminal["owner_approval_sha256"]
        or raw["foundation_authoring_context_receipt_sha256"]
        != foundation["receipt_sha256"]
        or foundation["full_canary_terminal_receipt"] != terminal
        or foundation["full_canary_terminal_receipt_sha256"] != terminal_sha256
    ):
        raise ValueError("capability plan authoring foundation chain drifted")
    bitrix = _validate_supplied_bitrix_foundation_receipt(
        raw["bitrix_foundation_receipt"],
        receipt_sha256=raw["bitrix_foundation_receipt_sha256"],
        terminal=terminal,
        context=foundation,
    )
    inputs = validate_plan_publication_inputs(raw["plan_inputs"])
    observations = _strict_mapping(
        raw["host_identity_observations"],
        {"browser", "execution", "mac_ops", "connector", "producer"},
        "capability plan host identity observations",
    )
    for role, observation in observations.items():
        if not isinstance(observation, Mapping):
            raise ValueError(f"capability {role} host identity observation is invalid")
        observed = copy.deepcopy(dict(observation))
        _validate_self_digest(
            observed,
            "receipt_sha256",
            f"capability {role} host identity observation",
        )
        if observed.get("plan_sha256") != raw["capability_plan_sha256"]:
            raise ValueError("capability host identity plan binding drifted")
        if role in {"mac_ops", "connector"} and (
            observed.get("schema") != CAPABILITY_SERVICE_HOST_IDENTITY_SCHEMA
            or observed.get("role") != role
            or observed.get("state")
            not in {
                "present_exact",
                "group_present_user_absent_create_only_slot",
                "absent_create_only_slot",
            }
            or observed.get("create_only_eligible") is not True
        ):
            raise ValueError("capability service host identity observation is invalid")
    if (
        observations["browser"].get("schema") != CAPABILITY_BROWSER_HOST_IDENTITY_SCHEMA
        or observations["execution"].get("schema")
        != CAPABILITY_EXECUTION_HOST_IDENTITY_SCHEMA
    ):
        raise ValueError("capability host identity observation schema is invalid")
    from gateway.canonical_capability_canary_producer_units import (
        PRODUCER_HOST_IDENTITY_SCHEMA,
        _validate_producer_host_identity_receipt,
    )

    if observations["producer"].get("schema") != PRODUCER_HOST_IDENTITY_SCHEMA:
        raise ValueError("capability producer host identity observation is invalid")
    _validate_producer_host_identity_receipt(
        observations["producer"],
        plan_sha256=raw["capability_plan_sha256"],
        require_present=False,
    )
    if (
        inputs["full_canary_terminal_receipt"] != terminal
        or inputs["full_canary_terminal_receipt_sha256"] != terminal_sha256
        or raw["capability_inputs_sha256"] != inputs["inputs_sha256"]
        or raw["bitrix_foundation_receipt_sha256"] != bitrix["receipt_sha256"]
    ):
        raise ValueError("capability plan authoring input chain drifted")
    for name in (
        "staged_plan_file_sha256",
        "full_canary_plan_sha256",
        "original_full_canary_owner_approval_sha256",
        "foundation_authoring_context_receipt_sha256",
        "bitrix_foundation_receipt_sha256",
        "capability_inputs_sha256",
        "capability_plan_sha256",
    ):
        _digest(raw[name], f"capability plan authoring context {name}")
    _validate_self_digest(raw, "receipt_sha256", "capability plan authoring context")
    return copy.deepcopy(dict(raw))


_PLAN_PUBLICATION_AUTHORITY_FIELDS = frozenset({
    "schema",
    "scope",
    "revision",
    "full_canary_plan_sha256",
    "full_canary_terminal_receipt",
    "full_canary_terminal_receipt_sha256",
    "original_full_canary_owner_approval_sha256",
    "plan_authoring_context",
    "plan_authoring_context_receipt_sha256",
    "plan_sha256",
    "owner_subject_sha256",
    "authority_kind",
    "cryptographic_owner_proof",
    "inputs",
    "secret_material_recorded",
    "secret_digest_recorded",
    "semantic_content_recorded",
    "authority_sha256",
})
_PLAN_PUBLICATION_RECEIPT_FIELDS = frozenset({
    "schema",
    "operation",
    "revision",
    "plan_sha256",
    "full_canary_plan_sha256",
    "full_canary_terminal_receipt",
    "full_canary_terminal_receipt_sha256",
    "original_full_canary_owner_approval_sha256",
    "plan_authoring_context",
    "plan_authoring_context_receipt_sha256",
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
})
_MAX_PLAN_PUBLICATION_AUTHORITY_BYTES = 4 * 1024 * 1024


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
        or raw["authority_kind"] != "trusted_gcloud_owner_explicit_plan_digest"
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
        "original_full_canary_owner_approval_sha256",
        "plan_authoring_context_receipt_sha256",
    ):
        _digest(raw[field], f"capability plan authority {field}")
    terminal, terminal_sha256 = _terminal_receipt_binding(
        raw["full_canary_terminal_receipt"],
        raw["full_canary_terminal_receipt_sha256"],
        revision=raw["revision"],
        full_canary_plan_sha256=raw["full_canary_plan_sha256"],
    )
    if (
        raw["original_full_canary_owner_approval_sha256"]
        != terminal["owner_approval_sha256"]
    ):
        raise ValueError("capability plan original owner approval drifted")
    authoring_context = validate_plan_authoring_context(raw["plan_authoring_context"])
    if (
        raw["plan_authoring_context_receipt_sha256"]
        != authoring_context["receipt_sha256"]
        or authoring_context["revision"] != raw["revision"]
        or authoring_context["full_canary_plan_sha256"]
        != raw["full_canary_plan_sha256"]
        or authoring_context["full_canary_terminal_receipt"] != terminal
        or authoring_context["full_canary_terminal_receipt_sha256"] != terminal_sha256
        or authoring_context["capability_plan_sha256"] != raw["plan_sha256"]
    ):
        raise ValueError("capability plan authoring context binding drifted")
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
        or discord["allowed_channel_ids"] != [PRODUCTION_CANARY_PUBLIC_CHANNEL_ID]
        or discord["allowed_user_ids"] != [PRODUCTION_OWNER_USER_ID]
        or LOCKED_NONPUBLIC_CHANNEL_IDS.intersection(discord["allowed_channel_ids"])
    ):
        raise ValueError("capability canary public Discord target is invalid")
    artifacts = _strict_mapping(
        inputs["artifacts"],
        _PLAN_PUBLICATION_ARTIFACT_FIELDS,
        "capability plan publication artifact hashes",
    )
    for field in _PLAN_PUBLICATION_ARTIFACT_FIELDS:
        _digest(artifacts[field], f"capability plan {field}")
    input_unsigned = {
        "schema": CAPABILITY_PLAN_INPUTS_SCHEMA,
        "full_canary_terminal_receipt": terminal,
        "full_canary_terminal_receipt_sha256": terminal_sha256,
        "identities": copy.deepcopy(dict(identities)),
        "discord": copy.deepcopy(dict(discord)),
        "artifacts": copy.deepcopy(dict(artifacts)),
    }
    validate_plan_publication_inputs({
        **input_unsigned,
        "inputs_sha256": _sha256_json(input_unsigned),
    })
    if authoring_context["plan_inputs"] != {
        **input_unsigned,
        "inputs_sha256": _sha256_json(input_unsigned),
    }:
        raise ValueError("capability plan publication inputs bypass authoring context")
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
        full_canary_terminal_receipt=authority["full_canary_terminal_receipt"],
        full_canary_terminal_receipt_sha256=authority[
            "full_canary_terminal_receipt_sha256"
        ],
        **_runtime_plan_identity_inputs(identities),
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


def _read_exact_publication_file(
    path: Path,
    *,
    maximum: int,
) -> tuple[bytes, os.stat_result]:
    """Read one immutable root publication and retain its stable identity."""

    raw, item = _read_stable_file(
        path,
        maximum=maximum,
        expected_uid=0,
        expected_gid=0,
        allowed_modes=frozenset({0o400}),
    )
    if (
        not stat.S_ISREG(item.st_mode)
        or item.st_uid != 0
        or item.st_gid != 0
        or stat.S_IMODE(item.st_mode) != 0o400
        or item.st_size != len(raw)
    ):
        raise RuntimeError("capability publication file identity drifted")
    return raw, item


def _require_same_file_identity(path: Path, expected: os.stat_result) -> None:
    """Reject replacement or mutation immediately before a paired publish."""

    current = os.lstat(path)
    if (
        not stat.S_ISREG(current.st_mode)
        or current.st_uid != expected.st_uid
        or current.st_gid != expected.st_gid
        or stat.S_IMODE(current.st_mode) != stat.S_IMODE(expected.st_mode)
        or current.st_dev != expected.st_dev
        or current.st_ino != expected.st_ino
        or current.st_size != expected.st_size
        or current.st_mtime_ns != expected.st_mtime_ns
        or current.st_ctime_ns != expected.st_ctime_ns
    ):
        raise RuntimeError("capability publication changed before reconciliation")


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
        key: copy.deepcopy(item) for key, item in raw.items() if key != "receipt_sha256"
    }
    if (
        raw["schema"] != CAPABILITY_PLAN_PUBLICATION_RECEIPT_SCHEMA
        or raw["operation"] != "publish_capability_plan"
        or raw["revision"] != plan.revision
        or raw["plan_sha256"] != plan.sha256
        or raw["full_canary_plan_sha256"] != plan.full_canary_plan_sha256
        or raw["full_canary_terminal_receipt"] != plan.full_canary_terminal_receipt
        or raw["full_canary_terminal_receipt_sha256"]
        != plan.full_canary_terminal_receipt_sha256
        or raw["original_full_canary_owner_approval_sha256"]
        != plan.original_full_canary_owner_approval_sha256
        or raw["plan_authoring_context"] != authority["plan_authoring_context"]
        or raw["plan_authoring_context_receipt_sha256"]
        != authority["plan_authoring_context_receipt_sha256"]
        or raw["plan_path"] != str(DEFAULT_PLAN_PATH)
        or raw["plan_file_sha256"] != _sha256_bytes(plan_payload)
        or raw["authority_sha256"] != authority["authority_sha256"]
        or raw["owner_subject_sha256"] != authority["owner_subject_sha256"]
        or raw["connector_bot_user_id"] != plan.connector_bot_user_id
        or raw["routeback_bot_user_id"] != plan.routeback_bot_user_id
        or raw["production_bot_user_id"] != PRODUCTION_DISCORD_BOT_USER_ID
        or raw["stopped_service_state_sha256"] != stopped_service_state_sha256
        or raw["prerequisite_evidence_sha256"] != prerequisite_evidence_sha256
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


def _build_plan_publication_receipt(
    *,
    authority: Mapping[str, Any],
    plan: CapabilityCanaryPlan,
    plan_payload: bytes,
    receipt_path: Path,
    stopped_service_state_sha256: str,
    prerequisite_evidence_sha256: str,
    published_at_unix: int,
) -> Mapping[str, Any]:
    """Build the deterministic receipt paired with an exact published plan."""

    if type(published_at_unix) is not int or published_at_unix < 0:
        raise ValueError("capability plan publication time is invalid")
    unsigned = {
        "schema": CAPABILITY_PLAN_PUBLICATION_RECEIPT_SCHEMA,
        "operation": "publish_capability_plan",
        "revision": plan.revision,
        "plan_sha256": plan.sha256,
        "full_canary_plan_sha256": plan.full_canary_plan_sha256,
        "full_canary_terminal_receipt": copy.deepcopy(
            dict(plan.full_canary_terminal_receipt)
        ),
        "full_canary_terminal_receipt_sha256": (
            plan.full_canary_terminal_receipt_sha256
        ),
        "original_full_canary_owner_approval_sha256": (
            plan.original_full_canary_owner_approval_sha256
        ),
        "plan_authoring_context": copy.deepcopy(
            dict(authority["plan_authoring_context"])
        ),
        "plan_authoring_context_receipt_sha256": authority[
            "plan_authoring_context_receipt_sha256"
        ],
        "plan_path": str(DEFAULT_PLAN_PATH),
        "plan_file_sha256": _sha256_bytes(plan_payload),
        "authority_sha256": authority["authority_sha256"],
        "owner_subject_sha256": authority["owner_subject_sha256"],
        "connector_bot_user_id": plan.connector_bot_user_id,
        "routeback_bot_user_id": plan.routeback_bot_user_id,
        "production_bot_user_id": PRODUCTION_DISCORD_BOT_USER_ID,
        "stopped_service_state_sha256": stopped_service_state_sha256,
        "prerequisite_evidence_sha256": prerequisite_evidence_sha256,
        "receipt_path": str(receipt_path),
        "published_at_unix": published_at_unix,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
        "semantic_content_recorded": False,
    }
    return {**unsigned, "receipt_sha256": _sha256_json(unsigned)}


def load_bound_plan_publication_receipt(
    plan: CapabilityCanaryPlan,
) -> Mapping[str, Any]:
    """Read and bind the immutable publication receipt without recomputation."""

    receipt_path = _plan_publication_receipt_path(plan)
    raw, _ = _read_stable_file(
        receipt_path,
        maximum=4 * 1024 * 1024,
        expected_uid=0,
        expected_gid=0,
        allowed_modes=frozenset({0o400}),
    )
    value = _decode_json(raw, label="capability plan publication receipt")
    if raw != _canonical_bytes(value):
        raise RuntimeError("capability plan publication receipt is not canonical")
    receipt = _strict_mapping(
        value,
        _PLAN_PUBLICATION_RECEIPT_FIELDS,
        "capability plan publication receipt",
    )
    unsigned = {
        key: copy.deepcopy(item)
        for key, item in receipt.items()
        if key != "receipt_sha256"
    }
    context = validate_plan_authoring_context(receipt["plan_authoring_context"])
    plan_raw = _read_published_plan_file(DEFAULT_PLAN_PATH, maximum=_MAX_PLAN_BYTES)
    if (
        receipt["schema"] != CAPABILITY_PLAN_PUBLICATION_RECEIPT_SCHEMA
        or receipt["operation"] != "publish_capability_plan"
        or receipt["revision"] != plan.revision
        or receipt["plan_sha256"] != plan.sha256
        or receipt["full_canary_plan_sha256"] != plan.full_canary_plan_sha256
        or receipt["full_canary_terminal_receipt"] != plan.full_canary_terminal_receipt
        or receipt["full_canary_terminal_receipt_sha256"]
        != plan.full_canary_terminal_receipt_sha256
        or receipt["original_full_canary_owner_approval_sha256"]
        != plan.original_full_canary_owner_approval_sha256
        or receipt["plan_authoring_context_receipt_sha256"] != context["receipt_sha256"]
        or context["capability_plan_sha256"] != plan.sha256
        or receipt["plan_path"] != str(DEFAULT_PLAN_PATH)
        or receipt["plan_file_sha256"] != _sha256_bytes(plan_raw)
        or plan_raw != _canonical_bytes(plan.to_mapping())
        or receipt["receipt_path"] != str(receipt_path)
        or receipt["receipt_sha256"] != _sha256_json(unsigned)
    ):
        raise RuntimeError("capability plan publication receipt binding drifted")
    return copy.deepcopy(dict(receipt))


def load_bound_reviewed_fixture_publication(
    plan: CapabilityCanaryPlan,
    full_plan: FullCanaryPlan,
) -> Mapping[str, Any]:
    """Load the reviewed fixture and its append-only publication receipt."""

    from gateway import canonical_capability_canary_e2e as evidence_contract
    from gateway.canonical_capability_canary_live_driver import (
        DEFAULT_FIXTURE_PUBLICATION_ROOT,
        DEFAULT_REVIEWED_FIXTURE,
        FIXTURE_PUBLICATION_RECEIPT_SCHEMA,
        MAX_FIXTURE_BYTES,
        _fixture_publication_receipt_path,
        _validate_fixture_plan_binding,
    )

    fixture_raw, fixture_item = _read_stable_file(
        DEFAULT_REVIEWED_FIXTURE,
        maximum=MAX_FIXTURE_BYTES,
        expected_uid=0,
        expected_gid=0,
        allowed_modes=frozenset({0o400}),
    )
    fixture = _decode_json(fixture_raw, label="reviewed capability fixture")
    if fixture_raw != _canonical_bytes(fixture):
        raise RuntimeError("reviewed capability fixture is not canonical")
    fixture_sha256 = _sha256_bytes(fixture_raw)
    evidence_contract._validate_fixture(fixture, fixture_sha256)
    _validate_fixture_plan_binding(fixture, plan=plan, full_plan=full_plan)
    receipt_path = _fixture_publication_receipt_path(
        root=DEFAULT_FIXTURE_PUBLICATION_ROOT,
        plan_sha256=plan.sha256,
        run_id=fixture["run_id"],
        fixture_sha256=fixture_sha256,
    )
    receipt_raw, _ = _read_stable_file(
        receipt_path,
        maximum=MAX_FIXTURE_BYTES,
        expected_uid=0,
        expected_gid=0,
        allowed_modes=frozenset({0o400}),
    )
    receipt = _decode_json(receipt_raw, label="reviewed fixture publication receipt")
    expected_fields = {
        "schema",
        "run_id",
        "release_sha",
        "capability_plan_sha256",
        "full_canary_plan_sha256",
        "full_canary_terminal_receipt",
        "full_canary_terminal_receipt_sha256",
        "original_full_canary_owner_approval_sha256",
        "plan_publication_receipt_sha256",
        "producer_foundation_sha256",
        "authority_sha256",
        "fixture_path",
        "fixture_sha256",
        "fixture_file_identity",
        "receipt_path",
        "published_at_unix_ms",
        "receipt_sha256",
    }
    unsigned = {
        key: copy.deepcopy(item)
        for key, item in receipt.items()
        if key != "receipt_sha256"
    }
    expected_fixture_identity = {
        "device": fixture_item.st_dev,
        "inode": fixture_item.st_ino,
        "uid": fixture_item.st_uid,
        "gid": fixture_item.st_gid,
        "mode": f"{stat.S_IMODE(fixture_item.st_mode):04o}",
        "size": fixture_item.st_size,
        "mtime_ns": fixture_item.st_mtime_ns,
    }
    plan_publication = load_bound_plan_publication_receipt(plan)
    if (
        receipt_raw != _canonical_bytes(receipt)
        or set(receipt) != expected_fields
        or receipt["schema"] != FIXTURE_PUBLICATION_RECEIPT_SCHEMA
        or receipt["run_id"] != fixture["run_id"]
        or receipt["release_sha"] != plan.revision
        or receipt["capability_plan_sha256"] != plan.sha256
        or receipt["full_canary_plan_sha256"] != full_plan.sha256
        or receipt["full_canary_terminal_receipt"] != plan.full_canary_terminal_receipt
        or receipt["full_canary_terminal_receipt_sha256"]
        != plan.full_canary_terminal_receipt_sha256
        or receipt["original_full_canary_owner_approval_sha256"]
        != plan.original_full_canary_owner_approval_sha256
        or receipt["plan_publication_receipt_sha256"]
        != plan_publication["receipt_sha256"]
        or fixture["plan_publication_receipt_sha256"]
        != plan_publication["receipt_sha256"]
        or receipt["producer_foundation_sha256"]
        != fixture["producer_foundation_sha256"]
        or receipt["fixture_path"] != str(DEFAULT_REVIEWED_FIXTURE)
        or receipt["fixture_sha256"] != fixture_sha256
        or receipt["fixture_file_identity"] != expected_fixture_identity
        or receipt["receipt_path"] != str(receipt_path)
        or receipt["published_at_unix_ms"] != fixture_item.st_mtime_ns // 1_000_000
        or receipt["receipt_sha256"] != _sha256_bytes(_canonical_bytes(unsigned))
    ):
        raise RuntimeError("reviewed fixture publication binding drifted")
    return {
        "fixture": copy.deepcopy(dict(fixture)),
        "fixture_sha256": fixture_sha256,
        "fixture_valid_until_unix_ms": fixture["valid_until_unix_ms"],
        "publication_receipt": copy.deepcopy(dict(receipt)),
        "publication_receipt_sha256": receipt["receipt_sha256"],
        "plan_publication_receipt_sha256": plan_publication["receipt_sha256"],
        "ready": True,
    }


def publish_capability_plan(authority_value: Any) -> Mapping[str, Any]:
    """Publish the one exact stopped-host capability plan and append-only receipt."""

    _require_root_linux()
    authority = validate_plan_publication_authority(authority_value)
    full_plan = load_full_canary_plan()
    plan = build_plan_from_publication_authority(authority, full_plan)
    with _lifecycle_lock():
        return _publish_capability_plan_locked(authority, full_plan, plan)


def _publish_capability_plan_locked(
    authority: Mapping[str, Any],
    full_plan: FullCanaryPlan,
    plan: CapabilityCanaryPlan,
) -> Mapping[str, Any]:
    """Observe and publish while the shared lifecycle lock remains held."""

    validate_dedicated_canary_host(full_plan)
    release_evidence = _validate_release_manifest(full_plan)
    from gateway.canonical_capability_canary_producer_units import (
        producer_host_identity_receipt,
    )

    current_host_identity_observations = {
        "browser": browser_host_identity_receipt(
            plan, full_plan, allow_create_only_absence=True
        ),
        "execution": execution_host_identity_receipt(
            plan, full_plan, allow_create_only_absence=True
        ),
        "mac_ops": service_host_identity_receipt(
            plan,
            full_plan,
            role="mac_ops",
            allow_create_only_absence=True,
        ),
        "connector": service_host_identity_receipt(
            plan,
            full_plan,
            role="connector",
            allow_create_only_absence=True,
        ),
        "producer": producer_host_identity_receipt(
            plan.sha256,
            allow_create_only_absence=True,
        ),
    }
    if (
        current_host_identity_observations
        != authority["plan_authoring_context"]["host_identity_observations"]
    ):
        raise RuntimeError("capability host identity changed after plan authoring")
    bitrix_identity_observation = _observe_bitrix_foundation_identity(
        service_uid=plan.identities.bitrix_operational_edge_uid,
        service_gid=plan.identities.bitrix_operational_edge_gid,
        socket_client_gid=plan.identities.bitrix_operational_edge_client_gid,
        allow_absence=False,
    )
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
    stopped_state_sha256 = _sha256_json({
        "states": states,
        "checks": dict(sorted(stopped_checks.items())),
    })
    prerequisite_evidence_sha256 = _sha256_json({
        "release": release_evidence,
        "runtime_dependencies": dependency_evidence,
        "browser": browser_evidence,
        "worker": worker_evidence,
        "bitrix_foundation": bitrix_foundation_evidence,
        "bitrix_identity": bitrix_identity_observation,
        "host_identities": current_host_identity_observations,
    })
    plan_payload = _canonical_bytes(plan.to_mapping())
    receipt_path = _plan_publication_receipt_path(plan)

    def commit_publication() -> Mapping[str, Any]:
        plan_exists = os.path.lexists(DEFAULT_PLAN_PATH)
        receipt_exists = os.path.lexists(receipt_path)
        if receipt_exists and not plan_exists:
            raise RuntimeError(
                "capability plan publication receipt exists without its plan"
            )
        if plan_exists:
            existing_plan, plan_item = _read_exact_publication_file(
                DEFAULT_PLAN_PATH,
                maximum=_MAX_PLAN_BYTES,
            )
            existing_receipt = (
                _read_published_plan_file(receipt_path, maximum=4 * 1024 * 1024)
                if receipt_exists
                else None
            )
            if existing_plan != plan_payload:
                raise RuntimeError("capability plan publication conflicts")
            decoded_plan = _decode_json(
                existing_plan, label="published capability plan"
            )
            if existing_plan != _canonical_bytes(decoded_plan):
                raise RuntimeError("capability plan orphan is not canonical")
            if existing_receipt is not None:
                receipt_value = _decode_json(
                    existing_receipt,
                    label="capability plan publication receipt",
                )
                if existing_receipt != _canonical_bytes(receipt_value):
                    raise RuntimeError(
                        "capability plan publication receipt is not canonical"
                    )
                return _validate_plan_publication_receipt(
                    receipt_value,
                    authority=authority,
                    plan=plan,
                    plan_payload=plan_payload,
                    receipt_path=receipt_path,
                    stopped_service_state_sha256=stopped_state_sha256,
                    prerequisite_evidence_sha256=prerequisite_evidence_sha256,
                )
        else:
            _atomic_publish_root_file(DEFAULT_PLAN_PATH, plan_payload)
            existing_plan, plan_item = _read_exact_publication_file(
                DEFAULT_PLAN_PATH,
                maximum=_MAX_PLAN_BYTES,
            )
            if existing_plan != plan_payload:
                raise RuntimeError("capability plan publication readback drifted")

        # A SIGKILL may leave only the exact immutable plan.  Under the same
        # lifecycle lock, and only after revalidating its stable identity and
        # complete canonical bytes, deterministically finish the paired
        # receipt.  Any conflicting orphan remains fail closed.
        receipt = _build_plan_publication_receipt(
            authority=authority,
            plan=plan,
            plan_payload=plan_payload,
            receipt_path=receipt_path,
            stopped_service_state_sha256=stopped_state_sha256,
            prerequisite_evidence_sha256=prerequisite_evidence_sha256,
            published_at_unix=plan_item.st_mtime_ns // 1_000_000_000,
        )
        _require_same_file_identity(DEFAULT_PLAN_PATH, plan_item)
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

    return commit_publication()


def runtime_contract() -> Mapping[str, Any]:
    unsigned = {
        "schema": CAPABILITY_CONTRACT_SCHEMA,
        "normal_gateway_loop": True,
        "model_semantic_authority": True,
        "model": "gpt-5.6-sol",
        "provider": "openai-codex",
        "toolsets": list(FIRST_WAVE_TOOLSETS),
        "kanban_auxiliary_planning_enabled": False,
        "kanban_auto_decompose": False,
        "kanban_dispatch_in_gateway": False,
        "approval_auxiliary_enabled": False,
        "retired_semantic_auxiliary_tasks_present": False,
        "cron_enabled": False,
        "goal_judge_enabled": False,
        "model_authored_goal_outcome_enabled": True,
        "goal_continuations_enabled": True,
        "goal_max_turns": 0,
        "goal_manager": "hermes_cli.goals.GoalManager",
        "goal_outcome_source": "todo.goal_outcome",
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
    raw, _ = _read_stable_file(
        path,
        maximum=_MAX_PLAN_BYTES,
        expected_uid=0,
        expected_gid=0,
        allowed_modes=frozenset({0o400}),
    )
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
    parser = argparse.ArgumentParser(
        description="Production-shaped Muncho capability-canary runtime"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("contract")
    sub.add_parser("collect-foundation-authoring-context")
    sub.add_parser("collect-plan-authoring-context")
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
        if args.command == "collect-foundation-authoring-context":
            _emit(
                collect_foundation_authoring_context(
                    read_foundation_authoring_request(sys.stdin.buffer)
                )
            )
            return 0
        if args.command == "collect-plan-authoring-context":
            _emit(
                collect_plan_authoring_context(
                    read_plan_authoring_request(sys.stdin.buffer)
                )
            )
            return 0
        if args.command == "publish-plan":
            _emit(
                publish_capability_plan(
                    read_plan_publication_authority(sys.stdin.buffer)
                )
            )
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
            metadata, secret = read_secret_lease_frame(
                sys.stdin.buffer, expected_kind=kind
            )
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
                phase=("stopped" if args.command == "preflight-stopped" else "live"),
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
                load_capability_approval()
            )
        elif args.command == "stop":
            result = CapabilityCanaryLifecycle(plan, full).stop()
        else:
            _require_root_linux()
            states = _capability_services(runner=_runner)
            if not all(_service_stopped(state) for state in states.values()):
                raise RuntimeError(
                    "credential retirement requires all services stopped"
                )
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
        failure = {
            "schema": "muncho-production-capability-runtime-failure.v1",
            "ok": False,
            "error_type": type(exc).__name__,
            "error_sha256": _sha256_bytes(
                f"{type(exc).__name__}:{exc}".encode("utf-8", errors="replace")
            ),
        }
        _emit({**failure, "receipt_sha256": _sha256_json(failure)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "APPROVAL_FRAME_MAGIC",
    "CAPABILITY_APPROVAL_INSTALL_RECEIPT_SCHEMA",
    "CAPABILITY_PLAN_PUBLICATION_AUTHORITY_SCHEMA",
    "CAPABILITY_PLAN_PUBLICATION_RECEIPT_SCHEMA",
    "CAPABILITY_ROUTEBACK_BOT_IDENTITY_SCHEMA",
    "CAPABILITY_PLAN_PUBLICATION_SCOPE",
    "CAPABILITY_APPROVAL_SCHEMA",
    "CAPABILITY_BROWSER_HOST_IDENTITY_SCHEMA",
    "CAPABILITY_BROWSER_IDENTITY_FOUNDATION_SCHEMA",
    "CAPABILITY_EXECUTION_HOST_IDENTITY_SCHEMA",
    "CAPABILITY_EXECUTION_IDENTITY_FOUNDATION_SCHEMA",
    "CAPABILITY_EXECUTION_READINESS_SCHEMA",
    "CAPABILITY_SERVICE_IDENTITY_FOUNDATION_SCHEMA",
    "CAPABILITY_SERVICE_HOST_IDENTITY_SCHEMA",
    "CAPABILITY_FOUNDATION_AUTHORING_REQUEST_SCHEMA",
    "CAPABILITY_FOUNDATION_AUTHORING_CONTEXT_SCHEMA",
    "CAPABILITY_PLAN_AUTHORING_REQUEST_SCHEMA",
    "CAPABILITY_PLAN_AUTHORING_CONTEXT_SCHEMA",
    "CAPABILITY_PLAN_INPUTS_SCHEMA",
    "CAPABILITY_OBSERVER_HOOKS",
    "CAPABILITY_OBSERVER_PLUGIN",
    "CAPABILITY_PLAN_SCHEMA",
    "CODEX_FRAME_MAGIC",
    "CAPABILITY_PRODUCTION_OBSERVATION_WAIT_REQUEST_SCHEMA",
    "CAPABILITY_PRODUCTION_OBSERVATION_STAGE_RECEIPT_SCHEMA",
    "CONNECTOR_FRAME_MAGIC",
    "CapabilityCanaryLifecycle",
    "CapabilityCanaryOwnerApproval",
    "CapabilityCanaryPlan",
    "FIRST_WAVE_TOOLSETS",
    "MAC_OPS_FRAME_MAGIC",
    "DEFAULT_SERVICE_IDENTITY_FOUNDATION_ROOT",
    "RuntimeIdentities",
    "attest_capability_execution_readiness",
    "browser_host_identity_receipt",
    "browser_principal_version_smoke",
    "browser_service_runtime_preflight",
    "browser_userns_preflight",
    "build_capability_plan",
    "build_secret_lease_frame",
    "build_capability_stop_proof",
    "build_credential_consumer_stop_proof",
    "build_capability_cleanup_facts",
    "publish_capability_cleanup_facts",
    "build_capability_observer_stop_receipt",
    "build_capability_cleanup_finalization",
    "capability_browser_controller_client_config",
    "capability_browser_controller_client_mapping",
    "capability_gateway_effective_environment_is_sealed",
    "collect_capability_preflight",
    "load_capability_approval",
    "collect_foundation_authoring_context",
    "collect_plan_authoring_context",
    "read_foundation_authoring_request",
    "read_plan_authoring_request",
    "validate_foundation_authoring_context",
    "validate_foundation_authoring_request",
    "validate_plan_authoring_context",
    "validate_plan_authoring_request",
    "validate_plan_publication_inputs",
    "validate_full_canary_terminal_receipt",
    "load_bound_plan_publication_receipt",
    "load_bound_reviewed_fixture_publication",
    "load_capability_plan",
    "install_capability_approval",
    "provision_secret_lease",
    "read_capability_approval",
    "read_secret_lease_frame",
    "render_browser_config",
    "render_browser_unit",
    "render_connector_config",
    "render_connector_unit",
    "render_gateway_config",
    "render_gateway_unit",
    "render_mac_ops_config",
    "render_mac_ops_unit",
    "render_worker_config",
    "render_worker_service_unit",
    "render_worker_socket_unit",
    "retire_secret_lease",
    "runtime_contract",
    "ensure_browser_identity_create_only",
    "ensure_execution_identities_create_only",
    "ensure_service_identities_create_only",
    "build_plan_from_publication_authority",
    "publish_capability_plan",
    "read_owner_signed_production_observation",
    "read_production_observation_wait_request",
    "read_plan_publication_authority",
    "validate_plan_publication_authority",
    "execution_host_identity_receipt",
    "service_host_identity_receipt",
    "load_service_identity_foundation_receipt",
    "stage_and_publish_owner_signed_production_observation",
    "wait_for_capability_production_observation_marker",
    "validate_capability_extension_surface",
    "validate_capability_agent_policy",
    "validate_capability_gateway_config",
    "validate_capability_model_runtime_route",
    "validate_capability_production_diff",
    "worker_executables_preflight",
    "worker_systemd252_preflight",
    "worker_tmpfs_runtime_preflight",
]

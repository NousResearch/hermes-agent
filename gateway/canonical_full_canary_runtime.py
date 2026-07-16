#!/usr/bin/env python3
"""Digest-bound three-service runtime for the isolated full Muncho canary.

This is a mechanical extension of the completed writer-only lifecycle.  It
does not replace that lifecycle and cannot infer approval from its historical
success receipt.  One separately owner-approved full-runtime plan binds the
same sealed release, the stopped writer-only receipt, exact config artifacts,
and exact systemd bytes for:

* the privileged Discord public-egress edge;
* the privileged Canonical Writer; and
* the full, model-running, credential-free Hermes gateway.

The module never enables a unit, creates a timer, reads a secret value into a
receipt, chooses a Discord route, or interprets task text.  Service start order
is edge -> collector/config gate -> writer -> gateway -> authenticated plugin
readiness.  Every stop path is gateway -> writer -> edge.
Successful start deliberately leaves the *disabled* canary units active for a
separate live E2E turn; ``verify-and-stop`` consumes only exact evidence
digests and always performs the ordered stop in ``finally``.
"""

from __future__ import annotations

import argparse
import base64
import copy
import ctypes
import errno
import fcntl
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import yaml

from gateway.canonical_canary_host_identity import (
    DEDICATED_CANARY_INSTANCE_ID,
    DEDICATED_CANARY_INSTANCE_NAME,
    DEDICATED_CANARY_PROJECT_ID,
    DEDICATED_CANARY_PROJECT_NUMBER,
    DEDICATED_CANARY_SERVICE_ACCOUNT,
    DEDICATED_CANARY_ZONE,
    FULL_CANARY_HOST_IDENTITY_SCHEMA,
    _GCE_METADATA_PATHS,
    _LOCAL_HOST_IDENTITY_PATHS,
    _bounded_identity_text,
    _dedicated_canary_gce_identity,
    _observe_dedicated_canary_host,
    _read_gce_metadata_value,
    _read_local_host_identity_value,
    collect_dedicated_canary_host_identity_receipt,
)
from gateway.canonical_writer_activation import ActivationPlan
from gateway.canonical_writer_bootstrap import (
    DEFAULT_WRITER_RUNTIME_ATTESTATION_PATH,
    WRITER_RUNTIME_ATTESTATION_VERSION,
)
from gateway.canonical_writer_db import (
    managed_cloudsqladmin_hba_receipt_from_mapping,
)
from gateway.canonical_writer_readiness import (
    DEFAULT_READINESS_RECEIPT_PATH as DEFAULT_GATEWAY_READINESS_PATH,
    READINESS_RECEIPT_VERSION,
    boot_identity,
    module_file_identity,
    process_start_time_ticks,
    readiness_receipt_sha256,
)
from gateway.canonical_writer_release_contract import (
    CANARY_WRITER_MAIN_CAPABILITY_CONTRACT,
    DEFAULT_RELEASE_BASE,
    GATEWAY_UNIT_NAME,
    PHASE_B_READINESS_UNIT_NAME,
    RELEASE_MANIFEST_NAME,
    RELEASE_SCHEMA,
    WRITER_UNIT_NAME,
)
from gateway.full_canary_discord_edge_bootstrap import (
    DEFAULT_EDGE_READINESS_PATH,
    EDGE_READINESS_SCHEMA,
)


FULL_CANARY_PLAN_SCHEMA = "muncho-full-canary-runtime-plan.v1"
FULL_CANARY_BUNDLE_SCHEMA = "muncho-full-canary-systemd-bundle.v1"
FULL_CANARY_PREFLIGHT_SCHEMA = "muncho-full-canary-preflight.v1"
FULL_CANARY_RECEIPT_SCHEMA = "muncho-full-canary-runtime-receipt.v1"
FULL_CANARY_APPROVAL_SCHEMA = "muncho-full-canary-owner-approval.v1"
COLLECTOR_READINESS_SCHEMA = "muncho-full-canary-collector-readiness.v1"
COLLECTOR_IDENTITY_SCHEMA = "muncho-full-canary-collector-identity.v1"
PLUGIN_READINESS_SCHEMA = "muncho-full-canary-plugin-readiness.v1"
PLUGIN_FRAME_SCHEMA = "muncho-canary-evidence-frame.v1"

EDGE_UNIT_NAME = "muncho-discord-egress.service"
EDGE_MODULE = "gateway.full_canary_discord_edge_bootstrap"
FULL_GATEWAY_MODULE = "gateway.run"
E2E_VERIFIER_MODULE = "gateway.canonical_full_canary_e2e"
CANARY_OBSERVER_PLUGIN = "muncho_canary_evidence"
FULL_CANARY_START_ORDER = (
    EDGE_UNIT_NAME,
    WRITER_UNIT_NAME,
    GATEWAY_UNIT_NAME,
)
FULL_CANARY_STOP_ORDER = (
    GATEWAY_UNIT_NAME,
    WRITER_UNIT_NAME,
    EDGE_UNIT_NAME,
    PHASE_B_READINESS_UNIT_NAME,
)

DEFAULT_PLAN_PATH = Path("/etc/muncho/full-canary/runtime-plan.json")
DEFAULT_APPROVAL_PATH = Path("/etc/muncho/full-canary/owner-approval.json")
DEFAULT_STAGED_PLAN_PATH = Path(
    "/etc/muncho/full-canary/staged/runtime-plan.json"
)
DEFAULT_WRITER_CONFIG_SOURCE = Path(
    "/etc/muncho/full-canary/staged/writer.json"
)
DEFAULT_GATEWAY_CONFIG_SOURCE = Path(
    "/etc/muncho/full-canary/staged/gateway.yaml"
)
DEFAULT_EDGE_CONFIG_SOURCE = Path(
    "/etc/muncho/full-canary/staged/discord-edge.json"
)
DEFAULT_WRITER_CONFIG = Path("/etc/muncho-canonical-writer/writer.json")
DEFAULT_GATEWAY_PROFILE_HOME = Path("/var/lib/hermes-gateway/.hermes")
DEFAULT_GATEWAY_CONFIG = DEFAULT_GATEWAY_PROFILE_HOME / "config.yaml"
DEFAULT_GATEWAY_USER_PLUGIN_ROOT = DEFAULT_GATEWAY_PROFILE_HOME / "plugins"
DEFAULT_EDGE_CONFIG = Path("/etc/muncho/discord-edge.json")
DEFAULT_WRITER_UNIT_PATH = Path("/etc/systemd/system") / WRITER_UNIT_NAME
DEFAULT_GATEWAY_UNIT_PATH = Path("/etc/systemd/system") / GATEWAY_UNIT_NAME
DEFAULT_EDGE_UNIT_PATH = Path("/etc/systemd/system") / EDGE_UNIT_NAME
DEFAULT_PHASE_B_READINESS_UNIT_PATH = (
    Path("/etc/systemd/system") / PHASE_B_READINESS_UNIT_NAME
)
DEFAULT_TMPFILES_PATH = Path("/etc/tmpfiles.d/muncho-full-canary.conf")
DEFAULT_EVIDENCE_ROOT = Path("/var/lib/muncho-full-canary")
DEFAULT_OBSERVER_CONFIG = Path("/etc/muncho/full-canary/observer.json")
DEFAULT_OBSERVER_CONFIG_SOURCE = Path(
    "/etc/muncho/full-canary/staged/observer.json"
)
DEFAULT_E2E_FIXTURE = Path("/etc/muncho/full-canary/fixture.json")
DEFAULT_HOST_IDENTITY_RECEIPT = Path(
    "/etc/muncho/full-canary/host-identity.json"
)
DEFAULT_COLLECTOR_RUNTIME = Path("/run/muncho-full-canary")
DEFAULT_COLLECTOR_SOCKET = DEFAULT_COLLECTOR_RUNTIME / "collector.sock"
DEFAULT_COLLECTOR_READINESS_PATH = Path(
    "/run/muncho-full-canary/collector-readiness.json"
)
DEFAULT_PLUGIN_READINESS_PATH = Path(
    "/run/muncho-full-canary/plugin-readiness.json"
)
DEFAULT_LOCK_PATH = Path("/run/muncho-full-canary.lock")
DEFAULT_GATEWAY_HOME = Path("/var/lib/hermes-gateway")
DEFAULT_GATEWAY_RUNTIME = Path("/run/hermes-cloud-gateway")
DEFAULT_DISABLED_MANAGED_SCOPE = (
    DEFAULT_COLLECTOR_RUNTIME / "managed-scope-disabled"
)
DEFAULT_GATEWAY_LOGS = Path("/var/log/hermes-gateway")
DEFAULT_GATEWAY_WORKSPACE = Path("/var/lib/hermes-gateway/workspace")
DEFAULT_GATEWAY_AUTH_STORE = DEFAULT_GATEWAY_PROFILE_HOME / "auth.json"
DEFAULT_GATEWAY_CA_BUNDLE = Path("/etc/ssl/certs/ca-certificates.crt")
DEFAULT_EDGE_RUNTIME = Path("/run/muncho-discord-egress")
DEFAULT_EDGE_SOCKET = DEFAULT_EDGE_RUNTIME / "edge.sock"
DEFAULT_EDGE_STATE = Path("/var/lib/muncho-discord-egress")
DEFAULT_EDGE_TOKEN_DIRECTORY = Path("/etc/muncho/discord-edge-credentials")
DEFAULT_EDGE_TOKEN_PATH = DEFAULT_EDGE_TOKEN_DIRECTORY / "bot-token"
DEFAULT_EDGE_RECEIPT_PRIVATE_KEY = Path(
    "/etc/muncho/keys/discord-edge-receipt-private.pem"
)
DEFAULT_WRITER_CAPABILITY_PRIVATE_KEY = Path(
    "/etc/muncho/keys/writer-capability-private.pem"
)
DEFAULT_WRITER_CAPABILITY_PUBLIC_KEY = Path(
    "/etc/muncho/keys/writer-capability-public.pem"
)
DEFAULT_API_SERVER_CONTROL_KEY = Path(
    "/etc/muncho/keys/api-server-control.key"
)
API_SERVER_CREDENTIAL_NAME = "api-server.key"

SYSTEMCTL = "/usr/bin/systemctl"
SYSTEMD_ANALYZE = "/usr/bin/systemd-analyze"
SYSTEMD_TMPFILES = "/usr/bin/systemd-tmpfiles"
SETPRIV = "/usr/bin/setpriv"

# The isolated gateway is deliberately configured from one sealed YAML file.
# These names can otherwise replace credentials, endpoints, model/tool policy,
# prompt content, plugin discovery, or the model-owned task loop after that
# file has passed preflight.  Keep the unit-time list explicit so systemd
# removes manager-provided values, and re-check the effective in-process names
# at readiness because dotenv/plugin startup happens after exec(2).
_GATEWAY_KANBAN_ENVIRONMENT_NAMES = (
    "HERMES_KANBAN_ATTACHMENTS_ROOT",
    "HERMES_KANBAN_BOARD",
    "HERMES_KANBAN_BRANCH",
    "HERMES_KANBAN_BUSY_TIMEOUT_MS",
    "HERMES_KANBAN_CLAIM_LOCK",
    "HERMES_KANBAN_CLAIM_TTL_SECONDS",
    "HERMES_KANBAN_CRASH_GRACE_SECONDS",
    "HERMES_KANBAN_DB",
    "HERMES_KANBAN_DISPATCH_IN_GATEWAY",
    "HERMES_KANBAN_GOAL_MAX_TURNS",
    "HERMES_KANBAN_GOAL_MODE",
    "HERMES_KANBAN_HOME",
    "HERMES_KANBAN_LOGS_ROOT",
    "HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS",
    "HERMES_KANBAN_ROOT",
    "HERMES_KANBAN_RUN_ID",
    "HERMES_KANBAN_SPECIFY_MAX_TOKENS",
    "HERMES_KANBAN_TASK",
    "HERMES_KANBAN_WORKSPACE",
    "HERMES_KANBAN_WORKSPACES_ROOT",
)
GATEWAY_FORBIDDEN_EFFECTIVE_ENVIRONMENT_NAMES = frozenset(
    {
        "CODEX_HOME",
        "ALL_PROXY",
        "CURL_CA_BUNDLE",
        "DISCORD_BOT_TOKEN",
        "HERMES_ACCEPT_HOOKS",
        "HERMES_AGENT_TIMEOUT",
        "HERMES_AGENT_TIMEOUT_WARNING",
        "HERMES_API_CALL_STALE_TIMEOUT",
        "HERMES_BUNDLED_PLUGINS",
        "HERMES_CODEX_BASE_URL",
        "HERMES_CODEX_TTFB_STRICT",
        "HERMES_CONCURRENT_TOOL_TIMEOUT_S",
        "HERMES_ENABLE_PROJECT_PLUGINS",
        "HERMES_ENVIRONMENT_HINT",
        "HERMES_EPHEMERAL_SYSTEM_PROMPT",
        "HERMES_FILE_MUTATION_VERIFIER",
        "HERMES_GATEWAY_BUSY_STEER_ACK_ENABLED",
        "HERMES_GATEWAY_PLATFORM_CONNECT_TIMEOUT",
        "HERMES_INFERENCE_MODEL",
        "HERMES_INFERENCE_PROVIDER",
        "HERMES_LANGUAGE",
        "HERMES_MAX_TOKENS",
        "HERMES_MODEL",
        "HERMES_PLATFORM",
        "HERMES_PREFILL_MESSAGES_FILE",
        "HERMES_PROFILE",
        "HERMES_REDACT_SECRETS",
        "HERMES_SAFE_MODE",
        "HERMES_TOOL_PROGRESS_MODE",
        "HERMES_TUI_PROVIDER",
        "HERMES_TURN_COMPLETION_EXPLAINER",
        "HERMES_WRITE_SAFE_ROOT",
        "HERMES_YOLO_MODE",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENROUTER_API_KEY",
        "OP_SERVICE_ACCOUNT_TOKEN",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "REQUESTS_CA_BUNDLE",
        "all_proxy",
        "http_proxy",
        "https_proxy",
        "no_proxy",
    }
    | set(_GATEWAY_KANBAN_ENVIRONMENT_NAMES)
)
_GATEWAY_SYSTEMD_UNSET_ENVIRONMENT_NAMES = tuple(
    sorted(
        GATEWAY_FORBIDDEN_EFFECTIVE_ENVIRONMENT_NAMES
        | {
            # Python isolation and config-derived values are also cleared at
            # exec time.  Some are populated later by reviewed gateway code,
            # so they are intentionally not all forbidden in the final
            # readiness inventory.
            "HERMES_EXEC_ASK",
            "HERMES_MAX_ITERATIONS",
            "LD_AUDIT",
            "LD_LIBRARY_PATH",
            "LD_PRELOAD",
            "PYTHONHOME",
            "PYTHONPATH",
            "PYTHONSTARTUP",
            "PYTHONUSERBASE",
        }
    )
)
_GATEWAY_UNSET_ENVIRONMENT_DIRECTIVE = (
    "UnsetEnvironment=" + " ".join(_GATEWAY_SYSTEMD_UNSET_ENVIRONMENT_NAMES)
)

_GATEWAY_REQUIRED_EFFECTIVE_ENVIRONMENT_NAMES = frozenset(
    {
        "CREDENTIALS_DIRECTORY",
        "HERMES_CONFIG",
        "HERMES_EXEC_ASK",
        "HERMES_HOME",
        "HERMES_MANAGED_DIR",
        "HERMES_MAX_ITERATIONS",
        "HERMES_QUIET",
        "HOME",
        "LANG",
        "LC_ALL",
        "LOGNAME",
        "NOTIFY_SOCKET",
        "PATH",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONNOUSERSITE",
        "SHELL",
        "SSL_CERT_FILE",
        "TERMINAL_CWD",
        "TZ",
        "USER",
        "_HERMES_GATEWAY",
    }
)
GATEWAY_ALLOWED_EFFECTIVE_ENVIRONMENT_NAMES = frozenset(
    _GATEWAY_REQUIRED_EFFECTIVE_ENVIRONMENT_NAMES
    | {
        "INVOCATION_ID",
        "JOURNAL_STREAM",
        "MEMORY_PRESSURE_WATCH",
        "MEMORY_PRESSURE_WRITE",
        "SYSTEMD_EXEC_PID",
        "SYSTEMD_NSS_DYNAMIC_BYPASS",
    }
)

_GATEWAY_SEALED_EMPTY_ENVIRONMENT_FILES = (
    DEFAULT_GATEWAY_HOME / ".env",
    DEFAULT_GATEWAY_HOME / ".op.env",
    DEFAULT_GATEWAY_PROFILE_HOME / ".env",
    DEFAULT_GATEWAY_PROFILE_HOME / ".op.env",
)
_GATEWAY_INACCESSIBLE_SEMANTIC_FILES = (
    DEFAULT_GATEWAY_PROFILE_HOME / "SOUL.md",
    DEFAULT_GATEWAY_PROFILE_HOME / "processes.json",
    DEFAULT_GATEWAY_HOME / ".hermes.md",
    DEFAULT_GATEWAY_HOME / "HERMES.md",
    DEFAULT_GATEWAY_HOME / "AGENTS.md",
    DEFAULT_GATEWAY_HOME / "agents.md",
    DEFAULT_GATEWAY_HOME / "CLAUDE.md",
    DEFAULT_GATEWAY_HOME / "claude.md",
    DEFAULT_GATEWAY_HOME / ".cursorrules",
)
_GATEWAY_INACCESSIBLE_SEMANTIC_DIRECTORIES = (
    DEFAULT_GATEWAY_USER_PLUGIN_ROOT,
    DEFAULT_GATEWAY_PROFILE_HOME / "hooks",
    DEFAULT_GATEWAY_PROFILE_HOME / "cron",
    DEFAULT_GATEWAY_PROFILE_HOME / "scripts",
    DEFAULT_GATEWAY_PROFILE_HOME / "memories",
    DEFAULT_GATEWAY_PROFILE_HOME / "skills",
    DEFAULT_GATEWAY_HOME / ".cursor",
)

_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTITY_RE = re.compile(r"^[a-z_][a-z0-9_-]{0,63}$")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_MAX_JSON_BYTES = 8 * 1024 * 1024
_MAX_CONFIG_BYTES = 2 * 1024 * 1024
_MAX_COMMAND_OUTPUT_BYTES = 4 * 1024 * 1024
_MAX_HOST_IDENTITY_RECEIPT_BYTES = 16 * 1024
_COMMAND_TIMEOUT_SECONDS = 180
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

def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8", errors="strict")
    except (TypeError, ValueError) as exc:
        raise ValueError("full-canary value is not canonical JSON") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be lowercase SHA-256")
    return value


def _revision(value: Any) -> str:
    if not isinstance(value, str) or _REVISION_RE.fullmatch(value) is None:
        raise ValueError("full-canary revision must be exact lowercase Git SHA")
    return value


def _absolute_path(value: Any, label: str) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError(f"{label} must be an absolute path")
    raw = os.fspath(value)
    path = Path(raw)
    if (
        not path.is_absolute()
        or str(path) != raw
        or ".." in path.parts
        or _CONTROL_RE.search(raw) is not None
    ):
        raise ValueError(f"{label} must be an absolute normalized path")
    return path


def _strict_mapping(
    value: Any,
    *,
    fields: frozenset[str],
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{label} fields are not exact")
    return value


def _positive_id(value: Any, label: str) -> int:
    if type(value) is not int or not 0 < value < 1 << 31:
        raise ValueError(f"{label} must be an exact positive numeric identity")
    return value


def _identity(value: Any, label: str) -> str:
    if not isinstance(value, str) or _IDENTITY_RE.fullmatch(value) is None:
        raise ValueError(f"{label} is invalid")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("full-canary JSON contains duplicate keys")
        result[key] = value
    return result


def _decode_json(raw: bytes, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-JSON constant {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is not strict JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def validate_dedicated_canary_host(
    plan: "FullCanaryPlan",
    *,
    metadata_reader: Callable[[str], bytes | str] | None = None,
    local_identity_reader: Callable[[str], bytes | str] | None = None,
) -> Mapping[str, Any]:
    """Revalidate the sealed receipt against both metadata and this boot."""

    try:
        artifact = plan.artifacts["host_identity_receipt"]
    except (AttributeError, KeyError) as exc:
        raise RuntimeError("full-canary host identity artifact is absent") from exc
    if (
        not isinstance(artifact, ExactArtifact)
        or artifact.source_path != DEFAULT_HOST_IDENTITY_RECEIPT
        or artifact.target_path != DEFAULT_HOST_IDENTITY_RECEIPT
        or artifact.mode != 0o400
        or artifact.uid != 0
        or artifact.gid != 0
        or artifact.maximum_bytes != _MAX_HOST_IDENTITY_RECEIPT_BYTES
    ):
        raise RuntimeError("full-canary host identity artifact is not pinned")
    raw = _validate_artifact_source(artifact, label="host_identity_receipt")
    receipt = _decode_json(raw, label="dedicated canary host identity receipt")
    observed_fields = frozenset(
        {
            "project_id",
            "project_number",
            "zone",
            "instance_name",
            "instance_id",
            "service_account_email",
            "gce_identity_sha256",
            "machine_id_sha256",
            "hostname_sha256",
            "host_identity_sha256",
            "boot_id_sha256",
        }
    )
    expected_fields = observed_fields | {
        "schema",
        "collector_authority",
        "observed_at_unix",
        "receipt_sha256",
    }
    if set(receipt) != expected_fields:
        raise RuntimeError("dedicated canary host receipt fields are not exact")
    unsigned = {
        name: copy.deepcopy(value)
        for name, value in receipt.items()
        if name != "receipt_sha256"
    }
    if (
        receipt["schema"] != FULL_CANARY_HOST_IDENTITY_SCHEMA
        or receipt["collector_authority"]
        != "trusted_root_read_only_host_collector"
        or type(receipt["observed_at_unix"]) is not int
        or receipt["observed_at_unix"] < 0
        or receipt["receipt_sha256"] != _sha256_json(unsigned)
        or any(
            _SHA256_RE.fullmatch(str(receipt[name])) is None
            for name in observed_fields
            if name.endswith("_sha256")
        )
    ):
        raise RuntimeError("dedicated canary host receipt is invalid")
    live = _observe_dedicated_canary_host(
        metadata_reader=metadata_reader,
        local_identity_reader=local_identity_reader,
    )
    if any(receipt[name] != live[name] for name in observed_fields):
        raise RuntimeError("dedicated canary host receipt is stale or mismatched")
    return {
        "artifact_sha256": artifact.sha256,
        "receipt_sha256": receipt["receipt_sha256"],
        "gce_identity_sha256": receipt["gce_identity_sha256"],
        "host_identity_sha256": receipt["host_identity_sha256"],
        "boot_id_sha256": receipt["boot_id_sha256"],
    }


@dataclass(frozen=True)
class FullCanaryIdentities:
    writer_user: str
    writer_group: str
    writer_uid: int
    writer_gid: int
    gateway_user: str
    gateway_group: str
    gateway_uid: int
    gateway_gid: int
    socket_client_group: str
    socket_client_gid: int
    edge_user: str
    edge_group: str
    edge_uid: int
    edge_gid: int

    @classmethod
    def from_mapping(cls, value: Any) -> "FullCanaryIdentities":
        fields = frozenset(cls.__dataclass_fields__)
        raw = _strict_mapping(value, fields=fields, label="full-canary identities")
        names = {
            name: _identity(raw[name], name)
            for name in fields
            if name.endswith("_user") or name.endswith("_group")
        }
        numbers = {
            name: _positive_id(raw[name], name)
            for name in fields
            if name.endswith("_uid") or name.endswith("_gid")
        }
        result = cls(**names, **numbers)
        if (
            len({result.writer_uid, result.gateway_uid, result.edge_uid}) != 3
            or len(
                {
                    result.writer_gid,
                    result.gateway_gid,
                    result.socket_client_gid,
                    result.edge_gid,
                }
            )
            != 4
            or result.edge_user != "muncho-discord-egress"
            or result.edge_group != "muncho-discord-egress"
        ):
            raise ValueError("full-canary service identities are not isolated")
        return result

    def to_mapping(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class ExactArtifact:
    source_path: Path
    target_path: Path
    sha256: str
    mode: int
    uid: int
    gid: int
    maximum_bytes: int = _MAX_CONFIG_BYTES

    @classmethod
    def from_mapping(cls, value: Any, *, label: str) -> "ExactArtifact":
        raw = _strict_mapping(
            value,
            fields=frozenset(
                {
                    "source_path",
                    "target_path",
                    "sha256",
                    "mode",
                    "uid",
                    "gid",
                    "maximum_bytes",
                }
            ),
            label=f"{label} artifact",
        )
        mode = raw["mode"]
        if not isinstance(mode, str) or re.fullmatch(r"0[0-7]{3}", mode) is None:
            raise ValueError(f"{label} artifact mode is invalid")
        maximum = raw["maximum_bytes"]
        if type(maximum) is not int or not 0 < maximum <= _MAX_JSON_BYTES:
            raise ValueError(f"{label} artifact maximum is invalid")
        uid = raw["uid"]
        gid = raw["gid"]
        if type(uid) is not int or type(gid) is not int or uid < 0 or gid < 0:
            raise ValueError(f"{label} artifact ownership is invalid")
        return cls(
            source_path=_absolute_path(raw["source_path"], f"{label} source"),
            target_path=_absolute_path(raw["target_path"], f"{label} target"),
            sha256=_digest(raw["sha256"], f"{label} artifact digest"),
            mode=int(mode, 8),
            uid=uid,
            gid=gid,
            maximum_bytes=maximum,
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "source_path": str(self.source_path),
            "target_path": str(self.target_path),
            "sha256": self.sha256,
            "mode": f"{self.mode:04o}",
            "uid": self.uid,
            "gid": self.gid,
            "maximum_bytes": self.maximum_bytes,
        }


@dataclass(frozen=True)
class FullCanarySystemdBundle:
    edge_service: str
    writer_service: str
    gateway_service: str
    tmpfiles: str
    sha256: str
    schema: str = FULL_CANARY_BUNDLE_SCHEMA

    @classmethod
    def from_mapping(cls, value: Any) -> "FullCanarySystemdBundle":
        raw = _strict_mapping(
            value,
            fields=frozenset(
                {
                    "schema",
                    "edge_service",
                    "writer_service",
                    "gateway_service",
                    "tmpfiles",
                    "sha256",
                }
            ),
            label="full-canary systemd bundle",
        )
        if raw["schema"] != FULL_CANARY_BUNDLE_SCHEMA:
            raise ValueError("full-canary bundle schema is invalid")
        for name in ("edge_service", "writer_service", "gateway_service", "tmpfiles"):
            text = raw[name]
            if (
                not isinstance(text, str)
                or not text.endswith("\n")
                or "\x00" in text
                or len(text.encode("utf-8")) > 256 * 1024
            ):
                raise ValueError(f"full-canary {name} is invalid")
        unsigned = {name: copy.deepcopy(item) for name, item in raw.items() if name != "sha256"}
        digest = _digest(raw["sha256"], "full-canary bundle digest")
        if _sha256_json(unsigned) != digest:
            raise ValueError("full-canary bundle digest drifted")
        forbidden = re.compile(r"(?im)^(?:EnvironmentFile|PassEnvironment)=")
        if any(forbidden.search(raw[name]) for name in ("edge_service", "writer_service", "gateway_service")):
            raise ValueError("full-canary unit injects process environment")
        load_credential = re.compile(r"(?m)^LoadCredential=(.+)$")
        expected_gateway_credential = (
            f"{API_SERVER_CREDENTIAL_NAME}:{DEFAULT_API_SERVER_CONTROL_KEY}"
        )
        if (
            load_credential.findall(raw["edge_service"])
            or load_credential.findall(raw["writer_service"])
            or load_credential.findall(raw["gateway_service"])
            != [expected_gateway_credential]
        ):
            raise ValueError("full-canary unit credential boundary is not exact")
        gateway_environment = re.compile(r"(?m)^Environment=(.+)$").findall(
            raw["gateway_service"]
        )
        gateway_unset_environment = re.compile(
            r"(?m)^UnsetEnvironment=(.*)$"
        ).findall(raw["gateway_service"])
        gateway_groups = re.compile(r"(?m)^Group=(.+)$").findall(
            raw["gateway_service"]
        )
        gateway_group = gateway_groups[0] if len(gateway_groups) == 1 else ""
        managed_parent_tmpfiles = (
            f"d {DEFAULT_COLLECTOR_RUNTIME} 0750 root {gateway_group} - -\n"
        )
        if (
            _IDENTITY_RE.fullmatch(gateway_group) is None
            or [
                item
                for item in gateway_environment
                if item.startswith("HERMES_HOME=")
            ]
            != [f"HERMES_HOME={DEFAULT_GATEWAY_PROFILE_HOME}"]
            or [
                item
                for item in gateway_environment
                if item.startswith("HERMES_MANAGED_DIR=")
            ]
            != [f"HERMES_MANAGED_DIR={DEFAULT_DISABLED_MANAGED_SCOPE}"]
            or [
                item
                for item in gateway_environment
                if item.startswith("SSL_CERT_FILE=")
            ]
            != [f"SSL_CERT_FILE={DEFAULT_GATEWAY_CA_BUNDLE}"]
            or raw["gateway_service"].count(
                f"AssertPathExists={DEFAULT_GATEWAY_CA_BUNDLE}\n"
            )
            != 1
            or raw["gateway_service"].count(
                f"ReadOnlyPaths={DEFAULT_GATEWAY_CA_BUNDLE}\n"
            )
            != 1
            or raw["gateway_service"].count(
                "InaccessiblePaths=-/etc/hermes\n"
            )
            != 1
            or raw["gateway_service"].count(
                f"InaccessiblePaths={DEFAULT_DISABLED_MANAGED_SCOPE}\n"
            )
            != 0
            or gateway_unset_environment
            != [" ".join(_GATEWAY_SYSTEMD_UNSET_ENVIRONMENT_NAMES)]
            or str(DEFAULT_DISABLED_MANAGED_SCOPE) in raw["tmpfiles"]
            or raw["tmpfiles"].count(managed_parent_tmpfiles)
            != 1
            or any(
                raw["gateway_service"].count(
                    f"ReadOnlyPaths={path}\n"
                )
                != 1
                or raw["gateway_service"].count(
                    f"InaccessiblePaths={path}\n"
                )
                != 0
                or raw["tmpfiles"].count(
                    f"f+ {path} 0444 root root - -\n"
                )
                != 1
                for path in _GATEWAY_SEALED_EMPTY_ENVIRONMENT_FILES
            )
            or any(
                raw["gateway_service"].count(
                    f"InaccessiblePaths={path}\n"
                )
                != 1
                or raw["tmpfiles"].count(
                    f"f {path} 0000 root root - -\n"
                )
                != 1
                for path in _GATEWAY_INACCESSIBLE_SEMANTIC_FILES
            )
            or any(
                raw["gateway_service"].count(
                    f"InaccessiblePaths={path}\n"
                )
                != 1
                or raw["tmpfiles"].count(
                    f"d {path} 0000 root root - -\n"
                )
                != 1
                for path in _GATEWAY_INACCESSIBLE_SEMANTIC_DIRECTORIES
            )
        ):
            raise ValueError(
                "full-canary managed configuration/environment boundary is not exact"
            )
        exec_start_pre = re.compile(r"(?m)^ExecStartPre=(.+)$")
        exec_stop_post = re.compile(r"(?m)^ExecStopPost=(.+)$")
        writer_pre_commands = exec_start_pre.findall(raw["writer_service"])
        writer_stop_commands = exec_stop_post.findall(raw["writer_service"])
        capability_bounding_set = re.compile(
            r"(?m)^CapabilityBoundingSet=(.*)$"
        )
        ambient_capabilities = re.compile(
            r"(?m)^AmbientCapabilities=(.*)$"
        )
        writer_users = re.compile(r"(?m)^User=(.+)$").findall(
            raw["writer_service"]
        )
        writer_exec_start = re.compile(r"(?m)^ExecStart=(.+)$").findall(
            raw["writer_service"]
        )
        expected_writer_exec_start = re.compile(
            r"/opt/muncho-canary-releases/[0-9a-f]{40}/venv/bin/python "
            r"-B -I -m gateway\.canonical_writer_bootstrap --config "
            + re.escape(str(DEFAULT_WRITER_CONFIG))
        )
        if (
            exec_start_pre.findall(raw["edge_service"])
            or exec_start_pre.findall(raw["gateway_service"])
            or exec_stop_post.findall(raw["edge_service"])
            or exec_stop_post.findall(raw["gateway_service"])
            or writer_pre_commands
            or writer_stop_commands
            or re.search(
                r"(?m)^Exec(?:Start|StartPre|StopPost)=\+",
                raw["writer_service"],
            )
            is not None
            or str(DEFAULT_WRITER_CONFIG_SOURCE) in raw["writer_service"]
            or capability_bounding_set.findall(raw["writer_service"])
            != [""]
            or capability_bounding_set.findall(raw["edge_service"]) != [""]
            or capability_bounding_set.findall(raw["gateway_service"])
            != [""]
            or ambient_capabilities.findall(raw["writer_service"]) != [""]
            or ambient_capabilities.findall(raw["edge_service"]) != [""]
            or ambient_capabilities.findall(raw["gateway_service"]) != [""]
            or raw["writer_service"].count("NoNewPrivileges=yes\n") != 1
            or len(writer_users) != 1
            or _IDENTITY_RE.fullmatch(writer_users[0]) is None
            or writer_users[0] == "root"
            or len(writer_exec_start) != 1
            or expected_writer_exec_start.fullmatch(writer_exec_start[0])
            is None
            or raw["writer_service"].count("Restart=no\n") != 1
            or "Restart=on-failure\n" in raw["writer_service"]
            or raw["writer_service"].count("RuntimeMaxSec=900s\n") != 1
        ):
            raise ValueError(
                "full-canary privileged writer boundary is not exact"
            )
        if re.search(r"(?im)^ExecStart=.*(?:systemctl|\.timer|\benable\b)", "\n".join(raw.values())):
            raise ValueError("full-canary unit contains forbidden lifecycle authority")
        return cls(
            edge_service=raw["edge_service"],
            writer_service=raw["writer_service"],
            gateway_service=raw["gateway_service"],
            tmpfiles=raw["tmpfiles"],
            sha256=digest,
        )

    def unsigned_mapping(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "edge_service": self.edge_service,
            "writer_service": self.writer_service,
            "gateway_service": self.gateway_service,
            "tmpfiles": self.tmpfiles,
        }

    def to_mapping(self) -> dict[str, Any]:
        return {**self.unsigned_mapping(), "sha256": self.sha256}


def _fixed_environment(*, user: str, home: Path) -> list[str]:
    return [
        f"Environment=HOME={home}",
        "Environment=LANG=C.UTF-8",
        "Environment=LC_ALL=C.UTF-8",
        f"Environment=LOGNAME={user}",
        "Environment=PATH=/usr/bin:/bin",
        "Environment=PYTHONDONTWRITEBYTECODE=1",
        "Environment=PYTHONNOUSERSITE=1",
        "Environment=SHELL=/usr/sbin/nologin",
        "Environment=TZ=UTC",
        f"Environment=USER={user}",
    ]


def _common_hardening() -> list[str]:
    return [
        "NoNewPrivileges=yes",
        "CapabilityBoundingSet=",
        "AmbientCapabilities=",
        "LockPersonality=yes",
        "MemoryDenyWriteExecute=yes",
        "PrivateDevices=yes",
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
        "RestrictNamespaces=yes",
        "RestrictRealtime=yes",
        "RestrictSUIDSGID=yes",
        "SystemCallArchitectures=native",
        "UMask=0077",
    ]


def _full_writer_service(
    writer_only: str,
    *,
    interpreter: Path,
    writer_user: str,
    writer_group: str,
) -> str:
    """Transition reviewed credential-free writer bytes to the full canary."""

    marker = "Wants=network-online.target\n"
    release_root = interpreter.parent.parent.parent
    expected_main = (
        f"ExecStart={interpreter} -B -I -m "
        "gateway.canonical_writer_bootstrap "
        f"--config {DEFAULT_WRITER_CONFIG}\n"
    )
    if (
        writer_only.count(marker) != 1
        or writer_only.count("[Install]\nWantedBy=multi-user.target\n") != 1
        or writer_only.count("\n[Unit]\n") > 1
        or writer_only.count("[Unit]\n") != 1
        or writer_only.count("\n[Service]\n") != 1
        or writer_only.count(f"User={writer_user}\n") != 1
        or writer_only.count(f"Group={writer_group}\n") != 1
        or writer_only.count(f"WorkingDirectory={release_root}\n") != 1
        or writer_only.count(expected_main) != 1
        or writer_only.count("Restart=on-failure\nRestartSec=5s\n") != 1
        or writer_only.count("NoNewPrivileges=yes\n") != 1
        or writer_only.count("CapabilityBoundingSet=\n") != 1
        or writer_only.count("AmbientCapabilities=\n") != 1
        or re.search(
            r"(?m)^(?:LoadCredential|ExecStartPre)=",
            writer_only,
        )
        is not None
    ):
        raise ValueError("writer-only unit cannot be extended exactly")
    result = writer_only.replace(
        marker,
        marker
        + f"BindsTo={EDGE_UNIT_NAME}\n"
        + f"After={EDGE_UNIT_NAME}\n",
        1,
    )
    result = result.replace(
        "Description=Muncho privileged Canonical Writer (isolated canary)",
        "Description=Muncho privileged Canonical Writer (full isolated canary)",
        1,
    )
    return result.replace(
        "Restart=on-failure\nRestartSec=5s\n",
        "Restart=no\nRestartSec=5s\nRuntimeMaxSec=900s\n",
        1,
    )


def render_full_canary_systemd_bundle(
    *,
    revision: str,
    artifact_sha256: str,
    interpreter: Path,
    writer_only_service: str,
    identities: FullCanaryIdentities,
    database_ip_allow: str,
) -> FullCanarySystemdBundle:
    """Render exact hardened bytes; never install, start, or enable them."""

    revision = _revision(revision)
    artifact_sha256 = _digest(artifact_sha256, "release artifact digest")
    release_root = DEFAULT_RELEASE_BASE / revision
    interpreter = _absolute_path(interpreter, "sealed interpreter")
    if interpreter != release_root / "venv/bin/python":
        raise ValueError("full-canary interpreter is not release-bound")
    identities = FullCanaryIdentities.from_mapping(identities.to_mapping())
    if not isinstance(database_ip_allow, str) or re.fullmatch(
        r"(?:[0-9]{1,3}\.){3}[0-9]{1,3}/32", database_ip_allow
    ) is None:
        raise ValueError("full-canary database allow-list is not one IPv4 host")

    edge_lines = [
        "# Generated from a digest-bound full-canary plan; do not edit.",
        f"# ArtifactSHA256={artifact_sha256}",
        "[Unit]",
        "Description=Muncho privileged Discord public-egress edge (full canary)",
        "After=network-online.target",
        "Wants=network-online.target",
        f"Before={WRITER_UNIT_NAME} {GATEWAY_UNIT_NAME}",
        f"AssertPathExists={DEFAULT_EDGE_CONFIG}",
        "",
        "[Service]",
        "Type=notify",
        "NotifyAccess=main",
        f"User={identities.edge_user}",
        f"Group={identities.edge_group}",
        f"WorkingDirectory={release_root}",
        (
            f"ExecStart={interpreter} -B -I -m {EDGE_MODULE} "
            f"--config {DEFAULT_EDGE_CONFIG}"
        ),
        "Restart=no",
        "RestartSec=5s",
        "RuntimeMaxSec=900s",
        "TimeoutStartSec=60s",
        "TimeoutStopSec=30s",
        "KillMode=mixed",
        "LimitCORE=0",
        *_fixed_environment(user=identities.edge_user, home=DEFAULT_EDGE_STATE),
        *_common_hardening(),
        "RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6",
        "IPAddressDeny=169.254.169.254/32",
        f"BindReadOnlyPaths={release_root}",
        f"ReadOnlyPaths={DEFAULT_EDGE_CONFIG}",
        f"ReadOnlyPaths={DEFAULT_WRITER_CAPABILITY_PUBLIC_KEY}",
        f"ReadOnlyPaths={DEFAULT_EDGE_RECEIPT_PRIVATE_KEY}",
        f"ReadOnlyPaths={DEFAULT_EDGE_TOKEN_PATH}",
        f"ReadWritePaths={DEFAULT_EDGE_RUNTIME}",
        f"ReadWritePaths={DEFAULT_EDGE_STATE}",
        "",
        "[Install]",
        "WantedBy=multi-user.target",
    ]
    gateway_lines = [
        "# Generated from a digest-bound full-canary plan; do not edit.",
        f"# ArtifactSHA256={artifact_sha256}",
        "# DiscordCredentialInGateway=false",
        "[Unit]",
        "Description=Muncho full model gateway (isolated canary)",
        f"BindsTo={WRITER_UNIT_NAME} {EDGE_UNIT_NAME}",
        f"After={WRITER_UNIT_NAME} {EDGE_UNIT_NAME}",
        f"AssertPathIsDirectory={DEFAULT_GATEWAY_HOME}",
        f"AssertPathExists={DEFAULT_GATEWAY_CONFIG}",
        f"AssertPathExists={DEFAULT_OBSERVER_CONFIG}",
        f"AssertPathExists={DEFAULT_GATEWAY_CA_BUNDLE}",
        "",
        "[Service]",
        "Type=notify",
        "NotifyAccess=main",
        f"User={identities.gateway_user}",
        f"Group={identities.gateway_group}",
        (
            f"SupplementaryGroups={identities.socket_client_group} "
            f"{identities.edge_group}"
        ),
        f"WorkingDirectory={release_root}",
        (
            f"ExecStart={interpreter} -B -I -m {FULL_GATEWAY_MODULE} "
            f"--config {DEFAULT_GATEWAY_CONFIG} --require-canonical-writer"
        ),
        "Restart=no",
        "RestartSec=5s",
        "RuntimeMaxSec=900s",
        "TimeoutStartSec=180s",
        "TimeoutStopSec=90s",
        "KillMode=mixed",
        "LimitCORE=0",
        (
            f"LoadCredential={API_SERVER_CREDENTIAL_NAME}:"
            f"{DEFAULT_API_SERVER_CONTROL_KEY}"
        ),
        *_fixed_environment(user=identities.gateway_user, home=DEFAULT_GATEWAY_HOME),
        f"Environment=HERMES_CONFIG={DEFAULT_GATEWAY_CONFIG}",
        f"Environment=HERMES_HOME={DEFAULT_GATEWAY_PROFILE_HOME}",
        f"Environment=HERMES_MANAGED_DIR={DEFAULT_DISABLED_MANAGED_SCOPE}",
        f"Environment=SSL_CERT_FILE={DEFAULT_GATEWAY_CA_BUNDLE}",
        _GATEWAY_UNSET_ENVIRONMENT_DIRECTIVE,
        *_common_hardening(),
        "RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6",
        "IPAddressDeny=169.254.169.254/32",
        f"BindReadOnlyPaths={release_root}",
        f"BindReadOnlyPaths={DEFAULT_GATEWAY_CONFIG}",
        f"ReadOnlyPaths={DEFAULT_GATEWAY_CONFIG}",
        f"ReadOnlyPaths={DEFAULT_OBSERVER_CONFIG}",
        f"ReadOnlyPaths={DEFAULT_E2E_FIXTURE}",
        f"ReadOnlyPaths={DEFAULT_GATEWAY_CA_BUNDLE}",
        f"InaccessiblePaths={DEFAULT_EDGE_TOKEN_DIRECTORY}",
        f"InaccessiblePaths={DEFAULT_EDGE_RECEIPT_PRIVATE_KEY}",
        f"InaccessiblePaths={DEFAULT_WRITER_CAPABILITY_PRIVATE_KEY}",
        f"InaccessiblePaths={DEFAULT_API_SERVER_CONTROL_KEY}",
        *(
            f"ReadOnlyPaths={path}"
            for path in _GATEWAY_SEALED_EMPTY_ENVIRONMENT_FILES
        ),
        *(
            f"InaccessiblePaths={path}"
            for path in _GATEWAY_INACCESSIBLE_SEMANTIC_FILES
        ),
        *(
            f"InaccessiblePaths={path}"
            for path in _GATEWAY_INACCESSIBLE_SEMANTIC_DIRECTORIES
        ),
        f"InaccessiblePaths=-{release_root}/.env",
        "InaccessiblePaths=-/etc/hermes",
        f"ReadWritePaths={DEFAULT_GATEWAY_RUNTIME}",
        f"ReadWritePaths={DEFAULT_GATEWAY_HOME}",
        f"ReadWritePaths={DEFAULT_GATEWAY_LOGS}",
        f"ReadWritePaths={DEFAULT_GATEWAY_WORKSPACE}",
        "",
        "[Install]",
        "WantedBy=multi-user.target",
    ]
    tmpfiles_lines = [
        "# type path mode user group age argument",
        (
            f"d {DEFAULT_COLLECTOR_RUNTIME} 0750 root "
            f"{identities.gateway_group} - -"
        ),
        *(
            f"f+ {path} 0444 root root - -"
            for path in _GATEWAY_SEALED_EMPTY_ENVIRONMENT_FILES
        ),
        *(
            f"f {path} 0000 root root - -"
            for path in _GATEWAY_INACCESSIBLE_SEMANTIC_FILES
        ),
        *(
            f"d {path} 0000 root root - -"
            for path in _GATEWAY_INACCESSIBLE_SEMANTIC_DIRECTORIES
        ),
        (
            f"d {DEFAULT_EDGE_RUNTIME} 0750 {identities.edge_user} "
            f"{identities.edge_group} - -"
        ),
        (
            f"d {DEFAULT_EDGE_STATE} 0700 {identities.edge_user} "
            f"{identities.edge_group} - -"
        ),
        (
            f"d {DEFAULT_GATEWAY_WORKSPACE} 0700 {identities.gateway_user} "
            f"{identities.gateway_group} - -"
        ),
    ]
    unsigned = {
        "schema": FULL_CANARY_BUNDLE_SCHEMA,
        "edge_service": "\n".join(edge_lines) + "\n",
        "writer_service": _full_writer_service(
            writer_only_service,
            interpreter=interpreter,
            writer_user=identities.writer_user,
            writer_group=identities.writer_group,
        ),
        "gateway_service": "\n".join(gateway_lines) + "\n",
        "tmpfiles": "\n".join(tmpfiles_lines) + "\n",
    }
    return FullCanarySystemdBundle(
        edge_service=unsigned["edge_service"],
        writer_service=unsigned["writer_service"],
        gateway_service=unsigned["gateway_service"],
        tmpfiles=unsigned["tmpfiles"],
        sha256=_sha256_json(unsigned),
    )


def _validate_writer_only_receipt(
    value: Any,
    *,
    plan: ActivationPlan,
) -> Mapping[str, Any]:
    fields = frozenset(
        {
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
        }
    )
    raw = _strict_mapping(
        value,
        fields=fields,
        label="writer-only activation receipt",
    )
    unsigned = {name: copy.deepcopy(item) for name, item in raw.items() if name != "receipt_sha256"}
    if (
        raw["schema"] != "muncho-writer-only-activation-receipt.v1"
        or raw["revision"] != plan.revision
        or raw["activation_plan_sha256"] != plan.sha256
        or raw["approved_plan_sha256"] != plan.sha256
        or raw["native_observation_plan_sha256"]
        != plan.digests.native_observation_plan_sha256
        or raw["native_observation_receipt_sha256"]
        != plan.digests.native_observation_receipt_sha256
        or raw["services_stopped"] is not True
        or raw["discord_started"] is not False
        or type(raw["completed_at_unix"]) is not int
        or raw["completed_at_unix"] < 0
        or raw["receipt_sha256"] != _sha256_json(unsigned)
    ):
        raise ValueError("writer-only activation receipt is not exact stopped truth")
    receipt_path = _absolute_path(
        raw["activation_receipt_path"],
        "writer-only activation receipt path",
    )
    expected_path = (
        plan.paths.evidence_root
        / "plans"
        / plan.revision
        / plan.sha256
        / "success/activation.json"
    )
    if receipt_path != expected_path:
        raise ValueError("writer-only activation receipt path is not plan-addressed")
    return copy.deepcopy(dict(raw))


_PHASE_B_READINESS_ANCHOR_FIELDS = frozenset(
    {
        "phase_b_release_revision",
        "phase_b_plan_sha256",
        "phase_b_approval_sha256",
        "phase_b_terminal_receipt_sha256",
        "phase_b_foundation_generation_sha256",
        "phase_b_readiness_receipt_sha256",
        "phase_b_readiness_handoff_file_sha256",
        "phase_b_readiness_sequence",
    }
)


def _validate_phase_b_readiness_anchor(
    value: Any,
    *,
    revision: str,
) -> Mapping[str, Any]:
    raw = _strict_mapping(
        value,
        fields=_PHASE_B_READINESS_ANCHOR_FIELDS,
        label="Phase-B readiness anchor",
    )
    if (
        raw["phase_b_release_revision"] != revision
        or type(raw["phase_b_readiness_sequence"]) is not int
        or raw["phase_b_readiness_sequence"] < 0
    ):
        raise ValueError("Phase-B readiness anchor is not release-bound")
    for name in _PHASE_B_READINESS_ANCHOR_FIELDS - {
        "phase_b_release_revision",
        "phase_b_readiness_sequence",
    }:
        _digest(raw[name], f"Phase-B readiness anchor {name}")
    return copy.deepcopy(dict(raw))


def _release_binding(plan: ActivationPlan) -> Mapping[str, str]:
    snapshot = plan.deployment_manifest["snapshot_template"]
    writer_policy = snapshot["writer_deployment"]["policy"]
    gateway_policy = snapshot["gateway_deployment"]["policy"]
    artifact_root = writer_policy["artifact_root"]
    artifact_sha256 = writer_policy["artifact_digest_sha256"]
    interpreter = writer_policy["interpreter"]
    if (
        gateway_policy["artifact_root"] != artifact_root
        or gateway_policy["artifact_digest_sha256"] != artifact_sha256
        or gateway_policy["interpreter"] != interpreter
        or Path(artifact_root) != DEFAULT_RELEASE_BASE / plan.revision
        or Path(interpreter) != Path(artifact_root) / "venv/bin/python"
    ):
        raise ValueError("writer-only plan does not bind one shared sealed release")
    return {
        "artifact_root": artifact_root,
        "artifact_sha256": _digest(artifact_sha256, "release artifact digest"),
        "interpreter": interpreter,
        "manifest_path": str(Path(artifact_root) / RELEASE_MANIFEST_NAME),
        "manifest_file_sha256": plan.digests.release_manifest_file_sha256,
    }


@dataclass(frozen=True)
class FullCanaryPlan:
    revision: str
    release: Mapping[str, str]
    identities: FullCanaryIdentities
    writer_activation_plan: Mapping[str, Any]
    writer_activation_receipt: Mapping[str, Any]
    writer_activation_receipt_file_sha256: str
    phase_b_readiness_anchor: Mapping[str, Any]
    artifacts: Mapping[str, ExactArtifact]
    allowed_previous_sha256: Mapping[str, str]
    unit_bundle: FullCanarySystemdBundle
    unit_paths: Mapping[str, str]
    e2e_verifier_module: str
    sha256: str
    schema: str = FULL_CANARY_PLAN_SCHEMA

    @classmethod
    def from_mapping(cls, value: Any) -> "FullCanaryPlan":
        raw = _strict_mapping(
            value,
            fields=frozenset(
                {
                    "schema",
                    "revision",
                    "release",
                    "identities",
                    "writer_activation_plan",
                    "writer_activation_receipt",
                    "writer_activation_receipt_file_sha256",
                    "phase_b_readiness_anchor",
                    "artifacts",
                    "allowed_previous_sha256",
                    "systemd_bundle",
                    "unit_paths",
                    "e2e_verifier_module",
                    "full_canary_plan_sha256",
                }
            ),
            label="full-canary plan",
        )
        if raw["schema"] != FULL_CANARY_PLAN_SCHEMA:
            raise ValueError("full-canary plan schema is invalid")
        revision = _revision(raw["revision"])
        release = _strict_mapping(
            raw["release"],
            fields=frozenset(
                {
                    "artifact_root",
                    "artifact_sha256",
                    "interpreter",
                    "manifest_path",
                    "manifest_file_sha256",
                }
            ),
            label="full-canary release",
        )
        artifact_root = _absolute_path(release["artifact_root"], "release root")
        interpreter = _absolute_path(release["interpreter"], "release interpreter")
        manifest_path = _absolute_path(release["manifest_path"], "release manifest")
        if (
            artifact_root != DEFAULT_RELEASE_BASE / revision
            or interpreter != artifact_root / "venv/bin/python"
            or manifest_path != artifact_root / RELEASE_MANIFEST_NAME
        ):
            raise ValueError("full-canary release paths are not revision-bound")
        clean_release = {
            "artifact_root": str(artifact_root),
            "artifact_sha256": _digest(
                release["artifact_sha256"], "release artifact digest"
            ),
            "interpreter": str(interpreter),
            "manifest_path": str(manifest_path),
            "manifest_file_sha256": _digest(
                release["manifest_file_sha256"], "release manifest file digest"
            ),
        }
        writer_plan = ActivationPlan.from_mapping(raw["writer_activation_plan"])
        if writer_plan.revision != revision or _release_binding(writer_plan) != clean_release:
            raise ValueError("full-canary release differs from writer-only plan")
        writer_receipt = _validate_writer_only_receipt(
            raw["writer_activation_receipt"],
            plan=writer_plan,
        )
        receipt_file_digest = _digest(
            raw["writer_activation_receipt_file_sha256"],
            "writer activation receipt file digest",
        )
        phase_b_anchor = _validate_phase_b_readiness_anchor(
            raw["phase_b_readiness_anchor"],
            revision=revision,
        )
        identities = FullCanaryIdentities.from_mapping(raw["identities"])
        artifact_raw = _strict_mapping(
            raw["artifacts"],
            fields=frozenset(
                {
                    "writer_config",
                    "gateway_config",
                    "edge_config",
                    "e2e_fixture",
                    "host_identity_receipt",
                }
            ),
            label="full-canary artifacts",
        )
        artifacts = {
            name: ExactArtifact.from_mapping(item, label=name)
            for name, item in artifact_raw.items()
        }
        expected_artifacts = {
            "writer_config": (
                DEFAULT_WRITER_CONFIG_SOURCE,
                DEFAULT_WRITER_CONFIG,
                0o440,
                0,
                identities.writer_gid,
            ),
            "gateway_config": (
                DEFAULT_GATEWAY_CONFIG_SOURCE,
                DEFAULT_GATEWAY_CONFIG,
                0o440,
                0,
                identities.gateway_gid,
            ),
            "edge_config": (
                DEFAULT_EDGE_CONFIG_SOURCE,
                DEFAULT_EDGE_CONFIG,
                0o440,
                0,
                identities.edge_gid,
            ),
        }
        for name, expected in expected_artifacts.items():
            artifact = artifacts[name]
            observed = (
                artifact.source_path,
                artifact.target_path,
                artifact.mode,
                artifact.uid,
                artifact.gid,
            )
            if observed != expected:
                raise ValueError(f"full-canary {name} artifact is not pinned")
        fixture = artifacts["e2e_fixture"]
        if (
            fixture.source_path != DEFAULT_E2E_FIXTURE
            or fixture.target_path != DEFAULT_E2E_FIXTURE
            or fixture.mode != 0o440
            or fixture.uid != 0
            or fixture.gid != identities.gateway_gid
        ):
            raise ValueError("full-canary E2E fixture is not fixed read-only state")
        host_identity = artifacts["host_identity_receipt"]
        if (
            host_identity.source_path != DEFAULT_HOST_IDENTITY_RECEIPT
            or host_identity.target_path != DEFAULT_HOST_IDENTITY_RECEIPT
            or host_identity.mode != 0o400
            or host_identity.uid != 0
            or host_identity.gid != 0
            or host_identity.maximum_bytes
            != _MAX_HOST_IDENTITY_RECEIPT_BYTES
        ):
            raise ValueError(
                "full-canary host identity receipt is not fixed root state"
            )
        bundle = FullCanarySystemdBundle.from_mapping(raw["systemd_bundle"])
        unit_paths = _strict_mapping(
            raw["unit_paths"],
            fields=frozenset({"edge", "writer", "gateway", "tmpfiles"}),
            label="full-canary unit paths",
        )
        expected_unit_paths = {
            "edge": str(DEFAULT_EDGE_UNIT_PATH),
            "writer": str(DEFAULT_WRITER_UNIT_PATH),
            "gateway": str(DEFAULT_GATEWAY_UNIT_PATH),
            "tmpfiles": str(DEFAULT_TMPFILES_PATH),
        }
        if dict(unit_paths) != expected_unit_paths:
            raise ValueError("full-canary unit paths are not fixed")
        previous_raw = _strict_mapping(
            raw["allowed_previous_sha256"],
            fields=frozenset({"writer_unit", "gateway_unit", "writer_config", "gateway_config"}),
            label="full-canary previous artifacts",
        )
        previous = {
            name: _digest(item, f"previous {name} digest")
            for name, item in previous_raw.items()
        }
        expected_previous = {
            "writer_unit": writer_plan.install_artifacts["writer_unit"].sha256,
            "gateway_unit": writer_plan.install_artifacts["gateway_unit"].sha256,
            "writer_config": writer_plan.install_artifacts["writer_config"].sha256,
            "gateway_config": writer_plan.install_artifacts["gateway_config"].sha256,
        }
        if previous != expected_previous:
            raise ValueError("full-canary previous state is not writer-only state")
        if raw["e2e_verifier_module"] != E2E_VERIFIER_MODULE:
            raise ValueError("full-canary E2E verifier module is not fixed")
        unsigned = {name: copy.deepcopy(item) for name, item in raw.items() if name != "full_canary_plan_sha256"}
        digest = _digest(raw["full_canary_plan_sha256"], "full-canary plan digest")
        if _sha256_json(unsigned) != digest:
            raise ValueError("full-canary plan self-digest drifted")
        result = cls(
            revision=revision,
            release=clean_release,
            identities=identities,
            writer_activation_plan=writer_plan.to_mapping(),
            writer_activation_receipt=writer_receipt,
            writer_activation_receipt_file_sha256=receipt_file_digest,
            phase_b_readiness_anchor=phase_b_anchor,
            artifacts=artifacts,
            allowed_previous_sha256=previous,
            unit_bundle=bundle,
            unit_paths=expected_unit_paths,
            e2e_verifier_module=E2E_VERIFIER_MODULE,
            sha256=digest,
        )
        result._validate_unit_bindings()
        return result

    def _validate_unit_bindings(self) -> None:
        interpreter = self.release["interpreter"]
        writer_plan = ActivationPlan.from_mapping(self.writer_activation_plan)
        expected_exec = {
            "edge": (
                f"ExecStart={interpreter} -B -I -m {EDGE_MODULE} "
                f"--config {DEFAULT_EDGE_CONFIG}"
            ),
            "gateway": (
                f"ExecStart={interpreter} -B -I -m {FULL_GATEWAY_MODULE} "
                f"--config {DEFAULT_GATEWAY_CONFIG} --require-canonical-writer"
            ),
            "writer": (
                f"ExecStart={interpreter} -B -I -m "
                "gateway.canonical_writer_bootstrap "
                f"--config {DEFAULT_WRITER_CONFIG}"
            ),
        }
        writer_pre_commands = re.compile(r"(?m)^ExecStartPre=(.+)$").findall(
            self.unit_bundle.writer_service
        )
        writer_stop_commands = re.compile(r"(?m)^ExecStopPost=(.+)$").findall(
            self.unit_bundle.writer_service
        )
        if (
            expected_exec["edge"] not in self.unit_bundle.edge_service
            or expected_exec["gateway"] not in self.unit_bundle.gateway_service
            or expected_exec["writer"] not in self.unit_bundle.writer_service
            or writer_pre_commands
            or writer_stop_commands
            or "LoadCredential=" in self.unit_bundle.writer_service
            or str(DEFAULT_WRITER_CONFIG_SOURCE)
            in self.unit_bundle.writer_service
            or "CapabilityBoundingSet=\n"
            not in self.unit_bundle.writer_service
            or "AmbientCapabilities=\n" not in self.unit_bundle.writer_service
            or (
                f"User={self.identities.writer_user}\n"
                not in self.unit_bundle.writer_service
            )
            or (
                f"Group={self.identities.writer_group}\n"
                not in self.unit_bundle.writer_service
            )
            or "Restart=no\n" not in self.unit_bundle.writer_service
            or (
                f"Requires={PHASE_B_READINESS_UNIT_NAME}\n"
                not in self.unit_bundle.writer_service
            )
            or (
                f"After=network-online.target {PHASE_B_READINESS_UNIT_NAME}\n"
                not in self.unit_bundle.writer_service
            )
            or _sha256_bytes(
                writer_plan.unit_bundle.phase_b_readiness_service.encode(
                    "utf-8"
                )
            )
            != writer_plan.install_artifacts[
                "phase_b_readiness_unit"
            ].sha256
            or f"After={EDGE_UNIT_NAME}" not in self.unit_bundle.writer_service
            or f"InaccessiblePaths={DEFAULT_EDGE_TOKEN_DIRECTORY}"
            not in self.unit_bundle.gateway_service
            or self.unit_bundle.gateway_service.count(
                _GATEWAY_UNSET_ENVIRONMENT_DIRECTIVE
            )
            != 1
            or f"Environment=HERMES_CONFIG={DEFAULT_GATEWAY_CONFIG}"
            not in self.unit_bundle.gateway_service
            or f"Environment=HERMES_HOME={DEFAULT_GATEWAY_PROFILE_HOME}"
            not in self.unit_bundle.gateway_service
            or (
                f"Environment=HERMES_MANAGED_DIR="
                f"{DEFAULT_DISABLED_MANAGED_SCOPE}"
            )
            not in self.unit_bundle.gateway_service
            or f"Environment=SSL_CERT_FILE={DEFAULT_GATEWAY_CA_BUNDLE}"
            not in self.unit_bundle.gateway_service
            or f"AssertPathExists={DEFAULT_GATEWAY_CA_BUNDLE}"
            not in self.unit_bundle.gateway_service
            or f"ReadOnlyPaths={DEFAULT_GATEWAY_CA_BUNDLE}"
            not in self.unit_bundle.gateway_service
            or "InaccessiblePaths=-/etc/hermes"
            not in self.unit_bundle.gateway_service
            or f"InaccessiblePaths={DEFAULT_DISABLED_MANAGED_SCOPE}"
            in self.unit_bundle.gateway_service
            or str(DEFAULT_DISABLED_MANAGED_SCOPE) in self.unit_bundle.tmpfiles
            or (
                f"d {DEFAULT_COLLECTOR_RUNTIME} 0750 root "
                f"{self.identities.gateway_group} - -"
            )
            not in self.unit_bundle.tmpfiles
            or any(
                f"ReadOnlyPaths={path}"
                not in self.unit_bundle.gateway_service
                or f"InaccessiblePaths={path}"
                in self.unit_bundle.gateway_service
                or f"f+ {path} 0444 root root - -"
                not in self.unit_bundle.tmpfiles
                for path in _GATEWAY_SEALED_EMPTY_ENVIRONMENT_FILES
            )
            or any(
                f"InaccessiblePaths={path}"
                not in self.unit_bundle.gateway_service
                or f"f {path} 0000 root root - -"
                not in self.unit_bundle.tmpfiles
                for path in _GATEWAY_INACCESSIBLE_SEMANTIC_FILES
            )
            or any(
                f"InaccessiblePaths={path}"
                not in self.unit_bundle.gateway_service
                or f"d {path} 0000 root root - -"
                not in self.unit_bundle.tmpfiles
                for path in _GATEWAY_INACCESSIBLE_SEMANTIC_DIRECTORIES
            )
            or f"BindReadOnlyPaths={DEFAULT_GATEWAY_CONFIG}"
            not in self.unit_bundle.gateway_service
            or (
                f"LoadCredential={API_SERVER_CREDENTIAL_NAME}:"
                f"{DEFAULT_API_SERVER_CONTROL_KEY}"
            ) not in self.unit_bundle.gateway_service
            or str(DEFAULT_EDGE_TOKEN_PATH) in self.unit_bundle.gateway_service
            or any(
                "Restart=no\n" not in service
                or "RuntimeMaxSec=900s\n" not in service
                or "Restart=on-failure" in service
                for service in (
                    self.unit_bundle.edge_service,
                    self.unit_bundle.writer_service,
                    self.unit_bundle.gateway_service,
                )
            )
        ):
            raise ValueError("full-canary unit security bindings drifted")

    def unsigned_mapping(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "revision": self.revision,
            "release": dict(self.release),
            "identities": self.identities.to_mapping(),
            "writer_activation_plan": copy.deepcopy(dict(self.writer_activation_plan)),
            "writer_activation_receipt": copy.deepcopy(dict(self.writer_activation_receipt)),
            "writer_activation_receipt_file_sha256": self.writer_activation_receipt_file_sha256,
            "phase_b_readiness_anchor": copy.deepcopy(
                dict(self.phase_b_readiness_anchor)
            ),
            "artifacts": {
                name: artifact.to_mapping()
                for name, artifact in sorted(self.artifacts.items())
            },
            "allowed_previous_sha256": dict(sorted(self.allowed_previous_sha256.items())),
            "systemd_bundle": self.unit_bundle.to_mapping(),
            "unit_paths": dict(self.unit_paths),
            "e2e_verifier_module": self.e2e_verifier_module,
        }

    def to_mapping(self) -> dict[str, Any]:
        return {**self.unsigned_mapping(), "full_canary_plan_sha256": self.sha256}


def build_full_canary_plan(
    *,
    writer_activation_plan: ActivationPlan,
    writer_activation_receipt: Mapping[str, Any],
    writer_activation_receipt_file_sha256: str,
    phase_b_readiness_anchor: Mapping[str, Any],
    identities: FullCanaryIdentities,
    writer_config: ExactArtifact,
    gateway_config: ExactArtifact,
    edge_config: ExactArtifact,
    e2e_fixture: ExactArtifact,
    host_identity_receipt: ExactArtifact,
) -> FullCanaryPlan:
    """Build one pure plan from a completed, stopped writer-only lifecycle."""

    if not isinstance(writer_activation_plan, ActivationPlan):
        raise TypeError("writer activation plan is required")
    writer_receipt = _validate_writer_only_receipt(
        writer_activation_receipt,
        plan=writer_activation_plan,
    )
    release = _release_binding(writer_activation_plan)
    database_host = writer_activation_plan.deployment_manifest["snapshot_template"]["database"]["connection"]["host"]
    bundle = render_full_canary_systemd_bundle(
        revision=writer_activation_plan.revision,
        artifact_sha256=release["artifact_sha256"],
        interpreter=Path(release["interpreter"]),
        writer_only_service=writer_activation_plan.unit_bundle.writer_service,
        identities=identities,
        database_ip_allow=f"{database_host}/32",
    )
    unsigned = {
        "schema": FULL_CANARY_PLAN_SCHEMA,
        "revision": writer_activation_plan.revision,
        "release": dict(release),
        "identities": identities.to_mapping(),
        "writer_activation_plan": writer_activation_plan.to_mapping(),
        "writer_activation_receipt": copy.deepcopy(dict(writer_receipt)),
        "writer_activation_receipt_file_sha256": _digest(
            writer_activation_receipt_file_sha256,
            "writer activation receipt file digest",
        ),
        "phase_b_readiness_anchor": _validate_phase_b_readiness_anchor(
            phase_b_readiness_anchor,
            revision=writer_activation_plan.revision,
        ),
        "artifacts": {
            "writer_config": writer_config.to_mapping(),
            "gateway_config": gateway_config.to_mapping(),
            "edge_config": edge_config.to_mapping(),
            "e2e_fixture": e2e_fixture.to_mapping(),
            "host_identity_receipt": host_identity_receipt.to_mapping(),
        },
        "allowed_previous_sha256": {
            "writer_unit": writer_activation_plan.install_artifacts["writer_unit"].sha256,
            "gateway_unit": writer_activation_plan.install_artifacts["gateway_unit"].sha256,
            "writer_config": writer_activation_plan.install_artifacts["writer_config"].sha256,
            "gateway_config": writer_activation_plan.install_artifacts["gateway_config"].sha256,
        },
        "systemd_bundle": bundle.to_mapping(),
        "unit_paths": {
            "edge": str(DEFAULT_EDGE_UNIT_PATH),
            "writer": str(DEFAULT_WRITER_UNIT_PATH),
            "gateway": str(DEFAULT_GATEWAY_UNIT_PATH),
            "tmpfiles": str(DEFAULT_TMPFILES_PATH),
        },
        "e2e_verifier_module": E2E_VERIFIER_MODULE,
    }
    return FullCanaryPlan.from_mapping(
        {**unsigned, "full_canary_plan_sha256": _sha256_json(unsigned)}
    )


@dataclass(frozen=True)
class FullCanaryOwnerApproval:
    value: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, value: Any) -> "FullCanaryOwnerApproval":
        raw = _strict_mapping(
            value,
            fields=frozenset(
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
            ),
            label="full-canary owner approval",
        )
        if (
            raw["schema"] != FULL_CANARY_APPROVAL_SCHEMA
            or raw["scope"] != "full_canary_runtime_start"
            or raw["authority_kind"] != "trusted_root_bootstrap_out_of_band_owner"
            or raw["cryptographic_owner_proof"] is not False
        ):
            raise ValueError("full-canary owner approval trust semantics are invalid")
        for name in (
            "plan_sha256",
            "owner_subject_sha256",
            "approval_source_sha256",
            "nonce_sha256",
        ):
            _digest(raw[name], f"owner approval {name}")
        approved = raw["approved_at_unix"]
        expires = raw["expires_at_unix"]
        if (
            type(approved) is not int
            or type(expires) is not int
            or approved < 0
            or not 1 <= expires - approved <= 900
        ):
            raise ValueError("full-canary owner approval window is invalid")
        return cls(copy.deepcopy(dict(raw)))

    def require(self, *, plan_sha256: str, now_unix: int) -> None:
        if (
            self.value["plan_sha256"] != plan_sha256
            or type(now_unix) is not int
            or not self.value["approved_at_unix"] <= now_unix <= self.value["expires_at_unix"]
        ):
            raise PermissionError("owner approval does not authorize this full-canary start")

    @property
    def sha256(self) -> str:
        return _sha256_json(self.value)


def _read_stable_file(
    path: Path,
    *,
    maximum: int,
    expected_uid: int | None = None,
    expected_gid: int | None = None,
    allowed_modes: frozenset[int] | None = None,
    allowed_link_counts: frozenset[int] = frozenset({1}),
) -> tuple[bytes, os.stat_result]:
    """Read one bounded regular file through a no-follow stable descriptor."""

    path = _absolute_path(path, "trusted file")
    before = path.lstat()
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or before.st_nlink not in allowed_link_counts
        or not 0 <= before.st_size <= maximum
        or (expected_uid is not None and before.st_uid != expected_uid)
        or (expected_gid is not None and before.st_gid != expected_gid)
        or (
            allowed_modes is not None
            and stat.S_IMODE(before.st_mode) not in allowed_modes
        )
    ):
        raise RuntimeError("trusted full-canary file identity is invalid")
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    chunks: list[bytes] = []
    total = 0
    try:
        opened = os.fstat(descriptor)
        while total <= maximum:
            chunk = os.read(descriptor, min(64 * 1024, maximum + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    reachable = path.lstat()

    def identity(item: os.stat_result) -> tuple[int, ...]:
        return (
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
        total > maximum
        or total != before.st_size
        or identity(before) != identity(opened)
        or identity(before) != identity(after)
        or identity(before) != identity(reachable)
    ):
        raise RuntimeError("trusted full-canary file changed during read")
    return b"".join(chunks), before


def _validate_release_manifest(plan: FullCanaryPlan) -> Mapping[str, Any]:
    raw, _item = _read_stable_file(
        Path(plan.release["manifest_path"]),
        maximum=_MAX_JSON_BYTES,
        expected_uid=0,
        expected_gid=0,
        allowed_modes=frozenset({0o400}),
    )
    if _sha256_bytes(raw) != plan.release["manifest_file_sha256"]:
        raise RuntimeError("sealed release manifest file digest drifted")
    manifest = _decode_json(raw, label="sealed release manifest")
    required = {
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
    }
    if (
        set(manifest) != required
        or manifest["schema"] != RELEASE_SCHEMA
        or manifest["revision"] != plan.revision
        or manifest["artifact_root"] != plan.release["artifact_root"]
        or manifest["interpreter"] != plan.release["interpreter"]
        or manifest["artifact_sha256"] != plan.release["artifact_sha256"]
    ):
        raise RuntimeError("sealed release manifest identity drifted")
    unsigned = {name: copy.deepcopy(item) for name, item in manifest.items() if name != "artifact_sha256"}
    if _sha256_json(unsigned) != manifest["artifact_sha256"]:
        raise RuntimeError("sealed release artifact digest drifted")
    return manifest


def _validated_release_file(
    plan: FullCanaryPlan,
    relative_path: Path,
    *,
    maximum_bytes: int,
) -> tuple[Path, bytes, str]:
    """Read one exact manifest-bound file from the sealed release."""

    if (
        relative_path.is_absolute()
        or ".." in relative_path.parts
        or relative_path.as_posix() != str(relative_path)
        or not 0 < maximum_bytes <= _MAX_JSON_BYTES
    ):
        raise RuntimeError("sealed release file request is invalid")
    manifest = _validate_release_manifest(plan)
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise RuntimeError("sealed release entries are invalid")
    matches = [
        entry
        for entry in entries
        if isinstance(entry, Mapping)
        and entry.get("path") == relative_path.as_posix()
    ]
    if len(matches) != 1:
        raise RuntimeError("sealed release file is not uniquely declared")
    entry = matches[0]
    if set(entry) != {"path", "kind", "mode", "size", "sha256"}:
        raise RuntimeError("sealed release file entry is not exact")
    mode = entry.get("mode")
    size = entry.get("size")
    if (
        entry.get("kind") != "file"
        or not isinstance(mode, str)
        or re.fullmatch(r"0[0-7]{3}", mode) is None
        or type(size) is not int
        or not 0 < size <= maximum_bytes
    ):
        raise RuntimeError("sealed release file metadata is invalid")
    digest = _digest(entry.get("sha256"), "sealed release file digest")
    path = Path(plan.release["artifact_root"]) / relative_path
    raw, item = _read_stable_file(
        path,
        maximum=maximum_bytes,
        expected_uid=0,
        expected_gid=0,
        allowed_modes=frozenset({int(mode, 8)}),
    )
    if item.st_size != size or _sha256_bytes(raw) != digest:
        raise RuntimeError("sealed release file content drifted")
    return path, raw, digest


class _StrictYamlLoader(yaml.SafeLoader):
    def compose_node(self, parent: Any, index: Any) -> Any:
        if self.check_event(yaml.events.AliasEvent):
            raise ValueError("full-canary config cannot contain YAML aliases")
        return super().compose_node(parent, index)

    def construct_mapping(self, node: Any, deep: bool = False) -> dict[str, Any]:
        if not isinstance(node, yaml.nodes.MappingNode):
            raise ValueError("full-canary YAML mapping is invalid")
        result: dict[str, Any] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str) or not key or key in result:
                raise ValueError("full-canary YAML contains duplicate or invalid key")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


_StrictYamlLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _StrictYamlLoader.construct_mapping,
)


def _validate_gateway_config(raw: bytes) -> Mapping[str, Any]:
    if b"DISCORD_BOT_TOKEN" in raw:
        raise RuntimeError("gateway config references the Discord credential")
    try:
        value = yaml.load(raw.decode("utf-8", errors="strict"), Loader=_StrictYamlLoader)
    except (UnicodeDecodeError, yaml.YAMLError, ValueError) as exc:
        raise RuntimeError("full gateway config is not strict YAML") from exc
    if not isinstance(value, Mapping):
        raise RuntimeError("full gateway config root is not a mapping")
    if set(value) != {
        "canonical_brain",
        "model",
        "agent",
        "memory",
        "cron",
        "kanban",
        "curator",
        "plugins",
        "platform_toolsets",
        "gateway",
        "platforms",
    }:
        raise RuntimeError("full gateway config root is not exact")
    canonical = value.get("canonical_brain")
    if canonical != {
        "writer_boundary": {"enabled": True},
        "discord_edge": {"enabled": True},
        "tools_enabled": True,
    }:
        raise RuntimeError("full gateway Canonical boundary is not exact enabled policy")
    model = value.get("model")
    agent = value.get("agent")
    adaptive = agent.get("adaptive_reasoning") if isinstance(agent, Mapping) else None
    if (
        model
        != {"default": "gpt-5.6-sol", "provider": "openai-codex"}
        or not isinstance(agent, Mapping)
        or set(agent)
        != {"reasoning_effort", "max_turns", "adaptive_reasoning"}
        or agent["reasoning_effort"] != "high"
        or agent["max_turns"] != 90
        or adaptive != {"enabled": True, "max_effort": "max"}
    ):
        raise RuntimeError("full gateway model/adaptive reasoning route is not exact")
    kanban = value.get("kanban")
    curator = value.get("curator")
    if (
        kanban
        != {
            "auxiliary_planning_enabled": False,
            "auto_decompose": False,
            "dispatch_in_gateway": False,
        }
        or not isinstance(kanban, Mapping)
        or type(kanban.get("dispatch_in_gateway")) is not bool
        or curator != {"enabled": False, "prune_builtins": False}
    ):
        raise RuntimeError("full gateway model-sovereignty policy is not exact")
    memory = value.get("memory")
    cron = value.get("cron")
    if memory != {
        "memory_enabled": False,
        "user_profile_enabled": False,
    } or (
        not isinstance(memory, Mapping)
        or type(memory.get("memory_enabled")) is not bool
        or type(memory.get("user_profile_enabled")) is not bool
    ) or cron != {"enabled": False} or (
        not isinstance(cron, Mapping)
        or type(cron.get("enabled")) is not bool
    ):
        raise RuntimeError("full gateway clean-room state policy is not exact")
    plugins = value.get("plugins")
    enabled_plugins = plugins.get("enabled") if isinstance(plugins, Mapping) else None
    if (
        not isinstance(plugins, Mapping)
        or set(plugins) != {"enabled"}
        or not isinstance(enabled_plugins, list)
        or any(not isinstance(item, str) for item in enabled_plugins)
        or enabled_plugins != sorted(set(enabled_plugins))
        or enabled_plugins != [CANARY_OBSERVER_PLUGIN]
    ):
        raise RuntimeError("full gateway has an unapproved in-process plugin")
    platform_toolsets = value.get("platform_toolsets")
    if platform_toolsets != {"api_server": ["canonical_brain", "todo"]}:
        raise RuntimeError("full gateway API tool surface is not exact")
    platforms = value.get("platforms")
    if not isinstance(platforms, Mapping) or set(platforms) != {"api_server"}:
        raise RuntimeError("full gateway exposes only the pinned API control surface")
    api_server = platforms.get("api_server")
    if (
        not isinstance(api_server, Mapping)
        or set(api_server) != {"enabled", "extra"}
        or api_server.get("enabled") is not True
    ):
        raise RuntimeError("full gateway API control surface is not exact")
    api_extra = api_server.get("extra")
    if (
        not isinstance(api_extra, Mapping)
        or set(api_extra) != {"host", "port", "key_credential"}
        or api_extra.get("host") != "127.0.0.1"
        or api_extra.get("port") != 8642
        or api_extra.get("key_credential") != API_SERVER_CREDENTIAL_NAME
    ):
        raise RuntimeError("full gateway API credential/loopback binding is not exact")
    gateway = value.get("gateway")
    api_runtime = gateway.get("api_server") if isinstance(gateway, Mapping) else None
    if (
        gateway
        != {
            "api_server": {"max_concurrent_runs": 1},
            "isolated_runtime": True,
        }
        or not isinstance(gateway, Mapping)
        or type(gateway.get("isolated_runtime")) is not bool
        or api_runtime != {"max_concurrent_runs": 1}
    ):
        raise RuntimeError("full gateway API runtime policy is not exact")
    return value


def _validate_writer_config(
    raw: bytes,
    identities: FullCanaryIdentities,
) -> Mapping[str, Any]:
    value = _decode_json(raw, label="full writer config")
    base_fields = {"service", "database", "privileges", "discord_edge_authority"}
    if set(value) != base_fields:
        raise RuntimeError("full writer config root is not exact")
    service = value.get("service")
    database = value.get("database")
    privileges = value.get("privileges")
    edge = value.get("discord_edge_authority")
    service_fields = {
        "socket_path",
        "gateway_unit",
        "gateway_uid",
        "writer_uid",
        "writer_gid",
        "socket_gid",
        "projector_gid",
        "owner_discord_user_ids",
        "connection_timeout_seconds",
        "max_connections",
    }
    database_fields = {
        "host",
        "tls_server_name",
        "port",
        "database",
        "user",
        "ca_file",
        "credential_file",
        "connect_timeout_seconds",
        "io_timeout_seconds",
    }
    privilege_fields = {
        "schema",
        "table_grants",
        "routine_identities",
        "helper_routine_identities",
        "schema_privileges",
        "database_privileges",
        "role_memberships",
        "private_schema_identity_sha256",
        "managed_cloudsqladmin_hba_rejection_receipt",
        "managed_cloudsqladmin_hba_rejection_sha256",
        "deployment_lock_key",
    }
    edge_fields = {
        "enabled",
        "capability_private_key_file",
        "edge_receipt_public_key_file",
        "edge_receipt_public_key_id",
        "request_timeout_seconds",
    }
    if (
        not isinstance(service, Mapping)
        or set(service) != service_fields
        or not isinstance(database, Mapping)
        or set(database) != database_fields
        or not isinstance(privileges, Mapping)
        or set(privileges) != privilege_fields
        or not isinstance(edge, Mapping)
        or set(edge) != edge_fields
        or service.get("gateway_unit") != GATEWAY_UNIT_NAME
        or service.get("gateway_uid") != identities.gateway_uid
        or service.get("writer_uid") != identities.writer_uid
        or service.get("writer_gid") != identities.writer_gid
        or service.get("socket_gid") != identities.socket_client_gid
        or edge.get("enabled") is not True
        or edge.get("capability_private_key_file")
        != str(DEFAULT_WRITER_CAPABILITY_PRIVATE_KEY)
    ):
        raise RuntimeError("full writer nested schema/binding is not exact")
    routine_fields = {
        "signature",
        "owner",
        "security_definer",
        "language",
        "configuration",
        "definition_sha256",
    }
    if (
        privileges.get("table_grants") != []
        or any(
            not isinstance(items, list)
            or any(
                not isinstance(item, Mapping)
                or set(item) != routine_fields
                for item in items
            )
            for items in (
                privileges.get("routine_identities"),
                privileges.get("helper_routine_identities"),
            )
        )
    ):
        raise RuntimeError("full writer privilege schema is not exact")
    return value


def _validate_edge_config(raw: bytes, identities: FullCanaryIdentities) -> Mapping[str, Any]:
    value = _decode_json(raw, label="Discord edge config")
    if set(value) != {"service", "keys", "discord", "journal", "runtime"}:
        raise RuntimeError("Discord edge config root is not exact")
    service = value.get("service")
    keys = value.get("keys")
    discord = value.get("discord")
    if (
        not isinstance(service, Mapping)
        or service.get("gateway_unit") != GATEWAY_UNIT_NAME
        or service.get("edge_unit") != EDGE_UNIT_NAME
        or service.get("gateway_uid") != identities.gateway_uid
        or service.get("edge_uid") != identities.edge_uid
        or service.get("edge_gid") != identities.edge_gid
        or not isinstance(keys, Mapping)
        or keys.get("writer_capability_public_key_file")
        != str(DEFAULT_WRITER_CAPABILITY_PUBLIC_KEY)
        or keys.get("edge_receipt_private_key_file")
        != str(DEFAULT_EDGE_RECEIPT_PRIVATE_KEY)
        or not isinstance(discord, Mapping)
        or discord.get("token_file") != str(DEFAULT_EDGE_TOKEN_PATH)
        or discord.get("credentials_directory")
        != str(DEFAULT_EDGE_TOKEN_DIRECTORY)
    ):
        raise RuntimeError("Discord edge config identity/path binding is invalid")
    if any(name in discord for name in ("token", "bot_token", "credential_value")):
        raise RuntimeError("Discord edge config embeds credential material")
    return value


def _validate_artifact_source(
    artifact: ExactArtifact,
    *,
    label: str,
) -> bytes:
    raw, _item = _read_stable_file(
        artifact.source_path,
        maximum=artifact.maximum_bytes,
        expected_uid=artifact.uid,
        expected_gid=artifact.gid,
        allowed_modes=frozenset({artifact.mode}),
    )
    if _sha256_bytes(raw) != artifact.sha256:
        raise RuntimeError(f"full-canary {label} source digest drifted")
    return raw


def _validate_secret_source_metadata(
    path: Path,
    *,
    expected_uid: int,
    expected_gid: int,
    expected_mode: int,
    maximum_bytes: int,
) -> bool:
    """Validate secret-file provenance without reading or hashing its value."""
    path = _absolute_path(path, "full-canary secret source")
    item = path.lstat()
    return (
        stat.S_ISREG(item.st_mode)
        and not stat.S_ISLNK(item.st_mode)
        and item.st_nlink == 1
        and item.st_uid == expected_uid
        and item.st_gid == expected_gid
        and stat.S_IMODE(item.st_mode) == expected_mode
        and 0 < item.st_size <= maximum_bytes
    )


def _validate_absent_managed_scope(
    managed_directory: Path,
    *,
    expected_parent_uid: int,
    expected_parent_gid: int,
    gateway_uid: int,
    gateway_gid: int,
    allow_parent_absent: bool = False,
) -> bool:
    """Prove managed scope is an absent child the gateway cannot create."""

    if any(
        type(value) is not int or value < 0
        for value in (
            expected_parent_uid,
            expected_parent_gid,
            gateway_uid,
            gateway_gid,
        )
    ) or type(allow_parent_absent) is not bool:
        raise TypeError("managed-scope identity boundary is invalid")
    managed_directory = _absolute_path(
        managed_directory,
        "disabled managed-scope child",
    )
    parent = managed_directory.parent
    if parent == managed_directory or gateway_uid == expected_parent_uid:
        raise RuntimeError("managed-scope parent identity is not isolating")

    try:
        managed_directory.lstat()
    except FileNotFoundError:
        pass
    else:
        raise RuntimeError("disabled managed-scope child exists")

    try:
        before = parent.lstat()
    except FileNotFoundError:
        if allow_parent_absent:
            return True
        raise RuntimeError("managed-scope parent is absent") from None
    if (
        not stat.S_ISDIR(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or before.st_uid != expected_parent_uid
        or before.st_gid != expected_parent_gid
        or stat.S_IMODE(before.st_mode) != 0o750
        or gateway_gid != expected_parent_gid
    ):
        raise RuntimeError("managed-scope parent boundary is not exact")

    after = parent.lstat()
    try:
        managed_directory.lstat()
    except FileNotFoundError:
        pass
    else:
        raise RuntimeError("disabled managed-scope child materialized")
    def identity(item: os.stat_result) -> tuple[int, ...]:
        return (
            item.st_dev,
            item.st_ino,
            item.st_mode,
            item.st_uid,
            item.st_gid,
        )
    if identity(before) != identity(after):
        raise RuntimeError("managed-scope parent changed during validation")
    return True


def _validate_inert_gateway_paths(
    *,
    environment_files: Sequence[Path] = _GATEWAY_SEALED_EMPTY_ENVIRONMENT_FILES,
    semantic_files: Sequence[Path] = _GATEWAY_INACCESSIBLE_SEMANTIC_FILES,
    semantic_directories: Sequence[
        Path
    ] = _GATEWAY_INACCESSIBLE_SEMANTIC_DIRECTORIES,
    managed_directory: Path = DEFAULT_DISABLED_MANAGED_SCOPE,
    expected_uid: int = 0,
    expected_gid: int = 0,
    gateway_uid: int,
    gateway_gid: int,
) -> bool:
    """Prove every pre-exec startup-input boundary is exact and readable."""

    if any(
        type(value) is not int or value < 0
        for value in (expected_uid, expected_gid, gateway_uid, gateway_gid)
    ):
        raise TypeError("inert gateway path ownership is invalid")
    for path in environment_files:
        raw, _item = _read_stable_file(
            path,
            maximum=0,
            expected_uid=expected_uid,
            expected_gid=expected_gid,
            allowed_modes=frozenset({0o444}),
        )
        if raw != b"":
            raise RuntimeError("sealed gateway environment file is not empty")
    for path in semantic_files:
        item = path.lstat()
        if (
            not stat.S_ISREG(item.st_mode)
            or stat.S_ISLNK(item.st_mode)
            or item.st_nlink != 1
            or item.st_uid != expected_uid
            or item.st_gid != expected_gid
            or stat.S_IMODE(item.st_mode) != 0
        ):
            raise RuntimeError("inert gateway file boundary is not exact")
    for path in semantic_directories:
        item = path.lstat()
        if (
            not stat.S_ISDIR(item.st_mode)
            or stat.S_ISLNK(item.st_mode)
            or item.st_uid != expected_uid
            or item.st_gid != expected_gid
            or stat.S_IMODE(item.st_mode) != 0
        ):
            raise RuntimeError("inert gateway directory boundary is not exact")
    _validate_absent_managed_scope(
        managed_directory,
        expected_parent_uid=expected_uid,
        expected_parent_gid=gateway_gid,
        gateway_uid=gateway_uid,
        gateway_gid=gateway_gid,
    )
    return True


def _gateway_startup_paths_are_sealed(
    plan: FullCanaryPlan,
    *,
    phase: str,
) -> bool:
    """Validate the absent sentinel always and all masks once installed."""

    identities = plan.identities
    _validate_absent_managed_scope(
        DEFAULT_DISABLED_MANAGED_SCOPE,
        expected_parent_uid=0,
        expected_parent_gid=identities.gateway_gid,
        gateway_uid=identities.gateway_uid,
        gateway_gid=identities.gateway_gid,
        allow_parent_absent=phase == "stopped",
    )
    if phase == "live":
        return _validate_inert_gateway_paths(
            gateway_uid=identities.gateway_uid,
            gateway_gid=identities.gateway_gid,
        )
    try:
        current_tmpfiles, _item = _read_stable_file(
            DEFAULT_TMPFILES_PATH,
            maximum=256 * 1024,
            expected_uid=0,
            expected_gid=0,
            allowed_modes=frozenset({0o644}),
        )
    except Exception:
        # The first stopped preflight runs before this plan is installed. Its
        # target inventory owns that state; the absent child is still required.
        return True
    if current_tmpfiles != plan.unit_bundle.tmpfiles.encode("utf-8"):
        return True
    return _validate_inert_gateway_paths(
        gateway_uid=identities.gateway_uid,
        gateway_gid=identities.gateway_gid,
    )


@dataclass(frozen=True)
class CollectorReadiness:
    receipt: Mapping[str, Any]
    file_sha256: str
    service_identity_sha256: str


@dataclass(frozen=True)
class PluginReadiness:
    receipt: Mapping[str, Any]
    file_sha256: str
    frame_sha256: str
    collector_hash_chain_head_sha256: str


def _validated_e2e_fixture(plan: FullCanaryPlan) -> Mapping[str, Any]:
    raw = _validate_artifact_source(
        plan.artifacts["e2e_fixture"],
        label="e2e_fixture",
    )
    value = _decode_json(raw, label="full-canary E2E fixture")
    from gateway.canonical_full_canary_e2e import _validate_fixture

    validated = _validate_fixture(value)
    if (
        validated.get("release_sha") != plan.revision
        or validated.get("release_artifact_sha256")
        != plan.release["artifact_sha256"]
    ):
        raise RuntimeError("full-canary fixture release binding drifted")
    return validated


def load_collector_readiness(
    plan: FullCanaryPlan,
    *,
    edge_pid: int,
    edge_service_identity_sha256: str,
    path: Path = DEFAULT_COLLECTOR_READINESS_PATH,
) -> CollectorReadiness:
    """Validate the root live-driver handoff after edge readiness."""
    if not isinstance(plan, FullCanaryPlan):
        raise TypeError("full-canary plan is required")
    edge_service_identity_sha256 = _digest(
        edge_service_identity_sha256,
        "edge service identity digest",
    )
    if type(edge_pid) is not int or edge_pid <= 1:
        raise RuntimeError("collector readiness edge PID is invalid")
    path = _absolute_path(path, "collector readiness receipt")
    if path != DEFAULT_COLLECTOR_READINESS_PATH:
        raise RuntimeError("collector readiness receipt path is not fixed")
    raw, _item = _read_stable_file(
        path,
        maximum=512 * 1024,
        expected_uid=0,
        expected_gid=0,
        allowed_modes=frozenset({0o400}),
    )
    value = _decode_json(raw, label="full-canary collector readiness")
    fields = {
        "schema",
        "release_sha",
        "full_canary_plan_sha256",
        "canary_run_id",
        "edge_pid",
        "edge_service_identity_sha256",
        "collector_socket",
        "service_identity",
        "service_identity_sha256",
        "observed_at_unix",
        "observed_at_boottime_ns",
        "receipt_sha256",
    }
    if set(value) != fields or value.get("schema") != COLLECTOR_READINESS_SCHEMA:
        raise RuntimeError("collector readiness receipt fields are not exact")
    unsigned = {
        name: copy.deepcopy(item)
        for name, item in value.items()
        if name != "receipt_sha256"
    }
    fixture = _validated_e2e_fixture(plan)
    if (
        value.get("receipt_sha256") != _sha256_json(unsigned)
        or value.get("release_sha") != plan.revision
        or value.get("full_canary_plan_sha256") != plan.sha256
        or value.get("canary_run_id") != fixture.get("canary_run_id")
        or value.get("edge_pid") != edge_pid
        or value.get("edge_service_identity_sha256")
        != edge_service_identity_sha256
    ):
        raise RuntimeError("collector readiness plan/edge binding drifted")

    identity = value.get("service_identity")
    identity_fields = {
        "schema",
        "release_sha",
        "collector_pid",
        "collector_start_time_ticks",
        "collector_uid",
        "collector_gid",
        "boot_id_sha256",
        "module_origin",
        "module_sha256",
    }
    if (
        not isinstance(identity, Mapping)
        or set(identity) != identity_fields
        or identity.get("schema") != COLLECTOR_IDENTITY_SCHEMA
        or identity.get("release_sha") != plan.revision
        or identity.get("collector_uid") != 0
        or identity.get("collector_gid") != 0
        or value.get("service_identity_sha256") != _sha256_json(identity)
    ):
        raise RuntimeError("collector service identity binding is invalid")
    collector_pid = identity.get("collector_pid")
    start_ticks = identity.get("collector_start_time_ticks")
    if (
        type(collector_pid) is not int
        or collector_pid <= 1
        or type(start_ticks) is not int
        or start_ticks <= 0
        or process_start_time_ticks(collector_pid) != start_ticks
    ):
        raise RuntimeError("collector process identity is not live")
    process_item = (Path("/proc") / str(collector_pid)).stat()
    if process_item.st_uid != 0 or process_item.st_gid != 0:
        raise RuntimeError("collector process is not root-owned")
    boot_sha256, now_boottime_ns = boot_identity()
    observed_boottime = value.get("observed_at_boottime_ns")
    if (
        identity.get("boot_id_sha256") != boot_sha256
        or type(observed_boottime) is not int
        or observed_boottime < 0
        or not 0 <= now_boottime_ns - observed_boottime <= 60_000_000_000
        or type(value.get("observed_at_unix")) is not int
        or value["observed_at_unix"] < 0
    ):
        raise RuntimeError("collector readiness is stale or from another boot")
    module_origin = _absolute_path(identity.get("module_origin"), "collector module")
    release_root = Path(plan.release["artifact_root"])
    if release_root not in module_origin.parents:
        raise RuntimeError("collector module is outside the sealed release")
    observed_origin, observed_module_sha256 = module_file_identity(module_origin)
    if (
        observed_origin != str(module_origin)
        or identity.get("module_sha256") != observed_module_sha256
    ):
        raise RuntimeError("collector module identity drifted")

    socket_receipt = value.get("collector_socket")
    if not isinstance(socket_receipt, Mapping) or set(socket_receipt) != {
        "path",
        "device",
        "inode",
        "uid",
        "gid",
        "mode",
    }:
        raise RuntimeError("collector socket receipt fields are not exact")
    socket_item = DEFAULT_COLLECTOR_SOCKET.lstat()
    if (
        socket_receipt.get("path") != str(DEFAULT_COLLECTOR_SOCKET)
        or socket_receipt.get("device") != socket_item.st_dev
        or socket_receipt.get("inode") != socket_item.st_ino
        or socket_receipt.get("uid") != 0
        or socket_receipt.get("gid") != plan.identities.gateway_gid
        or socket_receipt.get("mode") != "0660"
        or not stat.S_ISSOCK(socket_item.st_mode)
        or socket_item.st_uid != 0
        or socket_item.st_gid != plan.identities.gateway_gid
        or stat.S_IMODE(socket_item.st_mode) != 0o660
    ):
        raise RuntimeError("collector socket identity drifted")
    from gateway.canonical_writer_root_collector import _unix_listener_paths_for_pid

    if str(DEFAULT_COLLECTOR_SOCKET) not in _unix_listener_paths_for_pid(
        collector_pid
    ):
        raise RuntimeError("collector PID does not own the fixed listener")
    return CollectorReadiness(
        receipt=copy.deepcopy(dict(value)),
        file_sha256=_sha256_bytes(raw),
        service_identity_sha256=str(value["service_identity_sha256"]),
    )


def _validate_edge_collector_gate(
    plan: FullCanaryPlan,
    state: Mapping[str, Any],
) -> Mapping[str, Any]:
    if (
        state.get("LoadState") != "loaded"
        or state.get("ActiveState") != "active"
        or state.get("SubState") != "running"
        or state.get("UnitFileState") not in {"disabled", ""}
        or state.get("FragmentPath") != str(DEFAULT_EDGE_UNIT_PATH)
        or state.get("DropInPaths") not in {"", "[]"}
        or state.get("Type") != "notify"
        or state.get("NotifyAccess") != "main"
        or type(state.get("MainPID")) is not int
        or state["MainPID"] <= 1
    ):
        raise RuntimeError("Discord edge is not ready for collector handoff")
    receipt = _readiness_receipt(
        DEFAULT_EDGE_READINESS_PATH,
        uid=plan.identities.edge_uid,
        gid=plan.identities.edge_gid,
    )
    digest = readiness_receipt_sha256(receipt)
    if (
        receipt.get("version") != EDGE_READINESS_SCHEMA
        or receipt.get("edge_pid") != state["MainPID"]
        or receipt.get("config_sha256") != plan.artifacts["edge_config"].sha256
        or state.get("StatusText") != f"{EDGE_READINESS_SCHEMA}:{digest}"
        or receipt.get("allowed_target_types")
        != [
            "public_guild_channel",
            "public_guild_forum",
            "public_guild_thread",
        ]
        or receipt.get("forbidden_target_types")
        != [
            "direct_message",
            "dm",
            "group_dm",
            "private_channel",
            "private_thread",
        ]
    ):
        raise RuntimeError("Discord edge readiness receipt is not exact")
    return receipt


def _observer_config_mapping(
    plan: FullCanaryPlan,
    *,
    collector: CollectorReadiness,
    edge_pid: int,
    edge_service_identity_sha256: str,
) -> Mapping[str, Any]:
    fixture = _validated_e2e_fixture(plan)
    collector_identity = collector.receipt["service_identity"]
    return {
        "schema": "muncho-canary-evidence-config.v1",
        "release_sha": plan.revision,
        "release_sha256": plan.release["artifact_sha256"],
        "canary_run_id": fixture["canary_run_id"],
        "case_id": fixture["case_id"],
        "fixture_path": str(DEFAULT_E2E_FIXTURE),
        "fixture_sha256": plan.artifacts["e2e_fixture"].sha256,
        "collector": {
            "socket_path": str(DEFAULT_COLLECTOR_SOCKET),
            "expected_pid": collector_identity["collector_pid"],
            "expected_uid": 0,
            "expected_gid": 0,
            "socket_owner_uid": 0,
            "socket_owner_gid": plan.identities.gateway_gid,
            "socket_mode": "0660",
            "service_identity_sha256": collector.service_identity_sha256,
            "connect_timeout_ms": 1000,
            "ack_timeout_ms": 3000,
        },
        "discord_edge": {
            "socket_path": str(DEFAULT_EDGE_SOCKET),
            "expected_pid": edge_pid,
            "expected_uid": plan.identities.edge_uid,
            "expected_gid": plan.identities.edge_gid,
            "socket_owner_uid": plan.identities.edge_uid,
            "socket_owner_gid": plan.identities.edge_gid,
            "socket_mode": "0660",
            "service_identity_sha256": edge_service_identity_sha256,
            "connect_timeout_ms": 1000,
            "response_timeout_ms": 2000,
        },
    }


def _observer_static_binding_matches(
    value: Any,
    plan: FullCanaryPlan,
) -> bool:
    if not isinstance(value, Mapping):
        return False
    try:
        fixture = _validated_e2e_fixture(plan)
    except Exception:
        return False
    return (
        value.get("schema") == "muncho-canary-evidence-config.v1"
        and value.get("release_sha") == plan.revision
        and value.get("release_sha256") == plan.release["artifact_sha256"]
        and value.get("canary_run_id") == fixture.get("canary_run_id")
        and value.get("case_id") == fixture.get("case_id")
        and value.get("fixture_path") == str(DEFAULT_E2E_FIXTURE)
        and value.get("fixture_sha256")
        == plan.artifacts["e2e_fixture"].sha256
        and isinstance(value.get("collector"), Mapping)
        and value["collector"].get("socket_path")
        == str(DEFAULT_COLLECTOR_SOCKET)
        and isinstance(value.get("discord_edge"), Mapping)
        and value["discord_edge"].get("socket_path")
        == str(DEFAULT_EDGE_SOCKET)
    )


def materialize_observer_config(
    plan: FullCanaryPlan,
    *,
    collector: CollectorReadiness,
    edge_pid: int,
    edge_service_identity_sha256: str,
) -> Mapping[str, Any]:
    config = _observer_config_mapping(
        plan,
        collector=collector,
        edge_pid=edge_pid,
        edge_service_identity_sha256=edge_service_identity_sha256,
    )
    payload = _canonical_bytes(config)
    allowed_previous: set[str] = set()
    try:
        existing, _item = _read_stable_file(
            DEFAULT_OBSERVER_CONFIG,
            maximum=_MAX_CONFIG_BYTES,
            expected_uid=0,
            expected_gid=plan.identities.gateway_gid,
            allowed_modes=frozenset({0o440}),
        )
    except FileNotFoundError:
        existing = None
    if existing is not None:
        try:
            previous_value = _decode_json(existing, label="previous observer config")
        except ValueError:
            previous_value = None
        if _observer_static_binding_matches(previous_value, plan):
            allowed_previous.add(_sha256_bytes(existing))
    installed = _atomic_install_payload(
        plan,
        name="observer_config",
        path=DEFAULT_OBSERVER_CONFIG,
        payload=payload,
        mode=0o440,
        uid=0,
        gid=plan.identities.gateway_gid,
        allowed_previous=frozenset(allowed_previous),
    )
    return {
        **dict(installed),
        "config": copy.deepcopy(dict(config)),
        "collector_readiness_file_sha256": collector.file_sha256,
        "collector_service_identity_sha256": collector.service_identity_sha256,
        "edge_service_identity_sha256": edge_service_identity_sha256,
    }


def _socket_identity_sha256(
    path: Path,
    *,
    expected_uid: int,
    expected_gid: int,
) -> str:
    path = _absolute_path(path, "full-canary Unix socket")
    item = path.lstat()
    if (
        not stat.S_ISSOCK(item.st_mode)
        or item.st_nlink != 1
        or item.st_uid != expected_uid
        or item.st_gid != expected_gid
        or stat.S_IMODE(item.st_mode) != 0o660
    ):
        raise RuntimeError("full-canary Unix socket identity drifted")
    return _sha256_json(
        {
            "device": int(item.st_dev),
            "inode": int(item.st_ino),
            "mode": "0660",
            "owner_uid": int(item.st_uid),
            "owner_gid": int(item.st_gid),
        }
    )


def _process_owner_ids(pid: int) -> tuple[int, int]:
    if type(pid) is not int or pid <= 1:
        raise RuntimeError("full-canary process PID is invalid")
    item = (Path("/proc") / str(pid)).stat()
    return int(item.st_uid), int(item.st_gid)


def load_plugin_readiness(
    plan: FullCanaryPlan,
    *,
    collector: CollectorReadiness,
    gateway_pid: int,
    edge_pid: int,
    edge_service_identity_sha256: str,
    path: Path = DEFAULT_PLUGIN_READINESS_PATH,
) -> PluginReadiness:
    """Validate the collector's post-gateway authenticated plugin handoff."""

    if not isinstance(plan, FullCanaryPlan):
        raise TypeError("full-canary plan is required")
    if not isinstance(collector, CollectorReadiness):
        raise TypeError("full-canary collector readiness is required")
    if type(gateway_pid) is not int or gateway_pid <= 1:
        raise RuntimeError("plugin readiness gateway PID is invalid")
    if type(edge_pid) is not int or edge_pid <= 1:
        raise RuntimeError("plugin readiness edge PID is invalid")
    edge_service_identity_sha256 = _digest(
        edge_service_identity_sha256,
        "edge service identity digest",
    )
    path = _absolute_path(path, "plugin readiness receipt")
    if path != DEFAULT_PLUGIN_READINESS_PATH:
        raise RuntimeError("plugin readiness receipt path is not fixed")

    collector_raw, _collector_item = _read_stable_file(
        DEFAULT_COLLECTOR_READINESS_PATH,
        maximum=512 * 1024,
        expected_uid=0,
        expected_gid=0,
        allowed_modes=frozenset({0o400}),
    )
    if (
        _sha256_bytes(collector_raw) != collector.file_sha256
        or _decode_json(
            collector_raw,
            label="full-canary collector readiness",
        )
        != collector.receipt
    ):
        raise RuntimeError("collector readiness changed before plugin handoff")

    raw, _item = _read_stable_file(
        path,
        maximum=2 * 1024 * 1024,
        expected_uid=0,
        expected_gid=0,
        allowed_modes=frozenset({0o400}),
    )
    value = _decode_json(raw, label="full-canary plugin readiness")
    if raw != _canonical_bytes(value):
        raise RuntimeError("plugin readiness receipt is not canonical JSON")
    fields = {
        "schema",
        "full_canary_plan_sha256",
        "canary_run_id",
        "collector_readiness_file_sha256",
        "gateway_peer",
        "plugin_ready_frame",
        "plugin_ready_frame_sha256",
        "collector_hash_chain_head_sha256",
        "boot_id_sha256",
        "observed_at_unix",
        "observed_at_boottime_ns",
        "receipt_sha256",
    }
    if set(value) != fields or value.get("schema") != PLUGIN_READINESS_SCHEMA:
        raise RuntimeError("plugin readiness receipt fields are not exact")
    unsigned = {
        name: copy.deepcopy(item)
        for name, item in value.items()
        if name != "receipt_sha256"
    }
    fixture = _validated_e2e_fixture(plan)
    if (
        value.get("receipt_sha256") != _sha256_json(unsigned)
        or value.get("full_canary_plan_sha256") != plan.sha256
        or value.get("canary_run_id") != fixture.get("canary_run_id")
        or value.get("collector_readiness_file_sha256")
        != collector.file_sha256
    ):
        raise RuntimeError("plugin readiness plan/collector binding drifted")
    chain_head = _digest(
        value.get("collector_hash_chain_head_sha256"),
        "plugin readiness collector hash-chain head",
    )

    gateway_peer = value.get("gateway_peer")
    gateway_start_ticks = process_start_time_ticks(gateway_pid)
    process_uid, process_gid = _process_owner_ids(gateway_pid)
    if (
        not isinstance(gateway_peer, Mapping)
        or set(gateway_peer) != {"pid", "start_time_ticks", "uid", "gid"}
        or gateway_peer.get("pid") != gateway_pid
        or gateway_peer.get("start_time_ticks") != gateway_start_ticks
        or gateway_peer.get("uid") != plan.identities.gateway_uid
        or gateway_peer.get("gid") != plan.identities.gateway_gid
        or process_uid != plan.identities.gateway_uid
        or process_gid != plan.identities.gateway_gid
    ):
        raise RuntimeError("plugin readiness gateway peer is not exact/live")

    boot_sha256, now_boottime_ns = boot_identity()
    observed_boottime = value.get("observed_at_boottime_ns")
    observed_unix = value.get("observed_at_unix")
    if (
        value.get("boot_id_sha256") != boot_sha256
        or type(observed_boottime) is not int
        or observed_boottime < 0
        or not 0 <= now_boottime_ns - observed_boottime <= 60_000_000_000
        or type(observed_unix) is not int
        or observed_unix < 0
    ):
        raise RuntimeError("plugin readiness is stale or from another boot")

    frame = value.get("plugin_ready_frame")
    frame_fields = {
        "schema",
        "sequence",
        "event",
        "release_sha",
        "release_sha256",
        "canary_run_id",
        "case_id",
        "fixture_sha256",
        "collector_service_identity_sha256",
        "discord_edge_service_identity_sha256",
        "session_id",
        "turn_id",
        "observed_at_unix_ms",
        "payload",
    }
    if not isinstance(frame, Mapping) or set(frame) != frame_fields:
        raise RuntimeError("plugin-ready authenticated frame fields are not exact")
    frame_sha256 = _sha256_json(frame)
    frame_observed_ms = frame.get("observed_at_unix_ms")
    if (
        frame.get("schema") != PLUGIN_FRAME_SCHEMA
        or frame.get("sequence") != 1
        or frame.get("event") != "plugin_ready"
        or frame.get("release_sha") != plan.revision
        or frame.get("release_sha256") != plan.release["artifact_sha256"]
        or frame.get("canary_run_id") != fixture.get("canary_run_id")
        or frame.get("case_id") != fixture.get("case_id")
        or frame.get("fixture_sha256") != plan.artifacts["e2e_fixture"].sha256
        or frame.get("collector_service_identity_sha256")
        != collector.service_identity_sha256
        or frame.get("discord_edge_service_identity_sha256")
        != edge_service_identity_sha256
        or frame.get("session_id") is not None
        or frame.get("turn_id") is not None
        or type(frame_observed_ms) is not int
        or not fixture["valid_from_unix_ms"]
        <= frame_observed_ms
        <= fixture["valid_until_unix_ms"]
        or abs(observed_unix * 1000 - frame_observed_ms) > 60_000
        or value.get("plugin_ready_frame_sha256") != frame_sha256
    ):
        raise RuntimeError("plugin-ready authenticated frame binding drifted")

    observer_raw, _observer_item = _read_stable_file(
        DEFAULT_OBSERVER_CONFIG,
        maximum=_MAX_CONFIG_BYTES,
        expected_uid=0,
        expected_gid=plan.identities.gateway_gid,
        allowed_modes=frozenset({0o440}),
    )
    observer = _decode_json(observer_raw, label="live observer config")
    expected_observer = _observer_config_mapping(
        plan,
        collector=collector,
        edge_pid=edge_pid,
        edge_service_identity_sha256=edge_service_identity_sha256,
    )
    if observer != expected_observer or observer_raw != _canonical_bytes(observer):
        raise RuntimeError("plugin-ready observer config binding drifted")

    payload = frame.get("payload")
    payload_fields = {
        "plugin_name",
        "gateway_pid",
        "config_sha256",
        "fixture_sha256",
        "release_sha",
        "release_sha256",
        "api_session_key_sha256",
        "collector_service_identity_sha256",
        "collector_socket_identity_sha256",
        "discord_edge_service_identity_sha256",
        "discord_edge_socket_identity_sha256",
        "module_origin",
        "module_sha256",
    }
    if not isinstance(payload, Mapping) or set(payload) != payload_fields:
        raise RuntimeError("plugin-ready payload fields are not exact")
    module_origin = _absolute_path(payload.get("module_origin"), "canary plugin module")
    release_root = Path(plan.release["artifact_root"])
    try:
        relative_module = module_origin.relative_to(release_root)
    except ValueError as exc:
        raise RuntimeError("canary plugin module is outside sealed release") from exc
    observed_origin, observed_module_sha256 = module_file_identity(module_origin)
    expected_payload = {
        "plugin_name": CANARY_OBSERVER_PLUGIN,
        "gateway_pid": gateway_pid,
        "config_sha256": _sha256_bytes(observer_raw),
        "fixture_sha256": plan.artifacts["e2e_fixture"].sha256,
        "release_sha": plan.revision,
        "release_sha256": plan.release["artifact_sha256"],
        "api_session_key_sha256": fixture["api_session_key_sha256"],
        "collector_service_identity_sha256": collector.service_identity_sha256,
        "collector_socket_identity_sha256": _socket_identity_sha256(
            DEFAULT_COLLECTOR_SOCKET,
            expected_uid=0,
            expected_gid=plan.identities.gateway_gid,
        ),
        "discord_edge_service_identity_sha256": edge_service_identity_sha256,
        "discord_edge_socket_identity_sha256": _socket_identity_sha256(
            DEFAULT_EDGE_SOCKET,
            expected_uid=plan.identities.edge_uid,
            expected_gid=plan.identities.edge_gid,
        ),
        "module_origin": str(module_origin),
        "module_sha256": observed_module_sha256,
    }
    if (
        relative_module.parts[-3:]
        != ("plugins", "muncho_canary_evidence", "__init__.py")
        or observed_origin != str(module_origin)
        or dict(payload) != expected_payload
    ):
        raise RuntimeError("plugin-ready sealed module/config binding drifted")
    return PluginReadiness(
        receipt=copy.deepcopy(dict(value)),
        file_sha256=_sha256_bytes(raw),
        frame_sha256=frame_sha256,
        collector_hash_chain_head_sha256=chain_head,
    )


@dataclass(frozen=True)
class Command:
    argv: tuple[str, ...]
    timeout_seconds: int = _COMMAND_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        if (
            not self.argv
            or any(
                not isinstance(item, str)
                or not item
                or _CONTROL_RE.search(item) is not None
                for item in self.argv
            )
            or self.argv[0] in {"sh", "bash", "/bin/sh", "/bin/bash"}
            or any(item in {"enable", "reenable", "preset", "preset-all"} for item in self.argv)
        ):
            raise ValueError("full-canary command argv is invalid")


Runner = Callable[[Command], subprocess.CompletedProcess[bytes]]


def _runner(command: Command) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        list(command.argv),
        check=False,
        capture_output=True,
        shell=False,
        timeout=command.timeout_seconds,
        env={
            "HOME": "/root",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": "/usr/sbin:/usr/bin:/sbin:/bin",
            "TZ": "UTC",
        },
    )


def _run_checked(command: Command, *, runner: Runner, label: str) -> subprocess.CompletedProcess[bytes]:
    completed = runner(command)
    if not isinstance(completed, subprocess.CompletedProcess):
        raise TypeError("full-canary runner returned invalid result")
    if (
        len(completed.stdout) > _MAX_COMMAND_OUTPUT_BYTES
        or len(completed.stderr) > _MAX_COMMAND_OUTPUT_BYTES
    ):
        raise RuntimeError(f"{label} produced oversized output")
    if completed.returncode != 0:
        raise RuntimeError(
            f"{label} failed: rc={completed.returncode} "
            f"stdout_sha256={_sha256_bytes(completed.stdout)} "
            f"stderr_sha256={_sha256_bytes(completed.stderr)}"
        )
    return completed


def collect_service_state(unit: str, *, runner: Runner = _runner) -> Mapping[str, Any]:
    if unit not in {EDGE_UNIT_NAME, WRITER_UNIT_NAME, GATEWAY_UNIT_NAME}:
        raise ValueError("full-canary unit is not allowlisted")
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
        raise RuntimeError("systemd service state is not UTF-8") from exc
    values: dict[str, str] = {}
    for line in text.splitlines():
        if "=" not in line:
            raise RuntimeError("systemd service state is malformed")
        name, item = line.split("=", 1)
        if name in values:
            raise RuntimeError("systemd service state is ambiguous")
        values[name] = item
    if set(values) != set(_SERVICE_PROPERTIES):
        raise RuntimeError("systemd service state fields are not exact")
    try:
        main_pid = int(values.pop("MainPID"))
    except ValueError as exc:
        raise RuntimeError("systemd MainPID is invalid") from exc
    return {**values, "MainPID": main_pid}


def evaluate_service_states(
    states: Mapping[str, Mapping[str, Any]],
    *,
    phase: str,
) -> Mapping[str, bool]:
    if phase not in {"stopped", "live"}:
        raise ValueError("full-canary service phase is invalid")
    checks: dict[str, bool] = {}
    for unit, expected_path in (
        (EDGE_UNIT_NAME, DEFAULT_EDGE_UNIT_PATH),
        (WRITER_UNIT_NAME, DEFAULT_WRITER_UNIT_PATH),
        (GATEWAY_UNIT_NAME, DEFAULT_GATEWAY_UNIT_PATH),
    ):
        state = states.get(unit)
        if not isinstance(state, Mapping):
            checks[f"service.{unit}.present"] = False
            continue
        loaded = state.get("LoadState") == "loaded"
        absent = state.get("LoadState") == "not-found"
        disabled = state.get("UnitFileState") in {"disabled", ""}
        checks[f"service.{unit}.not_enabled"] = disabled
        checks[f"service.{unit}.no_dropins"] = state.get(
            "DropInPaths"
        ) in {"", "[]"}
        if phase == "live":
            checks[f"service.{unit}.loaded"] = loaded
            checks[f"service.{unit}.fragment"] = state.get("FragmentPath") == str(expected_path)
            checks[f"service.{unit}.active"] = (
                state.get("ActiveState") == "active"
                and state.get("SubState") == "running"
                and type(state.get("MainPID")) is int
                and state["MainPID"] > 1
                and state.get("Type") == "notify"
                and state.get("NotifyAccess") == "main"
            )
        else:
            checks[f"service.{unit}.stopped"] = (
                (absent or loaded)
                and state.get("ActiveState") in {"inactive", "failed"}
                and state.get("MainPID") == 0
            )
    return checks


class FullCanaryPreflightError(RuntimeError):
    def __init__(self, report: Mapping[str, Any]) -> None:
        self.report = copy.deepcopy(dict(report))
        super().__init__("full-canary preflight blocked")


def _readiness_receipt(
    path: Path,
    *,
    uid: int,
    gid: int,
) -> Mapping[str, Any]:
    raw, _item = _read_stable_file(
        path,
        maximum=512 * 1024,
        expected_uid=uid,
        expected_gid=gid,
        allowed_modes=frozenset({0o600}),
    )
    return _decode_json(raw, label="runtime readiness receipt")


def gateway_effective_environment_is_sealed(value: Any) -> bool:
    """Return whether the model gateway sees no external semantic selectors."""

    if (
        not isinstance(value, list)
        or value != sorted(set(value))
        or any(not isinstance(name, str) or not name for name in value)
    ):
        return False
    names = set(value)
    return (
        _GATEWAY_REQUIRED_EFFECTIVE_ENVIRONMENT_NAMES.issubset(names)
        and names.issubset(GATEWAY_ALLOWED_EFFECTIVE_ENVIRONMENT_NAMES)
        and not any(
            name in GATEWAY_FORBIDDEN_EFFECTIVE_ENVIRONMENT_NAMES
            or name.startswith("HERMES_KANBAN_")
            or (
                name.startswith("AUXILIARY_")
                and name.endswith(
                    ("_PROVIDER", "_MODEL", "_BASE_URL", "_API_KEY")
                )
            )
            for name in value
        )
    )


def _gateway_effective_environment_hashes_are_sealed(
    names: Any,
    hashes: Any,
    *,
    plan: FullCanaryPlan,
) -> bool:
    if (
        not gateway_effective_environment_is_sealed(names)
        or not isinstance(hashes, Mapping)
        or list(hashes) != names
        or any(
            not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None
            for value in hashes.values()
        )
    ):
        return False
    expected_values = {
        "CREDENTIALS_DIRECTORY": f"/run/credentials/{GATEWAY_UNIT_NAME}",
        "HERMES_CONFIG": str(DEFAULT_GATEWAY_CONFIG),
        "HERMES_EXEC_ASK": "1",
        "HERMES_HOME": str(DEFAULT_GATEWAY_PROFILE_HOME),
        "HERMES_MANAGED_DIR": str(DEFAULT_DISABLED_MANAGED_SCOPE),
        "HERMES_MAX_ITERATIONS": "90",
        "HERMES_QUIET": "1",
        "HOME": str(DEFAULT_GATEWAY_HOME),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "LOGNAME": plan.identities.gateway_user,
        "PATH": "/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "SHELL": "/usr/sbin/nologin",
        "SSL_CERT_FILE": str(DEFAULT_GATEWAY_CA_BUNDLE),
        "TERMINAL_CWD": str(DEFAULT_GATEWAY_HOME),
        "TZ": "UTC",
        "USER": plan.identities.gateway_user,
        "_HERMES_GATEWAY": "1",
    }
    return all(
        hashes[name] == _sha256_bytes(value.encode("utf-8"))
        for name, value in expected_values.items()
    )


def _validate_live_readiness(
    plan: FullCanaryPlan,
    states: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, bool]:
    checks: dict[str, bool] = {}
    writer = _readiness_receipt(
        DEFAULT_WRITER_RUNTIME_ATTESTATION_PATH,
        uid=plan.identities.writer_uid,
        gid=plan.identities.writer_gid,
    )
    gateway = _readiness_receipt(
        DEFAULT_GATEWAY_READINESS_PATH,
        uid=plan.identities.gateway_uid,
        gid=plan.identities.gateway_gid,
    )
    edge = _readiness_receipt(
        DEFAULT_EDGE_READINESS_PATH,
        uid=plan.identities.edge_uid,
        gid=plan.identities.edge_gid,
    )
    receipts = {
        WRITER_UNIT_NAME: (writer, WRITER_RUNTIME_ATTESTATION_VERSION, "writer_pid"),
        GATEWAY_UNIT_NAME: (gateway, READINESS_RECEIPT_VERSION, "gateway_pid"),
        EDGE_UNIT_NAME: (edge, EDGE_READINESS_SCHEMA, "edge_pid"),
    }
    release_root = Path(plan.release["artifact_root"])
    try:
        checks["readiness.gateway.sealed_startup_paths"] = (
            _gateway_startup_paths_are_sealed(plan, phase="live")
        )
    except Exception:
        checks["readiness.gateway.sealed_startup_paths"] = False
    for unit, (receipt, version, pid_field) in receipts.items():
        checks[f"readiness.{unit}.version"] = receipt.get("version") == version
        checks[f"readiness.{unit}.pid"] = receipt.get(pid_field) == states[unit].get("MainPID")
        environment_names = receipt.get("effective_environment_variable_names")
        checks[f"readiness.{unit}.no_discord_token_environment"] = (
            isinstance(environment_names, list)
            and "DISCORD_BOT_TOKEN" not in environment_names
        )
        if unit == GATEWAY_UNIT_NAME:
            checks["readiness.gateway.sealed_effective_environment"] = (
                _gateway_effective_environment_hashes_are_sealed(
                    environment_names,
                    receipt.get(
                        "effective_environment_variable_value_sha256"
                    ),
                    plan=plan,
                )
            )
        status = states[unit].get("StatusText")
        digest = readiness_receipt_sha256(receipt)
        if unit == WRITER_UNIT_NAME:
            checks[f"readiness.{unit}.systemd_status"] = isinstance(status, str) and (
                status == f"{version}:{digest}"
                or status.startswith("canonical-writer-liveness-v1:")
            )
            checks[f"readiness.{unit}.edge_authority"] = (
                receipt.get("discord_edge_authority_enabled") is True
            )
        else:
            checks[f"readiness.{unit}.systemd_status"] = status == f"{version}:{digest}"
    origins = gateway.get("loaded_module_origins")
    checks["readiness.gateway.immutable_code_closure"] = (
        isinstance(origins, list)
        and bool(origins)
        and all(
            isinstance(origin, str)
            and (Path(origin) == release_root or release_root in Path(origin).parents)
            for origin in origins
        )
    )
    checks["readiness.edge.public_only_targets"] = edge.get("allowed_target_types") == [
        "public_guild_channel",
        "public_guild_forum",
        "public_guild_thread",
    ] and edge.get("forbidden_target_types") == [
        "direct_message",
        "dm",
        "group_dm",
        "private_channel",
        "private_thread",
    ]
    checks["readiness.edge.config_digest"] = (
        edge.get("config_sha256") == plan.artifacts["edge_config"].sha256
    )
    collector: CollectorReadiness | None = None
    edge_identity_sha256 = ""
    try:
        edge_identity_sha256 = readiness_receipt_sha256(edge)
        collector = load_collector_readiness(
            plan,
            edge_pid=int(states[EDGE_UNIT_NAME]["MainPID"]),
            edge_service_identity_sha256=edge_identity_sha256,
        )
        expected_observer = _observer_config_mapping(
            plan,
            collector=collector,
            edge_pid=int(states[EDGE_UNIT_NAME]["MainPID"]),
            edge_service_identity_sha256=edge_identity_sha256,
        )
        observed_raw, _observer_item = _read_stable_file(
            DEFAULT_OBSERVER_CONFIG,
            maximum=_MAX_CONFIG_BYTES,
            expected_uid=0,
            expected_gid=plan.identities.gateway_gid,
            allowed_modes=frozenset({0o440}),
        )
        observed_observer = _decode_json(
            observed_raw,
            label="live observer config",
        )
        checks["readiness.collector.exact_live_peer"] = True
        checks["readiness.observer.exact_materialized_config"] = (
            observed_observer == expected_observer
        )
    except Exception:
        checks["readiness.collector.exact_live_peer"] = False
        checks["readiness.observer.exact_materialized_config"] = False
    if collector is not None:
        try:
            load_plugin_readiness(
                plan,
                collector=collector,
                gateway_pid=int(states[GATEWAY_UNIT_NAME]["MainPID"]),
                edge_pid=int(states[EDGE_UNIT_NAME]["MainPID"]),
                edge_service_identity_sha256=edge_identity_sha256,
            )
            checks["readiness.plugin.authenticated_registration"] = True
        except Exception:
            checks["readiness.plugin.authenticated_registration"] = False
    else:
        checks["readiness.plugin.authenticated_registration"] = False
    try:
        listener = _api_loopback_listener_identity(
            int(states[GATEWAY_UNIT_NAME]["MainPID"])
        )
    except Exception:
        listener = {}
    checks["readiness.gateway.api_loopback_main_pid"] = (
        listener.get("gateway_pid") == states[GATEWAY_UNIT_NAME].get("MainPID")
        and listener.get("host") == "127.0.0.1"
        and listener.get("port") == 8642
        and listener.get("protocol") == "tcp"
    )
    return checks


def _api_loopback_listener_identity(gateway_pid: int) -> Mapping[str, Any]:
    """Bind the exact gateway MainPID to its sole canary API listener."""
    if type(gateway_pid) is not int or gateway_pid <= 1:
        raise RuntimeError("gateway API listener PID is invalid")
    start_before = process_start_time_ticks(gateway_pid)
    fd_root = Path("/proc") / str(gateway_pid) / "fd"
    socket_inodes: set[int] = set()
    entries = list(fd_root.iterdir())
    if len(entries) > 4096:
        raise RuntimeError("gateway descriptor inventory exceeds its bound")
    for entry in entries:
        try:
            target = os.readlink(entry)
        except FileNotFoundError:
            continue
        match = re.fullmatch(r"socket:\[([1-9][0-9]*)\]", target)
        if match is not None:
            socket_inodes.add(int(match.group(1)))
    if not socket_inodes:
        raise RuntimeError("gateway owns no network sockets")

    matches: list[int] = []
    tcp_path = Path("/proc/net/tcp")
    raw = tcp_path.read_bytes()
    if len(raw) > 4 * 1024 * 1024:
        raise RuntimeError("kernel TCP inventory exceeds its bound")
    try:
        lines = raw.decode("ascii", errors="strict").splitlines()
    except UnicodeDecodeError as exc:
        raise RuntimeError("kernel TCP inventory is invalid") from exc
    for line in lines[1:]:
        fields = line.split()
        if len(fields) < 10 or fields[3] != "0A":
            continue
        local = fields[1].split(":", 1)
        if len(local) != 2:
            continue
        try:
            inode = int(fields[9])
            port = int(local[1], 16)
        except ValueError:
            continue
        if (
            local[0] == "0100007F"
            and port == 8642
            and inode in socket_inodes
        ):
            matches.append(inode)
    start_after = process_start_time_ticks(gateway_pid)
    current_targets: set[str] = set()
    for entry in fd_root.iterdir():
        try:
            current_targets.add(os.readlink(entry))
        except FileNotFoundError:
            continue
    if (
        start_before != start_after
        or len(matches) != 1
        or f"socket:[{matches[0]}]" not in current_targets
    ):
        raise RuntimeError("gateway API loopback listener identity is not exact")
    identity = {
        "schema": "muncho-full-canary-api-loopback-listener.v1",
        "gateway_pid": gateway_pid,
        "gateway_start_time_ticks": start_before,
        "protocol": "tcp",
        "host": "127.0.0.1",
        "port": 8642,
        "socket_inode": matches[0],
    }
    return {**identity, "identity_sha256": _sha256_json(identity)}


def _target_bindings(
    plan: FullCanaryPlan,
) -> Mapping[str, tuple[Path, bytes, int, int, int, frozenset[str]]]:
    writer_plan = ActivationPlan.from_mapping(plan.writer_activation_plan)
    config_payloads = {
        name: _validate_artifact_source(plan.artifacts[name], label=name)
        for name in ("writer_config", "gateway_config", "edge_config")
    }
    return {
        "writer_unit": (
            DEFAULT_WRITER_UNIT_PATH,
            plan.unit_bundle.writer_service.encode("utf-8"),
            0o644,
            0,
            0,
            frozenset({plan.allowed_previous_sha256["writer_unit"]}),
        ),
        "phase_b_readiness_unit": (
            DEFAULT_PHASE_B_READINESS_UNIT_PATH,
            writer_plan.unit_bundle.phase_b_readiness_service.encode("utf-8"),
            0o644,
            0,
            0,
            frozenset({
                writer_plan.install_artifacts["phase_b_readiness_unit"].sha256
            }),
        ),
        "gateway_unit": (
            DEFAULT_GATEWAY_UNIT_PATH,
            plan.unit_bundle.gateway_service.encode("utf-8"),
            0o644,
            0,
            0,
            frozenset({plan.allowed_previous_sha256["gateway_unit"]}),
        ),
        "edge_unit": (
            DEFAULT_EDGE_UNIT_PATH,
            plan.unit_bundle.edge_service.encode("utf-8"),
            0o644,
            0,
            0,
            frozenset(),
        ),
        "tmpfiles": (
            DEFAULT_TMPFILES_PATH,
            plan.unit_bundle.tmpfiles.encode("utf-8"),
            0o644,
            0,
            0,
            frozenset(),
        ),
        "writer_config": (
            DEFAULT_WRITER_CONFIG,
            config_payloads["writer_config"],
            plan.artifacts["writer_config"].mode,
            plan.artifacts["writer_config"].uid,
            plan.artifacts["writer_config"].gid,
            frozenset({plan.allowed_previous_sha256["writer_config"]}),
        ),
        "gateway_config": (
            DEFAULT_GATEWAY_CONFIG,
            config_payloads["gateway_config"],
            plan.artifacts["gateway_config"].mode,
            plan.artifacts["gateway_config"].uid,
            plan.artifacts["gateway_config"].gid,
            frozenset({plan.allowed_previous_sha256["gateway_config"]}),
        ),
        "edge_config": (
            DEFAULT_EDGE_CONFIG,
            config_payloads["edge_config"],
            plan.artifacts["edge_config"].mode,
            plan.artifacts["edge_config"].uid,
            plan.artifacts["edge_config"].gid,
            frozenset(),
        ),
    }


def _validate_target_state(plan: FullCanaryPlan, *, phase: str) -> Mapping[str, bool]:
    checks: dict[str, bool] = {}
    for name, (path, payload, mode, uid, gid, previous) in _target_bindings(plan).items():
        expected = _sha256_bytes(payload)
        try:
            raw, _item = _read_stable_file(
                path,
                maximum=max(len(payload), _MAX_CONFIG_BYTES),
                expected_uid=uid,
                expected_gid=gid,
                allowed_modes=frozenset({mode}),
            )
        except FileNotFoundError:
            checks[f"target.{name}.known"] = phase == "stopped"
            continue
        observed = _sha256_bytes(raw)
        allowed = {expected} if phase == "live" else {expected, *previous}
        known = observed in allowed
        if name == "writer_config" and not known:
            try:
                _retired_payload, retired_sha256 = (
                    _retired_writer_config_payload(plan)
                )
                if observed == retired_sha256:
                    _validate_bootstrap_retirement_tombstone(
                        plan,
                        expected_retired_sha256=retired_sha256,
                    )
                    known = True
            except Exception:
                known = False
        checks[f"target.{name}.known"] = known
    return checks


def _writer_receipt_matches_plan(path: Path, plan: FullCanaryPlan) -> bool:
    raw, _item = _read_stable_file(
        path,
        maximum=_MAX_JSON_BYTES,
        expected_uid=0,
        expected_gid=0,
        allowed_modes=frozenset({0o400}),
    )
    if _sha256_bytes(raw) != plan.writer_activation_receipt_file_sha256:
        return False
    observed = _decode_json(raw, label="writer-only activation receipt")
    return observed == plan.writer_activation_receipt


def collect_full_canary_preflight(
    plan: FullCanaryPlan,
    *,
    phase: str,
    runner: Runner = _runner,
    metadata_reader: Callable[[str], bytes | str] | None = None,
    local_identity_reader: Callable[[str], bytes | str] | None = None,
    bootstrap_reconciliation_evidence: (
        BootstrapReconciliationEvidence
        | BootstrapNeverAuthorizedEvidence
        | None
    ) = None,
) -> Mapping[str, Any]:
    """Collect and evaluate current exact state without repairing anything."""

    if not isinstance(plan, FullCanaryPlan):
        raise TypeError("full-canary plan is required")
    if phase not in {"stopped", "live"}:
        raise ValueError("full-canary preflight phase is invalid")
    if bootstrap_reconciliation_evidence is not None:
        raise RuntimeError("legacy bootstrap evidence is not startup authority")
    checks: dict[str, bool] = {}
    blockers: list[str] = []

    # This gate is deliberately first.  In particular, no systemctl runner is
    # reached on production, on another VM, or after a boot/host replacement.
    try:
        host_binding = validate_dedicated_canary_host(
            plan,
            metadata_reader=metadata_reader,
            local_identity_reader=local_identity_reader,
        )
    except Exception:
        host_binding = {}
    checks["host.dedicated_canary_exact"] = bool(host_binding)
    if not host_binding:
        report = {
            "schema": FULL_CANARY_PREFLIGHT_SCHEMA,
            "revision": plan.revision,
            "full_canary_plan_sha256": plan.sha256,
            "phase": phase,
            "checks": dict(sorted(checks.items())),
            "blockers": ["host.dedicated_canary_exact"],
            "ok": False,
            "observed_at_unix": int(time.time()),
        }
        report["report_sha256"] = _sha256_json(report)
        raise FullCanaryPreflightError(report)

    def check(name: str, operation: Callable[[], bool]) -> None:
        try:
            checks[name] = operation() is True
        except Exception:
            checks[name] = False
        if not checks[name]:
            blockers.append(name)

    def phase_b_readiness_is_current_descendant() -> bool:
        from gateway.canonical_writer_phase_b_runtime import (
            validate_fixed_phase_b_readiness_descendant,
            validate_fixed_phase_b_readiness_lineage,
        )

        return bool(
            (
                validate_fixed_phase_b_readiness_descendant
                if phase == "live"
                else validate_fixed_phase_b_readiness_lineage
            )(plan.phase_b_readiness_anchor)
        )

    check(
        (
            "phase_b.readiness_current_descendant"
            if phase == "live"
            else "phase_b.readiness_lineage"
        ),
        phase_b_readiness_is_current_descendant,
    )
    check(
        "gateway.sealed_startup_paths",
        lambda: _gateway_startup_paths_are_sealed(plan, phase=phase),
    )
    check("release.manifest", lambda: bool(_validate_release_manifest(plan)))
    writer_receipt_path = Path(plan.writer_activation_receipt["activation_receipt_path"])
    check(
        "writer_only.receipt",
        lambda: _writer_receipt_matches_plan(writer_receipt_path, plan),
    )
    check(
        "credential.api_server.metadata",
        lambda: _validate_secret_source_metadata(
            DEFAULT_API_SERVER_CONTROL_KEY,
            expected_uid=0,
            expected_gid=0,
            expected_mode=0o400,
            maximum_bytes=8 * 1024,
        ),
    )
    check(
        "credential.discord_bot_token.metadata",
        lambda: _validate_secret_source_metadata(
            DEFAULT_EDGE_TOKEN_PATH,
            expected_uid=plan.identities.edge_uid,
            expected_gid=plan.identities.edge_gid,
            expected_mode=0o400,
            maximum_bytes=64 * 1024,
        ),
    )
    check(
        "credential.openai_codex_auth_store.metadata",
        lambda: _validate_secret_source_metadata(
            DEFAULT_GATEWAY_AUTH_STORE,
            expected_uid=plan.identities.gateway_uid,
            expected_gid=plan.identities.gateway_gid,
            expected_mode=0o600,
            maximum_bytes=2 * 1024 * 1024,
        ),
    )
    config_readers = {
        "writer_config": lambda raw: _validate_writer_config(
            raw,
            plan.identities,
        ),
        "gateway_config": _validate_gateway_config,
        "edge_config": lambda raw: _validate_edge_config(raw, plan.identities),
        "e2e_fixture": lambda raw: _decode_json(raw, label="E2E fixture"),
    }
    for name, validator in config_readers.items():
        check(
            f"artifact.{name}",
            lambda name=name, validator=validator: bool(
                validator(_validate_artifact_source(plan.artifacts[name], label=name))
            ),
        )
    try:
        target_checks = _validate_target_state(plan, phase=phase)
    except Exception:
        target_checks = {"target.inventory": False}
    for name, result in target_checks.items():
        checks[name] = result
        if not result:
            blockers.append(name)
    states: dict[str, Mapping[str, Any]] = {}
    for unit in (EDGE_UNIT_NAME, WRITER_UNIT_NAME, GATEWAY_UNIT_NAME):
        try:
            states[unit] = collect_service_state(unit, runner=runner)
        except Exception:
            states[unit] = {}
    service_checks = evaluate_service_states(states, phase=phase)
    for name, result in service_checks.items():
        checks[name] = result
        if not result:
            blockers.append(name)
    if phase == "live" and all(states.values()):
        try:
            readiness_checks = _validate_live_readiness(plan, states)
        except Exception:
            readiness_checks = {"readiness.receipts": False}
        for name, result in readiness_checks.items():
            checks[name] = result
            if not result:
                blockers.append(name)
    report = {
        "schema": FULL_CANARY_PREFLIGHT_SCHEMA,
        "revision": plan.revision,
        "full_canary_plan_sha256": plan.sha256,
        "phase": phase,
        "writer_authority": "phase_b_persistent_writer",
        "checks": dict(sorted(checks.items())),
        "blockers": sorted(set(blockers)),
        "ok": not blockers,
        "observed_at_unix": int(time.time()),
    }
    report["report_sha256"] = _sha256_json(report)
    if blockers:
        raise FullCanaryPreflightError(report)
    return report


def edge_start_command() -> Command:
    """Start only the edge; the collector/config gate must follow."""
    return Command((SYSTEMCTL, "start", EDGE_UNIT_NAME), timeout_seconds=180)


def phase_b_readiness_start_command() -> Command:
    """Run the fixed root readiness oneshot while every live service is stopped."""

    return Command(
        (SYSTEMCTL, "start", PHASE_B_READINESS_UNIT_NAME),
        timeout_seconds=180,
    )


def post_collector_start_commands() -> tuple[Command, ...]:
    """Start writer then gateway only after collector/config verification."""
    return tuple(
        Command((SYSTEMCTL, "start", unit), timeout_seconds=180)
        for unit in (WRITER_UNIT_NAME, GATEWAY_UNIT_NAME)
    )


def _await_collector_readiness(
    plan: FullCanaryPlan,
    *,
    edge_pid: int,
    edge_service_identity_sha256: str,
    timeout_seconds: float = 30.0,
) -> CollectorReadiness:
    if not 0 < timeout_seconds <= 60:
        raise ValueError("collector readiness timeout is out of bounds")
    deadline = time.monotonic() + timeout_seconds
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        try:
            return load_collector_readiness(
                plan,
                edge_pid=edge_pid,
                edge_service_identity_sha256=edge_service_identity_sha256,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            last_error = exc
        time.sleep(0.1)
    raise RuntimeError("root live collector did not publish exact readiness") from last_error


def _await_plugin_readiness(
    plan: FullCanaryPlan,
    *,
    collector: CollectorReadiness,
    gateway_pid: int,
    edge_pid: int,
    edge_service_identity_sha256: str,
    timeout_seconds: float = 30.0,
) -> PluginReadiness:
    if not 0 < timeout_seconds <= 60:
        raise ValueError("plugin readiness timeout is out of bounds")
    deadline = time.monotonic() + timeout_seconds
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        try:
            return load_plugin_readiness(
                plan,
                collector=collector,
                gateway_pid=gateway_pid,
                edge_pid=edge_pid,
                edge_service_identity_sha256=edge_service_identity_sha256,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            last_error = exc
        time.sleep(0.1)
    raise RuntimeError(
        "root collector did not publish exact authenticated plugin readiness"
    ) from last_error


def stop_service_commands() -> tuple[Command, ...]:
    """Return the fail-safe reverse order used by every cleanup path."""

    return tuple(
        Command((SYSTEMCTL, "stop", unit), timeout_seconds=120)
        for unit in FULL_CANARY_STOP_ORDER
    )


def _require_root_linux() -> None:
    getter = getattr(os, "geteuid", None)
    if not callable(getter) or getter() != 0:
        raise PermissionError("full-canary lifecycle requires UID 0")
    if sys.platform != "linux":
        raise RuntimeError("full-canary lifecycle requires Linux")


def _open_root_directory_chain(
    path: Path,
    *,
    create: bool,
    mode: int = 0o700,
    expected_uid: int = 0,
    expected_gid: int = 0,
    trusted_anchor: Path = Path("/"),
) -> int:
    """Open one absolute directory through a no-follow, controlled chain.

    Each component is resolved relative to the already-open parent.  This is
    deliberately stronger than checking only the terminal directory: an
    existing symlink ancestor must never redirect a fixed evidence path.
    The returned descriptor owns the terminal directory and must be closed by
    the caller.
    """

    path = _absolute_path(path, "full-canary directory")
    trusted_anchor = _absolute_path(
        trusted_anchor,
        "full-canary trusted directory anchor",
    )
    try:
        relative = path.relative_to(trusted_anchor)
    except ValueError as exc:
        raise ValueError(
            "full-canary directory is outside its trusted anchor"
        ) from exc
    if type(create) is not bool:
        raise TypeError("full-canary directory creation policy is invalid")
    if (
        type(expected_uid) is not int
        or type(expected_gid) is not int
        or expected_uid < 0
        or expected_gid < 0
    ):
        raise ValueError("full-canary directory ownership is invalid")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(trusted_anchor, flags)
    try:
        anchor_item = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(anchor_item.st_mode)
            or anchor_item.st_uid != expected_uid
            or anchor_item.st_gid != expected_gid
            or stat.S_IMODE(anchor_item.st_mode) & 0o022
        ):
            raise RuntimeError(
                "full-canary trusted directory anchor is not controlled"
            )
        for component in relative.parts:
            created = False
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(component, mode, dir_fd=descriptor)
                os.fsync(descriptor)
                child = os.open(component, flags, dir_fd=descriptor)
                created = True
            except OSError as exc:
                raise RuntimeError(
                    "full-canary directory chain contains a symlink or invalid component"
                ) from exc
            try:
                item = os.fstat(child)
                if (
                    not stat.S_ISDIR(item.st_mode)
                    or item.st_uid != expected_uid
                    or item.st_gid != expected_gid
                    or stat.S_IMODE(item.st_mode) & 0o022
                ):
                    raise RuntimeError(
                        "full-canary evidence directory chain is not controlled"
                    )
                if created:
                    os.fsync(child)
            except BaseException:
                os.close(child)
                raise
            os.close(descriptor)
            descriptor = child
        root_item = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(root_item.st_mode)
            or root_item.st_uid != expected_uid
            or root_item.st_gid != expected_gid
            or stat.S_IMODE(root_item.st_mode) & 0o022
        ):
            raise RuntimeError(
                "full-canary evidence directory chain is not controlled"
            )
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _validate_root_directory_chain(path: Path) -> None:
    descriptor = _open_root_directory_chain(path, create=False)
    os.close(descriptor)


def _ensure_root_directory(path: Path, *, mode: int = 0o700) -> None:
    descriptor = _open_root_directory_chain(
        path,
        create=True,
        mode=mode,
    )
    os.close(descriptor)


def _directory_descriptor_identity(descriptor: int) -> tuple[int, int]:
    item = os.fstat(descriptor)
    if not stat.S_ISDIR(item.st_mode):
        raise RuntimeError("full-canary directory descriptor is invalid")
    return item.st_dev, item.st_ino


def _revalidate_root_directory_reachability(
    path: Path,
    descriptor: int,
    *,
    expected_uid: int = 0,
    expected_gid: int = 0,
    trusted_anchor: Path = Path("/"),
) -> None:
    """Prove that a held directory is still canonically reachable."""

    held = _directory_descriptor_identity(descriptor)
    reopened = _open_root_directory_chain(
        path,
        create=False,
        expected_uid=expected_uid,
        expected_gid=expected_gid,
        trusted_anchor=trusted_anchor,
    )
    try:
        if _directory_descriptor_identity(reopened) != held:
            raise RuntimeError(
                "full-canary evidence directory reachability changed"
            )
    finally:
        os.close(reopened)


def _entry_name(name: str) -> str:
    if (
        not isinstance(name, str)
        or not name
        or name in {".", ".."}
        or Path(name).name != name
        or "/" in name
    ):
        raise ValueError("full-canary evidence entry name is invalid")
    return name


def _read_stable_file_at(
    parent_descriptor: int,
    name: str,
    *,
    maximum: int,
    expected_uid: int | None = None,
    expected_gid: int | None = None,
    allowed_modes: frozenset[int] | None = None,
    allowed_link_counts: frozenset[int] = frozenset({1}),
) -> tuple[bytes, os.stat_result]:
    """Read a stable file relative to one already-validated parent fd."""

    name = _entry_name(name)
    before = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or before.st_nlink not in allowed_link_counts
        or not 0 <= before.st_size <= maximum
        or (expected_uid is not None and before.st_uid != expected_uid)
        or (expected_gid is not None and before.st_gid != expected_gid)
        or (
            allowed_modes is not None
            and stat.S_IMODE(before.st_mode) not in allowed_modes
        )
    ):
        raise RuntimeError("trusted full-canary file identity is invalid")
    descriptor = os.open(
        name,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=parent_descriptor,
    )
    chunks: list[bytes] = []
    total = 0
    try:
        opened = os.fstat(descriptor)
        while total <= maximum:
            chunk = os.read(descriptor, min(64 * 1024, maximum + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    reachable = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)

    def identity(item: os.stat_result) -> tuple[int, ...]:
        return (
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
        total > maximum
        or total != before.st_size
        or identity(before) != identity(opened)
        or identity(before) != identity(after)
        or identity(before) != identity(reachable)
    ):
        raise RuntimeError("trusted full-canary file changed during read")
    return b"".join(chunks), before


def _read_root_file_via_parent(
    path: Path,
    *,
    maximum: int,
    expected_uid: int = 0,
    expected_gid: int = 0,
    allowed_modes: frozenset[int] | None = None,
    allowed_link_counts: frozenset[int] = frozenset({1}),
    trusted_anchor: Path = Path("/"),
) -> tuple[bytes, os.stat_result]:
    """Read through a held controlled parent and revalidate reachability."""

    path = _absolute_path(path, "full-canary evidence path")
    parent_descriptor = _open_root_directory_chain(
        path.parent,
        create=False,
        expected_uid=expected_uid,
        expected_gid=expected_gid,
        trusted_anchor=trusted_anchor,
    )
    try:
        result = _read_stable_file_at(
            parent_descriptor,
            path.name,
            maximum=maximum,
            expected_uid=expected_uid,
            expected_gid=expected_gid,
            allowed_modes=allowed_modes,
            allowed_link_counts=allowed_link_counts,
        )
        _revalidate_root_directory_reachability(
            path.parent,
            parent_descriptor,
            expected_uid=expected_uid,
            expected_gid=expected_gid,
            trusted_anchor=trusted_anchor,
        )
        return result
    finally:
        os.close(parent_descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _exclusive_publication_temp_name(name: str) -> str:
    """Return the bounded, deterministic sibling used for one final name."""

    name = _entry_name(name)
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()
    return f".exclusive-{digest}.tmp"


def _optional_stat_at(parent_descriptor: int, name: str) -> os.stat_result | None:
    try:
        return os.stat(
            _entry_name(name),
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return None


def _publication_identity(item: os.stat_result) -> tuple[int, ...]:
    return (
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


def _unlink_stable_publication_entry_at(
    parent_descriptor: int,
    name: str,
    expected: os.stat_result,
) -> None:
    """Unlink only the still-reachable entry observed by the caller."""

    current = _optional_stat_at(parent_descriptor, name)
    if current is None:
        return
    if _publication_identity(current) != _publication_identity(expected):
        raise RuntimeError("full-canary evidence publication changed during cleanup")
    try:
        os.unlink(_entry_name(name), dir_fd=parent_descriptor)
    except FileNotFoundError:
        # A concurrent retry may already have completed the same bounded
        # cleanup.  The caller always revalidates the resulting state.
        return


def _rename_noreplace_at(
    parent_descriptor: int,
    source_name: str,
    destination_name: str,
) -> None:
    """Atomically publish a sibling without replacing an existing final."""

    source_name = _entry_name(source_name)
    destination_name = _entry_name(destination_name)
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        try:
            renameat2 = libc.renameat2
        except AttributeError as exc:  # pragma: no cover - modern glibc exports it
            raise RuntimeError("renameat2(RENAME_NOREPLACE) is unavailable") from exc
        renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
        ctypes.set_errno(0)
        result = renameat2(
            parent_descriptor,
            os.fsencode(source_name),
            parent_descriptor,
            os.fsencode(destination_name),
            1,  # RENAME_NOREPLACE
        )
        if result == 0:
            return
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise FileExistsError(
                error_number,
                os.strerror(error_number),
                destination_name,
            )
        raise OSError(error_number, os.strerror(error_number), destination_name)

    # Test/development hosts may not expose Linux renameat2.  A hard link has
    # the same create-only property.  The caller durably removes the temporary
    # name; if the process dies between those operations, retry recognizes only
    # the exact same inode with nlink == 2 and finishes that bounded cleanup.
    os.link(
        source_name,
        destination_name,
        src_dir_fd=parent_descriptor,
        dst_dir_fd=parent_descriptor,
        follow_symlinks=False,
    )


def _read_publication_entry_at(
    parent_descriptor: int,
    name: str,
    *,
    payload: bytes,
    mode: int,
    expected_uid: int,
    expected_gid: int,
    allowed_link_counts: frozenset[int] = frozenset({1}),
) -> tuple[str, os.stat_result]:
    raw, item = _read_stable_file_at(
        parent_descriptor,
        name,
        maximum=len(payload),
        expected_uid=expected_uid,
        expected_gid=expected_gid,
        allowed_modes=frozenset({mode}),
        allowed_link_counts=allowed_link_counts,
    )
    if raw == payload:
        return "exact", item
    if len(raw) < len(payload) and payload.startswith(raw):
        return "prefix", item
    raise RuntimeError("full-canary evidence publication payload drifted")


def _fsync_stable_publication_entry_at(
    parent_descriptor: int,
    name: str,
    expected: os.stat_result,
) -> os.stat_result:
    """Durably flush one still-reachable, identity-stable publication inode."""

    name = _entry_name(name)
    descriptor = os.open(
        name,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=parent_descriptor,
    )
    try:
        opened = os.fstat(descriptor)
        if _publication_identity(opened) != _publication_identity(expected):
            raise RuntimeError(
                "full-canary evidence publication changed before file fsync"
            )
        os.fsync(descriptor)
        flushed = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    reachable = _optional_stat_at(parent_descriptor, name)
    if (
        reachable is None
        or _publication_identity(flushed) != _publication_identity(opened)
        or _publication_identity(reachable) != _publication_identity(flushed)
    ):
        raise RuntimeError(
            "full-canary evidence publication changed during file fsync"
        )
    return flushed


def _write_exclusive_bytes_at(
    parent_descriptor: int,
    name: str,
    payload: bytes,
    *,
    mode: int,
    expected_uid: int,
    expected_gid: int,
) -> None:
    """Crash-atomically create one exact file below a validated parent fd."""

    name = _entry_name(name)
    if not isinstance(payload, bytes):
        raise TypeError("full-canary evidence payload must be bytes")
    if not isinstance(mode, int) or mode < 0 or mode > 0o7777:
        raise ValueError("full-canary evidence mode is invalid")
    temp_name = _exclusive_publication_temp_name(name)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )

    fcntl.flock(parent_descriptor, fcntl.LOCK_EX)
    try:
        final_item = _optional_stat_at(parent_descriptor, name)
        temp_item = _optional_stat_at(parent_descriptor, temp_name)

        if (
            final_item is not None
            and temp_item is not None
            and (final_item.st_dev, final_item.st_ino)
            == (temp_item.st_dev, temp_item.st_ino)
        ):
            state, linked_item = _read_publication_entry_at(
                parent_descriptor,
                name,
                payload=payload,
                mode=mode,
                expected_uid=expected_uid,
                expected_gid=expected_gid,
                allowed_link_counts=frozenset({2}),
            )
            if state != "exact" or temp_item.st_nlink != 2:
                raise RuntimeError(
                    "full-canary evidence linked publication is invalid"
                )
            linked_item = _fsync_stable_publication_entry_at(
                parent_descriptor,
                name,
                linked_item,
            )
            _unlink_stable_publication_entry_at(
                parent_descriptor,
                temp_name,
                linked_item,
            )
            os.fsync(parent_descriptor)
            final_item = _optional_stat_at(parent_descriptor, name)
            temp_item = _optional_stat_at(parent_descriptor, temp_name)

        final_state: str | None = None
        if final_item is not None:
            final_state, final_item = _read_publication_entry_at(
                parent_descriptor,
                name,
                payload=payload,
                mode=mode,
                expected_uid=expected_uid,
                expected_gid=expected_gid,
            )
        temp_state: str | None = None
        if temp_item is not None:
            temp_state, temp_item = _read_publication_entry_at(
                parent_descriptor,
                temp_name,
                payload=payload,
                mode=mode,
                expected_uid=expected_uid,
                expected_gid=expected_gid,
            )

        if final_state == "exact":
            if temp_item is not None:
                _unlink_stable_publication_entry_at(
                    parent_descriptor,
                    temp_name,
                    temp_item,
                )
            # This also commits a preceding retry whose rename reached the
            # final name but whose directory fsync had not yet completed.
            os.fsync(parent_descriptor)
            final_state, _ = _read_publication_entry_at(
                parent_descriptor,
                name,
                payload=payload,
                mode=mode,
                expected_uid=expected_uid,
                expected_gid=expected_gid,
            )
            if final_state != "exact" or _optional_stat_at(parent_descriptor, temp_name):
                raise RuntimeError("full-canary evidence publication did not converge")
            final_item = _optional_stat_at(parent_descriptor, name)
            if final_item is None:
                raise RuntimeError("full-canary evidence publication disappeared")
            _fsync_stable_publication_entry_at(
                parent_descriptor,
                name,
                final_item,
            )
            return

        if final_item is not None:
            # Only a strict prefix with exact provenance can be a legacy crash
            # from the former direct-to-final O_EXCL writer.
            if final_state != "prefix":
                raise RuntimeError("full-canary evidence publication is invalid")
            _unlink_stable_publication_entry_at(parent_descriptor, name, final_item)
            os.fsync(parent_descriptor)
            if _optional_stat_at(parent_descriptor, name) is not None:
                raise RuntimeError("full-canary evidence final cleanup did not converge")

        if temp_item is not None and temp_state == "prefix":
            _unlink_stable_publication_entry_at(
                parent_descriptor,
                temp_name,
                temp_item,
            )
            os.fsync(parent_descriptor)
            if _optional_stat_at(parent_descriptor, temp_name) is not None:
                raise RuntimeError("full-canary evidence temp cleanup did not converge")
            temp_item = None
            temp_state = None

        if temp_item is None:
            descriptor = os.open(
                temp_name,
                flags,
                mode,
                dir_fd=parent_descriptor,
            )
            try:
                offset = 0
                while offset < len(payload):
                    written = os.write(descriptor, payload[offset:])
                    if written <= 0:
                        raise RuntimeError(
                            "full-canary evidence write made no progress"
                        )
                    offset += written
                os.fchown(descriptor, expected_uid, expected_gid)
                os.fchmod(descriptor, mode)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            temp_state, temp_item = _read_publication_entry_at(
                parent_descriptor,
                temp_name,
                payload=payload,
                mode=mode,
                expected_uid=expected_uid,
                expected_gid=expected_gid,
            )
        if temp_state != "exact" or temp_item is None:
            raise RuntimeError("full-canary evidence temp publication is incomplete")
        temp_item = _fsync_stable_publication_entry_at(
            parent_descriptor,
            temp_name,
            temp_item,
        )

        published_item = temp_item
        published_here = False
        try:
            _rename_noreplace_at(parent_descriptor, temp_name, name)
            published_here = True
        except FileExistsError:
            # A cooperative retry is serialized by the directory lock.  Still
            # handle an independently published exact final without replacing
            # it; any other state remains a hard failure.
            final_state, _ = _read_publication_entry_at(
                parent_descriptor,
                name,
                payload=payload,
                mode=mode,
                expected_uid=expected_uid,
                expected_gid=expected_gid,
            )
            if final_state != "exact":
                raise RuntimeError("full-canary evidence final already exists")
        os.fsync(parent_descriptor)

        # Linux renameat2 consumes the temp name.  The non-Linux hard-link
        # fallback leaves it reachable until this durable cleanup.
        remaining_temp = _optional_stat_at(parent_descriptor, temp_name)
        if remaining_temp is not None:
            _unlink_stable_publication_entry_at(
                parent_descriptor,
                temp_name,
                remaining_temp,
            )
            os.fsync(parent_descriptor)
        final_state, final_item = _read_publication_entry_at(
            parent_descriptor,
            name,
            payload=payload,
            mode=mode,
            expected_uid=expected_uid,
            expected_gid=expected_gid,
        )
        if (
            final_state != "exact"
            or _optional_stat_at(parent_descriptor, temp_name)
            or (
                published_here
                and (final_item.st_dev, final_item.st_ino)
                != (published_item.st_dev, published_item.st_ino)
            )
        ):
            raise RuntimeError("full-canary evidence publication did not converge")
        _fsync_stable_publication_entry_at(
            parent_descriptor,
            name,
            final_item,
        )
    finally:
        fcntl.flock(parent_descriptor, fcntl.LOCK_UN)


def _write_exclusive_bytes(path: Path, payload: bytes, *, mode: int = 0o400) -> None:
    path = _absolute_path(path, "full-canary evidence path")
    parent_descriptor = _open_root_directory_chain(path.parent, create=True)
    try:
        _write_exclusive_bytes_at(
            parent_descriptor,
            path.name,
            payload,
            mode=mode,
            expected_uid=0,
            expected_gid=0,
        )
        _revalidate_root_directory_reachability(
            path.parent,
            parent_descriptor,
        )
    finally:
        os.close(parent_descriptor)


def _write_append_only_receipt(
    plan: FullCanaryPlan,
    *,
    stage: str,
    value: Mapping[str, Any],
) -> Mapping[str, Any]:
    if stage not in {"started", "verified_stopped", "stopped", "failure"}:
        raise ValueError("full-canary receipt stage is invalid")
    directory = DEFAULT_EVIDENCE_ROOT / "plans" / plan.revision / plan.sha256 / stage
    _ensure_root_directory(directory)
    receipt_path = directory / f"{time.time_ns()}-{os.getpid()}-{uuid.uuid4().hex}.json"
    unsigned = {
        **copy.deepcopy(dict(value)),
        "schema": FULL_CANARY_RECEIPT_SCHEMA,
        "stage": stage,
        "revision": plan.revision,
        "full_canary_plan_sha256": plan.sha256,
        "receipt_path": str(receipt_path),
    }
    receipt = {**unsigned, "receipt_sha256": _sha256_json(unsigned)}
    _write_exclusive_bytes(receipt_path, _canonical_bytes(receipt), mode=0o400)
    return receipt


@contextmanager
def _lifecycle_lock(path: Path = DEFAULT_LOCK_PATH):
    descriptor = os.open(
        path,
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        item = os.fstat(descriptor)
        if (
            not stat.S_ISREG(item.st_mode)
            or item.st_uid != 0
            or item.st_gid != 0
            or stat.S_IMODE(item.st_mode) != 0o600
        ):
            raise RuntimeError("full-canary lifecycle lock is not root-controlled")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _archive_existing_target(
    plan: FullCanaryPlan,
    *,
    name: str,
    path: Path,
    raw: bytes,
) -> Mapping[str, str]:
    digest = _sha256_bytes(raw)
    archive = (
        DEFAULT_EVIDENCE_ROOT
        / "plans"
        / plan.revision
        / plan.sha256
        / "prior-state"
        / name
        / f"{digest}.bin"
    )
    if archive.exists():
        archived, _item = _read_stable_file(
            archive,
            maximum=max(len(raw), 1),
            expected_uid=0,
            expected_gid=0,
            allowed_modes=frozenset({0o400}),
        )
        if archived != raw:
            raise RuntimeError("full-canary prior-state archive conflicts")
    else:
        _write_exclusive_bytes(archive, raw, mode=0o400)
    return {"path": str(archive), "sha256": digest}


def _atomic_install_payload(
    plan: FullCanaryPlan,
    *,
    name: str,
    path: Path,
    payload: bytes,
    mode: int,
    uid: int,
    gid: int,
    allowed_previous: frozenset[str],
) -> Mapping[str, Any]:
    expected = _sha256_bytes(payload)
    previous: Mapping[str, str] | None = None
    try:
        current, _item = _read_stable_file(
            path,
            maximum=max(len(payload), _MAX_CONFIG_BYTES),
            expected_uid=uid,
            expected_gid=gid,
            allowed_modes=frozenset({mode}),
        )
    except FileNotFoundError:
        current = None
    if current is not None:
        observed = _sha256_bytes(current)
        if observed == expected:
            return {
                "path": str(path),
                "sha256": expected,
                "changed": False,
                "prior_state": None,
            }
        if observed not in allowed_previous:
            raise RuntimeError(f"full-canary fixed target collision: {name}")
        previous = _archive_existing_target(plan, name=name, path=path, raw=current)
    path.parent.mkdir(parents=True, exist_ok=True)
    parent = path.parent.lstat()
    if (
        not stat.S_ISDIR(parent.st_mode)
        or stat.S_ISLNK(parent.st_mode)
        or parent.st_uid != 0
        or stat.S_IMODE(parent.st_mode) & 0o022
    ):
        raise RuntimeError("full-canary target parent is not root-controlled")
    temporary = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
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
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise RuntimeError("full-canary artifact write made no progress")
            offset += written
        os.fchmod(descriptor, mode)
        os.fchown(descriptor, uid, gid)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    _fsync_directory(path.parent)
    installed, _item = _read_stable_file(
        path,
        maximum=max(len(payload), 1),
        expected_uid=uid,
        expected_gid=gid,
        allowed_modes=frozenset({mode}),
    )
    if installed != payload:
        raise RuntimeError("full-canary artifact readback failed")
    return {
        "path": str(path),
        "sha256": expected,
        "changed": True,
        "prior_state": previous,
    }


def _install_plan_artifacts(plan: FullCanaryPlan) -> Mapping[str, Any]:
    installed: dict[str, Any] = {}
    for name, (path, payload, mode, uid, gid, previous) in _target_bindings(plan).items():
        installed[name] = _atomic_install_payload(
            plan,
            name=name,
            path=path,
            payload=payload,
            mode=mode,
            uid=uid,
            gid=gid,
            allowed_previous=previous,
        )
    return installed


def _service_identity_receipts(plan: FullCanaryPlan) -> Mapping[str, Any]:
    values: dict[str, Any] = {}
    for name, path, uid, gid in (
        (
            "edge",
            DEFAULT_EDGE_READINESS_PATH,
            plan.identities.edge_uid,
            plan.identities.edge_gid,
        ),
        (
            "writer",
            DEFAULT_WRITER_RUNTIME_ATTESTATION_PATH,
            plan.identities.writer_uid,
            plan.identities.writer_gid,
        ),
        (
            "gateway",
            DEFAULT_GATEWAY_READINESS_PATH,
            plan.identities.gateway_uid,
            plan.identities.gateway_gid,
        ),
    ):
        receipt = _readiness_receipt(path, uid=uid, gid=gid)
        values[name] = {
            "path": str(path),
            "receipt": receipt,
            "sha256": readiness_receipt_sha256(receipt),
        }
    return values


def _stop_all(*, runner: Runner) -> tuple[str, ...]:
    # The unit names are compile-time fixed but overlap the production naming
    # surface.  No stop mutation is reachable until fresh GCE/local identity
    # observation proves this is the one dedicated canary VM.  This boundary
    # intentionally consumes no plan or config bytes.
    _observe_dedicated_canary_host()
    stopped: list[str] = []
    errors: list[BaseException] = []
    for command in stop_service_commands():
        unit = command.argv[-1]
        try:
            _run_checked(command, runner=runner, label=f"stop {unit}")
            stopped.append(unit)
        except BaseException as exc:
            errors.append(exc)
    if errors:
        raise ExceptionGroup("full-canary ordered stop failed", errors)
    return tuple(stopped)


def mechanically_stop_full_canary_services(
    *,
    runner: Runner = _runner,
) -> tuple[str, ...]:
    """Expose only the deterministic reverse-order stop primitive."""

    return _stop_all(runner=runner)


class FullCanaryLifecycle:
    def __init__(
        self,
        plan: FullCanaryPlan,
        *,
        runner: Runner = _runner,
        metadata_reader: Callable[[str], bytes | str] | None = None,
        local_identity_reader: Callable[[str], bytes | str] | None = None,
    ) -> None:
        if not isinstance(plan, FullCanaryPlan):
            raise TypeError("full-canary plan is required")
        self.plan = plan
        self.runner = runner
        self.metadata_reader = metadata_reader
        self.local_identity_reader = local_identity_reader

    def _require_dedicated_host(self) -> Mapping[str, Any]:
        return validate_dedicated_canary_host(
            self.plan,
            metadata_reader=self.metadata_reader,
            local_identity_reader=self.local_identity_reader,
        )

    def _preflight(self, *, phase: str) -> Mapping[str, Any]:
        return collect_full_canary_preflight(
            self.plan,
            phase=phase,
            runner=self.runner,
            metadata_reader=self.metadata_reader,
            local_identity_reader=self.local_identity_reader,
        )

    def start(self, approval: FullCanaryOwnerApproval) -> Mapping[str, Any]:
        """Start only the already-founded persistent writer runtime."""

        return self._start_authorized(approval)

    def _start_authorized(
        self,
        approval: FullCanaryOwnerApproval,
    ) -> Mapping[str, Any]:
        """Install and start disabled one-shot services; leave them live."""

        _require_root_linux()
        if not isinstance(approval, FullCanaryOwnerApproval):
            raise PermissionError("full-canary owner approval is required")
        approval.require(plan_sha256=self.plan.sha256, now_unix=int(time.time()))
        self._require_dedicated_host()
        writer_config_raw = _validate_artifact_source(
            self.plan.artifacts["writer_config"],
            label="writer_config",
        )
        _validate_writer_config(
            writer_config_raw,
            self.plan.identities,
        )
        with _lifecycle_lock():
            preflight = self._preflight(phase="stopped")
            mutation_started = False
            installed: Mapping[str, Any] = {}
            started: list[str] = []
            phase_b_current: Mapping[str, Any] | None = None
            installed_phase_b_anchor: Mapping[str, Any] | None = None
            try:
                approval.require(
                    plan_sha256=self.plan.sha256,
                    now_unix=int(time.time()),
                )
                # Re-read the sealed receipt, metadata, machine/hostname, and
                # boot immediately before the first filesystem mutation.
                self._require_dedicated_host()
                mutation_started = True
                installed = _install_plan_artifacts(self.plan)
                _run_checked(
                    Command(
                        (
                            SYSTEMD_ANALYZE,
                            "verify",
                            str(DEFAULT_EDGE_UNIT_PATH),
                            str(DEFAULT_WRITER_UNIT_PATH),
                            str(DEFAULT_GATEWAY_UNIT_PATH),
                            str(DEFAULT_PHASE_B_READINESS_UNIT_PATH),
                        )
                    ),
                    runner=self.runner,
                    label="verify full-canary units",
                )
                _run_checked(
                    Command((SYSTEMCTL, "daemon-reload")),
                    runner=self.runner,
                    label="reload full-canary units",
                )
                _run_checked(
                    Command(
                        (
                            SYSTEMD_TMPFILES,
                            "--create",
                            str(DEFAULT_TMPFILES_PATH),
                        )
                    ),
                    runner=self.runner,
                    label="create full-canary runtime directories",
                )
                from gateway.canonical_writer_phase_b_runtime import (
                    install_fixed_phase_b_full_canary_anchor,
                    validate_fixed_phase_b_readiness_descendant,
                )

                # The root collector is zero-input and must finish before any
                # privileged or model-running canary service starts.
                # RemainAfterExit keeps its successful generation active so
                # the writer's Requires=/After= dependency cannot re-run it
                # after the privileged Discord edge is already live.
                _run_checked(
                    phase_b_readiness_start_command(),
                    runner=self.runner,
                    label=f"start {PHASE_B_READINESS_UNIT_NAME}",
                )
                _validate_inert_gateway_paths(
                    gateway_uid=self.plan.identities.gateway_uid,
                    gateway_gid=self.plan.identities.gateway_gid,
                )
                self._preflight(phase="stopped")
                phase_b_current = validate_fixed_phase_b_readiness_descendant(
                    self.plan.phase_b_readiness_anchor
                )
                installed_phase_b_anchor = (
                    install_fixed_phase_b_full_canary_anchor(
                        self.plan.phase_b_readiness_anchor
                    )
                )
                installed = {
                    **dict(installed),
                    "phase_b_full_canary_anchor": copy.deepcopy(
                        dict(installed_phase_b_anchor)
                    ),
                }
                approval.require(
                    plan_sha256=self.plan.sha256,
                    now_unix=int(time.time()),
                )
                edge_command = edge_start_command()
                _run_checked(
                    edge_command,
                    runner=self.runner,
                    label=f"start {EDGE_UNIT_NAME}",
                )
                started.append(EDGE_UNIT_NAME)
                edge_state = collect_service_state(
                    EDGE_UNIT_NAME,
                    runner=self.runner,
                )
                edge_readiness = _validate_edge_collector_gate(
                    self.plan,
                    edge_state,
                )
                edge_identity_sha256 = readiness_receipt_sha256(edge_readiness)
                collector = _await_collector_readiness(
                    self.plan,
                    edge_pid=int(edge_state["MainPID"]),
                    edge_service_identity_sha256=edge_identity_sha256,
                )
                observer = materialize_observer_config(
                    self.plan,
                    collector=collector,
                    edge_pid=int(edge_state["MainPID"]),
                    edge_service_identity_sha256=edge_identity_sha256,
                )
                installed = {**dict(installed), "observer_config": observer}
                approval.require(
                    plan_sha256=self.plan.sha256,
                    now_unix=int(time.time()),
                )
                self._require_dedicated_host()
                writer_command, gateway_command = (
                    post_collector_start_commands()
                )
                _run_checked(
                    writer_command,
                    runner=self.runner,
                    label=f"start {WRITER_UNIT_NAME}",
                )
                started.append(WRITER_UNIT_NAME)
                writer_readiness = _readiness_receipt(
                    DEFAULT_WRITER_RUNTIME_ATTESTATION_PATH,
                    uid=self.plan.identities.writer_uid,
                    gid=self.plan.identities.writer_gid,
                )
                _run_checked(
                    gateway_command,
                    runner=self.runner,
                    label=f"start {GATEWAY_UNIT_NAME}",
                )
                started.append(GATEWAY_UNIT_NAME)
                gateway_state = collect_service_state(
                    GATEWAY_UNIT_NAME,
                    runner=self.runner,
                )
                plugin_readiness = _await_plugin_readiness(
                    self.plan,
                    collector=collector,
                    gateway_pid=int(gateway_state["MainPID"]),
                    edge_pid=int(edge_state["MainPID"]),
                    edge_service_identity_sha256=edge_identity_sha256,
                )
                live = self._preflight(phase="live")
                identity_receipts = _service_identity_receipts(self.plan)
                api_loopback = _api_loopback_listener_identity(
                    int(gateway_state["MainPID"])
                )
                return _write_append_only_receipt(
                    self.plan,
                    stage="started",
                    value={
                        "owner_approval_receipt_sha256": approval.sha256,
                        "phase_b_full_canary_anchor": copy.deepcopy(
                            dict(installed_phase_b_anchor or {})
                        ),
                        "phase_b_current_readiness_receipt": copy.deepcopy(
                            dict(phase_b_current or {})
                        ),
                        "writer_runtime_readiness_receipt": copy.deepcopy(
                            dict(writer_readiness)
                        ),
                        "writer_only_activation_receipt_sha256": self.plan.writer_activation_receipt["receipt_sha256"],
                        "preflight_report_sha256": preflight["report_sha256"],
                        "live_report_sha256": live["report_sha256"],
                        "installed_artifacts": copy.deepcopy(dict(installed)),
                        "service_identity_receipts": identity_receipts,
                        "api_loopback_listener": api_loopback,
                        "collector_readiness_receipt": copy.deepcopy(
                            dict(collector.receipt)
                        ),
                        "collector_readiness_file_sha256": collector.file_sha256,
                        "plugin_readiness_receipt": copy.deepcopy(
                            dict(plugin_readiness.receipt)
                        ),
                        "plugin_readiness_file_sha256": (
                            plugin_readiness.file_sha256
                        ),
                        "plugin_ready_frame_sha256": (
                            plugin_readiness.frame_sha256
                        ),
                        "collector_hash_chain_head_sha256": (
                            plugin_readiness.collector_hash_chain_head_sha256
                        ),
                        "observer_config": copy.deepcopy(dict(observer)),
                        "start_order": started,
                        "units_enabled": False,
                        "runtime_max_seconds": 900,
                        "started_at_unix": int(time.time()),
                    },
                )
            except BaseException as error:
                cleanup_errors: list[BaseException] = []
                ordered_stop_complete = True
                if mutation_started:
                    try:
                        _stop_all(runner=self.runner)
                    except BaseException as exc:
                        ordered_stop_complete = False
                        cleanup_errors.append(exc)
                failure = _write_append_only_receipt(
                    self.plan,
                    stage="failure",
                    value={
                        "operation": "start",
                        "owner_approval_receipt": copy.deepcopy(
                            dict(approval.value)
                        ),
                        "owner_approval_receipt_sha256": approval.sha256,
                        "preflight_report_sha256": preflight["report_sha256"],
                        "started_before_failure": started,
                        "phase_b_full_canary_anchor": copy.deepcopy(
                            dict(installed_phase_b_anchor or {})
                        ),
                        "phase_b_current_readiness_receipt": copy.deepcopy(
                            dict(phase_b_current or {})
                        ),
                        "error_type": type(error).__name__,
                        "error_sha256": _sha256_bytes(
                            f"{type(error).__name__}:{error}".encode(
                                "utf-8", errors="replace"
                            )
                        ),
                        "ordered_stop_attempted": mutation_started,
                        "ordered_stop_complete": ordered_stop_complete,
                        "failed_at_unix": int(time.time()),
                    },
                )
                if cleanup_errors:
                    raise ExceptionGroup(
                        f"full-canary start failed; receipt={failure['receipt_path']}",
                        [error, *cleanup_errors],
                    ) from None
                raise RuntimeError(
                    f"full-canary start failed closed; receipt={failure['receipt_path']}"
                ) from error

    def _attest_stopped_locked(
        self,
        *,
        reason: str,
        stopped: Sequence[str],
    ) -> Mapping[str, Any]:
        if list(stopped) != list(FULL_CANARY_STOP_ORDER):
            raise RuntimeError("full-canary mechanical stop order is invalid")
        report = self._preflight(phase="stopped")
        return _write_append_only_receipt(
            self.plan,
            stage="stopped",
            value={
                "reason": reason,
                "stop_order": list(stopped),
                "phase_b_readiness_anchor": copy.deepcopy(
                    dict(self.plan.phase_b_readiness_anchor)
                ),
                "units_enabled": False,
                "stopped_report_sha256": report["report_sha256"],
                "stopped_at_unix": int(time.time()),
            },
        )

    def attest_stopped_after_mechanical_stop(
        self,
        *,
        reason: str,
        stopped: Sequence[str],
    ) -> Mapping[str, Any]:
        if reason not in {
            "operator_requested",
            "verification_complete",
            "verification_failed",
        }:
            raise ValueError("full-canary stop reason is invalid")
        _require_root_linux()
        self._require_dedicated_host()
        with _lifecycle_lock():
            return self._attest_stopped_locked(
                reason=reason,
                stopped=stopped,
            )

    def stop(self, *, reason: str = "operator_requested") -> Mapping[str, Any]:
        if reason not in {"operator_requested", "verification_complete", "verification_failed"}:
            raise ValueError("full-canary stop reason is invalid")
        _observe_dedicated_canary_host(
            metadata_reader=self.metadata_reader,
            local_identity_reader=self.local_identity_reader,
        )
        _require_root_linux()
        errors: list[BaseException] = []
        stopped: tuple[str, ...] = ()
        try:
            stopped = _stop_all(runner=self.runner)
        except BaseException as exc:
            errors.append(exc)
        try:
            self._require_dedicated_host()
        except BaseException as exc:
            errors.append(exc)
        if len(errors) > 1:
            raise BaseExceptionGroup(
                "full-canary stop/observation/plan-host validation failed",
                errors,
            )
        if errors:
            raise errors[0]
        with _lifecycle_lock():
            return self._attest_stopped_locked(
                reason=reason,
                stopped=stopped,
            )

    def verify_and_stop(
        self,
        *,
        start_receipt_path: Path,
        evidence_path: Path,
        evidence_sha256: str,
    ) -> Mapping[str, Any]:
        """Verify exact live evidence and always execute reverse-order stop."""

        # This immutable host allow-list is the only gate before the fixed
        # systemctl cleanup becomes reachable.  Plan/config semantics do not
        # participate in the mutation boundary.
        _observe_dedicated_canary_host(
            metadata_reader=self.metadata_reader,
            local_identity_reader=self.local_identity_reader,
        )
        loaded_start: LoadedStartReceipt | None = None
        start_receipt: Mapping[str, Any] | None = None
        validated_evidence_path: Path | None = None
        validated_evidence_sha256: str | None = None
        verifier_result: Mapping[str, Any] | None = None
        live_report: Mapping[str, Any] | None = None
        primary: BaseException | None = None
        stopped: tuple[str, ...] = ()
        stopped_report: Mapping[str, Any] | None = None
        semantic_cleanup_safe = False
        lifecycle_lock = None
        lifecycle_lock_entered = False
        try:
            _require_root_linux()
            self._require_dedicated_host()
            semantic_cleanup_safe = True
            validated_evidence_sha256 = _digest(
                evidence_sha256,
                "full-canary E2E evidence digest",
            )
            loaded_start = load_start_receipt(
                start_receipt_path,
                plan=self.plan,
            )
            start_receipt = loaded_start.value
            validated_evidence_path = _absolute_path(
                evidence_path,
                "E2E evidence path",
            )
            if validated_evidence_path != expected_live_evidence_path(self.plan):
                raise ValueError(
                    "full-canary evidence path is not plan-addressed"
                )
            lifecycle_lock = _lifecycle_lock()
            lifecycle_lock.__enter__()
            lifecycle_lock_entered = True
            live_report = self._preflight(phase="live")
            command = Command(
                (
                    self.plan.release["interpreter"],
                    "-B",
                    "-I",
                    "-m",
                    self.plan.e2e_verifier_module,
                    "verify",
                    "--fixture",
                    str(self.plan.artifacts["e2e_fixture"].source_path),
                    "--fixture-sha256",
                    self.plan.artifacts["e2e_fixture"].sha256,
                    "--fixture-gid",
                    str(self.plan.identities.gateway_gid),
                    "--evidence",
                    str(validated_evidence_path),
                    "--evidence-sha256",
                    validated_evidence_sha256,
                    "--start-receipt-sha256",
                    loaded_start.file_sha256,
                ),
                timeout_seconds=300,
            )
            completed = _run_checked(
                command,
                runner=self.runner,
                label="verify full-canary live E2E evidence",
            )
            verifier_result = _decode_json(
                completed.stdout,
                label="full-canary E2E verifier output",
            )
            from gateway.canonical_full_canary_e2e import (
                _INVARIANTS as verifier_invariants,
            )

            expected_invariants = list(verifier_invariants)
            if (
                verifier_result.get("schema")
                != "muncho-full-canary-e2e-verification.v1"
                or verifier_result.get("ok") is not True
                or verifier_result.get("fixture_sha256")
                != self.plan.artifacts["e2e_fixture"].sha256
                or verifier_result.get("evidence_sha256")
                != validated_evidence_sha256
                or verifier_result.get("full_canary_start_receipt_sha256")
                != loaded_start.file_sha256
                or verifier_result.get("invariants") != expected_invariants
                or _SHA256_RE.fullmatch(
                    str(verifier_result.get("invariant_receipt_sha256", ""))
                )
                is None
            ):
                raise RuntimeError("full-canary E2E verifier result is not exact")
        except BaseException as exc:
            primary = exc
        finally:
            cleanup_errors: list[BaseException] = []
            try:
                stopped = _stop_all(runner=self.runner)
            except BaseException as exc:
                cleanup_errors.append(exc)
            if not cleanup_errors and semantic_cleanup_safe:
                try:
                    stopped_report = self._preflight(phase="stopped")
                except BaseException as exc:
                    cleanup_errors.append(exc)
            if lifecycle_lock_entered:
                try:
                    assert lifecycle_lock is not None
                    lifecycle_lock.__exit__(None, None, None)
                except BaseException as exc:
                    cleanup_errors.append(exc)
            if cleanup_errors:
                combined = (
                    ([primary] if primary is not None else [])
                    + cleanup_errors
                )
                primary = BaseExceptionGroup(
                    "full-canary verification/stop failed",
                    combined,
                )
        if (
            loaded_start is None
            or start_receipt is None
            or validated_evidence_path is None
            or validated_evidence_sha256 is None
        ):
            if primary is None:
                raise AssertionError(
                    "full-canary verification inputs are unavailable"
                )
            raise primary
        receipt = _write_append_only_receipt(
            self.plan,
            stage="verified_stopped" if primary is None else "failure",
            value={
                "operation": "verify_and_stop",
                "full_canary_start_receipt_sha256": loaded_start.file_sha256,
                "full_canary_start_receipt_internal_sha256": start_receipt[
                    "receipt_sha256"
                ],
                "evidence_path": str(validated_evidence_path),
                "evidence_sha256": validated_evidence_sha256,
                "live_report_sha256": (
                    live_report.get("report_sha256") if live_report else None
                ),
                "verifier_result": copy.deepcopy(dict(verifier_result or {})),
                "stop_order": list(stopped),
                "phase_b_readiness_anchor": copy.deepcopy(
                    dict(self.plan.phase_b_readiness_anchor)
                ),
                "stopped_report_sha256": (
                    stopped_report.get("report_sha256") if stopped_report else None
                ),
                "units_enabled": False,
                "verified": primary is None,
                "error_type": type(primary).__name__ if primary is not None else None,
                "error_sha256": (
                    _sha256_bytes(
                        f"{type(primary).__name__}:{primary}".encode(
                            "utf-8", errors="replace"
                        )
                    )
                    if primary is not None
                    else None
                ),
                "completed_at_unix": int(time.time()),
            },
        )
        if primary is not None:
            raise RuntimeError(
                f"full-canary verification failed closed; receipt={receipt['receipt_path']}"
            ) from primary
        return receipt


@dataclass(frozen=True)
class LoadedStartReceipt:
    value: Mapping[str, Any]
    file_sha256: str


def expected_live_evidence_path(plan: FullCanaryPlan) -> Path:
    """Return the sole plan-addressed path accepted from the live collector."""
    if not isinstance(plan, FullCanaryPlan):
        raise TypeError("full-canary plan is required")
    return (
        DEFAULT_EVIDENCE_ROOT
        / "plans"
        / plan.revision
        / plan.sha256
        / "live"
        / "evidence.json"
    )


def load_start_receipt(
    path: Path,
    *,
    plan: FullCanaryPlan,
) -> LoadedStartReceipt:
    path = _absolute_path(path, "full-canary start receipt")
    raw, _item = _read_stable_file(
        path,
        maximum=_MAX_JSON_BYTES,
        expected_uid=0,
        expected_gid=0,
        allowed_modes=frozenset({0o400}),
    )
    value = _decode_json(raw, label="full-canary start receipt")
    unsigned = {name: copy.deepcopy(item) for name, item in value.items() if name != "receipt_sha256"}
    expected_root = DEFAULT_EVIDENCE_ROOT / "plans" / plan.revision / plan.sha256 / "started"
    receipt_path = _absolute_path(value.get("receipt_path"), "start receipt path")
    if (
        value.get("schema") != FULL_CANARY_RECEIPT_SCHEMA
        or value.get("stage") != "started"
        or value.get("revision") != plan.revision
        or value.get("full_canary_plan_sha256") != plan.sha256
        or value.get("units_enabled") is not False
        or value.get("runtime_max_seconds") != 900
        or value.get("start_order") != list(FULL_CANARY_START_ORDER)
        or receipt_path.parent != expected_root
        or receipt_path != path
        or value.get("receipt_sha256") != _sha256_json(unsigned)
    ):
        raise RuntimeError("full-canary start receipt is invalid")
    identities = value.get("service_identity_receipts")
    if not isinstance(identities, Mapping) or set(identities) != {"edge", "writer", "gateway"}:
        raise RuntimeError("full-canary start identity receipts are incomplete")
    for name, evidence in identities.items():
        if (
            not isinstance(evidence, Mapping)
            or set(evidence) != {"path", "receipt", "sha256"}
            or readiness_receipt_sha256(evidence["receipt"]) != evidence["sha256"]
        ):
            raise RuntimeError(f"full-canary {name} start identity receipt drifted")
    api_loopback = value.get("api_loopback_listener")
    gateway_receipt = identities["gateway"]["receipt"]
    if (
        not isinstance(api_loopback, Mapping)
        or set(api_loopback)
        != {
            "schema",
            "gateway_pid",
            "gateway_start_time_ticks",
            "protocol",
            "host",
            "port",
            "socket_inode",
            "identity_sha256",
        }
        or api_loopback.get("schema")
        != "muncho-full-canary-api-loopback-listener.v1"
        or api_loopback.get("gateway_pid") != gateway_receipt.get("gateway_pid")
        or api_loopback.get("host") != "127.0.0.1"
        or api_loopback.get("port") != 8642
        or api_loopback.get("protocol") != "tcp"
        or api_loopback.get("identity_sha256")
        != _sha256_json(
            {
                name: item
                for name, item in api_loopback.items()
                if name != "identity_sha256"
            }
        )
    ):
        raise RuntimeError("full-canary API loopback start identity drifted")

    edge_receipt = identities["edge"]["receipt"]
    edge_pid = edge_receipt.get("edge_pid")
    gateway_pid = gateway_receipt.get("gateway_pid")
    if type(edge_pid) is not int or type(gateway_pid) is not int:
        raise RuntimeError("full-canary start service PIDs are invalid")
    edge_identity_sha256 = readiness_receipt_sha256(edge_receipt)
    collector = load_collector_readiness(
        plan,
        edge_pid=edge_pid,
        edge_service_identity_sha256=edge_identity_sha256,
    )
    if (
        value.get("collector_readiness_receipt") != collector.receipt
        or value.get("collector_readiness_file_sha256") != collector.file_sha256
    ):
        raise RuntimeError("full-canary start collector receipt drifted")

    observer = value.get("observer_config")
    observer_fields = {
        "path",
        "sha256",
        "changed",
        "prior_state",
        "config",
        "collector_readiness_file_sha256",
        "collector_service_identity_sha256",
        "edge_service_identity_sha256",
    }
    expected_observer = _observer_config_mapping(
        plan,
        collector=collector,
        edge_pid=edge_pid,
        edge_service_identity_sha256=edge_identity_sha256,
    )
    observer_raw, _observer_item = _read_stable_file(
        DEFAULT_OBSERVER_CONFIG,
        maximum=_MAX_CONFIG_BYTES,
        expected_uid=0,
        expected_gid=plan.identities.gateway_gid,
        allowed_modes=frozenset({0o440}),
    )
    if (
        not isinstance(observer, Mapping)
        or set(observer) != observer_fields
        or observer.get("path") != str(DEFAULT_OBSERVER_CONFIG)
        or observer.get("sha256") != _sha256_bytes(observer_raw)
        or type(observer.get("changed")) is not bool
        or observer.get("prior_state") is not None
        and not isinstance(observer.get("prior_state"), Mapping)
        or observer.get("config") != expected_observer
        or observer.get("collector_readiness_file_sha256")
        != collector.file_sha256
        or observer.get("collector_service_identity_sha256")
        != collector.service_identity_sha256
        or observer.get("edge_service_identity_sha256")
        != edge_identity_sha256
        or _decode_json(observer_raw, label="start observer config")
        != expected_observer
    ):
        raise RuntimeError("full-canary start observer materialization drifted")

    plugin = load_plugin_readiness(
        plan,
        collector=collector,
        gateway_pid=gateway_pid,
        edge_pid=edge_pid,
        edge_service_identity_sha256=edge_identity_sha256,
    )
    if (
        value.get("plugin_readiness_receipt") != plugin.receipt
        or value.get("plugin_readiness_file_sha256") != plugin.file_sha256
        or value.get("plugin_ready_frame_sha256") != plugin.frame_sha256
        or value.get("collector_hash_chain_head_sha256")
        != plugin.collector_hash_chain_head_sha256
    ):
        raise RuntimeError("full-canary start plugin readiness drifted")
    return LoadedStartReceipt(
        value=copy.deepcopy(dict(value)),
        file_sha256=_sha256_bytes(raw),
    )


def load_full_canary_plan(path: Path = DEFAULT_PLAN_PATH) -> FullCanaryPlan:
    path = _absolute_path(path, "full-canary plan")
    if path != DEFAULT_PLAN_PATH:
        raise ValueError("full-canary plan path is not fixed")
    raw, _item = _read_stable_file(
        path,
        maximum=_MAX_JSON_BYTES,
        expected_uid=0,
        expected_gid=0,
        allowed_modes=frozenset({0o400}),
    )
    value = _decode_json(raw, label="full-canary runtime plan")
    if raw != _canonical_bytes(value):
        raise RuntimeError("full-canary runtime plan bytes are not canonical")
    return FullCanaryPlan.from_mapping(value)


def load_full_canary_approval(
    path: Path = DEFAULT_APPROVAL_PATH,
) -> FullCanaryOwnerApproval:
    path = _absolute_path(path, "full-canary approval")
    if path != DEFAULT_APPROVAL_PATH:
        raise ValueError("full-canary approval path is not fixed")
    raw, _item = _read_stable_file(
        path,
        maximum=128 * 1024,
        expected_uid=0,
        expected_gid=0,
        allowed_modes=frozenset({0o400}),
    )
    value = _decode_json(raw, label="full-canary owner approval")
    if raw != _canonical_bytes(value):
        raise RuntimeError("full-canary approval bytes are not canonical")
    return FullCanaryOwnerApproval.from_mapping(value)


def _cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Digest-bound isolated full Muncho canary lifecycle"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--phase", choices=("stopped", "live"), default="stopped")
    subparsers.add_parser("start")
    stop = subparsers.add_parser("stop")
    stop.add_argument(
        "--reason",
        choices=("operator_requested", "verification_complete", "verification_failed"),
        default="operator_requested",
    )
    verify = subparsers.add_parser("verify-and-stop")
    verify.add_argument("--start-receipt", required=True, type=Path)
    verify.add_argument("--evidence-sha256", required=True)
    return parser


def _cli_result(value: Mapping[str, Any]) -> None:
    print(_canonical_bytes(value).decode("utf-8", errors="strict"))


def main(argv: Sequence[str] | None = None) -> int:
    args = _cli_parser().parse_args(argv)
    try:
        if args.command == "validate":
            plan = load_full_canary_plan()
            result = collect_full_canary_preflight(
                plan,
                phase=args.phase,
            )
        elif args.command == "start":
            plan = load_full_canary_plan()
            result = FullCanaryLifecycle(plan).start(
                load_full_canary_approval()
            )
        elif args.command == "stop":
            terminal_errors: list[BaseException] = []
            stopped: tuple[str, ...] = ()
            try:
                stopped = mechanically_stop_full_canary_services()
            except BaseException as exc:
                terminal_errors.append(exc)
            plan: FullCanaryPlan | None = None
            try:
                plan = load_full_canary_plan()
            except BaseException as exc:
                terminal_errors.append(exc)
            if len(terminal_errors) > 1:
                raise BaseExceptionGroup(
                    "full-canary stop-first terminal loading failed",
                    terminal_errors,
                )
            if terminal_errors:
                raise terminal_errors[0]
            if plan is None:
                raise AssertionError("full-canary stop plan is unavailable")
            result = FullCanaryLifecycle(
                plan,
            ).attest_stopped_after_mechanical_stop(
                reason=args.reason,
                stopped=stopped,
            )
        elif args.command == "verify-and-stop":
            # This compile-time identity fence is deliberately independent of
            # plan/config bytes.  A wrong host must reach no systemctl call.
            _observe_dedicated_canary_host()
            primary: BaseException | None = None
            terminal_errors: list[BaseException] = []
            result = None
            try:
                plan = load_full_canary_plan()
                load_start_receipt(args.start_receipt, plan=plan)
                result = FullCanaryLifecycle(plan).verify_and_stop(
                    start_receipt_path=args.start_receipt,
                    evidence_path=expected_live_evidence_path(plan),
                    evidence_sha256=args.evidence_sha256,
                )
            except BaseException as exc:
                primary = exc
            finally:
                # Independent fixed cleanup covers failures before a lifecycle
                # exists (plan/evidence/start-receipt load or constructor) as
                # well as lifecycle verification failures.  The lifecycle's
                # own stop remains authoritative; this second stop is an
                # intentionally idempotent outer safety boundary.
                try:
                    mechanically_stop_full_canary_services()
                except BaseException as exc:
                    terminal_errors.append(exc)
            combined = ([primary] if primary is not None else []) + terminal_errors
            if len(combined) > 1:
                raise BaseExceptionGroup(
                    "full-canary verification and independent stop failed",
                    combined,
                )
            if combined:
                raise combined[0]
            if result is None:
                raise AssertionError("full-canary verify result is unavailable")
        else:  # pragma: no cover - argparse enforces the command set.
            raise RuntimeError("unsupported full-canary command")
        output = dict(result)
        receipt_path = output.get("receipt_path")
        if isinstance(receipt_path, str):
            raw, _item = _read_stable_file(
                Path(receipt_path),
                maximum=_MAX_JSON_BYTES,
                expected_uid=0,
                expected_gid=0,
                allowed_modes=frozenset({0o400}),
            )
            output["receipt_file_sha256"] = _sha256_bytes(raw)
        _cli_result(output)
        return 0
    except FullCanaryPreflightError as exc:
        _cli_result(exc.report)
        return 2
    except Exception as exc:
        failure = {
            "schema": "muncho-full-canary-cli-failure.v1",
            "ok": False,
            "error_type": type(exc).__name__,
            "error_sha256": _sha256_bytes(
                f"{type(exc).__name__}:{exc}".encode("utf-8", errors="replace")
            ),
        }
        _cli_result(failure)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

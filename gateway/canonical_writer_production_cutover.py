"""Exact, owner-approved production legacy-truth cutover coordinator.

The isolated-canary foundation deliberately refuses the production database.
This module does not weaken that boundary and does not contain a second SQL or
systemd implementation.  It composes four already-sealed mechanical edges:

* an exact service boundary for ``hermes-cloud-gateway.service`` and the
  Canonical Writer unit;
* a read-only legacy snapshot collector;
* an atomic database apply/rollback boundary; and
* an activation apply/rollback boundary.

There is one owner-approved authority before production is stopped.  It binds
the exact release, target, artifacts, host transition, capability collector
criteria, final-tail bounds, and rollback.  The final tail and dynamic
PID/socket readiness receipt are derived mechanically after that approval;
neither can be guessed or pre-signed.  No credential or secret value is
accepted by any public schema.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, ContextManager, Mapping, Protocol

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from gateway import canonical_writer_activation as activation
from gateway import production_cron_continuity_package
from gateway import production_cron_migration
from gateway import production_owner_runtime
from gateway.production_capability_prerequisites import (
    API_APPROVAL_CREDENTIAL_PATH,
    API_SERVER_CREDENTIAL_PATH,
    BROWSER_UNIT,
    BROWSER_CONFIG_PATH,
    MAC_OPS_CONFIG_PATH,
    MAC_OPS_UNIT,
    PHASE_B_RECEIPT_PATH,
    PHASE_B_UNIT,
    FIRST_WAVE_TOOLSETS,
    PREREQUISITE_LIFECYCLE_STAGED,
    ROUTEBACK_EDGE_CONFIG_PATH,
    ROUTEBACK_EDGE_UNIT,
    collect_and_install_from_production_config,
    load_production_capability_prerequisite_receipt,
    production_capability_topology_identity_sha256,
    validate_production_capability_topology,
)
from gateway.isolated_worker_units import (
    ISOLATED_WORKER_CONFIG,
    ISOLATED_WORKER_LEASE_BASE,
    ISOLATED_WORKER_SERVICE_UNIT,
    ISOLATED_WORKER_SOCKET_UNIT,
)
from gateway.operational_edge_catalog import (
    CREDENTIALS_BY_DOMAIN,
    required_cron_operations,
)
from gateway.operational_edge_bootstrap import (
    PRE_OWNER_STAGING_ROOT as OPERATIONAL_EDGE_KEY_STAGING_ROOT,
    validate_operational_edge_key_foundation,
)
from gateway.operational_edge_units import (
    CLIENT_CONFIG_PATH as OPERATIONAL_EDGE_CLIENT_CONFIG_PATH,
    service_config_path as operational_edge_config_path,
    service_unit as operational_edge_service_unit,
)
from gateway.support_ops_team_registry import (
    SKYVISION_APPROVED_OPERATIONAL_CHANNEL_IDS,
    SKYVISION_CONTROL_TOWER_CHANNEL_ID,
    SKYVISION_GUILD_ID,
    SKYVISION_NASI_AI_OPS_CHANNEL_ID,
)
from gateway.operational_edge_readiness import (
    OperationalEdgeReadinessError,
    validate_operational_edge_readiness,
)
from ops.muncho.runtime import mechanical_job_rail


FREEZE_PLAN_SCHEMA = "muncho-production-legacy-freeze-plan.v2"
CUTOVER_PLAN_SCHEMA = "muncho-production-legacy-cutover-plan.v2"
APPROVAL_SCHEMA = "muncho-production-legacy-cutover-approval.v1"
SNAPSHOT_SCHEMA = "muncho-production-legacy-snapshot.v1"
SERVICE_SCHEMA = "muncho-production-service-observation.v1"
FREEZE_RECEIPT_SCHEMA = "muncho-production-final-tail-receipt.v1"
JOURNAL_SCHEMA = "muncho-production-legacy-cutover-journal.v1"
TERMINAL_SCHEMA = "muncho-production-legacy-cutover-terminal.v1"
ROLLBACK_TERMINAL_SCHEMA = (
    "muncho-production-legacy-cutover-rollback-terminal.v1"
)
LEGACY_TRUTH_DECISION_SCHEMA = "muncho-production-legacy-truth-decision.v1"
LEGACY_RESEED_MANIFEST_SCHEMA = "muncho-production-legacy-reseed-manifest.v1"
NEW_TRUTH_EPOCH_SCHEMA = "muncho-production-new-truth-epoch.v1"
ISOLATED_CANARY_GOAL_PREREQUISITE_SCHEMA = (
    "muncho-production-isolated-canary-goal-prerequisite.v2"
)
ISOLATION_EQUIVALENCE_SCHEMA = (
    "muncho-production-isolation-equivalence-projection.v1"
)
CAPABILITY_PREREQUISITE_ACCEPTANCE_SCHEMA = (
    "muncho-production-capability-prerequisite-acceptance.v3"
)

PROJECT = "adventico-ai-platform"
ZONE = "europe-west3-a"
VM_NAME = "ai-platform-runtime-01"
DATABASE = "ai_platform_brain"
GATEWAY_UNIT = "hermes-cloud-gateway.service"
WRITER_UNIT = "muncho-canonical-writer.service"
CONNECTOR_UNIT = "muncho-discord-connector.service"
GATEWAY_FRAGMENT = "/etc/systemd/system/hermes-cloud-gateway.service"
WRITER_FRAGMENT = "/etc/systemd/system/muncho-canonical-writer.service"
CONNECTOR_FRAGMENT = "/etc/systemd/system/muncho-discord-connector.service"
PHASE_B_FRAGMENT = (
    "/etc/systemd/system/"
    "muncho-canonical-writer-phase-b-readiness.service"
)
ROUTEBACK_EDGE_FRAGMENT = (
    "/etc/systemd/system/muncho-discord-egress.service"
)
MAC_OPS_FRAGMENT = "/etc/systemd/system/muncho-mac-ops-edge.service"
BROWSER_FRAGMENT = "/etc/systemd/system/muncho-capability-browser.service"
BROWSER_CONFIG = str(BROWSER_CONFIG_PATH)
ISOLATED_WORKER_SOCKET_FRAGMENT = (
    "/etc/systemd/system/muncho-isolated-worker.socket"
)
ISOLATED_WORKER_SERVICE_FRAGMENT = (
    "/etc/systemd/system/muncho-isolated-worker.service"
)
ISOLATED_WORKER_CONFIG_PATH = str(ISOLATED_WORKER_CONFIG)
GATEWAY_CONNECTOR_DROP_IN = (
    "/etc/systemd/system/hermes-cloud-gateway.service.d/"
    "20-discord-connector.conf"
)
GATEWAY_CONFIG_PATH = "/opt/adventico-ai-platform/hermes-home/config.yaml"
CONNECTOR_CONFIG_PATH = "/etc/muncho/discord-public-connector.json"
WRITER_CONFIG_PATH = "/etc/muncho-canonical-writer/writer.json"
WRITER_CAPABILITY_PRIVATE_KEY_PATH = Path(
    "/etc/muncho/keys/writer-capability-private.pem"
)
WRITER_CAPABILITY_PUBLIC_KEY_PATH = Path(
    "/etc/muncho/keys/writer-capability-public.pem"
)
EDGE_RECEIPT_PRIVATE_KEY_PATH = Path(
    "/etc/muncho/keys/discord-edge-receipt-private.pem"
)
EDGE_RECEIPT_PUBLIC_KEY_PATH = Path(
    "/etc/muncho/keys/discord-edge-receipt-public.pem"
)
CONNECTOR_TOKEN_PATH = "/etc/muncho/discord-connector-credentials/bot-token"
EVIDENCE_ROOT = Path("/var/lib/muncho-production-legacy-cutover")
STAGED_APPROVAL_PASSKEY_PATH = EVIDENCE_ROOT / "staged" / "api-approval-passkey"
LEGACY_API_BEARER_PATH = Path("/etc/muncho/keys/api-server-control.key")
PRODUCTION_RELEASE_BASE = Path(
    "/opt/adventico-ai-platform/hermes-agent-releases"
)
SYSTEMCTL = "/usr/bin/systemctl"
OPERATIONAL_EDGE_UNITS = frozenset(
    operational_edge_service_unit(domain)
    for domain in CREDENTIALS_BY_DOMAIN
)
OPERATIONAL_EDGE_FRAGMENTS = {
    unit: f"/etc/systemd/system/{unit}" for unit in OPERATIONAL_EDGE_UNITS
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_ROLE = re.compile(r"^[a-z_][a-z0-9_-]{0,63}$")
_MAX_JSON = 8 * 1024 * 1024
_MAX_ARTIFACT = 8 * 1024 * 1024
_ARTIFACT_TIMEOUT_SECONDS = 900
_MAX_ACCEPTED_LEGACY_EVENTS = 40_000
ARTIFACT_REQUEST_SCHEMA = "muncho-production-cutover-artifact-request.v1"
FREEZE_ABORT_SCHEMA = "muncho-production-freeze-abort-receipt.v1"
STAGED_FREEZE_PLAN_PATH = EVIDENCE_ROOT / "staged" / "freeze-plan.json"
STAGED_FREEZE_APPROVAL_PATH = EVIDENCE_ROOT / "staged" / "freeze-approval.json"
STAGED_CUTOVER_PLAN_PATH = EVIDENCE_ROOT / "staged" / "cutover-plan.json"


class ProductionCutoverError(RuntimeError):
    """Stable, secret-free cutover failure."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8", errors="strict")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError("cutover value is not canonical JSON") from exc
    if len(payload) > _MAX_JSON:
        raise ValueError("cutover value is oversized")
    return payload


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be lowercase SHA-256")
    return value


def _exact(value: Any, fields: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{label} fields are not exact")
    return copy.deepcopy(dict(value))


def _hashed(
    value: Any,
    fields: frozenset[str],
    digest_field: str,
    label: str,
) -> dict[str, Any]:
    raw = _exact(value, fields, label)
    digest = _digest(raw[digest_field], f"{label} digest")
    unsigned = {key: item for key, item in raw.items() if key != digest_field}
    if _sha256_json(unsigned) != digest:
        raise ValueError(f"{label} self-digest is invalid")
    return raw


def _owner_runtime_attestation(value: Any, *, revision: str) -> dict[str, Any]:
    try:
        validated = production_owner_runtime.validate_owner_runtime_attestation(
            value,
            revision=revision,
        )
    except production_owner_runtime.ProductionOwnerRuntimeError as exc:
        raise ValueError("owner runtime attestation is invalid") from exc
    return copy.deepcopy(dict(validated))


_ISOLATED_CANARY_GOAL_PREREQUISITE_FIELDS = frozenset({
    "schema",
    "fixture",
    "fixture_sha256",
    "workspace_gateway",
    "workspace_gateway_receipt_sha256",
    "cleanup_receipt",
    "cleanup_receipt_sha256",
    "goal_continuation_terminal_schema",
    "goal_continuation_terminal_sha256",
    "isolation_equivalence_projection",
    "isolation_equivalence_projection_sha256",
    "production_diff_sha256",
    "production_diff",
    "production_diff_file_sha256",
    "run_id",
    "release_revision",
    "capability_plan_sha256",
    "full_canary_plan_sha256",
    "canary_owner_approval_receipt_sha256",
    "canary_production_mutation_observed",
    "secret_material_recorded",
    "secret_digest_recorded",
    "evidence_sha256",
})


def _validate_isolated_canary_goal_prerequisite(
    value: Any,
    *,
    revision: str,
) -> dict[str, Any]:
    """Validate the exact signed isolated-canary terminal selected by owner.

    The Ed25519 trust path and terminal semantics live in the canary verifier;
    this production edge adds only an exact owner-authority binding.  Keeping
    the full fixture and signed envelope inside the signed FreezePlan prevents
    substitution of a different run, release, fixture, or evidence envelope.
    """

    from gateway import canonical_capability_canary_e2e as canary_e2e
    from gateway import canonical_capability_canary_runtime as canary_runtime

    raw = _hashed(
        value,
        _ISOLATED_CANARY_GOAL_PREREQUISITE_FIELDS,
        "evidence_sha256",
        "isolated canary goal prerequisite",
    )
    fixture = raw["fixture"]
    envelope = raw["workspace_gateway"]
    cleanup_receipt = raw["cleanup_receipt"]
    production_diff = raw["production_diff"]
    if (
        not isinstance(fixture, Mapping)
        or not isinstance(envelope, Mapping)
        or not isinstance(cleanup_receipt, Mapping)
        or not isinstance(production_diff, Mapping)
        or raw["fixture_sha256"] != _sha256_json(fixture)
    ):
        raise ValueError("isolated canary fixture binding is invalid")
    try:
        terminal = canary_e2e.validate_goal_continuation_terminal_receipt(
            envelope,
            fixture=fixture,
            fixture_sha256=raw["fixture_sha256"],
        )
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("isolated canary goal terminal is invalid") from exc
    projection = terminal["isolation_equivalence_projection"]
    try:
        validated_diff = canary_runtime.validate_capability_production_diff(
            production_diff,
            run_id=terminal["run_id"],
            revision=terminal["release_sha"],
            capability_plan_sha256=terminal["capability_plan_sha256"],
            full_canary_plan_sha256=terminal[
                "full_canary_plan_sha256"
            ],
            fixture_sha256=terminal["fixture_sha256"],
        )
        validated_cleanup = (
            canary_e2e.validate_cleanup_production_diff_receipt(
                cleanup_receipt,
                fixture=fixture,
                fixture_sha256=raw["fixture_sha256"],
                production_diff_sha256=validated_diff["diff_sha256"],
            )
        )
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(
            "isolated canary native production diff is invalid"
        ) from exc
    if (
        raw["schema"] != ISOLATED_CANARY_GOAL_PREREQUISITE_SCHEMA
        or raw["release_revision"] != revision
        or terminal["release_sha"] != revision
        or raw["workspace_gateway_receipt_sha256"]
        != _sha256_json(envelope)
        or raw["cleanup_receipt_sha256"]
        != _sha256_json(validated_cleanup)
        or raw["goal_continuation_terminal_schema"]
        != canary_e2e.GOAL_CONTINUATION_TERMINAL_SCHEMA
        or raw["goal_continuation_terminal_schema"] != terminal["schema"]
        or raw["goal_continuation_terminal_sha256"]
        != terminal["terminal_sha256"]
        or raw["isolation_equivalence_projection"] != projection
        or raw["isolation_equivalence_projection_sha256"]
        != terminal["isolation_equivalence_projection_sha256"]
        or raw["isolation_equivalence_projection_sha256"]
        != _sha256_json(projection)
        or raw["production_diff_sha256"]
        != terminal["production_diff_sha256"]
        or raw["production_diff_sha256"] != validated_diff["diff_sha256"]
        or raw["production_diff_file_sha256"]
        != _sha256_json(validated_diff)
        or raw["run_id"] != terminal["run_id"]
        or raw["fixture_sha256"] != terminal["fixture_sha256"]
        or raw["capability_plan_sha256"]
        != terminal["capability_plan_sha256"]
        or raw["full_canary_plan_sha256"]
        != terminal["full_canary_plan_sha256"]
        or raw["canary_owner_approval_receipt_sha256"]
        != terminal["owner_approval_receipt_sha256"]
        or raw["canary_production_mutation_observed"] is not False
        or validated_diff["production_mutation_observed"] is not False
        or projection["production_mutation_observed"] is not False
        or raw["secret_material_recorded"] is not False
        or raw["secret_digest_recorded"] is not False
    ):
        raise ValueError("isolated canary goal prerequisite binding is invalid")
    return raw


def build_isolated_canary_goal_prerequisite(
    *,
    fixture: Mapping[str, Any],
    fixture_sha256: str,
    workspace_gateway: Mapping[str, Any],
    cleanup_receipt: Mapping[str, Any],
    production_diff: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Build the public evidence object that an owner may bind into cutover."""

    from gateway import canonical_capability_canary_e2e as canary_e2e

    terminal = canary_e2e.validate_goal_continuation_terminal_receipt(
        workspace_gateway,
        fixture=fixture,
        fixture_sha256=fixture_sha256,
    )
    projection = terminal["isolation_equivalence_projection"]
    from gateway import canonical_capability_canary_runtime as canary_runtime

    validated_diff = canary_runtime.validate_capability_production_diff(
        production_diff,
        run_id=terminal["run_id"],
        revision=terminal["release_sha"],
        capability_plan_sha256=terminal["capability_plan_sha256"],
        full_canary_plan_sha256=terminal["full_canary_plan_sha256"],
        fixture_sha256=terminal["fixture_sha256"],
    )
    validated_cleanup = canary_e2e.validate_cleanup_production_diff_receipt(
        cleanup_receipt,
        fixture=fixture,
        fixture_sha256=fixture_sha256,
        production_diff_sha256=validated_diff["diff_sha256"],
    )
    unsigned = {
        "schema": ISOLATED_CANARY_GOAL_PREREQUISITE_SCHEMA,
        "fixture": copy.deepcopy(dict(fixture)),
        "fixture_sha256": fixture_sha256,
        "workspace_gateway": copy.deepcopy(dict(workspace_gateway)),
        "workspace_gateway_receipt_sha256": _sha256_json(workspace_gateway),
        "cleanup_receipt": copy.deepcopy(dict(validated_cleanup)),
        "cleanup_receipt_sha256": _sha256_json(validated_cleanup),
        "goal_continuation_terminal_schema": terminal["schema"],
        "goal_continuation_terminal_sha256": terminal["terminal_sha256"],
        "isolation_equivalence_projection": copy.deepcopy(dict(projection)),
        "isolation_equivalence_projection_sha256": terminal[
            "isolation_equivalence_projection_sha256"
        ],
        "production_diff_sha256": validated_diff["diff_sha256"],
        "production_diff": copy.deepcopy(dict(validated_diff)),
        "production_diff_file_sha256": _sha256_json(validated_diff),
        "run_id": terminal["run_id"],
        "release_revision": terminal["release_sha"],
        "capability_plan_sha256": terminal["capability_plan_sha256"],
        "full_canary_plan_sha256": terminal["full_canary_plan_sha256"],
        "canary_owner_approval_receipt_sha256": terminal[
            "owner_approval_receipt_sha256"
        ],
        "canary_production_mutation_observed": False,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    return _validate_isolated_canary_goal_prerequisite(
        {**unsigned, "evidence_sha256": _sha256_json(unsigned)},
        revision=terminal["release_sha"],
    )


_SERVICE_FIELDS = frozenset({
    "schema", "name", "fragment_path", "fragment_sha256", "load_state",
    "active_state", "sub_state", "unit_file_state", "main_pid",
    "drop_in_paths", "drop_in_sha256", "need_daemon_reload",
    "triggered_by", "triggers",
    "observed_at_unix", "observation_sha256",
})


@dataclass(frozen=True)
class ServiceObservation:
    value: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, value: Any) -> "ServiceObservation":
        raw = _hashed(value, _SERVICE_FIELDS, "observation_sha256", "service observation")
        if raw["schema"] != SERVICE_SCHEMA or raw["name"] not in ({
            GATEWAY_UNIT,
            WRITER_UNIT,
            CONNECTOR_UNIT,
            PHASE_B_UNIT,
            ROUTEBACK_EDGE_UNIT,
            MAC_OPS_UNIT,
            BROWSER_UNIT,
            ISOLATED_WORKER_SOCKET_UNIT,
            ISOLATED_WORKER_SERVICE_UNIT,
        } | set(OPERATIONAL_EDGE_UNITS)):
            raise ValueError("service observation identity is invalid")
        expected_fragment = {
            GATEWAY_UNIT: GATEWAY_FRAGMENT,
            WRITER_UNIT: WRITER_FRAGMENT,
            CONNECTOR_UNIT: CONNECTOR_FRAGMENT,
            PHASE_B_UNIT: PHASE_B_FRAGMENT,
            ROUTEBACK_EDGE_UNIT: ROUTEBACK_EDGE_FRAGMENT,
            MAC_OPS_UNIT: MAC_OPS_FRAGMENT,
            BROWSER_UNIT: BROWSER_FRAGMENT,
            ISOLATED_WORKER_SOCKET_UNIT: ISOLATED_WORKER_SOCKET_FRAGMENT,
            ISOLATED_WORKER_SERVICE_UNIT: ISOLATED_WORKER_SERVICE_FRAGMENT,
            **OPERATIONAL_EDGE_FRAGMENTS,
        }[raw["name"]]
        if raw["load_state"] == "not-found":
            if (
                raw["name"]
                not in ({
                    WRITER_UNIT,
                    CONNECTOR_UNIT,
                    PHASE_B_UNIT,
                    ROUTEBACK_EDGE_UNIT,
                    MAC_OPS_UNIT,
                    BROWSER_UNIT,
                    ISOLATED_WORKER_SOCKET_UNIT,
                    ISOLATED_WORKER_SERVICE_UNIT,
                } | set(OPERATIONAL_EDGE_UNITS))
                or raw["fragment_path"] != ""
                or raw["fragment_sha256"] is not None
            ):
                raise ValueError("absent service observation is invalid")
        elif (
            raw["load_state"] != "loaded"
            or raw["fragment_path"] != expected_fragment
            or _SHA256.fullmatch(str(raw["fragment_sha256"])) is None
        ):
            raise ValueError("loaded service observation is invalid")
        state_identity = (
            raw["active_state"], raw["sub_state"], raw["main_pid"]
        )
        expected_triggered_by = (
            [ISOLATED_WORKER_SOCKET_UNIT]
            if raw["name"] == ISOLATED_WORKER_SERVICE_UNIT
            else []
        )
        expected_triggers = (
            [ISOLATED_WORKER_SERVICE_UNIT]
            if raw["name"] == ISOLATED_WORKER_SOCKET_UNIT
            else []
        )
        if (
            type(raw["main_pid"]) is not int
            or (
                state_identity != ("inactive", "dead", 0)
                and not (
                    raw["active_state"] == "active"
                    and raw["sub_state"] == "running"
                    and raw["main_pid"] > 0
                )
                and not (
                    raw["name"] == PHASE_B_UNIT
                    and state_identity == ("active", "exited", 0)
                )
                and not (
                    raw["name"] == ISOLATED_WORKER_SOCKET_UNIT
                    and state_identity == ("active", "listening", 0)
                )
            )
            or not isinstance(raw["drop_in_paths"], list)
            or raw["drop_in_paths"] not in ([], [GATEWAY_CONNECTOR_DROP_IN])
            or not isinstance(raw["drop_in_sha256"], Mapping)
            or set(raw["drop_in_sha256"]) != set(raw["drop_in_paths"])
            or any(
                _SHA256.fullmatch(str(item)) is None
                for item in raw["drop_in_sha256"].values()
            )
            or (
                raw["drop_in_paths"] == [GATEWAY_CONNECTOR_DROP_IN]
                and raw["name"] != GATEWAY_UNIT
            )
            or raw["need_daemon_reload"] is not False
            or raw["triggered_by"] != expected_triggered_by
            or raw["triggers"] != expected_triggers
            or type(raw["observed_at_unix"]) is not int
            or raw["observed_at_unix"] <= 0
        ):
            raise ValueError("service state is invalid")
        return cls(raw)

    @property
    def sha256(self) -> str:
        return str(self.value["observation_sha256"])

    @property
    def stopped(self) -> bool:
        return (
            self.value["active_state"] == "inactive"
            and self.value["sub_state"] == "dead"
            and self.value["main_pid"] == 0
        )

    def stable_identity(self) -> Mapping[str, Any]:
        return {
            key: copy.deepcopy(self.value[key])
            for key in (
                "name", "fragment_path", "fragment_sha256", "load_state",
                "unit_file_state", "drop_in_paths", "drop_in_sha256",
                "need_daemon_reload",
                "triggered_by", "triggers",
            )
        }

    def to_mapping(self) -> dict[str, Any]:
        return copy.deepcopy(dict(self.value))


_STABLE_SERVICE_IDENTITY_FIELDS = frozenset({
    "name", "fragment_path", "fragment_sha256", "load_state",
    "unit_file_state", "drop_in_paths", "drop_in_sha256",
    "need_daemon_reload",
    "triggered_by", "triggers",
})


def _validate_target_service_identity(
    value: Any,
    *,
    unit: str,
) -> dict[str, Any]:
    raw = _exact(
        value,
        _STABLE_SERVICE_IDENTITY_FIELDS,
        f"{unit} target identity",
    )
    fragment = {
        GATEWAY_UNIT: GATEWAY_FRAGMENT,
        WRITER_UNIT: WRITER_FRAGMENT,
        CONNECTOR_UNIT: CONNECTOR_FRAGMENT,
    }[unit]
    expected_drop_ins = (
        [GATEWAY_CONNECTOR_DROP_IN] if unit == GATEWAY_UNIT else []
    )
    if (
        raw["name"] != unit
        or raw["fragment_path"] != fragment
        or _SHA256.fullmatch(str(raw["fragment_sha256"])) is None
        or raw["load_state"] != "loaded"
        or raw["unit_file_state"] != "enabled"
        or raw["drop_in_paths"] != expected_drop_ins
        or not isinstance(raw["drop_in_sha256"], Mapping)
        or set(raw["drop_in_sha256"]) != set(expected_drop_ins)
        or any(
            _SHA256.fullmatch(str(item)) is None
            for item in raw["drop_in_sha256"].values()
        )
        or raw["need_daemon_reload"] is not False
        or raw["triggered_by"] != []
        or raw["triggers"] != []
    ):
        raise ValueError(f"{unit} target identity is invalid")
    return raw


_SNAPSHOT_FIELDS = frozenset({
    "schema", "database", "shape", "source_owner", "relation_oid",
    "source_row_count", "canonical14_sha256", "extended19_sha256",
    "occurred_at_cutoff", "inserted_at_cutoff", "relation_identity_sha256",
    "acl_identity_sha256", "index_identity_sha256", "observed_at_unix",
    "snapshot_sha256",
})


@dataclass(frozen=True)
class LegacySnapshot:
    value: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, value: Any) -> "LegacySnapshot":
        raw = _hashed(value, _SNAPSHOT_FIELDS, "snapshot_sha256", "legacy snapshot")
        if (
            raw["schema"] != SNAPSHOT_SCHEMA
            or raw["database"] != DATABASE
            or raw["shape"] != "legacy19"
            or not isinstance(raw["source_owner"], str)
            or _ROLE.fullmatch(raw["source_owner"]) is None
            or not isinstance(raw["relation_oid"], str)
            or re.fullmatch(r"[1-9][0-9]*", raw["relation_oid"]) is None
            or type(raw["source_row_count"]) is not int
            or raw["source_row_count"] <= 0
            or not isinstance(raw["occurred_at_cutoff"], str)
            or not raw["occurred_at_cutoff"]
            or not isinstance(raw["inserted_at_cutoff"], str)
            or not raw["inserted_at_cutoff"]
            or type(raw["observed_at_unix"]) is not int
            or raw["observed_at_unix"] <= 0
        ):
            raise ValueError("legacy snapshot identity is invalid")
        for field in (
            "canonical14_sha256", "extended19_sha256", "relation_identity_sha256",
            "acl_identity_sha256", "index_identity_sha256",
        ):
            _digest(raw[field], f"legacy snapshot {field}")
        return cls(raw)

    @property
    def sha256(self) -> str:
        return str(self.value["snapshot_sha256"])

    def to_mapping(self) -> dict[str, Any]:
        return copy.deepcopy(dict(self.value))

    def storage_identity(self) -> Mapping[str, Any]:
        """Return the immutable identity of the frozen legacy relation.

        Row counts, row digests, and cutoffs may advance while the live gateway
        is still appending.  The relation itself, its owner/OID, ACL, and index
        catalog must not change between the initial observation and the final
        stopped-tail observation.  Keeping this comparison separate from the
        row receipt makes the allowed append-only delta explicit.
        """

        return {
            key: copy.deepcopy(self.value[key])
            for key in (
                "database",
                "shape",
                "source_owner",
                "relation_oid",
                "relation_identity_sha256",
                "acl_identity_sha256",
                "index_identity_sha256",
            )
        }

    def frozen_truth(self) -> Mapping[str, Any]:
        """Return all database truth fields except collection time/digest."""

        return {
            key: copy.deepcopy(item)
            for key, item in self.value.items()
            if key not in {"observed_at_unix", "snapshot_sha256"}
        }


_LEGACY_EVENT_RECEIPT_FIELDS = frozenset({
    "event_id",
    "canonical14_row_sha256",
})
_LEGACY_TRUTH_DECISION_FIELDS = frozenset({
    "schema",
    "mode",
    "decision_id",
    "decision_event_id",
    "owner_subject_sha256",
    "reviewed_snapshot_sha256",
    "accepted_event_ids",
    "accepted_event_receipts",
    "reseed_manifest_sha256",
    "truth_epoch_id",
    "truth_epoch_sha256",
    "decision_sha256",
})


def _uuid_text(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a canonical UUID")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"{label} must be a canonical UUID") from exc
    if str(parsed) != value:
        raise ValueError(f"{label} must be a canonical UUID")
    return value


def _legacy_truth_decision(
    value: Any,
    *,
    snapshot: LegacySnapshot | None = None,
    owner_subject_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate one owner-authored, mechanically exact legacy-truth choice."""

    raw = _hashed(
        value,
        _LEGACY_TRUTH_DECISION_FIELDS,
        "decision_sha256",
        "legacy truth decision",
    )
    mode = raw["mode"]
    if (
        raw["schema"] != LEGACY_TRUTH_DECISION_SCHEMA
        or mode not in {"reseed_accepted_events", "start_new_truth_epoch"}
        or not isinstance(raw["decision_id"], str)
        or re.fullmatch(
            r"legacy-truth-decision:[0-9a-f]{8}-[0-9a-f]{4}-"
            r"[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            raw["decision_id"],
        )
        is None
    ):
        raise ValueError("legacy truth decision identity is invalid")
    _uuid_text(raw["decision_event_id"], "legacy truth decision event")
    _digest(raw["owner_subject_sha256"], "legacy truth decision owner")
    _digest(raw["reviewed_snapshot_sha256"], "legacy truth reviewed snapshot")
    if (
        snapshot is not None
        and raw["reviewed_snapshot_sha256"] != snapshot.sha256
    ):
        raise ValueError("legacy truth decision snapshot binding is invalid")
    if (
        owner_subject_sha256 is not None
        and raw["owner_subject_sha256"] != owner_subject_sha256
    ):
        raise ValueError("legacy truth decision owner binding is invalid")

    event_ids = raw["accepted_event_ids"]
    receipts = raw["accepted_event_receipts"]
    if (
        not isinstance(event_ids, list)
        or not isinstance(receipts, list)
        or len(event_ids) != len(receipts)
        or len(event_ids) > _MAX_ACCEPTED_LEGACY_EVENTS
        or len(set(event_ids)) != len(event_ids)
        or event_ids != sorted(event_ids)
        or raw["decision_event_id"] in event_ids
    ):
        raise ValueError("legacy truth accepted event set is invalid")
    normalized_receipts: list[dict[str, str]] = []
    for index, item in enumerate(receipts):
        receipt = _exact(
            item,
            _LEGACY_EVENT_RECEIPT_FIELDS,
            "legacy truth event receipt",
        )
        event_id = _uuid_text(
            receipt["event_id"],
            "legacy truth accepted event",
        )
        _digest(
            receipt["canonical14_row_sha256"],
            "legacy truth accepted event receipt",
        )
        if event_id != event_ids[index]:
            raise ValueError("legacy truth accepted event receipt order is invalid")
        normalized_receipts.append(receipt)
    if snapshot is not None and len(event_ids) > snapshot.value["source_row_count"]:
        raise ValueError("legacy truth accepted event set exceeds reviewed snapshot")

    if mode == "reseed_accepted_events":
        manifest_unsigned = {
            "schema": LEGACY_RESEED_MANIFEST_SCHEMA,
            "reviewed_snapshot_sha256": raw["reviewed_snapshot_sha256"],
            "accepted_event_ids": event_ids,
            "accepted_event_receipts": normalized_receipts,
        }
        if (
            not event_ids
            or raw["reseed_manifest_sha256"]
            != _sha256_json(manifest_unsigned)
            or raw["truth_epoch_id"] is not None
            or raw["truth_epoch_sha256"] is not None
        ):
            raise ValueError("legacy reseed manifest is invalid")
    else:
        epoch_id = raw["truth_epoch_id"]
        if (
            event_ids
            or receipts
            or raw["reseed_manifest_sha256"] is not None
            or not isinstance(epoch_id, str)
            or re.fullmatch(
                r"truth-epoch:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
                r"[0-9a-f]{4}-[0-9a-f]{12}",
                epoch_id,
            )
            is None
        ):
            raise ValueError("new truth epoch decision is invalid")
        epoch_unsigned = {
            "schema": NEW_TRUTH_EPOCH_SCHEMA,
            "reviewed_snapshot_sha256": raw["reviewed_snapshot_sha256"],
            "truth_epoch_id": epoch_id,
        }
        if raw["truth_epoch_sha256"] != _sha256_json(epoch_unsigned):
            raise ValueError("new truth epoch digest is invalid")
    return raw


def build_legacy_truth_decision(
    *,
    mode: str,
    decision_id: str,
    decision_event_id: str,
    owner_subject_sha256: str,
    reviewed_snapshot: LegacySnapshot,
    accepted_event_receipts: list[Mapping[str, Any]] | None = None,
    truth_epoch_id: str | None = None,
) -> Mapping[str, Any]:
    """Build one typed decision without interpreting any legacy event meaning."""

    receipts = [copy.deepcopy(dict(item)) for item in (accepted_event_receipts or [])]
    event_ids = [str(item.get("event_id") or "") for item in receipts]
    manifest_sha256: str | None = None
    epoch_sha256: str | None = None
    if mode == "reseed_accepted_events":
        manifest_sha256 = _sha256_json({
            "schema": LEGACY_RESEED_MANIFEST_SCHEMA,
            "reviewed_snapshot_sha256": reviewed_snapshot.sha256,
            "accepted_event_ids": event_ids,
            "accepted_event_receipts": receipts,
        })
    elif mode == "start_new_truth_epoch":
        epoch_sha256 = _sha256_json({
            "schema": NEW_TRUTH_EPOCH_SCHEMA,
            "reviewed_snapshot_sha256": reviewed_snapshot.sha256,
            "truth_epoch_id": truth_epoch_id,
        })
    unsigned = {
        "schema": LEGACY_TRUTH_DECISION_SCHEMA,
        "mode": mode,
        "decision_id": decision_id,
        "decision_event_id": decision_event_id,
        "owner_subject_sha256": owner_subject_sha256,
        "reviewed_snapshot_sha256": reviewed_snapshot.sha256,
        "accepted_event_ids": event_ids,
        "accepted_event_receipts": receipts,
        "reseed_manifest_sha256": manifest_sha256,
        "truth_epoch_id": truth_epoch_id,
        "truth_epoch_sha256": epoch_sha256,
    }
    return _legacy_truth_decision(
        {**unsigned, "decision_sha256": _sha256_json(unsigned)},
        snapshot=reviewed_snapshot,
        owner_subject_sha256=owner_subject_sha256,
    )


_TARGET_FIELDS = frozenset({
    "project", "zone", "vm", "database", "sql_instance", "sql_host",
    "tls_server_name", "port", "writer_login",
})


def _validate_target(value: Any) -> dict[str, Any]:
    raw = _exact(value, _TARGET_FIELDS, "production target")
    if (
        raw["project"] != PROJECT
        or raw["zone"] != ZONE
        or raw["vm"] != VM_NAME
        or raw["database"] != DATABASE
        or not isinstance(raw["sql_instance"], str)
        or not raw["sql_instance"]
        or not isinstance(raw["sql_host"], str)
        or not raw["sql_host"]
        or not isinstance(raw["tls_server_name"], str)
        or not raw["tls_server_name"]
        or raw["port"] != 5432
        or not isinstance(raw["writer_login"], str)
        or _ROLE.fullmatch(raw["writer_login"]) is None
    ):
        raise ValueError("production target is invalid")
    return raw


_BINDING_FIELDS = frozenset({"path", "sha256"})


def _artifact(value: Any, label: str, revision: str) -> dict[str, Any]:
    raw = _exact(value, _BINDING_FIELDS, label)
    path = Path(str(raw["path"]))
    release_root = PRODUCTION_RELEASE_BASE / f"hermes-agent-{revision[:12]}"
    if (
        not path.is_absolute()
        or ".." in path.parts
        or path == release_root
        or release_root not in path.parents
        or Path(os.path.normpath(str(path))) != path
    ):
        raise ValueError(f"{label} path is not release-addressed")
    _digest(raw["sha256"], f"{label} sha256")
    return raw


_FREEZE_FIELDS = frozenset({
    "schema", "release_revision", "target", "owner_subject_sha256",
    "owner_public_key_ed25519_hex", "owner_key_id", "gateway_before",
    "writer_before", "connector_before", "initial_snapshot",
    "owner_runtime_attestation", "observe_artifact", "cutover_authority", "states",
    "secret_material_recorded", "plan_sha256",
})

_FINAL_TAIL_BOUNDS_FIELDS = frozenset({
    "max_appended_rows",
    "max_capture_delay_seconds",
})
_CUTOVER_AUTHORITY_FIELDS = frozenset({
    "schema",
    "release_revision",
    "artifacts",
    "gateway_target_identity",
    "writer_target_identity",
    "connector_target_identity",
    "host_transition",
    "capability_topology",
    "cron_inventory",
    "cron_continuity_plan",
    "mechanical_job_host_facts",
    "mechanical_job_package",
    "isolated_canary_goal_prerequisite",
    "legacy_truth_decision",
    "final_tail_bounds",
    "rollback_contract",
    "secret_material_recorded",
    "authority_sha256",
})
_CUTOVER_AUTHORITY_SCHEMA = "muncho-production-cutover-authority.v3"


def _validate_cutover_authority(
    value: Any,
    *,
    revision: str,
    gateway_pre: "ServiceObservation",
    writer_pre: "ServiceObservation",
    connector_pre: "ServiceObservation",
) -> dict[str, Any]:
    raw = _hashed(
        value,
        _CUTOVER_AUTHORITY_FIELDS,
        "authority_sha256",
        "cutover authority",
    )
    if (
        raw["schema"] != _CUTOVER_AUTHORITY_SCHEMA
        or raw["release_revision"] != revision
        or raw["secret_material_recorded"] is not False
    ):
        raise ValueError("cutover authority identity is invalid")
    artifacts = _exact(
        raw["artifacts"],
        frozenset(_CUTOVER_ARTIFACT_NAMES),
        "cutover authority artifacts",
    )
    for name in _CUTOVER_ARTIFACT_NAMES:
        _artifact(artifacts[name], f"cutover authority {name}", revision)
    gateway_target = _validate_target_service_identity(
        raw["gateway_target_identity"], unit=GATEWAY_UNIT
    )
    writer_target = _validate_target_service_identity(
        raw["writer_target_identity"], unit=WRITER_UNIT
    )
    connector_target = _validate_target_service_identity(
        raw["connector_target_identity"], unit=CONNECTOR_UNIT
    )
    topology = validate_production_capability_topology(raw["capability_topology"])
    try:
        cron_inventory = production_cron_migration.validate_inventory(
            raw["cron_inventory"]
        )
        host_facts = mechanical_job_rail.validate_host_facts(
            raw["mechanical_job_host_facts"]
        )
        mechanical_package = mechanical_job_rail.validate_package_manifest(
            raw["mechanical_job_package"],
            revision=revision,
            host_facts_sha256=host_facts["host_facts_sha256"],
        )
        continuity_plan = production_cron_migration.validate_owner_approved_plan(
            cron_inventory,
            raw["cron_continuity_plan"],
            mechanical_package["manifest_sha256"],
        )
    except (
        production_cron_migration.ProductionCronMigrationError,
        mechanical_job_rail.MechanicalJobRailError,
    ) as exc:
        raise ValueError("cutover cron continuity authority is invalid") from exc
    if continuity_plan["cutover_executable"] is not True:
        raise ValueError("cutover cron continuity authority is not executable")
    canary_goal = _validate_isolated_canary_goal_prerequisite(
        raw["isolated_canary_goal_prerequisite"],
        revision=revision,
    )
    _validate_host_transition(
        raw["host_transition"],
        gateway_pre=gateway_pre,
        writer_pre=writer_pre,
        connector_pre=connector_pre,
        gateway_target=gateway_target,
        writer_target=writer_target,
        connector_target=connector_target,
        capability_topology=topology,
    )
    decision = _legacy_truth_decision(raw["legacy_truth_decision"])
    bounds = _exact(
        raw["final_tail_bounds"],
        _FINAL_TAIL_BOUNDS_FIELDS,
        "final-tail bounds",
    )
    if (
        type(bounds["max_appended_rows"]) is not int
        or not 0 <= bounds["max_appended_rows"] <= 10_000_000
        or type(bounds["max_capture_delay_seconds"]) is not int
        or not 1 <= bounds["max_capture_delay_seconds"] <= 3600
    ):
        raise ValueError("final-tail bounds are invalid")
    rollback = _exact(
        raw["rollback_contract"],
        frozenset({
            "database_rollback_sha256",
            "host_rollback_sha256",
            "requires_gateway_stopped",
            "requires_writer_stopped",
            "requires_connector_stopped",
            "requires_zero_canonical_writer_writes",
            "restart_legacy_gateway",
        }),
        "cutover authority rollback contract",
    )
    expected_rollback = {
        "database_rollback_sha256": artifacts["database_rollback"]["sha256"],
        "host_rollback_sha256": artifacts["host_rollback"]["sha256"],
        "requires_gateway_stopped": True,
        "requires_writer_stopped": True,
        "requires_connector_stopped": True,
        "requires_zero_canonical_writer_writes": True,
        "restart_legacy_gateway": True,
    }
    if rollback != expected_rollback:
        raise ValueError("cutover authority rollback contract is invalid")
    return {
        **raw,
        "artifacts": copy.deepcopy(dict(artifacts)),
        "capability_topology": copy.deepcopy(dict(topology)),
        "cron_inventory": copy.deepcopy(dict(cron_inventory)),
        "cron_continuity_plan": copy.deepcopy(dict(continuity_plan)),
        "mechanical_job_host_facts": copy.deepcopy(dict(host_facts)),
        "mechanical_job_package": copy.deepcopy(dict(mechanical_package)),
        "isolated_canary_goal_prerequisite": copy.deepcopy(dict(canary_goal)),
        "legacy_truth_decision": copy.deepcopy(dict(decision)),
    }


def build_cutover_authority(
    *,
    release_revision: str,
    artifacts: Mapping[str, Any],
    gateway_before: "ServiceObservation",
    writer_before: "ServiceObservation",
    connector_before: "ServiceObservation",
    gateway_target_identity: Mapping[str, Any],
    writer_target_identity: Mapping[str, Any],
    connector_target_identity: Mapping[str, Any],
    host_transition: Mapping[str, Any],
    capability_topology: Mapping[str, Any],
    cron_inventory: Mapping[str, Any],
    cron_continuity_plan: Mapping[str, Any],
    mechanical_job_host_facts: Mapping[str, Any],
    mechanical_job_package: Mapping[str, Any],
    isolated_canary_goal_prerequisite: Mapping[str, Any],
    legacy_truth_decision: Mapping[str, Any],
    max_appended_rows: int,
    max_capture_delay_seconds: int,
) -> Mapping[str, Any]:
    artifact_set = copy.deepcopy(dict(artifacts))
    unsigned = {
        "schema": _CUTOVER_AUTHORITY_SCHEMA,
        "release_revision": release_revision,
        "artifacts": artifact_set,
        "gateway_target_identity": copy.deepcopy(dict(gateway_target_identity)),
        "writer_target_identity": copy.deepcopy(dict(writer_target_identity)),
        "connector_target_identity": copy.deepcopy(dict(connector_target_identity)),
        "host_transition": copy.deepcopy(dict(host_transition)),
        "capability_topology": copy.deepcopy(dict(capability_topology)),
        "cron_inventory": copy.deepcopy(dict(cron_inventory)),
        "cron_continuity_plan": copy.deepcopy(dict(cron_continuity_plan)),
        "mechanical_job_host_facts": copy.deepcopy(
            dict(mechanical_job_host_facts)
        ),
        "mechanical_job_package": copy.deepcopy(dict(mechanical_job_package)),
        "isolated_canary_goal_prerequisite": copy.deepcopy(
            dict(isolated_canary_goal_prerequisite)
        ),
        "legacy_truth_decision": copy.deepcopy(dict(legacy_truth_decision)),
        "final_tail_bounds": {
            "max_appended_rows": max_appended_rows,
            "max_capture_delay_seconds": max_capture_delay_seconds,
        },
        "rollback_contract": {
            "database_rollback_sha256": artifact_set["database_rollback"]["sha256"],
            "host_rollback_sha256": artifact_set["host_rollback"]["sha256"],
            "requires_gateway_stopped": True,
            "requires_writer_stopped": True,
            "requires_connector_stopped": True,
            "requires_zero_canonical_writer_writes": True,
            "restart_legacy_gateway": True,
        },
        "secret_material_recorded": False,
    }
    return _validate_cutover_authority(
        {**unsigned, "authority_sha256": _sha256_json(unsigned)},
        revision=release_revision,
        gateway_pre=gateway_before,
        writer_pre=writer_before,
        connector_pre=connector_before,
    )


@dataclass(frozen=True)
class FreezePlan:
    value: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, value: Any) -> "FreezePlan":
        raw = _hashed(value, _FREEZE_FIELDS, "plan_sha256", "freeze plan")
        revision = raw["release_revision"]
        if raw["schema"] != FREEZE_PLAN_SCHEMA or not isinstance(revision, str) or _REVISION.fullmatch(revision) is None:
            raise ValueError("freeze plan identity is invalid")
        _validate_target(raw["target"])
        _owner_runtime_attestation(
            raw["owner_runtime_attestation"],
            revision=revision,
        )
        _digest(raw["owner_subject_sha256"], "freeze owner subject")
        public = raw["owner_public_key_ed25519_hex"]
        if not isinstance(public, str) or re.fullmatch(r"[0-9a-f]{64}", public) is None or raw["owner_key_id"] != hashlib.sha256(bytes.fromhex(public)).hexdigest():
            raise ValueError("freeze owner key is invalid")
        gateway = ServiceObservation.from_mapping(raw["gateway_before"])
        writer = ServiceObservation.from_mapping(raw["writer_before"])
        connector = ServiceObservation.from_mapping(raw["connector_before"])
        if (
            gateway.value["name"] != GATEWAY_UNIT
            or gateway.stopped
            or gateway.value["load_state"] != "loaded"
            or gateway.value["unit_file_state"] != "enabled"
            or writer.value["name"] != WRITER_UNIT
            or not writer.stopped
            or writer.value["load_state"] != "not-found"
            or writer.value["unit_file_state"] != ""
            or connector.value["name"] != CONNECTOR_UNIT
            or not connector.stopped
            or connector.value["load_state"] != "not-found"
            or connector.value["unit_file_state"] != ""
        ):
            raise ValueError("freeze service precondition is invalid")
        initial_snapshot = LegacySnapshot.from_mapping(raw["initial_snapshot"])
        authority = _validate_cutover_authority(
            raw["cutover_authority"],
            revision=revision,
            gateway_pre=gateway,
            writer_pre=writer,
            connector_pre=connector,
        )
        _legacy_truth_decision(
            authority["legacy_truth_decision"],
            snapshot=initial_snapshot,
            owner_subject_sha256=raw["owner_subject_sha256"],
        )
        observe = _artifact(raw["observe_artifact"], "observe artifact", revision)
        if observe != authority["artifacts"]["observe"]:
            raise ValueError("freeze observe artifact is not authority-bound")
        if raw["states"] != ["authority", "gateway_stopped", "final_tail_captured"] or raw["secret_material_recorded"] is not False:
            raise ValueError("freeze plan contract is invalid")
        return cls(raw)

    @property
    def sha256(self) -> str:
        return str(self.value["plan_sha256"])

    def to_mapping(self) -> dict[str, Any]:
        return copy.deepcopy(dict(self.value))


def build_freeze_plan(
    *,
    release_revision: str,
    target: Mapping[str, Any],
    owner_subject_sha256: str,
    owner_public_key_ed25519_hex: str,
    gateway_before: ServiceObservation,
    writer_before: ServiceObservation,
    connector_before: ServiceObservation,
    initial_snapshot: LegacySnapshot,
    cutover_authority: Mapping[str, Any],
    owner_runtime_attestation: Mapping[str, Any],
) -> FreezePlan:
    public = owner_public_key_ed25519_hex
    authority = _validate_cutover_authority(
        cutover_authority,
        revision=release_revision,
        gateway_pre=gateway_before,
        writer_pre=writer_before,
        connector_pre=connector_before,
    )
    _legacy_truth_decision(
        authority["legacy_truth_decision"],
        snapshot=initial_snapshot,
        owner_subject_sha256=owner_subject_sha256,
    )
    runtime_attestation = _owner_runtime_attestation(
        owner_runtime_attestation,
        revision=release_revision,
    )
    unsigned = {
        "schema": FREEZE_PLAN_SCHEMA,
        "release_revision": release_revision,
        "target": copy.deepcopy(dict(target)),
        "owner_subject_sha256": owner_subject_sha256,
        "owner_public_key_ed25519_hex": public,
        "owner_key_id": hashlib.sha256(bytes.fromhex(public)).hexdigest(),
        "gateway_before": gateway_before.to_mapping(),
        "writer_before": writer_before.to_mapping(),
        "connector_before": connector_before.to_mapping(),
        "initial_snapshot": initial_snapshot.to_mapping(),
        "owner_runtime_attestation": copy.deepcopy(
            dict(runtime_attestation)
        ),
        "observe_artifact": copy.deepcopy(dict(authority["artifacts"]["observe"])),
        "cutover_authority": copy.deepcopy(dict(authority)),
        "states": ["authority", "gateway_stopped", "final_tail_captured"],
        "secret_material_recorded": False,
    }
    return FreezePlan.from_mapping({**unsigned, "plan_sha256": _sha256_json(unsigned)})


_FREEZE_RECEIPT_FIELDS = frozenset({
    "schema", "freeze_plan_sha256", "approval_sha256", "gateway_stopped",
    "writer_stopped", "final_snapshot", "initial_row_count",
    "final_row_count", "append_only_row_floor_preserved", "captured_at_unix",
    "receipt_sha256",
})


@dataclass(frozen=True)
class FinalTailReceipt:
    value: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, value: Any, *, plan: FreezePlan | None = None) -> "FinalTailReceipt":
        raw = _hashed(value, _FREEZE_RECEIPT_FIELDS, "receipt_sha256", "final-tail receipt")
        snapshot = LegacySnapshot.from_mapping(raw["final_snapshot"])
        if (
            raw["schema"] != FREEZE_RECEIPT_SCHEMA
            or (plan is not None and raw["freeze_plan_sha256"] != plan.sha256)
            or raw["gateway_stopped"] is not True
            or raw["writer_stopped"] is not True
            or type(raw["initial_row_count"]) is not int
            or raw["final_row_count"] != snapshot.value["source_row_count"]
            or raw["final_row_count"] < raw["initial_row_count"]
            or raw["append_only_row_floor_preserved"] is not True
            or type(raw["captured_at_unix"]) is not int
        ):
            raise ValueError("final-tail receipt is invalid")
        _digest(raw["approval_sha256"], "final-tail approval")
        return cls(raw)

    @property
    def sha256(self) -> str:
        return str(self.value["receipt_sha256"])

    @property
    def snapshot(self) -> LegacySnapshot:
        return LegacySnapshot.from_mapping(self.value["final_snapshot"])

    def to_mapping(self) -> dict[str, Any]:
        return copy.deepcopy(dict(self.value))


_CUTOVER_ARTIFACT_NAMES = (
    "observe", "database_apply", "database_rollback", "database_postflight",
    "host_activation", "host_rollback",
)
_HOST_TRANSITION_SCHEMA = "muncho-production-host-transition-manifest.v1"
_HOST_TRANSITION_FILE_NAMES = (
    "gateway_unit",
    "writer_unit",
    "connector_unit",
    "phase_b_unit",
    "routeback_unit",
    "mac_ops_unit",
    "browser_unit",
    "browser_config",
    "isolated_worker_socket_unit",
    "isolated_worker_service_unit",
    "isolated_worker_config",
    "gateway_connector_drop_in",
    "gateway_config",
    "writer_config",
    "connector_config",
    "routeback_config",
    "mac_ops_config",
    "api_bearer_verifier",
    "api_approval_verifier",
    "operational_edge_client_config",
    *(
        f"operational_edge_unit_{domain}"
        for domain in sorted(CREDENTIALS_BY_DOMAIN)
    ),
    *(
        f"operational_edge_config_{domain}"
        for domain in sorted(CREDENTIALS_BY_DOMAIN)
    ),
)
_HOST_TRANSITION_FIELDS = frozenset({
    "schema", "files", "identity_foundation", "discord_key_foundation",
    "operational_edge_key_foundation",
    "operational_edge_key_foundation_sha256",
    "operational_edge_receipt_public_key_ids",
    "release_owner_uid", "release_owner_gid",
    "isolated_worker_lease_mountpoint",
    "connector_token",
    "gateway_retired_token_paths",
    "approval_passkey", "retired_approval_passkey_paths",
    "routeback_token_paths", "gateway_direct_discord_enabled",
    "gateway_relay_platforms", "connector_operation_class",
    "routeback_operation_class", "discord_dm_allowed",
    "discord_policy_continuity",
    "secret_material_recorded", "manifest_sha256",
})
_HOST_FILE_FIELDS = frozenset({
    "staged_path", "target_path", "sha256", "uid", "gid", "mode", "pre",
})
_HOST_FILE_PRE_FIELDS = frozenset({"state", "sha256", "uid", "gid", "mode"})
_HOST_DIRECTORY_FIELDS = frozenset({"target_path", "uid", "gid", "mode", "pre"})
_HOST_DIRECTORY_PRE_FIELDS = frozenset({"state", "uid", "gid", "mode"})
_CONNECTOR_TOKEN_FIELDS = frozenset({
    "path", "uid", "gid", "mode", "regular_one_link",
    "content_or_digest_recorded", "gateway_readable", "source_path",
    "source_uid", "source_gid", "source_mode",
})
_APPROVAL_PASSKEY_FIELDS = frozenset({
    "path", "uid", "gid", "mode", "regular_one_link",
    "content_or_digest_recorded", "gateway_readable", "source_path",
    "source_uid", "source_gid", "source_mode",
})
_DISCORD_KEY_FOUNDATION_SCHEMA = "muncho-production-discord-key-foundation.v1"
_DISCORD_KEY_FOUNDATION_FIELDS = frozenset({
    "schema", "writer", "edge", "pre_state", "keys_distinct",
    "private_content_or_digest_recorded", "secret_material_recorded",
    "foundation_sha256",
})
_DISCORD_KEY_PAIR_FIELDS = frozenset({
    "staged_private_path", "private_path", "private_uid", "private_gid",
    "private_mode", "public_path", "public_uid", "public_gid",
    "public_mode", "public_key_id",
})
_DISCORD_POLICY_FIELDS = frozenset({
    "allowed_guild_ids",
    "allowed_channel_ids",
    "allowed_user_ids",
    "allowed_role_ids",
    "allow_all_users",
    "allow_bot_authors",
    "require_mention",
    "auto_thread",
    "thread_require_mention",
    "discord_dm_allowed",
    "free_response_channel_ids",
    "public_only",
    "author_policy",
})
_DISCORD_POLICY_CONTINUITY_FIELDS = frozenset({
    "schema",
    "source_evidence_sha256",
    "legacy_policy",
    "target_policy",
    "exact_membership_preserved",
    "reviewed_reconciliation",
    "secret_material_recorded",
    "continuity_sha256",
})
_DISCORD_POLICY_CONTINUITY_SCHEMA = (
    "muncho-production-discord-policy-continuity.v2"
)
OWNER_DISCORD_USER_ID = "1279454038731264061"

_IDENTITY_FOUNDATION_SCHEMA = "muncho-production-host-identity-foundation.v1"
_OPERATIONAL_EDGE_USER_ROLES = tuple(
    f"operational_edge_{domain}" for domain in sorted(CREDENTIALS_BY_DOMAIN)
)
_OPERATIONAL_EDGE_CLIENT_GROUP_ROLES = tuple(
    f"operational_edge_{domain}_client"
    for domain in sorted(CREDENTIALS_BY_DOMAIN)
)
_IDENTITY_USER_ROLES = (
    "gateway",
    "writer",
    "projector",
    "routeback",
    "connector",
    "mac_ops",
    "browser",
    "worker",
) + _OPERATIONAL_EDGE_USER_ROLES
_IDENTITY_GROUP_ROLES = (
    "gateway",
    "writer",
    "projector",
    "writer_client",
    "routeback",
    "connector",
    "mac_ops",
    "browser",
    "worker",
    "worker_client",
) + _OPERATIONAL_EDGE_USER_ROLES + _OPERATIONAL_EDGE_CLIENT_GROUP_ROLES
_IDENTITY_USER_NAMES = {
    "gateway": "ai-platform-brain",
    "writer": "muncho-canonical-writer",
    "projector": "muncho-projector",
    "routeback": "muncho-discord-egress",
    "connector": "muncho-discord-connector",
    "mac_ops": "muncho-mac-ops-edge",
    "browser": "muncho-capability-browser",
    "worker": "muncho-worker",
    **{
        f"operational_edge_{domain}": f"muncho-edge-{domain}"
        for domain in sorted(CREDENTIALS_BY_DOMAIN)
    },
}
_IDENTITY_GROUP_NAMES = {
    **_IDENTITY_USER_NAMES,
    "writer_client": "muncho-writer-client",
    "worker_client": "muncho-worker-clients",
    **{
        f"operational_edge_{domain}_client": f"muncho-edge-{domain}-c"
        for domain in sorted(CREDENTIALS_BY_DOMAIN)
    },
}
_IDENTITY_SUPPLEMENTARY_ROLES = {
    "gateway": [
        "browser", "connector", "mac_ops", "routeback", "worker_client",
        "writer_client", *_OPERATIONAL_EDGE_CLIENT_GROUP_ROLES,
    ],
    "writer": ["projector"],
    "projector": [],
    "routeback": [],
    "connector": [],
    "mac_ops": [],
    "browser": [],
    "worker": [],
    **{
        f"operational_edge_{domain}": [
            f"operational_edge_{domain}_client"
        ]
        for domain in sorted(CREDENTIALS_BY_DOMAIN)
    },
}
_IDENTITY_FOUNDATION_FIELDS = frozenset({
    "schema",
    "users",
    "groups",
    "retain_created_dormant_on_rollback",
    "secret_material_recorded",
    "foundation_sha256",
})
_IDENTITY_USER_FIELDS = frozenset({
    "name",
    "uid",
    "primary_group",
    "home",
    "shell",
    "supplementary_groups",
    "pre",
})
_IDENTITY_USER_PRE_FIELDS = frozenset({
    "state", "uid", "gid", "home", "shell", "supplementary_group_names",
})
_IDENTITY_GROUP_FIELDS = frozenset({"name", "gid", "members", "pre"})
_IDENTITY_GROUP_PRE_FIELDS = frozenset({"state", "gid", "members"})
_SYSTEM_IDENTITY = re.compile(r"^[a-z_][a-z0-9_-]{0,63}$")


def _validate_discord_policy(
    value: Any,
    label: str,
    *,
    target: bool,
) -> dict[str, Any]:
    raw = _exact(value, _DISCORD_POLICY_FIELDS, label)
    for field in (
        "allowed_guild_ids",
        "allowed_channel_ids",
    ):
        values = raw[field]
        if (
            not isinstance(values, list)
            or not 1 <= len(values) <= 128
            or values != sorted(values)
            or len(set(values)) != len(values)
            or any(
                not isinstance(item, str)
                or not item.isdigit()
                or item.startswith("0")
                for item in values
            )
        ):
            raise ValueError(f"{label} {field} is invalid")
    for field in (
        "allowed_user_ids",
        "allowed_role_ids",
        "free_response_channel_ids",
    ):
        values = raw[field]
        if (
            not isinstance(values, list)
            or len(values) > 128
            or values != sorted(values)
            or len(set(values)) != len(values)
            or any(
                not isinstance(item, str)
                or not item.isdigit()
                or item.startswith("0")
                for item in values
            )
        ):
            raise ValueError(f"{label} {field} is invalid")
    for field in (
        "allow_all_users",
        "allow_bot_authors",
        "require_mention",
        "auto_thread",
        "thread_require_mention",
        "discord_dm_allowed",
        "public_only",
    ):
        if type(raw[field]) is not bool:
            raise ValueError(f"{label} {field} is invalid")
    if (
        raw["allow_all_users"] is not False
        or raw["allow_bot_authors"] is not False
        or raw["require_mention"] is not True
        or raw["auto_thread"] is not True
        or raw["thread_require_mention"] is not False
        or raw["discord_dm_allowed"] is not False
        or not set(raw["free_response_channel_ids"]).issubset(
            raw["allowed_channel_ids"]
        )
    ):
        raise ValueError(f"{label} safety contract is invalid")
    if target:
        expected_channels = sorted(SKYVISION_APPROVED_OPERATIONAL_CHANNEL_IDS)
        expected_free_response = sorted(
            {
                SKYVISION_CONTROL_TOWER_CHANNEL_ID,
                SKYVISION_NASI_AI_OPS_CHANNEL_ID,
            }
        )
        if (
            raw["allowed_guild_ids"] != [SKYVISION_GUILD_ID]
            or raw["allowed_channel_ids"] != expected_channels
            or raw["allowed_user_ids"] != []
            or raw["allowed_role_ids"] != []
            or raw["free_response_channel_ids"] != expected_free_response
            or raw["public_only"] is not False
            or raw["author_policy"] != "guild_acl"
        ):
            raise ValueError(f"{label} guild ACL contract is invalid")
    elif (
        raw["author_policy"] != "exact_ids_or_roles"
        or OWNER_DISCORD_USER_ID not in raw["allowed_user_ids"]
        or not raw["allowed_role_ids"]
    ):
        raise ValueError(f"{label} legacy authority is invalid")
    return raw


def validate_discord_policy_continuity(value: Any) -> dict[str, Any]:
    raw = _hashed(
        value,
        _DISCORD_POLICY_CONTINUITY_FIELDS,
        "continuity_sha256",
        "Discord policy continuity",
    )
    legacy = _validate_discord_policy(
        raw["legacy_policy"], "legacy Discord policy", target=False
    )
    target = _validate_discord_policy(
        raw["target_policy"], "target Discord policy", target=True
    )
    if (
        raw["schema"] != _DISCORD_POLICY_CONTINUITY_SCHEMA
        or _SHA256.fullmatch(str(raw["source_evidence_sha256"])) is None
        or legacy == target
        or raw["exact_membership_preserved"] is not False
        or raw["reviewed_reconciliation"] is not True
        or raw["secret_material_recorded"] is not False
    ):
        raise ValueError("Discord policy continuity is invalid")
    return raw


def build_discord_policy_continuity(
    *,
    source_evidence_sha256: str,
    legacy_policy: Mapping[str, Any],
    target_policy: Mapping[str, Any],
) -> Mapping[str, Any]:
    legacy = _validate_discord_policy(
        legacy_policy,
        "legacy Discord policy",
        target=False,
    )
    target = _validate_discord_policy(
        target_policy,
        "target Discord policy",
        target=True,
    )
    unsigned = {
        "schema": _DISCORD_POLICY_CONTINUITY_SCHEMA,
        "source_evidence_sha256": source_evidence_sha256,
        "legacy_policy": copy.deepcopy(legacy),
        "target_policy": copy.deepcopy(target),
        "exact_membership_preserved": False,
        "reviewed_reconciliation": True,
        "secret_material_recorded": False,
    }
    return validate_discord_policy_continuity(
        {**unsigned, "continuity_sha256": _sha256_json(unsigned)}
    )


def _safe_absolute_path(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} is invalid")
    path = Path(value)
    if (
        not path.is_absolute()
        or ".." in path.parts
        or Path(os.path.normpath(value)) != path
        or len(value.encode("utf-8")) > 4096
    ):
        raise ValueError(f"{label} is invalid")
    return value


def _validate_identity_foundation(value: Any) -> dict[str, Any]:
    raw = _hashed(
        value,
        _IDENTITY_FOUNDATION_FIELDS,
        "foundation_sha256",
        "identity foundation",
    )
    if (
        raw["schema"] != _IDENTITY_FOUNDATION_SCHEMA
        or raw["retain_created_dormant_on_rollback"] is not True
        or raw["secret_material_recorded"] is not False
    ):
        raise ValueError("identity foundation policy is invalid")
    users = _exact(
        raw["users"],
        frozenset(_IDENTITY_USER_ROLES),
        "identity foundation users",
    )
    groups = _exact(
        raw["groups"],
        frozenset(_IDENTITY_GROUP_ROLES),
        "identity foundation groups",
    )
    validated_groups: dict[str, dict[str, Any]] = {}
    group_ids: set[int] = set()
    for role in _IDENTITY_GROUP_ROLES:
        item = _exact(
            groups[role], _IDENTITY_GROUP_FIELDS, f"identity group {role}"
        )
        members = item["members"]
        pre = _exact(
            item["pre"],
            _IDENTITY_GROUP_PRE_FIELDS,
            f"identity group {role} pre",
        )
        if (
            item["name"] != _IDENTITY_GROUP_NAMES[role]
            or type(item["gid"]) is not int
            or not 0 < item["gid"] < (1 << 31)
            or item["gid"] in group_ids
            or not isinstance(members, list)
            or members != sorted(members)
            or len(set(members)) != len(members)
            or any(
                not isinstance(member, str)
                or _SYSTEM_IDENTITY.fullmatch(member) is None
                for member in members
            )
        ):
            raise ValueError(f"identity group {role} is invalid")
        group_ids.add(item["gid"])
        if pre["state"] == "absent":
            if pre != {"state": "absent", "gid": None, "members": None}:
                raise ValueError(f"identity group {role} pre-state is invalid")
        elif pre["state"] == "present":
            if (
                type(pre["gid"]) is not int
                or pre["gid"] != item["gid"]
                or not isinstance(pre["members"], list)
                or pre["members"] != sorted(pre["members"])
                or len(set(pre["members"])) != len(pre["members"])
                or any(
                    not isinstance(member, str)
                    or _SYSTEM_IDENTITY.fullmatch(member) is None
                    for member in pre["members"]
                )
            ):
                raise ValueError(f"identity group {role} pre-state is invalid")
        else:
            raise ValueError(f"identity group {role} pre-state is invalid")
        validated_groups[role] = item

    validated_users: dict[str, dict[str, Any]] = {}
    user_ids: set[int] = set()
    for role in _IDENTITY_USER_ROLES:
        item = _exact(
            users[role], _IDENTITY_USER_FIELDS, f"identity user {role}"
        )
        pre = _exact(
            item["pre"],
            _IDENTITY_USER_PRE_FIELDS,
            f"identity user {role} pre",
        )
        supplementary = item["supplementary_groups"]
        required = {
            _IDENTITY_GROUP_NAMES[group_role]
            for group_role in _IDENTITY_SUPPLEMENTARY_ROLES[role]
        }
        if (
            item["name"] != _IDENTITY_USER_NAMES[role]
            or item["primary_group"] != role
            or type(item["uid"]) is not int
            or not 0 < item["uid"] < (1 << 31)
            or item["uid"] in user_ids
            or _safe_absolute_path(item["home"], f"identity user {role} home")
            != item["home"]
            or _safe_absolute_path(item["shell"], f"identity user {role} shell")
            != item["shell"]
            or not isinstance(supplementary, list)
            or supplementary != sorted(supplementary)
            or len(set(supplementary)) != len(supplementary)
            or any(
                not isinstance(group, str)
                or _SYSTEM_IDENTITY.fullmatch(group) is None
                for group in supplementary
            )
            or supplementary != sorted(required)
        ):
            raise ValueError(f"identity user {role} is invalid")
        user_ids.add(item["uid"])
        if role != "gateway" and (
            item["home"] != "/nonexistent"
            or item["shell"] != "/usr/sbin/nologin"
        ):
            raise ValueError(f"identity user {role} login boundary is invalid")
        if pre["state"] == "absent":
            if pre != {
                "state": "absent",
                "uid": None,
                "gid": None,
                "home": None,
                "shell": None,
                "supplementary_group_names": None,
            }:
                raise ValueError(f"identity user {role} pre-state is invalid")
            if supplementary != sorted(required):
                raise ValueError(f"identity user {role} target groups are invalid")
        elif pre["state"] == "present":
            before_groups = pre["supplementary_group_names"]
            if (
                pre["uid"] != item["uid"]
                or pre["gid"] != validated_groups[role]["gid"]
                or pre["home"] != item["home"]
                or pre["shell"] != item["shell"]
                or not isinstance(before_groups, list)
                or before_groups != sorted(before_groups)
                or len(set(before_groups)) != len(before_groups)
            ):
                raise ValueError(f"identity user {role} pre-state is invalid")
        else:
            raise ValueError(f"identity user {role} pre-state is invalid")
        validated_users[role] = item

    if (
        validated_users["gateway"]["pre"]["state"] != "present"
        or validated_groups["gateway"]["pre"]["state"] != "present"
    ):
        raise ValueError("identity foundation existing anchors are invalid")
    derived_members = {role: set() for role in _IDENTITY_GROUP_ROLES}
    for user in validated_users.values():
        for role, group in validated_groups.items():
            if group["name"] in user["supplementary_groups"]:
                derived_members[role].add(user["name"])
    for role, group in validated_groups.items():
        if group["members"] != sorted(derived_members[role]):
            raise ValueError(f"identity group {role} target members are invalid")
    return raw


def _validate_discord_key_foundation(
    value: Any,
    *,
    identity_foundation: Mapping[str, Any],
) -> dict[str, Any]:
    raw = _hashed(
        value,
        _DISCORD_KEY_FOUNDATION_FIELDS,
        "foundation_sha256",
        "Discord key foundation",
    )
    writer = _exact(
        raw["writer"],
        _DISCORD_KEY_PAIR_FIELDS,
        "writer capability key pair",
    )
    edge = _exact(
        raw["edge"],
        _DISCORD_KEY_PAIR_FIELDS,
        "Discord edge receipt key pair",
    )
    staged_root = EVIDENCE_ROOT / "staged" / "keys"
    expected = {
        "writer": {
            "staged_private_path": str(
                staged_root / "writer-capability-private.pem"
            ),
            "private_path": str(WRITER_CAPABILITY_PRIVATE_KEY_PATH),
            "private_uid": identity_foundation["users"]["writer"]["uid"],
            "private_gid": identity_foundation["groups"]["writer"]["gid"],
            "private_mode": 0o400,
            "public_path": str(WRITER_CAPABILITY_PUBLIC_KEY_PATH),
            "public_uid": 0,
            "public_gid": identity_foundation["groups"]["routeback"]["gid"],
            "public_mode": 0o440,
        },
        "edge": {
            "staged_private_path": str(
                staged_root / "discord-edge-receipt-private.pem"
            ),
            "private_path": str(EDGE_RECEIPT_PRIVATE_KEY_PATH),
            "private_uid": 0,
            "private_gid": 0,
            "private_mode": 0o400,
            "public_path": str(EDGE_RECEIPT_PUBLIC_KEY_PATH),
            "public_uid": 0,
            "public_gid": identity_foundation["groups"]["writer"]["gid"],
            "public_mode": 0o440,
        },
    }
    for role, item in (("writer", writer), ("edge", edge)):
        if (
            any(item[field] != expected[role][field] for field in expected[role])
            or _SHA256.fullmatch(str(item["public_key_id"])) is None
        ):
            raise ValueError(f"Discord {role} key foundation is invalid")
    all_paths = [
        writer["staged_private_path"], writer["private_path"],
        writer["public_path"], edge["staged_private_path"],
        edge["private_path"], edge["public_path"],
    ]
    if (
        raw["schema"] != _DISCORD_KEY_FOUNDATION_SCHEMA
        or raw["pre_state"] != "absent"
        or raw["keys_distinct"] is not True
        or raw["private_content_or_digest_recorded"] is not False
        or raw["secret_material_recorded"] is not False
        or writer["public_key_id"] == edge["public_key_id"]
        or len(set(all_paths)) != len(all_paths)
    ):
        raise ValueError("Discord key foundation policy is invalid")
    return raw


def _validate_host_transition(
    value: Any,
    *,
    gateway_pre: ServiceObservation,
    writer_pre: ServiceObservation,
    connector_pre: ServiceObservation,
    gateway_target: Mapping[str, Any],
    writer_target: Mapping[str, Any],
    connector_target: Mapping[str, Any],
    capability_topology: Mapping[str, Any],
) -> dict[str, Any]:
    raw = _hashed(
        value,
        _HOST_TRANSITION_FIELDS,
        "manifest_sha256",
        "host transition manifest",
    )
    if (
        raw["schema"] != _HOST_TRANSITION_SCHEMA
        or raw["gateway_direct_discord_enabled"] is not False
        or raw["gateway_relay_platforms"] != ["discord"]
        or raw["connector_operation_class"]
        != "ordinary_guild_acl_session_only"
        or raw["routeback_operation_class"]
        != "canonical_guild_acl_routeback_rest_only"
        or raw["discord_dm_allowed"] is not False
        or raw["secret_material_recorded"] is not False
    ):
        raise ValueError("host transition policy is invalid")
    validate_discord_policy_continuity(raw["discord_policy_continuity"])
    files = _exact(
        raw["files"],
        frozenset(_HOST_TRANSITION_FILE_NAMES),
        "host transition files",
    )
    identity_foundation = _validate_identity_foundation(
        raw["identity_foundation"]
    )
    discord_key_foundation = _validate_discord_key_foundation(
        raw["discord_key_foundation"],
        identity_foundation=identity_foundation,
    )
    try:
        operational_key_foundation = (
            validate_operational_edge_key_foundation(
                raw["operational_edge_key_foundation"],
                expected_writer_public_key_id=(
                    discord_key_foundation["writer"]["public_key_id"]
                ),
                key_root=OPERATIONAL_EDGE_KEY_STAGING_ROOT,
                trust_root=OPERATIONAL_EDGE_KEY_STAGING_ROOT,
                expected_uid=0,
                expected_gid=0,
            )
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(
            "operational edge key foundation is invalid"
        ) from exc
    operational_key_ids = raw["operational_edge_receipt_public_key_ids"]
    observed_operational_key_ids = {
        row["domain"]: row["public_key_id"]
        for row in operational_key_foundation["keys"]
    }
    if (
        _SHA256.fullmatch(
            str(raw["operational_edge_key_foundation_sha256"])
        )
        is None
        or raw["operational_edge_key_foundation_sha256"]
        != operational_key_foundation["receipt_sha256"]
        or operational_key_ids != observed_operational_key_ids
        or not isinstance(operational_key_ids, Mapping)
        or set(operational_key_ids) != set(CREDENTIALS_BY_DOMAIN)
        or any(
            _SHA256.fullmatch(str(key_id)) is None
            for key_id in operational_key_ids.values()
        )
        or len(set(operational_key_ids.values())) != len(operational_key_ids)
        or discord_key_foundation["writer"]["public_key_id"]
        in set(operational_key_ids.values())
        or raw["release_owner_uid"]
        != identity_foundation["users"]["gateway"]["uid"]
        or raw["release_owner_gid"]
        != identity_foundation["groups"]["gateway"]["gid"]
    ):
        raise ValueError("operational edge key foundation is invalid")
    expected_paths = {
        "gateway_unit": GATEWAY_FRAGMENT,
        "writer_unit": WRITER_FRAGMENT,
        "connector_unit": CONNECTOR_FRAGMENT,
        "phase_b_unit": PHASE_B_FRAGMENT,
        "routeback_unit": ROUTEBACK_EDGE_FRAGMENT,
        "mac_ops_unit": MAC_OPS_FRAGMENT,
        "browser_unit": BROWSER_FRAGMENT,
        "browser_config": BROWSER_CONFIG,
        "isolated_worker_socket_unit": ISOLATED_WORKER_SOCKET_FRAGMENT,
        "isolated_worker_service_unit": ISOLATED_WORKER_SERVICE_FRAGMENT,
        "isolated_worker_config": ISOLATED_WORKER_CONFIG_PATH,
        "gateway_connector_drop_in": GATEWAY_CONNECTOR_DROP_IN,
        "gateway_config": GATEWAY_CONFIG_PATH,
        "writer_config": WRITER_CONFIG_PATH,
        "connector_config": CONNECTOR_CONFIG_PATH,
        "routeback_config": str(ROUTEBACK_EDGE_CONFIG_PATH),
        "mac_ops_config": str(MAC_OPS_CONFIG_PATH),
        "api_bearer_verifier": str(API_SERVER_CREDENTIAL_PATH),
        "api_approval_verifier": str(API_APPROVAL_CREDENTIAL_PATH),
        "operational_edge_client_config": str(
            OPERATIONAL_EDGE_CLIENT_CONFIG_PATH
        ),
        **{
            f"operational_edge_unit_{domain}": (
                f"/etc/systemd/system/{operational_edge_service_unit(domain)}"
            )
            for domain in sorted(CREDENTIALS_BY_DOMAIN)
        },
        **{
            f"operational_edge_config_{domain}": str(
                operational_edge_config_path(domain)
            )
            for domain in sorted(CREDENTIALS_BY_DOMAIN)
        },
    }
    capability_topology = validate_production_capability_topology(
        capability_topology
    )
    service_digests = {
        "gateway_unit": gateway_target["fragment_sha256"],
        "writer_unit": writer_target["fragment_sha256"],
        "connector_unit": connector_target["fragment_sha256"],
        "phase_b_unit": capability_topology["phase_b"]["fragment_sha256"],
        "routeback_unit": capability_topology["routeback_edge"][
            "fragment_sha256"
        ],
        "mac_ops_unit": capability_topology["mac_ops"]["fragment_sha256"],
        "browser_unit": capability_topology["browser"]["fragment_sha256"],
        "browser_config": capability_topology["browser"]["config_sha256"],
        "isolated_worker_socket_unit": capability_topology[
            "isolated_worker"
        ]["socket_fragment_sha256"],
        "isolated_worker_service_unit": capability_topology[
            "isolated_worker"
        ]["service_fragment_sha256"],
        "isolated_worker_config": capability_topology[
            "isolated_worker"
        ]["config_sha256"],
        "gateway_connector_drop_in": gateway_target["drop_in_sha256"][
            GATEWAY_CONNECTOR_DROP_IN
        ],
        "routeback_config": capability_topology["routeback_edge"][
            "config_sha256"
        ],
        "mac_ops_config": capability_topology["mac_ops"]["config_sha256"],
    }
    pre_services = {
        "gateway_unit": gateway_pre,
        "writer_unit": writer_pre,
        "connector_unit": connector_pre,
    }
    staged_root = EVIDENCE_ROOT / "staged" / "host"
    for name in _HOST_TRANSITION_FILE_NAMES:
        item = _exact(files[name], _HOST_FILE_FIELDS, f"host file {name}")
        expected_staged = str(staged_root / Path(expected_paths[name]).name)
        if (
            item["staged_path"] != expected_staged
            or item["target_path"] != expected_paths[name]
            or _SHA256.fullmatch(str(item["sha256"])) is None
            or type(item["uid"]) is not int
            or type(item["gid"]) is not int
            or item["uid"] < 0
            or item["gid"] < 0
            or type(item["mode"]) is not int
        ):
            raise ValueError(f"host file {name} target is invalid")
        if name in service_digests and item["sha256"] != service_digests[name]:
            raise ValueError(f"host file {name} service digest is invalid")
        if name in {
            "gateway_unit", "writer_unit", "connector_unit", "phase_b_unit",
            "routeback_unit", "mac_ops_unit", "browser_unit",
            "isolated_worker_socket_unit", "isolated_worker_service_unit",
            "gateway_connector_drop_in",
        } | {
            f"operational_edge_unit_{domain}"
            for domain in CREDENTIALS_BY_DOMAIN
        } and (item["uid"], item["gid"], item["mode"]) != (0, 0, 0o644):
            raise ValueError(f"host file {name} mode is invalid")
        if name == "gateway_config" and (
            item["uid"] <= 0
            or item["gid"] <= 0
            or item["mode"] not in {0o600, 0o640}
        ):
            raise ValueError("gateway config target identity is invalid")
        if name == "writer_config" and (
            item["uid"] != 0
            or item["gid"] != identity_foundation["groups"]["writer"]["gid"]
            or item["mode"] != 0o440
        ):
            raise ValueError("host file writer_config identity is invalid")
        if name in {"connector_config", "routeback_config", "mac_ops_config"} and (
            item["uid"] != 0 or item["gid"] <= 0 or item["mode"] != 0o440
        ):
            raise ValueError(f"host file {name} config identity is invalid")
        if name == "browser_config" and (
            item["uid"] != 0
            or item["gid"] != identity_foundation["groups"]["browser"]["gid"]
            or item["mode"] != 0o440
        ):
            raise ValueError("host file browser_config identity is invalid")
        if name == "isolated_worker_config" and (
            item["uid"] != 0
            or item["gid"] != identity_foundation["groups"]["worker"]["gid"]
            or item["mode"] != 0o440
        ):
            raise ValueError(
                "host file isolated_worker_config identity is invalid"
            )
        if name in {
            f"operational_edge_config_{domain}"
            for domain in CREDENTIALS_BY_DOMAIN
        } and (item["uid"], item["gid"], item["mode"]) != (0, 0, 0o400):
            raise ValueError(
                f"host file {name} operational config identity is invalid"
            )
        if name == "operational_edge_client_config" and (
            item["uid"], item["gid"], item["mode"]
        ) != (0, 0, 0o444):
            raise ValueError(
                "host file operational_edge_client_config identity is invalid"
            )
        if name in {"api_bearer_verifier", "api_approval_verifier"} and (
            item["uid"] != 0 or item["gid"] != 0 or item["mode"] != 0o400
        ):
            raise ValueError(f"host file {name} verifier identity is invalid")
        pre = _exact(item["pre"], _HOST_FILE_PRE_FIELDS, f"host file {name} pre")
        if pre["state"] == "absent":
            if any(pre[field] is not None for field in ("sha256", "uid", "gid", "mode")):
                raise ValueError(f"host file {name} absent pre-state is invalid")
        elif pre["state"] == "present":
            if (
                _SHA256.fullmatch(str(pre["sha256"])) is None
                or type(pre["uid"]) is not int
                or type(pre["gid"]) is not int
                or type(pre["mode"]) is not int
                or pre["uid"] < 0
                or pre["gid"] < 0
                or not 0 <= pre["mode"] <= 0o777
            ):
                raise ValueError(f"host file {name} present pre-state is invalid")
        else:
            raise ValueError(f"host file {name} pre-state is invalid")
        if name in pre_services:
            service = pre_services[name]
            expected_state = (
                "present" if service.value["load_state"] == "loaded" else "absent"
            )
            if pre["state"] != expected_state:
                raise ValueError(f"host file {name} pre-service state is invalid")
            if expected_state == "present" and pre["sha256"] != service.value[
                "fragment_sha256"
            ]:
                raise ValueError(f"host file {name} pre-service digest is invalid")
        if name == "gateway_connector_drop_in" and pre["state"] != "absent":
            raise ValueError("gateway connector drop-in pre-state is invalid")
        if name in {
            "writer_config", "connector_config", "phase_b_unit", "routeback_unit",
            "mac_ops_unit", "browser_unit", "routeback_config",
            "browser_config", "isolated_worker_socket_unit",
            "isolated_worker_service_unit", "isolated_worker_config",
            "mac_ops_config", "api_bearer_verifier", "api_approval_verifier",
            "operational_edge_client_config",
        } | {
            f"operational_edge_unit_{domain}"
            for domain in CREDENTIALS_BY_DOMAIN
        } | {
            f"operational_edge_config_{domain}"
            for domain in CREDENTIALS_BY_DOMAIN
        } and pre["state"] != "absent":
            raise ValueError(f"host file {name} pre-state is invalid")

    worker = capability_topology["isolated_worker"]
    browser = capability_topology["browser"]
    if (
        identity_foundation["users"]["gateway"]["uid"]
        != capability_topology["gateway_identity"]["uid"]
        or identity_foundation["groups"]["gateway"]["gid"]
        != capability_topology["gateway_identity"]["gid"]
        or identity_foundation["users"]["worker"]["uid"]
        != worker["server_uid"]
        or identity_foundation["groups"]["worker"]["gid"]
        != worker["server_gid"]
        or identity_foundation["groups"]["worker_client"]["gid"]
        != worker["socket_gid"]
        or identity_foundation["users"]["browser"]["uid"]
        != browser["service_uid"]
        or identity_foundation["groups"]["browser"]["gid"]
        != browser["service_gid"]
        or "docker"
        in identity_foundation["users"]["gateway"]["supplementary_groups"]
    ):
        raise ValueError("host capability identities are invalid")
    directory = _exact(
        raw["isolated_worker_lease_mountpoint"],
        _HOST_DIRECTORY_FIELDS,
        "isolated worker lease mountpoint",
    )
    directory_pre = _exact(
        directory["pre"],
        _HOST_DIRECTORY_PRE_FIELDS,
        "isolated worker lease mountpoint pre",
    )
    if (
        directory["target_path"] != str(ISOLATED_WORKER_LEASE_BASE)
        or (directory["uid"], directory["gid"], directory["mode"])
        != (0, 0, 0o700)
    ):
        raise ValueError("isolated worker lease mountpoint is invalid")
    if directory_pre["state"] == "absent":
        if any(
            directory_pre[field] is not None for field in ("uid", "gid", "mode")
        ):
            raise ValueError("isolated worker lease mountpoint pre is invalid")
    elif directory_pre["state"] == "present":
        if any(
            type(directory_pre[field]) is not int
            for field in ("uid", "gid", "mode")
        ):
            raise ValueError("isolated worker lease mountpoint pre is invalid")
    else:
        raise ValueError("isolated worker lease mountpoint pre is invalid")

    token = _exact(
        raw["connector_token"],
        _CONNECTOR_TOKEN_FIELDS,
        "connector token identity",
    )
    if (
        token["path"] != CONNECTOR_TOKEN_PATH
        or type(token["uid"]) is not int
        or type(token["gid"]) is not int
        or token["uid"] <= 0
        or token["gid"] <= 0
        or token["mode"] != 0o400
        or token["regular_one_link"] is not True
        or token["content_or_digest_recorded"] is not False
        or token["gateway_readable"] is not False
        or token["gid"] != files["connector_config"]["gid"]
        or _safe_absolute_path(
            token["source_path"], "connector token source path"
        )
        != token["source_path"]
        or type(token["source_uid"]) is not int
        or type(token["source_gid"]) is not int
        or token["source_uid"] < 0
        or token["source_gid"] < 0
        or token["source_mode"] != 0o400
        or token["source_uid"] != files["gateway_config"]["uid"]
        or token["uid"] == files["gateway_config"]["uid"]
        or token["uid"]
        != identity_foundation["users"]["connector"]["uid"]
        or token["gid"]
        != identity_foundation["groups"]["connector"]["gid"]
        or files["gateway_config"]["uid"]
        != identity_foundation["users"]["gateway"]["uid"]
        or files["gateway_config"]["gid"]
        != identity_foundation["groups"]["gateway"]["gid"]
        or files["connector_config"]["gid"]
        != identity_foundation["groups"]["connector"]["gid"]
        or files["writer_config"]["gid"]
        != identity_foundation["groups"]["writer"]["gid"]
        or files["routeback_config"]["gid"]
        != identity_foundation["groups"]["routeback"]["gid"]
        or files["mac_ops_config"]["gid"]
        != identity_foundation["groups"]["mac_ops"]["gid"]
    ):
        raise ValueError("connector token identity is invalid")
    retired = raw["gateway_retired_token_paths"]
    preserved = raw["routeback_token_paths"]
    if (
        not isinstance(retired, list)
        or not 1 <= len(retired) <= 8
        or len(set(retired)) != len(retired)
        or not isinstance(preserved, list)
        or not 1 <= len(preserved) <= 8
        or len(set(preserved)) != len(preserved)
    ):
        raise ValueError("Discord credential path contract is invalid")
    for label, values in (
        ("retired gateway token path", retired),
        ("preserved routeback token path", preserved),
    ):
        for path in values:
            _safe_absolute_path(path, label)
    if (
        CONNECTOR_TOKEN_PATH in retired
        or CONNECTOR_TOKEN_PATH in preserved
        or set(retired) & set(preserved)
        or token["source_path"] not in retired
    ):
        raise ValueError("Discord credential leases are not disjoint")

    approval = _exact(
        raw["approval_passkey"],
        _APPROVAL_PASSKEY_FIELDS,
        "approval passkey identity",
    )
    approval_source = _safe_absolute_path(
        approval["source_path"], "approval passkey source path"
    )
    approval_retired = raw["retired_approval_passkey_paths"]
    if (
        not isinstance(approval_retired, list)
        or not 1 <= len(approval_retired) <= 8
        or any(not isinstance(path, str) for path in approval_retired)
        or approval_retired != sorted(approval_retired)
        or len(set(approval_retired)) != len(approval_retired)
    ):
        raise ValueError("approval passkey lease is invalid")
    for path in approval_retired:
        _safe_absolute_path(path, "retired approval passkey path")
    if (
        approval["path"] != str(API_APPROVAL_CREDENTIAL_PATH)
        or any(
            type(approval[field]) is not int
            for field in (
                "uid", "gid", "mode", "source_uid", "source_gid", "source_mode"
            )
        )
        or (approval["uid"], approval["gid"], approval["mode"])
        != (0, 0, 0o400)
        or approval["regular_one_link"] is not True
        or approval["content_or_digest_recorded"] is not False
        or approval["gateway_readable"] is not False
        or approval_source != str(STAGED_APPROVAL_PASSKEY_PATH)
        or (approval["source_uid"], approval["source_gid"], approval["source_mode"])
        != (0, 0, 0o400)
        or approval_source not in approval_retired
        or str(API_APPROVAL_CREDENTIAL_PATH) in approval_retired
        or approval_source in retired
        or approval_source in preserved
        or str(API_APPROVAL_CREDENTIAL_PATH) in retired
        or str(API_APPROVAL_CREDENTIAL_PATH) in preserved
    ):
        raise ValueError("approval passkey lease is invalid")
    if discord_key_foundation["writer"]["public_gid"] != files[
        "routeback_config"
    ]["gid"]:
        raise ValueError("Discord writer public key lease is invalid")
    return raw


_CUTOVER_FIELDS = frozenset({
    "schema", "release_revision", "target", "owner_subject_sha256",
    "owner_public_key_ed25519_hex", "owner_key_id", "freeze_plan",
    "freeze_plan_sha256", "freeze_approval_sha256", "final_tail_receipt",
    "final_tail_receipt_sha256", "artifacts", "gateway_legacy_identity",
    "writer_pre_identity", "connector_pre_identity",
    "gateway_target_identity", "writer_target_identity",
    "connector_target_identity", "host_transition", "rollback_contract", "states",
    "capability_topology",
    "cron_inventory",
    "cron_continuity_plan",
    "mechanical_job_host_facts",
    "mechanical_job_package",
    "legacy_truth_decision",
    "owner_runtime_attestation",
    "secret_material_recorded", "plan_sha256",
})


def _require_final_tail_within_authority(
    freeze: FreezePlan,
    tail: FinalTailReceipt,
) -> None:
    initial = LegacySnapshot.from_mapping(freeze.value["initial_snapshot"])
    bounds = freeze.value["cutover_authority"]["final_tail_bounds"]
    if (
        tail.value["initial_row_count"] != initial.value["source_row_count"]
        or tail.snapshot.storage_identity() != initial.storage_identity()
        or tail.value["final_row_count"] - tail.value["initial_row_count"]
        > bounds["max_appended_rows"]
        or tail.value["captured_at_unix"] < initial.value["observed_at_unix"]
        or tail.value["captured_at_unix"] - initial.value["observed_at_unix"]
        > bounds["max_capture_delay_seconds"]
    ):
        raise ValueError("cutover final-tail authority bounds are invalid")


@dataclass(frozen=True)
class CutoverPlan:
    value: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, value: Any) -> "CutoverPlan":
        raw = _hashed(value, _CUTOVER_FIELDS, "plan_sha256", "cutover plan")
        revision = raw["release_revision"]
        if raw["schema"] != CUTOVER_PLAN_SCHEMA or not isinstance(revision, str) or _REVISION.fullmatch(revision) is None:
            raise ValueError("cutover plan identity is invalid")
        _validate_target(raw["target"])
        _digest(raw["owner_subject_sha256"], "cutover owner subject")
        public = raw["owner_public_key_ed25519_hex"]
        if not isinstance(public, str) or re.fullmatch(r"[0-9a-f]{64}", public) is None or raw["owner_key_id"] != hashlib.sha256(bytes.fromhex(public)).hexdigest():
            raise ValueError("cutover owner key is invalid")
        freeze = FreezePlan.from_mapping(raw["freeze_plan"])
        if (
            raw["freeze_plan_sha256"] != freeze.sha256
            or revision != freeze.value["release_revision"]
            or raw["target"] != freeze.value["target"]
            or raw["owner_subject_sha256"] != freeze.value["owner_subject_sha256"]
            or raw["owner_public_key_ed25519_hex"]
            != freeze.value["owner_public_key_ed25519_hex"]
            or raw["owner_key_id"] != freeze.value["owner_key_id"]
            or raw["owner_runtime_attestation"]
            != freeze.value["owner_runtime_attestation"]
        ):
            raise ValueError("cutover freeze authority binding is invalid")
        tail = FinalTailReceipt.from_mapping(raw["final_tail_receipt"], plan=freeze)
        if (
            raw["final_tail_receipt_sha256"] != tail.sha256
            or raw["freeze_approval_sha256"] != tail.value["approval_sha256"]
        ):
            raise ValueError("cutover final-tail binding is invalid")
        _require_final_tail_within_authority(freeze, tail)
        authority = freeze.value["cutover_authority"]
        decision = _legacy_truth_decision(
            raw["legacy_truth_decision"],
            snapshot=LegacySnapshot.from_mapping(
                freeze.value["initial_snapshot"]
            ),
            owner_subject_sha256=raw["owner_subject_sha256"],
        )
        if decision != authority["legacy_truth_decision"]:
            raise ValueError("cutover legacy truth authority drifted")
        topology = validate_production_capability_topology(raw["capability_topology"])
        if topology != authority["capability_topology"]:
            raise ValueError("cutover capability topology authority drifted")
        if any(
            raw[name] != authority[name]
            for name in (
                "cron_inventory",
                "cron_continuity_plan",
                "mechanical_job_host_facts",
                "mechanical_job_package",
            )
        ):
            raise ValueError("cutover cron continuity authority drifted")
        artifacts = _exact(raw["artifacts"], frozenset(_CUTOVER_ARTIFACT_NAMES), "cutover artifacts")
        for name in _CUTOVER_ARTIFACT_NAMES:
            _artifact(artifacts[name], name, revision)
        gateway = ServiceObservation.from_mapping(raw["gateway_legacy_identity"])
        writer = ServiceObservation.from_mapping(raw["writer_pre_identity"])
        connector = ServiceObservation.from_mapping(raw["connector_pre_identity"])
        if (
            not gateway.stopped
            or not writer.stopped
            or not connector.stopped
            or gateway.stable_identity()
            != ServiceObservation.from_mapping(
                freeze.value["gateway_before"]
            ).stable_identity()
            or writer.stable_identity()
            != ServiceObservation.from_mapping(
                freeze.value["writer_before"]
            ).stable_identity()
            or connector.stable_identity()
            != ServiceObservation.from_mapping(
                freeze.value["connector_before"]
            ).stable_identity()
        ):
            raise ValueError("cutover service precondition is invalid")
        gateway_target = _validate_target_service_identity(
            raw["gateway_target_identity"], unit=GATEWAY_UNIT
        )
        writer_target = _validate_target_service_identity(
            raw["writer_target_identity"], unit=WRITER_UNIT
        )
        connector_target = _validate_target_service_identity(
            raw["connector_target_identity"], unit=CONNECTOR_UNIT
        )
        _validate_host_transition(
            raw["host_transition"],
            gateway_pre=gateway,
            writer_pre=writer,
            connector_pre=connector,
            gateway_target=gateway_target,
            writer_target=writer_target,
            connector_target=connector_target,
            capability_topology=topology,
        )
        if (
            artifacts != authority["artifacts"]
            or raw["gateway_target_identity"]
            != authority["gateway_target_identity"]
            or raw["writer_target_identity"]
            != authority["writer_target_identity"]
            or raw["connector_target_identity"]
            != authority["connector_target_identity"]
            or raw["host_transition"] != authority["host_transition"]
        ):
            raise ValueError("cutover plan differs from approved authority")
        rollback = _exact(raw["rollback_contract"], frozenset({
            "database_rollback_sha256", "host_rollback_sha256",
            "requires_gateway_stopped", "requires_writer_stopped",
            "requires_connector_stopped",
            "requires_zero_canonical_writer_writes", "restart_legacy_gateway",
        }), "rollback contract")
        if rollback != {
            "database_rollback_sha256": artifacts["database_rollback"]["sha256"],
            "host_rollback_sha256": artifacts["host_rollback"]["sha256"],
            "requires_gateway_stopped": True,
            "requires_writer_stopped": True,
            "requires_connector_stopped": True,
            "requires_zero_canonical_writer_writes": True,
            "restart_legacy_gateway": True,
        }:
            raise ValueError("cutover rollback contract is invalid")
        if rollback != authority["rollback_contract"]:
            raise ValueError("cutover rollback authority drifted")
        if raw["states"] != [
            "authority", "host_applied", "prerequisites_started",
            "capability_prerequisites_validated", "final_tail_reobserved",
            "preflight", "database_applied", "writer_started",
            "database_terminal_validated", "activation_commit_intent",
            "boot_committed", "gateway_started", "terminal",
        ] or raw["secret_material_recorded"] is not False:
            raise ValueError("cutover state contract is invalid")
        return cls(raw)

    @property
    def sha256(self) -> str:
        return str(self.value["plan_sha256"])

    @property
    def final_snapshot(self) -> LegacySnapshot:
        return FinalTailReceipt.from_mapping(self.value["final_tail_receipt"]).snapshot

    def to_mapping(self) -> dict[str, Any]:
        return copy.deepcopy(dict(self.value))


def build_cutover_plan(
    *,
    freeze_plan: FreezePlan,
    final_tail_receipt: FinalTailReceipt,
    gateway_stopped: ServiceObservation,
    writer_stopped: ServiceObservation,
    connector_stopped: ServiceObservation,
) -> CutoverPlan:
    FinalTailReceipt.from_mapping(final_tail_receipt.to_mapping(), plan=freeze_plan)
    _require_final_tail_within_authority(freeze_plan, final_tail_receipt)
    authority = freeze_plan.value["cutover_authority"]
    artifact_set = copy.deepcopy(dict(authority["artifacts"]))
    unsigned = {
        "schema": CUTOVER_PLAN_SCHEMA,
        "release_revision": freeze_plan.value["release_revision"],
        "target": copy.deepcopy(freeze_plan.value["target"]),
        "owner_subject_sha256": freeze_plan.value["owner_subject_sha256"],
        "owner_public_key_ed25519_hex": freeze_plan.value["owner_public_key_ed25519_hex"],
        "owner_key_id": freeze_plan.value["owner_key_id"],
        "freeze_plan": freeze_plan.to_mapping(),
        "freeze_plan_sha256": freeze_plan.sha256,
        "freeze_approval_sha256": final_tail_receipt.value["approval_sha256"],
        "final_tail_receipt": final_tail_receipt.to_mapping(),
        "final_tail_receipt_sha256": final_tail_receipt.sha256,
        "artifacts": artifact_set,
        "gateway_legacy_identity": gateway_stopped.to_mapping(),
        "writer_pre_identity": writer_stopped.to_mapping(),
        "connector_pre_identity": connector_stopped.to_mapping(),
        "gateway_target_identity": copy.deepcopy(
            dict(authority["gateway_target_identity"])
        ),
        "writer_target_identity": copy.deepcopy(
            dict(authority["writer_target_identity"])
        ),
        "connector_target_identity": copy.deepcopy(
            dict(authority["connector_target_identity"])
        ),
        "host_transition": copy.deepcopy(dict(authority["host_transition"])),
        "capability_topology": copy.deepcopy(
            dict(authority["capability_topology"])
        ),
        "cron_inventory": copy.deepcopy(dict(authority["cron_inventory"])),
        "cron_continuity_plan": copy.deepcopy(
            dict(authority["cron_continuity_plan"])
        ),
        "mechanical_job_host_facts": copy.deepcopy(
            dict(authority["mechanical_job_host_facts"])
        ),
        "mechanical_job_package": copy.deepcopy(
            dict(authority["mechanical_job_package"])
        ),
        "legacy_truth_decision": copy.deepcopy(
            dict(authority["legacy_truth_decision"])
        ),
        "owner_runtime_attestation": copy.deepcopy(
            dict(freeze_plan.value["owner_runtime_attestation"])
        ),
        "rollback_contract": copy.deepcopy(dict(authority["rollback_contract"])),
        "states": [
            "authority", "host_applied", "prerequisites_started",
            "capability_prerequisites_validated", "final_tail_reobserved",
            "preflight", "database_applied", "writer_started",
            "database_terminal_validated", "activation_commit_intent",
            "boot_committed", "gateway_started", "terminal",
        ],
        "secret_material_recorded": False,
    }
    return CutoverPlan.from_mapping({**unsigned, "plan_sha256": _sha256_json(unsigned)})


_APPROVAL_FIELDS = frozenset({
    "schema", "plan_kind", "purpose", "sequence", "previous_approval_sha256",
    "plan_sha256", "owner_subject_sha256", "owner_public_key_ed25519_hex",
    "owner_key_id", "nonce_sha256", "issued_at_unix", "expires_at_unix",
    "approved", "signature_ed25519_hex", "approval_sha256",
})
_APPROVAL_SIGNED_FIELDS = _APPROVAL_FIELDS - {"signature_ed25519_hex", "approval_sha256"}


def approval_signature_payload(value: Mapping[str, Any]) -> bytes:
    if set(value) != _APPROVAL_FIELDS:
        raise ValueError("approval fields are not exact")
    return _canonical_bytes({key: value[key] for key in _APPROVAL_SIGNED_FIELDS})


@dataclass(frozen=True)
class CutoverApproval:
    value: Mapping[str, Any]

    @classmethod
    def from_mapping(
        cls,
        value: Any,
        *,
        plan: FreezePlan | CutoverPlan,
        now_unix: int | None = None,
    ) -> "CutoverApproval":
        raw = _hashed(value, _APPROVAL_FIELDS, "approval_sha256", "cutover approval")
        current = int(time.time()) if now_unix is None else now_unix
        authority_plan = (
            FreezePlan.from_mapping(plan.value["freeze_plan"])
            if isinstance(plan, CutoverPlan)
            else plan
        )
        kind = "freeze"
        expected_purpose = f"{kind}_{'apply' if raw['sequence'] == 0 else 'resume'}"
        public = authority_plan.value["owner_public_key_ed25519_hex"]
        if (
            raw["schema"] != APPROVAL_SCHEMA
            or raw["plan_kind"] != kind
            or raw["purpose"] != expected_purpose
            or type(raw["sequence"]) is not int
            or raw["sequence"] < 0
            or (raw["sequence"] == 0 and raw["previous_approval_sha256"] is not None)
            or (raw["sequence"] > 0 and _SHA256.fullmatch(str(raw["previous_approval_sha256"])) is None)
            or raw["plan_sha256"] != authority_plan.sha256
            or raw["owner_subject_sha256"]
            != authority_plan.value["owner_subject_sha256"]
            or raw["owner_public_key_ed25519_hex"] != public
            or raw["owner_key_id"] != authority_plan.value["owner_key_id"]
            or _SHA256.fullmatch(str(raw["nonce_sha256"])) is None
            or type(raw["issued_at_unix"]) is not int
            or type(raw["expires_at_unix"]) is not int
            or not raw["issued_at_unix"] <= current < raw["expires_at_unix"]
            or not 1 <= raw["expires_at_unix"] - raw["issued_at_unix"] <= 3600
            or raw["approved"] is not True
            or not isinstance(raw["signature_ed25519_hex"], str)
            or re.fullmatch(r"[0-9a-f]{128}", raw["signature_ed25519_hex"]) is None
        ):
            raise PermissionError("cutover approval is invalid")
        try:
            Ed25519PublicKey.from_public_bytes(bytes.fromhex(public)).verify(
                bytes.fromhex(raw["signature_ed25519_hex"]),
                approval_signature_payload(raw),
            )
        except (InvalidSignature, ValueError) as exc:
            raise PermissionError("cutover owner signature is invalid") from exc
        return cls(raw)

    @property
    def sha256(self) -> str:
        return str(self.value["approval_sha256"])


_JOURNAL_FIELDS = frozenset({
    "schema", "plan_sha256", "sequence", "event", "previous_entry_sha256",
    "evidence", "recorded_at_unix", "entry_sha256",
})


@dataclass(frozen=True)
class JournalEntry:
    value: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, value: Any, *, plan_sha256: str) -> "JournalEntry":
        raw = _hashed(value, _JOURNAL_FIELDS, "entry_sha256", "cutover journal entry")
        if (
            raw["schema"] != JOURNAL_SCHEMA
            or raw["plan_sha256"] != plan_sha256
            or type(raw["sequence"]) is not int
            or raw["sequence"] < 0
            or not isinstance(raw["event"], str)
            or not isinstance(raw["evidence"], Mapping)
            or type(raw["recorded_at_unix"]) is not int
        ):
            raise ValueError("cutover journal entry is invalid")
        return cls(raw)

    @property
    def sha256(self) -> str:
        return str(self.value["entry_sha256"])


class CutoverJournal(Protocol):
    def load(self, plan_sha256: str) -> list[JournalEntry]: ...
    def append(self, plan_sha256: str, event: str, evidence: Mapping[str, Any], now_unix: int) -> JournalEntry: ...


class RootCutoverJournal:
    """Root-owned append-only, hash-chained journal composed from activation I/O."""

    def __init__(self, root: Path = EVIDENCE_ROOT) -> None:
        self.root = root

    def _entries(self, plan_sha256: str) -> Path:
        _digest(plan_sha256, "journal plan")
        return self.root / "plans" / plan_sha256 / "entries"

    def load(self, plan_sha256: str) -> list[JournalEntry]:
        root = self._entries(plan_sha256)
        if not os.path.lexists(root):
            return []
        result: list[JournalEntry] = []
        for expected, path in enumerate(sorted(root.iterdir())):
            if path.name != f"{expected:06d}.json":
                raise ProductionCutoverError("cutover_journal_sequence_invalid")
            raw = activation._read_trusted_file(
                path, expected_uid=0, expected_gid=0,
                allowed_modes=frozenset({0o400}), maximum=_MAX_JSON,
            )
            value = json.loads(raw.decode("utf-8", errors="strict"))
            if raw != _canonical_bytes(value):
                raise ProductionCutoverError("cutover_journal_not_canonical")
            entry = JournalEntry.from_mapping(value, plan_sha256=plan_sha256)
            previous = None if not result else result[-1].sha256
            if entry.value["sequence"] != expected or entry.value["previous_entry_sha256"] != previous:
                raise ProductionCutoverError("cutover_journal_chain_invalid")
            result.append(entry)
        return result

    def append(self, plan_sha256: str, event: str, evidence: Mapping[str, Any], now_unix: int) -> JournalEntry:
        entries = self.load(plan_sha256)
        root = self._entries(plan_sha256)
        activation._ensure_root_directory(root)
        unsigned = {
            "schema": JOURNAL_SCHEMA,
            "plan_sha256": plan_sha256,
            "sequence": len(entries),
            "event": event,
            "previous_entry_sha256": None if not entries else entries[-1].sha256,
            "evidence": copy.deepcopy(dict(evidence)),
            "recorded_at_unix": now_unix,
        }
        entry = JournalEntry.from_mapping(
            {**unsigned, "entry_sha256": _sha256_json(unsigned)},
            plan_sha256=plan_sha256,
        )
        activation._write_root_receipt(root / f"{len(entries):06d}.json", entry.value)
        return entry


class ServiceBoundary(Protocol):
    def observe_gateway(self) -> ServiceObservation: ...
    def observe_writer(self) -> ServiceObservation: ...
    def observe_connector(self) -> ServiceObservation: ...
    def stop_gateway(self) -> None: ...
    def stop_writer(self) -> None: ...
    def stop_connector(self) -> None: ...
    def start_gateway(self) -> None: ...


class ProductionSystemdServiceBoundary:
    """Concrete fixed-unit adapter over the existing activation primitives."""

    def __init__(
        self,
        *,
        runner: Callable[[tuple[str, ...]], Any] = activation._runner,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._runner = runner
        self._clock = clock

    @staticmethod
    def _assert_gateway_credential_free(main_pid: int) -> None:
        """Fail closed without serializing any gateway credential value."""

        if main_pid <= 0:
            raise ProductionCutoverError(
                "production_gateway_process_identity_invalid"
            )
        proc = Path("/proc") / str(main_pid)
        try:
            proc_identity = os.stat(proc, follow_symlinks=False)
            with (proc / "environ").open("rb", buffering=0) as stream:
                environment = stream.read(1024 * 1024 + 1)
            descriptors = list((proc / "fd").iterdir())
        except OSError as exc:
            raise ProductionCutoverError(
                "production_gateway_process_unavailable"
            ) from exc
        if (
            proc_identity.st_uid == 0
            or len(environment) > 1024 * 1024
            or any(
                item.partition(b"=")[0]
                in {b"DISCORD_BOT_TOKEN", b"DISCORD_TOKEN"}
                for item in environment.split(b"\0")
                if item
            )
        ):
            raise ProductionCutoverError(
                "production_gateway_discord_credential_present"
            )
        protected = {
            CONNECTOR_TOKEN_PATH,
            "/etc/muncho/discord-edge-credentials/bot-token",
        }
        try:
            for descriptor in descriptors:
                if os.readlink(descriptor) in protected:
                    raise ProductionCutoverError(
                        "production_gateway_discord_credential_open"
                    )
            for value in protected:
                item = os.lstat(value)
                if (
                    not stat.S_ISREG(item.st_mode)
                    or stat.S_ISLNK(item.st_mode)
                    or item.st_nlink != 1
                    or stat.S_IMODE(item.st_mode) != 0o400
                    or item.st_uid == proc_identity.st_uid
                ):
                    raise ProductionCutoverError(
                        "production_gateway_discord_lease_invalid"
                    )
        except ProductionCutoverError:
            raise
        except OSError as exc:
            raise ProductionCutoverError(
                "production_gateway_discord_lease_unavailable"
            ) from exc

    def _observe(self, unit: str) -> ServiceObservation:
        if unit not in {GATEWAY_UNIT, WRITER_UNIT, CONNECTOR_UNIT}:
            raise ValueError("production service unit is not fixed")
        state = activation._systemd_show(unit, runner=self._runner)
        path = {
            GATEWAY_UNIT: GATEWAY_FRAGMENT,
            WRITER_UNIT: WRITER_FRAGMENT,
            CONNECTOR_UNIT: CONNECTOR_FRAGMENT,
        }[unit]
        fragment_sha256: str | None = None
        if state["LoadState"] == "loaded":
            if state["FragmentPath"] != path:
                raise ProductionCutoverError("production_service_fragment_drifted")
            payload = activation._read_trusted_file(
                Path(path),
                expected_uid=0,
                expected_gid=0,
                allowed_modes=frozenset({0o644}),
                maximum=1024 * 1024,
            )
            fragment_sha256 = hashlib.sha256(payload).hexdigest()
        elif state["LoadState"] != "not-found":
            raise ProductionCutoverError("production_service_load_state_invalid")
        drop_in_paths = (
            []
            if state["DropInPaths"] == ""
            else sorted(state["DropInPaths"].split())
        )
        drop_in_sha256: dict[str, str] = {}
        for drop_in in drop_in_paths:
            if drop_in != GATEWAY_CONNECTOR_DROP_IN or unit != GATEWAY_UNIT:
                raise ProductionCutoverError(
                    "production_service_drop_in_drifted"
                )
            payload = activation._read_trusted_file(
                Path(drop_in),
                expected_uid=0,
                expected_gid=0,
                allowed_modes=frozenset({0o644}),
                maximum=1024 * 1024,
            )
            drop_in_sha256[drop_in] = hashlib.sha256(payload).hexdigest()
        if (
            unit == GATEWAY_UNIT
            and state["ActiveState"] == "active"
            and drop_in_paths == [GATEWAY_CONNECTOR_DROP_IN]
        ):
            self._assert_gateway_credential_free(int(state["MainPID"]))
        raw = {
            "schema": SERVICE_SCHEMA,
            "name": unit,
            "fragment_path": state["FragmentPath"],
            "fragment_sha256": fragment_sha256,
            "load_state": state["LoadState"],
            "active_state": state["ActiveState"],
            "sub_state": state["SubState"],
            "unit_file_state": state["UnitFileState"],
            "main_pid": int(state["MainPID"]),
            "drop_in_paths": drop_in_paths,
            "drop_in_sha256": drop_in_sha256,
            "need_daemon_reload": state["NeedDaemonReload"] == "yes",
            "triggered_by": [],
            "triggers": [],
            "observed_at_unix": int(self._clock()),
        }
        return ServiceObservation.from_mapping(
            {**raw, "observation_sha256": _sha256_json(raw)}
        )

    def _change(self, action: str, unit: str) -> None:
        if action not in {"start", "stop"} or unit not in {
            GATEWAY_UNIT,
            WRITER_UNIT,
            CONNECTOR_UNIT,
        }:
            raise ValueError("production systemd action is not fixed")
        result = self._runner(
            activation.Command(
                (SYSTEMCTL, action, "--", unit),
                timeout_seconds=30,
            )
        )
        if getattr(result, "returncode", 1) != 0:
            raise ProductionCutoverError("production_systemd_action_failed")

    def observe_gateway(self) -> ServiceObservation:
        return self._observe(GATEWAY_UNIT)

    def observe_writer(self) -> ServiceObservation:
        return self._observe(WRITER_UNIT)

    def observe_connector(self) -> ServiceObservation:
        return self._observe(CONNECTOR_UNIT)

    def stop_gateway(self) -> None:
        self._change("stop", GATEWAY_UNIT)

    def stop_writer(self) -> None:
        self._change("stop", WRITER_UNIT)

    def stop_connector(self) -> None:
        self._change("stop", CONNECTOR_UNIT)

    def start_gateway(self) -> None:
        self._change("start", GATEWAY_UNIT)


class SnapshotBoundary(Protocol):
    def observe_final_tail(self, plan: FreezePlan) -> LegacySnapshot: ...
    def observe_before_apply(self, plan: CutoverPlan) -> LegacySnapshot: ...


class DatabaseCutoverBoundary(Protocol):
    def preflight(self, plan: CutoverPlan) -> Mapping[str, Any]: ...
    def apply(self, plan: CutoverPlan) -> Mapping[str, Any]: ...
    def terminal(self, plan: CutoverPlan) -> Mapping[str, Any]: ...
    def rollback(
        self,
        plan: CutoverPlan,
        apply_receipt: Mapping[str, Any] | None,
    ) -> Mapping[str, Any]: ...


class HostActivationBoundary(Protocol):
    def apply_stopped(self, plan: CutoverPlan) -> Mapping[str, Any]: ...
    def start_prerequisites(self, plan: CutoverPlan) -> Mapping[str, Any]: ...
    def start_writer(self, plan: CutoverPlan) -> Mapping[str, Any]: ...
    def commit_boot(self, plan: CutoverPlan) -> Mapping[str, Any]: ...
    def rollback(self, plan: CutoverPlan, apply_receipt: Mapping[str, Any] | None) -> Mapping[str, Any]: ...


class CronCutoverBoundary(Protocol):
    """Transactional packaged-cron edge owned by the core coordinator."""

    def preflight(self, plan: CutoverPlan) -> Mapping[str, Any]: ...
    def apply(
        self,
        plan: CutoverPlan,
        preflight_receipt: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...
    def postflight(
        self,
        plan: CutoverPlan,
        apply_receipt: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...
    def activate(
        self,
        plan: CutoverPlan,
        postflight_receipt: Mapping[str, Any],
        activation_authority: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...
    def rollback(
        self,
        plan: CutoverPlan,
        apply_receipt: Mapping[str, Any] | None,
    ) -> Mapping[str, Any]: ...


class ProductionCronCutoverBoundary:
    """Invoke only the digest-attested isolated release entrypoint.

    The semantic migration plan is already owner-bound inside ``CutoverPlan``.
    This adapter performs no routing or classification: it constructs a fixed
    argv, strips the inherited environment, and validates the one canonical
    receipt emitted by the exact release runtime.
    """

    _ACTIONS = frozenset({
        "preflight", "apply", "postflight", "activate", "rollback",
    })

    def __init__(
        self,
        *,
        runner: Callable[..., Any] = subprocess.run,
    ) -> None:
        self._runner = runner

    @staticmethod
    def _identity(plan: CutoverPlan) -> tuple[Path, Path, Mapping[str, Any]]:
        continuity = plan.value["cron_continuity_plan"]
        if (
            continuity.get("schema")
            != production_cron_continuity_package.PLAN_SCHEMA
        ):
            raise ProductionCutoverError(
                "production_cron_cutover_plan_not_packaged"
            )
        release_root = Path(
            continuity["trusted_collector_package"]["release_root"]
        )
        expected_root = PRODUCTION_RELEASE_BASE / (
            f"hermes-agent-{plan.value['release_revision'][:12]}"
        )
        entrypoint = release_root / (
            production_cron_continuity_package.CUTOVER_ENTRYPOINT_RELATIVE_PATH
        )
        if release_root != expected_root:
            raise ProductionCutoverError(
                "production_cron_cutover_release_identity_invalid"
            )
        return release_root / ".venv/bin/python", entrypoint, continuity

    def _run(
        self,
        *,
        action: str,
        plan: CutoverPlan,
        expected_prior_sha256: str | None = None,
        expected_activation_authority_sha256: str | None = None,
    ) -> Mapping[str, Any]:
        if action not in self._ACTIONS:
            raise ValueError("production cron cutover action is not fixed")
        interpreter, entrypoint, continuity = self._identity(plan)
        argv = [
            str(interpreter),
            "-B",
            "-I",
            str(entrypoint),
            action,
            "--expected-cutover-plan-sha256",
            plan.sha256,
            "--expected-entrypoint-sha256",
            str(continuity["cutover_entrypoint_sha256"]),
            "--expected-runtime-sha256",
            str(continuity["cutover_runtime_sha256"]),
        ]
        if expected_prior_sha256 is not None:
            argv.extend((
                "--expected-prior-receipt-sha256",
                expected_prior_sha256,
            ))
        if expected_activation_authority_sha256 is not None:
            argv.extend((
                "--expected-activation-authority-sha256",
                expected_activation_authority_sha256,
            ))
        try:
            result = self._runner(
                tuple(argv),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd="/",
                env={
                    "LC_ALL": "C.UTF-8",
                    "PATH": "/usr/bin:/bin",
                    "PYTHONNOUSERSITE": "1",
                },
                timeout=_ARTIFACT_TIMEOUT_SECONDS,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ProductionCutoverError(
                "production_cron_cutover_boundary_failed"
            ) from exc
        stdout = getattr(result, "stdout", b"")
        stderr = getattr(result, "stderr", b"")
        if (
            getattr(result, "returncode", 1) != 0
            or not isinstance(stdout, bytes)
            or not isinstance(stderr, bytes)
            or stderr
            or not 1 < len(stdout) <= _MAX_JSON + 1
        ):
            raise ProductionCutoverError(
                "production_cron_cutover_boundary_failed"
            )
        try:
            value = json.loads(
                stdout.decode("utf-8", errors="strict"),
                object_pairs_hook=activation._reject_duplicate_keys,
                parse_constant=activation._reject_json_constant,
            )
        except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
            raise ProductionCutoverError(
                "production_cron_cutover_receipt_invalid"
            ) from exc
        if (
            not isinstance(value, Mapping)
            or stdout != _canonical_bytes(value) + b"\n"
        ):
            raise ProductionCutoverError(
                "production_cron_cutover_receipt_invalid"
            )
        receipt = _require_cron_cutover_receipt(
            value,
            action=action,
            plan=plan,
            expected_prior_sha256=expected_prior_sha256,
        )
        if (
            action == "activate"
            and receipt["activation_authority_sha256"]
            != expected_activation_authority_sha256
        ):
            raise ProductionCutoverError(
                "production_cron_activation_lineage_invalid"
            )
        return receipt

    def preflight(self, plan: CutoverPlan) -> Mapping[str, Any]:
        return self._run(action="preflight", plan=plan)

    def apply(
        self,
        plan: CutoverPlan,
        preflight_receipt: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        return self._run(
            action="apply",
            plan=plan,
            expected_prior_sha256=str(preflight_receipt["receipt_sha256"]),
        )

    def postflight(
        self,
        plan: CutoverPlan,
        apply_receipt: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        return self._run(
            action="postflight",
            plan=plan,
            expected_prior_sha256=str(apply_receipt["receipt_sha256"]),
        )

    def activate(
        self,
        plan: CutoverPlan,
        postflight_receipt: Mapping[str, Any],
        activation_authority: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        from gateway import production_cron_cutover_runtime as cron_runtime

        authority = cron_runtime.validate_activation_authority(
            activation_authority,
            cutover_plan_sha256=plan.sha256,
            cron_postflight_receipt_sha256=str(
                postflight_receipt["receipt_sha256"]
            ),
            expected_authority_sha256=str(
                activation_authority.get("authority_sha256") or ""
            ),
        )
        activation._ensure_root_directory(
            cron_runtime.STAGED_ACTIVATION_AUTHORITY_PATH.parent
        )
        activation._write_root_receipt(
            cron_runtime.STAGED_ACTIVATION_AUTHORITY_PATH,
            authority,
        )
        return self._run(
            action="activate",
            plan=plan,
            expected_prior_sha256=str(
                postflight_receipt["receipt_sha256"]
            ),
            expected_activation_authority_sha256=str(
                authority["authority_sha256"]
            ),
        )

    def rollback(
        self,
        plan: CutoverPlan,
        apply_receipt: Mapping[str, Any] | None,
    ) -> Mapping[str, Any]:
        return self._run(
            action="rollback",
            plan=plan,
            expected_prior_sha256=(
                None
                if apply_receipt is None
                else str(apply_receipt["receipt_sha256"])
            ),
        )


class CapabilityPrerequisiteBoundary(Protocol):
    def collect_and_validate(self, plan: CutoverPlan) -> Mapping[str, Any]: ...


_ISOLATION_EQUIVALENCE_FIELDS = frozenset({
    "schema",
    "plan_sha256",
    "run_id",
    "release_revision",
    "fixture_sha256",
    "workspace_gateway_receipt_sha256",
    "goal_continuation_terminal_sha256",
    "canary_projection_sha256",
    "normalized_canary_projection_sha256",
    "normalized_production_projection_sha256",
    "environment_specific_fields_normalized",
    "production_gateway_config_sha256",
    "production_model_route",
    "canary_semantic_config_contract_sha256",
    "production_semantic_config_contract",
    "production_semantic_config_contract_sha256",
    "canary_ordered_toolsets_sha256",
    "production_ordered_toolsets",
    "production_ordered_toolsets_sha256",
    "canary_capability_role_topology_contract_sha256",
    "production_capability_role_topology_contract",
    "production_capability_role_topology_contract_sha256",
    "production_capability_topology_identity_sha256",
    "production_mutation_observed",
    "equivalent",
    "projection_sha256",
})

_PRE_DB_ZERO_WRITE_OBSERVATION_FIELDS = frozenset({
    "schema",
    "plan_sha256",
    "gateway_fenced",
    "writer_stopped",
    "connector_active",
    "final_legacy_snapshot",
    "frozen_truth_sha256",
    "canonical_database_write_path_stopped",
    "legacy_schema_not_applied",
    "canonical_database_mutation_observed",
    "staging_and_service_lifecycle_mutations_observed",
    "observed_at_unix",
    "observation_sha256",
})


def _build_pre_db_zero_write_observation(
    *,
    plan: CutoverPlan,
    gateway: ServiceObservation,
    writer: ServiceObservation,
    connector: ServiceObservation,
    snapshot: LegacySnapshot,
) -> dict[str, Any]:
    """Bind the exact fenced pre-DB state, not a self-asserted boolean."""

    def _expected_staged_identity(
        identity: Mapping[str, Any],
    ) -> dict[str, Any]:
        # The owner-approved target identity describes the committed boot
        # state.  Before the activation intent every newly installed unit is
        # deliberately disabled, even when the connector is already running
        # for readiness.  Normalize only that lifecycle field; every file,
        # drop-in and executable identity remains byte-exact.
        expected = copy.deepcopy(dict(identity))
        expected["unit_file_state"] = "disabled"
        return expected

    if (
        not gateway.stopped
        or not writer.stopped
        or connector.value["active_state"] != "active"
        or gateway.stable_identity()
        != _expected_staged_identity(plan.value["gateway_target_identity"])
        or writer.stable_identity()
        != _expected_staged_identity(plan.value["writer_target_identity"])
        or connector.stable_identity()
        != _expected_staged_identity(plan.value["connector_target_identity"])
        or snapshot.frozen_truth() != plan.final_snapshot.frozen_truth()
        or snapshot.value["shape"] != "legacy19"
    ):
        raise ProductionCutoverError(
            "production_pre_db_zero_write_observation_invalid"
        )
    observed = max(
        gateway.value["observed_at_unix"],
        writer.value["observed_at_unix"],
        connector.value["observed_at_unix"],
        snapshot.value["observed_at_unix"],
    )
    unsigned = {
        "schema": "muncho-production-pre-db-zero-write-observation.v2",
        "plan_sha256": plan.sha256,
        "gateway_fenced": gateway.to_mapping(),
        "writer_stopped": writer.to_mapping(),
        "connector_active": connector.to_mapping(),
        "final_legacy_snapshot": snapshot.to_mapping(),
        "frozen_truth_sha256": _sha256_json(snapshot.frozen_truth()),
        "canonical_database_write_path_stopped": True,
        "legacy_schema_not_applied": True,
        "canonical_database_mutation_observed": False,
        "staging_and_service_lifecycle_mutations_observed": True,
        "observed_at_unix": observed,
    }
    return {**unsigned, "observation_sha256": _sha256_json(unsigned)}


def _require_pre_db_zero_write_observation(
    value: Any,
    *,
    plan: CutoverPlan,
) -> dict[str, Any]:
    raw = _hashed(
        value,
        _PRE_DB_ZERO_WRITE_OBSERVATION_FIELDS,
        "observation_sha256",
        "production pre-DB zero-write observation",
    )
    try:
        expected = _build_pre_db_zero_write_observation(
            plan=plan,
            gateway=ServiceObservation.from_mapping(raw["gateway_fenced"]),
            writer=ServiceObservation.from_mapping(raw["writer_stopped"]),
            connector=ServiceObservation.from_mapping(raw["connector_active"]),
            snapshot=LegacySnapshot.from_mapping(raw["final_legacy_snapshot"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ProductionCutoverError(
            "production_pre_db_zero_write_observation_invalid"
        ) from exc
    if raw != expected:
        raise ProductionCutoverError(
            "production_pre_db_zero_write_observation_invalid"
        )
    return raw


def _build_production_isolation_equivalence(
    *,
    plan: CutoverPlan,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Project canary and production onto one environment-neutral contract.

    Only the reviewed Discord channel is environment-specific.  Every model,
    semantic-authority, task-workspace, transport, restart, prompt-cache, and
    tool-schema field stays byte-exact.  The production release, config,
    toolset and service-topology identities remain explicit in the receipt.
    """

    from gateway import canonical_capability_canary_e2e as canary_e2e

    canary = copy.deepcopy(dict(evidence["isolation_equivalence_projection"]))
    production_semantic = canary_e2e.build_semantic_config_contract()
    production_toolsets = list(FIRST_WAVE_TOOLSETS)
    production_topology = canary_e2e.build_capability_role_topology_contract(
        public_discord_target={
            "target_type": "public_channel",
            "guild_id": SKYVISION_GUILD_ID,
            "channel_id": SKYVISION_CONTROL_TOWER_CHANNEL_ID,
        },
        discord_bot_identities={
            "production_bot_user_id": (
                canary_e2e.PRODUCTION_DISCORD_BOT_USER_ID
            ),
            "connector_bot_user_id": (
                canary_e2e.PRODUCTION_DISCORD_CONNECTOR_BOT_USER_ID
            ),
            "routeback_bot_user_id": (
                canary_e2e.PRODUCTION_DISCORD_ROUTEBACK_BOT_USER_ID
            ),
        },
    )
    production_semantic_sha256 = _sha256_json(production_semantic)
    production_toolsets_sha256 = _sha256_json(
        {"toolsets": production_toolsets}
    )
    production_topology_sha256 = _sha256_json(production_topology)
    production = {
        "model": "gpt-5.6-sol",
        "provider": "openai-codex",
        "api_mode": "codex_responses",
        "fallback_configured": False,
        "fallback_used": False,
        "goal_manager": production_semantic["goals"]["manager"],
        "model_semantic_authority": True,
        "auxiliary_semantic_authority": False,
        "canonical_task_workspace": "durable_restart_recovery_v1",
        "discord_transport": production_semantic["discord"][
            "public_transport"
        ],
        "discord_guild_id": SKYVISION_GUILD_ID,
        "discord_channel_id": SKYVISION_CONTROL_TOWER_CHANNEL_ID,
        "direct_discord_in_gateway": production_semantic["discord"][
            "direct_discord_in_gateway"
        ],
        "discord_dm_enabled": production_semantic["discord"]["dm_enabled"],
        "full_gateway_restart": True,
        "min_model_authored_continue_outcomes": 2,
        "user_preemption_queue_preserved": True,
        "production_mutation_observed": False,
        "prompt_cache_stable": True,
        "tool_schema_stable": True,
        "semantic_config_contract": production_semantic,
        "semantic_config_contract_sha256": production_semantic_sha256,
        "ordered_toolsets": production_toolsets,
        "ordered_toolsets_sha256": production_toolsets_sha256,
        "capability_role_topology_contract": production_topology,
        "capability_role_topology_contract_sha256": (
            production_topology_sha256
        ),
    }
    normalized_canary = copy.deepcopy(canary)
    normalized_production = copy.deepcopy(production)
    for projection in (normalized_canary, normalized_production):
        projection["discord_channel_id"] = "<environment-specific-channel>"
        projection["capability_role_topology_contract"][
            "public_discord_target"
        ]["channel_id"] = "<environment-specific-channel>"
        projection["capability_role_topology_contract_sha256"] = _sha256_json(
            projection["capability_role_topology_contract"]
        )
    normalized_canary_sha256 = _sha256_json(normalized_canary)
    normalized_production_sha256 = _sha256_json(normalized_production)
    capability_topology = plan.value["capability_topology"]
    topology_sha256 = production_capability_topology_identity_sha256(
        capability_topology
    )
    production_model_route = {
        "model": "gpt-5.6-sol",
        "provider": "openai-codex",
        "api_mode": "codex_responses",
        "fallback_configured": False,
    }
    host_transition = plan.value["host_transition"]
    if (
        evidence["release_revision"] != plan.value["release_revision"]
        or canary["discord_guild_id"] != SKYVISION_GUILD_ID
        or canary["model"] != production_model_route["model"]
        or canary["provider"] != production_model_route["provider"]
        or canary["api_mode"] != production_model_route["api_mode"]
        or canary["fallback_configured"] is not False
        or canary["fallback_used"] is not False
        or canary["production_mutation_observed"] is not False
        or canary["semantic_config_contract"] != production_semantic
        or canary["semantic_config_contract_sha256"]
        != production_semantic_sha256
        or canary["ordered_toolsets"] != production_toolsets
        or canary["ordered_toolsets_sha256"]
        != production_toolsets_sha256
        or plan.value["gateway_target_identity"]["name"]
        != production_topology["gateway"]["service_unit"]
        or plan.value["writer_target_identity"]["name"]
        != production_topology["canonical_writer"]["service_unit"]
        or capability_topology["public_connector"]["unit"]
        != production_topology["discord_connector"]["service_unit"]
        or capability_topology["routeback_edge"]["unit"]
        != production_topology["discord_routeback"]["service_unit"]
        or host_transition["gateway_direct_discord_enabled"] is not False
        or host_transition["discord_dm_allowed"] is not False
        or production_semantic["goals"]["max_turns"] != 0
        or production_semantic["kanban"]
        != {
            "auxiliary_planning_enabled": False,
            "auto_decompose": False,
            "dispatch_in_gateway": False,
        }
        or production_semantic["canonical_writer"]
        != {
            "privileged_writer_boundary_enabled": True,
            "gateway_direct_write_enabled": False,
            "dangerous_mutation_requires_owner_approval": True,
        }
        or production_semantic["discord"]
        != {
            "public_transport": "credential_free_local_connector_relay",
            "direct_discord_in_gateway": False,
            "dm_enabled": False,
        }
        or normalized_canary_sha256 != normalized_production_sha256
    ):
        raise ProductionCutoverError(
            "production_isolation_equivalence_projection_invalid"
        )
    unsigned = {
        "schema": ISOLATION_EQUIVALENCE_SCHEMA,
        "plan_sha256": plan.sha256,
        "run_id": evidence["run_id"],
        "release_revision": plan.value["release_revision"],
        "fixture_sha256": evidence["fixture_sha256"],
        "workspace_gateway_receipt_sha256": evidence[
            "workspace_gateway_receipt_sha256"
        ],
        "goal_continuation_terminal_sha256": evidence[
            "goal_continuation_terminal_sha256"
        ],
        "canary_projection_sha256": evidence[
            "isolation_equivalence_projection_sha256"
        ],
        "normalized_canary_projection_sha256": normalized_canary_sha256,
        "normalized_production_projection_sha256": (
            normalized_production_sha256
        ),
        "environment_specific_fields_normalized": [
            "discord_channel_id",
            (
                "capability_role_topology_contract."
                "public_discord_target.channel_id"
            ),
        ],
        "production_gateway_config_sha256": plan.value["host_transition"][
            "files"
        ]["gateway_config"]["sha256"],
        "production_model_route": production_model_route,
        "canary_semantic_config_contract_sha256": canary[
            "semantic_config_contract_sha256"
        ],
        "production_semantic_config_contract": production_semantic,
        "production_semantic_config_contract_sha256": (
            production_semantic_sha256
        ),
        "canary_ordered_toolsets_sha256": canary[
            "ordered_toolsets_sha256"
        ],
        "production_ordered_toolsets": production_toolsets,
        "production_ordered_toolsets_sha256": production_toolsets_sha256,
        "canary_capability_role_topology_contract_sha256": canary[
            "capability_role_topology_contract_sha256"
        ],
        "production_capability_role_topology_contract": production_topology,
        "production_capability_role_topology_contract_sha256": (
            production_topology_sha256
        ),
        "production_capability_topology_identity_sha256": topology_sha256,
        "production_mutation_observed": False,
        "equivalent": True,
    }
    return {**unsigned, "projection_sha256": _sha256_json(unsigned)}


class ProductionCapabilityPrerequisiteBoundary:
    """Re-attest the exact root-collected receipt before any forward edge."""

    def __init__(
        self,
        *,
        services: ServiceBoundary,
        snapshots: SnapshotBoundary,
    ) -> None:
        self._services = services
        self._snapshots = snapshots

    def collect_and_validate(self, plan: CutoverPlan) -> Mapping[str, Any]:
        topology = plan.value["capability_topology"]
        canary_evidence = _validate_isolated_canary_goal_prerequisite(
            plan.value["freeze_plan"]["cutover_authority"][
                "isolated_canary_goal_prerequisite"
            ],
            revision=plan.value["release_revision"],
        )
        equivalence = _build_production_isolation_equivalence(
            plan=plan,
            evidence=canary_evidence,
        )
        try:
            collect_and_install_from_production_config(
                revision=plan.value["release_revision"],
                config_sha256=plan.value["host_transition"]["files"][
                    "gateway_config"
                ]["sha256"],
                lifecycle_phase=PREREQUISITE_LIFECYCLE_STAGED,
            )
            current = load_production_capability_prerequisite_receipt(
                revision=plan.value["release_revision"],
                topology=topology,
                lifecycle_phase=PREREQUISITE_LIFECYCLE_STAGED,
            )
            pre_db_observation = _build_pre_db_zero_write_observation(
                plan=plan,
                gateway=self._services.observe_gateway(),
                writer=self._services.observe_writer(),
                connector=self._services.observe_connector(),
                snapshot=self._snapshots.observe_before_apply(plan),
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ProductionCutoverError(
                "production_capability_prerequisite_invalid"
            ) from exc
        unsigned = {
            "schema": CAPABILITY_PREREQUISITE_ACCEPTANCE_SCHEMA,
            "plan_sha256": plan.sha256,
            "production_owner_approval_sha256": plan.value[
                "freeze_approval_sha256"
            ],
            "prerequisite_receipt_sha256": current["receipt_sha256"],
            "prerequisite_file_sha256": _sha256_json(current),
            "topology_identity_sha256": (
                production_capability_topology_identity_sha256(topology)
            ),
            "boot_id_sha256": current["boot_id_sha256"],
            "pre_db_zero_write_observation": pre_db_observation,
            "pre_db_zero_write_observation_sha256": pre_db_observation[
                "observation_sha256"
            ],
            "isolated_canary_evidence_sha256": canary_evidence[
                "evidence_sha256"
            ],
            "workspace_gateway_receipt_sha256": canary_evidence[
                "workspace_gateway_receipt_sha256"
            ],
            "goal_continuation_terminal_schema": canary_evidence[
                "goal_continuation_terminal_schema"
            ],
            "goal_continuation_terminal_sha256": canary_evidence[
                "goal_continuation_terminal_sha256"
            ],
            "canary_run_id": canary_evidence["run_id"],
            "canary_release_revision": canary_evidence["release_revision"],
            "canary_fixture_sha256": canary_evidence["fixture_sha256"],
            "canary_capability_plan_sha256": canary_evidence[
                "capability_plan_sha256"
            ],
            "canary_full_canary_plan_sha256": canary_evidence[
                "full_canary_plan_sha256"
            ],
            "canary_owner_approval_receipt_sha256": canary_evidence[
                "canary_owner_approval_receipt_sha256"
            ],
            "production_diff_sha256": canary_evidence[
                "production_diff_sha256"
            ],
            "isolation_equivalence_projection": equivalence,
            "isolation_equivalence_projection_sha256": equivalence[
                "projection_sha256"
            ],
            "zero_canonical_database_mutation_observed": (
                pre_db_observation["canonical_database_mutation_observed"]
                is False
            ),
            "ok": True,
            "secret_material_recorded": False,
            "secret_digest_recorded": False,
        }
        return {**unsigned, "receipt_sha256": _sha256_json(unsigned)}


@dataclass(frozen=True)
class FreezeDependencies:
    services: ServiceBoundary
    snapshots: SnapshotBoundary
    journal: CutoverJournal
    lock: Callable[[], ContextManager[Any]] = activation._host_activation_lock


@dataclass(frozen=True)
class CutoverDependencies:
    services: ServiceBoundary
    snapshots: SnapshotBoundary
    database: DatabaseCutoverBoundary
    host: HostActivationBoundary
    journal: CutoverJournal
    prerequisites: CapabilityPrerequisiteBoundary
    lock: Callable[[], ContextManager[Any]] = activation._host_activation_lock
    cron: CronCutoverBoundary | None = None


class ProductionArtifactProcessBoundary:
    """Run only plan-bound, root-materialized mechanical edge executables.

    The coordinator intentionally does not duplicate PostgreSQL reconciliation
    SQL or host activation logic.  Each reviewed release supplies sealed,
    self-contained executables for those edges.  This adapter copies the exact
    owner-approved bytes from the fixed production release into the root-owned
    journal tree before executing them with a secret-free canonical request.
    The child reads any credential it needs from its own fixed privileged path;
    no credential or inherited environment crosses this boundary.
    """

    _ACTIONS = frozenset({
        "observe_final_tail",
        "observe_before_apply",
        "database_preflight",
        "database_apply",
        "database_terminal",
        "database_rollback",
        "host_apply_stopped",
        "host_start_prerequisites",
        "host_start_writer",
        "host_commit_boot",
        "host_rollback",
    })

    def __init__(
        self,
        *,
        runner: Callable[..., Any] = subprocess.run,
    ) -> None:
        self._runner = runner

    @staticmethod
    def _read_release_file(path: Path, *, maximum: int) -> bytes:
        try:
            resolved = path.resolve(strict=True)
            observed = os.lstat(path)
        except OSError as exc:
            raise ProductionCutoverError(
                "production_cutover_artifact_unavailable"
            ) from exc
        if (
            resolved != path
            or stat.S_ISLNK(observed.st_mode)
            or not stat.S_ISREG(observed.st_mode)
            or observed.st_nlink != 1
            or observed.st_size < 1
            or observed.st_size > maximum
        ):
            raise ProductionCutoverError(
                "production_cutover_artifact_identity_invalid"
            )
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(path, flags)
            try:
                opened = os.fstat(descriptor)
                if (
                    (opened.st_dev, opened.st_ino)
                    != (observed.st_dev, observed.st_ino)
                    or opened.st_size != observed.st_size
                ):
                    raise ProductionCutoverError(
                        "production_cutover_artifact_raced"
                    )
                chunks: list[bytes] = []
                remaining = maximum + 1
                while remaining:
                    chunk = os.read(descriptor, min(remaining, 64 * 1024))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    remaining -= len(chunk)
                payload = b"".join(chunks)
                if len(payload) > maximum or len(payload) != opened.st_size:
                    raise ProductionCutoverError(
                        "production_cutover_artifact_size_invalid"
                    )
                return payload
            finally:
                os.close(descriptor)
        except ProductionCutoverError:
            raise
        except OSError as exc:
            raise ProductionCutoverError(
                "production_cutover_artifact_read_failed"
            ) from exc

    def _materialize(
        self,
        *,
        plan: FreezePlan | CutoverPlan,
        binding: Mapping[str, Any],
        label: str,
    ) -> Path:
        revision = str(plan.value["release_revision"])
        artifact = _artifact(binding, label, revision)
        source = Path(artifact["path"])
        release_root = (
            PRODUCTION_RELEASE_BASE / f"hermes-agent-{revision[:12]}"
        )
        marker = self._read_release_file(
            release_root / ".codex-source-commit", maximum=128
        )
        if marker != (revision + "\n").encode("ascii"):
            raise ProductionCutoverError(
                "production_cutover_release_identity_invalid"
            )
        payload = self._read_release_file(source, maximum=_MAX_ARTIFACT)
        if hashlib.sha256(payload).hexdigest() != artifact["sha256"]:
            raise ProductionCutoverError(
                "production_cutover_artifact_digest_invalid"
            )
        if not (payload.startswith(b"#!") or payload.startswith(b"\x7fELF")):
            raise ProductionCutoverError(
                "production_cutover_artifact_format_invalid"
            )
        root = EVIDENCE_ROOT / "plans" / plan.sha256 / "executables"
        activation._ensure_root_directory(root)
        target = root / f"{artifact['sha256']}.exec"
        activation._install_exact_bytes(
            target,
            payload,
            uid=0,
            gid=0,
            mode=0o500,
        )
        return target

    def _invoke(
        self,
        *,
        plan: FreezePlan | CutoverPlan,
        binding: Mapping[str, Any],
        label: str,
        action: str,
        apply_receipt: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        if action not in self._ACTIONS:
            raise ValueError("production cutover artifact action is invalid")
        executable = self._materialize(
            plan=plan,
            binding=binding,
            label=label,
        )
        unsigned = {
            "schema": ARTIFACT_REQUEST_SCHEMA,
            "action": action,
            "plan": plan.to_mapping(),
            "apply_receipt": (
                None
                if apply_receipt is None
                else copy.deepcopy(dict(apply_receipt))
            ),
            "secret_material_recorded": False,
        }
        request = {**unsigned, "request_sha256": _sha256_json(unsigned)}
        try:
            completed = self._runner(
                (str(executable), action),
                input=_canonical_bytes(request) + b"\n",
                capture_output=True,
                check=False,
                timeout=_ARTIFACT_TIMEOUT_SECONDS,
                env={
                    "LC_ALL": "C.UTF-8",
                    "PATH": "/usr/bin:/bin",
                    "PYTHONNOUSERSITE": "1",
                },
                cwd="/",
                close_fds=True,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise ProductionCutoverError(
                "production_cutover_artifact_execution_failed"
            ) from exc
        stdout = getattr(completed, "stdout", None)
        stderr = getattr(completed, "stderr", None)
        if (
            getattr(completed, "returncode", 1) != 0
            or not isinstance(stdout, bytes)
            or not isinstance(stderr, bytes)
            or stderr
            or not stdout
            or len(stdout) > _MAX_JSON + 1
        ):
            raise ProductionCutoverError(
                "production_cutover_artifact_execution_failed"
            )
        payload = stdout[:-1] if stdout.endswith(b"\n") else b""
        try:
            value = json.loads(
                payload.decode("utf-8", errors="strict"),
                object_pairs_hook=activation._reject_duplicate_keys,
                parse_constant=activation._reject_json_constant,
            )
        except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
            raise ProductionCutoverError(
                "production_cutover_artifact_response_invalid"
            ) from exc
        if (
            not isinstance(value, Mapping)
            or payload != _canonical_bytes(value)
        ):
            raise ProductionCutoverError(
                "production_cutover_artifact_response_invalid"
            )
        return copy.deepcopy(dict(value))

    def observe_final_tail(self, plan: FreezePlan) -> LegacySnapshot:
        return LegacySnapshot.from_mapping(
            self._invoke(
                plan=plan,
                binding=plan.value["observe_artifact"],
                label="observe artifact",
                action="observe_final_tail",
            )
        )

    def observe_before_apply(self, plan: CutoverPlan) -> LegacySnapshot:
        return LegacySnapshot.from_mapping(
            self._invoke(
                plan=plan,
                binding=plan.value["artifacts"]["observe"],
                label="observe artifact",
                action="observe_before_apply",
            )
        )

    def preflight(self, plan: CutoverPlan) -> Mapping[str, Any]:
        return self._invoke(
            plan=plan,
            binding=plan.value["artifacts"]["database_postflight"],
            label="database postflight",
            action="database_preflight",
        )

    def apply(self, plan: CutoverPlan) -> Mapping[str, Any]:
        return self._invoke(
            plan=plan,
            binding=plan.value["artifacts"]["database_apply"],
            label="database apply",
            action="database_apply",
        )

    def terminal(self, plan: CutoverPlan) -> Mapping[str, Any]:
        return self._invoke(
            plan=plan,
            binding=plan.value["artifacts"]["database_postflight"],
            label="database postflight",
            action="database_terminal",
        )

    def database_rollback(
        self,
        plan: CutoverPlan,
        apply_receipt: Mapping[str, Any] | None,
    ) -> Mapping[str, Any]:
        return self._invoke(
            plan=plan,
            binding=plan.value["artifacts"]["database_rollback"],
            label="database rollback",
            action="database_rollback",
            apply_receipt=apply_receipt,
        )

    def apply_stopped(self, plan: CutoverPlan) -> Mapping[str, Any]:
        return self._invoke(
            plan=plan,
            binding=plan.value["artifacts"]["host_activation"],
            label="host activation",
            action="host_apply_stopped",
        )

    def start_writer(self, plan: CutoverPlan) -> Mapping[str, Any]:
        return self._invoke(
            plan=plan,
            binding=plan.value["artifacts"]["host_activation"],
            label="host activation",
            action="host_start_writer",
        )

    def start_prerequisites(self, plan: CutoverPlan) -> Mapping[str, Any]:
        return self._invoke(
            plan=plan,
            binding=plan.value["artifacts"]["host_activation"],
            label="host activation",
            action="host_start_prerequisites",
        )

    def commit_boot(self, plan: CutoverPlan) -> Mapping[str, Any]:
        return self._invoke(
            plan=plan,
            binding=plan.value["artifacts"]["host_activation"],
            label="host activation",
            action="host_commit_boot",
        )


    def host_rollback(
        self,
        plan: CutoverPlan,
        apply_receipt: Mapping[str, Any] | None,
    ) -> Mapping[str, Any]:
        return self._invoke(
            plan=plan,
            binding=plan.value["artifacts"]["host_rollback"],
            label="host rollback",
            action="host_rollback",
            apply_receipt=apply_receipt,
        )


@dataclass(frozen=True)
class ProductionDatabaseArtifactBoundary:
    process: ProductionArtifactProcessBoundary

    def preflight(self, plan: CutoverPlan) -> Mapping[str, Any]:
        return self.process.preflight(plan)

    def apply(self, plan: CutoverPlan) -> Mapping[str, Any]:
        return self.process.apply(plan)

    def terminal(self, plan: CutoverPlan) -> Mapping[str, Any]:
        return self.process.terminal(plan)

    def rollback(
        self,
        plan: CutoverPlan,
        apply_receipt: Mapping[str, Any] | None,
    ) -> Mapping[str, Any]:
        return self.process.database_rollback(plan, apply_receipt)


@dataclass(frozen=True)
class ProductionHostArtifactBoundary:
    process: ProductionArtifactProcessBoundary

    def apply_stopped(self, plan: CutoverPlan) -> Mapping[str, Any]:
        return self.process.apply_stopped(plan)

    def start_writer(self, plan: CutoverPlan) -> Mapping[str, Any]:
        return self.process.start_writer(plan)

    def start_prerequisites(self, plan: CutoverPlan) -> Mapping[str, Any]:
        return self.process.start_prerequisites(plan)

    def commit_boot(self, plan: CutoverPlan) -> Mapping[str, Any]:
        return self.process.commit_boot(plan)


    def rollback(
        self,
        plan: CutoverPlan,
        apply_receipt: Mapping[str, Any] | None,
    ) -> Mapping[str, Any]:
        return self.process.host_rollback(plan, apply_receipt)


def _append_authority(
    journal: CutoverJournal,
    plan_sha256: str,
    approval: CutoverApproval,
    now_unix: int,
) -> list[JournalEntry]:
    entries = journal.load(plan_sha256)
    authorities = [entry for entry in entries if entry.value["event"] == "authority"]
    if authorities and authorities[-1].value["evidence"]["approval_sha256"] == approval.sha256:
        return entries
    expected_sequence = len(authorities)
    previous = None if not authorities else authorities[-1].value["evidence"]["approval_sha256"]
    if approval.value["sequence"] != expected_sequence or approval.value["previous_approval_sha256"] != previous:
        raise PermissionError("cutover approval chain does not resume this journal")
    journal.append(plan_sha256, "authority", {
        "approval_sha256": approval.sha256,
        "sequence": approval.value["sequence"],
    }, now_unix)
    return journal.load(plan_sha256)


def _last(entries: list[JournalEntry], event: str) -> JournalEntry | None:
    return next((item for item in reversed(entries) if item.value["event"] == event), None)


def _require_same_service_identity(expected: ServiceObservation, current: ServiceObservation) -> None:
    if expected.stable_identity() != current.stable_identity():
        raise ProductionCutoverError("production_service_identity_drifted")


_FREEZE_ABORT_FIELDS = frozenset({
    "schema", "freeze_plan_sha256", "approval_sha256", "trigger",
    "gateway_legacy_restarted", "writer_stopped", "connector_stopped",
    "database_mutated", "host_mutated", "secret_material_recorded",
    "completed_at_unix", "receipt_sha256",
})


def _validate_freeze_abort_receipt(
    value: Any,
    *,
    plan: FreezePlan,
) -> dict[str, Any]:
    raw = _hashed(
        value,
        _FREEZE_ABORT_FIELDS,
        "receipt_sha256",
        "freeze abort receipt",
    )
    if (
        raw["schema"] != FREEZE_ABORT_SCHEMA
        or raw["freeze_plan_sha256"] != plan.sha256
        or _SHA256.fullmatch(str(raw["approval_sha256"])) is None
        or raw["trigger"] not in {"capture_failure", "owner_abort"}
        or raw["gateway_legacy_restarted"] is not True
        or raw["writer_stopped"] is not True
        or raw["connector_stopped"] is not True
        or raw["database_mutated"] is not False
        or raw["host_mutated"] is not False
        or raw["secret_material_recorded"] is not False
        or type(raw["completed_at_unix"]) is not int
        or raw["completed_at_unix"] <= 0
    ):
        raise ProductionCutoverError("production_freeze_abort_receipt_invalid")
    return raw


def _restore_legacy_gateway_after_freeze(
    *,
    plan: FreezePlan,
    approval: CutoverApproval,
    dependencies: FreezeDependencies,
    entries: list[JournalEntry],
    trigger: str,
    now_unix: int,
) -> Mapping[str, Any]:
    terminal = _last(entries, "freeze_aborted")
    if terminal is not None:
        return _validate_freeze_abort_receipt(
            terminal.value["evidence"],
            plan=plan,
        )
    gateway_expected = ServiceObservation.from_mapping(plan.value["gateway_before"])
    writer_expected = ServiceObservation.from_mapping(plan.value["writer_before"])
    connector_expected = ServiceObservation.from_mapping(
        plan.value["connector_before"]
    )
    gateway = dependencies.services.observe_gateway()
    writer = dependencies.services.observe_writer()
    connector = dependencies.services.observe_connector()
    _require_same_service_identity(gateway_expected, gateway)
    _require_same_service_identity(writer_expected, writer)
    _require_same_service_identity(connector_expected, connector)
    if not writer.stopped or not connector.stopped:
        raise ProductionCutoverError(
            "production_freeze_abort_auxiliary_service_active"
        )
    if gateway.stopped:
        dependencies.services.start_gateway()
    gateway = dependencies.services.observe_gateway()
    writer = dependencies.services.observe_writer()
    connector = dependencies.services.observe_connector()
    _require_same_service_identity(gateway_expected, gateway)
    _require_same_service_identity(writer_expected, writer)
    _require_same_service_identity(connector_expected, connector)
    if (
        gateway.value["active_state"] != "active"
        or gateway.value["sub_state"] != "running"
        or gateway.value["main_pid"] <= 0
        or not writer.stopped
        or not connector.stopped
    ):
        raise ProductionCutoverError("production_freeze_abort_restart_failed")
    unsigned = {
        "schema": FREEZE_ABORT_SCHEMA,
        "freeze_plan_sha256": plan.sha256,
        "approval_sha256": approval.sha256,
        "trigger": trigger,
        "gateway_legacy_restarted": True,
        "writer_stopped": True,
        "connector_stopped": True,
        "database_mutated": False,
        "host_mutated": False,
        "secret_material_recorded": False,
        "completed_at_unix": now_unix,
    }
    receipt = _validate_freeze_abort_receipt(
        {**unsigned, "receipt_sha256": _sha256_json(unsigned)},
        plan=plan,
    )
    dependencies.journal.append(
        plan.sha256,
        "freeze_aborted",
        receipt,
        now_unix,
    )
    return receipt


def abort_freeze(
    plan: FreezePlan,
    approval_value: Mapping[str, Any],
    dependencies: FreezeDependencies,
    *,
    cutover_plan_staged: bool,
    now_unix: int | None = None,
) -> Mapping[str, Any]:
    """Restore the exact legacy gateway before any cutover mutation.

    The recorded freeze authority is revalidated at the time it entered the
    append-only journal, so recovery remains possible after a short approval
    lease expires.  A staged cutover plan or any non-freeze journal event
    blocks this path; forward mutations use the full rollback state machine.
    """

    if type(cutover_plan_staged) is not bool or cutover_plan_staged:
        raise ProductionCutoverError(
            "production_freeze_abort_cutover_plan_present"
        )
    now = int(time.time()) if now_unix is None else now_unix
    with dependencies.lock():
        entries = dependencies.journal.load(plan.sha256)
        if any(
            entry.value["event"]
            not in {
                "authority",
                "gateway_stopped",
                "final_tail_captured",
                "freeze_aborted",
            }
            for entry in entries
        ):
            raise ProductionCutoverError(
                "production_freeze_abort_journal_not_pre_mutation"
            )
        authorities = [
            entry for entry in entries if entry.value["event"] == "authority"
        ]
        matching = next(
            (
                entry
                for entry in authorities
                if entry.value["evidence"].get("approval_sha256")
                == approval_value.get("approval_sha256")
            ),
            None,
        )
        if matching is None:
            raise PermissionError(
                "freeze abort requires a recorded owner authority"
            )
        approval = CutoverApproval.from_mapping(
            approval_value,
            plan=plan,
            now_unix=matching.value["recorded_at_unix"],
        )
        return _restore_legacy_gateway_after_freeze(
            plan=plan,
            approval=approval,
            dependencies=dependencies,
            entries=entries,
            trigger="owner_abort",
            now_unix=now,
        )


def execute_final_tail_capture(
    plan: FreezePlan,
    approval_value: Mapping[str, Any],
    dependencies: FreezeDependencies,
    *,
    now_unix: int | None = None,
) -> FinalTailReceipt:
    """Stop the exact gateway and capture the final append-only legacy tail."""

    now = int(time.time()) if now_unix is None else now_unix
    approval = CutoverApproval.from_mapping(approval_value, plan=plan, now_unix=now)
    with dependencies.lock():
        entries = _append_authority(dependencies.journal, plan.sha256, approval, now)
        terminal = _last(entries, "final_tail_captured")
        if terminal is not None:
            return FinalTailReceipt.from_mapping(terminal.value["evidence"], plan=plan)
        gateway_expected = ServiceObservation.from_mapping(plan.value["gateway_before"])
        writer_expected = ServiceObservation.from_mapping(plan.value["writer_before"])
        connector_expected = ServiceObservation.from_mapping(
            plan.value["connector_before"]
        )
        gateway = dependencies.services.observe_gateway()
        writer = dependencies.services.observe_writer()
        connector = dependencies.services.observe_connector()
        _require_same_service_identity(gateway_expected, gateway)
        _require_same_service_identity(writer_expected, writer)
        _require_same_service_identity(connector_expected, connector)
        if not writer.stopped:
            dependencies.services.stop_writer()
            writer = dependencies.services.observe_writer()
        if not connector.stopped:
            dependencies.services.stop_connector()
            connector = dependencies.services.observe_connector()
        if not gateway.stopped:
            dependencies.services.stop_gateway()
            gateway = dependencies.services.observe_gateway()
        _require_same_service_identity(gateway_expected, gateway)
        _require_same_service_identity(writer_expected, writer)
        _require_same_service_identity(connector_expected, connector)
        if not gateway.stopped or not writer.stopped or not connector.stopped:
            raise ProductionCutoverError("production_services_not_stopped")
        if _last(entries, "gateway_stopped") is None:
            dependencies.journal.append(plan.sha256, "gateway_stopped", {
                "gateway": gateway.to_mapping(), "writer": writer.to_mapping(),
            }, now)
        try:
            snapshot = dependencies.snapshots.observe_final_tail(plan)
            initial = LegacySnapshot.from_mapping(plan.value["initial_snapshot"])
            if snapshot.storage_identity() != initial.storage_identity():
                raise ProductionCutoverError("legacy_storage_identity_drifted")
            if snapshot.value["source_row_count"] < initial.value["source_row_count"]:
                raise ProductionCutoverError("legacy_append_only_row_floor_regressed")
            bounds = plan.value["cutover_authority"]["final_tail_bounds"]
            if (
                snapshot.value["source_row_count"]
                - initial.value["source_row_count"]
                > bounds["max_appended_rows"]
                or now < initial.value["observed_at_unix"]
                or now - initial.value["observed_at_unix"]
                > bounds["max_capture_delay_seconds"]
            ):
                raise ProductionCutoverError("legacy_final_tail_bounds_exceeded")
            unsigned = {
                "schema": FREEZE_RECEIPT_SCHEMA,
                "freeze_plan_sha256": plan.sha256,
                "approval_sha256": approval.sha256,
                "gateway_stopped": True,
                "writer_stopped": True,
                "final_snapshot": snapshot.to_mapping(),
                "initial_row_count": initial.value["source_row_count"],
                "final_row_count": snapshot.value["source_row_count"],
                "append_only_row_floor_preserved": True,
                "captured_at_unix": now,
            }
            receipt = FinalTailReceipt.from_mapping(
                {**unsigned, "receipt_sha256": _sha256_json(unsigned)},
                plan=plan,
            )
            dependencies.journal.append(
                plan.sha256,
                "final_tail_captured",
                receipt.to_mapping(),
                now,
            )
            return receipt
        except BaseException as primary:
            try:
                _restore_legacy_gateway_after_freeze(
                    plan=plan,
                    approval=approval,
                    dependencies=dependencies,
                    entries=dependencies.journal.load(plan.sha256),
                    trigger="capture_failure",
                    now_unix=now,
                )
            except BaseException as recovery:
                raise ExceptionGroup(
                    "freeze capture failed and legacy gateway recovery was incomplete",
                    [primary, recovery],
                ) from None
            raise


def _require_bound_receipt(
    value: Mapping[str, Any],
    *,
    schema: str,
    plan: CutoverPlan,
    artifact_name: str,
) -> dict[str, Any]:
    required = frozenset({
        "schema", "plan_sha256", "artifact_sha256", "ok",
        "secret_material_recorded", "receipt_sha256",
    })
    raw = _hashed(value, required, "receipt_sha256", schema)
    if (
        raw["schema"] != schema
        or raw["plan_sha256"] != plan.sha256
        or raw["artifact_sha256"] != plan.value["artifacts"][artifact_name]["sha256"]
        or raw["ok"] is not True
        or raw["secret_material_recorded"] is not False
    ):
        raise ProductionCutoverError(f"{schema}_invalid")
    return raw


_CAPABILITY_PREREQUISITE_ACCEPTANCE_FIELDS = frozenset(
    {
        "schema",
        "plan_sha256",
        "production_owner_approval_sha256",
        "prerequisite_receipt_sha256",
        "prerequisite_file_sha256",
        "topology_identity_sha256",
        "boot_id_sha256",
        "pre_db_zero_write_observation",
        "pre_db_zero_write_observation_sha256",
        "isolated_canary_evidence_sha256",
        "workspace_gateway_receipt_sha256",
        "goal_continuation_terminal_schema",
        "goal_continuation_terminal_sha256",
        "canary_run_id",
        "canary_release_revision",
        "canary_fixture_sha256",
        "canary_capability_plan_sha256",
        "canary_full_canary_plan_sha256",
        "canary_owner_approval_receipt_sha256",
        "production_diff_sha256",
        "isolation_equivalence_projection",
        "isolation_equivalence_projection_sha256",
        "zero_canonical_database_mutation_observed",
        "ok",
        "secret_material_recorded",
        "secret_digest_recorded",
        "receipt_sha256",
    }
)


def _require_capability_prerequisite_acceptance(
    value: Mapping[str, Any],
    *,
    plan: CutoverPlan,
) -> dict[str, Any]:
    raw = _hashed(
        value,
        _CAPABILITY_PREREQUISITE_ACCEPTANCE_FIELDS,
        "receipt_sha256",
        "production capability prerequisite acceptance",
    )
    canary_evidence = _validate_isolated_canary_goal_prerequisite(
        plan.value["freeze_plan"]["cutover_authority"][
            "isolated_canary_goal_prerequisite"
        ],
        revision=plan.value["release_revision"],
    )
    expected_equivalence = _build_production_isolation_equivalence(
        plan=plan,
        evidence=canary_evidence,
    )
    pre_db_observation = _require_pre_db_zero_write_observation(
        raw["pre_db_zero_write_observation"],
        plan=plan,
    )
    if (
        raw["schema"] != CAPABILITY_PREREQUISITE_ACCEPTANCE_SCHEMA
        or raw["plan_sha256"] != plan.sha256
        or raw["production_owner_approval_sha256"]
        != plan.value["freeze_approval_sha256"]
        or _SHA256.fullmatch(str(raw["prerequisite_receipt_sha256"])) is None
        or _SHA256.fullmatch(str(raw["prerequisite_file_sha256"])) is None
        or raw["topology_identity_sha256"]
        != production_capability_topology_identity_sha256(
            plan.value["capability_topology"]
        )
        or _SHA256.fullmatch(str(raw["boot_id_sha256"])) is None
        or raw["pre_db_zero_write_observation_sha256"]
        != pre_db_observation["observation_sha256"]
        or raw["goal_continuation_terminal_schema"]
        != canary_evidence["goal_continuation_terminal_schema"]
        or raw["canary_release_revision"] != plan.value["release_revision"]
        or raw["isolated_canary_evidence_sha256"]
        != canary_evidence["evidence_sha256"]
        or raw["workspace_gateway_receipt_sha256"]
        != canary_evidence["workspace_gateway_receipt_sha256"]
        or raw["goal_continuation_terminal_sha256"]
        != canary_evidence["goal_continuation_terminal_sha256"]
        or raw["canary_run_id"] != canary_evidence["run_id"]
        or raw["canary_fixture_sha256"] != canary_evidence["fixture_sha256"]
        or raw["canary_capability_plan_sha256"]
        != canary_evidence["capability_plan_sha256"]
        or raw["canary_full_canary_plan_sha256"]
        != canary_evidence["full_canary_plan_sha256"]
        or raw["canary_owner_approval_receipt_sha256"]
        != canary_evidence["canary_owner_approval_receipt_sha256"]
        or raw["production_diff_sha256"]
        != canary_evidence["production_diff_sha256"]
        or any(
            _SHA256.fullmatch(str(raw[field])) is None
            for field in (
                "isolated_canary_evidence_sha256",
                "workspace_gateway_receipt_sha256",
                "goal_continuation_terminal_sha256",
                "canary_fixture_sha256",
                "canary_capability_plan_sha256",
                "canary_full_canary_plan_sha256",
                "canary_owner_approval_receipt_sha256",
                "production_diff_sha256",
                "isolation_equivalence_projection_sha256",
            )
        )
        or not isinstance(raw["canary_run_id"], str)
        or not raw["canary_run_id"]
        or not isinstance(raw["isolation_equivalence_projection"], Mapping)
        or set(raw["isolation_equivalence_projection"])
        != _ISOLATION_EQUIVALENCE_FIELDS
        or raw["isolation_equivalence_projection_sha256"]
        != raw["isolation_equivalence_projection"]["projection_sha256"]
        or raw["isolation_equivalence_projection"] != expected_equivalence
        or _sha256_json({
            key: item
            for key, item in raw["isolation_equivalence_projection"].items()
            if key != "projection_sha256"
        })
        != raw["isolation_equivalence_projection_sha256"]
        or raw["isolation_equivalence_projection"]["plan_sha256"]
        != plan.sha256
        or raw["isolation_equivalence_projection"]["equivalent"] is not True
        or raw["isolation_equivalence_projection"][
            "production_mutation_observed"
        ] is not False
        or raw["zero_canonical_database_mutation_observed"]
        is not (
            pre_db_observation["canonical_database_mutation_observed"]
            is False
        )
        or pre_db_observation[
            "staging_and_service_lifecycle_mutations_observed"
        ] is not True
        or raw["ok"] is not True
        or raw["secret_material_recorded"] is not False
        or raw["secret_digest_recorded"] is not False
    ):
        raise ProductionCutoverError(
            "production_capability_prerequisite_acceptance_invalid"
        )
    return raw


_OPERATIONAL_EDGE_RUNTIME_ROW_FIELDS = frozenset({
    "service", "process_uid", "process_gid", "socket_path", "socket_uid",
    "socket_gid", "socket_mode", "socket_device", "socket_inode", "ready",
})


def _require_operational_edge_runtime(
    value: Mapping[str, Any],
    *,
    plan: CutoverPlan,
    unit_file_state: str,
) -> dict[str, Any]:
    domains = set(CREDENTIALS_BY_DOMAIN)
    if (
        not isinstance(value, Mapping)
        or set(value) != domains
        or unit_file_state not in {"disabled", "enabled"}
    ):
        raise ProductionCutoverError(
            "production operational edge runtime invalid"
        )
    transition = plan.value["host_transition"]
    identity = transition["identity_foundation"]
    parsed: dict[str, Any] = {}
    for domain in sorted(domains):
        service_role = f"operational_edge_{domain}"
        socket_role = f"operational_edge_{domain}_client"
        service_uid = identity["users"][service_role]["uid"]
        service_gid = identity["groups"][service_role]["gid"]
        socket_gid = identity["groups"][socket_role]["gid"]
        row = value[domain]
        if (
            not isinstance(row, Mapping)
            or set(row) != _OPERATIONAL_EDGE_RUNTIME_ROW_FIELDS
        ):
            raise ProductionCutoverError(
                "production operational edge runtime invalid"
            )
        service = ServiceObservation.from_mapping(row["service"])
        unit = operational_edge_service_unit(domain)
        if (
            service.value["name"] != unit
            or service.value["fragment_path"]
            != f"/etc/systemd/system/{unit}"
            or service.value["fragment_sha256"]
            != transition["files"][f"operational_edge_unit_{domain}"][
                "sha256"
            ]
            or service.value["load_state"] != "loaded"
            or service.value["active_state"] != "active"
            or service.value["sub_state"] != "running"
            or service.value["unit_file_state"] != unit_file_state
            or service.value["main_pid"] <= 0
            or row["process_uid"] != service_uid
            or row["process_gid"] != service_gid
            or row["socket_path"]
            != f"/run/muncho-operational-edge/{domain}/edge.sock"
            or row["socket_uid"] != service_uid
            or row["socket_gid"] != socket_gid
            or row["socket_mode"] != "0660"
            or type(row["socket_device"]) is not int
            or row["socket_device"] <= 0
            or type(row["socket_inode"]) is not int
            or row["socket_inode"] <= 0
            or row["ready"] is not True
        ):
            raise ProductionCutoverError(
                "production operational edge runtime invalid"
            )
        parsed[domain] = {**dict(row), "service": service.to_mapping()}
    return parsed


def _require_operational_edge_readiness(
    value: Mapping[str, Any],
    *,
    plan: CutoverPlan,
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProductionCutoverError(
            "production operational edge readiness invalid"
        )
    try:
        parsed = validate_operational_edge_readiness(
            value,
            revision=plan.value["release_revision"],
            required_jobs=required_cron_operations(),
            expected_boot_id_sha256=str(value.get("boot_id_sha256") or ""),
            now_unix=value.get("observed_at_unix"),
        )
    except (OperationalEdgeReadinessError, TypeError, ValueError) as exc:
        raise ProductionCutoverError(
            "production operational edge readiness invalid"
        ) from exc
    for row in parsed["jobs"]:
        live = runtime.get(row["domain"])
        service = live.get("service") if isinstance(live, Mapping) else None
        if (
            not isinstance(live, Mapping)
            or not isinstance(service, Mapping)
            or row["service_unit"] != service["name"]
            or row["service_uid"] != live["process_uid"]
            or row["service_gid"] != live["process_gid"]
            or row["socket_path"] != live["socket_path"]
            or row["socket_uid"] != live["socket_uid"]
            or row["socket_gid"] != live["socket_gid"]
            or row["socket_mode"] != live["socket_mode"]
            or row["main_pid"] != service["main_pid"]
        ):
            raise ProductionCutoverError(
                "production operational edge readiness invalid"
            )
    return parsed


_HOST_APPLY_RECEIPT_FIELDS = frozenset({
    "schema", "plan_sha256", "artifact_sha256", "gateway_stopped",
    "writer_stopped", "connector_stopped", "phase_b_stopped",
    "routeback_stopped", "mac_ops_stopped", "browser_stopped",
    "isolated_worker_socket_stopped", "isolated_worker_service_stopped",
    "isolated_worker_lease_mountpoint_prepared",
    "identity_foundation_sha256", "identity_apply_receipt_sha256",
    "discord_key_foundation_sha256", "discord_foundation_receipt_sha256",
    "writer_public_key_id", "edge_public_key_id",
    "operational_edge_key_foundation_sha256",
    "operational_edge_key_provision_receipt_sha256",
    "operational_edge_receipt_public_key_ids",
    "operational_edge_staged_key_copies_retained",
    "operational_edge_keys_ready",
    "operational_edge_runtime",
    "operational_edge_asset_readback_receipt_sha256",
    "operational_edge_readiness",
    "only_operational_edge_services_started",
    "normal_services_remained_stopped",
    "discord_journals_clean_before_service_start",
    "normal_startup_journal_creation_disabled",
    "preexisting_memberships_converged_to_allowlist",
    "direct_discord_disabled",
    "discord_dm_allowed", "api_verifier_foundation_receipt_sha256",
    "api_bearer_verifier_installed", "api_approval_verifier_installed",
    "api_source_secrets_root_only", "api_source_secrets_loaded_by_gateway",
    "ok", "secret_material_recorded",
    "secret_digest_recorded", "receipt_sha256",
})


def _disabled_target_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    target = copy.deepcopy(dict(value))
    target["unit_file_state"] = "disabled"
    return target


def _require_host_apply_receipt(
    value: Mapping[str, Any],
    *,
    plan: CutoverPlan,
) -> dict[str, Any]:
    """Validate and retain the exact post-install stopped unit identities.

    A resumed process must compare the live host with the units installed by
    the already-journaled host transition.  Comparing against the legacy
    pre-install observations would reject every legitimate resume; comparing
    only fragment hashes would accept unit-file or drop-in drift.
    """

    raw = _hashed(
        value,
        _HOST_APPLY_RECEIPT_FIELDS,
        "receipt_sha256",
        "production host apply receipt",
    )
    gateway = ServiceObservation.from_mapping(raw["gateway_stopped"])
    writer = ServiceObservation.from_mapping(raw["writer_stopped"])
    connector = ServiceObservation.from_mapping(raw["connector_stopped"])
    phase_b = ServiceObservation.from_mapping(raw["phase_b_stopped"])
    routeback = ServiceObservation.from_mapping(raw["routeback_stopped"])
    mac_ops = ServiceObservation.from_mapping(raw["mac_ops_stopped"])
    browser = ServiceObservation.from_mapping(raw["browser_stopped"])
    worker_socket = ServiceObservation.from_mapping(
        raw["isolated_worker_socket_stopped"]
    )
    worker_service = ServiceObservation.from_mapping(
        raw["isolated_worker_service_stopped"]
    )
    topology = validate_production_capability_topology(
        plan.value["capability_topology"]
    )
    operational_runtime = _require_operational_edge_runtime(
        raw["operational_edge_runtime"],
        plan=plan,
        unit_file_state="disabled",
    )
    _require_operational_edge_readiness(
        raw["operational_edge_readiness"],
        plan=plan,
        runtime=operational_runtime,
    )
    if (
        raw["schema"] != "muncho-production-writer-host-apply.v2"
        or raw["plan_sha256"] != plan.sha256
        or raw["artifact_sha256"]
        != plan.value["artifacts"]["host_activation"]["sha256"]
        or gateway.value["name"] != GATEWAY_UNIT
        or writer.value["name"] != WRITER_UNIT
        or connector.value["name"] != CONNECTOR_UNIT
        or phase_b.value["name"] != PHASE_B_UNIT
        or routeback.value["name"] != ROUTEBACK_EDGE_UNIT
        or mac_ops.value["name"] != MAC_OPS_UNIT
        or browser.value["name"] != BROWSER_UNIT
        or worker_socket.value["name"] != ISOLATED_WORKER_SOCKET_UNIT
        or worker_service.value["name"] != ISOLATED_WORKER_SERVICE_UNIT
        or not gateway.stopped
        or not writer.stopped
        or not connector.stopped
        or not phase_b.stopped
        or not routeback.stopped
        or not mac_ops.stopped
        or not browser.stopped
        or not worker_socket.stopped
        or not worker_service.stopped
        or gateway.value["load_state"] != "loaded"
        or writer.value["load_state"] != "loaded"
        or connector.value["load_state"] != "loaded"
        or gateway.value["fragment_path"] != GATEWAY_FRAGMENT
        or writer.value["fragment_path"] != WRITER_FRAGMENT
        or connector.value["fragment_path"] != CONNECTOR_FRAGMENT
        or phase_b.value["fragment_path"] != PHASE_B_FRAGMENT
        or routeback.value["fragment_path"] != ROUTEBACK_EDGE_FRAGMENT
        or mac_ops.value["fragment_path"] != MAC_OPS_FRAGMENT
        or browser.value["fragment_path"] != BROWSER_FRAGMENT
        or worker_socket.value["fragment_path"]
        != ISOLATED_WORKER_SOCKET_FRAGMENT
        or worker_service.value["fragment_path"]
        != ISOLATED_WORKER_SERVICE_FRAGMENT
        or any(
            service.value["unit_file_state"] != "disabled"
            for service in (
                gateway, writer, connector, phase_b, routeback, mac_ops,
                browser,
                worker_socket,
            )
        )
        or worker_service.value["unit_file_state"] != "static"
        or gateway.stable_identity()
        != _disabled_target_identity(plan.value["gateway_target_identity"])
        or writer.stable_identity()
        != _disabled_target_identity(plan.value["writer_target_identity"])
        or connector.stable_identity()
        != _disabled_target_identity(plan.value["connector_target_identity"])
        or phase_b.value["fragment_sha256"]
        != topology["phase_b"]["fragment_sha256"]
        or routeback.value["fragment_sha256"]
        != topology["routeback_edge"]["fragment_sha256"]
        or mac_ops.value["fragment_sha256"]
        != topology["mac_ops"]["fragment_sha256"]
        or browser.value["fragment_sha256"]
        != topology["browser"]["fragment_sha256"]
        or worker_socket.value["fragment_sha256"]
        != topology["isolated_worker"]["socket_fragment_sha256"]
        or worker_service.value["fragment_sha256"]
        != topology["isolated_worker"]["service_fragment_sha256"]
        or raw["isolated_worker_lease_mountpoint_prepared"] is not True
        or raw["identity_foundation_sha256"]
        != plan.value["host_transition"]["identity_foundation"][
            "foundation_sha256"
        ]
        or _SHA256.fullmatch(str(raw["identity_apply_receipt_sha256"])) is None
        or raw["discord_key_foundation_sha256"]
        != plan.value["host_transition"]["discord_key_foundation"][
            "foundation_sha256"
        ]
        or _SHA256.fullmatch(
            str(raw["discord_foundation_receipt_sha256"])
        ) is None
        or raw["writer_public_key_id"]
        != plan.value["host_transition"]["discord_key_foundation"]["writer"][
            "public_key_id"
        ]
        or raw["edge_public_key_id"]
        != plan.value["host_transition"]["discord_key_foundation"]["edge"][
            "public_key_id"
        ]
        or raw["writer_public_key_id"] == raw["edge_public_key_id"]
        or raw["operational_edge_key_foundation_sha256"]
        != plan.value["host_transition"][
            "operational_edge_key_foundation_sha256"
        ]
        or _SHA256.fullmatch(
            str(raw["operational_edge_key_provision_receipt_sha256"])
        ) is None
        or raw["operational_edge_receipt_public_key_ids"]
        != plan.value["host_transition"][
            "operational_edge_receipt_public_key_ids"
        ]
        or raw["operational_edge_staged_key_copies_retained"] is not True
        or raw["operational_edge_keys_ready"] is not True
        or _SHA256.fullmatch(
            str(raw["operational_edge_asset_readback_receipt_sha256"])
        ) is None
        or raw["only_operational_edge_services_started"] is not True
        or raw["normal_services_remained_stopped"] is not True
        or raw["discord_journals_clean_before_service_start"] is not True
        or raw["normal_startup_journal_creation_disabled"] is not True
        or raw["preexisting_memberships_converged_to_allowlist"] is not True
        or raw["direct_discord_disabled"] is not True
        or raw["discord_dm_allowed"] is not False
        or _SHA256.fullmatch(
            str(raw["api_verifier_foundation_receipt_sha256"])
        ) is None
        or raw["api_bearer_verifier_installed"] is not True
        or raw["api_approval_verifier_installed"] is not True
        or raw["api_source_secrets_root_only"] is not True
        or raw["api_source_secrets_loaded_by_gateway"] is not False
        or raw["ok"] is not True
        or raw["secret_material_recorded"] is not False
        or raw["secret_digest_recorded"] is not False
    ):
        raise ProductionCutoverError(
            "muncho-production-writer-host-apply.v2_invalid"
        )
    return raw


_HOST_BOOT_COMMIT_RECEIPT_FIELDS = frozenset({
    "schema", "plan_sha256", "artifact_sha256", "gateway_observed",
    "writer_active", "connector_active", "phase_b_active",
    "routeback_active", "mac_ops_active", "browser_active",
    "isolated_worker_socket_active", "isolated_worker_service_active",
    "operational_edge_runtime",
    "boot_fence_released", "ok", "secret_material_recorded",
    "secret_digest_recorded", "receipt_sha256",
})


def _require_host_boot_commit_receipt(
    value: Mapping[str, Any],
    *,
    plan: CutoverPlan,
) -> dict[str, Any]:
    """Accept only the final, plan-bound release of the reboot fence."""

    raw = _hashed(
        value,
        _HOST_BOOT_COMMIT_RECEIPT_FIELDS,
        "receipt_sha256",
        "production host boot commit receipt",
    )
    gateway = ServiceObservation.from_mapping(raw["gateway_observed"])
    writer = ServiceObservation.from_mapping(raw["writer_active"])
    connector = ServiceObservation.from_mapping(raw["connector_active"])
    phase_b = ServiceObservation.from_mapping(raw["phase_b_active"])
    routeback = ServiceObservation.from_mapping(raw["routeback_active"])
    mac_ops = ServiceObservation.from_mapping(raw["mac_ops_active"])
    browser = ServiceObservation.from_mapping(raw["browser_active"])
    worker_socket = ServiceObservation.from_mapping(
        raw["isolated_worker_socket_active"]
    )
    worker_service = ServiceObservation.from_mapping(
        raw["isolated_worker_service_active"]
    )
    topology = validate_production_capability_topology(
        plan.value["capability_topology"]
    )
    _require_operational_edge_runtime(
        raw["operational_edge_runtime"],
        plan=plan,
        unit_file_state="enabled",
    )
    running = (
        writer, connector, routeback, mac_ops, browser, worker_service,
    )
    if (
        raw["schema"] != "muncho-production-host-boot-commit.v2"
        or raw["plan_sha256"] != plan.sha256
        or raw["artifact_sha256"]
        != plan.value["artifacts"]["host_activation"]["sha256"]
        or [
            item.value["name"]
            for item in (
                gateway, writer, connector, phase_b, routeback, mac_ops,
                browser,
                worker_socket,
                worker_service,
            )
        ]
        != [
            GATEWAY_UNIT, WRITER_UNIT, CONNECTOR_UNIT, PHASE_B_UNIT,
            ROUTEBACK_EDGE_UNIT, MAC_OPS_UNIT, BROWSER_UNIT,
            ISOLATED_WORKER_SOCKET_UNIT, ISOLATED_WORKER_SERVICE_UNIT,
        ]
        or any(
            item.value["active_state"] != "active"
            or item.value["sub_state"] != "running"
            or item.value["main_pid"] <= 0
            for item in running
        )
        or not (
            gateway.stopped
            or (
                gateway.value["active_state"] == "active"
                and gateway.value["sub_state"] == "running"
                and gateway.value["main_pid"] > 0
            )
        )
        or phase_b.value["active_state"] != "active"
        or phase_b.value["sub_state"] != "exited"
        or phase_b.value["main_pid"] != 0
        or any(
            item.value["unit_file_state"] != "enabled"
            for item in (
                gateway, writer, connector, phase_b, routeback, mac_ops,
                browser, worker_socket,
            )
        )
        or worker_service.value["unit_file_state"] != "static"
        or gateway.stable_identity() != plan.value["gateway_target_identity"]
        or writer.stable_identity() != plan.value["writer_target_identity"]
        or connector.stable_identity()
        != plan.value["connector_target_identity"]
        or phase_b.value["fragment_sha256"]
        != topology["phase_b"]["fragment_sha256"]
        or routeback.value["fragment_sha256"]
        != topology["routeback_edge"]["fragment_sha256"]
        or mac_ops.value["fragment_sha256"]
        != topology["mac_ops"]["fragment_sha256"]
        or browser.value["fragment_sha256"]
        != topology["browser"]["fragment_sha256"]
        or worker_socket.value["active_state"] != "active"
        or worker_socket.value["sub_state"] != "listening"
        or worker_socket.value["main_pid"] != 0
        or worker_socket.value["fragment_sha256"]
        != topology["isolated_worker"]["socket_fragment_sha256"]
        or worker_service.value["fragment_sha256"]
        != topology["isolated_worker"]["service_fragment_sha256"]
        or raw["boot_fence_released"] is not True
        or raw["ok"] is not True
        or raw["secret_material_recorded"] is not False
        or raw["secret_digest_recorded"] is not False
    ):
        raise ProductionCutoverError(
            "muncho-production-host-boot-commit.v2_invalid"
        )
    return raw


_HOST_ROLLBACK_RECEIPT_FIELDS = frozenset({
    "schema", "plan_sha256", "artifact_sha256",
    "api_verifiers_removed", "api_source_secrets_preserved", "ok",
    "phase_b_removed", "routeback_removed", "mac_ops_removed",
    "browser_removed",
    "isolated_worker_socket_removed", "isolated_worker_service_removed",
    "isolated_worker_lease_mountpoint_restored",
    "identity_foundation_sha256", "identity_rollback_receipt_sha256",
    "discord_key_rollback_receipt_sha256",
    "operational_edge_key_retention_receipt_sha256",
    "operational_edge_keys_retained_dormant",
    "operational_edge_staged_key_copies_retained",
    "operational_edge_units_removed",
    "operational_edge_readiness_removed",
    "discord_journal_rollback_receipt_sha256",
    "staged_private_keys_restored", "discord_journals_removed",
    "created_identities_retained_dormant",
    "preexisting_memberships_restored",
    "secret_material_recorded", "secret_digest_recorded", "receipt_sha256",
})


def _require_host_rollback_receipt(
    value: Mapping[str, Any],
    *,
    plan: CutoverPlan,
) -> dict[str, Any]:
    raw = _hashed(
        value,
        _HOST_ROLLBACK_RECEIPT_FIELDS,
        "receipt_sha256",
        "production host rollback receipt",
    )
    if (
        raw["schema"] != "muncho-production-writer-host-rollback.v2"
        or raw["plan_sha256"] != plan.sha256
        or raw["artifact_sha256"]
        != plan.value["artifacts"]["host_rollback"]["sha256"]
        or raw["api_verifiers_removed"] is not True
        or raw["api_source_secrets_preserved"] is not True
        or raw["phase_b_removed"] is not True
        or raw["routeback_removed"] is not True
        or raw["mac_ops_removed"] is not True
        or raw["browser_removed"] is not True
        or raw["isolated_worker_socket_removed"] is not True
        or raw["isolated_worker_service_removed"] is not True
        or raw["isolated_worker_lease_mountpoint_restored"] is not True
        or raw["identity_foundation_sha256"]
        != plan.value["host_transition"]["identity_foundation"][
            "foundation_sha256"
        ]
        or _SHA256.fullmatch(str(raw["identity_rollback_receipt_sha256"])) is None
        or _SHA256.fullmatch(
            str(raw["discord_key_rollback_receipt_sha256"])
        ) is None
        or _SHA256.fullmatch(
            str(raw["discord_journal_rollback_receipt_sha256"])
        ) is None
        or _SHA256.fullmatch(
            str(raw["operational_edge_key_retention_receipt_sha256"])
        ) is None
        or raw["operational_edge_keys_retained_dormant"] is not True
        or raw["operational_edge_staged_key_copies_retained"] is not True
        or raw["operational_edge_units_removed"] != [
            operational_edge_service_unit(domain)
            for domain in sorted(CREDENTIALS_BY_DOMAIN)
        ]
        or raw["operational_edge_readiness_removed"] is not True
        or raw["staged_private_keys_restored"] is not True
        or raw["discord_journals_removed"] is not True
        or raw["created_identities_retained_dormant"] is not True
        or raw["preexisting_memberships_restored"] is not True
        or raw["ok"] is not True
        or raw["secret_material_recorded"] is not False
        or raw["secret_digest_recorded"] is not False
    ):
        raise ProductionCutoverError(
            "muncho-production-writer-host-rollback.v2_invalid"
        )
    return raw


def _require_database_receipt(
    value: Mapping[str, Any],
    *,
    schema: str,
    plan: CutoverPlan,
    artifact_name: str,
) -> dict[str, Any]:
    """Validate the lossless row/digest proof emitted by the sealed DB edge."""

    required = frozenset({
        "schema", "plan_sha256", "artifact_sha256", "final_snapshot_sha256",
        "source_row_count", "archive_row_count", "canonical_row_count",
        "archive_extended19_sha256", "canonical14_sha256",
        "relation_identity_sha256", "acl_identity_sha256",
        "index_identity_sha256", "roles_acl_sha256",
        "zero_canonical_writer_writes", "legacy_shape_restored", "ok",
        "legacy_truth_mode", "legacy_truth_decision_sha256",
        "legacy_truth_decision_event_id", "accepted_event_set_sha256",
        "trusted_legacy_event_count", "truth_epoch_sha256",
        "secret_material_recorded", "receipt_sha256",
    })
    raw = _hashed(value, required, "receipt_sha256", schema)
    snapshot = plan.final_snapshot.value
    decision = _legacy_truth_decision(plan.value["legacy_truth_decision"])
    legacy_state = schema in {
        "muncho-production-legacy-cutover-preflight.v1",
        "muncho-production-legacy-database-rollback.v1",
    }
    accepted_ids = decision["accepted_event_ids"]
    accepted_set_sha256 = _sha256_json(accepted_ids)
    trusted_count = 0 if legacy_state else len(accepted_ids)
    if (
        raw["schema"] != schema
        or raw["plan_sha256"] != plan.sha256
        or raw["artifact_sha256"] != plan.value["artifacts"][artifact_name]["sha256"]
        or raw["final_snapshot_sha256"] != plan.final_snapshot.sha256
        or raw["source_row_count"] != snapshot["source_row_count"]
        or raw["archive_row_count"] != snapshot["source_row_count"]
        or raw["canonical_row_count"]
        != (0 if legacy_state else snapshot["source_row_count"] + 1)
        or raw["archive_extended19_sha256"] != snapshot["extended19_sha256"]
        or raw["canonical14_sha256"] != snapshot["canonical14_sha256"]
        or raw["relation_identity_sha256"] != snapshot["relation_identity_sha256"]
        or raw["acl_identity_sha256"] != snapshot["acl_identity_sha256"]
        or raw["index_identity_sha256"] != snapshot["index_identity_sha256"]
        or _SHA256.fullmatch(str(raw["roles_acl_sha256"])) is None
        or raw["zero_canonical_writer_writes"] is not True
        or raw["legacy_shape_restored"] is not legacy_state
        or raw["legacy_truth_mode"] != decision["mode"]
        or raw["legacy_truth_decision_sha256"] != decision["decision_sha256"]
        or raw["legacy_truth_decision_event_id"]
        != decision["decision_event_id"]
        or raw["accepted_event_set_sha256"] != accepted_set_sha256
        or raw["trusted_legacy_event_count"] != trusted_count
        or raw["truth_epoch_sha256"] != decision["truth_epoch_sha256"]
        or raw["ok"] is not True
        or raw["secret_material_recorded"] is not False
    ):
        raise ProductionCutoverError(f"{schema}_invalid")
    return raw


_ROLLBACK_TERMINAL_FIELDS = frozenset({
    "schema", "plan_sha256", "approval_sha256",
    "database_restore_required", "database_restored",
    "database_rollback_receipt_sha256", "host_restore_required",
    "host_restored", "host_rollback_receipt_sha256", "legacy_gateway",
    "legacy_writer", "legacy_connector", "secret_material_recorded",
    "legacy_gateway_restarted", "direct_discord_disabled",
    "discord_dm_allowed", "completed_at_unix",
    "receipt_sha256",
})


def _validate_rollback_terminal(
    value: Mapping[str, Any],
    *,
    plan: CutoverPlan,
    entries: list[JournalEntry] | None = None,
) -> dict[str, Any]:
    raw = _hashed(
        value,
        _ROLLBACK_TERMINAL_FIELDS,
        "receipt_sha256",
        "cutover rollback terminal",
    )
    gateway = ServiceObservation.from_mapping(raw["legacy_gateway"])
    writer = ServiceObservation.from_mapping(raw["legacy_writer"])
    connector = ServiceObservation.from_mapping(raw["legacy_connector"])
    expected_gateway = ServiceObservation.from_mapping(
        plan.value["gateway_legacy_identity"]
    )
    expected_writer = ServiceObservation.from_mapping(
        plan.value["writer_pre_identity"]
    )
    expected_connector = ServiceObservation.from_mapping(
        plan.value["connector_pre_identity"]
    )
    database_required = raw["database_restore_required"]
    host_required = raw["host_restore_required"]
    if (
        raw["schema"] != ROLLBACK_TERMINAL_SCHEMA
        or raw["plan_sha256"] != plan.sha256
        or _SHA256.fullmatch(str(raw["approval_sha256"])) is None
        or type(database_required) is not bool
        or raw["database_restored"] is not database_required
        or (
            database_required
            != (raw["database_rollback_receipt_sha256"] is not None)
        )
        or (
            raw["database_rollback_receipt_sha256"] is not None
            and _SHA256.fullmatch(
                str(raw["database_rollback_receipt_sha256"])
            )
            is None
        )
        or type(host_required) is not bool
        or raw["host_restored"] is not host_required
        or host_required != (raw["host_rollback_receipt_sha256"] is not None)
        or (
            raw["host_rollback_receipt_sha256"] is not None
            and _SHA256.fullmatch(str(raw["host_rollback_receipt_sha256"]))
            is None
        )
        or gateway.value["name"] != GATEWAY_UNIT
        or writer.value["name"] != WRITER_UNIT
        or connector.value["name"] != CONNECTOR_UNIT
        or gateway.value["active_state"] != "active"
        or gateway.value["sub_state"] != "running"
        or gateway.value["main_pid"] <= 0
        or not writer.stopped
        or not connector.stopped
        or gateway.stable_identity() != expected_gateway.stable_identity()
        or writer.stable_identity() != expected_writer.stable_identity()
        or connector.stable_identity() != expected_connector.stable_identity()
        or raw["legacy_gateway_restarted"] is not True
        or raw["direct_discord_disabled"] is not False
        or raw["discord_dm_allowed"] is not False
        or raw["secret_material_recorded"] is not False
        or type(raw["completed_at_unix"]) is not int
        or raw["completed_at_unix"] <= 0
    ):
        raise ProductionCutoverError("production_cutover_rollback_terminal_invalid")
    if entries is not None:
        authority = _last(entries, "authority")
        database_entry = _last(entries, "database_rolled_back")
        host_entry = _last(entries, "host_rolled_back")
        if (
            authority is None
            or authority.value["evidence"].get("approval_sha256")
            != raw["approval_sha256"]
            or (database_entry is not None) is not database_required
            or (host_entry is not None) is not host_required
            or (
                database_entry is not None
                and database_entry.value["evidence"].get("receipt_sha256")
                != raw["database_rollback_receipt_sha256"]
            )
            or (
                host_entry is not None
                and host_entry.value["evidence"].get("receipt_sha256")
                != raw["host_rollback_receipt_sha256"]
            )
        ):
            raise ProductionCutoverError(
                "production_cutover_rollback_terminal_lineage_invalid"
            )
    return raw


def _accepted_database_apply(
    entries: list[JournalEntry],
    plan: CutoverPlan,
) -> Mapping[str, Any] | None:
    entry = _last(entries, "database_applied")
    if entry is None:
        return None
    return _require_database_receipt(
        entry.value["evidence"],
        schema="muncho-production-legacy-database-apply.v1",
        plan=plan,
        artifact_name="database_apply",
    )


def _accepted_host_apply(
    entries: list[JournalEntry],
    plan: CutoverPlan,
) -> Mapping[str, Any] | None:
    entry = _last(entries, "host_applied")
    if entry is None:
        return None
    return _require_host_apply_receipt(entry.value["evidence"], plan=plan)


def _accepted_host_boot_commit(
    entries: list[JournalEntry],
    plan: CutoverPlan,
) -> Mapping[str, Any] | None:
    entry = _last(entries, "boot_committed")
    if entry is None:
        return None
    return _require_host_boot_commit_receipt(
        entry.value["evidence"], plan=plan
    )


def _accepted_database_terminal(
    entries: list[JournalEntry],
    plan: CutoverPlan,
) -> Mapping[str, Any] | None:
    entry = _last(entries, "database_terminal_validated")
    if entry is None:
        return None
    return _require_database_receipt(
        entry.value["evidence"],
        schema="muncho-production-legacy-cutover-postflight.v1",
        plan=plan,
        artifact_name="database_postflight",
    )


def _uses_packaged_cron_cutover(plan: CutoverPlan) -> bool:
    return (
        plan.value["cron_continuity_plan"].get("schema")
        == production_cron_continuity_package.PLAN_SCHEMA
    )


def _require_cron_cutover_receipt(
    value: Mapping[str, Any],
    *,
    action: str,
    plan: CutoverPlan,
    expected_prior_sha256: str | None = None,
) -> dict[str, Any]:
    from gateway import production_cron_cutover_runtime as cron_runtime

    try:
        trusted = cron_runtime.validate_cutover_receipt(
            value,
            action=action,
            plan_sha256=plan.sha256,
            expected_prior_sha256=expected_prior_sha256,
        )
    except cron_runtime.ProductionCronCutoverRuntimeError as exc:
        raise ProductionCutoverError(
            "production_cron_cutover_receipt_invalid"
        ) from exc
    continuity = plan.value["cron_continuity_plan"]
    if (
        trusted["continuity_plan_sha256"] != continuity["plan_sha256"]
        or trusted["source_store_sha256"]
        != continuity["source_store_sha256"]
        or action == "rollback"
        and trusted["apply_receipt_sha256"] != expected_prior_sha256
    ):
        raise ProductionCutoverError(
            "production_cron_cutover_receipt_invalid"
        )
    return trusted


@dataclass(frozen=True)
class _CronCutoverState:
    preflight: Mapping[str, Any] | None = None
    apply: Mapping[str, Any] | None = None
    postflight: Mapping[str, Any] | None = None
    activation_authority: Mapping[str, Any] | None = None
    activation: Mapping[str, Any] | None = None
    rollback: Mapping[str, Any] | None = None


def _build_cron_activation_authority(
    *,
    plan: CutoverPlan,
    postflight_receipt: Mapping[str, Any],
    entries: list[JournalEntry],
) -> dict[str, Any]:
    from gateway import production_cron_cutover_runtime as cron_runtime

    required = {
        name: _last(entries, name)
        for name in (
            "database_terminal_validated",
            "activation_commit_intent",
            "boot_committed",
            "gateway_started",
        )
    }
    if any(entry is None for entry in required.values()):
        raise ProductionCutoverError(
            "production_cron_activation_lineage_missing"
        )
    try:
        return cron_runtime.build_activation_authority(
            cutover_plan_sha256=plan.sha256,
            cron_postflight_receipt_sha256=str(
                postflight_receipt["receipt_sha256"]
            ),
            database_terminal_entry_sha256=required[
                "database_terminal_validated"
            ].sha256,
            activation_commit_intent_entry_sha256=required[
                "activation_commit_intent"
            ].sha256,
            boot_committed_entry_sha256=required["boot_committed"].sha256,
            gateway_started_entry_sha256=required["gateway_started"].sha256,
        )
    except cron_runtime.ProductionCronCutoverRuntimeError as exc:
        raise ProductionCutoverError(
            "production_cron_activation_authority_invalid"
        ) from exc


def _accepted_cron_cutover_state(
    entries: list[JournalEntry],
    plan: CutoverPlan,
) -> _CronCutoverState:
    if not _uses_packaged_cron_cutover(plan):
        return _CronCutoverState()

    def evidence(event: str) -> Mapping[str, Any] | None:
        entry = _last(entries, event)
        return None if entry is None else entry.value["evidence"]

    preflight_value = evidence("cron_preflight_validated")
    if (
        preflight_value is not None
        and _last(entries, "cron_preflight_started") is None
    ):
        raise ProductionCutoverError("production_cron_cutover_lineage_invalid")
    preflight = (
        None
        if preflight_value is None
        else _require_cron_cutover_receipt(
            preflight_value,
            action="preflight",
            plan=plan,
        )
    )
    apply_value = evidence("cron_applied")
    if apply_value is not None and (
        preflight is None or _last(entries, "cron_apply_started") is None
    ):
        raise ProductionCutoverError("production_cron_cutover_lineage_invalid")
    applied = (
        None
        if apply_value is None
        else _require_cron_cutover_receipt(
            apply_value,
            action="apply",
            plan=plan,
            expected_prior_sha256=str(preflight["receipt_sha256"]),
        )
    )
    postflight_value = evidence("cron_postflight_validated")
    if postflight_value is not None and (
        applied is None or _last(entries, "cron_postflight_started") is None
    ):
        raise ProductionCutoverError("production_cron_cutover_lineage_invalid")
    postflight = (
        None
        if postflight_value is None
        else _require_cron_cutover_receipt(
            postflight_value,
            action="postflight",
            plan=plan,
            expected_prior_sha256=str(applied["receipt_sha256"]),
        )
    )

    authority_value = evidence("cron_activation_authority")
    authority: Mapping[str, Any] | None = None
    if authority_value is not None:
        if postflight is None:
            raise ProductionCutoverError(
                "production_cron_activation_lineage_invalid"
            )
        expected = _build_cron_activation_authority(
            plan=plan,
            postflight_receipt=postflight,
            entries=entries,
        )
        if authority_value != expected:
            raise ProductionCutoverError(
                "production_cron_activation_authority_invalid"
            )
        authority = expected

    activation_value = evidence("cron_activated")
    if activation_value is not None and (
        postflight is None
        or authority is None
        or _last(entries, "cron_activation_started") is None
    ):
        raise ProductionCutoverError(
            "production_cron_activation_lineage_invalid"
        )
    activated = (
        None
        if activation_value is None
        else _require_cron_cutover_receipt(
            activation_value,
            action="activation",
            plan=plan,
            expected_prior_sha256=str(postflight["receipt_sha256"]),
        )
    )
    if (
        activated is not None
        and activated["activation_authority_sha256"]
        != authority["authority_sha256"]
    ):
        raise ProductionCutoverError(
            "production_cron_activation_lineage_invalid"
        )

    rollback_value = evidence("cron_rolled_back")
    if (
        rollback_value is not None
        and _last(entries, "cron_rollback_started") is None
    ):
        raise ProductionCutoverError("production_cron_cutover_lineage_invalid")
    rolled_back = (
        None
        if rollback_value is None
        else _require_cron_cutover_receipt(
            rollback_value,
            action="rollback",
            plan=plan,
            expected_prior_sha256=(
                None
                if applied is None
                else str(applied["receipt_sha256"])
            ),
        )
    )
    if activated is not None and rolled_back is not None:
        raise ProductionCutoverError("production_cron_cutover_lineage_invalid")
    return _CronCutoverState(
        preflight=preflight,
        apply=applied,
        postflight=postflight,
        activation_authority=authority,
        activation=activated,
        rollback=rolled_back,
    )


_ACTIVATION_COMMIT_INTENT_FIELDS = frozenset({
    "schema", "plan_sha256", "approval_sha256",
    "host_apply_receipt_sha256", "database_apply_receipt_sha256",
    "database_postflight_receipt_sha256",
    "prerequisite_start_receipt_sha256", "writer_start_receipt_sha256",
    "capability_prerequisite_receipt_sha256",
    "gateway_stopped", "connector_active", "forward_only",
    "secret_material_recorded", "receipt_sha256",
})


def _require_activation_commit_intent(
    value: Mapping[str, Any],
    *,
    plan: CutoverPlan,
    entries: list[JournalEntry],
) -> dict[str, Any]:
    raw = _hashed(
        value,
        _ACTIVATION_COMMIT_INTENT_FIELDS,
        "receipt_sha256",
        "production activation commit intent",
    )
    host = _accepted_host_apply(entries, plan)
    database = _accepted_database_apply(entries, plan)
    terminal = _accepted_database_terminal(entries, plan)
    prerequisite_entry = _last(entries, "prerequisites_started")
    writer_entry = _last(entries, "writer_started")
    capability_entry = _last(entries, "capability_prerequisites_validated")
    authorities = [
        entry.value["evidence"].get("approval_sha256")
        for entry in entries
        if entry.value["event"] == "authority"
    ]
    prerequisite = (
        None
        if prerequisite_entry is None
        else _require_bound_receipt(
            prerequisite_entry.value["evidence"],
            schema="muncho-production-prerequisite-services-start.v1",
            plan=plan,
            artifact_name="host_activation",
        )
    )
    writer = (
        None
        if writer_entry is None
        else _require_bound_receipt(
            writer_entry.value["evidence"],
            schema="muncho-production-writer-start.v1",
            plan=plan,
            artifact_name="host_activation",
        )
    )
    capability = (
        None
        if capability_entry is None
        else _require_capability_prerequisite_acceptance(
            capability_entry.value["evidence"], plan=plan
        )
    )
    if (
        raw["schema"] != "muncho-production-activation-commit-intent.v1"
        or raw["plan_sha256"] != plan.sha256
        or raw["approval_sha256"] not in authorities
        or host is None
        or database is None
        or terminal is None
        or prerequisite is None
        or writer is None
        or capability is None
        or raw["host_apply_receipt_sha256"] != host["receipt_sha256"]
        or raw["database_apply_receipt_sha256"]
        != database["receipt_sha256"]
        or raw["database_postflight_receipt_sha256"]
        != terminal["receipt_sha256"]
        or raw["prerequisite_start_receipt_sha256"]
        != prerequisite["receipt_sha256"]
        or raw["writer_start_receipt_sha256"] != writer["receipt_sha256"]
        or raw["capability_prerequisite_receipt_sha256"]
        != capability["receipt_sha256"]
        or raw["gateway_stopped"] is not True
        or raw["connector_active"] is not True
        or raw["forward_only"] is not True
        or raw["secret_material_recorded"] is not False
    ):
        raise ProductionCutoverError(
            "production_activation_commit_intent_invalid"
        )
    return raw


def _accepted_activation_commit_intent(
    entries: list[JournalEntry],
    plan: CutoverPlan,
) -> Mapping[str, Any] | None:
    entry = _last(entries, "activation_commit_intent")
    if entry is None:
        return None
    return _require_activation_commit_intent(
        entry.value["evidence"], plan=plan, entries=entries
    )


def _resume_requires_rollback(entries: list[JournalEntry]) -> bool:
    """Return whether a prior process crossed an unreceipted mutation edge."""

    if _last(entries, "host_rolled_back") is not None:
        return True
    if _last(entries, "database_rolled_back") is not None:
        return True
    if (
        _last(entries, "database_apply_started") is not None
        and _last(entries, "database_applied") is None
    ):
        return True
    if (
        _last(entries, "host_apply_started") is not None
        and _last(entries, "host_applied") is None
    ):
        return True
    if (
        _last(entries, "cron_apply_started") is not None
        and _last(entries, "cron_applied") is None
    ):
        return True
    if (
        _last(entries, "prerequisites_start_started") is not None
        and _last(entries, "prerequisites_started") is None
    ):
        return True
    if (
        _last(entries, "writer_start_started") is not None
        and _last(entries, "writer_started") is None
    ):
        return True
    if (
        _last(entries, "gateway_start_started") is not None
        and _last(entries, "gateway_started") is None
    ):
        return True
    return (
        _last(entries, "boot_commit_started") is not None
        and _last(entries, "boot_committed") is None
    )


def _mutation_was_attempted(entries: list[JournalEntry]) -> bool:
    return any(
        _last(entries, event) is not None
        for event in (
            "database_apply_started",
            "database_applied",
            "host_apply_started",
            "host_applied",
            "cron_apply_started",
            "cron_applied",
            "cron_postflight_started",
            "cron_postflight_validated",
            "prerequisites_start_started",
            "prerequisites_started",
            "writer_start_started",
            "writer_started",
            "gateway_start_started",
            "gateway_started",
            "boot_commit_started",
            "boot_committed",
            "host_rolled_back",
            "database_rolled_back",
        )
    )


def _rollback_cutover(
    *,
    plan: CutoverPlan,
    approval: CutoverApproval,
    dependencies: CutoverDependencies,
    entries: list[JournalEntry],
    database_receipt: Mapping[str, Any] | None,
    host_receipt: Mapping[str, Any] | None,
    now_unix: int,
) -> Mapping[str, Any]:
    """Idempotently converge an attempted cutover back to exact legacy state.

    Mutation-intent journal entries are authority to *reconcile*, never to
    replay forward.  Both rollback boundaries must therefore tolerate a
    missing apply receipt: the apply may have committed or installed its host
    artifacts before the caller lost the response.
    """

    database_required = (
        _last(entries, "database_apply_started") is not None
        or _last(entries, "database_applied") is not None
        or _last(entries, "database_rolled_back") is not None
    )
    host_required = (
        _last(entries, "host_apply_started") is not None
        or _last(entries, "host_applied") is not None
        or _last(entries, "prerequisites_start_started") is not None
        or _last(entries, "prerequisites_started") is not None
        or _last(entries, "writer_start_started") is not None
        or _last(entries, "writer_started") is not None
        or _last(entries, "gateway_start_started") is not None
        or _last(entries, "gateway_started") is not None
        or _last(entries, "boot_commit_started") is not None
        or _last(entries, "boot_committed") is not None
        or _last(entries, "host_rolled_back") is not None
    )
    cron_required = _uses_packaged_cron_cutover(plan) and any(
        _last(entries, event) is not None
        for event in (
            "cron_apply_started",
            "cron_applied",
            "cron_postflight_started",
            "cron_postflight_validated",
            "cron_rollback_started",
            "cron_rolled_back",
        )
    )
    rollback_errors: list[BaseException] = []

    for stop in (
        dependencies.services.stop_gateway,
        dependencies.services.stop_writer,
        dependencies.services.stop_connector,
    ):
        try:
            stop()
        except BaseException as exc:
            rollback_errors.append(exc)
    if not rollback_errors:
        try:
            if (
                not dependencies.services.observe_gateway().stopped
                or not dependencies.services.observe_writer().stopped
                or not dependencies.services.observe_connector().stopped
            ):
                raise ProductionCutoverError(
                    "production_rollback_services_not_stopped"
                )
        except BaseException as exc:
            rollback_errors.append(exc)

    cron_rollback: Mapping[str, Any] | None = None
    host_rollback: Mapping[str, Any] | None = None
    database_rollback: Mapping[str, Any] | None = None
    # Cron owns jobs.json and its packaged unit files.  Restore those first,
    # while the exact owner-bound service identity and release still exist;
    # host/database rollback may remove prerequisites needed by this edge.
    if not rollback_errors and cron_required:
        prior = _last(entries, "cron_rolled_back")
        try:
            cron_state = _accepted_cron_cutover_state(entries, plan)
            if prior is None:
                if dependencies.cron is None:
                    raise ProductionCutoverError(
                        "production_cron_cutover_boundary_missing"
                    )
                if _last(entries, "cron_rollback_started") is None:
                    dependencies.journal.append(
                        plan.sha256,
                        "cron_rollback_started",
                        {
                            "apply_receipt_sha256": (
                                None
                                if cron_state.apply is None
                                else cron_state.apply["receipt_sha256"]
                            )
                        },
                        now_unix,
                    )
                cron_rollback = _require_cron_cutover_receipt(
                    dependencies.cron.rollback(plan, cron_state.apply),
                    action="rollback",
                    plan=plan,
                    expected_prior_sha256=(
                        None
                        if cron_state.apply is None
                        else str(cron_state.apply["receipt_sha256"])
                    ),
                )
                dependencies.journal.append(
                    plan.sha256,
                    "cron_rolled_back",
                    cron_rollback,
                    now_unix,
                )
                entries = dependencies.journal.load(plan.sha256)
            elif cron_state.rollback is None:
                raise ProductionCutoverError(
                    "production_cron_cutover_rollback_invalid"
                )
            else:
                cron_rollback = cron_state.rollback
        except BaseException as exc:
            rollback_errors.append(exc)

    if not rollback_errors and host_required:
        prior = _last(entries, "host_rolled_back")
        if prior is None:
            try:
                host_rollback = _require_host_rollback_receipt(
                    dependencies.host.rollback(plan, host_receipt),
                    plan=plan,
                )
                dependencies.journal.append(
                    plan.sha256, "host_rolled_back", host_rollback, now_unix
                )
                entries = dependencies.journal.load(plan.sha256)
            except BaseException as exc:
                rollback_errors.append(exc)
        else:
            try:
                host_rollback = _require_host_rollback_receipt(
                    prior.value["evidence"],
                    plan=plan,
                )
            except BaseException as exc:
                rollback_errors.append(exc)

    if not rollback_errors and database_required:
        prior = _last(entries, "database_rolled_back")
        if prior is None:
            try:
                database_rollback = _require_database_receipt(
                    dependencies.database.rollback(plan, database_receipt),
                    schema="muncho-production-legacy-database-rollback.v1",
                    plan=plan,
                    artifact_name="database_rollback",
                )
                dependencies.journal.append(
                    plan.sha256,
                    "database_rolled_back",
                    database_rollback,
                    now_unix,
                )
                entries = dependencies.journal.load(plan.sha256)
            except BaseException as exc:
                rollback_errors.append(exc)
        else:
            try:
                database_rollback = _require_database_receipt(
                    prior.value["evidence"],
                    schema="muncho-production-legacy-database-rollback.v1",
                    plan=plan,
                    artifact_name="database_rollback",
                )
            except BaseException as exc:
                rollback_errors.append(exc)

    if rollback_errors:
        raise ExceptionGroup("cutover rollback was incomplete", rollback_errors)

    dependencies.services.start_gateway()
    gateway = dependencies.services.observe_gateway()
    writer = dependencies.services.observe_writer()
    connector = dependencies.services.observe_connector()
    expected_gateway = ServiceObservation.from_mapping(
        plan.value["gateway_legacy_identity"]
    )
    expected_writer = ServiceObservation.from_mapping(
        plan.value["writer_pre_identity"]
    )
    expected_connector = ServiceObservation.from_mapping(
        plan.value["connector_pre_identity"]
    )
    if (
        gateway.value["active_state"] != "active"
        or gateway.value["sub_state"] != "running"
        or gateway.value["main_pid"] <= 0
        or not writer.stopped
        or not connector.stopped
        or gateway.stable_identity() != expected_gateway.stable_identity()
        or writer.stable_identity() != expected_writer.stable_identity()
        or connector.stable_identity() != expected_connector.stable_identity()
    ):
        raise ProductionCutoverError("production_rollback_legacy_state_invalid")
    unsigned = {
        "schema": ROLLBACK_TERMINAL_SCHEMA,
        "plan_sha256": plan.sha256,
        "approval_sha256": approval.sha256,
        "database_restore_required": database_required,
        "database_restored": database_required,
        "database_rollback_receipt_sha256": (
            None
            if database_rollback is None
            else database_rollback["receipt_sha256"]
        ),
        "host_restore_required": host_required,
        "host_restored": host_required,
        "host_rollback_receipt_sha256": (
            None if host_rollback is None else host_rollback["receipt_sha256"]
        ),
        "legacy_gateway": gateway.to_mapping(),
        "legacy_writer": writer.to_mapping(),
        "legacy_connector": connector.to_mapping(),
        "legacy_gateway_restarted": True,
        "direct_discord_disabled": False,
        "discord_dm_allowed": False,
        "secret_material_recorded": False,
        "completed_at_unix": now_unix,
    }
    terminal = _validate_rollback_terminal(
        {**unsigned, "receipt_sha256": _sha256_json(unsigned)},
        plan=plan,
    )
    dependencies.journal.append(
        plan.sha256, "rollback_terminal", terminal, now_unix
    )
    return terminal


def _rollback_and_raise(
    *,
    primary: BaseException,
    plan: CutoverPlan,
    approval: CutoverApproval,
    dependencies: CutoverDependencies,
    entries: list[JournalEntry],
    database_receipt: Mapping[str, Any] | None,
    host_receipt: Mapping[str, Any] | None,
    now_unix: int,
) -> None:
    try:
        _rollback_cutover(
            plan=plan,
            approval=approval,
            dependencies=dependencies,
            entries=entries,
            database_receipt=database_receipt,
            host_receipt=host_receipt,
            now_unix=now_unix,
        )
    except BaseException as rollback_error:
        raise ExceptionGroup(
            "cutover failed and rollback was incomplete",
            [primary, rollback_error],
        ) from None
    raise ProductionCutoverError("production_cutover_rolled_back") from primary


def execute_cutover(
    plan: CutoverPlan,
    approval_value: Mapping[str, Any],
    dependencies: CutoverDependencies,
    *,
    now_unix: int | None = None,
) -> Mapping[str, Any]:
    """Apply or resume the mechanically derived, pre-stop-approved cutover."""

    now = int(time.time()) if now_unix is None else now_unix
    freeze = FreezePlan.from_mapping(plan.value["freeze_plan"])
    approval = CutoverApproval.from_mapping(
        approval_value,
        plan=freeze,
        now_unix=now,
    )
    with dependencies.lock():
        entries = dependencies.journal.load(plan.sha256)
        rollback_terminal = _last(entries, "rollback_terminal")
        if rollback_terminal is not None:
            _validate_rollback_terminal(
                rollback_terminal.value["evidence"],
                plan=plan,
                entries=entries,
            )
            raise ProductionCutoverError("production_cutover_already_rolled_back")
        terminal_entry = _last(entries, "terminal")
        if terminal_entry is not None:
            if (
                _uses_packaged_cron_cutover(plan)
                and _accepted_cron_cutover_state(entries, plan).activation
                is None
            ):
                raise ProductionCutoverError(
                    "production_cron_cutover_terminal_lineage_invalid"
                )
            return copy.deepcopy(dict(terminal_entry.value["evidence"]))
        if not entries and approval.sha256 != plan.value["freeze_approval_sha256"]:
            raise PermissionError("cutover does not reuse the pre-stop owner approval")
        entries = _append_authority(
            dependencies.journal,
            plan.sha256,
            approval,
            now,
        )

        database_receipt: Mapping[str, Any] | None = None
        host_receipt: Mapping[str, Any] | None = None
        boot_commit_receipt: Mapping[str, Any] | None = None
        activation_commit_intent: Mapping[str, Any] | None = None
        database_terminal: Mapping[str, Any] | None = None
        prerequisite_acceptance: Mapping[str, Any] | None = None
        cron_state = _CronCutoverState()
        try:
            database_receipt = _accepted_database_apply(entries, plan)
            host_receipt = _accepted_host_apply(entries, plan)
            database_terminal = _accepted_database_terminal(entries, plan)
            activation_commit_intent = _accepted_activation_commit_intent(
                entries, plan
            )
            boot_commit_receipt = _accepted_host_boot_commit(entries, plan)
            cron_state = _accepted_cron_cutover_state(entries, plan)
            if (
                activation_commit_intent is None
                and _resume_requires_rollback(entries)
            ):
                raise ProductionCutoverError(
                    "production_cutover_unreceipted_mutation"
                )

            expected_gateway = ServiceObservation.from_mapping(
                plan.value["gateway_legacy_identity"]
            )
            expected_writer = ServiceObservation.from_mapping(
                plan.value["writer_pre_identity"]
            )
            expected_connector = ServiceObservation.from_mapping(
                plan.value["connector_pre_identity"]
            )
            current_gateway = dependencies.services.observe_gateway()
            current_writer = dependencies.services.observe_writer()
            current_connector = dependencies.services.observe_connector()
            prerequisites_started = _last(entries, "prerequisites_started") is not None
            writer_started = _last(entries, "writer_started") is not None
            gateway_started = _last(entries, "gateway_started") is not None
            if boot_commit_receipt is not None and (
                database_receipt is None
                or host_receipt is None
                or activation_commit_intent is None
                or database_terminal is None
                or not prerequisites_started
                or not writer_started
                or _uses_packaged_cron_cutover(plan)
                and cron_state.postflight is None
            ):
                raise ProductionCutoverError(
                    "production_cutover_boot_commit_lineage_invalid"
                )
            if host_receipt is None:
                _require_same_service_identity(expected_gateway, current_gateway)
                _require_same_service_identity(expected_writer, current_writer)
                _require_same_service_identity(expected_connector, current_connector)
                if (
                    not current_gateway.stopped
                    or not current_writer.stopped
                    or not current_connector.stopped
                ):
                    raise ProductionCutoverError(
                        "production_cutover_services_not_stopped"
                    )
            elif activation_commit_intent is None:
                target_gateway = ServiceObservation.from_mapping(
                    host_receipt["gateway_stopped"]
                )
                target_writer = ServiceObservation.from_mapping(
                    host_receipt["writer_stopped"]
                )
                target_connector = ServiceObservation.from_mapping(
                    host_receipt["connector_stopped"]
                )
                expected_gateway_identity = (
                    ServiceObservation.from_mapping(
                        boot_commit_receipt["gateway_observed"]
                    )
                    if boot_commit_receipt is not None
                    else target_gateway
                )
                expected_writer_identity = (
                    ServiceObservation.from_mapping(
                        boot_commit_receipt["writer_active"]
                    )
                    if boot_commit_receipt is not None
                    else target_writer
                )
                expected_connector_identity = (
                    ServiceObservation.from_mapping(
                        boot_commit_receipt["connector_active"]
                    )
                    if boot_commit_receipt is not None
                    else target_connector
                )
                _require_same_service_identity(
                    expected_gateway_identity, current_gateway
                )
                _require_same_service_identity(
                    expected_writer_identity, current_writer
                )
                _require_same_service_identity(
                    expected_connector_identity, current_connector
                )
                if gateway_started:
                    if current_gateway.value["active_state"] != "active":
                        if not current_gateway.stopped:
                            raise ProductionCutoverError(
                                "production_cutover_gateway_resume_state_invalid"
                            )
                elif not current_gateway.stopped:
                    raise ProductionCutoverError(
                        "production_cutover_gateway_started_early"
                    )
                if writer_started:
                    if current_writer.value["active_state"] != "active":
                        raise ProductionCutoverError(
                            "production_cutover_writer_resume_state_invalid"
                        )
                elif not current_writer.stopped:
                    raise ProductionCutoverError(
                        "production_cutover_writer_started_early"
                    )
                if prerequisites_started:
                    if current_connector.value["active_state"] != "active":
                        if not current_connector.stopped:
                            raise ProductionCutoverError(
                                "production_cutover_connector_resume_state_invalid"
                            )
                elif not current_connector.stopped:
                    raise ProductionCutoverError(
                        "production_cutover_connector_started_early"
                    )

            if host_receipt is None:
                dependencies.journal.append(
                    plan.sha256,
                    "host_apply_started",
                    {
                        "artifact_sha256": plan.value["artifacts"]
                        ["host_activation"]["sha256"]
                    },
                    now,
                )
                host_receipt = _require_host_apply_receipt(
                    dependencies.host.apply_stopped(plan),
                    plan=plan,
                )
                dependencies.journal.append(
                    plan.sha256,
                    "host_applied",
                    host_receipt,
                    now,
                )
                entries = dependencies.journal.load(plan.sha256)

            if _uses_packaged_cron_cutover(plan):
                cron_state = _accepted_cron_cutover_state(entries, plan)
                if cron_state.rollback is not None:
                    raise ProductionCutoverError(
                        "production_cron_cutover_already_rolled_back"
                    )
                if (
                    activation_commit_intent is not None
                    and cron_state.postflight is None
                ):
                    raise ProductionCutoverError(
                        "production_cron_cutover_commit_lineage_invalid"
                    )
                if activation_commit_intent is None:
                    if dependencies.cron is None:
                        raise ProductionCutoverError(
                            "production_cron_cutover_boundary_missing"
                        )
                    if (
                        not dependencies.services.observe_gateway().stopped
                        or not dependencies.services.observe_writer().stopped
                        or not dependencies.services.observe_connector().stopped
                    ):
                        raise ProductionCutoverError(
                            "production_cron_cutover_services_not_stopped"
                        )
                    if cron_state.preflight is None:
                        if _last(entries, "cron_preflight_started") is None:
                            dependencies.journal.append(
                                plan.sha256,
                                "cron_preflight_started",
                                {
                                    "continuity_plan_sha256": plan.value[
                                        "cron_continuity_plan"
                                    ]["plan_sha256"]
                                },
                                now,
                            )
                        cron_preflight = _require_cron_cutover_receipt(
                            dependencies.cron.preflight(plan),
                            action="preflight",
                            plan=plan,
                        )
                        dependencies.journal.append(
                            plan.sha256,
                            "cron_preflight_validated",
                            cron_preflight,
                            now,
                        )
                        entries = dependencies.journal.load(plan.sha256)
                        cron_state = _accepted_cron_cutover_state(entries, plan)
                    if cron_state.apply is None:
                        dependencies.journal.append(
                            plan.sha256,
                            "cron_apply_started",
                            {
                                "preflight_receipt_sha256": cron_state.preflight[
                                    "receipt_sha256"
                                ]
                            },
                            now,
                        )
                        cron_apply = _require_cron_cutover_receipt(
                            dependencies.cron.apply(
                                plan,
                                cron_state.preflight,
                            ),
                            action="apply",
                            plan=plan,
                            expected_prior_sha256=str(
                                cron_state.preflight["receipt_sha256"]
                            ),
                        )
                        dependencies.journal.append(
                            plan.sha256,
                            "cron_applied",
                            cron_apply,
                            now,
                        )
                        entries = dependencies.journal.load(plan.sha256)
                        cron_state = _accepted_cron_cutover_state(entries, plan)
                    if cron_state.postflight is None:
                        if _last(entries, "cron_postflight_started") is None:
                            dependencies.journal.append(
                                plan.sha256,
                                "cron_postflight_started",
                                {
                                    "apply_receipt_sha256": cron_state.apply[
                                        "receipt_sha256"
                                    ]
                                },
                                now,
                            )
                        cron_postflight = _require_cron_cutover_receipt(
                            dependencies.cron.postflight(
                                plan,
                                cron_state.apply,
                            ),
                            action="postflight",
                            plan=plan,
                            expected_prior_sha256=str(
                                cron_state.apply["receipt_sha256"]
                            ),
                        )
                        dependencies.journal.append(
                            plan.sha256,
                            "cron_postflight_validated",
                            cron_postflight,
                            now,
                        )
                        entries = dependencies.journal.load(plan.sha256)
                        cron_state = _accepted_cron_cutover_state(entries, plan)

            if _last(entries, "prerequisites_started") is None:
                dependencies.journal.append(
                    plan.sha256,
                    "prerequisites_start_started",
                    {
                        "artifact_sha256": plan.value["artifacts"]
                        ["host_activation"]["sha256"]
                    },
                    now,
                )
                prerequisite_start = _require_bound_receipt(
                    dependencies.host.start_prerequisites(plan),
                    schema="muncho-production-prerequisite-services-start.v1",
                    plan=plan,
                    artifact_name="host_activation",
                )
                dependencies.journal.append(
                    plan.sha256,
                    "prerequisites_started",
                    prerequisite_start,
                    now,
                )
                entries = dependencies.journal.load(plan.sha256)

            current_gateway = dependencies.services.observe_gateway()
            current_writer = dependencies.services.observe_writer()
            current_connector = dependencies.services.observe_connector()
            writer_stage_valid = (
                current_writer.value["active_state"] == "active"
                if writer_started
                else current_writer.stopped
            )
            if activation_commit_intent is None and (
                not current_gateway.stopped
                or not writer_stage_valid
                or current_connector.value["active_state"] != "active"
            ):
                raise ProductionCutoverError(
                    "production_cutover_prerequisite_service_state_invalid"
                )

            if activation_commit_intent is None:
                prerequisite_acceptance = (
                    _require_capability_prerequisite_acceptance(
                        dependencies.prerequisites.collect_and_validate(plan),
                        plan=plan,
                    )
                )
                dependencies.journal.append(
                    plan.sha256,
                    "capability_prerequisites_validated",
                    prerequisite_acceptance,
                    now,
                )
                entries = dependencies.journal.load(plan.sha256)
            else:
                accepted = _last(entries, "capability_prerequisites_validated")
                if accepted is None:
                    raise ProductionCutoverError(
                        "production_capability_prerequisite_acceptance_missing"
                    )
                prerequisite_acceptance = (
                    _require_capability_prerequisite_acceptance(
                        accepted.value["evidence"], plan=plan
                    )
                )

            if database_receipt is None:
                observed = dependencies.snapshots.observe_before_apply(plan)
                if observed.frozen_truth() != plan.final_snapshot.frozen_truth():
                    raise ProductionCutoverError(
                        "production_final_tail_changed_after_approval"
                    )
                dependencies.journal.append(
                    plan.sha256,
                    "final_tail_reobserved",
                    {
                        "snapshot_sha256": observed.sha256,
                        "frozen_truth_sha256": _sha256_json(
                            observed.frozen_truth()
                        ),
                    },
                    now,
                )

            if _last(entries, "preflight") is None:
                preflight = _require_database_receipt(
                    dependencies.database.preflight(plan),
                    schema="muncho-production-legacy-cutover-preflight.v1",
                    plan=plan,
                    artifact_name="database_postflight",
                )
                dependencies.journal.append(
                    plan.sha256,
                    "preflight",
                    preflight,
                    now,
                )
                entries = dependencies.journal.load(plan.sha256)

            if database_receipt is None:
                dependencies.journal.append(
                    plan.sha256,
                    "database_apply_started",
                    {
                        "artifact_sha256": plan.value["artifacts"]
                        ["database_apply"]["sha256"]
                    },
                    now,
                )
                database_receipt = _require_database_receipt(
                    dependencies.database.apply(plan),
                    schema="muncho-production-legacy-database-apply.v1",
                    plan=plan,
                    artifact_name="database_apply",
                )
                dependencies.journal.append(
                    plan.sha256,
                    "database_applied",
                    database_receipt,
                    now,
                )
                entries = dependencies.journal.load(plan.sha256)

            if _last(entries, "writer_started") is None:
                dependencies.journal.append(
                    plan.sha256,
                    "writer_start_started",
                    {
                        "artifact_sha256": plan.value["artifacts"]
                        ["host_activation"]["sha256"]
                    },
                    now,
                )
                writer_receipt = _require_bound_receipt(
                    dependencies.host.start_writer(plan),
                    schema="muncho-production-writer-start.v1",
                    plan=plan,
                    artifact_name="host_activation",
                )
                dependencies.journal.append(
                    plan.sha256,
                    "writer_started",
                    writer_receipt,
                    now,
                )
                entries = dependencies.journal.load(plan.sha256)

            if database_terminal is None:
                database_terminal = _require_database_receipt(
                    dependencies.database.terminal(plan),
                    schema="muncho-production-legacy-cutover-postflight.v1",
                    plan=plan,
                    artifact_name="database_postflight",
                )
                dependencies.journal.append(
                    plan.sha256,
                    "database_terminal_validated",
                    database_terminal,
                    now,
                )
                entries = dependencies.journal.load(plan.sha256)

            if activation_commit_intent is None:
                gateway_fenced = dependencies.services.observe_gateway()
                writer_ready = dependencies.services.observe_writer()
                connector_ready = dependencies.services.observe_connector()
                if (
                    not gateway_fenced.stopped
                    or writer_ready.value["active_state"] != "active"
                    or connector_ready.value["active_state"] != "active"
                    or gateway_fenced.stable_identity()
                    != ServiceObservation.from_mapping(
                        host_receipt["gateway_stopped"]
                    ).stable_identity()
                    or writer_ready.stable_identity()
                    != ServiceObservation.from_mapping(
                        host_receipt["writer_stopped"]
                    ).stable_identity()
                    or connector_ready.stable_identity()
                    != ServiceObservation.from_mapping(
                        host_receipt["connector_stopped"]
                    ).stable_identity()
                ):
                    raise ProductionCutoverError(
                        "production_cutover_ingress_fence_invalid"
                    )
                prerequisite_start = _last(entries, "prerequisites_started")
                writer_start = _last(entries, "writer_started")
                if prerequisite_start is None or writer_start is None:
                    raise ProductionCutoverError(
                        "production_activation_commit_lineage_missing"
                    )
                prerequisite_start_receipt = _require_bound_receipt(
                    prerequisite_start.value["evidence"],
                    schema="muncho-production-prerequisite-services-start.v1",
                    plan=plan,
                    artifact_name="host_activation",
                )
                writer_start_receipt = _require_bound_receipt(
                    writer_start.value["evidence"],
                    schema="muncho-production-writer-start.v1",
                    plan=plan,
                    artifact_name="host_activation",
                )
                intent_unsigned = {
                    "schema": "muncho-production-activation-commit-intent.v1",
                    "plan_sha256": plan.sha256,
                    "approval_sha256": approval.sha256,
                    "host_apply_receipt_sha256": host_receipt["receipt_sha256"],
                    "database_apply_receipt_sha256": database_receipt[
                        "receipt_sha256"
                    ],
                    "database_postflight_receipt_sha256": database_terminal[
                        "receipt_sha256"
                    ],
                    "prerequisite_start_receipt_sha256": (
                        prerequisite_start_receipt["receipt_sha256"]
                    ),
                    "writer_start_receipt_sha256": writer_start_receipt[
                        "receipt_sha256"
                    ],
                    "capability_prerequisite_receipt_sha256": (
                        prerequisite_acceptance["receipt_sha256"]
                    ),
                    "gateway_stopped": True,
                    "connector_active": True,
                    "forward_only": True,
                    "secret_material_recorded": False,
                }
                activation_commit_intent = {
                    **intent_unsigned,
                    "receipt_sha256": _sha256_json(intent_unsigned),
                }
                # CutoverJournal.append fsyncs the entry and its parent before
                # returning.  This is the irreversible authority boundary.
                dependencies.journal.append(
                    plan.sha256,
                    "activation_commit_intent",
                    activation_commit_intent,
                    now,
                )
                entries = dependencies.journal.load(plan.sha256)
                activation_commit_intent = _require_activation_commit_intent(
                    activation_commit_intent,
                    plan=plan,
                    entries=entries,
                )

            if boot_commit_receipt is None:
                if _last(entries, "boot_commit_started") is None:
                    dependencies.journal.append(
                        plan.sha256,
                        "boot_commit_started",
                        {
                            "artifact_sha256": plan.value["artifacts"]
                            ["host_activation"]["sha256"]
                        },
                        now,
                    )
                boot_commit_receipt = _require_host_boot_commit_receipt(
                    dependencies.host.commit_boot(plan), plan=plan
                )
                dependencies.journal.append(
                    plan.sha256, "boot_committed", boot_commit_receipt, now
                )
                entries = dependencies.journal.load(plan.sha256)

            if _last(entries, "gateway_started") is None:
                if _last(entries, "gateway_start_started") is None:
                    dependencies.journal.append(
                        plan.sha256,
                        "gateway_start_started",
                        {"unit": GATEWAY_UNIT},
                        now,
                    )
                dependencies.services.start_gateway()
            gateway_active = dependencies.services.observe_gateway()
            writer_active = dependencies.services.observe_writer()
            connector_active = dependencies.services.observe_connector()
            if (
                gateway_active.value["active_state"] != "active"
                or writer_active.value["active_state"] != "active"
                or connector_active.value["active_state"] != "active"
                or gateway_active.stable_identity()
                != plan.value["gateway_target_identity"]
                or writer_active.stable_identity()
                != plan.value["writer_target_identity"]
                or connector_active.stable_identity()
                != plan.value["connector_target_identity"]
            ):
                raise ProductionCutoverError(
                    "production_cutover_terminal_services_invalid"
                )
            if _last(entries, "gateway_started") is None:
                dependencies.journal.append(
                    plan.sha256,
                    "gateway_started",
                    {
                        "gateway_observation_sha256": gateway_active.sha256,
                        "writer_observation_sha256": writer_active.sha256,
                        "connector_observation_sha256": connector_active.sha256,
                    },
                    now,
                )
            entries = dependencies.journal.load(plan.sha256)
            if _uses_packaged_cron_cutover(plan):
                cron_state = _accepted_cron_cutover_state(entries, plan)
                if cron_state.postflight is None:
                    raise ProductionCutoverError(
                        "production_cron_cutover_postflight_missing"
                    )
                expected_cron_authority = _build_cron_activation_authority(
                    plan=plan,
                    postflight_receipt=cron_state.postflight,
                    entries=entries,
                )
                authority_entry = _last(entries, "cron_activation_authority")
                if authority_entry is None:
                    dependencies.journal.append(
                        plan.sha256,
                        "cron_activation_authority",
                        expected_cron_authority,
                        now,
                    )
                    entries = dependencies.journal.load(plan.sha256)
                    cron_state = _accepted_cron_cutover_state(entries, plan)
                elif authority_entry.value["evidence"] != expected_cron_authority:
                    raise ProductionCutoverError(
                        "production_cron_activation_authority_invalid"
                    )
                if cron_state.activation is None:
                    if dependencies.cron is None:
                        raise ProductionCutoverError(
                            "production_cron_cutover_boundary_missing"
                        )
                    if _last(entries, "cron_activation_started") is None:
                        dependencies.journal.append(
                            plan.sha256,
                            "cron_activation_started",
                            {
                                "activation_authority_sha256": (
                                    expected_cron_authority["authority_sha256"]
                                )
                            },
                            now,
                        )
                    cron_activation = _require_cron_cutover_receipt(
                        dependencies.cron.activate(
                            plan,
                            cron_state.postflight,
                            expected_cron_authority,
                        ),
                        action="activation",
                        plan=plan,
                        expected_prior_sha256=str(
                            cron_state.postflight["receipt_sha256"]
                        ),
                    )
                    if (
                        cron_activation["activation_authority_sha256"]
                        != expected_cron_authority["authority_sha256"]
                    ):
                        raise ProductionCutoverError(
                            "production_cron_activation_lineage_invalid"
                        )
                    dependencies.journal.append(
                        plan.sha256,
                        "cron_activated",
                        cron_activation,
                        now,
                    )
                    entries = dependencies.journal.load(plan.sha256)
                    cron_state = _accepted_cron_cutover_state(entries, plan)
                if cron_state.activation is None:
                    raise ProductionCutoverError(
                        "production_cron_cutover_activation_missing"
                    )
            unsigned = {
                "schema": TERMINAL_SCHEMA,
                "plan_sha256": plan.sha256,
                "freeze_plan_sha256": plan.value["freeze_plan_sha256"],
                "freeze_approval_sha256": plan.value["freeze_approval_sha256"],
                "approval_sha256": approval.sha256,
                "final_tail_receipt_sha256": plan.value[
                    "final_tail_receipt_sha256"
                ],
                "capability_prerequisite_receipt_sha256": (
                    prerequisite_acceptance["prerequisite_receipt_sha256"]
                ),
                "capability_prerequisite_file_sha256": (
                    prerequisite_acceptance["prerequisite_file_sha256"]
                ),
                "isolated_canary_goal_continuation_terminal_sha256": (
                    prerequisite_acceptance[
                        "goal_continuation_terminal_sha256"
                    ]
                ),
                "isolated_canary_workspace_gateway_receipt_sha256": (
                    prerequisite_acceptance[
                        "workspace_gateway_receipt_sha256"
                    ]
                ),
                "isolation_equivalence_projection_sha256": (
                    prerequisite_acceptance[
                        "isolation_equivalence_projection_sha256"
                    ]
                ),
                "zero_canonical_database_mutation_observed": (
                    prerequisite_acceptance[
                        "zero_canonical_database_mutation_observed"
                    ]
                ),
                "pre_db_zero_write_observation_sha256": (
                    prerequisite_acceptance[
                        "pre_db_zero_write_observation_sha256"
                    ]
                ),
                "capability_topology_identity_sha256": (
                    production_capability_topology_identity_sha256(
                        plan.value["capability_topology"]
                    )
                ),
                "database_apply_receipt_sha256": database_receipt[
                    "receipt_sha256"
                ],
                "host_apply_receipt_sha256": host_receipt["receipt_sha256"],
                "host_boot_commit_receipt_sha256": boot_commit_receipt[
                    "receipt_sha256"
                ],
                "activation_commit_intent_receipt_sha256": (
                    activation_commit_intent["receipt_sha256"]
                ),
                "database_postflight_receipt_sha256": database_terminal[
                    "receipt_sha256"
                ],
                "gateway_observation_sha256": gateway_active.sha256,
                "writer_observation_sha256": writer_active.sha256,
                "connector_observation_sha256": connector_active.sha256,
                "direct_discord_disabled": True,
                "discord_dm_allowed": False,
                "rollback_used": False,
                "secret_material_recorded": False,
                "completed_at_unix": now,
            }
            terminal = {**unsigned, "receipt_sha256": _sha256_json(unsigned)}
            dependencies.journal.append(plan.sha256, "terminal", terminal, now)
            return terminal
        except BaseException as primary:
            try:
                entries = dependencies.journal.load(plan.sha256)
            except BaseException as journal_error:
                raise ExceptionGroup(
                    "cutover failed and rollback was incomplete",
                    [primary, journal_error],
                ) from None
            try:
                committed = _accepted_activation_commit_intent(entries, plan)
            except BaseException as intent_error:
                raise ExceptionGroup(
                    "cutover activation authority could not be reconciled",
                    [primary, intent_error],
                ) from None
            if committed is not None:
                raise ProductionCutoverError(
                    "production_cutover_forward_recovery_required"
                ) from primary
            try:
                accepted_database = _accepted_database_apply(entries, plan)
                accepted_host = _accepted_host_apply(entries, plan)
            except BaseException:
                database_receipt = None
                host_receipt = None
            else:
                if accepted_database is not None:
                    database_receipt = accepted_database
                if accepted_host is not None:
                    host_receipt = accepted_host
            _rollback_and_raise(
                primary=primary,
                plan=plan,
                approval=approval,
                dependencies=dependencies,
                entries=entries,
                database_receipt=database_receipt,
                host_receipt=host_receipt,
                now_unix=now,
            )


def _require_production_runtime() -> None:
    getuid = getattr(os, "geteuid", None)
    if not callable(getuid) or getuid() != 0:
        raise PermissionError("production_cutover_requires_uid_0")
    if not sys.platform.startswith("linux"):
        raise RuntimeError("production_cutover_requires_linux")


def _load_staged_json(path: Path) -> Mapping[str, Any]:
    if path not in {
        STAGED_FREEZE_PLAN_PATH,
        STAGED_FREEZE_APPROVAL_PATH,
        STAGED_CUTOVER_PLAN_PATH,
    }:
        raise ValueError("production cutover staged path is not fixed")
    raw = activation._read_trusted_file(
        path,
        expected_uid=0,
        expected_gid=0,
        allowed_modes=frozenset({0o400}),
        maximum=_MAX_JSON,
    )
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=activation._reject_duplicate_keys,
            parse_constant=activation._reject_json_constant,
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise ProductionCutoverError(
            "production_cutover_staged_json_invalid"
        ) from exc
    if not isinstance(value, Mapping) or raw != _canonical_bytes(value):
        raise ProductionCutoverError(
            "production_cutover_staged_json_invalid"
        )
    return copy.deepcopy(dict(value))


def execute_fixed_staged(command: str) -> Mapping[str, Any]:
    """Run one fixed root-staged production step with packaged boundaries."""

    _require_production_runtime()
    services = ProductionSystemdServiceBoundary()
    journal = RootCutoverJournal()
    process = ProductionArtifactProcessBoundary()
    if command == "abort-freeze":
        plan = FreezePlan.from_mapping(
            _load_staged_json(STAGED_FREEZE_PLAN_PATH)
        )
        receipt = abort_freeze(
            plan,
            _load_staged_json(STAGED_FREEZE_APPROVAL_PATH),
            FreezeDependencies(
                services=services,
                snapshots=process,
                journal=journal,
            ),
            cutover_plan_staged=os.path.lexists(STAGED_CUTOVER_PLAN_PATH),
        )
        return receipt
    if command == "phase-b-preflight":
        plan = CutoverPlan.from_mapping(
            _load_staged_json(STAGED_CUTOVER_PLAN_PATH)
        )
        freeze = FreezePlan.from_mapping(plan.value["freeze_plan"])
        CutoverApproval.from_mapping(
            _load_staged_json(STAGED_FREEZE_APPROVAL_PATH),
            plan=freeze,
        )
        receipt = _require_database_receipt(
            ProductionDatabaseArtifactBoundary(process).preflight(plan),
            schema="muncho-production-legacy-cutover-preflight.v1",
            plan=plan,
            artifact_name="database_postflight",
        )
        activation._ensure_root_directory(PHASE_B_RECEIPT_PATH.parent)
        payload = _canonical_bytes(receipt)
        activation._install_exact_bytes(
            PHASE_B_RECEIPT_PATH,
            payload,
            uid=0,
            gid=0,
            mode=0o444,
        )
        installed = activation._read_trusted_file(
            PHASE_B_RECEIPT_PATH,
            expected_uid=0,
            expected_gid=0,
            allowed_modes=frozenset({0o444}),
            maximum=_MAX_JSON,
        )
        if installed != payload:
            raise ProductionCutoverError(
                "production_phase_b_receipt_install_unconfirmed"
            )
        return receipt
    if command == "capture-final-tail":
        plan = FreezePlan.from_mapping(
            _load_staged_json(STAGED_FREEZE_PLAN_PATH)
        )
        receipt = execute_final_tail_capture(
            plan,
            _load_staged_json(STAGED_FREEZE_APPROVAL_PATH),
            FreezeDependencies(
                services=services,
                snapshots=process,
                journal=journal,
            ),
        )
        return receipt.to_mapping()
    if command == "apply-cutover":
        plan = CutoverPlan.from_mapping(
            _load_staged_json(STAGED_CUTOVER_PLAN_PATH)
        )
        return execute_cutover(
            plan,
            _load_staged_json(STAGED_FREEZE_APPROVAL_PATH),
            CutoverDependencies(
                services=services,
                snapshots=process,
                database=ProductionDatabaseArtifactBoundary(process),
                host=ProductionHostArtifactBoundary(process),
                prerequisites=ProductionCapabilityPrerequisiteBoundary(
                    services=services,
                    snapshots=process,
                ),
                journal=journal,
                cron=ProductionCronCutoverBoundary(),
            ),
        )
    raise ValueError("production cutover command is invalid")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the fixed owner-approved production legacy cutover",
    )
    parser.add_argument(
        "command",
        choices=(
            "capture-final-tail",
            "abort-freeze",
            "phase-b-preflight",
            "apply-cutover",
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = execute_fixed_staged(args.command)
    except BaseException as exc:
        error_code = (
            str(exc)
            if isinstance(exc, ProductionCutoverError)
            and re.fullmatch(r"[a-z0-9_]+", str(exc)) is not None
            else "production_cutover_failed"
        )
        print(
            json.dumps(
                {"ok": False, "error_code": error_code},
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


__all__ = [
    "APPROVAL_SCHEMA", "CUTOVER_PLAN_SCHEMA", "CutoverApproval",
    "CutoverDependencies", "CutoverPlan", "FinalTailReceipt",
    "FreezeDependencies", "FreezePlan", "LegacySnapshot",
    "ProductionArtifactProcessBoundary", "ProductionCutoverError",
    "ProductionDatabaseArtifactBoundary", "ProductionHostArtifactBoundary",
    "ProductionSystemdServiceBoundary",
    "RootCutoverJournal", "ServiceObservation",
    "abort_freeze", "approval_signature_payload", "build_cutover_plan",
    "build_freeze_plan", "build_isolated_canary_goal_prerequisite",
    "execute_cutover", "execute_final_tail_capture", "execute_fixed_staged",
    "main",
]


if __name__ == "__main__":  # pragma: no cover - installed runtime entry.
    raise SystemExit(main())

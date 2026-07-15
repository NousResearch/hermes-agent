#!/usr/bin/env python3
"""Transactional production cutover for packaged cron continuity.

The runtime is root-only and consumes only owner-bound staged artifacts.  It
never executes a cron job while applying the package.  The legacy gateway,
writer, and connector must be stopped before ``preflight``, ``apply``,
``postflight``, or ``rollback``.  ``activate`` is a distinct terminal step and
requires those three services to be active, so collector timers cannot race
the Canonical Brain cutover.

All semantic work remains in the persisted primary-model records.  This module
only archives exact bytes, installs digest-bound files, performs an exact-ID
record transform, controls systemd timers, and emits replay-safe receipts.
"""

from __future__ import annotations

import argparse
import base64
import copy
import grp
import hashlib
import json
import os
import pwd
import re
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from agent.file_safety import get_read_block_error
from gateway import canonical_writer_production_cutover as cutover
from gateway import production_cron_continuity_package as continuity
from gateway import production_cron_migration
from gateway.production_model_sovereignty_runtime import (
    ProductionContractError,
    validate_production_cron_jobs,
)
from ops.muncho.runtime import trusted_cron_collector_rail as collector_rail


REQUEST_SCHEMA = "muncho-production-cron-cutover-request.v1"
PREFLIGHT_SCHEMA = "muncho-production-cron-cutover-preflight.v3"
APPLY_SCHEMA = "muncho-production-cron-cutover-apply.v2"
POSTFLIGHT_SCHEMA = "muncho-production-cron-cutover-postflight.v1"
ACTIVATION_SCHEMA = "muncho-production-cron-cutover-activation.v1"
ACTIVATION_AUTHORITY_SCHEMA = (
    "muncho-production-cron-activation-authority.v1"
)
ROLLBACK_SCHEMA = "muncho-production-cron-cutover-rollback.v1"
PREPARED_SCHEMA = "muncho-production-cron-cutover-prepared.v1"
HOST_SNAPSHOT_SCHEMA = "muncho-production-cron-host-snapshot.v1"

EVIDENCE_ROOT = Path(
    "/var/lib/muncho-production-legacy-cutover/cron-continuity"
)
STAGED_CUTOVER_PLAN_PATH = Path(
    "/var/lib/muncho-production-legacy-cutover/staged/cutover-plan.json"
)
STAGED_ARTIFACT_ROOT = Path(
    "/var/lib/muncho-production-legacy-cutover/staged/cron-continuity"
)
STAGED_ACTIVATION_AUTHORITY_PATH = Path(
    "/var/lib/muncho-production-legacy-cutover/staged/"
    "cron-activation-authority.json"
)
JOBS_PATH = production_cron_migration.DEFAULT_JOBS_PATH
SYSTEMD_ROOT = Path("/etc/systemd/system")
SYSTEMCTL = "/usr/bin/systemctl"
NOLOGIN = "/usr/sbin/nologin"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_UUID4 = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
_UTC = re.compile(r"^20[0-9]{2}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")
_MAX_FILE = 8 * 1024 * 1024
_CORE_UNITS = (
    cutover.GATEWAY_UNIT,
    cutover.WRITER_UNIT,
    cutover.CONNECTOR_UNIT,
)


class ProductionCronCutoverRuntimeError(RuntimeError):
    """Stable, secret-free transactional cron cutover failure."""


@dataclass(frozen=True)
class RuntimeContext:
    cutover_plan: Mapping[str, Any]
    inventory: Mapping[str, Any]
    continuity_plan: Mapping[str, Any]
    replacement_bundle: Mapping[str, Any]
    collector_package: Mapping[str, Any]
    artifact_index: Mapping[str, Any]

    @property
    def cutover_plan_sha256(self) -> str:
        return str(self.cutover_plan["plan_sha256"])


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii", errors="strict")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ProductionCronCutoverRuntimeError(
            "production_cron_cutover_json_invalid"
        ) from exc


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _stable_read(path: Path, *, maximum: int = _MAX_FILE) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ProductionCronCutoverRuntimeError(
            "production_cron_cutover_file_unavailable"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or not 0 < before.st_size <= maximum
        ):
            raise ProductionCronCutoverRuntimeError(
                "production_cron_cutover_file_metadata_invalid"
            )
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
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
    raw = b"".join(chunks)
    if len(raw) != before.st_size or identity(before) != identity(after):
        raise ProductionCronCutoverRuntimeError(
            "production_cron_cutover_file_changed"
        )
    return raw


def _json_file(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(_stable_read(path).decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ProductionCronCutoverRuntimeError(
            "production_cron_cutover_json_file_invalid"
        ) from exc
    if not isinstance(value, dict):
        raise ProductionCronCutoverRuntimeError(
            "production_cron_cutover_json_file_invalid"
        )
    return value


def _atomic_write(
    path: Path,
    value: bytes,
    *,
    mode: int,
    uid: int | None = None,
    gid: int | None = None,
    replay_exact: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.is_symlink():
        raise ProductionCronCutoverRuntimeError(
            "production_cron_cutover_symlink_forbidden"
        )
    if replay_exact and path.exists():
        metadata = path.stat(follow_symlinks=False)
        if (
            _stable_read(path) != value
            or stat.S_IMODE(metadata.st_mode) != mode
        ):
            raise ProductionCronCutoverRuntimeError(
                "production_cron_cutover_evidence_replay_mismatch"
            )
        return
    descriptor, temporary = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        os.fchmod(descriptor, mode)
        if uid is not None or gid is not None:
            os.fchown(descriptor, -1 if uid is None else uid, -1 if gid is None else gid)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _receipt(unsigned: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(unsigned))
    return {**payload, "receipt_sha256": _sha256(_canonical(payload))}


def build_activation_authority(
    *,
    cutover_plan_sha256: str,
    cron_postflight_receipt_sha256: str,
    database_terminal_entry_sha256: str,
    activation_commit_intent_entry_sha256: str,
    boot_committed_entry_sha256: str,
    gateway_started_entry_sha256: str,
) -> dict[str, Any]:
    """Build coordinator-derived forward-only timer activation authority."""

    digests = (
        cutover_plan_sha256,
        cron_postflight_receipt_sha256,
        database_terminal_entry_sha256,
        activation_commit_intent_entry_sha256,
        boot_committed_entry_sha256,
        gateway_started_entry_sha256,
    )
    if any(_SHA256.fullmatch(value or "") is None for value in digests):
        raise ProductionCronCutoverRuntimeError(
            "production_cron_activation_authority_invalid"
        )
    unsigned = {
        "schema": ACTIVATION_AUTHORITY_SCHEMA,
        "cutover_plan_sha256": cutover_plan_sha256,
        "cron_postflight_receipt_sha256": cron_postflight_receipt_sha256,
        "database_terminal_entry_sha256": database_terminal_entry_sha256,
        "activation_commit_intent_entry_sha256": (
            activation_commit_intent_entry_sha256
        ),
        "boot_committed_entry_sha256": boot_committed_entry_sha256,
        "gateway_started_entry_sha256": gateway_started_entry_sha256,
        "database_terminal_validated": True,
        "activation_commit_intent_recorded": True,
        "boot_committed": True,
        "gateway_started": True,
        "forward_recovery_only": True,
        "secret_material_recorded": False,
    }
    return {
        **unsigned,
        "authority_sha256": _sha256(_canonical(unsigned)),
    }


def validate_activation_authority(
    value: Mapping[str, Any],
    *,
    cutover_plan_sha256: str,
    cron_postflight_receipt_sha256: str,
    expected_authority_sha256: str,
) -> dict[str, Any]:
    expected_fields = {
        "schema",
        "cutover_plan_sha256",
        "cron_postflight_receipt_sha256",
        "database_terminal_entry_sha256",
        "activation_commit_intent_entry_sha256",
        "boot_committed_entry_sha256",
        "gateway_started_entry_sha256",
        "database_terminal_validated",
        "activation_commit_intent_recorded",
        "boot_committed",
        "gateway_started",
        "forward_recovery_only",
        "secret_material_recorded",
        "authority_sha256",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != expected_fields
        or value.get("schema") != ACTIVATION_AUTHORITY_SCHEMA
        or value.get("cutover_plan_sha256") != cutover_plan_sha256
        or value.get("cron_postflight_receipt_sha256")
        != cron_postflight_receipt_sha256
        or value.get("authority_sha256") != expected_authority_sha256
        or any(
            _SHA256.fullmatch(str(value.get(field) or "")) is None
            for field in (
                "database_terminal_entry_sha256",
                "activation_commit_intent_entry_sha256",
                "boot_committed_entry_sha256",
                "gateway_started_entry_sha256",
                "authority_sha256",
            )
        )
        or any(
            value.get(field) is not True
            for field in (
                "database_terminal_validated",
                "activation_commit_intent_recorded",
                "boot_committed",
                "gateway_started",
                "forward_recovery_only",
            )
        )
        or value.get("secret_material_recorded") is not False
        or _sha256(
            _canonical(
                {key: item for key, item in value.items() if key != "authority_sha256"}
            )
        )
        != value.get("authority_sha256")
    ):
        raise ProductionCronCutoverRuntimeError(
            "production_cron_activation_authority_invalid"
        )
    return copy.deepcopy(dict(value))


_RECEIPT_FIELDS = {
    "preflight": {
        "schema", "created_at", "cutover_plan_sha256",
        "continuity_plan_sha256", "artifact_index_sha256",
        "source_store_sha256", "expected_target_store_sha256",
        "collector_timer_count", "gateway_writer_connector_stopped",
        "artifacts_valid", "source_store_unchanged",
        "prepared_recovery_sha256", "jobs_archive_sha256",
        "host_snapshot_sha256", "spool_prestate_sha256",
        "manifest_directory_prestate_sha256",
        "legacy_auto_sync_disabled", "legacy_auto_sync_no_active_claim",
        "legacy_auto_sync_next_run_reconciled",
        "collector_execution_readiness_sha256",
        "operational_edge_readiness_receipt_sha256",
        "operational_edge_boot_id_sha256",
        "operational_edge_observed_at_unix",
        "operational_edge_maximum_age_seconds",
        "operational_edge_collector_nonce",
        "operational_edge_meaningful_packet_count",
        "collector_execution_ready",
        "recovery_evidence_persisted", "runtime_target_mutation_performed",
        "provider_or_model_invoked", "discord_delivery_attempted",
        "secret_material_recorded", "receipt_sha256",
    },
    "apply": {
        "schema", "created_at", "cutover_plan_sha256",
        "continuity_plan_sha256", "artifact_index_sha256",
        "preflight_receipt_sha256", "source_store_sha256",
        "target_store_sha256", "jobs_archive_path", "jobs_archive_sha256",
        "host_snapshot_path", "host_snapshot_sha256",
        "replacement_agent_record_count", "collector_only_inert_record_count",
        "preserved_inert_record_count", "collector_unit_file_count",
        "collector_timer_count", "collector_manifest_installed",
        "collector_timers_disabled", "collector_timers_active",
        "collector_execution_readiness_sha256",
        "operational_edge_readiness_receipt_sha256",
        "operational_edge_boot_id_sha256",
        "operational_edge_observed_at_unix",
        "operational_edge_maximum_age_seconds",
        "operational_edge_collector_nonce",
        "operational_edge_meaningful_packet_count",
        "collector_execution_ready",
        "service_identity_reused_from_owner_bound_foundation",
        "records_deleted", "jobs_executed", "provider_or_model_invoked",
        "discord_delivery_attempted", "secret_material_recorded",
        "receipt_sha256",
    },
    "postflight": {
        "schema", "created_at", "cutover_plan_sha256",
        "continuity_plan_sha256", "artifact_index_sha256",
        "apply_receipt_sha256", "source_store_sha256", "target_store_sha256",
        "jobs_store_matches_target", "collector_manifest_matches",
        "collector_units_match", "collector_timers_disabled",
        "collector_timers_active", "packet_root_gateway_readable",
        "production_mutation_performed", "provider_or_model_invoked",
        "discord_delivery_attempted", "secret_material_recorded",
        "receipt_sha256",
    },
    "activation": {
        "schema", "created_at", "cutover_plan_sha256",
        "continuity_plan_sha256", "artifact_index_sha256",
        "postflight_receipt_sha256", "activation_authority_sha256",
        "source_store_sha256", "target_store_sha256",
        "gateway_writer_connector_active", "collector_timer_count",
        "collector_timers_enabled", "collector_timers_active",
        "jobs_executed_by_activation_action", "provider_or_model_invoked",
        "discord_delivery_attempted", "secret_material_recorded",
        "receipt_sha256",
    },
    "rollback": {
        "schema", "created_at", "cutover_plan_sha256",
        "continuity_plan_sha256", "artifact_index_sha256",
        "apply_receipt_sha256", "source_store_sha256", "restored_store_sha256",
        "collector_timers_disabled", "collector_timers_stopped",
        "host_file_prestate_restored", "jobs_store_byte_exact_restored",
        "collector_spool_prestate_restored",
        "collector_manifest_directory_prestate_restored",
        "owner_bound_service_identity_unchanged", "provider_or_model_invoked",
        "discord_delivery_attempted", "secret_material_recorded",
        "receipt_sha256",
    },
}
_RECEIPT_SCHEMAS = {
    "preflight": PREFLIGHT_SCHEMA,
    "apply": APPLY_SCHEMA,
    "postflight": POSTFLIGHT_SCHEMA,
    "activation": ACTIVATION_SCHEMA,
    "rollback": ROLLBACK_SCHEMA,
}
_PRIOR_FIELDS = {
    "apply": "preflight_receipt_sha256",
    "postflight": "apply_receipt_sha256",
    "activation": "postflight_receipt_sha256",
    "rollback": "apply_receipt_sha256",
}


def validate_cutover_receipt(
    value: Mapping[str, Any],
    *,
    action: str,
    plan_sha256: str,
    expected_sha256: str | None = None,
    expected_prior_sha256: str | None = None,
) -> dict[str, Any]:
    expected_fields = _RECEIPT_FIELDS.get(action)
    schema = _RECEIPT_SCHEMAS.get(action)
    prior_field = _PRIOR_FIELDS.get(action)
    if (
        not isinstance(value, Mapping)
        or expected_fields is None
        or schema is None
        or set(value) != expected_fields
        or value.get("schema") != schema
        or value.get("cutover_plan_sha256") != plan_sha256
        or _UTC.fullmatch(str(value.get("created_at") or "")) is None
        or _SHA256.fullmatch(str(value.get("receipt_sha256") or "")) is None
        or _sha256(
            _canonical(
                {key: item for key, item in value.items() if key != "receipt_sha256"}
            )
        )
        != value.get("receipt_sha256")
        or expected_sha256 is not None
        and value.get("receipt_sha256") != expected_sha256
        or value.get("secret_material_recorded") is not False
        or value.get("provider_or_model_invoked") is not False
        or value.get("discord_delivery_attempted") is not False
        or any(
            _SHA256.fullmatch(str(item or "")) is None
            for field, item in value.items()
            if field.endswith("_sha256")
            and field != "apply_receipt_sha256"
        )
        or value.get("apply_receipt_sha256") is not None
        and _SHA256.fullmatch(str(value.get("apply_receipt_sha256"))) is None
        or expected_prior_sha256 is not None
        and (
            prior_field is None
            or value.get(prior_field) != expected_prior_sha256
        )
    ):
        raise ProductionCronCutoverRuntimeError(
            "production_cron_cutover_receipt_invalid"
        )
    expected_values: dict[str, Any] = {
        "collector_timer_count": 21,
        "replacement_agent_record_count": 24,
        "collector_only_inert_record_count": 2,
        "preserved_inert_record_count": 1,
        "collector_unit_file_count": 42,
        "operational_edge_meaningful_packet_count": 14,
        "provider_or_model_invoked": False,
        "discord_delivery_attempted": False,
        "secret_material_recorded": False,
    }
    true_fields = {
        "gateway_writer_connector_stopped", "artifacts_valid",
        "source_store_unchanged", "recovery_evidence_persisted",
        "legacy_auto_sync_disabled", "legacy_auto_sync_no_active_claim",
        "legacy_auto_sync_next_run_reconciled",
        "collector_execution_ready",
        "collector_manifest_installed", "collector_timers_disabled",
        "service_identity_reused_from_owner_bound_foundation",
        "jobs_store_matches_target", "collector_manifest_matches",
        "collector_units_match", "packet_root_gateway_readable",
        "gateway_writer_connector_active", "collector_timers_enabled",
        "collector_timers_stopped", "host_file_prestate_restored",
        "jobs_store_byte_exact_restored", "collector_spool_prestate_restored",
        "collector_manifest_directory_prestate_restored",
        "owner_bound_service_identity_unchanged",
    }
    false_fields = {
        "runtime_target_mutation_performed", "collector_timers_active",
        "records_deleted", "jobs_executed", "production_mutation_performed",
        "jobs_executed_by_activation_action",
    }
    if action == "activation":
        true_fields.add("collector_timers_active")
        false_fields.discard("collector_timers_active")
    values_invalid = any(
        field in value and value[field] != expected
        for field, expected in expected_values.items()
    )
    required_true_invalid = any(
        value.get(field) is not True for field in true_fields if field in value
    )
    required_false_invalid = any(
        value.get(field) is not False for field in false_fields if field in value
    )
    if values_invalid or required_true_invalid or required_false_invalid:
        raise ProductionCronCutoverRuntimeError(
            "production_cron_cutover_receipt_invalid"
        )
    if action in {"preflight", "apply"} and (
        _UUID4.fullmatch(
            str(value.get("operational_edge_collector_nonce") or "")
        )
        is None
        or type(value.get("operational_edge_observed_at_unix")) is not int
        or value.get("operational_edge_observed_at_unix", 0) < 1
        or value.get("operational_edge_maximum_age_seconds") != 120
    ):
        raise ProductionCronCutoverRuntimeError(
            "production_cron_cutover_receipt_invalid"
        )
    if (
        action == "rollback"
        and value["restored_store_sha256"] != value["source_store_sha256"]
        or action == "preflight"
        and value["jobs_archive_sha256"] != value["source_store_sha256"]
        or action == "apply"
        and value["jobs_archive_sha256"] != value["source_store_sha256"]
    ):
        raise ProductionCronCutoverRuntimeError(
            "production_cron_cutover_receipt_invalid"
        )
    return copy.deepcopy(dict(value))


def _plan_root(plan_sha256: str, *, evidence_root: Path = EVIDENCE_ROOT) -> Path:
    if _SHA256.fullmatch(plan_sha256 or "") is None:
        raise ProductionCronCutoverRuntimeError(
            "production_cron_cutover_plan_digest_invalid"
        )
    return evidence_root / plan_sha256


def _receipt_path(
    plan_sha256: str,
    action: str,
    *,
    evidence_root: Path = EVIDENCE_ROOT,
) -> Path:
    return _plan_root(plan_sha256, evidence_root=evidence_root) / "receipts" / f"{action}.json"


def _publish_receipt(
    receipt: Mapping[str, Any],
    *,
    action: str,
    evidence_root: Path,
) -> dict[str, Any]:
    trusted = validate_cutover_receipt(
        receipt,
        action=action,
        plan_sha256=str(receipt["cutover_plan_sha256"]),
    )
    _atomic_write(
        _receipt_path(
            trusted["cutover_plan_sha256"],
            action,
            evidence_root=evidence_root,
        ),
        _canonical(trusted) + b"\n",
        mode=0o600,
        replay_exact=True,
    )
    return trusted


def _prior_receipt(
    *,
    plan_sha256: str,
    action: str,
    expected_sha256: str,
    evidence_root: Path,
) -> dict[str, Any]:
    if _SHA256.fullmatch(expected_sha256 or "") is None:
        raise ProductionCronCutoverRuntimeError(
            "production_cron_cutover_prior_receipt_digest_invalid"
        )
    return validate_cutover_receipt(
        _json_file(
            _receipt_path(plan_sha256, action, evidence_root=evidence_root)
        ),
        action=action,
        plan_sha256=plan_sha256,
        expected_sha256=expected_sha256,
    )


def _existing_receipt(
    *,
    plan_sha256: str,
    action: str,
    evidence_root: Path,
) -> dict[str, Any] | None:
    path = _receipt_path(plan_sha256, action, evidence_root=evidence_root)
    if not path.exists():
        return None
    return validate_cutover_receipt(
        _json_file(path),
        action=action,
        plan_sha256=plan_sha256,
    )


def load_runtime_context(
    *,
    expected_cutover_plan_sha256: str,
    cutover_plan_path: Path = STAGED_CUTOVER_PLAN_PATH,
    artifact_root: Path = STAGED_ARTIFACT_ROOT,
) -> RuntimeContext:
    """Load and cross-bind the final CutoverPlan and staged cron package."""

    try:
        cutover_plan = cutover.CutoverPlan.from_mapping(
            _json_file(cutover_plan_path)
        ).value
    except (ValueError, cutover.ProductionCutoverError) as exc:
        raise ProductionCronCutoverRuntimeError(
            "production_cron_cutover_plan_invalid"
        ) from exc
    if cutover_plan.get("plan_sha256") != expected_cutover_plan_sha256:
        raise ProductionCronCutoverRuntimeError(
            "production_cron_cutover_plan_drifted"
        )
    inventory = production_cron_migration.validate_inventory(
        cutover_plan["cron_inventory"]
    )
    continuity_plan = continuity.validate_packaged_continuity_plan(
        cutover_plan["cron_continuity_plan"],
        inventory=inventory,
        expected_mechanical_job_package_manifest_sha256=cutover_plan[
            "mechanical_job_package"
        ]["manifest_sha256"],
        require_executable=True,
    )
    index = _json_file(artifact_root / continuity.ARTIFACT_INDEX_RELATIVE_PATH)
    continuity.validate_packaged_continuity_artifacts(
        output_root=artifact_root,
        artifact_index=index,
        inventory=inventory,
        expected_revision=cutover_plan["release_revision"],
    )
    bundle = continuity.validate_replacement_bundle(
        _json_file(artifact_root / continuity.REPLACEMENT_BUNDLE_RELATIVE_PATH)
    )
    if (
        index["plan_sha256"] != continuity_plan["plan_sha256"]
        or bundle["bundle_sha256"]
        != continuity_plan["replacement_bundle_sha256"]
    ):
        raise ProductionCronCutoverRuntimeError(
            "production_cron_cutover_artifact_drifted"
        )
    release_root = Path(continuity_plan["trusted_collector_package"]["release_root"])
    runtime_path = release_root / continuity.CUTOVER_RUNTIME_RELATIVE_PATH
    entrypoint_path = release_root / continuity.CUTOVER_ENTRYPOINT_RELATIVE_PATH
    try:
        executing_runtime = Path(__file__).resolve(strict=True)
    except OSError as exc:
        raise ProductionCronCutoverRuntimeError(
            "production_cron_cutover_runtime_unavailable"
        ) from exc
    if (
        executing_runtime != runtime_path
        or _sha256(_stable_read(runtime_path))
        != continuity_plan["cutover_runtime_sha256"]
        or _sha256(_stable_read(entrypoint_path))
        != continuity_plan["cutover_entrypoint_sha256"]
    ):
        raise ProductionCronCutoverRuntimeError(
            "production_cron_cutover_runtime_drifted"
        )
    return RuntimeContext(
        cutover_plan=cutover_plan,
        inventory=inventory,
        continuity_plan=continuity_plan,
        replacement_bundle=bundle,
        collector_package=continuity_plan["trusted_collector_package"],
        artifact_index=index,
    )


def _systemctl(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            [SYSTEMCTL, *args],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProductionCronCutoverRuntimeError(
            "production_cron_cutover_systemctl_failed"
        ) from exc
    if check and result.returncode != 0:
        raise ProductionCronCutoverRuntimeError(
            "production_cron_cutover_systemctl_failed"
        )
    return result


def _unit_active(unit: str) -> bool:
    return _systemctl("is-active", "--quiet", unit, check=False).returncode == 0


def _unit_enabled(unit: str) -> bool:
    return _systemctl("is-enabled", "--quiet", unit, check=False).returncode == 0


def _core_services_stopped() -> bool:
    return all(not _unit_active(unit) for unit in _CORE_UNITS)


def _core_services_active() -> bool:
    return all(_unit_active(unit) for unit in _CORE_UNITS)


def _timer_names(context: RuntimeContext) -> list[str]:
    return sorted(
        item["timer"] for item in context.collector_package["units"].values()
    )


def _unit_paths(context: RuntimeContext) -> dict[Path, bytes]:
    rendered = collector_rail.render_package_unit_files(
        context.collector_package
    )
    return {SYSTEMD_ROOT / Path(relative).name: raw for relative, raw in rendered.items()}


def _parse_jobs(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ProductionCronCutoverRuntimeError(
            "production_cron_cutover_jobs_invalid"
        ) from exc
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("jobs"), list)
        or any(not isinstance(job, dict) for job in value["jobs"])
    ):
        raise ProductionCronCutoverRuntimeError(
            "production_cron_cutover_jobs_invalid"
        )
    return value


def _legacy_auto_sync_prestate(source: Mapping[str, Any]) -> dict[str, bool]:
    matches = [
        job
        for job in source["jobs"]
        if job.get("id") == production_cron_migration.LEGACY_AUTO_SYNC_JOB_ID
        and job.get("name")
        == production_cron_migration.LEGACY_AUTO_SYNC_JOB_NAME
    ]
    if (
        len(matches) != 1
        or matches[0].get("enabled") is not False
        or matches[0].get("fire_claim") is not None
        or matches[0].get("run_claim") is not None
        or matches[0].get("next_run_at") is not None
        and not isinstance(matches[0].get("next_run_at"), str)
    ):
        raise ProductionCronCutoverRuntimeError(
            "production_cron_legacy_auto_sync_not_quiescent"
        )
    return {
        "disabled": True,
        "no_active_claim": True,
        "next_run_reconciled": True,
    }


def _collect_runtime_execution_readiness(
    context: RuntimeContext,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        from gateway import operational_edge_readiness as edge_readiness
        from gateway.operational_edge_catalog import required_cron_operations

        operational = edge_readiness.collect_and_publish_operational_edge_readiness(
            revision=context.collector_package["release_revision"],
            required_jobs=required_cron_operations(),
        )
        execution = collector_rail.collect_execution_readiness(
            context.collector_package,
            operational_edge_receipt=operational,
        )
        execution = collector_rail.validate_execution_readiness(
            execution,
            manifest=context.collector_package,
            operational_edge_receipt=operational,
        )
    except (AttributeError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise ProductionCronCutoverRuntimeError(
            "production_cron_execution_readiness_unavailable"
        ) from exc
    if execution["activation_ready"] is not True:
        raise ProductionCronCutoverRuntimeError(
            "production_cron_execution_readiness_incomplete"
        )
    return operational, execution


def _validate_runtime_execution_readiness_pair(
    context: RuntimeContext,
    operational: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> None:
    """Fail closed unless a just-collected pair is exact and live-bound."""

    try:
        from gateway.operational_edge_catalog import required_cron_operations
        from gateway.operational_edge_readiness import (
            OPERATIONAL_EDGE_READINESS_SCHEMA,
            READINESS_MAXIMUM_AGE_SECONDS,
        )

        required_count = len(required_cron_operations())
    except (ImportError, TypeError, ValueError) as exc:
        raise ProductionCronCutoverRuntimeError(
            "production_cron_execution_readiness_incomplete"
        ) from exc
    if (
        execution.get("activation_ready") is not True
        or _SHA256.fullmatch(
            str(execution.get("readiness_sha256") or "")
        ) is None
        or execution.get("scoped_execution_edge_receipt_sha256")
        != operational.get("receipt_sha256")
        or execution.get("scoped_execution_edge_meaningful_packet_count")
        != required_count
        or operational.get("schema") != OPERATIONAL_EDGE_READINESS_SCHEMA
        or operational.get("release_revision")
        != context.collector_package.get("release_revision")
        or operational.get("job_count") != required_count
        or operational.get("required_job_count") != required_count
        or _SHA256.fullmatch(
            str(operational.get("receipt_sha256") or "")
        ) is None
        or _SHA256.fullmatch(
            str(operational.get("boot_id_sha256") or "")
        ) is None
        or type(operational.get("observed_at_unix")) is not int
        or operational.get("observed_at_unix", 0) < 1
        or operational.get("maximum_age_seconds")
        != READINESS_MAXIMUM_AGE_SECONDS
        or _UUID4.fullmatch(str(operational.get("collector_nonce") or ""))
        is None
    ):
        raise ProductionCronCutoverRuntimeError(
            "production_cron_execution_readiness_incomplete"
        )


def build_target_jobs_bytes(
    *,
    source_store: bytes,
    continuity_plan: Mapping[str, Any],
    replacement_bundle: Mapping[str, Any],
) -> bytes:
    """Pure exact-ID transform used by apply and E2E tests."""

    source = _parse_jobs(source_store)
    _legacy_auto_sync_prestate(source)
    bundle = continuity.validate_replacement_bundle(replacement_bundle)
    records = {record["id"]: record for record in bundle["records"]}
    rows = {row["job_id"]: row for row in continuity_plan["records"]}
    target_jobs = copy.deepcopy(source["jobs"])
    for job in target_jobs:
        if (
            job.get("id")
            == production_cron_migration.LEGACY_AUTO_SYNC_JOB_ID
            and job.get("name")
            == production_cron_migration.LEGACY_AUTO_SYNC_JOB_NAME
        ):
            job["next_run_at"] = None
            break
    for job_id, row in rows.items():
        index = row["index"]
        if (
            not 0 <= index < len(target_jobs)
            or target_jobs[index].get("id") != job_id
            or _sha256(_canonical(target_jobs[index]))
            != row["source_record_sha256"]
        ):
            raise ProductionCronCutoverRuntimeError(
                "production_cron_cutover_source_record_drifted"
            )
        disposition = row["disposition"]
        replacement = records.get(job_id)
        if disposition == continuity.DISPOSITION_KEEP:
            continue
        if disposition == continuity.DISPOSITION_AGENT or (
            disposition == continuity.DISPOSITION_COLLECTOR
            and row["target"]["replacement_agent_record_sha256"] is not None
        ):
            if (
                replacement is None
                or _sha256(_canonical(replacement))
                not in {
                    row["target"].get("replacement_record_sha256"),
                    row["target"].get("replacement_agent_record_sha256"),
                }
            ):
                raise ProductionCronCutoverRuntimeError(
                    "production_cron_cutover_replacement_drifted"
                )
            target_jobs[index] = copy.deepcopy(replacement)
            continue
        if disposition in {
            continuity.DISPOSITION_COLLECTOR,
            continuity.DISPOSITION_PRESERVE,
        }:
            inert = copy.deepcopy(target_jobs[index])
            inert["enabled"] = False
            inert["state"] = "paused"
            target_jobs[index] = inert
            continue
        raise ProductionCronCutoverRuntimeError(
            "production_cron_cutover_disposition_invalid"
        )
    target = {**source, "jobs": target_jobs}
    enabled = [job for job in target_jobs if job.get("enabled") is not False]
    try:
        validate_production_cron_jobs(enabled)
    except ProductionContractError as exc:
        raise ProductionCronCutoverRuntimeError(
            "production_cron_cutover_target_jobs_invalid"
        ) from exc
    inventory = production_cron_migration.inventory_jobs_bytes(
        _canonical(target) + b"\n"
    )
    if inventory["incompatible_enabled_count"] != 0:
        raise ProductionCronCutoverRuntimeError(
            "production_cron_cutover_target_jobs_invalid"
        )
    return _canonical(target) + b"\n"


def _target_snapshot(context: RuntimeContext) -> dict[str, Any]:
    paths = [collector_rail.MANIFEST_PATH, *_unit_paths(context)]
    entries: list[dict[str, Any]] = []
    for path in sorted(paths, key=str):
        if path.is_symlink():
            raise ProductionCronCutoverRuntimeError(
                "production_cron_cutover_target_symlink_forbidden"
            )
        if not path.exists():
            entries.append({"path": str(path), "state": "absent"})
            continue
        raw = _stable_read(path)
        metadata = path.stat(follow_symlinks=False)
        entries.append(
            {
                "path": str(path),
                "state": "present",
                "sha256": _sha256(raw),
                "mode": stat.S_IMODE(metadata.st_mode),
                "uid": metadata.st_uid,
                "gid": metadata.st_gid,
                "content_base64": base64.b64encode(raw).decode("ascii"),
            }
        )
    unsigned = {
        "schema": HOST_SNAPSHOT_SCHEMA,
        "entries": entries,
        "entry_count": len(entries),
        "secret_material_recorded": False,
    }
    return {**unsigned, "snapshot_sha256": _sha256(_canonical(unsigned))}


def _live_snapshot_matches(
    context: RuntimeContext,
    expected: Mapping[str, Any],
) -> bool:
    """Compare every install target and its metadata to prepared prestate."""

    try:
        trusted = _validate_snapshot(
            expected,
            expected_paths={
                str(collector_rail.MANIFEST_PATH),
                *(str(path) for path in _unit_paths(context)),
            },
        )
        return _target_snapshot(context) == trusted
    except (OSError, ProductionCronCutoverRuntimeError):
        return False


def _directory_prestate(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ProductionCronCutoverRuntimeError(
            "production_cron_cutover_spool_symlink_forbidden"
        )
    if not path.exists():
        unsigned = {"path": str(path), "state": "absent"}
    else:
        metadata = path.stat(follow_symlinks=False)
        if not stat.S_ISDIR(metadata.st_mode):
            raise ProductionCronCutoverRuntimeError(
                "production_cron_cutover_spool_invalid"
            )
        unsigned = {
            "path": str(path),
            "state": "present",
            "uid": metadata.st_uid,
            "gid": metadata.st_gid,
            "mode": stat.S_IMODE(metadata.st_mode),
        }
    return {**unsigned, "prestate_sha256": _sha256(_canonical(unsigned))}


def _validate_directory_prestate(
    value: Mapping[str, Any],
    *,
    path: Path,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProductionCronCutoverRuntimeError(
            "production_cron_cutover_spool_prestate_invalid"
        )
    state = value.get("state")
    expected = (
        {"path", "state", "prestate_sha256"}
        if state == "absent"
        else {"path", "state", "uid", "gid", "mode", "prestate_sha256"}
    )
    if (
        set(value) != expected
        or value.get("path") != str(path)
        or state not in {"absent", "present"}
        or state == "present"
        and any(type(value.get(field)) is not int for field in ("uid", "gid", "mode"))
        or _SHA256.fullmatch(str(value.get("prestate_sha256") or "")) is None
        or _sha256(
            _canonical(
                {key: item for key, item in value.items() if key != "prestate_sha256"}
            )
        )
        != value.get("prestate_sha256")
    ):
        raise ProductionCronCutoverRuntimeError(
            "production_cron_cutover_spool_prestate_invalid"
        )
    return copy.deepcopy(dict(value))


def _restore_directory_prestate(
    value: Mapping[str, Any],
    *,
    path: Path,
) -> None:
    trusted = _validate_directory_prestate(value, path=path)
    if trusted["state"] == "absent":
        if path.exists():
            try:
                path.rmdir()
            except OSError as exc:
                raise ProductionCronCutoverRuntimeError(
                    "production_cron_cutover_spool_not_empty"
                ) from exc
        return
    if not path.is_dir() or path.is_symlink():
        raise ProductionCronCutoverRuntimeError(
            "production_cron_cutover_spool_invalid"
        )
    os.chown(path, trusted["uid"], trusted["gid"])
    os.chmod(path, trusted["mode"])


def _validate_snapshot(
    value: Mapping[str, Any],
    *,
    expected_paths: set[str] | None = None,
) -> dict[str, Any]:
    if (
        not isinstance(value, Mapping)
        or value.get("schema") != HOST_SNAPSHOT_SCHEMA
        or not isinstance(value.get("entries"), list)
        or value.get("entry_count") != len(value["entries"])
        or value.get("secret_material_recorded") is not False
        or _SHA256.fullmatch(str(value.get("snapshot_sha256") or "")) is None
        or _sha256(
            _canonical(
                {key: item for key, item in value.items() if key != "snapshot_sha256"}
            )
        )
        != value.get("snapshot_sha256")
    ):
        raise ProductionCronCutoverRuntimeError(
            "production_cron_cutover_snapshot_invalid"
        )
    observed: set[str] = set()
    for item in value["entries"]:
        if (
            not isinstance(item, Mapping)
            or not isinstance(item.get("path"), str)
            or item["path"] in observed
            or item.get("state") not in {"absent", "present"}
            or item.get("state") == "absent"
            and set(item) != {"path", "state"}
            or item.get("state") == "present"
            and (
                set(item)
                != {
                    "path",
                    "state",
                    "sha256",
                    "mode",
                    "uid",
                    "gid",
                    "content_base64",
                }
                or _SHA256.fullmatch(str(item.get("sha256") or "")) is None
                or type(item.get("mode")) is not int
                or type(item.get("uid")) is not int
                or type(item.get("gid")) is not int
                or not isinstance(item.get("content_base64"), str)
            )
        ):
            raise ProductionCronCutoverRuntimeError(
                "production_cron_cutover_snapshot_invalid"
            )
        observed.add(item["path"])
    if expected_paths is not None and observed != expected_paths:
        raise ProductionCronCutoverRuntimeError(
            "production_cron_cutover_snapshot_paths_invalid"
        )
    return copy.deepcopy(dict(value))


def _gateway_group() -> grp.struct_group:
    try:
        return grp.getgrnam(collector_rail.READER_GROUP)
    except KeyError as exc:
        raise ProductionCronCutoverRuntimeError(
            "production_cron_cutover_gateway_group_missing"
        ) from exc


def _ensure_service_identity_and_spool(
    context: RuntimeContext,
) -> tuple[int, int]:
    foundation = context.cutover_plan["host_transition"]["identity_foundation"]
    expected_user = foundation["users"]["projector"]
    expected_primary_group = foundation["groups"]["projector"]
    expected_reader_group = foundation["groups"]["gateway"]
    group = _gateway_group()
    try:
        account = pwd.getpwnam(collector_rail.SERVICE_USER)
        primary = grp.getgrnam(expected_primary_group["name"])
    except KeyError as exc:
        raise ProductionCronCutoverRuntimeError(
            "production_cron_cutover_service_identity_missing"
        ) from exc
    if (
        collector_rail.SERVICE_USER != expected_user["name"]
        or account.pw_uid != expected_user["uid"]
        or account.pw_gid != expected_primary_group["gid"]
        or primary.gr_gid != expected_primary_group["gid"]
        or group.gr_gid != expected_reader_group["gid"]
        or account.pw_dir != expected_user["home"]
        or account.pw_shell != expected_user["shell"]
        or account.pw_dir != "/nonexistent"
        or account.pw_shell != NOLOGIN
    ):
        raise ProductionCronCutoverRuntimeError(
            "production_cron_cutover_service_user_invalid"
        )
    collector_rail.STATE_ROOT.mkdir(parents=True, exist_ok=True, mode=0o2750)
    os.chown(collector_rail.STATE_ROOT, account.pw_uid, group.gr_gid)
    os.chmod(collector_rail.STATE_ROOT, 0o2750)
    return account.pw_uid, group.gr_gid


def _install_package_files(
    context: RuntimeContext,
    *,
    artifact_root: Path,
    gateway_gid: int,
) -> None:
    manifest_source = artifact_root / continuity.COLLECTOR_MANIFEST_RELATIVE_PATH
    collector_rail.MANIFEST_PATH.parent.mkdir(
        parents=True, exist_ok=True, mode=0o750
    )
    os.chown(collector_rail.MANIFEST_PATH.parent, 0, gateway_gid)
    os.chmod(collector_rail.MANIFEST_PATH.parent, 0o750)
    _atomic_write(
        collector_rail.MANIFEST_PATH,
        _stable_read(manifest_source),
        mode=0o640,
        uid=0,
        gid=gateway_gid,
    )
    rendered = _unit_paths(context)
    for target, expected in rendered.items():
        source = (
            artifact_root
            / "cron/trusted-collector/systemd"
            / target.name
        )
        raw = _stable_read(source)
        if raw != expected:
            raise ProductionCronCutoverRuntimeError(
                "production_cron_cutover_unit_drifted"
            )
        _atomic_write(target, raw, mode=0o644, uid=0, gid=0)
    _systemctl("daemon-reload")
    for timer in _timer_names(context):
        _systemctl("disable", "--now", timer, check=False)


def _timers_match(
    context: RuntimeContext,
    *,
    enabled: bool,
    active: bool,
) -> bool:
    return all(
        _unit_enabled(timer) is enabled and _unit_active(timer) is active
        for timer in _timer_names(context)
    )


def _installed_package_files_match(context: RuntimeContext) -> bool:
    try:
        group = _gateway_group()
        manifest_metadata = collector_rail.MANIFEST_PATH.stat(
            follow_symlinks=False
        )
        directory_metadata = collector_rail.MANIFEST_PATH.parent.stat(
            follow_symlinks=False
        )
        if (
            _json_file(collector_rail.MANIFEST_PATH)
            != context.collector_package
            or (manifest_metadata.st_uid, manifest_metadata.st_gid)
            != (0, group.gr_gid)
            or stat.S_IMODE(manifest_metadata.st_mode) != 0o640
            or (directory_metadata.st_uid, directory_metadata.st_gid)
            != (0, group.gr_gid)
            or stat.S_IMODE(directory_metadata.st_mode) != 0o750
        ):
            return False
        for path, raw in _unit_paths(context).items():
            metadata = path.stat(follow_symlinks=False)
            if (
                _stable_read(path) != raw
                or (metadata.st_uid, metadata.st_gid) != (0, 0)
                or stat.S_IMODE(metadata.st_mode) != 0o644
            ):
                return False
    except (OSError, ProductionCronCutoverRuntimeError):
        return False
    return True


def _packet_root_gateway_readable() -> bool:
    try:
        account = pwd.getpwnam(collector_rail.SERVICE_USER)
        group = _gateway_group()
        metadata = collector_rail.STATE_ROOT.stat(follow_symlinks=False)
    except (KeyError, OSError):
        return False
    return (
        account.pw_name == collector_rail.SERVICE_USER
        and metadata.st_gid == group.gr_gid
        and stat.S_IMODE(metadata.st_mode) == 0o2750
        and bool(metadata.st_mode & stat.S_IRGRP)
        and bool(metadata.st_mode & stat.S_IXGRP)
        and get_read_block_error(str(collector_rail.PACKET_ROOT)) is None
    )


def _load_prepared(
    *,
    context: RuntimeContext,
    evidence_root: Path,
) -> dict[str, Any]:
    root = _plan_root(context.cutover_plan_sha256, evidence_root=evidence_root)
    prepared = _json_file(root / "prepared.json")
    expected_fields = {
        "schema",
        "cutover_plan_sha256",
        "source_store_sha256",
        "expected_target_store_sha256",
        "jobs_archive_path",
        "host_snapshot_path",
        "host_snapshot_sha256",
        "spool_prestate",
        "manifest_directory_prestate",
        "prepared_sha256",
    }
    if (
        set(prepared) != expected_fields
        or prepared.get("schema") != PREPARED_SCHEMA
        or prepared.get("cutover_plan_sha256") != context.cutover_plan_sha256
        or prepared.get("jobs_archive_path")
        != str(root / "archive/jobs.source.json")
        or prepared.get("host_snapshot_path")
        != str(root / "archive/host-files.pre.json")
        or any(
            _SHA256.fullmatch(str(prepared.get(field) or "")) is None
            for field in (
                "source_store_sha256",
                "expected_target_store_sha256",
                "host_snapshot_sha256",
                "prepared_sha256",
            )
        )
        or _sha256(
            _canonical(
                {key: item for key, item in prepared.items() if key != "prepared_sha256"}
            )
        )
        != prepared.get("prepared_sha256")
    ):
        raise ProductionCronCutoverRuntimeError(
            "production_cron_cutover_prepared_invalid"
        )
    _validate_directory_prestate(
        prepared["spool_prestate"],
        path=collector_rail.STATE_ROOT,
    )
    _validate_directory_prestate(
        prepared["manifest_directory_prestate"],
        path=collector_rail.MANIFEST_PATH.parent,
    )
    return prepared


def preflight(
    *,
    expected_cutover_plan_sha256: str,
    cutover_plan_path: Path = STAGED_CUTOVER_PLAN_PATH,
    artifact_root: Path = STAGED_ARTIFACT_ROOT,
    jobs_path: Path = JOBS_PATH,
    evidence_root: Path = EVIDENCE_ROOT,
    activation_authority_path: Path = STAGED_ACTIVATION_AUTHORITY_PATH,
    services_stopped: Callable[[], bool] = _core_services_stopped,
    execution_readiness_collector: Callable[
        [RuntimeContext], tuple[dict[str, Any], dict[str, Any]]
    ] = _collect_runtime_execution_readiness,
) -> dict[str, Any]:
    context = load_runtime_context(
        expected_cutover_plan_sha256=expected_cutover_plan_sha256,
        cutover_plan_path=cutover_plan_path,
        artifact_root=artifact_root,
    )
    existing = _existing_receipt(
        plan_sha256=context.cutover_plan_sha256,
        action="preflight",
        evidence_root=evidence_root,
    )
    if (
        activation_authority_path.exists()
        or activation_authority_path.is_symlink()
    ):
        raise ProductionCronCutoverRuntimeError(
            "production_cron_activation_authority_premature"
        )
    operational_readiness, execution_readiness = (
        execution_readiness_collector(context)
    )
    _validate_runtime_execution_readiness_pair(
        context,
        operational_readiness,
        execution_readiness,
    )
    if existing is not None:
        prepared = _load_prepared(context=context, evidence_root=evidence_root)
        archive = _stable_read(Path(prepared["jobs_archive_path"]))
        snapshot = _json_file(Path(prepared["host_snapshot_path"]))
        current_sha256 = _sha256(_stable_read(jobs_path))
        source_state = current_sha256 == prepared["source_store_sha256"]
        target_state = (
            current_sha256 == prepared["expected_target_store_sha256"]
        )
        source_host_safe = _live_snapshot_matches(context, snapshot)
        target_host_safe = (
            _installed_package_files_match(context)
            and _packet_root_gateway_readable()
        )
        if (
            not services_stopped()
            or not _timers_match(context, enabled=False, active=False)
            or prepared["prepared_sha256"]
            != existing["prepared_recovery_sha256"]
            or prepared["source_store_sha256"]
            != existing["source_store_sha256"]
            or prepared["expected_target_store_sha256"]
            != existing["expected_target_store_sha256"]
            or existing["operational_edge_boot_id_sha256"]
            != operational_readiness["boot_id_sha256"]
            or _sha256(archive) != existing["jobs_archive_sha256"]
            or snapshot.get("snapshot_sha256")
            != existing["host_snapshot_sha256"]
            or not (
                source_state and (source_host_safe or target_host_safe)
                or target_state and target_host_safe
            )
        ):
            raise ProductionCronCutoverRuntimeError(
                "production_cron_cutover_preflight_replay_drifted"
            )
        return existing
    source = _stable_read(jobs_path)
    source_jobs = _parse_jobs(source)
    legacy_auto_sync = _legacy_auto_sync_prestate(source_jobs)
    host_snapshot = _target_snapshot(context)
    if (
        _sha256(source) != context.continuity_plan["source_store_sha256"]
        or not services_stopped()
        or any(
            _unit_active(timer) or _unit_enabled(timer)
            for timer in _timer_names(context)
        )
        or any(item["state"] != "absent" for item in host_snapshot["entries"])
    ):
        raise ProductionCronCutoverRuntimeError(
            "production_cron_cutover_preflight_failed"
        )
    target = build_target_jobs_bytes(
        source_store=source,
        continuity_plan=context.continuity_plan,
        replacement_bundle=context.replacement_bundle,
    )
    root = _plan_root(context.cutover_plan_sha256, evidence_root=evidence_root)
    archive = root / "archive/jobs.source.json"
    snapshot_path = root / "archive/host-files.pre.json"
    _atomic_write(archive, source, mode=0o600, replay_exact=True)
    _atomic_write(
        snapshot_path,
        _canonical(host_snapshot) + b"\n",
        mode=0o600,
        replay_exact=True,
    )
    spool_prestate = _directory_prestate(collector_rail.STATE_ROOT)
    manifest_directory_prestate = _directory_prestate(
        collector_rail.MANIFEST_PATH.parent
    )
    prepared_unsigned = {
        "schema": PREPARED_SCHEMA,
        "cutover_plan_sha256": context.cutover_plan_sha256,
        "source_store_sha256": _sha256(source),
        "expected_target_store_sha256": _sha256(target),
        "jobs_archive_path": str(archive),
        "host_snapshot_path": str(snapshot_path),
        "host_snapshot_sha256": host_snapshot["snapshot_sha256"],
        "spool_prestate": spool_prestate,
        "manifest_directory_prestate": manifest_directory_prestate,
    }
    prepared = {
        **prepared_unsigned,
        "prepared_sha256": _sha256(_canonical(prepared_unsigned)),
    }
    _atomic_write(
        root / "prepared.json",
        _canonical(prepared) + b"\n",
        mode=0o600,
        replay_exact=True,
    )
    result = _receipt(
        {
            "schema": PREFLIGHT_SCHEMA,
            "created_at": _now(),
            "cutover_plan_sha256": context.cutover_plan_sha256,
            "continuity_plan_sha256": context.continuity_plan["plan_sha256"],
            "artifact_index_sha256": context.artifact_index[
                "artifact_index_sha256"
            ],
            "source_store_sha256": _sha256(source),
            "expected_target_store_sha256": _sha256(target),
            "collector_timer_count": len(_timer_names(context)),
            "gateway_writer_connector_stopped": True,
            "artifacts_valid": True,
            "source_store_unchanged": True,
            "prepared_recovery_sha256": prepared["prepared_sha256"],
            "jobs_archive_sha256": _sha256(source),
            "host_snapshot_sha256": host_snapshot["snapshot_sha256"],
            "spool_prestate_sha256": spool_prestate["prestate_sha256"],
            "manifest_directory_prestate_sha256": (
                manifest_directory_prestate["prestate_sha256"]
            ),
            "legacy_auto_sync_disabled": legacy_auto_sync["disabled"],
            "legacy_auto_sync_no_active_claim": legacy_auto_sync[
                "no_active_claim"
            ],
            "legacy_auto_sync_next_run_reconciled": legacy_auto_sync[
                "next_run_reconciled"
            ],
            "collector_execution_readiness_sha256": execution_readiness[
                "readiness_sha256"
            ],
            "operational_edge_readiness_receipt_sha256": (
                operational_readiness["receipt_sha256"]
            ),
            "operational_edge_boot_id_sha256": operational_readiness[
                "boot_id_sha256"
            ],
            "operational_edge_observed_at_unix": operational_readiness[
                "observed_at_unix"
            ],
            "operational_edge_maximum_age_seconds": operational_readiness[
                "maximum_age_seconds"
            ],
            "operational_edge_collector_nonce": operational_readiness[
                "collector_nonce"
            ],
            "operational_edge_meaningful_packet_count": operational_readiness[
                "job_count"
            ],
            "collector_execution_ready": True,
            "recovery_evidence_persisted": True,
            "runtime_target_mutation_performed": False,
            "provider_or_model_invoked": False,
            "discord_delivery_attempted": False,
            "secret_material_recorded": False,
        }
    )
    return _publish_receipt(result, action="preflight", evidence_root=evidence_root)


def apply(
    *,
    expected_cutover_plan_sha256: str,
    expected_preflight_receipt_sha256: str,
    cutover_plan_path: Path = STAGED_CUTOVER_PLAN_PATH,
    artifact_root: Path = STAGED_ARTIFACT_ROOT,
    jobs_path: Path = JOBS_PATH,
    evidence_root: Path = EVIDENCE_ROOT,
    activation_authority_path: Path = STAGED_ACTIVATION_AUTHORITY_PATH,
    services_stopped: Callable[[], bool] = _core_services_stopped,
    execution_readiness_collector: Callable[
        [RuntimeContext], tuple[dict[str, Any], dict[str, Any]]
    ] = _collect_runtime_execution_readiness,
) -> dict[str, Any]:
    context = load_runtime_context(
        expected_cutover_plan_sha256=expected_cutover_plan_sha256,
        cutover_plan_path=cutover_plan_path,
        artifact_root=artifact_root,
    )
    prior = _prior_receipt(
        plan_sha256=context.cutover_plan_sha256,
        action="preflight",
        expected_sha256=expected_preflight_receipt_sha256,
        evidence_root=evidence_root,
    )
    existing = _existing_receipt(
        plan_sha256=context.cutover_plan_sha256,
        action="apply",
        evidence_root=evidence_root,
    )
    if (
        activation_authority_path.exists()
        or activation_authority_path.is_symlink()
    ):
        raise ProductionCronCutoverRuntimeError(
            "production_cron_activation_authority_premature"
        )
    if not services_stopped():
        raise ProductionCronCutoverRuntimeError(
            "production_cron_cutover_services_not_stopped"
        )
    operational_readiness, execution_readiness = (
        execution_readiness_collector(context)
    )
    _validate_runtime_execution_readiness_pair(
        context,
        operational_readiness,
        execution_readiness,
    )
    prepared = _load_prepared(context=context, evidence_root=evidence_root)
    if (
        prepared["prepared_sha256"] != prior["prepared_recovery_sha256"]
        or prepared["source_store_sha256"] != prior["source_store_sha256"]
        or prepared["expected_target_store_sha256"]
        != prior["expected_target_store_sha256"]
    ):
        raise ProductionCronCutoverRuntimeError(
            "production_cron_cutover_prepared_drifted"
        )
    archive = Path(prepared["jobs_archive_path"])
    source = _stable_read(archive)
    if _sha256(source) != prepared["source_store_sha256"]:
        raise ProductionCronCutoverRuntimeError(
            "production_cron_cutover_archive_invalid"
        )
    target = build_target_jobs_bytes(
        source_store=source,
        continuity_plan=context.continuity_plan,
        replacement_bundle=context.replacement_bundle,
    )
    if _sha256(target) != prepared["expected_target_store_sha256"]:
        raise ProductionCronCutoverRuntimeError(
            "production_cron_cutover_target_store_drifted"
        )
    current_sha256 = _sha256(_stable_read(jobs_path))
    if current_sha256 not in {_sha256(source), _sha256(target)}:
        raise ProductionCronCutoverRuntimeError(
            "production_cron_cutover_source_store_drifted"
        )
    if existing is not None:
        if (
            existing.get("preflight_receipt_sha256")
            != prior["receipt_sha256"]
            or existing.get("operational_edge_boot_id_sha256")
            != operational_readiness["boot_id_sha256"]
            or current_sha256 != existing.get("target_store_sha256")
            or not _installed_package_files_match(context)
            or not _timers_match(context, enabled=False, active=False)
            or not _packet_root_gateway_readable()
        ):
            raise ProductionCronCutoverRuntimeError(
                "production_cron_cutover_receipt_lineage_invalid"
            )
        return existing
    snapshot_path = Path(prepared["host_snapshot_path"])
    expected_snapshot_paths = {
        str(collector_rail.MANIFEST_PATH),
        *(str(path) for path in _unit_paths(context)),
    }
    snapshot = _validate_snapshot(
        _json_file(snapshot_path),
        expected_paths=expected_snapshot_paths,
    )
    if snapshot["snapshot_sha256"] != prepared["host_snapshot_sha256"]:
        raise ProductionCronCutoverRuntimeError(
            "production_cron_cutover_snapshot_drifted"
        )
    source_state = current_sha256 == _sha256(source)
    target_host_safe = (
        _installed_package_files_match(context)
        and _packet_root_gateway_readable()
    )
    if (
        source_state
        and not (_live_snapshot_matches(context, snapshot) or target_host_safe)
        or not source_state
        and not target_host_safe
    ):
        raise ProductionCronCutoverRuntimeError(
            "production_cron_cutover_host_prestate_drifted"
        )
    _service_uid, gateway_gid = (
        _ensure_service_identity_and_spool(context)
    )
    _install_package_files(
        context,
        artifact_root=artifact_root,
        gateway_gid=gateway_gid,
    )
    metadata = jobs_path.stat(follow_symlinks=False)
    _atomic_write(
        jobs_path,
        target,
        mode=stat.S_IMODE(metadata.st_mode),
        uid=metadata.st_uid,
        gid=metadata.st_gid,
    )
    if (
        _sha256(_stable_read(jobs_path)) != _sha256(target)
        or not _timers_match(context, enabled=False, active=False)
    ):
        raise ProductionCronCutoverRuntimeError(
            "production_cron_cutover_apply_unconfirmed"
        )
    result = _receipt(
        {
            "schema": APPLY_SCHEMA,
            "created_at": _now(),
            "cutover_plan_sha256": context.cutover_plan_sha256,
            "continuity_plan_sha256": context.continuity_plan["plan_sha256"],
            "artifact_index_sha256": context.artifact_index[
                "artifact_index_sha256"
            ],
            "preflight_receipt_sha256": prior["receipt_sha256"],
            "source_store_sha256": _sha256(source),
            "target_store_sha256": _sha256(target),
            "jobs_archive_path": str(archive),
            "jobs_archive_sha256": _sha256(source),
            "host_snapshot_path": str(snapshot_path),
            "host_snapshot_sha256": snapshot["snapshot_sha256"],
            "replacement_agent_record_count": 24,
            "collector_only_inert_record_count": 2,
            "preserved_inert_record_count": 1,
            "collector_unit_file_count": 42,
            "collector_timer_count": len(_timer_names(context)),
            "collector_manifest_installed": True,
            "collector_timers_disabled": True,
            "collector_timers_active": False,
            "collector_execution_readiness_sha256": execution_readiness[
                "readiness_sha256"
            ],
            "operational_edge_readiness_receipt_sha256": (
                operational_readiness["receipt_sha256"]
            ),
            "operational_edge_boot_id_sha256": operational_readiness[
                "boot_id_sha256"
            ],
            "operational_edge_observed_at_unix": operational_readiness[
                "observed_at_unix"
            ],
            "operational_edge_maximum_age_seconds": operational_readiness[
                "maximum_age_seconds"
            ],
            "operational_edge_collector_nonce": operational_readiness[
                "collector_nonce"
            ],
            "operational_edge_meaningful_packet_count": operational_readiness[
                "job_count"
            ],
            "collector_execution_ready": True,
            "service_identity_reused_from_owner_bound_foundation": True,
            "records_deleted": False,
            "jobs_executed": False,
            "provider_or_model_invoked": False,
            "discord_delivery_attempted": False,
            "secret_material_recorded": False,
        }
    )
    return _publish_receipt(result, action="apply", evidence_root=evidence_root)


def postflight(
    *,
    expected_cutover_plan_sha256: str,
    expected_apply_receipt_sha256: str,
    cutover_plan_path: Path = STAGED_CUTOVER_PLAN_PATH,
    artifact_root: Path = STAGED_ARTIFACT_ROOT,
    jobs_path: Path = JOBS_PATH,
    evidence_root: Path = EVIDENCE_ROOT,
    activation_authority_path: Path = STAGED_ACTIVATION_AUTHORITY_PATH,
    services_stopped: Callable[[], bool] = _core_services_stopped,
) -> dict[str, Any]:
    context = load_runtime_context(
        expected_cutover_plan_sha256=expected_cutover_plan_sha256,
        cutover_plan_path=cutover_plan_path,
        artifact_root=artifact_root,
    )
    applied = _prior_receipt(
        plan_sha256=context.cutover_plan_sha256,
        action="apply",
        expected_sha256=expected_apply_receipt_sha256,
        evidence_root=evidence_root,
    )
    existing = _existing_receipt(
        plan_sha256=context.cutover_plan_sha256,
        action="postflight",
        evidence_root=evidence_root,
    )
    if (
        activation_authority_path.exists()
        or activation_authority_path.is_symlink()
    ):
        raise ProductionCronCutoverRuntimeError(
            "production_cron_activation_authority_premature"
        )
    if existing is not None:
        if existing.get("apply_receipt_sha256") != applied["receipt_sha256"]:
            raise ProductionCronCutoverRuntimeError(
                "production_cron_cutover_receipt_lineage_invalid"
            )
    if (
        not services_stopped()
        or _sha256(_stable_read(jobs_path)) != applied["target_store_sha256"]
        or not _installed_package_files_match(context)
        or not _timers_match(context, enabled=False, active=False)
        or not _packet_root_gateway_readable()
    ):
        raise ProductionCronCutoverRuntimeError(
            "production_cron_cutover_postflight_failed"
        )
    if existing is not None:
        return existing
    result = _receipt(
        {
            "schema": POSTFLIGHT_SCHEMA,
            "created_at": _now(),
            "cutover_plan_sha256": context.cutover_plan_sha256,
            "continuity_plan_sha256": context.continuity_plan["plan_sha256"],
            "artifact_index_sha256": context.artifact_index[
                "artifact_index_sha256"
            ],
            "apply_receipt_sha256": applied["receipt_sha256"],
            "source_store_sha256": applied["source_store_sha256"],
            "target_store_sha256": applied["target_store_sha256"],
            "jobs_store_matches_target": True,
            "collector_manifest_matches": True,
            "collector_units_match": True,
            "collector_timers_disabled": True,
            "collector_timers_active": False,
            "packet_root_gateway_readable": True,
            "production_mutation_performed": False,
            "provider_or_model_invoked": False,
            "discord_delivery_attempted": False,
            "secret_material_recorded": False,
        }
    )
    return _publish_receipt(result, action="postflight", evidence_root=evidence_root)


def activate(
    *,
    expected_cutover_plan_sha256: str,
    expected_postflight_receipt_sha256: str,
    expected_activation_authority_sha256: str,
    cutover_plan_path: Path = STAGED_CUTOVER_PLAN_PATH,
    artifact_root: Path = STAGED_ARTIFACT_ROOT,
    evidence_root: Path = EVIDENCE_ROOT,
    activation_authority_path: Path = STAGED_ACTIVATION_AUTHORITY_PATH,
    services_active: Callable[[], bool] = _core_services_active,
) -> dict[str, Any]:
    context = load_runtime_context(
        expected_cutover_plan_sha256=expected_cutover_plan_sha256,
        cutover_plan_path=cutover_plan_path,
        artifact_root=artifact_root,
    )
    post = _prior_receipt(
        plan_sha256=context.cutover_plan_sha256,
        action="postflight",
        expected_sha256=expected_postflight_receipt_sha256,
        evidence_root=evidence_root,
    )
    authority = validate_activation_authority(
        _json_file(activation_authority_path),
        cutover_plan_sha256=context.cutover_plan_sha256,
        cron_postflight_receipt_sha256=post["receipt_sha256"],
        expected_authority_sha256=expected_activation_authority_sha256,
    )
    existing = _existing_receipt(
        plan_sha256=context.cutover_plan_sha256,
        action="activation",
        evidence_root=evidence_root,
    )
    if existing is not None:
        if (
            existing.get("postflight_receipt_sha256")
            != post["receipt_sha256"]
            or existing.get("activation_authority_sha256")
            != authority["authority_sha256"]
            or not _timers_match(context, enabled=True, active=True)
        ):
            raise ProductionCronCutoverRuntimeError(
                "production_cron_cutover_receipt_lineage_invalid"
            )
        return existing
    if not services_active():
        raise ProductionCronCutoverRuntimeError(
            "production_cron_cutover_terminal_services_not_ready"
        )
    for timer in _timer_names(context):
        _systemctl("enable", "--now", timer)
    if not _timers_match(context, enabled=True, active=True):
        raise ProductionCronCutoverRuntimeError(
            "production_cron_cutover_activation_unconfirmed"
        )
    result = _receipt(
        {
            "schema": ACTIVATION_SCHEMA,
            "created_at": _now(),
            "cutover_plan_sha256": context.cutover_plan_sha256,
            "continuity_plan_sha256": context.continuity_plan["plan_sha256"],
            "artifact_index_sha256": context.artifact_index[
                "artifact_index_sha256"
            ],
            "postflight_receipt_sha256": post["receipt_sha256"],
            "activation_authority_sha256": authority["authority_sha256"],
            "source_store_sha256": post["source_store_sha256"],
            "target_store_sha256": post["target_store_sha256"],
            "gateway_writer_connector_active": True,
            "collector_timer_count": len(_timer_names(context)),
            "collector_timers_enabled": True,
            "collector_timers_active": True,
            "jobs_executed_by_activation_action": False,
            "provider_or_model_invoked": False,
            "discord_delivery_attempted": False,
            "secret_material_recorded": False,
        }
    )
    return _publish_receipt(result, action="activation", evidence_root=evidence_root)


def _restore_snapshot(
    snapshot: Mapping[str, Any],
    *,
    expected_paths: set[str],
) -> None:
    trusted = _validate_snapshot(snapshot, expected_paths=expected_paths)
    for item in trusted["entries"]:
        path = Path(item["path"])
        if item["state"] == "absent":
            if path.exists() or path.is_symlink():
                path.unlink()
            continue
        try:
            raw = base64.b64decode(
                item["content_base64"].encode("ascii"), validate=True
            )
        except (ValueError, UnicodeError) as exc:
            raise ProductionCronCutoverRuntimeError(
                "production_cron_cutover_snapshot_invalid"
            ) from exc
        if _sha256(raw) != item["sha256"]:
            raise ProductionCronCutoverRuntimeError(
                "production_cron_cutover_snapshot_invalid"
            )
        _atomic_write(
            path,
            raw,
            mode=item["mode"],
            uid=item["uid"],
            gid=item["gid"],
        )


def rollback(
    *,
    expected_cutover_plan_sha256: str,
    expected_apply_receipt_sha256: str | None = None,
    cutover_plan_path: Path = STAGED_CUTOVER_PLAN_PATH,
    artifact_root: Path = STAGED_ARTIFACT_ROOT,
    jobs_path: Path = JOBS_PATH,
    evidence_root: Path = EVIDENCE_ROOT,
    activation_authority_path: Path = STAGED_ACTIVATION_AUTHORITY_PATH,
    services_stopped: Callable[[], bool] = _core_services_stopped,
) -> dict[str, Any]:
    context = load_runtime_context(
        expected_cutover_plan_sha256=expected_cutover_plan_sha256,
        cutover_plan_path=cutover_plan_path,
        artifact_root=artifact_root,
    )
    existing = _existing_receipt(
        plan_sha256=context.cutover_plan_sha256,
        action="rollback",
        evidence_root=evidence_root,
    )
    if existing is not None:
        if (
            expected_apply_receipt_sha256 is not None
            and existing.get("apply_receipt_sha256")
            != expected_apply_receipt_sha256
        ):
            raise ProductionCronCutoverRuntimeError(
                "production_cron_cutover_receipt_lineage_invalid"
            )
        return existing
    # The coordinator writes this authority only after its durable activation
    # intent and terminal database/boot/gateway journal entries exist.  From
    # that point onward recovery is forward-only, including the crash window
    # after timers start but before this runtime can publish its activation
    # receipt.  Mere existence therefore fails closed; parsing an attacker-
    # controlled or partially-written authority must never re-open rollback.
    if (
        activation_authority_path.exists()
        or activation_authority_path.is_symlink()
        or _receipt_path(
            context.cutover_plan_sha256,
            "activation",
            evidence_root=evidence_root,
        ).exists()
    ):
        raise ProductionCronCutoverRuntimeError(
            "production_cron_cutover_forward_recovery_required"
        )
    if not services_stopped():
        raise ProductionCronCutoverRuntimeError(
            "production_cron_cutover_services_not_stopped"
        )
    applied: dict[str, Any] | None = None
    if expected_apply_receipt_sha256 is not None:
        applied = _prior_receipt(
            plan_sha256=context.cutover_plan_sha256,
            action="apply",
            expected_sha256=expected_apply_receipt_sha256,
            evidence_root=evidence_root,
        )
    prepared = _load_prepared(context=context, evidence_root=evidence_root)
    root = _plan_root(context.cutover_plan_sha256, evidence_root=evidence_root)
    for timer in _timer_names(context):
        _systemctl("disable", "--now", timer, check=False)
    archive_path = Path(prepared["jobs_archive_path"])
    snapshot_path = Path(prepared["host_snapshot_path"])
    if (
        archive_path != root / "archive/jobs.source.json"
        or snapshot_path != root / "archive/host-files.pre.json"
    ):
        raise ProductionCronCutoverRuntimeError(
            "production_cron_cutover_prepared_paths_invalid"
        )
    archive = _stable_read(archive_path)
    if _sha256(archive) != prepared["source_store_sha256"]:
        raise ProductionCronCutoverRuntimeError(
            "production_cron_cutover_archive_invalid"
        )
    current_metadata = jobs_path.stat(follow_symlinks=False)
    _atomic_write(
        jobs_path,
        archive,
        mode=stat.S_IMODE(current_metadata.st_mode),
        uid=current_metadata.st_uid,
        gid=current_metadata.st_gid,
    )
    expected_snapshot_paths = {
        str(collector_rail.MANIFEST_PATH),
        *(str(path) for path in _unit_paths(context)),
    }
    _restore_snapshot(
        _json_file(snapshot_path),
        expected_paths=expected_snapshot_paths,
    )
    _restore_directory_prestate(
        prepared["spool_prestate"],
        path=collector_rail.STATE_ROOT,
    )
    _restore_directory_prestate(
        prepared["manifest_directory_prestate"],
        path=collector_rail.MANIFEST_PATH.parent,
    )
    _systemctl("daemon-reload")
    if _sha256(_stable_read(jobs_path)) != prepared["source_store_sha256"]:
        raise ProductionCronCutoverRuntimeError(
            "production_cron_cutover_rollback_unconfirmed"
        )
    result = _receipt(
        {
            "schema": ROLLBACK_SCHEMA,
            "created_at": _now(),
            "cutover_plan_sha256": context.cutover_plan_sha256,
            "continuity_plan_sha256": context.continuity_plan["plan_sha256"],
            "artifact_index_sha256": context.artifact_index[
                "artifact_index_sha256"
            ],
            "apply_receipt_sha256": (
                None if applied is None else applied["receipt_sha256"]
            ),
            "source_store_sha256": prepared["source_store_sha256"],
            "restored_store_sha256": _sha256(archive),
            "collector_timers_disabled": True,
            "collector_timers_stopped": True,
            "host_file_prestate_restored": True,
            "jobs_store_byte_exact_restored": True,
            "collector_spool_prestate_restored": True,
            "collector_manifest_directory_prestate_restored": True,
            "owner_bound_service_identity_unchanged": True,
            "provider_or_model_invoked": False,
            "discord_delivery_attempted": False,
            "secret_material_recorded": False,
        }
    )
    return _publish_receipt(result, action="rollback", evidence_root=evidence_root)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transactional Muncho production cron cutover"
    )
    parser.add_argument(
        "action",
        choices=("preflight", "apply", "postflight", "activate", "rollback"),
    )
    parser.add_argument("--expected-cutover-plan-sha256", required=True)
    parser.add_argument("--expected-prior-receipt-sha256")
    parser.add_argument("--expected-activation-authority-sha256")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if os.geteuid() != 0:
        raise ProductionCronCutoverRuntimeError(
            "production_cron_cutover_root_required"
        )
    if (
        args.action != "activate"
        and args.expected_activation_authority_sha256 is not None
    ):
        raise ProductionCronCutoverRuntimeError(
            "production_cron_activation_authority_unexpected"
        )
    common = {
        "expected_cutover_plan_sha256": args.expected_cutover_plan_sha256,
    }
    if args.action == "preflight":
        if args.expected_prior_receipt_sha256 is not None:
            raise ProductionCronCutoverRuntimeError(
                "production_cron_cutover_prior_receipt_unexpected"
            )
        result = preflight(**common)
    elif args.action == "apply":
        result = apply(
            **common,
            expected_preflight_receipt_sha256=(
                args.expected_prior_receipt_sha256 or ""
            ),
        )
    elif args.action == "postflight":
        result = postflight(
            **common,
            expected_apply_receipt_sha256=(
                args.expected_prior_receipt_sha256 or ""
            ),
        )
    elif args.action == "activate":
        if not args.expected_activation_authority_sha256:
            raise ProductionCronCutoverRuntimeError(
                "production_cron_activation_authority_required"
            )
        result = activate(
            **common,
            expected_postflight_receipt_sha256=(
                args.expected_prior_receipt_sha256 or ""
            ),
            expected_activation_authority_sha256=(
                args.expected_activation_authority_sha256
            ),
        )
    else:
        result = rollback(
            **common,
            expected_apply_receipt_sha256=args.expected_prior_receipt_sha256,
        )
    sys.stdout.buffer.write(_canonical(result) + b"\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ProductionCronCutoverRuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from None


__all__ = [
    "ACTIVATION_AUTHORITY_SCHEMA",
    "ACTIVATION_SCHEMA",
    "APPLY_SCHEMA",
    "EVIDENCE_ROOT",
    "JOBS_PATH",
    "POSTFLIGHT_SCHEMA",
    "PREFLIGHT_SCHEMA",
    "ProductionCronCutoverRuntimeError",
    "ROLLBACK_SCHEMA",
    "RuntimeContext",
    "STAGED_ARTIFACT_ROOT",
    "STAGED_ACTIVATION_AUTHORITY_PATH",
    "STAGED_CUTOVER_PLAN_PATH",
    "activate",
    "apply",
    "build_activation_authority",
    "build_target_jobs_bytes",
    "load_runtime_context",
    "postflight",
    "preflight",
    "rollback",
    "validate_activation_authority",
    "validate_cutover_receipt",
]

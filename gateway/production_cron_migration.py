"""Exact inventory and inert migration for legacy production cron records.

This is a schema boundary, not a task classifier.  It calls the same static
production cron validator used by gateway startup and records only stable
validation codes and cryptographic record identities.  No prompt or script is
interpreted, rewritten, executed, or promoted to the external mechanical-job
rail.

An apply operation is deliberately separate from inventory.  It requires an
owner-approved, digest-bound plan, an unchanged source store, and an inactive
gateway.  Incompatible enabled records are preserved byte-for-byte in a
private archive and retained in the live store with only ``enabled=False`` and
``state=paused``.  Compatible jobs and already-disabled history are untouched.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import stat
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from gateway.production_model_sovereignty_runtime import (
    GATEWAY_UNIT,
    PRODUCTION_HOME,
    ProductionContractError,
    validate_production_cron_jobs,
)


INVENTORY_SCHEMA = "muncho-production-cron-migration-inventory.v2"
PLAN_SCHEMA = "muncho-production-cron-continuity-plan.v2"
RECEIPT_SCHEMA = "muncho-production-cron-inert-migration-receipt.v1"
PREPARED_SCHEMA = "muncho-production-cron-inert-migration-prepared.v1"
ROLLBACK_SCHEMA = "muncho-production-cron-inert-migration-rollback.v1"
LEGACY_AUTO_SYNC_JOB_ID = "808bddb875ee"
LEGACY_AUTO_SYNC_JOB_NAME = "Fork upstream auto sync PR routine"
MECHANICAL_RAIL_JOB_ID = "fork_upstream_auto_sync_pr"
DEFAULT_JOBS_PATH = PRODUCTION_HOME / "cron" / "jobs.json"
DEFAULT_EVIDENCE_ROOT = Path(
    "/var/lib/muncho-production-legacy-cutover/cron-migration"
)
MAX_STORE_BYTES = 2 * 1024 * 1024

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_APPROVAL_ID = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
_DISPOSITIONS = frozenset(
    {
        "pending_review",
        "keep_compatible",
        "migrate_agent",
        "migrate_mechanical",
        "preserve_inert",
        "retire_stale",
        "replaced_by_packaged_rail",
    }
)


class ProductionCronMigrationError(RuntimeError):
    """Stable non-secret inventory or migration failure."""


def _now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


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
        raise ProductionCronMigrationError(
            "production_cron_migration_json_invalid"
        ) from exc


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _stable_read(path: Path) -> tuple[bytes, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ProductionCronMigrationError(
            "production_cron_store_unavailable"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or not 0 < before.st_size <= MAX_STORE_BYTES
        ):
            raise ProductionCronMigrationError(
                "production_cron_store_metadata_invalid"
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
        raise ProductionCronMigrationError("production_cron_store_changed")
    return raw, before


def _parse_store(raw: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ProductionCronMigrationError(
            "production_cron_store_invalid"
        ) from exc
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("jobs"), list)
        or any(not isinstance(job, dict) for job in payload["jobs"])
    ):
        raise ProductionCronMigrationError(
            "production_cron_store_shape_invalid"
        )
    return payload


def _record_identity(index: int, job: Mapping[str, Any]) -> dict[str, Any]:
    job_id = job.get("id")
    if not isinstance(job_id, str) or not job_id or len(job_id) > 256:
        job_id = None
    return {
        "index": index,
        "job_id": job_id,
        "record_sha256": _sha256(_canonical(job)),
    }


def inventory_jobs_bytes(raw: bytes) -> dict[str, Any]:
    """Return a redaction-safe startup-policy inventory without mutation."""

    payload = _parse_store(raw)
    compatible: list[dict[str, Any]] = []
    incompatible: list[dict[str, Any]] = []
    disabled: list[dict[str, Any]] = []
    legacy_auto_sync: dict[str, Any] = {
        "present": False,
        "legacy_job_id": LEGACY_AUTO_SYNC_JOB_ID,
        "replacement_rail_job_id": MECHANICAL_RAIL_JOB_ID,
    }
    for index, job in enumerate(payload["jobs"]):
        identity = _record_identity(index, job)
        if (
            job.get("id") == LEGACY_AUTO_SYNC_JOB_ID
            and job.get("name") == LEGACY_AUTO_SYNC_JOB_NAME
        ):
            legacy_auto_sync = {
                **identity,
                "present": True,
                "legacy_job_id": LEGACY_AUTO_SYNC_JOB_ID,
                "legacy_job_name_sha256": _sha256(
                    LEGACY_AUTO_SYNC_JOB_NAME.encode("utf-8")
                ),
                "enabled": job.get("enabled") is not False,
                "replacement_rail_job_id": MECHANICAL_RAIL_JOB_ID,
                "legacy_record_executed": False,
            }
        if job.get("enabled") is False:
            disabled.append(identity)
            continue
        try:
            validate_production_cron_jobs([job])
        except ProductionContractError as exc:
            incompatible.append({**identity, "validation_code": exc.code})
        else:
            compatible.append(identity)
    continuity_template = [
        {
            **record,
            "validation_code": None,
            "disposition": "keep_compatible",
            "target": None,
        }
        for record in compatible
    ] + [
        {
            **record,
            "disposition": "pending_review",
            "target": None,
        }
        for record in incompatible
    ]
    if legacy_auto_sync.get("present") and not legacy_auto_sync.get("enabled"):
        continuity_template.append(
            {
                "index": legacy_auto_sync["index"],
                "job_id": legacy_auto_sync["job_id"],
                "record_sha256": legacy_auto_sync["record_sha256"],
                "validation_code": None,
                "disposition": "pending_review",
                "target": {
                    "kind": "packaged_mechanical_job",
                    "job_id": MECHANICAL_RAIL_JOB_ID,
                    "package_manifest_sha256": None,
                },
            }
        )
    plan_payload = {
        "schema": PLAN_SCHEMA,
        "source_store_sha256": _sha256(raw),
        "continuity_dispositions": continuity_template,
        "review_complete": False,
        "cutover_executable": False,
        "blanket_inert_migration_allowed": False,
        "execute_job_during_migration": False,
    }
    result = {
        "schema": INVENTORY_SCHEMA,
        "created_at": _now(),
        "source_store_sha256": _sha256(raw),
        "job_count": len(payload["jobs"]),
        "enabled_count": len(compatible) + len(incompatible),
        "compatible_enabled_count": len(compatible),
        "incompatible_enabled_count": len(incompatible),
        "disabled_count": len(disabled),
        "compatible_enabled_records": compatible,
        "incompatible_enabled_records": incompatible,
        "disabled_records": disabled,
        "legacy_auto_sync": legacy_auto_sync,
        "migration_required_for_gateway_startup": bool(incompatible),
        "migration_plan_payload": plan_payload,
        "migration_plan_sha256": _sha256(_canonical(plan_payload)),
        "prompt_or_script_content_recorded": False,
        "job_executed": False,
    }
    return {
        **result,
        "inventory_sha256": _sha256(_canonical(result)),
    }


def inventory_jobs_file(path: Path = DEFAULT_JOBS_PATH) -> dict[str, Any]:
    raw, _metadata = _stable_read(path)
    return inventory_jobs_bytes(raw)


def validate_inventory(inventory: Mapping[str, Any]) -> dict[str, Any]:
    """Pure self-digest and exact-field validator for owner-side binding."""

    expected_fields = {
        "schema",
        "created_at",
        "source_store_sha256",
        "job_count",
        "enabled_count",
        "compatible_enabled_count",
        "incompatible_enabled_count",
        "disabled_count",
        "compatible_enabled_records",
        "incompatible_enabled_records",
        "disabled_records",
        "legacy_auto_sync",
        "migration_required_for_gateway_startup",
        "migration_plan_payload",
        "migration_plan_sha256",
        "prompt_or_script_content_recorded",
        "job_executed",
        "inventory_sha256",
    }
    if (
        not isinstance(inventory, Mapping)
        or set(inventory) != expected_fields
        or inventory.get("schema") != INVENTORY_SCHEMA
        or not isinstance(inventory.get("inventory_sha256"), str)
        or _SHA256.fullmatch(inventory["inventory_sha256"]) is None
        or _sha256(
            _canonical(
                {
                    key: value
                    for key, value in inventory.items()
                    if key != "inventory_sha256"
                }
            )
        )
        != inventory["inventory_sha256"]
        or not isinstance(inventory.get("migration_plan_payload"), Mapping)
        or _sha256(_canonical(inventory["migration_plan_payload"]))
        != inventory.get("migration_plan_sha256")
        or inventory["migration_plan_payload"].get("schema") != PLAN_SCHEMA
        or inventory["migration_plan_payload"].get("review_complete") is not False
        or inventory["migration_plan_payload"].get("cutover_executable") is not False
        or inventory["migration_plan_payload"].get(
            "blanket_inert_migration_allowed"
        )
        is not False
        or set(inventory["migration_plan_payload"])
        != {
            "schema",
            "source_store_sha256",
            "continuity_dispositions",
            "review_complete",
            "cutover_executable",
            "blanket_inert_migration_allowed",
            "execute_job_during_migration",
        }
        or inventory["migration_plan_payload"].get("source_store_sha256")
        != inventory.get("source_store_sha256")
        or inventory["migration_plan_payload"].get(
            "execute_job_during_migration"
        )
        is not False
        or not isinstance(inventory.get("source_store_sha256"), str)
        or _SHA256.fullmatch(inventory["source_store_sha256"]) is None
        or inventory.get("prompt_or_script_content_recorded") is not False
        or inventory.get("job_executed") is not False
    ):
        raise ProductionCronMigrationError("production_cron_inventory_invalid")
    for field in (
        "compatible_enabled_records",
        "incompatible_enabled_records",
        "disabled_records",
    ):
        if not isinstance(inventory.get(field), list):
            raise ProductionCronMigrationError("production_cron_inventory_invalid")
    counts = {
        name: inventory.get(name)
        for name in (
            "job_count",
            "enabled_count",
            "compatible_enabled_count",
            "incompatible_enabled_count",
            "disabled_count",
        )
    }
    if any(type(value) is not int or value < 0 for value in counts.values()):
        raise ProductionCronMigrationError("production_cron_inventory_invalid")
    if (
        counts["compatible_enabled_count"]
        != len(inventory["compatible_enabled_records"])
        or counts["incompatible_enabled_count"]
        != len(inventory["incompatible_enabled_records"])
        or counts["disabled_count"] != len(inventory["disabled_records"])
        or counts["enabled_count"]
        != counts["compatible_enabled_count"]
        + counts["incompatible_enabled_count"]
        or counts["job_count"] != counts["enabled_count"] + counts["disabled_count"]
        or inventory.get("migration_required_for_gateway_startup")
        is not bool(counts["incompatible_enabled_count"])
    ):
        raise ProductionCronMigrationError("production_cron_inventory_invalid")
    observed_indexes: set[int] = set()
    for field in (
        "compatible_enabled_records",
        "incompatible_enabled_records",
        "disabled_records",
    ):
        for row in inventory[field]:
            expected_row_fields = {"index", "job_id", "record_sha256"}
            if field == "incompatible_enabled_records":
                expected_row_fields.add("validation_code")
            if (
                not isinstance(row, Mapping)
                or set(row) != expected_row_fields
                or type(row.get("index")) is not int
                or not 0 <= row["index"] < counts["job_count"]
                or row["index"] in observed_indexes
                or row.get("job_id") is not None
                and (
                    not isinstance(row.get("job_id"), str)
                    or not row["job_id"]
                    or len(row["job_id"]) > 256
                )
                or not isinstance(row.get("record_sha256"), str)
                or _SHA256.fullmatch(row["record_sha256"]) is None
                or field == "incompatible_enabled_records"
                and (
                    not isinstance(row.get("validation_code"), str)
                    or not row["validation_code"].startswith("production_cron_")
                )
            ):
                raise ProductionCronMigrationError(
                    "production_cron_inventory_invalid"
                )
            observed_indexes.add(row["index"])
    if observed_indexes != set(range(counts["job_count"])):
        raise ProductionCronMigrationError("production_cron_inventory_invalid")
    incompatible = inventory["incompatible_enabled_records"]
    template = inventory["migration_plan_payload"].get("continuity_dispositions")
    if not isinstance(template, list):
        raise ProductionCronMigrationError("production_cron_inventory_invalid")
    pending = {
        (row.get("index"), row.get("record_sha256"), row.get("validation_code"))
        for row in template
        if isinstance(row, Mapping) and row.get("disposition") == "pending_review"
    }
    expected_pending = {
        (row.get("index"), row.get("record_sha256"), row.get("validation_code"))
        for row in incompatible
    }
    if not expected_pending.issubset(pending):
        raise ProductionCronMigrationError("production_cron_inventory_invalid")
    return dict(inventory)


def _decision_target(disposition: str, value: Any) -> Mapping[str, Any] | None:
    if disposition == "preserve_inert":
        return None if value is None else _invalid_target()
    if disposition == "migrate_agent":
        if (
            isinstance(value, Mapping)
            and set(value) == {"kind", "replacement_record_sha256"}
            and value.get("kind") == "production_agent_job"
            and isinstance(value.get("replacement_record_sha256"), str)
            and _SHA256.fullmatch(value["replacement_record_sha256"]) is not None
        ):
            return dict(value)
        return _invalid_target()
    if disposition in {"migrate_mechanical", "replaced_by_packaged_rail"}:
        if (
            isinstance(value, Mapping)
            and set(value) == {"kind", "job_id", "package_manifest_sha256"}
            and value.get("kind") == "packaged_mechanical_job"
            and isinstance(value.get("job_id"), str)
            and value.get("job_id")
            and isinstance(value.get("package_manifest_sha256"), str)
            and _SHA256.fullmatch(value["package_manifest_sha256"]) is not None
        ):
            return dict(value)
        return _invalid_target()
    if disposition == "retire_stale":
        if value == {"kind": "retirement_receipt", "required": True}:
            return dict(value)
        return _invalid_target()
    if disposition == "keep_compatible" and value is None:
        return None
    return _invalid_target()


def _invalid_target() -> Mapping[str, Any]:
    raise ProductionCronMigrationError(
        "production_cron_continuity_disposition_target_invalid"
    )


def build_owner_approved_plan(
    inventory: Mapping[str, Any],
    *,
    dispositions: Sequence[Mapping[str, Any]],
    approval_id: str,
    approved_by: str,
) -> dict[str, Any]:
    """Bind exhaustive reviewed continuity dispositions to one inventory."""

    trusted_inventory = validate_inventory(inventory)
    if not isinstance(approval_id, str) or _APPROVAL_ID.fullmatch(approval_id) is None:
        raise ProductionCronMigrationError(
            "production_cron_migration_approval_id_invalid"
        )
    if approved_by != "owner":
        raise ProductionCronMigrationError(
            "production_cron_migration_owner_approval_required"
        )
    if not isinstance(dispositions, Sequence) or isinstance(
        dispositions, (str, bytes)
    ):
        raise ProductionCronMigrationError(
            "production_cron_continuity_dispositions_invalid"
        )
    expected: dict[tuple[int, str], Mapping[str, Any]] = {
        (row["index"], row["record_sha256"]): row
        for row in trusted_inventory["incompatible_enabled_records"]
    }
    legacy = trusted_inventory["legacy_auto_sync"]
    legacy_key: tuple[int, str] | None = None
    if legacy.get("present"):
        legacy_key = (legacy["index"], legacy["record_sha256"])
        expected.setdefault(
            legacy_key,
            {
                "index": legacy["index"],
                "job_id": legacy["job_id"],
                "record_sha256": legacy["record_sha256"],
                "validation_code": None,
            },
        )
    supplied: dict[tuple[int, str], Mapping[str, Any]] = {}
    for decision in dispositions:
        if not isinstance(decision, Mapping) or set(decision) != {
            "index",
            "record_sha256",
            "disposition",
            "target",
        }:
            raise ProductionCronMigrationError(
                "production_cron_continuity_dispositions_invalid"
            )
        key = (decision.get("index"), decision.get("record_sha256"))
        if key not in expected or key in supplied:
            raise ProductionCronMigrationError(
                "production_cron_continuity_dispositions_not_exhaustive"
            )
        disposition = decision.get("disposition")
        if disposition not in _DISPOSITIONS or disposition in {
            "pending_review",
            "keep_compatible",
        }:
            raise ProductionCronMigrationError(
                "production_cron_continuity_disposition_invalid"
            )
        if key == legacy_key:
            if disposition != "replaced_by_packaged_rail":
                raise ProductionCronMigrationError(
                    "production_cron_legacy_auto_sync_disposition_invalid"
                )
        elif disposition == "replaced_by_packaged_rail":
            raise ProductionCronMigrationError(
                "production_cron_continuity_disposition_invalid"
            )
        target = _decision_target(str(disposition), decision.get("target"))
        if disposition == "replaced_by_packaged_rail" and target != {
            "kind": "packaged_mechanical_job",
            "job_id": MECHANICAL_RAIL_JOB_ID,
            "package_manifest_sha256": target.get("package_manifest_sha256"),
        }:
            raise ProductionCronMigrationError(
                "production_cron_legacy_auto_sync_disposition_invalid"
            )
        supplied[key] = {**dict(decision), "target": target}
    if set(supplied) != set(expected):
        raise ProductionCronMigrationError(
            "production_cron_continuity_dispositions_not_exhaustive"
        )
    compatible = [
        {
            **row,
            "validation_code": None,
            "disposition": "keep_compatible",
            "target": None,
        }
        for row in trusted_inventory["compatible_enabled_records"]
    ]
    reviewed = compatible + [
        {
            "index": expected[key]["index"],
            "job_id": expected[key].get("job_id"),
            "record_sha256": expected[key]["record_sha256"],
            "validation_code": expected[key].get("validation_code"),
            "disposition": supplied[key]["disposition"],
            "target": supplied[key]["target"],
        }
        for key in sorted(expected)
    ]
    incompatible_dispositions = [
        row["disposition"]
        for row in reviewed
        if row.get("validation_code") is not None
    ]
    blanket_inert = (
        len(incompatible_dispositions) > 1
        and set(incompatible_dispositions) == {"preserve_inert"}
    )
    inert_apply_executable = (
        not blanket_inert
        and all(
            disposition == "preserve_inert"
            for disposition in incompatible_dispositions
        )
    )
    unsigned = {
        "schema": PLAN_SCHEMA,
        "source_store_sha256": trusted_inventory["source_store_sha256"],
        "inventory_sha256": trusted_inventory["inventory_sha256"],
        "inventory_plan_template_sha256": trusted_inventory[
            "migration_plan_sha256"
        ],
        "continuity_dispositions": reviewed,
        "review_complete": True,
        "cutover_executable": inert_apply_executable,
        "blanket_inert_migration_allowed": False,
        "execute_job_during_migration": False,
        "approval_id": approval_id,
        "approved_by": "owner",
    }
    return {
        **unsigned,
        "owner_approved_plan_sha256": _sha256(_canonical(unsigned)),
    }


def validate_owner_approved_plan(
    inventory: Mapping[str, Any],
    plan: Mapping[str, Any],
    expected_package_manifest_sha256: str,
) -> dict[str, Any]:
    """Pure exact validator for owner/cutover binding; performs no mutation."""

    trusted_inventory = validate_inventory(inventory)
    if _SHA256.fullmatch(expected_package_manifest_sha256 or "") is None:
        raise ProductionCronMigrationError(
            "production_cron_package_manifest_sha256_invalid"
        )
    # The packaged continuity plan supersedes the legacy inert-only schema for
    # the reviewed July 2026 production store.  Keep this dispatch here so all
    # existing cutover/launcher callers use the same validation boundary and
    # do not need a second authority field or a schema-specific router.
    if isinstance(plan, Mapping):
        from gateway import production_cron_continuity_package as packaged

        if plan.get("schema") == packaged.PLAN_SCHEMA:
            try:
                return packaged.validate_packaged_continuity_plan(
                    plan,
                    inventory=trusted_inventory,
                    expected_mechanical_job_package_manifest_sha256=(
                        expected_package_manifest_sha256
                    ),
                    require_executable=True,
                )
            except packaged.ProductionCronContinuityPackageError as exc:
                raise ProductionCronMigrationError(
                    "production_cron_packaged_continuity_plan_invalid"
                ) from exc
    expected_fields = {
        "schema",
        "source_store_sha256",
        "inventory_sha256",
        "inventory_plan_template_sha256",
        "continuity_dispositions",
        "review_complete",
        "cutover_executable",
        "blanket_inert_migration_allowed",
        "execute_job_during_migration",
        "approval_id",
        "approved_by",
        "owner_approved_plan_sha256",
    }
    if (
        not isinstance(plan, Mapping)
        or set(plan) != expected_fields
        or plan.get("schema") != PLAN_SCHEMA
        or plan.get("source_store_sha256")
        != trusted_inventory["source_store_sha256"]
        or plan.get("inventory_sha256") != trusted_inventory["inventory_sha256"]
        or plan.get("inventory_plan_template_sha256")
        != trusted_inventory["migration_plan_sha256"]
        or plan.get("review_complete") is not True
        or plan.get("blanket_inert_migration_allowed") is not False
        or plan.get("execute_job_during_migration") is not False
        or plan.get("approved_by") != "owner"
        or not isinstance(plan.get("approval_id"), str)
        or _APPROVAL_ID.fullmatch(plan["approval_id"]) is None
        or not isinstance(plan.get("owner_approved_plan_sha256"), str)
        or _SHA256.fullmatch(plan["owner_approved_plan_sha256"]) is None
        or _sha256(
            _canonical(
                {
                    key: value
                    for key, value in plan.items()
                    if key != "owner_approved_plan_sha256"
                }
            )
        )
        != plan["owner_approved_plan_sha256"]
        or not isinstance(plan.get("continuity_dispositions"), list)
    ):
        raise ProductionCronMigrationError(
            "production_cron_owner_plan_invalid"
        )
    expected: dict[tuple[int, str], Mapping[str, Any]] = {
        (row["index"], row["record_sha256"]): {
            **row,
            "record_class": "compatible_enabled",
        }
        for row in trusted_inventory["compatible_enabled_records"]
    }
    expected.update(
        {
            (row["index"], row["record_sha256"]): {
                **row,
                "record_class": "incompatible_enabled",
            }
            for row in trusted_inventory["incompatible_enabled_records"]
        }
    )
    legacy = trusted_inventory["legacy_auto_sync"]
    legacy_key: tuple[int, str] | None = None
    if legacy.get("present"):
        legacy_key = (legacy["index"], legacy["record_sha256"])
        expected.setdefault(
            legacy_key,
            {
                "index": legacy["index"],
                "job_id": legacy["job_id"],
                "record_sha256": legacy["record_sha256"],
                "validation_code": None,
                "record_class": "disabled_legacy_auto_sync",
            },
        )
    observed: dict[tuple[int, str], Mapping[str, Any]] = {}
    for row in plan["continuity_dispositions"]:
        if not isinstance(row, Mapping) or set(row) != {
            "index",
            "job_id",
            "record_sha256",
            "validation_code",
            "disposition",
            "target",
        }:
            raise ProductionCronMigrationError(
                "production_cron_continuity_dispositions_invalid"
            )
        key = (row.get("index"), row.get("record_sha256"))
        source = expected.get(key)
        if source is None or key in observed or row.get("job_id") != source.get(
            "job_id"
        ):
            raise ProductionCronMigrationError(
                "production_cron_continuity_dispositions_not_exhaustive"
            )
        disposition = row.get("disposition")
        if source["record_class"] == "compatible_enabled":
            if (
                row.get("validation_code") is not None
                or disposition != "keep_compatible"
                or row.get("target") is not None
            ):
                raise ProductionCronMigrationError(
                    "production_cron_continuity_disposition_invalid"
                )
        else:
            if row.get("validation_code") != source.get("validation_code"):
                raise ProductionCronMigrationError(
                    "production_cron_continuity_disposition_invalid"
                )
            if disposition not in _DISPOSITIONS or disposition in {
                "pending_review",
                "keep_compatible",
            }:
                raise ProductionCronMigrationError(
                    "production_cron_continuity_disposition_invalid"
                )
            target = _decision_target(str(disposition), row.get("target"))
            if target != row.get("target"):
                raise ProductionCronMigrationError(
                    "production_cron_continuity_disposition_target_invalid"
                )
            if key == legacy_key:
                if (
                    disposition != "replaced_by_packaged_rail"
                    or target
                    != {
                        "kind": "packaged_mechanical_job",
                        "job_id": MECHANICAL_RAIL_JOB_ID,
                        "package_manifest_sha256": expected_package_manifest_sha256,
                    }
                ):
                    raise ProductionCronMigrationError(
                        "production_cron_legacy_auto_sync_disposition_invalid"
                    )
            elif disposition == "replaced_by_packaged_rail":
                raise ProductionCronMigrationError(
                    "production_cron_continuity_disposition_invalid"
                )
            if disposition == "migrate_mechanical" and target.get(
                "package_manifest_sha256"
            ) != expected_package_manifest_sha256:
                raise ProductionCronMigrationError(
                    "production_cron_continuity_disposition_target_invalid"
                )
        observed[key] = row
    if set(observed) != set(expected):
        raise ProductionCronMigrationError(
            "production_cron_continuity_dispositions_not_exhaustive"
        )
    incompatible_dispositions = [
        observed[key]["disposition"]
        for key, source in expected.items()
        if source["record_class"] == "incompatible_enabled"
    ]
    blanket = (
        len(incompatible_dispositions) > 1
        and set(incompatible_dispositions) == {"preserve_inert"}
    )
    expected_executable = (
        not blanket
        and all(
            disposition == "preserve_inert"
            for disposition in incompatible_dispositions
        )
    )
    if plan.get("cutover_executable") is not expected_executable:
        raise ProductionCronMigrationError(
            "production_cron_owner_plan_executable_state_invalid"
        )
    return dict(plan)


def _gateway_inactive() -> bool:
    command = [
        "/usr/bin/systemctl",
        "show",
        GATEWAY_UNIT,
        "--property=MainPID",
        "--value",
    ]
    try:
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and result.stdout.strip() == "0"


def _atomic_replace_preserving_metadata(
    path: Path,
    value: bytes,
    metadata: os.stat_result,
) -> None:
    descriptor, temporary = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        os.fchmod(descriptor, stat.S_IMODE(metadata.st_mode))
        if os.geteuid() == 0:  # windows-footgun: ok — Linux production/canary boundary
            os.fchown(descriptor, metadata.st_uid, metadata.st_gid)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _write_exclusive_or_verify(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    if path.exists():
        existing, _metadata = _stable_read(path)
        if existing != value:
            raise ProductionCronMigrationError(
                "production_cron_migration_evidence_replay_mismatch"
            )
        return
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _build_receipt(
    *,
    created_at: str,
    owner_plan_sha256: str,
    inventory_sha256: str,
    source_store_sha256: str,
    target_store_sha256: str,
    archive: Path,
    migrated: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    unsigned = {
        "schema": RECEIPT_SCHEMA,
        "created_at": created_at,
        "owner_approved_plan_sha256": owner_plan_sha256,
        "inventory_sha256": inventory_sha256,
        "source_store_sha256": source_store_sha256,
        "target_store_sha256": target_store_sha256,
        "archive_path": str(archive),
        "archive_sha256": source_store_sha256,
        "migrated_record_count": len(migrated),
        "migrated_records": list(migrated),
        "compatible_records_unchanged": True,
        "disabled_records_unchanged": True,
        "records_deleted": False,
        "job_executed": False,
        "provider_or_model_invoked": False,
        "discord_delivery_attempted": False,
        "prompt_or_script_content_recorded": False,
    }
    return {**unsigned, "receipt_sha256": _sha256(_canonical(unsigned))}


def _publish_receipt(evidence_root: Path, receipt: Mapping[str, Any]) -> None:
    encoded = _canonical(dict(receipt)) + b"\n"
    _write_exclusive_or_verify(
        evidence_root
        / "receipts"
        / f"migration-{receipt['receipt_sha256']}.json",
        encoded,
    )
    latest = evidence_root / "latest.json"
    latest.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary = tempfile.mkstemp(
        dir=str(latest.parent), prefix=".latest.", suffix=".tmp"
    )
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, latest)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def apply_inert_migration(
    *,
    jobs_path: Path,
    inventory: Mapping[str, Any],
    owner_approved_plan: Mapping[str, Any],
    expected_owner_plan_sha256: str,
    expected_package_manifest_sha256: str,
    evidence_root: Path = DEFAULT_EVIDENCE_ROOT,
    gateway_inactive: Callable[[], bool] = _gateway_inactive,
) -> dict[str, Any]:
    """Preserve incompatible records inertly; never execute or reinterpret them."""

    trusted_inventory = validate_inventory(inventory)
    validate_owner_approved_plan(
        trusted_inventory,
        owner_approved_plan,
        expected_package_manifest_sha256,
    )
    if (
        not isinstance(expected_owner_plan_sha256, str)
        or _SHA256.fullmatch(expected_owner_plan_sha256) is None
        or not isinstance(owner_approved_plan, Mapping)
        or owner_approved_plan.get("owner_approved_plan_sha256")
        != expected_owner_plan_sha256
        or _sha256(
            _canonical(
                {
                    key: value
                    for key, value in owner_approved_plan.items()
                    if key != "owner_approved_plan_sha256"
                }
            )
        )
        != expected_owner_plan_sha256
        or owner_approved_plan.get("approved_by") != "owner"
        or owner_approved_plan.get("inventory_sha256")
        != trusted_inventory.get("inventory_sha256")
        or owner_approved_plan.get("inventory_plan_template_sha256")
        != trusted_inventory.get("migration_plan_sha256")
        or owner_approved_plan.get("schema") != PLAN_SCHEMA
        or owner_approved_plan.get("review_complete") is not True
        or owner_approved_plan.get("cutover_executable") is not True
        or owner_approved_plan.get("blanket_inert_migration_allowed") is not False
    ):
        raise ProductionCronMigrationError(
            "production_cron_owner_plan_invalid"
        )
    continuity = owner_approved_plan.get("continuity_dispositions")
    if not isinstance(continuity, list):
        raise ProductionCronMigrationError(
            "production_cron_continuity_dispositions_invalid"
        )
    preserve_rows = [
        row
        for row in continuity
        if isinstance(row, Mapping)
        and row.get("validation_code") is not None
        and row.get("disposition") == "preserve_inert"
        and row.get("target") is None
    ]
    incompatible_rows = [
        row
        for row in continuity
        if isinstance(row, Mapping) and row.get("validation_code") is not None
    ]
    if (
        not preserve_rows
        or len(preserve_rows) != len(incompatible_rows)
        or len(preserve_rows) > 1
    ):
        raise ProductionCronMigrationError(
            "production_cron_inert_apply_not_explicit_or_blanket_forbidden"
        )
    if not gateway_inactive():
        raise ProductionCronMigrationError(
            "production_cron_gateway_must_be_inactive"
        )
    lock_path = jobs_path.parent / ".jobs.lock"
    lock_descriptor = os.open(
        lock_path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ProductionCronMigrationError(
                "production_cron_store_lock_unavailable"
            ) from exc
        raw, metadata = _stable_read(jobs_path)
        source_sha = str(owner_approved_plan.get("source_store_sha256") or "")
        if _SHA256.fullmatch(source_sha) is None:
            raise ProductionCronMigrationError(
                "production_cron_owner_plan_invalid"
            )
        archive = evidence_root / "stores" / f"jobs-{source_sha}.json"
        prepared_path = (
            evidence_root
            / "prepared"
            / f"migration-{expected_owner_plan_sha256}.json"
        )
        if _sha256(raw) != source_sha:
            if not prepared_path.exists():
                raise ProductionCronMigrationError(
                    "production_cron_store_digest_drifted"
                )
            prepared_raw, _prepared_metadata = _stable_read(prepared_path)
            try:
                prepared = json.loads(prepared_raw.decode("ascii", errors="strict"))
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise ProductionCronMigrationError(
                    "production_cron_migration_prepared_invalid"
                ) from exc
            if (
                not isinstance(prepared, dict)
                or prepared.get("schema") != PREPARED_SCHEMA
                or prepared.get("owner_approved_plan_sha256")
                != expected_owner_plan_sha256
                or prepared.get("source_store_sha256") != source_sha
                or prepared.get("target_store_sha256") != _sha256(raw)
                or prepared.get("archive_path") != str(archive)
                or prepared.get("archive_sha256") != source_sha
                or not isinstance(prepared.get("migrated_records"), list)
                or not isinstance(prepared.get("created_at"), str)
            ):
                raise ProductionCronMigrationError(
                    "production_cron_store_digest_drifted"
                )
            archive_raw, _archive_metadata = _stable_read(archive)
            if _sha256(archive_raw) != source_sha:
                raise ProductionCronMigrationError(
                    "production_cron_migration_archive_invalid"
                )
            receipt = _build_receipt(
                created_at=prepared["created_at"],
                owner_plan_sha256=expected_owner_plan_sha256,
                inventory_sha256=str(inventory["inventory_sha256"]),
                source_store_sha256=source_sha,
                target_store_sha256=_sha256(raw),
                archive=archive,
                migrated=prepared["migrated_records"],
            )
            _publish_receipt(evidence_root, receipt)
            return receipt
        fresh_inventory = inventory_jobs_bytes(raw)
        if fresh_inventory.get("inventory_sha256") != inventory.get(
            "inventory_sha256"
        ):
            raise ProductionCronMigrationError(
                "production_cron_inventory_drifted"
            )
        payload = _parse_store(raw)
        expected_records = {
            (item["index"], item["record_sha256"]): item
            for item in preserve_rows
        }
        migrated: list[dict[str, Any]] = []
        jobs = list(payload["jobs"])
        for index, job in enumerate(jobs):
            identity = _record_identity(index, job)
            key = (index, identity["record_sha256"])
            if key not in expected_records:
                continue
            if job.get("enabled") is False:
                raise ProductionCronMigrationError(
                    "production_cron_migration_record_state_drifted"
                )
            updated = dict(job)
            updated["enabled"] = False
            updated["state"] = "paused"
            jobs[index] = updated
            migrated.append(identity)
        if len(migrated) != len(expected_records):
            raise ProductionCronMigrationError(
                "production_cron_migration_record_set_drifted"
            )
        _write_exclusive_or_verify(archive, raw)
        updated_payload = dict(payload)
        updated_payload["jobs"] = jobs
        migration_time = str(inventory.get("created_at") or "")
        if not migration_time:
            raise ProductionCronMigrationError(
                "production_cron_inventory_created_at_invalid"
            )
        updated_payload["updated_at"] = migration_time
        updated_raw = json.dumps(
            updated_payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8") + b"\n"
        # Prove the exact post-state before the only live-store mutation.
        post_inventory = inventory_jobs_bytes(updated_raw)
        if post_inventory["incompatible_enabled_count"] != 0:
            raise ProductionCronMigrationError(
                "production_cron_migration_post_state_invalid"
            )
        validate_production_cron_jobs(updated_payload["jobs"])
        prepared = {
            "schema": PREPARED_SCHEMA,
            "created_at": migration_time,
            "owner_approved_plan_sha256": expected_owner_plan_sha256,
            "source_store_sha256": _sha256(raw),
            "target_store_sha256": _sha256(updated_raw),
            "archive_path": str(archive),
            "archive_sha256": _sha256(raw),
            "migrated_records": migrated,
            "job_executed": False,
        }
        _write_exclusive_or_verify(prepared_path, _canonical(prepared) + b"\n")
        _atomic_replace_preserving_metadata(jobs_path, updated_raw, metadata)
        receipt = _build_receipt(
            created_at=migration_time,
            owner_plan_sha256=expected_owner_plan_sha256,
            inventory_sha256=str(inventory["inventory_sha256"]),
            source_store_sha256=_sha256(raw),
            target_store_sha256=_sha256(updated_raw),
            archive=archive,
            migrated=migrated,
        )
        _publish_receipt(evidence_root, receipt)
        return receipt
    finally:
        os.close(lock_descriptor)


def restore_inert_migration(
    *,
    jobs_path: Path,
    migration_receipt: Mapping[str, Any],
    expected_receipt_sha256: str,
    evidence_root: Path = DEFAULT_EVIDENCE_ROOT,
    gateway_inactive: Callable[[], bool] = _gateway_inactive,
) -> dict[str, Any]:
    """Restore the exact archived pre-migration store while gateway is stopped."""

    if (
        not isinstance(migration_receipt, Mapping)
        or migration_receipt.get("schema") != RECEIPT_SCHEMA
        or migration_receipt.get("receipt_sha256") != expected_receipt_sha256
        or _SHA256.fullmatch(expected_receipt_sha256 or "") is None
        or _sha256(
            _canonical(
                {
                    key: value
                    for key, value in migration_receipt.items()
                    if key != "receipt_sha256"
                }
            )
        )
        != expected_receipt_sha256
    ):
        raise ProductionCronMigrationError(
            "production_cron_migration_receipt_invalid"
        )
    source_sha = migration_receipt.get("source_store_sha256")
    target_sha = migration_receipt.get("target_store_sha256")
    if (
        not isinstance(source_sha, str)
        or _SHA256.fullmatch(source_sha) is None
        or not isinstance(target_sha, str)
        or _SHA256.fullmatch(target_sha) is None
    ):
        raise ProductionCronMigrationError(
            "production_cron_migration_receipt_invalid"
        )
    archive = evidence_root / "stores" / f"jobs-{source_sha}.json"
    if migration_receipt.get("archive_path") != str(archive):
        raise ProductionCronMigrationError(
            "production_cron_migration_receipt_invalid"
        )
    if not gateway_inactive():
        raise ProductionCronMigrationError(
            "production_cron_gateway_must_be_inactive"
        )
    lock_path = jobs_path.parent / ".jobs.lock"
    lock_descriptor = os.open(
        lock_path,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ProductionCronMigrationError(
                "production_cron_store_lock_unavailable"
            ) from exc
        current, metadata = _stable_read(jobs_path)
        archive_raw, _archive_metadata = _stable_read(archive)
        if _sha256(archive_raw) != source_sha:
            raise ProductionCronMigrationError(
                "production_cron_migration_archive_invalid"
            )
        current_sha = _sha256(current)
        if current_sha not in {target_sha, source_sha}:
            raise ProductionCronMigrationError(
                "production_cron_rollback_store_drifted"
            )
        if current_sha == target_sha:
            _atomic_replace_preserving_metadata(jobs_path, archive_raw, metadata)
        unsigned = {
            "schema": ROLLBACK_SCHEMA,
            "created_at": migration_receipt["created_at"],
            "migration_receipt_sha256": expected_receipt_sha256,
            "restored_store_sha256": source_sha,
            "replaced_store_sha256": target_sha,
            "archive_path": str(archive),
            "outcome": "restored_or_already_exact",
            "gateway_start_allowed": False,
            "reason": "restored_store_requires_new_production_cron_inventory",
            "job_executed": False,
        }
        receipt = {
            **unsigned,
            "rollback_receipt_sha256": _sha256(_canonical(unsigned)),
        }
        _write_exclusive_or_verify(
            evidence_root
            / "rollbacks"
            / f"rollback-{receipt['rollback_receipt_sha256']}.json",
            _canonical(receipt) + b"\n",
        )
        return receipt
    finally:
        os.close(lock_descriptor)


def _read_json_file(path: Path) -> dict[str, Any]:
    raw, _metadata = _stable_read(path)
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ProductionCronMigrationError(
            "production_cron_migration_input_invalid"
        ) from exc
    if not isinstance(value, dict):
        raise ProductionCronMigrationError(
            "production_cron_migration_input_invalid"
        )
    return value


def _write_output(path: Path, value: Mapping[str, Any]) -> None:
    encoded = _canonical(dict(value)) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inventory or inertly migrate production cron jobs"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    inventory = subparsers.add_parser("inventory")
    inventory.add_argument("--jobs", type=Path, default=DEFAULT_JOBS_PATH)
    inventory.add_argument("--output", type=Path, required=True)
    apply = subparsers.add_parser("apply-inert-migration")
    apply.add_argument("--jobs", type=Path, default=DEFAULT_JOBS_PATH)
    apply.add_argument("--inventory", type=Path, required=True)
    apply.add_argument("--owner-approved-plan", type=Path, required=True)
    apply.add_argument("--expected-owner-plan-sha256", required=True)
    apply.add_argument("--expected-package-manifest-sha256", required=True)
    apply.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    restore = subparsers.add_parser("restore-inert-migration")
    restore.add_argument("--jobs", type=Path, default=DEFAULT_JOBS_PATH)
    restore.add_argument("--migration-receipt", type=Path, required=True)
    restore.add_argument("--expected-receipt-sha256", required=True)
    restore.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "inventory":
        result = inventory_jobs_file(args.jobs)
        _write_output(args.output, result)
        return 0 if not result["migration_required_for_gateway_startup"] else 2
    if args.command == "apply-inert-migration":
        apply_inert_migration(
            jobs_path=args.jobs,
            inventory=_read_json_file(args.inventory),
            owner_approved_plan=_read_json_file(args.owner_approved_plan),
            expected_owner_plan_sha256=args.expected_owner_plan_sha256,
            expected_package_manifest_sha256=args.expected_package_manifest_sha256,
            evidence_root=args.evidence_root,
        )
        return 0
    if args.command == "restore-inert-migration":
        restore_inert_migration(
            jobs_path=args.jobs,
            migration_receipt=_read_json_file(args.migration_receipt),
            expected_receipt_sha256=args.expected_receipt_sha256,
            evidence_root=args.evidence_root,
        )
        return 0
    raise ProductionCronMigrationError("production_cron_migration_command_invalid")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ProductionCronMigrationError as exc:
        print(str(exc), file=os.sys.stderr)
        raise SystemExit(1) from None


__all__ = [
    "DEFAULT_EVIDENCE_ROOT",
    "DEFAULT_JOBS_PATH",
    "INVENTORY_SCHEMA",
    "LEGACY_AUTO_SYNC_JOB_ID",
    "LEGACY_AUTO_SYNC_JOB_NAME",
    "MECHANICAL_RAIL_JOB_ID",
    "PLAN_SCHEMA",
    "ProductionCronMigrationError",
    "RECEIPT_SCHEMA",
    "ROLLBACK_SCHEMA",
    "apply_inert_migration",
    "build_owner_approved_plan",
    "inventory_jobs_bytes",
    "inventory_jobs_file",
    "restore_inert_migration",
    "validate_inventory",
    "validate_owner_approved_plan",
]

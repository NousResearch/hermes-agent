"""Bind reviewed legacy cron continuity to exact replacement artifacts.

This module does not mutate ``jobs.json`` and does not execute a collector.  It
joins three already-reviewed identities:

* the redaction-safe production observation;
* the exact trusted-collector package; and
* normalized primary-model replacement records.

All selection is by exact source job ID.  Packet content, job names, prompts,
and command output never influence a disposition.  Five previously blocked
records have an architecture-mandated raw-collector + primary-model target.
The single local-Mac maintenance record is preserved byte-for-byte and made
inert because its referenced files are absent in production.  That reversible
fail-closed disposition does not claim the job was discontinued and does not
block cutover; rebuilding it as a bounded Mac agent or retiring the preserved
record remains an explicit post-cutover owner follow-up.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from gateway import production_cron_continuity_review as review
from gateway import production_cron_migration
from gateway.operational_edge_catalog import required_cron_operations
from gateway.operational_edge_readiness import (
    OperationalEdgeReadinessError,
    validate_operational_edge_readiness,
)
from gateway.production_model_sovereignty_runtime import (
    ProductionContractError,
    validate_production_cron_jobs,
)
from gateway.support_ops_team_registry import (
    SKYVISION_CONTROL_TOWER_CHANNEL_ID,
    SKYVISION_GUILD_ID,
)
from ops.muncho.runtime import trusted_cron_collector_rail as collector_rail
from ops.muncho.runtime import mechanical_job_rail


PLAN_SCHEMA = "muncho-production-cron-packaged-continuity-plan.v4"
REPLACEMENT_BUNDLE_SCHEMA = "muncho-production-cron-agent-replacements.v1"
ARTIFACT_INDEX_SCHEMA = "muncho-production-cron-continuity-artifacts.v1"

ARTIFACT_INDEX_RELATIVE_PATH = Path("cron/continuity-artifacts.json")
PLAN_RELATIVE_PATH = Path("cron/continuity-plan.json")
REPLACEMENT_BUNDLE_RELATIVE_PATH = Path(
    "cron/agent-replacements.private.json"
)
COLLECTOR_MANIFEST_RELATIVE_PATH = Path(
    "cron/trusted-collector/manifest.json"
)
CUTOVER_RUNTIME_RELATIVE_PATH = Path(
    "gateway/production_cron_cutover_runtime.py"
)
CUTOVER_ENTRYPOINT_RELATIVE_PATH = Path(
    "scripts/canary/production_cron_cutover_entrypoint.py"
)

DISPOSITION_KEEP = "keep_compatible"
DISPOSITION_AGENT = "migrate_primary_agent"
DISPOSITION_COLLECTOR = "migrate_trusted_collector"
DISPOSITION_PRESERVE = "preserve_inert"

OWNER_JOB_ID = "fecd0675f91e"
OWNER_FOLLOWUP_CODE = "rebuild_local_mac_agent_or_retire_preserved_record"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DISCORD_SNOWFLAKE = re.compile(r"^[0-9]{17,20}$")
VOICE_CONTEXT_THREAD_ID = "1524321461714681976"
REVIEWED_GUILD_TARGETS: Mapping[str, Mapping[str, Any]] = {
    "06ef64d72891": {
        "platform": "discord",
        "guild_id": SKYVISION_GUILD_ID,
        "channel_id": SKYVISION_CONTROL_TOWER_CHANNEL_ID,
        "thread_id": None,
        "target_type": "guild_channel",
        "authorization_source": "exact_reviewed_job_guild_acl_projection",
    },
    "e62f55ca93ca": {
        "platform": "discord",
        "guild_id": SKYVISION_GUILD_ID,
        "channel_id": VOICE_CONTEXT_THREAD_ID,
        "parent_channel_id": SKYVISION_CONTROL_TOWER_CHANNEL_ID,
        "thread_id": VOICE_CONTEXT_THREAD_ID,
        "target_type": "guild_thread",
        "authorization_source": "exact_reviewed_job_guild_acl_projection",
    },
}
_DYNAMIC_FIELDS = frozenset(
    {
        "fire_claim",
        "last_delivery_error",
        "last_delivery_status",
        "last_delivery_confirmed_at",
        "last_error",
        "last_run_at",
        "last_status",
        "next_run_at",
        "paused_at",
        "paused_reason",
        "state",
    }
)


class ProductionCronContinuityPackageError(RuntimeError):
    """Stable, non-secret package-plan failure."""


@dataclass(frozen=True)
class ContinuityPackageBuild:
    plan: Mapping[str, Any]
    replacement_bundle: Mapping[str, Any]
    inventory: Mapping[str, Any]


@dataclass(frozen=True)
class HostContinuityDerivation:
    build: ContinuityPackageBuild
    inventory: Mapping[str, Any]


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
        raise ProductionCronContinuityPackageError(
            "production_cron_continuity_json_invalid"
        ) from exc


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _parse_store(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ProductionCronContinuityPackageError(
            "production_cron_continuity_store_invalid"
        ) from exc
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("jobs"), list)
        or any(not isinstance(job, dict) for job in value["jobs"])
    ):
        raise ProductionCronContinuityPackageError(
            "production_cron_continuity_store_invalid"
        )
    return value


def _target_from_delivery(
    source: Mapping[str, Any],
    *,
    source_job_id: str | None = None,
) -> dict[str, Any] | None:
    """Project an exact reviewed guild-ACL target, never infer authority.

    Legacy ``origin``/``deliver`` fields are historical provenance. A numeric
    Discord shape never grants a target. Only the two owner-reviewed job
    identities project to their exact existing production lane/thread. The
    connector still re-proves live bot ACL before every read or send. This is a
    permission boundary, not semantic routing; GPT still decides whether any
    message is useful.
    """

    job_id = str(source_job_id or source.get("id") or "").strip()
    if source.get("id") != job_id or job_id not in REVIEWED_GUILD_TARGETS:
        return None
    return copy.deepcopy(dict(REVIEWED_GUILD_TARGETS[job_id]))


def _source_packet_path(source_job_id: str) -> str:
    return str(collector_rail.PACKET_ROOT / source_job_id / "latest.json")


def _agent_note() -> str:
    return (
        "\n\nProduction tool note: use discord_guild_history for the exact "
        "ACL-authorized guild channel/thread stated above. Discord DMs, group "
        "DMs, and type-12 private threads are forbidden. GPT-5.6-sol is the "
        "semantic authority; do not "
        "delegate interpretation to a classifier or routing rule. Record "
        "route_back.sent only after a real connector send and readback receipt; "
        "otherwise record route_back.blocked and explain the blocker."
    )


def _collector_prompt(source: Mapping[str, Any], spec: collector_rail.CollectorSpec) -> str:
    target = _target_from_delivery(source, source_job_id=spec.source_job_id)
    target_text = json.dumps(target, ensure_ascii=True, sort_keys=True)
    authorized_history = (
        " When guild conversation context is needed, read only that exact "
        "ACL-authorized channel/thread with discord_guild_history."
        if target is not None
        else ""
    )
    voice = ""
    if spec.mode == "voice_stage":
        voice = (
            " The voice packet cursor must remain unchanged unless the "
            "connector send receipt, connector readback receipt, and Canonical "
            "route_back.sent event all exist for the exact packet. On any "
            "problem append route_back.blocked; never claim sent."
        )
    return (
        "You are the pinned production GPT-5.6-sol agent for one reviewed "
        "scheduled task. Read the exact self-digesting raw packet at "
        f"{_source_packet_path(spec.source_job_id)} with the file tool. Verify "
        f"schema={collector_rail.PACKET_SCHEMA}, source_job_id="
        f"{spec.source_job_id}, and its packet_sha256 before using it. The "
        "collector performed no semantic judgment: inspect the evidence "
        "yourself, decide what it means, and choose the next safe step. Use "
        "canonical_brain_query when prior case state matters and append the "
        "model-authored review outcome to Canonical Brain. Do not infer from "
        "filenames alone. Do not perform a dangerous mutation without the "
        "applicable owner/passkey approval. "
        f"Approved guild target, if any: {target_text}. Legacy delivery fields "
        "are ineligible historical provenance and must never be used as a "
        "send/read authorization. If a guild "
        "message is useful, use the Canonical route-back operation so delivery "
        "is receipt-backed; otherwise keep the run local."
        f"{authorized_history}"
        f"{voice}"
    )


def _normalized_source(source: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        key: copy.deepcopy(value)
        for key, value in source.items()
        if key not in _DYNAMIC_FIELDS
    }
    result["enabled"] = True
    result["state"] = "scheduled"
    for field in (
        "fallback_model",
        "fallback_models",
        "fallback_provider",
        "fallback_providers",
    ):
        result.pop(field, None)
    return result


def build_agent_replacement_record(
    source: Mapping[str, Any],
    *,
    catalog_entry: review.ReviewDisposition | None = None,
    collector_spec: collector_rail.CollectorSpec | None = None,
) -> dict[str, Any]:
    """Build one actual persisted primary-model record, never a classifier."""

    if (catalog_entry is None) == (collector_spec is None):
        raise ProductionCronContinuityPackageError(
            "production_cron_replacement_kind_invalid"
        )
    job = _normalized_source(source)
    job["no_agent"] = False
    job["provider"] = review.PRIMARY_PROVIDER
    job["model"] = review.PRIMARY_MODEL
    job["base_url"] = None
    job["script"] = None
    job["workdir"] = None
    if catalog_entry is not None:
        if catalog_entry.disposition != review.DISPOSITION_AGENT:
            raise ProductionCronContinuityPackageError(
                "production_cron_replacement_kind_invalid"
            )
        prompt = source.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ProductionCronContinuityPackageError(
                "production_cron_replacement_prompt_invalid"
            )
        job["prompt"] = prompt.rstrip() + _agent_note()
        job["enabled_toolsets"] = list(catalog_entry.required_toolsets or ())
        job["deliver"] = "origin"
    else:
        assert collector_spec is not None
        if not collector_spec.model_review_required:
            raise ProductionCronContinuityPackageError(
                "production_cron_replacement_not_required"
            )
        job["prompt"] = _collector_prompt(source, collector_spec)
        job["enabled_toolsets"] = ["file", "canonical_brain"]
        approved_target = _target_from_delivery(
            source,
            source_job_id=collector_spec.source_job_id,
        )
        if approved_target is not None:
            job["enabled_toolsets"].append(review.GUILD_HISTORY_TOOLSET)
            source_origin = source.get("origin")
            source_user_id = (
                source_origin.get("user_id")
                if isinstance(source_origin, Mapping)
                else None
            )
            is_thread = approved_target["target_type"] == "guild_thread"
            job["origin"] = {
                "platform": "discord",
                "chat_id": (
                    approved_target["parent_channel_id"]
                    if is_thread
                    else approved_target["channel_id"]
                ),
                "chat_name": "control-tower",
                "thread_id": (
                    approved_target["channel_id"] if is_thread else None
                ),
                "user_id": (
                    source_user_id
                    if isinstance(source_user_id, str)
                    and _DISCORD_SNOWFLAKE.fullmatch(source_user_id)
                    else None
                ),
            }
        # Guild delivery is a model tool action through the privileged
        # connector, never an automatic no-agent/direct transport.
        job["deliver"] = "local"
    try:
        validate_production_cron_jobs([job])
    except ProductionContractError as exc:
        raise ProductionCronContinuityPackageError(
            "production_cron_replacement_record_invalid"
        ) from exc
    return job


def _replacement_bundle(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [copy.deepcopy(dict(item)) for item in records]
    unsigned = {
        "schema": REPLACEMENT_BUNDLE_SCHEMA,
        "records": rows,
        "record_count": len(rows),
        "all_primary_route": True,
        "local_script_present": False,
        "workdir_present": False,
    }
    return {
        **unsigned,
        "bundle_sha256": _sha256(_canonical(unsigned)),
    }


def validate_replacement_bundle(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the private, exact persisted replacement records."""

    expected_fields = {
        "schema",
        "records",
        "record_count",
        "all_primary_route",
        "local_script_present",
        "workdir_present",
        "bundle_sha256",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != expected_fields
        or value.get("schema") != REPLACEMENT_BUNDLE_SCHEMA
        or not isinstance(value.get("records"), list)
        or type(value.get("record_count")) is not int
        or value.get("record_count") != len(value["records"])
        or value.get("all_primary_route") is not True
        or value.get("local_script_present") is not False
        or value.get("workdir_present") is not False
        or _SHA256.fullmatch(str(value.get("bundle_sha256") or "")) is None
        or _sha256(
            _canonical(
                {key: item for key, item in value.items() if key != "bundle_sha256"}
            )
        )
        != value.get("bundle_sha256")
    ):
        raise ProductionCronContinuityPackageError(
            "production_cron_replacement_bundle_invalid"
        )
    observed: set[str] = set()
    for record in value["records"]:
        if (
            not isinstance(record, Mapping)
            or not isinstance(record.get("id"), str)
            or not record["id"]
            or record["id"] in observed
            or record.get("no_agent") is not False
            or record.get("provider") != review.PRIMARY_PROVIDER
            or record.get("model") != review.PRIMARY_MODEL
            or record.get("base_url") is not None
            or record.get("script") is not None
            or record.get("workdir") is not None
        ):
            raise ProductionCronContinuityPackageError(
                "production_cron_replacement_bundle_invalid"
            )
        try:
            validate_production_cron_jobs([record])
        except ProductionContractError as exc:
            raise ProductionCronContinuityPackageError(
                "production_cron_replacement_bundle_invalid"
            ) from exc
        observed.add(record["id"])
    return copy.deepcopy(dict(value))


def build_packaged_continuity_plan(
    *,
    source_store: bytes,
    collector_package: Mapping[str, Any],
    collector_execution_readiness: Mapping[str, Any],
    operational_edge_readiness: Mapping[str, Any],
    mechanical_job_package_manifest_sha256: str,
    cutover_runtime_sha256: str,
    cutover_entrypoint_sha256: str,
    expected_boot_id_sha256: str | None = None,
    now_unix: int | None = None,
) -> ContinuityPackageBuild:
    """Build an exhaustive plan and private replacement-record bundle."""

    package = collector_rail.validate_package_manifest(collector_package)
    try:
        operational_readiness = validate_operational_edge_readiness(
            operational_edge_readiness,
            revision=package["release_revision"],
            required_jobs=required_cron_operations(),
            expected_boot_id_sha256=expected_boot_id_sha256,
            now_unix=now_unix,
        )
        execution_readiness = collector_rail.validate_execution_readiness(
            collector_execution_readiness,
            manifest=package,
            operational_edge_receipt=operational_readiness,
            expected_boot_id_sha256=expected_boot_id_sha256,
            now_unix=now_unix,
        )
    except (
        OperationalEdgeReadinessError,
        collector_rail.TrustedCronCollectorError,
    ) as exc:
        raise ProductionCronContinuityPackageError(
            "production_cron_execution_readiness_invalid"
        ) from exc
    if execution_readiness["activation_ready"] is not True:
        raise ProductionCronContinuityPackageError(
            "production_cron_execution_readiness_incomplete"
        )
    if any(
        _SHA256.fullmatch(digest or "") is None
        for digest in (
            mechanical_job_package_manifest_sha256,
            cutover_runtime_sha256,
            cutover_entrypoint_sha256,
        )
    ):
        raise ProductionCronContinuityPackageError(
            "production_cron_mechanical_package_invalid"
        )
    observation = review.observe_enabled_jobs_bytes(source_store)
    review_plan = review.build_owner_review_plan(observation)
    inventory = production_cron_migration.inventory_jobs_bytes(source_store)
    store = _parse_store(source_store)
    enabled = {
        job.get("id"): job for job in store["jobs"] if job.get("enabled") is not False
    }
    if set(enabled) != {row["job_id"] for row in review_plan["records"]}:
        raise ProductionCronContinuityPackageError(
            "production_cron_continuity_source_not_exhaustive"
        )
    review_catalog = {item.job_id: item for item in review.REVIEW_CATALOG}
    collector_catalog = {
        item.source_job_id: item for item in collector_rail.COLLECTOR_SPECS
    }
    package_units = package["units"]
    replacements: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for reviewed in review_plan["records"]:
        job_id = reviewed["job_id"]
        source = enabled[job_id]
        entry = review_catalog[job_id]
        if entry.disposition == review.DISPOSITION_KEEP:
            disposition = DISPOSITION_KEEP
            target: dict[str, Any] | None = {
                "kind": "retained_compatible_record",
                "record_sha256": reviewed["record_sha256"],
            }
            owner_required = False
        elif entry.disposition == review.DISPOSITION_AGENT:
            replacement = build_agent_replacement_record(
                source,
                catalog_entry=entry,
            )
            replacements.append(replacement)
            disposition = DISPOSITION_AGENT
            target = {
                "kind": "primary_agent_record",
                "replacement_record_sha256": _sha256(_canonical(replacement)),
                "provider": review.PRIMARY_PROVIDER,
                "model": review.PRIMARY_MODEL,
                "enabled_toolsets": list(entry.required_toolsets or ()),
            }
            owner_required = False
        elif job_id in collector_catalog:
            spec = collector_catalog[job_id]
            approved_guild_target = _target_from_delivery(
                source,
                source_job_id=job_id,
            )
            replacement_sha: str | None = None
            if spec.model_review_required:
                replacement = build_agent_replacement_record(
                    source,
                    collector_spec=spec,
                )
                replacements.append(replacement)
                replacement_sha = _sha256(_canonical(replacement))
            disposition = DISPOSITION_COLLECTOR
            target = {
                "kind": "trusted_raw_collector",
                "rail_id": spec.rail_id,
                "collector_package_manifest_sha256": package["manifest_sha256"],
                "service_unit": package_units[job_id]["service"],
                "service_unit_sha256": package_units[job_id]["service_sha256"],
                "timer_unit": package_units[job_id]["timer"],
                "timer_unit_sha256": package_units[job_id]["timer_sha256"],
                "packet_path": _source_packet_path(job_id),
                "replacement_agent_record_sha256": replacement_sha,
                "approved_guild_target": approved_guild_target,
                "historical_source_delivery": spec.historical_source_delivery,
                "historical_source_delivery_eligible": False,
                "semantic_judgment_allowed_in_collector": False,
                "provider_or_model_allowed_in_collector": False,
                "direct_discord_allowed_in_collector": False,
            }
            owner_required = False
        elif job_id == OWNER_JOB_ID:
            disposition = DISPOSITION_PRESERVE
            target = {
                "kind": "preserved_inert_record",
                "archive_byte_exact_required": True,
                "archive_source_store_sha256": observation[
                    "source_store_sha256"
                ],
                "archive_record_sha256": reviewed["record_sha256"],
                "live_record_enabled": False,
                "live_record_state": "paused",
                "delete_allowed": False,
                "claimed_discontinued": False,
                "owner_followup_code": OWNER_FOLLOWUP_CODE,
                "owner_followup_blocks_cutover": False,
            }
            owner_required = False
        else:
            raise ProductionCronContinuityPackageError(
                "production_cron_continuity_source_not_exhaustive"
            )
        rows.append(
            {
                "index": reviewed["index"],
                "job_id": job_id,
                "source_record_sha256": reviewed["record_sha256"],
                "source_definition_sha256": reviewed["definition_sha256"],
                "validation_code": reviewed["validation_code"],
                "disposition": disposition,
                "target": target,
                "owner_input_required": owner_required,
            }
        )
    bundle = _replacement_bundle(replacements)
    validate_replacement_bundle(bundle)
    unresolved = [row["job_id"] for row in rows if row["owner_input_required"]]
    counts = {
        disposition: sum(row["disposition"] == disposition for row in rows)
        for disposition in (
            DISPOSITION_KEEP,
            DISPOSITION_AGENT,
            DISPOSITION_COLLECTOR,
            DISPOSITION_PRESERVE,
        )
    }
    unsigned = {
        "schema": PLAN_SCHEMA,
        "source_store_sha256": observation["source_store_sha256"],
        "observation_sha256": observation["observation_sha256"],
        "review_plan_sha256": review_plan["plan_sha256"],
        "inventory_sha256": inventory["inventory_sha256"],
        "collector_package_manifest_sha256": package["manifest_sha256"],
        "trusted_collector_package": copy.deepcopy(package),
        "collector_execution_readiness": copy.deepcopy(execution_readiness),
        "operational_edge_readiness": copy.deepcopy(operational_readiness),
        "mechanical_job_package_manifest_sha256": (
            mechanical_job_package_manifest_sha256
        ),
        "cutover_runtime_sha256": cutover_runtime_sha256,
        "cutover_entrypoint_sha256": cutover_entrypoint_sha256,
        "replacement_bundle_sha256": bundle["bundle_sha256"],
        "replacement_record_count": bundle["record_count"],
        "enabled_count": len(rows),
        "records": rows,
        "disposition_counts": counts,
        "owner_input_required_job_ids": unresolved,
        "post_cutover_owner_followups": [
            {
                "job_id": OWNER_JOB_ID,
                "followup_code": OWNER_FOLLOWUP_CODE,
                "current_disposition": DISPOSITION_PRESERVE,
                "blocks_cutover": False,
            }
        ],
        "catalog_complete": True,
        "review_complete": True,
        "cutover_executable": True,
        "blanket_inert_migration_allowed": False,
        "byte_exact_source_archive_required": True,
        "execute_job_during_packaging": False,
        "prompt_or_script_content_recorded": False,
    }
    plan = {**unsigned, "plan_sha256": _sha256(_canonical(unsigned))}
    validate_packaged_continuity_plan(
        plan,
        collector_package=package,
        replacement_bundle=bundle,
        inventory=inventory,
        expected_mechanical_job_package_manifest_sha256=(
            mechanical_job_package_manifest_sha256
        ),
        require_executable=True,
    )
    return ContinuityPackageBuild(
        plan=plan,
        replacement_bundle=bundle,
        inventory=copy.deepcopy(inventory),
    )


def validate_packaged_continuity_plan(
    value: Mapping[str, Any],
    *,
    collector_package: Mapping[str, Any] | None = None,
    collector_execution_readiness: Mapping[str, Any] | None = None,
    operational_edge_readiness: Mapping[str, Any] | None = None,
    replacement_bundle: Mapping[str, Any] | None = None,
    inventory: Mapping[str, Any] | None = None,
    expected_mechanical_job_package_manifest_sha256: str | None = None,
    require_executable: bool = False,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProductionCronContinuityPackageError(
            "production_cron_packaged_plan_invalid"
        )
    embedded_package = value.get("trusted_collector_package")
    package = collector_rail.validate_package_manifest(embedded_package)
    if collector_package is not None and dict(collector_package) != package:
        raise ProductionCronContinuityPackageError(
            "production_cron_packaged_plan_collector_drifted"
        )
    embedded_edge = value.get("operational_edge_readiness")
    embedded_boot_id = (
        embedded_edge.get("boot_id_sha256")
        if isinstance(embedded_edge, Mapping)
        and isinstance(embedded_edge.get("boot_id_sha256"), str)
        else ""
    )
    embedded_observed_at = (
        embedded_edge.get("observed_at_unix")
        if isinstance(embedded_edge, Mapping)
        and type(embedded_edge.get("observed_at_unix")) is int
        else 0
    )
    try:
        embedded_operational_readiness = validate_operational_edge_readiness(
            embedded_edge,
            revision=package["release_revision"],
            required_jobs=required_cron_operations(),
            # This receipt is historical evidence inside an owner-bound,
            # digest-addressed plan.  Current-boot freshness is re-proved by
            # the transactional cutover preflight; it must not make the
            # immutable plan expire after 120 seconds.
            expected_boot_id_sha256=embedded_boot_id,
            now_unix=embedded_observed_at,
        )
        embedded_execution_readiness = (
            collector_rail.validate_execution_readiness(
                value.get("collector_execution_readiness"),
                manifest=package,
                operational_edge_receipt=embedded_operational_readiness,
                expected_boot_id_sha256=embedded_boot_id,
                now_unix=embedded_observed_at,
            )
        )
    except (
        OperationalEdgeReadinessError,
        collector_rail.TrustedCronCollectorError,
    ) as exc:
        raise ProductionCronContinuityPackageError(
            "production_cron_packaged_plan_readiness_invalid"
        ) from exc
    if (
        embedded_execution_readiness["activation_ready"] is not True
        or collector_execution_readiness is not None
        and dict(collector_execution_readiness)
        != embedded_execution_readiness
        or operational_edge_readiness is not None
        and dict(operational_edge_readiness)
        != embedded_operational_readiness
    ):
        raise ProductionCronContinuityPackageError(
            "production_cron_packaged_plan_readiness_invalid"
        )
    expected_fields = {
        "schema",
        "source_store_sha256",
        "observation_sha256",
        "review_plan_sha256",
        "inventory_sha256",
        "collector_package_manifest_sha256",
        "trusted_collector_package",
        "collector_execution_readiness",
        "operational_edge_readiness",
        "mechanical_job_package_manifest_sha256",
        "cutover_runtime_sha256",
        "cutover_entrypoint_sha256",
        "replacement_bundle_sha256",
        "replacement_record_count",
        "enabled_count",
        "records",
        "disposition_counts",
        "owner_input_required_job_ids",
        "post_cutover_owner_followups",
        "catalog_complete",
        "review_complete",
        "cutover_executable",
        "blanket_inert_migration_allowed",
        "byte_exact_source_archive_required",
        "execute_job_during_packaging",
        "prompt_or_script_content_recorded",
        "plan_sha256",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != expected_fields
        or value.get("schema") != PLAN_SCHEMA
        or any(
            _SHA256.fullmatch(str(value.get(field) or "")) is None
            for field in (
                "source_store_sha256",
                "observation_sha256",
                "review_plan_sha256",
                "inventory_sha256",
                "collector_package_manifest_sha256",
                "mechanical_job_package_manifest_sha256",
                "cutover_runtime_sha256",
                "cutover_entrypoint_sha256",
                "replacement_bundle_sha256",
                "plan_sha256",
            )
        )
        or value.get("collector_package_manifest_sha256")
        != package["manifest_sha256"]
        or _sha256(
            _canonical({key: item for key, item in value.items() if key != "plan_sha256"})
        )
        != value.get("plan_sha256")
        or type(value.get("enabled_count")) is not int
        or value.get("enabled_count") != 28
        or type(value.get("replacement_record_count")) is not int
        or value.get("replacement_record_count") != 24
        or not isinstance(value.get("records"), list)
        or len(value["records"]) != 28
        or value.get("catalog_complete") is not True
        or value.get("review_complete") is not True
        or value.get("cutover_executable") is not True
        or value.get("blanket_inert_migration_allowed") is not False
        or value.get("byte_exact_source_archive_required") is not True
        or value.get("execute_job_during_packaging") is not False
        or value.get("prompt_or_script_content_recorded") is not False
        or value.get("owner_input_required_job_ids") != []
        or value.get("post_cutover_owner_followups")
        != [
            {
                "job_id": OWNER_JOB_ID,
                "followup_code": OWNER_FOLLOWUP_CODE,
                "current_disposition": DISPOSITION_PRESERVE,
                "blocks_cutover": False,
            }
        ]
        or require_executable and value.get("cutover_executable") is not True
    ):
        raise ProductionCronContinuityPackageError(
            "production_cron_packaged_plan_invalid"
        )
    mechanical_manifest_sha256 = value[
        "mechanical_job_package_manifest_sha256"
    ]
    if (
        expected_mechanical_job_package_manifest_sha256 is not None
        and mechanical_manifest_sha256
        != expected_mechanical_job_package_manifest_sha256
    ):
        raise ProductionCronContinuityPackageError(
            "production_cron_packaged_plan_mechanical_drifted"
        )
    expected_job_ids = {item.job_id for item in review.REVIEW_CATALOG}
    review_catalog = {item.job_id: item for item in review.REVIEW_CATALOG}
    collector_catalog = {
        item.source_job_id: item for item in collector_rail.COLLECTOR_SPECS
    }
    observed: set[str] = set()
    observed_indexes: set[int] = set()
    replacement_hashes: dict[str, str] = {}
    for row in value["records"]:
        if (
            not isinstance(row, Mapping)
            or set(row)
            != {
                "index",
                "job_id",
                "source_record_sha256",
                "source_definition_sha256",
                "validation_code",
                "disposition",
                "target",
                "owner_input_required",
            }
            or row.get("job_id") not in expected_job_ids
            or row["job_id"] in observed
            or type(row.get("index")) is not int
            or row["index"] < 0
            or row["index"] in observed_indexes
            or _SHA256.fullmatch(str(row.get("source_record_sha256") or ""))
            is None
            or _SHA256.fullmatch(str(row.get("source_definition_sha256") or ""))
            is None
            or row.get("owner_input_required") is not False
        ):
            raise ProductionCronContinuityPackageError(
                "production_cron_packaged_plan_invalid"
            )
        observed.add(row["job_id"])
        observed_indexes.add(row["index"])
        entry = review_catalog[row["job_id"]]
        if (
            row.get("source_definition_sha256")
            != entry.expected_definition_sha256
            or row.get("validation_code") != entry.expected_validation_code
        ):
            raise ProductionCronContinuityPackageError(
                "production_cron_packaged_plan_source_drifted"
            )
        target = row.get("target")
        if entry.disposition == review.DISPOSITION_KEEP:
            if (
                row.get("disposition") != DISPOSITION_KEEP
                or target
                != {
                    "kind": "retained_compatible_record",
                    "record_sha256": row["source_record_sha256"],
                }
            ):
                raise ProductionCronContinuityPackageError(
                    "production_cron_packaged_plan_invalid"
                )
        elif entry.disposition == review.DISPOSITION_AGENT:
            if (
                row.get("disposition") != DISPOSITION_AGENT
                or not isinstance(target, Mapping)
                or set(target)
                != {
                    "kind",
                    "replacement_record_sha256",
                    "provider",
                    "model",
                    "enabled_toolsets",
                }
                or target.get("kind") != "primary_agent_record"
                or _SHA256.fullmatch(
                    str(target.get("replacement_record_sha256") or "")
                )
                is None
                or target.get("provider") != review.PRIMARY_PROVIDER
                or target.get("model") != review.PRIMARY_MODEL
                or target.get("enabled_toolsets")
                != list(entry.required_toolsets or ())
            ):
                raise ProductionCronContinuityPackageError(
                    "production_cron_packaged_plan_invalid"
                )
            replacement_hashes[row["job_id"]] = target[
                "replacement_record_sha256"
            ]
        elif row["job_id"] in collector_catalog:
            spec = collector_catalog[row["job_id"]]
            units = package["units"].get(row["job_id"])
            if (
                row.get("disposition") != DISPOSITION_COLLECTOR
                or not isinstance(target, Mapping)
                or not isinstance(units, Mapping)
                or set(target)
                != {
                    "kind",
                    "rail_id",
                    "collector_package_manifest_sha256",
                    "service_unit",
                    "service_unit_sha256",
                    "timer_unit",
                    "timer_unit_sha256",
                    "packet_path",
                    "replacement_agent_record_sha256",
                    "approved_guild_target",
                    "historical_source_delivery",
                    "historical_source_delivery_eligible",
                    "semantic_judgment_allowed_in_collector",
                    "provider_or_model_allowed_in_collector",
                    "direct_discord_allowed_in_collector",
                }
                or target.get("kind") != "trusted_raw_collector"
                or target.get("rail_id") != spec.rail_id
                or target.get("collector_package_manifest_sha256")
                != package["manifest_sha256"]
                or target.get("service_unit") != units["service"]
                or target.get("service_unit_sha256") != units["service_sha256"]
                or target.get("timer_unit") != units["timer"]
                or target.get("timer_unit_sha256") != units["timer_sha256"]
                or target.get("packet_path") != _source_packet_path(row["job_id"])
                or target.get("approved_guild_target")
                != _target_from_delivery(
                    {"id": row["job_id"]},
                    source_job_id=row["job_id"],
                )
                or target.get("historical_source_delivery")
                != spec.historical_source_delivery
                or target.get("historical_source_delivery_eligible") is not False
                or spec.model_review_required
                and _SHA256.fullmatch(
                    str(target.get("replacement_agent_record_sha256") or "")
                )
                is None
                or not spec.model_review_required
                and target.get("replacement_agent_record_sha256") is not None
                or target.get("semantic_judgment_allowed_in_collector") is not False
                or target.get("provider_or_model_allowed_in_collector") is not False
                or target.get("direct_discord_allowed_in_collector") is not False
            ):
                raise ProductionCronContinuityPackageError(
                    "production_cron_packaged_plan_invalid"
                )
            if spec.model_review_required:
                replacement_hashes[row["job_id"]] = target[
                    "replacement_agent_record_sha256"
                ]
        elif row["job_id"] == OWNER_JOB_ID:
            expected_target = {
                "kind": "preserved_inert_record",
                "archive_byte_exact_required": True,
                "archive_source_store_sha256": value[
                    "source_store_sha256"
                ],
                "archive_record_sha256": row["source_record_sha256"],
                "live_record_enabled": False,
                "live_record_state": "paused",
                "delete_allowed": False,
                "claimed_discontinued": False,
                "owner_followup_code": OWNER_FOLLOWUP_CODE,
                "owner_followup_blocks_cutover": False,
            }
            if (
                row.get("disposition") != DISPOSITION_PRESERVE
                or target != expected_target
            ):
                raise ProductionCronContinuityPackageError(
                    "production_cron_packaged_plan_invalid"
                )
        else:
            raise ProductionCronContinuityPackageError(
                "production_cron_packaged_plan_not_exhaustive"
            )
    if observed != expected_job_ids or len(observed_indexes) != 28:
        raise ProductionCronContinuityPackageError(
            "production_cron_packaged_plan_not_exhaustive"
        )
    expected_counts = {
        DISPOSITION_KEEP: 1,
        DISPOSITION_AGENT: 5,
        DISPOSITION_COLLECTOR: 21,
        DISPOSITION_PRESERVE: 1,
    }
    if value.get("disposition_counts") != expected_counts:
        raise ProductionCronContinuityPackageError(
            "production_cron_packaged_plan_invalid"
        )
    if replacement_bundle is not None:
        bundle = validate_replacement_bundle(replacement_bundle)
        bundle_by_id = {item["id"]: item for item in bundle["records"]}
        if (
            bundle["bundle_sha256"] != value["replacement_bundle_sha256"]
            or bundle["record_count"] != value["replacement_record_count"]
            or set(bundle_by_id) != set(replacement_hashes)
            or any(
                _sha256(_canonical(bundle_by_id[job_id])) != digest
                for job_id, digest in replacement_hashes.items()
            )
        ):
            raise ProductionCronContinuityPackageError(
                "production_cron_replacement_bundle_drifted"
            )
    if inventory is not None:
        trusted_inventory = production_cron_migration.validate_inventory(inventory)
        inventory_rows = {
            row["job_id"]: row
            for field in (
                "compatible_enabled_records",
                "incompatible_enabled_records",
            )
            for row in trusted_inventory[field]
        }
        plan_rows = {row["job_id"]: row for row in value["records"]}
        legacy = trusted_inventory["legacy_auto_sync"]
        if (
            trusted_inventory["source_store_sha256"]
            != value["source_store_sha256"]
            or trusted_inventory["inventory_sha256"] != value["inventory_sha256"]
            or trusted_inventory["enabled_count"] != value["enabled_count"]
            or set(inventory_rows) != set(plan_rows)
            or any(
                source["index"] != plan_rows[job_id]["index"]
                or source["record_sha256"]
                != plan_rows[job_id]["source_record_sha256"]
                for job_id, source in inventory_rows.items()
            )
            or legacy.get("present") is True
            and (
                legacy.get("enabled") is not False
                or legacy.get("replacement_rail_job_id")
                != production_cron_migration.MECHANICAL_RAIL_JOB_ID
            )
        ):
            raise ProductionCronContinuityPackageError(
                "production_cron_packaged_plan_inventory_drifted"
            )
    return dict(value)


def _artifact_bytes(path: Path, *, maximum: int = 8 * 1024 * 1024) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ProductionCronContinuityPackageError(
            "production_cron_artifact_unavailable"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or not 0 < before.st_size <= maximum
        ):
            raise ProductionCronContinuityPackageError(
                "production_cron_artifact_metadata_invalid"
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
        raise ProductionCronContinuityPackageError(
            "production_cron_artifact_changed"
        )
    return raw


def _write_exact(path: Path, value: bytes, *, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    if path.is_symlink():
        raise ProductionCronContinuityPackageError(
            "production_cron_artifact_symlink_forbidden"
        )
    if path.exists():
        existing = _artifact_bytes(path)
        metadata = path.stat(follow_symlinks=False)
        if existing != value or stat.S_IMODE(metadata.st_mode) != mode:
            raise ProductionCronContinuityPackageError(
                "production_cron_artifact_replay_mismatch"
            )
        return
    descriptor, temporary = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        os.fchmod(descriptor, mode)
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


def _safe_artifact_path(root: Path, relative_text: str) -> Path:
    relative = Path(relative_text)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ProductionCronContinuityPackageError(
            "production_cron_artifact_path_invalid"
        )
    candidate = root / relative
    if Path(os.path.normpath(str(candidate))) != candidate:
        raise ProductionCronContinuityPackageError(
            "production_cron_artifact_path_invalid"
        )
    return candidate


def write_packaged_continuity_artifacts(
    *,
    output_root: Path,
    build: ContinuityPackageBuild,
    inventory: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a replay-safe, non-installing package consumed by cutover."""

    trusted_inventory = build.inventory if inventory is None else inventory
    plan = validate_packaged_continuity_plan(
        build.plan,
        replacement_bundle=build.replacement_bundle,
        inventory=trusted_inventory,
        require_executable=True,
    )
    bundle = validate_replacement_bundle(build.replacement_bundle)
    package = collector_rail.validate_package_manifest(
        plan["trusted_collector_package"]
    )
    if not output_root.is_absolute() or output_root.is_symlink():
        raise ProductionCronContinuityPackageError(
            "production_cron_artifact_root_invalid"
        )
    unit_files = collector_rail.render_package_unit_files(package)
    file_bytes: dict[str, bytes] = {
        str(PLAN_RELATIVE_PATH): _canonical(plan) + b"\n",
        str(REPLACEMENT_BUNDLE_RELATIVE_PATH): _canonical(bundle) + b"\n",
        str(COLLECTOR_MANIFEST_RELATIVE_PATH): _canonical(package) + b"\n",
    }
    for relative, raw in unit_files.items():
        file_bytes[str(Path("cron/trusted-collector") / relative)] = raw
    rows: list[dict[str, Any]] = []
    for relative in sorted(file_bytes):
        private = relative == str(REPLACEMENT_BUNDLE_RELATIVE_PATH)
        mode = 0o600 if private else 0o640
        raw = file_bytes[relative]
        _write_exact(
            _safe_artifact_path(output_root, relative),
            raw,
            mode=mode,
        )
        rows.append(
            {
                "relative_path": relative,
                "sha256": _sha256(raw),
                "mode": mode,
                "private": private,
            }
        )
    unsigned = {
        "schema": ARTIFACT_INDEX_SCHEMA,
        "release_revision": package["release_revision"],
        "plan_relative_path": str(PLAN_RELATIVE_PATH),
        "plan_sha256": plan["plan_sha256"],
        "replacement_bundle_relative_path": str(
            REPLACEMENT_BUNDLE_RELATIVE_PATH
        ),
        "replacement_bundle_sha256": bundle["bundle_sha256"],
        "collector_manifest_relative_path": str(
            COLLECTOR_MANIFEST_RELATIVE_PATH
        ),
        "collector_manifest_sha256": package["manifest_sha256"],
        "cutover_runtime_sha256": plan["cutover_runtime_sha256"],
        "cutover_entrypoint_sha256": plan["cutover_entrypoint_sha256"],
        "files": rows,
        "file_count": len(rows),
        "units_installed": False,
        "timers_enabled": False,
        "timers_started": False,
        "jobs_store_mutated": False,
        "secret_material_recorded": False,
    }
    index = {**unsigned, "artifact_index_sha256": _sha256(_canonical(unsigned))}
    _write_exact(
        output_root / ARTIFACT_INDEX_RELATIVE_PATH,
        _canonical(index) + b"\n",
        mode=0o640,
    )
    return validate_packaged_continuity_artifacts(
        output_root=output_root,
        artifact_index=index,
        inventory=trusted_inventory,
    )


def derive_packaged_continuity_from_host(
    *,
    revision: str,
    mechanical_job_package: Mapping[str, Any],
    source_jobs_path: Path,
    operational_edge_receipt: Mapping[str, Any] | None = None,
) -> HostContinuityDerivation:
    """Derive the complete package in memory without any filesystem write.

    This is the read-only producer used before owner authority construction.
    It reads and hashes runtime dependencies but neither stages artifacts,
    installs units, nor changes ``jobs.json``.
    """

    try:
        mechanical = mechanical_job_rail.validate_package_manifest(
            mechanical_job_package,
            revision=revision,
            host_facts_sha256=str(
                mechanical_job_package.get("host_facts_sha256") or ""
            ),
        )
    except mechanical_job_rail.MechanicalJobRailError as exc:
        raise ProductionCronContinuityPackageError(
            "production_cron_mechanical_package_invalid"
        ) from exc
    release_root = collector_rail.RELEASES_ROOT / f"hermes-agent-{revision[:12]}"
    rail_path = release_root / collector_rail.RAIL_RELATIVE
    runtime_path = release_root / CUTOVER_RUNTIME_RELATIVE_PATH
    entrypoint_path = release_root / CUTOVER_ENTRYPOINT_RELATIVE_PATH
    source_store = _artifact_bytes(source_jobs_path, maximum=2 * 1024 * 1024)
    dependency_facts = {
        path: _sha256(_artifact_bytes(Path(path), maximum=16 * 1024 * 1024))
        for path in sorted(
            {
                path
                for spec in collector_rail.COLLECTOR_SPECS
                for path in spec.dependency_paths
            }
        )
    }
    package = collector_rail.build_package_manifest(
        revision=revision,
        rail_sha256=_sha256(_artifact_bytes(rail_path, maximum=4 * 1024 * 1024)),
        dependency_facts=dependency_facts,
    )
    try:
        if operational_edge_receipt is None:
            from gateway import operational_edge_readiness as edge_readiness

            operational_edge_receipt = (
                edge_readiness.load_operational_edge_readiness(
                    path=edge_readiness.OPERATIONAL_EDGE_READINESS_PATH,
                    revision=revision,
                    required_jobs=required_cron_operations(),
                )
            )
        execution_readiness = collector_rail.collect_execution_readiness(
            package,
            operational_edge_receipt=operational_edge_receipt,
        )
    except (
        AttributeError,
        OperationalEdgeReadinessError,
        collector_rail.TrustedCronCollectorError,
    ) as exc:
        raise ProductionCronContinuityPackageError(
            "production_cron_execution_readiness_unavailable"
        ) from exc
    build = build_packaged_continuity_plan(
        source_store=source_store,
        collector_package=package,
        collector_execution_readiness=execution_readiness,
        operational_edge_readiness=operational_edge_receipt,
        mechanical_job_package_manifest_sha256=mechanical["manifest_sha256"],
        cutover_runtime_sha256=_sha256(
            _artifact_bytes(runtime_path, maximum=4 * 1024 * 1024)
        ),
        cutover_entrypoint_sha256=_sha256(
            _artifact_bytes(entrypoint_path, maximum=1024 * 1024)
        ),
    )
    return HostContinuityDerivation(
        build=build,
        inventory=copy.deepcopy(build.inventory),
    )


def stage_packaged_continuity_from_host(
    *,
    revision: str,
    mechanical_job_package: Mapping[str, Any],
    source_jobs_path: Path,
    output_root: Path,
    expected_continuity_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Post-freeze stage after exact equality with signed authority."""

    derived = derive_packaged_continuity_from_host(
        revision=revision,
        mechanical_job_package=mechanical_job_package,
        source_jobs_path=source_jobs_path,
    )
    expected = validate_packaged_continuity_plan(
        expected_continuity_plan,
        inventory=derived.inventory,
        expected_mechanical_job_package_manifest_sha256=(
            mechanical_job_package["manifest_sha256"]
        ),
        require_executable=True,
    )
    if expected != derived.build.plan:
        raise ProductionCronContinuityPackageError(
            "production_cron_signed_continuity_plan_drifted"
        )
    return write_packaged_continuity_artifacts(
        output_root=output_root,
        build=derived.build,
        inventory=derived.inventory,
    )


def validate_packaged_continuity_artifacts(
    *,
    output_root: Path,
    artifact_index: Mapping[str, Any],
    inventory: Mapping[str, Any] | None = None,
    expected_revision: str | None = None,
) -> dict[str, Any]:
    """Read back and validate the exact plan, private records, and units."""

    expected_fields = {
        "schema",
        "release_revision",
        "plan_relative_path",
        "plan_sha256",
        "replacement_bundle_relative_path",
        "replacement_bundle_sha256",
        "collector_manifest_relative_path",
        "collector_manifest_sha256",
        "cutover_runtime_sha256",
        "cutover_entrypoint_sha256",
        "files",
        "file_count",
        "units_installed",
        "timers_enabled",
        "timers_started",
        "jobs_store_mutated",
        "secret_material_recorded",
        "artifact_index_sha256",
    }
    if (
        not isinstance(artifact_index, Mapping)
        or set(artifact_index) != expected_fields
        or artifact_index.get("schema") != ARTIFACT_INDEX_SCHEMA
        or _SHA256.fullmatch(
            str(artifact_index.get("artifact_index_sha256") or "")
        )
        is None
        or _sha256(
            _canonical(
                {
                    key: item
                    for key, item in artifact_index.items()
                    if key != "artifact_index_sha256"
                }
            )
        )
        != artifact_index.get("artifact_index_sha256")
        or expected_revision is not None
        and artifact_index.get("release_revision") != expected_revision
        or artifact_index.get("plan_relative_path") != str(PLAN_RELATIVE_PATH)
        or artifact_index.get("replacement_bundle_relative_path")
        != str(REPLACEMENT_BUNDLE_RELATIVE_PATH)
        or artifact_index.get("collector_manifest_relative_path")
        != str(COLLECTOR_MANIFEST_RELATIVE_PATH)
        or artifact_index.get("cutover_runtime_sha256")
        is None
        or _SHA256.fullmatch(
            str(artifact_index.get("cutover_runtime_sha256") or "")
        )
        is None
        or _SHA256.fullmatch(
            str(artifact_index.get("cutover_entrypoint_sha256") or "")
        )
        is None
        or not isinstance(artifact_index.get("files"), list)
        or type(artifact_index.get("file_count")) is not int
        or artifact_index.get("file_count") != len(artifact_index["files"])
        or artifact_index.get("file_count") != 45
        or any(
            artifact_index.get(field) is not False
            for field in (
                "units_installed",
                "timers_enabled",
                "timers_started",
                "jobs_store_mutated",
                "secret_material_recorded",
            )
        )
    ):
        raise ProductionCronContinuityPackageError(
            "production_cron_artifact_index_invalid"
        )
    observed: dict[str, bytes] = {}
    for item in artifact_index["files"]:
        if (
            not isinstance(item, Mapping)
            or set(item) != {"relative_path", "sha256", "mode", "private"}
            or not isinstance(item.get("relative_path"), str)
            or item["relative_path"] in observed
            or _SHA256.fullmatch(str(item.get("sha256") or "")) is None
            or item.get("mode") not in {0o600, 0o640}
            or type(item.get("private")) is not bool
            or item["private"]
            is not (
                item["relative_path"]
                == str(REPLACEMENT_BUNDLE_RELATIVE_PATH)
            )
        ):
            raise ProductionCronContinuityPackageError(
                "production_cron_artifact_index_invalid"
            )
        path = _safe_artifact_path(output_root, item["relative_path"])
        raw = _artifact_bytes(path)
        metadata = path.stat(follow_symlinks=False)
        if (
            _sha256(raw) != item["sha256"]
            or stat.S_IMODE(metadata.st_mode) != item["mode"]
        ):
            raise ProductionCronContinuityPackageError(
                "production_cron_artifact_digest_mismatch"
            )
        observed[item["relative_path"]] = raw
    try:
        plan_value = json.loads(
            observed[str(PLAN_RELATIVE_PATH)].decode("ascii", errors="strict")
        )
        bundle_value = json.loads(
            observed[str(REPLACEMENT_BUNDLE_RELATIVE_PATH)].decode(
                "ascii", errors="strict"
            )
        )
        package_value = json.loads(
            observed[str(COLLECTOR_MANIFEST_RELATIVE_PATH)].decode(
                "ascii", errors="strict"
            )
        )
    except (KeyError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProductionCronContinuityPackageError(
            "production_cron_artifact_json_invalid"
        ) from exc
    package = collector_rail.validate_package_manifest(
        package_value,
        revision=artifact_index["release_revision"],
    )
    bundle = validate_replacement_bundle(bundle_value)
    plan = validate_packaged_continuity_plan(
        plan_value,
        collector_package=package,
        replacement_bundle=bundle,
        inventory=inventory,
        require_executable=True,
    )
    expected_units = collector_rail.render_package_unit_files(package)
    expected_unit_paths = {
        str(Path("cron/trusted-collector") / relative): raw
        for relative, raw in expected_units.items()
    }
    if (
        plan["plan_sha256"] != artifact_index["plan_sha256"]
        or bundle["bundle_sha256"]
        != artifact_index["replacement_bundle_sha256"]
        or package["manifest_sha256"]
        != artifact_index["collector_manifest_sha256"]
        or plan["cutover_runtime_sha256"]
        != artifact_index["cutover_runtime_sha256"]
        or plan["cutover_entrypoint_sha256"]
        != artifact_index["cutover_entrypoint_sha256"]
        or any(observed.get(relative) != raw for relative, raw in expected_unit_paths.items())
    ):
        raise ProductionCronContinuityPackageError(
            "production_cron_artifact_binding_invalid"
        )
    return copy.deepcopy(dict(artifact_index))


__all__ = [
    "ARTIFACT_INDEX_RELATIVE_PATH",
    "ARTIFACT_INDEX_SCHEMA",
    "COLLECTOR_MANIFEST_RELATIVE_PATH",
    "ContinuityPackageBuild",
    "HostContinuityDerivation",
    "DISPOSITION_AGENT",
    "DISPOSITION_COLLECTOR",
    "DISPOSITION_KEEP",
    "DISPOSITION_PRESERVE",
    "CUTOVER_ENTRYPOINT_RELATIVE_PATH",
    "CUTOVER_RUNTIME_RELATIVE_PATH",
    "OWNER_JOB_ID",
    "OWNER_FOLLOWUP_CODE",
    "PLAN_RELATIVE_PATH",
    "PLAN_SCHEMA",
    "ProductionCronContinuityPackageError",
    "REPLACEMENT_BUNDLE_RELATIVE_PATH",
    "REPLACEMENT_BUNDLE_SCHEMA",
    "build_agent_replacement_record",
    "build_packaged_continuity_plan",
    "derive_packaged_continuity_from_host",
    "stage_packaged_continuity_from_host",
    "validate_packaged_continuity_plan",
    "validate_packaged_continuity_artifacts",
    "validate_replacement_bundle",
    "write_packaged_continuity_artifacts",
]

"""Redaction-safe, exact-ID review plan for the legacy Muncho cron store.

This module is deliberately an owner-review artifact, not a migration engine.
It never executes, rewrites, disables, or schedules a job.  It reads a cron
store only to produce cryptographic identities and a fixed, code-owned mapping
for the exact production job IDs reviewed in July 2026.

The mapping is not a task classifier.  No prompt, job name, script output, or
keyword is interpreted at runtime.  An unknown ID or any drift in a reviewed
job's static definition fails closed.  Prompt and script bodies never appear in
the observation or plan; only their SHA-256 identities are retained.

The resulting plan is intentionally not cutover-executable.  A separate owner
approval must bind replacement agent records and packaged collector manifests
to a fresh :mod:`gateway.production_cron_migration` inventory.  Records marked
``unresolved_block`` cannot be made inert merely to pass a deployment gate.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shlex
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from gateway.production_model_sovereignty_runtime import (
    ProductionContractError,
    validate_production_cron_jobs,
)


OBSERVATION_SCHEMA = "muncho-production-cron-review-observation.v1"
PLAN_SCHEMA = "muncho-production-cron-owner-review-plan.v2"
MAX_STORE_BYTES = 2 * 1024 * 1024

DISPOSITION_KEEP = "keep_compatible"
DISPOSITION_AGENT = "migrate_strict_primary_agent"
DISPOSITION_COLLECTOR = "replace_packaged_trusted_collector"
DISPOSITION_BLOCK = "unresolved_block"

PRIMARY_PROVIDER = "openai-codex"
PRIMARY_MODEL = "gpt-5.6-sol"
# Compatibility toolset name; its service-gated contract is policy-neutral
# authorized guild history. The connector enforces public_only for canaries and
# guild_acl for production.
GUILD_HISTORY_TOOLSET = "discord_guild_read"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TARGET_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_DYNAMIC_RECORD_FIELDS = frozenset(
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
_SENSITIVE_KEY_PARTS = (
    "api_key",
    "credential",
    "password",
    "passkey",
    "secret",
    "token",
)
_OBSERVATION_RECORD_FIELDS = frozenset(
    {
        "index",
        "job_id",
        "name",
        "record_sha256",
        "definition_sha256",
        "validation_code",
        "schedule",
        "schedule_sha256",
        "repeat_times",
        "no_agent",
        "prompt_sha256",
        "script_present",
        "script_sha256",
        "script_basename",
        "workdir_present",
        "workdir_sha256",
        "deliver",
        "origin_binding_sha256",
        "provider",
        "model",
        "base_url_present",
        "enabled_toolsets",
    }
)


class ProductionCronContinuityReviewError(RuntimeError):
    """Stable, non-secret failure while creating a review-only plan."""


@dataclass(frozen=True)
class ReviewDisposition:
    job_id: str
    expected_definition_sha256: str
    expected_validation_code: str | None
    disposition: str
    target_id: str | None
    required_toolsets: tuple[str, ...] | None = None
    required_delivery: str | None = None
    remove_workdir: bool = False
    remove_script: bool = False
    owner_semantic_decision_required: bool = False
    blocker_code: str | None = None
    inferred_target_id: str | None = None
    semantic_logic_outside_model_observed: bool = False


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
        raise ProductionCronContinuityReviewError(
            "production_cron_review_json_invalid"
        ) from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8", errors="strict"))


def _text_identity(value: Any) -> dict[str, Any]:
    if value is None:
        return {"present": False, "sha256": None}
    if not isinstance(value, str):
        raise ProductionCronContinuityReviewError(
            "production_cron_review_text_field_invalid"
        )
    return {"present": True, "sha256": _sha256_text(value)}


def _origin_identity(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ProductionCronContinuityReviewError(
            "production_cron_review_origin_invalid"
        )
    unexpected = set(value) - {
        "platform",
        "chat_id",
        "chat_name",
        "thread_id",
        "user_id",
    }
    if unexpected:
        raise ProductionCronContinuityReviewError(
            "production_cron_review_origin_invalid"
        )
    chat_name = value.get("chat_name")
    if chat_name is not None and not isinstance(chat_name, str):
        raise ProductionCronContinuityReviewError(
            "production_cron_review_origin_invalid"
        )
    return {
        "platform": value.get("platform"),
        "chat_id": value.get("chat_id"),
        "thread_id": value.get("thread_id"),
        "user_id": value.get("user_id"),
        "chat_name_sha256": (
            _sha256_text(chat_name) if isinstance(chat_name, str) else None
        ),
    }


def _ensure_no_sensitive_keys(value: Any, *, path: str = "job") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ProductionCronContinuityReviewError(
                    "production_cron_review_definition_invalid"
                )
            folded = key.casefold()
            if any(part in folded for part in _SENSITIVE_KEY_PARTS):
                raise ProductionCronContinuityReviewError(
                    "production_cron_review_sensitive_field_forbidden"
                )
            _ensure_no_sensitive_keys(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _ensure_no_sensitive_keys(item, path=f"{path}[{index}]")


def _static_definition(job: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact static job definition with bodies replaced by hashes."""

    _ensure_no_sensitive_keys(job)
    definition: dict[str, Any] = {}
    for key, value in job.items():
        if key in _DYNAMIC_RECORD_FIELDS:
            continue
        if key in {"prompt", "script", "workdir"}:
            definition[key] = _text_identity(value)
        elif key == "origin":
            definition[key] = _origin_identity(value)
        elif key == "repeat":
            if value is None:
                definition[key] = None
            elif isinstance(value, Mapping) and set(value).issubset(
                {"completed", "times"}
            ):
                definition[key] = {"times": value.get("times")}
            else:
                raise ProductionCronContinuityReviewError(
                    "production_cron_review_repeat_invalid"
                )
        else:
            definition[key] = value
    return definition


def _script_basename(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ProductionCronContinuityReviewError(
            "production_cron_review_text_field_invalid"
        )
    try:
        parts = shlex.split(value)
    except ValueError as exc:
        raise ProductionCronContinuityReviewError(
            "production_cron_review_script_invalid"
        ) from exc
    if not parts:
        return None
    return os.path.basename(parts[0])


def _validation_code(job: Mapping[str, Any]) -> str | None:
    try:
        validate_production_cron_jobs([job])
    except ProductionContractError as exc:
        return exc.code
    return None


def observe_enabled_jobs_bytes(raw: bytes) -> dict[str, Any]:
    """Produce a deterministic redaction-safe observation of enabled jobs."""

    if not isinstance(raw, bytes) or not 0 < len(raw) <= MAX_STORE_BYTES:
        raise ProductionCronContinuityReviewError(
            "production_cron_review_store_size_invalid"
        )
    try:
        payload = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ProductionCronContinuityReviewError(
            "production_cron_review_store_invalid"
        ) from exc
    if (
        not isinstance(payload, Mapping)
        or not isinstance(payload.get("jobs"), list)
        or any(not isinstance(job, Mapping) for job in payload["jobs"])
    ):
        raise ProductionCronContinuityReviewError(
            "production_cron_review_store_shape_invalid"
        )

    records: list[dict[str, Any]] = []
    observed_ids: set[str] = set()
    for index, job in enumerate(payload["jobs"]):
        if job.get("enabled") is False:
            continue
        job_id = job.get("id")
        if (
            not isinstance(job_id, str)
            or not job_id
            or len(job_id) > 256
            or job_id in observed_ids
        ):
            raise ProductionCronContinuityReviewError(
                "production_cron_review_job_id_invalid"
            )
        observed_ids.add(job_id)
        name = job.get("name")
        if name is not None and not isinstance(name, str):
            raise ProductionCronContinuityReviewError(
                "production_cron_review_name_invalid"
            )
        definition = _static_definition(job)
        prompt_identity = _text_identity(job.get("prompt"))
        script_identity = _text_identity(job.get("script"))
        workdir_identity = _text_identity(job.get("workdir"))
        schedule = job.get("schedule")
        if not isinstance(schedule, Mapping):
            raise ProductionCronContinuityReviewError(
                "production_cron_review_schedule_invalid"
            )
        record = {
            "index": index,
            "job_id": job_id,
            "name": name,
            "record_sha256": _sha256_bytes(_canonical(job)),
            "definition_sha256": _sha256_bytes(_canonical(definition)),
            "validation_code": _validation_code(job),
            "schedule": dict(schedule),
            "schedule_sha256": _sha256_bytes(_canonical(schedule)),
            "repeat_times": (
                job.get("repeat", {}).get("times")
                if isinstance(job.get("repeat"), Mapping)
                else None
            ),
            "no_agent": job.get("no_agent"),
            "prompt_sha256": prompt_identity["sha256"],
            "script_present": script_identity["present"],
            "script_sha256": script_identity["sha256"],
            "script_basename": _script_basename(job.get("script")),
            "workdir_present": workdir_identity["present"],
            "workdir_sha256": workdir_identity["sha256"],
            "deliver": job.get("deliver"),
            "origin_binding_sha256": (
                _sha256_bytes(_canonical(_origin_identity(job.get("origin"))))
                if job.get("origin") is not None
                else None
            ),
            "provider": job.get("provider"),
            "model": job.get("model"),
            "base_url_present": job.get("base_url") not in (None, ""),
            "enabled_toolsets": job.get("enabled_toolsets"),
        }
        records.append(record)

    unsigned = {
        "schema": OBSERVATION_SCHEMA,
        "source_store_sha256": _sha256_bytes(raw),
        "source_job_count": len(payload["jobs"]),
        "enabled_count": len(records),
        "records": records,
        "prompt_or_script_content_recorded": False,
        "job_executed": False,
    }
    return {
        **unsigned,
        "observation_sha256": _sha256_bytes(_canonical(unsigned)),
    }


def observe_enabled_jobs_file(path: Path) -> dict[str, Any]:
    """Read a local store without following symlinks, then observe it."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ProductionCronContinuityReviewError(
            "production_cron_review_store_unavailable"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or not 0 < before.st_size <= MAX_STORE_BYTES
        ):
            raise ProductionCronContinuityReviewError(
                "production_cron_review_store_metadata_invalid"
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
        raise ProductionCronContinuityReviewError(
            "production_cron_review_store_changed"
        )
    return observe_enabled_jobs_bytes(raw)


# Expected definition digests come from one redaction-safe, read-only
# production observation on 2026-07-14.  They bind schedule, prompt/script
# identity, origin, route, tools, and all other static record fields while
# excluding only runtime counters/status.  A changed definition must be
# reviewed as a new record; the catalog never guesses that it is equivalent.
_EXPECTED_DEFINITION_SHA256 = {
    "fecd0675f91e": "ff395509477207be3f14ff33f989fda160749182e22554ccb7adf79d3aa0c807",
    "8d09136f7da5": "6bf852341be5394bc5b9decc8dd0d2fc0211db4bcfdd0b02b5eedf6c40220763",
    "81ed8a3ea0d9": "56e558e57b96be4314c104a2d97f138994a344a6d6e1fbb2d65de5a47bc1ebcc",
    "58344347b373": "7fa28c51bc569898233c977fbf67d1bd8668a6ba5c4d5ad33aa32e94a548d5e9",
    "a05143c24275": "df1a2b47a4ef5c6613eded3991e71576c8d59bd624b4f988b194eb37047ef0c8",
    "06ef64d72891": "d3743597d70202b06234ef48b7410348aebe3322525d5402ef26a937f3c620ed",
    "a7b15e3dea75": "e359ce89dd691923390f180b229b9524fd0ada375b4fc06d3d73aa8e3eaec064",
    "a77d64526f9a": "7569a084b024a27aed90def1d08f8a5810d969fd6e1cac9d434dc35b19020f1e",
    "0d446b5df20c": "9d6a7a8d0eeb730939235849975bf88566955daa83c869a9f946fb2bd7f7b865",
    "54976db7a384": "401335a6ffcbd1b7823cd9e9a697ebd3d81899520f819eed2fe2abe0b6dcaf38",
    "d84d45a86b80": "8e00450558ff5a2408c9772e9d543d8d0af6280aa4405130d2c6eb6aa2b2e645",
    "b35767232c37": "43c70335a4246629b1e439b0cb146a614c94c0fd3563e7c87f63f25428093eb0",
    "27ab4a64f8ad": "72fc516d96d815f13fe0da576b9dfa064daa85953ce075d55a5dd13feca29af9",
    "2b2035630202": "1cc40f0f969e068b63828c54315743c2f030b00dae5e0f09f43631d3873eee6f",
    "2a9f6be53fec": "1710e27ccb39eec2dc808778c207e2b15b2f452545d0ac92d82f834b05ea977a",
    "90857403360d": "3b0e6c1f205e207b5ad6a421805f7bdc5f92e13363f474bfae7a2e7a3471bee8",
    "2c9a05136051": "eda2a6dd8a226c701cd29587677fefb4d9ee48ab1d2d151db3ab13c43aafb7c2",
    "e873367f6019": "90987d4bce9566b3d1ffa4e4d4750778281cd796b10121d55f8aa3d390e99314",
    "cd778104fc92": "b3051076d5cac8a7644fd2ef22f279d76e3fa1faab999627685dfb57e140b407",
    "a1dfd5c2a7ab": "a158cb710306188be39f4ecb64dd1d9fbf0a93a5c8a542273b7f1bd461bd9b2c",
    "457b208c90ab": "c3d6d31bc29a0a54cb75850f2e40e0f6a824bdc2a77a9323c31db03d4e471cb2",
    "969248a7da45": "4ba0c5a3b507c8e31a47ad6ca7574b887c26f70b1bbf363a34bdc0ff76642874",
    "e62f55ca93ca": "6861092e5d1941eaa4f5505dad1b1b492f3a8272088570a93bd4d99c7a5bfbe6",
    "7e4a90bdeff0": "3fb3e2bb2e4188e66e8c691911b586106066badfaf9137c6a4b50abfa2a511ca",
    "27f7f59fa0ca": "a3f35f4e32029b2a0082db878aa27b320d2dfe043c0837d62e2233a65f22089c",
    "6faf380f3512": "868f2d6aab796c94921e91a9d7288e631493e90d0118c60e98a2ed2e0bf23eec",
    "90ac99d45130": "eea03924f54a1e0a8a6eaf2b5e3847cf2b0aacf4b05b460c7020b6a1eaa37095",
    "dee523e6f47b": "eeff2015f69cf6a637eea2bdfffb454e9318ab57c6d4a7da751df6ef33fe1454",
}


def _entry(
    job_id: str,
    *,
    code: str | None,
    disposition: str,
    target_id: str | None,
    tools: tuple[str, ...] | None = None,
    delivery: str | None = None,
    remove_workdir: bool = False,
    remove_script: bool = False,
    owner_decision: bool = False,
    blocker: str | None = None,
    inferred_target_id: str | None = None,
    external_semantics: bool = False,
) -> ReviewDisposition:
    return ReviewDisposition(
        job_id=job_id,
        expected_definition_sha256=_EXPECTED_DEFINITION_SHA256[job_id],
        expected_validation_code=code,
        disposition=disposition,
        target_id=target_id,
        required_toolsets=tools,
        required_delivery=delivery,
        remove_workdir=remove_workdir,
        remove_script=remove_script,
        owner_semantic_decision_required=owner_decision,
        blocker_code=blocker,
        inferred_target_id=inferred_target_id,
        semantic_logic_outside_model_observed=external_semantics,
    )


_SCRIPT_FORBIDDEN = "production_cron_local_script_forbidden"
_DISCORD_TOOLSET_FORBIDDEN = "production_cron_enabled_toolset_not_allowed"


REVIEW_CATALOG: tuple[ReviewDisposition, ...] = (
    _entry(
        "fecd0675f91e",
        code="production_cron_workdir_forbidden",
        disposition=DISPOSITION_BLOCK,
        target_id=None,
        owner_decision=True,
        blocker="local_mac_workspace_authority_undefined",
    ),
    _entry("8d09136f7da5", code=_SCRIPT_FORBIDDEN, disposition=DISPOSITION_COLLECTOR, target_id="canonical-brain-heartbeat"),
    _entry("81ed8a3ea0d9", code=_SCRIPT_FORBIDDEN, disposition=DISPOSITION_COLLECTOR, target_id="canonical-brain-projection-refresh"),
    _entry("58344347b373", code=_SCRIPT_FORBIDDEN, disposition=DISPOSITION_COLLECTOR, target_id="routeback-violation-monitor"),
    _entry("a05143c24275", code=_SCRIPT_FORBIDDEN, disposition=DISPOSITION_COLLECTOR, target_id="muncho-parity-drift-monitor"),
    _entry("06ef64d72891", code=_SCRIPT_FORBIDDEN, disposition=DISPOSITION_COLLECTOR, target_id="fork-upstream-sync-drift-monitor"),
    _entry("a7b15e3dea75", code=_SCRIPT_FORBIDDEN, disposition=DISPOSITION_COLLECTOR, target_id="canonical-brain-daily-ops-brief"),
    _entry("a77d64526f9a", code=_SCRIPT_FORBIDDEN, disposition=DISPOSITION_COLLECTOR, target_id="knowledge-artifact-skyvision-voucher-monitor"),
    _entry("0d446b5df20c", code=_SCRIPT_FORBIDDEN, disposition=DISPOSITION_COLLECTOR, target_id="knowledge-artifact-all-monitor"),
    _entry("54976db7a384", code=_SCRIPT_FORBIDDEN, disposition=DISPOSITION_COLLECTOR, target_id="canonical-routeback-state-watchdog"),
    _entry("d84d45a86b80", code=_SCRIPT_FORBIDDEN, disposition=DISPOSITION_COLLECTOR, target_id="canonical-operational-persistence-watchdog"),
    _entry("b35767232c37", code=_SCRIPT_FORBIDDEN, disposition=DISPOSITION_COLLECTOR, target_id="private-skill-manifest-monitor"),
    _entry(
        "27ab4a64f8ad",
        code=_SCRIPT_FORBIDDEN,
        disposition=DISPOSITION_BLOCK,
        target_id=None,
        blocker="legacy_learning_pattern_semantic_projection_not_admissible",
        inferred_target_id="learning-loop-model-authored-review",
        external_semantics=True,
    ),
    _entry(
        "2b2035630202",
        code=_SCRIPT_FORBIDDEN,
        disposition=DISPOSITION_BLOCK,
        target_id=None,
        blocker="legacy_learning_orchestrator_classifier_not_admissible",
        inferred_target_id="learning-loop-model-authored-review",
        external_semantics=True,
    ),
    _entry(
        "2a9f6be53fec",
        code=_SCRIPT_FORBIDDEN,
        disposition=DISPOSITION_BLOCK,
        target_id=None,
        blocker="path_keyword_knowledge_classifier_not_admissible",
        inferred_target_id="knowledge-ingestion-model-authored-review",
        external_semantics=True,
    ),
    _entry(
        "90857403360d",
        code=_SCRIPT_FORBIDDEN,
        disposition=DISPOSITION_BLOCK,
        target_id=None,
        blocker="keyword_candidate_generator_and_prerun_split_not_admissible",
        inferred_target_id="knowledge-ingestion-model-authored-review",
        external_semantics=True,
    ),
    _entry(
        "2c9a05136051",
        code=_DISCORD_TOOLSET_FORBIDDEN,
        disposition=DISPOSITION_AGENT,
        target_id="strict-primary-agent",
        tools=(GUILD_HISTORY_TOOLSET, "canonical_brain"),
        delivery="origin",
        remove_workdir=True,
        remove_script=True,
    ),
    _entry(
        "e873367f6019",
        code=_DISCORD_TOOLSET_FORBIDDEN,
        disposition=DISPOSITION_AGENT,
        target_id="strict-primary-agent",
        tools=(GUILD_HISTORY_TOOLSET, "canonical_brain"),
        delivery="origin",
        remove_workdir=True,
        remove_script=True,
    ),
    _entry(
        "cd778104fc92",
        code=_DISCORD_TOOLSET_FORBIDDEN,
        disposition=DISPOSITION_AGENT,
        target_id="strict-primary-agent",
        tools=(GUILD_HISTORY_TOOLSET,),
        delivery="origin",
        remove_workdir=True,
        remove_script=True,
    ),
    _entry(
        "a1dfd5c2a7ab",
        code=_DISCORD_TOOLSET_FORBIDDEN,
        disposition=DISPOSITION_AGENT,
        target_id="strict-primary-agent",
        tools=(GUILD_HISTORY_TOOLSET,),
        delivery="origin",
        remove_workdir=True,
        remove_script=True,
    ),
    _entry(
        "457b208c90ab",
        code=None,
        disposition=DISPOSITION_KEEP,
        target_id=None,
    ),
    _entry(
        "969248a7da45",
        code=_DISCORD_TOOLSET_FORBIDDEN,
        disposition=DISPOSITION_AGENT,
        target_id="strict-primary-agent",
        tools=(GUILD_HISTORY_TOOLSET,),
        delivery="origin",
        remove_workdir=True,
        remove_script=True,
    ),
    _entry(
        "e62f55ca93ca",
        code=_SCRIPT_FORBIDDEN,
        disposition=DISPOSITION_BLOCK,
        target_id=None,
        blocker="voice_collector_stage_and_delivery_commit_protocol_not_packaged",
        inferred_target_id="voice-context-model-authored-digest",
    ),
    _entry("7e4a90bdeff0", code=_SCRIPT_FORBIDDEN, disposition=DISPOSITION_COLLECTOR, target_id="devops-watchtower-public"),
    _entry("27f7f59fa0ca", code=_SCRIPT_FORBIDDEN, disposition=DISPOSITION_COLLECTOR, target_id="devops-watchtower-infrastructure"),
    _entry("6faf380f3512", code=_SCRIPT_FORBIDDEN, disposition=DISPOSITION_COLLECTOR, target_id="devops-watchtower-tls-dns"),
    _entry("90ac99d45130", code=_SCRIPT_FORBIDDEN, disposition=DISPOSITION_COLLECTOR, target_id="devops-watchtower-digest"),
    _entry("dee523e6f47b", code=_SCRIPT_FORBIDDEN, disposition=DISPOSITION_COLLECTOR, target_id="skyvision-from-heart-weekly-count"),
)


# A second read-only review inspected the exact six initially unresolved jobs,
# their redacted prompts, script implementations, latest local receipts, and
# (for the voice digest) exact Discord guild target/history metadata. These facts
# are codes rather than executable policy.  They explain why five migrations
# are objectively dictated by the approved model-sovereignty architecture and
# why only the repository-maintenance job needs an owner business choice.
_DEEP_REVIEW = {
    "fecd0675f91e": {
        "purpose_code": "weekly_local_hermes_agent_os_standards_maintenance",
        "business_impact_code": "currently_no_effect_in_cloud",
        "dependency_codes": [
            "local_mac_hermes_checkout",
            "agent_os_discover_and_index_sources",
            "writable_agent_os_standards_subtree",
        ],
        "evidence_codes": [
            "latest_cloud_run_failed_because_users_path_absent",
            "configured_local_agent_os_paths_currently_absent",
        ],
        "safest_option_kind": "owner_choice",
        "safest_option_codes": [
            "recreate_as_bounded_local_mac_agent_job",
            "discontinue_unsuperseded_maintenance_job",
        ],
        "objectively_inferable": False,
    },
    "27ab4a64f8ad": {
        "purpose_code": "daily_private_learning_packet_pattern_report",
        "business_impact_code": "repeats_static_pilot_evidence_gap_report",
        "dependency_codes": [
            "five_static_private_learning_packets",
            "latest_review_metadata",
            "filesystem_report_projection",
        ],
        "evidence_codes": [
            "explicit_metadata_is_aggregated",
            "review_status_prefix_selects_candidates",
            "one_missing_evidence_case_and_zero_candidates_observed",
        ],
        "safest_option_kind": "trusted_raw_collector_plus_primary_agent",
        "safest_option_codes": [
            "collect_only_packet_identities_and_explicit_fields",
            "gpt_5_6_sol_authors_pattern_and_candidate_judgment",
            "append_review_outcome_through_canonical_writer",
        ],
        "objectively_inferable": True,
    },
    "2b2035630202": {
        "purpose_code": "daily_learning_review_analytics_and_transfer_queue",
        "business_impact_code": "owner_review_queue_for_private_operational_knowledge",
        "dependency_codes": [
            "learning_review_runner",
            "learning_pattern_analytics",
            "filesystem_report_projection",
        ],
        "evidence_codes": [
            "deterministic_transfer_class_mapper_observed",
            "deterministic_review_recommendation_classifier_observed",
            "one_blocked_nontransferable_item_observed",
        ],
        "safest_option_kind": "trusted_raw_collector_plus_primary_agent",
        "safest_option_codes": [
            "remove_external_transfer_and_recommendation_classifiers",
            "gpt_5_6_sol_authors_review_and_transfer_recommendations",
            "canonical_event_is_truth_not_latest_report_file",
        ],
        "objectively_inferable": True,
    },
    "2a9f6be53fec": {
        "purpose_code": "daily_approved_root_metadata_change_inventory",
        "business_impact_code": "feeds_weekly_knowledge_candidate_review",
        "dependency_codes": [
            "canonical_brain_reports_and_docs_roots",
            "filesystem_manifest_and_queue",
            "path_keyword_domain_sensitivity_action_functions",
        ],
        "evidence_codes": [
            "metadata_hash_and_diff_collection_is_mechanical",
            "path_keyword_semantic_classifier_observed",
            "774_entries_and_10_current_queue_changes_observed",
        ],
        "safest_option_kind": "trusted_raw_collector_plus_primary_agent",
        "safest_option_codes": [
            "collector_emits_only_path_size_mtime_and_sha256",
            "gpt_5_6_sol_authors_domain_sensitivity_and_action",
            "canonical_event_records_review_outcome",
        ],
        "objectively_inferable": True,
    },
    "90857403360d": {
        "purpose_code": "weekly_deeper_knowledge_candidate_review",
        "business_impact_code": "currently_blocked_after_model_upgrade",
        "dependency_codes": [
            "daily_knowledge_metadata_inventory",
            "legacy_keyword_candidate_generator",
            "primary_model_report",
        ],
        "evidence_codes": [
            "last_run_made_no_inference_due_stale_model_snapshot",
            "legacy_generator_grouped_and_selected_by_derived_labels",
            "three_candidates_from_eleven_queued_files_last_generated",
        ],
        "safest_option_kind": "trusted_raw_collector_plus_primary_agent",
        "safest_option_codes": [
            "remove_keyword_candidate_generator",
            "gpt_5_6_sol_reads_digest_bound_raw_metadata_snapshot",
            "model_authors_candidates_sensitivity_and_eval_prompts",
        ],
        "objectively_inferable": True,
    },
    "e62f55ca93ca": {
        "purpose_code": "hourly_voice_transcript_owner_review_digest",
        "business_impact_code": "proven_owner_digest_value_when_voice_events_exist",
        "dependency_codes": [
            "private_voice_jsonl_events",
            "checkpoint_and_bounded_redaction_collector",
            "primary_model_bulgarian_summary",
            "approved_control_tower_guild_acl_delivery",
        ],
        "evidence_codes": [
            "collector_is_mechanical_and_model_authors_semantics",
            "nineteen_private_and_public_packet_artifacts_observed",
            "legacy_thread_target_is_historical_ineligible_provenance",
            "replacement_requires_exact_reviewed_guild_acl_projection",
            "checkpoint_currently_advances_before_delivery_receipt",
        ],
        "safest_option_kind": "staged_collector_plus_primary_agent_with_receipt_commit",
        "safest_option_codes": [
            "collector_stages_bounded_redacted_packet_without_committing_cursor",
            "gpt_5_6_sol_authors_owner_digest",
            "guild_connector_send_and_readback_receipt_commits_checkpoint",
            "canonical_event_records_sent_or_blocked_outcome",
        ],
        "objectively_inferable": True,
    },
}


def _catalog() -> dict[str, ReviewDisposition]:
    result = {entry.job_id: entry for entry in REVIEW_CATALOG}
    if len(result) != len(REVIEW_CATALOG):
        raise ProductionCronContinuityReviewError(
            "production_cron_review_catalog_duplicate"
        )
    for entry in REVIEW_CATALOG:
        if (
            _SHA256.fullmatch(entry.expected_definition_sha256) is None
            or entry.expected_validation_code is not None
            and not entry.expected_validation_code.startswith("production_cron_")
            or entry.disposition
            not in {
                DISPOSITION_KEEP,
                DISPOSITION_AGENT,
                DISPOSITION_COLLECTOR,
                DISPOSITION_BLOCK,
            }
        ):
            raise ProductionCronContinuityReviewError(
                "production_cron_review_catalog_invalid"
            )
        if entry.disposition == DISPOSITION_KEEP:
            valid = (
                entry.expected_validation_code is None
                and entry.target_id is None
                and entry.required_toolsets is None
                and entry.required_delivery is None
                and entry.blocker_code is None
                and entry.owner_semantic_decision_required is False
                and entry.inferred_target_id is None
                and entry.semantic_logic_outside_model_observed is False
            )
        elif entry.disposition == DISPOSITION_AGENT:
            valid = (
                entry.expected_validation_code is not None
                and entry.target_id == "strict-primary-agent"
                and isinstance(entry.required_toolsets, tuple)
                and bool(entry.required_toolsets)
                and len(set(entry.required_toolsets))
                == len(entry.required_toolsets)
                and entry.required_delivery == "origin"
                and entry.remove_workdir is True
                and entry.remove_script is True
                and entry.blocker_code is None
                and entry.owner_semantic_decision_required is False
                and entry.inferred_target_id is None
                and entry.semantic_logic_outside_model_observed is False
            )
        elif entry.disposition == DISPOSITION_COLLECTOR:
            valid = (
                entry.expected_validation_code == _SCRIPT_FORBIDDEN
                and isinstance(entry.target_id, str)
                and _TARGET_ID.fullmatch(entry.target_id) is not None
                and entry.required_toolsets is None
                and entry.required_delivery is None
                and entry.blocker_code is None
                and entry.owner_semantic_decision_required is False
                and entry.inferred_target_id is None
                and entry.semantic_logic_outside_model_observed is False
            )
        else:
            valid = (
                entry.expected_validation_code is not None
                and entry.target_id is None
                and entry.required_toolsets is None
                and entry.required_delivery is None
                and isinstance(entry.blocker_code, str)
                and bool(entry.blocker_code)
                and (
                    entry.owner_semantic_decision_required is True
                    and entry.inferred_target_id is None
                    or entry.owner_semantic_decision_required is False
                    and isinstance(entry.inferred_target_id, str)
                    and _TARGET_ID.fullmatch(entry.inferred_target_id) is not None
                )
            )
        if not valid:
            raise ProductionCronContinuityReviewError(
                "production_cron_review_catalog_invalid"
            )
    blocked = {
        entry.job_id: entry
        for entry in REVIEW_CATALOG
        if entry.disposition == DISPOSITION_BLOCK
    }
    if set(blocked) != set(_DEEP_REVIEW):
        raise ProductionCronContinuityReviewError(
            "production_cron_review_deep_review_invalid"
        )
    for job_id, deep in _DEEP_REVIEW.items():
        entry = blocked[job_id]
        if (
            not isinstance(deep, Mapping)
            or set(deep)
            != {
                "purpose_code",
                "business_impact_code",
                "dependency_codes",
                "evidence_codes",
                "safest_option_kind",
                "safest_option_codes",
                "objectively_inferable",
            }
            or any(
                not isinstance(deep.get(field), str) or not deep[field]
                for field in (
                    "purpose_code",
                    "business_impact_code",
                    "safest_option_kind",
                )
            )
            or any(
                not isinstance(deep.get(field), list)
                or not deep[field]
                or any(not isinstance(value, str) or not value for value in deep[field])
                for field in (
                    "dependency_codes",
                    "evidence_codes",
                    "safest_option_codes",
                )
            )
            or deep.get("objectively_inferable")
            is entry.owner_semantic_decision_required
        ):
            raise ProductionCronContinuityReviewError(
                "production_cron_review_deep_review_invalid"
            )
    return result


def _valid_optional_sha256(value: Any) -> bool:
    return value is None or (
        isinstance(value, str) and _SHA256.fullmatch(value) is not None
    )


def _validate_observation_record(record: Any) -> Mapping[str, Any]:
    if not isinstance(record, Mapping) or set(record) != _OBSERVATION_RECORD_FIELDS:
        raise ProductionCronContinuityReviewError(
            "production_cron_review_observation_invalid"
        )
    schedule = record.get("schedule")
    toolsets = record.get("enabled_toolsets")
    if (
        type(record.get("index")) is not int
        or record["index"] < 0
        or not isinstance(record.get("job_id"), str)
        or not record["job_id"]
        or len(record["job_id"]) > 256
        or record.get("name") is not None
        and not isinstance(record.get("name"), str)
        or not isinstance(record.get("record_sha256"), str)
        or _SHA256.fullmatch(record["record_sha256"]) is None
        or not isinstance(record.get("definition_sha256"), str)
        or _SHA256.fullmatch(record["definition_sha256"]) is None
        or record.get("validation_code") is not None
        and (
            not isinstance(record.get("validation_code"), str)
            or not record["validation_code"].startswith("production_cron_")
        )
        or not isinstance(schedule, Mapping)
        or not isinstance(record.get("schedule_sha256"), str)
        or _SHA256.fullmatch(record["schedule_sha256"]) is None
        or _sha256_bytes(_canonical(schedule)) != record["schedule_sha256"]
        or record.get("repeat_times") is not None
        and type(record.get("repeat_times")) is not int
        or type(record.get("no_agent")) is not bool
        or not _valid_optional_sha256(record.get("prompt_sha256"))
        or type(record.get("script_present")) is not bool
        or not _valid_optional_sha256(record.get("script_sha256"))
        or record["script_present"] is (record.get("script_sha256") is None)
        or record.get("script_basename") is not None
        and not isinstance(record.get("script_basename"), str)
        or type(record.get("workdir_present")) is not bool
        or not _valid_optional_sha256(record.get("workdir_sha256"))
        or record["workdir_present"] is (record.get("workdir_sha256") is None)
        or record.get("deliver") is not None
        and not isinstance(record.get("deliver"), str)
        or not _valid_optional_sha256(record.get("origin_binding_sha256"))
        or record.get("provider") is not None
        and not isinstance(record.get("provider"), str)
        or record.get("model") is not None
        and not isinstance(record.get("model"), str)
        or type(record.get("base_url_present")) is not bool
        or toolsets is not None
        and (
            type(toolsets) is not list
            or any(type(item) is not str or not item for item in toolsets)
            or len(set(toolsets)) != len(toolsets)
        )
    ):
        raise ProductionCronContinuityReviewError(
            "production_cron_review_observation_invalid"
        )
    return record


def build_owner_review_plan(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Map the exact observed records to fixed, non-executable dispositions."""

    expected_observation_fields = {
        "schema",
        "source_store_sha256",
        "source_job_count",
        "enabled_count",
        "records",
        "prompt_or_script_content_recorded",
        "job_executed",
        "observation_sha256",
    }
    if (
        not isinstance(observation, Mapping)
        or set(observation) != expected_observation_fields
        or observation.get("schema") != OBSERVATION_SCHEMA
        or observation.get("prompt_or_script_content_recorded") is not False
        or observation.get("job_executed") is not False
        or not isinstance(observation.get("records"), list)
        or type(observation.get("source_job_count")) is not int
        or observation["source_job_count"] < len(observation["records"])
        or type(observation.get("enabled_count")) is not int
        or observation.get("enabled_count") != len(observation["records"])
        or not isinstance(observation.get("source_store_sha256"), str)
        or _SHA256.fullmatch(observation["source_store_sha256"]) is None
        or not isinstance(observation.get("observation_sha256"), str)
        or _SHA256.fullmatch(observation["observation_sha256"]) is None
        or _sha256_bytes(
            _canonical(
                {
                    key: value
                    for key, value in observation.items()
                    if key != "observation_sha256"
                }
            )
        )
        != observation["observation_sha256"]
    ):
        raise ProductionCronContinuityReviewError(
            "production_cron_review_observation_invalid"
        )

    catalog = _catalog()
    observed: dict[str, Mapping[str, Any]] = {}
    for record in observation["records"]:
        record = _validate_observation_record(record)
        job_id = record.get("job_id")
        if job_id in observed:
            raise ProductionCronContinuityReviewError(
                "production_cron_review_observation_invalid"
            )
        observed[job_id] = record
    indexes = [record["index"] for record in observed.values()]
    if len(indexes) != len(set(indexes)):
        raise ProductionCronContinuityReviewError(
            "production_cron_review_observation_invalid"
        )
    if set(observed) != set(catalog):
        raise ProductionCronContinuityReviewError(
            "production_cron_review_catalog_not_exhaustive"
        )

    records: list[dict[str, Any]] = []
    for source in observation["records"]:
        entry = catalog[source["job_id"]]
        if (
            source.get("definition_sha256")
            != entry.expected_definition_sha256
            or source.get("validation_code") != entry.expected_validation_code
        ):
            raise ProductionCronContinuityReviewError(
                "production_cron_review_definition_drifted"
            )
        if entry.disposition == DISPOSITION_KEEP:
            target: dict[str, Any] | None = None
        elif entry.disposition == DISPOSITION_AGENT:
            target = {
                "kind": "strict_primary_agent_job",
                "provider": PRIMARY_PROVIDER,
                "model": PRIMARY_MODEL,
                "base_url": None,
                "fallback_route_allowed": False,
                "enabled_toolsets": list(entry.required_toolsets or ()),
                "deliver": entry.required_delivery,
                "origin_binding_sha256": source.get("origin_binding_sha256"),
                "prompt_sha256": source.get("prompt_sha256"),
                "prompt_identity_retained": True,
                "authorized_guild_history_tool": "discord_guild_history",
                "authorized_guild_history_binding_source": (
                    "internal_reviewed_cron_context"
                ),
                "authorized_guild_history_known_message_cursor_required": True,
                "source_script_removal_required": entry.remove_script,
                "source_workdir_removal_required": entry.remove_workdir,
                "script": None,
                "workdir": None,
                "replacement_record_sha256": None,
                "guild_target_readiness_receipt_sha256": None,
            }
        elif entry.disposition == DISPOSITION_COLLECTOR:
            target = {
                "kind": "packaged_trusted_collector",
                "systemd_rail_id": entry.target_id,
                "source_script_sha256": source.get("script_sha256"),
                "schedule_sha256": source.get("schedule_sha256"),
                "historical_source_delivery": source.get("deliver"),
                "historical_source_delivery_eligible": False,
                "source_origin_binding_sha256": source.get(
                    "origin_binding_sha256"
                ),
                "source_workdir_must_not_be_inherited": True,
                "model_or_provider_allowed": False,
                "gateway_ambient_credentials_allowed": False,
                "direct_discord_credential_allowed": False,
                "guild_discord_send_requires_connector_receipt": True,
                "package_manifest_sha256": None,
                "readiness_receipt_sha256": None,
            }
        elif entry.disposition == DISPOSITION_BLOCK:
            target = None
        else:
            raise ProductionCronContinuityReviewError(
                "production_cron_review_disposition_invalid"
            )
        records.append(
            {
                "index": source.get("index"),
                "job_id": entry.job_id,
                "name": source.get("name"),
                "record_sha256": source.get("record_sha256"),
                "definition_sha256": source.get("definition_sha256"),
                "validation_code": source.get("validation_code"),
                "schedule": source.get("schedule"),
                "disposition": entry.disposition,
                "target": target,
                "owner_semantic_decision_required": (
                    entry.owner_semantic_decision_required
                ),
                "objectively_inferable": (
                    not entry.owner_semantic_decision_required
                ),
                "semantic_logic_outside_model_observed": (
                    entry.semantic_logic_outside_model_observed
                ),
                "blocker_code": entry.blocker_code,
                "inferred_target_id": entry.inferred_target_id,
                "deep_review": (
                    {
                        **copy.deepcopy(_DEEP_REVIEW[entry.job_id]),
                        "current_delivery": source.get("deliver"),
                        "current_origin_binding_sha256": source.get(
                            "origin_binding_sha256"
                        ),
                    }
                    if entry.job_id in _DEEP_REVIEW
                    else None
                ),
                "retirement_evidence": None,
            }
        )

    disposition_counts = {
        disposition: sum(
            row["disposition"] == disposition for row in records
        )
        for disposition in (
            DISPOSITION_KEEP,
            DISPOSITION_AGENT,
            DISPOSITION_COLLECTOR,
            DISPOSITION_BLOCK,
        )
    }
    owner_decisions = [
        row["job_id"]
        for row in records
        if row["owner_semantic_decision_required"]
    ]
    objectively_inferable_deep_review = [
        row["job_id"]
        for row in records
        if row["deep_review"] is not None and row["objectively_inferable"]
    ]
    unsigned = {
        "schema": PLAN_SCHEMA,
        "source_store_sha256": observation["source_store_sha256"],
        "observation_sha256": observation["observation_sha256"],
        "enabled_count": len(records),
        "incompatible_count": sum(
            row["validation_code"] is not None for row in records
        ),
        "records": records,
        "disposition_counts": disposition_counts,
        "owner_semantic_decision_job_ids": owner_decisions,
        "objectively_inferable_deep_review_job_ids": (
            objectively_inferable_deep_review
        ),
        "retire_stale_count": 0,
        "catalog_complete": True,
        "owner_review_complete": False,
        "cutover_executable": False,
        "blanket_inert_migration_allowed": False,
        "prompt_or_script_content_recorded": False,
        "job_executed": False,
    }
    return {
        **unsigned,
        "plan_sha256": _sha256_bytes(_canonical(unsigned)),
    }


def build_owner_review_plan_bytes(raw: bytes) -> dict[str, Any]:
    return build_owner_review_plan(observe_enabled_jobs_bytes(raw))


__all__ = [
    "DISPOSITION_AGENT",
    "DISPOSITION_BLOCK",
    "DISPOSITION_COLLECTOR",
    "DISPOSITION_KEEP",
    "OBSERVATION_SCHEMA",
    "PLAN_SCHEMA",
    "ProductionCronContinuityReviewError",
    "REVIEW_CATALOG",
    "build_owner_review_plan",
    "build_owner_review_plan_bytes",
    "observe_enabled_jobs_bytes",
    "observe_enabled_jobs_file",
]

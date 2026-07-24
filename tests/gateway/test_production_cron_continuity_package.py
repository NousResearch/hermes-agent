from __future__ import annotations

import copy
import dataclasses
import json
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import production_cron_continuity_package as package
from gateway import production_cron_continuity_review as review
from gateway import production_cron_cutover_runtime as cutover_runtime
from gateway import production_cron_migration
from gateway.production_capability_prerequisites import FIRST_WAVE_TOOLSETS
from gateway.production_model_sovereignty_runtime import (
    resolve_production_cron_enabled_toolsets,
)
from ops.muncho.runtime import trusted_cron_collector_rail as rail
from toolsets import resolve_toolset


LEGACY_LOCKED_CHANNEL = "1504852355588423801"
LEGACY_LOCKED_THREAD = "1524321461714681976"
def _agent_job(entry: review.ReviewDisposition, index: int) -> dict:
    origin = {
        "platform": "discord",
        "chat_id": LEGACY_LOCKED_CHANNEL,
        "thread_id": None,
        "user_id": "1279454038731264061",
        "chat_name": "Public operations",
    }
    job = {
        "id": entry.job_id,
        "name": f"reviewed-{index}",
        "enabled": True,
        "no_agent": False,
        "prompt": f"Use your judgment for reviewed operation {index}.",
        "script": None,
        "workdir": None,
        "deliver": "origin",
        "origin": origin,
        "provider": review.PRIMARY_PROVIDER,
        "model": review.PRIMARY_MODEL,
        "base_url": None,
        "enabled_toolsets": None,
        "schedule": {"kind": "interval", "minutes": index + 1},
        "repeat": {"times": None, "completed": 0},
        "state": "scheduled",
    }
    if entry.job_id == package.OWNER_JOB_ID:
        job["workdir"] = "/Users/emil/local-only"
    elif entry.disposition in {
        review.DISPOSITION_COLLECTOR,
        review.DISPOSITION_BLOCK,
    }:
        job.update(
            no_agent=True,
            script=f"/opt/legacy/collector-{entry.job_id}.py",
            deliver="local",
            origin=None,
            provider=None,
            model=None,
        )
        # Exercise the authorized-guild read surface on reviewed collector
        # replacements whose exact production destination was owner-reviewed.
        if entry.job_id in {"06ef64d72891", "e62f55ca93ca"}:
            job["origin"] = copy.deepcopy(origin)
            if entry.job_id == "e62f55ca93ca":
                job["origin"]["thread_id"] = LEGACY_LOCKED_THREAD
    elif entry.disposition == review.DISPOSITION_AGENT:
        job["enabled_toolsets"] = ["discord"]
    return job


def _source_store(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[bytes, tuple[review.ReviewDisposition, ...]]:
    jobs = [
        _agent_job(entry, index)
        for index, entry in enumerate(review.REVIEW_CATALOG)
    ]
    jobs.append(
        {
            "id": production_cron_migration.LEGACY_AUTO_SYNC_JOB_ID,
            "name": production_cron_migration.LEGACY_AUTO_SYNC_JOB_NAME,
            "enabled": False,
            "no_agent": True,
            "prompt": "",
            "script": "/opt/legacy/fork-sync.py --execute",
            "deliver": "origin",
            "provider": None,
            "model": None,
            "state": "paused",
            "next_run_at": "2026-07-14T18:01:00Z",
        }
    )
    raw = json.dumps(
        {"jobs": jobs, "updated_at": "2026-07-15T00:00:00Z"},
        ensure_ascii=False,
        indent=2,
    ).encode("utf-8") + b"\n"
    observed = review.observe_enabled_jobs_bytes(raw)
    by_id = {row["job_id"]: row for row in observed["records"]}
    rebound = tuple(
        dataclasses.replace(
            entry,
            expected_definition_sha256=by_id[entry.job_id][
                "definition_sha256"
            ],
            expected_validation_code=by_id[entry.job_id]["validation_code"],
        )
        for entry in review.REVIEW_CATALOG
    )
    monkeypatch.setattr(review, "REVIEW_CATALOG", rebound)
    return raw, rebound


def _collector_package() -> dict:
    dependency_paths = {
        path
        for spec in rail.COLLECTOR_SPECS
        for path in spec.dependency_paths
    }
    dependency_paths.add(str(rail.SETPRIV))
    dependencies = {
        path: f"{index + 1:064x}"
        for index, path in enumerate(
            sorted(dependency_paths)
        )
    }
    return rail.build_package_manifest(
        revision="a" * 40,
        rail_sha256="b" * 64,
        dependency_facts=dependencies,
    )


def _collector_execution_readiness() -> dict:
    account_ids = {
        rail.ISOLATED_SERVICE_USER: 1000,
        rail.SCOPED_SERVICE_USER: 2004,
    }
    group_ids = {
        rail.ISOLATED_SERVICE_GROUP: 1000,
        rail.SCOPED_SERVICE_GROUP: 2004,
    }
    return rail.collect_execution_readiness(
        _collector_package(),
        operational_edge_receipt=None,
        account_lookup=lambda name: SimpleNamespace(pw_uid=account_ids[name]),
        group_lookup=lambda name: SimpleNamespace(gr_gid=group_ids[name]),
    )


def _build(monkeypatch: pytest.MonkeyPatch) -> tuple[bytes, package.ContinuityPackageBuild]:
    raw, _catalog = _source_store(monkeypatch)
    build = package.build_packaged_continuity_plan(
        source_store=raw,
        collector_package=_collector_package(),
        collector_execution_readiness=_collector_execution_readiness(),
        mechanical_job_package_manifest_sha256="c" * 64,
        cutover_runtime_sha256="d" * 64,
        cutover_entrypoint_sha256="e" * 64,
    )
    return raw, build


def test_plan_is_exhaustive_primary_model_authored_and_owner_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _raw, build = _build(monkeypatch)
    inventory = build.inventory
    plan = package.validate_packaged_continuity_plan(
        build.plan,
        collector_package=_collector_package(),
        replacement_bundle=build.replacement_bundle,
        inventory=inventory,
        expected_mechanical_job_package_manifest_sha256="c" * 64,
        require_executable=True,
    )

    assert plan["disposition_counts"] == {
        package.DISPOSITION_KEEP: 1,
        package.DISPOSITION_AGENT: 5,
        package.DISPOSITION_COLLECTOR: 21,
        package.DISPOSITION_PRESERVE: 1,
    }
    assert plan["replacement_record_count"] == 24
    assert plan["owner_input_required_job_ids"] == []
    assert plan["cutover_executable"] is True
    assert plan["blanket_inert_migration_allowed"] is False
    assert plan["byte_exact_source_archive_required"] is True
    assert plan["execute_job_during_packaging"] is False
    assert plan["collector_execution_readiness"]["activation_ready"] is False
    assert plan["collector_execution_readiness"][
        "unit_namespace_readiness_packaged"
    ] is False
    assert plan["collector_execution_readiness"][
        "unit_namespace_readiness_receipt_sha256"
    ] is None
    assert plan["collector_execution_readiness"][
        "unit_namespace_readiness_job_count"
    ] == 0
    assert plan["collector_execution_readiness"][
        "unit_namespace_readiness_boot_id_sha256"
    ] is None
    assert plan["collector_execution_readiness"][
        "unit_namespace_readiness_observed_at_unix"
    ] is None
    assert plan["collector_execution_readiness"][
        "unit_namespace_readiness_maximum_age_seconds"
    ] == 0
    assert plan["collector_execution_readiness"][
        "direct_dependencies_ready"
    ] is False
    assert (
        plan["collector_execution_readiness"][
            "scoped_execution_edge_packaged"
        ]
        is False
    )
    assert plan["operational_edge_requirements"][
        "readiness_collection_stage"
    ] == "transactional_cutover_preflight"
    assert plan["operational_edge_requirements"][
        "readiness_receipt_embedded"
    ] is False
    assert plan["post_cutover_owner_followups"] == [
        {
            "job_id": package.OWNER_JOB_ID,
            "followup_code": package.OWNER_FOLLOWUP_CODE,
            "current_disposition": package.DISPOSITION_PRESERVE,
            "blocks_cutover": False,
        }
    ]

    records = build.replacement_bundle["records"]
    assert len(records) == 24
    for record in records:
        assert record["provider"] == review.PRIMARY_PROVIDER
        assert record["model"] == review.PRIMARY_MODEL
        assert record["script"] is None
        assert record["workdir"] is None
        assert record.get("fallback_model") is None
        assert record.get("fallback_provider") is None
        assert "last_delivery_status" not in record
        assert "last_delivery_confirmed_at" not in record
        assert resolve_production_cron_enabled_toolsets(
            record,
            {"platform_toolsets": {"cron": list(FIRST_WAVE_TOOLSETS)}},
        ) == record["enabled_toolsets"]

    guild_history_records = {
        record["id"]: record
        for record in records
        if "discord_guild_read" in record["enabled_toolsets"]
    }
    assert {
        "2c9a05136051",
        "e873367f6019",
        "cd778104fc92",
        "a1dfd5c2a7ab",
        "969248a7da45",
        "06ef64d72891",
        "e62f55ca93ca",
    }.issubset(guild_history_records)
    assert resolve_toolset("discord_guild_read") == [
        "discord_guild_history"
    ]
    assert all("discord" not in record["enabled_toolsets"] for record in records)
    root_record = guild_history_records["06ef64d72891"]
    assert root_record["deliver"] == "local"
    assert root_record["origin"] == {
            "platform": "discord",
            "chat_id": LEGACY_LOCKED_CHANNEL,
            "chat_name": "control-tower",
            "thread_id": None,
            "user_id": "1279454038731264061",
    }
    thread_record = guild_history_records["e62f55ca93ca"]
    assert thread_record["deliver"] == "local"
    assert thread_record["origin"] == {
        "platform": "discord",
        "chat_id": LEGACY_LOCKED_CHANNEL,
        "chat_name": "control-tower",
        "thread_id": LEGACY_LOCKED_THREAD,
        "user_id": "1279454038731264061",
    }
    assert LEGACY_LOCKED_CHANNEL in root_record["prompt"]
    assert LEGACY_LOCKED_CHANNEL in thread_record["prompt"]
    assert LEGACY_LOCKED_THREAD in thread_record["prompt"]
    assert "1526870121677848636" not in root_record["prompt"]
    assert "1526870121677848636" not in thread_record["prompt"]

    plan_rows = {row["job_id"]: row for row in plan["records"]}
    for job_id in {"06ef64d72891", "e62f55ca93ca"}:
        target = plan_rows[job_id]["target"]
        assert target["approved_guild_target"] == package.REVIEWED_GUILD_TARGETS[
            job_id
        ]
        assert target["historical_source_delivery_eligible"] is False


def test_plan_bootstraps_from_static_edge_requirements_without_live_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbid_live_readiness(*_args, **_kwargs):
        raise AssertionError("live readiness cannot be collected during packaging")

    monkeypatch.setattr(
        package,
        "validate_operational_edge_readiness",
        forbid_live_readiness,
        raising=False,
    )
    _raw, build = _build(monkeypatch)

    requirements = build.plan["operational_edge_requirements"]
    assert requirements == package._operational_edge_requirements("a" * 40)
    assert requirements["fresh_runtime_readiness_required"] is True
    assert requirements["readiness_receipt_embedded"] is False
    assert build.plan["collector_execution_readiness"][
        "scoped_execution_edge_receipt_sha256"
    ] is None


def test_build_carries_one_exact_inventory_across_clock_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_at = iter(
        ("2027-01-15T08:00:00Z", "2027-01-15T08:00:01Z")
    )
    monkeypatch.setattr(
        production_cron_migration,
        "_now",
        lambda: next(observed_at),
    )
    raw, build = _build(monkeypatch)

    assert build.inventory["inventory_sha256"] == build.plan["inventory_sha256"]
    assert package.validate_packaged_continuity_plan(
        build.plan,
        inventory=build.inventory,
        require_executable=True,
    ) == build.plan

    fresh_envelope = production_cron_migration.inventory_jobs_bytes(raw)
    assert fresh_envelope["source_store_sha256"] == build.inventory[
        "source_store_sha256"
    ]
    assert fresh_envelope["inventory_sha256"] != build.inventory[
        "inventory_sha256"
    ]
    with pytest.raises(
        package.ProductionCronContinuityPackageError,
        match="packaged_plan_inventory_drifted",
    ):
        package.validate_packaged_continuity_plan(
            build.plan,
            inventory=fresh_envelope,
            require_executable=True,
        )


@pytest.mark.parametrize(
    "source",
    [
        {
            "id": "arbitrary001",
            "deliver": f"discord:{LEGACY_LOCKED_CHANNEL}",
        },
        {
            "id": "arbitrary002",
            "origin": {
                "platform": "discord",
                "chat_id": LEGACY_LOCKED_CHANNEL,
                "thread_id": LEGACY_LOCKED_THREAD,
            },
        },
        {
            "id": "arbitrary003",
            "deliver": "discord:1526999999999999999",
        },
    ],
)
def test_delivery_shape_never_infers_guild_authority(source: dict) -> None:
    assert package._target_from_delivery(source) is None


def test_artifact_stage_is_complete_private_and_replay_exact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _raw, build = _build(monkeypatch)
    inventory = build.inventory
    root = tmp_path / "stage"

    first = package.write_packaged_continuity_artifacts(
        output_root=root,
        build=build,
        inventory=inventory,
    )
    second = package.write_packaged_continuity_artifacts(
        output_root=root,
        build=build,
        inventory=inventory,
    )

    assert first == second
    assert first["file_count"] == 45
    assert first["units_installed"] is False
    assert first["timers_enabled"] is False
    assert first["timers_started"] is False
    assert first["jobs_store_mutated"] is False
    private = root / package.REPLACEMENT_BUNDLE_RELATIVE_PATH
    public_plan = root / package.PLAN_RELATIVE_PATH
    assert stat.S_IMODE(private.stat().st_mode) == 0o600
    assert stat.S_IMODE(public_plan.stat().st_mode) == 0o640

    private.write_bytes(private.read_bytes() + b" ")
    with pytest.raises(
        package.ProductionCronContinuityPackageError,
        match="artifact_digest_mismatch",
    ):
        package.validate_packaged_continuity_artifacts(
            output_root=root,
            artifact_index=first,
            inventory=inventory,
        )


def test_target_transform_replaces_twenty_four_and_preserves_archived_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw, build = _build(monkeypatch)
    original = json.loads(raw)

    target_raw = cutover_runtime.build_target_jobs_bytes(
        source_store=raw,
        continuity_plan=build.plan,
        replacement_bundle=build.replacement_bundle,
    )
    target = json.loads(target_raw)
    by_id = {job["id"]: job for job in target["jobs"]}

    assert json.loads(raw) == original
    assert by_id[package.OWNER_JOB_ID]["enabled"] is False
    assert by_id[package.OWNER_JOB_ID]["state"] == "paused"
    expected_legacy = copy.deepcopy(original["jobs"][-1])
    expected_legacy["next_run_at"] = None
    assert by_id[production_cron_migration.LEGACY_AUTO_SYNC_JOB_ID] == (
        expected_legacy
    )
    assert sum(
        job.get("enabled") is not False
        and job.get("provider") == review.PRIMARY_PROVIDER
        and job.get("model") == review.PRIMARY_MODEL
        for job in target["jobs"]
    ) == 25
    assert all(
        job.get("enabled") is False
        or job.get("script") is None
        for job in target["jobs"]
    )


def test_target_transform_refuses_inflight_disabled_legacy_auto_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw, build = _build(monkeypatch)
    source = json.loads(raw)
    source["jobs"][-1]["fire_claim"] = {
        "at": "2026-07-15T00:00:00+00:00",
        "by": "legacy-host:123",
    }
    claimed = json.dumps(source, sort_keys=True).encode("utf-8") + b"\n"

    with pytest.raises(
        cutover_runtime.ProductionCronCutoverRuntimeError,
        match="legacy_auto_sync_not_quiescent",
    ):
        cutover_runtime.build_target_jobs_bytes(
            source_store=claimed,
            continuity_plan=build.plan,
            replacement_bundle=build.replacement_bundle,
        )


def test_plan_or_bundle_tamper_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _raw, build = _build(monkeypatch)
    inventory = build.inventory
    drifted = copy.deepcopy(build.replacement_bundle)
    drifted["records"][0]["model"] = "gpt-5.5"

    with pytest.raises(
        package.ProductionCronContinuityPackageError,
        match="replacement_bundle_invalid",
    ):
        package.validate_replacement_bundle(drifted)

    drifted_plan = copy.deepcopy(build.plan)
    drifted_plan["owner_input_required_job_ids"] = [package.OWNER_JOB_ID]
    with pytest.raises(
        package.ProductionCronContinuityPackageError,
        match="packaged_plan_invalid",
    ):
        package.validate_packaged_continuity_plan(
            drifted_plan,
            inventory=inventory,
            require_executable=True,
        )

    drifted_requirements = copy.deepcopy(build.plan)
    requirements = drifted_requirements["operational_edge_requirements"]
    requirements["required_jobs"][0]["operation_id"] = "tampered.operation"
    requirements["requirements_sha256"] = package._sha256(
        package._canonical({
            key: value
            for key, value in requirements.items()
            if key != "requirements_sha256"
        })
    )
    drifted_requirements["plan_sha256"] = package._sha256(
        package._canonical({
            key: value
            for key, value in drifted_requirements.items()
            if key != "plan_sha256"
        })
    )
    with pytest.raises(
        package.ProductionCronContinuityPackageError,
        match="packaged_plan_readiness_invalid",
    ):
        package.validate_packaged_continuity_plan(
            drifted_requirements,
            inventory=inventory,
            require_executable=True,
        )

    drifted_namespace = copy.deepcopy(build.plan)
    readiness = drifted_namespace["collector_execution_readiness"]
    readiness["unit_namespace_readiness_packaged"] = True
    readiness["unit_namespace_readiness_receipt_sha256"] = "9" * 64
    readiness["unit_namespace_readiness_job_count"] = 21
    readiness["direct_dependencies_ready"] = True
    readiness["readiness_sha256"] = package._sha256(
        package._canonical({
            key: value
            for key, value in readiness.items()
            if key != "readiness_sha256"
        })
    )
    drifted_namespace["plan_sha256"] = package._sha256(
        package._canonical({
            key: value
            for key, value in drifted_namespace.items()
            if key != "plan_sha256"
        })
    )
    with pytest.raises(
        package.ProductionCronContinuityPackageError,
        match="packaged_plan_readiness_invalid",
    ):
        package.validate_packaged_continuity_plan(
            drifted_namespace,
            inventory=inventory,
            require_executable=True,
        )

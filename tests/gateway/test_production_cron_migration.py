from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from gateway import production_cron_migration as migration
from gateway.production_model_sovereignty_runtime import (
    validate_production_cron_jobs,
)


APPROVAL_ID = "11111111-1111-4111-8111-111111111111"


def _agent_job(job_id: str = "agent-ok", **overrides):
    value = {
        "id": job_id,
        "name": "Primary model task",
        "enabled": True,
        "no_agent": False,
        "prompt": "Inspect the evidence and decide the next step.",
        "script": None,
        "workdir": None,
        "deliver": "local",
        "provider": "openai-codex",
        "model": "gpt-5.6-sol",
        "base_url": None,
        "enabled_toolsets": None,
    }
    value.update(overrides)
    return value


def _legacy_auto_sync(*, enabled: bool = False):
    return {
        "id": migration.LEGACY_AUTO_SYNC_JOB_ID,
        "name": migration.LEGACY_AUTO_SYNC_JOB_NAME,
        "enabled": enabled,
        "no_agent": True,
        "prompt": "",
        "script": "/opt/legacy/fork-sync.py --execute",
        "deliver": "origin",
        "provider": None,
        "model": None,
    }


def _store(jobs) -> bytes:
    return json.dumps(
        {"jobs": jobs, "updated_at": "2026-07-14T00:00:00Z"},
        indent=2,
    ).encode() + b"\n"


def _review_dispositions(
    inventory,
    *,
    incompatible_disposition: str = "preserve_inert",
):
    decisions = [
        {
            "index": item["index"],
            "record_sha256": item["record_sha256"],
            "disposition": incompatible_disposition,
            "target": (
                None
                if incompatible_disposition == "preserve_inert"
                else {
                    "kind": "production_agent_job",
                    "replacement_record_sha256": "a" * 64,
                }
            ),
        }
        for item in inventory["incompatible_enabled_records"]
    ]
    legacy = inventory["legacy_auto_sync"]
    if legacy.get("present"):
        key = (legacy["index"], legacy["record_sha256"])
        decisions = [
            decision
            for decision in decisions
            if (decision["index"], decision["record_sha256"]) != key
        ]
        decisions.append(
            {
                "index": legacy["index"],
                "record_sha256": legacy["record_sha256"],
                "disposition": "replaced_by_packaged_rail",
                "target": {
                    "kind": "packaged_mechanical_job",
                    "job_id": migration.MECHANICAL_RAIL_JOB_ID,
                    "package_manifest_sha256": "b" * 64,
                },
            }
        )
    return decisions


def test_inventory_uses_exact_gateway_validator_and_records_no_job_content() -> None:
    secret_prompt = "private operational prompt must not enter inventory"
    raw = _store(
        [
            _agent_job(),
            _agent_job(
                "script-job",
                prompt=secret_prompt,
                script="/srv/private/collector.sh",
            ),
            _agent_job("widened", enabled_toolsets=["code_execution"]),
            _legacy_auto_sync(enabled=False),
        ]
    )
    result = migration.inventory_jobs_bytes(raw)
    encoded = json.dumps(result)

    assert result["job_count"] == 4
    assert result["enabled_count"] == 3
    assert result["compatible_enabled_count"] == 1
    assert result["incompatible_enabled_count"] == 2
    assert result["disabled_count"] == 1
    assert [
        item["validation_code"]
        for item in result["incompatible_enabled_records"]
    ] == [
        "production_cron_local_script_forbidden",
        "production_cron_enabled_toolset_not_allowed",
    ]
    assert result["legacy_auto_sync"]["present"] is True
    assert result["legacy_auto_sync"]["enabled"] is False
    assert (
        result["legacy_auto_sync"]["replacement_rail_job_id"]
        == migration.MECHANICAL_RAIL_JOB_ID
    )
    assert result["migration_required_for_gateway_startup"] is True
    assert result["job_executed"] is False
    assert result["prompt_or_script_content_recorded"] is False
    assert secret_prompt not in encoded
    assert "/srv/private/collector.sh" not in encoded
    assert "/opt/legacy/fork-sync.py" not in encoded
    assert result["migration_plan_payload"]["review_complete"] is False
    assert result["migration_plan_payload"]["cutover_executable"] is False
    assert result["migration_plan_payload"]["blanket_inert_migration_allowed"] is False
    assert migration.validate_inventory(result) == result


def test_owner_plan_is_exact_inventory_bound() -> None:
    inventory = migration.inventory_jobs_bytes(
        _store([_agent_job(script="collect.py")])
    )
    plan = migration.build_owner_approved_plan(
        inventory,
        dispositions=_review_dispositions(inventory),
        approval_id=APPROVAL_ID,
        approved_by="owner",
    )
    assert plan["source_store_sha256"] == inventory["source_store_sha256"]
    assert plan["inventory_sha256"] == inventory["inventory_sha256"]
    assert plan["approved_by"] == "owner"
    assert plan["review_complete"] is True
    assert plan["cutover_executable"] is True
    assert plan["execute_job_during_migration"] is False
    assert migration.validate_owner_approved_plan(
        inventory,
        plan,
        "b" * 64,
    ) == plan

    tampered = copy.deepcopy(inventory)
    tampered["incompatible_enabled_count"] = 0
    with pytest.raises(
        migration.ProductionCronMigrationError,
        match="inventory_invalid",
    ):
        migration.build_owner_approved_plan(
            tampered,
            dispositions=_review_dispositions(inventory),
            approval_id=APPROVAL_ID,
            approved_by="owner",
        )
    with pytest.raises(
        migration.ProductionCronMigrationError,
        match="owner_approval_required",
    ):
        migration.build_owner_approved_plan(
            inventory,
            dispositions=_review_dispositions(inventory),
            approval_id=APPROVAL_ID,
            approved_by="operator",
        )


def test_owner_validator_dispatches_the_current_packaged_plan_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gateway import production_cron_continuity_package as packaged

    inventory = migration.inventory_jobs_bytes(_store([_agent_job()]))
    plan = {"schema": packaged.PLAN_SCHEMA}
    validated = {"validated": True}
    calls = []

    def validate(value, **kwargs):
        calls.append((value, kwargs))
        return validated

    monkeypatch.setattr(packaged, "validate_packaged_continuity_plan", validate)

    assert migration.validate_owner_approved_plan(
        inventory,
        plan,
        "b" * 64,
    ) == validated
    assert calls == [
        (
            plan,
            {
                "inventory": inventory,
                "expected_mechanical_job_package_manifest_sha256": "b" * 64,
                "require_executable": True,
            },
        )
    ]


def test_continuity_plan_requires_every_incompatible_record_and_forbids_blanket_inert() -> None:
    inventory = migration.inventory_jobs_bytes(
        _store(
            [
                _agent_job("script", script="collect.py"),
                _agent_job("workdir", workdir="/srv/legacy"),
            ]
        )
    )
    decisions = _review_dispositions(inventory)
    with pytest.raises(
        migration.ProductionCronMigrationError,
        match="not_exhaustive",
    ):
        migration.build_owner_approved_plan(
            inventory,
            dispositions=decisions[:1],
            approval_id=APPROVAL_ID,
            approved_by="owner",
        )

    plan = migration.build_owner_approved_plan(
        inventory,
        dispositions=decisions,
        approval_id=APPROVAL_ID,
        approved_by="owner",
    )
    assert plan["review_complete"] is True
    assert plan["blanket_inert_migration_allowed"] is False
    assert plan["cutover_executable"] is False
    assert migration.validate_owner_approved_plan(
        inventory,
        plan,
        "b" * 64,
    ) == plan


def test_apply_preserves_all_records_and_only_makes_incompatible_enabled_inert(
    tmp_path: Path,
) -> None:
    original_jobs = [
        _agent_job(),
        _agent_job("script-job", script="collect.py"),
        _legacy_auto_sync(enabled=False),
    ]
    original = _store(original_jobs)
    jobs_path = tmp_path / "cron/jobs.json"
    jobs_path.parent.mkdir()
    jobs_path.write_bytes(original)
    jobs_path.chmod(0o600)
    inventory = migration.inventory_jobs_bytes(original)
    plan = migration.build_owner_approved_plan(
        inventory,
        dispositions=_review_dispositions(inventory),
        approval_id=APPROVAL_ID,
        approved_by="owner",
    )
    evidence = tmp_path / "evidence"

    receipt = migration.apply_inert_migration(
        jobs_path=jobs_path,
        inventory=inventory,
        owner_approved_plan=plan,
        expected_owner_plan_sha256=plan["owner_approved_plan_sha256"],
        expected_package_manifest_sha256="b" * 64,
        evidence_root=evidence,
        gateway_inactive=lambda: True,
    )
    updated = json.loads(jobs_path.read_text())["jobs"]

    assert len(updated) == len(original_jobs)
    assert updated[0] == original_jobs[0]
    assert updated[1] == {**original_jobs[1], "enabled": False, "state": "paused"}
    assert updated[2] == original_jobs[2]
    validate_production_cron_jobs(updated)
    assert jobs_path.stat().st_mode & 0o777 == 0o600
    archive = Path(receipt["archive_path"])
    assert archive.read_bytes() == original
    assert archive.stat().st_mode & 0o777 == 0o600
    assert receipt["migrated_record_count"] == 1
    assert receipt["records_deleted"] is False
    assert receipt["job_executed"] is False
    assert receipt["provider_or_model_invoked"] is False
    assert receipt["discord_delivery_attempted"] is False
    assert receipt["prompt_or_script_content_recorded"] is False
    assert (evidence / "latest.json").is_file()

    # Same owner-bound transaction is idempotent after the source store has
    # become the prepared target; it verifies the archive and reuses one exact
    # receipt instead of pausing or rewriting records again.
    second = migration.apply_inert_migration(
        jobs_path=jobs_path,
        inventory=inventory,
        owner_approved_plan=plan,
        expected_owner_plan_sha256=plan["owner_approved_plan_sha256"],
        expected_package_manifest_sha256="b" * 64,
        evidence_root=evidence,
        gateway_inactive=lambda: True,
    )
    assert second == receipt
    assert json.loads(jobs_path.read_text())["jobs"] == updated
    assert len(list((evidence / "receipts").glob("migration-*.json"))) == 1

    rollback = migration.restore_inert_migration(
        jobs_path=jobs_path,
        migration_receipt=receipt,
        expected_receipt_sha256=receipt["receipt_sha256"],
        evidence_root=evidence,
        gateway_inactive=lambda: True,
    )
    assert jobs_path.read_bytes() == original
    assert rollback["gateway_start_allowed"] is False
    assert rollback["restored_store_sha256"] == inventory["source_store_sha256"]
    second_rollback = migration.restore_inert_migration(
        jobs_path=jobs_path,
        migration_receipt=receipt,
        expected_receipt_sha256=receipt["receipt_sha256"],
        evidence_root=evidence,
        gateway_inactive=lambda: True,
    )
    assert second_rollback == rollback


def test_enabled_legacy_auto_sync_is_only_preserved_inert_never_admitted_to_rail(
    tmp_path: Path,
) -> None:
    original = _store([_legacy_auto_sync(enabled=True)])
    jobs_path = tmp_path / "cron/jobs.json"
    jobs_path.parent.mkdir()
    jobs_path.write_bytes(original)
    inventory = migration.inventory_jobs_bytes(original)
    plan = migration.build_owner_approved_plan(
        inventory,
        dispositions=_review_dispositions(inventory),
        approval_id=APPROVAL_ID,
        approved_by="owner",
    )
    assert plan["cutover_executable"] is False
    assert plan["continuity_dispositions"][0]["disposition"] == (
        "replaced_by_packaged_rail"
    )
    with pytest.raises(
        migration.ProductionCronMigrationError,
        match="owner_plan_invalid|not_explicit",
    ):
        migration.apply_inert_migration(
            jobs_path=jobs_path,
            inventory=inventory,
            owner_approved_plan=plan,
            expected_owner_plan_sha256=plan["owner_approved_plan_sha256"],
            expected_package_manifest_sha256="b" * 64,
            evidence_root=tmp_path / "evidence",
            gateway_inactive=lambda: True,
        )
    assert jobs_path.read_bytes() == original


def test_apply_fails_closed_before_write_without_stopped_gateway_or_exact_plan(
    tmp_path: Path,
) -> None:
    original = _store([_agent_job(script="collect.py")])
    jobs_path = tmp_path / "cron/jobs.json"
    jobs_path.parent.mkdir()
    jobs_path.write_bytes(original)
    inventory = migration.inventory_jobs_bytes(original)
    plan = migration.build_owner_approved_plan(
        inventory,
        dispositions=_review_dispositions(inventory),
        approval_id=APPROVAL_ID,
        approved_by="owner",
    )
    with pytest.raises(
        migration.ProductionCronMigrationError,
        match="gateway_must_be_inactive",
    ):
        migration.apply_inert_migration(
            jobs_path=jobs_path,
            inventory=inventory,
            owner_approved_plan=plan,
            expected_owner_plan_sha256=plan["owner_approved_plan_sha256"],
            expected_package_manifest_sha256="b" * 64,
            evidence_root=tmp_path / "evidence",
            gateway_inactive=lambda: False,
        )
    assert jobs_path.read_bytes() == original

    with pytest.raises(
        migration.ProductionCronMigrationError,
        match="owner_plan_invalid",
    ):
        migration.apply_inert_migration(
            jobs_path=jobs_path,
            inventory=inventory,
            owner_approved_plan=plan,
            expected_owner_plan_sha256="f" * 64,
            expected_package_manifest_sha256="b" * 64,
            evidence_root=tmp_path / "evidence",
            gateway_inactive=lambda: True,
        )
    assert jobs_path.read_bytes() == original


def test_apply_rejects_source_drift_without_partial_migration(tmp_path: Path) -> None:
    original = _store([_agent_job(script="collect.py")])
    jobs_path = tmp_path / "cron/jobs.json"
    jobs_path.parent.mkdir()
    jobs_path.write_bytes(original)
    inventory = migration.inventory_jobs_bytes(original)
    plan = migration.build_owner_approved_plan(
        inventory,
        dispositions=_review_dispositions(inventory),
        approval_id=APPROVAL_ID,
        approved_by="owner",
    )
    drifted = _store(
        [
            _agent_job(script="collect.py"),
            _agent_job("new-job", script="other.py"),
        ]
    )
    jobs_path.write_bytes(drifted)
    with pytest.raises(
        migration.ProductionCronMigrationError,
        match="store_digest_drifted",
    ):
        migration.apply_inert_migration(
            jobs_path=jobs_path,
            inventory=inventory,
            owner_approved_plan=plan,
            expected_owner_plan_sha256=plan["owner_approved_plan_sha256"],
            expected_package_manifest_sha256="b" * 64,
            evidence_root=tmp_path / "evidence",
            gateway_inactive=lambda: True,
        )
    assert jobs_path.read_bytes() == drifted

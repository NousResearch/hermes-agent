from __future__ import annotations

import subprocess

import pytest

from scripts.canary.foundation import (
    FORBIDDEN_CANARY_SECRET_NAMES,
    FoundationSpec,
    build_plan,
    execute_plan,
)


def test_phase_one_plan_is_dedicated_private_and_contains_no_host_or_secret_step():
    plan = build_plan()
    rendered = str(plan.report())
    argv = [value for step in plan.steps for value in step.argv]

    assert plan.spec.network == "muncho-canary-vpc"
    assert plan.spec.subnet_cidr == "10.90.0.0/24"
    assert plan.spec.private_service_range_cidr == "10.91.0.0/24"
    assert plan.spec.service_account_name == "muncho-canary-v2-runtime"
    assert plan.spec.sql_instance == "muncho-canary-pg18-v2"
    assert "--tier=db-f1-micro" in rendered
    assert "--database-version=POSTGRES_18" in rendered
    assert "--no-assign-ip" in rendered
    assert "--ssl-mode=ENCRYPTED_ONLY" in rendered
    assert "--no-backup" in rendered
    assert "--no-storage-auto-increase" in rendered
    assert "--deletion-protection" in rendered
    assert "roles/cloudsql.client" not in rendered
    assert "roles/secretmanager.secretAccessor" not in rendered
    assert not any(step.argv[:4] == ("gcloud", "compute", "instances", "create") for step in plan.steps)
    assert not any(value == "firewall-rules" for value in argv)
    assert not any(value == "secrets" for value in argv)
    assert all(name not in rendered for name in FORBIDDEN_CANARY_SECRET_NAMES)
    assert plan.architecture["creates_vm"] is False
    assert plan.architecture["creates_network_rules"] is False
    assert plan.architecture["creates_secret_manager_resources"] is False


def test_phase_one_step_order_is_dependency_bounded():
    plan = build_plan()

    assert [step.name for step in plan.steps] == [
        "create_isolated_vpc",
        "create_isolated_subnet",
        "reserve_private_service_range",
        "connect_private_service_networking",
        "create_runtime_service_account",
        "grant_logging_writer",
        "grant_monitoring_writer",
        "create_isolated_postgres",
        "create_canonical_database",
    ]


def test_plan_contains_no_secret_values_or_secret_input_flags():
    argv = [argument for step in build_plan().steps for argument in step.argv]

    assert not any("password=" in value.casefold() for value in argv)
    assert "--data-file=-" not in argv
    assert not any("secret" in value.casefold() for value in argv)


def test_plan_is_deterministic():
    first = build_plan()
    second = build_plan()

    assert first.sha256 == second.sha256
    assert len(first.sha256) == 64


def test_apply_rejects_a_digest_mismatch_before_any_command():
    plan = build_plan()
    called = []

    with pytest.raises(RuntimeError, match="digest"):
        execute_plan(
            plan,
            approved_plan_sha256="0" * 64,
            preflight={},
            runner=lambda argv: called.append(argv),
            now_unix=1_000,
        )

    assert called == []


def test_apply_executes_exact_steps_after_fresh_bound_preflight():
    plan = build_plan()
    called = []

    def runner(argv):
        called.append(tuple(argv))
        return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")

    receipt = execute_plan(
        plan,
        approved_plan_sha256=plan.sha256,
        preflight={
            "schema": "muncho-isolated-canary-foundation-preflight.v2",
            "ok": True,
            "plan_sha256": plan.sha256,
            "collected_at_unix": 1_000,
            "satisfied_steps": [],
        },
        runner=runner,
        now_unix=1_001,
    )

    assert receipt["ok"] is True
    assert called == [step.argv for step in plan.steps]
    assert all(
        set(item)
        == {"name", "returncode", "stdout_sha256", "stderr_sha256", "result"}
        for item in receipt["receipts"]
    )


def test_apply_skips_only_preflight_verified_existing_steps():
    plan = build_plan()
    skipped = plan.steps[0].name
    called = []

    def runner(argv):
        called.append(tuple(argv))
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    receipt = execute_plan(
        plan,
        approved_plan_sha256=plan.sha256,
        preflight={
            "schema": "muncho-isolated-canary-foundation-preflight.v2",
            "ok": True,
            "plan_sha256": plan.sha256,
            "collected_at_unix": 1_000,
            "satisfied_steps": [skipped],
        },
        runner=runner,
        now_unix=1_001,
    )

    assert receipt["receipts"][0]["result"] == "verified_existing"
    assert called == [step.argv for step in plan.steps[1:]]


def test_spec_rejects_a_zone_outside_the_region():
    with pytest.raises(ValueError, match="zone"):
        build_plan(FoundationSpec(zone="europe-west1-b"))

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from gateway import canonical_writer_production_cutover as cutover
from gateway import production_cron_continuity_package as cron_continuity
from gateway import production_cron_migration as cron_migration
from ops.muncho.runtime import mechanical_job_rail
from scripts.canary import package_production_cutover_artifacts as package
from scripts.canary import production_cutover_initial_collector as collector
from tests.gateway.test_canonical_writer_production_cutover import (
    NOW,
    Services as CutoverServices,
    _approval,
    _freeze,
    _mechanical_package,
    _service,
    _snapshot,
)
from tests.gateway.test_production_cron_continuity_package import (
    EDGE_BOOT_ID_SHA256,
    EDGE_OBSERVED_AT_UNIX,
    _collector_package,
    _collector_execution_readiness,
    _operational_edge_receipt,
    _source_store,
)
from tests.scripts.canary.test_package_production_cutover_artifacts import (
    REVISION,
    _load_artifact,
    _release,
    _sha_json,
    _unit_inputs,
)


def _target() -> dict:
    return {
        "project": "adventico-ai-platform",
        "zone": "europe-west3-a",
        "vm": "ai-platform-runtime-01",
        "database": "ai_platform_brain",
        "sql_instance": "production-pg18",
        "sql_host": "10.20.30.40",
        "tls_server_name": "production.example.internal",
        "port": 5432,
        "writer_login": "muncho_production_writer_login",
    }


def _observation() -> dict:
    return {
        "source_owner": "legacy_event_owner",
        "relation_oid": "16421",
        "source_row_count": 14_073,
        "canonical14_sha256": "1" * 64,
        "extended19_sha256": "2" * 64,
        "occurred_at_cutoff": "2026-07-14T09:00:00+00:00",
        "inserted_at_cutoff": "2026-07-14T09:00:01+00:00",
        "relation_identity_sha256": "3" * 64,
        "acl_identity_sha256": "4" * 64,
        "index_identity_sha256": "5" * 64,
        "observed_at_unix": 1_800_000_000,
    }


def _iso(unix: int) -> str:
    return datetime.fromtimestamp(unix, tz=timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


def _cron_inventory(now: int) -> dict:
    value = cron_migration.inventory_jobs_bytes(b'{"jobs":[]}')
    value["created_at"] = _iso(now)
    value["inventory_sha256"] = package._sha256(
        package._canonical_bytes({
            name: item
            for name, item in value.items()
            if name != "inventory_sha256"
        })
    )
    return value


def _host_facts(now: int) -> dict:
    unsigned = {
        "schema": mechanical_job_rail.HOST_FACTS_SCHEMA,
        "collected_at": _iso(now),
        "github_cli": {
            "path": "/usr/bin/gh",
            "regular": True,
            "nlink": 1,
            "uid": 0,
            "gid": 0,
            "mode": "0755",
            "group_or_other_writable": False,
            "sha256": "8" * 64,
        },
        "git": {
            "path": "/usr/bin/git",
            "regular": True,
            "nlink": 1,
            "uid": 0,
            "gid": 0,
            "mode": "0755",
            "group_or_other_writable": False,
            "sha256": "9" * 64,
        },
        "github_credential": {
            "path": "/etc/muncho/fork-auto-sync/github-token",
            "regular": True,
            "nlink": 1,
            "uid": 0,
            "gid": 0,
            "mode": "0400",
            "content_recorded": False,
            "size_recorded": False,
            "digest_recorded": False,
        },
        "provider_or_model_credential_observed": False,
        "discord_credential_observed": False,
    }
    return {
        **unsigned,
        "host_facts_sha256": package._sha256(
            package._canonical_bytes(unsigned)
        ),
    }


def _cron_continuity_fixture(
    monkeypatch: pytest.MonkeyPatch,
    *,
    now: int,
    mechanical_package: dict,
) -> tuple[dict, cron_continuity.HostContinuityDerivation]:
    monkeypatch.setattr(cron_migration, "_now", lambda: _iso(now))
    raw, _catalog = _source_store(monkeypatch)
    inventory = cron_migration.inventory_jobs_bytes(raw)
    build = cron_continuity.build_packaged_continuity_plan(
        source_store=raw,
        collector_package=_collector_package(),
        collector_execution_readiness=_collector_execution_readiness(),
        operational_edge_readiness=_operational_edge_receipt(),
        mechanical_job_package_manifest_sha256=(
            mechanical_package["manifest_sha256"]
        ),
        cutover_runtime_sha256="d" * 64,
        cutover_entrypoint_sha256="e" * 64,
        expected_boot_id_sha256=EDGE_BOOT_ID_SHA256,
        now_unix=EDGE_OBSERVED_AT_UNIX,
    )
    return inventory, cron_continuity.HostContinuityDerivation(
        build=build,
        inventory=inventory,
    )


def test_initial_observe_is_release_bound_without_a_freeze_plan(
    tmp_path: Path,
    monkeypatch,
) -> None:
    release = _release(tmp_path)
    manifest = package.build_release_artifacts(
        release,
        REVISION,
        unit_inputs=_unit_inputs(),
    )
    item = manifest["artifacts"]["production-observe"]
    runtime = _load_artifact(
        Path(item["path"]),
        "production_initial_observe_artifact",
    )
    monkeypatch.setattr(runtime, "_self_sha256", lambda: item["sha256"])
    artifact_path = (
        "/opt/adventico-ai-platform/hermes-agent-releases/"
        f"hermes-agent-{REVISION[:12]}/ops/muncho/cutover/artifacts/"
        "production-observe"
    )
    unsigned = {
        "schema": runtime.INITIAL_OBSERVATION_REQUEST_SCHEMA,
        "action": "observe_initial",
        "release_revision": REVISION,
        "target": _target(),
        "artifact": {"path": artifact_path, "sha256": item["sha256"]},
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    request = {**unsigned, "request_sha256": _sha_json(unsigned)}

    action, context, apply_receipt = runtime._validate_request(request)

    assert action == "observe_initial"
    assert apply_receipt is None
    assert "plan" not in context
    assert "initial_snapshot" not in context
    assert "source_owner" not in context
    monkeypatch.setattr(runtime, "_database_state", lambda _value: "legacy")
    monkeypatch.setattr(
        runtime,
        "_legacy_observation",
        lambda _value, *, archived: _observation(),
    )
    monkeypatch.setattr(
        runtime,
        "_pgpass_user",
        lambda _target_value: "legacy_event_owner",
    )
    snapshot = runtime.execute(action, context, apply_receipt)
    assert snapshot["source_owner"] == "legacy_event_owner"
    assert snapshot["source_row_count"] == 14_073


def test_initial_observe_derives_source_owner_from_fixed_pgpass(
    tmp_path: Path,
    monkeypatch,
) -> None:
    release = _release(tmp_path)
    manifest = package.build_release_artifacts(
        release,
        REVISION,
        unit_inputs=_unit_inputs(),
    )
    item = manifest["artifacts"]["production-observe"]
    runtime = _load_artifact(
        Path(item["path"]),
        "production_initial_pgpass_artifact",
    )
    monkeypatch.setattr(
        runtime,
        "_read_exact_file",
        lambda *_args, **_kwargs: (
            b"production.example.internal:5432:ai_platform_brain:"
            b"legacy_event_owner:opaque\\:password\n"
        ),
    )

    assert runtime._pgpass_user(_target()) == "legacy_event_owner"

    wrong = _target()
    wrong["tls_server_name"] = "other.example.internal"
    with pytest.raises(runtime.ArtifactError, match="pgpass_identity_invalid"):
        runtime._pgpass_user(wrong)


def test_initial_observe_rejects_nonproduction_host_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    release = _release(tmp_path)
    manifest = package.build_release_artifacts(
        release,
        REVISION,
        unit_inputs=_unit_inputs(),
    )
    item = manifest["artifacts"]["production-observe"]
    runtime = _load_artifact(
        Path(item["path"]),
        "production_initial_wrong_host_artifact",
    )
    monkeypatch.setattr(runtime, "_self_sha256", lambda: item["sha256"])
    target = _target()
    target["vm"] = "other-runtime"
    unsigned = {
        "schema": runtime.INITIAL_OBSERVATION_REQUEST_SCHEMA,
        "action": "observe_initial",
        "release_revision": REVISION,
        "target": target,
        "artifact": {
            "path": (
                "/opt/adventico-ai-platform/hermes-agent-releases/"
                f"hermes-agent-{REVISION[:12]}/ops/muncho/cutover/artifacts/"
                "production-observe"
            ),
            "sha256": item["sha256"],
        },
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    request = {**unsigned, "request_sha256": _sha_json(unsigned)}

    with pytest.raises(runtime.ArtifactError, match="initial_request_invalid"):
        runtime._validate_request(request)


def test_root_initial_collector_uses_only_sealed_target_and_public_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 1_800_000_000
    release = _release(tmp_path)
    fixed = (
        cutover.PRODUCTION_RELEASE_BASE / f"hermes-agent-{REVISION[:12]}"
    )
    package.build_release_artifacts(
        release,
        REVISION,
        release_address=fixed,
        unit_inputs=_unit_inputs(),
    )
    mechanical = _mechanical_package(_host_facts(now))
    cron_inventory, continuity = _cron_continuity_fixture(
        monkeypatch,
        now=now,
        mechanical_package=mechanical,
    )
    captured: dict = {}

    def runner(command, **kwargs):
        captured["command"] = command
        captured["request"] = json.loads(kwargs["input"][:-1])
        return SimpleNamespace(
            returncode=0,
            stdout=package._canonical_bytes(
                _snapshot(14_073, observed_at=now).to_mapping()
            )
            + b"\n",
            stderr=b"",
        )

    class Services:
        def observe_gateway(self):
            return _service(
                cutover.GATEWAY_UNIT,
                active=True,
                digest="7" * 64,
                observed_at=now,
            )

        def observe_writer(self):
            return _service(
                cutover.WRITER_UNIT,
                active=False,
                digest=None,
                observed_at=now,
            )

        def observe_connector(self):
            return _service(
                cutover.CONNECTOR_UNIT,
                active=False,
                digest=None,
                observed_at=now,
            )

    receipt = collector.collect_initial_observations(
        REVISION,
        release_root=release,
        unit_inputs=_unit_inputs(),
        runner=runner,
        services=Services(),
        clock=lambda: now,
        boot_reader=lambda: "f" * 64,
        cron_inventory_collector=lambda: cron_inventory,
        mechanical_host_facts_collector=lambda: _host_facts(now),
        mechanical_package_collector=lambda _revision, _facts: mechanical,
        cron_continuity_collector=lambda _revision, _mechanical: continuity,
        require_root=False,
    )

    assert captured["command"][1] == "observe_initial"
    assert captured["request"]["target"] == _unit_inputs()["target"]
    assert "source_owner" not in captured["request"]
    assert "plan" not in captured["request"]
    assert receipt["initial_snapshot"]["source_owner"] == "legacy_event_owner"
    assert receipt["source_boot_id_sha256"] == "f" * 64
    assert receipt["cron_inventory"]["job_count"] == 29
    assert receipt["cron_continuity_plan"] == continuity.build.plan
    assert receipt["mechanical_job_host_facts"]["github_cli"]["path"] == (
        "/usr/bin/gh"
    )
    assert receipt["mechanical_job_host_facts"]["github_credential"] == {
        "path": "/etc/muncho/fork-auto-sync/github-token",
        "regular": True,
        "nlink": 1,
        "uid": 0,
        "gid": 0,
        "mode": "0400",
        "content_recorded": False,
        "size_recorded": False,
        "digest_recorded": False,
    }
    assert receipt["mechanical_job_package"]["host_facts_sha256"] == (
        receipt["mechanical_job_host_facts"]["host_facts_sha256"]
    )
    assert receipt["mechanical_job_package"]["timer_started_by_package"] is False
    assert receipt["secret_material_recorded"] is False
    assert receipt["secret_digest_recorded"] is False

    with pytest.raises(
        collector.InitialCollectorError,
        match="initial_collector_auxiliary_fact_invalid",
    ):
        collector.collect_initial_observations(
            REVISION,
            release_root=release,
            unit_inputs=_unit_inputs(),
            runner=runner,
            services=Services(),
            clock=lambda: now,
            boot_reader=lambda: "f" * 64,
            cron_inventory_collector=lambda: _cron_inventory(now),
            mechanical_host_facts_collector=lambda: _host_facts(now),
            mechanical_package_collector=lambda _revision, _facts: mechanical,
            cron_continuity_collector=(
                lambda _revision, _mechanical: continuity
            ),
            require_root=False,
        )


def test_root_initial_collector_rejects_boot_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 1_800_000_000
    release = _release(tmp_path)
    fixed = (
        cutover.PRODUCTION_RELEASE_BASE / f"hermes-agent-{REVISION[:12]}"
    )
    package.build_release_artifacts(
        release,
        REVISION,
        release_address=fixed,
        unit_inputs=_unit_inputs(),
    )
    mechanical = _mechanical_package(_host_facts(now))
    cron_inventory, continuity = _cron_continuity_fixture(
        monkeypatch,
        now=now,
        mechanical_package=mechanical,
    )

    class Services:
        def observe_gateway(self):
            return _service(
                cutover.GATEWAY_UNIT,
                active=True,
                digest="7" * 64,
                observed_at=now,
            )

        def observe_writer(self):
            return _service(
                cutover.WRITER_UNIT,
                active=False,
                digest=None,
                observed_at=now,
            )

        def observe_connector(self):
            return _service(
                cutover.CONNECTOR_UNIT,
                active=False,
                digest=None,
                observed_at=now,
            )

    boot_ids = iter(("e" * 64, "f" * 64))

    with pytest.raises(collector.InitialCollectorError, match="live_state_invalid"):
        collector.collect_initial_observations(
            REVISION,
            release_root=release,
            unit_inputs=_unit_inputs(),
            runner=lambda *_args, **_kwargs: SimpleNamespace(
                returncode=0,
                stdout=(
                    package._canonical_bytes(
                        _snapshot(14_073, observed_at=now).to_mapping()
                    )
                    + b"\n"
                ),
                stderr=b"",
            ),
            services=Services(),
            clock=lambda: now,
            boot_reader=lambda: next(boot_ids),
            cron_inventory_collector=lambda: cron_inventory,
            mechanical_host_facts_collector=lambda: _host_facts(now),
            mechanical_package_collector=lambda _revision, _facts: mechanical,
            cron_continuity_collector=(
                lambda _revision, _mechanical: continuity
            ),
            require_root=False,
        )


def test_stopped_collector_binds_exact_freeze_and_service_identities() -> None:
    private = Ed25519PrivateKey.generate()
    services = CutoverServices()
    freeze = _freeze(private, services)
    approval = _approval(private, freeze)
    services.stop_gateway()

    receipt = collector.collect_stopped_services(
        REVISION,
        freeze_plan=freeze.to_mapping(),
        freeze_approval=approval,
        services=services,
        clock=lambda: NOW,
        boot_reader=lambda: "f" * 64,
        require_root=False,
    )

    assert receipt["freeze_plan_sha256"] == freeze.sha256
    assert receipt["freeze_approval_sha256"] == approval["approval_sha256"]
    assert cutover.ServiceObservation.from_mapping(
        receipt["gateway_stopped"]
    ).stopped
    assert receipt["secret_material_recorded"] is False
    assert receipt["secret_digest_recorded"] is False

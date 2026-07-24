from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest

from gateway.mac_ops_edge_service import DEFAULT_PROJECT_ID
from scripts.canary import package_production_cutover_artifacts as package
from scripts.canary import production_cutover_host_plan as producer
from tests.scripts.canary.test_package_production_cutover_artifacts import (
    REVISION,
    _release,
    _unit_inputs,
)


def test_effective_identity_fails_closed_without_posix_apis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(producer.os, "geteuid")
    monkeypatch.delattr(producer.os, "getegid")

    assert producer._effective_identity() is None


def test_every_fixed_target_has_exactly_one_producer_source() -> None:
    partitions = producer.HOST_ARTIFACT_SOURCE_PARTITIONS

    assert set().union(*partitions) == set(package.HOST_ARTIFACT_TARGETS)
    assert sum(len(group) for group in partitions) == len(
        package.HOST_ARTIFACT_TARGETS
    )
    assert producer.ROOT_VERIFIER_ARTIFACT_NAMES == {
        "api_bearer_verifier",
        "api_approval_verifier",
    }
    assert producer.REVIEWED_RELEASE_ARTIFACT_NAMES == {
        "gateway_connector_drop_in"
    }


def test_release_sealed_payloads_reproduce_the_manifest(tmp_path: Path) -> None:
    release = _release(tmp_path)
    inputs = _unit_inputs()
    manifest = package.build_release_artifacts(
        release,
        REVISION,
        unit_inputs=inputs,
    )

    payloads, descriptor, observed_manifest = (
        package.render_release_sealed_host_payloads(
            release_root=release,
            revision=REVISION,
            unit_inputs=inputs,
        )
    )

    assert set(payloads) == producer.RELEASE_SEALED_ARTIFACT_NAMES
    assert descriptor == manifest["sealed_runtime_artifact_request"]
    assert observed_manifest == manifest


def test_stage_uses_runtime_pinned_mac_ops_project(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class MacOpsConfigObserved(RuntimeError):
        pass

    inputs = _unit_inputs()
    assert inputs["target"]["project"] != DEFAULT_PROJECT_ID
    monkeypatch.setattr(
        producer,
        "_validate_reconciliation_intent",
        lambda *_args, **_kwargs: ({}, {}),
    )
    monkeypatch.setattr(
        producer.package,
        "render_release_sealed_host_payloads",
        lambda **_kwargs: (
            {"mac_ops_unit": b"mac-ops-unit"},
            {},
            {"host_artifact_contract": {}},
        ),
    )
    monkeypatch.setattr(
        producer,
        "_read_regular",
        lambda *_args, **_kwargs: (b"source", {}),
    )
    monkeypatch.setattr(
        producer,
        "_render_connector_unit",
        lambda *_args, **_kwargs: b"connector-unit",
    )
    monkeypatch.setattr(
        producer,
        "_render_connector_config",
        lambda *_args, **_kwargs: b"connector-config",
    )
    monkeypatch.setattr(
        producer,
        "render_production_routeback_config",
        lambda **_kwargs: b"routeback-config",
    )

    def observe_mac_ops_config(**kwargs: object) -> bytes:
        assert kwargs["project_id"] == DEFAULT_PROJECT_ID
        raise MacOpsConfigObserved

    monkeypatch.setattr(
        producer,
        "render_production_mac_ops_config",
        observe_mac_ops_config,
    )

    with pytest.raises(MacOpsConfigObserved):
        producer.stage_fixed_host_artifacts(
            REVISION,
            release_root=tmp_path,
            filesystem_root=tmp_path,
            unit_inputs=inputs,
            require_root=False,
        )


def test_staging_receipt_validates_projected_secret_foundation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _unit_inputs()
    foundation_sha256 = inputs["operational_edge_key_foundation_sha256"]
    secret_foundation = {
        "schema": producer.production_secret_stager.STAGING_SCHEMA,
        "bearer_verifier_path": str(
            producer.production_secret_stager.STAGED_API_BEARER_VERIFIER_PATH
        ),
        "bearer_verifier_sha256": "a" * 64,
        "approval_verifier_path": str(
            producer.production_secret_stager.STAGED_API_APPROVAL_VERIFIER_PATH
        ),
        "approval_verifier_sha256": "b" * 64,
        "writer_private_path": str(
            producer.production_secret_stager.STAGED_WRITER_PRIVATE_KEY_PATH
        ),
        "writer_public_key_id": inputs["writer_capability_public_key_id"],
        "edge_private_path": str(
            producer.production_secret_stager.STAGED_EDGE_PRIVATE_KEY_PATH
        ),
        "edge_public_key_id": inputs["discord_edge_receipt_public_key_id"],
        "operational_edge_key_foundation": {
            "receipt_sha256": foundation_sha256,
        },
        "operational_edge_key_foundation_sha256": foundation_sha256,
        "operational_edge_receipt_public_key_ids": inputs[
            "operational_edge_receipt_public_key_ids"
        ],
        "private_content_or_digest_recorded": False,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    staged = {
        name: {"artifact": name}
        for name in package.HOST_ARTIFACT_TARGETS
    }
    unsigned = {
        "schema": producer.STAGING_SCHEMA,
        "release_revision": REVISION,
        "release_manifest_sha256": "c" * 64,
        "host_artifact_contract_sha256": "d" * 64,
        "unit_inputs_authority_plan_sha256": inputs[
            "authority_plan_sha256"
        ],
        "unit_inputs_authority_approval_sha256": inputs[
            "authority_approval_sha256"
        ],
        "source_gateway_config_sha256": "e" * 64,
        "source_writer_config_sha256": "f" * 64,
        "secret_foundation": secret_foundation,
        "capability_topology": {},
        "staged_file_count": len(staged),
        "staged_files": staged,
        "staged_set_sha256": producer._sha(
            producer._canonical({"files": staged})
        ),
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    receipt = {
        **unsigned,
        "receipt_sha256": producer._sha(producer._canonical(unsigned)),
    }
    monkeypatch.setattr(
        producer,
        "validate_operational_edge_key_foundation",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        producer,
        "validate_production_capability_topology",
        lambda value: value,
    )

    assert producer._validate_staging_receipt(
        receipt,
        revision=REVISION,
        inputs=inputs,
    ) == receipt

    tampered = copy.deepcopy(receipt)
    tampered["secret_foundation"]["secret_material_recorded"] = True
    tampered_unsigned = {
        name: item
        for name, item in tampered.items()
        if name != "receipt_sha256"
    }
    tampered["receipt_sha256"] = producer._sha(
        producer._canonical(tampered_unsigned)
    )
    with pytest.raises(
        producer.HostPlanProducerError,
        match="host_plan_secret_foundation_invalid",
    ):
        producer._validate_staging_receipt(
            tampered,
            revision=REVISION,
            inputs=inputs,
        )


def test_create_only_staging_resumes_and_rejects_conflicts(
    tmp_path: Path,
) -> None:
    logical = Path("/staged/exact.json")
    staged = tmp_path / "staged"
    staged.mkdir(mode=0o700)
    payload = b'{"safe":true}'
    uid = os.geteuid()
    gid = os.getegid()

    producer._create_or_validate(
        logical,
        payload,
        filesystem_root=tmp_path,
        uid=uid,
        gid=gid,
    )
    producer._create_or_validate(
        logical,
        payload,
        filesystem_root=tmp_path,
        uid=uid,
        gid=gid,
    )

    assert (staged / "exact.json").read_bytes() == payload
    assert (staged / "exact.json").stat().st_mode & 0o777 == 0o400
    with pytest.raises(
        producer.HostPlanProducerError,
        match="host_plan_staging_conflict",
    ):
        producer._create_or_validate(
            logical,
            b'{"safe":false}',
            filesystem_root=tmp_path,
            uid=uid,
            gid=gid,
        )


@pytest.mark.parametrize("kind", ("symlink", "hardlink"))
def test_create_only_staging_rejects_link_substitution(
    tmp_path: Path,
    kind: str,
) -> None:
    staged = tmp_path / "staged"
    staged.mkdir(mode=0o700)
    source = staged / "source"
    source.write_bytes(b"fixed")
    source.chmod(0o400)
    target = staged / "target"
    if kind == "symlink":
        target.symlink_to(source)
    else:
        os.link(source, target)

    with pytest.raises(
        producer.HostPlanProducerError,
        match="host_plan_file_identity_invalid",
    ):
        producer._create_or_validate(
            Path("/staged/target"),
            b"fixed",
            filesystem_root=tmp_path,
            uid=os.geteuid(),
            gid=os.getegid(),
        )


def test_signed_reconciliation_intent_binds_exact_reviewed_policies() -> None:
    inputs = copy.deepcopy(_unit_inputs())
    legacy = producer._legacy_discord_policy()
    target = producer._target_discord_policy()
    inputs["discord_reconciliation_intent"].update(
        {
            "legacy_public_policy_sha256": producer._sha(
                producer._canonical(legacy)
            ),
            "target_public_policy_sha256": producer._sha(
                producer._canonical(target)
            ),
        }
    )

    assert producer._validate_reconciliation_intent(
        inputs, revision=REVISION
    ) == (legacy, target)

    inputs["discord_reconciliation_intent"][
        "target_public_policy_sha256"
    ] = "f" * 64
    with pytest.raises(
        producer.HostPlanProducerError,
        match="host_plan_reconciliation_intent_mismatch",
    ):
        producer._validate_reconciliation_intent(inputs, revision=REVISION)


def test_connector_renderer_projects_the_complete_target_policy() -> None:
    inputs = _unit_inputs()
    target = producer._target_discord_policy()
    template = (
        Path("ops/muncho/systemd/discord-public-connector.json.in")
        .resolve()
        .read_bytes()
    )

    rendered = producer._render_connector_config(
        template,
        inputs=inputs,
        target_policy=target,
    )
    value = json.loads(rendered)

    assert all(value["discord"][name] == item for name, item in target.items())
    assert "opaque" not in rendered.decode("utf-8")


def test_writer_unit_invokes_release_bound_production_readiness() -> None:
    rendered = producer._render_writer_unit(
        revision=REVISION,
        inputs=_unit_inputs(),
    ).decode("utf-8")
    receipt = (
        "/var/lib/muncho/canonical-writer-phase-b/runtime-receipt.json"
    )

    assert (
        f"--production-release-revision {REVISION} "
        f"--production-phase-b-receipt {receipt}"
    ) in rendered
    assert f"AssertPathExists={receipt}\n" in rendered
    assert f"ReadOnlyPaths={receipt}\n" in rendered
    assert (
        "Requires=muncho-canonical-writer-phase-b-readiness.service\n"
        in rendered
    )

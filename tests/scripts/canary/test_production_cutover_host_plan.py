from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest

from scripts.canary import package_production_cutover_artifacts as package
from scripts.canary import production_cutover_host_plan as producer
from tests.scripts.canary.test_package_production_cutover_artifacts import (
    REVISION,
    _release,
    _unit_inputs,
)


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

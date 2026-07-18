from __future__ import annotations

import copy

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from scripts.canary import owner_gate_cloud_observation_author as author
from scripts.canary import owner_gate_foundation as foundation
from scripts.canary import owner_gate_preflight as preflight
from tests.scripts.canary.test_owner_gate_foundation import NOW
from tests.scripts.canary.test_owner_gate_preflight import (
    _attest,
    _cloud,
    _host,
    _plan,
)


def _pair_context():
    network_key = Ed25519PrivateKey.generate()
    cloud_key = Ed25519PrivateKey.generate()
    host_key = Ed25519PrivateKey.generate()
    plan = _plan(network_key, cloud_key, host_key)
    cloud = _cloud(plan, cloud_key, iam=False)
    host = _host(plan, host_key, iam=False)
    return plan, cloud_key, host_key, cloud, host


def test_bound_pair_constructor_is_not_a_caller_surface() -> None:
    with pytest.raises(
        author.OwnerGateCloudObservationAuthorError,
        match="owner_gate_bound_observation_pair_factory_required",
    ):
        author.BoundObservationPair()


def test_factory_pair_is_canonical_single_use_and_preflight_consumable() -> None:
    plan, cloud_key, host_key, cloud, host = _pair_context()
    pair = author.BoundObservationPair._create(
        cloud_observation=cloud,
        host_observation=host,
        plan_sha256=plan.sha256,
        phase="inert",
    )
    observed_cloud, observed_host = author.consume_bound_observation_pair(
        pair,
        plan=plan,
        phase="inert",
    )
    report = preflight.build_preflight_report(
        plan=plan,
        cloud_observation=observed_cloud,
        host_observation=observed_host,
        cloud_collector_public_key=cloud_key.public_key(),
        host_collector_public_key=host_key.public_key(),
        now_unix=NOW,
    )
    assert report["cloud_observation_sha256"] == cloud["report_sha256"]
    assert report["host_observation_sha256"] == host["report_sha256"]
    assert pair._cloud_raw == foundation.canonical_json_bytes(cloud)
    assert pair._host_raw == foundation.canonical_json_bytes(host)
    with pytest.raises(
        author.OwnerGateCloudObservationAuthorError,
        match="owner_gate_bound_observation_pair_invalid",
    ):
        author.consume_bound_observation_pair(
            pair,
            plan=plan,
            phase="inert",
        )


def test_factory_rejects_individually_signed_swapped_host() -> None:
    plan, _cloud_key, host_key, cloud, host = _pair_context()
    other_body = {
        key: copy.deepcopy(value)
        for key, value in host.items()
        if key not in {"report_sha256", "attestation"}
    }
    other_body["observation_binding_sha256"] = "b" * 64
    other_host = _attest(other_body, host_key)
    preflight._validate_host(
        other_host,
        spec=plan.spec,
        plan_sha256=plan.sha256,
        public_key=host_key.public_key(),
        expected_public_key_id=plan.spec.host_collector_public_key_id,
        mutation_binding_present=False,
    )
    with pytest.raises(
        author.OwnerGateCloudObservationAuthorError,
        match="owner_gate_bound_observation_pair_invalid",
    ):
        author.BoundObservationPair._create(
            cloud_observation=cloud,
            host_observation=other_host,
            plan_sha256=plan.sha256,
            phase="inert",
        )


def test_stale_pair_is_consumed_and_cannot_be_reused() -> None:
    plan, cloud_key, host_key, cloud, host = _pair_context()
    pair = author.BoundObservationPair._create(
        cloud_observation=cloud,
        host_observation=host,
        plan_sha256=plan.sha256,
        phase="inert",
    )
    observed_cloud, observed_host = author.consume_bound_observation_pair(
        pair,
        plan=plan,
        phase="inert",
    )
    with pytest.raises(
        preflight.OwnerGatePreflightError,
        match="owner_gate_preflight_stale",
    ):
        preflight.build_preflight_report(
            plan=plan,
            cloud_observation=observed_cloud,
            host_observation=observed_host,
            cloud_collector_public_key=cloud_key.public_key(),
            host_collector_public_key=host_key.public_key(),
            now_unix=NOW + foundation.PREFLIGHT_MAX_AGE_SECONDS + 1,
        )
    with pytest.raises(
        author.OwnerGateCloudObservationAuthorError,
        match="owner_gate_bound_observation_pair_invalid",
    ):
        author.consume_bound_observation_pair(
            pair,
            plan=plan,
            phase="inert",
        )


def test_legacy_cloud_only_return_shape_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, _cloud_key, _host_key, cloud, host = _pair_context()

    class Handoff:
        host_observation = host

    calls: list[str] = []

    def components(**_kwargs):
        calls.append("composite")
        return cloud, Handoff()

    monkeypatch.setattr(author, "_collect_and_author_components", components)
    arguments = {
        "plan": plan,
        "foundation_apply_chain": object(),
        "phase": "inert",
        "collected_at_unix": NOW,
        "gcloud_executable": object(),
        "gcloud_configuration": object(),
        "owner_identity": object(),
        "stage0_transport": object(),
        "kit_stream": object(),
        "bundle_stream": object(),
    }
    assert author.collect_and_author(**arguments) == cloud
    pair = author.collect_and_author_bound_pair(**arguments)
    assert type(pair) is author.BoundObservationPair
    assert calls == ["composite", "composite"]

from __future__ import annotations

import base64
import copy
import hashlib
import os

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from scripts.canary import owner_gate_cloud_observation_author as author
from scripts.canary import owner_gate_cloud_observation_signer as signer
from scripts.canary import owner_gate_foundation as foundation
from scripts.canary import owner_gate_preflight as preflight
from scripts.canary import storage_growth_trusted_collector as trusted
from tests.scripts.canary.test_owner_gate_cloud_observation_author import (
    NOW,
    _context,
    _identities,
    _raw,
    _verified_probe,
)
from tests.scripts.canary.test_owner_gate_foundation import IMAGE
from tests.scripts.canary.test_owner_gate_preflight import (
    _host as _preflight_host,
)


def _key_id(key: Ed25519PrivateKey) -> str:
    return hashlib.sha256(key.public_key().public_bytes_raw()).hexdigest()


def _package(
    *,
    cloud_key: Ed25519PrivateKey,
    host_key: Ed25519PrivateKey,
) -> dict[str, object]:
    return {
        "release_revision": "a" * 40,
        "source_tree_oid": "9" * 40,
        "package_sha256": "3" * 64,
        "package_inventory_sha256": "5" * 64,
        "pre_foundation_authority_sha256": "a" * 64,
        "foundation_apply_receipt_sha256": "b" * 64,
        "project_ancestry_evidence_sha256": "c" * 64,
        "project_ancestry_chain_sha256": "d" * 64,
        "resource_ancestor_chain": ["organizations/123456789012"],
        "interpreter_sha256": "6" * 64,
        "interpreter_image": {
            "project": "debian-cloud",
            "image_name": IMAGE.rsplit("/", 1)[-1],
            "image_numeric_id": "1234567890123456789",
            "image_self_link": ("https://www.googleapis.com/compute/v1/" + IMAGE),
            "python_version": "3.11.2",
            "interpreter_sha256": "6" * 64,
        },
        "collector_public_key_ids": {
            "network": "7" * 64,
            "cloud": _key_id(cloud_key),
            "host": _key_id(host_key),
        },
    }


def _terminal(package: dict[str, object]) -> dict[str, object]:
    value: dict[str, object] = {name: None for name in signer._TERMINAL_FIELDS}
    value.update({
        "schema": "muncho-owner-gate-inert-cloud-bundle-terminal.v1",
        "release_sha": package["release_revision"],
        "source_tree_oid": package["source_tree_oid"],
        "package_sha256": package["package_sha256"],
        "kit_release_id": "6" * 64,
        "trusted_runner_path": "/opt/muncho-stage0/releases/x/runner",
        "bundle_path": "/var/lib/muncho-stage0/incoming/x",
        "pre_foundation_authority_sha256": package["pre_foundation_authority_sha256"],
        "foundation_apply_receipt_sha256": package["foundation_apply_receipt_sha256"],
        "project_ancestry_evidence_sha256": package["project_ancestry_evidence_sha256"],
        "project_ancestry_chain_sha256": package["project_ancestry_chain_sha256"],
        "resource_ancestor_chain": package["resource_ancestor_chain"],
        "operation_order": [
            "transport_exact_stage0_and_bundle",
            "cloud-verify",
            "cloud-preflight",
            "cloud-install",
        ],
        "transport_receipt_sha256": "1" * 64,
        "cloud_verify_receipt_sha256": "2" * 64,
        "cloud_preflight_receipt_sha256": "3" * 64,
        "cloud_install_receipt_sha256": "4" * 64,
        "cloud_install_receipt_file_sha256": "5" * 64,
        "cloud_install_receipt": {"receipt_sha256": "4" * 64},
        "cloud_install_signature_framing_validated": True,
        "cloud_install_signature_cryptographically_verified": False,
        "inert_cloud_bundle_installed": True,
        "host_filesystem_materialization_performed": True,
        "current_release_selected": False,
        "systemd_units_enabled": [],
        "service_activation_performed": False,
        "activation_performed": False,
        "activation_seal_created": False,
        "iam_binding_created": False,
        "caddy_cutover_performed": False,
        "cloud_mutation_performed": False,
        "cloud_control_plane_mutation_performed": False,
    })
    unsigned = {
        name: item for name, item in value.items() if name != "terminal_receipt_sha256"
    }
    value["terminal_receipt_sha256"] = foundation.sha256_json(unsigned)
    return value


def _attest_host(body: dict[str, object], key: Ed25519PrivateKey) -> dict[str, object]:
    report = {**body, "report_sha256": foundation.sha256_json(body)}
    signature = key.sign(foundation.canonical_json_bytes(report))
    return {
        **report,
        "attestation": {
            "schema": "muncho-owner-gate-observation-attestation.v1",
            "public_key_id": _key_id(key),
            "signature_ed25519_b64url": base64
            .urlsafe_b64encode(signature)
            .rstrip(b"=")
            .decode("ascii"),
        },
    }


def _request_fixture(
    *, phase: str = "inert"
) -> tuple[
    dict[str, object],
    dict[str, object],
    Ed25519PrivateKey,
    Ed25519PrivateKey,
]:
    plan, ancestry, _unused_cloud_key = _context()
    cloud_key = Ed25519PrivateKey.generate()
    host_key = Ed25519PrivateKey.generate()
    package = _package(cloud_key=cloud_key, host_key=host_key)
    package["release_revision"] = plan.spec.release_revision
    package["package_inventory_sha256"] = plan.spec.package_inventory_sha256
    package["project_ancestry_evidence_sha256"] = ancestry.signed_evidence_sha256
    package["project_ancestry_chain_sha256"] = ancestry.value["stable_chain_sha256"]
    package["resource_ancestor_chain"] = [
        item["resource_name"] for item in ancestry.ordered_chain[1:]
    ]
    terminal = _terminal(package)
    probe = _verified_probe(phase)
    release = {
        "revision": package["release_revision"],
        "source_tree_oid": package["source_tree_oid"],
        "package_sha256": package["package_sha256"],
        "package_inventory_sha256": package["package_inventory_sha256"],
        "pre_foundation_authority_sha256": package["pre_foundation_authority_sha256"],
        "foundation_apply_receipt_sha256": package["foundation_apply_receipt_sha256"],
        "project_ancestry_evidence_sha256": package["project_ancestry_evidence_sha256"],
        "project_ancestry_chain_sha256": package["project_ancestry_chain_sha256"],
        "resource_ancestor_chain": package["resource_ancestor_chain"],
        "attached_sa_permission_probe_report_sha256": (
            probe.attached_sa_permission_probe_report_sha256
        ),
        "cloud_signer_provisioning_receipt_sha256": (
            probe.cloud_signer_provisioning_receipt_sha256
        ),
        "cloud_signer_readiness_sha256": probe.cloud_signer_readiness_sha256,
        "host_signer_provisioning_receipt_sha256": (
            probe.host_signer_provisioning_receipt_sha256
        ),
        "host_signer_readiness_sha256": probe.host_signer_readiness_sha256,
    }
    host_template = _preflight_host(plan, host_key, iam=phase == "post_iam")
    host_body = {
        name: copy.deepcopy(item)
        for name, item in host_template.items()
        if name not in {"report_sha256", "attestation"}
    }
    host_body["collected_at_unix"] = NOW
    host_body["completed_at_unix"] = NOW
    host_body["fresh_through_unix"] = NOW + preflight.HOST_OBSERVATION_FRESHNESS_SECONDS
    host_body["release"].update(release)
    host_body["release"].update({
        "root": ("/opt/muncho-owner-gate/releases/" + str(package["release_revision"])),
        "install_receipt_sha256": "7" * 64,
        "install_receipt_file_sha256": "8" * 64,
        "entrypoints": list(preflight.HOST_RELEASE_ENTRYPOINTS),
        "observation_dispatcher_schemas": list(
            preflight.HOST_OBSERVATION_DISPATCHER_SCHEMAS
        ),
        "python_executable": (
            "/opt/muncho-owner-gate/releases/"
            + str(package["release_revision"])
            + "/venv/bin/python"
        ),
        "python_executable_sha256": package["interpreter_sha256"],
    })
    host = _attest_host(host_body, host_key)
    identities = _identities(plan, ancestry)
    identities = author._FoundationIdentities(**{
        **identities.__dict__,
        "foundation_source_tree_oid": str(package["source_tree_oid"]),
    })
    unsigned = author._unsigned_from_raw(
        plan=plan,
        ancestry_evidence=ancestry,
        phase=phase,
        raw=_raw(plan, ancestry, phase=phase),
        collected_at_unix=NOW,
        package_sha256=str(package["package_sha256"]),
        foundation_identities=identities,
        verified_probe=probe,
    )
    unsigned["release_binding"] = {
        "phase": phase,
        "release_revision": package["release_revision"],
        "source_tree_oid": package["source_tree_oid"],
        "package_sha256": package["package_sha256"],
        "package_inventory_sha256": package["package_inventory_sha256"],
        "pre_foundation_authority_sha256": package["pre_foundation_authority_sha256"],
        "foundation_apply_receipt_sha256": package["foundation_apply_receipt_sha256"],
        "project_ancestry_evidence_sha256": package["project_ancestry_evidence_sha256"],
        "project_ancestry_chain_sha256": package["project_ancestry_chain_sha256"],
        "resource_ancestor_chain": package["resource_ancestor_chain"],
        "terminal_receipt_sha256": terminal["terminal_receipt_sha256"],
        "host_observation_report_sha256": host["report_sha256"],
        "host_observation_binding_sha256": host["observation_binding_sha256"],
        "attached_sa_permission_probe_report_sha256": release[
            "attached_sa_permission_probe_report_sha256"
        ],
        "cloud_signer_provisioning_receipt_sha256": release[
            "cloud_signer_provisioning_receipt_sha256"
        ],
        "cloud_signer_readiness_sha256": release["cloud_signer_readiness_sha256"],
        "host_signer_provisioning_receipt_sha256": release[
            "host_signer_provisioning_receipt_sha256"
        ],
        "host_signer_readiness_sha256": release["host_signer_readiness_sha256"],
        "effective_permission_probe_sha256": foundation.sha256_json(
            probe.permission_probe
        ),
    }
    request_body = {
        "schema": signer.REQUEST_SCHEMA,
        "phase": phase,
        "release_revision": package["release_revision"],
        "unsigned_observation": unsigned,
        "terminal_receipt": terminal,
        "host_observation": host,
    }
    request = {
        **request_body,
        "request_sha256": foundation.sha256_json(request_body),
    }
    return request, package, cloud_key, host_key


def test_request_binds_full_unsigned_host_probe_and_terminal() -> None:
    request, package, _cloud_key, host_key = _request_fixture()
    checked, unsigned = signer._validate_request(
        request,
        revision=str(package["release_revision"]),
        package=package,
        host_public_key=host_key.public_key(),
        now_unix=NOW,
    )
    assert checked["request_sha256"] == request["request_sha256"]
    assert (
        unsigned["release_binding"]["host_observation_report_sha256"]
        == (request["host_observation"]["report_sha256"])
    )


@pytest.mark.parametrize(
    "mutation",
    ["host_signature", "sidecar_report", "effective_probe", "terminal", "release"],
)
def test_hostile_binding_substitution_is_rejected(mutation: str) -> None:
    request, package, _cloud_key, host_key = _request_fixture()
    changed = copy.deepcopy(request)
    if mutation == "host_signature":
        changed["host_observation"]["effective_permission_probe"]["disk"][
            "granted_permissions"
        ].append("compute.disks.update")
    elif mutation == "sidecar_report":
        changed["unsigned_observation"]["release_binding"][
            "attached_sa_permission_probe_report_sha256"
        ] = "0" * 64
    elif mutation == "effective_probe":
        changed["unsigned_observation"]["service_account"][
            "effective_permission_probe"
        ]["disk"]["granted_permissions"].append("compute.disks.update")
    elif mutation == "terminal":
        changed["terminal_receipt"]["package_sha256"] = "0" * 64
    else:
        changed["release_revision"] = "0" * 40
    body = {name: item for name, item in changed.items() if name != "request_sha256"}
    changed["request_sha256"] = foundation.sha256_json(body)
    with pytest.raises(signer.OwnerGateCloudObservationSignerError):
        signer._validate_request(
            changed,
            revision=str(package["release_revision"]),
            package=package,
            host_public_key=host_key.public_key(),
            now_unix=NOW,
        )


def test_request_freshness_is_not_caller_extendable() -> None:
    request, package, _cloud_key, host_key = _request_fixture()
    with pytest.raises(
        signer.OwnerGateCloudObservationSignerError,
        match="request_stale",
    ):
        signer._validate_request(
            request,
            revision=str(package["release_revision"]),
            package=package,
            host_public_key=host_key.public_key(),
            now_unix=NOW + signer.FRESHNESS_SECONDS + 1,
        )


@pytest.mark.parametrize("mutation", ["unit_identity", "stale_host"])
def test_validly_attested_host_still_requires_full_fresh_schema(
    mutation: str,
) -> None:
    request, package, _cloud_key, host_key = _request_fixture()
    changed = copy.deepcopy(request)
    host_body = {
        name: item
        for name, item in changed["host_observation"].items()
        if name not in {"report_sha256", "attestation"}
    }
    if mutation == "unit_identity":
        host_body["units"]["web"]["User"] = "root"
    else:
        host_body["collected_at_unix"] = NOW - signer.FRESHNESS_SECONDS - 1
    changed["host_observation"] = _attest_host(host_body, host_key)
    changed["unsigned_observation"]["release_binding"][
        "host_observation_report_sha256"
    ] = changed["host_observation"]["report_sha256"]
    request_body = {
        name: item for name, item in changed.items() if name != "request_sha256"
    }
    changed["request_sha256"] = foundation.sha256_json(request_body)

    with pytest.raises(
        signer.OwnerGateCloudObservationSignerError,
        match="host_invalid",
    ):
        signer._validate_request(
            changed,
            revision=str(package["release_revision"]),
            package=package,
            host_public_key=host_key.public_key(),
            now_unix=NOW,
        )


def test_executor_signer_replays_exactly_and_never_outputs_private_key(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, package, cloud_key, host_key = _request_fixture()
    key_path = tmp_path / "cloud.key"
    replay = tmp_path / "replay"
    key_path.write_bytes(cloud_key.private_bytes_raw())
    key_path.chmod(0o400)
    replay.mkdir(mode=0o700)
    uid = os.getuid()
    gid = os.getgid()
    os.chown(key_path, -1, gid)
    os.chown(replay, -1, gid)
    config = {
        "schema": trusted.CLOUD_CONFIG_SCHEMA,
        "role": "cloud",
        "private_key_path": str(key_path),
        "private_key_uid": uid,
        "private_key_gid": gid,
        "private_key_mode": "0400",
        "public_key_id": _key_id(cloud_key),
        "replay_directory": str(replay),
        "replay_directory_uid": uid,
        "replay_directory_gid": gid,
        "replay_directory_mode": "0700",
    }
    monkeypatch.setattr(signer.os, "geteuid", lambda: signer.EXECUTOR_UID)
    monkeypatch.setattr(signer, "_load_package", lambda _revision: package)
    monkeypatch.setattr(
        signer,
        "_load_public_key",
        lambda _path, *, expected_id: host_key.public_key(),
    )
    monkeypatch.setattr(trusted, "load_cloud_attestor_config", lambda: config)

    first = signer.sign_request(
        request,
        release_revision=str(package["release_revision"]),
        now_unix=NOW,
    )
    second = signer.sign_request(
        request,
        release_revision=str(package["release_revision"]),
        now_unix=NOW,
    )

    assert first == second
    encoded = foundation.canonical_json_bytes(first)
    assert cloud_key.private_bytes_raw() not in encoded
    assert (
        base64.urlsafe_b64encode(cloud_key.private_bytes_raw()).rstrip(b"=")
        not in encoded
    )
    assert len(list(replay.glob("owner-gate-cloud-*.json"))) == 1


def test_signer_rejects_wrong_uid_before_loading_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, package, _cloud_key, _host_key = _request_fixture()
    monkeypatch.setattr(signer.os, "geteuid", lambda: signer.EXECUTOR_UID + 1)
    monkeypatch.setattr(
        signer,
        "_load_package",
        lambda _revision: pytest.fail("package must not be read"),
    )
    with pytest.raises(
        signer.OwnerGateCloudObservationSignerError,
        match="runtime_invalid",
    ):
        signer.sign_request(
            request,
            release_revision=str(package["release_revision"]),
            now_unix=NOW,
        )

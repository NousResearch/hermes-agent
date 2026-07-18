from __future__ import annotations

import base64
import hashlib
import multiprocessing
import os
import select
import shutil
import signal
import stat
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from scripts.canary import owner_gate_activation_seal as activation
from scripts.canary import owner_gate_foundation as foundation
from scripts.canary import owner_gate_owner_reauth as owner_reauth
from scripts.canary import owner_gate_package as package
from scripts.canary import owner_gate_preflight as preflight
from scripts.canary import owner_gate_trust as trust
from scripts.canary import passkey_v2_protocol as protocol
from scripts.canary import passkey_v2_service as service
from tests.scripts.canary.test_owner_gate_foundation import (
    IMAGE,
    NOW,
    REVISION,
    _signed_network_evidence,
)
from tests.scripts.canary.test_owner_gate_package import (
    _source,
    _trusted_spec,
    _wheelhouse,
    _write,
)
from tests.scripts.canary.test_owner_gate_preflight import (
    _attest,
    _cloud,
    _host,
)


def _chmod_directories(root: Path, mode: int) -> None:
    for path in sorted(
        (item for item in root.rglob("*") if item.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        path.chmod(mode)
    root.chmod(mode)


def _write_exact(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(foundation.canonical_json_bytes(value))
    path.chmod(0o444)


def _host_for_release(
    plan: foundation.OwnerGateFoundationPlan,
    key: Ed25519PrivateKey,
    *,
    iam: bool,
    package_manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    original = _host(plan, key, iam=iam)
    body = {
        name: value
        for name, value in original.items()
        if name not in {"report_sha256", "attestation"}
    }
    release = dict(body["release"])
    release.update({
        "revision": plan.spec.release_revision,
        "root": str(plan.spec.release_root),
        "package_sha256": package_manifest["package_sha256"],
        "package_inventory_sha256": package_manifest[
            "package_inventory_sha256"
        ],
        "python_executable": f"{plan.spec.release_root}/venv/bin/python",
        "python_executable_sha256": package_manifest["interpreter_sha256"],
    })
    body["release"] = release
    return _attest(body, key)


def _activation_owner_reauth_receipt(
    private_key: Ed25519PrivateKey,
    *,
    project_number: str,
    issued_at_unix: int = NOW,
    expires_at_unix: int | None = None,
) -> Mapping[str, Any]:
    if expires_at_unix is None:
        expires_at_unix = (
            issued_at_unix + owner_reauth.MAX_RECEIPT_TTL_SECONDS
        )
    key_id = hashlib.sha256(
        private_key.public_key().public_bytes_raw()
    ).hexdigest()
    body = {
        "schema": owner_reauth.RECEIPT_SCHEMA,
        "purpose": owner_reauth.RECEIPT_PURPOSE,
        "trusted_runtime_identity": {
            "release_revision": REVISION,
            "sealed_runtime_identity_sha256": "1" * 64,
            "command_prefix_sha256": "2" * 64,
            "python_executable_sha256": "3" * 64,
            "gcloud_module_sha256": "4" * 64,
            "sdk_root": "/opt/google-cloud-sdk",
            "sdk_python_config_identity_sha256": "5" * 64,
            "closed_environment_sha256": "6" * 64,
            "configuration": owner_reauth.GCLOUD_CONFIGURATION,
            "account": owner_reauth.OWNER_ACCOUNT,
            "project": owner_reauth.PROJECT,
            "zone": owner_reauth.ZONE,
        },
        "interactive_reauthentication": {
            "method": "gcloud_auth_login_force_interactive",
            "started_at_unix": NOW - 2,
            "completed_at_unix": NOW - 1,
            "command_sha256": "7" * 64,
            "interactive_tty_verified": True,
            "access_token_requested": False,
            "credential_material_captured": False,
        },
        "authenticated_probe": {
            "command_sha256": "8" * 64,
            "output_sha256": "9" * 64,
            "project_id": foundation.PROJECT,
            "project_number": project_number,
        },
        "issued_at_unix": issued_at_unix,
        "expires_at_unix": expires_at_unix,
        "signer_key_id": key_id,
    }
    return owner_reauth._sign_owner_reauth_receipt(
        body,
        private_key=private_key,
    )


def _release_trust_manifest(
    *,
    inventory: Mapping[str, Any],
    direct_raw: bytes,
    collector_ids: Mapping[str, str],
    private_key: Ed25519PrivateKey,
) -> Mapping[str, Any]:
    public_raw = private_key.public_key().public_bytes_raw()
    unsigned = {
        "schema": trust.TRUST_SCHEMA,
        "approved_for_offline_install": True,
        "fork_repository": trust.FORK_REPOSITORY,
        "release_revision": inventory["release_revision"],
        "source_tree_oid": inventory["source_tree_oid"],
        "package_inventory_sha256": foundation.sha256_json(inventory),
        "boot_image_self_link": IMAGE,
        "collector_public_key_ids": dict(collector_ids),
        "credential_migration_envelope_sha256": "7" * 64,
        "direct_iam_identity_authority_sha256": hashlib.sha256(
            direct_raw
        ).hexdigest(),
        "pre_foundation_authority_sha256": inventory[
            "pre_foundation_authority_sha256"
        ],
        "foundation_apply_receipt_sha256": inventory[
            "foundation_apply_receipt_sha256"
        ],
        "project_ancestry_evidence_sha256": "c" * 64,
        "project_ancestry_chain_sha256": "d" * 64,
        "resource_ancestor_chain": inventory["resource_ancestor_chain"],
        "interpreter_image": {
            "project": "debian-cloud",
            "image_name": IMAGE.rsplit("/", 1)[-1],
            "image_numeric_id": "1234567890123456789",
            "image_self_link": (
                "https://www.googleapis.com/compute/v1/" + IMAGE
            ),
            "python_version": package.PYTHON_VERSION,
            "interpreter_sha256": inventory["interpreter_sha256"],
        },
        "release_attestation": {
            "purpose": trust.ATTESTATION_PURPOSE,
            "attested_at_unix": NOW - 10,
        },
        "signer_key_id": hashlib.sha256(public_raw).hexdigest(),
    }
    signature = private_key.sign(foundation.canonical_json_bytes(unsigned))
    return {
        **unsigned,
        "signature_ed25519_b64url": base64.urlsafe_b64encode(signature)
        .rstrip(b"=")
        .decode("ascii"),
    }


def _environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Mapping[str, Any]:
    install_root = tmp_path / "opt/muncho-owner-gate"
    release_base = install_root / "releases"
    release = release_base / REVISION
    state_root = tmp_path / "var/lib/muncho-owner-gate"
    evidence_base = state_root / "activation-evidence"
    evidence_root = evidence_base / REVISION
    receipt_base = state_root / "activation-receipts"
    etc_root = tmp_path / "etc/muncho-owner-gate"
    run_root = tmp_path / "run/muncho-owner-gate"
    for path, mode in (
        (install_root, 0o755),
        (release_base, 0o755),
        (state_root, 0o711),
        (evidence_base, 0o700),
        (evidence_root, 0o700),
        (receipt_base, 0o700),
        (etc_root, 0o755),
        (run_root, 0o755),
    ):
        path.mkdir(parents=True, exist_ok=True)
        path.chmod(mode)

    uid = install_root.stat().st_uid
    gid = install_root.stat().st_gid

    monkeypatch.setattr(foundation, "RELEASE_BASE", release_base)
    monkeypatch.setattr(activation, "RELEASE_BASE", release_base)
    monkeypatch.setattr(activation, "ROOT_UID", uid)
    monkeypatch.setattr(activation, "ROOT_GID", gid)
    monkeypatch.setattr(activation, "EXECUTOR_GID", gid)
    monkeypatch.setattr(activation, "ACTIVATION_EVIDENCE_BASE", evidence_base)
    monkeypatch.setattr(activation, "ACTIVATION_RECEIPT_BASE", receipt_base)
    monkeypatch.setattr(
        activation,
        "ACTIVATION_SEAL_PATH",
        etc_root / "storage-executor-enabled",
    )
    monkeypatch.setattr(
        activation,
        "ACTIVATION_LOCK_PATH",
        run_root / "storage-executor-activation.lock",
    )

    source = _source(tmp_path, monkeypatch)
    _write(
        source / "scripts/canary/owner_gate_activation_seal.py",
        b"VALUE = 'activation-author'\n",
    )
    monkeypatch.setattr(
        package,
        "ROOT_RUNTIME_FILES",
        (
            "scripts/canary/owner_gate_activation_seal.py",
            "scripts/canary/passkey_v2_service.py",
        ),
    )
    wheel_root, wheel_manifest = _wheelhouse(tmp_path)
    lock_unsigned = {
        "bootstrap_pip": {
            **wheel_manifest["bootstrap_pip"],
            "active_dependencies": [],
        },
        "schema": package.RUNTIME_LOCK_SCHEMA,
        "python_version": package.PYTHON_VERSION,
        "platform": package.WHEELHOUSE_PLATFORM,
        "network_required": False,
        "source_build_allowed": False,
        "complete_transitive_closure": True,
        "wheels": sorted(
            (
                {
                    **item,
                    "active_dependencies": sorted(
                        package.EXPECTED_DIRECT_DEPENDENCIES.get(
                            item["project"],
                            set(),
                        )
                    ),
                }
                for item in wheel_manifest["wheels"]
            ),
            key=lambda item: item["project"],
        ),
    }
    runtime_lock = {
        **lock_unsigned,
        "lock_sha256": foundation.sha256_json(lock_unsigned),
    }
    runtime_lock_path = source / package.RUNTIME_LOCK_RELATIVE
    runtime_lock_path.chmod(0o644)
    runtime_lock_path.write_bytes(
        foundation.canonical_json_bytes(runtime_lock) + b"\n"
    )
    runtime_lock_path.chmod(0o444)
    wheel_unsigned = {
        key: value
        for key, value in wheel_manifest.items()
        if key != "manifest_sha256"
    }
    wheel_unsigned["runtime_lock_sha256"] = package.runtime_lock_file_sha256(
        runtime_lock
    )
    wheel_manifest = {
        **wheel_unsigned,
        "manifest_sha256": foundation.sha256_json(wheel_unsigned),
    }
    spec = _trusted_spec(
        source=source,
        wheel_root=wheel_root,
        wheel_manifest=wheel_manifest,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    inventory = package.build_inventory(spec)
    direct_path = spec.direct_iam_identity_authority_path
    assert direct_path is not None
    direct_raw = direct_path.read_bytes()
    direct_authority = protocol.decode_canonical_json(direct_raw)

    collector_private = {
        name: Ed25519PrivateKey.generate()
        for name in ("network", "cloud", "host")
    }
    collector_ids = {
        name: hashlib.sha256(key.public_key().public_bytes_raw()).hexdigest()
        for name, key in collector_private.items()
    }
    release_key = Ed25519PrivateKey.generate()
    trust_manifest = _release_trust_manifest(
        inventory=inventory,
        direct_raw=direct_raw,
        collector_ids=collector_ids,
        private_key=release_key,
    )
    trust_raw = foundation.canonical_json_bytes(trust_manifest)
    public_raw = release_key.public_key().public_bytes_raw()
    monkeypatch.setattr(
        trust,
        "PINNED_RELEASE_TRUST_PUBLIC_KEY_SHA256",
        hashlib.sha256(public_raw).hexdigest(),
    )
    unsigned_manifest = {
        **inventory,
        "package_inventory_sha256": foundation.sha256_json(inventory),
        "trust_manifest_sha256": hashlib.sha256(trust_raw).hexdigest(),
        "trust_public_key_sha256": hashlib.sha256(public_raw).hexdigest(),
        "interpreter_image": trust_manifest["interpreter_image"],
        "release_supply_chain_attestation": trust_manifest[
            "release_attestation"
        ],
        "collector_public_key_ids": dict(collector_ids),
        "credential_migration_envelope_sha256": trust_manifest[
            "credential_migration_envelope_sha256"
        ],
        "project_ancestry_evidence_sha256": trust_manifest[
            "project_ancestry_evidence_sha256"
        ],
        "project_ancestry_chain_sha256": trust_manifest[
            "project_ancestry_chain_sha256"
        ],
        "caller_self_hash_is_authority": False,
    }
    package_manifest = {
        **unsigned_manifest,
        "package_sha256": foundation.sha256_json(unsigned_manifest),
    }

    release.mkdir(mode=0o700)
    for item in inventory["payloads"]:
        source_path = source / item["source_relative"]
        destination = release / item["release_relative"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, destination)
        destination.chmod(int(item["mode"], 8))
    _write_exact(release / "package-manifest.json", package_manifest)
    _write_exact(release / "trust/release-trust.json", trust_manifest)
    trust_key_path = release / "trust/release-trust-signing.pub"
    trust_key_path.write_bytes(public_raw)
    trust_key_path.chmod(0o444)
    direct_target = release / "trust/direct-iam-identity-authority.json"
    direct_target.write_bytes(direct_raw)
    direct_target.chmod(0o444)
    for name, key in collector_private.items():
        path = release / "trust" / f"{name}-observation-attestation.pub"
        path.write_bytes(key.public_key().public_bytes_raw())
        path.chmod(0o444)
    _chmod_directories(release, 0o555)

    network_mapping = _signed_network_evidence(
        collector_private["network"],
        collected_at=NOW - 2,
    )
    _write_exact(evidence_root / activation.NETWORK_EVIDENCE_NAME, network_mapping)
    network = foundation.ProductionNetworkEvidence.from_mapping(
        network_mapping,
        public_key=collector_private["network"].public_key(),
        expected_public_key_id=collector_ids["network"],
        now_unix=NOW - 2,
    )
    plan = foundation.build_plan(
        spec=foundation.OwnerGateSpec(
            release_revision=REVISION,
            boot_image_self_link=IMAGE,
            package_inventory_sha256=package_manifest[
                "package_inventory_sha256"
            ],
            interpreter_sha256=package_manifest["interpreter_sha256"],
            network_collector_public_key_id=collector_ids["network"],
            organization_id="123456789012",
            ancestry_evidence_sha256="c" * 64,
            cloud_collector_public_key_id=collector_ids["cloud"],
            host_collector_public_key_id=collector_ids["host"],
        ),
        network_evidence=network,
        network_collector_public_key=collector_private["network"].public_key(),
        now_unix=NOW - 2,
    )
    inert_cloud = _cloud(plan, collector_private["cloud"], iam=False)
    post_cloud = _cloud(plan, collector_private["cloud"], iam=True)
    inert_host = _host_for_release(
        plan,
        collector_private["host"],
        iam=False,
        package_manifest=package_manifest,
    )
    post_host = _host_for_release(
        plan,
        collector_private["host"],
        iam=True,
        package_manifest=package_manifest,
    )
    inert = preflight.build_preflight_report(
        plan=plan,
        cloud_observation=inert_cloud,
        host_observation=inert_host,
        cloud_collector_public_key=collector_private["cloud"].public_key(),
        host_collector_public_key=collector_private["host"].public_key(),
        now_unix=NOW,
    )
    post = preflight.build_post_iam_preflight_report(
        plan=plan,
        cloud_observation=post_cloud,
        host_observation=post_host,
        cloud_collector_public_key=collector_private["cloud"].public_key(),
        host_collector_public_key=collector_private["host"].public_key(),
        now_unix=NOW,
    )
    for name, value in (
        (activation.INERT_CLOUD_OBSERVATION_NAME, inert_cloud),
        (activation.INERT_HOST_OBSERVATION_NAME, inert_host),
        (activation.INERT_PREFLIGHT_NAME, inert),
        (activation.POST_IAM_CLOUD_OBSERVATION_NAME, post_cloud),
        (activation.POST_IAM_HOST_OBSERVATION_NAME, post_host),
        (activation.POST_IAM_PREFLIGHT_NAME, post),
        (
            activation.ACTIVATION_OWNER_REAUTH_NAME,
            _activation_owner_reauth_receipt(
                release_key,
                project_number=str(direct_authority["project_number"]),
            ),
        ),
    ):
        _write_exact(evidence_root / name, value)
    evidence_root.chmod(0o500)
    return {
        "release": release,
        "evidence_root": evidence_root,
        "receipt": receipt_base / f"{REVISION}.json",
        "seal": etc_root / "storage-executor-enabled",
        "lock": run_root / "storage-executor-activation.lock",
        "manifest": package_manifest,
        "post": post,
        "release_key": release_key,
        "project_number": str(direct_authority["project_number"]),
    }


def _install(environment: Mapping[str, Any], *, now_unix: int = NOW):
    return activation.install_activation_seal(
        release=environment["release"],
        evidence_root=environment["evidence_root"],
        receipt_path=environment["receipt"],
        seal_path=environment["seal"],
        lock_path=environment["lock"],
        now_unix=now_unix,
    )


def test_exact_evidence_authors_installs_and_replays_service_seal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path, monkeypatch)
    assert not environment["seal"].exists()
    first = _install(environment)
    assert first["disposition"] == "installed"
    state = environment["seal"].stat()
    assert stat.S_IMODE(state.st_mode) == 0o440
    assert state.st_uid == environment["release"].stat().st_uid
    assert state.st_gid == environment["release"].stat().st_gid
    assert state.st_nlink == 1
    seal_raw = environment["seal"].read_bytes()
    assert len(seal_raw) <= service.MAX_SEAL_BYTES
    seal = protocol.decode_canonical_json(seal_raw)
    assert service.validate_activation_seal(
        seal,
        expected_release_revision=REVISION,
        now_unix=NOW,
    ) == seal
    assert seal["package_sha256"] == environment["manifest"]["package_sha256"]
    assert seal["iam_repreflight_receipt_sha256"] == environment["post"][
        "report_sha256"
    ]
    assert seal["authorization_record_complete"] is True
    assert seal["activation_installed"] is True
    assert seal["cloud_mutation_performed"] is False
    assert set(seal["evidence_file_sha256"]) == set(
        activation.EVIDENCE_NAMES
    )
    lineage = seal["verified_release_lineage"]
    assert lineage["foundation_apply_receipt_sha256"] == environment[
        "manifest"
    ]["foundation_apply_receipt_sha256"]
    assert lineage["source_tree_oid"] == environment["manifest"][
        "source_tree_oid"
    ]
    assert lineage["post_iam_preflight_receipt_sha256"] == environment[
        "post"
    ]["report_sha256"]
    assert lineage["activation_owner_reauthentication_receipt_sha256"]
    stored_receipt = protocol.decode_canonical_json(
        environment["receipt"].read_bytes()
    )
    assert stored_receipt["activation_record_is_canonical_authorization"] is True
    assert stored_receipt["verified_release_lineage"] == lineage
    assert stored_receipt["evidence_file_sha256"] == seal[
        "evidence_file_sha256"
    ]
    replay = _install(environment, now_unix=NOW + 10_000)
    assert replay["disposition"] == "exact_replay"
    assert replay["activation_seal_sha256"] == first["activation_seal_sha256"]


def test_stale_or_tampered_evidence_never_creates_seal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path, monkeypatch)
    with pytest.raises(
        activation.OwnerGateActivationSealError,
        match="owner_gate_activation_evidence_stale",
    ):
        _install(
            environment,
            now_unix=NOW + foundation.PREFLIGHT_MAX_AGE_SECONDS + 1,
        )
    assert not environment["seal"].exists()
    tampered = environment["evidence_root"] / activation.POST_IAM_PREFLIGHT_NAME
    environment["evidence_root"].chmod(0o700)
    tampered.chmod(0o644)
    tampered.write_bytes(b"{}")
    tampered.chmod(0o444)
    environment["evidence_root"].chmod(0o500)
    with pytest.raises(activation.OwnerGateActivationSealError):
        _install(environment)
    assert not environment["seal"].exists()


def test_seal_only_recovery_is_complete_truth_but_has_no_freshness_waiver(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path, monkeypatch)
    _install(environment)
    seal_raw = environment["seal"].read_bytes()
    environment["receipt"].unlink()

    with pytest.raises(
        activation.OwnerGateActivationSealError,
        match="owner_gate_activation_evidence_stale",
    ):
        _install(
            environment,
            now_unix=NOW + foundation.PREFLIGHT_MAX_AGE_SECONDS + 1,
        )

    assert environment["seal"].read_bytes() == seal_raw
    assert not environment["receipt"].exists()
    seal = protocol.decode_canonical_json(seal_raw)
    assert service.validate_activation_seal(
        seal,
        expected_release_revision=REVISION,
        now_unix=NOW + foundation.PREFLIGHT_MAX_AGE_SECONDS + 1,
    ) == seal
    assert seal["authorization_record_complete"] is True
    assert seal["activation_installed"] is True
    assert seal["verified_release_lineage"]["release_revision"] == REVISION


def test_receipt_publication_failure_leaves_complete_canonical_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path, monkeypatch)
    publish = activation._publish_no_replace

    def fail_audit_mirror(path: Path, **kwargs: Any) -> bool:
        if path == environment["receipt"]:
            raise activation.OwnerGateActivationSealError(
                "injected_audit_mirror_failure"
            )
        return publish(path, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(activation, "_publish_no_replace", fail_audit_mirror)
        with pytest.raises(
            activation.OwnerGateActivationSealError,
            match="injected_audit_mirror_failure",
        ):
            _install(environment)

    assert environment["seal"].exists()
    assert not environment["receipt"].exists()
    seal = protocol.decode_canonical_json(environment["seal"].read_bytes())
    assert service.validate_activation_seal(
        seal,
        expected_release_revision=REVISION,
        now_unix=NOW,
    ) == seal
    assert seal["authorization_record_complete"] is True
    assert seal["activation_installed"] is True
    assert seal["cloud_mutation_performed"] is False
    assert set(seal["evidence_file_sha256"]) == set(
        activation.EVIDENCE_NAMES
    )

    recovered = _install(environment)
    assert recovered["disposition"] == "exact_replay"
    assert environment["receipt"].exists()


def test_expired_activation_owner_reauth_never_creates_seal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path, monkeypatch)
    evidence = (
        environment["evidence_root"]
        / activation.ACTIVATION_OWNER_REAUTH_NAME
    )
    environment["evidence_root"].chmod(0o700)
    evidence.chmod(0o644)
    evidence.write_bytes(
        foundation.canonical_json_bytes(
            _activation_owner_reauth_receipt(
                environment["release_key"],
                project_number=environment["project_number"],
                expires_at_unix=NOW + 1,
            )
        )
    )
    evidence.chmod(0o444)
    environment["evidence_root"].chmod(0o500)
    with pytest.raises(
        activation.OwnerGateActivationSealError,
        match="owner_gate_activation_owner_reauth_invalid",
    ):
        _install(environment, now_unix=NOW + 2)
    assert not environment["seal"].exists()


@pytest.mark.parametrize("mismatch", ("wrong_project", "issued_before_post"))
def test_critical_owner_reauthentication_mismatch_never_activates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mismatch: str,
) -> None:
    environment = _environment(tmp_path, monkeypatch)
    evidence = (
        environment["evidence_root"]
        / activation.ACTIVATION_OWNER_REAUTH_NAME
    )
    environment["evidence_root"].chmod(0o700)
    evidence.chmod(0o644)
    evidence.write_bytes(
        foundation.canonical_json_bytes(
            _activation_owner_reauth_receipt(
                environment["release_key"],
                project_number=(
                    "999999999999"
                    if mismatch == "wrong_project"
                    else environment["project_number"]
                ),
                issued_at_unix=(
                    NOW - 1 if mismatch == "issued_before_post" else NOW
                ),
            )
        )
    )
    evidence.chmod(0o444)
    environment["evidence_root"].chmod(0o500)

    with pytest.raises(
        activation.OwnerGateActivationSealError,
        match="owner_gate_activation_owner_reauth_invalid",
    ):
        _install(environment)
    assert not environment["seal"].exists()


def test_release_payload_hash_mismatch_never_activates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path, monkeypatch)
    payload = (
        environment["release"]
        / "scripts/canary/owner_gate_activation_seal.py"
    )
    payload.chmod(0o644)
    payload.write_bytes(payload.read_bytes() + b"# tampered\n")
    payload.chmod(0o444)

    with pytest.raises(
        activation.OwnerGateActivationSealError,
        match="owner_gate_activation_package_payload_invalid",
    ):
        _install(environment)
    assert not environment["seal"].exists()


def test_no_replace_rejects_existing_different_seal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path, monkeypatch)
    environment["seal"].write_bytes(b"{}")
    environment["seal"].chmod(0o440)
    with pytest.raises(
        activation.OwnerGateActivationSealError,
        match="owner_gate_activation_publication_drift",
    ):
        _install(environment)
    assert environment["seal"].read_bytes() == b"{}"


def test_concurrent_replays_and_interrupted_hardlink_cleanup_are_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path, monkeypatch)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _index: _install(environment), range(8)))
    assert sum(item["disposition"] == "installed" for item in results) == 1
    assert len({item["activation_seal_sha256"] for item in results}) == 1
    stage = environment["seal"].with_name(
        f".{environment['seal'].name}.staged"
    )
    os.link(environment["seal"], stage)
    assert environment["seal"].stat().st_nlink == 2
    replay = _install(environment, now_unix=NOW + 10_000)
    assert replay["disposition"] == "exact_replay"
    assert not stage.exists()
    assert environment["seal"].stat().st_nlink == 1


def test_sigkill_after_fsynced_record_never_leaves_authority_without_truth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("requires fork to inherit the fixed-path test boundary")
    environment = _environment(tmp_path, monkeypatch)
    context = multiprocessing.get_context("fork")
    ready_read, ready_write = os.pipe()
    publish = activation._publish_no_replace

    def pause_after_record(path: Path, **kwargs: Any) -> bool:
        installed = publish(path, **kwargs)
        if path == environment["seal"]:
            os.write(ready_write, b"1")
            signal.pause()
        return installed

    def child_install() -> None:
        _install(environment)

    with monkeypatch.context() as scoped:
        scoped.setattr(activation, "_publish_no_replace", pause_after_record)
        process = context.Process(target=child_install)
        process.start()
        os.close(ready_write)
        try:
            readable, _, _ = select.select([ready_read], [], [], 20)
            assert readable, "child did not publish the canonical record"
            assert os.read(ready_read, 1) == b"1"
        finally:
            if process.is_alive():
                os.kill(process.pid, signal.SIGKILL)
            process.join(timeout=20)
        assert not process.is_alive()
        assert process.exitcode == -signal.SIGKILL
    os.close(ready_read)

    assert environment["seal"].exists()
    assert not environment["receipt"].exists()
    seal = protocol.decode_canonical_json(environment["seal"].read_bytes())
    assert service.validate_activation_seal(
        seal,
        expected_release_revision=REVISION,
        now_unix=NOW,
    ) == seal
    assert seal["authorization_record_complete"] is True
    assert seal["verified_release_lineage"]["lineage_sha256"]
    assert set(seal["evidence_file_sha256"]) == set(
        activation.EVIDENCE_NAMES
    )

    recovered = _install(environment)
    assert recovered["disposition"] == "exact_replay"
    assert environment["receipt"].exists()


def test_multiprocess_first_install_has_one_exact_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("requires fork to inherit the fixed-path test boundary")
    environment = _environment(tmp_path, monkeypatch)
    context = multiprocessing.get_context("fork")
    start = context.Event()
    results = context.Queue()

    def child_install() -> None:
        start.wait(timeout=20)
        try:
            results.put(("ok", dict(_install(environment))))
        except Exception as exc:  # pragma: no cover - diagnostic transport
            results.put(("error", type(exc).__name__, str(exc)))

    processes = [context.Process(target=child_install) for _ in range(6)]
    for process in processes:
        process.start()
    start.set()
    observed = [results.get(timeout=30) for _ in processes]
    for process in processes:
        process.join(timeout=30)
        assert not process.is_alive()
        assert process.exitcode == 0

    assert all(item[0] == "ok" for item in observed), observed
    responses = [item[1] for item in observed]
    assert sum(item["disposition"] == "installed" for item in responses) == 1
    assert len({item["activation_seal_sha256"] for item in responses}) == 1
    assert environment["seal"].stat().st_nlink == 1
    assert environment["receipt"].stat().st_nlink == 1


def test_symlinked_evidence_and_caller_selected_paths_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path, monkeypatch)
    selected = environment["evidence_root"] / activation.INERT_PREFLIGHT_NAME
    environment["evidence_root"].chmod(0o700)
    original = selected.with_name("inert-preflight.original")
    selected.rename(original)
    selected.symlink_to(original)
    environment["evidence_root"].chmod(0o500)
    with pytest.raises(activation.OwnerGateActivationSealError):
        _install(environment)
    assert not environment["seal"].exists()
    with pytest.raises(
        activation.OwnerGateActivationSealError,
        match="owner_gate_activation_fixed_path_required",
    ):
        activation.install_activation_seal(
            release=environment["release"],
            evidence_root=environment["evidence_root"],
            receipt_path=tmp_path / "caller-selected.json",
            seal_path=environment["seal"],
            lock_path=environment["lock"],
            now_unix=NOW,
        )


def test_file_boundary_traceback_suppresses_sensitive_os_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "evidence.json"
    source.write_bytes(b"{}")
    source.chmod(0o444)
    sensitive_detail = "https://token.invalid/?credential=do-not-render"

    def fail_open(*_args, **_kwargs) -> int:
        raise OSError(sensitive_detail)

    with monkeypatch.context() as scoped:
        scoped.setattr(activation.os, "open", fail_open)
        with pytest.raises(
            activation.OwnerGateActivationSealError,
            match="^stable_activation_read_failure$",
        ) as captured:
            activation._read_regular(
                source,
                maximum=1024,
                uid=source.stat().st_uid,
                gid=source.stat().st_gid,
                modes=frozenset({0o444}),
                code="stable_activation_read_failure",
            )

    rendered = "".join(
        traceback.format_exception(
            captured.type,
            captured.value,
            captured.tb,
        )
    )
    assert rendered.rstrip().endswith(
        ": stable_activation_read_failure"
    )
    assert sensitive_detail not in rendered
    assert captured.value.__cause__ is None
    assert captured.value.__suppress_context__ is True


def test_publication_unlink_error_is_stable_and_secret_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path, monkeypatch)
    sensitive_detail = "https://token.invalid/?credential=unlink-secret"

    def fail_unlink(*_args: Any, **_kwargs: Any) -> None:
        raise OSError(sensitive_detail)

    with monkeypatch.context() as scoped:
        scoped.setattr(activation.os, "unlink", fail_unlink)
        with pytest.raises(
            activation.OwnerGateActivationSealError,
            match="^owner_gate_activation_publication_failed$",
        ) as captured:
            _install(environment)

    rendered = "".join(
        traceback.format_exception(
            captured.type,
            captured.value,
            captured.tb,
        )
    )
    assert sensitive_detail not in rendered
    assert captured.value.__cause__ is None
    assert captured.value.__suppress_context__ is True


def test_lock_release_error_is_stable_and_secret_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _environment(tmp_path, monkeypatch)
    sensitive_detail = "https://token.invalid/?credential=lock-secret"
    flock = activation.fcntl.flock

    def fail_unlock(descriptor: int, operation: int) -> None:
        if operation == activation.fcntl.LOCK_UN:
            raise OSError(sensitive_detail)
        flock(descriptor, operation)

    with monkeypatch.context() as scoped:
        scoped.setattr(activation.fcntl, "flock", fail_unlock)
        with pytest.raises(
            activation.OwnerGateActivationSealError,
            match="^owner_gate_activation_lock_release_failed$",
        ) as captured:
            _install(environment)

    rendered = "".join(
        traceback.format_exception(
            captured.type,
            captured.value,
            captured.tb,
        )
    )
    assert sensitive_detail not in rendered
    assert captured.value.__cause__ is None
    assert captured.value.__suppress_context__ is True
    assert environment["seal"].exists()
    assert environment["receipt"].exists()

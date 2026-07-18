from __future__ import annotations

import hashlib
import inspect
import json
import os
import pwd
import stat
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from scripts.canary import direct_iam_identity_authority as direct_iam
from scripts.canary import owner_gate_foundation as foundation
from scripts.canary import owner_gate_foundation_apply as foundation_apply
from scripts.canary import owner_gate_foundation_journal as foundation_journal
from scripts.canary import owner_gate_trust as trust
from scripts.canary import owner_gate_trust_author as author
from scripts.canary import trusted_signer_author
from tests.scripts.canary import test_owner_gate_pre_foundation as pre_fixture


_REAL_VALIDATE_FOUNDATION_CHAIN = author._validate_foundation_chain_files


@pytest.fixture(autouse=True)
def _isolated_authority_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "authority-parent"
    parent.mkdir(mode=0o700)
    directory = parent / "owner-gate-release-authority"
    monkeypatch.setattr(author, "AUTHORITY_PARENT", parent)
    monkeypatch.setattr(author, "KEY_DIRECTORY", directory)
    monkeypatch.setattr(author, "MANIFEST_DIRECTORY", directory / "manifests")
    monkeypatch.setattr(
        author,
        "_validate_foundation_chain_files",
        lambda **_kwargs: author._ValidatedFoundationChain._create(
            final_release_revision="a" * 40,
            pre_foundation_authority_sha256="4" * 64,
            foundation_apply_receipt_sha256="5" * 64,
            bootstrap_network_collector_public_key_id="9" * 64,
            foundation_source_revision="0" * 40,
            direct_iam_identity_authority_sha256="e" * 64,
            project_ancestry_evidence_sha256="6" * 64,
            project_ancestry_chain_sha256="7" * 64,
            resource_ancestor_chain=("organizations/123456789012",),
            interpreter_sha256="f" * 64,
            interpreter_version="3.11.2",
        ),
    )


def _initialize(tmp_path: Path) -> tuple[Path, Path, Path, str]:
    del tmp_path
    directory = author.KEY_DIRECTORY
    receipt = author.initialize_keypair(directory)
    private_path = directory / author.PRIVATE_KEY_NAME
    public_path = directory / author.PUBLIC_KEY_NAME
    return directory, private_path, public_path, receipt["public_key_sha256"]


def _unsigned(key_id: str) -> dict[str, object]:
    image_name = "debian-12-bookworm-v20260710"
    return {
        "schema": trust.TRUST_SCHEMA,
        "approved_for_offline_install": True,
        "fork_repository": trust.FORK_REPOSITORY,
        "release_revision": "a" * 40,
        "source_tree_oid": "b" * 40,
        "package_inventory_sha256": "c" * 64,
        "boot_image_self_link": (
            f"projects/debian-cloud/global/images/{image_name}"
        ),
        "collector_public_key_ids": {
            "network": "1" * 64,
            "cloud": "2" * 64,
            "host": "3" * 64,
        },
        "credential_migration_envelope_sha256": "d" * 64,
        "direct_iam_identity_authority_sha256": "e" * 64,
        "pre_foundation_authority_sha256": "4" * 64,
        "foundation_apply_receipt_sha256": "5" * 64,
        "project_ancestry_evidence_sha256": "6" * 64,
        "project_ancestry_chain_sha256": "7" * 64,
        "resource_ancestor_chain": ["organizations/123456789012"],
        "interpreter_image": {
            "project": "debian-cloud",
            "image_name": image_name,
            "image_numeric_id": "1234567890123456789",
            "image_self_link": (
                "https://www.googleapis.com/compute/v1/projects/"
                f"debian-cloud/global/images/{image_name}"
            ),
            "python_version": "3.11.2",
            "interpreter_sha256": "f" * 64,
        },
        "release_attestation": {
            "purpose": trust.ATTESTATION_PURPOSE,
            "attested_at_unix": 1_784_300_000,
        },
        "signer_key_id": key_id,
    }


def _write_unsigned(path: Path, value: dict[str, object]) -> None:
    path.write_bytes(foundation.canonical_json_bytes(value))
    path.chmod(0o444)


def _chain_paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "pre_foundation_authority_path": tmp_path / "pre-foundation.json",
        "owner_reauthentication_receipt_path": tmp_path / "owner-reauth.json",
        "network_evidence_path": tmp_path / "network-evidence.json",
        "network_collector_public_key_path": tmp_path / "network.pub",
        "project_ancestry_evidence_path": tmp_path / "ancestry.json",
        "project_ancestry_collector_public_key_path": (
            tmp_path / "ancestry.pub"
        ),
        "direct_iam_identity_authority_path": tmp_path / "direct-iam.json",
    }


def _real_chain_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    direct_overrides: dict[str, object] | None = None,
) -> tuple[dict[str, Path], author._ValidatedFoundationChain]:
    _directory, private_path, public_path, release_key_id = _initialize(tmp_path)
    release_key = Ed25519PrivateKey.from_private_bytes(private_path.read_bytes())
    monkeypatch.setattr(
        trust,
        "PINNED_RELEASE_TRUST_PUBLIC_KEY_SHA256",
        release_key_id,
    )
    observation_root = author.KEY_DIRECTORY / "observation-signers"
    monkeypatch.setattr(trusted_signer_author, "OBSERVATION_ROOT", observation_root)
    initialization = trusted_signer_author.initialize_observation_keys(
        pre_fixture.REVISION,
        mode="network-bootstrap",
    )
    network_private = Ed25519PrivateKey.from_private_bytes(
        trusted_signer_author._private_path(
            pre_fixture.REVISION, "network"
        ).read_bytes()
    )
    network_public_path = Path(
        initialization["public_keys"]["network"]["path"]
    )
    network_key_id = initialization["public_keys"]["network"][
        "public_key_id"
    ]
    monkeypatch.setattr(pre_fixture, "RELEASE_KEY", release_key)
    monkeypatch.setattr(pre_fixture, "RELEASE_KEY_ID", release_key_id)
    monkeypatch.setattr(pre_fixture, "NETWORK_KEY", network_private)
    monkeypatch.setattr(pre_fixture, "NETWORK_KEY_ID", network_key_id)
    original_signed_ancestry = pre_fixture._signed_ancestry_raw
    monkeypatch.setattr(
        pre_fixture,
        "_signed_ancestry_raw",
        lambda **kwargs: original_signed_ancestry(
            key=network_private,
            **kwargs,
        ),
    )
    reauthentication = pre_fixture._owner_reauth_receipt()
    evidence = pre_fixture._evidence(network_private)
    plan = pre_fixture._plan(evidence=evidence)
    authority_value, plan, _evidence = pre_fixture._authority(
        plan=plan,
        evidence=evidence,
    )
    apply_value = pre_fixture._apply_receipt(authority_value, plan)
    paths = _chain_paths(tmp_path)
    values = {
        paths["pre_foundation_authority_path"]: authority_value,
        paths["owner_reauthentication_receipt_path"]: reauthentication,
        paths["network_evidence_path"]: pre_fixture._signed_network_evidence(
            network_private
        ),
    }
    for path, value in values.items():
        path.write_bytes(foundation.canonical_json_bytes(value))
        path.chmod(0o444)
    paths["project_ancestry_evidence_path"].write_bytes(
        pre_fixture._signed_ancestry_raw()
    )
    paths["project_ancestry_evidence_path"].chmod(0o444)
    paths["network_collector_public_key_path"] = network_public_path
    paths["project_ancestry_collector_public_key_path"].write_bytes(
        network_public_path.read_bytes()
    )
    paths["project_ancestry_collector_public_key_path"].chmod(0o444)

    foundation_a = foundation_apply.decode_validated_foundation_a_chain(
        pre_foundation_authority_raw=foundation.canonical_json_bytes(
            authority_value
        ),
        owner_reauthentication_receipt_raw=foundation.canonical_json_bytes(
            reauthentication
        ),
        network_evidence_raw=foundation.canonical_json_bytes(
            pre_fixture._signed_network_evidence(network_private)
        ),
        project_ancestry_evidence_raw=(
            paths["project_ancestry_evidence_path"].read_bytes()
        ),
        release_public_key=release_key.public_key(),
        network_collector_public_key=network_private.public_key(),
        project_ancestry_collector_public_key=network_private.public_key(),
        now_unix=pre_fixture.NOW + 20,
    )
    journal_parent = tmp_path.stat()
    store = foundation_journal.FoundationApplyJournal(
        _root=tmp_path / "foundation-journal",
        _owner_uid=journal_parent.st_uid,
        _owner_gid=journal_parent.st_gid,
    )
    store._require_owner_process = lambda: None  # type: ignore[method-assign]
    transaction_id = foundation_apply._transaction_id(foundation_a)
    foundation_apply._publish_transition(
        journal=store,
        transaction_id=transaction_id,
        name="success",
        body=foundation_apply._transition_body(
            chain=foundation_a,
            transaction_id=transaction_id,
            phase="success",
            payload={"receipt": apply_value},
        ),
        private_key=release_key,
    )
    monkeypatch.setattr(
        foundation_apply.foundation_journal,
        "FoundationApplyJournal",
        lambda: store,
    )
    monkeypatch.setattr(
        foundation_apply.time,
        "time",
        lambda: float(pre_fixture.NOW + 40),
    )
    direct_raw = foundation.canonical_json_bytes({"fixture": "direct-iam"})
    paths["direct_iam_identity_authority_path"].write_bytes(direct_raw)
    paths["direct_iam_identity_authority_path"].chmod(0o444)
    identities = {
        item["step_name"]: item["resource_identity"]
        for item in apply_value["applied_steps"]
    }
    service_account = identities["create_dedicated_service_account"]
    project_role = identities["create_narrow_iam_observation_reader_role"]
    project_binding = identities[
        "bind_narrow_iam_observation_reader_to_owner_gate_service_account"
    ]
    mutation_role = identities["create_narrow_storage_executor_role"]
    ancestor_role = identities[
        "create_narrow_organization_iam_observation_reader_role"
    ]
    ancestor_binding = identities[
        "bind_narrow_organization_iam_observation_reader_to_owner_gate_service_account"
    ]
    vm = identities["create_private_owner_gate_vm"]
    direct_value: dict[str, object] = {
        "pre_foundation_authority_sha256": authority_value[
            "pre_foundation_authority_sha256"
        ],
        "foundation_apply_receipt_sha256": apply_value[
            "foundation_apply_receipt_sha256"
        ],
        "owner_reauthentication_receipt_sha256": reauthentication[
            "owner_reauthentication_receipt_sha256"
        ],
        "collected_at_unix": pre_fixture.NOW + 30,
        "resource_ancestor_chain": [
            f"folders/{pre_fixture.FOLDER_ID}",
            f"organizations/{pre_fixture.ORGANIZATION_ID}",
        ],
        "owner_gate_vm_numeric_id": vm["numeric_id"],
        "owner_gate_vm_name": vm["name"],
        "owner_gate_service_account_email": service_account["email"],
        "owner_gate_service_account_unique_id": service_account["unique_id"],
        "project_read_role": project_role["name"],
        "project_read_role_etag": project_role["etag"],
        "project_read_binding_member": project_binding["member"],
        "ancestor_read_role": ancestor_role["name"],
        "ancestor_read_role_etag": ancestor_role["etag"],
        "ancestor_binding_member": ancestor_binding["member"],
        "mutation_role": mutation_role["name"],
        "mutation_role_etag": mutation_role["etag"],
        "external_gcp_admin_trust_root": {
            "resource_policy_generations": [
                {
                    "resource": f"projects/{foundation.PROJECT}",
                    "etag": project_binding["policy_etag"],
                },
                {
                    "resource": ancestor_binding["resource_name"],
                    "etag": ancestor_binding["policy_etag"],
                },
            ]
        },
    }
    direct_value.update(direct_overrides or {})

    def decode_direct(raw: bytes, *, release_revision: str | None = None):
        assert raw == direct_raw
        assert release_revision == pre_fixture.REVISION
        return dict(direct_value)

    monkeypatch.setattr(direct_iam, "decode_canonical", decode_direct)
    chain = _REAL_VALIDATE_FOUNDATION_CHAIN(
        **paths,
        release_public_key=release_key.public_key(),
        final_release_revision="c" * 40,
        now_unix=pre_fixture.NOW + 40,
    )
    assert public_path.read_bytes() == release_key.public_key().public_bytes_raw()
    return paths, chain


def _decode_foundation_a_from_paths(
    paths: dict[str, Path],
) -> foundation_apply.ValidatedFoundationAChain:
    release_key = Ed25519PrivateKey.from_private_bytes(
        (author.KEY_DIRECTORY / author.PRIVATE_KEY_NAME).read_bytes()
    )
    network_key = Ed25519PublicKey.from_public_bytes(
        paths["network_collector_public_key_path"].read_bytes()
    )
    ancestry_key = Ed25519PublicKey.from_public_bytes(
        paths["project_ancestry_collector_public_key_path"].read_bytes()
    )
    return foundation_apply.decode_validated_foundation_a_chain(
        pre_foundation_authority_raw=paths[
            "pre_foundation_authority_path"
        ].read_bytes(),
        owner_reauthentication_receipt_raw=paths[
            "owner_reauthentication_receipt_path"
        ].read_bytes(),
        network_evidence_raw=paths["network_evidence_path"].read_bytes(),
        project_ancestry_evidence_raw=paths[
            "project_ancestry_evidence_path"
        ].read_bytes(),
        release_public_key=release_key.public_key(),
        network_collector_public_key=network_key,
        project_ancestry_collector_public_key=ancestry_key,
        now_unix=pre_fixture.NOW + 40,
    )


def test_initialize_keypair_is_private_replay_safe_and_secret_free(
    tmp_path: Path,
) -> None:
    directory, private_path, public_path, key_id = _initialize(tmp_path)
    receipt = author.initialize_keypair(directory)

    assert receipt == {
        "schema": "muncho-owner-gate-release-key-initialization.v1",
        "key_initialized": True,
        "private_key_material_printed": False,
        "private_key_digest_printed": False,
        "public_key_path": str(public_path),
        "public_key_sha256": key_id,
    }
    assert private_path.stat().st_size == 32
    assert public_path.stat().st_size == 32
    assert stat.S_IMODE(private_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(public_path.stat().st_mode) == 0o444
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert private_path.read_bytes() not in foundation.canonical_json_bytes(receipt)
    assert hashlib.sha256(public_path.read_bytes()).hexdigest() == key_id


def test_initialize_recovers_public_half_only_from_exact_private_key(
    tmp_path: Path,
) -> None:
    directory, private_path, public_path, key_id = _initialize(tmp_path)
    public_path.chmod(0o600)
    public_path.unlink()

    receipt = author.initialize_keypair(directory)

    assert receipt["public_key_sha256"] == key_id
    assert private_path.exists()
    assert stat.S_IMODE(public_path.stat().st_mode) == 0o444


def test_initialize_recovers_crash_after_public_stage_open(
    tmp_path: Path,
) -> None:
    directory, _private_path, public_path, key_id = _initialize(tmp_path)
    public_path.chmod(0o600)
    public_path.unlink()
    stage = public_path.with_name(f".{public_path.name}.stage")
    descriptor = os.open(stage, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)

    receipt = author.initialize_keypair(directory)

    assert receipt["public_key_sha256"] == key_id
    assert not stage.exists()
    assert stat.S_IMODE(public_path.stat().st_mode) == 0o444


def test_initialize_recovers_complete_and_discards_partial_private_stage(
    tmp_path: Path,
) -> None:
    directory = author.KEY_DIRECTORY
    directory.mkdir(mode=0o700)
    author.MANIFEST_DIRECTORY.mkdir(mode=0o700)
    private_path = directory / author.PRIVATE_KEY_NAME
    stage = private_path.with_name(f".{private_path.name}.stage")
    stage.write_bytes(b"K" * 32)
    stage.chmod(0o600)

    author.initialize_keypair(directory)

    assert private_path.read_bytes() == b"K" * 32
    assert not stage.exists()

    for item in directory.iterdir():
        if item.is_file():
            item.chmod(0o600)
            item.unlink()
    stage.write_bytes(b"partial")
    stage.chmod(0o600)
    author.initialize_keypair(directory)
    assert private_path.stat().st_size == 32
    assert private_path.read_bytes() != b"partial"
    assert not stage.exists()


def test_initialize_rejects_symlink_directory_and_public_mismatch(
    tmp_path: Path,
) -> None:
    real = author.AUTHORITY_PARENT / "real"
    real.mkdir(mode=0o700)
    linked = author.KEY_DIRECTORY
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(
        author.OwnerGateTrustAuthorError,
        match="owner_gate_trust_author_directory_invalid",
    ):
        author.initialize_keypair(linked)
    linked.unlink()

    directory, _private_path, public_path, _key_id = _initialize(tmp_path)
    public_path.chmod(0o600)
    public_path.write_bytes(b"X" * 32)
    public_path.chmod(0o444)
    with pytest.raises(
        author.OwnerGateTrustAuthorError,
        match="owner_gate_trust_author_keypair_mismatch",
    ):
        author.initialize_keypair(directory)


def test_sign_manifest_requires_pinned_matching_key_and_postverifies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _directory, private_path, public_path, key_id = _initialize(tmp_path)
    unsigned_path = tmp_path / "unsigned.json"
    output_path = author.MANIFEST_DIRECTORY / f"{'a' * 40}.trust.json"
    _write_unsigned(unsigned_path, _unsigned(key_id))
    monkeypatch.setattr(
        trust,
        "PINNED_RELEASE_TRUST_PUBLIC_KEY_SHA256",
        key_id,
    )

    receipt = author.sign_manifest(
        unsigned_path=unsigned_path,
        private_key_path=private_path,
        public_key_path=public_path,
        output_path=output_path,
        **_chain_paths(tmp_path),
    )

    assert receipt["manifest_authored"] is True
    assert receipt["release_revision"] == "a" * 40
    assert receipt["public_key_sha256"] == key_id
    assert receipt["project_ancestry_evidence_sha256"] == "6" * 64
    assert receipt["project_ancestry_chain_sha256"] == "7" * 64
    assert receipt["resource_ancestor_chain"] == [
        "organizations/123456789012"
    ]
    assert receipt["private_key_material_printed"] is False
    assert receipt["private_key_digest_printed"] is False
    assert stat.S_IMODE(output_path.stat().st_mode) == 0o444
    assert hashlib.sha256(output_path.read_bytes()).hexdigest() == receipt[
        "manifest_sha256"
    ]
    verified = trust.load_pinned_release_trust(
        manifest_path=output_path,
        public_key_path=public_path,
        expected_uid=os.geteuid(),
    )
    assert verified["release_revision"] == "a" * 40


def test_real_foundation_chain_accepts_distinct_collector_paths_with_equal_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, chain = _real_chain_inputs(tmp_path, monkeypatch)

    assert (
        paths["network_collector_public_key_path"]
        != paths["project_ancestry_collector_public_key_path"]
    )
    assert paths["network_collector_public_key_path"].read_bytes() == paths[
        "project_ancestry_collector_public_key_path"
    ].read_bytes()
    assert chain.pre_foundation_authority_sha256
    assert chain.foundation_apply_receipt_sha256
    assert chain.foundation_source_revision == pre_fixture.REVISION
    assert chain.project_ancestry_evidence_sha256
    assert chain.project_ancestry_chain_sha256
    assert chain.resource_ancestor_chain == (
        f"folders/{pre_fixture.FOLDER_ID}",
        f"organizations/{pre_fixture.ORGANIZATION_ID}",
    )
    assert chain.interpreter_sha256 == pre_fixture.INTERPRETER_SHA256
    assert chain.interpreter_version == "3.11.2"


def test_real_foundation_chain_rejects_one_path_for_two_collector_authorities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, _chain = _real_chain_inputs(tmp_path, monkeypatch)
    paths["project_ancestry_collector_public_key_path"] = paths[
        "network_collector_public_key_path"
    ]
    release_key = Ed25519PrivateKey.from_private_bytes(
        (author.KEY_DIRECTORY / author.PRIVATE_KEY_NAME).read_bytes()
    )

    with pytest.raises(
        author.OwnerGateTrustAuthorError,
        match="owner_gate_trust_author_foundation_chain_invalid",
    ):
        _REAL_VALIDATE_FOUNDATION_CHAIN(
            **paths,
            release_public_key=release_key.public_key(),
            final_release_revision="c" * 40,
            now_unix=pre_fixture.NOW + 40,
        )


def test_real_foundation_chain_rejects_a_success_journal_for_b(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, _chain = _real_chain_inputs(tmp_path, monkeypatch)
    network_private = Ed25519PrivateKey.from_private_bytes(
        trusted_signer_author._private_path(
            pre_fixture.REVISION, "network"
        ).read_bytes()
    )
    evidence_b = pre_fixture._evidence(
        network_private,
        collected_at_unix=pre_fixture.NOW - 2,
    )
    authority_b, _plan_b, _evidence_b = pre_fixture._authority(
        evidence=evidence_b,
        expires_at_unix=pre_fixture.NOW + 299,
    )
    authority_path = paths["pre_foundation_authority_path"]
    authority_path.chmod(0o600)
    authority_path.write_bytes(foundation.canonical_json_bytes(authority_b))
    authority_path.chmod(0o444)
    network_path = paths["network_evidence_path"]
    network_path.chmod(0o600)
    network_path.write_bytes(
        foundation.canonical_json_bytes(
            pre_fixture._signed_network_evidence(
                network_private,
                collected_at_unix=pre_fixture.NOW - 2,
            )
        )
    )
    network_path.chmod(0o444)
    release_key = Ed25519PrivateKey.from_private_bytes(
        (author.KEY_DIRECTORY / author.PRIVATE_KEY_NAME).read_bytes()
    )

    with pytest.raises(
        author.OwnerGateTrustAuthorError,
        match="owner_gate_trust_author_foundation_chain_invalid",
    ):
        _REAL_VALIDATE_FOUNDATION_CHAIN(
            **paths,
            release_public_key=release_key.public_key(),
            final_release_revision="c" * 40,
            now_unix=pre_fixture.NOW + 40,
        )


def test_real_foundation_chain_rejects_conflicting_failure_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, _chain = _real_chain_inputs(tmp_path, monkeypatch)
    foundation_a = _decode_foundation_a_from_paths(paths)
    store = foundation_apply.foundation_journal.FoundationApplyJournal()
    transaction_id = foundation_apply._transaction_id(foundation_a)
    release_key = Ed25519PrivateKey.from_private_bytes(
        (author.KEY_DIRECTORY / author.PRIVATE_KEY_NAME).read_bytes()
    )
    foundation_apply._publish_transition(
        journal=store,
        transaction_id=transaction_id,
        name="failure-intent",
        body=foundation_apply._transition_body(
            chain=foundation_a,
            transaction_id=transaction_id,
            phase="failure_intent",
            payload={"test_conflict": True},
        ),
        private_key=release_key,
    )

    with pytest.raises(
        author.OwnerGateTrustAuthorError,
        match="owner_gate_trust_author_foundation_chain_invalid",
    ):
        _REAL_VALIDATE_FOUNDATION_CHAIN(
            **paths,
            release_public_key=release_key.public_key(),
            final_release_revision="c" * 40,
            now_unix=pre_fixture.NOW + 40,
        )


def test_real_foundation_chain_never_recovers_pending_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, _chain = _real_chain_inputs(tmp_path, monkeypatch)
    foundation_a = _decode_foundation_a_from_paths(paths)
    store = foundation_apply.foundation_journal.FoundationApplyJournal()
    transaction_id = foundation_apply._transaction_id(foundation_a)
    pending = store.root / transaction_id / ".failure-intent.pending"
    pending.write_bytes(b"{}")
    pending.chmod(0o600)
    release_key = Ed25519PrivateKey.from_private_bytes(
        (author.KEY_DIRECTORY / author.PRIVATE_KEY_NAME).read_bytes()
    )

    with pytest.raises(
        author.OwnerGateTrustAuthorError,
        match="owner_gate_trust_author_foundation_chain_invalid",
    ):
        _REAL_VALIDATE_FOUNDATION_CHAIN(
            **paths,
            release_public_key=release_key.public_key(),
            final_release_revision="c" * 40,
            now_unix=pre_fixture.NOW + 40,
        )
    assert pending.read_bytes() == b"{}"


@pytest.mark.parametrize(
    "direct_overrides",
    (
        {"owner_reauthentication_receipt_sha256": "7" * 64},
        {"collected_at_unix": pre_fixture.NOW + 19},
        {"collected_at_unix": pre_fixture.NOW + 301},
    ),
)
def test_real_foundation_chain_rejects_unrelated_or_stale_direct_iam_collection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    direct_overrides: dict[str, object],
) -> None:
    with pytest.raises(
        author.OwnerGateTrustAuthorError,
        match="owner_gate_trust_author_foundation_chain_invalid",
    ):
        _real_chain_inputs(
            tmp_path,
            monkeypatch,
            direct_overrides=direct_overrides,
        )


@pytest.mark.parametrize(
    "direct_overrides",
    (
        {"owner_gate_vm_numeric_id": "9999999999999999999"},
        {"owner_gate_service_account_unique_id": "999999999999999999999"},
        {"project_read_role_etag": "valid-looking-recreated-role-etag"},
        {"mutation_role_etag": "valid-looking-recreated-mutation-role-etag"},
    ),
)
def test_real_foundation_chain_rejects_recreated_security_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    direct_overrides: dict[str, object],
) -> None:
    with pytest.raises(
        author.OwnerGateTrustAuthorError,
        match="owner_gate_trust_author_foundation_identity_mismatch",
    ):
        _real_chain_inputs(
            tmp_path,
            monkeypatch,
            direct_overrides=direct_overrides,
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("pre_foundation_authority_sha256", "6" * 64),
        ("foundation_apply_receipt_sha256", "6" * 64),
        ("direct_iam_identity_authority_sha256", "6" * 64),
    ),
)
def test_sign_manifest_rejects_caller_hash_drift_from_validated_raw_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: str,
) -> None:
    _directory, private_path, public_path, key_id = _initialize(tmp_path)
    monkeypatch.setattr(
        trust,
        "PINNED_RELEASE_TRUST_PUBLIC_KEY_SHA256",
        key_id,
    )
    unsigned = _unsigned(key_id)
    unsigned[field] = replacement
    unsigned_path = tmp_path / "unsigned.json"
    _write_unsigned(unsigned_path, unsigned)
    output_path = author.MANIFEST_DIRECTORY / f"{'a' * 40}.trust.json"

    with pytest.raises(
        author.OwnerGateTrustAuthorError,
        match="owner_gate_trust_author_foundation_chain_mismatch",
    ):
        author.sign_manifest(
            unsigned_path=unsigned_path,
            private_key_path=private_path,
            public_key_path=public_path,
            output_path=output_path,
            **_chain_paths(tmp_path),
        )
    assert not output_path.exists()


def test_sign_manifest_fails_closed_for_unpinned_key_without_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _directory, private_path, public_path, key_id = _initialize(tmp_path)
    unsigned_path = tmp_path / "unsigned.json"
    output_path = author.MANIFEST_DIRECTORY / f"{'a' * 40}.trust.json"
    _write_unsigned(unsigned_path, _unsigned(key_id))
    monkeypatch.setattr(
        trust,
        "PINNED_RELEASE_TRUST_PUBLIC_KEY_SHA256",
        "0" * 64,
    )

    with pytest.raises(
        author.OwnerGateTrustAuthorError,
        match="owner_gate_trust_author_key_not_pinned",
    ):
        author.sign_manifest(
            unsigned_path=unsigned_path,
            private_key_path=private_path,
            public_key_path=public_path,
            output_path=output_path,
            **_chain_paths(tmp_path),
        )
    assert not output_path.exists()


def test_sign_manifest_rejects_noncanonical_input_and_no_clobber(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _directory, private_path, public_path, key_id = _initialize(tmp_path)
    monkeypatch.setattr(
        trust,
        "PINNED_RELEASE_TRUST_PUBLIC_KEY_SHA256",
        key_id,
    )
    unsigned_path = tmp_path / "unsigned.json"
    unsigned_path.write_text(json.dumps(_unsigned(key_id), indent=2), "utf-8")
    unsigned_path.chmod(0o444)
    output_path = author.MANIFEST_DIRECTORY / f"{'a' * 40}.trust.json"

    with pytest.raises(
        author.OwnerGateTrustAuthorError,
        match="owner_gate_trust_author_unsigned_manifest_not_canonical",
    ):
        author.sign_manifest(
            unsigned_path=unsigned_path,
            private_key_path=private_path,
            public_key_path=public_path,
            output_path=output_path,
            **_chain_paths(tmp_path),
        )

    unsigned_path.chmod(0o600)
    _write_unsigned(unsigned_path, _unsigned(key_id))
    output_path.write_bytes(b"do-not-replace")
    original = output_path.read_bytes()
    with pytest.raises(
        author.OwnerGateTrustAuthorError,
        match="owner_gate_trust_author_manifest_write_failed",
    ):
        author.sign_manifest(
            unsigned_path=unsigned_path,
            private_key_path=private_path,
            public_key_path=public_path,
            output_path=output_path,
            **_chain_paths(tmp_path),
        )
    assert output_path.read_bytes() == original


def test_sign_manifest_recovers_crash_after_manifest_stage_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _directory, private_path, public_path, key_id = _initialize(tmp_path)
    unsigned_path = tmp_path / "unsigned.json"
    _write_unsigned(unsigned_path, _unsigned(key_id))
    output_path = author.MANIFEST_DIRECTORY / f"{'a' * 40}.trust.json"
    stage = output_path.with_name(f".{output_path.name}.stage")
    descriptor = os.open(stage, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    monkeypatch.setattr(
        trust,
        "PINNED_RELEASE_TRUST_PUBLIC_KEY_SHA256",
        key_id,
    )

    receipt = author.sign_manifest(
        unsigned_path=unsigned_path,
        private_key_path=private_path,
        public_key_path=public_path,
        output_path=output_path,
        **_chain_paths(tmp_path),
    )

    assert receipt["manifest_authored"] is True
    assert not stage.exists()
    assert stat.S_IMODE(output_path.stat().st_mode) == 0o444


def test_authority_home_ignores_home_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "attacker-controlled-repo"))
    account_home = Path(pwd.getpwuid(os.geteuid()).pw_dir)

    assert author.OWNER_HOME == account_home
    assert author.AUTHORITY_PARENT != Path(os.environ["HOME"]) / ".hermes"


def test_trust_author_cli_uses_fixed_journal_and_distinct_ancestry_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameters = inspect.signature(author.sign_manifest).parameters
    assert "foundation_apply_receipt_path" not in parameters
    assert "project_ancestry_collector_public_key_path" in parameters

    captured: dict[str, object] = {}

    def sign_manifest_probe(**kwargs: object) -> dict[str, bool]:
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(author, "sign_manifest", sign_manifest_probe)
    ancestry_key = tmp_path / "project-ancestry-collector.pub"
    arguments = [
        "sign",
        "--unsigned",
        str(tmp_path / "unsigned.json"),
        "--output",
        str(tmp_path / "signed.json"),
        "--pre-foundation-authority",
        str(tmp_path / "pre-foundation.json"),
        "--owner-reauth-receipt",
        str(tmp_path / "owner-reauth.json"),
        "--network-evidence",
        str(tmp_path / "network-evidence.json"),
        "--network-collector-public-key",
        str(tmp_path / "network-collector.pub"),
        "--project-ancestry-evidence",
        str(tmp_path / "project-ancestry.json"),
        "--project-ancestry-collector-public-key",
        str(ancestry_key),
        "--direct-iam-identity-authority",
        str(tmp_path / "direct-iam.json"),
    ]

    assert author.main(arguments) == 0
    assert captured["project_ancestry_collector_public_key_path"] == ancestry_key
    assert "foundation_apply_receipt_path" not in captured
    with pytest.raises(SystemExit):
        author.main(
            [
                *arguments,
                "--foundation-apply-receipt",
                str(tmp_path / "caller-selected-apply.json"),
            ]
        )

    assert not hasattr(author, "ValidatedFoundationChain")
    with pytest.raises(
        author.OwnerGateTrustAuthorError,
        match="owner_gate_trust_author_foundation_chain_factory_required",
    ):
        author._ValidatedFoundationChain()

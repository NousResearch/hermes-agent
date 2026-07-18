from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path

import pytest

from scripts.canary import owner_gate_foundation as foundation
from scripts.canary import owner_gate_trust_author as release_author
from scripts.canary import trusted_signer_author as author
from scripts.canary import trusted_signer_provisioning as provisioning


RELEASE = "a" * 40
PACKAGE_SHA256 = "b" * 64
OWNER_RECEIPT_SHA256 = "c" * 64


@pytest.fixture(autouse=True)
def _isolated_authority_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "authority-parent"
    parent.mkdir(mode=0o700)
    key_directory = parent / "owner-gate-release-authority"
    manifests = key_directory / "manifests"
    observation = key_directory / "observation-signers"
    monkeypatch.setattr(release_author, "AUTHORITY_PARENT", parent)
    monkeypatch.setattr(release_author, "KEY_DIRECTORY", key_directory)
    monkeypatch.setattr(release_author, "MANIFEST_DIRECTORY", manifests)
    monkeypatch.setattr(author, "OBSERVATION_ROOT", observation)
    release_author.initialize_keypair()


def test_initialize_observation_keys_is_replay_safe_and_secret_free() -> None:
    receipt = author.initialize_observation_keys(
        RELEASE,
        mode="full-release",
    )
    replay = author.initialize_observation_keys(
        RELEASE,
        mode="full-release",
    )

    assert replay == receipt
    assert receipt["initialization_mode"] == "full-release"
    assert receipt["initialized_roles"] == list(author.ROLES)
    assert receipt["keys_initialized"] is True
    assert receipt["private_key_material_printed"] is False
    assert receipt["private_key_digest_printed"] is False
    assert receipt["provisioning_envelope_materialized"] is False
    receipt_raw = foundation.canonical_json_bytes(receipt)
    for role in author.ROLES:
        private_path = author._private_path(RELEASE, role)
        public_path = author._public_path(RELEASE, role)
        assert stat.S_IMODE(private_path.stat().st_mode) == 0o600
        assert stat.S_IMODE(public_path.stat().st_mode) == 0o444
        assert private_path.stat().st_nlink == 1
        assert public_path.stat().st_nlink == 1
        assert private_path.read_bytes() not in receipt_raw
        assert receipt["public_keys"][role] == {
            "path": str(public_path),
            "public_key_id": hashlib.sha256(public_path.read_bytes()).hexdigest(),
        }


def test_network_bootstrap_creates_only_network_signer() -> None:
    receipt = author.initialize_observation_keys(
        RELEASE,
        mode="network-bootstrap",
    )
    replay = author.initialize_observation_keys(
        RELEASE,
        mode="network-bootstrap",
    )

    assert replay == receipt
    assert receipt["initialization_mode"] == "network-bootstrap"
    assert receipt["initialized_roles"] == ["network"]
    assert set(receipt["public_keys"]) == {"network"}
    assert author._private_path(RELEASE, "network").is_file()
    assert author._public_path(RELEASE, "network").is_file()
    for role in ("cloud", "host"):
        assert not author._private_path(RELEASE, role).exists()
        assert not author._public_path(RELEASE, role).exists()


def test_network_bootstrap_rejects_any_out_of_scope_signer_material() -> None:
    author.initialize_observation_keys(RELEASE, mode="full-release")

    with pytest.raises(
        author.TrustedSignerAuthorError,
        match="^trusted_signer_author_role_scope_invalid$",
    ):
        author.initialize_observation_keys(
            RELEASE,
            mode="network-bootstrap",
        )


def test_initialize_rejects_invalid_release_and_entropy() -> None:
    with pytest.raises(
        author.TrustedSignerAuthorError,
        match="trusted_signer_author_release_invalid",
    ):
        author.initialize_observation_keys("main", mode="full-release")
    with pytest.raises(
        author.TrustedSignerAuthorError,
        match="trusted_signer_author_entropy_invalid",
    ):
        author.initialize_observation_keys(
            RELEASE,
            mode="full-release",
            entropy=lambda _size: b"x",
        )
    with pytest.raises(
        author.TrustedSignerAuthorError,
        match="trusted_signer_author_mode_invalid",
    ):
        author.initialize_observation_keys(RELEASE, mode="network,cloud,host")


@pytest.mark.parametrize("role", ["cloud", "host"])
def test_build_provisioning_envelope_self_validates_and_can_be_wiped(
    role: str,
) -> None:
    receipt = author.initialize_observation_keys(RELEASE, mode="full-release")

    frame = author.build_provisioning_envelope(
        role=role,
        release_revision=RELEASE,
        package_sha256=PACKAGE_SHA256,
        owner_authorization_receipt_sha256=OWNER_RECEIPT_SHA256,
    )

    decoded, seed = provisioning.decode_provisioning_envelope(
        bytes(frame),
        role=role,
        release_revision=RELEASE,
        package_sha256=PACKAGE_SHA256,
    )
    assert decoded["owner_authorization_receipt_sha256"] == OWNER_RECEIPT_SHA256
    assert decoded["public_key_id"] == receipt["public_keys"][role][
        "public_key_id"
    ]
    assert len(seed) == 32
    author.wipe_secret_frame(frame)
    assert frame == bytearray(len(frame))


def test_build_envelope_rejects_network_role_and_wrong_digest() -> None:
    author.initialize_observation_keys(RELEASE, mode="full-release")
    with pytest.raises(
        author.TrustedSignerAuthorError,
        match="trusted_signer_author_envelope_invalid",
    ):
        author.build_provisioning_envelope(
            role="network",
            release_revision=RELEASE,
            package_sha256=PACKAGE_SHA256,
            owner_authorization_receipt_sha256=OWNER_RECEIPT_SHA256,
        )
    with pytest.raises(
        author.TrustedSignerAuthorError,
        match="trusted_signer_author_envelope_invalid",
    ):
        author.build_provisioning_envelope(
            role="cloud",
            release_revision=RELEASE,
            package_sha256="wrong",
            owner_authorization_receipt_sha256=OWNER_RECEIPT_SHA256,
        )


def test_build_envelope_rejects_public_key_drift() -> None:
    author.initialize_observation_keys(RELEASE, mode="full-release")
    public_path = author._public_path(RELEASE, "cloud")
    public_path.chmod(0o600)
    public_path.write_bytes(os.urandom(32))
    public_path.chmod(0o444)

    with pytest.raises(
        author.TrustedSignerAuthorError,
        match="trusted_signer_author_keypair_mismatch",
    ):
        author.build_provisioning_envelope(
            role="cloud",
            release_revision=RELEASE,
            package_sha256=PACKAGE_SHA256,
            owner_authorization_receipt_sha256=OWNER_RECEIPT_SHA256,
        )


def test_wipe_rejects_immutable_input() -> None:
    with pytest.raises(
        author.TrustedSignerAuthorError,
        match="trusted_signer_author_frame_invalid",
    ):
        author.wipe_secret_frame(b"ok")  # type: ignore[arg-type]

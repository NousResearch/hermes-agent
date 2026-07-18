from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.canary import owner_gate_foundation as foundation
from scripts.canary import owner_gate_trust_author as release_author
from scripts.canary import trusted_signer_author as author
from scripts.canary import trusted_signer_provisioning as provisioning


RELEASE = "d" * 40
PACKAGE_SHA256 = "e" * 64
OWNER_RECEIPT_SHA256 = "f" * 64


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


def test_cli_stdout_and_stderr_never_contain_seed_or_seed_digest(
    capfd: pytest.CaptureFixture[str],
) -> None:
    assert author.main([
        "--release-revision",
        RELEASE,
        "--mode",
        "full-release",
    ]) == 0
    captured = capfd.readouterr()
    receipt = json.loads(captured.out)
    assert captured.out.encode("ascii") == foundation.canonical_json_bytes(receipt) + b"\n"

    assert captured.err == ""
    assert receipt["private_key_material_printed"] is False
    assert receipt["private_key_digest_printed"] is False
    assert receipt["initialization_mode"] == "full-release"
    assert receipt["initialized_roles"] == list(author.ROLES)
    for role in author.ROLES:
        seed = author._private_path(RELEASE, role).read_bytes()
        assert seed not in captured.out.encode("utf-8")
        assert hashlib.sha256(seed).hexdigest() not in captured.out


@pytest.mark.parametrize("role", ["cloud", "host"])
def test_secret_frame_is_one_canonical_lf_terminated_document(role: str) -> None:
    author.initialize_observation_keys(RELEASE, mode="full-release")
    frame = author.build_provisioning_envelope(
        role=role,
        release_revision=RELEASE,
        package_sha256=PACKAGE_SHA256,
        owner_authorization_receipt_sha256=OWNER_RECEIPT_SHA256,
    )
    try:
        assert frame.endswith(b"\n")
        assert frame.count(b"\n") == 1
        decoded, _ = provisioning.decode_provisioning_envelope(
            bytes(frame),
            role=role,
            release_revision=RELEASE,
            package_sha256=PACKAGE_SHA256,
        )
        assert bytes(frame[:-1]) == foundation.canonical_json_bytes(decoded)
    finally:
        author.wipe_secret_frame(frame)
    assert frame == bytearray(len(frame))


def test_failed_self_decode_wipes_the_mutable_frame_before_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    author.initialize_observation_keys(RELEASE, mode="full-release")
    wiped: list[bytes] = []
    original_wipe = author.wipe_secret_frame

    def fail_decode(*_: object, **__: object) -> object:
        raise provisioning.TrustedSignerProvisioningError(
            "trusted_signer_provisioning_envelope_invalid"
        )

    def observe_wipe(frame: bytearray) -> None:
        original_wipe(frame)
        wiped.append(bytes(frame))

    monkeypatch.setattr(provisioning, "decode_provisioning_envelope", fail_decode)
    monkeypatch.setattr(author, "wipe_secret_frame", observe_wipe)

    with pytest.raises(
        author.TrustedSignerAuthorError,
        match="^trusted_signer_author_envelope_invalid$",
    ):
        author.build_provisioning_envelope(
            role="cloud",
            release_revision=RELEASE,
            package_sha256=PACKAGE_SHA256,
            owner_authorization_receipt_sha256=OWNER_RECEIPT_SHA256,
        )

    assert len(wiped) == 1
    assert wiped[0]
    assert set(wiped[0]) == {0}

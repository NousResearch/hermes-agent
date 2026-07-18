from __future__ import annotations

import os
from pathlib import Path

import pytest

from scripts.canary import owner_gate_trust_author as release_author
from scripts.canary import owner_gate_v1_credential_author as author
from scripts.canary import owner_gate_v1_credential_migration as migration
from scripts.canary import trusted_signer_author as signer_author


REVISION = "b" * 40


@pytest.fixture
def authority_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    parent = tmp_path / ".hermes"
    parent.mkdir(mode=0o700)
    key_directory = parent / "owner-gate-release-authority"
    key_directory.mkdir(mode=0o700)
    observation = key_directory / "observation-signers"
    monkeypatch.setattr(release_author, "AUTHORITY_PARENT", parent)
    monkeypatch.setattr(release_author, "KEY_DIRECTORY", key_directory)
    monkeypatch.setattr(
        release_author,
        "MANIFEST_DIRECTORY",
        key_directory / "manifests",
    )
    monkeypatch.setattr(signer_author, "OBSERVATION_ROOT", observation)
    signer_author.initialize_observation_keys(
        REVISION,
        mode="full-release",
        entropy=lambda size: b"k" * size,
    )
    return observation / REVISION


def _source() -> dict[str, object]:
    return {
        "schema": migration.SOURCE_RECEIPT_SCHEMA,
        "signed_at_unix": 1_700_000_000,
        "receipt_sha256": "1" * 64,
        "public_fixture": True,
    }


def _envelope() -> dict[str, object]:
    return {
        "schema": migration.MIGRATION_SCHEMA,
        "envelope_sha256": "2" * 64,
        "public_fixture": True,
    }


def _stub_live_collection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        migration,
        "V1CredentialMigrationTransport",
        lambda *, revision: {"revision": revision},
    )
    monkeypatch.setattr(
        migration,
        "collect_and_sign_migration",
        lambda transport, *, release_revision, host_private_key: (
            _envelope(),
            _source(),
        ),
    )
    monkeypatch.setattr(
        migration,
        "validate_source_receipt",
        lambda value, **_kwargs: dict(value),
    )
    monkeypatch.setattr(
        migration,
        "sign_migration_from_source_receipt",
        lambda source_receipt, **_kwargs: _envelope(),
    )


def test_author_publishes_fixed_private_artifacts_and_replays(
    authority_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_live_collection(monkeypatch)

    first = author.author_live_migration(REVISION)
    second = author.author_live_migration(REVISION)

    assert first == second
    source = authority_root / author.SOURCE_FILENAME
    envelope = authority_root / author.ENVELOPE_FILENAME
    receipt = authority_root / author.RECEIPT_FILENAME
    assert os.stat(source).st_mode & 0o777 == 0o400
    assert os.stat(envelope).st_mode & 0o777 == 0o400
    assert os.stat(receipt).st_mode & 0o777 == 0o444
    assert first["source_receipt_path"] == str(source)
    assert first["migration_envelope_path"] == str(envelope)
    assert first["private_key_material_recorded"] is False
    assert first["private_key_digest_recorded"] is False
    assert "credential_id" not in first
    assert "public_key" not in first


def test_source_only_crash_is_recovered_without_live_recollection(
    authority_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_live_collection(monkeypatch)
    author.author_live_migration(REVISION)
    (authority_root / author.ENVELOPE_FILENAME).unlink()
    (authority_root / author.RECEIPT_FILENAME).unlink()
    invoked = False

    def forbidden(*_args: object, **_kwargs: object) -> object:
        nonlocal invoked
        invoked = True
        raise AssertionError("live collection must not replay")

    monkeypatch.setattr(migration, "collect_and_sign_migration", forbidden)

    receipt = author.author_live_migration(REVISION)

    assert invoked is False
    assert receipt["source_receipt_sha256"] == _source()["receipt_sha256"]
    assert (authority_root / author.ENVELOPE_FILENAME).is_file()


def test_existing_mismatched_envelope_fails_closed(
    authority_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_live_collection(monkeypatch)
    author.author_live_migration(REVISION)
    envelope = authority_root / author.ENVELOPE_FILENAME
    envelope.chmod(0o600)
    envelope.write_bytes(b'{"different":true}\n')
    envelope.chmod(0o400)

    with pytest.raises(
        author.V1CredentialAuthorError,
        match="^owner_gate_v1_credential_author_publish_failed$",
    ):
        author.author_live_migration(REVISION)

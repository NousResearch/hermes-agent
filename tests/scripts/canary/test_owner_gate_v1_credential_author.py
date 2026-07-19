from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from scripts.canary import full_canary_owner_launcher as launcher
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
    source_raw = source.read_bytes()
    envelope_raw = envelope.read_bytes()
    receipt_raw = receipt.read_bytes()
    assert source_raw == migration._canonical(_source()) + b"\n"
    assert envelope_raw == migration._canonical(_envelope())
    assert receipt_raw.endswith(b"\n")
    assert not receipt_raw.endswith(b"\n\n")
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


def test_owner_launcher_parser_exposes_only_pathless_credential_action() -> None:
    arguments = launcher._cli_parser().parse_args(
        [
            "--release-sha",
            REVISION,
            "--author-v1-credential-migration",
        ]
    )

    assert arguments.author_v1_credential_migration is True
    with pytest.raises(SystemExit):
        launcher._cli_parser().parse_args(
            [
                "--release-sha",
                REVISION,
                "--author-v1-credential-migration",
                "--publish-stopped-release",
            ]
        )
    with pytest.raises(SystemExit):
        launcher._cli_parser().parse_args(
            [
                "--release-sha",
                REVISION,
                "--author-v1-credential-migration",
                "--credential-migration-envelope",
                "/tmp/caller-selected.json",
            ]
        )


def test_owner_launcher_rejects_credential_external_iam_before_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    emitted: list[dict[str, Any]] = []
    monkeypatch.setattr(
        launcher,
        "require_trusted_owner_runtime",
        lambda _revision: pytest.fail("runtime must not be opened"),
    )
    monkeypatch.setattr(launcher, "_emit_canonical_line", emitted.append)

    result = launcher.main(
        [
            "--release-sha",
            REVISION,
            "--author-v1-credential-migration",
            "--external-iam-policy-sha256",
            "a" * 64,
        ]
    )

    assert result == 2
    assert len(emitted) == 1
    assert emitted[0]["error_code"] == (
        "owner_gate_v1_credential_author_cli_invalid"
    )


def test_owner_launcher_dispatches_credential_author_after_sealed_activation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []

    class Runtime:
        def trusted_command_prefix(self) -> tuple[str, ...]:
            events.append("trusted_command_prefix")
            return ("/fixed/python",)

    runtime = Runtime()
    receipt = {"schema": "credential-author-receipt", "ok": True}
    monkeypatch.setattr(
        launcher,
        "require_trusted_owner_runtime",
        lambda revision: events.append(("runtime", revision)) or runtime,
    )
    monkeypatch.setattr(
        launcher,
        "activate_trusted_owner_support",
        lambda selected, *, release_sha: events.append(
            ("activate", selected, release_sha)
        ),
    )
    monkeypatch.setattr(
        launcher,
        "require_local_launcher_provenance",
        lambda revision: events.append(("provenance", revision)),
    )
    monkeypatch.setattr(
        launcher,
        "_install_canonical_launcher_bridge",
        lambda revision: events.append(("bridge", revision)),
    )
    monkeypatch.setattr(
        author,
        "author_live_migration",
        lambda revision: events.append(("author", revision)) or receipt,
    )
    monkeypatch.setattr(
        launcher,
        "_validate_owner_interpreter_invocation",
        lambda path: events.append(("interpreter", path)),
    )
    monkeypatch.setattr(
        launcher,
        "require_trusted_owner_support_activation",
        lambda selected, *, release_sha: events.append(
            ("support", selected, release_sha)
        ),
    )
    monkeypatch.setattr(
        launcher,
        "_emit_canonical_line",
        lambda value: events.append(("emit", value)),
    )

    result = launcher.main(
        ["--release-sha", REVISION, "--author-v1-credential-migration"]
    )

    assert result == 0
    assert events == [
        ("runtime", REVISION),
        ("activate", runtime, REVISION),
        ("provenance", REVISION),
        ("bridge", REVISION),
        ("author", REVISION),
        "trusted_command_prefix",
        ("interpreter", "/fixed/python"),
        ("support", runtime, REVISION),
        ("provenance", REVISION),
        ("emit", receipt),
    ]


def test_owner_launcher_preserves_stable_credential_error_after_post_guards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []

    class Runtime:
        def trusted_command_prefix(self) -> tuple[str, ...]:
            events.append("trusted_command_prefix")
            return ("/fixed/python",)

    runtime = Runtime()
    monkeypatch.setattr(
        launcher,
        "require_trusted_owner_runtime",
        lambda _revision: runtime,
    )
    monkeypatch.setattr(
        launcher,
        "activate_trusted_owner_support",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        launcher,
        "require_local_launcher_provenance",
        lambda revision: events.append(("provenance", revision)),
    )
    monkeypatch.setattr(
        launcher,
        "_install_canonical_launcher_bridge",
        lambda _revision: None,
    )

    def fail_author(_revision: str) -> dict[str, Any]:
        raise author.V1CredentialAuthorError(
            "owner_gate_v1_credential_author_artifact_invalid"
        )

    monkeypatch.setattr(author, "author_live_migration", fail_author)
    monkeypatch.setattr(
        launcher,
        "_validate_owner_interpreter_invocation",
        lambda path: events.append(("interpreter", path)),
    )
    monkeypatch.setattr(
        launcher,
        "require_trusted_owner_support_activation",
        lambda selected, *, release_sha: events.append(
            ("support", selected, release_sha)
        ),
    )
    emitted: list[dict[str, Any]] = []
    monkeypatch.setattr(launcher, "_emit_canonical_line", emitted.append)

    result = launcher.main(
        ["--release-sha", REVISION, "--author-v1-credential-migration"]
    )

    assert result == 2
    assert events[-4:] == [
        "trusted_command_prefix",
        ("interpreter", "/fixed/python"),
        ("support", runtime, REVISION),
        ("provenance", REVISION),
    ]
    assert len(emitted) == 1
    assert emitted[0]["error_code"] == (
        "owner_gate_v1_credential_author_artifact_invalid"
    )

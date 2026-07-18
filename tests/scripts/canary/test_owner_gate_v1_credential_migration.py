from __future__ import annotations

import base64
import copy
import hashlib
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from scripts.canary import owner_gate_bootstrap as bootstrap
from scripts.canary import owner_gate_v1_credential_migration as migration


REVISION = "a" * 40
NOW = 1_784_400_000
CREDENTIAL_ID = b"public-credential-id"
PUBLIC_KEY = b"public-cose-key"
USER_HANDLE = migration.OWNER_DISCORD_USER_ID.encode("ascii")


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


@pytest.fixture(autouse=True)
def _fixture_digests(monkeypatch: pytest.MonkeyPatch) -> None:
    values = {
        "EXPECTED_CREDENTIAL_ID_SHA256": hashlib.sha256(CREDENTIAL_ID).hexdigest(),
        "EXPECTED_PUBLIC_KEY_SHA256": hashlib.sha256(PUBLIC_KEY).hexdigest(),
        "EXPECTED_USER_HANDLE_SHA256": hashlib.sha256(USER_HANDLE).hexdigest(),
    }
    for name, value in values.items():
        monkeypatch.setattr(migration, name, value)
        monkeypatch.setattr(bootstrap, name, value)


def _registry() -> bytes:
    value = {
        "schema": "muncho.passkey.credentials.v1",
        "credentials": [
            {
                "schema": "muncho.passkey.credential.v1",
                "discord_user_id": migration.OWNER_DISCORD_USER_ID,
                "credential_id": _b64url(CREDENTIAL_ID),
                "credential_id_hash": hashlib.sha256(CREDENTIAL_ID).hexdigest(),
                "public_key": _b64url(PUBLIC_KEY),
                "sign_count": 0,
                "credential_device_type": "multi_device",
                "credential_backed_up": True,
                "aaguid": "00000000-0000-0000-0000-000000000000",
                "enabled": True,
                "label": "owner passkey",
                "created_at": "2026-07-01T00:00:00+00:00",
                "last_used_at": "2026-07-02T00:00:00+00:00",
            }
        ],
    }
    return migration._canonical(value) + b"\n"


def _observation() -> dict[str, object]:
    public = migration._public_credential(_registry())
    unsigned = {
        "schema": migration.OBSERVATION_SCHEMA,
        "release_revision": REVISION,
        "target": {
            "project": migration.PROJECT,
            "zone": migration.ZONE,
            "vm": migration.VM_NAME,
            "instance_id": migration.INSTANCE_ID,
        },
        "source_service": {
            "path": str(migration.SOURCE_SERVICE),
            "uid": migration.SOURCE_UID,
            "gid": migration.SOURCE_GID,
            "mode": f"{migration.SOURCE_MODE:04o}",
            "size": 1234,
            "stable_nofollow_read": True,
            "sha256": migration.EXPECTED_SOURCE_SERVICE_SHA256,
        },
        "credentials_file": {
            "path": str(migration.CREDENTIALS_FILE),
            "uid": migration.CREDENTIALS_UID,
            "gid": migration.CREDENTIALS_GID,
            "mode": f"{migration.CREDENTIALS_MODE:04o}",
            "size": len(_registry()),
            "stable_nofollow_read": True,
        },
        "service": {
            "unit": migration.UNIT,
            "load_state": "loaded",
            "active_state": "active",
            "sub_state": "running",
            "unit_file_state": "enabled",
            "fragment_path": str(migration.UNIT_FRAGMENT),
            "read_only_systemctl_show": True,
        },
        "credential_public_material": public,
        "collected_at_unix": NOW - 2,
        "completed_at_unix": NOW - 1,
        "fresh_through_unix": NOW - 1 + migration.FRESHNESS_SECONDS,
        "collector_authority": "production_root_read_only_fixed_projection",
        "caller_selected_input_accepted": False,
        "production_mutation_performed": False,
        "secret_material_recorded": False,
        "secret_digest_recorded": False,
    }
    return {**unsigned, "report_sha256": migration._sha256_json(unsigned)}


def _transport_authority() -> dict[str, object]:
    return {
        "kind": "pinned_owner_gcloud_iap_ssh_read_only",
        "project": migration.PROJECT,
        "zone": migration.ZONE,
        "vm": migration.VM_NAME,
        "instance_id": migration.INSTANCE_ID,
        "known_hosts_file_sha256": "1" * 64,
        "collector_source_sha256": "2" * 64,
        "instance_authorization_sha256": "3" * 64,
        "project_authorization_sha256": "4" * 64,
        "oslogin_authorization_sha256": "5" * 64,
    }


def test_registry_projection_contains_only_required_public_material() -> None:
    projected = migration._public_credential(_registry())

    assert projected["credential_id_b64url"] == _b64url(CREDENTIAL_ID)
    assert projected["public_key_cose_b64url"] == _b64url(PUBLIC_KEY)
    assert projected["expected_user_handle_b64url"] == _b64url(USER_HANDLE)
    assert "label" not in projected
    assert "aaguid" not in projected
    assert "last_used_at" not in projected


def test_observation_is_exact_and_tamper_evident() -> None:
    value = _observation()
    assert migration.validate_observation(
        value,
        release_revision=REVISION,
        now_unix=NOW,
    ) == value

    changed = copy.deepcopy(value)
    changed["credential_public_material"]["sign_count"] = 1  # type: ignore[index]
    changed["report_sha256"] = migration._sha256_json(
        {key: item for key, item in changed.items() if key != "report_sha256"}
    )
    with pytest.raises(
        migration.V1CredentialMigrationError,
        match="^owner_gate_v1_credential_observation_invalid$",
    ):
        migration.validate_observation(
            changed,
            release_revision=REVISION,
            now_unix=NOW,
        )


def test_host_signed_envelope_matches_offline_bootstrap_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = object.__new__(migration.V1CredentialMigrationTransport)
    monkeypatch.setattr(
        migration.V1CredentialMigrationTransport,
        "observe",
        lambda self, *, release_revision: (
            _observation(),
            _transport_authority(),
        ),
    )
    private = Ed25519PrivateKey.generate()

    envelope, source = migration.collect_and_sign_migration(
        transport,
        release_revision=REVISION,
        host_private_key=private,
        now_unix=NOW,
    )

    key_id = hashlib.sha256(private.public_key().public_bytes_raw()).hexdigest()
    assert migration.validate_source_receipt(
        source,
        release_revision=REVISION,
        host_key_id=key_id,
        now_unix=NOW,
    ) == source
    assert bootstrap.validate_migration(
        envelope,
        release_revision=REVISION,
        host_public_key=private.public_key(),
        host_key_id=key_id,
    ) == envelope
    assert envelope["source_receipt_sha256"] == source["receipt_sha256"]


def test_source_receipt_cannot_be_rebound_to_another_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = object.__new__(migration.V1CredentialMigrationTransport)
    monkeypatch.setattr(
        migration.V1CredentialMigrationTransport,
        "observe",
        lambda self, *, release_revision: (
            _observation(),
            _transport_authority(),
        ),
    )
    private = Ed25519PrivateKey.generate()
    _envelope, source = migration.collect_and_sign_migration(
        transport,
        release_revision=REVISION,
        host_private_key=private,
        now_unix=NOW,
    )
    changed = copy.deepcopy(source)
    changed["transport_authority"]["vm"] = "other-vm"  # type: ignore[index]
    changed["receipt_sha256"] = migration._sha256_json(
        {key: item for key, item in changed.items() if key != "receipt_sha256"}
    )
    key_id = hashlib.sha256(private.public_key().public_bytes_raw()).hexdigest()

    with pytest.raises(
        migration.V1CredentialMigrationError,
        match="^owner_gate_v1_credential_transport_authority_invalid$",
    ):
        migration.validate_source_receipt(
            changed,
            release_revision=REVISION,
            host_key_id=key_id,
            now_unix=NOW,
        )


def test_stable_file_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_bytes(b"public")
    link = tmp_path / "link"
    link.symlink_to(target)

    with pytest.raises(
        migration.V1CredentialMigrationError,
        match="^owner_gate_v1_credential_file_invalid$",
    ):
        migration._stable_file(
            link,
            uid=target.stat().st_uid,
            gid=target.stat().st_gid,
            mode=target.stat().st_mode & 0o777,
            maximum=1024,
            include_sha256=False,
        )

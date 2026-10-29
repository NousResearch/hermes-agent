"""A completed split-layout import must not replay an old approval."""

import json

import pytest

from gateway.pairing import PairingStore


@pytest.mark.parametrize("already_active", [False, True])
def test_migrated_approval_stays_revoked_after_store_reopen(tmp_path, monkeypatch, already_active):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    home = tmp_path / "profiles" / "sample"
    active = home / "pairing"
    alternate = home / "platforms" / "pairing"
    active.mkdir(parents=True)
    alternate.mkdir(parents=True)
    users = {"900000001": {"user_name": "Revoked"}, "900000002": {"user_name": "Retained"}}
    (active / "telegram-approved.json").write_text(json.dumps(users if already_active else {}))
    (alternate / "telegram-approved.json").write_text(json.dumps(users))
    store = PairingStore(profile="sample")
    assert store.is_approved("telegram", "900000001")
    assert store.revoke("telegram", "900000001")
    assert not store.is_approved("telegram", "900000001")
    reopened = PairingStore(profile="sample")
    assert not reopened.is_approved("telegram", "900000001")
    assert not store.is_approved("telegram", "900000001")
    assert reopened.is_approved("telegram", "900000002")


def test_failed_migration_write_preserves_the_source(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    home = tmp_path / "profiles" / "sample"
    active = home / "pairing"
    alternate = home / "platforms" / "pairing"
    active.mkdir(parents=True)
    (active / "telegram-approved.json").write_text("{}")
    alternate.mkdir(parents=True)
    source = alternate / "telegram-approved.json"
    original = '{"900000001": {"user_name": "Retained"}}'
    source.write_text(original)

    def fail_save(*args, **kwargs):
        raise OSError("fixture write failure")

    monkeypatch.setattr("gateway.pairing._save_json_file", fail_save)
    with pytest.raises(OSError, match="fixture write failure"):
        PairingStore(profile="sample")
    assert source.read_text() == original

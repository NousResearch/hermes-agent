"""Redacted profile migration: dry-run, apply, rollback, crash restore.
All roots are temp directories with synthetic legacy entries."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from agent.credential_pool import CredentialPool, PooledCredential
from agent.oauth_broker.migration import (
    GROUP_ORDER,
    MigrationError,
    apply_migration,
    plan_migration,
    rollback_migration,
    save_snapshot,
)

PORT = 17880
GROUPS = {"p1": "A", "p2": "B", "p3": "C"}


def _legacy_entry(idx, profile):
    return {
        "id": f"legacy-{profile}-{idx}",
        "label": f"legacy-{profile}-{idx}",
        "auth_type": "oauth",
        "priority": idx,
        "source": "device_code" if idx == 0 else f"manual:clone{idx}",
        "access_token": f"synthetic-legacy-access-{profile}-{idx}",
        "refresh_token": f"synthetic-legacy-refresh-{profile}-{idx}",
        "base_url": "https://chatgpt.com/backend-api/codex",
    }


def _make_root(tmp_path, profiles=("p1", "p2", "p3")):
    root = tmp_path / "hermes-root"
    for name in profiles:
        profile_dir = root / "profiles" / name
        profile_dir.mkdir(parents=True)
        store = {
            "version": 1,
            "credential_pool": {
                "openai-codex": [_legacy_entry(i, name) for i in range(3)]
            },
        }
        (profile_dir / "auth.json").write_text(json.dumps(store, indent=2))
    return root


def _add_default_profile(root):
    store = {
        "version": 1,
        "credential_pool": {
            "openai-codex": [_legacy_entry(i, "default") for i in range(3)]
        },
    }
    (root / "auth.json").write_text(json.dumps(store, indent=2))


def _auth_path(root, name):
    if name == "default":
        return root / "auth.json"
    return root / "profiles" / name / "auth.json"


def _pool(root, name):
    payload = json.loads(_auth_path(root, name).read_text())
    return payload["credential_pool"]["openai-codex"]


def _sha(root, name):
    raw = _auth_path(root, name).read_bytes()
    return hashlib.sha256(raw).hexdigest()


# ---------------------------------------------------------------------------
# Dry-run planning and the redacted snapshot
# ---------------------------------------------------------------------------


def test_plan_builds_redacted_snapshot_without_writing(tmp_path):
    root = _make_root(tmp_path)
    before = {name: _sha(root, name) for name in GROUPS}
    snapshot = plan_migration(root, GROUPS, port=PORT)
    assert snapshot["mode"] == "dry-run"
    assert snapshot["port"] == PORT
    assert snapshot["group_counts"] == {"A": 1, "B": 1, "C": 1}
    assert set(snapshot["profiles"]) == {"p1", "p2", "p3"}
    p2 = snapshot["profiles"]["p2"]
    assert p2["group"] == "B"
    assert p2["auth_sha256"] == before["p2"]
    assert p2["added_entry_ids"] == ["broker-B", "broker-C", "broker-A"]
    for legacy in p2["legacy"]:
        # Only keys actually present on the entry are captured (exact-state
        # snapshots); the fixtures carry these four.
        assert set(legacy) == {"id", "label", "source", "priority"}
    # Dry run never mutates any profile file.
    assert {name: _sha(root, name) for name in GROUPS} == before


def test_default_root_profile_is_in_exact_set_and_round_trips(tmp_path):
    root = _make_root(tmp_path, profiles=("p1",))
    _add_default_profile(root)
    groups = {"default": "A", "p1": "B"}
    originals = {name: _auth_path(root, name).read_bytes() for name in groups}

    snapshot = plan_migration(root, groups, port=PORT)
    assert set(snapshot["profiles"]) == {"default", "p1"}
    assert snapshot["group_counts"] == {"A": 1, "B": 1, "C": 0}

    apply_migration(root, snapshot)
    assert _pool(root, "default")[0]["id"] == "broker-A"
    assert _pool(root, "p1")[0]["id"] == "broker-B"

    rollback_migration(root, snapshot)
    assert {name: _auth_path(root, name).read_bytes() for name in groups} == originals


def test_snapshot_serialization_contains_no_secrets(tmp_path):
    root = _make_root(tmp_path)
    snapshot = plan_migration(root, GROUPS, port=PORT)
    path = tmp_path / "snapshot.json"
    save_snapshot(snapshot, path)
    text = path.read_text()
    assert "synthetic-legacy-access" not in text
    assert "synthetic-legacy-refresh" not in text
    assert '"access_token"' not in text
    assert '"refresh_token"' not in text


def test_plan_rejects_group_profile_set_mismatch(tmp_path):
    root = _make_root(tmp_path)
    with pytest.raises(MigrationError):
        plan_migration(root, {**GROUPS, "ghost": "A"}, port=PORT)
    with pytest.raises(MigrationError):
        plan_migration(root, {"p1": "A", "p2": "B"}, port=PORT)  # missing p3
    with pytest.raises(MigrationError):
        plan_migration(root, {**GROUPS, "p3": "D"}, port=PORT)  # bad group


def test_plan_rejects_malformed_group_and_pool_entry_with_migration_error(tmp_path):
    root = _make_root(tmp_path)
    with pytest.raises(MigrationError):
        plan_migration(root, {**GROUPS, "p3": ["C"]}, port=PORT)

    path = root / "profiles" / "p2" / "auth.json"
    payload = json.loads(path.read_text())
    payload["credential_pool"]["openai-codex"].append("not-an-object")
    path.write_text(json.dumps(payload))
    with pytest.raises(MigrationError):
        plan_migration(root, GROUPS, port=PORT)


def test_apply_rejects_malformed_snapshot_structure_as_migration_error(tmp_path):
    root = _make_root(tmp_path)
    snapshot = plan_migration(root, GROUPS, port=PORT)
    malformed = json.loads(json.dumps(snapshot))
    malformed["groups"]["p2"] = ["B"]
    with pytest.raises(MigrationError):
        apply_migration(root, malformed)


def test_snapshot_rejects_wrong_mode_counts_hash_and_secret_fields(tmp_path):
    root = _make_root(tmp_path)
    snapshot = plan_migration(root, GROUPS, port=PORT)

    invalid_variants = []
    wrong_mode = json.loads(json.dumps(snapshot))
    wrong_mode["mode"] = "apply"
    invalid_variants.append(wrong_mode)
    wrong_counts = json.loads(json.dumps(snapshot))
    wrong_counts["group_counts"]["A"] = 99
    invalid_variants.append(wrong_counts)
    bad_hash = json.loads(json.dumps(snapshot))
    bad_hash["profiles"]["p1"]["auth_sha256"] = "not-a-sha256"
    invalid_variants.append(bad_hash)
    secret_field = json.loads(json.dumps(snapshot))
    secret_field["profiles"]["p1"]["legacy"][0]["refresh_token"] = "synthetic"
    invalid_variants.append(secret_field)
    bad_priority = json.loads(json.dumps(snapshot))
    bad_priority["profiles"]["p1"]["legacy"][0]["priority"] = {
        "malformed": "accepted"
    }
    invalid_variants.append(bad_priority)
    bad_disabled = json.loads(json.dumps(snapshot))
    bad_disabled["profiles"]["p1"]["legacy"][0]["disabled"] = "true"
    invalid_variants.append(bad_disabled)

    for malformed in invalid_variants:
        with pytest.raises(MigrationError):
            apply_migration(root, malformed)

    target = tmp_path / "must-not-write-invalid-snapshot.json"
    with pytest.raises(MigrationError):
        save_snapshot(secret_field, target)
    assert not target.exists()


def _add_profile_after_snapshot(root, name="p4"):
    profile_dir = root / "profiles" / name
    profile_dir.mkdir(parents=True)
    store = {
        "version": 1,
        "credential_pool": {
            "openai-codex": [_legacy_entry(i, name) for i in range(3)]
        },
    }
    (profile_dir / "auth.json").write_text(json.dumps(store, indent=2))


def test_apply_and_rollback_reject_active_profile_set_drift(tmp_path):
    apply_root = _make_root(tmp_path / "apply")
    apply_snapshot = plan_migration(apply_root, GROUPS, port=PORT)
    _add_profile_after_snapshot(apply_root)
    with pytest.raises(MigrationError):
        apply_migration(apply_root, apply_snapshot)

    rollback_root = _make_root(tmp_path / "rollback")
    rollback_snapshot = plan_migration(rollback_root, GROUPS, port=PORT)
    apply_migration(rollback_root, rollback_snapshot)
    _add_profile_after_snapshot(rollback_root)
    with pytest.raises(MigrationError):
        rollback_migration(rollback_root, rollback_snapshot)


def test_plan_rejects_already_migrated_profile(tmp_path):
    root = _make_root(tmp_path)
    pool = _pool(root, "p1")
    pool.append({"id": "broker-A", "source": "keychain_reference", "priority": 9})
    path = root / "profiles" / "p1" / "auth.json"
    payload = json.loads(path.read_text())
    payload["credential_pool"]["openai-codex"] = pool
    path.write_text(json.dumps(payload))
    with pytest.raises(MigrationError):
        plan_migration(root, GROUPS, port=PORT)


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------


def test_apply_adds_cyclic_refs_and_disables_legacy_without_deleting(tmp_path):
    root = _make_root(tmp_path)
    snapshot = plan_migration(root, GROUPS, port=PORT)
    report = apply_migration(root, snapshot)
    assert report["applied"] is True
    assert report["written"] == ["p1", "p2", "p3"]

    for name, first_alias in (("p1", "A"), ("p2", "B"), ("p3", "C")):
        pool = _pool(root, name)
        brokers = [e for e in pool if e["source"] == "keychain_reference"]
        legacy = [e for e in pool if e["source"] != "keychain_reference"]
        order = GROUP_ORDER[first_alias]
        assert [e["id"] for e in sorted(brokers, key=lambda e: e["priority"])] == [
            f"broker-{alias}" for alias in order
        ]
        for entry in brokers:
            alias = entry["id"].removeprefix("broker-")
            assert entry["auth_type"] == "api_key"
            assert entry["base_url"] == (
                f"http://127.0.0.1:{PORT}/accounts/{alias}/backend-api/codex"
            )
            assert entry["secret_source"] == (
                "keychain://ai.hermes.oauth-broker.client/local"
            )
            assert "access_token" not in entry
            assert "refresh_token" not in entry
        assert [e["priority"] for e in legacy] == [3, 4, 5]
        for idx, entry in enumerate(legacy):
            assert entry["disabled"] is True
            # Archived, not deleted: original secret material stays in place.
            assert entry["access_token"] == (
                f"synthetic-legacy-access-{name}-{idx}"
            )
            assert entry["refresh_token"] == (
                f"synthetic-legacy-refresh-{name}-{idx}"
            )


def test_apply_aborts_on_hash_drift_without_writing_anything(tmp_path):
    root = _make_root(tmp_path)
    snapshot = plan_migration(root, GROUPS, port=PORT)
    drift_path = root / "profiles" / "p2" / "auth.json"
    payload = json.loads(drift_path.read_text())
    payload["credential_pool"]["openai-codex"][0]["priority"] = 42
    drift_path.write_text(json.dumps(payload, indent=2))
    before = {name: _sha(root, name) for name in GROUPS}
    with pytest.raises(MigrationError):
        apply_migration(root, snapshot)
    assert {name: _sha(root, name) for name in GROUPS} == before


def test_apply_write_failure_auto_restores_already_written_profiles(
    tmp_path, monkeypatch
):
    import agent.oauth_broker.migration as migration_mod

    root = _make_root(tmp_path)
    original = {name: _pool(root, name) for name in GROUPS}
    snapshot = plan_migration(root, GROUPS, port=PORT)

    real_write = migration_mod._write_profile_auth
    calls = []

    def flaky_write(path, store):
        calls.append(path)
        if len(calls) == 2:
            raise OSError("synthetic disk failure")
        real_write(path, store)

    monkeypatch.setattr(migration_mod, "_write_profile_auth", flaky_write)
    with pytest.raises(MigrationError):
        apply_migration(root, snapshot)
    # p1 was written then auto-restored from the snapshot; p2/p3 untouched.
    for name in GROUPS:
        assert _pool(root, name) == original[name]


def test_apply_keyboard_interrupt_restores_all_profiles_and_removes_journal(
    tmp_path, monkeypatch
):
    import agent.oauth_broker.migration as migration_mod

    root = _make_root(tmp_path)
    originals = {
        name: _auth_path(root, name).read_bytes() for name in GROUPS
    }
    snapshot = plan_migration(root, GROUPS, port=PORT)
    journal = tmp_path / "apply.journal"
    real_write = migration_mod._write_profile_auth
    calls = []

    def interrupted_write(path, store):
        calls.append(path)
        if len(calls) == 2:
            raise KeyboardInterrupt
        real_write(path, store)

    monkeypatch.setattr(migration_mod, "_write_profile_auth", interrupted_write)
    with pytest.raises(KeyboardInterrupt):
        apply_migration(root, snapshot, journal_path=journal)

    assert not journal.exists()
    assert {
        name: _auth_path(root, name).read_bytes() for name in GROUPS
    } == originals


def test_keyboard_interrupt_during_staging_write_removes_secret_temp_file(
    tmp_path, monkeypatch
):
    import agent.oauth_broker.migration as migration_mod

    root = _make_root(tmp_path, profiles=("p1",))
    auth_path = _auth_path(root, "p1")
    original = auth_path.read_bytes()
    real_write = migration_mod.os.write
    writes = 0

    def write_then_interrupt(fd, data):
        nonlocal writes
        writes += 1
        if writes == 1:
            chunk = data[: max(1, len(data) // 2)]
            return real_write(fd, chunk)
        real_write(fd, data)
        raise KeyboardInterrupt

    monkeypatch.setattr(migration_mod.os, "write", write_then_interrupt)
    sensitive = b'{"access_token":"synthetic-secret-staging"}'
    with pytest.raises(KeyboardInterrupt):
        migration_mod._durable_write_bytes(auth_path, sensitive)

    assert auth_path.read_bytes() == original
    assert list(auth_path.parent.glob(f".{auth_path.name}.*.tmp")) == []


def test_apply_recovers_after_uncatchable_process_exit(tmp_path):
    root = _make_root(tmp_path)
    snapshot = plan_migration(root, GROUPS, port=PORT)
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot))
    journal = tmp_path / "apply.journal"
    repo_root = Path(__file__).resolve().parents[3]
    script = f"""
import json
import os
from pathlib import Path
import agent.oauth_broker.migration as migration

root = Path({str(root)!r})
snapshot = json.loads(Path({str(snapshot_path)!r}).read_text())
journal = Path({str(journal)!r})
real_write = migration._write_profile_auth
calls = 0

def crash_after_first_replace(path, store):
    global calls
    real_write(path, store)
    calls += 1
    if calls == 1:
        os._exit(77)

migration._write_profile_auth = crash_after_first_replace
migration.apply_migration(root, snapshot, journal_path=journal)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        check=False,
    )
    assert result.returncode == 77
    assert journal.is_file()
    assert _pool(root, "p1")[0]["id"] == "broker-A"
    assert _pool(root, "p2")[0]["id"] == "legacy-p2-0"

    report = apply_migration(root, snapshot, journal_path=journal)

    assert report["written"] == ["p1", "p2", "p3"]
    assert not journal.exists()
    assert _pool(root, "p1")[0]["id"] == "broker-A"
    assert _pool(root, "p2")[0]["id"] == "broker-B"
    assert _pool(root, "p3")[0]["id"] == "broker-C"


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------


def test_rollback_restores_priorities_and_reenables_legacy(tmp_path):
    root = _make_root(tmp_path)
    original = {name: _pool(root, name) for name in GROUPS}
    snapshot = plan_migration(root, GROUPS, port=PORT)
    apply_migration(root, snapshot)
    report = rollback_migration(root, snapshot)
    assert report["restored"] == ["p1", "p2", "p3"]
    for name in GROUPS:
        assert _pool(root, name) == original[name]


def test_rollback_recovers_after_uncatchable_process_exit(tmp_path):
    root = _make_root(tmp_path)
    original = {name: _pool(root, name) for name in GROUPS}
    snapshot = plan_migration(root, GROUPS, port=PORT)
    apply_migration(root, snapshot)
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot))
    journal = tmp_path / "rollback.journal"
    repo_root = Path(__file__).resolve().parents[3]
    script = f"""
import json
import os
from pathlib import Path
import agent.oauth_broker.migration as migration

root = Path({str(root)!r})
snapshot = json.loads(Path({str(snapshot_path)!r}).read_text())
journal = Path({str(journal)!r})
real_write = migration._write_profile_auth
calls = 0

def crash_after_first_replace(path, store):
    global calls
    real_write(path, store)
    calls += 1
    if calls == 1:
        os._exit(78)

migration._write_profile_auth = crash_after_first_replace
migration.rollback_migration(root, snapshot, journal_path=journal)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        check=False,
    )
    assert result.returncode == 78
    assert journal.is_file()
    assert _pool(root, "p1")[0]["id"] == "legacy-p1-0"
    assert _pool(root, "p2")[0]["id"] == "broker-B"

    report = rollback_migration(root, snapshot, journal_path=journal)

    assert report["restored"] == ["p1", "p2", "p3"]
    assert not journal.exists()
    for name in GROUPS:
        assert _pool(root, name) == original[name]


# ---------------------------------------------------------------------------
# Security repair checkpoint: exact-state snapshots, preflights, durability
# ---------------------------------------------------------------------------


def test_snapshot_and_restore_preserve_exact_disabled_and_priority_state(
    tmp_path,
):
    root = _make_root(tmp_path, profiles=("p1",))
    path = root / "profiles" / "p1" / "auth.json"
    payload = json.loads(path.read_text())
    entries = payload["credential_pool"]["openai-codex"]
    entries[0]["disabled"] = True  # already disabled before migration
    del entries[1]["priority"]  # entry lacking a priority key entirely
    path.write_text(json.dumps(payload, indent=2))
    original = _pool(root, "p1")

    snapshot = plan_migration(root, {"p1": "A"}, port=PORT)
    legacy = {e["id"]: e for e in snapshot["profiles"]["p1"]["legacy"]}
    assert legacy["legacy-p1-0"]["disabled"] is True
    assert "priority" not in legacy["legacy-p1-1"]

    apply_migration(root, snapshot)
    rollback_migration(root, snapshot)
    assert _pool(root, "p1") == original  # exact key presence and values


def test_plan_rejects_missing_and_duplicate_legacy_ids(tmp_path):
    root = _make_root(tmp_path, profiles=("p1",))
    path = root / "profiles" / "p1" / "auth.json"
    payload = json.loads(path.read_text())
    del payload["credential_pool"]["openai-codex"][0]["id"]
    path.write_text(json.dumps(payload))
    with pytest.raises(MigrationError):
        plan_migration(root, {"p1": "A"}, port=PORT)

    payload = json.loads(path.read_text())
    payload["credential_pool"]["openai-codex"][0]["id"] = "legacy-p1-0"
    payload["credential_pool"]["openai-codex"][1]["id"] = "legacy-p1-0"
    path.write_text(json.dumps(payload))
    with pytest.raises(MigrationError):
        plan_migration(root, {"p1": "A"}, port=PORT)


def test_rollback_preflights_broker_identity_before_any_write(tmp_path):
    root = _make_root(tmp_path)
    snapshot = plan_migration(root, GROUPS, port=PORT)
    apply_migration(root, snapshot)
    tampered = root / "profiles" / "p2" / "auth.json"
    payload = json.loads(tampered.read_text())
    for entry in payload["credential_pool"]["openai-codex"]:
        if entry["id"] == "broker-B":
            entry["base_url"] = "http://127.0.0.1:9999/accounts/B/backend-api/codex"
    tampered.write_text(json.dumps(payload, indent=2))
    before = {name: _sha(root, name) for name in GROUPS}
    with pytest.raises(MigrationError):
        rollback_migration(root, snapshot)
    assert {name: _sha(root, name) for name in GROUPS} == before


def test_rollback_write_failure_reapplies_migrated_state(tmp_path, monkeypatch):
    import agent.oauth_broker.migration as migration_mod

    root = _make_root(tmp_path)
    snapshot = plan_migration(root, GROUPS, port=PORT)
    apply_migration(root, snapshot)
    migrated = {name: _pool(root, name) for name in GROUPS}

    real_write = migration_mod._write_profile_auth
    calls = []

    def flaky_write(path, store):
        calls.append(path)
        if len(calls) == 2:
            raise OSError("synthetic disk failure")
        real_write(path, store)

    monkeypatch.setattr(migration_mod, "_write_profile_auth", flaky_write)
    with pytest.raises(MigrationError):
        rollback_migration(root, snapshot)
    monkeypatch.setattr(migration_mod, "_write_profile_auth", real_write)
    # Consistent end state: every profile is back in the migrated shape.
    for name in GROUPS:
        assert _pool(root, name) == migrated[name]


def test_symlinked_profile_dir_or_auth_json_rejected(tmp_path):
    root = _make_root(tmp_path, profiles=("p1",))
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "auth.json").write_text(json.dumps({"credential_pool": {}}))
    import os as _os

    _os.symlink(outside, root / "profiles" / "evil")
    with pytest.raises(MigrationError):
        plan_migration(root, {"p1": "A", "evil": "B"}, port=PORT)
    (root / "profiles" / "evil").unlink()

    real = root / "profiles" / "p1" / "auth.json"
    moved = root / "profiles" / "p1" / "auth-real.json"
    real.rename(moved)
    _os.symlink(moved, real)
    with pytest.raises(MigrationError):
        plan_migration(root, {"p1": "A"}, port=PORT)


def test_profile_names_escaping_root_are_rejected(tmp_path):
    root = _make_root(tmp_path, profiles=("p1",))
    snapshot = plan_migration(root, {"p1": "A"}, port=PORT)
    apply_migration(root, snapshot)
    evil = json.loads(json.dumps(snapshot))
    evil["profiles"]["../escape"] = evil["profiles"]["p1"]
    with pytest.raises(MigrationError):
        rollback_migration(root, evil)


@pytest.mark.parametrize("port", [0, -1, 65536, 70000])
def test_port_range_is_validated(tmp_path, port):
    root = _make_root(tmp_path, profiles=("p1",))
    with pytest.raises(MigrationError):
        plan_migration(root, {"p1": "A"}, port=port)


def test_durable_write_fsyncs_and_cleans_temp_on_failure(tmp_path, monkeypatch):
    import os as _os

    import agent.oauth_broker.migration as migration_mod

    fsyncs = []
    real_fsync = _os.fsync
    monkeypatch.setattr(
        migration_mod.os, "fsync", lambda fd: fsyncs.append(fd) or real_fsync(fd)
    )
    target = tmp_path / "auth.json"
    migration_mod._durable_write_bytes(target, b"synthetic-content")
    assert target.read_bytes() == b"synthetic-content"
    assert len(fsyncs) >= 2  # file fd and parent directory fd
    assert list(tmp_path.glob("*.tmp")) == []

    def boom(src, dst):
        raise OSError("synthetic replace failure")

    monkeypatch.setattr(migration_mod.os, "replace", boom)
    with pytest.raises(OSError):
        migration_mod._durable_write_bytes(target, b"new-content")
    assert target.read_bytes() == b"synthetic-content"  # original intact
    assert list(tmp_path.glob("*.tmp")) == []  # staging file cleaned up


def test_durable_write_handles_short_writes(tmp_path, monkeypatch):
    import agent.oauth_broker.migration as migration_mod

    real_write = migration_mod.os.write

    def short_write(fd, data):
        chunk_size = max(1, len(data) // 2)
        return real_write(fd, data[:chunk_size])

    monkeypatch.setattr(migration_mod.os, "write", short_write)
    target = tmp_path / "auth.json"
    payload = b"synthetic-content-long-enough-for-many-short-writes"
    migration_mod._durable_write_bytes(target, payload)
    assert target.read_bytes() == payload


def test_durable_write_does_not_follow_planted_fixed_temp_symlink(tmp_path):
    import os as _os

    import agent.oauth_broker.migration as migration_mod

    target = tmp_path / "auth.json"
    victim = tmp_path / "victim.txt"
    victim.write_bytes(b"victim-must-stay-unchanged")
    _os.symlink(victim, target.with_name(target.name + ".tmp"))

    migration_mod._durable_write_bytes(target, b"new-auth-content")

    assert victim.read_bytes() == b"victim-must-stay-unchanged"
    assert target.read_bytes() == b"new-auth-content"
    assert not target.is_symlink()


def test_apply_failure_after_replace_restores_current_and_all_profiles(
    tmp_path, monkeypatch
):
    import agent.oauth_broker.migration as migration_mod

    root = _make_root(tmp_path)
    originals = {
        name: (root / "profiles" / name / "auth.json").read_bytes()
        for name in GROUPS
    }
    snapshot = plan_migration(root, GROUPS, port=PORT)
    real_write = migration_mod._write_profile_auth
    raised = False

    def write_then_raise(path, store):
        nonlocal raised
        real_write(path, store)
        if path.parent.name == "p2" and not raised:
            raised = True
            raise OSError("synthetic failure after replace")

    monkeypatch.setattr(migration_mod, "_write_profile_auth", write_then_raise)
    with pytest.raises(MigrationError):
        apply_migration(root, snapshot)

    for name in GROUPS:
        assert (root / "profiles" / name / "auth.json").read_bytes() == originals[name]


def test_rollback_failure_after_replace_reapplies_current_and_all_profiles(
    tmp_path, monkeypatch
):
    import agent.oauth_broker.migration as migration_mod

    root = _make_root(tmp_path)
    snapshot = plan_migration(root, GROUPS, port=PORT)
    apply_migration(root, snapshot)
    migrated = {
        name: (root / "profiles" / name / "auth.json").read_bytes()
        for name in GROUPS
    }
    real_write = migration_mod._write_profile_auth
    raised = False

    def write_then_raise(path, store):
        nonlocal raised
        real_write(path, store)
        if path.parent.name == "p2" and not raised:
            raised = True
            raise OSError("synthetic failure after replace")

    monkeypatch.setattr(migration_mod, "_write_profile_auth", write_then_raise)
    with pytest.raises(MigrationError):
        rollback_migration(root, snapshot)

    for name in GROUPS:
        assert (root / "profiles" / name / "auth.json").read_bytes() == migrated[name]


def test_snapshot_schema_and_legacy_set_are_validated_before_rollback(tmp_path):
    root = _make_root(tmp_path)
    snapshot = plan_migration(root, GROUPS, port=PORT)
    assert snapshot["snapshot_schema_version"] == 2
    apply_migration(root, snapshot)

    bad_schema = json.loads(json.dumps(snapshot))
    bad_schema["snapshot_schema_version"] = 999
    with pytest.raises(MigrationError):
        rollback_migration(root, bad_schema)

    path = root / "profiles" / "p2" / "auth.json"
    payload = json.loads(path.read_text())
    payload["credential_pool"]["openai-codex"].append(_legacy_entry(99, "p2"))
    path.write_text(json.dumps(payload, indent=2))
    with pytest.raises(MigrationError):
        rollback_migration(root, snapshot)


def test_snapshot_rejects_boolean_group_counts(tmp_path):
    root = _make_root(tmp_path)
    snapshot = plan_migration(root, GROUPS, port=PORT)
    snapshot["group_counts"] = {"A": True, "B": True, "C": True}

    with pytest.raises(MigrationError, match="group counts"):
        apply_migration(root, snapshot)


def test_symlinked_profiles_base_and_unsafe_profile_names_are_rejected(tmp_path):
    import os as _os

    outside_root = _make_root(tmp_path / "outside", profiles=("p1",))
    root = tmp_path / "linked-root"
    root.mkdir()
    _os.symlink(outside_root / "profiles", root / "profiles")
    with pytest.raises(MigrationError):
        plan_migration(root, {"p1": "A"}, port=PORT)

    unsafe_root = _make_root(tmp_path / "unsafe", profiles=("bad\nname",))
    with pytest.raises(MigrationError):
        plan_migration(unsafe_root, {"bad\nname": "A"}, port=PORT)


def test_apply_rejects_symlinked_or_non_owner_only_journal(tmp_path):
    import os as _os

    root = _make_root(tmp_path)
    snapshot = plan_migration(root, GROUPS, port=PORT)
    before = {name: _sha(root, name) for name in GROUPS}
    victim = tmp_path / "victim"
    victim.write_text("victim-must-not-change")
    symlink_journal = tmp_path / "apply-symlink.journal"
    _os.symlink(victim, symlink_journal)

    with pytest.raises(MigrationError):
        apply_migration(root, snapshot, journal_path=symlink_journal)
    assert victim.read_text() == "victim-must-not-change"
    assert {name: _sha(root, name) for name in GROUPS} == before

    weak_journal = tmp_path / "apply-weak.journal"
    weak_journal.write_text("{}")
    weak_journal.chmod(0o644)
    with pytest.raises(MigrationError, match="permissions"):
        apply_migration(root, snapshot, journal_path=weak_journal)
    assert {name: _sha(root, name) for name in GROUPS} == before


def test_apply_rejects_journal_bound_to_another_snapshot(tmp_path):
    import agent.oauth_broker.migration as migration_mod

    root = _make_root(tmp_path)
    snapshot = plan_migration(root, GROUPS, port=PORT)
    other = json.loads(json.dumps(snapshot))
    other["groups"]["p1"] = "B"
    journal = tmp_path / "apply.journal"
    migration_mod._write_journal(
        journal,
        other,
        sorted(GROUPS),
        [],
        operation="apply",
    )
    before = {name: _sha(root, name) for name in GROUPS}

    with pytest.raises(MigrationError, match="does not match"):
        apply_migration(root, snapshot, journal_path=journal)

    assert {name: _sha(root, name) for name in GROUPS} == before


def test_rollback_accepts_known_secret_free_broker_runtime_metadata(tmp_path):
    root = _make_root(tmp_path)
    original = {name: _pool(root, name) for name in GROUPS}
    snapshot = plan_migration(root, GROUPS, port=PORT)
    apply_migration(root, snapshot)
    path = _auth_path(root, "p2")
    payload = json.loads(path.read_text())
    broker = payload["credential_pool"]["openai-codex"][0]
    broker.update(
        {
            "access_token": "",
            "refresh_token": None,
            "last_status": "ready",
            "last_status_at": 1234.5,
            "last_error_code": None,
            "last_error_reason": None,
            "last_error_message": None,
            "last_error_reset_at": None,
            "request_count": 7,
        }
    )
    path.write_text(json.dumps(payload, indent=2))

    rollback_migration(root, snapshot)

    for name in GROUPS:
        assert _pool(root, name) == original[name]


def test_rollback_rejects_secret_persisted_on_broker_entry(tmp_path):
    root = _make_root(tmp_path)
    snapshot = plan_migration(root, GROUPS, port=PORT)
    apply_migration(root, snapshot)
    path = _auth_path(root, "p2")
    payload = json.loads(path.read_text())
    payload["credential_pool"]["openai-codex"][0]["access_token"] = (
        "synthetic-must-not-persist"
    )
    path.write_text(json.dumps(payload, indent=2))

    with pytest.raises(MigrationError, match="persisted secret"):
        rollback_migration(root, snapshot)


def test_rollback_rejects_secret_bearing_third_state(tmp_path):
    root = _make_root(tmp_path)
    snapshot = plan_migration(root, GROUPS, port=PORT)
    apply_migration(root, snapshot)
    path = _auth_path(root, "p2")
    payload = json.loads(path.read_text())
    payload["credential_pool"]["openai-codex"][3]["access_token"] = (
        "synthetic-external-drift"
    )
    path.write_text(json.dumps(payload, indent=2))
    before = {name: _sha(root, name) for name in GROUPS}

    with pytest.raises(MigrationError, match="secret-bearing state drifted"):
        rollback_migration(root, snapshot)

    assert {name: _sha(root, name) for name in GROUPS} == before


# ---------------------------------------------------------------------------
# Credential pool honors the disabled flag
# ---------------------------------------------------------------------------


def _pooled(payload):
    return PooledCredential.from_dict("openai-codex", payload)


def test_pool_selection_skips_disabled_entries():
    disabled = _pooled(
        {
            "id": "legacy-1",
            "label": "legacy-1",
            "auth_type": "oauth",
            "priority": 0,
            "source": "manual:clone1",
            "access_token": "synthetic-legacy-access",
            "refresh_token": "synthetic-legacy-refresh",
            "disabled": True,
        }
    )
    enabled = _pooled(
        {
            "id": "manual-1",
            "label": "manual-1",
            "auth_type": "api_key",
            "priority": 1,
            "source": "manual",
            "access_token": "synthetic-manual-token",
        }
    )
    pool = CredentialPool("openai-codex", [disabled, enabled])
    selected = pool.select()
    assert selected is not None and selected.id == "manual-1"


def test_disabled_flag_round_trips_through_serialization():
    entry = _pooled(
        {
            "id": "legacy-1",
            "label": "legacy-1",
            "auth_type": "oauth",
            "priority": 0,
            "source": "manual:clone1",
            "access_token": "synthetic-legacy-access",
            "disabled": True,
        }
    )
    assert entry.to_dict()["disabled"] is True

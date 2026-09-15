"""Recovery obligations survive pruning their original omission manifests."""

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest


def _database(path, value="recovery", large=False):
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS evidence (value TEXT)")
        conn.execute("INSERT INTO evidence VALUES (?)", (value,))
        if large:
            conn.execute("INSERT INTO evidence VALUES (?)", ("x" * 32768,))


@pytest.fixture
def snapshots(tmp_path, monkeypatch):
    from hermes_cli import backup

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text("model: test\n")
    clock = [datetime(2026, 9, 12, tzinfo=timezone.utc)]

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock[0].astimezone(tz)

    monkeypatch.setattr(backup, "datetime", Clock)
    failed = set()
    safe_copy = backup._safe_copy_db

    def selective_copy(src, dst):
        return False if src.name in failed else safe_copy(src, dst)

    monkeypatch.setattr(backup, "_safe_copy_db", selective_copy)

    def capture(label="manual", keep=1, cap=None):
        clock[0] += timedelta(seconds=1)
        snap_id = backup.create_quick_snapshot(
            label=label, hermes_home=home, keep=keep, max_file_size=cap,
        )
        assert snap_id is not None
        return home / "state-snapshots" / snap_id

    return SimpleNamespace(home=home, backup=backup, capture=capture, failed=failed, clock=clock)


def _meta(directory):
    return json.loads((directory / "manifest.json").read_text())


@pytest.mark.parametrize("omission", ["failed", "oversized"])
@pytest.mark.parametrize("label", ["manual", "pre-update"])
def test_missing_database_obligation_outlives_omission_witness(snapshots, omission, label):
    s = snapshots
    _database(s.home / "state.db", large=True)  # B
    _database(s.home / "memory_store.db")  # C
    complete = s.capture(label)
    _database(s.home / "projects.db", "sole A recovery")
    cap = 16384 if omission == "oversized" else None
    s.failed.update({"state.db"} if omission == "failed" else set())
    recovery = s.capture(label, cap=cap)
    recovery_manifest = (recovery / "manifest.json").read_bytes()
    if omission == "oversized":
        _database(s.home / "projects.db", large=True)
    else:
        s.failed.add("projects.db")
    witness = s.capture(label, cap=cap)
    (s.home / "projects.db").unlink()
    latest = None
    for _ in range(5):
        latest = s.capture(label, cap=cap)
        assert recovery.exists(), "sole recovery copy was pruned after live A disappeared"
        assert set(_meta(latest)["recovery_required_dbs"]) == {"projects.db", "state.db"}
        current_key = "failed_dbs" if omission == "failed" else "oversized_skipped"
        assert _meta(latest)[current_key] == ["state.db"]
        s.backup.prune_quick_snapshots(keep=0, hermes_home=s.home)
        assert recovery.exists() and latest.exists()
    assert not witness.exists()
    assert {d for d in latest.parent.iterdir() if d.is_dir()} == {complete, recovery, latest}
    assert (recovery / "manifest.json").read_bytes() == recovery_manifest
    with sqlite3.connect(recovery / "projects.db") as conn:
        assert conn.execute("SELECT value FROM evidence").fetchall() == [("sole A recovery",)]


def test_verified_capture_clears_obligation_without_resurrecting_old_omissions(snapshots):
    s = snapshots
    _database(s.home / "state.db")
    _database(s.home / "projects.db")
    s.capture()
    s.failed.add("projects.db")
    failed = s.capture()
    assert "projects.db" in _meta(failed)["recovery_required_dbs"]
    s.failed.clear()
    cleared = s.capture(keep=10)
    assert _meta(cleared)["recovery_required_dbs"] == []
    (s.home / "projects.db").unlink()
    # Keep older omission evidence temporarily, so each new checkpoint must
    # supersede it rather than union it back into the outstanding set.
    for _ in range(3):
        latest = s.capture(keep=10)
        assert _meta(latest)["recovery_required_dbs"] == []
    s.backup.prune_quick_snapshots(keep=0, hermes_home=s.home)
    assert list(latest.parent.iterdir()) == [latest]
    s.backup.prune_quick_snapshots(keep=0, hermes_home=s.home)
    assert list(latest.parent.iterdir()) == [latest]


def test_namespace_does_not_borrow_other_namespace_recovery(snapshots):
    s = snapshots
    _database(s.home / "projects.db")
    manual_copy = s.capture("manual")
    s.failed.add("projects.db")
    manual_failed = s.capture("manual")
    s.failed.clear()
    updater = s.capture("pre-update")
    assert _meta(updater)["recovery_required_dbs"] == []
    (s.home / "projects.db").unlink()
    for _ in range(3):
        manual_latest = s.capture("manual")
        s.capture("pre-update")
    assert _meta(manual_latest)["recovery_required_dbs"] == ["projects.db"]
    assert manual_copy.exists()
    assert not manual_failed.exists()


def test_public_prune_respects_existing_backup_lock(snapshots):
    s = snapshots
    _database(s.home / "state.db")
    captured = s.capture()
    with s.backup._backup_operation_lock(s.home):
        with pytest.raises(s.backup.BackupInProgressError):
            s.backup.prune_quick_snapshots(keep=0, hermes_home=s.home)
    assert captured.exists()


def _legacy_generation(s, index, databases, omissions):
    directory = s.home / "state-snapshots" / f"20260911-00000{index}-legacy"
    directory.mkdir(parents=True)
    (directory / "config.yaml").write_text("model: legacy\n")
    for name in databases:
        _database(directory / name)
    files = {p.name: p.stat().st_size for p in directory.iterdir()}
    (directory / "manifest.json").write_text(json.dumps({
        "id": directory.name, "label": "legacy", "files": files,
        "failed_dbs": omissions, "oversized_skipped": [],
    }))
    return directory


def test_legacy_witnesses_survive_pruning_then_consolidate(snapshots):
    s = snapshots
    complete = _legacy_generation(s, 0, ["state.db", "memory_store.db"], [])
    recovery = _legacy_generation(s, 1, ["projects.db", "memory_store.db"], ["state.db"])
    witness = _legacy_generation(s, 2, ["memory_store.db"], ["projects.db", "state.db"])
    latest_legacy = _legacy_generation(s, 3, ["memory_store.db"], ["state.db"])
    originals = {d: (d / "manifest.json").read_bytes() for d in (complete, recovery, witness, latest_legacy)}
    for _ in range(3):
        s.backup.prune_quick_snapshots(keep=0, hermes_home=s.home)
        assert recovery.exists() and witness.exists()
    _database(s.home / "memory_store.db")
    latest = s.capture(keep=0)
    assert set(_meta(latest)["recovery_required_dbs"]) == {"projects.db", "state.db"}
    assert set(latest.parent.iterdir()) == {complete, recovery, latest}
    for d in (complete, recovery):
        assert (d / "manifest.json").read_bytes() == originals[d]


def test_empty_checkpoint_survives_if_older_retained_generation_has_obligations(snapshots):
    s = snapshots
    _database(s.home / "projects.db")
    _database(s.home / "state.db")
    s.capture(keep=10)
    s.failed.add("projects.db")
    s.capture(keep=10)
    (s.home / "projects.db").unlink()
    s.failed.clear()
    old_complete = s.capture(keep=10)
    assert _meta(old_complete)["recovery_required_dbs"] == ["projects.db"]
    _database(s.home / "projects.db", "recaptured")
    cleared = s.capture(keep=10)
    assert _meta(cleared)["recovery_required_dbs"] == []
    # The clearing checkpoint remains readable, but its generation no longer
    # qualifies as complete. keep=0 must not fall back to the old obligation.
    (cleared / "config.yaml").unlink()
    for _ in range(3):
        s.backup.prune_quick_snapshots(keep=0, hermes_home=s.home)
        assert set(cleared.parent.iterdir()) == {old_complete, cleared}
    (s.home / "projects.db").unlink()
    latest = s.capture(keep=0)
    assert _meta(latest)["recovery_required_dbs"] == []
    assert list(latest.parent.iterdir()) == [latest]


def test_newer_legacy_clear_is_not_lost_when_checkpoint_is_retained(snapshots):
    s = snapshots
    prior = _legacy_generation(s, 0, ["state.db"], ["projects.db"])
    meta = _meta(prior)
    meta["recovery_required_dbs"] = ["projects.db"]
    (prior / "manifest.json").write_text(json.dumps(meta))
    cleared = _legacy_generation(s, 1, ["projects.db"], ["state.db"])
    for _ in range(3):
        s.backup.prune_quick_snapshots(keep=0, hermes_home=s.home)
        assert prior.exists() and cleared.exists()
    latest = s.capture(keep=0)
    assert _meta(latest)["recovery_required_dbs"] == ["state.db"]
    assert set(latest.parent.iterdir()) == {prior, latest}


@pytest.mark.parametrize("invalid", ["projects.db", [17], ["../projects.db"]])
def test_malformed_checkpoint_falls_back_without_clearing(snapshots, invalid):
    s = snapshots
    _database(s.home / "projects.db")
    recovery = s.capture(keep=10)
    s.failed.add("projects.db")
    s.capture(keep=10)
    (s.home / "projects.db").unlink()
    damaged = s.capture(keep=10)
    meta = _meta(damaged)
    meta["recovery_required_dbs"] = invalid
    (damaged / "manifest.json").write_text(json.dumps(meta))
    original = (damaged / "manifest.json").read_bytes()
    latest = s.capture(keep=10)
    assert _meta(latest)["recovery_required_dbs"] == ["projects.db"]
    assert recovery.exists()
    assert (damaged / "manifest.json").read_bytes() == original


def test_unverified_or_unlisted_capture_does_not_clear_obligation(snapshots, monkeypatch):
    s = snapshots
    _database(s.home / "projects.db")
    recovery = s.capture(keep=10)
    s.failed.add("projects.db")
    s.capture(keep=10)
    s.failed.clear()

    def corrupt_copy(_src, dst):
        dst.write_bytes(b"not a SQLite database")
        return True

    monkeypatch.setattr(s.backup, "_safe_copy_db", corrupt_copy)
    corrupt = s.capture(keep=10)
    assert "projects.db" in _meta(corrupt)["files"]
    assert _meta(corrupt)["recovery_required_dbs"] == ["projects.db"]
    # A physically readable but unlisted payload cannot clear a legacy omission.
    unlisted = _legacy_generation(s, 4, [], ["projects.db"])
    _database(unlisted / "projects.db")
    # Exercise legacy-only reconstruction independently of the newer checkpoint.
    missing, _ = s.backup._snapshot_recovery_state([unlisted], {unlisted: _meta(unlisted)})
    assert missing == {"projects.db"}
    s.backup.prune_quick_snapshots(keep=0, hermes_home=s.home)
    assert recovery.exists()


def test_profile_obligations_are_not_inherited(snapshots, tmp_path):
    s = snapshots
    _database(s.home / "projects.db")
    s.failed.add("projects.db")
    s.capture()
    other_home = tmp_path / "other-profile"
    other_home.mkdir()
    (other_home / "config.yaml").write_text("model: other\n")
    snap_id = s.backup.create_quick_snapshot(hermes_home=other_home)
    assert snap_id is not None
    assert _meta(other_home / "state-snapshots" / snap_id)["recovery_required_dbs"] == []


def test_same_second_snapshot_keeps_newest_recovery_checkpoint(snapshots):
    s = snapshots
    _database(s.home / "projects.db")
    recovery = s.capture("z-first")
    s.failed.add("projects.db")
    s.clock[0] -= timedelta(seconds=1)
    omitted = s.capture("a-second")
    assert omitted.exists(), "newly published checkpoint was mistaken for an older snapshot"
    (s.home / "projects.db").unlink()
    for _ in range(3):
        latest = s.capture()
        assert _meta(latest)["recovery_required_dbs"] == ["projects.db"]
        assert recovery.exists()


def test_same_second_numeric_suffix_does_not_regress_checkpoint(snapshots):
    s = snapshots
    _database(s.home / "projects.db")
    first = s.capture("collision", keep=20)
    previous_sequence = _meta(first)["recovery_sequence"]
    for index in range(12):
        if index == 9:
            s.failed.add("projects.db")
        s.clock[0] -= timedelta(seconds=1)
        latest = s.capture("collision", keep=20)
        assert _meta(latest)["recovery_sequence"] > previous_sequence
        previous_sequence = _meta(latest)["recovery_sequence"]
    (s.home / "projects.db").unlink()
    for _ in range(3):
        latest = s.capture(keep=0)
        assert _meta(latest)["recovery_required_dbs"] == ["projects.db"]
        assert _meta(latest)["recovery_sequence"] > previous_sequence
        previous_sequence = _meta(latest)["recovery_sequence"]


@pytest.mark.parametrize("replacement", ["healthy", "size-mismatch", "null", "string", "float", "bool", "negative"])
@pytest.mark.parametrize("entrypoint", ["public-prune", "publication"])
def test_legacy_valid_db_must_match_manifest_before_clearing_obligation(snapshots, replacement, entrypoint):
    s = snapshots
    recovery = _legacy_generation(s, 0, ["projects.db"], ["state.db"])
    _legacy_generation(s, 1, [], ["projects.db", "state.db"])
    cleared = _legacy_generation(s, 2, ["projects.db"], ["state.db"])
    meta = _meta(cleared)
    size = meta["files"]["projects.db"]
    if replacement == "size-mismatch":
        # Still valid SQLite, but no longer the payload described by the manifest.
        _database(cleared / "projects.db", large=True)
    elif replacement != "healthy":
        meta["files"]["projects.db"] = {
            "null": None, "string": str(size), "float": float(size), "bool": True, "negative": -1,
        }[replacement]
        (cleared / "manifest.json").write_text(json.dumps(meta))
    manifest_bytes = (cleared / "manifest.json").read_bytes()
    integrity = s.backup.verify_sqlite_integrity(cleared / "projects.db")
    assert integrity["valid"]
    if replacement == "size-mismatch":
        assert integrity["size"] != size
    unverified = replacement != "healthy"
    expected = {"state.db", "projects.db"} if unverified else {"state.db"}
    if entrypoint == "public-prune":
        for _ in range(2):
            s.backup.prune_quick_snapshots(keep=0, hermes_home=s.home)
            assert recovery.exists() is unverified
        assert (cleared / "manifest.json").read_bytes() == manifest_bytes
    complete_meta = {**_meta(cleared), "failed_dbs": []}
    assert s.backup._is_complete_quick_snapshot(cleared, complete_meta) is (not unverified)
    # Publication must consolidate only size-bound clears, even when the live DB
    # has disappeared. Repeated keep=0 must not lose or resurrect obligations.
    for _ in range(2):
        latest = s.capture(keep=0)
        assert set(_meta(latest)["recovery_required_dbs"]) == expected
        assert recovery.exists() is unverified


@pytest.mark.parametrize("fully_cleared", [True, False])
@pytest.mark.parametrize("damage", ["missing", "corrupt", "size-mismatch"])
@pytest.mark.parametrize("entrypoint", ["public-prune", "publication"])
def test_damaged_clearing_checkpoint_restores_database_obligation(snapshots, damage, entrypoint, fully_cleared):
    s = snapshots
    _database(s.home / "state.db")
    complete = s.capture(keep=10)
    _database(s.home / "projects.db", "sole older recovery")
    s.failed.add("state.db")
    recovery = s.capture(keep=10)
    s.failed.add("projects.db")
    s.capture(keep=10)
    s.failed.remove("projects.db")
    if fully_cleared:
        s.failed.clear()
    cleared = s.capture(keep=10)
    assert _meta(cleared)["recovery_required_dbs"] == ([] if fully_cleared else ["state.db"])
    s.failed.add("state.db")
    payload = cleared / "projects.db"
    if damage == "missing":
        payload.unlink()
    elif damage == "corrupt":
        payload.write_bytes(b"x" * payload.stat().st_size)
    else:
        _database(payload, large=True)
    (s.home / "projects.db").unlink()
    original = (cleared / "manifest.json").read_bytes()
    if entrypoint == "public-prune":
        s.backup.prune_quick_snapshots(keep=0, hermes_home=s.home)
        assert recovery.exists(), "damaged checkpoint pruned the only valid database copy"
        assert (cleared / "manifest.json").read_bytes() == original
    for _ in range(3):
        latest = s.capture(keep=0)
        assert recovery.exists(), "empty inherited checkpoint pruned the only valid database copy"
        assert set(_meta(latest)["recovery_required_dbs"]) == {"state.db", "projects.db"}
        assert complete.exists()
        with sqlite3.connect(recovery / "projects.db") as conn:
            assert conn.execute("SELECT value FROM evidence").fetchall() == [("sole older recovery",)]

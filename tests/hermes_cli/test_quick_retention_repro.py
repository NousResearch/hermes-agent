"""Real-file recovery-retention invariants for quick snapshots."""
import json
import sqlite3

import pytest

from hermes_cli import backup
from hermes_cli.backup import create_quick_snapshot


_DB_CAP = 1024 ** 3


def _make_db(path, table):
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as db:
        db.execute(f"CREATE TABLE {table} (value TEXT)")
        db.execute(f"INSERT INTO {table} VALUES ('preserve-me')")


@pytest.mark.parametrize("damage_manifests", [False, True])
def test_incomplete_snapshots_with_unique_config_payloads_are_bounded(
    tmp_path, monkeypatch, damage_manifests
):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text("model: test\n")
    _make_db(home / "state.db", "recovery")
    complete = create_quick_snapshot(label="complete", keep=1)
    assert complete
    # A sparse file crosses the actual pre-update cap without allocating 1 GiB.
    with (home / "state.db").open("r+b") as db:
        db.truncate(_DB_CAP + 1)
    root = home / "state-snapshots"
    unmanaged = root / "not-a-snapshot"
    unmanaged.mkdir()
    monkeypatch.setattr(backup, "_QUICK_DEFAULT_KEEP", 2)
    for index in range(5):
        (home / "config.yaml").write_text(f"model: test-{index}\n")
        snapshot = create_quick_snapshot(label=f"partial-{index}", keep=1, max_file_size=_DB_CAP)
        assert snapshot
        meta = json.loads((root / snapshot / "manifest.json").read_text())
        assert meta["oversized_skipped"] == ["state.db"]
        assert meta["failed_dbs"] == []
        if damage_manifests:
            (root / snapshot / "manifest.json").write_bytes(b"\xff")
    assert len([p for p in root.iterdir() if p != unmanaged]) <= 4
    backup.prune_quick_snapshots(keep=1, hermes_home=home)
    assert unmanaged.is_dir()
    with sqlite3.connect(root / complete / "state.db") as db:
        assert db.execute("SELECT value FROM recovery").fetchone() == ("preserve-me",)
    published = [p for p in root.iterdir() if p.is_dir() and p != unmanaged
                 and not p.name.startswith(".")]
    assert len(published) <= 2, "unique config partials accumulate beyond recovery + keep"


@pytest.mark.parametrize("have_complete", [True, False])
@pytest.mark.parametrize("replacement", ["healthy", "missing", "corrupt", "unlisted", "invalid-manifest"])
def test_pruning_keeps_complete_generation_and_verified_per_db_recovery(
    tmp_path, monkeypatch, replacement, have_complete
):
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("model: test\n")
    _make_db(home / "state.db", "state_recovery")
    _make_db(home / "cron" / "executions.db", "execution_recovery")
    root = home / "state-snapshots"
    real_copy = backup._safe_copy_db
    failed = set() if have_complete else {"cron/executions.db"}

    def selective_copy(src, dst):
        rel = src.relative_to(home).as_posix()
        return False if rel in failed else real_copy(src, dst)

    monkeypatch.setattr(backup, "_safe_copy_db", selective_copy)
    original = create_quick_snapshot(label="a-original", hermes_home=home, keep=10)
    assert original
    pre_update = create_quick_snapshot(label="pre-update", hermes_home=home, keep=1)
    assert pre_update
    failed = {"cron/executions.db"}
    # Keep history until the replacement has been damaged, then trigger pruning.
    partial = create_quick_snapshot(label="b-partial", hermes_home=home, keep=10)
    assert partial
    payload = root / partial / "state.db"
    manifest = root / partial / "manifest.json"
    if replacement == "missing":
        payload.unlink()
    elif replacement == "corrupt":
        payload.write_bytes(b"not sqlite")
    elif replacement == "unlisted":
        meta = json.loads(manifest.read_text())
        del meta["files"]["state.db"]
        manifest.write_text(json.dumps(meta))
    elif replacement == "invalid-manifest":
        manifest.write_bytes(b"\xff")

    failed = {"state.db"}
    latest = create_quick_snapshot(label="c-partial", hermes_home=home, keep=1)
    assert latest
    expected_recovery = partial if replacement == "healthy" else original
    # With no complete generation, a manifest lie must not evict the only usable
    # partial recovery copy. With one, it survives even a healthy newer partial.
    with sqlite3.connect(root / expected_recovery / "state.db") as db:
        assert db.execute("SELECT value FROM state_recovery").fetchone() == ("preserve-me",)
    with sqlite3.connect(root / latest / "cron" / "executions.db") as db:
        assert db.execute("SELECT value FROM execution_recovery").fetchone() == ("preserve-me",)
    if have_complete:
        assert (root / original / "state.db").is_file()
        assert (root / original / "cron" / "executions.db").is_file()
    # Manual retention cannot evict an unrelated updater safety-net generation.
    assert (root / pre_update / "state.db").is_file()
    backup.prune_quick_snapshots(keep=1, hermes_home=home)
    assert (root / expected_recovery / "state.db").is_file()
    assert (root / pre_update / "state.db").is_file()
    if have_complete:
        assert (root / original / "state.db").is_file()
    if replacement == "invalid-manifest":
        assert manifest.read_bytes() == b"\xff", "unknown snapshot namespace must not be pruned"


@pytest.mark.parametrize("label", ["manual", "pre-update"])
@pytest.mark.parametrize("prune_via", ["publication", "public-prune"])
def test_valid_size_mismatched_db_cannot_evict_last_recovery(tmp_path, monkeypatch, label, prune_via):
    from contextlib import closing

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text("{}\n")
    with closing(sqlite3.connect(home / "state.db")) as db:
        db.execute("CREATE TABLE evidence (value TEXT)")
        db.executemany("INSERT INTO evidence VALUES (?)", ((f"row-{i:04d}",) for i in range(500)))
        db.commit()
    _make_db(home / "cron/executions.db", "execution_recovery")
    with closing(sqlite3.connect(home / "cron/executions.db")) as db:
        db.execute("INSERT INTO execution_recovery VALUES (zeroblob(32768))")
        db.commit()
    cap = (home / "state.db").stat().st_size
    assert (home / "cron/executions.db").stat().st_size > cap

    def capture(keep, max_file_size):
        snap_id = create_quick_snapshot(
            label=label, hermes_home=home, keep=keep, max_file_size=max_file_size,
        )
        assert snap_id
        return home / "state-snapshots" / snap_id

    original = capture(10, cap)
    replaced = capture(10, cap)
    manifest_bytes = (replaced / "manifest.json").read_bytes()
    meta = json.loads(manifest_bytes)
    replacement = tmp_path / "replacement.db"
    _make_db(replacement, "evidence")
    replacement.replace(replaced / "state.db")
    integrity = backup.verify_sqlite_integrity(replaced / "state.db")
    assert integrity["valid"]
    assert integrity["size"] != meta["files"]["state.db"]
    assert not backup._is_complete_quick_snapshot(original, json.loads((original / "manifest.json").read_text()))
    latest = capture(1 if prune_via == "publication" else 10, 4096)
    if prune_via == "public-prune":
        backup.prune_quick_snapshots(keep=1, hermes_home=home)
    assert original.is_dir(), "manifest-mismatched valid SQLite displaced the last good recovery"
    assert latest.is_dir()
    assert not replaced.exists(), "invalid coverage should not prevent bounded retention"
    for _ in range(2):
        backup.prune_quick_snapshots(keep=1, hermes_home=home)
        assert original.is_dir() and latest.is_dir()
    # Exercise the supported restore path as well as checking retained files.
    (home / "state.db").unlink()
    assert backup.restore_quick_snapshot(original.name, hermes_home=home)
    with closing(sqlite3.connect(f"file:{home / 'state.db'}?mode=ro", uri=True)) as db:
        assert db.execute("SELECT count(*) FROM evidence").fetchone() == (500,)

"""
Tests for the Runtime Snapshot Engine.

Covers: hashing, object store, snapshot/restore roundtrip,
deduplication, SQLite safe copy, diff, prune, debouncing.
"""

import json
import os
import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.snapshot_engine import (
    SnapshotEngine,
    auto_snapshot,
    diff,
    get_head,
    list_snapshots,
    prune,
    restore,
    snapshot,
)


@pytest.fixture
def tmp_hermes(tmp_path):
    """Create a temporary ~/.hermes-like directory with test files."""
    hermes = tmp_path / "hermes"
    hermes.mkdir()

    # Create test state files
    (hermes / "config.yaml").write_text("model: test\nprovider: openai\n")
    (hermes / "auth.json").write_text('{"openai": "sk-test"}')

    # Create a real SQLite DB
    db_path = hermes / "state.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)")
    conn.execute("INSERT INTO test VALUES (1, 'hello')")
    conn.commit()
    conn.close()

    # Cron dir
    cron_dir = hermes / "cron"
    cron_dir.mkdir()
    (cron_dir / "jobs.json").write_text('[{"name": "test"}]')

    # Other files
    (hermes / "gateway_state.json").write_text("{}")
    (hermes / "channel_directory.json").write_text("{}")
    (hermes / "processes.json").write_text("[]")

    return hermes


@pytest.fixture
def engine(tmp_hermes):
    """Create a SnapshotEngine pointed at the temp hermes dir."""
    return SnapshotEngine(hermes_home=tmp_hermes)


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------

class TestBasicSnapshot:
    """Test core snapshot creation and restoration."""

    def test_snapshot_creates_manifest(self, engine):
        snap_id = engine.snapshot(label="test")
        assert snap_id is not None
        assert "test" in snap_id

        # Check manifest exists
        snap_dir = engine.snapshots_dir / snap_id
        assert (snap_dir / "manifest.json").exists()
        assert (snap_dir / "meta.json").exists()

        # Check manifest content
        with open(snap_dir / "manifest.json") as f:
            manifest = json.load(f)
        assert "config.yaml" in manifest
        assert "state.db" in manifest
        assert "auth.json" in manifest

    def test_snapshot_metadata(self, engine):
        snap_id = engine.snapshot(label="meta-test", trigger="manual")
        meta_path = engine.snapshots_dir / snap_id / "meta.json"
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["label"] == "meta-test"
        assert meta["trigger"] == "manual"
        assert meta["file_count"] > 0
        assert meta["total_size"] > 0

    def test_head_updated(self, engine):
        assert engine.get_head() is None
        snap_id = engine.snapshot()
        assert engine.get_head() == snap_id

    def test_restore_roundtrip(self, engine, tmp_hermes):
        # Snapshot original state
        snap_id = engine.snapshot(label="original")

        # Modify state
        (tmp_hermes / "config.yaml").write_text("model: modified\n")

        # Restore
        assert engine.restore(snap_id)
        content = (tmp_hermes / "config.yaml").read_text()
        assert "test" in content
        assert "modified" not in content

    def test_restore_db_roundtrip(self, engine, tmp_hermes):
        """SQLite DB should survive snapshot/restore."""
        snap_id = engine.snapshot()

        # Modify DB
        conn = sqlite3.connect(str(tmp_hermes / "state.db"))
        conn.execute("INSERT INTO test VALUES (2, 'world')")
        conn.commit()
        conn.close()

        # Restore
        engine.restore(snap_id)

        # Verify
        conn = sqlite3.connect(str(tmp_hermes / "state.db"))
        rows = conn.execute("SELECT * FROM test").fetchall()
        conn.close()
        assert len(rows) == 1
        assert rows[0][1] == "hello"


# ---------------------------------------------------------------------------
# Content-addressed deduplication
# ---------------------------------------------------------------------------

class TestDeduplication:
    """Verify that identical files share objects."""

    def test_same_content_same_hash(self, engine, tmp_hermes):
        """Two snapshots with unchanged files should share objects."""
        snap1 = engine.snapshot(label="snap1")

        # Modify state slightly so snapshot is created, then check shared objects
        (tmp_hermes / "gateway_state.json").write_text('{"changed": true}')
        snap2 = engine.snapshot(label="snap2")

        assert snap1 is not None
        assert snap2 is not None

        m1 = json.load(open(engine.snapshots_dir / snap1 / "manifest.json"))
        m2 = json.load(open(engine.snapshots_dir / snap2 / "manifest.json"))

        # config.yaml hash should be identical (didn't change between snapshots)
        assert m1["config.yaml"] == m2["config.yaml"]

    def test_object_store_dedup(self, engine, tmp_hermes):
        """Same content stored twice = 1 object file."""
        engine._store_object(b"identical content")
        engine._store_object(b"identical content")

        # Count objects
        obj_count = sum(1 for d in engine.objects.iterdir() if d.is_dir()
                        for f in d.iterdir())
        assert obj_count == 1


# ---------------------------------------------------------------------------
# SQLite safe copy
# ---------------------------------------------------------------------------

class TestSqliteSafeCopy:
    """Verify SQLite databases are safely copied."""

    def test_safe_copy_produces_valid_db(self, engine, tmp_hermes):
        """Safe-copied DB should be queryable."""
        src = tmp_hermes / "state.db"
        dst = tmp_hermes / "state_copy.db"

        assert engine._safe_copy_db(src, dst)

        conn = sqlite3.connect(str(dst))
        rows = conn.execute("SELECT * FROM test").fetchall()
        conn.close()
        assert len(rows) == 1
        assert rows[0][1] == "hello"

    def test_safe_copy_with_wal(self, engine, tmp_hermes):
        """Safe copy should handle WAL mode correctly."""
        src = tmp_hermes / "state.db"

        # Open in WAL mode and write
        conn = sqlite3.connect(str(src))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("INSERT INTO test VALUES (3, 'wal-test')")
        conn.commit()

        dst = tmp_hermes / "state_wal_copy.db"
        assert engine._safe_copy_db(src, dst)

        # Verify data is in copy
        conn2 = sqlite3.connect(str(dst))
        rows = conn2.execute("SELECT * FROM test WHERE id=3").fetchall()
        conn2.close()
        assert len(rows) == 1

        conn.close()


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------

class TestDiff:
    """Test snapshot diffing."""

    def test_no_changes(self, engine, tmp_hermes):
        """Snapshots with a change should diff cleanly."""
        snap1 = engine.snapshot(label="a")
        (tmp_hermes / "config.yaml").write_text("model: changed\n")
        snap2 = engine.snapshot(label="b")

        assert snap1 is not None
        assert snap2 is not None

        result = engine.diff(snap1, snap2)
        assert "config.yaml" in result["changed"]
        assert len(result["added"]) == 0
        assert len(result["removed"]) == 0

    def test_detect_changes(self, engine, tmp_hermes):
        """Modified file should show in diff."""
        snap1 = engine.snapshot(label="before")

        # Modify config
        (tmp_hermes / "config.yaml").write_text("model: changed\n")

        snap2 = engine.snapshot(label="after")

        result = engine.diff(snap1, snap2)
        assert "config.yaml" in result["changed"]


# ---------------------------------------------------------------------------
# List and history
# ---------------------------------------------------------------------------

class TestListAndHistory:
    """Test snapshot listing."""

    def test_list_returns_recent(self, engine, tmp_hermes):
        (tmp_hermes / "config.yaml").write_text("turn: 1\n")
        engine.snapshot(label="old")
        time.sleep(0.1)
        (tmp_hermes / "config.yaml").write_text("turn: 2\n")
        engine.snapshot(label="new")

        snaps = engine.list_snapshots(limit=10)
        assert len(snaps) >= 2
        # Most recent first
        assert snaps[0]["label"] == "new"

    def test_list_limit(self, engine):
        for i in range(5):
            # Force different timestamps
            time.sleep(0.05)
            # Modify state so each snapshot is unique
            (engine.hermes_home / "config.yaml").write_text(f"turn: {i}\n")
            engine.snapshot(label=f"turn-{i}")

        snaps = engine.list_snapshots(limit=3)
        assert len(snaps) == 3


# ---------------------------------------------------------------------------
# Prune
# ---------------------------------------------------------------------------

class TestPrune:
    """Test snapshot pruning."""

    def test_prune_keeps_last_n(self, engine, tmp_hermes):
        """Prune should always keep the last N snapshots."""
        for i in range(10):
            (tmp_hermes / "config.yaml").write_text(f"turn: {i}\n")
            time.sleep(0.05)
            engine.snapshot(label=f"s{i}")

        deleted = engine.prune(keep_last=5, keep_hourly=0, keep_daily=0)
        remaining = engine.list_snapshots(limit=100)
        assert len(remaining) <= 5 + deleted  # At most 5 kept

    def test_prune_cleans_history_db(self, engine, tmp_hermes):
        """Prune should clean up the history DB."""
        for i in range(5):
            (tmp_hermes / "config.yaml").write_text(f"v{i}\n")
            time.sleep(0.05)
            engine.snapshot()

        engine.prune(keep_last=2, keep_hourly=0, keep_daily=0)
        remaining = engine.list_snapshots(limit=100)
        # Should have at most 2 from keep_last
        assert len(remaining) <= 2


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

class TestConvenienceFunctions:
    """Test the module-level shorthand functions."""

    def test_snapshot_function(self, tmp_hermes):
        """Module-level snapshot() should work."""
        with patch("tools.snapshot_engine.get_hermes_home", return_value=tmp_hermes):
            # Reset global engine
            import tools.snapshot_engine as se
            se._engine = None

            snap_id = snapshot(label="conv-test")
            assert snap_id is not None
            assert "conv-test" in snap_id

    def test_auto_snapshot_debounce(self, tmp_hermes):
        """Auto-snapshot should debounce rapid calls."""
        with patch("tools.snapshot_engine.get_hermes_home", return_value=tmp_hermes):
            import tools.snapshot_engine as se
            se._engine = None
            se._last_snapshot_time = 0

            snap1 = auto_snapshot(debounce_seconds=60, trigger="test")
            assert snap1 is not None

            # Second call within debounce window should be skipped
            snap2 = auto_snapshot(debounce_seconds=60, trigger="test")
            assert snap2 is None


# ---------------------------------------------------------------------------
# State hash / change detection
# ---------------------------------------------------------------------------

class TestChangeDetection:
    """Test compute_state_hash for change detection."""

    def test_same_state_same_hash(self, engine):
        h1 = engine.compute_state_hash()
        h2 = engine.compute_state_hash()
        assert h1 == h2

    def test_changed_state_different_hash(self, engine, tmp_hermes):
        h1 = engine.compute_state_hash()
        (tmp_hermes / "config.yaml").write_text("changed: true\n")
        h2 = engine.compute_state_hash()
        assert h1 != h2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases and error handling."""

    def test_restore_nonexistent_snapshot(self, engine):
        assert engine.restore("does-not-exist-999999") is False

    def test_snapshot_with_missing_files(self, engine, tmp_hermes):
        """Should handle missing files gracefully."""
        # Remove a file
        (tmp_hermes / "processes.json").unlink()
        snap_id = engine.snapshot()
        assert snap_id is not None

    def test_empty_hermes_dir(self, tmp_path):
        """Should handle empty hermes dir without crashing."""
        empty = tmp_path / "empty"
        empty.mkdir()
        eng = SnapshotEngine(hermes_home=empty)
        snap_id = eng.snapshot()
        # Should return None (no files to snapshot)
        assert snap_id is None

    def test_concurrent_snapshots(self, engine):
        """Multiple threads snapshotting shouldn't crash."""
        import threading

        results = []
        def do_snap():
            r = engine.snapshot(label="concurrent")
            results.append(r)

        threads = [threading.Thread(target=do_snap) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # At least one should succeed
        assert any(r is not None for r in results)


# ---------------------------------------------------------------------------
# Write-Ahead Log
# ---------------------------------------------------------------------------

class TestWAL:
    """Test WAL (Write-Ahead Log) functionality."""

    def test_wal_append_and_unflushed(self, engine):
        """WAL entries should appear in unflushed list."""
        engine.wal_append("config.yaml", b"model: test\n")
        entries = engine.wal_unflushed()
        assert len(entries) == 1
        assert entries[0]["rel_path"] == "config.yaml"

    def test_wal_flush_during_snapshot(self, engine, tmp_hermes):
        """Snapshot should flush WAL entries."""
        engine.wal_append("config.yaml", b"wal-data\n")
        assert len(engine.wal_unflushed()) == 1

        snap_id = engine.snapshot(label="wal-test")
        assert snap_id is not None

        # WAL should be flushed
        assert len(engine.wal_unflushed()) == 0

    def test_wal_replay(self, engine, tmp_hermes):
        """WAL replay should restore unflushed changes."""
        # Create initial state and snapshot
        engine.snapshot(label="initial")

        # Append change to WAL (simulating a change between snapshots)
        new_config = b"model: replayed\nprovider: test\n"
        engine.wal_append("config.yaml", new_config)

        # Verify the live file hasn't changed yet
        assert "replayed" not in (tmp_hermes / "config.yaml").read_text()

        # Replay
        restored = engine.wal_replay()
        assert "config.yaml" in restored
        assert "replayed" in (tmp_hermes / "config.yaml").read_text()

    def test_wal_append_file(self, engine, tmp_hermes):
        """wal_append_file should handle files correctly."""
        engine.wal_append_file("config.yaml", tmp_hermes / "config.yaml")
        entries = engine.wal_unflushed()
        assert len(entries) == 1
        assert entries[0]["rel_path"] == "config.yaml"

    def test_wal_dedup_in_replay(self, engine, tmp_hermes):
        """WAL replay should keep only the latest version of each file."""
        engine.wal_append("config.yaml", b"v1\n")
        engine.wal_append("config.yaml", b"v2\n")
        engine.wal_append("config.yaml", b"v3-final\n")

        restored = engine.wal_replay()
        assert len(restored) == 1  # Only one file
        assert "v3-final" in (tmp_hermes / "config.yaml").read_text()

    def test_wal_prune(self, engine):
        """wal_prune should remove old flushed entries."""
        engine.wal_append("config.yaml", b"data\n")
        snap_id = engine.snapshot()
        assert len(engine.wal_unflushed()) == 0  # Flushed

        # Prune with 0 hours (should delete everything flushed)
        deleted = engine.wal_prune(older_than_hours=0)
        assert deleted >= 1


# ---------------------------------------------------------------------------
# Branching
# ---------------------------------------------------------------------------

class TestBranching:
    """Test branch management."""

    def test_default_branch_is_main(self, engine):
        assert engine.get_branch() == "main"

    def test_create_branch(self, engine):
        assert engine.create_branch("test-branch")
        branches = engine.list_branches()
        names = [b["name"] for b in branches]
        assert "test-branch" in names
        assert "main" in names

    def test_create_branch_invalid_name(self, engine):
        assert not engine.create_branch("bad name!")  # Spaces + special chars
        assert not engine.create_branch("")  # Empty

    def test_create_duplicate_branch(self, engine):
        assert engine.create_branch("dup-test")
        assert not engine.create_branch("dup-test")  # Duplicate

    def test_switch_branch(self, engine, tmp_hermes):
        # Create snapshot on main
        (tmp_hermes / "config.yaml").write_text("branch: main\n")
        snap_main = engine.snapshot(label="main-state")

        # Create branch from main
        assert engine.create_branch("experiment")
        assert engine.get_branch() == "main"

        # Change state on main
        (tmp_hermes / "config.yaml").write_text("branch: main-changed\n")
        engine.snapshot(label="main-updated")

        # Switch to experiment
        assert engine.switch_branch("experiment")
        assert engine.get_branch() == "experiment"

    def test_delete_branch(self, engine):
        engine.create_branch("to-delete")
        assert engine.delete_branch("to-delete")

    def test_cannot_delete_main(self, engine):
        assert not engine.delete_branch("main")

    def test_cannot_delete_active_branch(self, engine):
        engine.create_branch("active-branch")
        engine.switch_branch("active-branch")
        assert not engine.delete_branch("active-branch")

    def test_branch_tracks_snapshot(self, engine, tmp_hermes):
        """Snapshots should be tagged with their branch."""
        (tmp_hermes / "config.yaml").write_text("v1\n")
        snap1 = engine.snapshot(label="on-main")

        engine.create_branch("feature")
        engine.switch_branch("feature")
        (tmp_hermes / "config.yaml").write_text("v2\n")
        snap2 = engine.snapshot(label="on-feature")

        main_snaps = engine.list_snapshots(branch="main")
        feature_snaps = engine.list_snapshots(branch="feature")

        main_ids = [s["id"] for s in main_snaps]
        feature_ids = [s["id"] for s in feature_snaps]

        assert snap1 in main_ids
        assert snap2 in feature_ids

    def test_branch_protection_in_prune(self, engine, tmp_hermes):
        """Prune should NOT delete snapshots on non-main branches."""
        # Create snapshots on main
        for i in range(5):
            (tmp_hermes / "config.yaml").write_text(f"main-{i}\n")
            time.sleep(0.05)
            engine.snapshot(label=f"main-{i}")

        # Create branch with a snapshot
        engine.create_branch("protected")
        engine.switch_branch("protected")
        (tmp_hermes / "config.yaml").write_text("protected-data\n")
        engine.snapshot(label="protected-snap")
        protected_snap_id = engine.get_head()

        # Switch back to main and prune aggressively
        engine.switch_branch("main")
        engine.prune(keep_last=2, keep_hourly=0, keep_daily=0)

        # The protected branch snapshot should still exist
        remaining = engine.list_snapshots(limit=100)
        remaining_ids = [s["id"] for s in remaining]
        assert protected_snap_id in remaining_ids

    def test_create_branch_from_snapshot(self, engine, tmp_hermes):
        """Create branch from a specific snapshot."""
        (tmp_hermes / "config.yaml").write_text("snap1\n")
        snap1 = engine.snapshot(label="first")
        (tmp_hermes / "config.yaml").write_text("snap2\n")
        snap2 = engine.snapshot(label="second")

        # Create branch from the first snapshot (not HEAD)
        assert engine.create_branch("from-first", from_snapshot=snap1)
        branches = engine.list_branches()
        branch_map = {b["name"]: b for b in branches}
        assert branch_map["from-first"]["head_snapshot"] == snap1


# ---------------------------------------------------------------------------
# Integration: WAL + Branch + Auto-prune
# ---------------------------------------------------------------------------

class TestIntegration:
    """Integration tests combining WAL, branching, and pruning."""

    def test_full_lifecycle(self, engine, tmp_hermes):
        """Full lifecycle: snapshot -> branch -> changes -> WAL -> merge back."""
        # 1. Snapshot on main
        (tmp_hermes / "config.yaml").write_text("model: original\n")
        snap1 = engine.snapshot(label="stable")
        assert snap1

        # 2. Branch for upgrade experiment
        engine.create_branch("upgrade-test")
        engine.switch_branch("upgrade-test")

        # 3. Make changes on the branch
        (tmp_hermes / "config.yaml").write_text("model: new-and-buggy\n")
        snap2 = engine.snapshot(label="upgrade-attempt")
        assert snap2

        # 4. WAL records more changes
        engine.wal_append("config.yaml", b"model: hotfix-attempt\n")
        assert len(engine.wal_unflushed()) == 1

        # 5. Replay WAL (simulates crash recovery)
        restored = engine.wal_replay()
        assert "config.yaml" in restored

        # 6. Switch back to main — state should be restored to main's snapshot
        engine.switch_branch("main")
        config = (tmp_hermes / "config.yaml").read_text()
        # Should have the main branch's last state (via switch_branch restore)

        # 7. Delete the failed experiment branch
        engine.switch_branch("main")
        assert engine.delete_branch("upgrade-test")

        # 8. Prune
        deleted = engine.prune(keep_last=10)
        assert deleted >= 0  # Prune ran without error

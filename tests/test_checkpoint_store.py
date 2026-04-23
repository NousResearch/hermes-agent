import pytest
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from agent.checkpoint_store import CheckpointStore


@pytest.fixture
def store(tmp_path):
    return CheckpointStore(checkpoints_dir=tmp_path / "checkpoints")


@pytest.fixture
def sample_checkpoint():
    return {
        "task": "Red team QA on PR #14303",
        "status": "in_progress",
        "created": "2026-04-22T22:58:00",
        "updated": "2026-04-22T23:15:00",
        "progress": [
            {"step": "Read code", "status": "completed"},
            {"step": "Write tests", "status": "in_progress"},
        ],
        "state": {
            "active_branch": "fix/delegation",
            "files_changed": ["tools/delegate_tool.py"],
            "tests_status": "22/22 passing",
            "last_commit": "abc123",
            "pushed": True,
            "working_directory": "/tmp/repo",
        },
        "decisions": ["No SSRF fix needed"],
        "blocked": [],
        "unresolved": [],
    }


class TestCheckpointStoreWrite:
    def test_write_creates_yaml_file(self, store, sample_checkpoint):
        store.write("sess_001", sample_checkpoint)
        path = store._path_for("sess_001")
        assert path.exists()

    def test_write_content_is_valid_yaml(self, store, sample_checkpoint):
        store.write("sess_001", sample_checkpoint)
        path = store._path_for("sess_001")
        data = yaml.safe_load(path.read_text())
        assert data["task"] == "Red team QA on PR #14303"

    def test_write_overwrites_existing(self, store, sample_checkpoint):
        store.write("sess_001", sample_checkpoint)
        sample_checkpoint["status"] = "completed"
        store.write("sess_001", sample_checkpoint)
        data = store.read("sess_001")
        assert data["status"] == "completed"


class TestCheckpointStoreRead:
    def test_read_existing(self, store, sample_checkpoint):
        store.write("sess_001", sample_checkpoint)
        data = store.read("sess_001")
        assert data["task"] == "Red team QA on PR #14303"

    def test_read_nonexistent_returns_none(self, store):
        assert store.read("no_such_session") is None

    def test_read_corrupt_yaml_returns_none(self, store):
        path = store._path_for("sess_bad")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{{invalid yaml: [")
        assert store.read("sess_bad") is None


class TestCheckpointStoreDelete:
    def test_delete_removes_file(self, store, sample_checkpoint):
        store.write("sess_001", sample_checkpoint)
        store.delete("sess_001")
        assert not store._path_for("sess_001").exists()

    def test_delete_nonexistent_is_noop(self, store):
        store.delete("no_such_session")  # no error


class TestCheckpointStoreList:
    def test_list_returns_all_session_ids(self, store, sample_checkpoint):
        store.write("sess_001", sample_checkpoint)
        store.write("sess_002", sample_checkpoint)
        ids = store.list_sessions()
        assert sorted(ids) == ["sess_001", "sess_002"]

    def test_list_empty_dir(self, store):
        assert store.list_sessions() == []


class TestCheckpointStoreGC:
    def test_gc_removes_old_checkpoints(self, store, sample_checkpoint):
        # Write an old checkpoint directly to disk (bypass write() which stamps now())
        old_time = (datetime.now() - timedelta(days=10)).isoformat()
        sample_checkpoint["updated"] = old_time
        path_old = store._path_for("sess_old")
        path_old.parent.mkdir(parents=True, exist_ok=True)
        path_old.write_text(
            yaml.dump(sample_checkpoint, default_flow_style=False, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        # Write a recent one through the normal path
        store.write("sess_new", sample_checkpoint)
        removed = store.garbage_collect(max_age_days=7)
        assert "sess_old" in removed
        assert "sess_new" not in removed
        assert store.read("sess_old") is None
        assert store.read("sess_new") is not None

    def test_gc_keeps_all_when_fresh(self, store, sample_checkpoint):
        store.write("sess_001", sample_checkpoint)
        removed = store.garbage_collect(max_age_days=7)
        assert removed == []
        assert store.read("sess_001") is not None
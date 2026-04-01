"""Tests for recording/store.py — recording CRUD operations."""

import pytest

import recording.store as store_mod
from recording.store import (
    create_recording,
    get_recording,
    list_recordings,
    delete_recording,
    add_step,
    validate_name,
)


@pytest.fixture(autouse=True)
def _patch_recordings_dir(tmp_path, monkeypatch):
    """Ensure RECORDINGS_DIR points to the test-local temp directory."""
    rec_dir = tmp_path / "hermes_test" / "recordings"
    rec_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(store_mod, "RECORDINGS_DIR", rec_dir)


class TestValidateName:
    def test_valid_names(self):
        assert validate_name("daily-report") == "daily-report"
        assert validate_name("test_recording") == "test_recording"
        assert validate_name("a123") == "a123"
        assert validate_name("Report1") == "Report1"

    def test_empty_name(self):
        with pytest.raises(ValueError, match="non-empty"):
            validate_name("")

    def test_none_name(self):
        with pytest.raises(ValueError, match="non-empty"):
            validate_name(None)

    def test_invalid_chars(self):
        with pytest.raises(ValueError, match="alphanumeric"):
            validate_name("has spaces")
        with pytest.raises(ValueError, match="alphanumeric"):
            validate_name("has/slash")
        with pytest.raises(ValueError, match="alphanumeric"):
            validate_name("has.dot")

    def test_starts_with_hyphen(self):
        with pytest.raises(ValueError, match="alphanumeric"):
            validate_name("-starts-bad")

    def test_too_long(self):
        with pytest.raises(ValueError, match="64 characters"):
            validate_name("a" * 65)

    def test_max_length_ok(self):
        name = "a" * 64
        assert validate_name(name) == name


class TestCreateRecording:
    def test_basic_create(self):
        rec = create_recording("test-rec", description="A test recording")
        assert rec["name"] == "test-rec"
        assert rec["description"] == "A test recording"
        assert rec["steps"] == []
        assert "created_at" in rec

    def test_create_no_description(self):
        rec = create_recording("minimal")
        assert rec["description"] == ""

    def test_create_duplicate(self):
        create_recording("dup")
        with pytest.raises(ValueError, match="already exists"):
            create_recording("dup")

    def test_create_invalid_name(self):
        with pytest.raises(ValueError):
            create_recording("bad name!")


class TestGetRecording:
    def test_get_existing(self):
        create_recording("get-test", description="desc")
        rec = get_recording("get-test")
        assert rec is not None
        assert rec["name"] == "get-test"
        assert rec["description"] == "desc"

    def test_get_nonexistent(self):
        assert get_recording("does-not-exist") is None


class TestListRecordings:
    def test_list_empty(self):
        result = list_recordings()
        assert result == []

    def test_list_multiple(self):
        create_recording("rec-a")
        create_recording("rec-b")
        create_recording("rec-c")

        result = list_recordings()
        names = [r["name"] for r in result]
        assert "rec-a" in names
        assert "rec-b" in names
        assert "rec-c" in names

    def test_list_includes_step_count(self):
        create_recording("counted")
        add_step("counted", "terminal", {"command": "echo hi"}, "hi", True)
        add_step("counted", "terminal", {"command": "echo bye"}, "bye", True)

        result = list_recordings()
        rec = next(r for r in result if r["name"] == "counted")
        assert rec["step_count"] == 2


class TestDeleteRecording:
    def test_delete_existing(self):
        create_recording("to-delete")
        assert delete_recording("to-delete") is True
        assert get_recording("to-delete") is None

    def test_delete_nonexistent(self):
        assert delete_recording("ghost") is False


class TestAddStep:
    def test_add_single_step(self):
        create_recording("steps-test")
        step = add_step("steps-test", "terminal", {"command": "ls"}, "file1\nfile2", True)
        assert step["tool"] == "terminal"
        assert step["arguments"]["command"] == "ls"
        assert step["expected_status"] == "success"

    def test_add_error_step(self):
        create_recording("error-step")
        step = add_step("error-step", "terminal", {"command": "bad"}, "Error: not found", False)
        assert step["expected_status"] == "error"

    def test_steps_accumulate(self):
        create_recording("multi-step")
        add_step("multi-step", "terminal", {"command": "a"}, "out_a", True)
        add_step("multi-step", "terminal", {"command": "b"}, "out_b", True)
        add_step("multi-step", "write_file", {"path": "/tmp/x"}, "ok", True)

        rec = get_recording("multi-step")
        assert len(rec["steps"]) == 3
        assert rec["steps"][0]["arguments"]["command"] == "a"
        assert rec["steps"][2]["tool"] == "write_file"

    def test_add_step_nonexistent(self):
        with pytest.raises(ValueError, match="not found"):
            add_step("nope", "terminal", {}, "", True)

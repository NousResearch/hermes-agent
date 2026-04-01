"""Tests for hermes_cli/record.py — CLI record subcommand."""

import pytest
from unittest.mock import patch, MagicMock
from io import StringIO

import recording.store as store_mod
from recording.store import create_recording, add_step, get_recording


@pytest.fixture(autouse=True)
def _patch_recordings_dir(tmp_path, monkeypatch):
    """Ensure RECORDINGS_DIR points to the test-local temp directory."""
    rec_dir = tmp_path / "hermes_test" / "recordings"
    rec_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(store_mod, "RECORDINGS_DIR", rec_dir)


class TestRecordList:
    def test_list_empty(self, capsys):
        from hermes_cli.record import record_list
        record_list()
        out = capsys.readouterr().out
        assert "No saved recordings" in out

    def test_list_with_recordings(self, capsys):
        create_recording("list-a", "Description A")
        create_recording("list-b")

        from hermes_cli.record import record_list
        record_list()
        out = capsys.readouterr().out
        assert "list-a" in out
        assert "Description A" in out
        assert "list-b" in out


class TestRecordShow:
    def test_show_existing(self, capsys):
        create_recording("show-test", "A recording")
        add_step("show-test", "terminal", {"command": "ls"}, "files", True)

        from hermes_cli.record import record_show
        result = record_show("show-test")
        assert result == 0
        out = capsys.readouterr().out
        assert "show-test" in out
        assert "terminal" in out

    def test_show_nonexistent(self, capsys):
        from hermes_cli.record import record_show
        result = record_show("nope")
        assert result == 1
        out = capsys.readouterr().out
        assert "not found" in out


class TestRecordDelete:
    def test_delete_existing(self, capsys):
        create_recording("del-test")

        from hermes_cli.record import record_delete
        result = record_delete("del-test")
        assert result == 0
        assert get_recording("del-test") is None

    def test_delete_nonexistent(self, capsys):
        from hermes_cli.record import record_delete
        result = record_delete("ghost")
        assert result == 1


class TestRecordSchedule:
    @patch("cron.jobs.create_job")
    def test_schedule_creates_job(self, mock_create_job, capsys):
        create_recording("sched-test")
        add_step("sched-test", "terminal", {"command": "echo"}, "out", True)

        mock_create_job.return_value = {
            "id": "job-123",
            "name": "recording:sched-test",
            "next_run_at": "2026-04-02T08:00:00",
        }

        from hermes_cli.record import record_schedule
        result = record_schedule("sched-test", "0 8 * * *")
        assert result == 0

        mock_create_job.assert_called_once()
        call_kwargs = mock_create_job.call_args
        assert "REPLAY_RECORDING:sched-test" in call_kwargs.kwargs.get("prompt", "") or \
               "REPLAY_RECORDING:sched-test" in str(call_kwargs)

    def test_schedule_nonexistent(self, capsys):
        from hermes_cli.record import record_schedule
        result = record_schedule("no-such", "0 8 * * *")
        assert result == 1


class TestRecordRun:
    @patch("model_tools.handle_function_call")
    def test_run_basic(self, mock_handle, capsys):
        import json
        mock_handle.return_value = json.dumps({"success": True})

        create_recording("run-test")
        add_step("run-test", "terminal", {"command": "echo hi"}, "hi", True)

        from hermes_cli.record import record_run
        result = record_run("run-test")
        assert result == 0
        out = capsys.readouterr().out
        assert "completed" in out.lower() or "Replay" in out

    def test_run_nonexistent(self, capsys):
        from hermes_cli.record import record_run
        result = record_run("nope")
        assert result == 1

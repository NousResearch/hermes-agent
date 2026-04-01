"""Tests for recording/replay.py — replay engine."""

import json
from unittest.mock import patch, MagicMock

import pytest

from recording.replay import replay_recording, _is_error_result


class TestIsErrorResult:
    def test_success_json(self):
        assert _is_error_result(json.dumps({"success": True, "data": "ok"})) is False

    def test_failure_json(self):
        assert _is_error_result(json.dumps({"success": False, "error": "bad"})) is True

    def test_error_key(self):
        assert _is_error_result(json.dumps({"error": "something broke"})) is True

    def test_error_prefix(self):
        assert _is_error_result("Error executing tool 'terminal': boom") is True

    def test_plain_text(self):
        assert _is_error_result("hello world") is False

    def test_empty(self):
        assert _is_error_result("") is False


class TestReplayRecording:
    @patch("model_tools.handle_function_call")
    def test_basic_replay(self, mock_handle):
        mock_handle.return_value = json.dumps({"success": True})

        recording = {
            "name": "test",
            "steps": [
                {"tool": "terminal", "arguments": {"command": "echo hi"}, "expected_status": "success"},
                {"tool": "terminal", "arguments": {"command": "echo bye"}, "expected_status": "success"},
            ],
        }

        result = replay_recording(recording)
        assert result["success"] is True
        assert result["steps_completed"] == 2
        assert result["steps_total"] == 2
        assert result["error"] is None
        assert mock_handle.call_count == 2

    @patch("model_tools.handle_function_call")
    def test_empty_recording(self, mock_handle):
        recording = {"name": "empty", "steps": []}
        result = replay_recording(recording)
        assert result["success"] is True
        assert result["steps_completed"] == 0
        assert mock_handle.call_count == 0

    @patch("model_tools.handle_function_call")
    def test_deviation_continues_by_default(self, mock_handle):
        # Step expects success but tool returns error
        mock_handle.return_value = json.dumps({"success": False, "error": "oops"})

        recording = {
            "name": "deviate",
            "steps": [
                {"tool": "terminal", "arguments": {"command": "fail"}, "expected_status": "success"},
                {"tool": "terminal", "arguments": {"command": "ok"}, "expected_status": "error"},
            ],
        }

        result = replay_recording(recording)
        # Without on_deviation callback, deviations are logged but execution continues
        assert result["success"] is True
        assert result["steps_completed"] == 2

    @patch("model_tools.handle_function_call")
    def test_deviation_callback_abort(self, mock_handle):
        mock_handle.return_value = json.dumps({"success": False, "error": "fail"})

        recording = {
            "name": "abort",
            "steps": [
                {"tool": "terminal", "arguments": {"command": "fail"}, "expected_status": "success"},
                {"tool": "terminal", "arguments": {"command": "never"}, "expected_status": "success"},
            ],
        }

        def abort_on_deviation(i, step, result):
            return False

        result = replay_recording(recording, on_deviation=abort_on_deviation)
        assert result["success"] is False
        assert result["steps_completed"] == 1
        assert "Aborted" in result["error"]
        # Second step should not have been called
        assert mock_handle.call_count == 1

    @patch("model_tools.handle_function_call")
    def test_deviation_callback_continue(self, mock_handle):
        mock_handle.return_value = json.dumps({"success": False, "error": "fail"})

        recording = {
            "name": "continue",
            "steps": [
                {"tool": "terminal", "arguments": {"command": "fail"}, "expected_status": "success"},
                {"tool": "terminal", "arguments": {"command": "also-fail"}, "expected_status": "success"},
            ],
        }

        def continue_on_deviation(i, step, result):
            return True

        result = replay_recording(recording, on_deviation=continue_on_deviation)
        assert result["success"] is True
        assert result["steps_completed"] == 2

    @patch("model_tools.handle_function_call")
    def test_on_step_callback(self, mock_handle):
        mock_handle.return_value = json.dumps({"success": True})

        recording = {
            "name": "callback",
            "steps": [
                {"tool": "terminal", "arguments": {"command": "a"}, "expected_status": "success"},
                {"tool": "terminal", "arguments": {"command": "b"}, "expected_status": "success"},
            ],
        }

        steps_seen = []

        def on_step(i, step, result):
            steps_seen.append(i)

        replay_recording(recording, on_step=on_step)
        assert steps_seen == [0, 1]

    @patch("model_tools.handle_function_call")
    def test_tool_exception_is_caught(self, mock_handle):
        mock_handle.side_effect = RuntimeError("tool crashed")

        recording = {
            "name": "crash",
            "steps": [
                {"tool": "terminal", "arguments": {"command": "crash"}, "expected_status": "success"},
            ],
        }

        result = replay_recording(recording)
        # Exception is caught and treated as error
        assert result["steps_completed"] == 1
        assert result["results"][0]["status"] == "error"
        assert result["results"][0]["deviated"] is True

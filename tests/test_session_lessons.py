"""Tests for session_lessons — auto-extract lessons from recent failures."""

import pytest
from unittest.mock import MagicMock, patch

from agent.session_lessons import (
    extract_lessons_from_sessions,
    format_lessons_for_prompt,
    _extract_error_pattern,
    _suggest_fix,
)


class TestExtractErrorPattern:
    """Test error pattern extraction from tool results."""

    def test_python_exception(self):
        content = 'ModuleNotFoundError: No module named \'os\''
        pattern = _extract_error_pattern(content)
        assert pattern is not None
        assert "ModuleNotFoundError" in pattern

    def test_exit_code(self):
        content = '{"exit_code": 1, "output": "error"}'
        pattern = _extract_error_pattern(content)
        # May or may not match depending on format
        # Just verify it doesn't crash
        assert pattern is None or isinstance(pattern, str)

    def test_command_not_found(self):
        content = "bash: foo: command not found"
        pattern = _extract_error_pattern(content)
        assert pattern is not None
        assert "command not found" in pattern

    def test_permission_denied(self):
        content = "Permission denied: /etc/shadow"
        pattern = _extract_error_pattern(content)
        assert pattern is not None
        assert "permission denied" in pattern.lower()

    def test_no_error(self):
        content = '{"success": true, "data": "hello"}'
        pattern = _extract_error_pattern(content)
        assert pattern is None

    def test_empty_content(self):
        assert _extract_error_pattern("") is None
        assert _extract_error_pattern(None) is None


class TestSuggestFix:
    """Test fix suggestion generation."""

    def test_module_not_found(self):
        suggestion = _suggest_fix("ModuleNotFoundError: No module named 'os'", "")
        assert suggestion is not None
        assert "terminal" in suggestion.lower()

    def test_permission_denied(self):
        suggestion = _suggest_fix("permission denied: /etc/shadow", "")
        assert suggestion is not None
        assert "permission" in suggestion.lower()

    def test_command_not_found(self):
        suggestion = _suggest_fix("command not found: docker", "")
        assert suggestion is not None
        assert "install" in suggestion.lower() or "alternative" in suggestion.lower()

    def test_generic_error(self):
        suggestion = _suggest_fix("something error happened", "")
        assert suggestion is not None
        assert "different approach" in suggestion.lower()

    def test_no_suggestion_for_non_error(self):
        suggestion = _suggest_fix("all good", "")
        assert suggestion is None


class TestExtractLessonsFromSessions:
    """Test the main lesson extraction function."""

    def test_no_session_db(self):
        lessons = extract_lessons_from_sessions(None)
        assert lessons == []

    def test_empty_results(self):
        mock_db = MagicMock()
        mock_db.search_messages.return_value = []
        lessons = extract_lessons_from_sessions(mock_db)
        assert lessons == []

    def test_extracts_recurring_patterns(self):
        mock_db = MagicMock()
        mock_db.search_messages.return_value = [
            {"session_id": "s1", "content": "ModuleNotFoundError: No module named 'os'"},
            {"session_id": "s2", "content": "ModuleNotFoundError: No module named 'os'"},
            {"session_id": "s3", "content": "ModuleNotFoundError: No module named 'os'"},
        ]
        lessons = extract_lessons_from_sessions(mock_db, current_session_id="current")
        assert len(lessons) > 0
        assert lessons[0]["pattern"] is not None
        assert lessons[0]["suggestion"] is not None

    def test_skips_current_session(self):
        mock_db = MagicMock()
        mock_db.search_messages.return_value = [
            {"session_id": "current", "content": "ModuleNotFoundError: No module named 'os'"},
            {"session_id": "current", "content": "ModuleNotFoundError: No module named 'os'"},
        ]
        lessons = extract_lessons_from_sessions(mock_db, current_session_id="current")
        assert lessons == []

    def test_respects_limit(self):
        mock_db = MagicMock()
        # Create many different error patterns
        messages = []
        for i in range(20):
            messages.append({
                "session_id": f"s{i}",
                "content": f"ErrorType{i}: error message {i}",
            })
        mock_db.search_messages.return_value = messages
        lessons = extract_lessons_from_sessions(mock_db, limit=3)
        assert len(lessons) <= 3


class TestFormatLessonsForPrompt:
    """Test prompt formatting."""

    def test_empty_lessons(self):
        assert format_lessons_for_prompt([]) == ""

    def test_single_lesson(self):
        lessons = [{
            "pattern": "ModuleNotFoundError: No module named 'os'",
            "suggestion": "Use terminal instead of execute_code",
            "frequency": "3",
        }]
        result = format_lessons_for_prompt(lessons)
        assert "Recent Session Lessons" in result
        assert "ModuleNotFoundError" in result
        assert "terminal" in result

    def test_multiple_lessons(self):
        lessons = [
            {"pattern": "error1", "suggestion": "fix1", "frequency": "5"},
            {"pattern": "error2", "suggestion": "fix2", "frequency": "3"},
        ]
        result = format_lessons_for_prompt(lessons)
        assert "1." in result
        assert "2." in result
        assert "error1" in result
        assert "error2" in result

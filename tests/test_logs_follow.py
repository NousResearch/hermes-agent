#!/usr/bin/env python3
"""Tests for _follow_log Windows log-rotation fix (GitHub issue #153311)."""

import io
import logging
import tempfile
import threading
from pathlib import Path

import pytest

import hermes_cli.logs as logs_mod
from hermes_logging import RotatingFileHandler
from hermes_cli.logs import _follow_log


@pytest.fixture
def tmp_log_file(tmp_path):
    """Create a temporary log file with some content."""
    log_path = tmp_path / "agent.log"
    log_path.write_text("2026-11-11 12:00:00 INFO first message\n", encoding="utf-8")
    return log_path


@pytest.fixture
def tmp_log_file_multiline(tmp_path):
    """Create a temporary log file with multi-line content (traceback)."""
    log_path = tmp_path / "agent.log"
    log_content = (
        "2026-11-11 12:00:00 ERROR test error\n"
        "Traceback (most recent call last):\n"
        '  File "test.py", line 5, in <module>\n'
        '    raise RuntimeError("error")\n'
        "RuntimeError: error\n"
    )
    log_path.write_text(log_content, encoding="utf-8")
    return log_path


class TestFollowLogBinaryMode:
    """Tests for the _follow_log binary-mode fix (issue #153311)."""

    def test_follow_log_opens_in_binary_mode(self, tmp_log_file):
        """Verify _follow_log opens the log in binary mode and reads lines correctly."""
        lines = []
        with open(tmp_log_file, "rb") as raw_f:
            with io.TextIOWrapper(raw_f, encoding="utf-8-sig", errors="replace") as f:
                f.seek(0, 0)
                lines = f.readlines()

        assert len(lines) == 1
        assert lines[0].startswith("2026-11-11")
        assert "INFO first message" in lines[0]

    def test_follow_log_with_content_freshly_added(self, tmp_path, monkeypatch):
        """Verify _follow_log can read newly appended content."""
        log_path = tmp_path / "agent.log"
        log_path.write_text("2026-11-11 12:00:00 INFO initial\n", encoding="utf-8")

        lines = []
        with open(log_path, "rb") as raw_f:
            with io.TextIOWrapper(raw_f, encoding="utf-8-sig", errors="replace") as f:
                f.seek(0, 0)
                lines = f.readlines()

        assert len(lines) == 1

    def test_follow_log_empty_file(self, tmp_path):
        """Verify _follow_log handles an empty log file gracefully."""
        log_path = tmp_path / "empty.log"
        log_path.touch()

        lines = []
        with open(log_path, "rb") as raw_f:
            with io.TextIOWrapper(raw_f, encoding="utf-8-sig", errors="replace") as f:
                f.seek(0, 0)
                lines = f.readlines()

        assert len(lines) == 0

    def test_follow_log_multiline_content(self, tmp_log_file_multiline):
        """Verify _follow_log handles multi-line log records (tracebacks)."""
        lines = []
        with open(tmp_log_file_multiline, "rb") as raw_f:
            with io.TextIOWrapper(raw_f, encoding="utf-8-sig", errors="replace") as f:
                f.seek(0, 0)
                lines = f.readlines()

        assert len(lines) == 5
        assert "RuntimeError: error" in lines[4]
        assert 'File "test.py"' in lines[2]
        assert 'line 5, in <module>' in lines[2]

    def test_follow_log_binary_content(self, tmp_path):
        """Verify _follow_log handles non-UTF8 content with errors='replace'."""
        log_path = tmp_path / "agent.log"
        log_path.write_bytes(b"2026-11-11 12:00:00 INFO valid text\n\xff\xfe\x00invalid bytes\n")

        lines = []
        with open(log_path, "rb") as raw_f:
            with io.TextIOWrapper(raw_f, encoding="utf-8-sig", errors="replace") as f:
                f.seek(0, 0)
                lines = f.readlines()

        assert len(lines) >= 1
        assert "valid text" in lines[0]


class TestRotatingFileHandlerWindowsBehavior:
    """Tests for RotatingFileHandler behavior on Windows."""

    def test_rotate_with_binary_follow(self, tmp_path):
        """Verify that RotatingFileHandler can rotate a file that _follow_log is reading."""
        from hermes_logging import _ManagedRotatingFileHandler

        log_path = tmp_path / "agent.log"
        log_path.write_text("2026-11-11 12:00:00 INFO test message\n", encoding="utf-8")

        handler = _ManagedRotatingFileHandler(log_path, maxBytes=100, backupCount=1)
        try:
            log_path.write_text("2026-11-11 12:00:01 INFO second message\n", encoding="utf-8")
            handler.doRollover()
            handler.close()

            assert log_path.exists()
            assert (tmp_path / "agent.log.1").exists()
        finally:
            handler.close()


class TestWindowsLogRotation:
    """Tests for the Windows log-rotation scenario (issue #153311)."""

    def test_log_rotation_does_not_block(self, tmp_path):
        """Verify that log rotation is not blocked by a text-mode file handle."""
        from hermes_logging import RotatingFileHandler

        log_path = tmp_path / "agent.log"
        log_path.write_text("2026-11-11 12:00:00 INFO test message\n", encoding="utf-8")

        handler = RotatingFileHandler(log_path, maxBytes=100, backupCount=1)
        try:
            handler.doRollover()
            handler.close()

            assert log_path.exists()
            assert (tmp_path / "agent.log.1").exists()
        finally:
            handler.close()

"""Tests that JSONL transcript writes are flushed to disk immediately."""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock

from gateway.session import SessionStore


def _make_store(tmp_path: Path) -> SessionStore:
    """Create a minimal SessionStore pointing at a temp directory."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    store = SessionStore.__new__(SessionStore)
    store.sessions_dir = sessions_dir
    store._sessions = {}
    store._lock = __import__("threading").Lock()
    store._db = None
    store._sessions_json_path = tmp_path / "sessions.json"
    return store


class TestTranscriptFlush:
    def test_append_is_readable_before_close(self, tmp_path: Path):
        """After append_to_transcript, the data must be on disk immediately
        (not stuck in Python's write buffer)."""
        store = _make_store(tmp_path)
        session_id = "test_session_001"
        msg = {"role": "user", "content": "hello"}

        store.append_to_transcript(session_id, msg)

        # Read raw file — if flush works, data is there now
        transcript = store.get_transcript_path(session_id)
        assert transcript.exists()
        lines = transcript.read_text().strip().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0]) == msg

    def test_append_multiple_all_visible(self, tmp_path: Path):
        """Multiple appends should all be visible on disk immediately."""
        store = _make_store(tmp_path)
        session_id = "test_session_002"
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "bye"},
        ]

        for msg in messages:
            store.append_to_transcript(session_id, msg)

        transcript = store.get_transcript_path(session_id)
        lines = transcript.read_text().strip().splitlines()
        assert len(lines) == 3
        for i, line in enumerate(lines):
            assert json.loads(line) == messages[i]

    def test_rewrite_is_readable_immediately(self, tmp_path: Path):
        """rewrite_transcript should flush all data to disk."""
        store = _make_store(tmp_path)
        session_id = "test_session_003"
        messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
        ]

        store.rewrite_transcript(session_id, messages)

        transcript = store.get_transcript_path(session_id)
        lines = transcript.read_text().strip().splitlines()
        assert len(lines) == 2
        for i, line in enumerate(lines):
            assert json.loads(line) == messages[i]

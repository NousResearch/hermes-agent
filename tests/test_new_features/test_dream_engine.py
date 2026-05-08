"""Tests for Dream Engine."""
import json
import time
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from agent.dream_engine import DreamEngine


@pytest.fixture
def mock_session_db():
    db = MagicMock()
    db.search_sessions.return_value = [
        {"id": "s1", "title": "Test Session 1", "started_at": time.time() - 3600, "source": "cli", "message_count": 5},
        {"id": "s2", "title": "Test Session 2", "started_at": time.time() - 7200, "source": "cli", "message_count": 3},
    ]
    db.get_messages.return_value = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    return db


@pytest.fixture
def engine(mock_session_db):
    return DreamEngine(mock_session_db)


def test_gather_sessions(engine, mock_session_db):
    sessions = engine.gather_sessions(hours=24, limit=10)
    assert len(sessions) == 2
    mock_session_db.search_sessions.assert_called_once()


def test_gather_sessions_empty(engine, mock_session_db):
    mock_session_db.search_sessions.return_value = []
    sessions = engine.gather_sessions()
    assert sessions == []


def test_extract_insights(engine):
    sessions = [
        {"id": "s1", "title": "Memory Setup", "source": "cli", "message_count": 5, "transcript": "user did memory setup"},
        {"id": "s2", "title": "Debug Session", "source": "cli", "message_count": 3, "transcript": "user debugged an issue"},
    ]
    prompt = engine.extract_insights(sessions)
    assert "memory consolidation" in prompt.lower() or "dreaming" in prompt.lower()
    assert "Memory Setup" in prompt
    assert "Debug Session" in prompt


def test_extract_insights_empty(engine):
    prompt = engine.extract_insights([])
    assert prompt == ""


def test_run_dream_result_format(engine, mock_session_db):
    """run_dream() returns a dict with success, insights_count, entries_written, etc."""
    with patch.object(engine, '_create_dream_agent') as mock_create:
        mock_agent = MagicMock()
        mock_agent.run_conversation.return_value = {"final_response": "[]"}
        mock_create.return_value = mock_agent

        result = engine.run_dream(hours=24, limit=10)

    assert isinstance(result, dict)
    assert "success" in result
    assert "insights_count" in result
    assert "entries_written" in result
    assert "sessions_reviewed" in result
    assert result["success"] is True
    assert result["sessions_reviewed"] == 2


def test_run_dream_no_sessions(engine, mock_session_db):
    """run_dream() handles the case of no sessions gracefully."""
    mock_session_db.search_sessions.return_value = []
    result = engine.run_dream(hours=24, limit=10)
    assert result["success"] is True
    assert result["sessions_reviewed"] == 0


def test_apply_consolidation(engine):
    """apply_consolidation writes entries via MemoryStore and returns count."""
    entries = [
        {"target": "memory", "content": "User prefers Python 3.12"},
        {"target": "memory", "content": "macOS ARM64 environment"},
    ]
    with patch("agent.dream_engine.MemoryStore") as MockStore:
        mock_store = MagicMock()
        mock_store.add.return_value = {"success": True}
        MockStore.return_value = mock_store

        count = engine.apply_consolidation(entries)

    assert count == 2
    assert mock_store.add.call_count == 2
    mock_store.load_from_disk.assert_called_once()

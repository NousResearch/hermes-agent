"""Pytest configuration for Mission Control adapter tests."""

import pytest
import tempfile
import os


@pytest.fixture
def temp_db_path():
    """Provide a temporary database file path."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def mock_config():
    """Provide a mock PlatformConfig."""
    from unittest.mock import Mock
    config = Mock()
    config.extra = {}
    return config


@pytest.fixture
def sample_task_data():
    """Provide sample task data for tests."""
    return {
        "id": 12345,
        "title": "Test Task",
        "description": "Test Description",
        "status": "pending",
        "priority": "medium",
        "assigned_to": "hermes-cli",
        "project_id": 1,
        "workspace_id": 1,
        "metadata": "{}"
    }


@pytest.fixture
def sample_agent_data():
    """Provide sample agent data for tests."""
    return {
        "id": "agent-1",
        "agent_id": "agent-1",
        "name": "Test Agent",
        "agent_name": "Test Agent",
        "status": "online",
        "last_seen": 1710774000
    }


@pytest.fixture
def sample_chat_message():
    """Provide sample chat message data."""
    return {
        "id": "msg-123",
        "content": "Hello world",
        "sender": "user@example.com",
        "channel": "general",
        "timestamp": 1710774000
    }


@pytest.fixture
def sample_notification():
    """Provide sample notification data."""
    return {
        "id": "notif-123",
        "title": "Test Notification",
        "message": "This is a test notification",
        "level": "info"
    }


@pytest.fixture
def sample_security_event():
    """Provide sample security event data."""
    return {
        "type": "suspicious_activity",
        "event_type": "suspicious_activity",
        "severity": "high",
        "description": "Multiple failed login attempts detected"
    }

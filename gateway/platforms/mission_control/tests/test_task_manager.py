"""Tests for Mission Control task manager."""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch

from ..database import MissionControlDatabase
from ..notifications import CLINotifier
from ..task_manager import TaskManager


class TestTaskManager:
    """Test task management business logic."""

    @pytest.fixture
    def task_manager(self):
        """Create task manager with mock dependencies."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        
        db = MissionControlDatabase(db_path)
        notifier = Mock(spec=CLINotifier)
        
        tm = TaskManager(db, notifier, agent_name="hermes-cli", auto_accept=True)
        
        yield tm, db, notifier
        
        os.unlink(db_path)

    def test_generate_event_id(self, task_manager):
        """Test event ID generation is deterministic."""
        tm, _, _ = task_manager
        
        event_id1 = tm.generate_event_id("test.event", 12345, {"key": "value"})
        event_id2 = tm.generate_event_id("test.event", 12345, {"key": "value"})
        
        assert event_id1 == event_id2
        assert "test.event" in event_id1
        assert "12345" in event_id1

    def test_generate_event_id_unique(self, task_manager):
        """Test different events generate different IDs."""
        tm, _, _ = task_manager
        
        event_id1 = tm.generate_event_id("test.event", 12345, {"key": "value"})
        event_id2 = tm.generate_event_id("test.event", 12346, {"key": "value"})
        
        assert event_id1 != event_id2

    def test_is_duplicate(self, task_manager):
        """Test duplicate event detection."""
        tm, db, _ = task_manager
        
        event_id = "test:123:abc"
        
        # First time - not duplicate
        assert tm.is_duplicate(event_id) is False
        
        # Log and mark processed
        db.log_webhook_delivery(event_id, "test", "hash")
        db.mark_event_processed(event_id)
        
        # Now is duplicate
        assert tm.is_duplicate(event_id) is True

    def test_handle_task_created(self, task_manager):
        """Test task creation handling."""
        tm, db, notifier = task_manager
        
        data = {
            "id": 123,
            "title": "New Task",
            "priority": "high",
            "assigned_to": "other-agent"
        }
        
        result = tm.handle_task_created(data)
        
        assert result is True
        notifier.task_created.assert_called_once_with(123, "New Task", "high", "other-agent")

    def test_handle_task_created_auto_accept(self, task_manager):
        """Test auto-accept when assigned to our agent."""
        tm, db, notifier = task_manager
        
        data = {
            "id": 123,
            "title": "My Task",
            "priority": "medium",
            "assigned_to": "hermes-cli"  # Matches agent_name
        }
        
        result = tm.handle_task_created(data)
        
        assert result is True
        # Should have auto-accepted
        notifier.task_accepted.assert_called_once_with(123, "My Task")

    def test_handle_task_created_no_auto_accept(self, task_manager):
        """Test no auto-accept when disabled."""
        tm, db, notifier = task_manager
        tm._auto_accept = False  # Disable auto-accept
        
        data = {
            "id": 123,
            "title": "My Task",
            "assigned_to": "hermes-cli"
        }
        
        result = tm.handle_task_created(data)
        
        assert result is True
        # Should NOT have auto-accepted
        notifier.task_accepted.assert_not_called()

    def test_handle_task_updated(self, task_manager):
        """Test task update handling."""
        tm, db, notifier = task_manager
        
        # Create task first
        db.create_task({"id": 123, "title": "Old Title", "status": "pending"})
        
        data = {
            "id": 123,
            "title": "New Title",
            "status": "in_progress"
        }
        
        result = tm.handle_task_updated(data)
        
        assert result is True
        
        # Verify updated
        task = db.get_task(123)
        assert task["title"] == "New Title"
        assert task["status"] == "in_progress"

    def test_handle_task_updated_new_task(self, task_manager):
        """Test update creates task if not exists."""
        tm, db, notifier = task_manager
        
        data = {
            "id": 123,
            "title": "New Title",
            "status": "pending"
        }
        
        result = tm.handle_task_updated(data)
        
        assert result is True
        
        # Should have created
        task = db.get_task(123)
        assert task is not None
        assert task["title"] == "New Title"

    def test_handle_task_status_changed(self, task_manager):
        """Test status change handling."""
        tm, db, notifier = task_manager
        
        # Create task
        db.create_task({"id": 123, "title": "Task", "status": "pending"})
        
        data = {
            "id": 123,
            "old_status": "pending",
            "new_status": "completed"
        }
        
        result = tm.handle_task_status_changed(data)
        
        assert result is True
        
        task = db.get_task(123)
        assert task["status"] == "completed"

    def test_handle_agent_status_changed(self, task_manager):
        """Test agent status change handling."""
        tm, db, notifier = task_manager
        
        data = {
            "agent_id": "agent-1",
            "agent_name": "hermes-cli",
            "status": "online",
            "last_seen": 1234567890
        }
        
        result = tm.handle_agent_status_changed(data)
        
        assert result is True

    def test_accept_task(self, task_manager):
        """Test task acceptance."""
        tm, db, notifier = task_manager
        
        # Create task
        db.create_task({"id": 123, "title": "Task", "status": "pending"})
        
        tm.accept_task(123, "Test Task")
        
        notifier.task_accepted.assert_called_once_with(123, "Test Task")
        
        task = db.get_task(123)
        assert task["accepted_at"] is not None

    def test_accept_task_not_found(self, task_manager):
        """Test accepting non-existent task fails gracefully."""
        tm, _, notifier = task_manager
        
        # Should not raise
        tm.accept_task(99999, "Non-existent")
        
        # Notification still sent
        notifier.task_accepted.assert_called_once_with(99999, "Non-existent")

    def test_handle_unknown_event(self, task_manager):
        """Test unknown event handling."""
        tm, _, _ = task_manager
        
        result = tm.handle_unknown_event("custom.event", {"data": "test"})
        
        # Should return True (acknowledged but not processed)
        assert result is True

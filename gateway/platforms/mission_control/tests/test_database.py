"""Tests for Mission Control database operations."""

import pytest
import sqlite3
import tempfile
import os
from pathlib import Path

from ..database import MissionControlDatabase


class TestMissionControlDatabase:
    """Test SQLite database operations."""

    @pytest.fixture
    def db(self):
        """Create temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        
        database = MissionControlDatabase(db_path)
        yield database
        
        # Cleanup
        os.unlink(db_path)

    def test_database_initialization(self, db):
        """Test database creates required tables."""
        conn = sqlite3.connect(db._db_path)
        cursor = conn.cursor()
        
        # Check tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        
        assert "mc_tasks" in tables
        assert "mc_agents" in tables
        assert "mc_webhook_deliveries" in tables
        
        conn.close()

    def test_create_task(self, db):
        """Test task creation."""
        task_data = {
            "id": 123,
            "title": "Test Task",
            "description": "Test Description",
            "status": "pending",
            "priority": "high",
            "assigned_to": "hermes-cli",
            "project_id": 1,
            "workspace_id": 1
        }
        
        result = db.create_task(task_data)
        assert result is True
        
        # Verify task exists
        task = db.get_task(123)
        assert task is not None
        assert task["title"] == "Test Task"
        assert task["status"] == "pending"

    def test_create_task_duplicate(self, db):
        """Test creating duplicate task updates existing."""
        task_data = {
            "id": 123,
            "title": "Test Task",
            "status": "pending"
        }
        
        db.create_task(task_data)
        
        # Create again with different title
        task_data["title"] = "Updated Task"
        result = db.create_task(task_data)
        assert result is True
        
        # Should have updated
        task = db.get_task(123)
        assert task["title"] == "Updated Task"

    def test_update_task(self, db):
        """Test task update with valid columns."""
        # Create task first
        task_data = {
            "id": 123,
            "title": "Test Task",
            "status": "pending"
        }
        db.create_task(task_data)
        
        # Update
        result = db.update_task(123, {"status": "in_progress", "title": "New Title"})
        assert result is True
        
        task = db.get_task(123)
        assert task["status"] == "in_progress"
        assert task["title"] == "New Title"

    def test_update_task_sql_injection_protection(self, db):
        """Test that invalid columns are rejected."""
        # Create task
        db.create_task({"id": 123, "title": "Test", "status": "pending"})
        
        # Try to update with invalid column (should be ignored)
        result = db.update_task(123, {"status": "completed", "invalid_column": "value"})
        assert result is True
        
        # Task should still be updated for valid column
        task = db.get_task(123)
        assert task["status"] == "completed"

    def test_accept_task(self, db):
        """Test task acceptance."""
        db.create_task({"id": 123, "title": "Test", "status": "pending"})
        
        result = db.accept_task(123, "session-123")
        assert result is True
        
        task = db.get_task(123)
        assert task["hermes_session_id"] == "session-123"
        assert task["accepted_at"] is not None

    def test_complete_task(self, db):
        """Test task completion."""
        db.create_task({"id": 123, "title": "Test", "status": "in_progress"})
        
        result = db.complete_task(123)
        assert result is True
        
        task = db.get_task(123)
        assert task["completed_at"] is not None

    def test_get_nonexistent_task(self, db):
        """Test getting task that doesn't exist."""
        task = db.get_task(99999)
        assert task is None

    def test_is_duplicate_event(self, db):
        """Test event idempotency checking."""
        event_id = "test:event:123"
        
        # First check should be False
        assert db.is_duplicate_event(event_id) is False
        
        # Log the delivery
        db.log_webhook_delivery(event_id, "test.ping", "hash123")
        
        # Now should be marked as processed (but is_duplicate_event only checks mc_webhook_deliveries)
        # Actually the logic checks if processed_at is null
        # Let's check again after marking processed
        db.mark_event_processed(event_id)
        
        # After processing, a new event with same ID should be duplicate
        # But is_duplicate_event just checks existence in mc_webhook_deliveries
        assert db.is_duplicate_event(event_id) is True

    def test_log_webhook_delivery(self, db):
        """Test webhook delivery logging."""
        db.log_webhook_delivery("event:123", "test.event", "payload_hash")
        
        # Verify logged
        conn = sqlite3.connect(db._db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT event_id, event_type FROM mc_webhook_deliveries WHERE event_id = ?",
            ("event:123",)
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[1] == "test.event"
        conn.close()

    def test_list_tasks(self, db):
        """Test listing tasks."""
        db.create_task({"id": 1, "title": "Task 1", "status": "pending"})
        db.create_task({"id": 2, "title": "Task 2", "status": "completed"})
        
        tasks = db.list_tasks()
        assert len(tasks) == 2

    def test_list_tasks_by_status(self, db):
        """Test filtering tasks by status."""
        db.create_task({"id": 1, "title": "Task 1", "status": "pending"})
        db.create_task({"id": 2, "title": "Task 2", "status": "completed"})
        
        tasks = db.list_tasks(status="pending")
        assert len(tasks) == 1
        assert tasks[0]["title"] == "Task 1"

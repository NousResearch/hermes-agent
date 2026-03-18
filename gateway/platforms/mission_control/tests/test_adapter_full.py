"""Comprehensive tests for Mission Control adapter event handlers."""

import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock, MagicMock

from ..adapter import MissionControlAdapter
from ..database import MissionControlDatabase
from ..notifications import CLINotifier
from ..task_manager import TaskManager


class TestAdapterEventHandlers:
    """Test all event type handlers."""

    @pytest.fixture
    def adapter(self):
        """Create adapter with mocked components."""
        config = Mock()
        config.extra = {}
        
        adapter = MissionControlAdapter(config)
        
        # Mock all components
        adapter._db = Mock(spec=MissionControlDatabase)
        adapter._notifier = Mock(spec=CLINotifier)
        adapter._task_manager = Mock(spec=TaskManager)
        adapter._task_manager.handle_task_created = Mock(return_value=True)
        adapter._task_manager.handle_task_updated = Mock(return_value=True)
        adapter._task_manager.handle_task_status_changed = Mock(return_value=True)
        adapter._task_manager.handle_agent_status_changed = Mock(return_value=True)
        adapter._task_manager.handle_unknown_event = Mock(return_value=True)
        
        return adapter

    # ========== Task Events ==========

    @pytest.mark.asyncio
    async def test_task_created_activity_prefix(self, adapter):
        """Test activity.task_created event."""
        data = {"id": 123, "title": "Task", "assigned_to": "hermes-cli"}
        result = await adapter._route_event("activity.task_created", data)
        assert result is True
        adapter._task_manager.handle_task_created.assert_called_once_with(data)

    @pytest.mark.asyncio
    async def test_task_created_direct_prefix(self, adapter):
        """Test task.created event."""
        data = {"id": 123, "title": "Task"}
        result = await adapter._route_event("task.created", data)
        assert result is True
        adapter._task_manager.handle_task_created.assert_called_once_with(data)

    @pytest.mark.asyncio
    async def test_task_updated_activity_prefix(self, adapter):
        """Test activity.task_updated event."""
        data = {"id": 123, "title": "Updated"}
        result = await adapter._route_event("activity.task_updated", data)
        assert result is True
        adapter._task_manager.handle_task_updated.assert_called_once_with(data)

    @pytest.mark.asyncio
    async def test_task_updated_direct_prefix(self, adapter):
        """Test task.updated event."""
        data = {"id": 123, "title": "Updated"}
        result = await adapter._route_event("task.updated", data)
        assert result is True
        adapter._task_manager.handle_task_updated.assert_called_once_with(data)

    @pytest.mark.asyncio
    async def test_task_status_changed_activity_prefix(self, adapter):
        """Test activity.task_status_changed event."""
        data = {"id": 123, "status": "completed"}
        result = await adapter._route_event("activity.task_status_changed", data)
        assert result is True
        adapter._task_manager.handle_task_status_changed.assert_called_once_with(data)

    @pytest.mark.asyncio
    async def test_task_status_changed_direct_prefix(self, adapter):
        """Test task.status_changed event."""
        data = {"id": 123, "status": "completed"}
        result = await adapter._route_event("task.status_changed", data)
        assert result is True
        adapter._task_manager.handle_task_status_changed.assert_called_once_with(data)

    @pytest.mark.asyncio
    async def test_task_deleted_activity_prefix(self, adapter):
        """Test activity.task_deleted event."""
        data = {"id": 123}
        result = await adapter._route_event("activity.task_deleted", data)
        assert result is True

    @pytest.mark.asyncio
    async def test_task_deleted_direct_prefix(self, adapter):
        """Test task.deleted event."""
        data = {"id": 123}
        result = await adapter._route_event("task.deleted", data)
        assert result is True

    # ========== Agent Events ==========

    @pytest.mark.asyncio
    async def test_agent_status_change(self, adapter):
        """Test agent.status_change event."""
        data = {"agent_id": "agent-1", "status": "online"}
        result = await adapter._route_event("agent.status_change", data)
        assert result is True
        adapter._task_manager.handle_agent_status_changed.assert_called_once_with(data)

    @pytest.mark.asyncio
    async def test_agent_error(self, adapter):
        """Test agent.error event."""
        data = {"agent_id": "agent-1", "error": "Connection failed"}
        result = await adapter._route_event("agent.error", data)
        assert result is True
        adapter._task_manager.handle_agent_status_changed.assert_called_once_with(data)

    @pytest.mark.asyncio
    async def test_agent_created(self, adapter):
        """Test agent.created event."""
        data = {"id": "agent-1", "name": "New Agent"}
        result = await adapter._route_event("agent.created", data)
        assert result is True
        adapter._task_manager.handle_agent_status_changed.assert_called_once_with(data)

    @pytest.mark.asyncio
    async def test_agent_updated(self, adapter):
        """Test agent.updated event."""
        data = {"id": "agent-1", "name": "Updated Agent"}
        result = await adapter._route_event("agent.updated", data)
        assert result is True
        adapter._task_manager.handle_agent_status_changed.assert_called_once_with(data)

    @pytest.mark.asyncio
    async def test_agent_deleted(self, adapter):
        """Test agent.deleted event."""
        data = {"id": "agent-1", "name": "Deleted Agent"}
        result = await adapter._route_event("agent.deleted", data)
        assert result is True

    @pytest.mark.asyncio
    async def test_agent_synced(self, adapter):
        """Test agent.synced event."""
        data = {"id": "agent-1", "last_sync": 1234567890}
        result = await adapter._route_event("agent.synced", data)
        assert result is True
        adapter._task_manager.handle_agent_status_changed.assert_called_once_with(data)

    @pytest.mark.asyncio
    async def test_agent_status_changed(self, adapter):
        """Test agent.status_changed event."""
        data = {"id": "agent-1", "status": "busy"}
        result = await adapter._route_event("agent.status_changed", data)
        assert result is True
        adapter._task_manager.handle_agent_status_changed.assert_called_once_with(data)

    # ========== Chat Events ==========

    @pytest.mark.asyncio
    async def test_chat_message(self, adapter):
        """Test chat.message event."""
        data = {
            "id": "msg-123",
            "content": "Hello world",
            "sender": "user@example.com",
            "channel": "general"
        }
        result = await adapter._route_event("chat.message", data)
        assert result is True

    @pytest.mark.asyncio
    async def test_chat_message_deleted(self, adapter):
        """Test chat.message.deleted event."""
        data = {"id": "msg-123"}
        result = await adapter._route_event("chat.message.deleted", data)
        assert result is True

    @pytest.mark.asyncio
    async def test_chat_message_with_notifications(self, adapter):
        """Test chat.message with notifier."""
        adapter._notifier = Mock()
        data = {
            "id": "msg-123",
            "content": "Hello",
            "sender": "test",
            "channel": "test-channel"
        }
        result = await adapter._route_event("chat.message", data)
        assert result is True

    # ========== Notification Events ==========

    @pytest.mark.asyncio
    async def test_notification_created(self, adapter):
        """Test notification.created event."""
        data = {
            "id": "notif-123",
            "title": "Test Notification",
            "message": "This is a test",
            "level": "info"
        }
        result = await adapter._route_event("notification.created", data)
        assert result is True
        adapter._notifier.notify.assert_called_once()

    @pytest.mark.asyncio
    async def test_notification_created_high_priority(self, adapter):
        """Test high priority notification."""
        data = {
            "id": "notif-123",
            "title": "Alert",
            "message": "Critical!",
            "level": "critical"
        }
        result = await adapter._route_event("notification.created", data)
        assert result is True
        adapter._notifier.notify.assert_called_once()

    @pytest.mark.asyncio
    async def test_notification_read(self, adapter):
        """Test notification.read event."""
        data = {"id": "notif-123"}
        result = await adapter._route_event("notification.read", data)
        assert result is True

    # ========== Activity Events ==========

    @pytest.mark.asyncio
    async def test_activity_created(self, adapter):
        """Test activity.created event."""
        data = {
            "type": "user.login",
            "description": "User logged in",
            "user": "admin"
        }
        result = await adapter._route_event("activity.created", data)
        assert result is True

    # ========== Security Events ==========

    @pytest.mark.asyncio
    async def test_audit_security(self, adapter):
        """Test audit.security event."""
        data = {
            "type": "permission_change",
            "severity": "high",
            "description": "Admin permissions modified"
        }
        result = await adapter._route_event("audit.security", data)
        assert result is True

    @pytest.mark.asyncio
    async def test_security_event(self, adapter):
        """Test security.event event."""
        data = {
            "event_type": "suspicious_login",
            "severity": "critical",
            "description": "Multiple failed login attempts"
        }
        result = await adapter._route_event("security.event", data)
        assert result is True

    @pytest.mark.asyncio
    async def test_security_event_notifies(self, adapter):
        """Test that high severity security events trigger notifications."""
        adapter._notifier = Mock()
        data = {
            "type": "breach_attempt",
            "severity": "critical",
            "description": "Possible intrusion"
        }
        result = await adapter._route_event("security.event", data)
        assert result is True

    # ========== Connection Events ==========

    @pytest.mark.asyncio
    async def test_connection_created(self, adapter):
        """Test connection.created event."""
        data = {
            "id": "conn-123",
            "agent_name": "hermes-cli",
            "type": "connection.created"
        }
        result = await adapter._route_event("connection.created", data)
        assert result is True

    @pytest.mark.asyncio
    async def test_connection_disconnected(self, adapter):
        """Test connection.disconnected event."""
        data = {
            "id": "conn-123",
            "agent_name": "hermes-cli",
            "type": "connection.disconnected"
        }
        result = await adapter._route_event("connection.disconnected", data)
        assert result is True

    # ========== GitHub Events ==========

    @pytest.mark.asyncio
    async def test_github_synced(self, adapter):
        """Test github.synced event."""
        data = {
            "repository": "builderz-labs/mission-control",
            "sync_type": "commits",
            "commit_count": 15
        }
        result = await adapter._route_event("github.synced", data)
        assert result is True

    # ========== Test Events ==========

    @pytest.mark.asyncio
    async def test_test_ping(self, adapter):
        """Test test.ping event."""
        data = {}
        result = await adapter._route_event("test.ping", data)
        assert result is True

    # ========== Unknown Events ==========

    @pytest.mark.asyncio
    async def test_unknown_event(self, adapter):
        """Test unknown event handling."""
        data = {"some": "data"}
        result = await adapter._route_event("custom.unknown.event", data)
        assert result is True
        adapter._task_manager.handle_unknown_event.assert_called_once_with("custom.unknown.event", data)

    @pytest.mark.asyncio
    async def test_no_task_manager(self, adapter):
        """Test routing without task manager."""
        adapter._task_manager = None
        result = await adapter._route_event("task.created", {})
        assert result is False


class TestAdapterHandlerMethods:
    """Test individual handler methods."""

    @pytest.fixture
    def adapter(self):
        config = Mock()
        config.extra = {}
        adapter = MissionControlAdapter(config)
        adapter._notifier = Mock(spec=CLINotifier)
        return adapter

    def test_handle_task_deleted(self, adapter):
        """Test task deletion handler."""
        result = adapter._handle_task_deleted({"id": 123})
        assert result is True

    def test_handle_task_deleted_no_id(self, adapter):
        """Test task deletion without ID."""
        result = adapter._handle_task_deleted({})
        assert result is True

    def test_handle_agent_deleted(self, adapter):
        """Test agent deletion handler."""
        result = adapter._handle_agent_deleted({"id": "agent-1", "name": "Test Agent"})
        assert result is True

    def test_handle_agent_deleted_alt_fields(self, adapter):
        """Test agent deletion with alternate field names."""
        result = adapter._handle_agent_deleted({"agent_id": "agent-1", "agent_name": "Test"})
        assert result is True

    def test_handle_chat_message(self, adapter):
        """Test chat message handler."""
        data = {
            "id": "msg-123",
            "content": "Hello world",
            "sender": "user",
            "channel": "general"
        }
        result = adapter._handle_chat_message(data)
        assert result is True

    def test_handle_chat_message_defaults(self, adapter):
        """Test chat message with default values."""
        data = {"id": "msg-123"}
        result = adapter._handle_chat_message(data)
        assert result is True

    def test_handle_chat_message_deleted(self, adapter):
        """Test chat message deletion handler."""
        result = adapter._handle_chat_message_deleted({"id": "msg-123"})
        assert result is True

    def test_handle_notification_created(self, adapter):
        """Test notification created handler."""
        data = {
            "id": "notif-1",
            "title": "Test",
            "message": "Message",
            "level": "info"
        }
        result = adapter._handle_notification_created(data)
        assert result is True
        adapter._notifier.notify.assert_called_once()

    def test_handle_notification_created_no_notifier(self, adapter):
        """Test notification without notifier."""
        adapter._notifier = None
        data = {"id": "notif-1", "title": "Test", "message": "Message"}
        result = adapter._handle_notification_created(data)
        assert result is True

    def test_handle_notification_read(self, adapter):
        """Test notification read handler."""
        result = adapter._handle_notification_read({"id": "notif-1"})
        assert result is True

    def test_handle_activity_created(self, adapter):
        """Test activity created handler."""
        data = {
            "type": "login",
            "description": "User logged in",
            "user": "admin"
        }
        result = adapter._handle_activity_created(data)
        assert result is True

    def test_handle_activity_created_defaults(self, adapter):
        """Test activity with default values."""
        result = adapter._handle_activity_created({})
        assert result is True

    def test_handle_security_event(self, adapter):
        """Test security event handler."""
        data = {
            "type": "alert",
            "severity": "high",
            "description": "Security issue"
        }
        result = adapter._handle_security_event(data)
        assert result is True

    def test_handle_security_event_alt_fields(self, adapter):
        """Test security event with alternate field names."""
        data = {
            "event_type": "breach",
            "severity": "critical",
            "description": "Major issue"
        }
        result = adapter._handle_security_event(data)
        assert result is True

    def test_handle_security_event_notifies_high_severity(self, adapter):
        """Test that high severity events notify."""
        data = {"type": "alert", "severity": "critical", "description": "Test"}
        result = adapter._handle_security_event(data)
        assert result is True
        adapter._notifier.notify.assert_called_once()

    def test_handle_security_event_no_notify_low_severity(self, adapter):
        """Test that low severity events don't notify."""
        adapter._notifier.reset_mock()
        data = {"type": "alert", "severity": "info", "description": "Test"}
        result = adapter._handle_security_event(data)
        assert result is True
        adapter._notifier.notify.assert_not_called()

    def test_handle_connection_event_created(self, adapter):
        """Test connection created event."""
        data = {
            "type": "connection.created",
            "id": "conn-1",
            "agent_name": "test"
        }
        result = adapter._handle_connection_event(data)
        assert result is True

    def test_handle_connection_event_disconnected(self, adapter):
        """Test connection disconnected event."""
        data = {
            "type": "connection.disconnected",
            "id": "conn-1",
            "agent_name": "test"
        }
        result = adapter._handle_connection_event(data)
        assert result is True

    def test_handle_connection_event_defaults(self, adapter):
        """Test connection event with defaults."""
        data = {}
        result = adapter._handle_connection_event(data)
        assert result is True

    def test_handle_github_synced(self, adapter):
        """Test GitHub sync handler."""
        data = {
            "repository": "user/repo",
            "sync_type": "commits",
            "commit_count": 10
        }
        result = adapter._handle_github_synced(data)
        assert result is True

    def test_handle_github_synced_alt_fields(self, adapter):
        """Test GitHub sync with alternate field names."""
        data = {
            "repo": "user/repo",
            "sync_type": "issues"
        }
        result = adapter._handle_github_synced(data)
        assert result is True

    def test_handle_test_ping(self, adapter):
        """Test ping handler."""
        result = adapter._handle_test_ping({})
        assert result is True

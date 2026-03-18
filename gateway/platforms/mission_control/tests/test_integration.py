"""Integration tests for Mission Control adapter.

These tests verify end-to-end webhook processing with real HTTP requests.
"""

import pytest
import json
import hmac
import hashlib
import tempfile
import os
from unittest.mock import Mock, patch

# Mark all tests as async
pytestmark = pytest.mark.asyncio


class TestWebhookIntegration:
    """Integration tests using real aiohttp server."""

    async def test_full_webhook_flow_task_created(self, aiohttp_client):
        """Test complete task creation webhook flow."""
        from aiohttp import web
        from ..adapter import MissionControlAdapter
        from ..database import MissionControlDatabase
        from ..notifications import CLINotifier
        from ..task_manager import TaskManager
        
        # Create adapter with temp database
        config = Mock()
        config.extra = {}
        
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        
        try:
            adapter = MissionControlAdapter(config)
            adapter._db_path = db_path
            adapter._secret = "test-secret"
            adapter._port = 8889  # Use different port for tests
            
            # Initialize components
            adapter._db = MissionControlDatabase(db_path)
            adapter._notifier = CLINotifier()
            adapter._task_manager = TaskManager(
                adapter._db,
                adapter._notifier,
                "hermes-cli",
                auto_accept=True
            )
            
            # Create aiohttp app
            from aiohttp import web
            app = web.Application()
            app.router.add_post("/webhooks/mc", adapter._handle_webhook)
            
            client = await aiohttp_client(app)
            
            # Prepare webhook payload
            payload = {
                "event": "task.created",
                "timestamp": 1710774000,
                "data": {
                    "id": 12345,
                    "title": "Integration Test Task",
                    "description": "Test description",
                    "status": "pending",
                    "priority": "high",
                    "assigned_to": "hermes-cli",
                    "project_id": 1,
                    "workspace_id": 1
                }
            }
            
            body = json.dumps(payload, separators=(',', ':')).encode('utf-8')
            signature = hmac.new(
                b"test-secret",
                body,
                hashlib.sha256
            ).hexdigest()
            
            # Send request
            resp = await client.post(
                "/webhooks/mc",
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "X-MC-Signature": f"sha256={signature}"
                }
            )
            
            assert resp.status == 200
            
            # Verify task was stored
            task = adapter._db.get_task(12345)
            assert task is not None
            assert task["title"] == "Integration Test Task"
            assert task["assigned_to"] == "hermes-cli"
            
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    async def test_webhook_invalid_signature(self, aiohttp_client):
        """Test webhook rejected with invalid signature."""
        from aiohttp import web
        from unittest.mock import Mock
        from ..adapter import MissionControlAdapter
        
        config = Mock()
        config.extra = {}
        
        adapter = MissionControlAdapter(config)
        adapter._secret = "test-secret"
        adapter._db = Mock()
        adapter._task_manager = Mock()
        
        app = web.Application()
        app.router.add_post("/webhooks/mc", adapter._handle_webhook)
        
        client = await aiohttp_client(app)
        
        payload = {"event": "test", "timestamp": 123, "data": {}}
        body = json.dumps(payload).encode()
        
        resp = await client.post(
            "/webhooks/mc",
            data=body,
            headers={
                "Content-Type": "application/json",
                "X-MC-Signature": "sha256=invalid"
            }
        )
        
        assert resp.status == 401

    async def test_webhook_missing_signature(self, aiohttp_client):
        """Test webhook rejected without signature."""
        from aiohttp import web
        from unittest.mock import Mock
        from ..adapter import MissionControlAdapter
        
        config = Mock()
        config.extra = {}
        
        adapter = MissionControlAdapter(config)
        adapter._secret = "test-secret"
        adapter._db = Mock()
        adapter._task_manager = Mock()
        
        app = web.Application()
        app.router.add_post("/webhooks/mc", adapter._handle_webhook)
        
        client = await aiohttp_client(app)
        
        payload = {"event": "test", "timestamp": 123, "data": {}}
        body = json.dumps(payload).encode()
        
        resp = await client.post(
            "/webhooks/mc",
            data=body,
            headers={"Content-Type": "application/json"}
        )
        
        assert resp.status == 401

    async def test_webhook_duplicate_event(self, aiohttp_client):
        """Test duplicate events are ignored."""
        from aiohttp import web
        from unittest.mock import Mock
        from ..adapter import MissionControlAdapter
        from ..database import MissionControlDatabase
        
        config = Mock()
        config.extra = {}
        
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        
        try:
            adapter = MissionControlAdapter(config)
            adapter._db_path = db_path
            adapter._secret = "test-secret"
            adapter._db = MissionControlDatabase(db_path)
            adapter._notifier = Mock()
            adapter._task_manager = Mock()
            adapter._task_manager.generate_event_id = Mock(return_value="event:123:abc")
            adapter._task_manager.is_duplicate = Mock(return_value=True)
            
            app = web.Application()
            app.router.add_post("/webhooks/mc", adapter._handle_webhook)
            
            client = await aiohttp_client(app)
            
            payload = {
                "event": "task.created",
                "timestamp": 1710774000,
                "data": {"id": 123}
            }
            body = json.dumps(payload, separators=(',', ':')).encode()
            signature = hmac.new(b"test-secret", body, hashlib.sha256).hexdigest()
            
            resp = await client.post(
                "/webhooks/mc",
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "X-MC-Signature": f"sha256={signature}"
                }
            )
            
            assert resp.status == 200
            text = await resp.text()
            assert "Already processed" in text
            
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    async def test_webhook_invalid_json(self, aiohttp_client):
        """Test webhook with invalid JSON body."""
        from aiohttp import web
        from unittest.mock import Mock
        from ..adapter import MissionControlAdapter
        
        config = Mock()
        config.extra = {}
        
        adapter = MissionControlAdapter(config)
        adapter._secret = "test-secret"
        adapter._db = Mock()
        adapter._task_manager = Mock()
        
        app = web.Application()
        app.router.add_post("/webhooks/mc", adapter._handle_webhook)
        
        client = await aiohttp_client(app)
        
        body = b"not valid json"
        signature = hmac.new(b"test-secret", body, hashlib.sha256).hexdigest()
        
        resp = await client.post(
            "/webhooks/mc",
            data=body,
            headers={
                "Content-Type": "application/json",
                "X-MC-Signature": f"sha256={signature}"
            }
        )
        
        assert resp.status == 400

    async def test_webhook_large_payload(self, aiohttp_client):
        """Test webhook with payload exceeding size limit."""
        from aiohttp import web
        from unittest.mock import Mock, patch
        from ..adapter import MissionControlAdapter
        
        config = Mock()
        config.extra = {}
        
        adapter = MissionControlAdapter(config)
        adapter._secret = "test-secret"
        adapter._db = Mock()
        adapter._task_manager = Mock()
        
        app = web.Application()
        app.router.add_post("/webhooks/mc", adapter._handle_webhook)
        
        client = await aiohttp_client(app)
        
        # Create large payload (> 1MB)
        large_data = "x" * (1024 * 1024 + 1000)
        payload = {"event": "test", "data": large_data}
        body = json.dumps(payload).encode()
        
        # Mock content length to exceed limit
        resp = await client.post(
            "/webhooks/mc",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
                "X-MC-Signature": "sha256=test"
            }
        )
        
        assert resp.status == 413

    async def test_health_endpoint(self, aiohttp_client):
        """Test health check endpoint."""
        from aiohttp import web
        from unittest.mock import Mock
        from ..adapter import MissionControlAdapter
        
        config = Mock()
        config.extra = {}
        
        adapter = MissionControlAdapter(config)
        adapter._running = True
        
        app = web.Application()
        app.router.add_get("/health", adapter._handle_health)
        
        client = await aiohttp_client(app)
        
        resp = await client.get("/health")
        assert resp.status == 200
        
        data = await resp.json()
        assert data["status"] == "ok"
        assert data["platform"] == "mission_control"
        assert data["running"] is True

    async def test_all_event_types(self, aiohttp_client):
        """Test all supported event types return 200."""
        from aiohttp import web
        from unittest.mock import Mock
        from ..adapter import MissionControlAdapter
        
        config = Mock()
        config.extra = {}
        
        adapter = MissionControlAdapter(config)
        adapter._secret = "test-secret"
        adapter._db = Mock()
        adapter._db.log_webhook_delivery = Mock()
        adapter._db.mark_event_processed = Mock()
        adapter._notifier = Mock()
        adapter._task_manager = Mock()
        adapter._task_manager.generate_event_id = Mock(return_value="test:123")
        adapter._task_manager.is_duplicate = Mock(return_value=False)
        adapter._task_manager.handle_unknown_event = Mock(return_value=True)
        
        app = web.Application()
        app.router.add_post("/webhooks/mc", adapter._handle_webhook)
        
        client = await aiohttp_client(app)
        
        events = [
            "task.created",
            "task.updated",
            "task.status_changed",
            "task.deleted",
            "activity.task_created",
            "activity.task_updated",
            "activity.task_status_changed",
            "activity.task_deleted",
            "agent.created",
            "agent.updated",
            "agent.deleted",
            "agent.synced",
            "agent.status_changed",
            "agent.status_change",
            "agent.error",
            "chat.message",
            "chat.message.deleted",
            "notification.created",
            "notification.read",
            "activity.created",
            "audit.security",
            "security.event",
            "connection.created",
            "connection.disconnected",
            "github.synced",
            "test.ping",
        ]
        
        for event in events:
            payload = {
                "event": event,
                "timestamp": 1710774000,
                "data": {"id": "test-123", "test": True}
            }
            body = json.dumps(payload, separators=(',', ':')).encode()
            signature = hmac.new(b"test-secret", body, hashlib.sha256).hexdigest()
            
            resp = await client.post(
                "/webhooks/mc",
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "X-MC-Signature": f"sha256={signature}"
                }
            )
            
            assert resp.status == 200, f"Event {event} failed with {resp.status}"

"""Tests for Mission Control webhook adapter."""

import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock

from ..adapter import MissionControlAdapter, check_mc_requirements
from ..database import MissionControlDatabase
from ..notifications import CLINotifier
from ..task_manager import TaskManager


class TestCheckMCRequirements:
    """Test adapter requirements checking."""

    def test_aiohttp_available(self):
        """Test passes when aiohttp is available."""
        assert check_mc_requirements() is True


class TestMissionControlAdapter:
    """Test webhook adapter functionality."""

    @pytest.fixture
    def adapter(self):
        """Create adapter with mocked config."""
        config = Mock()
        config.extra = {}
        
        adapter = MissionControlAdapter(config)
        
        # Use temp database
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            adapter._db_path = f.name
        
        yield adapter
        
        # Cleanup
        if os.path.exists(adapter._db_path):
            os.unlink(adapter._db_path)

    def test_init_default_values(self, adapter):
        """Test adapter initializes with defaults."""
        assert adapter._port == 8888
        assert adapter._path == "/webhooks/mc"
        assert adapter._agent_name == "hermes-cli"
        assert adapter._auto_accept is True

    @patch.dict(os.environ, {"MC_WEBHOOK_PORT": "9999"})
    def test_init_env_port(self):
        """Test port from environment variable."""
        config = Mock()
        config.extra = {}
        
        adapter = MissionControlAdapter(config)
        assert adapter._port == 9999

    @patch.dict(os.environ, {"MC_AGENT_NAME": "custom-agent", "MC_AUTO_ACCEPT": "false"})
    def test_init_env_agent(self):
        """Test agent config from environment."""
        config = Mock()
        config.extra = {}
        
        adapter = MissionControlAdapter(config)
        assert adapter._agent_name == "custom-agent"
        assert adapter._auto_accept is False

    def test_hash_payload(self, adapter):
        """Test payload hashing."""
        body = b'{"test": "data"}'
        hash1 = adapter._hash_payload(body)
        hash2 = adapter._hash_payload(body)
        
        assert len(hash1) == 16  # Truncated to 16 chars
        assert hash1 == hash2  # Deterministic

    def test_hash_payload_different(self, adapter):
        """Test different payloads produce different hashes."""
        hash1 = adapter._hash_payload(b'{"a": 1}')
        hash2 = adapter._hash_payload(b'{"a": 2}')
        
        assert hash1 != hash2


class TestAdapterEventRouting:
    """Test event routing to handlers."""

    @pytest.fixture
    async def adapter(self):
        """Create adapter with mocked components."""
        config = Mock()
        config.extra = {}
        
        adapter = MissionControlAdapter(config)
        
        # Mock components
        adapter._db = Mock(spec=MissionControlDatabase)
        adapter._notifier = Mock(spec=CLINotifier)
        adapter._task_manager = Mock(spec=TaskManager)
        
        return adapter

    @pytest.mark.asyncio
    async def test_route_task_created(self, adapter):
        """Test routing task.created event."""
        adapter._task_manager.handle_task_created = Mock(return_value=True)
        
        result = await adapter._route_event("activity.task_created", {"id": 123})
        
        assert result is True
        adapter._task_manager.handle_task_created.assert_called_once_with({"id": 123})

    @pytest.mark.asyncio
    async def test_route_task_updated(self, adapter):
        """Test routing task.updated event."""
        adapter._task_manager.handle_task_updated = Mock(return_value=True)
        
        result = await adapter._route_event("activity.task_updated", {"id": 123})
        
        assert result is True
        adapter._task_manager.handle_task_updated.assert_called_once_with({"id": 123})

    @pytest.mark.asyncio
    async def test_route_unknown_event(self, adapter):
        """Test routing unknown event."""
        adapter._task_manager.handle_unknown_event = Mock(return_value=True)
        
        result = await adapter._route_event("custom.unknown", {"data": "test"})
        
        assert result is True
        adapter._task_manager.handle_unknown_event.assert_called_once()


class TestAdapterHealth:
    """Test health check endpoint."""

    @pytest.fixture
    async def adapter(self):
        """Create running adapter."""
        config = Mock()
        config.extra = {}
        
        adapter = MissionControlAdapter(config)
        adapter._running = True
        
        return adapter

    @pytest.mark.asyncio
    async def test_health_endpoint_running(self, adapter):
        """Test health check when running."""
        # Create mock request
        request = Mock()
        
        response = await adapter._handle_health(request)
        
        assert response.status == 200
        body = json.loads(response.text)
        assert body["status"] == "ok"
        assert body["running"] is True

    @pytest.mark.asyncio
    async def test_health_endpoint_stopped(self, adapter):
        """Test health check when stopped."""
        adapter._running = False
        request = Mock()
        
        response = await adapter._handle_health(request)
        
        body = json.loads(response.text)
        assert body["running"] is False


class TestAdapterIntegration:
    """Integration tests requiring full setup."""

    @pytest.mark.skip(reason="Requires full async setup")
    @pytest.mark.asyncio
    async def test_full_webhook_flow(self):
        """Test complete webhook processing flow."""
        # This would test the full aiohttp server integration
        # Skipped in unit tests, covered by integration tests
        pass

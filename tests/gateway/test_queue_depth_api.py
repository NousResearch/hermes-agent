"""Tests for get_queue_depth and get_queue_status public API on GatewayRunner."""

from unittest.mock import MagicMock

from gateway.run import GatewayRunner


class TestQueueDepthAPI:
    """Tests for the public queue depth/status API."""

    def _make_runner(self, adapters=None, queued_events=None):
        """Create a GatewayRunner with optional pre-set state."""
        runner = GatewayRunner.__new__(GatewayRunner)
        runner.adapters = adapters or {}
        runner._queued_events = queued_events or {}
        return runner

    def test_get_queue_depth_returns_none_when_no_adapter_no_queue(self):
        """Session not found → None."""
        runner = self._make_runner(adapters={}, queued_events={})
        result = runner.get_queue_depth("feishu:user:chat")
        assert result is None

    def test_get_queue_depth_returns_zero_when_empty_queue(self):
        """Adapter exists but queue is empty → 0."""
        mock_adapter = MagicMock()
        mock_adapter._pending_messages = {}
        runner = self._make_runner(
            adapters={"feishu": mock_adapter},
            queued_events={},
        )
        result = runner.get_queue_depth("feishu:user:chat")
        assert result == 0

    def test_get_queue_depth_counts_slot_message(self):
        """Adapter has slot message → depth includes it."""
        mock_adapter = MagicMock()
        mock_adapter._pending_messages = {"feishu:user:chat": MagicMock()}
        runner = self._make_runner(
            adapters={"feishu": mock_adapter},
            queued_events={},
        )
        result = runner.get_queue_depth("feishu:user:chat")
        assert result == 1

    def test_get_queue_depth_counts_overflow_only(self):
        """Overflow has 2 items, no slot → depth is 2."""
        runner = self._make_runner(
            adapters={},
            queued_events={"feishu:user:chat": [MagicMock(), MagicMock()]},
        )
        result = runner.get_queue_depth("feishu:user:chat")
        assert result == 2

    def test_get_queue_depth_counts_slot_plus_overflow(self):
        """Slot occupied + 3 overflow → depth is 4."""
        mock_adapter = MagicMock()
        mock_adapter._pending_messages = {"feishu:user:chat": MagicMock()}
        runner = self._make_runner(
            adapters={"feishu": mock_adapter},
            queued_events={"feishu:user:chat": [MagicMock(), MagicMock(), MagicMock()]},
        )
        result = runner.get_queue_depth("feishu:user:chat")
        assert result == 4


class TestQueueStatusAPI:
    """Tests for the richer get_queue_status method."""

    def _make_runner(self, adapters=None, queued_events=None):
        runner = GatewayRunner.__new__(GatewayRunner)
        runner.adapters = adapters or {}
        runner._queued_events = queued_events or {}
        return runner

    def test_get_queue_status_returns_none_when_not_found(self):
        """Session with no adapter and no queue → None."""
        runner = self._make_runner(adapters={}, queued_events={})
        result = runner.get_queue_status("feishu:user:chat")
        assert result is None

    def test_get_queue_status_empty_queue(self):
        """Adapter exists, empty queue → status with depth 0."""
        mock_adapter = MagicMock()
        mock_adapter._pending_messages = {}
        runner = self._make_runner(
            adapters={"feishu": mock_adapter},
            queued_events={},
        )
        result = runner.get_queue_status("feishu:user:chat")
        assert result is not None
        assert result["session_key"] == "feishu:user:chat"
        assert result["depth"] == 0
        assert result["has_slot_message"] is False
        assert result["overflow_count"] == 0

    def test_get_queue_status_with_slot_and_overflow(self):
        """Slot occupied + 2 overflow → correct structured status."""
        mock_adapter = MagicMock()
        mock_adapter._pending_messages = {"feishu:user:chat": MagicMock()}
        runner = self._make_runner(
            adapters={"feishu": mock_adapter},
            queued_events={"feishu:user:chat": [MagicMock(), MagicMock()]},
        )
        result = runner.get_queue_status("feishu:user:chat")
        assert result["depth"] == 3
        assert result["has_slot_message"] is True
        assert result["overflow_count"] == 2

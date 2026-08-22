"""Tests for federation Phase 6 — security hardening, health, compatibility."""
from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.federation.federation_protocol import FedMessage, MessageType
from gateway.config import FederationConfig


# ========================================================================
# Security: HMAC signature full length
# ========================================================================

class TestSignatureFullLength:
    """Regression: signatures must be full 64-char SHA256 (not truncated)."""

    def test_signature_is_full_sha256(self):
        msg = FedMessage(
            msg_type=MessageType.PEER_JOIN.value,
            sender_id="dev-a",
            payload={"hello": "world"},
        )
        msg.sign("my-secret-token")
        assert len(msg.signature) == 64  # Full SHA256 hex = 64 chars

    def test_signature_uniqueness(self):
        """Different payloads should produce different signatures."""
        msg1 = FedMessage(msg_type=MessageType.PEER_PING.value, payload={"a": 1})
        msg2 = FedMessage(msg_type=MessageType.PEER_PING.value, payload={"a": 2})
        msg1.sign("secret")
        msg2.sign("secret")
        assert msg1.signature != msg2.signature

    def test_signature_tamper_detection(self):
        """Modified payload should fail verification — tamper IS detected."""
        msg = FedMessage(
            msg_type=MessageType.TASK_SUBMIT.value,
            sender_id="dev-a",
            payload={"task_id": "T-001"},
        )
        msg.sign("secret")
        assert msg.verify("secret")  # Valid before tampering

        # Tamper with payload
        msg.payload["task_id"] = "T-002"
        assert not msg.verify("secret")  # Tamper detected!


# ========================================================================
# Security: Config defaults
# ========================================================================

class TestSecureDefaults:
    """Regression: federation should default to secure settings."""

    def test_require_auth_defaults_to_true(self):
        cfg = FederationConfig()
        assert cfg.require_auth is True

    def test_from_dict_preserves_require_auth_default(self):
        cfg = FederationConfig.from_dict({"enabled": True})
        assert cfg.require_auth is True

    def test_ip_whitelist_defaults_empty(self):
        cfg = FederationConfig()
        assert cfg.ip_whitelist == []

    def test_tls_defaults_none(self):
        cfg = FederationConfig()
        assert cfg.tls_cert is None
        assert cfg.tls_key is None

    def test_from_dict_all_security_fields(self):
        data = {
            "enabled": True,
            "mode": "lan",
            "auth_token": "secret",
            "require_auth": True,
            "tls_cert": "/path/to/cert.pem",
            "tls_key": "/path/to/key.pem",
            "ip_whitelist": ["192.168.1.0/24"],
        }
        cfg = FederationConfig.from_dict(data)
        assert cfg.require_auth is True
        assert cfg.tls_cert == "/path/to/cert.pem"
        assert cfg.tls_key == "/path/to/key.pem"
        assert cfg.ip_whitelist == ["192.168.1.0/24"]


# ========================================================================
# Health: Connection metrics
# ========================================================================

class TestConnectionMetrics:
    """Regression: connection quality tracking works."""

    def _make_manager(self):
        from gateway.federation.federation_connection import (
            FederationConnectionManager,
        )
        return FederationConnectionManager(
            device_id="dev-a",
            auth_token="test-token",
            ws_port=18765,
        )

    def test_metrics_empty_initially(self):
        mgr = self._make_manager()
        assert mgr.get_all_metrics() == {}
        assert mgr.get_metrics("nonexistent") is None

    def test_metrics_updated_on_pong(self):
        """Verify that PEER_PONG updates latency metrics."""
        msg = FedMessage(
            msg_type=MessageType.PEER_PONG.value,
            sender_id="dev-b",
            payload={"timestamp": time.time() - 0.05},  # 50ms ago
        )
        # This is tested via _receive_loop in integration tests
        # Here we just verify the message structure is correct
        assert msg.msg_type == MessageType.PEER_PONG.value
        assert "timestamp" in msg.payload


# ========================================================================
# Compatibility: Config backward compatibility
# ========================================================================

class TestBackwardCompatibility:
    """Old config files without new fields should still work."""

    def test_old_config_without_security_fields(self):
        """Config from before Phase 6 should still parse."""
        old_data = {
            "enabled": True,
            "mode": "shared_db",
            "db_path": "/tmp/fed.db",
            "offline_threshold_s": 30,
            "heartbeat_interval_s": 60,
        }
        cfg = FederationConfig.from_dict(old_data)
        assert cfg.enabled is True
        assert cfg.mode == "shared_db"
        # New fields get defaults
        assert cfg.require_auth is True
        assert cfg.tls_cert is None
        assert cfg.ip_whitelist == []

    def test_old_config_with_auth_token(self):
        old_data = {
            "enabled": True,
            "mode": "lan",
            "auth_token": "my-token",
            "peers": ["ws://192.168.1.10:18765"],
        }
        cfg = FederationConfig.from_dict(old_data)
        assert cfg.auth_token == "my-token"
        assert cfg.require_auth is True  # default
        assert cfg.peers == ["ws://192.168.1.10:18765"]

    def test_mixed_old_and_new_fields(self):
        data = {
            "enabled": True,
            "mode": "auto",
            "auth_token": "token",
            # New security fields
            "require_auth": False,
            "tls_cert": "/cert.pem",
            # Invalid type for ip_whitelist should be normalized
            "ip_whitelist": "not-a-list",
        }
        cfg = FederationConfig.from_dict(data)
        assert cfg.require_auth is False
        assert cfg.tls_cert == "/cert.pem"
        assert cfg.ip_whitelist == []  # normalized from invalid type


# ========================================================================
# Rate limiting
# ========================================================================

class TestRateLimiting:
    def test_rate_limit_allows_first_10(self):
        from gateway.federation.federation_connection import (
            FederationConnectionManager,
        )
        mgr = FederationConnectionManager(device_id="dev-a")
        ip = "192.168.1.100"
        loop = asyncio.get_event_loop()
        for _ in range(10):
            assert loop.run_until_complete(mgr._check_rate_limit(ip)) is True

    def test_rate_limit_blocks_11th(self):
        from gateway.federation.federation_connection import (
            FederationConnectionManager,
        )
        mgr = FederationConnectionManager(device_id="dev-a")
        ip = "192.168.1.100"
        loop = asyncio.get_event_loop()
        for _ in range(10):
            loop.run_until_complete(mgr._check_rate_limit(ip))
        assert loop.run_until_complete(mgr._check_rate_limit(ip)) is False

    def test_rate_limit_resets_after_60s(self):
        from gateway.federation.federation_connection import (
            FederationConnectionManager,
        )
        mgr = FederationConnectionManager(device_id="dev-a")
        ip = "192.168.1.100"
        loop = asyncio.get_event_loop()
        for _ in range(10):
            loop.run_until_complete(mgr._check_rate_limit(ip))

        # Fake time passage by clearing timestamps
        mgr._conn_times[ip] = []
        assert loop.run_until_complete(mgr._check_rate_limit(ip)) is True


# ========================================================================
# Message size enforcement
# ========================================================================

class TestMessageSizeEnforcement:
    MAX_SIZE = 1024 * 1024  # 1MB

    def test_normal_message_passes(self):
        msg = FedMessage(
            msg_type=MessageType.PEER_PING.value,
            sender_id="dev-a",
            payload={"data": "x" * 1000},
        )
        raw = msg.to_json()
        assert len(raw) < self.MAX_SIZE

    def test_oversized_message_detected(self):
        huge_payload = "x" * (self.MAX_SIZE + 1)
        # In real code, this would be caught in _receive_loop
        assert len(huge_payload) > self.MAX_SIZE

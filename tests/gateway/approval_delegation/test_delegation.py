"""Tests for gateway.approval_delegation — native approval delegation."""

import time
from unittest.mock import patch, MagicMock

import pytest


# ── Config tests ────────────────────────────────────────────────────────


class TestDelegationConfig:
    """Test config loading and admin lookup."""

    def _reset_config(self):
        """Reset the cached config between tests."""
        import gateway.approval_delegation as mod
        mod._delegation_config = None

    def test_delegation_disabled_by_default(self):
        """Delegation is disabled when config has no delegation section."""
        self._reset_config()
        with patch("hermes_cli.config.load_config", return_value={}):
            from gateway.approval_delegation import is_delegation_enabled
            assert is_delegation_enabled() is False

    def test_delegation_enabled(self):
        """Delegation is enabled when config says so."""
        self._reset_config()
        config = {
            "approvals": {
                "delegation": {
                    "enabled": True,
                    "admins": [
                        {"platform": "feishu", "user_id": "admin1"}
                    ]
                }
            }
        }
        with patch("hermes_cli.config.load_config", return_value=config):
            from gateway.approval_delegation import is_delegation_enabled, get_admins
            assert is_delegation_enabled() is True
            admins = get_admins()
            assert len(admins) == 1
            assert admins[0]["platform"] == "feishu"
            assert admins[0]["user_id"] == "admin1"

    def test_admin_user_detection(self):
        """is_admin_user correctly identifies admins."""
        self._reset_config()
        config = {
            "approvals": {
                "delegation": {
                    "enabled": True,
                    "admins": [
                        {"platform": "feishu", "user_id": "admin1"},
                        {"platform": "weixin", "user_id": "admin2"},
                    ]
                }
            }
        }
        with patch("hermes_cli.config.load_config", return_value=config):
            from gateway.approval_delegation import is_admin_user
            assert is_admin_user("feishu", "admin1") is True
            assert is_admin_user("weixin", "admin2") is True
            assert is_admin_user("feishu", "regular_user") is False
            assert is_admin_user("telegram", "admin1") is False

    def test_admin_chat_id_defaults_to_user_id(self):
        """chat_id defaults to user_id when not specified."""
        self._reset_config()
        config = {
            "approvals": {
                "delegation": {
                    "enabled": True,
                    "admins": [
                        {"platform": "feishu", "user_id": "admin1"}
                    ]
                }
            }
        }
        with patch("hermes_cli.config.load_config", return_value=config):
            from gateway.approval_delegation import get_admins
            admins = get_admins()
            assert admins[0]["chat_id"] == "admin1"

    def test_admin_explicit_chat_id(self):
        """chat_id can be explicitly set."""
        self._reset_config()
        config = {
            "approvals": {
                "delegation": {
                    "enabled": True,
                    "admins": [
                        {"platform": "feishu", "user_id": "admin1", "chat_id": "oc_123"}
                    ]
                }
            }
        }
        with patch("hermes_cli.config.load_config", return_value=config):
            from gateway.approval_delegation import get_admins
            admins = get_admins()
            assert admins[0]["chat_id"] == "oc_123"

    def test_multiple_admins(self):
        """Multiple admins are loaded correctly."""
        self._reset_config()
        config = {
            "approvals": {
                "delegation": {
                    "enabled": True,
                    "admins": [
                        {"platform": "feishu", "user_id": "admin1"},
                        {"platform": "weixin", "user_id": "admin2"},
                    ]
                }
            }
        }
        with patch("hermes_cli.config.load_config", return_value=config):
            from gateway.approval_delegation import get_admins
            admins = get_admins()
            assert len(admins) == 2
            assert admins[0]["platform"] == "feishu"
            assert admins[1]["platform"] == "weixin"


# ── Delegation state tests ──────────────────────────────────────────────


class TestDelegationState:
    """Test delegation registration, resolution, and cleanup."""

    def setup_method(self):
        """Clear delegation state before each test."""
        from gateway.approval_delegation import clear_all_delegations
        clear_all_delegations()

    def test_register_and_resolve(self):
        """Can register and resolve a delegation."""
        from gateway.approval_delegation import register_delegation, resolve_delegation

        register_delegation(
            admin_platform="feishu",
            admin_chat_id="oc_123",
            session_key="sk_abc",
            user_platform="weixin",
            user_chat_id="wx_user",
            command="rm -rf /",
            description="destructive delete",
        )

        entry = resolve_delegation("feishu", "oc_123")
        assert entry is not None
        assert entry["session_key"] == "sk_abc"
        assert entry["user_platform"] == "weixin"
        assert entry["user_chat_id"] == "wx_user"
        assert entry["command"] == "rm -rf /"

    def test_resolve_nonexistent(self):
        """Resolving a nonexistent delegation returns None."""
        from gateway.approval_delegation import resolve_delegation
        assert resolve_delegation("feishu", "nonexistent") is None

    def test_clear_delegation(self):
        """Clearing a delegation removes it."""
        from gateway.approval_delegation import (
            register_delegation, resolve_delegation, clear_delegation,
        )

        register_delegation(
            admin_platform="feishu",
            admin_chat_id="oc_123",
            session_key="sk_abc",
            user_platform="weixin",
            user_chat_id="wx_user",
        )

        clear_delegation("feishu", "oc_123", session_key="sk_abc")
        assert resolve_delegation("feishu", "oc_123") is None

    def test_stale_delegation_expires(self):
        """Delegations older than TTL are automatically pruned."""
        from gateway.approval_delegation import (
            register_delegation, resolve_delegation, _DELEGATION_TTL,
            _delegation_map,
        )

        register_delegation(
            admin_platform="feishu",
            admin_chat_id="oc_123",
            session_key="sk_abc",
            user_platform="weixin",
            user_chat_id="wx_user",
        )

        # Manually age the entry
        _delegation_map["feishu:oc_123"]["sk_abc"]["created_at"] = time.monotonic() - _DELEGATION_TTL - 1

        assert resolve_delegation("feishu", "oc_123") is None

    def test_concurrent_delegations_to_same_admin(self):
        """Multiple concurrent delegations to the same admin coexist."""
        from gateway.approval_delegation import (
            register_delegation, resolve_delegation,
        )

        register_delegation(
            admin_platform="feishu",
            admin_chat_id="oc_admin",
            session_key="sk_1",
            user_platform="weixin",
            user_chat_id="wx_1",
            command="cmd1",
        )
        register_delegation(
            admin_platform="feishu",
            admin_chat_id="oc_admin",
            session_key="sk_2",
            user_platform="telegram",
            user_chat_id="tg_2",
            command="cmd2",
        )

        # resolve_delegation returns the most recent
        entry = resolve_delegation("feishu", "oc_admin")
        assert entry is not None
        assert entry["session_key"] == "sk_2"

    def test_concurrent_delegations_independent_clear(self):
        """Clearing one delegation doesn't affect others."""
        from gateway.approval_delegation import (
            register_delegation, resolve_delegation, clear_delegation,
        )

        register_delegation(
            admin_platform="feishu",
            admin_chat_id="oc_admin",
            session_key="sk_1",
            user_platform="weixin",
            user_chat_id="wx_1",
        )
        register_delegation(
            admin_platform="feishu",
            admin_chat_id="oc_admin",
            session_key="sk_2",
            user_platform="telegram",
            user_chat_id="tg_2",
        )

        clear_delegation("feishu", "oc_admin", session_key="sk_1")

        # sk_2 should still be there
        entry = resolve_delegation("feishu", "oc_admin")
        assert entry is not None
        assert entry["session_key"] == "sk_2"

    def test_clear_all_delegations(self):
        """clear_all_delegations removes everything."""
        from gateway.approval_delegation import (
            register_delegation, resolve_delegation, clear_all_delegations,
        )

        register_delegation(
            admin_platform="feishu",
            admin_chat_id="oc_111",
            session_key="sk_1",
            user_platform="weixin",
            user_chat_id="wx_1",
        )
        register_delegation(
            admin_platform="feishu",
            admin_chat_id="oc_222",
            session_key="sk_2",
            user_platform="telegram",
            user_chat_id="tg_2",
        )

        clear_all_delegations()
        assert resolve_delegation("feishu", "oc_111") is None
        assert resolve_delegation("feishu", "oc_222") is None

    def test_multiple_admins_different_platforms(self):
        """Delegations to admins on different platforms are independent."""
        from gateway.approval_delegation import (
            register_delegation, resolve_delegation,
        )

        register_delegation(
            admin_platform="feishu",
            admin_chat_id="oc_feishu",
            session_key="sk_1",
            user_platform="weixin",
            user_chat_id="wx_1",
        )
        register_delegation(
            admin_platform="telegram",
            admin_chat_id="tg_admin",
            session_key="sk_2",
            user_platform="weixin",
            user_chat_id="wx_2",
        )

        e1 = resolve_delegation("feishu", "oc_feishu")
        e2 = resolve_delegation("telegram", "tg_admin")
        assert e1["session_key"] == "sk_1"
        assert e2["session_key"] == "sk_2"

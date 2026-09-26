"""Unit tests verifying Feishu card action approval operator authorization fail-closed behavior."""

import os
from unittest.mock import MagicMock
from plugins.platforms.feishu.adapter import FeishuAdapter


def _make_feishu_adapter(admins=None, allowed_group_users=None):
    adapter = object.__new__(FeishuAdapter)
    adapter._admins = set(admins) if admins else set()
    adapter._allowed_group_users = set(allowed_group_users) if allowed_group_users else set()
    return adapter


def test_feishu_approval_operator_authorized_with_explicit_allowlist():
    adapter = _make_feishu_adapter(admins=["ou_admin_123"])

    # Authorized admin user
    assert adapter._is_interactive_operator_authorized("ou_admin_123") is True
    # Unauthorized user
    assert adapter._is_interactive_operator_authorized("ou_stranger_999") is False


def test_feishu_approval_operator_fail_closed_when_no_allowlist_configured(monkeypatch):
    monkeypatch.delenv("FEISHU_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
    adapter = _make_feishu_adapter(admins=None, allowed_group_users=None)

    # Empty allowlist + no opt-in → FAIL CLOSED (False)
    assert adapter._is_interactive_operator_authorized("ou_user_123") is False


def test_feishu_approval_operator_allowed_with_feishu_allow_all_env(monkeypatch):
    monkeypatch.setenv("FEISHU_ALLOW_ALL_USERS", "true")
    monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
    adapter = _make_feishu_adapter(admins=None, allowed_group_users=None)

    # Empty allowlist + FEISHU_ALLOW_ALL_USERS=true → True
    assert adapter._is_interactive_operator_authorized("ou_user_123") is True


def test_feishu_approval_operator_allowed_with_gateway_allow_all_env(monkeypatch):
    monkeypatch.delenv("FEISHU_ALLOW_ALL_USERS", raising=False)
    monkeypatch.setenv("GATEWAY_ALLOW_ALL_USERS", "true")
    adapter = _make_feishu_adapter(admins=None, allowed_group_users=None)

    # Empty allowlist + GATEWAY_ALLOW_ALL_USERS=true → True
    assert adapter._is_interactive_operator_authorized("ou_user_123") is True

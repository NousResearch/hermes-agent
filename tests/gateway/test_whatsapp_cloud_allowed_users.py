"""Regression tests for PR #58448 salvage: the documented
WHATSAPP_CLOUD_ALLOWED_USERS / WHATSAPP_CLOUD_ALLOW_ALL_USERS env vars
must actually drive the DM intake gate.

Before the fix, the adapter only read WHATSAPP_CLOUD_ALLOW_FROM and the
dm_policy default was "open" (which fails closed without an allow-all
opt-in), so a wizard-configured install using the documented vars
silently dropped every inbound message.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from gateway.config import Platform


def _build_adapter(monkeypatch, env: dict[str, str], extra: dict | None = None):
    """Construct a real WhatsAppCloudAdapter through __init__ with env vars."""
    from gateway.platforms.whatsapp_cloud import WhatsAppCloudAdapter

    for var in (
        "WHATSAPP_CLOUD_ALLOW_FROM",
        "WHATSAPP_CLOUD_ALLOWED_USERS",
        "WHATSAPP_CLOUD_ALLOW_ALL_USERS",
        "WHATSAPP_CLOUD_DM_POLICY",
        "WHATSAPP_DM_POLICY",
        "GATEWAY_ALLOW_ALL_USERS",
        "WHATSAPP_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(var, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    config = MagicMock()
    config.extra = {
        "phone_number_id": "1234567890",
        "access_token": "test-token",
        **(extra or {}),
    }
    return WhatsAppCloudAdapter(config)


def _dm_message(sender: str) -> dict:
    return {"from": sender, "id": "wamid.test", "type": "text"}


def test_allowed_users_env_populates_allowlist_and_enforces_it(monkeypatch):
    adapter = _build_adapter(
        monkeypatch, {"WHATSAPP_CLOUD_ALLOWED_USERS": "15551234567"}
    )

    # The documented var must populate the allowlist...
    assert "15551234567" in adapter._allow_from
    # ...and flip the default dm_policy to allowlist so it is enforced.
    assert adapter._dm_policy == "allowlist"
    # Allowlisted sender passes the intake gate; others are dropped.
    assert adapter._is_dm_allowed("15551234567") is True
    assert adapter._is_dm_allowed("19998887777") is False


def test_allow_from_still_takes_precedence(monkeypatch):
    adapter = _build_adapter(
        monkeypatch,
        {
            "WHATSAPP_CLOUD_ALLOW_FROM": "15550000001",
            "WHATSAPP_CLOUD_ALLOWED_USERS": "15559999999",
        },
    )

    # Legacy ALLOW_FROM wins when both are set (documented precedence).
    assert "15550000001" in adapter._allow_from
    assert "15559999999" not in adapter._allow_from
    assert adapter._dm_allowlist_source == "WHATSAPP_CLOUD_ALLOW_FROM"
    assert adapter._is_dm_allowed("15550000001") is True
    assert adapter._is_dm_allowed("15559999999") is False


def test_explicit_config_beats_cloud_env_on_live_checks(monkeypatch):
    """Live DM auth must keep explicit config above both cloud env carriers."""
    adapter = _build_adapter(
        monkeypatch,
        {
            "WHATSAPP_CLOUD_ALLOW_FROM": "15550000002",
            "WHATSAPP_CLOUD_ALLOWED_USERS": "15550000003",
        },
        extra={"dm_policy": "allowlist", "allow_from": ["15550000001"]},
    )

    assert adapter._dm_allowlist_source == "config"
    assert adapter._is_dm_allowed("15550000001") is True
    assert adapter._is_dm_intake_allowed("15550000001") is True
    assert adapter._is_dm_allowed("15550000002") is False
    assert adapter._is_dm_allowed("15550000003") is False
    assert adapter._is_dm_intake_allowed("15550000002") is False

    # Mutating lower-precedence env must not replace the config allowlist.
    monkeypatch.setenv("WHATSAPP_CLOUD_ALLOWED_USERS", "15550000003,15550000004")
    assert adapter._is_dm_allowed("15550000001") is True
    assert adapter._is_dm_allowed("15550000003") is False
    assert adapter._is_dm_allowed("15550000004") is False


def test_explicit_empty_allow_from_blocks_cloud_env_grants(monkeypatch):
    """allow_from: [] is present config — conflicting cloud env must not authorize."""
    adapter = _build_adapter(
        monkeypatch,
        {
            "WHATSAPP_CLOUD_ALLOW_FROM": "15550000002",
            "WHATSAPP_CLOUD_ALLOWED_USERS": "15550000003",
        },
        extra={"dm_policy": "allowlist", "allow_from": []},
    )

    assert adapter._dm_allowlist_source == "config"
    assert adapter._allow_from == set()
    assert adapter._is_dm_allowed("15550000002") is False
    assert adapter._is_dm_allowed("15550000003") is False
    assert adapter._is_dm_intake_allowed("15550000002") is False
    assert adapter._is_dm_intake_allowed("15550000003") is False


def test_cloud_allowed_users_live_reread_when_env_seeded(monkeypatch):
    adapter = _build_adapter(
        monkeypatch,
        {"WHATSAPP_CLOUD_ALLOWED_USERS": "15551234567"},
    )

    assert adapter._dm_allowlist_source == "WHATSAPP_CLOUD_ALLOWED_USERS"
    assert adapter._is_dm_allowed("15551234567") is True

    monkeypatch.setenv("WHATSAPP_CLOUD_ALLOWED_USERS", "")
    assert adapter._is_dm_allowed("15551234567") is False


def test_cloud_live_allowlist_denies_when_env_key_removed(monkeypatch):
    """Sole-entry revoke removes the key — stale construction snapshot must not authorize."""
    adapter = _build_adapter(
        monkeypatch,
        {"WHATSAPP_CLOUD_ALLOWED_USERS": "15551234567"},
    )
    assert "15551234567" in adapter._allow_from

    monkeypatch.delenv("WHATSAPP_CLOUD_ALLOWED_USERS", raising=False)
    assert adapter._live_dm_allow_from() == set()
    assert adapter._is_dm_allowed("15551234567") is False

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


def test_allow_all_users_env_opts_into_open_dms(monkeypatch):
    adapter = _build_adapter(
        monkeypatch, {"WHATSAPP_CLOUD_ALLOW_ALL_USERS": "true"}
    )

    assert adapter._dm_policy == "open"
    assert adapter._open_dm_opted_in() is True
    assert adapter._is_dm_allowed("19998887777") is True


def test_explicit_dm_policy_still_wins_over_derived_default(monkeypatch):
    adapter = _build_adapter(
        monkeypatch,
        {
            "WHATSAPP_CLOUD_ALLOWED_USERS": "15551234567",
            "WHATSAPP_CLOUD_DM_POLICY": "disabled",
        },
    )

    # Operator's explicit policy beats the allowlist-derived default.
    assert adapter._dm_policy == "disabled"


def test_unconfigured_default_unchanged(monkeypatch):
    adapter = _build_adapter(monkeypatch, {})

    # No allowlist, no opt-in: default stays "open" (which fails closed
    # in the shared mixin without an allow-all opt-in) — pre-fix behavior
    # for unconfigured installs is preserved.
    assert adapter._dm_policy == "open"
    assert adapter._allow_from == set()
    assert adapter._open_dm_opted_in() is False


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

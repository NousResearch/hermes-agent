"""Regression tests for PR #58448 salvage: the documented
WHATSAPP_CLOUD_ALLOWED_USERS / WHATSAPP_CLOUD_ALLOW_ALL_USERS env vars
must actually drive the DM intake gate.

Before the fix, the adapter only read WHATSAPP_CLOUD_ALLOW_FROM and the
dm_policy default was "open" (which fails closed without an allow-all
opt-in), so a wizard-configured install using the documented vars
silently dropped every inbound message.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform


def _build_adapter(monkeypatch, env: dict[str, str], extra: dict | None = None):
    """Construct a real WhatsAppCloudAdapter through __init__ with env vars."""
    from plugins.platforms.whatsapp_cloud.adapter import WhatsAppCloudAdapter

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


_CLOUD_ALLOWLIST_VARS = (
    "WHATSAPP_CLOUD_ALLOW_FROM",
    "WHATSAPP_CLOUD_ALLOWED_USERS",
)
_BSUID = "DE.1A2B3C4D5E6F7G"
_OTHER_BSUID = "DE.1234567ABCDEFG"


async def _dispatch_cloud_message(adapter, sender, wamid, *, button_id=None):
    message = {"from": sender, "id": wamid, "type": "text", "text": {"body": "hello"}}
    if button_id is not None:
        message.update(type="interactive", interactive={
            "type": "button_reply",
            "button_reply": {"id": button_id, "title": "1"},
        })
    await adapter._dispatch_payload({
        "object": "whatsapp_business_account",
        "entry": [{"changes": [{"field": "messages", "value": {
            "metadata": {"phone_number_id": "1234567890"},
            "messages": [message],
        }}]}],
    })


@pytest.mark.parametrize("env_var", _CLOUD_ALLOWLIST_VARS)
@pytest.mark.asyncio
async def test_suffixed_bsuid_live_allowlist_message_intake(monkeypatch, env_var):
    adapter = _build_adapter(monkeypatch, {
        env_var: f"{_BSUID}@s.whatsapp.net,+1 (555) 123-4567@s.whatsapp.net",
    })
    adapter.handle_message = AsyncMock()

    async def expect_intake(sender, wamid, allowed):
        adapter.handle_message.reset_mock()
        await _dispatch_cloud_message(adapter, sender, wamid)
        if allowed:
            adapter.handle_message.assert_awaited_once()
            assert adapter.handle_message.call_args.args[0].source.user_id == sender
        else:
            adapter.handle_message.assert_not_awaited()

    await expect_intake(_BSUID, "initial", True)
    await expect_intake(_OTHER_BSUID, "collision", False)
    await expect_intake(_BSUID.replace("A", "a"), "case", False)
    await expect_intake("1234567", "projection", False)
    await expect_intake("15551234567", "phone", True)

    monkeypatch.setenv(env_var, f"{_OTHER_BSUID}@s.whatsapp.net")
    await expect_intake(_OTHER_BSUID, "update", True)
    await expect_intake(_BSUID, "old", False)
    await expect_intake("15551234567", "phone-revoked", False)
    monkeypatch.setenv(env_var, "")
    await expect_intake(_OTHER_BSUID, "empty", False)
    monkeypatch.setenv(env_var, "*@s.whatsapp.net")
    await expect_intake(_BSUID, "wildcard", True)
    monkeypatch.delenv(env_var)
    await expect_intake(_BSUID, "removed", False)


@pytest.mark.parametrize("env_var", _CLOUD_ALLOWLIST_VARS)
@pytest.mark.asyncio
async def test_suffixed_bsuid_live_allowlist_interactive_dispatch(monkeypatch, env_var):
    adapter = _build_adapter(monkeypatch, {env_var: f"{_BSUID}@s.whatsapp.net"})
    adapter.handle_message = AsyncMock()
    resolver = MagicMock(return_value=True)
    monkeypatch.setattr("tools.clarify_gateway.resolve_gateway_clarify", resolver)

    async def expect_tap(sender, prompt_id, allowed):
        adapter._clarify_state[prompt_id] = "session"
        resolver.reset_mock()
        await _dispatch_cloud_message(
            adapter, sender, prompt_id, button_id=f"cl:{prompt_id}:0",
        )
        if allowed:
            resolver.assert_called_once_with(prompt_id, "1")
            assert prompt_id not in adapter._clarify_state
        else:
            resolver.assert_not_called()
            assert adapter._clarify_state[prompt_id] == "session"
        adapter.handle_message.assert_not_awaited()

    await expect_tap(_BSUID, "initial", True)
    await expect_tap(_OTHER_BSUID, "collision", False)
    await expect_tap("1234567", "projection", False)
    await expect_tap(_BSUID.replace("A", "a"), "case", False)
    monkeypatch.setenv(env_var, f"{_OTHER_BSUID}@s.whatsapp.net")
    await expect_tap(_OTHER_BSUID, "update", True)
    await expect_tap(_BSUID, "old", False)
    monkeypatch.setenv(env_var, "")
    await expect_tap(_OTHER_BSUID, "empty", False)
    monkeypatch.setenv(env_var, f"{_OTHER_BSUID}@s.whatsapp.net")
    await expect_tap(_OTHER_BSUID, "restored", True)
    monkeypatch.delenv(env_var)
    await expect_tap(_OTHER_BSUID, "removed", False)

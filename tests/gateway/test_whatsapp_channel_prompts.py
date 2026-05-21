"""Tests for WhatsApp channel_prompts resolution (per-user/per-group)."""

from types import SimpleNamespace
from unittest.mock import MagicMock


def _make_adapter(channel_prompts: dict | None = None):
    """Build a WhatsAppAdapter shell with just enough config for channel-prompt lookup."""
    from gateway.platforms.whatsapp import WhatsAppAdapter

    adapter = object.__new__(WhatsAppAdapter)
    adapter.config = MagicMock()
    adapter.config.extra = {"channel_prompts": channel_prompts or {}}
    return adapter


def _resolve_for(adapter, data: dict) -> str | None:
    """Replicate the adapter's per-event lookup logic.

    Keeps the test focused on the *resolution* contract — full event-building
    requires a live bridge and DB and is exercised separately.
    """
    from gateway.platforms.base import resolve_channel_prompt

    primary_key = data.get("chatId") if data.get("isGroup") else data.get("senderId")
    return resolve_channel_prompt(
        adapter.config.extra,
        str(primary_key or ""),
        str(data.get("chatId") or "") if not data.get("isGroup") else None,
    )


def test_dm_resolves_by_sender_id():
    adapter = _make_adapter({
        "436642361246@s.whatsapp.net": "You are Eleana's assistant. Reply in Spanish.",
    })
    prompt = _resolve_for(adapter, {
        "isGroup": False,
        "senderId": "436642361246@s.whatsapp.net",
        "chatId": "436642361246@s.whatsapp.net",
    })
    assert prompt == "You are Eleana's assistant. Reply in Spanish."


def test_dm_falls_back_to_chat_id():
    """When senderId is missing but chatId matches, the prompt still resolves."""
    adapter = _make_adapter({
        "436642361246@s.whatsapp.net": "Eleana prompt",
    })
    prompt = _resolve_for(adapter, {
        "isGroup": False,
        "senderId": None,
        "chatId": "436642361246@s.whatsapp.net",
    })
    assert prompt == "Eleana prompt"


def test_group_resolves_by_chat_id():
    adapter = _make_adapter({
        "120363025@g.us": "Family group assistant.",
    })
    prompt = _resolve_for(adapter, {
        "isGroup": True,
        "chatId": "120363025@g.us",
        "senderId": "436641147220@s.whatsapp.net",  # individual member — ignored for groups
    })
    assert prompt == "Family group assistant."


def test_no_match_returns_none():
    adapter = _make_adapter({
        "436642361246@s.whatsapp.net": "Eleana prompt",
    })
    prompt = _resolve_for(adapter, {
        "isGroup": False,
        "senderId": "999999999999@s.whatsapp.net",
        "chatId": "999999999999@s.whatsapp.net",
    })
    assert prompt is None


def test_empty_config_returns_none():
    adapter = _make_adapter({})
    prompt = _resolve_for(adapter, {
        "isGroup": False,
        "senderId": "436642361246@s.whatsapp.net",
    })
    assert prompt is None


def test_whitespace_prompt_treated_as_absent():
    adapter = _make_adapter({
        "436642361246@s.whatsapp.net": "   \n  ",
    })
    prompt = _resolve_for(adapter, {
        "isGroup": False,
        "senderId": "436642361246@s.whatsapp.net",
    })
    assert prompt is None


def test_group_does_not_match_individual_sender():
    """In a group, sender's personal prompt must not leak in — only the group's prompt counts."""
    adapter = _make_adapter({
        "436642361246@s.whatsapp.net": "Eleana DM prompt",
        "120363025@g.us": "Group prompt",
    })
    # Eleana writes in the family group — she should get the group prompt, not her DM one.
    prompt = _resolve_for(adapter, {
        "isGroup": True,
        "chatId": "120363025@g.us",
        "senderId": "436642361246@s.whatsapp.net",
    })
    assert prompt == "Group prompt"

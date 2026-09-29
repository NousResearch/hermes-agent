"""WhatsApp inbound @mention metadata on MessageEvent."""

from __future__ import annotations

import asyncio

import pytest

from tests.gateway.test_whatsapp_from_owner import _dm_payload, _make_adapter


@pytest.fixture(autouse=True)
def _whatsapp_open_optin(monkeypatch):
    # The adapter fails closed on dm_policy "open" without the allow-all opt-in.
    monkeypatch.setenv("WHATSAPP_ALLOW_ALL_USERS", "true")


def test_mentioned_ids_surface_in_metadata():
    adapter = _make_adapter()
    payload = _dm_payload(mentionedIds=["15550001111@s.whatsapp.net"])

    event = asyncio.run(adapter._build_message_event(payload))

    assert event.metadata["whatsapp_mentioned_ids"] == ["15550001111@s.whatsapp.net"]


def test_no_mentions_leaves_metadata_key_absent():
    adapter = _make_adapter()

    event = asyncio.run(adapter._build_message_event(_dm_payload()))

    assert "whatsapp_mentioned_ids" not in event.metadata

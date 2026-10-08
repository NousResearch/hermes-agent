"""Replay and capability-boundary contracts for WhatsApp Cloud callbacks."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.whatsapp_cloud import WhatsAppCloudAdapter
from tools import slash_confirm


_SECRET = "synthetic-callback-secret"
_USER = "15550001111"
_FOREIGN_USER = "15550002222"


@pytest.fixture(autouse=True)
def _isolate_callback_state(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("WHATSAPP_CLOUD_ALLOW_ALL_USERS", "true")
    slash_confirm._pending.clear()
    yield
    slash_confirm._pending.clear()


def _response(message_id: str):
    response = MagicMock(status_code=200)
    response.json.return_value = {"messages": [{"id": message_id}]}
    response.text = "{}"
    return response


def _adapter(*message_ids: str) -> WhatsAppCloudAdapter:
    adapter = WhatsAppCloudAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "phone_number_id": "test-phone-scope",
                "access_token": "test-access-token",
                "app_secret": _SECRET,
                "dm_policy": "open",
            },
        )
    )
    adapter._http_client = MagicMock()
    adapter._http_client.post = AsyncMock(
        side_effect=[_response(mid) for mid in message_ids]
    )
    adapter.handle_message = AsyncMock()
    return adapter


def _signed_request(payload: dict):
    body = json.dumps(payload, separators=(",", ":")).encode()
    signature = hmac.new(_SECRET.encode(), body, hashlib.sha256).hexdigest()
    request = MagicMock()
    content = MagicMock()

    async def readexactly(size: int) -> bytes:
        raise asyncio.IncompleteReadError(body, size)

    content.readexactly = readexactly
    request.content = content
    request.headers = {"X-Hub-Signature-256": f"sha256={signature}"}
    return request


def _interactive_payload(
    *, sender: str, wamid: str, button_id: str, prompt_wamid: str
) -> dict:
    return {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "synthetic-entry",
                "changes": [
                    {
                        "field": "messages",
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {"display_phone_number": "15559990000", "phone_number_id": "test-phone-scope"},
                            "contacts": [
                                {"wa_id": sender, "profile": {"name": "Synthetic User"}}
                            ],
                            "messages": [
                                {
                                    "from": sender,
                                    "id": wamid,
                                    "type": "interactive",
                                    "context": {
                                        "from": "15559990000",
                                        "id": prompt_wamid,
                                    },
                                    "interactive": {
                                        "type": "button_reply",
                                        "button_reply": {
                                            "id": button_id,
                                            "title": "Approve Once",
                                        },
                                    },
                                }
                            ],
                        },
                    }
                ],
            }
        ],
    }


def _text_payload(*, wamid: str) -> dict:
    return {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "synthetic-entry",
                "changes": [
                    {
                        "field": "messages",
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {"display_phone_number": "15559990000", "phone_number_id": "test-phone-scope"},
                            "contacts": [
                                {"wa_id": _USER, "profile": {"name": "Synthetic User"}}
                            ],
                            "messages": [
                                {
                                    "from": _USER,
                                    "id": wamid,
                                    "type": "text",
                                    "text": {"body": "run this once"},
                                }
                            ],
                        },
                    }
                ],
            }
        ],
    }


@pytest.mark.asyncio
async def test_restart_fences_random_callback_to_origin_user_chat_and_message():
    old = _adapter("wamid.prompt.old")
    await old.send_slash_confirm(
        _USER,
        "Confirm",
        "old operation",
        "session-a",
        "1",
    )
    old_button = old._http_client.post.call_args.kwargs["json"]["interactive"][
        "action"
    ]["buttons"][0]["reply"]["id"]

    # A restarted gateway reuses the caller's small confirm id for a different operation.
    slash_confirm._pending.clear()
    calls: list[str] = []

    async def new_handler(choice: str):
        calls.append(choice)
        return "new operation ran"

    slash_confirm.register("session-a", "1", "new-operation", new_handler)
    restarted = _adapter("wamid.prompt.new", "wamid.confirmation")
    await restarted.send_slash_confirm(
        _USER,
        "Confirm",
        "new operation",
        "session-a",
        "1",
    )
    new_button = restarted._http_client.post.call_args.kwargs["json"]["interactive"][
        "action"
    ]["buttons"][0]["reply"]["id"]

    stale = _interactive_payload(
        sender=_USER,
        wamid="wamid.tap.stale",
        button_id=old_button,
        prompt_wamid="wamid.prompt.old",
    )
    assert (await restarted._handle_webhook(_signed_request(stale))).status == 200
    assert calls == []

    foreign_user = _interactive_payload(
        sender=_FOREIGN_USER,
        wamid="wamid.tap.foreign-user",
        button_id=new_button,
        prompt_wamid="wamid.prompt.new",
    )
    assert (
        await restarted._handle_webhook(_signed_request(foreign_user))
    ).status == 200
    assert calls == []

    foreign_message = _interactive_payload(
        sender=_USER,
        wamid="wamid.tap.foreign-message",
        button_id=new_button,
        prompt_wamid="wamid.prompt.other",
    )
    assert (
        await restarted._handle_webhook(_signed_request(foreign_message))
    ).status == 200
    assert calls == []

    valid = _interactive_payload(
        sender=_USER,
        wamid="wamid.tap.valid",
        button_id=new_button,
        prompt_wamid="wamid.prompt.new",
    )
    assert (await restarted._handle_webhook(_signed_request(valid))).status == 200
    assert calls == ["once"]
    assert old_button != new_button
    restarted.handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_wamid_claim_survives_restart_and_suppresses_replay():
    payload = _text_payload(wamid="wamid.replayed-after-restart")

    first = _adapter()
    assert (await first._handle_webhook(_signed_request(payload))).status == 200
    first.handle_message.assert_awaited_once()

    restarted = _adapter()
    assert (await restarted._handle_webhook(_signed_request(payload))).status == 200
    restarted.handle_message.assert_not_called()
    assert restarted._duplicate_count == 1

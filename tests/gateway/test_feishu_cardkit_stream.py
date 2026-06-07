import json
from types import SimpleNamespace

import pytest

from gateway.platforms.base import SendResult
from gateway.platforms.feishu import FeishuCardKitStreamSession


@pytest.mark.asyncio
async def test_cardkit_stream_lifecycle_creates_appends_updates_note_and_closes():
    requests = []
    sent_refs = []

    async def request_json(method, url, *, headers=None, json_body=None):
        requests.append((method, url, headers or {}, json_body))
        if url.endswith('/auth/v3/tenant_access_token/internal'):
            return {"code": 0, "tenant_access_token": "tenant-token", "expire": 7200}
        if url.endswith('/cardkit/v1/cards') and method == 'POST':
            card_data = json.loads(json_body["data"])
            assert card_data["schema"] == "2.0"
            assert card_data["config"]["streaming_mode"] is True
            assert card_data["body"]["elements"][0] == {
                "tag": "markdown",
                "content": "",
                "element_id": "content",
            }
            assert card_data["body"]["elements"][2]["element_id"] == "note"
            assert "<font color='grey'>Agent: Hermes" in card_data["body"]["elements"][2]["content"]
            return {"code": 0, "data": {"card_id": "card-1"}}
        return {"code": 0, "msg": "ok"}

    async def send_card_reference(chat_id, card_id, *, reply_to=None, metadata=None):
        sent_refs.append((chat_id, card_id, reply_to, metadata))
        return SendResult(success=True, message_id="msg-1")

    session = FeishuCardKitStreamSession(
        app_id="app-id",
        app_secret="app-secret",
        domain_name="feishu",
        request_json=request_json,
        send_card_reference=send_card_reference,
        note="Agent: Hermes | Model: gpt-5.5 | Provider: OpenAI | Generating…",
    )

    start = await session.start("chat-1", reply_to="root-msg", metadata={"thread_id": "topic-1"})
    assert start.success is True
    assert start.message_id == "msg-1"
    assert sent_refs == [("chat-1", "card-1", "root-msg", {"thread_id": "topic-1"})]

    assert await session.update("hello") is True
    assert await session.update("hello world") is True
    await session.update_note("Agent: Hermes | Model: gpt-5.5 | Provider: OpenAI")
    assert await session.close("hello world", note="Agent: Hermes | Model: gpt-5.5 | Provider: OpenAI") is True

    update_calls = [c for c in requests if c[0] == "PUT" and c[1].endswith('/elements/content/content')]
    assert [c[3]["content"] for c in update_calls] == ["hello", " world"]
    assert [c[3]["sequence"] for c in update_calls] == [1, 2]
    assert update_calls[0][3]["uuid"] == "s_card-1_1"

    note_calls = [c for c in requests if c[0] == "PUT" and c[1].endswith('/elements/note/content')]
    assert note_calls[-1][3]["content"] == "<font color='grey'>Agent: Hermes | Model: gpt-5.5 | Provider: OpenAI</font>"

    close_calls = [c for c in requests if c[0] == "PATCH" and c[1].endswith('/cardkit/v1/cards/card-1/settings')]
    assert len(close_calls) == 1
    close_settings = json.loads(close_calls[0][3]["settings"])
    assert close_settings["config"]["streaming_mode"] is False
    assert close_settings["config"]["summary"]["content"] == "hello world"


@pytest.mark.asyncio
async def test_cardkit_stream_replaces_content_when_text_is_not_append_only():
    requests = []

    async def request_json(method, url, *, headers=None, json_body=None):
        requests.append((method, url, json_body))
        if url.endswith('/auth/v3/tenant_access_token/internal'):
            return {"code": 0, "tenant_access_token": "tenant-token", "expire": 7200}
        if url.endswith('/cardkit/v1/cards') and method == 'POST':
            return {"code": 0, "data": {"card_id": "card-2"}}
        return {"code": 0}

    async def send_card_reference(*args, **kwargs):
        return SendResult(success=True, message_id="msg-2")

    session = FeishuCardKitStreamSession(
        app_id="app-id",
        app_secret="app-secret",
        domain_name="feishu",
        request_json=request_json,
        send_card_reference=send_card_reference,
    )
    await session.start("chat")
    assert await session.update("abcdef") is True
    assert await session.update("XYZ") is True

    replace_calls = [c for c in requests if c[0] == "PUT" and c[1].endswith('/elements/content')]
    assert len(replace_calls) == 1
    element = json.loads(replace_calls[0][2]["element"])
    assert element == {"tag": "markdown", "content": "XYZ", "element_id": "content"}


def test_feishu_adapter_creates_cardkit_stream_session_with_footer_note():
    from gateway.config import PlatformConfig
    from gateway.platforms.feishu import FeishuAdapter

    adapter = FeishuAdapter(PlatformConfig(enabled=True, extra={
        "app_id": "app-id",
        "app_secret": "app-secret",
        "domain": "feishu",
        "card_streaming": True,
    }))

    session = adapter.create_stream_session(metadata={
        "stream_footer_note": "Agent: Hermes | Model: gpt-5.5 | Provider: OpenAI"
    })

    assert isinstance(session, FeishuCardKitStreamSession)
    assert session.app_id == "app-id"
    assert session.app_secret == "app-secret"
    assert session.note == "Agent: Hermes | Model: gpt-5.5 | Provider: OpenAI"


def test_feishu_adapter_does_not_offer_cardkit_stream_when_disabled():
    from gateway.config import PlatformConfig
    from gateway.platforms.feishu import FeishuAdapter

    adapter = FeishuAdapter(PlatformConfig(enabled=True, extra={
        "app_id": "app-id",
        "app_secret": "app-secret",
        "domain": "feishu",
        "card_streaming": False,
    }))

    assert not hasattr(adapter, "create_stream_session") or adapter.create_stream_session(metadata={}) is None

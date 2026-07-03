import asyncio
from types import SimpleNamespace

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource
from plugins.platforms.whatsapp.adapter import WhatsAppAdapter


def _event(
    *,
    text="hello",
    message_type=MessageType.TEXT,
    raw_message=None,
    media_urls=None,
    media_types=None,
):
    media_urls = list(media_urls or [])
    media_types = list(media_types or [])
    raw = {"agentDispatchAllowed": False}
    if raw_message:
        raw.update(raw_message)
    source = SessionSource(
        platform=Platform.WHATSAPP,
        chat_id="48500111222@s.whatsapp.net",
        chat_type="dm",
        user_id="48500111222@s.whatsapp.net",
        user_name="Customer",
    )
    return MessageEvent(
        text=text,
        message_type=message_type,
        source=source,
        raw_message=raw,
        message_id=raw.get("messageId", "MSG1"),
        media_urls=media_urls,
        media_types=media_types,
    )


def test_ingest_only_handler_invokes_pre_gateway_dispatch_and_drops(monkeypatch):
    adapter = object.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    calls = []

    def fake_invoke_hook(name, **kwargs):
        calls.append((name, kwargs))
        return [{"action": "skip", "reason": "whatsapp_business_ingested"}]

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", fake_invoke_hook)

    asyncio.run(adapter._handle_ingest_only_event(_event()))

    assert len(calls) == 1
    name, kwargs = calls[0]
    assert name == "pre_gateway_dispatch"
    assert kwargs["event"].raw_message["agentDispatchAllowed"] is False
    assert kwargs["gateway"] is None
    assert kwargs["session_store"] is None


def test_ingest_only_media_event_passes_media_to_business_hook_and_drops(monkeypatch):
    adapter = object.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    calls = []
    media_path = "/home/adamf/.hermes/cache/images/customer.jpg"

    def fake_invoke_hook(name, **kwargs):
        calls.append((name, kwargs))
        return [{"action": "skip", "reason": "whatsapp_business_ingested"}]

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", fake_invoke_hook)

    event = _event(
        text="[image received]",
        message_type=MessageType.PHOTO,
        raw_message={
            "agentDispatchAllowed": False,
            "messageId": "MEDIA1",
            "hasMedia": True,
            "mediaType": "image",
            "mediaUrls": [media_path],
        },
        media_urls=[media_path],
        media_types=["image/jpeg"],
    )
    asyncio.run(adapter._handle_ingest_only_event(event))

    assert len(calls) == 1
    _, kwargs = calls[0]
    observed = kwargs["event"]
    assert observed.message_type == MessageType.PHOTO
    assert observed.raw_message["agentDispatchAllowed"] is False
    assert observed.raw_message["hasMedia"] is True
    assert observed.media_urls == [media_path]
    assert observed.media_types == ["image/jpeg"]


def test_ingest_only_handler_fail_closed_when_hook_crashes(monkeypatch):
    adapter = object.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP

    def boom(name, **kwargs):
        raise RuntimeError("hook exploded")

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", boom)

    # No exception: ingest-only events are dropped even if the plugin fails.
    asyncio.run(adapter._handle_ingest_only_event(_event()))

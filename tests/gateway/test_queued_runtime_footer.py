"""Queued replies retain the configured footer without repeating a delivered footer."""

from types import SimpleNamespace

import pytest

from gateway import run as gateway_run
from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource


class _Transport:
    name = "fake"
    extract_media = staticmethod(BasePlatformAdapter.extract_media)
    extract_images = staticmethod(BasePlatformAdapter.extract_images)

    def __init__(self, accepted=True):
        self.accepted = accepted
        self.sent = []

    async def send(self, chat_id, content, **kwargs):
        self.sent.append((chat_id, content, kwargs.get("metadata")))
        return SendResult(success=self.accepted, message_id="sent", error=None if self.accepted else "refused")


def _queued_turn(tmp_path, monkeypatch, *, streamed, enabled=True, accepted=True):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    (tmp_path / "config.yaml").write_text(
        "display:\n  runtime_footer:\n    enabled: " + str(enabled).lower() + "\n    fields: [model]\n",
        encoding="utf-8",
    )
    runner = gateway_run.GatewayRunner(GatewayConfig())
    transport = _Transport(accepted=accepted)
    runner._delivery_adapter_for = lambda _source: transport
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    source = SessionSource(platform=Platform.DISCORD, chat_id="room-1", thread_id="thread-1")
    consumer = SimpleNamespace(final_response_sent=True, delivered_final_matches=lambda text: text == "first reply")
    context = SimpleNamespace(
        source=source, session_key="conversation", stream_consumer_holder=[consumer if streamed else None],
        mute_notification_reply=False, persist_user_display_kind=None, reply_expected=True,
        _status_thread_metadata={"thread_id": source.thread_id}, event_message_id="input-1",
        inbound_message_id="input-1", run_generation=1,
    )
    result = {"final_response": "first reply", "model": "example/model-one", "failed": False}
    return runner, transport, context, result


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
async def test_queued_reply_delivers_configured_footer_once_even_if_followup_is_refused(
    tmp_path, monkeypatch, streamed, enabled,
):
    runner, transport, context, result = _queued_turn(
        tmp_path, monkeypatch, streamed=streamed, enabled=enabled,
    )

    await runner._run_agent_deliver_first_response(context, transport, result, result, None)
    expected = (["model-one"] if enabled else []) if streamed else [
        "first reply\n\nmodel-one" if enabled else "first reply"
    ]
    assert [content for _, content, _ in transport.sent] == expected
    # If the queued follow-up is refused, the ordinary completion path receives this same result.
    footer = runner._hmwa_runtime_footer_line(result, context.source, 1)
    completion = await runner._hmwa_deliver_turn_response(
        MessageEvent(text="first", source=context.source, message_id="input-1"), context.source,
        SimpleNamespace(session_id="session"), context.session_key, 1, result, [], "first reply", footer, False,
    )

    assert completion is None
    assert [content for _, content, _ in transport.sent] == expected
    assert all(chat == context.source.chat_id and metadata["thread_id"] == context.source.thread_id
               for chat, _, metadata in transport.sent)


@pytest.mark.asyncio
async def test_refused_trailing_footer_does_not_reopen_the_already_delivered_body(tmp_path, monkeypatch):
    runner, transport, context, result = _queued_turn(
        tmp_path, monkeypatch, streamed=True, accepted=False,
    )

    await runner._run_agent_deliver_first_response(context, transport, result, result, None)

    assert [content for _, content, _ in transport.sent] == ["model-one"]
    assert result["already_sent"]
    transport.accepted = True
    footer = runner._hmwa_runtime_footer_line(result, context.source, 1)
    assert footer == "model-one"
    assert await runner._hmwa_deliver_turn_response(
        MessageEvent(text="first", source=context.source, message_id="input-1"), context.source,
        SimpleNamespace(session_id="session"), context.session_key, 1, result, [], "first reply", footer, False,
    ) is None
    assert [content for _, content, _ in transport.sent] == ["model-one", "model-one"]

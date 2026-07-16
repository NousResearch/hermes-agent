from types import SimpleNamespace

from agent.chat_completion_helpers import build_assistant_message
from gateway import run as gateway_run
from gateway.platforms import base


def _event(text, *, message_type=None, media_urls=None):
    return SimpleNamespace(
        text=text,
        message_type=message_type or base.MessageType.TEXT,
        media_urls=media_urls or [],
    )


def test_busy_queue_batches_plain_text_but_stops_before_command():
    runner = object.__new__(gateway_run.GatewayRunner)
    head = _event("first")
    runner._queued_events = {
        "session": [_event("second"), _event("/new"), _event("fourth")]
    }

    result = runner._batch_queued_text_events("session", head)

    assert result.text == "[Queued message 1]\nfirst\n\n[Queued message 2]\nsecond"
    assert [event.text for event in runner._queued_events["session"]] == [
        "/new",
        "fourth",
    ]


def test_busy_queue_does_not_batch_media():
    runner = object.__new__(gateway_run.GatewayRunner)
    head = _event("caption", media_urls=["voice.ogg"])
    runner._queued_events = {"session": [_event("next")]}

    result = runner._batch_queued_text_events("session", head)

    assert result.text == "caption"
    assert runner._queued_events["session"][0].text == "next"


def test_reasoning_persistence_can_be_disabled():
    agent = SimpleNamespace(
        _persist_reasoning=False,
        _extract_reasoning=lambda _message: "hidden scratchpad",
        verbose_logging=False,
        reasoning_callback=None,
        stream_delta_callback=None,
        _stream_callback=None,
        _strip_think_blocks=lambda content: content,
        _needs_thinking_reasoning_pad=lambda: False,
    )
    response = SimpleNamespace(
        content="answer",
        tool_calls=None,
        reasoning_content="hidden scratchpad",
        reasoning_details=[{"type": "reasoning", "text": "hidden"}],
        anthropic_content_blocks=None,
        codex_reasoning_items=None,
        codex_message_items=None,
    )

    message = build_assistant_message(agent, response, "stop")

    assert message["content"] == "answer"
    assert message["reasoning"] is None
    assert "reasoning_content" not in message
    assert "reasoning_details" not in message


def test_media_outbox_stages_acks_and_deduplicates(tmp_path, monkeypatch):
    monkeypatch.setattr(base, "_MEDIA_OUTBOX_DB", tmp_path / "outbox.sqlite3")
    kwargs = {
        "session_key": "telegram:123",
        "tool_call_id": "call-tts",
        "platform": "telegram",
        "chat_id": "123",
        "thread_id": None,
        "media_path": str(tmp_path / "voice.ogg"),
        "is_voice": True,
    }

    delivery_key, should_send = base._media_outbox_stage(**kwargs)
    assert should_send is True
    assert len(base._media_outbox_pending("telegram", "123")) == 1

    base._media_outbox_ack(delivery_key, "telegram-message-7")
    _, should_send_again = base._media_outbox_stage(**kwargs)

    assert should_send_again is False
    assert base._media_outbox_pending("telegram", "123") == []


def test_media_outbox_marker_never_reaches_visible_text():
    text = "[[media_outbox:call-tts]]\nMEDIA:C:/audio/voice.ogg"
    assert base._strip_media_tag_directives(text).strip() == ""

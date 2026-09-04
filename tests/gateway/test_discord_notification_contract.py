# pyright: reportMissingImports=false
"""Discord ``notifications: important`` delivery contract.

The policy is metadata-driven: Hermes internals decide whether a logical send is
interim or notification-worthy; Discord decides suppression per physical
message so a split response produces at most one push notification.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, SendResult
from gateway.run_turn_runner import TurnRunner
from gateway.session import build_session_key
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from plugins.platforms.discord import adapter as discord_adapter_module
from plugins.platforms.discord import notifications as discord_notifications
from plugins.platforms.discord.adapter import DiscordAdapter


_SUPPRESS_NOTIFICATIONS = 1 << 12


def _adapter(mode: str) -> DiscordAdapter:
    adapter = DiscordAdapter(
        PlatformConfig(enabled=True, token="test", extra={"notifications": mode})
    )
    setattr(adapter, "format_message", lambda content: content)
    return adapter


def _silent_values(calls: list[dict]) -> list[bool | None]:
    return [call.get("silent") for call in calls]


@pytest.mark.asyncio
async def test_notification_contract_for_channels_threads_retries_and_edits(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    # The Discord plugin owns config resolution. Direct adapter construction
    # uses PlatformConfig.extra; the profile-specific display setting overrides it.
    from hermes_cli import config as hermes_config

    with monkeypatch.context() as config_patch:
        config_patch.setattr(
            hermes_config,
            "load_config_readonly",
            lambda: {},
        )
        assert discord_notifications.resolve_notification_mode(
            PlatformConfig(enabled=True, token="test", extra={"notifications": "important"})
        ) == "important"
        config_patch.setattr(
            hermes_config,
            "load_config_readonly",
            lambda: {
                "display": {
                    "platforms": {"discord": {"notifications": "all"}},
                }
            },
        )
        assert discord_notifications.resolve_notification_mode(
            PlatformConfig(enabled=True, token="test", extra={"notifications": "important"})
        ) == "all"

    adapter = _adapter("important")
    calls: list[dict] = []
    fail_reference_once = False

    async def send(**kwargs):
        nonlocal fail_reference_once
        calls.append(dict(kwargs))
        if fail_reference_once and kwargs.get("reference") is not None:
            fail_reference_once = False
            raise RuntimeError("400 Bad Request (error code: 10008): Unknown Message")
        message_id = 9000 + len(calls)
        return SimpleNamespace(
            id=message_id,
            to_reference=lambda **_kwargs: SimpleNamespace(message_id=message_id),
        )

    channel = SimpleNamespace(id=555, send=send)
    thread = SimpleNamespace(id=777, send=send)
    setattr(
        adapter,
        "_client",
        SimpleNamespace(
            get_channel=lambda channel_id: thread if channel_id == 777 else channel,
            fetch_channel=AsyncMock(),
        ),
    )
    setattr(adapter, "truncate_message", lambda _content, _limit, _len_fn=None: ["one"])

    # Reasoning, tool activity, commentary, lifecycle text, and ordinary status
    # remain visible but silent. The content label is not used for classification.
    for label, metadata in (
        ("reasoning", {"_interim_send": True}),
        ("tool progress", {"non_conversational": True}),
        ("commentary", {"_interim_send": True}),
        ("lifecycle status", {"non_conversational": True}),
        ("progress", {}),
    ):
        before = len(calls)
        original = dict(metadata)
        result = await adapter.send("555", label, metadata=metadata)
        assert result.success is True
        assert calls[before].get("silent") is True
        assert metadata == original
        assert "notify" not in calls[before]
        assert "_interim_send" not in calls[before]

    # A runtime-marked user-attention event is notification-worthy even if it is
    # not ordinary final-answer prose.
    before = len(calls)
    result = await adapter.send(
        "555",
        "Approval required",
        metadata={"notify": True, "_interim_send": True, "kind": "approval"},
    )
    assert result.success is True
    assert "silent" not in calls[before]

    # A one-message completed response is notification-capable.
    before = len(calls)
    result = await adapter.send(
        "555", "single-message final", metadata={"notify": True, "turn_id": "turn-0"}
    )
    assert result.success is True
    assert "silent" not in calls[before]

    # Finality is runtime metadata plus physical chunk position, never prose.
    calls.clear()
    final_metadata = {"notify": True, "turn_id": "turn-1"}
    setattr(
        adapter,
        "truncate_message",
        lambda _content, _limit, _len_fn=None: ["one", "two", "three"],
    )
    result = await adapter.send("555", "not parsed for finality", metadata=final_metadata)
    assert result.success is True
    assert _silent_values(calls) == [True, True, None]
    assert final_metadata == {"notify": True, "turn_id": "turn-1"}

    # Thread routing and a deleted-reference retry retain the same policy and
    # reply anchoring. The retry cannot turn a silent chunk into a loud one.
    calls.clear()
    reference = object()
    setattr(adapter, "_reply_reference_for_send", lambda reply_to, channel: reference)
    fail_reference_once = True
    thread_metadata = {"notify": True, "thread_id": "777"}
    result = await adapter.send(
        "555", "thread final", reply_to="99", metadata=thread_metadata
    )
    assert result.success is True
    assert _silent_values(calls) == [True, True, True, None]
    assert [call.get("reference") for call in calls] == [reference, None, None, None]
    assert thread_metadata == {"notify": True, "thread_id": "777"}

    # Explicit legacy mode preserves the pre-feature wire shape for every chunk.
    legacy = _adapter("all")
    legacy_calls: list[dict] = []

    async def legacy_send(**kwargs):
        legacy_calls.append(dict(kwargs))
        return SimpleNamespace(id=10000 + len(legacy_calls))

    legacy_channel = SimpleNamespace(id=888, send=legacy_send)
    setattr(
        legacy,
        "_client",
        SimpleNamespace(
            get_channel=lambda _channel_id: legacy_channel,
            fetch_channel=AsyncMock(),
        ),
    )
    setattr(legacy, "truncate_message", lambda _content, _limit, _len_fn=None: ["one"])
    await legacy.send("888", "legacy interim", metadata={"_interim_send": True})
    await legacy.send("888", "legacy final", metadata={"notify": True})
    setattr(legacy, "truncate_message", lambda _content, _limit, _len_fn=None: ["one", "two", "three"])
    result = await legacy.send(
        "888", "legacy split final", metadata={"notify": True}
    )
    assert result.success is True
    assert _silent_values(legacy_calls) == [None, None, None, None, None]

    # Interactive prompts are already user-attention egress and bypass routine
    # turn-progress suppression.
    calls.clear()
    setattr(adapter, "truncate_message", lambda _content, _limit, _len_fn=None: ["one"])
    await adapter._send_prompt(
        "555",
        None,
        lambda _channel: ({"content": "Choose an option"}, None),
    )
    assert len(calls) == 1
    assert "silent" not in calls[0]

    # The plain-text dangerous-command fallback is an interim boundary but an
    # actionable prompt. Its producer must set notify=True before sealing the
    # metadata as an interim send.
    approval_calls: list[dict] = []

    class PlainApprovalAdapter:
        typed_command_prefix = "/"

        def pause_typing_for_chat(self, _chat_id):
            return None

        def send(self, chat_id, content, metadata=None):
            approval_calls.append(
                {"chat_id": chat_id, "content": content, "metadata": metadata}
            )
            return "scheduled approval"

    approval_ctx = SimpleNamespace(
        _status_adapter=PlainApprovalAdapter(),
        _status_chat_id="555",
        _status_thread_metadata={"thread_id": "777"},
        session_key="approval-session",
    )
    approval_runner = TurnRunner(
        cast(Any, SimpleNamespace()), cast(Any, approval_ctx)
    )
    setattr(approval_runner, "_close_native_stream_boundary", lambda _reason: None)

    def schedule_approval(operation, _label):
        assert operation == "scheduled approval"
        return SimpleNamespace(result=lambda timeout=None: None)

    setattr(approval_runner, "_schedule", schedule_approval)
    approval_runner._approval_notify_sync(
        {"command": "printf safe", "description": "test approval"}
    )
    assert approval_calls[0]["metadata"] == {
        "thread_id": "777",
        "is_approval_prompt": True,
        "notify": True,
        "_interim_send": True,
    }

    # Fatal turn errors are also explicit user-attention events.
    calls.clear()
    error_event = MessageEvent(
        text="trigger",
        source=adapter.build_source("555", chat_type="channel"),
    )
    error_metadata = await adapter._notify_turn_error(
        error_event, RuntimeError("test failure")
    )
    assert error_metadata == {"notify": True}
    assert len(calls) == 1
    assert "silent" not in calls[0]

    # Routine thread infrastructure is also visible but silent. User-attention
    # prompts and failure warnings remain on their dedicated loud paths.
    thread_seed_calls: list[dict] = []

    async def thread_seed_send(content, **kwargs):
        thread_seed_calls.append({"content": content, **kwargs})
        return SimpleNamespace(id=12001)

    created_thread = SimpleNamespace(
        id=12000,
        name="test thread",
        send=thread_seed_send,
    )
    thread_parent = SimpleNamespace(
        create_thread=AsyncMock(return_value=created_thread),
    )
    interaction = SimpleNamespace(
        channel=thread_parent,
        user=SimpleNamespace(display_name="Tester"),
    )
    result = await adapter._create_thread(
        cast(Any, interaction),
        name="test thread",
        message="starter message",
    )
    assert result["success"] is True
    assert _silent_values(thread_seed_calls) == [True]

    auto_seed_calls: list[dict] = []
    auto_thread = SimpleNamespace(id=13000, name="auto thread")
    auto_seed = SimpleNamespace(
        create_thread=AsyncMock(return_value=auto_thread),
    )

    async def auto_seed_send(content, **kwargs):
        auto_seed_calls.append({"content": content, **kwargs})
        return auto_seed

    auto_message = SimpleNamespace(
        content="auto thread",
        author=SimpleNamespace(display_name="Tester"),
        create_thread=AsyncMock(side_effect=RuntimeError("direct create rejected")),
        channel=SimpleNamespace(send=auto_seed_send),
    )
    assert await adapter._auto_create_thread(auto_message) is auto_thread
    assert _silent_values(auto_seed_calls) == [True]

    handoff_seed_calls: list[dict] = []
    handoff_thread = SimpleNamespace(id=14000)
    handoff_seed = SimpleNamespace(
        create_thread=AsyncMock(return_value=handoff_thread),
    )

    async def handoff_seed_send(content, **kwargs):
        handoff_seed_calls.append({"content": content, **kwargs})
        return handoff_seed

    handoff_parent = SimpleNamespace(
        create_thread=AsyncMock(side_effect=RuntimeError("direct create rejected")),
        send=handoff_seed_send,
    )
    previous_client = adapter._client
    setattr(
        adapter,
        "_client",
        SimpleNamespace(
            get_channel=lambda _channel_id: handoff_parent,
            fetch_channel=AsyncMock(),
        ),
    )
    assert await adapter.create_handoff_thread("555", "handoff") == "14000"
    adapter._client = previous_client
    assert _silent_values(handoff_seed_calls) == [True]

    # Discord edits cannot create a push notification. For a finalized overflow,
    # only continuation sends can notify, and only the last continuation does.
    calls.clear()
    setattr(
        adapter,
        "truncate_message",
        lambda _content, _limit, _len_fn=None: ["one", "two", "three"],
    )
    original_message = SimpleNamespace(
        id=42,
        edit=AsyncMock(),
        to_reference=MagicMock(return_value=object()),
    )
    result = await adapter._edit_overflow_split(  # pyright: ignore[reportCallIssue]
        channel,
        original_message,
        "42",
        "oversized final",
        metadata={"notify": True},
    )
    assert result.success is True
    assert _silent_values(calls) == [True, None]

    # A segment-boundary edit stays interim. A turn-final edit receives the
    # same authoritative notify metadata as a fresh final send, allowing its
    # overflow continuations to select only the last notification candidate.
    edit_calls: list[dict] = []

    async def edit_message(**kwargs):
        edit_calls.append(dict(kwargs))
        return SendResult(success=True, message_id=kwargs["message_id"])

    setattr(adapter, "edit_message", edit_message)
    consumer = GatewayStreamConsumer(
        adapter,
        "555",
        StreamConsumerConfig(),
        metadata={"thread_id": "777"},
    )
    await consumer._edit_message(
        message_id="42", content="segment", finalize=True, notify=False
    )
    await consumer._edit_message(
        message_id="42", content="completed", finalize=True, notify=True
    )
    assert edit_calls[0]["metadata"] == {"thread_id": "777"}
    assert edit_calls[1]["metadata"] == {"thread_id": "777", "notify": True}

    # Edit-based streaming starts with a visible silent preview. Because an edit
    # cannot create a Discord push, important mode replaces that preview with one
    # fresh, notification-capable completed message and removes the stale preview.
    streaming = _adapter("important")
    streaming_calls: list[dict] = []
    preview_landed = asyncio.Event()

    async def streaming_send(**kwargs):
        streaming_calls.append(dict(kwargs))
        preview_landed.set()
        message_id = 15000 + len(streaming_calls)
        return SimpleNamespace(
            id=message_id,
            to_reference=lambda **_kwargs: SimpleNamespace(message_id=message_id),
        )

    streaming_channel = SimpleNamespace(id=555, send=streaming_send)
    setattr(
        streaming,
        "_client",
        SimpleNamespace(
            get_channel=lambda _channel_id: streaming_channel,
            fetch_channel=AsyncMock(),
        ),
    )
    setattr(streaming, "truncate_message", lambda text, _limit, _len_fn=None: [text])
    delete_preview = AsyncMock(return_value=True)
    setattr(streaming, "delete_message", delete_preview)
    stream_consumer = GatewayStreamConsumer(
        streaming,
        "555",
        StreamConsumerConfig(
            transport="edit",
            chat_type="channel",
            edit_interval=0.01,
            buffer_threshold=1,
            cursor="",
            fresh_final_after_seconds=0.0,
        ),
    )
    stream_consumer.on_delta("streamed final")
    stream_task = asyncio.create_task(stream_consumer.run())
    await asyncio.wait_for(preview_landed.wait(), timeout=1)
    stream_consumer.finish()
    await asyncio.wait_for(stream_task, timeout=1)
    assert _silent_values(streaming_calls) == [True, None]
    delete_preview.assert_awaited_once_with("555", "15001")
    assert stream_consumer.final_response_sent is True

    # The same guarantee holds after streaming overflow has sealed earlier
    # chunks. Those heads stay visible and silent; only the fresh tail notifies,
    # and cleanup deletes only the active tail preview.
    overflow_streaming = _adapter("important")
    overflow_calls: list[dict] = []
    overflow_heads_landed = asyncio.Event()
    overflow_tail_preview_landed = asyncio.Event()

    async def overflow_send(**kwargs):
        overflow_calls.append(dict(kwargs))
        if len(overflow_calls) == 2:
            overflow_heads_landed.set()
        elif len(overflow_calls) == 3:
            overflow_tail_preview_landed.set()
        message_id = 16000 + len(overflow_calls)
        return SimpleNamespace(
            id=message_id,
            to_reference=lambda **_kwargs: SimpleNamespace(message_id=message_id),
        )

    overflow_channel = SimpleNamespace(id=556, send=overflow_send)
    setattr(
        overflow_streaming,
        "_client",
        SimpleNamespace(
            get_channel=lambda _channel_id: overflow_channel,
            fetch_channel=AsyncMock(),
        ),
    )
    delete_overflow_preview = AsyncMock(return_value=True)
    setattr(overflow_streaming, "delete_message", delete_overflow_preview)
    overflow_consumer = GatewayStreamConsumer(
        overflow_streaming,
        "556",
        StreamConsumerConfig(
            transport="edit",
            chat_type="channel",
            edit_interval=0.01,
            buffer_threshold=1,
            cursor="",
            fresh_final_after_seconds=0.0,
        ),
    )
    overflow_consumer.on_delta("x" * 4500)
    overflow_task = asyncio.create_task(overflow_consumer.run())
    await asyncio.wait_for(overflow_heads_landed.wait(), timeout=1)
    overflow_consumer.on_delta("y")
    await asyncio.wait_for(overflow_tail_preview_landed.wait(), timeout=1)
    overflow_consumer.finish()
    await asyncio.wait_for(overflow_task, timeout=1)
    assert len(overflow_calls) >= 3
    assert _silent_values(overflow_calls[:-1]) == [True] * (
        len(overflow_calls) - 1
    ), overflow_calls
    assert "silent" not in overflow_calls[-1]
    deleted_overflow_ids = {
        call.args[1] for call in delete_overflow_preview.await_args_list
    }
    assert deleted_overflow_ids == {str(16000 + len(overflow_calls) - 1)}
    assert overflow_consumer.final_response_sent is True

    # A tool-boundary segment can finalize its transport without finalizing the
    # turn. A first physical send at that boundary must remain silent.
    segment_adapter = _adapter("important")
    segment_sends: list[dict] = []

    async def segment_send(**kwargs):
        segment_sends.append(dict(kwargs))
        return SendResult(success=True, message_id="segment-1")

    setattr(segment_adapter, "send", segment_send)
    segment_consumer = GatewayStreamConsumer(
        segment_adapter,
        "555",
        StreamConsumerConfig(transport="edit", cursor=""),
    )
    assert await segment_consumer._send_or_edit(
        "tool preamble", finalize=True, is_turn_final=False
    ) is True
    assert not segment_sends[0]["metadata"] or (
        "notify" not in segment_sends[0]["metadata"]
    )
    assert segment_consumer.final_response_sent is False

    # Discord important mode explicitly needs a fresh final because edits cannot
    # create a push. If that send fails, do not downgrade to a successful edit
    # and falsely complete the turn without a notification.
    failed_fresh_adapter = _adapter("important")
    setattr(
        failed_fresh_adapter,
        "send",
        AsyncMock(return_value=SendResult(success=False, error="send rejected")),
    )
    edit_after_failure = AsyncMock(
        return_value=SendResult(success=True, message_id="preview-1")
    )
    setattr(failed_fresh_adapter, "edit_message", edit_after_failure)
    failed_fresh_consumer = GatewayStreamConsumer(
        failed_fresh_adapter,
        "555",
        StreamConsumerConfig(transport="edit", cursor=""),
    )
    failed_fresh_consumer._message_id = "preview-1"
    failed_fresh_consumer._last_sent_text = "preview"
    assert await failed_fresh_consumer._edit_existing(
        "completed", finalize=True, is_turn_final=True
    ) is False
    edit_after_failure.assert_not_awaited()
    assert failed_fresh_consumer.final_response_sent is False
    assert failed_fresh_consumer._fallback_final_send is True


@pytest.mark.asyncio
async def test_notification_contract_for_forums_and_media(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _adapter("important")
    forum_starters: list[dict] = []
    forum_followups: list[list[dict]] = []

    async def create_thread(**kwargs):
        forum_starters.append(dict(kwargs))
        followup_calls: list[dict] = []
        forum_followups.append(followup_calls)

        async def followup_send(**send_kwargs):
            followup_calls.append(dict(send_kwargs))
            message_id = 8000 + len(followup_calls)
            return SimpleNamespace(
                id=message_id,
                to_reference=lambda **_kwargs: SimpleNamespace(message_id=message_id),
            )

        attachments = []
        if kwargs.get("file") is not None:
            attachments = [kwargs["file"]]
        elif kwargs.get("files"):
            attachments = list(kwargs["files"])
        return SimpleNamespace(
            id=7000 + len(forum_starters),
            thread=SimpleNamespace(
                id=7000 + len(forum_starters),
                send=followup_send,
            ),
            message=SimpleNamespace(
                id=6000 + len(forum_starters),
                attachments=attachments,
            ),
        )

    forum = SimpleNamespace(id=666, create_thread=create_thread)
    setattr(adapter, "_is_forum_parent", lambda channel: channel is forum)
    setattr(
        adapter,
        "_client",
        SimpleNamespace(
            get_channel=lambda _channel_id: forum,
            fetch_channel=AsyncMock(),
        ),
    )

    def forum_chunks(content, _limit):
        if content == "final single":
            return ["single"]
        return ["one", "two", "three"]

    setattr(adapter, "truncate_message", forum_chunks)

    progress_metadata = {"_interim_send": True, "phase": "tool"}
    result = await adapter.send("666", "progress", metadata=progress_metadata)
    assert result.success is True
    assert forum_starters[-1].get("silent") is True
    assert _silent_values(forum_followups[-1]) == [True, True]
    assert progress_metadata == {"_interim_send": True, "phase": "tool"}

    result = await adapter.send("666", "final single", metadata={"notify": True})
    assert result.success is True
    assert "silent" not in forum_starters[-1]
    assert forum_followups[-1] == []

    result = await adapter.send("666", "final split", metadata={"notify": True})
    assert result.success is True
    assert forum_starters[-1].get("silent") is True
    assert _silent_values(forum_followups[-1]) == [True, None]

    # Forum attachments use the same starter-message policy.
    media_path = tmp_path / "report.txt"
    media_path.write_text("evidence", encoding="utf-8")
    monkeypatch.setattr(
        discord_adapter_module.discord,
        "File",
        lambda path, filename=None, **_kwargs: SimpleNamespace(
            path=path, filename=filename
        ),
    )
    result = await adapter.send_document(
        "666", str(media_path), metadata={"_interim_send": True}
    )
    assert result.success is True
    assert forum_starters[-1].get("silent") is True

    # Multi-image batches are physical Discord messages. A notify-worthy batch
    # sequence emits at most one push, on its final batch.
    regular_calls: list[dict] = []

    async def regular_send(**kwargs):
        regular_calls.append(dict(kwargs))
        return SimpleNamespace(
            id=5000 + len(regular_calls),
            attachments=list(kwargs.get("files") or []),
        )

    regular = SimpleNamespace(id=555, send=regular_send)
    client = SimpleNamespace(
        get_channel=lambda _channel_id: regular,
        fetch_channel=AsyncMock(),
        http=SimpleNamespace(request=AsyncMock(return_value={"id": "4444"})),
    )
    setattr(adapter, "_client", client)
    setattr(adapter, "_is_forum_parent", lambda channel: False)

    # An unsafe or failed remote image falls back to the base URL send without
    # dropping the notification metadata.
    url_fallbacks: list[dict] = []

    async def base_send_image(
        self, chat_id, image_url, caption=None, reply_to=None, metadata=None
    ):
        url_fallbacks.append(
            {
                "chat_id": chat_id,
                "image_url": image_url,
                "caption": caption,
                "reply_to": reply_to,
                "metadata": metadata,
            }
        )
        return SendResult(success=True, message_id="fallback")

    with monkeypatch.context() as media_patch:
        media_patch.setattr(BasePlatformAdapter, "send_image", base_send_image)
        fallback_metadata = {"notify": True, "thread_id": "777"}
        result = await adapter.send_image(
            "555",
            "http://127.0.0.1/image.png",
            caption="fallback",
            metadata=fallback_metadata,
        )
    assert result.success is True
    assert url_fallbacks[0]["metadata"] == fallback_metadata

    # A completed response containing both text and an attachment emits exactly
    # one notification candidate: the attachment is silent and sent first, then
    # the completed text is loud.
    regular_calls.clear()
    attachment = tmp_path / "final-report.txt"
    attachment.write_text("report", encoding="utf-8")
    adapter.config.typing_indicator = False
    setattr(adapter, "truncate_message", lambda _content, _limit, _len_fn=None: ["Final body"])
    setattr(adapter, "on_processing_start", AsyncMock())
    setattr(adapter, "on_processing_complete", AsyncMock())

    async def final_with_attachment(_event):
        return f"Final body\nMEDIA: {attachment}"

    adapter.set_message_handler(final_with_attachment)
    final_event = MessageEvent(
        text="request",
        source=adapter.build_source("555", chat_type="channel"),
    )
    await adapter._process_message_background(
        final_event, build_session_key(final_event.source)
    )
    assert len(regular_calls) == 2
    assert regular_calls[0].get("files")
    assert regular_calls[1].get("content") == "Final body"
    assert _silent_values(regular_calls) == [True, None]

    regular_calls.clear()
    images = []
    for index in range(11):
        path = tmp_path / f"image-{index}.png"
        path.write_bytes(b"png")
        images.append((path.as_uri(), f"image {index}"))
    await adapter.send_multiple_images(
        "555", images, metadata={"notify": True}, human_delay=0
    )
    assert _silent_values(regular_calls) == [True, None]

    # A skipped planned tail must not consume the final notification: the last
    # actual batch remains loud.
    regular_calls.clear()
    missing = tmp_path / "missing-tail.png"
    await adapter.send_multiple_images(
        "555",
        [*images[:10], (missing.as_uri(), "missing")],
        metadata={"notify": True},
        human_delay=0,
    )
    assert _silent_values(regular_calls) == [None]

    # If a native image batch fails, the per-image fallback preserves the same
    # last-actual-message invariant.
    fallback_image_metadata: list[dict] = []

    async def base_send_multiple_images(
        self, chat_id, fallback_images, metadata=None, human_delay=0.0
    ):
        fallback_image_metadata.append(dict(metadata or {}))

    async def fail_batch_send(**_kwargs):
        raise RuntimeError("native batch rejected")

    previous_client = adapter._client
    failing_channel = SimpleNamespace(id=555, send=fail_batch_send)
    setattr(
        adapter,
        "_client",
        SimpleNamespace(
            get_channel=lambda _channel_id: failing_channel,
            fetch_channel=AsyncMock(),
        ),
    )
    with monkeypatch.context() as fallback_patch:
        fallback_patch.setattr(
            BasePlatformAdapter,
            "send_multiple_images",
            base_send_multiple_images,
        )
        await adapter.send_multiple_images(
            "555", images[:3], metadata={"notify": True}, human_delay=0
        )
    adapter._client = previous_client
    assert [metadata.get("notify") for metadata in fallback_image_metadata] == [
        None,
        None,
        True,
    ]

    # Native voice messages use a raw Discord payload. Preserve the voice flag
    # and add SUPPRESS_NOTIFICATIONS only for non-notify delivery.
    audio_path = tmp_path / "voice.ogg"
    audio_path.write_bytes(b"not-real-ogg")
    await adapter.send_voice(
        "555", str(audio_path), metadata={"_interim_send": True}
    )
    form = client.http.request.await_args.kwargs["form"]
    payload = json.loads(next(part["value"] for part in form if part["name"] == "payload_json"))
    assert payload["flags"] & _SUPPRESS_NOTIFICATIONS

    await adapter.send_voice("555", str(audio_path), metadata={"notify": True})
    form = client.http.request.await_args.kwargs["form"]
    payload = json.loads(next(part["value"] for part in form if part["name"] == "payload_json"))
    assert not payload["flags"] & _SUPPRESS_NOTIFICATIONS

    # Fallback final chunking assigns notification intent per physical message,
    # so only the last fallback chunk can produce a Discord push.
    fallback_adapter = _adapter("important")
    fallback_chunk_metadata: list[dict] = []

    async def fallback_send(**kwargs):
        fallback_chunk_metadata.append(dict(kwargs.get("metadata") or {}))
        return SendResult(
            success=True, message_id=f"fallback-{len(fallback_chunk_metadata)}"
        )

    setattr(fallback_adapter, "send", fallback_send)
    fallback_consumer = GatewayStreamConsumer(
        fallback_adapter,
        "555",
        StreamConsumerConfig(transport="edit", cursor=""),
    )
    setattr(
        fallback_consumer,
        "_split_text_chunks",
        lambda _text, _limit, len_fn=None: ["one", "two", "three"],
    )
    await fallback_consumer._send_fallback_final("completed response")
    assert [metadata.get("notify") for metadata in fallback_chunk_metadata] == [
        None,
        None,
        True,
    ]

    # A failed attachment warning is itself actionable. It must restore notify
    # intent even when the failed non-last attachment had that marker removed.
    media_failure_adapter = _adapter("important")
    media_failure_sends: list[dict] = []

    async def media_failure_send(**kwargs):
        media_failure_sends.append(dict(kwargs))
        return SendResult(success=True, message_id="warning-1")

    setattr(media_failure_adapter, "send", media_failure_send)
    original_failure_metadata = {"notify": False, "thread_id": "777"}
    await media_failure_adapter._notify_media_delivery_failure(
        "555", "/tmp/report.txt", metadata=original_failure_metadata
    )
    assert media_failure_sends[0]["metadata"] == {
        "notify": True,
        "thread_id": "777",
    }
    assert original_failure_metadata == {"notify": False, "thread_id": "777"}

"""Regression test for issue #27970 Bug 2.

The auto Telegram voice reply (``GatewayRunner._send_voice_reply``) is the
final response of a turn. It must mark its metadata as ``notify=True`` so
adapters that gate push notifications (Telegram's "important" mode) deliver
it as a normal push instead of a silent message — mirroring the existing
final-text path in ``gateway/platforms/base.py``.
"""

import json
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _make_event(thread_id=None):
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="208214988",
        user_id="208214988",
        chat_type="dm",
        thread_id=thread_id,
    )
    return MessageEvent(
        text="hi",
        message_type=MessageType.TEXT,
        source=source,
        message_id="m1",
    )


def _runner_with_adapter(send_voice_mock):
    runner = object.__new__(GatewayRunner)
    adapter = SimpleNamespace(
        send_voice=send_voice_mock,
        is_in_voice_channel=lambda *_a, **_k: False,
    )
    runner.adapters = {Platform.TELEGRAM: adapter}
    return runner


def _fake_tts_call(monkeypatch, audio_bytes=b"\x00" * 32):
    """Patch the TTS tool so it writes a real file at the requested path."""

    def _fake_text_to_speech_tool(*, text, output_path, **_kwargs):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as fh:
            fh.write(audio_bytes)
        return json.dumps({"success": True, "file_path": output_path})

    monkeypatch.setattr(
        "tools.tts_tool.text_to_speech_tool",
        _fake_text_to_speech_tool,
    )
    monkeypatch.setattr(
        "tools.tts_text_normalize._strip_markdown_for_tts",
        lambda text: text,
    )


@pytest.mark.asyncio
async def test_voice_reply_marks_existing_thread_metadata_without_mutation(monkeypatch, tmp_path):
    """When thread metadata exists (Telegram DM-topic), notify=True is added without mutating the source dict."""
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    _fake_tts_call(monkeypatch)

    send_voice = AsyncMock()
    runner = _runner_with_adapter(send_voice)
    # Use a DM topic source so _thread_metadata_for_source returns a non-None dict.
    event = _make_event(thread_id="17585")
    source_meta_snapshot = runner._thread_metadata_for_source(
        event.source, runner._reply_anchor_for_event(event)
    )
    assert source_meta_snapshot is not None
    snapshot_copy = dict(source_meta_snapshot)

    await runner._send_voice_reply(event, "Hello there.")

    send_voice.assert_awaited_once()
    kwargs = send_voice.await_args.kwargs
    assert kwargs["metadata"].get("notify") is True
    # All pre-existing thread keys are preserved.
    for k, v in snapshot_copy.items():
        assert kwargs["metadata"].get(k) == v
    # The freshly-computed source-side metadata must NOT have been mutated
    # (would otherwise leak notify=True into the typing-indicator state).
    fresh = runner._thread_metadata_for_source(
        event.source, runner._reply_anchor_for_event(event)
    )
    assert "notify" not in fresh


@pytest.mark.asyncio
@pytest.mark.parametrize("platform", [Platform.FEISHU, Platform.TELEGRAM])
@pytest.mark.parametrize("outcome", ["terminal", "transient", "success"])
async def test_multipart_voice_stops_only_for_terminal_policy(platform, outcome):
    from gateway.platforms.base import SendResult
    from gateway.platforms.base_thread_metadata import _thread_metadata_for_event

    receipt = SendResult(
        success=outcome == "success", error=None if outcome == "success" else "voice delivery failed",
        retry_suppressed=outcome == "terminal",
    )
    send_voice = AsyncMock(return_value=receipt)
    runner = _runner_with_adapter(send_voice)
    adapter = runner.adapters.pop(Platform.TELEGRAM)
    runner.adapters[platform] = adapter
    event = _make_event(thread_id="omt_topic" if platform == Platform.FEISHU else "17585")
    event.source.platform = platform
    metadata = _thread_metadata_for_event(event)
    if platform == Platform.FEISHU:
        metadata["_feishu_topic_delivery"]["destination"] = "parent_chat"

    await runner._deliver_voice_reply(event, ["first.ogg", "second.ogg"])

    assert send_voice.await_count == (1 if outcome == "terminal" else 2)
    for call in send_voice.await_args_list:
        assert call.kwargs["metadata"]["notify"] is True
        if platform == Platform.FEISHU:
            assert call.kwargs["metadata"]["_feishu_topic_delivery"] is metadata["_feishu_topic_delivery"]
    if outcome == "terminal":
        assert event._delivery_retry_suppressed_result is receipt
        if platform == Platform.FEISHU:
            assert metadata["_feishu_topic_delivery"]["terminal"] is receipt
    else:
        assert not hasattr(event, "_delivery_retry_suppressed_result")


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["parent_chat", "error_notice", "silent"])
@pytest.mark.parametrize("text_first", [False, True])
async def test_auto_voice_and_final_text_share_feishu_policy_on_the_wire(tmp_path, policy, text_first):
    from unittest.mock import Mock
    from gateway.config import PlatformConfig
    from gateway.platforms.base_thread_metadata import _thread_metadata_for_event
    from plugins.platforms.feishu.adapter import FeishuAdapter

    pytest.importorskip("lark_oapi")
    from plugins.platforms.feishu.adapter import _load_lark_oapi
    assert _load_lark_oapi()
    adapter = FeishuAdapter(PlatformConfig(extra={"topic_delivery_fallback": policy}))
    ok = SimpleNamespace(success=lambda: True, data=SimpleNamespace(message_id="om_sent"))
    wire = SimpleNamespace(
        reply=Mock(return_value=SimpleNamespace(success=lambda: False, code=230011, msg="withdrawn")),
        list=Mock(return_value=SimpleNamespace(success=lambda: True, data=SimpleNamespace(items=[]))),
        create=Mock(return_value=ok),
    )
    upload = Mock(return_value=SimpleNamespace(success=lambda: True, data=SimpleNamespace(file_key="file_key")))
    adapter._client = SimpleNamespace(im=SimpleNamespace(v1=SimpleNamespace(
        message=wire, file=SimpleNamespace(create=upload))))
    async def blocking(func, *args):
        return func(*args)
    adapter._run_blocking = blocking
    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.FEISHU: adapter}
    event = _make_event(thread_id="omt_topic")
    event.source.platform = Platform.FEISHU
    event.message_id = "om_trigger"
    metadata = runner._event_thread_metadata(event, event.source)
    paths = [tmp_path / "first.ogg", tmp_path / "second.ogg"]
    for path in paths:
        path.write_bytes(b"OggS fake opus")
    if text_first:
        await adapter.send(event.source.chat_id, "Answer", metadata=metadata)

    await runner._deliver_voice_reply(event, [str(path) for path in paths])
    final = await adapter.send(event.source.chat_id, "Final answer", metadata=_thread_metadata_for_event(event))

    assert wire.reply.call_count == 1
    assert wire.list.call_count == 1
    state = metadata["_feishu_topic_delivery"]
    if policy == "parent_chat":
        assert final.success
        assert state["destination"] == "parent_chat"
        assert upload.call_count == 2
        assert wire.create.call_count == 3 + int(text_first)
    else:
        assert not final.success and final.retry_suppressed
        assert upload.call_count == (0 if text_first else 1)
        assert wire.create.call_count == (1 if policy == "error_notice" else 0)
        assert event._delivery_retry_suppressed_result is state["terminal"]

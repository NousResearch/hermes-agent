"""Adapter warnings are separate from captions, attachment receipts and failed state."""
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from plugins.platforms.discord.adapter import DiscordAdapter
from plugins.platforms.slack.adapter import SlackAdapter
from plugins.platforms.matrix.adapter import MatrixAdapter


class LegacyAdapter(BasePlatformAdapter):
    async def connect(self, **kwargs):
        return True

    async def disconnect(self):
        pass

    async def get_chat_info(self, chat_id):
        return {"name": "fixture", "type": "dm"}

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append((content, reply_to, metadata))
        return SendResult(success=True, message_id="actual-text-send")


@pytest.mark.asyncio
@pytest.mark.parametrize("setting", [None, False, True])
async def test_media_fallbacks_preserve_default_receipts_and_caption(tmp_path, monkeypatch, setting, caplog):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    if setting is not None:
        (tmp_path / "config.yaml").write_text(f"display: {{suppress_warning_notifications: {str(setting).lower()}}}")
    base = LegacyAdapter(PlatformConfig(), Platform.TELEGRAM)
    base.sent = []
    result = await base.send_document("chat", str(tmp_path / "missing"), caption="caption", reply_to="reply", metadata={"thread_id": "thread"})
    assert result.success is (setting is not True)
    assert result.message_id == (None if setting is True else "actual-text-send")
    assert base.sent[0][0] == ("caption" if setting is True else "caption\n⚠️ Couldn't deliver the file attachment.")
    assert "native file send unavailable" in caplog.text
    for adapter, invoke in (
        (SlackAdapter(PlatformConfig()), lambda a: a._send_failure_notice("chat", "caption", "diagnostic", "reply", {"thread_id": "thread"})),
        (MatrixAdapter(PlatformConfig()), lambda a: a._send_local_file("chat", str(tmp_path / "missing"), "m.file", "caption", "reply")),
    ):
        adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="actual-text-send"))
        outcome = await invoke(adapter)
        assert outcome.success is (setting is not True)
        if setting is True:
            assert adapter.send.call_args.args[1] == "caption"
    await base._notify_media_delivery_failure("chat", "missing")
    assert len(base.sent) == (1 if setting is True else 2)


@pytest.mark.asyncio
@pytest.mark.parametrize("suppress", [False, True])
async def test_discord_oversize_and_plain_fallback_keep_material_content(tmp_path, monkeypatch, suppress):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(f"display: {{suppress_warning_notifications: {str(suppress).lower()}}}")
    adapter = DiscordAdapter(PlatformConfig())
    adapter._discord_upload_limit_bytes = lambda channel: 1
    adapter._is_forum_parent = lambda channel: False
    channel = type("Channel", (), {"send": AsyncMock()})()
    path = tmp_path / "upload.txt"
    path.write_bytes(b"too large")
    result = await adapter._reject_oversized_upload(channel, str(path), path.name)
    assert not result.success and "too large" in result.error.lower()
    assert channel.send.await_count == int(not suppress)
    base = LegacyAdapter(PlatformConfig(), Platform.TELEGRAM)
    base.sent = []
    outcome = await base._send_plain_fallback("chat", "requested content", reply_to=None, metadata=None)
    assert outcome.success
    assert base.sent[0][0] == ("requested content" if suppress else "(Response formatting failed, plain text:)\n\nrequested content")


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit", [False, True])
async def test_relay_media_diagnostic_uses_logical_destination_without_sealing(tmp_path, monkeypatch, explicit):
    from tests.gateway.relay.test_relay_prompt_ack_stream_isolation import _adapter, _open_turn_draft
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("display:\n  suppress_warning_notifications: false\n  platforms:\n    slack: {suppress_warning_notifications: true}\n")
    adapter = _adapter()
    adapter._platform_by_chat["D01"] = "telegram" if explicit else "slack"
    metadata = {"_relay_logical_platform": "slack"} if explicit else None
    key = await _open_turn_draft(adapter)
    before = list(adapter._transport.frames)
    await adapter._notify_media_delivery_failure("D01", "missing.pdf", metadata=metadata)
    assert adapter._transport.frames == before
    assert key in adapter._open_draft_by_chat
    (tmp_path / "config.yaml").write_text("display: {suppress_warning_notifications: false}")
    await adapter._notify_media_delivery_failure("D01", "missing.pdf", metadata=metadata)
    assert adapter._transport.frames[-1][0]["op"] == "send"
    assert key in adapter._open_draft_by_chat

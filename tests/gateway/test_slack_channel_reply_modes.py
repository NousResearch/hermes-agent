"""Outbound regression coverage for per-channel Slack reply modes."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.slack.adapter import SlackAdapter


@pytest.fixture
def adapter():
    config = PlatformConfig(enabled=True, token="«redacted:xox…»")
    config.extra.update({
        "reply_in_thread": True,
        "channel_reply_modes": {
            "C_FLAT": "channel",
            "C_THREADED": "thread",
        },
    })
    result = SlackAdapter(config)
    result._app = MagicMock()
    client = MagicMock()
    client.chat_postMessage = AsyncMock(return_value={"ok": True, "ts": "200.001"})
    client.files_upload_v2 = AsyncMock(return_value={"ok": True, "ts": "200.002"})
    result._get_client = MagicMock(return_value=client)
    result._ensure_dm_conversation = AsyncMock(
        side_effect=lambda chat_id, **_kwargs: chat_id
    )
    result.stop_typing = AsyncMock()
    result._record_uploaded_file_thread = MagicMock()
    return result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("chat_id", "metadata", "reply_to", "expected_thread"),
    [
        ("C_FLAT", None, "100.001", None),
        ("C_THREADED", {"thread_id": "100.002"}, "100.002", "100.002"),
        ("C_FLAT", {"thread_id": "100.003"}, "100.004", "100.003"),
    ],
)
async def test_final_text_payload_uses_channel_mode(
    adapter, chat_id, metadata, reply_to, expected_thread
):
    await adapter.send(chat_id, "done", reply_to=reply_to, metadata=metadata)

    kwargs = adapter._get_client.return_value.chat_postMessage.await_args.kwargs
    assert kwargs.get("thread_ts") == expected_thread


@pytest.mark.asyncio
async def test_flat_status_payload_has_no_thread_anchor(adapter):
    await adapter.send_or_update_status("C_FLAT", "turn", "working", metadata=None)

    kwargs = adapter._get_client.return_value.chat_postMessage.await_args.kwargs
    assert "thread_ts" not in kwargs


@pytest.mark.asyncio
async def test_flat_attachment_payload_has_no_thread_anchor(adapter, tmp_path):
    document = tmp_path / "report.txt"
    document.write_text("report", encoding="utf-8")

    await adapter.send_document("C_FLAT", str(document), metadata=None)

    kwargs = adapter._get_client.return_value.files_upload_v2.await_args.kwargs
    assert kwargs["thread_ts"] is None


@pytest.mark.asyncio
async def test_flat_interactive_prompt_payload_has_no_thread_anchor(adapter):
    await adapter.send_exec_approval(
        "C_FLAT",
        command="true",
        session_key="session",
        metadata=None,
    )

    kwargs = adapter._get_client.return_value.chat_postMessage.await_args.kwargs
    assert "thread_ts" not in kwargs

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
            "C_PROJECT": "project",
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
        ("C_PROJECT", None, "100.005", None),
        ("C_PROJECT", {"thread_id": "100.006"}, "100.006", "100.006"),
    ],
)
async def test_final_text_payload_uses_channel_mode(
    adapter, chat_id, metadata, reply_to, expected_thread
):
    await adapter.send(chat_id, "done", reply_to=reply_to, metadata=metadata)

    kwargs = adapter._get_client.return_value.chat_postMessage.await_args.kwargs
    assert kwargs.get("thread_ts") == expected_thread


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("chat_id", "metadata", "expected_thread"),
    [
        ("C_FLAT", None, None),
        ("C_PROJECT", None, None),
        ("C_PROJECT", {"thread_id": "100.101"}, "100.101"),
    ],
)
async def test_status_payload_uses_resolved_turn_route(
    adapter, chat_id, metadata, expected_thread
):
    await adapter.send_or_update_status(
        chat_id, "turn", "working", metadata=metadata
    )

    kwargs = adapter._get_client.return_value.chat_postMessage.await_args.kwargs
    assert kwargs.get("thread_ts") == expected_thread


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("chat_id", "metadata", "expected_thread"),
    [
        ("C_FLAT", None, None),
        ("C_PROJECT", None, None),
        ("C_PROJECT", {"thread_id": "100.201"}, "100.201"),
    ],
)
async def test_attachment_payload_uses_resolved_turn_route(
    adapter, tmp_path, chat_id, metadata, expected_thread
):
    document = tmp_path / "report.txt"
    document.write_text("report", encoding="utf-8")

    await adapter.send_document(chat_id, str(document), metadata=metadata)

    kwargs = adapter._get_client.return_value.files_upload_v2.await_args.kwargs
    assert kwargs["thread_ts"] == expected_thread


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("chat_id", "metadata", "expected_thread"),
    [
        ("C_FLAT", None, None),
        ("C_PROJECT", None, None),
        ("C_PROJECT", {"thread_id": "100.301"}, "100.301"),
    ],
)
async def test_interactive_prompt_payload_uses_resolved_turn_route(
    adapter, chat_id, metadata, expected_thread
):
    await adapter.send_exec_approval(
        chat_id,
        command="true",
        session_key="session",
        metadata=metadata,
    )

    kwargs = adapter._get_client.return_value.chat_postMessage.await_args.kwargs
    assert kwargs.get("thread_ts") == expected_thread

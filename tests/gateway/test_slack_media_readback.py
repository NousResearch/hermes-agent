"""Slack media delivery is successful only when the posted message has files."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.slack.adapter import SlackAdapter


def _adapter(client):
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-fake"))
    adapter._app = MagicMock()
    adapter._app.client = client
    adapter._running = True
    return adapter


@pytest.mark.asyncio
async def test_upload_readback_with_files_succeeds(tmp_path):
    artifact = tmp_path / "report.pdf"
    artifact.write_bytes(b"%PDF-1.4")
    client = AsyncMock()
    client.files_upload_v2.return_value = {
        "ok": True,
        "files": [{"id": "F123", "shares": {"public": {"C123": [{"ts": "10.20"}]}}}],
    }
    client.conversations_replies.return_value = {
        "ok": True,
        "messages": [{"ts": "10.20", "files": [{"id": "F123"}]}],
    }
    adapter = _adapter(client)

    result = await adapter.send_document("C123", str(artifact), caption="report")

    assert result.success is True
    client.conversations_replies.assert_awaited_once_with(
        channel="C123", ts="10.20", inclusive=True, limit=100
    )


@pytest.mark.asyncio
async def test_thread_upload_reads_back_from_root_and_matches_child(tmp_path):
    artifact = tmp_path / "report.pdf"
    artifact.write_bytes(b"%PDF-1.4")
    client = AsyncMock()
    client.files_upload_v2.return_value = {
        "ok": True,
        "files": [{"id": "F123", "shares": {"private": {"C123": [{"ts": "10.20"}]}}}],
    }
    client.conversations_replies.return_value = {
        "ok": True,
        "messages": [
            {"ts": "9.00", "text": "root"},
            {"ts": "10.20", "files": [{"id": "F123"}]},
        ],
    }
    adapter = _adapter(client)

    result = await adapter.send_document(
        "C123", str(artifact), caption="report", reply_to="9.00")

    assert result.success is True
    client.conversations_replies.assert_awaited_once_with(
        channel="C123", ts="9.00", inclusive=True, limit=100
    )


@pytest.mark.asyncio
async def test_readback_error_does_not_reupload(tmp_path):
    artifact = tmp_path / "report.pdf"
    artifact.write_bytes(b"%PDF-1.4")
    client = AsyncMock()
    client.files_upload_v2.return_value = {"ok": True, "ts": "10.20"}
    client.conversations_replies.side_effect = RuntimeError("readback unavailable")
    client.chat_postMessage.return_value = {"ok": True, "ts": "10.21"}
    adapter = _adapter(client)

    result = await adapter.send_document("C123", str(artifact), caption="report")

    assert result.success is False
    assert "could not be verified" in (result.error or "")
    client.files_upload_v2.assert_awaited_once()


@pytest.mark.asyncio
async def test_upload_readback_without_files_reports_definite_failure(tmp_path):
    artifact = tmp_path / "report.pdf"
    artifact.write_bytes(b"%PDF-1.4")
    client = AsyncMock()
    client.files_upload_v2.return_value = {
        "ok": True,
        "files": [{"id": "F123", "shares": {"public": {"C123": [{"ts": "10.20"}]}}}],
    }
    client.conversations_replies.return_value = {
        "ok": True,
        "messages": [{"ts": "10.20", "text": "report"}],
    }
    client.chat_postMessage.return_value = {"ok": True, "ts": "10.21"}
    adapter = _adapter(client)

    result = await adapter.send_document("C123", str(artifact), caption="report")

    assert result.success is False
    assert "files is empty" in (result.error or "")
    client.chat_postMessage.assert_awaited_once()
    assert "attachment" in client.chat_postMessage.await_args.kwargs["text"].lower()

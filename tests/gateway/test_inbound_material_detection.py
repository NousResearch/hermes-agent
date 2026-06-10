"""Smoke tests for inbound material-handling detection."""

from types import SimpleNamespace

import pytest

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource


def _make_runner():
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.config = SimpleNamespace(
        group_sessions_per_user=True,
        thread_sessions_per_user=False,
    )
    runner._session_key_for_source = lambda source: "session-1"
    runner._consume_pending_native_image_paths = lambda session_key: None
    return runner


def _make_source():
    return SessionSource(
        platform=Platform.WECOM,
        chat_id="group-1",
        chat_type="group",
        user_id="user-1",
    )


@pytest.mark.asyncio
async def test_plain_text_does_not_get_material_context():
    runner = _make_runner()
    event = MessageEvent(
        text="hostname",
        message_type=MessageType.TEXT,
        source=_make_source(),
        message_id="m-1",
    )

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=event.source,
        history=[],
    )

    assert result == "hostname"


@pytest.mark.asyncio
async def test_github_actions_jobs_url_does_not_get_material_context():
    runner = _make_runner()
    text = "https://github.com/example/repo/actions/runs/123456789/jobs/987654321"
    event = MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=_make_source(),
        message_id="m-2",
    )

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=event.source,
        history=[],
    )

    assert result == text


@pytest.mark.asyncio
async def test_github_pr_url_does_not_get_material_context():
    runner = _make_runner()
    text = "https://github.com/example/repo/pull/42"
    event = MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=_make_source(),
        message_id="m-3",
    )

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=event.source,
        history=[],
    )

    assert result == text


@pytest.mark.asyncio
async def test_file_attachment_still_gets_material_context(tmp_path):
    runner = _make_runner()
    file_path = tmp_path / "report.pdf"
    file_path.write_bytes(b"%PDF-1.4")
    event = MessageEvent(
        text="please review",
        message_type=MessageType.DOCUMENT,
        source=_make_source(),
        message_id="m-4",
        media_urls=[str(file_path)],
        media_types=["application/pdf"],
    )

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=event.source,
        history=[],
    )

    assert "The user sent a document" in result
    assert "please review" in result


@pytest.mark.asyncio
async def test_explicit_archive_request_can_get_material_context():
    runner = _make_runner()
    text = "把这个链接资料入库：https://github.com/example/repo/pull/42"
    event = MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=_make_source(),
        message_id="m-5",
    )

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=event.source,
        history=[],
    )

    assert "explicitly asked" in result
    assert text in result

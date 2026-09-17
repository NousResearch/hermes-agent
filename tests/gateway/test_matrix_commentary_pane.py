"""Matrix sticky interim-commentary pane — send-then-edit contract."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.matrix_commentary import matrix_commentary_bodies
from gateway.matrix_commentary_pane import MatrixCommentaryPane
from gateway.platforms.base import SendResult
from gateway.run_turn_runner import TurnRunner
from gateway.turn_context import TurnContext
from plugins.platforms.matrix.adapter import MatrixAdapter, _sanitize_matrix_html


class _FakeAdapter:
    name = "matrix"

    def __init__(self):
        self.sends = []
        self.edits = []
        self.send_results = []
        self.edit_results = []

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sends.append(
            {
                "chat_id": chat_id,
                "content": content,
                "reply_to": reply_to,
                "metadata": dict(metadata or {}),
            }
        )
        if self.send_results:
            return self.send_results.pop(0)
        return SendResult(success=True, message_id="$commentary-root")

    async def edit_message(self, chat_id, message_id, content, *, finalize=False, metadata=None):
        self.edits.append(
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "content": content,
                "metadata": dict(metadata or {}),
            }
        )
        if self.edit_results:
            return self.edit_results.pop(0)
        return SendResult(success=True, message_id=message_id)


def test_commentary_bodies_title_and_list():
    body, html = matrix_commentary_bodies(
        ["I'll inspect the config next.", "Writing the packet locally."]
    )
    assert body == "💬 Commentary (2 updates)"
    assert "<ol>" in html
    assert html.count("<li>") == 2
    assert "<details>" not in html
    assert "```" not in html
    assert "inspect the config next." in html


def test_commentary_bodies_preserve_paragraph_breaks_and_cap():
    body, html = matrix_commentary_bodies(["First paragraph.\n\nSecond paragraph." + ("x" * 800)])
    assert body == "💬 Commentary (1 update)"
    assert "<br>" in html
    assert "Second paragraph." in html
    assert "x" * 600 not in html
    assert "..." in html


def test_commentary_bodies_window_keeps_newest():
    items = [f"oldest-{i} " + ("y" * 400) for i in range(80)]
    body, html = matrix_commentary_bodies(items)
    assert body == "💬 Commentary (80 updates)"
    assert "Showing latest" in html
    assert "oldest-79" in html
    assert "oldest-0 " not in html


@pytest.mark.asyncio
async def test_first_commentary_sends_root_with_unprefixed_html():
    adapter = _FakeAdapter()
    pane = MatrixCommentaryPane(
        adapter=adapter, chat_id="!room:example", reply_to="$user", metadata={"thread_id": "$t"}
    )
    result = await pane.append("I'll inspect the config next.")
    assert result.success
    assert pane.root_event_id == "$commentary-root"
    assert len(adapter.sends) == 1
    assert not adapter.edits
    sent = adapter.sends[0]
    assert sent["content"] == "💬 Commentary (1 update)"
    assert sent["metadata"]["_interim_send"] is True
    assert sent["metadata"]["matrix_formatted_body_unprefixed"] is True
    assert sent["metadata"]["thread_id"] == "$t"
    assert "<ol>" in sent["metadata"]["matrix_formatted_body"]
    assert "inspect the config next." in sent["metadata"]["matrix_formatted_body"]


@pytest.mark.asyncio
async def test_second_commentary_edits_same_root():
    adapter = _FakeAdapter()
    pane = MatrixCommentaryPane(adapter=adapter, chat_id="!room:example")
    await pane.append("I'll inspect the config next.")
    await pane.append("Writing the packet locally.")
    assert len(adapter.sends) == 1
    assert len(adapter.edits) == 1
    edit = adapter.edits[0]
    assert edit["message_id"] == "$commentary-root"
    html = edit["metadata"]["matrix_formatted_body"]
    assert "inspect the config next." in html
    assert "Writing the packet locally." in html
    assert edit["content"] == "💬 Commentary (2 updates)"


@pytest.mark.asyncio
async def test_edit_failure_retries_same_root_without_second_send():
    adapter = _FakeAdapter()
    pane = MatrixCommentaryPane(adapter=adapter, chat_id="!room:example")
    await pane.append("first")
    adapter.edit_results = [SendResult(success=False, error="forbidden")]
    await pane.append("second")
    assert len(adapter.sends) == 1
    assert adapter.edits[0]["message_id"] == "$commentary-root"
    adapter.edit_results = [SendResult(success=True, message_id="$commentary-root")]
    await pane.close()
    assert all(e["message_id"] == "$commentary-root" for e in adapter.edits)
    assert len(adapter.sends) == 1


@pytest.mark.asyncio
async def test_empty_commentary_does_not_send():
    adapter = _FakeAdapter()
    pane = MatrixCommentaryPane(adapter=adapter, chat_id="!room:example")
    assert await pane.append("   ") is None
    assert adapter.sends == []
    assert pane.root_event_id is None


def test_interim_callback_skips_already_streamed_and_empty():
    adapter = _FakeAdapter()
    pane = MatrixCommentaryPane(adapter=adapter, chat_id="!room:example")
    ctx = TurnContext()
    ctx._run_still_current = lambda: True
    ctx.scheduled_heartbeat = False
    ctx.interim_assistant_messages_enabled = True
    ctx.streaming_tts_consumer_holder = [None]
    ctx.stream_consumer_holder = [None]
    ctx.user_config = {}
    ctx.source = SimpleNamespace(chat_id="!room:example", platform=Platform.MATRIX)
    ctx.matrix_commentary_pane = pane
    ctx.resolve_display_setting = lambda *a, **k: None
    ctx.progress_queue = None
    ctx.event_message_id = None
    scheduled = []

    runner = TurnRunner(SimpleNamespace(config=None, _adapter_for_source=lambda s: adapter), ctx)

    def _schedule(coro, log_message, loop=None):
        scheduled.append(True)
        coro.close()
        return None

    runner._schedule = _schedule  # type: ignore[method-assign]
    _, _, cb, want = runner._setup_stream_consumer("matrix")
    assert want is True
    cb("   ")
    cb("shown already", already_streamed=True)
    assert scheduled == []
    cb("I'll inspect the config next.")
    assert len(scheduled) == 1


def test_non_matrix_without_pane_does_not_schedule_commentary_pane():
    ctx = TurnContext()
    ctx._run_still_current = lambda: True
    ctx.scheduled_heartbeat = False
    ctx.interim_assistant_messages_enabled = True
    ctx.streaming_tts_consumer_holder = [None]
    ctx.stream_consumer_holder = [None]
    ctx.user_config = {}
    ctx.source = SimpleNamespace(chat_id="123", platform=Platform.TELEGRAM)
    ctx.matrix_commentary_pane = None
    ctx.resolve_display_setting = lambda *a, **k: None
    ctx.progress_queue = None
    ctx.event_message_id = None
    ctx._status_adapter = None
    scheduled = []
    runner = TurnRunner(SimpleNamespace(config=None, _adapter_for_source=lambda s: None), ctx)
    runner._schedule = lambda *a, **k: scheduled.append(a)  # type: ignore[method-assign]
    _, _, cb, want = runner._setup_stream_consumer("telegram")
    assert want is True
    cb("I'll inspect the config next.")
    assert scheduled == []


@pytest.mark.asyncio
async def test_adapter_sanitizes_html_on_send_and_edit_and_keeps_star_prefix_without_flag():
    adapter = object.__new__(MatrixAdapter)
    adapter._client = MagicMock()
    adapter._encryption = False
    adapter.format_message = lambda c: c
    adapter.truncate_message = lambda c, n: [c]
    adapter._build_text_message_content = lambda c: {"msgtype": "m.text", "body": c}
    adapter._apply_relation_metadata = lambda *a, **k: None
    events = []

    async def _send_evt(room, etype, content):
        events.append(content)
        return f"$e{len(events)}"

    adapter._client.send_message_event = _send_evt
    dirty = '<p><strong>💬 Commentary (1 update)</strong></p><ol><li>ok</li></ol><script>alert(1)</script><img src=x onerror="alert(1)">'
    await MatrixAdapter.send(
        adapter,
        "!room:ex",
        "💬 Commentary (1 update)",
        metadata={
            "matrix_formatted_body": dirty,
            "matrix_formatted_body_unprefixed": True,
            "_interim_send": True,
        },
    )
    html = events[0]["formatted_body"]
    assert "<script>" not in html
    assert "onerror" not in html
    assert "<ol>" in html
    await MatrixAdapter.edit_message(
        adapter,
        "!room:ex",
        "$e1",
        "💬 Commentary (1 update)",
        metadata={
            "matrix_formatted_body": dirty,
            "matrix_formatted_body_unprefixed": True,
        },
    )
    edit = events[1]
    assert edit["m.relates_to"]["rel_type"] == "m.replace"
    assert edit["m.relates_to"]["event_id"] == "$e1"
    assert not edit["formatted_body"].startswith("* ")
    assert edit["m.new_content"]["formatted_body"].startswith("<p>")
    await MatrixAdapter.edit_message(
        adapter,
        "!room:ex",
        "$e1",
        "approval card",
        metadata={"matrix_formatted_body": "<p>Approve</p>"},
    )
    approval = events[2]
    assert approval["formatted_body"].startswith("* ")
    stripped = _sanitize_matrix_html(dirty)
    assert "<script>" not in stripped

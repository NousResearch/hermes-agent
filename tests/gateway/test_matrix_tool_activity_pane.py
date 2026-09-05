"""Matrix Tool activity pane — first-principles contract tests."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.matrix_tool_activity import matrix_tool_activity_bodies
from gateway.platforms.base import SendResult
from gateway.run_turn_runner import TurnRunner
from gateway.turn_context import TurnContext
from plugins.platforms.matrix.adapter import MatrixAdapter, _sanitize_matrix_html


def test_matrix_body_is_counter_only():
    body, html = matrix_tool_activity_bodies(
        [
            "🔍 Searching past sessions",
            "💻 terminal: ls -la /tmp",
            "```",
            "💻 terminal\n```\nset -euo pipefail\n```",
        ]
    )
    assert body == "🛠 Tool activity (3 updates)"
    assert "```" not in body
    assert "terminal" not in body  # plain body has no tool labels
    assert body.count("\n") == 0


def test_matrix_html_is_single_ol_no_fences_or_details():
    body, html = matrix_tool_activity_bodies(
        [
            "🔍 Searching past sessions",
            "```",
            "💻 terminal: rg -n progress gateway/run.py",
            "⚙️ hindsight_recall: resume website",
        ]
    )
    assert body == "🛠 Tool activity (3 updates)"
    assert html.count("<ol>") == 1
    assert html.count("<li>") == 3
    assert "<details>" not in html
    assert "data-mx-spoiler" not in html
    assert "<pre>" not in html
    assert "```" not in html
    assert "Searching past sessions" in html
    assert html.count("Searching past sessions") == 1


def test_plain_fallback_hides_arguments_and_rich_lines_are_bounded():
    private_path = "/home/alice/private/customer-records.csv"
    body, html = matrix_tool_activity_bodies(
        [f"📖 read_file: {private_path} " + ("x" * 240) + "\nignored second line"]
    )

    assert body == "🛠 Tool activity (1 update)"
    assert private_path not in body
    assert private_path in html
    assert "ignored second line" not in html
    assert ("x" * 160) not in html
    assert "..." in html


def test_sanitize_keeps_ol_li_and_strips_details():
    html = (
        "<p><strong>🛠 Tool activity (1 update)</strong></p>"
        "<ol><li>💻 terminal: ls</li></ol>"
        "<details><summary>x</summary>secret</details>"
    )
    out = _sanitize_matrix_html(html)
    assert "<ol>" in out and "<li>" in out
    assert "terminal: ls" in out
    assert "<details>" not in out
    assert "<summary>" not in out


@pytest.mark.asyncio
async def test_matrix_send_and_edit_carry_html():
    adapter = object.__new__(MatrixAdapter)
    adapter._client = MagicMock()
    adapter._encryption = False
    adapter.format_message = lambda c: c
    adapter.truncate_message = lambda c, n: [c]
    adapter._build_text_message_content = lambda c: {"msgtype": "m.text", "body": c}
    adapter._apply_relation_metadata = lambda *a, **k: None

    sent = {}

    async def _send_evt(room, etype, content):
        sent.setdefault("events", []).append(content)
        return f"$e{len(sent['events'])}"

    adapter._client.send_message_event = _send_evt
    body, html = matrix_tool_activity_bodies(["💻 terminal: ls", "📖 read_file: x"])

    res = await MatrixAdapter.send(
        adapter,
        "!room:ex",
        body,
        metadata={
            "matrix_formatted_body": html,
            "matrix_formatted_body_unprefixed": True,
        },
    )
    assert res.success
    assert sent["events"][0]["format"] == "org.matrix.custom.html"
    assert "<ol>" in sent["events"][0]["formatted_body"]

    root_id = str(res.message_id or "")
    res2 = await MatrixAdapter.edit_message(
        adapter,
        "!room:ex",
        root_id,
        body,
        metadata={
            "matrix_formatted_body": html,
            "matrix_formatted_body_unprefixed": True,
        },
    )
    assert res2.success
    assert res2.message_id == root_id  # sticky root
    edit = sent["events"][1]
    assert edit["m.relates_to"]["rel_type"] == "m.replace"
    assert edit["m.new_content"]["formatted_body"].startswith("<p>")
    assert not edit["formatted_body"].startswith("*")
    assert "```" not in edit["m.new_content"]["formatted_body"]
    assert "<details>" not in edit["m.new_content"]["formatted_body"]


@pytest.mark.asyncio
@pytest.mark.parametrize("flag", [None, False, "false"])
async def test_unflagged_rich_edit_keeps_matrix_replacement_prefix(flag):
    adapter = object.__new__(MatrixAdapter)
    sent = {}

    async def _send_evt(room, etype, content):
        sent["content"] = content
        return "$replacement"

    adapter._client = MagicMock(send_message_event=_send_evt)
    adapter.format_message = lambda c: c
    adapter._build_text_message_content = lambda c: {"msgtype": "m.text", "body": c}

    result = await MatrixAdapter.edit_message(
        adapter,
        "!room:ex",
        "$root",
        "Updated content",
        metadata={"matrix_formatted_body": "<p>Updated content</p>", "matrix_formatted_body_unprefixed": flag},
    )

    assert result.success
    assert sent["content"]["formatted_body"].startswith("* ")
    assert not sent["content"]["m.new_content"]["formatted_body"].startswith("* ")


def _progress_runner():
    adapter = SimpleNamespace(
        name="matrix", MAX_MESSAGE_LENGTH=40,
        edit_message=AsyncMock(return_value=SendResult(success=True, message_id="$root")),
        send=AsyncMock(return_value=SendResult(success=True, message_id="$root")),
    )
    ctx = TurnContext()
    ctx.source = SimpleNamespace(chat_id="!room:example.org", platform="matrix")
    ctx._progress_metadata = {"thread_id": "$thread"}
    ctx._progress_reply_to = "$user"
    ctx.progress_grouping = "accumulate"
    return TurnRunner(SimpleNamespace(), ctx), adapter


@pytest.mark.asyncio
async def test_progress_producer_sends_html_then_edits_same_root_without_plain_labels():
    runner, adapter = _progress_runner()
    state = runner._progress_edit_state(adapter)
    runner._progress_absorb(state, "read_file: private-looking-path")
    await runner._progress_send_or_edit(state, state.progress_lines[-1])
    runner._progress_absorb(state, "terminal: second tool")
    assert not await runner._roll_progress_overflow_if_needed(state)
    await runner._progress_send_or_edit(state, state.progress_lines[-1])
    first = adapter.send.call_args.kwargs
    edit = adapter.edit_message.call_args.kwargs
    assert first["content"] == "🛠 Tool activity (1 update)"
    assert first["metadata"]["thread_id"] == "$thread"
    assert first["metadata"]["_interim_send"] is True
    assert first["metadata"]["matrix_formatted_body_unprefixed"] is True
    assert "private-looking-path" in first["metadata"]["matrix_formatted_body"]
    assert edit["content"] == "🛠 Tool activity (2 updates)"
    assert edit["message_id"] == "$root"
    assert "<ol>" in edit["metadata"]["matrix_formatted_body"]
    assert "matrix_formatted_body" not in runner._ctx._progress_metadata


@pytest.mark.asyncio
@pytest.mark.parametrize("error", ["forbidden", "flood control", "timeout"])
async def test_progress_edit_failure_never_falls_back_to_new_message(error):
    runner, adapter = _progress_runner()
    state = runner._progress_edit_state(adapter)
    runner._progress_absorb(state, "read_file: x")
    await runner._progress_send_or_edit(state, "read_file: x")
    adapter.edit_message.return_value = SendResult(success=False, error=error)
    runner._progress_absorb(state, "terminal: next")
    await runner._progress_send_or_edit(state, "terminal: next")
    await runner._progress_send_or_edit(state, "terminal: next")
    assert adapter.send.call_count == 1
    assert state.progress_msg_id == "$root"
    assert state.can_edit

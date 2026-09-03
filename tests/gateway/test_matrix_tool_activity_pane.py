"""Matrix Tool activity pane — first-principles contract tests."""

from __future__ import annotations

import asyncio
import inspect
from unittest.mock import MagicMock

import pytest

from gateway.matrix_activity_pane import MatrixActivityPane
from gateway.matrix_tool_activity import matrix_tool_activity_bodies
from gateway.run import _send_or_update_status_coro
from gateway.platforms.base import SendResult
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


def test_matrix_activity_keeps_a_bounded_recent_window():
    max_lines = 16
    lines = [f"tool update {index}" for index in range(max_lines + 1)]

    body, html = matrix_tool_activity_bodies(lines)

    assert body == (
        f"🛠 Tool activity ({len(lines)} updates, latest {max_lines} shown)"
    )
    assert html.count("<li>") == max_lines
    assert "tool update 0" not in html
    assert f"tool update {max_lines}" in html


def test_matrix_tools_and_footer_render_in_locked_order():
    body, html = matrix_tool_activity_bodies(
        ["🔍 Recall & inspect", "💻 terminal: a < b"],
        "⏳ Working — 2 min & waiting <now>",
    )

    assert body == (
        "🛠 Tool activity (2 updates) · "
        "⏳ Working — 2 min & waiting <now>"
    )
    assert html == (
        "<p><strong>🛠 Tool activity (2 updates)</strong></p>"
        "<ol><li>🔍 Recall &amp; inspect</li>"
        "<li>💻 terminal: a &lt; b</li></ol>"
        "<p>⏳ Working — 2 min &amp; waiting &lt;now&gt;</p>"
    )


def test_matrix_footer_only_has_no_empty_list_or_zero_count():
    body, html = matrix_tool_activity_bodies([], "⏳ Working — 1 min")

    assert body == "🛠 Tool activity · ⏳ Working — 1 min"
    assert html == (
        "<p><strong>🛠 Tool activity</strong></p>"
        "<p>⏳ Working — 1 min</p>"
    )
    assert "<ol>" not in html
    assert "0 updates" not in html


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
async def test_unflagged_rich_edit_keeps_matrix_replacement_prefix():
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
        metadata={"matrix_formatted_body": "<p>Updated content</p>"},
    )

    assert result.success
    assert sent["content"]["formatted_body"].startswith("* ")
    assert not sent["content"]["m.new_content"]["formatted_body"].startswith("* ")


def test_edit_message_accepts_metadata():
    assert "metadata" in inspect.signature(MatrixAdapter.edit_message).parameters


def test_matrix_one_line_label_contract_for_terminal():
    """What progress_callback should enqueue for terminal on Matrix."""
    cmd = "set -euo pipefail\nscp foo bar\n"
    first = cmd.strip().splitlines()[0]
    label = f"💻 terminal: {first[:80]}"
    assert "```" not in label
    assert "\n" not in label
    assert label.startswith("💻 terminal:")


class _PaneAdapter:
    name = "matrix"

    def __init__(self):
        self.sends = []
        self.edits = []
        self.send_results = []
        self.edit_results = []
        self.send_entered = None
        self.send_release = None
        self.edit_entered = None
        self.edit_release = None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sends.append(
            {
                "chat_id": chat_id,
                "content": content,
                "reply_to": reply_to,
                "metadata": dict(metadata or {}),
            }
        )
        if self.send_entered is not None:
            self.send_entered.set()
        if self.send_release is not None:
            await self.send_release.wait()
        if self.send_results:
            return self.send_results.pop(0)
        return SendResult(success=True, message_id="$root")

    async def edit_message(
        self, chat_id, message_id, content, *, finalize=False, metadata=None
    ):
        self.edits.append(
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "content": content,
                "metadata": dict(metadata or {}),
            }
        )
        if self.edit_entered is not None:
            self.edit_entered.set()
        if self.edit_release is not None:
            await self.edit_release.wait()
        if self.edit_results:
            return self.edit_results.pop(0)
        return SendResult(success=True, message_id=message_id)


def _pane(adapter=None, *, enabled=True):
    return MatrixActivityPane(
        adapter=adapter or _PaneAdapter(),
        chat_id="!room:example",
        reply_to="$user",
        metadata={"thread": "kept"},
        coalescing_enabled=enabled,
    )


@pytest.mark.asyncio
async def test_pane_tools_only_then_heartbeat_edits_original_root():
    adapter = _PaneAdapter()
    pane = _pane(adapter)

    await pane.append_activity("💻 terminal: ls")
    await pane.set_footer("⏳ Working — 1 min")

    assert len(adapter.sends) == 1
    assert len(adapter.edits) == 1
    assert adapter.sends[0]["content"] == "🛠 Tool activity (1 update)"
    assert adapter.sends[0]["metadata"]["_interim_send"] is True
    assert adapter.edits[0]["message_id"] == "$root"
    assert adapter.edits[0]["content"] == (
        "🛠 Tool activity (1 update) · ⏳ Working — 1 min"
    )


@pytest.mark.asyncio
async def test_pane_second_heartbeat_replaces_footer_then_tool_stays_above_it():
    adapter = _PaneAdapter()
    pane = _pane(adapter)

    await pane.set_footer("⏳ Working — 1 min")
    await pane.set_footer("⏳ Working — 2 min")
    await pane.append_activity("📖 read_file: x")

    assert len(adapter.sends) == 1
    assert len(adapter.edits) == 2
    final = adapter.edits[-1]
    assert final["content"] == (
        "🛠 Tool activity (1 update) · ⏳ Working — 2 min"
    )
    assert "1 min" not in final["metadata"]["matrix_formatted_body"]
    assert final["metadata"]["matrix_formatted_body"].index("<ol>") < (
        final["metadata"]["matrix_formatted_body"].index("⏳ Working")
    )


@pytest.mark.asyncio
async def test_pane_status_then_tool_preserves_activity_order():
    adapter = _PaneAdapter()
    pane = _pane(adapter)

    await pane.append_activity("🔍 Searching past sessions")
    await pane.append_activity("💻 terminal: rg Matrix")

    html = adapter.edits[-1]["metadata"]["matrix_formatted_body"]
    assert html.index("Searching past sessions") < html.index("terminal: rg Matrix")


@pytest.mark.asyncio
async def test_concurrent_first_tool_and_heartbeat_send_exactly_one_root():
    adapter = _PaneAdapter()
    adapter.send_entered = asyncio.Event()
    adapter.send_release = asyncio.Event()
    pane = _pane(adapter)

    tool_task = asyncio.create_task(pane.append_activity("💻 terminal: ls"))
    await adapter.send_entered.wait()
    heartbeat_task = asyncio.create_task(pane.set_footer("⏳ Working — 1 min"))
    await asyncio.sleep(0)
    adapter.send_release.set()
    await asyncio.gather(tool_task, heartbeat_task)

    assert len(adapter.sends) == 1
    assert len(adapter.edits) == 1
    assert adapter.edits[0]["message_id"] == "$root"
    assert "terminal: ls" in adapter.edits[0]["metadata"]["matrix_formatted_body"]
    assert "⏳ Working — 1 min" in adapter.edits[0]["metadata"]["matrix_formatted_body"]


@pytest.mark.asyncio
async def test_concurrent_tool_and_heartbeat_edits_finish_with_both():
    adapter = _PaneAdapter()
    pane = _pane(adapter)
    await pane.append_activity("🔍 recall")
    adapter.edit_entered = asyncio.Event()
    adapter.edit_release = asyncio.Event()

    tool_task = asyncio.create_task(pane.append_activity("💻 terminal: ls"))
    await adapter.edit_entered.wait()
    heartbeat_task = asyncio.create_task(pane.set_footer("⏳ Working — 1 min"))
    await asyncio.sleep(0)
    adapter.edit_release.set()
    await asyncio.gather(tool_task, heartbeat_task)

    final_html = adapter.edits[-1]["metadata"]["matrix_formatted_body"]
    assert "terminal: ls" in final_html
    assert "⏳ Working — 1 min" in final_html


@pytest.mark.asyncio
async def test_failed_initial_send_retries_full_snapshot_on_later_update():
    adapter = _PaneAdapter()
    adapter.send_results = [
        SendResult(success=False, error="offline"),
        SendResult(success=True, message_id="$retry-root"),
    ]
    pane = _pane(adapter)

    await pane.append_activity("🔍 recall")
    assert pane.root_event_id is None
    await pane.set_footer("⏳ Working — 1 min")

    assert len(adapter.sends) == 2
    assert len(adapter.edits) == 0
    assert pane.root_event_id == "$retry-root"
    retry_html = adapter.sends[-1]["metadata"]["matrix_formatted_body"]
    assert "recall" in retry_html and "⏳ Working — 1 min" in retry_html


@pytest.mark.asyncio
async def test_edit_failure_keeps_root_and_never_sends_second_root():
    adapter = _PaneAdapter()
    adapter.edit_results = [
        SendResult(success=False, error="temporary"),
        SendResult(success=True, message_id="$root"),
    ]
    pane = _pane(adapter)

    await pane.append_activity("🔍 recall")
    await pane.set_footer("⏳ Working — 1 min")
    await pane.append_activity("💻 terminal: ls")

    assert len(adapter.sends) == 1
    assert len(adapter.edits) == 2
    assert pane.root_event_id == "$root"
    assert "terminal: ls" in adapter.edits[-1]["metadata"]["matrix_formatted_body"]
    assert "⏳ Working — 1 min" in adapter.edits[-1]["metadata"]["matrix_formatted_body"]


@pytest.mark.asyncio
async def test_transport_exception_is_fail_soft_and_snapshot_retries():
    adapter = _PaneAdapter()
    original_send = adapter.send
    attempts = 0

    async def _raise_once(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("offline")
        return await original_send(*args, **kwargs)

    adapter.send = _raise_once
    pane = _pane(adapter)

    assert await pane.append_activity("🔍 recall") is None
    await pane.append_activity("💻 terminal: ls")

    assert pane.root_event_id == "$root"
    html = adapter.sends[-1]["metadata"]["matrix_formatted_body"]
    assert "recall" in html and "terminal: ls" in html


@pytest.mark.asyncio
async def test_close_blocks_later_heartbeat_edit():
    adapter = _PaneAdapter()
    pane = _pane(adapter)
    await pane.append_activity("🔍 recall")

    await pane.close()
    await pane.set_footer("⏳ Working — too late")

    assert pane.closed
    assert len(adapter.sends) == 1
    assert adapter.edits == []


@pytest.mark.asyncio
async def test_pane_metadata_is_unprefixed_but_standalone_status_is_not():
    pane_adapter = _PaneAdapter()
    pane = _pane(pane_adapter)
    await pane.set_footer("⏳ Working — 1 min")

    standalone_adapter = _PaneAdapter()
    await _send_or_update_status_coro(
        standalone_adapter,
        "!room:example",
        "recall",
        "🔍 recall",
        {"thread": "kept"},
        None,
    )

    assert pane_adapter.sends[0]["metadata"][
        "matrix_formatted_body_unprefixed"
    ] is True
    assert "matrix_formatted_body_unprefixed" not in (
        standalone_adapter.sends[0]["metadata"]
    )


@pytest.mark.asyncio
async def test_disabled_pane_does_not_capture_status():
    pane_adapter = _PaneAdapter()
    pane = _pane(pane_adapter, enabled=False)
    standalone_adapter = _PaneAdapter()

    await _send_or_update_status_coro(
        standalone_adapter,
        "!room:example",
        "recall",
        "🔍 recall",
        None,
        pane,
    )

    assert pane_adapter.sends == []
    assert len(standalone_adapter.sends) == 1

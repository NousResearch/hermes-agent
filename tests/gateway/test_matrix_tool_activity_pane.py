"""Matrix Tool activity pane — first-principles contract tests."""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import pytest

from gateway.platforms.base import SendResult
from plugins.platforms.matrix.adapter import MatrixAdapter, _sanitize_matrix_html


def _render_like_production(lines: list[str]) -> tuple[str, str]:
    """Mirror gateway/run.py _matrix_tool_activity_bodies contract for unit tests.

    Production helper is nested inside send_progress_messages; keep this in sync
    via the assertions below (no fences, counter body, single ol list).
    """
    import html as _html
    import re

    cleaned: list[str] = []
    for line in lines:
        s = str(line or "").strip()
        if not s:
            continue
        if s in {"```", "~~~"} or set(s) <= {"`", "~", " "}:
            continue
        if s.startswith("```") or s.startswith("~~~"):
            continue
        s = s.splitlines()[0].strip()
        s = re.sub(r"\s+", " ", s)
        if len(s) > 160:
            s = s[:157] + "..."
        cleaned.append(s)
    n = len(cleaned)
    body = f"🛠 Tool activity ({n} update{'s' if n != 1 else ''})"
    if not cleaned:
        return body, f"<p><strong>{_html.escape(body)}</strong></p>"
    items = "".join(f"<li>{_html.escape(item)}</li>" for item in cleaned)
    html_body = f"<p><strong>{_html.escape(body)}</strong></p><ol>{items}</ol>"
    return body, html_body


def test_matrix_body_is_counter_only():
    body, html = _render_like_production(
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
    body, html = _render_like_production(
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


def test_sanitize_keeps_ol_li():
    html = (
        "<p><strong>🛠 Tool activity (1 update)</strong></p>"
        "<ol><li>💻 terminal: ls</li></ol>"
    )
    out = _sanitize_matrix_html(html)
    assert "<ol>" in out and "<li>" in out
    assert "terminal: ls" in out


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
    body, html = _render_like_production(["💻 terminal: ls", "📖 read_file: x"])

    res = await MatrixAdapter.send(
        adapter,
        "!room:ex",
        body,
        metadata={"matrix_formatted_body": html},
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
        metadata={"matrix_formatted_body": html},
    )
    assert res2.success
    assert res2.message_id == root_id  # sticky root
    edit = sent["events"][1]
    assert edit["m.relates_to"]["rel_type"] == "m.replace"
    assert edit["m.new_content"]["formatted_body"].startswith("<p>")
    assert not edit["formatted_body"].startswith("*")
    assert "```" not in edit["m.new_content"]["formatted_body"]


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

"""Golden file regression for outbound message rendering.

Each fixture pair (`<name>.input.txt` + `<name>.expected.json`) under
`fixtures/outbound/` triggers `adapter.send(chat_id, content)` and the
captured lark API call is compared against the expected JSON.

Add new fixtures by creating both files; no code change needed.
"""

import asyncio
import json

import pytest

pytest.importorskip("lark_oapi.channel")

from .conftest import FIXTURES_DIR, load_fixture, load_text_fixture

FIXTURES = FIXTURES_DIR


def _discover_outbound_fixtures():
    outbound_dir = FIXTURES_DIR / "outbound"
    return sorted(p.stem.removesuffix(".input") for p in outbound_dir.glob("*.input.txt"))


def _project_send(captured) -> dict:
    """Stable projection of a CapturedSend into the dict shape used in fixtures."""
    body = captured.body
    content_str = body.get("content", "{}")
    try:
        content_parsed = json.loads(content_str) if isinstance(content_str, str) else content_str
    except json.JSONDecodeError:
        content_parsed = content_str
    return {
        "endpoint": captured.endpoint,
        "msg_type": body.get("msg_type"),
        "content": content_parsed,
        "extra": {k: v for k, v in captured.extra.items() if k == "receive_id_type"},
    }


@pytest.mark.parametrize("fixture_name", _discover_outbound_fixtures())
def test_outbound_golden(fixture_name: str, adapter_harness):
    input_text = load_text_fixture("outbound", f"{fixture_name}.input.txt")
    expected = load_fixture("outbound", f"{fixture_name}.expected.json")

    async def _send():
        await adapter_harness.adapter.send(
            chat_id="oc_testchat",
            content=input_text,
        )

    asyncio.run(_send())

    assert len(adapter_harness.captured_sends) >= 1, (
        f"Expected at least 1 captured send for fixture {fixture_name!r}, got 0"
    )
    actual = _project_send(adapter_harness.captured_sends[0])
    assert actual == expected, (
        f"Outbound projection mismatch for {fixture_name!r}.\n"
        f"  expected: {json.dumps(expected, indent=2, ensure_ascii=False)}\n"
        f"  actual:   {json.dumps(actual,   indent=2, ensure_ascii=False)}"
    )


def test_markdown_table_sent_as_text_to_render_visible(adapter_harness):
    """Markdown tables must be sent with ``msg_type=text``.

    The Feishu client does not render markdown tables inside post
    ``tag:md`` nodes (the format produced by SDK ``tag_md_mode="native"``),
    so a table delivered as a post arrives blank. The adapter detects
    table syntax and ships the chunk as plain text so the cells stay
    visible to the user.
    """
    src = (FIXTURES / "outbound" / "markdown_table.input.txt").read_text(encoding="utf-8")

    async def _send():
        await adapter_harness.adapter.send(chat_id="oc_testchat", content=src)

    asyncio.run(_send())

    assert len(adapter_harness.captured_sends) >= 1, (
        "Expected at least 1 captured send for markdown table fixture."
    )
    captured = adapter_harness.captured_sends[0]
    body = captured.body
    assert body.get("msg_type") == "text", (
        "Markdown tables must be sent as msg_type=text; got "
        f"{body.get('msg_type')!r}.\nbody: "
        f"{json.dumps(body, ensure_ascii=False, indent=2)}"
    )
    raw_content = body.get("content", "")
    parsed = json.loads(raw_content) if isinstance(raw_content, str) else raw_content
    text = parsed.get("text", "") if isinstance(parsed, dict) else ""
    assert "Header A" in text
    assert "Cell 1" in text
    assert "Cell 4" in text


def test_markdown_table_only_downgrades_table_chunk(adapter_harness):
    """Mixed markdown should keep rich rendering outside table chunks."""
    src = "\n".join(
        [
            "# E2E H1",
            "",
            "- bullet-a",
            "",
            "| col1 | col2 |",
            "|---|---|",
            "| A | B |",
            "",
            "```python",
            "print(1)",
            "```",
        ]
    )

    async def _send():
        await adapter_harness.adapter.send(chat_id="oc_testchat", content=src)

    asyncio.run(_send())

    assert len(adapter_harness.captured_sends) == 3
    first, table, last = adapter_harness.captured_sends

    assert first.body.get("msg_type") == "post"
    first_content = json.loads(first.body["content"])
    first_md_nodes = [
        item
        for row in first_content["zh_cn"]["content"]
        for item in row
        if item.get("tag") == "md"
    ]
    assert any("# E2E H1" in item.get("text", "") for item in first_md_nodes)

    assert table.body.get("msg_type") == "text"
    table_content = json.loads(table.body["content"])
    assert table_content["text"] == "| col1 | col2 |\n|---|---|\n| A | B |"

    assert last.body.get("msg_type") == "post"
    last_content = json.loads(last.body["content"])
    last_md_nodes = [
        item
        for row in last_content["zh_cn"]["content"]
        for item in row
        if item.get("tag") == "md"
    ]
    assert any("```python\nprint(1)\n```" in item.get("text", "") for item in last_md_nodes)

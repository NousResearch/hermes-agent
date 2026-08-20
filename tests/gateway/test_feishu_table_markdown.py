"""Tests for Feishu adapter outbound markdown payload construction.

Reproduces the bug tracked in hermes-agent issue #52786:
`_build_outbound_payload` was force-downgrading any message containing a
markdown pipe table to ``msg_type=text``, so Feishu clients rendered the raw
pipe-and-dash source instead of a table.  Empirically current Feishu clients
render ``post``+``md`` tables natively, so the downgrade branch must be removed.

These tests guard the fix.  They invoke the real adapter via the project's
plugin-loader helper so that no ``sys.path`` / ``sys.modules`` games are
needed.
"""

from __future__ import annotations

import json

from tests.gateway._plugin_adapter_loader import load_plugin_adapter

_adapter = load_plugin_adapter("feishu")


def _call_build_outbound_payload(content: str) -> tuple[str, str]:
    """Invoke ``_build_outbound_payload`` on a bare adapter instance.

    ``_build_outbound_payload`` is a method that only uses module-level
    helpers (``_MARKDOWN_TABLE_RE``, ``_MARKDOWN_HINT_RE``,
    ``_build_markdown_post_payload``) and never touches ``self.*``, so a bare
    object is sufficient.
    """
    inst = object.__new__(_adapter.FeishuAdapter)
    return inst._build_outbound_payload(content)


def _md_texts_from_post_payload(payload_str: str) -> list[str]:
    """Pull every ``{tag:'md', text:'...'}`` element out of a Feishu post payload.

    Real payload shape::

        {"zh_cn": {"content": [[{"tag": "md", "text": "..."}], ...]}}

    Helpers and tests need to introspect the ``md`` blocks regardless of
    locale, so we walk the structure generically.
    """
    payload = json.loads(payload_str)
    if not isinstance(payload, dict):
        return []
    texts: list[str] = []
    for lang_val in payload.values():
        if not isinstance(lang_val, dict):
            continue
        content = lang_val.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, list):
                candidates = block
            else:
                candidates = [block]
            for el in candidates:
                if isinstance(el, dict) and el.get("tag") == "md":
                    texts.append(el.get("text", ""))
    return texts


def test_markdown_table_uses_post_not_text():
    """Regression test for issue #52786 (and its older sibling #23938).

    A message whose only markdown is a table must take the ``post`` path,
    not be downgraded to plain text.
    """
    content = (
        "| col A | col B |\n"
        "| ----- | ----- |\n"
        "| 1     | 2     |"
    )
    msg_type, payload_str = _call_build_outbound_payload(content)
    assert msg_type == "post", (
        f"expected 'post' for a markdown table (issue #52786), got {msg_type!r}; "
        "the table-downgrade branch in _build_outbound_payload has been re-introduced"
    )
    md_texts = _md_texts_from_post_payload(payload_str)
    assert md_texts, f"post payload must include at least one md element; got {payload_str!r}"
    joined = "".join(md_texts)
    assert "col A" in joined and "|" in joined, (
        "table text was lost or reformatted when switching from text to post"
    )


def test_indented_code_block_does_not_swallow_trailing_content():
    """Regression test: Feishu's ``md`` renderer swallows prose that follows a
    4-space-indented code block when it lives inside one large markdown element.

    ``_build_markdown_post_rows`` isolates indented code blocks into their own
    post row (like fenced blocks already were), so content after the block —
    commonly a ``xxx:`` line followed by an indented list — stays visible.
    Indented segments are emitted as ``text`` rows so they survive the Feishu
    renderer, while surrounding prose stays in ``md`` rows.
    """
    from plugins.platforms.feishu.adapter import _build_markdown_post_rows
    content = (
        "**加粗触发 post 渲染**\n"
        "前文冒号：\n"
        "\n"
        "    配置项一  值一\n"
        "    配置项二  值二\n"
        "\n"
        "冒号后的关键结论必须保留"
    )
    rows = _build_markdown_post_rows(content)
    md_texts = [row[0]["text"] for row in rows if row]
    assert len(md_texts) >= 3, (
        f"indented code block must be isolated into its own row (>=3 rows), "
        f"got {len(md_texts)}: {md_texts!r}"
    )
    joined = "\n".join(md_texts)
    assert "冒号后的关键结论必须保留" in joined, (
        "content after the indented code block was swallowed by the Feishu md "
        "renderer; the isolation fix in _build_markdown_post_rows is missing"
    )
    assert "配置项一" in joined, "indented block content itself was lost"


def test_indented_code_block_emitted_as_text_row_preserving_markdown() -> None:
    """Regression test: a message carrying BOTH a markdown hint (e.g. ``**``)
    and a 4-space-indented block must stay a ``post`` message — markdown
    formatting for the non-indented part must NOT be sacrificed. The indented
    block itself is emitted as a ``text`` row (Feishu renders those verbatim,
    indentation preserved), and the rest keeps ``md`` rows.

    This is the user-visible bug: a reply with ``**加粗**`` heading + indented
    body arrived showing only the heading lines — every indented paragraph was
    silently dropped by Feishu's post renderer. The fix must keep ``**``
    rendering bold while making the indented body visible again.
    """
    content = (
        "**⚠️ 出发前必办两件事**\n"
        "\n"
        "    1. 门票必须提前线上预约！\n"
        "       暑期（7/23-8/31）景区只开网络预约。\n"
        "\n"
        "    2. 学生证必须带实体证件！\n"
        "\n"
        "结尾正常显示"
    )
    msg_type, payload_str = _call_build_outbound_payload(content)
    assert msg_type == "post", (
        f"expected 'post' (markdown preserved) for indented block + markdown, "
        f"got {msg_type!r}"
    )
    payload = json.loads(payload_str)
    rows = payload["zh_cn"]["content"]
    tags = [row[0]["tag"] for row in rows if row]
    assert "md" in tags, "markdown rows must still be emitted"
    assert "text" in tags, "indented block must be emitted as a text row"
    md_texts = [row[0]["text"] for row in rows if row and row[0]["tag"] == "md"]
    text_rows = [row[0]["text"] for row in rows if row and row[0]["tag"] == "text"]
    joined_md = "\n".join(md_texts)
    assert "**⚠️ 出发前必办两件事**" in joined_md, (
        "markdown hint must stay in an md row (so Feishu renders it bold)"
    )
    assert "结尾正常显示" in joined_md, "trailing prose must survive"
    joined_text = "\n".join(text_rows)
    assert "门票必须提前线上预约" in joined_text, "indented content must survive in text row"
    assert "    " in joined_text, "indentation must be preserved in text row"



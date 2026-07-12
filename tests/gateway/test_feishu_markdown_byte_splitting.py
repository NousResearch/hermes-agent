"""Tests for UTF-8 byte-safe markdown element splitting in Feishu Card renderer."""

import json
import re
import subprocess
import sys
import textwrap

import pytest

import gateway.platforms.feishu_card_renderer as renderer
from gateway.platforms.feishu_card_renderer import build_feishu_card_v2_payloads


def _markdown_elements(payloads):
    """Extract all markdown elements from card payloads."""
    elements = []
    for p in payloads:
        card = json.loads(p)
        for el in card.get("body", {}).get("elements", []):
            if el.get("tag") == "markdown":
                elements.append(el)
    return elements


def test_long_unbroken_ascii_is_split_by_byte_budget():
    """Long ASCII-only content with no newlines must be split so each piece
    fits within the 3000-byte UTF-8 budget."""
    payloads = build_feishu_card_v2_payloads("x" * 9000)
    for el in _markdown_elements(payloads):
        assert len(el["content"].encode("utf-8")) <= 3000, (
            f"element exceeds 3000 bytes: {len(el['content'].encode('utf-8'))}"
        )


def test_long_chinese_is_split_by_utf8_bytes():
    """Each Chinese character is 3 bytes in UTF-8; 4000 chars = 12000 bytes.
    Each resulting element must stay within the 3000-byte budget."""
    payloads = build_feishu_card_v2_payloads("中" * 4000)
    for el in _markdown_elements(payloads):
        assert len(el["content"].encode("utf-8")) <= 3000


def test_split_code_blocks_have_balanced_fences():
    """When a code block is split, each piece must have balanced ``` fences."""
    code = "```python\n" + "x = 1\n" * 500 + "```"
    payloads = build_feishu_card_v2_payloads(code)
    for el in _markdown_elements(payloads):
        assert el["content"].count("```") % 2 == 0, "unbalanced fence"
        assert len(el["content"].encode("utf-8")) <= 3000


def test_fenced_multibyte_chunks_include_fence_overhead_in_byte_budget():
    code = "```python\n" + ("中" * 5000) + "\n```"

    elements = _markdown_elements(build_feishu_card_v2_payloads(code))

    assert len(elements) > 1
    assert all(len(element["content"].encode("utf-8")) <= 3000 for element in elements)
    assert all(element["content"].count("```") % 2 == 0 for element in elements)


def test_public_pipeline_degrades_oversized_fence_opener_losslessly_with_timeout():
    """An oversized info string must terminate and remain visible losslessly."""
    script = textwrap.dedent(
        r'''
        import json

        from gateway.platforms.feishu_card_renderer import build_feishu_card_v2_payloads

        source = "```" + ("lang" * 1000) + "\nBODY\n```"
        payloads = build_feishu_card_v2_payloads(source)
        chunks = [
            element["content"]
            for payload in payloads
            for element in json.loads(payload)["body"]["elements"]
            if element.get("tag") == "markdown"
        ]
        assert chunks
        assert all(len(chunk.encode("utf-8")) <= 3000 for chunk in chunks)
        assert "".join(chunks).replace("\\`", "`") == source
        print(json.dumps({"chunks": len(chunks)}))
        '''
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(completed.stdout)["chunks"] >= 2


def test_splitter_raises_if_a_round_does_not_reduce_remaining_utf8_bytes(monkeypatch):
    calls = 0

    def fake_open_fence_language(_text):
        nonlocal calls
        calls += 1
        if calls > 2:
            raise AssertionError("splitter started another non-progressing round")
        return "x" * 4000

    monkeypatch.setattr(renderer, "_open_fence_language", fake_open_fence_language)

    with pytest.raises(renderer.FeishuCardRenderingError, match="made no UTF-8 byte progress"):
        renderer._split_markdown_by_utf8_bytes("a" * 4000, 3000)


def test_markdown_link_is_not_cut_mid_token():
    """Links [text](url) should not be cut in the middle of the URL."""
    text = "See [this is a very long link](https://example.com/" + "x" * 200 + ") for details."
    text = text * 20
    payloads = build_feishu_card_v2_payloads(text)
    for el in _markdown_elements(payloads):
        content = el["content"]
        # Check that there's no unclosed ]( — every ]( must have a matching )
        # Simple heuristic: count of ]( should equal count of ) after the last ]
        open_count = content.count("](")
        close_count = content.count(")")
        # If there are ]( without enough ), it's a broken link
        assert open_count <= close_count, (
            f"Unclosed markdown link: {open_count} ]( vs {close_count} )"
        )


def test_single_link_larger_than_budget_is_degraded_without_cutting_link_token():
    url = "https://example.com/" + ("路径" * 1800)
    source = f"[完整链接]({url})"

    elements = _markdown_elements(build_feishu_card_v2_payloads(source))

    assert len(elements) > 1
    assert all(len(element["content"].encode("utf-8")) <= 3000 for element in elements)
    combined = "".join(element["content"] for element in elements)
    visible = re.sub(r"\\([\\`*_{}\[\]()#+\-.!|>~])", r"\1", combined)
    assert source in visible
    assert all(
        element["content"].count("](") <= element["content"].count(")")
        for element in elements
    )


def test_single_inline_code_token_larger_than_budget_is_safely_degraded():
    source = "`" + ("变量值" * 1800) + "`"

    elements = _markdown_elements(build_feishu_card_v2_payloads(source))

    assert len(elements) > 1
    assert all(len(element["content"].encode("utf-8")) <= 3000 for element in elements)
    combined = "".join(element["content"] for element in elements)
    visible = combined.replace("\\`", "`")
    assert source in visible
    assert all(
        not re.search(r"(?<!\\)`", element["content"])
        for element in elements
    )


def _expected_atom_degradation(raw, special):
    """Independently escape only delimiters that are not escaped already."""
    result = []
    for index, char in enumerate(raw):
        if char in special:
            slash_count = 0
            cursor = index - 1
            while cursor >= 0 and raw[cursor] == "\\":
                slash_count += 1
                cursor -= 1
            if slash_count % 2 == 0:
                result.append("\\")
        result.append(char)
    return "".join(result)


def test_public_pipeline_preserves_existing_escape_parity_in_oversized_link():
    source = "[" + ("a" * 100) + r"\`" + ("b" * 3100) + r"\`" + "](u)"

    elements = _markdown_elements(build_feishu_card_v2_payloads(source))
    chunks = [element["content"] for element in elements]

    assert chunks
    assert all(len(chunk.encode("utf-8")) <= 3000 for chunk in chunks)
    assert "".join(chunks) == _expected_atom_degradation(source, "`[]()")


@pytest.mark.parametrize("slash_count", range(4))
def test_oversized_atom_escape_parity_combinations_are_lossless(slash_count):
    marker = ("\\" * slash_count) + "`"
    source = "[" + ("p" * 120) + marker + ("x" * 3100) + marker + "](url)"
    expected = _expected_atom_degradation(source, "`[]()")

    elements = _markdown_elements(build_feishu_card_v2_payloads(source))
    chunks = [element["content"] for element in elements]

    assert chunks
    assert all(len(chunk.encode("utf-8")) <= 3000 for chunk in chunks)
    assert "".join(chunks) == expected


def test_each_remaining_round_degrades_newly_exposed_oversized_atom():
    nested_link = "[" + ("x" * 3100) + "](url)"
    source = "`" + ("prefix" * 20) + nested_link + "`"
    expected = r"\`" + ("prefix" * 20) + _expected_atom_degradation(
        nested_link, "`[]()"
    ) + r"\`"

    elements = _markdown_elements(build_feishu_card_v2_payloads(source))
    chunks = [element["content"] for element in elements]

    assert len(chunks) > 1
    assert all(len(chunk.encode("utf-8")) <= 3000 for chunk in chunks)
    assert "".join(chunks) == expected


def test_no_empty_chunks():
    """No split piece should be empty."""
    payloads = build_feishu_card_v2_payloads("x" * 9000)
    for el in _markdown_elements(payloads):
        assert len(el["content"]) > 0


def test_short_content_unchanged():
    """Short content should produce a single card without splitting."""
    payloads = build_feishu_card_v2_payloads("short text")
    assert len(payloads) == 1


def test_chinese_no_mojibake():
    """Splitting must not break multi-byte UTF-8 characters.
    Each piece must be valid UTF-8 and decodable."""
    payloads = build_feishu_card_v2_payloads("中文测试" * 1000)
    for el in _markdown_elements(payloads):
        content = el["content"]
        # Must be encodable and re-decodable without error
        raw = content.encode("utf-8")
        assert raw.decode("utf-8") == content
        # No partial characters — the content should not end or start with
        # a continuation byte
        assert len(raw) <= 3000

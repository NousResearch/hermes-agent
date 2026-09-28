"""Keep complete links in one chunk after prose (regression for #125885)."""
import re

import pytest

from gateway.platforms.base import BasePlatformAdapter, utf16_len
from plugins.platforms.telegram.adapter import TelegramAdapter


def test_formatted_links_survive_chunk_boundaries():
    fmt = TelegramAdapter.format_message.__get__(object.__new__(TelegramAdapter))
    prose = "This synthetic statement is supported by a retained source. " * 90
    prefix = next(
        prose[:i + 1] for i in range(len(prose) - 1, -1, -1)
        if prose[i] == " " and utf16_len(fmt(prose[:i + 1])) <= 4040
    )
    for raw_link in [
        "[Mission Control issue 7](https://example.invalid/projects/synthetic/issues/7)",
        "[Mission Control issue 7](https://example.invalid/a_(b))",
        "[Mission Control 😀 issue 7](https://example.invalid/path)",
        "[Mission Control [issue 7](https://example.invalid/path)",
    ]:
        formatted = fmt(prefix + raw_link + " tail " * 30)
        link = fmt(raw_link)
        chunks = BasePlatformAdapter.truncate_message(formatted, 4096, len_fn=utf16_len)
        assert len(chunks) > 1
        assert sum(link in chunk for chunk in chunks) == 1
        assert all(utf16_len(chunk) <= 4096 for chunk in chunks)
        joined = "".join(re.sub(r" \(\d+/\d+\)$", "", c) for c in chunks)
        assert re.sub(r"\s+", "", joined) == re.sub(r"\s+", "", formatted)


@pytest.mark.parametrize("limit", [40, 80])
def test_link_chunking_preserves_fallback_and_short_messages(limit):
    for content in ["", "plain short", "[short link](https://example.invalid)"]:
        if len(content) <= limit:
            assert BasePlatformAdapter.truncate_message(content, limit) == [content]
    link = "[" + "word " * 7 + "](https://example.invalid)"
    if len(link) <= limit - 14:
        chunks = BasePlatformAdapter.truncate_message(link + " tail " * 30, limit)
        assert sum(link in chunk for chunk in chunks) == 1
        assert all(len(chunk) <= limit for chunk in chunks)
    for content in [
        "prefix " + "[overlong label " * 15 + "](https://example.invalid)",
        "ordinary words " * 30,
        "prefix [unclosed link " * 20,
    ]:
        chunks = BasePlatformAdapter.truncate_message(content, limit)
        assert chunks and all(len(chunk) <= limit for chunk in chunks)
        joined = "".join(re.sub(r" \(\d+/\d+\)$", "", c) for c in chunks)
        assert re.sub(r"\s+", "", joined) == re.sub(r"\s+", "", content)

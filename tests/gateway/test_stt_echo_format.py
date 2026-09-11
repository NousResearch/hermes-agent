"""Tests for gateway STT transcript echo formatting."""

import html
import re

from gateway.config import Platform
from gateway.platforms.base import utf16_len
from gateway.stt_echo import (
    chunk_telegram_stt_echo_html,
    format_stt_transcript_echo,
    stt_echo_metadata,
)

_OPEN = "<blockquote expandable>"
_CLOSE = "</blockquote>"
_INDICATOR_RE = re.compile(r" \((\d+)/(\d+)\)$")


def _quote_body(chunk: str) -> str:
    start = chunk.index(_OPEN) + len(_OPEN)
    end = chunk.rindex(_CLOSE)
    return chunk[start:end]


def test_non_telegram_keeps_classic_quoted_line():
    assert format_stt_transcript_echo("hello once", Platform.DISCORD) == '🎙️ "hello once"'
    assert format_stt_transcript_echo("hello once", "whatsapp") == '🎙️ "hello once"'
    assert stt_echo_metadata(Platform.DISCORD, {"thread_id": 1}) == {"thread_id": 1}


def test_telegram_uses_html_expandable_blockquote():
    formatted = format_stt_transcript_echo("hello once", Platform.TELEGRAM)

    assert formatted == "🎙️\n<blockquote expandable>hello once</blockquote>"
    assert stt_echo_metadata(Platform.TELEGRAM, None) == {"telegram_html": True}
    assert stt_echo_metadata("telegram", {"thread_id": 7}) == {
        "thread_id": 7,
        "telegram_html": True,
    }


def test_telegram_multiline_and_markdown_chars_are_html_escaped():
    formatted = format_stt_transcript_echo(
        "line one\n**bold** & <tag>\nline three",
        "telegram",
    )

    assert formatted == (
        "🎙️\n"
        "<blockquote expandable>"
        "line one\n"
        "**bold** &amp; &lt;tag&gt;\n"
        "line three"
        "</blockquote>"
    )


def test_chunker_returns_none_for_non_quote_html():
    assert chunk_telegram_stt_echo_html("plain text", 80, utf16_len) is None
    assert chunk_telegram_stt_echo_html("<b>bold</b>", 80) is None


def test_short_telegram_echo_stays_one_unindexed_chunk():
    formatted = format_stt_transcript_echo("hello once", "telegram")
    chunks = chunk_telegram_stt_echo_html(formatted, 4096, utf16_len)

    assert chunks == [formatted]
    assert "(1/" not in chunks[0]


def test_long_telegram_echo_wraps_every_chunk_as_expandable_quote():
    paragraphs = [f"paragraph {index:03d} " + ("word " * 20) for index in range(40)]
    transcript = "\n".join(paragraphs)
    formatted = format_stt_transcript_echo(transcript, "telegram")
    max_length = 400
    chunks = chunk_telegram_stt_echo_html(formatted, max_length, utf16_len)

    assert chunks is not None
    assert len(chunks) > 1
    total = len(chunks)
    for index, chunk in enumerate(chunks):
        assert chunk.startswith("🎙️\n" + _OPEN)
        assert _CLOSE in chunk
        assert utf16_len(chunk) <= max_length
        match = _INDICATOR_RE.search(chunk)
        assert match is not None
        assert int(match.group(1)) == index + 1
        assert int(match.group(2)) == total
        # * Indicator stays visible outside the collapsed quote.
        assert chunk.rstrip().endswith(f"({index + 1}/{total})")
        assert chunk.rindex(_CLOSE) < match.start()

    bodies = [_quote_body(chunk) for chunk in chunks]
    joined = html.unescape(" ".join(bodies))
    for paragraph in paragraphs:
        assert paragraph.strip() in joined


def test_chunker_does_not_split_inside_html_entities():
    transcript = ("alpha " * 30) + "& <tag> " + ("omega " * 30)
    formatted = format_stt_transcript_echo(transcript, "telegram")
    chunks = chunk_telegram_stt_echo_html(formatted, 180, utf16_len)

    assert chunks is not None
    assert len(chunks) > 1
    bodies = [_quote_body(chunk) for chunk in chunks]
    joined = "".join(bodies)
    assert "&amp;" in joined
    assert "&lt;tag&gt;" in joined
    # * A cut inside &amp; / &lt; / &gt; would leave a dangling &... with no ';'.
    assert re.search(r"&(?:amp|lt|gt);", joined)
    for body in bodies:
        assert not re.search(r"&(?:amp|lt|gt)(?!;)", body)

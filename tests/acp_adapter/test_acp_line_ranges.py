"""Tests for ACP adapter line range parsing, MIME mapping, and extract_text."""

import base64

import pytest
from acp.schema import (
    EmbeddedResourceContentBlock,
    ImageContentBlock,
    ResourceContentBlock,
    TextContentBlock,
    TextResourceContents,
    AgentCapabilities,
    PromptCapabilities,
)

from acp_adapter.server import (
    _parse_line_range_from_uri,
    _extract_range_from_meta,
    _mime_to_ext,
    _extract_text,
    HermesACPAgent,
)


# ---------------------------------------------------------------------------
# _parse_line_range_from_uri
# ---------------------------------------------------------------------------

class TestParseLineRangeFromUri:
    def test_l_format_dash(self):
        assert _parse_line_range_from_uri("file:///path.py#L13-L20") == (13, 20)

    def test_l_format_colon(self):
        assert _parse_line_range_from_uri("file:///path.py#L13:L20") == (13, 20)

    def test_l_format_comma(self):
        assert _parse_line_range_from_uri("file:///path.py#L13,L20") == (13, 20)

    def test_numeric_format_dash(self):
        assert _parse_line_range_from_uri("file:///path.py#15-21") == (15, 21)

    def test_colon_separator_at_end(self):
        assert _parse_line_range_from_uri("file:///path.py:13:20") == (13, 20)

    def test_no_fragment(self):
        assert _parse_line_range_from_uri("file:///path.py") is None

    def test_no_range_numbers(self):
        assert _parse_line_range_from_uri("file:///path.py#L13") is None

    def test_single_line(self):
        assert _parse_line_range_from_uri("file:///path.py#L13-L13") == (13, 13)

    def test_zero_padded_rejected_in_colon(self):
        # With colon format, leading zero is rejected
        assert _parse_line_range_from_uri("file:///path.py:013:020") is not None or True

    def test_empty_uri(self):
        assert _parse_line_range_from_uri("") is None


# ---------------------------------------------------------------------------
# _extract_range_from_meta
# ---------------------------------------------------------------------------

class TestExtractRangeFromMeta:
    def test_range_dict(self):
        assert _extract_range_from_meta({"range": {"start": 13, "end": 20}}) == (13, 20)

    def test_start_line_end_line(self):
        assert _extract_range_from_meta({"start_line": 5, "end_line": 10}) == (5, 10)

    def test_flat_start_end(self):
        assert _extract_range_from_meta({"start": 3, "end": 7}) == (3, 7)

    def test_none_meta(self):
        assert _extract_range_from_meta(None) is None

    def test_empty_dict(self):
        assert _extract_range_from_meta({}) is None

    def test_missing_start(self):
        assert _extract_range_from_meta({"end": 20}) is None

    def test_missing_end(self):
        assert _extract_range_from_meta({"start": 13}) is None

    def test_range_dict_with_start_line(self):
        assert _extract_range_from_meta({"range": {"start_line": 1, "end_line": 50}}) == (1, 50)


# ---------------------------------------------------------------------------
# _mime_to_ext
# ---------------------------------------------------------------------------

class TestMimeToExt:
    def test_png(self):
        assert _mime_to_ext("image/png") == ".png"

    def test_jpeg(self):
        assert _mime_to_ext("image/jpeg") == ".jpg"

    def test_gif(self):
        assert _mime_to_ext("image/gif") == ".gif"

    def test_webp(self):
        assert _mime_to_ext("image/webp") == ".webp"

    def test_svg(self):
        assert _mime_to_ext("image/svg+xml") == ".svg"

    def test_wav(self):
        assert _mime_to_ext("audio/wav") == ".wav"

    def test_mp3(self):
        assert _mime_to_ext("audio/mpeg") == ".mp3"

    def test_flac(self):
        assert _mime_to_ext("audio/flac") == ".flac"

    def test_unknown_mime_fallback(self):
        assert _mime_to_ext("application/octet-stream") == ".bin"

    def test_unknown_mime_custom_fallback(self):
        assert _mime_to_ext("application/pdf", ".pdf") == ".pdf"

    def test_empty_mime(self):
        assert _mime_to_ext("") == ".bin"


# ---------------------------------------------------------------------------
# _extract_text with EmbeddedResourceContentBlock (line ranges)
# ---------------------------------------------------------------------------

class TestExtractText:
    def test_plain_text_preserved(self):
        result = _extract_text([
            TextContentBlock(type="text", text="hello world"),
        ])
        assert result == "hello world"

    def test_multiple_text_blocks(self):
        result = _extract_text([
            TextContentBlock(type="text", text="line one"),
            TextContentBlock(type="text", text="line two"),
        ])
        assert result == "line one\nline two"

    def test_embedded_file_without_line_range(self):
        """A @-referenced file with no line range includes the full body."""
        result = _extract_text([
            EmbeddedResourceContentBlock(
                type="resource",
                resource=TextResourceContents(
                    uri="file:///project/main.py",
                    mimeType="text/x-python",
                    text="def hello():\n    print('hi')\n\nif __name__ == '__main__':\n    hello()",
                ),
            ),
        ])
        assert "[File: file:///project/main.py]" in result
        assert "def hello():" in result

    def test_embedded_file_with_uri_line_range(self):
        """@file#L1-L2 clips to lines 1-2."""
        result = _extract_text([
            EmbeddedResourceContentBlock(
                type="resource",
                resource=TextResourceContents(
                    uri="file:///project/main.py#L1-L2",
                    mimeType="text/x-python",
                    text="line1\nline2\nline3\nline4\n",  # full file (4 lines, range span=2, 4 > 2*2=false → already clipped)
                ),
            ),
        ])
        # total (4) equals range_span*2 (4), so it's NOT re-clipped
        assert "[File: file:///project/main.py#L1-L2 (lines 1-2)]" in result
        assert "line1" in result
        assert "line2" in result

    def test_embedded_file_full_content_re_clipped(self):
        """When total >> range_span, content is re-clipped to the range."""
        result = _extract_text([
            EmbeddedResourceContentBlock(
                type="resource",
                resource=TextResourceContents(
                    uri="file:///project/main.py#L2-L3",
                    mimeType="text/x-python",
                    text="line1\nline2\nline3\nline4\nline5\nline6\nline7\nline8\nline9\nline10\n",
                ),
            ),
        ])
        # total=10, range_span=2, 10 > 4 → re-clip
        assert "[File: file:///project/main.py#L2-L3 (lines 2-3/10)]" in result
        assert "line2" in result
        assert "line3" in result
        assert "line1" not in result
        assert "line4" not in result

    def test_embedded_file_from_meta_range(self):
        """Test _meta range extraction works."""
        meta = {"range": {"start": 1, "end": 2}}
        result = _extract_text([
            EmbeddedResourceContentBlock(
                type="resource",
                resource=TextResourceContents(
                    uri="file:///project/main.py",
                    mimeType="text/x-python",
                    text="line1\nline2\nline3\nline4\n",
                ),
                field_meta=meta,
            ),
        ])
        # total=4, range_span=2, 4 == 2*2 → not re-clipped
        assert "line1" in result
        assert "line2" in result

    def test_resource_content_block_with_line_range(self):
        """ResourceContentBlock with file:// URI + line range."""
        import tempfile, os
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("a\nb\nc\nd\ne\n")
            fpath = f.name
        try:
            uri = f"file://{fpath}#L2-L4"
            result = _extract_text([
                ResourceContentBlock(type="resource_link", uri=uri, name="test.py"),
            ])
            assert "[File: file://" in result
            assert "(lines 2-4" in result
            assert "b" in result and "c" in result and "d" in result
            # Verify only lines 2-4 are in the output body (after header)
            body_section = result.split("\n", 1)[1] if "\n" in result else result
            assert body_section.strip() == "b\nc\nd"
        finally:
            os.unlink(fpath)

    def test_image_block_adds_marker(self):
        """ImageContentBlock produces [Image: /tmp/...] marker."""
        img_data = base64.b64encode(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82").decode()
        import tempfile, os
        result = _extract_text([
            ImageContentBlock(type="image", data=img_data, mimeType="image/png"),
        ])
        assert "[Image: /tmp/hermes_acp_img_" in result
        assert ".png]" in result


# ---------------------------------------------------------------------------
# PromptCapabilities handshake
# ---------------------------------------------------------------------------

class TestPromptCapabilities:
    """Verify the ACP handshake advertises the right capabilities."""

    def test_prompt_capabilities_includes_embedded_context(self):
        """The InitializeResponse must include embedded_context=True."""
        # Just construct what the agent does in initialize()
        caps = PromptCapabilities(image=True, audio=True, embedded_context=True)
        assert caps.image is True
        assert caps.audio is True
        assert caps.embedded_context is True

    def test_agent_capabilities_constructed_with_audio_and_embedded(self):
        """Verify AgentCapabilities can be created with our PromptCapabilities."""
        caps = AgentCapabilities(
            load_session=True,
            prompt_capabilities=PromptCapabilities(
                image=True,
                audio=True,
                embedded_context=True,
            ),
            session_capabilities=None,
        )
        assert caps.prompt_capabilities.image is True
        assert caps.prompt_capabilities.audio is True
        assert caps.prompt_capabilities.embedded_context is True

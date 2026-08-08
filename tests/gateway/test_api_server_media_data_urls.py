"""MEDIA: tag → base64 data-URL resolution for the API server (salvage of #2696).

Remote OpenAI-compatible frontends can't read local file paths, so
``MEDIA:<path>`` image tags in final responses are inlined as markdown
data URLs before crossing the HTTP boundary.
"""

import base64
import unittest

import pytest

pytest.importorskip("aiohttp")

from gateway.platforms.api_server import (  # noqa: E402
    StreamingMediaTagResolver,
    _resolve_media_to_data_urls,
)

# 1x1 transparent PNG
_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQAB"
    "h6FO1AAAAABJRU5ErkJggg=="
)


class TestResolveMediaToDataUrls(unittest.TestCase):
    def _write_png(self, tmpdir_name="hermes_media_test"):
        import tempfile
        from pathlib import Path

        d = Path(tempfile.mkdtemp(prefix=tmpdir_name))
        p = d / "shot.png"
        p.write_bytes(_PNG_BYTES)
        return p

    def test_media_tag_inlined(self):
        p = self._write_png()
        out = _resolve_media_to_data_urls(f"Here you go: MEDIA:{p}")
        self.assertIn("data:image/png;base64,", out)
        self.assertNotIn("MEDIA:", out)

    def test_backtick_wrapped_tag(self):
        p = self._write_png()
        out = _resolve_media_to_data_urls(f"See `MEDIA:{p}` above")
        self.assertIn("data:image/png;base64,", out)

    def test_missing_file_left_untouched(self):
        text = "MEDIA:/nonexistent/path/shot.png"
        self.assertEqual(_resolve_media_to_data_urls(text), text)

    def test_non_image_left_untouched(self):
        text = "MEDIA:/tmp/archive.zip"
        self.assertEqual(_resolve_media_to_data_urls(text), text)


class TestStreamingMediaTagResolver(unittest.TestCase):
    """Real per-token LLM streaming can split a MEDIA:<path> tag's characters
    across many separate delta chunks -- confirmed in production (2026-07-28)
    across every SSE/streaming endpoint in api_server.py: only the
    non-streaming responses ever called _resolve_media_to_data_urls, so a
    split tag reached live clients as literal "MEDIA:/path..." text instead
    of an image, every time. These tests drive the resolver the same way the
    real streaming call sites do: many small .feed() calls, then .flush().
    """

    def setUp(self):
        import tempfile
        from pathlib import Path

        d = Path(tempfile.mkdtemp(prefix="hermes_media_stream_test"))
        self.png_path = d / "shot.png"
        self.png_path.write_bytes(_PNG_BYTES)

    def _feed_all(self, resolver, chunks):
        out = [resolver.feed(c) for c in chunks]
        out.append(resolver.flush())
        return "".join(out)

    def test_tag_split_across_many_chunks_resolves_to_data_url(self):
        path_str = str(self.png_path)
        # Split the tag character-by-character around "MEDIA:" plus the path
        # in a few pieces, mimicking real token-by-token streaming.
        chunks = ["Here you go: ", "MED", "IA:", path_str[:5], path_str[5:], " done!"]
        resolver = StreamingMediaTagResolver()
        out = self._feed_all(resolver, chunks)
        self.assertIn("data:image/png;base64,", out)
        self.assertNotIn("MEDIA:", out)
        self.assertIn("Here you go: ", out)
        self.assertIn(" done!", out)

    def test_plain_text_with_no_media_mention_flows_through_immediately(self):
        resolver = StreamingMediaTagResolver()
        # Each feed() call should return its own text immediately -- no
        # buffering latency when there's nothing that could be a tag.
        self.assertEqual(resolver.feed("Hello there, "), "Hello there, ")
        self.assertEqual(resolver.feed("how are you?"), "how are you?")
        self.assertEqual(resolver.flush(), "")

    def test_bogus_media_word_in_prose_eventually_releases_as_plain_text(self):
        resolver = StreamingMediaTagResolver()
        chunks = ["I love using social MEDIA", " apps", " every day", " to stay in touch."]
        out = self._feed_all(resolver, chunks)
        self.assertEqual(out, "I love using social MEDIA apps every day to stay in touch.")

    def test_holdback_cap_releases_a_media_word_that_never_completes(self):
        """A "MEDIA" mention followed by a very long run of unrelated text
        (well past _MEDIA_HOLDBACK_CAP) must not stall the stream forever --
        it releases as plain text instead of buffering indefinitely."""
        resolver = StreamingMediaTagResolver()
        out = resolver.feed("Talk about MEDIA" + "x" * 500)
        self.assertIn("MEDIAxxx", out)  # released, not held back forever

    def test_two_tags_in_separate_feed_calls_both_resolve(self):
        path_str = str(self.png_path)
        resolver = StreamingMediaTagResolver()
        out = self._feed_all(resolver, [
            f"First: MEDIA:{path_str} ",
            f"Second: MEDIA:{path_str}",
        ])
        self.assertEqual(out.count("data:image/png;base64,"), 2)
        self.assertNotIn("MEDIA:", out)

    def test_invalid_path_tag_split_across_chunks_stays_visible(self):
        """A tag that never validates (e.g. traversal outside the allowed
        roots) must still surface as visible text after reassembly -- the
        resolver only changes WHEN resolution happens, not the safety
        decision _resolve_media_to_data_urls already makes."""
        resolver = StreamingMediaTagResolver()
        out = self._feed_all(resolver, ["MED", "IA:/etc/nonexistent_", "file123.png"])
        self.assertIn("MEDIA:/etc/nonexistent_file123.png", out)


if __name__ == "__main__":
    unittest.main()

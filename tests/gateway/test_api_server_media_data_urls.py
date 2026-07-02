"""MEDIA: tag → base64 data-URL resolution for the API server (salvage of #2696).

Remote OpenAI-compatible frontends can't read local file paths, so
``MEDIA:<path>`` image tags in final responses are inlined as markdown
data URLs before crossing the HTTP boundary.
"""

import base64
import unittest

import pytest

pytest.importorskip("aiohttp")

from gateway.platforms.api_server import _resolve_media_to_data_urls  # noqa: E402

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

    def test_text_without_media_passthrough(self):
        self.assertEqual(_resolve_media_to_data_urls("plain text"), "plain text")
        self.assertEqual(_resolve_media_to_data_urls(""), "")

    def test_oversized_image_skipped(self):
        from gateway.platforms import api_server as mod

        p = self._write_png()
        orig = mod._MEDIA_DATA_URL_MAX_BYTES
        mod._MEDIA_DATA_URL_MAX_BYTES = 1
        try:
            text = f"MEDIA:{p}"
            self.assertEqual(_resolve_media_to_data_urls(text), text)
        finally:
            mod._MEDIA_DATA_URL_MAX_BYTES = orig

    def test_multiple_tags(self):
        p1 = self._write_png()
        p2 = self._write_png("hermes_media_test2")
        out = _resolve_media_to_data_urls(f"MEDIA:{p1}\nand MEDIA:{p2}")
        self.assertEqual(out.count("data:image/png;base64,"), 2)


if __name__ == "__main__":
    unittest.main()

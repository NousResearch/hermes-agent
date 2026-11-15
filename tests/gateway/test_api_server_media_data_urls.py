"""MEDIA: tag → base64 data-URL resolution for the API server (salvage of #2696).

Remote OpenAI-compatible frontends can't read local file paths, so
``MEDIA:<path>`` image tags in final responses are inlined as markdown
data URLs before crossing the HTTP boundary.
"""

import base64
import unittest

import pytest

pytest.importorskip("aiohttp")

from gateway.platforms.api_server import APIServerAdapter, _resolve_media_to_data_urls

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

    def test_terminal_eos_sentinel_does_not_block_inlining(self):
        """A leaked terminal ``<|eos|>`` glued to the tag (#111046) inlines exactly like the clean
        response; the control token is dropped rather than returned to the HTTP client."""
        p = self._write_png()
        clean = f"Here you go: MEDIA:{p}"
        self.assertEqual(_resolve_media_to_data_urls(clean + "<|eos|>"), _resolve_media_to_data_urls(clean))

    def test_missing_file_left_untouched(self):
        text = "MEDIA:/nonexistent/path/shot.png"
        self.assertEqual(_resolve_media_to_data_urls(text), text)

    def test_non_image_left_untouched(self):
        text = "MEDIA:/tmp/archive.zip"
        self.assertEqual(_resolve_media_to_data_urls(text), text)


class TestHistoryMessagesInlineMedia(unittest.TestCase):
    """``GET /api/sessions/{id}/messages`` was the one outbound path that skipped inlining.

    A stored reply keeps the raw ``MEDIA:<path>`` form, so a remote frontend opening an old
    session got a tag it has no filesystem to resolve. The projection now opts in, per request.
    """

    def _png(self):
        import tempfile
        from pathlib import Path

        d = Path(tempfile.mkdtemp(prefix="hermes_history_media"))
        p = d / "shot.png"
        p.write_bytes(_PNG_BYTES)
        return p

    @staticmethod
    def _row(role: str, content):
        return {"id": 1, "session_id": "s1", "role": role, "content": content, "timestamp": 0.0}

    def test_default_keeps_the_wire_shape(self):
        p = self._png()
        out = APIServerAdapter._message_response(self._row("assistant", f"Here: MEDIA:{p}"))
        self.assertIn("MEDIA:", out["content"])
        self.assertNotIn("data:image", out["content"])

    def test_opted_in_assistant_row_is_inlined(self):
        p = self._png()
        out = APIServerAdapter._message_response(
            self._row("assistant", f"Here: MEDIA:{p}"), inline_media=True)
        self.assertIn("data:image/png;base64,", out["content"])
        self.assertNotIn("MEDIA:", out["content"])

    def test_opted_in_leaves_non_assistant_rows_alone(self):
        p = self._png()
        out = APIServerAdapter._message_response(self._row("tool", f"MEDIA:{p}"), inline_media=True)
        self.assertIn("MEDIA:", out["content"])

    def test_null_content_is_not_a_crash(self):
        out = APIServerAdapter._message_response(self._row("tool", None), inline_media=True)
        self.assertIsNone(out["content"])


if __name__ == "__main__":
    unittest.main()

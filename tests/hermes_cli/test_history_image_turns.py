"""`/history` shows a native image turn as its text plus ``[image]``.

With native image input the staged user message holds the content-parts list from
``build_native_content_parts``; `/history` printed ``str()`` of that list, so the preview was a
Python repr filled with the image's base64 data URL. The `/resume` recap already renders such
rows through ``_user_display_text``.
"""

import base64
import io
import sys

from agent.image_routing import build_native_content_parts

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")


def _render_history(history):
    from cli import HermesCLI

    h = HermesCLI.__new__(HermesCLI)
    h.show_timestamps = False
    h.conversation_history = history
    buf, old = io.StringIO(), sys.stdout
    sys.stdout = buf
    try:
        h.show_history()
    finally:
        sys.stdout = old
    return buf.getvalue()


def test_native_image_turn_shows_text_and_image_marker(tmp_path):
    image = tmp_path / "photo.png"
    image.write_bytes(PNG)
    parts, skipped = build_native_content_parts("describe this", [str(image)])
    assert not skipped and any(p.get("type") == "image_url" for p in parts)

    out = _render_history([{"role": "user", "content": parts},
                           {"role": "assistant", "content": "A single pixel."}])

    assert "describe this" in out
    assert "[image]" in out
    assert "base64" not in out
    assert "{'type'" not in out
    assert "A single pixel." in out

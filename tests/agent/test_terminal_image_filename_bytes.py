"""POSIX image previews accept the same filesystem names as Pillow."""

import os

import pytest

from agent.terminal_image import render_inline_image

Image = pytest.importorskip("PIL.Image")


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("filename", [b"photo-\xff.png", b"photo-\x80.png"])
def test_kitty_preview_accepts_non_utf8_filename(tmp_path, filename):
    path = tmp_path / os.fsdecode(filename)
    Image.new("RGB", (40, 20), "red").save(path)

    preview = render_inline_image(path, mode="kitty", max_cols=20, max_rows=8)

    assert preview is not None
    assert preview.transmit and preview.placeholder
    again = render_inline_image(path, mode="kitty", max_cols=20, max_rows=8)
    assert again is not None and again.color == preview.color

"""Terminal graphics: capability detection and the inline-image API (hermes_agent/agent/terminal_image.py)."""

from __future__ import annotations

import json

import pytest

from hermes_agent.agent import terminal_image
from hermes_agent.agent.terminal_image import render_inline_image, resolve_inline_mode, tool_image_path

PIL = pytest.importorskip("PIL.Image")


def _png(path, size=(400, 200), color=(200, 30, 60, 255)):
    PIL.new("RGBA", size, color).save(path)
    return str(path)


# ---------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------


def _clear_graphics_env(monkeypatch):
    for key in ("KITTY_WINDOW_ID", "TERM_PROGRAM", "ITERM_SESSION_ID", "WEZTERM_PANE", "TERM", "KONSOLE_VERSION"):
        monkeypatch.delenv(key, raising=False)


def test_vscode_terminal_ignores_leaked_graphics_env(monkeypatch):
    # The VS Code / Cursor integrated terminal can't show inline images by
    # default, yet inherits ITERM_SESSION_ID/KITTY_WINDOW_ID when launched from
    # those terminals. TERM_PROGRAM=vscode must win → unicode, never a protocol
    # whose escapes the embedded terminal would silently drop.
    _clear_graphics_env(monkeypatch)
    monkeypatch.setenv("TERM_PROGRAM", "vscode")

    assert terminal_image.detect_terminal_graphics() == "unicode"
    for leaked in ("ITERM_SESSION_ID", "KITTY_WINDOW_ID", "WEZTERM_PANE"):
        monkeypatch.setenv(leaked, "1")
        assert terminal_image.detect_terminal_graphics() == "unicode"
        monkeypatch.delenv(leaked)
        assert terminal_image.supports_kitty_placeholders() is False


def test_kitty_placeholders_on_kitty_and_ghostty(monkeypatch):
    _clear_graphics_env(monkeypatch)
    monkeypatch.setenv("TERM", "xterm-kitty")
    monkeypatch.setenv("KITTY_WINDOW_ID", "1")
    assert terminal_image.supports_kitty_placeholders() is True

    _clear_graphics_env(monkeypatch)
    monkeypatch.setenv("TERM", "xterm-ghostty")
    monkeypatch.setenv("TERM_PROGRAM", "ghostty")
    assert terminal_image.supports_kitty_placeholders() is True


@pytest.mark.parametrize("multiplexer", ["TMUX", "STY"])
def test_auto_stays_on_half_blocks_inside_a_multiplexer(monkeypatch, multiplexer):
    # tmux and GNU screen inherit KITTY_WINDOW_ID from the outer kitty yet swallow the transmit.
    _clear_graphics_env(monkeypatch)
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.delenv("STY", raising=False)
    monkeypatch.setenv("TERM", "xterm-kitty")
    monkeypatch.setenv("KITTY_WINDOW_ID", "1")
    monkeypatch.setenv(multiplexer, "1")
    assert resolve_inline_mode("auto") == "unicode"
    assert resolve_inline_mode("kitty") == "kitty"


def test_wezterm_is_not_placeholder_capable(monkeypatch):
    # WezTerm speaks kitty APC but not U+10EEEE. Placeholders would tofu.
    _clear_graphics_env(monkeypatch)
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("TERM_PROGRAM", "WezTerm")
    assert terminal_image.detect_terminal_graphics() == "kitty"
    assert terminal_image.supports_kitty_placeholders() is False

    monkeypatch.setenv("WEZTERM_PANE", "1")
    assert terminal_image.supports_kitty_placeholders() is False


def test_auto_mode_uses_kitty_placeholders_only_where_they_paint(monkeypatch):
    _clear_graphics_env(monkeypatch)
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.setenv("TERM", "xterm-ghostty")
    monkeypatch.setenv("TERM_PROGRAM", "ghostty")
    assert resolve_inline_mode("auto") == "kitty"
    # tmux eats the transmit without allow-passthrough: placeholders would paint nothing.
    monkeypatch.setenv("TMUX", "/tmp/tmux-1000/default,1,0")
    assert resolve_inline_mode("auto") == "unicode"
    assert resolve_inline_mode("kitty") == "kitty"

    _clear_graphics_env(monkeypatch)
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.setenv("TERM_PROGRAM", "WezTerm")
    assert resolve_inline_mode("auto") == "unicode"
    monkeypatch.setenv("TERM_PROGRAM", "iTerm.app")
    assert resolve_inline_mode("auto") == "unicode"
    # The classic CLI can reserve rows for a cell-pinned image: iTerm2/WezTerm/Konsole get pixels
    # there, while Ink (no overlays) keeps half-blocks even when ``iterm`` is pinned.
    assert resolve_inline_mode("auto", overlays=True) == "iterm"
    assert resolve_inline_mode("iterm") == "unicode"
    _clear_graphics_env(monkeypatch)
    monkeypatch.setenv("KONSOLE_VERSION", "251203")
    assert resolve_inline_mode("auto", overlays=True) == "iterm"
    monkeypatch.setenv("TMUX", "/tmp/tmux-1000/default,1,0")
    assert resolve_inline_mode("auto", overlays=True) == "unicode"


def test_inline_mode_off_and_no_terminal(monkeypatch):
    _clear_graphics_env(monkeypatch)
    monkeypatch.setenv("TERM", "xterm-kitty")
    assert resolve_inline_mode("off") == "off"
    assert resolve_inline_mode(False) == "off"
    # A painter without a real terminal can still draw half-blocks, never escapes.
    assert resolve_inline_mode("kitty", graphics=False) == "unicode"
    assert resolve_inline_mode("off", graphics=False) == "off"


# ---------------------------------------------------------------------------
# which image a tool call shows
# ---------------------------------------------------------------------------


def test_tool_image_path_reads_named_fields(tmp_path):
    shot = _png(tmp_path / "shot.png")
    assert tool_image_path("browser_exec", {}, json.dumps({"screenshot_path": shot}), is_error=False) == shot
    assert tool_image_path("vision_analyze", {"image_url": shot}, "analysis text, not JSON", is_error=False) == shot
    assert tool_image_path("image_generate", {"prompt": "x"}, json.dumps({"success": True, "image": shot}),
                           is_error=False) == shot
    envelope = {"_multimodal": True, "content": [], "meta": {"screenshot_path": shot}}
    assert tool_image_path("browser_exec", {}, envelope, is_error=False) == shot


def test_tool_image_path_ignores_listings_urls_and_missing_files(tmp_path):
    shot = _png(tmp_path / "shot.png")
    # A file listing names images in free text and nested lists: never a preview.
    assert tool_image_path("search_files", {"pattern": "*.png"}, json.dumps({"matches": [shot], "output": shot}),
                           is_error=False) is None
    for url in ("https://example.com/a.png", str(tmp_path / "gone.png"), str(tmp_path)):
        assert tool_image_path("vision_analyze", {"image_url": url}, "{}", is_error=False) is None
    (tmp_path / "notes.txt").write_text("x")
    assert tool_image_path("vision_analyze", {"image_url": str(tmp_path / "notes.txt")}, "{}", is_error=False) is None


def test_failed_or_writing_calls_never_show_the_image_they_name(tmp_path):
    # A refused read, a write/patch to an image path and a failed vision call all name a real
    # image; drawing it would read as success.
    shot = _png(tmp_path / "shot.png")
    refused = json.dumps({"error": f"Cannot read binary file '{shot}' (.png)."})
    assert tool_image_path("read_file", {"path": shot}, refused, is_error=True) is None
    assert tool_image_path("read_file", {"path": shot}, json.dumps({"content": "", "is_image": True}),
                           is_error=False) is None
    assert tool_image_path("write_file", {"path": shot, "content": "x"}, json.dumps({"bytes_written": 1}),
                           is_error=False) is None
    assert tool_image_path("patch", {"path": shot}, json.dumps({"error": "Refusing to overwrite"}), is_error=True) is None
    assert tool_image_path("vision_analyze", {"image_url": shot}, json.dumps({"success": False, "error": "x"}),
                           is_error=True) is None


# ---------------------------------------------------------------------------
# layout + encoding
# ---------------------------------------------------------------------------


def test_kitty_render_is_a_virtual_placement_sized_to_its_placeholder(tmp_path):
    image = render_inline_image(_png(tmp_path / "wide.png", (1600, 400)), mode="kitty", max_cols=60, max_rows=16)
    assert image is not None
    assert image.cols == 60 and 1 <= image.rows <= 16
    assert len(image.placeholder) == image.rows
    assert all(row.count("\U0010eeee") == image.cols for row in image.placeholder)
    assert image.transmit.startswith("\x1b_G") and image.transmit.endswith("\x1b\\")
    image_id = int(image.color[1:], 16)  # the placeholder color names the image
    assert "U=1" in image.transmit and f"i={image_id}," in image.transmit
    assert f"c={image.cols}" in image.transmit and f"r={image.rows}" in image.transmit
    # Pet ids own [1, 0x7FFF]; an image must never replace the mascot.
    assert 0x7FFF < image_id <= 0xFFFFFF


def test_render_respects_caps_and_never_upscales(tmp_path):
    tall = render_inline_image(_png(tmp_path / "tall.png", (300, 3000)), mode="unicode", max_cols=60, max_rows=10)
    assert tall is not None and tall.rows <= 10 and tall.cols < 60
    tiny = render_inline_image(_png(tmp_path / "tiny.png", (32, 32)), mode="kitty", max_cols=60, max_rows=16)
    assert tiny is not None and tiny.cols == 4  # 32 px / 8 px per cell


def test_unicode_render_returns_half_block_cells(tmp_path):
    image = render_inline_image(_png(tmp_path / "red.png"), mode="unicode", max_cols=20, max_rows=16)
    assert image is not None and not image.transmit
    assert len(image.cells) == image.rows and all(len(row) == image.cols for row in image.cells)
    top, bottom = image.cells[0][0]
    assert top[:3] == (200, 30, 60) and bottom[3] == 255


def test_edited_file_gets_a_new_image_id(tmp_path):
    path = _png(tmp_path / "a.png")
    first = render_inline_image(path, mode="kitty", max_cols=20, max_rows=8)
    _png(tmp_path / "a.png", (400, 201))
    second = render_inline_image(path, mode="kitty", max_cols=20, max_rows=8)
    assert first is not None and second is not None and first.color != second.color


def test_two_sizes_of_one_file_get_their_own_image_ids(tmp_path):
    # kitty keeps one virtual placement per id: a shared id lets the later size repaint the earlier one.
    path = _png(tmp_path / "a.png", (800, 400))
    wide = render_inline_image(path, mode="kitty", max_cols=60, max_rows=16)
    narrow = render_inline_image(path, mode="kitty", max_cols=20, max_rows=16)
    again = render_inline_image(path, mode="kitty", max_cols=60, max_rows=16)
    assert wide is not None and narrow is not None and again is not None
    assert (wide.cols, wide.rows) != (narrow.cols, narrow.rows)
    assert wide.color != narrow.color and wide.color == again.color


def test_unreadable_image_renders_nothing(tmp_path):
    bogus = tmp_path / "bogus.png"
    bogus.write_bytes(b"not a png")
    assert render_inline_image(str(bogus), mode="kitty", max_cols=20, max_rows=8) is None
    assert render_inline_image(_png(tmp_path / "ok.png"), mode="off", max_cols=20, max_rows=8) is None


def test_iterm_size_is_the_decoded_byte_count(tmp_path):
    # xterm.js (VS Code) refuses an OSC 1337 image whose ``size`` is the base64 length.
    import base64
    import re

    for transmit in (
        render_inline_image(_png(tmp_path / "a.png"), mode="iterm", max_cols=20, max_rows=8).transmit,
        terminal_image.encode_iterm(PIL.open(_png(tmp_path / "b.png")).convert("RGBA")),
    ):
        size, payload = re.search(r"size=(\d+);.*:([A-Za-z0-9+/=]+)\x07$", transmit).groups()
        assert int(size) == len(base64.b64decode(payload))


def test_preview_past_the_decode_cap_is_refused_before_decoding(tmp_path, monkeypatch):
    # A huge PNG (no draft scaling) must not fully decode on the agent thread for a 64-column preview.
    big = _png(tmp_path / "big.png", (400, 300))
    monkeypatch.setattr(terminal_image, "_MAX_DECODE_PIXELS", 100_000, raising=False)
    assert render_inline_image(big, mode="unicode", max_cols=20, max_rows=8) is None
    assert render_inline_image(_png(tmp_path / "small.png", (300, 300)), mode="unicode", max_cols=20, max_rows=8)


def test_oversized_png_is_refused_without_entering_load(tmp_path, monkeypatch):
    # A PNG's EXIF lookup decodes the whole file, so the cap must come first.
    from PIL import ImageFile

    big = _png(tmp_path / "big.png", (400, 300))
    monkeypatch.setattr(terminal_image, "_MAX_DECODE_PIXELS", 100_000, raising=False)
    loads = []
    real_load = ImageFile.ImageFile.load
    monkeypatch.setattr(ImageFile.ImageFile, "load", lambda self: loads.append(self.size) or real_load(self))
    assert render_inline_image(big, mode="unicode", max_cols=20, max_rows=8) is None
    assert loads == []


def test_decompression_bomb_warning_is_a_refusal(tmp_path, monkeypatch):
    # Between MAX_IMAGE_PIXELS and twice that Pillow only warns; a preview treats it as a bomb.
    path = _png(tmp_path / "bomb.png", (400, 400))
    monkeypatch.setattr(PIL, "MAX_IMAGE_PIXELS", 100_000)
    assert render_inline_image(path, mode="kitty", max_cols=20, max_rows=8) is None

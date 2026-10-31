"""Terminal graphics: capability detection and the inline-image API (agent/terminal_image.py)."""

from __future__ import annotations

import json

import pytest

from agent import terminal_image
from agent.terminal_image import render_inline_image, resolve_inline_mode, tool_image_path

PIL = pytest.importorskip("PIL.Image")


def _png(path, size=(400, 200), color=(200, 30, 60, 255)):
    PIL.new("RGBA", size, color).save(path)
    return str(path)


# ---------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------


def _clear_graphics_env(monkeypatch):
    for key in ("KITTY_WINDOW_ID", "TERM_PROGRAM", "ITERM_SESSION_ID", "WEZTERM_PANE", "TERM"):
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
    assert tool_image_path({}, json.dumps({"screenshot_path": shot})) == shot
    assert tool_image_path({"path": shot}, json.dumps({"content": "binary"})) == shot
    assert tool_image_path({"image_url": shot}, "analysis text, not JSON") == shot


def test_tool_image_path_ignores_listings_urls_and_missing_files(tmp_path):
    shot = _png(tmp_path / "shot.png")
    # A file listing names images in free text and nested lists: never a preview.
    assert tool_image_path({"pattern": "*.png"}, json.dumps({"matches": [shot], "output": shot})) is None
    assert tool_image_path({"image_url": "https://example.com/a.png"}, "{}") is None
    assert tool_image_path({"path": str(tmp_path / "gone.png")}, "{}") is None
    assert tool_image_path({"path": str(tmp_path)}, "{}") is None
    (tmp_path / "notes.txt").write_text("x")
    assert tool_image_path({"path": str(tmp_path / "notes.txt")}, "{}") is None


# ---------------------------------------------------------------------------
# layout + encoding
# ---------------------------------------------------------------------------


def test_kitty_render_is_a_virtual_placement_sized_to_its_placeholder(tmp_path):
    image = render_inline_image(_png(tmp_path / "wide.png", (1600, 400)), mode="kitty", max_cols=60, max_rows=16)
    assert image is not None
    assert image.cols == 60 and 1 <= image.rows <= 16
    assert len(image.placeholder) == image.rows
    assert all(row.count(terminal_image.KITTY_PLACEHOLDER) == image.cols for row in image.placeholder)
    assert image.transmit.startswith("\x1b_G") and image.transmit.endswith("\x1b\\")
    assert "U=1" in image.transmit and f"i={image.image_id}" in image.transmit
    assert f"c={image.cols}" in image.transmit and f"r={image.rows}" in image.transmit
    assert image.color == terminal_image.kitty_color_hex(image.image_id)
    # Pet ids own [1, 0x7FFF]; an image must never replace the mascot.
    assert 0x7FFF < image.image_id <= 0xFFFFFF


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
    assert first is not None and second is not None and first.image_id != second.image_id


def test_unreadable_image_renders_nothing(tmp_path):
    bogus = tmp_path / "bogus.png"
    bogus.write_bytes(b"not a png")
    assert render_inline_image(str(bogus), mode="kitty", max_cols=20, max_rows=8) is None
    assert render_inline_image(_png(tmp_path / "ok.png"), mode="off", max_cols=20, max_rows=8) is None

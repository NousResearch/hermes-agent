"""Terminal graphics capability detection (agent/terminal_image.py)."""

from __future__ import annotations

from agent import terminal_image


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

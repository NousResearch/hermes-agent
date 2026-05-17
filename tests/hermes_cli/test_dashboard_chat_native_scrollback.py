"""Regression tests for Dashboard Chat native text selection scrollback."""
from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CHAT_PAGE = REPO_ROOT / "web" / "src" / "pages" / "ChatPage.tsx"
APP_LAYOUT = REPO_ROOT / "ui-tui" / "src" / "components" / "appLayout.tsx"


def test_dashboard_chat_pty_uses_inline_tui_for_native_scrollback(monkeypatch):
    """Dashboard embeds xterm; native selection needs the TUI in primary-buffer mode."""
    from hermes_cli import web_server

    monkeypatch.setattr(
        "hermes_cli.main._make_tui_argv",
        lambda _root, tui_dev=False: (["node", "entry.js"], "/tmp/hermes-ui-tui"),
    )

    _argv, _cwd, env = web_server._resolve_chat_argv()

    assert env["HERMES_TUI_DISABLE_MOUSE"] == "1"
    assert env["HERMES_TUI_INLINE"] == "1"


def test_chat_page_keeps_xterm_native_scrollback_and_wheel():
    """The browser xterm must own scrollback for drag-to-edge selection."""
    source = CHAT_PAGE.read_text(encoding="utf-8")

    assert "const DASHBOARD_CHAT_SCROLLBACK" in source
    assert "scrollback: DASHBOARD_CHAT_SCROLLBACK" in source
    assert "attachCustomWheelEventHandler" not in source


def test_inline_chat_layout_does_not_virtualize_primary_buffer_transcript():
    """Inline mode must not render ScrollBox virtual spacers into xterm scrollback."""
    source = APP_LAYOUT.read_text(encoding="utf-8")

    assert "const InlineTranscriptPane" in source
    assert "const TranscriptShell = INLINE_MODE ? InlineTranscriptPane : TranscriptPane" in source

    inline_start = source.index("const InlineTranscriptPane")
    inline_end = source.index("const TranscriptPane")
    inline_source = source[inline_start:inline_end]

    assert "<ScrollBox" not in inline_source
    assert "virtualHistory.topSpacer" not in inline_source
    assert "virtualHistory.bottomSpacer" not in inline_source
    assert "transcript.historyItems.map" in inline_source

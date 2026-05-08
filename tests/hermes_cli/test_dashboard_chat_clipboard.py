"""Dashboard Chat clipboard regression tests.

The web Chat page embeds xterm.js. Keyboard shortcuts can be handled via
xterm's custom key handler, but browser/system-menu copy and context-menu copy
fire the standard DOM ``copy`` event instead. A selected terminal range must be
written synchronously to ``ClipboardEvent.clipboardData`` in that event so the
browser's native copy path works without relying on async Clipboard API user
activation.
"""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CHAT_PAGE = REPO_ROOT / "web" / "src" / "pages" / "ChatPage.tsx"


def _chat_page_source() -> str:
    return CHAT_PAGE.read_text(encoding="utf-8")


def test_chat_page_handles_native_copy_event_for_xterm_selection() -> None:
    """Right-click/system-menu copy must use the DOM copy event path."""
    src = _chat_page_source()

    assert "const handleCopyEvent = (ev: ClipboardEvent) =>" in src
    assert 'host.addEventListener("copy", handleCopyEvent)' in src
    assert "const sel = term.getSelection();" in src
    assert 'ev.clipboardData?.setData("text/plain", sel)' in src
    assert "ev.preventDefault();" in src
    assert "term.clearSelection();" in src


def test_chat_page_removes_native_copy_event_listener_on_unmount() -> None:
    """The copy event listener must not leak after ChatPage unmounts/remounts."""
    src = _chat_page_source()

    assert 'host.removeEventListener("copy", handleCopyEvent)' in src

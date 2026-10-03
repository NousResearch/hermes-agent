"""Guard the dashboard chat scroll contract.

The embedded dashboard chat is a PTY-backed view of ``hermes --tui``. The
browser host must not invent a second transcript history, or the TUI loses its
sticky-tail and "pause auto-follow while reviewing history" behavior.

This regression test intentionally inspects ``web/src/pages/ChatPage.tsx``
directly because the repo currently has no lightweight frontend interaction
harness for the embedded xterm bridge. The contract we care about is narrow:

1. browser-side xterm scrollback stays disabled
2. wheel gestures are routed back into the TUI's existing Shift+Up/Down path
3. the browser host does not call ``term.scrollLines(...)`` locally
"""

from __future__ import annotations

import re
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHAT_PAGE = _REPO_ROOT / "web" / "src" / "pages" / "ChatPage.tsx"


def _chat_page_source() -> str:
    return _CHAT_PAGE.read_text(encoding="utf-8")


def _wheel_handler_block(source: str) -> str:
    match = re.search(
        r"term\.attachCustomWheelEventHandler\(\(ev\) => \{(?P<body>.*?)\n    \}\);",
        source,
        re.DOTALL,
    )
    assert match, "ChatPage must define a custom wheel handler for the embedded TUI bridge"
    return match.group("body")


def test_dashboard_chat_disables_browser_scrollback() -> None:
    source = _chat_page_source()
    assert "scrollback: 0," in source, "dashboard xterm must not keep its own scrollback buffer"


def test_dashboard_chat_routes_wheel_to_tui_scroll_keys() -> None:
    body = _wheel_handler_block(_chat_page_source())

    assert '"\\x1b[1;2A"' in body, "wheel-up must map to the TUI's Shift+Up scroll input"
    assert '"\\x1b[1;2B"' in body, "wheel-down must map to the TUI's Shift+Down scroll input"
    assert "ws.send(seq);" in body, "wheel handler must forward scroll input through the PTY websocket"
    assert "ev.preventDefault();" in body, "wheel handler must stop browser-native scrolling"
    assert "ev.stopPropagation();" in body, "wheel handler must not leak wheel events to outer panes"


def test_dashboard_chat_does_not_scroll_browser_xterm_locally() -> None:
    body = _wheel_handler_block(_chat_page_source())
    assert "scrollLines" not in body, "wheel handler must not bypass the inner TUI with local xterm scrollback"

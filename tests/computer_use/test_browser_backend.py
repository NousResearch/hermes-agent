"""Browser backend click/type/scroll/tabs."""

from __future__ import annotations

import pytest

from tools.computer_use.backend import UIElement
from tools.computer_use.browser_backend import BrowserBackend


class _FakePage:
    def __init__(self, title="fixture", url="https://example.com"):
        self.calls = []
        self.title_value = title
        self.url = url
        self.mouse = self

    def evaluate(self, script, arg=None):
        self.calls.append((script, arg))
        if arg is None:
            return [{"index": 1, "role": "textbox", "label": "From", "enabled": True},
                    {"index": 2, "role": "button", "label": "Search", "enabled": True}]
        idx = arg[0] if isinstance(arg, list) else arg
        return True if idx in (1, 2) else None

    def title(self):
        return self.title_value

    def wait_for_timeout(self, _ms):
        return None

    def wait_for_load_state(self, *_a, **_k):
        return None

    def wheel(self, dx, dy):
        self.calls.append(("wheel", (dx, dy)))

    def goto(self, url, wait_until=None):
        self.url = url
        self.calls.append(("goto", url))


def test_type_text_fills_last_clicked_element():
    backend = BrowserBackend("https://example.com")
    backend._page = _FakePage()
    backend._pages = [backend._page]
    backend._elements = [
        UIElement(index=1, role="textbox", label="From"),
        UIElement(index=2, role="button", label="Search"),
    ]
    clicked = backend.click(element=1)
    assert clicked.ok
    typed = backend.type_text("Zurich")
    assert typed.ok
    assert "typed 6 chars" in typed.message
    script, arg = backend._page.calls[-2]
    assert arg == [1, "Zurich"]
    assert "el.value = extra" in script


def test_type_text_without_target_fails():
    backend = BrowserBackend("https://example.com")
    backend._page = _FakePage()
    result = backend.type_text("nope")
    assert result.ok is False
    assert "focused element" in result.message


def test_new_backend_defaults_to_about_blank(monkeypatch):
    from tools.computer_use import tool as cu_tool

    monkeypatch.setenv("HERMES_COMPUTER_USE_BACKEND", "browser")
    monkeypatch.delenv("HERMES_CU_BROWSER_URL", raising=False)
    monkeypatch.delenv("HERMES_CU_BROWSER_HTML", raising=False)
    backend = cu_tool._new_backend("standard")
    assert backend._target == "about:blank"


def test_scroll_and_tabs():
    backend = BrowserBackend("https://example.com")
    p1 = _FakePage(title="one", url="https://a.example")
    p2 = _FakePage(title="two", url="https://b.example")
    backend._page = p1
    backend._pages = [p1, p2]
    scrolled = backend.scroll(direction="down", amount=2)
    assert scrolled.ok
    assert ("wheel", (0, 240)) in p1.calls
    wins = backend.list_windows()
    assert [w["window_id"] for w in wins] == [1, 2]
    backend.capture(window_id=2)
    assert backend._page is p2
    focused = backend.focus_app("tab:1")
    assert focused.ok
    assert backend._page is p1


def test_navigate_replaces_current_tab():
    backend = BrowserBackend("https://example.com")
    page = _FakePage(title="one", url="https://a.example")
    backend._page = page
    backend._pages = [page]
    res = backend.navigate("https://b.example")
    assert res.ok
    assert len(backend._pages) == 1
    assert page.url == "https://b.example"


class _FakeBrowser:
    def new_page(self, viewport=None):
        return _FakePage(title="new", url="about:blank")


def test_navigate_new_tab_appends():
    backend = BrowserBackend("https://example.com")
    page = _FakePage(title="one", url="https://a.example")
    backend._page = page
    backend._pages = [page]
    backend._browser = _FakeBrowser()
    res = backend.navigate("https://b.example", new_tab=True)
    assert res.ok
    assert len(backend._pages) == 2
    assert backend._page is not page
    assert backend._page.url == "https://b.example"

def test_playwright_type_into_input(tmp_path):
    pytest.importorskip("playwright.sync_api")
    html = tmp_path / "form.html"
    html.write_text(
        '<input aria-label="From"><button>Search</button>',
        encoding="utf-8",
    )
    backend = BrowserBackend(html)
    backend.start()
    try:
        cap = backend.capture()
        assert any(el.label == "From" for el in cap.elements)
        assert backend.click(element=1).ok
        assert backend.type_text("Zurich").ok
        value = backend._page.evaluate("() => document.querySelector('input').value")
        assert value == "Zurich"
    finally:
        backend.stop()

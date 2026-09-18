"""Browser backend click/type without launching Chromium."""

from __future__ import annotations

import pytest

from tools.computer_use.backend import UIElement
from tools.computer_use.browser_backend import BrowserBackend


class _FakePage:
    def __init__(self):
        self.calls = []
        self.title_value = "fixture"

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


def test_type_text_fills_last_clicked_element():
    backend = BrowserBackend("https://example.com")
    backend._page = _FakePage()
    backend._elements = [
        UIElement(index=1, role="textbox", label="From"),
        UIElement(index=2, role="button", label="Search"),
    ]
    clicked = backend.click(element=1)
    assert clicked.ok
    typed = backend.type_text("Zurich")
    assert typed.ok
    assert "typed 6 chars" in typed.message
    script, arg = backend._page.calls[-2]  # type evaluate before refresh
    assert arg == [1, "Zurich"]
    assert "el.value = extra" in script


def test_type_text_without_target_fails():
    backend = BrowserBackend("https://example.com")
    backend._page = _FakePage()
    result = backend.type_text("nope")
    assert result.ok is False
    assert "focused element" in result.message


def test_new_backend_requires_browser_url(monkeypatch):
    from tools.computer_use import tool as cu_tool

    monkeypatch.setenv("HERMES_COMPUTER_USE_BACKEND", "browser")
    monkeypatch.delenv("HERMES_CU_BROWSER_URL", raising=False)
    monkeypatch.delenv("HERMES_CU_BROWSER_HTML", raising=False)
    try:
        cu_tool._new_backend("standard")
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "HERMES_CU_BROWSER_URL" in str(exc)


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

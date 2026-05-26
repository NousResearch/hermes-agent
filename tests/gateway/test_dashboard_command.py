"""Unit tests for the /dashboard slash command helper.

Exercises ``gateway.run._build_dashboard_keyboard`` — the pure helper that
constructs the Telegram InlineKeyboardMarkup. Avoids spinning up a real
Telegram client.

Works against both the real python-telegram-bot package and the
MagicMock stand-in installed by tests/gateway/conftest.py: we inject
lightweight local fakes for ``InlineKeyboardButton`` /
``InlineKeyboardMarkup`` into ``sys.modules['telegram']`` for the
duration of each test so attribute round-tripping is observable.
"""
import sys
import types

import pytest

pytest.importorskip("telegram")

from gateway.run import _build_dashboard_keyboard  # noqa: E402


class _FakeButton:
    def __init__(self, *, text=None, url=None, **_):
        self.text = text
        self.url = url


class _FakeMarkup:
    def __init__(self, rows):
        self.inline_keyboard = tuple(tuple(r) for r in rows)


@pytest.fixture
def fake_telegram(monkeypatch):
    """Swap telegram.InlineKeyboardButton/Markup for local fakes.

    Restored automatically via monkeypatch teardown — no cross-test
    pollution of ``sys.modules``.
    """
    tg = sys.modules["telegram"]
    monkeypatch.setattr(tg, "InlineKeyboardButton", _FakeButton, raising=False)
    monkeypatch.setattr(tg, "InlineKeyboardMarkup", _FakeMarkup, raising=False)
    yield


def _flatten(kb):
    rows = getattr(kb, "inline_keyboard", None)
    assert rows is not None, "keyboard missing inline_keyboard attribute"
    return [btn for row in rows for btn in row]


def test_build_dashboard_keyboard_empty_returns_none():
    # No telegram import needed — empty short-circuits.
    assert _build_dashboard_keyboard([]) is None
    assert _build_dashboard_keyboard(None) is None


def test_build_dashboard_keyboard_with_hosts(fake_telegram):
    hosts = [
        {"name": "emma", "url": "https://x"},
        {"name": "i7", "url": "https://y"},
    ]
    kb = _build_dashboard_keyboard(hosts)
    assert isinstance(kb, _FakeMarkup)
    buttons = _flatten(kb)
    assert [b.text for b in buttons] == ["emma", "i7"]
    assert [b.url for b in buttons] == ["https://x", "https://y"]


def test_build_dashboard_keyboard_skips_invalid_entries(fake_telegram):
    hosts = [
        {"name": "ok", "url": "https://ok"},
        {"name": "no-url"},  # missing url -> skipped
        "not-a-dict",         # wrong type -> skipped
    ]
    kb = _build_dashboard_keyboard(hosts)
    assert isinstance(kb, _FakeMarkup)
    buttons = _flatten(kb)
    assert len(buttons) == 1
    assert buttons[0].text == "ok"
    assert buttons[0].url == "https://ok"

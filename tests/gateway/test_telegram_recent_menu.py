"""Regression tests: 🕘 recent-models row in the Telegram /model picker (opt-in flat menu)."""
from __future__ import annotations

import asyncio

import pytest

import plugins.platforms.telegram.adapter as telegram_mod  # noqa: E402
from hermes_cli.telegram_recent_models import record_recent  # noqa: E402
from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402

PROVIDERS = [
    {"slug": "opencode-zen", "name": "OpenCode Zen", "models": ["zen-a"], "total_models": 1},
    {"slug": "opencode-go", "name": "OpenCode Go", "models": ["go-a"], "total_models": 1},
    {"slug": "openrouter", "name": "OpenRouter",
     "models": ["google/gemini-4-pro"], "total_models": 1},
]


def _bare() -> TelegramAdapter:
    """Adapter instance without running __init__ (pure-helper level tests)."""
    return object.__new__(TelegramAdapter)


class _Btn:
    """Faithful stand-in for telegram.InlineKeyboardButton (see tests/gateway/conftest.py)."""

    def __init__(self, text, callback_data=None, **_):
        self.text = text
        self.callback_data = callback_data


class _Kb:
    """Faithful stand-in for telegram.InlineKeyboardMarkup."""

    def __init__(self, rows):
        self.inline_keyboard = rows


@pytest.fixture
def keys(monkeypatch):
    """Bind real (assertable) keyboard objects — the telegram SDK is mocked in gateway tests."""
    monkeypatch.setattr(telegram_mod, "InlineKeyboardButton", _Btn)
    monkeypatch.setattr(telegram_mod, "InlineKeyboardMarkup", _Kb)


def _labels(keyboard):
    return [b.text for row in keyboard.inline_keyboard for b in row]


def _callbacks(keyboard):
    return [b.callback_data for row in keyboard.inline_keyboard for b in row]


class _FakeQuery:
    """Callback-query stand-in capturing the re-rendered message."""

    def __init__(self):
        self.text = None
        self.keyboard = None
        self.answers: list[str] = []

    async def edit_message_text(self, text=None, parse_mode=None, reply_markup=None):
        self.text = text
        self.keyboard = reply_markup

    async def answer(self, text=None, **_):
        self.answers.append(text)


def _adapter_for_render(keys) -> TelegramAdapter:
    ad = _bare()
    ad.format_message = lambda text: text  # type: ignore[assignment]
    return ad


def test_flat_menu_offers_recent_button(keys):
    kb, _ = _bare()._build_provider_keyboard(PROVIDERS, 0, flat=True)
    assert any("🕘" in lbl for lbl in _labels(kb)), _labels(kb)
    assert "mr" in _callbacks(kb), _callbacks(kb)


def test_grouped_menu_has_no_recent_button_by_default(keys):
    kb, _ = _bare()._build_provider_keyboard(PROVIDERS, 0, flat=False)
    assert "mr" not in _callbacks(kb), _callbacks(kb)


def test_recent_entries_follow_history_and_carry_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    record_recent("openrouter", "google/gemini-4-pro")
    record_recent("opencode-zen", "zen-a")

    entries = _bare()._recent_entries()
    assert entries == [("opencode-zen", "zen-a"), ("openrouter", "google/gemini-4-pro")]


def test_recent_entries_empty_without_history(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert _bare()._recent_entries() == []


def test_recent_page_labels_show_model_and_short_provider(tmp_path, monkeypatch, keys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    record_recent("opencode-zen", "zen-a")
    record_recent("openrouter", "google/gemini-4-pro")

    ad = _adapter_for_render(keys)
    state = {"providers": PROVIDERS}
    query = _FakeQuery()
    asyncio.run(ad._picker_show_recent(query, state, 0))

    labels = _labels(query.keyboard)
    assert any("gemini-4-pro" in lbl for lbl in labels), labels
    assert any("OpenRouter" in lbl for lbl in labels), labels
    # "OpenCode Zen" shortens to "Zen".
    assert any(lbl.endswith("· Zen") for lbl in labels), labels
    assert "mrsel:0" in _callbacks(query.keyboard), _callbacks(query.keyboard)
    assert "◀ Back" in labels and "✗ Cancel" in labels, labels
    assert "🕘" in (query.text or ""), query.text
    assert state["recent_page"] == 0


def test_recent_labels_fall_back_to_the_raw_slug(tmp_path, monkeypatch, keys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    record_recent("gizemli-saglayici", "gizli-model")

    ad = _adapter_for_render(keys)
    state = {"providers": PROVIDERS, "recent_entries": [("gizemli-saglayici", "gizli-model")]}
    query = _FakeQuery()
    asyncio.run(ad._picker_show_recent(query, state, 0))

    labels = _labels(query.keyboard)
    assert any("gizemli-saglayici" in lbl for lbl in labels), labels


def test_recent_button_without_history_answers_instead_of_rendering(tmp_path, monkeypatch, keys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    ad = _adapter_for_render(keys)
    ad._model_picker_state = {"4242": {"providers": PROVIDERS, "flat": True}}
    query = _FakeQuery()
    asyncio.run(ad._handle_model_picker_callback(query, "mr", "4242"))

    assert query.keyboard is None, "no keyboard should be rendered without history"
    assert query.answers and "model seçilmedi" in query.answers[0], query.answers

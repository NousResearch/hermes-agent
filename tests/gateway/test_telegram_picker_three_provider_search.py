"""Regression tests: config-gated Telegram /model menu (3 providers + Search).

Feature: when ``model_catalog.telegram_three_provider_menu`` is true, the
Telegram /model picker must render the three providers as separate buttons
(no OpenCode group fold) and offer a working Search button that consumes the
next free-text message in the same chat as a model query instead of sending
it to the agent.
"""
from __future__ import annotations

import asyncio

import pytest

import plugins.platforms.telegram.adapter as telegram_mod  # noqa: E402
from gateway.platforms.base import BasePlatformAdapter  # noqa: E402
from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402

PROVIDERS = [
    {
        "slug": "opencode-zen",
        "name": "OpenCode Zen",
        "models": ["zen-a", "zen-b", "google/gemini-4-flash"],
        "total_models": 3,
        "is_current": True,
    },
    {
        "slug": "opencode-go",
        "name": "OpenCode Go",
        "models": ["go-gemini-fast", "go-x"],
        "total_models": 2,
    },
    {
        "slug": "openrouter",
        "name": "OpenRouter",
        "models": ["google/gemini-4-pro", "openai/gpt-6"],
        "total_models": 2,
    },
]


def _bare() -> TelegramAdapter:
    """Adapter instance without running __init__ (pure-helper level tests)."""
    return object.__new__(TelegramAdapter)


class _Btn:
    """Faithful stand-in for telegram.InlineKeyboardButton (the SDK is mocked in gateway tests)."""

    def __init__(self, text, callback_data=None, **_):
        self.text = text
        self.callback_data = callback_data


class _Kb:
    """Faithful stand-in for telegram.InlineKeyboardMarkup."""

    def __init__(self, rows):
        self.inline_keyboard = rows


@pytest.fixture
def keys(monkeypatch):
    """Bind real (assertable) keyboard objects."""
    monkeypatch.setattr(telegram_mod, "InlineKeyboardButton", _Btn)
    monkeypatch.setattr(telegram_mod, "InlineKeyboardMarkup", _Kb)


def _labels(keyboard) -> list[str]:
    return [b.text for row in keyboard.inline_keyboard for b in row]


def _callbacks(keyboard) -> list[str]:
    return [b.callback_data for row in keyboard.inline_keyboard for b in row]


def test_flat_provider_keyboard_keeps_three_separate_buttons_and_search(keys):
    kb, _info = _bare()._build_provider_keyboard(PROVIDERS, 0, flat=True)
    labels = _labels(kb)

    assert any("OpenCode Zen" in lbl for lbl in labels), labels
    assert any("OpenCode Go" in lbl for lbl in labels), labels
    assert any(lbl.startswith("OpenRouter") for lbl in labels), labels

    # Zen and Go must NOT be folded into a single "OpenCode" group button.
    assert not any("▸" in lbl for lbl in labels), labels
    assert "ms" in _callbacks(kb), _callbacks(kb)


def test_grouped_provider_keyboard_is_unchanged_by_default(keys):
    kb, _info = _bare()._build_provider_keyboard(PROVIDERS, 0, flat=False)
    labels = _labels(kb)

    assert "ms" not in _callbacks(kb), _callbacks(kb)
    # Default behaviour still folds the OpenCode family.
    assert any("OpenCode" in lbl for lbl in labels), labels


def test_search_button_marks_and_consumes_pending_query_once():
    ad = _bare()
    ad._model_picker_state = {"42": {"providers": PROVIDERS}}

    assert ad._take_pending_model_search("42") is None
    ad._mark_pending_model_search("42")
    assert ad._take_pending_model_search("42") == {"providers": PROVIDERS}
    # Second read must not consume again (query already handled).
    assert ad._take_pending_model_search("42") is None


def test_search_models_matches_across_providers_case_insensitively():
    hits = _bare()._search_models(PROVIDERS, "GEMINI")

    assert ("opencode-zen", "google/gemini-4-flash") in hits
    assert ("opencode-go", "go-gemini-fast") in hits
    assert ("openrouter", "google/gemini-4-pro") in hits
    assert all("gemini" in mid.lower() for _slug, mid in hits)
    assert len(hits) == 3


def test_search_models_empty_query_and_no_match():
    ad = _bare()
    assert ad._search_models(PROVIDERS, "   ") == []
    assert ad._search_models(PROVIDERS, "zzz-nope") == []


def test_flag_reader_defaults_off(monkeypatch, tmp_path):
    ad = _bare()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("model_catalog:\n  excluded_providers: []\n")
    assert ad._three_provider_menu_enabled() is False

    (tmp_path / "config.yaml").write_text(
        "model_catalog:\n  telegram_three_provider_menu: true\n"
    )
    assert telegram_mod.TelegramAdapter._three_provider_menu_enabled(ad) is True


class _FakeSource:
    chat_id = "4242"


class _FakeEvent:
    def __init__(self, text: str):
        self.text = text
        self.source = _FakeSource()
        self.metadata = None


def _search_consuming_adapter(monkeypatch, captured: list, base_calls: list) -> TelegramAdapter:
    ad = _bare()
    ad._model_picker_state = {"4242": {"providers": PROVIDERS, "flat": True}}
    ad._should_drop_delayed_delivery = lambda: False  # type: ignore[assignment]
    ad._three_provider_menu_enabled = lambda: True  # type: ignore[assignment]

    async def _run(event, state, text):
        captured.append((state, text))

    ad._run_model_search = _run  # type: ignore[assignment]
    monkeypatch.setattr(
        BasePlatformAdapter, "_enqueue_text_event", lambda self, event: base_calls.append(event))
    return ad


@pytest.mark.asyncio
async def test_pending_search_consumes_the_next_free_text_message(monkeypatch):
    captured: list = []
    base_calls: list = []
    ad = _search_consuming_adapter(monkeypatch, captured, base_calls)

    ad._mark_pending_model_search("4242")
    ad._enqueue_text_event(_FakeEvent("gemini"))
    await asyncio.sleep(0)  # the consume path schedules the search task

    # Drained synchronously: the query must never reach the agent.
    assert len(captured) == 1, captured
    state, text = captured[0]
    assert text == "gemini", captured
    assert state["providers"] == PROVIDERS and state["flat"] is True, state
    assert "awaiting_search" not in state, state  # flag consumed, not left armed
    assert base_calls == [], "the query message was also dispatched to the agent"
    # Flag is one-shot.
    assert ad._take_pending_model_search("4242") is None


@pytest.mark.asyncio
async def test_plain_text_goes_to_the_agent_when_no_search_is_pending(monkeypatch):
    captured: list = []
    base_calls: list = []
    ad = _search_consuming_adapter(monkeypatch, captured, base_calls)

    ad._enqueue_text_event(_FakeEvent("normal chat message"))
    await asyncio.sleep(0)

    assert captured == [], "no search was pending — the message must not be treated as a query"
    assert len(base_calls) == 1, base_calls


@pytest.mark.asyncio
async def test_slash_commands_bypass_a_pending_search(monkeypatch):
    captured: list = []
    base_calls: list = []
    ad = _search_consuming_adapter(monkeypatch, captured, base_calls)

    ad._mark_pending_model_search("4242")
    ad._enqueue_text_event(_FakeEvent("/status"))
    await asyncio.sleep(0)

    assert captured == [], "slash commands must pass through"
    assert len(base_calls) == 1, base_calls
    # The flag survives so the next real query is still consumed.
    assert ad._take_pending_model_search("4242") is not None
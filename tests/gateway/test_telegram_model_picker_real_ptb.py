"""``/model`` picker keyboards built with the REAL python-telegram-bot classes.

``tests/gateway/conftest.py`` installs a MagicMock ``telegram`` package so the
adapter imports without the library. A MagicMock accepts every attribute access,
so ``InlineKeyboardMarkup(rows).inline_keyboard`` yields another mock and any
assertion about the rendered keyboard passes whether or not it is correct — the
mask that let the #94986 symptoms survive a green suite.

This file re-imports the real package (skipping when it is genuinely absent) and
builds the picker keyboards through it, so the PTB constructors that would reject
what the adapter emits actually run.
"""

import importlib
import sys

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.telegram.adapter import TelegramAdapter


@pytest.fixture
def real_ptb(monkeypatch):
    """Rebind the adapter's button classes to the real PTB ones for this test."""
    for name in [n for n in list(sys.modules) if n == "telegram" or n.startswith("telegram.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    try:
        telegram = importlib.import_module("telegram")
    except ImportError:  # pragma: no cover - env without the optional dependency
        pytest.skip("python-telegram-bot not installed")
    if not hasattr(telegram, "__file__"):  # pragma: no cover - defensive
        pytest.skip("telegram is mocked in this environment")

    from plugins.platforms.telegram import adapter as telegram_adapter

    monkeypatch.setattr(telegram_adapter, "InlineKeyboardButton", telegram.InlineKeyboardButton)
    monkeypatch.setattr(telegram_adapter, "InlineKeyboardMarkup", telegram.InlineKeyboardMarkup)
    return telegram_adapter


# Trimmed from a real ``provider_model_ids("bedrock")`` listing: the same model
# advertised bare, regionally and globally, across several vendors.
#
# Amazon carries enough entries to span THREE pages, so the walk below actually
# reaches a middle page — the only place ``◀ Prev | n/N | Next ▶`` renders all
# three navigation buttons. A fixture whose every vendor fits one page let the
# row-width rule below look satisfied while a real 17-model vendor rendered a
# row it forbids (#94990 review).
BEDROCK_IDS = [
    "global.anthropic.claude-opus-5",
    "us.anthropic.claude-opus-5",
    "anthropic.claude-opus-5",
    "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "amazon.nova-lite-v1:0",
    "us.amazon.nova-lite-v1:0",
    *[f"amazon.nova-{name}-v1:0" for name in (
        "micro", "pro", "premier", "canvas", "reel", "sonic",
        "embed-text", "embed-multimodal", "rerank", "act", "sonic-2")],
    *[f"us.amazon.nova-{name}-v1:0" for name in ("micro", "pro", "premier", "sonic")],
    "openai.gpt-5.6-terra",
    "us.openai.gpt-5.6-terra",
    "moonshot.kimi-k2-thinking",
    "moonshotai.kimi-k2.5",
]


def test_every_picker_keyboard_is_a_valid_ptb_keyboard(real_ptb):
    """Real PTB constructors run over the vendor step and every model page.

    Each button must carry non-empty text (Telegram answers
    ``BUTTON_TEXT_INVALID`` and drops the whole message otherwise) and a
    ``callback_data`` within the 64-byte wire limit. Model rows keep the two
    columns the layout promises; the navigation row is the deliberate exception
    (``◀ Prev | n/N | Next ▶`` is three buttons on any middle page).
    """
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))

    from plugins.platforms.telegram.model_picker_callbacks import parse_selection
    from plugins.platforms.telegram.model_picker_display import group_models_by_vendor

    keyboards = [adapter._build_vendor_keyboard(BEDROCK_IDS)]
    # Each vendor's pages must, together, offer a tap for every model it scopes:
    # a model missing from every page is unreachable in the UI even though its ID
    # is in the catalog. The payload names a canonical position in the FULL list,
    # so reachability is measured there — that is the list a tap resolves against.
    for group in group_models_by_vendor(BEDROCK_IDS):
        scoped = [BEDROCK_IDS[i] for i in group["indices"]]
        reachable: set[int] = set()
        page = 0
        while True:
            keyboard, _info = adapter._build_model_keyboard(
                scoped, page, listing=1, canonical=list(group["indices"]))
            keyboards.append(keyboard)
            buttons = [b for row in keyboard.inline_keyboard for b in row]
            for b in buttons:
                if str(b.callback_data).startswith("mm:"):
                    parsed = parse_selection(str(b.callback_data)[3:])
                    assert parsed is not None, b.callback_data
                    listing, index = parsed
                    assert listing == 1, b.callback_data
                    reachable.add(index)
            if not any(str(b.callback_data) == f"mg:{page + 1}" for b in buttons):
                break
            page += 1
        assert reachable == set(group["indices"]), f"{group['label']}: {sorted(reachable)}"

    for keyboard in keyboards:
        assert keyboard.inline_keyboard, "an empty keyboard is rejected by Telegram"
        for row in keyboard.inline_keyboard:
            assert 1 <= len(row) <= _max_row_width(row), f"row wider than the layout allows: {row}"
            for button in row:
                assert button.text.strip(), f"blank button text in {keyboard.inline_keyboard}"
                assert len(str(button.callback_data).encode()) <= 64, button.callback_data


def _is_nav_row(row) -> bool:
    """True for a ``◀ Prev | n/N | Next ▶`` pagination row.

    Its middle button is the inert page counter (``mx:noop``), which no other row
    carries — so the layout rule can stay strict for model rows while a middle
    page legitimately renders three buttons.
    """
    return any(str(b.callback_data) == "mx:noop" for b in row)


def _max_row_width(row) -> int:
    return 3 if _is_nav_row(row) else 2


def test_a_middle_page_renders_prev_counter_and_next(real_ptb):
    """A vendor big enough for three pages must show all three nav buttons.

    ``_picker_nav_row`` emits Prev only past page 0 and Next only before the last
    page, so the three-button form exists exclusively on a middle page. A vendor
    of one page — or two — never reaches it, which is how a two-column assertion
    could hold over the trimmed fixture while a real 17-model vendor renders a
    row the rule forbids (#94990 review).
    """
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    # 3 pages at 8 models/page: enough for page 1 to have a page on each side.
    models = [f"amazon.nova-variant-{i}-v1:0" for i in range(17)]

    keyboard, _info = adapter._build_model_keyboard(models, 1, listing=1)
    rows = keyboard.inline_keyboard
    nav = [row for row in rows if _is_nav_row(row)]
    assert len(nav) == 1, f"expected exactly one navigation row, got {len(nav)}"
    assert [str(b.callback_data) for b in nav[0]] == ["mg:0", "mx:noop", "mg:2"]

    model_rows = [row for row in rows
                  if any(str(b.callback_data).startswith("mm:") for b in row)]
    assert model_rows, "a middle page must still render models"
    for row in model_rows:
        assert len(row) <= 2, f"model row wider than two columns: {[b.text for b in row]}"


@pytest.mark.asyncio
@pytest.mark.parametrize("slug", ["bedrock", "openai-codex"])
async def test_provider_scope_and_other_ids_survive_real_router(real_ptb, monkeypatch, slug):
    """Real PTB keyboards, real callback dispatch; only transport/switch are mocked."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from hermes_cli import model_selection_guards

    monkeypatch.setattr(model_selection_guards, "combined_selection_warning", lambda *a, **k: None)
    if slug != "bedrock":
        def unexpected_region_lookup():
            pytest.fail("Non-Bedrock picker looked up an AWS region")
        monkeypatch.setattr(real_ptb, "configured_region_geo", unexpected_region_lookup)
    models = ["openai.example", "global.openai.example", "meta.example"]
    unknown = [f"unlisted-vendor.example-{i}" for i in range(9)]
    models += unknown
    callback = AsyncMock(return_value="switched")
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._model_picker_state["12345"] = {
        "providers": [{"slug": slug, "name": slug, "models": models}],
        "current_model": models[0], "current_provider": slug,
        "on_model_selected": callback,
    }

    async def tap(data):
        query = SimpleNamespace(
            data=data, message=SimpleNamespace(chat_id=12345), from_user=None,
            answer=AsyncMock(), edit_message_text=AsyncMock())
        await adapter._handle_callback_query(SimpleNamespace(callback_query=query), None)
        return query

    query = await tap(f"mp:{slug}")
    payload = query.edit_message_text.call_args.kwargs
    if slug == "bedrock":
        vendors = [b.callback_data for row in payload["reply_markup"].inline_keyboard for b in row]
        assert "mvd:other" in vendors
        await tap("mvd:other")
        query = await tap("mg:1")
        payload = query.edit_message_text.call_args.kwargs
        chosen = next(b for row in payload["reply_markup"].inline_keyboard for b in row
                      if b.text == unknown[8])
        await tap(chosen.callback_data)
        callback.assert_awaited_once_with("12345", unknown[8], slug)
    else:
        buttons = [b for row in payload["reply_markup"].inline_keyboard for b in row]
        assert not any(b.callback_data.startswith("mvd:") for b in buttons)
        picks = [b for b in buttons if b.callback_data.startswith("mm:")]
        assert [b.text for b in picks] == models[:8]
        assert "in-region" not in payload["text"]
        await tap(picks[1].callback_data)
        callback.assert_awaited_once_with("12345", models[1], slug)

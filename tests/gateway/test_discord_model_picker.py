"""Regression tests for the Discord /model picker.

Uses the shared discord mock from tests/gateway/conftest.py (installed
at collection time via _ensure_discord_mock()). Previously this file
installed its own mock at module-import time and clobbered sys.modules,
breaking other gateway tests under pytest-xdist.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.platforms.base import utf16_len
from plugins.platforms.discord.adapter import ModelPickerView


@pytest.mark.asyncio
async def test_model_picker_clears_controls_before_running_switch_callback():
    events: list[object] = []

    async def on_model_selected(chat_id: str, model_id: str, provider_slug: str) -> str:
        events.append(("switch", chat_id, model_id, provider_slug))
        return "Model switched"

    async def edit_message(**kwargs):
        events.append(
            (
                "initial-edit",
                kwargs["embed"].title,
                kwargs["embed"].description,
                kwargs["view"],
            )
        )

    async def edit_original_response(**kwargs):
        events.append((
            "final-edit",
            kwargs["embed"].title,
            kwargs["embed"].description,
            kwargs["view"],
        ))

    view = ModelPickerView(
        providers=[
            {
                "slug": "copilot",
                "name": "GitHub Copilot",
                "models": ["gpt-5.4"],
                "total_models": 1,
                "is_current": True,
            }
        ],
        current_model="gpt-5-mini",
        current_provider="copilot",
        session_key="session-1",
        on_model_selected=on_model_selected,
        allowed_user_ids={"123"},  # matches the interaction user; empty = fail-closed
    )
    view._selected_provider = "copilot"

    interaction = SimpleNamespace(
        user=SimpleNamespace(id=123),
        channel_id=456,
        data={"values": ["gpt-5.4"]},
        response=SimpleNamespace(
            defer=AsyncMock(),
            send_message=AsyncMock(),
            edit_message=AsyncMock(side_effect=edit_message),
        ),
        edit_original_response=AsyncMock(side_effect=edit_original_response),
    )

    await view._on_model_selected(interaction)

    assert events == [
        ("initial-edit", "⚙ Switching Model", "Switching to `gpt-5.4`...", None),
        ("switch", "456", "gpt-5.4", "copilot"),
        ("final-edit", "⚙ Model Switched", "Model switched", None),
    ]
    interaction.response.edit_message.assert_awaited_once()
    interaction.response.defer.assert_not_called()
    interaction.edit_original_response.assert_awaited_once()


def test_model_picker_provider_labels_fit_discord_utf16_limit():
    provider_name = "Provider " + ("\U0001f600" * 80)

    view = ModelPickerView(
        providers=[
            {
                "slug": "emoji",
                "name": provider_name,
                "models": ["gpt-5-mini"],
                "total_models": 1,
                "is_current": False,
            }
        ],
        current_model="gpt-5-mini",
        current_provider="emoji",
        session_key="session-1",
        on_model_selected=AsyncMock(return_value="ok"),
        allowed_user_ids={"123"},
    )

    provider_select = view.children[0]
    option = provider_select.options[0]
    assert utf16_len(option.label) <= 100


def test_model_picker_model_labels_and_values_fit_discord_utf16_limit():
    model_id = "emoji/" + ("\U0001f600" * 80)

    view = ModelPickerView(
        providers=[
            {
                "slug": "emoji",
                "name": "Emoji",
                "models": [model_id],
                "total_models": 1,
                "is_current": False,
            }
        ],
        current_model="gpt-5-mini",
        current_provider="emoji",
        session_key="session-1",
        on_model_selected=AsyncMock(return_value="ok"),
        allowed_user_ids={"123"},
    )

    view._build_model_select("emoji")
    model_select = view.children[0]
    option = model_select.options[0]
    assert utf16_len(option.label) <= 100
    assert utf16_len(option.value) <= 100


@pytest.mark.asyncio
async def test_expensive_model_requires_confirmation(monkeypatch):
    events: list[object] = []

    async def on_model_selected(chat_id: str, model_id: str, provider_slug: str) -> str:
        events.append(("switch", chat_id, model_id, provider_slug))
        return "Model switched"

    async def edit_message(**kwargs):
        events.append(
            (
                "edit",
                kwargs["embed"].title,
                kwargs["embed"].description,
                kwargs["view"],
            )
        )

    async def edit_original_response(**kwargs):
        events.append((
            "final-edit",
            kwargs["embed"].title,
            kwargs["embed"].description,
            kwargs["view"],
        ))

    monkeypatch.setattr(
        "hermes_cli.model_cost_guard.expensive_model_warning",
        lambda *_args, **_kwargs: SimpleNamespace(
            message="!!! EXPENSIVE MODEL WARNING !!!\ndid you mean to select openai/gpt-5.5?"
        ),
    )

    view = ModelPickerView(
        providers=[
            {
                "slug": "openrouter",
                "name": "OpenRouter",
                "models": ["openai/gpt-5.5-pro"],
                "total_models": 1,
                "is_current": True,
            }
        ],
        current_model="openai/gpt-5.5",
        current_provider="openrouter",
        session_key="session-1",
        on_model_selected=on_model_selected,
        allowed_user_ids={"123"},  # matches the interaction user; empty = fail-closed
    )
    view._selected_provider = "openrouter"

    interaction = SimpleNamespace(
        user=SimpleNamespace(id=123),
        channel_id=456,
        data={"values": ["openai/gpt-5.5-pro"]},
        response=SimpleNamespace(
            send_message=AsyncMock(),
            edit_message=AsyncMock(side_effect=edit_message),
        ),
        edit_original_response=AsyncMock(side_effect=edit_original_response),
    )

    await view._on_model_selected(interaction)

    assert events == [
        (
            "edit",
            "⚠ Expensive Model Warning",
            "!!! EXPENSIVE MODEL WARNING !!!\ndid you mean to select openai/gpt-5.5?",
            view,
        ),
    ]
    assert view.resolved is False

    await view._on_expensive_confirm(interaction)

    assert events[1:] == [
        (
            "edit",
            "⚙ Switching Model",
            "Switching to `openai/gpt-5.5-pro`...",
            None,
        ),
        ("switch", "456", "openai/gpt-5.5-pro", "openrouter"),
        ("final-edit", "⚙ Model Switched", "Model switched", None),
    ]

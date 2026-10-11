"""Regression tests for the Discord /model picker.

Uses the shared discord mock from tests/gateway/conftest.py (installed
at collection time via _ensure_discord_mock()). Previously this file
installed its own mock at module-import time and clobbered sys.modules,
breaking other gateway tests in the same process.
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


def _select_items(view):
    """All select menus currently attached to the view."""
    return [item for item in view.children if getattr(item, "options", None)]


def test_provider_select_skips_duplicate_slug_values():
    """A duplicated provider slug must not render two options with the same value: Discord
    rejects the whole component payload (50035, "The specified option value is already used")
    and the /model interaction times out (#134258)."""
    view = ModelPickerView(
        providers=[
            {
                "slug": "openrouter",
                "name": "OpenRouter",
                "models": [],
                "total_models": 1,
            },
            {
                "slug": "openrouter",
                "name": "OpenRouter",
                "models": [],
                "total_models": 1,
            },
            {
                "slug": "copilot",
                "name": "GitHub Copilot",
                "models": [],
                "total_models": 2,
            },
        ],
        current_model="gpt-5-mini",
        current_provider="copilot",
        session_key="session-1",
        on_model_selected=None,
        allowed_user_ids={"123"},
    )

    (select,) = _select_items(view)
    values = [opt.value for opt in select.options]

    assert values == ["openrouter", "copilot"]


def test_model_select_skips_truncation_colliding_values():
    """Model ids that collide after the 100-char option-value truncation keep only the first
    rendering — two identical values would trip the same duplicate-value rejection (#134258)."""
    colliding_a = "m" * 99 + "a" + "-tail-one"
    colliding_b = "m" * 99 + "a" + "-tail-two"
    view = ModelPickerView(
        providers=[
            {
                "slug": "openrouter",
                "name": "OpenRouter",
                "models": [colliding_a, colliding_b, "deepseek-chat"],
                "total_models": 3,
            },
        ],
        current_model="deepseek-chat",
        current_provider="openrouter",
        session_key="session-1",
        on_model_selected=None,
        allowed_user_ids={"123"},
    )

    view._build_model_select("openrouter")
    (select,) = _select_items(view)
    values = [opt.value for opt in select.options]

    assert len(values) == len(set(values))
    assert values[0] == "m" * 99 + "a"
    assert values[-1] == "deepseek-chat"

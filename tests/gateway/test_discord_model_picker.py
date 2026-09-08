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


@pytest.mark.asyncio
async def test_model_picker_on_timeout_does_not_overwrite_completed_switch():
    """Regression: once a model switch resolves, the 120s Discord view timeout
    must NOT edit the stored message to the expired embed — otherwise the
    green "Model Switched" confirmation is clobbered with a stale "Selection
    expired" message. (#69685)"""
    from unittest.mock import AsyncMock

    async def on_model_selected(chat_id, model_id, provider_slug):
        return "Model switched"

    view = ModelPickerView(
        providers=[{
            "slug": "kimi",
            "name": "Kimi",
            "models": ["k3"],
            "total_models": 1,
            "is_current": False,
        }],
        current_model="x",
        current_provider="kimi",
        session_key="session-1",
        on_model_selected=on_model_selected,
        allowed_user_ids={"123"},
    )
    view._selected_provider = "kimi"

    # Simulate a completed switch: resolved=True, message edit was awaited.
    view.resolved = True
    view._message = SimpleNamespace(
        edit=AsyncMock(),
    )

    # The stale 120s timer fires.
    await view.on_timeout()

    # The message edit must NOT have been awaited — the confirmation
    # embed is the authoritative post-switch state.
    view._message.edit.assert_not_awaited()
    # resolved is still True (idempotent).
    assert view.resolved is True



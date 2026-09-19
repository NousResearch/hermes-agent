"""Regression coverage for #110784's gateway switch confirmation."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.run import GatewayRunner
from gateway.slash_commands_model import _ModelSwitchContext
from hermes_cli.model_switch import ModelSwitchResult


@pytest.mark.asyncio
async def test_gateway_confirmation_shows_ordered_openrouter_route(monkeypatch):
    import gateway.run as gateway_run

    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {"model": {}, "provider_routing": {"models": {
            "z-ai/glm-5.3:exacto": {"order": ["together", "fireworks"]},
        }}},
    )
    monkeypatch.setattr(
        "hermes_cli.model_switch.resolve_display_context_length_async",
        AsyncMock(return_value=None),
    )
    result = ModelSwitchResult(
        success=True,
        new_model="z-ai/glm-5.3:exacto",
        target_provider="openrouter",
        provider_label="OpenRouter",
    )
    context = _ModelSwitchContext(session_key="test", source=SimpleNamespace(), config_path=None, persist_global=False)

    confirmation = await GatewayRunner._model_switch_confirmation(
        object.__new__(GatewayRunner), result, context, one_turn=False, picker=False,
    )

    assert "Inference routing: Together → Fireworks" in confirmation

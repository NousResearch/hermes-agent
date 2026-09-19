"""Gateway model-switch confirmations expose configured OpenRouter routing.

Regression for #110784.
"""

from datetime import date
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from gateway.slash_commands_model import _configured_provider_routing_line


@pytest.mark.parametrize("routing", [{"sort": "bogus"}, {"only": [date(2026, 1, 1)]}])
def test_routing_display_omits_values_the_request_path_cannot_emit(monkeypatch, routing):
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {})

    assert _configured_provider_routing_line(routing, "test/model") is None


@pytest.mark.asyncio
async def test_model_command_reads_routing_from_profile_config(tmp_path, monkeypatch):
    import gateway.run as gateway_run
    from hermes_cli.model_switch import ModelSwitchResult

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        """
model:
  default: old-model
  provider: openrouter
provider_routing:
  sort: price
  models:
    "z-ai/glm-5.3:exacto":
      only: [fireworks]
      data_collection: deny
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: hermes_home)
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: hermes_home)
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(
        "hermes_cli.model_cost_guard.expensive_model_warning",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model",
        lambda **_kwargs: ModelSwitchResult(
            success=True,
            new_model="z-ai/glm-5.3:exacto",
            target_provider="openrouter",
            provider_changed=False,
            api_key="test",
            base_url="https://openrouter.ai/api/v1",
            api_mode="chat_completions",
            provider_label="OpenRouter",
        ),
    )

    async def _no_context(*_args, **_kwargs):
        return None

    monkeypatch.setattr(
        "hermes_cli.model_switch.resolve_display_context_length_async",
        _no_context,
    )

    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._session_model_overrides = {}
    runner._pending_one_turn_model_restores = {}
    runner._running_agents = {}
    runner._provider_routing = {"sort": " Throughput "}
    store = MagicMock()
    store.set_model_override = AsyncMock()
    store._store = None
    setattr(runner, "session_store", None)
    runner._async_session_store = store
    event = MessageEvent(
        text="/model glm53",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.SLACK,
            user_id="user",
            chat_id="channel",
            chat_type="group",
        ),
    )

    confirmation = await runner._handle_model_command(event)

    assert confirmation is not None
    assert (
        'provider_routing: {"only":["fireworks"],"sort":"throughput",'
        '"data_collection":"deny"}' in confirmation
    )

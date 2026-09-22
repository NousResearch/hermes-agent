"""/reasoning offers the levels the route declares, not the shared ladder.

A relay that takes four levels used to be presented with all seven: the other three either
400 at the endpoint or are silently clamped. Undeclared routes stay fail-open.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource

CONFIG_DECLARED = """\
agent:
  reasoning_effort: medium
model:
  provider: relay
  default: qwen-next
providers:
  relay:
    base_url: "http://relay.invalid/v1"
    models:
      qwen-next:
        context_length: 262144
        reasoning_efforts: [none, low, high, max]
"""

CONFIG_UNDECLARED = """\
agent:
  reasoning_effort: medium
model:
  provider: relay
  default: qwen-next
providers:
  relay:
    base_url: "http://relay.invalid/v1"
    models:
      qwen-next:
        context_length: 262144
"""


def _make_event():
    return MessageEvent(
        text="/reasoning",
        source=SessionSource(
            platform=Platform.TELEGRAM, user_id="12345", chat_id="67890", user_name="testuser"
        ),
    )


def _make_runner(captured):
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    runner._ephemeral_system_prompt = ""
    runner._prefill_messages = []
    runner._reasoning_config = {"enabled": True, "effort": "high"}
    runner._session_reasoning_overrides = {}
    runner._session_model_overrides = {}
    runner._show_reasoning = False
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._running_agents = {}
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.hooks.loaded_hooks = []
    runner._session_db = None
    runner._get_or_create_gateway_honcho = lambda session_key: (None, None)

    async def _capture(event, session_key, title, choices, on_choice_selected):
        captured.extend(choices)
        return True

    runner._try_send_choice_picker = _capture
    return runner


async def _collect_offered_values(tmp_path, monkeypatch, config_text):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    config_path = hermes_home / "config.yaml"
    config_path.write_text(config_text, encoding="utf-8")
    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)

    captured: list = []
    runner = _make_runner(captured)
    runner.config_path = config_path

    assert await runner._handle_reasoning_command(_make_event()) is None
    return [choice["value"] for choice in captured]


def test_picker_offers_only_the_declared_levels(tmp_path, monkeypatch):
    values = asyncio.run(_collect_offered_values(tmp_path, monkeypatch, CONFIG_DECLARED))

    assert [v for v in values if v not in ("reset", "show", "hide")] == ["none", "low", "high", "max"]
    # The controls the command has always carried are untouched.
    assert values[-3:] == ["reset", "show", "hide"]
    for undeclared in ("minimal", "medium", "xhigh", "ultra"):
        assert undeclared not in values


def test_undeclared_route_keeps_the_whole_ladder(tmp_path, monkeypatch):
    from hermes_constants import VALID_REASONING_EFFORTS

    values = asyncio.run(_collect_offered_values(tmp_path, monkeypatch, CONFIG_UNDECLARED))

    assert [v for v in values if v not in ("none", "reset", "show", "hide")] == list(
        VALID_REASONING_EFFORTS
    )

"""A per-turn or per-request reasoning level survives a provider fallback mid-turn.

``/model <m> --once --reasoning X`` (the CLI-turn record's ``model_override.reasoning``) and an API
request's ``model_options`` reasoning replace the session's level for one turn. The caller chose X
explicitly, so a fallback activated during that turn must keep it. A turn with neither override
keeps the session rule: unpinned, so the fallback model's configured level applies.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.session import SessionSource
from gateway.session_cli_turn import cli_turn_scope
from hermes_constants import get_hermes_home
from hermes_cli.model_switch import ModelSwitchResult
from run_agent import AIAgent

_FALLBACK = {"provider": "openai", "model": "fb-model"}
_CONFIG = (
    "agent:\n  reasoning_effort: medium\n  reasoning_overrides:\n    fb-model: low\n"
    "fallback_providers:\n  - provider: openai\n    model: fb-model\n"
)
_ADMISSION = "adm-1"


class _FallsBackAgent(AIAgent):
    """The real agent; its turn records the level it starts at, activates the configured fallback
    (``bind_route_entry``) and reports the model, level and pin it runs the rest of the turn on."""

    def run_conversation(self, user_message, conversation_history=None, task_id=None, **_kwargs):
        before = (self.reasoning_config or {}).get("effort")
        client = MagicMock()
        client.base_url, client.api_key = "https://api.openai.com/v1", "fb-key"
        with patch("agent.auxiliary_client.resolve_provider_client", return_value=(client, _FALLBACK["model"])):
            assert self._try_activate_fallback() is True
        pinned = bool(getattr(self, "reasoning_pinned", False))
        return {"final_response": f"{before}>{self.model}:{self.reasoning_config['effort']} pinned={pinned}",
                "messages": [], "api_calls": 1}


@pytest.fixture
def gateway_home(monkeypatch):
    home = get_hermes_home()
    (home / "config.yaml").write_text(_CONFIG, encoding="utf-8")
    monkeypatch.setattr(gateway_run, "_hermes_home", home)
    monkeypatch.setattr(gateway_run, "_env_path", home / ".env")
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {
        "provider": "openrouter", "api_mode": "chat_completions",
        "base_url": "https://openrouter.ai/api/v1", "api_key": "test-key"})
    monkeypatch.setattr("run_agent.AIAgent", _FallsBackAgent)
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda *_a, **_k: [])
    monkeypatch.setattr("model_tools.check_toolset_requirements", lambda *_a, **_k: {})
    monkeypatch.setattr("agent.process_bootstrap.OpenAI", MagicMock())
    # `/model --once <m>` resolves its target through switch_model; offline, the target is a model on
    # the session's own provider.
    monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda **_k: ModelSwitchResult(
        success=True, new_model="once-model", target_provider="openrouter", api_key="",
        base_url="https://openrouter.ai/api/v1", api_mode="chat_completions", request_overrides={}))
    monkeypatch.setattr("hermes_cli.config.get_compatible_custom_providers", lambda _config: [])
    return home


def _config_levels():
    """The levels the fixture's config hands each model of the turn (per-model, else global)."""
    from hermes_cli.config import load_config
    from hermes_constants import resolve_reasoning_config
    return {resolve_reasoning_config(load_config(), model)["effort"] for model in ("once-model", "fb-model")}


def _runner():
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {}
    runner._ephemeral_system_prompt = ""
    runner._prefill_messages = []
    runner._reasoning_config = None
    runner._session_reasoning_overrides = {}
    runner._show_reasoning = False
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._running_agents = {}
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.hooks.loaded_hooks = []
    runner._session_db = None
    runner._get_or_create_gateway_honcho = lambda session_key: (None, None)
    return runner


def _turn(cli_fields=None):
    runner = _runner()
    source = SessionSource(platform=Platform.TELEGRAM, user_id="12345", chat_id="67890", user_name="testuser")
    with cli_turn_scope(_ADMISSION, cli_fields):
        return asyncio.run(runner._run_agent(
            message="ping", context_prompt="", history=[], source=source, session_id="session-1",
            session_key=runner._session_key_for_source(source), inbound_message_id=_ADMISSION))


def _api_turn(model_options):
    """An API-server request on the session's own route (no requested model), carrying *model_options*."""
    from types import SimpleNamespace
    from gateway.session_api_turn import api_execution
    adapter = SimpleNamespace(_model_routes={}, _select_agent_runtime=lambda _rt, model, **_k: (model, None, None, None))
    token = api_execution.set({"adapter": adapter, "settings": {"model_options": model_options}, "history": None})
    try:
        return _turn()
    finally:
        api_execution.reset(token)


def test_a3_once_reasoning_level_survives_fallback(gateway_home):
    """A3-once: `/model once-model --once --reasoning high` keeps high after the fallback bind."""
    assert _config_levels() == {"medium", "low"}  # config never yields the requested "high"

    result = _turn({"model_override": {"model": "once-model", "reasoning": "high"}})

    assert result["final_response"] == "high>fb-model:high pinned=True"


def test_a3_once_level_equal_to_the_config_level_still_pins(gateway_home):
    """A3-once-equal: `--once --reasoning medium` names the profile's own global level. It is still
    the caller's choice, so the fallback keeps medium instead of fb-model's configured low."""
    result = _turn({"model_override": {"model": "once-model", "reasoning": "medium"}})

    assert result["final_response"] == "medium>fb-model:medium pinned=True"


def test_a3_api_request_reasoning_level_survives_fallback(gateway_home):
    """A3-api: an API request's model_options reasoning keeps its level after the fallback bind."""
    assert _config_levels() == {"medium", "low"}  # config never yields the requested "high"

    result = _api_turn({"reasoning": {"effort": "high"}})

    assert result["final_response"] == "high>fb-model:high pinned=True"


def test_a3_none_turn_without_override_resolves_the_fallback_models_config(gateway_home):
    """A3-none: no per-turn level and no session pin, so the fallback follows config (low)."""
    result = _turn()

    assert result["final_response"] == "medium>fb-model:low pinned=False"


def test_a3_once_without_reasoning_does_not_pin(gateway_home):
    """A3-once-noreasoning: `/model once-model --once` changes the route only, not who owns the level."""
    result = _turn({"model_override": {"model": "once-model"}})

    assert result["final_response"] == "medium>fb-model:low pinned=False"


def test_a3_api_request_without_reasoning_does_not_pin(gateway_home):
    """An API request whose model_options carry no reasoning keeps today's unpinned fallback."""
    result = _api_turn({})

    assert result["final_response"] == "medium>fb-model:low pinned=False"


def _direct_api_agent(monkeypatch, model_options):
    """The API server's direct (non-canonical) path: ``APIServerAdapter._create_agent`` builds the
    request's agent itself; the turn then activates the configured fallback."""
    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda: "primary-model")
    agent = APIServerAdapter(PlatformConfig())._create_agent(model_options=model_options)
    return agent.run_conversation("ping")["final_response"]


def test_a3_api_direct_request_reasoning_level_survives_fallback(gateway_home, monkeypatch):
    """R4 api_server ``_create_agent``: a request's model_options reasoning keeps its level."""
    assert _config_levels() == {"medium", "low"}  # config never yields the requested "high"

    assert _direct_api_agent(monkeypatch, {"reasoning_effort": "high"}) == "high>fb-model:high pinned=True"


def test_a3_api_direct_request_without_reasoning_resolves_the_fallback_models_config(gateway_home, monkeypatch):
    """The same direct path without a requested level stays unpinned and follows config."""
    assert _direct_api_agent(monkeypatch, None) == "medium>fb-model:low pinned=False"

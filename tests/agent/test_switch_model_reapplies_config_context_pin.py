"""A live model/provider switch must re-derive the durable ``model.context_length`` pin.

``model.context_length`` is the user's profile-wide ceiling. It is read from config.yaml exactly once,
at agent construction (``agent_init._resolve_context_length``), then cached on
``agent._config_context_length``. Every live path that re-resolves a runtime cleared that cached copy
instead of re-reading the config, so a long-running Desktop/CLI session silently lost the ceiling and
fell through to provider probing / catalog metadata / the 256K fallback — while the ceiling was still
on disk and still shown by other surfaces.
"""

from unittest.mock import MagicMock, patch

from agent.context_compressor import ContextCompressor
from run_agent import AIAgent

PIN = 249_000
ROUTE = "http://127.0.0.1:8123/v1"
OTHER_ROUTE = "https://openrouter.ai/api/v1"


def _cfg_with_pin():
    return {
        "model": {
            "default": "model-b",
            "provider": "custom:acme",
            "base_url": ROUTE,
            "context_length": PIN,
        },
        "custom_providers": [{"name": "acme", "base_url": ROUTE, "models": {}}],
    }


def _make_agent(config_context_length=None):
    agent = AIAgent.__new__(AIAgent)
    agent.model = "model-a"
    agent.provider = "openrouter"
    agent.base_url = OTHER_ROUTE
    agent.api_key = "sk-primary"
    agent.api_mode = "chat_completions"
    agent.client = MagicMock()
    agent.quiet_mode = True
    agent._config_context_length = config_context_length
    agent._custom_providers = []
    agent._primary_runtime = {}
    agent._create_openai_client = lambda *_args, **_kwargs: MagicMock()
    agent.context_compressor = ContextCompressor(
        model="model-a", threshold_percent=0.50, base_url=OTHER_ROUTE,
        api_key="sk-primary", provider="openrouter", quiet_mode=True,
    )
    return agent


def _switch(agent, model, provider, base_url):
    """Switch and return the ``config_context_length`` the compressor resolution ran with."""
    with patch("agent.model_metadata.get_model_context_length", return_value=200_000) as ctx_len:
        agent.switch_model(model, provider, api_key="sk-new", base_url=base_url)
    return ctx_len.call_args.kwargs.get("config_context_length")


def test_switch_to_configured_default_route_reapplies_context_pin():
    """Switching onto the route ``model.context_length`` describes keeps the ceiling."""
    cfg = _cfg_with_pin()
    agent = _make_agent(config_context_length=None)

    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("hermes_cli.config.load_config_readonly", return_value=cfg),
    ):
        applied = _switch(agent, "model-b", "custom:acme", ROUTE)

    assert applied == PIN
    assert agent._config_context_length == PIN


def test_switch_back_to_configured_default_route_restores_context_pin():
    """A round-trip away from and back onto the configured route restores the ceiling."""
    cfg = _cfg_with_pin()
    agent = _make_agent(config_context_length=PIN)

    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch("agent.model_metadata.get_model_context_length", return_value=1_000_000),
    ):
        agent.switch_model("other-model", "openrouter", api_key="sk-new", base_url=OTHER_ROUTE)
        assert agent._config_context_length is None  # pin does not describe the other route
        applied = _switch(agent, "model-b", "custom:acme", ROUTE)

    assert applied == PIN
    assert agent._config_context_length == PIN


def test_switch_to_other_route_does_not_inherit_context_pin():
    """The pin stays scoped to its own route (no leak onto an unrelated model)."""
    cfg = _cfg_with_pin()
    agent = _make_agent(config_context_length=PIN)

    with (
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("hermes_cli.config.load_config_readonly", return_value=cfg),
    ):
        applied = _switch(agent, "unrelated-model", "openrouter", OTHER_ROUTE)

    assert applied is None
    assert agent._config_context_length is None

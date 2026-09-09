"""context_window is accepted as an alias for model.context_length (issue #8015)."""

import types
from unittest.mock import patch


def test_gateway_model_context_resolves_context_window():
    """_resolve_gateway_model_context honors model.context_window alone."""
    from gateway import run as gw_run

    cfg = {"model": {"default": "gpt5.4", "provider": "custom",
                     "base_url": "http://localhost:4000/v1",
                     "context_window": 1000000}}
    with (
        patch.object(gw_run, "_load_gateway_config", return_value=cfg),
        patch.object(gw_run, "_resolve_runtime_agent_kwargs", return_value={}),
        patch("agent.model_metadata.get_model_context_length",
              side_effect=lambda model, **kw: kw.get("config_context_length") or 128000),
    ):
        resolved = gw_run._resolve_gateway_model_context()
    assert resolved.context_length == 1000000
    assert resolved.context_source == "config"


def test_gateway_model_context_prefers_context_length():
    """When both keys are set, the canonical context_length wins."""
    from gateway import run as gw_run

    cfg = {"model": {"default": "gpt5.4", "provider": "custom",
                     "base_url": "http://localhost:4000/v1",
                     "context_length": 256000, "context_window": 1000000}}
    with (
        patch.object(gw_run, "_load_gateway_config", return_value=cfg),
        patch.object(gw_run, "_resolve_runtime_agent_kwargs", return_value={}),
        patch("agent.model_metadata.get_model_context_length",
              side_effect=lambda model, **kw: kw.get("config_context_length") or 128000),
    ):
        resolved = gw_run._resolve_gateway_model_context()
    assert resolved.context_length == 256000


def test_hygiene_read_config_resolves_context_window():
    """_hmwa_hygiene_read_config honors model.context_window alone."""
    from gateway.run_turn import GatewayTurnMixin

    hs = types.SimpleNamespace(model="m", config_context_length=None,
                               provider=None, base_url=None, compression_enabled=True,
                               hard_msg_limit=5000, timeout_seconds=30.0,
                               total_ceiling_seconds=600.0, max_turn_hold_seconds=10.0,
                               failure_cooldown_seconds=300.0)
    GatewayTurnMixin._hmwa_hygiene_read_config(
        hs, {"model": {"default": "gpt5.4", "context_window": 1000000}})
    assert hs.config_context_length == 1000000


def test_context_window_busts_agent_cache():
    """A context_window edit must invalidate the cached gateway agent."""
    from gateway.run import GatewayRunner

    assert ("model", "context_window") in GatewayRunner._CACHE_BUSTING_CONFIG_KEYS

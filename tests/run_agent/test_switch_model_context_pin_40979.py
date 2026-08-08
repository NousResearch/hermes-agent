"""Regression tests for #40979: the global ``model.context_length`` pin must
survive a /model switch back to the configured default model+route.

Before the fix, ``switch_model()`` unconditionally set
``agent._config_context_length = None`` and never re-read config.yaml, so the
global pin worked at startup but silently disappeared after ANY /model switch
— even ``/model <other>`` followed by ``/model <configured-default>``. The
compressor then sized itself from the full catalog window (e.g. 1M for
long-context models the user deliberately capped).

The fix re-resolves the pin from live config at switch time, scoped exactly
like startup (agent_init.py): restored ONLY when the switched-to model+route
equals the configured default model+route, so the pin still cannot leak onto
other models (#62152) and the stale-inheritance behavior the original
clearing protected against is preserved.
"""

from unittest.mock import MagicMock, patch

from run_agent import AIAgent
from agent.context_compressor import ContextCompressor


NOUS_URL = "https://inference-api.nousresearch.com/v1"


def _make_agent(model, provider, base_url, config_context_length=None):
    agent = AIAgent.__new__(AIAgent)
    agent.model = model
    agent.provider = provider
    agent.base_url = base_url
    agent.api_key = "sk-test"
    agent.api_mode = "chat_completions"
    agent.client = MagicMock()
    agent.quiet_mode = True
    agent._config_context_length = config_context_length
    agent.context_compressor = ContextCompressor(
        model=model,
        threshold_percent=0.50,
        base_url=base_url,
        api_key="sk-test",
        provider=provider,
        quiet_mode=True,
        config_context_length=config_context_length,
    )
    agent._primary_runtime = {}
    return agent


def _cfg(default, provider, context_length, base_url=None):
    model_cfg = {
        "default": default,
        "provider": provider,
        "context_length": context_length,
    }
    if base_url:
        model_cfg["base_url"] = base_url
    return {"model": model_cfg}


def test_pin_restored_when_switching_back_to_configured_default():
    """/model away and back must restore the configured default's pin."""
    cfg = _cfg("poolside/laguna-s-2.1:free", "nous", 262_144)
    agent = _make_agent(
        "tencent/hy3:free", "nous", NOUS_URL, config_context_length=None
    )

    with (
        patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch(
            "agent.model_metadata.get_model_context_length",
            side_effect=lambda *a, **k: k.get("config_context_length")
            or 1_048_576,
        ) as mock_ctx,
    ):
        agent.switch_model(
            "poolside/laguna-s-2.1:free", "nous",
            api_key="sk-test", base_url=NOUS_URL,
        )

    assert agent._config_context_length == 262_144
    assert (
        mock_ctx.call_args.kwargs.get("config_context_length") == 262_144
    )
    assert agent.context_compressor.context_length == 262_144


def test_pin_not_leaked_onto_other_models():
    """Switching to a model other than the configured default clears the pin."""
    cfg = _cfg("poolside/laguna-s-2.1:free", "nous", 262_144)
    agent = _make_agent(
        "poolside/laguna-s-2.1:free", "nous", NOUS_URL,
        config_context_length=262_144,
    )

    with (
        patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch(
            "agent.model_metadata.get_model_context_length",
            return_value=131_072,
        ) as mock_ctx,
    ):
        agent.switch_model(
            "tencent/hy3:free", "nous", api_key="sk-test", base_url=NOUS_URL
        )

    assert agent._config_context_length is None
    assert mock_ctx.call_args.kwargs.get("config_context_length") is None


def test_pin_not_restored_across_route_change():
    """Same model name on a different provider/route must not reuse the pin."""
    cfg = _cfg("shared-model", "custom:big-route", 1_048_576,
               base_url="https://big.example/v1")
    agent = _make_agent(
        "other-model", "openrouter", "https://openrouter.ai/api/v1"
    )

    with (
        patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch(
            "agent.model_metadata.get_model_context_length",
            return_value=131_072,
        ),
    ):
        agent.switch_model(
            "shared-model", "openrouter",
            api_key="sk-test", base_url="https://openrouter.ai/api/v1",
        )

    assert agent._config_context_length is None


def test_reselecting_same_default_model_keeps_pin():
    """/model to the model the session already runs must keep the pin."""
    cfg = _cfg("tencent/hy3:free", "nous", 200_000)
    agent = _make_agent(
        "tencent/hy3:free", "nous", NOUS_URL, config_context_length=200_000
    )

    with (
        patch("hermes_cli.config.load_config_readonly", return_value=cfg),
        patch(
            "agent.model_metadata.get_model_context_length",
            side_effect=lambda *a, **k: k.get("config_context_length")
            or 262_144,
        ),
    ):
        agent.switch_model(
            "tencent/hy3:free", "nous", api_key="sk-test", base_url=NOUS_URL
        )

    assert agent._config_context_length == 200_000
    assert agent.context_compressor.context_length == 200_000


def test_unreadable_config_falls_back_to_clearing():
    """A config read failure mid-switch must clear, not inherit, the pin."""
    agent = _make_agent(
        "primary-model", "openrouter", "https://openrouter.ai/api/v1",
        config_context_length=32_768,
    )

    with (
        patch(
            "hermes_cli.config.load_config_readonly",
            side_effect=OSError("disk error"),
        ),
        patch(
            "agent.model_metadata.get_model_context_length",
            return_value=131_072,
        ) as mock_ctx,
    ):
        agent.switch_model(
            "new-model", "openrouter",
            api_key="sk-test", base_url="https://openrouter.ai/api/v1",
        )

    assert agent._config_context_length is None
    assert mock_ctx.call_args.kwargs.get("config_context_length") is None

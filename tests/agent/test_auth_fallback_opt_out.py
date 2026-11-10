"""#132337: an auth verdict that sets ``should_fallback=False`` (a provider profile's
``classify_api_error`` saying "no login where the CLI runs; another provider cannot fix this") must
fail closed instead of escalating to the fallback chain and billing a different provider.
"""

import dataclasses
from unittest.mock import MagicMock, patch

import pytest

import providers
from run_agent import AIAgent

# Assembled at runtime so no credential-shaped literal sits in source (scanner guard).
_TEST_KEY = "test-" + "key-12345678"


class _NotLoggedIn(RuntimeError):
    status_code = 401


def _make_agent():
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("agent.context_compressor.get_model_context_length", return_value=200_000),
    ):
        agent = AIAgent(
            api_key=_TEST_KEY, base_url="https://my-llm.example.com/v1", provider="custom",
            quiet_mode=True, skip_context_files=True, skip_memory=True,
            fallback_model={"provider": "custom", "model": "paid-model", "base_url": "https://paid.example.com/v1"},
        )
        agent.client = MagicMock()
        return agent


@pytest.mark.parametrize("hook_verdict, falls_back", [
    ({"reason": "auth_permanent", "retryable": False, "should_fallback": False}, False),
    (None, True),  # no opinion: the built-in 401 verdict keeps escalating once
])
def test_auth_fallback_follows_the_verdict(hook_verdict, falls_back):
    agent = _make_agent()
    real = providers.get_provider_profile
    hooked = dataclasses.replace(real("custom"), classify_api_error=lambda error, **_: hook_verdict)
    with (
        patch("providers.get_provider_profile", side_effect=lambda name, *a, **k: hooked if name == "custom" else real(name, *a, **k)),
        patch.object(agent, "_interruptible_api_call", side_effect=_NotLoggedIn("Not logged in")),
        patch.object(agent, "_try_activate_fallback", return_value=False) as activate,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("hello")
    assert activate.called is falls_back
    assert result.get("completed") is not True

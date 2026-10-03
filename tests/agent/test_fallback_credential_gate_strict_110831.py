"""Regression coverage for fallback destinations that cannot authenticate (#110831)."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import time

from openai import OpenAI
import pytest

from agent.chat_completion_helpers import _fallback_destination_auth_failure
from run_agent import AIAgent


_TEST_KEY = "test-key-12345678"


def _make_agent(fallback_model):
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key=_TEST_KEY,
            base_url="https://primary.example.com/v1",
            provider="custom",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallback_model,
        )
        agent.client = MagicMock()
        return agent


def _client(base_url, api_key=""):
    return SimpleNamespace(base_url=base_url, api_key=api_key)


def _pool(*, available):
    pool = MagicMock()
    pool.has_credentials.return_value = True
    pool.has_available.return_value = available
    # A short cooldown reaches the auth gate instead of the long-exhaustion guard.
    pool.next_available_at.return_value = time.time() + 60
    return pool


def _activate(agent, client, pool):
    with (
        patch("agent.auxiliary_client.resolve_provider_client", return_value=(client, "fallback-model")),
        patch("agent.credential_pool.load_pool", return_value=pool),
    ):
        return agent._try_activate_fallback()


def test_paid_destination_without_credentials_is_skipped():
    agent = _make_agent({"provider": "opencode", "model": "deepseek-v4-flash-free"})
    assert _activate(agent, _client("https://opencode.ai/zen/v1"), None) is False
    assert agent.provider == "custom"
    assert agent._unavailable_fallback_keys == {("opencode", "deepseek-v4-flash-free", "")}


def test_exhausted_pool_is_not_treated_as_a_valid_fallback():
    agent = _make_agent({"provider": "zai", "model": "glm-5.2"})
    assert _activate(agent, _client("https://api.z.ai/api/paas/v4/"), _pool(available=False)) is False
    assert agent.provider == "custom"


def test_available_pool_allows_a_remote_destination_without_client_key():
    agent = _make_agent({"provider": "zai", "model": "glm-5.2"})
    assert _activate(agent, _client("https://api.z.ai/api/paas/v4/"), _pool(available=True)) is True
    assert agent.provider == "zai"


def test_pool_read_failure_fails_closed_instead_of_entering_a_401_loop():
    client = _client("https://api.z.ai/api/paas/v4/")
    with patch("agent.credential_pool.load_pool", side_effect=OSError("auth store unavailable")):
        reason = _fallback_destination_auth_failure("zai", client.base_url, client)
    assert reason == "credential availability could not be verified"


def test_non_api_key_provider_still_activates_without_a_pool():
    agent = _make_agent({"provider": "bedrock", "model": "amazon.nova-lite-v1:0"})
    assert _activate(
        agent,
        _client("https://bedrock-runtime.us-east-1.amazonaws.com"),
        None,
    ) is True
    assert agent.provider == "bedrock"


def test_local_endpoint_still_activates_without_a_key():
    agent = _make_agent({
        "provider": "custom",
        "model": "local-model",
        "base_url": "http://127.0.0.1:11434/v1",
    })
    assert _activate(agent, _client("http://127.0.0.1:11434/v1"), None) is True
    assert agent.provider == "custom"


def test_auth_header_counts_when_client_key_is_empty():
    client = SimpleNamespace(
        api_key="",
        default_headers={"Authorization": "Bearer configured"},
    )
    assert _fallback_destination_auth_failure("zai", "https://api.z.ai/v1", client) is None

def test_no_key_placeholder_does_not_authenticate_remote_destination():
    client = _client("https://api.z.ai/v1", api_key="no-key-required")
    assert _fallback_destination_auth_failure(
        "zai", client.base_url, client
    ) == "no usable credentials are available"


@pytest.mark.parametrize(("api_key", "headers", "usable"), [
    ("no-key-required", None, False),
    ("", None, False),
    ("  ", None, False),
    ("", {"Authorization": "Bearer"}, False),
    ("", {"Authorization": "Bearer "}, False),
    ("", {"Authorization": "bEaReR\t no-key-required "}, False),
    ("", {"Authorization": "Basic "}, False),
    ("no-key-required", {"Authorization": "Bearer configured"}, True),
    ("", {"authorization": "Basic dXNlcjpwYXNz"}, True),
    ("", {"Authorization": "Token configured"}, True),
    (_TEST_KEY, None, True),
    (lambda: _TEST_KEY, None, True),
])
def test_real_client_requires_usable_authorization_credentials(api_key, headers, usable):
    # The SDK synthesizes Authorization from api_key, including Hermes' placeholder.
    with OpenAI(
        api_key=api_key,
        base_url="https://fallback.example.com/v1",
        default_headers=headers,
    ) as client:
        reason = _fallback_destination_auth_failure("custom", str(client.base_url), client)
        assert reason == (None if usable else "no usable credentials are available")
        assert _fallback_destination_auth_failure(
            "custom", "http://127.0.0.1:11434/v1", client,
        ) is None


def test_real_custom_fallback_without_credentials_keeps_primary_runtime():
    base_url = "https://fallback.example.com/v1"
    agent = _make_agent({
        "provider": "custom", "model": "fallback-model", "base_url": base_url,
    })
    primary_runtime = (agent.model, agent.base_url, agent.client)

    # Use the real resolver and credential store in the isolated HERMES_HOME.
    assert agent._try_activate_fallback() is False
    assert (agent.model, agent.base_url, agent.client) == primary_runtime
    assert agent._unavailable_fallback_keys == {("custom", "fallback-model", base_url)}

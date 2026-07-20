"""Per-turn adoption of ~/.hermes/.env credential edits (#67821).

A Settings save (desktop ``PUT /api/env``, ``hermes setup``) updates .env and
the saving process's os.environ, but a live session worker keeps the
base_url/api_key captured at agent init until restart — an open chat silently
kept calling the old endpoint (e.g. a local-server key sent to
api.openai.com → opaque 401).

``AIAgent._try_refresh_env_client_credentials`` re-resolves env-sourced
credentials at the start of each conversation turn and rebuilds the client
when the user edited them. It must react only to env *edits*, never to mere
divergence from the agent's current values: credential-pool rotation and
failover legitimately move the session off the env credential, and config
``model.base_url`` has higher precedence than the env override.
"""

import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from run_agent import AIAgent

DEFAULT_BASE = "https://api.openai.com/v1"
LOCAL_BASE = "http://127.0.0.1:39080"


def _make_agent(*, provider="openai-api", base_url=DEFAULT_BASE, api_key="sk-old"):
    agent = object.__new__(AIAgent)
    agent.provider = provider
    agent.api_mode = "chat_completions"
    agent.base_url = base_url
    agent.api_key = api_key
    agent._client_kwargs = {"base_url": base_url, "api_key": api_key}
    agent._fallback_activated = False
    agent._replace_primary_openai_client = MagicMock(return_value=True)
    return agent


@pytest.fixture
def env(monkeypatch):
    """Dict-driven stand-in for the .env/os.environ resolution chain."""
    values = {}
    import agent.credential_pool as cp

    monkeypatch.setattr(
        cp, "get_env_prefer_dotenv", lambda key: values.get(key, "")
    )
    return values


class TestAdoptsEnvEdits:
    def test_boot_default_adopts_override_on_first_look(self, env):
        """The reported scenario: worker spawned before the user saved the
        override — first turn after the save must switch to the local URL."""
        agent = _make_agent()
        env["OPENAI_API_KEY"] = "sk-old"
        env["OPENAI_BASE_URL"] = LOCAL_BASE

        assert agent._try_refresh_env_client_credentials() is True
        assert agent.base_url == LOCAL_BASE
        assert agent._client_kwargs["base_url"] == LOCAL_BASE
        agent._replace_primary_openai_client.assert_called_once_with(
            reason="env_credential_refresh"
        )

    def test_edit_between_turns_is_adopted(self, env):
        """No-op first turn, then the user saves an override → next turn
        rebuilds the client onto the new endpoint."""
        agent = _make_agent()
        env["OPENAI_API_KEY"] = "sk-old"

        assert agent._try_refresh_env_client_credentials() is False

        env["OPENAI_BASE_URL"] = LOCAL_BASE
        assert agent._try_refresh_env_client_credentials() is True
        assert agent.base_url == LOCAL_BASE

    def test_key_rotation_in_env_is_adopted(self, env):
        agent = _make_agent()
        env["OPENAI_API_KEY"] = "sk-old"

        assert agent._try_refresh_env_client_credentials() is False

        env["OPENAI_API_KEY"] = "sk-new"
        assert agent._try_refresh_env_client_credentials() is True
        assert agent.api_key == "sk-new"
        assert agent._client_kwargs["api_key"] == "sk-new"


class TestLeavesNonEnvStateAlone:
    def test_unchanged_env_is_a_noop(self, env):
        agent = _make_agent()
        env["OPENAI_API_KEY"] = "sk-old"

        assert agent._try_refresh_env_client_credentials() is False
        agent._replace_primary_openai_client.assert_not_called()

    def test_pool_rotation_is_not_stomped(self, env):
        """After the pool rotates the session onto a different key, an
        unchanged env must not flap the session back every turn."""
        agent = _make_agent()
        env["OPENAI_API_KEY"] = "sk-old"
        assert agent._try_refresh_env_client_credentials() is False

        agent.api_key = "sk-rotated-pool-entry"
        assert agent._try_refresh_env_client_credentials() is False
        assert agent.api_key == "sk-rotated-pool-entry"

    def test_custom_endpoint_wins_over_env_edit(self, env):
        """A session running on a config/pool custom endpoint (not the
        registry default, not a previously-seen env value) keeps it."""
        agent = _make_agent(base_url="https://my-proxy.corp.example/v1")
        env["OPENAI_API_KEY"] = "sk-old"
        env["OPENAI_BASE_URL"] = LOCAL_BASE

        assert agent._try_refresh_env_client_credentials() is False
        assert agent.base_url == "https://my-proxy.corp.example/v1"

    def test_skipped_while_failed_over(self, env):
        agent = _make_agent()
        agent._fallback_activated = True
        env["OPENAI_API_KEY"] = "sk-old"
        env["OPENAI_BASE_URL"] = LOCAL_BASE

        assert agent._try_refresh_env_client_credentials() is False

    def test_skipped_for_non_api_key_provider(self, env):
        agent = _make_agent(provider="openai-codex")
        assert agent._try_refresh_env_client_credentials() is False

    def test_skipped_for_non_chat_completions_api_mode(self, env):
        agent = _make_agent()
        agent.api_mode = "anthropic_messages"
        assert agent._try_refresh_env_client_credentials() is False

    def test_skipped_when_no_key_resolves(self, env):
        agent = _make_agent()
        env["OPENAI_BASE_URL"] = LOCAL_BASE

        assert agent._try_refresh_env_client_credentials() is False

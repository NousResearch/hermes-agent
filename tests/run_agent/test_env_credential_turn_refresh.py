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
    agent.requested_provider = provider
    agent.api_mode = "chat_completions"
    agent.base_url = base_url
    agent.api_key = api_key
    agent._client_kwargs = {"base_url": base_url, "api_key": api_key}
    agent._fallback_activated = False
    agent._replace_primary_openai_client = MagicMock(return_value=True)
    agent._reapply_route_client_config = MagicMock()
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

    def test_first_look_does_not_stomp_rotated_pool_key(self, env):
        """#79156: first look with a pool-rotated key must seed the baseline,
        not rewrite api_key back to the env primary."""
        agent = _make_agent(api_key="sk-backup")
        agent._credential_pool = object()  # any non-None pool binding
        agent._credential_pool_entry_id = "entry-backup"
        env["OPENAI_API_KEY"] = "sk-primary"

        assert agent._try_refresh_env_client_credentials() is False
        assert agent.api_key == "sk-backup"
        assert agent._credential_pool_entry_id == "entry-backup"

    def test_adopted_key_rebinds_pool_entry_id(self, env, monkeypatch):
        """#79156: adopting a new env key must rebind _credential_pool_entry_id
        so the next 429 is attributed to the key that actually ran."""
        agent = _make_agent()
        env["OPENAI_API_KEY"] = "sk-old"
        assert agent._try_refresh_env_client_credentials() is False
        agent._credential_pool_entry_id = "stale-rotated-id"

        called = {}

        def fake_sync(a):
            called["agent"] = a
            a._credential_pool_entry_id = "entry-for-sk-new"

        monkeypatch.setattr(
            "agent.agent_runtime_helpers.sync_credential_pool_entry_id",
            fake_sync,
        )

        env["OPENAI_API_KEY"] = "sk-new"
        assert agent._try_refresh_env_client_credentials() is True
        assert agent.api_key == "sk-new"
        assert called.get("agent") is agent
        assert agent._credential_pool_entry_id == "entry-for-sk-new"

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


class TestFailedRebuildRetries:
    def test_failed_rebuild_rolls_back_and_retries_next_turn(self, env):
        """A failed client rebuild must not advance the edit baseline: the
        agent rolls back to the still-live old client's state and the same
        unchanged edit is retried on the next turn."""
        agent = _make_agent()
        env["OPENAI_API_KEY"] = "sk-old"
        assert agent._try_refresh_env_client_credentials() is False

        env["OPENAI_BASE_URL"] = LOCAL_BASE
        agent._replace_primary_openai_client.return_value = False
        assert agent._try_refresh_env_client_credentials() is False
        # Rolled back: agent state still matches the old client.
        assert agent.base_url == DEFAULT_BASE
        assert agent.api_key == "sk-old"
        assert agent._client_kwargs == {"base_url": DEFAULT_BASE, "api_key": "sk-old"}

        agent._replace_primary_openai_client.return_value = True
        assert agent._try_refresh_env_client_credentials() is True
        assert agent.base_url == LOCAL_BASE
        assert agent._client_kwargs["base_url"] == LOCAL_BASE


class TestRouteConfigRefresh:
    def test_base_url_change_recomputes_route_tls_and_headers(self, env):
        """Moving to a new endpoint must recompute route-derived TLS material
        and default headers, exactly as credential-pool rotation does."""
        agent = _make_agent()
        env["OPENAI_API_KEY"] = "sk-old"
        env["OPENAI_BASE_URL"] = LOCAL_BASE

        assert agent._try_refresh_env_client_credentials() is True
        agent._reapply_route_client_config.assert_called_once_with(route_changed=True)

    def test_key_only_change_keeps_route_config(self, env):
        agent = _make_agent()
        env["OPENAI_API_KEY"] = "sk-old"
        assert agent._try_refresh_env_client_credentials() is False

        env["OPENAI_API_KEY"] = "sk-new"
        assert agent._try_refresh_env_client_credentials() is True
        agent._reapply_route_client_config.assert_called_once_with(route_changed=False)


CUSTOM_BASE = "https://api.longcat.example/openai/v1"


@pytest.fixture
def named_custom_provider(monkeypatch):
    """Register a named custom provider (config `providers.longcat` block)."""
    block = {"name": "longcat", "base_url": CUSTOM_BASE, "key_env": "LONGCAT_API_KEY"}
    import hermes_cli.runtime_provider as rp

    monkeypatch.setattr(
        rp,
        "_get_named_custom_provider",
        lambda requested: block if requested == "longcat" else None,
    )
    return block


class TestNamedCustomProviders:
    """#67935: named custom providers resolve to provider="custom" with no
    PROVIDER_REGISTRY entry — their `key_env` credential must refresh too."""

    def _make_custom_agent(self, *, api_key="no-key-required"):
        agent = _make_agent(provider="custom", base_url=CUSTOM_BASE, api_key=api_key)
        agent.requested_provider = "longcat"
        return agent

    def test_key_added_mid_session_is_adopted(self, env, named_custom_provider):
        """The #67935 repro: long-lived worker spawned before the key_env var
        was written to .env — the first turn after the save must pick it up."""
        agent = self._make_custom_agent()
        env["LONGCAT_API_KEY"] = "lc-fresh"

        assert agent._try_refresh_env_client_credentials() is True
        assert agent.api_key == "lc-fresh"
        assert agent._client_kwargs["api_key"] == "lc-fresh"
        agent._replace_primary_openai_client.assert_called_once_with(
            reason="env_credential_refresh"
        )

    def test_key_rotation_between_turns_is_adopted(self, env, named_custom_provider):
        agent = self._make_custom_agent(api_key="lc-old")
        env["LONGCAT_API_KEY"] = "lc-old"
        assert agent._try_refresh_env_client_credentials() is False

        env["LONGCAT_API_KEY"] = "lc-new"
        assert agent._try_refresh_env_client_credentials() is True
        assert agent.api_key == "lc-new"

    def test_unchanged_env_is_a_noop(self, env, named_custom_provider):
        agent = self._make_custom_agent(api_key="lc-old")
        env["LONGCAT_API_KEY"] = "lc-old"

        assert agent._try_refresh_env_client_credentials() is False
        agent._replace_primary_openai_client.assert_not_called()

    def test_skipped_without_key_env(self, env, named_custom_provider):
        """Inline `api_key` / pool-backed entries have no env-sourced
        credential to watch."""
        named_custom_provider.pop("key_env")
        agent = self._make_custom_agent()
        env["LONGCAT_API_KEY"] = "lc-fresh"

        assert agent._try_refresh_env_client_credentials() is False

    def test_skipped_for_unknown_custom_provider(self, env, named_custom_provider):
        agent = self._make_custom_agent()
        agent.requested_provider = "someone-else"
        env["LONGCAT_API_KEY"] = "lc-fresh"

        assert agent._try_refresh_env_client_credentials() is False


class TestOpenRouter:
    """OpenRouter is an aggregator with no PROVIDER_REGISTRY entry, so the
    registry branch never matches it — but its OPENROUTER_API_KEY is
    env-sourced exactly like a registry api-key provider. It must refresh
    mid-session so a depleted key can be swapped without a restart."""

    def _make_openrouter_agent(self, *, api_key="no-key-required"):
        return _make_agent(
            provider="openrouter",
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

    def test_key_added_mid_session_is_adopted(self, env):
        """Long-lived worker spawned before the key was written to .env — the
        first turn after the save must pick it up."""
        agent = self._make_openrouter_agent(api_key="sk-or-old")
        env["OPENROUTER_API_KEY"] = "sk-or-fresh"

        assert agent._try_refresh_env_client_credentials() is True
        assert agent.api_key == "sk-or-fresh"
        assert agent._client_kwargs["api_key"] == "sk-or-fresh"
        agent._replace_primary_openai_client.assert_called_once_with(
            reason="env_credential_refresh"
        )

    def test_key_rotation_between_turns_is_adopted(self, env):
        agent = self._make_openrouter_agent(api_key="sk-or-old")
        env["OPENROUTER_API_KEY"] = "sk-or-old"
        assert agent._try_refresh_env_client_credentials() is False

        env["OPENROUTER_API_KEY"] = "sk-or-new"
        assert agent._try_refresh_env_client_credentials() is True
        assert agent.api_key == "sk-or-new"
        assert agent._client_kwargs["api_key"] == "sk-or-new"

    def test_unchanged_env_is_a_noop(self, env):
        agent = self._make_openrouter_agent(api_key="sk-or-old")
        env["OPENROUTER_API_KEY"] = "sk-or-old"

        assert agent._try_refresh_env_client_credentials() is False
        agent._replace_primary_openai_client.assert_not_called()

    def test_skipped_when_no_key_resolves(self, env):
        agent = self._make_openrouter_agent(api_key="sk-or-old")

        assert agent._try_refresh_env_client_credentials() is False

    def test_custom_base_url_wins_over_env_edit(self, env):
        """A session pointed at a proxy/non-default OpenRouter endpoint keeps
        its base_url — the env edit only watches key rotation on the default
        route."""
        agent = self._make_openrouter_agent(api_key="sk-or-old")
        agent.base_url = "https://my-proxy.example/v1"
        env["OPENROUTER_API_KEY"] = "sk-or-new"

        # First look adopts nothing: current base isn't the default route.
        assert agent._try_refresh_env_client_credentials() is False
        assert agent.base_url == "https://my-proxy.example/v1"


class TestResolveEnvKeyBinding:
    """Unit tests for the shared provider→env credential resolver. Its whole
    job is: every env-sourced api-key credential is covered, regardless of
    provider, with no per-provider branch needed in the consumer."""

    def _resolve(self, provider, requested_provider=None):
        import agent.credential_pool as cp

        return cp.resolve_env_key_binding(provider, requested_provider)

    def test_openrouter(self):
        binding = self._resolve("openrouter")
        assert binding["key_env_vars"] == ("OPENROUTER_API_KEY",)
        assert binding["default_base_url"].rstrip("/") == "https://openrouter.ai/api/v1"

    def test_registry_api_key_provider(self):
        binding = self._resolve("openai-api")
        assert binding["key_env_vars"] == ("OPENAI_API_KEY",)
        assert binding["base_url_env_var"] == "OPENAI_BASE_URL"
        assert binding["default_base_url"].rstrip("/") == "https://api.openai.com/v1"

    def test_anthropic_three_token_override(self):
        binding = self._resolve("anthropic")
        assert binding["key_env_vars"] == (
            "ANTHROPIC_TOKEN",
            "CLAUDE_CODE_OAUTH_TOKEN",
            "ANTHROPIC_API_KEY",
        )

    def test_registry_oauth_provider_has_no_env_binding(self):
        # nous is OAuth device-code — nothing env-sourced to watch.
        assert self._resolve("nous") is None

    def test_unknown_provider_has_no_env_binding(self):
        assert self._resolve("not-a-real-provider") is None

    def test_provider_is_lowercased(self):
        assert self._resolve("OpenRouter") is not None

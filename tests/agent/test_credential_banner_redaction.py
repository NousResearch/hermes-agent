"""Credential init banners must not preview secret material (issue #60319).

``agent/agent_init.py`` prints status banners on non-quiet agent startup.
Those lines are routinely captured into orchestrator logs and transcripts,
so partial head/tail previews of tokens or API keys are an exposure surface
with no operational upside over a fixed ``[configured]`` marker.

Tests drive the real ``init_agent()`` Anthropic and OpenAI-compatible paths
with client construction mocked out, and assert on captured stdout.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import patch

from run_agent import AIAgent


def _bare_agent() -> AIAgent:
    """Minimal AIAgent instance without running ``__init__``."""
    agent = object.__new__(AIAgent)
    agent._base_url = ""
    agent._base_url_lower = ""
    agent._base_url_hostname = ""
    # Methods used during the OpenAI client path.
    agent._create_openai_client = lambda *a, **k: SimpleNamespace()
    agent._apply_user_default_headers = lambda: None
    return agent


@contextmanager
def _common_patches():
    """Patches that keep init_agent off the network and out of config IO."""
    with ExitStack() as stack:
        stack.enter_context(
            patch("agent.auxiliary_client.resolve_provider_client", return_value=(None, None))
        )
        stack.enter_context(patch("run_agent.get_tool_definitions", return_value=[]))
        stack.enter_context(
            patch(
                "agent.anthropic_adapter.build_anthropic_client",
                return_value=SimpleNamespace(),
            )
        )
        stack.enter_context(
            patch("agent.anthropic_adapter.resolve_anthropic_token", return_value="")
        )
        stack.enter_context(
            patch("agent.anthropic_adapter._is_oauth_token", return_value=False)
        )
        stack.enter_context(
            patch(
                "hermes_cli.model_normalize.normalize_model_for_provider",
                side_effect=lambda model, *a, **k: model or "test-model",
            )
        )
        stack.enter_context(
            patch("agent.credential_pool.load_pool", return_value=SimpleNamespace())
        )
        stack.enter_context(patch("hermes_cli.config.load_config", return_value={}))
        stack.enter_context(
            patch("hermes_cli.config.load_config_readonly", return_value={})
        )
        stack.enter_context(
            patch("hermes_cli.config.get_compatible_custom_providers", return_value=[])
        )
        stack.enter_context(patch("agent.iteration_budget.IterationBudget"))
        stack.enter_context(patch("hermes_cli.config.cfg_get", return_value=None))
        stack.enter_context(patch("agent.ssl_guard.verify_ca_bundle_with_fallback"))
        stack.enter_context(
            patch(
                "hermes_cli.config.apply_custom_provider_tls_to_client_kwargs",
                return_value=None,
            )
        )
        stack.enter_context(
            patch(
                "hermes_cli.config.apply_custom_provider_extra_headers_to_client_kwargs",
                return_value=None,
            )
        )
        # Callable Entra keys break the Anthropic context-length probe
        # (it assumes a string api_key). Stub both import sites so init
        # can finish after the banner has already printed.
        stack.enter_context(
            patch(
                "agent.model_metadata.get_model_context_length",
                return_value=200000,
            )
        )
        stack.enter_context(
            patch(
                "agent.context_compressor.get_model_context_length",
                return_value=200000,
            )
        )
        yield


class TestAnthropicInitBanner:
    def test_token_banner_never_leaks_secret_material(self, capsys):
        from agent.agent_init import init_agent

        secret = "sk-ant-api03-REALLYSECRETTOKENVALUE99"
        agent = _bare_agent()
        with _common_patches():
            with patch(
                "agent.azure_identity_adapter.is_token_provider", return_value=False
            ):
                init_agent(
                    agent,
                    base_url="https://api.anthropic.com",
                    api_key=secret,
                    provider="anthropic",
                    api_mode="anthropic_messages",
                    model="claude-opus-4.8",
                    skip_context_files=True,
                    skip_memory=True,
                    quiet_mode=False,
                )
        out = capsys.readouterr().out
        assert "Using token: [configured]" in out
        assert secret not in out
        assert secret[:8] not in out
        assert secret[-4:] not in out

    def test_entra_provider_prints_static_label_without_invoking(self, capsys):
        from agent.agent_init import init_agent

        called = {"n": 0}

        def provider():
            called["n"] += 1
            return "should-never-be-read"

        agent = _bare_agent()
        with _common_patches():
            with patch(
                "agent.azure_identity_adapter.is_token_provider",
                side_effect=lambda key: callable(key),
            ):
                init_agent(
                    agent,
                    base_url="https://api.anthropic.com",
                    api_key=provider,
                    provider="anthropic",
                    api_mode="anthropic_messages",
                    model="claude-opus-4.8",
                    skip_context_files=True,
                    skip_memory=True,
                    quiet_mode=False,
                )
        out = capsys.readouterr().out
        assert "Microsoft Entra ID" in out
        assert called["n"] == 0
        assert "should-never-be-read" not in out


class TestOpenAICompatInitBanner:
    def test_api_key_banner_never_leaks_secret_material(self, capsys):
        from agent.agent_init import init_agent

        secret = "sk-proj-SUPERSECRETVALUE1234567890"
        agent = _bare_agent()
        with _common_patches():
            with patch(
                "agent.azure_identity_adapter.is_token_provider", return_value=False
            ):
                init_agent(
                    agent,
                    base_url="https://openrouter.ai/api/v1",
                    api_key=secret,
                    provider="openrouter",
                    api_mode="chat_completions",
                    model="openai/gpt-4o",
                    skip_context_files=True,
                    skip_memory=True,
                    quiet_mode=False,
                )
        out = capsys.readouterr().out
        assert "Using API key: [configured]" in out
        assert secret not in out
        assert secret[:8] not in out
        assert secret[-4:] not in out

    def test_invalid_or_missing_key_still_warns(self, capsys):
        from agent.agent_init import init_agent

        agent = _bare_agent()
        with _common_patches():
            with patch(
                "agent.azure_identity_adapter.is_token_provider", return_value=False
            ):
                init_agent(
                    agent,
                    base_url="https://openrouter.ai/api/v1",
                    api_key="short",
                    provider="openrouter",
                    api_mode="chat_completions",
                    model="openai/gpt-4o",
                    skip_context_files=True,
                    skip_memory=True,
                    quiet_mode=False,
                )
        out = capsys.readouterr().out
        assert "API key appears invalid or missing" in out

    def test_dummy_key_still_warns(self, capsys):
        from agent.agent_init import init_agent

        agent = _bare_agent()
        with _common_patches():
            with patch(
                "agent.azure_identity_adapter.is_token_provider", return_value=False
            ):
                init_agent(
                    agent,
                    base_url="https://openrouter.ai/api/v1",
                    api_key="dummy-key",
                    provider="openrouter",
                    api_mode="chat_completions",
                    model="openai/gpt-4o",
                    skip_context_files=True,
                    skip_memory=True,
                    quiet_mode=False,
                )
        out = capsys.readouterr().out
        assert "API key appears invalid or missing" in out

    def test_entra_provider_prints_static_label(self, capsys):
        from agent.agent_init import init_agent

        called = {"n": 0}

        def provider():
            called["n"] += 1
            return "should-never-be-read"

        agent = _bare_agent()
        with _common_patches():
            with patch(
                "agent.azure_identity_adapter.is_token_provider",
                side_effect=lambda key: callable(key),
            ):
                init_agent(
                    agent,
                    base_url="https://openrouter.ai/api/v1",
                    api_key=provider,
                    provider="openrouter",
                    api_mode="chat_completions",
                    model="openai/gpt-4o",
                    skip_context_files=True,
                    skip_memory=True,
                    quiet_mode=False,
                )
        out = capsys.readouterr().out
        assert "Microsoft Entra ID" in out
        assert called["n"] == 0
        assert "should-never-be-read" not in out

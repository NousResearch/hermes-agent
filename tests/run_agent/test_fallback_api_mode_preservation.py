"""Fallback activation must preserve the Anthropic wire signal (PR #79787).

Three behaviors salvaged from PR #79787:

1. An explicit ``api_mode`` on the fallback entry is honored — and always
   wins, including an explicit ``chat_completions`` that would otherwise be
   overridden by codex_responses / bedrock re-detection.
2. ``fb_api_mode`` is detected from the ORIGINAL ``base_url`` hint before
   ``resolve_provider_client`` / ``_to_openai_base_url`` can rewrite a
   dual-surface ``/anthropic`` base to ``/v1``.
3. ``api_mode`` is passed into ``resolve_provider_client`` at the fallback
   call site so the resolver keeps the Anthropic wire for custom bases.
"""

from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def _make_agent(fallback_model=None):
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallback_model,
        )
        agent.client = MagicMock()
        return agent


def _mock_client(base_url="https://openrouter.ai/api/v1", api_key="fb-key"):
    mock = MagicMock()
    mock.base_url = base_url
    mock.api_key = api_key
    return mock


def _activate(agent, resolved_base_url, resolved_model, build_anthropic=None):
    """Run _try_activate_fallback with the standard mock stack.

    Returns the mock for resolve_provider_client so callers can assert on
    the api_mode kwarg passed at the fallback call site.
    """
    patches = [
        patch(
            "agent.chat_completion_helpers._fallback_entry_unavailable_without_network",
            return_value=None,
        ),
        patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(
                _mock_client(base_url=resolved_base_url),
                resolved_model,
            ),
        ),
        patch(
            "hermes_cli.model_normalize.normalize_model_for_provider",
            side_effect=lambda m, p: m,
        ),
        patch(
            "agent.anthropic_adapter.build_anthropic_client",
            side_effect=build_anthropic
            or (lambda api_key, base_url, timeout=None, **kw: MagicMock()),
        ),
    ]
    with patches[0], patches[1] as mock_rpc, patches[2], patches[3]:
        assert agent._try_activate_fallback() is True
    return mock_rpc


class TestExplicitApiModeHonored:
    def test_explicit_anthropic_messages_honored(self):
        fbs = [{
            "provider": "custom",
            "model": "claude-opus-4-6",
            "base_url": "https://gateway.example.com/v1",
            "api_key": "k",
            "api_mode": "anthropic_messages",
        }]
        agent = _make_agent(fallback_model=fbs)
        mock_rpc = _activate(agent, "https://gateway.example.com/v1", "claude-opus-4-6")
        assert agent.api_mode == "anthropic_messages"
        # Behavior (3): api_mode forwarded to the resolver.
        assert mock_rpc.call_args.kwargs["api_mode"] == "anthropic_messages"

    def test_explicit_chat_completions_not_overridden_by_redetection(self):
        """An explicit chat_completions must survive re-detection.

        api.openai.com would normally re-detect to codex_responses via
        _is_direct_openai_url; the explicit config field must win.
        """
        fbs = [{
            "provider": "custom",
            "model": "gpt-5.2",
            "base_url": "https://api.openai.com/v1",
            "api_key": "k",
            "api_mode": "chat_completions",
        }]
        agent = _make_agent(fallback_model=fbs)
        _activate(agent, "https://api.openai.com/v1", "gpt-5.2")
        assert agent.api_mode == "chat_completions"

    def test_explicit_chat_completions_not_overridden_by_bedrock_redetection(self):
        fbs = [{
            "provider": "bedrock",
            "model": "anthropic.claude-3-5-sonnet",
            "api_key": "k",
            "api_mode": "chat_completions",
        }]
        agent = _make_agent(fallback_model=fbs)
        _activate(
            agent,
            "https://bedrock-runtime.us-east-1.amazonaws.com",
            "anthropic.claude-3-5-sonnet",
        )
        assert agent.api_mode == "chat_completions"


class TestOriginalUrlDetection:
    def test_anthropic_suffix_hint_survives_rewrite(self):
        """Dual-surface /anthropic base rewritten to /v1 by the resolver:
        detection must run on the ORIGINAL hint, not the rewritten client URL.
        """
        fbs = [{
            "provider": "custom",
            "model": "MiniMax-M2.5",
            "base_url": "https://api.minimax.io/anthropic",
            "api_key": "k",
        }]
        agent = _make_agent(fallback_model=fbs)
        mock_rpc = _activate(agent, "https://api.minimax.io/v1", "MiniMax-M2.5")
        assert agent.api_mode == "anthropic_messages"
        assert mock_rpc.call_args.kwargs["api_mode"] == "anthropic_messages"

    def test_anthropic_host_hint_detected(self):
        fbs = [{
            "provider": "custom",
            "model": "claude-opus-4-6",
            "base_url": "https://api.anthropic.com",
            "api_key": "k",
        }]
        agent = _make_agent(fallback_model=fbs)
        _activate(agent, "https://api.anthropic.com", "claude-opus-4-6")
        assert agent.api_mode == "anthropic_messages"

    def test_provider_anthropic_without_base_url(self):
        """provider: anthropic with no explicit base_url must still resolve
        to anthropic_messages (follow-up commit 38303343 in PR #79787)."""
        fbs = [{"provider": "anthropic", "model": "claude-opus-4-6", "api_key": "k"}]
        agent = _make_agent(fallback_model=fbs)
        mock_rpc = _activate(agent, "https://api.anthropic.com", "claude-opus-4-6")
        assert agent.api_mode == "anthropic_messages"
        assert mock_rpc.call_args.kwargs["api_mode"] == "anthropic_messages"


class TestPlainFallbackUnchanged:
    def test_plain_openrouter_fallback_stays_chat_completions(self):
        fbs = [{
            "provider": "openrouter",
            "model": "z-ai/glm-5",
            "api_key": "k",
        }]
        agent = _make_agent(fallback_model=fbs)
        mock_rpc = _activate(agent, "https://openrouter.ai/api/v1", "z-ai/glm-5")
        assert agent.api_mode == "chat_completions"
        assert mock_rpc.call_args.kwargs["api_mode"] == "chat_completions"

    def test_post_resolve_anthropic_host_still_detected(self):
        """Named custom providers resolve api.anthropic.com from config, not
        the fallback entry — post-resolve detection must still catch it
        (#32243, #49247)."""
        fbs = [{"provider": "cron-anthropic", "model": "claude-opus-4-6", "api_key": "k"}]
        agent = _make_agent(fallback_model=fbs)
        _activate(agent, "https://api.anthropic.com/v1", "claude-opus-4-6")
        assert agent.api_mode == "anthropic_messages"

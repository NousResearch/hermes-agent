"""Tests for agent.adaptive_reasoning — per-turn reasoning effort classification."""

from unittest.mock import MagicMock, patch

import pytest

from agent.adaptive_reasoning import (
    apply_adaptive_reasoning_intent,
    classify_reasoning_effort,
)


def _mock_response(content: str):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"choices": [{"message": {"content": content}}]}
    return resp


def _mock_anthropic_response(text: str):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"content": [{"type": "text", "text": text}]}
    return resp


def _mock_responses_api(text: str):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"output_text": text}
    return resp


class TestClassifyReasoningEffort:
    def test_returns_valid_effort_on_clean_response(self):
        with patch("agent.adaptive_reasoning.requests.post", return_value=_mock_response("high")):
            effort = classify_reasoning_effort(
                "Debug this segfault", base_url="http://127.0.0.1:8090/v1", model="local-model"
            )
        assert effort == "high"

    def test_parses_effort_out_of_extra_text(self):
        with patch(
            "agent.adaptive_reasoning.requests.post",
            return_value=_mock_response("I'd say high."),
        ):
            effort = classify_reasoning_effort(
                "Hello", base_url="http://127.0.0.1:8090/v1", model="local-model"
            )
        assert effort == "high"

    def test_falls_back_to_medium_on_unparseable_response(self):
        with patch(
            "agent.adaptive_reasoning.requests.post",
            return_value=_mock_response("banana"),
        ):
            effort = classify_reasoning_effort(
                "Hello", base_url="http://127.0.0.1:8090/v1", model="local-model"
            )
        assert effort == "medium"

    def test_falls_back_to_medium_on_empty_response(self):
        with patch(
            "agent.adaptive_reasoning.requests.post",
            return_value=_mock_response(""),
        ):
            effort = classify_reasoning_effort(
                "Hello", base_url="http://127.0.0.1:8090/v1", model="local-model"
            )
        assert effort == "medium"

    def test_falls_back_to_medium_on_request_exception(self):
        with patch(
            "agent.adaptive_reasoning.requests.post",
            side_effect=RuntimeError("connection refused"),
        ):
            effort = classify_reasoning_effort(
                "Hello", base_url="http://127.0.0.1:8090/v1", model="local-model"
            )
        assert effort == "medium"

    def test_falls_back_to_medium_when_no_base_url(self):
        effort = classify_reasoning_effort("Hello", base_url=None)
        assert effort == "medium"

    def test_falls_back_to_medium_on_empty_message(self):
        effort = classify_reasoning_effort("   ", base_url="http://127.0.0.1:8090/v1")
        assert effort == "medium"

    def test_disables_thinking_on_the_classification_call(self):
        mock_post = MagicMock(return_value=_mock_response("minimal"))
        with patch("agent.adaptive_reasoning.requests.post", mock_post):
            classify_reasoning_effort(
                "Hello", base_url="http://127.0.0.1:8090/v1", model="local-model"
            )
        payload = mock_post.call_args.kwargs["json"]
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}

    def test_sends_bearer_auth_when_api_key_provided(self):
        mock_post = MagicMock(return_value=_mock_response("minimal"))
        with patch("agent.adaptive_reasoning.requests.post", mock_post):
            classify_reasoning_effort(
                "Hello",
                base_url="http://127.0.0.1:8090/v1",
                api_key="sk-real-key",
            )
        headers = mock_post.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer sk-real-key"

    @pytest.mark.parametrize(
        "placeholder", ["not-needed", "none", "None", "NOT-NEEDED", "no-key", "sk-noauth", "local"]
    )
    def test_omits_auth_header_for_placeholder_key(self, placeholder):
        mock_post = MagicMock(return_value=_mock_response("minimal"))
        with patch("agent.adaptive_reasoning.requests.post", mock_post):
            classify_reasoning_effort(
                "Hello",
                base_url="http://127.0.0.1:8090/v1",
                api_key=placeholder,
            )
        headers = mock_post.call_args.kwargs["headers"]
        assert "Authorization" not in headers


class TestModeAwareTransport:
    """Classification must speak the active api_mode, not always /chat/completions."""

    def test_chat_completions_default(self):
        mock_post = MagicMock(return_value=_mock_response("low"))
        with patch("agent.adaptive_reasoning.requests.post", mock_post):
            effort = classify_reasoning_effort(
                "Hello",
                base_url="http://127.0.0.1:8090/v1",
                api_mode="chat_completions",
            )
        assert effort == "low"
        assert mock_post.call_args.args[0].endswith("/chat/completions")

    def test_anthropic_messages_transport(self):
        mock_post = MagicMock(return_value=_mock_anthropic_response("high"))
        with patch("agent.adaptive_reasoning.requests.post", mock_post):
            effort = classify_reasoning_effort(
                "Debug carefully",
                base_url="https://api.anthropic.com",
                api_key="sk-ant-test",
                model="claude-sonnet-4-20250514",
                api_mode="anthropic_messages",
                provider="anthropic",
            )
        assert effort == "high"
        assert mock_post.call_args.args[0] == "https://api.anthropic.com/v1/messages"
        headers = mock_post.call_args.kwargs["headers"]
        assert headers["x-api-key"] == "sk-ant-test"
        assert headers["anthropic-version"] == "2023-06-01"
        payload = mock_post.call_args.kwargs["json"]
        assert payload["system"]
        assert payload["messages"][0]["role"] == "user"
        assert "chat_template_kwargs" not in payload

    def test_anthropic_messages_does_not_double_v1(self):
        mock_post = MagicMock(return_value=_mock_anthropic_response("medium"))
        with patch("agent.adaptive_reasoning.requests.post", mock_post):
            classify_reasoning_effort(
                "Hello",
                base_url="https://api.anthropic.com/v1",
                api_key="sk-ant-test",
                api_mode="anthropic_messages",
                provider="anthropic",
            )
        assert mock_post.call_args.args[0] == "https://api.anthropic.com/v1/messages"

    def test_codex_responses_transport(self):
        mock_post = MagicMock(return_value=_mock_responses_api("xhigh"))
        with patch("agent.adaptive_reasoning.requests.post", mock_post), patch(
            "agent.adaptive_reasoning._resolve_oauth_credentials", return_value=None
        ):
            effort = classify_reasoning_effort(
                "Prove this theorem",
                base_url="https://chatgpt.com/backend-api/codex",
                api_key="codex-token",
                model="gpt-5.5",
                api_mode="codex_responses",
                provider="openai-codex",
            )
        assert effort == "xhigh"
        assert mock_post.call_args.args[0].endswith("/responses")
        payload = mock_post.call_args.kwargs["json"]
        assert payload["instructions"]
        assert payload["input"] == "Prove this theorem"
        assert payload["max_output_tokens"] == 20

    def test_unsupported_mode_falls_back_to_medium(self):
        with patch("agent.adaptive_reasoning.requests.post") as mock_post:
            effort = classify_reasoning_effort(
                "Hello",
                base_url="https://bedrock.amazonaws.com",
                api_mode="bedrock_converse",
                provider="bedrock",
            )
        assert effort == "medium"
        mock_post.assert_not_called()


class TestOAuthProviderAutoDetection:
    """For OAuth-gated providers (xai-oauth, openai-codex, nous), the caller's plain
    ``api_key`` kwarg is never the live credential — it's static/None, while the real
    bearer token rotates and lives in Hermes's credential pool / auth store.
    """

    def test_resolves_live_credentials_for_xai_oauth(self):
        mock_post = MagicMock(return_value=_mock_response("high"))
        with patch("agent.adaptive_reasoning.requests.post", mock_post), patch(
            "agent.adaptive_reasoning._resolve_oauth_credentials",
            return_value=("live-token-abc", "https://api.x.ai/v1"),
        ) as mock_resolve:
            effort = classify_reasoning_effort(
                "Debug this segfault",
                provider="xai-oauth",
                model="grok-4.5",
                base_url="stale-or-placeholder-base-url",
                api_key=None,
            )
        mock_resolve.assert_called_once_with("xai-oauth")
        assert effort == "high"
        # The resolved (not the passed-in) base_url and api_key must be used.
        assert mock_post.call_args.args[0] == "https://api.x.ai/v1/chat/completions"
        assert mock_post.call_args.kwargs["headers"]["Authorization"] == "Bearer live-token-abc"

    def test_resolves_live_credentials_for_openai_codex(self):
        mock_post = MagicMock(return_value=_mock_responses_api("medium"))
        with patch("agent.adaptive_reasoning.requests.post", mock_post), patch(
            "agent.adaptive_reasoning._resolve_oauth_credentials",
            return_value=("codex-live", "https://chatgpt.com/backend-api/codex"),
        ) as mock_resolve:
            effort = classify_reasoning_effort(
                "Hi",
                provider="openai-codex",
                model="gpt-5.5",
                base_url="stale",
                api_key=None,
                api_mode="codex_responses",
            )
        mock_resolve.assert_called_once_with("openai-codex")
        assert effort == "medium"
        assert mock_post.call_args.kwargs["headers"]["Authorization"] == "Bearer codex-live"

    def test_resolves_live_credentials_for_nous(self):
        mock_post = MagicMock(return_value=_mock_response("low"))
        with patch("agent.adaptive_reasoning.requests.post", mock_post), patch(
            "agent.adaptive_reasoning._resolve_oauth_credentials",
            return_value=("nous-live", "https://inference.nousresearch.com/v1"),
        ) as mock_resolve:
            effort = classify_reasoning_effort(
                "Hi",
                provider="nous",
                model="hermes",
                base_url="stale",
                api_key=None,
            )
        mock_resolve.assert_called_once_with("nous")
        assert effort == "low"
        assert mock_post.call_args.kwargs["headers"]["Authorization"] == "Bearer nous-live"

    def test_falls_back_to_passed_in_credentials_when_oauth_resolution_fails(self):
        mock_post = MagicMock(return_value=_mock_response("minimal"))
        with patch("agent.adaptive_reasoning.requests.post", mock_post), patch(
            "agent.adaptive_reasoning._resolve_oauth_credentials", return_value=None
        ):
            effort = classify_reasoning_effort(
                "Hi",
                provider="xai-oauth",
                model="grok-4.5",
                base_url="https://api.x.ai/v1",
                api_key="stale-static-key",
            )
        # Resolution failed, but the call still goes through with whatever was
        # passed in rather than hard-failing differently from a non-OAuth provider.
        assert effort == "minimal"
        assert mock_post.call_args.kwargs["headers"]["Authorization"] == "Bearer stale-static-key"

    def test_does_not_attempt_oauth_resolution_for_non_oauth_providers(self):
        mock_post = MagicMock(return_value=_mock_response("low"))
        with patch("agent.adaptive_reasoning.requests.post", mock_post), patch(
            "agent.adaptive_reasoning._resolve_oauth_credentials"
        ) as mock_resolve:
            classify_reasoning_effort(
                "Hello",
                provider="openrouter",
                base_url="https://openrouter.ai/api/v1",
                api_key="sk-real-key",
            )
        mock_resolve.assert_not_called()

    def test_oauth_resolve_helper_dispatches_codex_and_nous(self):
        """_resolve_oauth_credentials itself must attempt codex/nous helpers."""
        from agent.adaptive_reasoning import _resolve_oauth_credentials

        with patch(
            "hermes_cli.auth.resolve_codex_runtime_credentials",
            return_value={"api_key": "c", "base_url": "https://codex.example"},
        ):
            assert _resolve_oauth_credentials("openai-codex") == ("c", "https://codex.example")

        with patch(
            "hermes_cli.auth.resolve_nous_runtime_credentials",
            return_value={"api_key": "n", "base_url": "https://nous.example/v1"},
        ):
            assert _resolve_oauth_credentials("nous") == ("n", "https://nous.example/v1")


class TestCloudProviderChatTemplateGating:
    """chat_template_kwargs.enable_thinking is a llama.cpp-specific extension —
    meaningless (and an unrecognized-field risk) against a real hosted API, so it
    should only be sent for local-style deployments.
    """

    @pytest.mark.parametrize(
        "provider,base_url",
        [
            ("xai-oauth", "https://api.x.ai/v1"),
            ("xai", "https://api.x.ai/v1"),
            ("openai", "https://api.openai.com/v1"),
            ("anthropic", "https://api.anthropic.com/v1"),
            (None, "https://api.x.ai/v1"),  # no provider hint, but a clearly-remote host
        ],
    )
    def test_omits_chat_template_kwargs_for_cloud_providers(self, provider, base_url):
        mock_post = MagicMock(return_value=_mock_response("minimal"))
        with patch("agent.adaptive_reasoning.requests.post", mock_post), patch(
            "agent.adaptive_reasoning._resolve_oauth_credentials", return_value=None
        ):
            classify_reasoning_effort(
                "Hello", provider=provider, base_url=base_url, api_key="sk-real-key"
            )
        payload = mock_post.call_args.kwargs["json"]
        assert "chat_template_kwargs" not in payload

    @pytest.mark.parametrize(
        "base_url", ["http://127.0.0.1:8090/v1", "http://localhost:8090/v1", "http://0.0.0.0:8090/v1"]
    )
    def test_includes_chat_template_kwargs_for_local_hosts(self, base_url):
        mock_post = MagicMock(return_value=_mock_response("minimal"))
        with patch("agent.adaptive_reasoning.requests.post", mock_post):
            classify_reasoning_effort("Hello", base_url=base_url)
        payload = mock_post.call_args.kwargs["json"]
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}


class TestAdaptiveIntentHelper:
    def test_apply_intent_enables_and_clears_cache(self):
        agent = MagicMock()
        agent._adaptive_reasoning_cache = {"message": "hi", "effort": "high"}
        apply_adaptive_reasoning_intent(agent, {"enabled": True, "effort": "adaptive"})
        assert agent._adaptive_reasoning is True
        assert agent._adaptive_reasoning_cache is None

    def test_apply_intent_disables_for_fixed_effort(self):
        agent = MagicMock()
        agent._adaptive_reasoning = True
        agent._adaptive_reasoning_cache = {"message": "hi", "effort": "high"}
        apply_adaptive_reasoning_intent(agent, {"enabled": True, "effort": "high"})
        assert agent._adaptive_reasoning is False
        assert agent._adaptive_reasoning_cache is None

    def test_apply_intent_disables_for_none(self):
        agent = MagicMock()
        agent._adaptive_reasoning = True
        apply_adaptive_reasoning_intent(agent, {"enabled": False})
        assert agent._adaptive_reasoning is False


class TestTurnCacheOnAgent:
    """_resolve_adaptive_reasoning must not re-fire classification mid-turn."""

    def test_cache_reuses_effort_for_same_user_message(self):
        from run_agent import AIAgent

        agent = MagicMock(spec=AIAgent)
        # Bind the real method
        agent._resolve_adaptive_reasoning = AIAgent._resolve_adaptive_reasoning.__get__(
            agent, AIAgent
        )
        agent.provider = "openrouter"
        agent.model = "test"
        agent.base_url = "https://openrouter.ai/api/v1"
        agent.api_key = "sk-test"
        agent.api_mode = "chat_completions"
        agent._adaptive_reasoning_cache = None

        with patch(
            "agent.adaptive_reasoning.classify_reasoning_effort", return_value="high"
        ) as mock_cls:
            agent._resolve_adaptive_reasoning(
                [{"role": "user", "content": "Debug this properly"}]
            )
            agent._resolve_adaptive_reasoning(
                [
                    {"role": "user", "content": "Debug this properly"},
                    {"role": "assistant", "content": "..."},
                    {"role": "tool", "content": "ok"},
                ]
            )
        assert mock_cls.call_count == 1
        assert agent.reasoning_config == {"enabled": True, "effort": "high"}
        assert agent._adaptive_reasoning_cache["effort"] == "high"

    def test_cache_misses_on_new_user_message(self):
        from run_agent import AIAgent

        agent = MagicMock(spec=AIAgent)
        agent._resolve_adaptive_reasoning = AIAgent._resolve_adaptive_reasoning.__get__(
            agent, AIAgent
        )
        agent.provider = "openrouter"
        agent.model = "test"
        agent.base_url = "https://openrouter.ai/api/v1"
        agent.api_key = "sk-test"
        agent.api_mode = "chat_completions"
        agent._adaptive_reasoning_cache = None

        with patch(
            "agent.adaptive_reasoning.classify_reasoning_effort",
            side_effect=["low", "xhigh"],
        ) as mock_cls:
            agent._resolve_adaptive_reasoning([{"role": "user", "content": "hi"}])
            agent._resolve_adaptive_reasoning(
                [{"role": "user", "content": "prove Fermat's last theorem"}]
            )
        assert mock_cls.call_count == 2
        assert agent.reasoning_config == {"enabled": True, "effort": "xhigh"}

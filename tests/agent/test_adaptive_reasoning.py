"""Tests for agent.adaptive_reasoning — per-turn reasoning effort classification."""

from unittest.mock import MagicMock, patch

import pytest

from agent.adaptive_reasoning import classify_reasoning_effort


def _mock_response(content: str):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"choices": [{"message": {"content": content}}]}
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


class TestOAuthProviderAutoDetection:
    """For OAuth-gated providers (xai-oauth, openai-codex, nous), the caller's plain
    ``api_key`` kwarg is never the live credential — it's static/None, while the real
    bearer token rotates and lives in Hermes's credential pool / auth store. Sending
    no/stale auth to these providers doesn't fail fast; several hang until the
    client-side timeout, which is what silently made every adaptive-reasoning
    classification against xai-oauth eat the full 25s and fall back to 'medium',
    regardless of actual message content, for as long as it went unnoticed in
    production."""

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


class TestCloudProviderChatTemplateGating:
    """chat_template_kwargs.enable_thinking is a llama.cpp-specific extension —
    meaningless (and an unrecognized-field risk) against a real hosted API, so it
    should only be sent for local-style deployments."""

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

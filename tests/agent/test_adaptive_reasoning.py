"""Tests for agent.adaptive_reasoning — per-turn reasoning effort classification."""

from unittest.mock import MagicMock, patch

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

    def test_omits_auth_header_for_placeholder_key(self):
        mock_post = MagicMock(return_value=_mock_response("minimal"))
        with patch("agent.adaptive_reasoning.requests.post", mock_post):
            classify_reasoning_effort(
                "Hello",
                base_url="http://127.0.0.1:8090/v1",
                api_key="not-needed",
            )
        headers = mock_post.call_args.kwargs["headers"]
        assert "Authorization" not in headers

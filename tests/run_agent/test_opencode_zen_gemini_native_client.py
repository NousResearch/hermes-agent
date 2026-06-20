"""Regression tests for OpenCode Zen Gemini-native routing."""

from unittest.mock import patch

from run_agent import AIAgent


ZEN_TEST_KEY = "zen-test-key"


def _make_agent(model="gemini-3.5-flash"):
    return AIAgent(
        api_key=ZEN_TEST_KEY,
        base_url="https://opencode.ai/zen/v1",
        provider="opencode-zen",
        model=model,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )


@patch("run_agent.OpenAI")
def test_opencode_zen_gemini_model_uses_native_gemini_client(mock_openai):
    from agent.gemini_native_adapter import GeminiNativeClient

    agent = _make_agent()
    mock_openai.reset_mock()

    client = agent._create_openai_client(
        {"api_key": ZEN_TEST_KEY, "base_url": "https://opencode.ai/zen/v1"},
        reason="test",
        shared=False,
    )
    try:
        assert isinstance(client, GeminiNativeClient)
        assert client.base_url == "https://opencode.ai/zen/v1"
        assert client._headers()["x-goog-api-key"] == ZEN_TEST_KEY
        mock_openai.assert_not_called()
    finally:
        client.close()


@patch("run_agent.OpenAI")
def test_opencode_zen_non_gemini_model_keeps_openai_transport(mock_openai):
    agent = _make_agent(model="glm-5")
    mock_openai.reset_mock()

    client = agent._create_openai_client(
        {"api_key": ZEN_TEST_KEY, "base_url": "https://opencode.ai/zen/v1"},
        reason="test",
        shared=False,
    )

    assert client is mock_openai.return_value
    mock_openai.assert_called_once()

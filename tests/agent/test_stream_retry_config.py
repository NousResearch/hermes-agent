"""The mid-stream retry count is configurable and governs both streaming paths.

The count used to be split: chat-completions read ``HERMES_STREAM_RETRIES``, while the Codex
Responses path hardcoded ``1``. These tests pin the resolution order (config, then env, then
default) and drive the real retry loops on both paths, so a configured value is proven to reach
them rather than a formula being re-derived in the test.
"""
from unittest.mock import MagicMock, patch

import pytest

from agent.agent_init import _apply_agent_section


def _agent(api_mode="chat_completions"):
    from run_agent import AIAgent

    agent = AIAgent(
        api_key="test-key",
        base_url="https://example.com/v1",
        model="test/model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent.api_mode = api_mode
    agent._interrupt_requested = False
    return agent


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("HERMES_STREAM_RETRIES", raising=False)


class TestResolutionOrder:
    """``agent.max_stream_retries`` resolves through the real init path."""

    def _resolve(self, agent_section):
        agent = _agent()
        _apply_agent_section(agent, {"agent": agent_section})
        return agent._max_stream_retries

    def test_default_when_unset_anywhere(self):
        assert self._resolve({}) == 2

    def test_config_value_wins(self):
        assert self._resolve({"max_stream_retries": 5}) == 5

    def test_zero_is_allowed_and_means_no_reconnect(self):
        assert self._resolve({"max_stream_retries": 0}) == 0

    def test_env_is_the_fallback_when_config_is_absent(self, monkeypatch):
        monkeypatch.setenv("HERMES_STREAM_RETRIES", "3")
        assert self._resolve({}) == 3

    def test_config_beats_env(self, monkeypatch):
        monkeypatch.setenv("HERMES_STREAM_RETRIES", "9")
        assert self._resolve({"max_stream_retries": 4}) == 4

    def test_negative_is_clamped_to_zero(self):
        assert self._resolve({"max_stream_retries": -3}) == 0

    def test_unparseable_config_falls_back_to_default(self):
        assert self._resolve({"max_stream_retries": "not-a-number"}) == 2


class TestCodexResponsesPath:
    """``run_codex_stream`` honours the configured count instead of hardcoding 1."""

    def _attempts_for(self, configured, exc_type):
        agent = _agent(api_mode="codex_responses")
        agent._max_stream_retries = configured
        calls = {"n": 0}

        def _create(**kwargs):
            calls["n"] += 1
            raise exc_type("peer closed connection")

        client = MagicMock()
        client.responses.create.side_effect = _create
        with pytest.raises(exc_type):
            agent._run_codex_stream({"model": "test/model"}, client=client)
        return calls["n"]

    def test_configured_count_drives_the_real_loop(self):
        import httpx

        attempts = self._attempts_for(3, httpx.RemoteProtocolError)
        assert attempts == 4, "1 initial attempt + 3 configured reconnects"

    def test_zero_configured_means_a_single_attempt(self):
        import httpx

        attempts = self._attempts_for(0, httpx.RemoteProtocolError)
        assert attempts == 1

    def test_default_is_not_the_old_hardcoded_one(self):
        import httpx

        agent = _agent(api_mode="codex_responses")
        assert agent._max_stream_retries == 2

        calls = {"n": 0}

        def _create(**kwargs):
            calls["n"] += 1
            raise httpx.RemoteProtocolError("peer closed connection")

        client = MagicMock()
        client.responses.create.side_effect = _create
        with pytest.raises(httpx.RemoteProtocolError):
            agent._run_codex_stream({"model": "test/model"}, client=client)

        assert calls["n"] == 3, "default 2 reconnects, not the previous hardcoded 1"


class TestChatCompletionsPath:
    """``_interruptible_streaming_api_call`` reads the resolved attribute, not the env var."""

    @patch("run_agent.AIAgent._close_request_openai_client")
    @patch("run_agent.AIAgent._create_request_openai_client")
    def test_configured_count_drives_the_real_loop(self, mock_create, mock_close):
        import httpx

        agent = _agent(api_mode="chat_completions")
        agent._max_stream_retries = 3
        calls = {"n": 0}

        def _create(**kwargs):
            calls["n"] += 1
            raise httpx.ConnectError("transient failure")

        client = MagicMock()
        client.chat.completions.create.side_effect = _create
        mock_create.return_value = client

        with pytest.raises(httpx.ConnectError):
            agent._interruptible_streaming_api_call({})

        assert calls["n"] == 4, "1 initial attempt + 3 configured reconnects"

    @patch("run_agent.AIAgent._close_request_openai_client")
    @patch("run_agent.AIAgent._create_request_openai_client")
    def test_config_attribute_beats_the_env_var(self, mock_create, mock_close, monkeypatch):
        import httpx

        monkeypatch.setenv("HERMES_STREAM_RETRIES", "0")
        agent = _agent(api_mode="chat_completions")
        agent._max_stream_retries = 2
        calls = {"n": 0}

        def _create(**kwargs):
            calls["n"] += 1
            raise httpx.ConnectError("transient failure")

        client = MagicMock()
        client.chat.completions.create.side_effect = _create
        mock_create.return_value = client

        with pytest.raises(httpx.ConnectError):
            agent._interruptible_streaming_api_call({})

        assert calls["n"] == 3, "the resolved attribute wins over the env var"

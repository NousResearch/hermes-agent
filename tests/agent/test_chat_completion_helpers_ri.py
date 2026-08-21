"""Tests for RI llm-pipeline dispatch gating in chat completion helpers."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.chat_completion_helpers import (
    _dispatch_nonstreaming_api_request,
    interruptible_streaming_api_call,
)
from agent.transports.ri_llm import _should_use_ri_pipeline


def test_dispatch_nonstreaming_uses_ri_when_enabled():
    agent = SimpleNamespace(api_mode="chat_completions", provider="ollama-launch")
    api_kwargs = {"model": "llama", "messages": [{"role": "user", "content": "hi"}]}
    response = SimpleNamespace(id="ri-response")

    with patch(
        "agent.chat_completion_helpers._should_use_ri_pipeline", return_value=True
    ) as can_use_ri, patch(
        "agent.chat_completion_helpers.ri_pipeline_chat_completion",
        return_value=response,
    ) as ri_call:
        result = _dispatch_nonstreaming_api_request(
            agent, api_kwargs, make_client=MagicMock()
        )

    assert result is response
    can_use_ri.assert_called_once_with(agent)
    ri_call.assert_called_once_with(agent, api_kwargs)


def test_dispatch_nonstreaming_reverts_to_openai_when_ri_disabled():
    agent = SimpleNamespace(api_mode="chat_completions", provider="openrouter")
    api_kwargs = {"model": "llama", "messages": [{"role": "user", "content": "hi"}]}
    response = SimpleNamespace(id="openai-response")
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = response

    with patch("agent.chat_completion_helpers._should_use_ri_pipeline", return_value=False) as can_use_ri, patch(
        "agent.chat_completion_helpers.ri_pipeline_chat_completion",
    ) as ri_call:
        result = _dispatch_nonstreaming_api_request(
            agent, api_kwargs, make_client=lambda *_: mock_client
        )

    assert result is response
    can_use_ri.assert_called_once_with(agent)
    ri_call.assert_not_called()
    mock_client.chat.completions.create.assert_called_once_with(**api_kwargs)


def test_dispatch_nonstreaming_does_not_use_ri_for_streamed_requests():
    agent = SimpleNamespace(api_mode="chat_completions", provider="ollama-launch")
    api_kwargs = {
        "model": "llama",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }
    response = SimpleNamespace(id="openai-response")
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = response

    with patch("agent.chat_completion_helpers._should_use_ri_pipeline", return_value=True), patch(
        "agent.chat_completion_helpers.ri_pipeline_chat_completion",
    ) as ri_call:
        result = _dispatch_nonstreaming_api_request(
            agent, api_kwargs, make_client=lambda *_: mock_client
        )

    assert result is response
    ri_call.assert_not_called()
    mock_client.chat.completions.create.assert_called_once_with(**api_kwargs)


def test_interruptible_streaming_chat_completions_with_ri_uses_nonstreaming_call():
    agent = SimpleNamespace(
        api_mode="chat_completions",
        provider="ollama-launch",
        _interrupt_requested=False,
    )
    agent._interruptible_api_call = MagicMock(return_value=SimpleNamespace(id="streamed-via-nonstream"))

    with patch("agent.chat_completion_helpers._should_use_ri_pipeline", return_value=True):
        result = interruptible_streaming_api_call(
            agent,
            {"model": "llama", "messages": [{"role": "user", "content": "hi"}], "stream": True},
            on_first_delta=lambda: None,
        )

    assert result.id == "streamed-via-nonstream"
    call_kwargs = agent._interruptible_api_call.call_args.args[0]
    assert call_kwargs["stream"] is False


def test_interruptible_streaming_chat_completions_with_ri_for_cron_routes_inline_nonstreaming():
    agent = SimpleNamespace(
        api_mode="chat_completions",
        provider="ollama-launch",
        platform="cron",
        _interrupt_requested=False,
    )
    response = SimpleNamespace(id="cron-nonstream")
    agent._interruptible_api_call = MagicMock(return_value=response)

    with patch("agent.chat_completion_helpers._should_use_ri_pipeline", return_value=True):
        out = interruptible_streaming_api_call(
            agent,
            {"model": "llama", "messages": [{"role": "user", "content": "hi"}], "stream": True},
            on_first_delta=lambda: None,
        )

    assert out.id == "cron-nonstream"
    call_kwargs = agent._interruptible_api_call.call_args.args[0]
    assert call_kwargs["stream"] is False


def test_interruptible_streaming_codex_path_not_intercepted_by_ri_gate():
    agent = SimpleNamespace(
        api_mode="codex_responses",
        provider="openai",
        _interrupt_requested=False,
    )
    response = SimpleNamespace(id="codex-response")
    agent._interruptible_api_call = MagicMock(return_value=response)

    with patch("agent.chat_completion_helpers._should_use_ri_pipeline", return_value=True):
        out = interruptible_streaming_api_call(
            agent,
            {"model": "gpt-4", "input": "x", "stream": True},
            on_first_delta=lambda: None,
        )

    assert out is response
    assert agent._interruptible_api_call.call_args.args[0]["stream"] is True


def test_should_use_ri_pipeline_defaults_to_native_plus_config_path(monkeypatch):
    """Use RiPipeline when native is available and provider filters allow it."""
    agent = SimpleNamespace(
        provider="openrouter",
        _ri_pipeline_enabled=True,
        _ri_pipeline_providers=[],
    )
    with patch("agent.transports.ri_llm._NATIVE_AVAILABLE", True):
        with monkeypatch.context() as cm:
            cm.delenv("HERMES_RI_PIPELINE", raising=False)
            cm.delenv("HERMES_RI_PIPELINE_PROVIDERS", raising=False)
            assert _should_use_ri_pipeline(agent) is True


def test_should_use_ri_pipeline_honors_agent_disable_config(monkeypatch):
    agent = SimpleNamespace(
        provider="ollama-launch",
        _ri_pipeline_enabled=False,
        _ri_pipeline_providers=[],
    )
    with patch("agent.transports.ri_llm._NATIVE_AVAILABLE", True):
        with monkeypatch.context() as cm:
            cm.delenv("HERMES_RI_PIPELINE", raising=False)
            cm.delenv("HERMES_RI_PIPELINE_PROVIDERS", raising=False)
            assert _should_use_ri_pipeline(agent) is False


def test_should_use_ri_pipeline_honors_env_disable(monkeypatch):
    agent = SimpleNamespace(_ri_pipeline_enabled=True, _ri_pipeline_providers=[])
    with patch("agent.transports.ri_llm._NATIVE_AVAILABLE", True):
        with monkeypatch.context() as cm:
            cm.setenv("HERMES_RI_PIPELINE", "0")
            cm.delenv("HERMES_RI_PIPELINE_PROVIDERS", raising=False)
            assert _should_use_ri_pipeline(agent) is False


def test_should_use_ri_pipeline_honors_config_provider_whitelist(monkeypatch):
    agent = SimpleNamespace(
        provider="Ollama-Launch",
        _ri_pipeline_enabled=True,
        _ri_pipeline_providers=["ollama-launch", "deepseek"],
    )
    with patch("agent.transports.ri_llm._NATIVE_AVAILABLE", True):
        with monkeypatch.context() as cm:
            cm.delenv("HERMES_RI_PIPELINE", raising=False)
            cm.delenv("HERMES_RI_PIPELINE_PROVIDERS", raising=False)
            assert _should_use_ri_pipeline(agent) is True


def test_should_use_ri_pipeline_config_provider_whitelist_rejects_unlisted(monkeypatch):
    agent = SimpleNamespace(
        provider="openrouter",
        _ri_pipeline_enabled=True,
        _ri_pipeline_providers=["ollama-launch", "deepseek"],
    )
    with patch("agent.transports.ri_llm._NATIVE_AVAILABLE", True):
        with monkeypatch.context() as cm:
            cm.delenv("HERMES_RI_PIPELINE", raising=False)
            cm.delenv("HERMES_RI_PIPELINE_PROVIDERS", raising=False)
            assert _should_use_ri_pipeline(agent) is False


def test_should_use_ri_pipeline_env_provider_whitelist_precedes_config(monkeypatch):
    """Runtime env provider list must override config provider list."""
    agent = SimpleNamespace(
        provider="openrouter",
        _ri_pipeline_enabled=True,
        _ri_pipeline_providers=["ollama-launch", "deepseek"],
    )
    with patch("agent.transports.ri_llm._NATIVE_AVAILABLE", True):
        with monkeypatch.context() as cm:
            cm.delenv("HERMES_RI_PIPELINE", raising=False)
            cm.setenv("HERMES_RI_PIPELINE_PROVIDERS", "openrouter,openai")
            assert _should_use_ri_pipeline(agent) is True


def test_should_use_ri_pipeline_requires_native_extension(monkeypatch):
    agent = SimpleNamespace(
        provider="openrouter",
        _ri_pipeline_enabled=True,
        _ri_pipeline_providers=[],
    )
    with patch("agent.transports.ri_llm._NATIVE_AVAILABLE", False):
        with monkeypatch.context() as cm:
            cm.delenv("HERMES_RI_PIPELINE", raising=False)
            cm.delenv("HERMES_RI_PIPELINE_PROVIDERS", raising=False)
            assert _should_use_ri_pipeline(agent) is False

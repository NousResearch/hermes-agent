"""Tests for ordered provider fallback chain (salvage of PR #1761).

Extends the single-fallback tests in test_fallback_model.py to cover
the new list-based ``fallback_providers`` config format and chain
advancement through multiple providers.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from run_agent import AIAgent, _parse_chat_completion_sse_payload, _runtime_unhealthy_state


def _make_agent(fallback_model=None):
    """Create a minimal AIAgent with optional fallback config."""
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
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


def _mock_response(content="Hello", finish_reason="stop", reasoning=None, reasoning_content=None):
    message = SimpleNamespace(content=content, tool_calls=None)
    if reasoning is not None:
        message.reasoning = reasoning
    if reasoning_content is not None:
        message.reasoning_content = reasoning_content
    return SimpleNamespace(
        choices=[SimpleNamespace(index=0, message=message, finish_reason=finish_reason)],
        usage=None,
        model="mock-model",
    )


# ── Chain initialisation ──────────────────────────────────────────────────


class TestFallbackChainInit:
    def test_no_fallback(self):
        agent = _make_agent(fallback_model=None)
        assert agent._fallback_chain == []
        assert agent._fallback_index == 0
        assert agent._fallback_model is None

    def test_single_dict_backwards_compat(self):
        fb = {"provider": "openai", "model": "gpt-4o"}
        agent = _make_agent(fallback_model=fb)
        assert agent._fallback_chain == [fb]
        assert agent._fallback_model == fb

    def test_list_of_providers(self):
        fbs = [
            {"provider": "openai", "model": "gpt-4o"},
            {"provider": "zai", "model": "glm-4.7"},
        ]
        agent = _make_agent(fallback_model=fbs)
        assert len(agent._fallback_chain) == 2
        assert agent._fallback_model == fbs[0]

    def test_disabled_entries_are_skipped(self):
        fbs = [
            {"provider": "openai", "model": "gpt-4o", "enabled": False},
            {"provider": "zai", "model": "glm-4.7", "enabled": "off"},
            {"provider": "custom", "model": "gpt-5.4", "base_url": "https://pay.kxaug.xyz/v1"},
        ]
        agent = _make_agent(fallback_model=fbs)
        assert agent._fallback_chain == [
            {"provider": "custom", "model": "gpt-5.4", "base_url": "https://pay.kxaug.xyz/v1"}
        ]
        assert agent._fallback_model == agent._fallback_chain[0]

    def test_invalid_entries_filtered(self):
        fbs = [
            {"provider": "openai", "model": "gpt-4o"},
            {"provider": "", "model": "glm-4.7"},
            {"provider": "zai"},
            "not-a-dict",
        ]
        agent = _make_agent(fallback_model=fbs)
        assert len(agent._fallback_chain) == 1
        assert agent._fallback_chain[0]["provider"] == "openai"

    def test_empty_list(self):
        agent = _make_agent(fallback_model=[])
        assert agent._fallback_chain == []
        assert agent._fallback_model is None

    def test_invalid_dict_no_provider(self):
        agent = _make_agent(fallback_model={"model": "gpt-4o"})
        assert agent._fallback_chain == []


# ── Chain advancement ─────────────────────────────────────────────────────


class TestFallbackChainAdvancement:
    def test_interruptible_api_call_recovers_mislabelled_sse_chat_completion(self):
        agent = _make_agent(fallback_model=None)
        agent.provider = "custom"
        agent.model = "grok-4.20-0309"
        agent.base_url = "https://wududu.edu.kg/v1"
        agent.api_key = "test-key"

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = AttributeError("'str' object has no attribute 'choices'")
        mock_http_response = MagicMock()
        mock_http_response.text = (
            'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"grok-4.20-0309",'
            '"choices":[{"index":0,"delta":{"role":"assistant","content":"OK"}}]}\n\n'
            'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"grok-4.20-0309",'
            '"choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":"stop"}]}\n\n'
            "data: [DONE]\n\n"
        )
        mock_http_response.raise_for_status.return_value = None

        with (
            patch.object(agent, "_create_request_openai_client", return_value=mock_client),
            patch.object(agent, "_close_request_openai_client"),
            patch("httpx.post", return_value=mock_http_response),
        ):
            response = agent._interruptible_api_call(
                {"model": agent.model, "messages": [{"role": "user", "content": "只回复OK"}]}
            )

        assert response.choices[0].message.content == "OK"
        assert response.choices[0].finish_reason == "stop"

    def test_interruptible_api_call_parses_string_sse_payload(self):
        agent = _make_agent(fallback_model=None)
        agent.provider = "custom"
        agent.model = "grok-4.20-0309"
        agent.base_url = "https://wududu.edu.kg/v1"
        agent.api_key = "test-key"

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = (
            'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"grok-4.20-0309",'
            '"choices":[{"index":0,"delta":{"role":"assistant","content":"OK"}}]}\n\n'
            'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"grok-4.20-0309",'
            '"choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":"stop"}]}\n\n'
            "data: [DONE]\n\n"
        )

        with (
            patch.object(agent, "_create_request_openai_client", return_value=mock_client),
            patch.object(agent, "_close_request_openai_client"),
        ):
            response = agent._interruptible_api_call(
                {"model": agent.model, "messages": [{"role": "user", "content": "只回复OK"}]}
            )

        assert response.choices[0].message.content == "OK"
        assert response.choices[0].finish_reason == "stop"

    def test_interruptible_api_call_parses_string_sse_tool_calls(self):
        agent = _make_agent(fallback_model=None)
        agent.provider = "custom"
        agent.model = "grok-4.20-0309"
        agent.base_url = "https://wududu.edu.kg/v1"
        agent.api_key = "test-key"

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = (
            'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"grok-4.20-0309",'
            '"choices":[{"index":0,"delta":{"role":"assistant","content":null,"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"read_file","arguments":"{\\"path\\":\\"/tmp/demo.txt\\"}"}}]}}]}\n\n'
            'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"grok-4.20-0309",'
            '"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}\n\n'
            "data: [DONE]\n\n"
        )

        with (
            patch.object(agent, "_create_request_openai_client", return_value=mock_client),
            patch.object(agent, "_close_request_openai_client"),
        ):
            response = agent._interruptible_api_call(
                {"model": agent.model, "messages": [{"role": "user", "content": "读文件"}]}
            )

        tool_calls = response.choices[0].message.tool_calls
        assert response.choices[0].finish_reason == "tool_calls"
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].id == "call_1"
        assert tool_calls[0].function.name == "read_file"
        assert tool_calls[0].function.arguments == '{"path":"/tmp/demo.txt"}'

    def test_parse_chat_completion_sse_payload_handles_cumulative_tool_call_frames(self):
        payload = (
            'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"grok-4.20-0309",'
            '"choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"read_file","arguments":"{\\"path\\":"}}]}}]}\n\n'
            'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"grok-4.20-0309",'
            '"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"read_file","arguments":"{\\"path\\":\\"/tmp/demo.txt\\"}"}}]}}]}\n\n'
            'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"grok-4.20-0309",'
            '"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}\n\n'
            "data: [DONE]\n\n"
        )

        response = _parse_chat_completion_sse_payload(payload)

        tool_calls = response.choices[0].message.tool_calls
        assert response.choices[0].finish_reason == "tool_calls"
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].id == "call_1"
        assert tool_calls[0].function.name == "read_file"
        assert tool_calls[0].function.arguments == '{"path":"/tmp/demo.txt"}'

    def test_parse_chat_completion_sse_payload_merges_same_tool_id_without_index(self):
        payload = (
            'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"grok-4.20-0309",'
            '"choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"read_","arguments":"{\\"path\\":"}}]}}]}\n\n'
            'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"grok-4.20-0309",'
            '"choices":[{"index":0,"delta":{"tool_calls":[{"id":"call_1","type":"function","function":{"name":"file","arguments":"\\"/tmp/demo.txt\\"}"}}]}}]}\n\n'
            'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","model":"grok-4.20-0309",'
            '"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}\n\n'
            "data: [DONE]\n\n"
        )

        response = _parse_chat_completion_sse_payload(payload)

        tool_calls = response.choices[0].message.tool_calls
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].id == "call_1"
        assert tool_calls[0].function.name == "read_file"
        assert tool_calls[0].function.arguments == '{"path":"/tmp/demo.txt"}'

    def test_interruptible_api_call_retries_without_optional_tool_controls_on_400(self):
        agent = _make_agent(fallback_model=None)
        agent.provider = "custom"
        agent.model = "grok-4.20-multi-agent-0309"
        agent.base_url = "https://wududu.edu.kg/v1"
        agent.api_key = "test-key"

        class _BadRequest(RuntimeError):
            def __init__(self, message):
                super().__init__(message)
                self.status_code = 400

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            _BadRequest("Unknown field: parallel_tool_calls"),
            _mock_response(content="OK", finish_reason="stop"),
        ]

        api_kwargs = {
            "model": agent.model,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "read_file"}}],
            "tool_choice": "auto",
            "parallel_tool_calls": True,
        }

        with (
            patch.object(agent, "_create_request_openai_client", return_value=mock_client),
            patch.object(agent, "_close_request_openai_client"),
        ):
            response = agent._interruptible_api_call(api_kwargs)

        assert response.choices[0].message.content == "OK"
        assert mock_client.chat.completions.create.call_count == 2
        retry_kwargs = mock_client.chat.completions.create.call_args_list[1].kwargs
        assert "parallel_tool_calls" not in retry_kwargs
        assert "tool_choice" not in retry_kwargs

    def test_exhausted_returns_false(self):
        agent = _make_agent(fallback_model=None)
        assert agent._try_activate_fallback() is False

    def test_advances_index(self):
        fbs = [
            {"provider": "openai", "model": "gpt-4o"},
            {"provider": "zai", "model": "glm-4.7"},
        ]
        agent = _make_agent(fallback_model=fbs)
        with patch("agent.auxiliary_client.resolve_provider_client",
                    return_value=(_mock_client(), "gpt-4o")):
            assert agent._try_activate_fallback() is True
            assert agent._fallback_index == 1
            assert agent.model == "gpt-4o"
            assert agent._fallback_activated is True

    def test_second_fallback_works(self):
        fbs = [
            {"provider": "openai", "model": "gpt-4o"},
            {"provider": "zai", "model": "glm-4.7"},
        ]
        agent = _make_agent(fallback_model=fbs)
        with patch("agent.auxiliary_client.resolve_provider_client",
                    return_value=(_mock_client(), "resolved")):
            assert agent._try_activate_fallback() is True
            assert agent.model == "gpt-4o"
            assert agent._try_activate_fallback() is True
            assert agent.model == "glm-4.7"
            assert agent._fallback_index == 2

    def test_all_exhausted_returns_false(self):
        fbs = [{"provider": "openai", "model": "gpt-4o"}]
        agent = _make_agent(fallback_model=fbs)
        with patch("agent.auxiliary_client.resolve_provider_client",
                    return_value=(_mock_client(), "gpt-4o")):
            assert agent._try_activate_fallback() is True
            assert agent._try_activate_fallback() is False

    def test_skips_unconfigured_provider_to_next(self):
        """If resolve_provider_client returns None, skip to next in chain."""
        fbs = [
            {"provider": "broken", "model": "nope"},
            {"provider": "openai", "model": "gpt-4o"},
        ]
        agent = _make_agent(fallback_model=fbs)
        with patch("agent.auxiliary_client.resolve_provider_client") as mock_rpc:
            mock_rpc.side_effect = [
                (None, None),                    # broken provider
                (_mock_client(), "gpt-4o"),       # fallback succeeds
            ]
            assert agent._try_activate_fallback() is True
            assert agent.model == "gpt-4o"
            assert agent._fallback_index == 2

    def test_skips_provider_that_raises_to_next(self):
        """If resolve_provider_client raises, skip to next in chain."""
        fbs = [
            {"provider": "broken", "model": "nope"},
            {"provider": "openai", "model": "gpt-4o"},
        ]
        agent = _make_agent(fallback_model=fbs)
        with patch("agent.auxiliary_client.resolve_provider_client") as mock_rpc:
            mock_rpc.side_effect = [
                RuntimeError("auth failed"),
                (_mock_client(), "gpt-4o"),
            ]
            assert agent._try_activate_fallback() is True
            assert agent.model == "gpt-4o"

    def test_empty_response_retries_primary_once_before_fallback_provider(self):
        fbs = [
            {"provider": "custom", "model": "fallback-model", "base_url": "https://fallback.example/v1"},
        ]
        agent = _make_agent(fallback_model=fbs)
        agent.model = "primary-model"
        agent.provider = "custom"
        agent.base_url = "https://primary.example/v1"
        calls = []

        def fake_api_call(*_args, **_kwargs):
            calls.append(agent.model)
            if agent.model == "primary-model":
                return _mock_response(content=None, finish_reason="stop")
            if agent.model == "fallback-model":
                return _mock_response(content="fallback answer", finish_reason="stop")
            raise AssertionError(f"Unexpected model during test: {agent.model}")

        with (
            patch.object(agent, "_interruptible_api_call", side_effect=fake_api_call),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch("agent.auxiliary_client.resolve_provider_client",
                  return_value=(_mock_client(base_url="https://fallback.example/v1"), "fallback-model")),
        ):
            result = agent.run_conversation("在? 用一句话回复。")

        assert result["final_response"] == "fallback answer"
        assert result["completed"] is True
        assert result["api_calls"] == 3
        assert calls == ["primary-model", "primary-model", "fallback-model"]
        assert agent.model == "fallback-model"

    def test_empty_response_same_runtime_retry_can_recover_without_fallback(self):
        fbs = [
            {"provider": "custom", "model": "fallback-model", "base_url": "https://fallback.example/v1"},
        ]
        agent = _make_agent(fallback_model=fbs)
        agent.model = "primary-model"
        agent.provider = "custom"
        agent.base_url = "https://primary.example/v1"
        calls = []

        def fake_api_call(*_args, **_kwargs):
            calls.append(agent.model)
            if calls == ["primary-model"]:
                return _mock_response(content=None, finish_reason="stop")
            if agent.model == "primary-model":
                return _mock_response(content="primary recovered", finish_reason="stop")
            raise AssertionError(f"Unexpected model during test: {agent.model}")

        with (
            patch.object(agent, "_interruptible_api_call", side_effect=fake_api_call),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("在?")

        assert result["final_response"] == "primary recovered"
        assert result["completed"] is True
        assert result["api_calls"] == 2
        assert calls == ["primary-model", "primary-model"]
        assert agent.model == "primary-model"

    def test_empty_response_without_fallback_marks_runtime_unhealthy(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        agent = _make_agent(fallback_model=None)
        agent.model = "primary-model"
        agent.provider = "custom"
        agent.base_url = "https://primary.example/v1"
        agent._primary_runtime["model"] = agent.model
        agent._primary_runtime["provider"] = agent.provider
        agent._primary_runtime["base_url"] = agent.base_url
        agent._primary_runtime["api_mode"] = agent.api_mode

        calls = []

        def fake_api_call(*_args, **_kwargs):
            calls.append(agent.model)
            return _mock_response(content=None, finish_reason="stop")

        with (
            patch.object(agent, "_interruptible_api_call", side_effect=fake_api_call),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("在?")

        remaining, reason = _runtime_unhealthy_state(
            provider="custom",
            model="primary-model",
            base_url="https://primary.example/v1",
            api_mode=agent.api_mode,
        )

        assert result["final_response"] == "(empty)"
        assert result["completed"] is True
        assert calls == ["primary-model", "primary-model"]
        assert remaining > 0
        assert reason == "empty_response"

    def test_empty_response_pins_fallback_for_following_turn(self):
        fbs = [
            {"provider": "custom", "model": "fallback-model", "base_url": "https://fallback.example/v1"},
        ]
        agent = _make_agent(fallback_model=fbs)
        agent.model = "primary-model"
        agent.provider = "custom"
        agent.base_url = "https://primary.example/v1"
        calls = []

        def fake_api_call(*_args, **_kwargs):
            calls.append(agent.model)
            if agent.model == "primary-model":
                return _mock_response(content=None, finish_reason="stop")
            if agent.model == "fallback-model":
                return _mock_response(content="fallback answer", finish_reason="stop")
            raise AssertionError(f"Unexpected model during test: {agent.model}")

        with (
            patch.object(agent, "_interruptible_api_call", side_effect=fake_api_call),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch("run_agent.OpenAI", return_value=MagicMock()),
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(_mock_client(base_url="https://fallback.example/v1"), "fallback-model"),
            ),
        ):
            first = agent.run_conversation("第一轮")
            second = agent.run_conversation("第二轮")

        assert first["final_response"] == "fallback answer"
        assert second["final_response"] == "fallback answer"
        assert calls == ["primary-model", "primary-model", "fallback-model", "fallback-model"]

    def test_empty_response_marks_primary_unhealthy_across_agents(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        fbs = [
            {"provider": "custom", "model": "fallback-model", "base_url": "https://fallback.example/v1"},
        ]
        agent1 = _make_agent(fallback_model=fbs)
        agent1.model = "primary-model"
        agent1.provider = "custom"
        agent1.base_url = "https://primary.example/v1"
        agent1._primary_runtime["model"] = agent1.model
        agent1._primary_runtime["provider"] = agent1.provider
        agent1._primary_runtime["base_url"] = agent1.base_url
        agent1._primary_runtime["api_mode"] = agent1.api_mode

        first_calls = []

        def first_api_call(*_args, **_kwargs):
            first_calls.append(agent1.model)
            if agent1.model == "primary-model":
                return _mock_response(content=None, finish_reason="stop")
            if agent1.model == "fallback-model":
                return _mock_response(content="fallback answer", finish_reason="stop")
            raise AssertionError(f"Unexpected model during first run: {agent1.model}")

        agent2 = _make_agent(fallback_model=fbs)
        agent2.model = "primary-model"
        agent2.provider = "custom"
        agent2.base_url = "https://primary.example/v1"
        agent2._primary_runtime["model"] = agent2.model
        agent2._primary_runtime["provider"] = agent2.provider
        agent2._primary_runtime["base_url"] = agent2.base_url
        agent2._primary_runtime["api_mode"] = agent2.api_mode

        second_calls = []

        def second_api_call(*_args, **_kwargs):
            second_calls.append(agent2.model)
            if agent2.model == "primary-model":
                raise AssertionError("Primary runtime should have been skipped after shared empty-response failure")
            if agent2.model == "fallback-model":
                return _mock_response(content="fallback answer", finish_reason="stop")
            raise AssertionError(f"Unexpected model during second run: {agent2.model}")

        with (
            patch.object(agent1, "_interruptible_api_call", side_effect=first_api_call),
            patch.object(agent2, "_interruptible_api_call", side_effect=second_api_call),
            patch.object(agent1, "_persist_session"),
            patch.object(agent2, "_persist_session"),
            patch.object(agent1, "_save_trajectory"),
            patch.object(agent2, "_save_trajectory"),
            patch.object(agent1, "_cleanup_task_resources"),
            patch.object(agent2, "_cleanup_task_resources"),
            patch("agent.auxiliary_client.resolve_provider_client",
                  return_value=(_mock_client(base_url="https://fallback.example/v1"), "fallback-model")),
        ):
            first = agent1.run_conversation("第一轮")
            second = agent2.run_conversation("第二轮")

        assert first["final_response"] == "fallback answer"
        assert second["final_response"] == "fallback answer"
        assert first_calls == ["primary-model", "primary-model", "fallback-model"]
        assert second_calls == ["fallback-model"]

    def test_empty_response_fallback_is_silent_to_user(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        fbs = [
            {"provider": "custom", "model": "fallback-model", "base_url": "https://fallback.example/v1"},
        ]
        agent = _make_agent(fallback_model=fbs)
        agent.model = "primary-model"
        agent.provider = "custom"
        agent.base_url = "https://primary.example/v1"
        agent._primary_runtime["model"] = agent.model
        agent._primary_runtime["provider"] = agent.provider
        agent._primary_runtime["base_url"] = agent.base_url
        agent._primary_runtime["api_mode"] = agent.api_mode

        statuses = []

        def fake_api_call(*_args, **_kwargs):
            if agent.model == "primary-model":
                return _mock_response(content=None, finish_reason="stop")
            if agent.model == "fallback-model":
                return _mock_response(content="fallback answer", finish_reason="stop")
            raise AssertionError(f"Unexpected model during test: {agent.model}")

        with (
            patch.object(agent, "_interruptible_api_call", side_effect=fake_api_call),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch.object(agent, "_emit_status", side_effect=statuses.append),
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(_mock_client(base_url="https://fallback.example/v1"), "fallback-model"),
            ),
        ):
            result = agent.run_conversation("在?")

        assert result["final_response"] == "fallback answer"
        assert statuses == []

    def test_invalid_response_pins_fallback_for_following_turn(self):
        fbs = [
            {"provider": "custom", "model": "fallback-model", "base_url": "https://fallback.example/v1"},
        ]
        agent = _make_agent(fallback_model=fbs)
        agent.model = "primary-model"
        agent.provider = "custom"
        agent.base_url = "https://primary.example/v1"
        agent._primary_runtime["model"] = agent.model
        agent._primary_runtime["provider"] = agent.provider
        agent._primary_runtime["base_url"] = agent.base_url
        agent._primary_runtime["api_mode"] = agent.api_mode
        calls = []

        bad_resp = SimpleNamespace(choices=[], usage=None, model="primary-model")

        def fake_api_call(*_args, **_kwargs):
            calls.append(agent.model)
            if agent.model == "primary-model":
                return bad_resp
            if agent.model == "fallback-model":
                return _mock_response(content="fallback answer", finish_reason="stop")
            raise AssertionError(f"Unexpected model during test: {agent.model}")

        with (
            patch.object(agent, "_interruptible_api_call", side_effect=fake_api_call),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch("run_agent.OpenAI", return_value=MagicMock()),
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(_mock_client(base_url="https://fallback.example/v1"), "fallback-model"),
            ),
            patch("run_agent.time.sleep", return_value=None),
        ):
            first = agent.run_conversation("第一轮")
            second = agent.run_conversation("第二轮")

        assert first["final_response"] == "fallback answer"
        assert second["final_response"] == "fallback answer"
        assert calls == ["primary-model", "fallback-model", "fallback-model"]

    def test_auth_failures_do_not_spam_multiple_fallback_notices_in_one_turn(self):
        fbs = [
            {"provider": "custom", "model": "fallback-model", "base_url": "https://fallback.example/v1"},
        ]
        agent = _make_agent(fallback_model=fbs)
        agent.model = "primary-model"
        agent.provider = "custom"
        agent.base_url = "https://primary.example/v1"
        agent._primary_runtime["model"] = agent.model
        agent._primary_runtime["provider"] = agent.provider
        agent._primary_runtime["base_url"] = agent.base_url
        agent._primary_runtime["api_mode"] = agent.api_mode
        statuses = []
        agent.status_callback = lambda _kind, message: statuses.append(message)

        class _UnauthorizedError(RuntimeError):
            def __init__(self):
                super().__init__("Error code: 401 - unauthorized")
                self.status_code = 401
                self.body = {"error": {"message": "Unauthorized"}}

        class _ForbiddenError(RuntimeError):
            def __init__(self):
                super().__init__("Error code: 403 - blocked")
                self.status_code = 403
                self.body = {"error": {"message": "Blocked"}}

        def fake_api_call(*_args, **_kwargs):
            if agent.model == "primary-model":
                raise _UnauthorizedError()
            if agent.model == "fallback-model":
                raise _ForbiddenError()
            raise AssertionError(f"Unexpected model during test: {agent.model}")

        with (
            patch.object(agent, "_interruptible_api_call", side_effect=fake_api_call),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(_mock_client(base_url="https://fallback.example/v1"), "fallback-model"),
            ),
        ):
            result = agent.run_conversation("在?")

        assert result["completed"] is False
        assert result["failed"] is True
        fallback_statuses = [msg for msg in statuses if "fallback" in msg.lower()]
        assert fallback_statuses == [
            "🔄 Primary model failed — switching to fallback: fallback-model via custom",
        ]
        assert any("❌ Non-retryable error (HTTP 403)" in msg for msg in statuses)

    def test_auth_failure_final_error_preserves_primary_failure_context(self):
        fbs = [
            {"provider": "custom", "model": "fallback-model", "base_url": "https://fallback.example/v1"},
        ]
        agent = _make_agent(fallback_model=fbs)
        agent.model = "primary-model"
        agent.provider = "custom"
        agent.base_url = "https://primary.example/v1"
        agent._primary_runtime["model"] = agent.model
        agent._primary_runtime["provider"] = agent.provider
        agent._primary_runtime["base_url"] = agent.base_url
        agent._primary_runtime["api_mode"] = agent.api_mode

        class _ForbiddenPrimary(RuntimeError):
            def __init__(self):
                super().__init__("Error code: 403 - blocked")
                self.status_code = 403
                self.body = {"error": {"message": "Your request was blocked."}}

        def fake_api_call(*_args, **_kwargs):
            if agent.model == "primary-model":
                raise _ForbiddenPrimary()
            if agent.model == "fallback-model":
                raise RuntimeError("No available accounts for this model tier")
            raise AssertionError(f"Unexpected model during test: {agent.model}")

        with (
            patch.object(agent, "_interruptible_api_call", side_effect=fake_api_call),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(_mock_client(base_url="https://fallback.example/v1"), "fallback-model"),
            ),
            patch("run_agent.time.sleep", return_value=None),
        ):
            result = agent.run_conversation("在?")

        assert result["completed"] is False
        assert result["failed"] is True
        assert "Primary runtime failed first" in result["final_response"]
        assert "Your request was blocked." in result["final_response"]
        assert "Fallback runtime then failed" in result["final_response"]
        assert "No available accounts for this model tier" in result["final_response"]

    def test_auth_failure_marks_primary_unhealthy_with_shorter_window(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        fbs = [
            {"provider": "custom", "model": "fallback-model", "base_url": "https://fallback.example/v1"},
        ]
        agent = _make_agent(fallback_model=fbs)
        agent.model = "primary-model"
        agent.provider = "custom"
        agent.base_url = "https://primary.example/v1"
        agent._primary_runtime["model"] = agent.model
        agent._primary_runtime["provider"] = agent.provider
        agent._primary_runtime["base_url"] = agent.base_url
        agent._primary_runtime["api_mode"] = agent.api_mode

        class _ForbiddenPrimary(RuntimeError):
            def __init__(self):
                super().__init__("Error code: 403 - blocked")
                self.status_code = 403
                self.body = {"error": {"message": "Your request was blocked."}}

        def fake_api_call(*_args, **_kwargs):
            if agent.model == "primary-model":
                raise _ForbiddenPrimary()
            return _mock_response(content="fallback answer", finish_reason="stop")

        with (
            patch.object(agent, "_interruptible_api_call", side_effect=fake_api_call),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(_mock_client(base_url="https://fallback.example/v1"), "fallback-model"),
            ),
        ):
            result = agent.run_conversation("在?")

        assert result["final_response"] == "fallback answer"
        remaining, reason = _runtime_unhealthy_state(
            provider="custom",
            model="primary-model",
            base_url="https://primary.example/v1",
            api_mode=agent.api_mode,
        )
        assert remaining > 0
        assert remaining <= agent._AUTH_FAILURE_UNHEALTHY_SECONDS
        assert reason == "http_403"

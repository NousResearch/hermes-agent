"""Subscription-safe auxiliary routing for the Claude Agent SDK."""

from __future__ import annotations

import asyncio
import sys
from types import ModuleType

import pytest

from agent import auxiliary_client as M
from agent import claude_sdk_aux_client as AUX
from agent.claude_sdk_aux_client import ClaudeSdkAuxClient, ClaudeSdkAuxError
from agent.transports import claude_agent_sdk_session as SESSION


def _plant_sdk(monkeypatch, messages):
    """Install the optional SDK's minimal typed surface for one test."""
    module = ModuleType("claude_agent_sdk")

    class TextBlock:
        def __init__(self, text):
            self.text = text

    class AssistantMessage:
        def __init__(self, content):
            self.content = content

    class ResultMessage:
        def __init__(
            self,
            *,
            subtype="success",
            is_error=False,
            result="",
            errors=None,
            usage=None,
            stop_reason=None,
        ):
            self.subtype = subtype
            self.is_error = is_error
            self.result = result
            self.errors = errors or []
            self.usage = usage
            self.stop_reason = stop_reason

    captured = {}

    class ClaudeAgentOptions:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    async def query(*, prompt, options):
        captured["prompt"] = prompt
        captured["options"] = options
        for message in messages:
            yield message

    module.TextBlock = TextBlock
    module.AssistantMessage = AssistantMessage
    module.ResultMessage = ResultMessage
    module.ClaudeAgentOptions = ClaudeAgentOptions
    module.query = query
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", module)
    return module, captured


def test_auto_sdk_runtime_uses_one_shot_subscription_aux(monkeypatch):
    monkeypatch.setattr(
        M,
        "_normalize_main_runtime",
        lambda runtime: {
            "api_mode": "claude_agent_sdk",
            "model": "claude-sonnet-5",
            "provider": "claude-agent-sdk",
        },
    )

    client, model, provider = M._resolve_auto_route(main_runtime={})

    assert isinstance(client, ClaudeSdkAuxClient)
    assert model == "claude-sonnet-5"
    assert provider == "claude-agent-sdk"


def test_explicit_sdk_aux_provider_returns_sdk_client():
    client, model = M.resolve_provider_client(
        "claude-agent-sdk", model="claude-sonnet-5"
    )

    assert isinstance(client, ClaudeSdkAuxClient)
    assert model == "claude-sonnet-5"


def test_explicit_sdk_async_aux_provider_uses_subscription_facade(monkeypatch):
    """Async SDK aux calls must not fall through to AsyncOpenAI."""
    client, model = M.resolve_provider_client(
        "claude-agent-sdk", model="claude-sonnet-5", async_mode=True
    )

    assert type(client).__name__ == "AsyncClaudeSdkAuxClient"
    assert model == "claude-sonnet-5"
    assert client.base_url == ""

    async def _fake_collect(prompt, *, model):
        assert prompt
        assert model == "claude-sonnet-5"
        return "summary", {"input_tokens": 2}, "stop"

    monkeypatch.setattr(AUX, "_collect_text", _fake_collect)
    result = asyncio.run(
        client.chat.completions.create(
            model="claude-sonnet-5",
            messages=[{"role": "user", "content": "summarize"}],
        )
    )

    assert result.choices[0].message.content == "summary"


def test_auto_sdk_async_aux_provider_retains_subscription_identity():
    client, model = M.resolve_provider_client(
        "auto",
        async_mode=True,
        main_runtime={
            "api_mode": "claude_agent_sdk",
            "model": "claude-sonnet-5",
            "provider": "claude-agent-sdk",
        },
    )

    assert type(client).__name__ == "AsyncClaudeSdkAuxClient"
    assert model == "claude-sonnet-5"
    assert getattr(client, "_hermes_aux_effective_provider", None) == "claude-agent-sdk"


def _payment_error():
    error = RuntimeError("Payment Required: subscription request failed")
    error.status_code = 402
    return error


def _fail_if_called(*_args, **_kwargs):
    raise AssertionError("metered fallback helper must not run for claude-agent-sdk")


def test_auto_sdk_sync_failure_is_fail_closed_before_every_fallback(monkeypatch):
    """An auto-selected subscription client must never open a metered route."""
    client = type("SdkClient", (), {})()
    client.base_url = ""
    client._hermes_aux_effective_provider = "claude-agent-sdk"
    client.chat = type("Chat", (), {})()
    client.chat.completions = type("Completions", (), {})()
    client.chat.completions.create = lambda **_kwargs: (_ for _ in ()).throw(_payment_error())

    monkeypatch.setattr(
        M,
        "_resolve_task_provider_model",
        lambda *_args, **_kwargs: ("auto", None, None, None, None),
    )
    monkeypatch.setattr(M, "_get_cached_client", lambda *_args, **_kwargs: (client, "claude-sonnet-5"))
    for helper in (
        "_try_configured_fallback_chain",
        "_try_main_fallback_chain",
        "_try_payment_fallback",
        "_try_main_agent_model_fallback",
    ):
        monkeypatch.setattr(M, helper, _fail_if_called)

    with pytest.raises(RuntimeError, match="Payment Required"):
        M.call_llm(task="compression", messages=[{"role": "user", "content": "summarize"}])


def test_auto_sdk_async_failure_is_fail_closed_before_every_fallback(monkeypatch):
    """Async auto SDK failures retain the subscription identity too."""
    class Completions:
        async def create(self, **_kwargs):
            raise _payment_error()

    client = type("AsyncSdkClient", (), {})()
    client.base_url = ""
    client._hermes_aux_effective_provider = "claude-agent-sdk"
    client.chat = type("Chat", (), {"completions": Completions()})()

    monkeypatch.setattr(
        M,
        "_resolve_task_provider_model",
        lambda *_args, **_kwargs: ("auto", None, None, None, None),
    )
    monkeypatch.setattr(M, "_get_cached_client", lambda *_args, **_kwargs: (client, "claude-sonnet-5"))
    for helper in (
        "_try_configured_fallback_chain",
        "_try_main_fallback_chain",
        "_try_payment_fallback",
        "_try_main_agent_model_fallback",
    ):
        monkeypatch.setattr(M, helper, _fail_if_called)

    with pytest.raises(RuntimeError, match="Payment Required"):
        asyncio.run(M.async_call_llm(
            task="compression", messages=[{"role": "user", "content": "summarize"}]
        ))


def test_prompt_formatter_preserves_roles_and_only_text_content():
    prompt = AUX._messages_to_prompt(
        [
            {"role": "system", "content": "Summarize precisely."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello"},
                    {"type": "image_url", "image_url": {"url": "secret-url"}},
                ],
            },
            {"role": "tool", "content": {"content": "tool output"}},
        ]
    )

    assert "System:\nSummarize precisely." in prompt
    assert "User:\nhello" in prompt
    assert "Tool result:\ntool output" in prompt
    assert "secret-url" not in prompt


def test_one_shot_query_has_no_tools_and_scrubs_child_env(monkeypatch):
    module, captured = _plant_sdk(monkeypatch, [])
    messages = [
        module.AssistantMessage([module.TextBlock("answer")]),
        module.ResultMessage(usage={"input_tokens": 2}),
    ]
    # The async generator closes over this list, so populate it after the
    # stand-in classes are available.
    monkeypatch.setattr(
        sys.modules["claude_agent_sdk"],
        "query",
        lambda **kwargs: _async_messages(messages, captured, kwargs),
    )
    monkeypatch.setattr(
        SESSION,
        "_sdk_env_overrides",
        lambda: {"ANTHROPIC_API_KEY": ""},
    )

    text, usage, _ = asyncio.run(AUX._collect_text("prompt", model="claude-sonnet-5"))

    assert text == "answer"
    assert usage == {"input_tokens": 2}
    assert captured["tools"] == []
    assert captured["allowed_tools"] == []
    assert captured["mcp_servers"] == {}
    assert captured["env"] == {"ANTHROPIC_API_KEY": ""}


async def _async_messages(messages, captured, kwargs):
    captured["prompt"] = kwargs["prompt"]
    captured["options"] = kwargs["options"]
    for message in messages:
        yield message


@pytest.mark.parametrize(
    "result_kwargs",
    [
        {"is_error": True, "subtype": "error_during_execution", "result": "boom"},
        {"subtype": "error_max_turns", "result": "limit"},
    ],
)
def test_terminal_error_never_returns_partial_text(monkeypatch, result_kwargs):
    module, captured = _plant_sdk(monkeypatch, [])
    messages = [
        module.AssistantMessage([module.TextBlock("partial")]),
        module.ResultMessage(**result_kwargs),
    ]
    monkeypatch.setattr(
        module,
        "query",
        lambda **kwargs: _async_messages(messages, captured, kwargs),
    )
    monkeypatch.setattr(SESSION, "_sdk_env_overrides", lambda: {})

    with pytest.raises(ClaudeSdkAuxError):
        asyncio.run(AUX._collect_text("prompt", model="claude-sonnet-5"))


def test_terminal_error_is_redacted(monkeypatch):
    module, captured = _plant_sdk(monkeypatch, [])
    secret = "sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789"
    messages = [module.ResultMessage(is_error=True, result=f"bad key {secret}")]
    monkeypatch.setattr(
        module,
        "query",
        lambda **kwargs: _async_messages(messages, captured, kwargs),
    )
    monkeypatch.setattr(SESSION, "_sdk_env_overrides", lambda: {})

    with pytest.raises(ClaudeSdkAuxError) as raised:
        asyncio.run(AUX._collect_text("prompt", model="claude-sonnet-5"))

    assert secret not in str(raised.value)


def test_stream_ending_without_result_is_not_partial_success(monkeypatch):
    module, captured = _plant_sdk(monkeypatch, [])
    messages = [module.AssistantMessage([module.TextBlock("partial")])]
    monkeypatch.setattr(
        module,
        "query",
        lambda **kwargs: _async_messages(messages, captured, kwargs),
    )
    monkeypatch.setattr(SESSION, "_sdk_env_overrides", lambda: {})

    with pytest.raises(ClaudeSdkAuxError, match="without a terminal result"):
        asyncio.run(AUX._collect_text("prompt", model="claude-sonnet-5"))


def test_sdk_exception_is_redacted_at_openai_facade(monkeypatch):
    secret = "sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789"

    async def _boom(prompt, *, model):
        raise RuntimeError(f"transport exposed {secret}")

    monkeypatch.setattr(AUX, "_collect_text", _boom)
    client = ClaudeSdkAuxClient()

    with pytest.raises(ClaudeSdkAuxError) as raised:
        client.chat.completions.create(
            messages=[{"role": "user", "content": "summarize this"}]
        )

    assert secret not in str(raised.value)

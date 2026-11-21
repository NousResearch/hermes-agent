"""Task reasoning and output budgets override short-response defaults without changing generic precedence."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.parametrize("raw_reasoning", [None, {"effort": "high"}])
def test_explicit_auxiliary_caller_disable_still_wins(monkeypatch, raw_reasoning):
    from agent import auxiliary_client as aux

    monkeypatch.setattr(aux, "_get_auxiliary_task_config", lambda _task: {"reasoning_effort": "high"})
    payload = aux._build_call_kwargs(
        provider="custom", model="test-model", base_url="https://relay.example/v1",
        messages=[{"role": "user", "content": "hello"}], task="title_generation",
        extra_body=({"reasoning": raw_reasoning} if raw_reasoning else aux._get_task_extra_body("title_generation")),
        reasoning_config={"enabled": False},
    )
    assert payload["reasoning_effort"] == "none"
    assert "reasoning" not in payload.get("extra_body", {})


@pytest.mark.parametrize("base", ["https://api.openai.com/v1", "https://chatgpt.com/backend-api/codex"])
@pytest.mark.parametrize("effort", ["none", "medium", "high", "xhigh"])
def test_terra_supported_settings_still_reach_the_responses_wire(base, effort):
    from agent.transports.codex import ResponsesApiTransport

    wire = ResponsesApiTransport().build_kwargs(
        model="gpt-6.1-terra", messages=[{"role": "user", "content": "hello"}],
        base_url=base, is_codex_backend="chatgpt.com" in base,
        reasoning_config={"enabled": effort != "none", "effort": effort},
    )
    assert wire["reasoning"]["effort"] == effort


@pytest.mark.parametrize("task,answer", [("title_generation", '{"title":"Configured Task Reasoning"}'), ("approval", "APPROVE")])
@pytest.mark.parametrize("extra_reasoning", [False, True])
@pytest.mark.parametrize("provider", ["openrouter", "custom"])
def test_reasoning_and_output_limit_overrides_reach_the_auxiliary_wire(task, answer, extra_reasoning, provider):
    from agent import auxiliary_client as aux
    from agent.auxiliary_task_config import session_auxiliary_scope
    from agent.title_generator import generate_title
    from tools.approval_smart import _smart_approve

    config = {"max_tokens": 4096}
    config.update({"extra_body": {"reasoning": {"enabled": True, "effort": "medium"}}}
                  if extra_reasoning else {"reasoning_effort": "medium"})
    client = MagicMock()
    client.base_url = "https://openrouter.ai/api/v1" if provider == "openrouter" else "https://api.openai.com/v1"

    def create(**kwargs):
        # Simulate a route that budgets reasoning before visible output: the old 16-token cap cannot work.
        cap = kwargs.get("max_completion_tokens", kwargs.get("max_tokens", 0))
        text = answer if cap >= 1024 else ""
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")])

    client.chat.completions.create.side_effect = create
    with session_auxiliary_scope({task: config}), \
         patch.object(aux, "_resolve_task_provider_model", return_value=(
             provider, "test-model", client.base_url, "fake-key", None)), \
         patch.object(aux, "_get_cached_client", return_value=(client, "test-model")):
        if task == "title_generation":
            assert generate_title("Configure my task reasoning") == "Configured Task Reasoning"
        else:
            assert _smart_approve("echo hello", "test command") == "approve"
    wire = client.chat.completions.create.call_args.kwargs
    assert wire.get("max_completion_tokens", wire.get("max_tokens")) == 4096
    if provider == "custom":
        assert wire["reasoning_effort"] == "medium"
    else:
        assert wire["extra_body"]["reasoning"]["effort"] == "medium"


@pytest.mark.parametrize("base,expected_cap", [("https://api.openai.com/v1", 4096),
                                              ("https://chatgpt.com/backend-api/codex", None)])
def test_responses_budget_reaches_public_api_and_omits_unsupported_codex_cap(base, expected_cap):
    from agent.auxiliary_client import _CodexCompletionsAdapter

    adapter = _CodexCompletionsAdapter(SimpleNamespace(base_url=base), "gpt-6.1-terra")
    wire, _, _ = adapter._build_responses_kwargs({
        "model": "gpt-6.1-terra", "messages": [{"role": "user", "content": "hello"}], "max_completion_tokens": 4096,
    })
    assert wire.get("max_output_tokens") == expected_cap


@pytest.mark.parametrize("task,default", [("title_generation", 512), ("approval", 16)])
def test_output_budget_default_and_profile_scoping(task, default):
    from agent.auxiliary_task_config import auxiliary_output_limit, session_auxiliary_scope

    with session_auxiliary_scope({task: {"max_tokens": 2048}}):
        assert auxiliary_output_limit(task, default) == 2048
    with session_auxiliary_scope({}):
        assert auxiliary_output_limit(task, default) == default


@pytest.mark.parametrize("value", [0, -1, True, "4096", None])
def test_invalid_budget_override_is_rejected(value):
    from agent.auxiliary_task_config import auxiliary_output_limit, session_auxiliary_scope

    with session_auxiliary_scope({"approval": {"max_tokens": value}}), pytest.raises(ValueError, match="positive integer"):
        auxiliary_output_limit("approval", 16)


@pytest.mark.parametrize("effort", ["medium", "high"])
def test_configured_session_reasoning_overrides_automatic_continuation_disable(effort):
    from tests.agent.test_reasoning_disable_recovery import _WireAgent, _wire_reasoning_config

    agent = _WireAgent({"enabled": True, "effort": effort})
    agent._ephemeral_reasoning_off = True
    assert _wire_reasoning_config(agent) == {"enabled": True, "effort": effort}
    assert not agent._ephemeral_reasoning_off


@pytest.mark.parametrize("separate_voice_route", [False, True])
def test_voice_reasoning_override_reaches_wire_and_main_setting_is_restored(tmp_path, monkeypatch, separate_voice_route):
    from run_agent import AIAgent
    from tests.fakes.fake_llm_provider import FakeLLMServer, Text, write_hermes_home

    with FakeLLMServer([Text("main reply"), Text("next main reply")]) as main, \
         FakeLLMServer([Text("voice reply")]) as voice:
        route = (f"    provider: custom\n    model: voice-model\n    base_url: {voice.base_url}\n"
                 "    api_key: fake-voice-key\n" if separate_voice_route else "")
        home = write_hermes_home(tmp_path / ".hermes", main.base_url, extra_config=(
            "auxiliary:\n  voice_chat:\n    reasoning_effort: medium\n" + route))
        monkeypatch.setenv("HERMES_HOME", str(home))
        agent = AIAgent(model="main-model", provider="custom", base_url=main.base_url, api_key="fake-main-key",
                        quiet_mode=True, skip_context_files=True, skip_memory=True, enabled_toolsets=["memory"],
                        reasoning_config={"enabled": True, "effort": "high"})
        agent._voice_turn_pending = True
        spoken = agent.run_conversation("spoken question")
        requests = voice.main_requests() if separate_voice_route else main.main_requests()
        assert requests[0]["reasoning_effort"] == "medium"
        assert agent.reasoning_config == {"enabled": True, "effort": "high"}
        agent.run_conversation("typed question", conversation_history=spoken["messages"])
        assert main.main_requests()[-1]["reasoning_effort"] == "high"


@pytest.mark.parametrize("answer,finish_reason", [("", "length"), ("", "stop"), ("I would approve", "length")])
def test_empty_or_invalid_approval_output_remains_fail_safe(answer, finish_reason):
    from tools.approval_smart import _smart_approve

    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=answer), finish_reason=finish_reason)])
    with patch("agent.auxiliary_client.call_llm", return_value=response):
        assert _smart_approve("echo hello", "test command") == "escalate"

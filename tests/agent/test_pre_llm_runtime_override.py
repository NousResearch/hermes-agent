from types import SimpleNamespace

from run_agent import AIAgent


def test_collect_pre_llm_hook_result_preserves_context_only_returns():
    assert AIAgent._collect_pre_llm_hook_result("memory note") == ("memory note", {})
    assert AIAgent._collect_pre_llm_hook_result({"context": "memory note"}) == (
        "memory note",
        {},
    )
    assert AIAgent._collect_pre_llm_hook_result(None) == (None, {})


def test_collect_pre_llm_hook_result_extracts_runtime_override():
    context, override = AIAgent._collect_pre_llm_hook_result(
        {
            "context": "route note",
            "runtime_override": {
                "provider": "openrouter",
                "model": "openrouter/example-specialist",
            },
        }
    )

    assert context == "route note"
    assert override == {
        "provider": "openrouter",
        "model": "openrouter/example-specialist",
    }


def test_collect_pre_llm_hook_result_accepts_direct_override_keys():
    context, override = AIAgent._collect_pre_llm_hook_result(
        {
            "model": "openrouter/example-specialist",
            "provider": "openrouter",
            "system_prompt": "Use the specialist rubric.",
            "ignored": "value",
        }
    )

    assert context is None
    assert override == {
        "model": "openrouter/example-specialist",
        "provider": "openrouter",
        "system_prompt": "Use the specialist rubric.",
    }


def _agent_for_runtime_override():
    agent = object.__new__(AIAgent)
    agent.model = "openrouter/main"
    agent.provider = "openrouter"
    agent.base_url = "https://openrouter.ai/api/v1"
    agent.api_key = "main-key"
    agent.api_mode = "chat_completions"
    agent._primary_runtime = {
        "model": agent.model,
        "provider": agent.provider,
        "base_url": agent.base_url,
        "api_key": agent.api_key,
        "api_mode": agent.api_mode,
    }
    agent.switch_calls = []

    def fake_switch_model(new_model, new_provider, api_key="", base_url="", api_mode=""):
        agent.switch_calls.append(
            {
                "model": new_model,
                "provider": new_provider,
                "api_key": api_key,
                "base_url": base_url,
                "api_mode": api_mode,
            }
        )
        agent.model = new_model
        agent.provider = new_provider
        if api_key:
            agent.api_key = api_key
        if base_url:
            agent.base_url = base_url
        if api_mode:
            agent.api_mode = api_mode

    agent.switch_model = fake_switch_model
    return agent


def test_apply_pre_llm_runtime_override_switches_model_and_system_prompt(monkeypatch):
    agent = _agent_for_runtime_override()

    def fake_resolve_provider_client(*args, **kwargs):
        return (
            SimpleNamespace(
                api_key="specialist-key",
                base_url="https://openrouter.ai/api/v1",
            ),
            "openrouter/specialist-resolved",
        )

    monkeypatch.setattr(
        "agent.auxiliary_client.resolve_provider_client",
        fake_resolve_provider_client,
    )

    active_system_prompt = agent._apply_pre_llm_runtime_override(
        {
            "provider": "openrouter",
            "model": "openrouter/specialist",
            "system_prompt": "Use the specialist rubric.",
        },
        "Base system prompt.",
    )

    assert agent.switch_calls == [
        {
            "model": "openrouter/specialist-resolved",
            "provider": "openrouter",
            "api_key": "specialist-key",
            "base_url": "https://openrouter.ai/api/v1",
            "api_mode": "",
        }
    ]
    assert agent.model == "openrouter/specialist-resolved"
    assert active_system_prompt == "Base system prompt.\n\nUse the specialist rubric."


def test_apply_pre_llm_runtime_override_can_restore_main_model():
    agent = _agent_for_runtime_override()
    agent._apply_pre_llm_runtime_override(
        {"provider": "openrouter", "model": "openrouter/specialist"},
        "Base system prompt.",
    )

    agent._apply_pre_llm_runtime_override({"restore_main": True}, "Base system prompt.")

    assert agent.switch_calls[-1] == {
        "model": "openrouter/main",
        "provider": "openrouter",
        "api_key": "main-key",
        "base_url": "https://openrouter.ai/api/v1",
        "api_mode": "chat_completions",
    }
    assert agent.model == "openrouter/main"

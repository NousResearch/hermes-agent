from agent.transports.chat_completions import ChatCompletionsTransport
from agent.transports.codex import ResponsesApiTransport
from run_agent import AIAgent


def _configured_platform_agent(monkeypatch, tmp_path, *, api_mode, base_url):
    cfg = {
        "platform_request_overrides": {
            "api_server": {
                "reasoning_effort": "minimal",
                "service_tier": "priority",
                "extra_body": {
                    "chat_template_kwargs": {"enable_thinking": False},
                    "shared": "platform",
                },
            },
        },
    }
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr("run_agent._hermes_home", hermes_home)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: cfg)

    return AIAgent(
        base_url=base_url,
        api_key="test-key",
        api_mode=api_mode,
        provider="custom",
        model="test-model",
        platform="api_server",
        reasoning_config={"enabled": True, "effort": "high"},
        request_overrides={
            "reasoning_effort": "low",
            "extra_body": {"shared": "caller", "caller_only": True},
        },
        quiet_mode=True,
        skip_memory=True,
        skip_context_files=True,
    )


def _assert_platform_overrides_reach_transport(kwargs):
    assert kwargs["service_tier"] == "priority"
    assert kwargs["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False},
        "shared": "caller",
        "caller_only": True,
    }


def test_platform_request_overrides_reach_chat_completions_transport(
    monkeypatch, tmp_path
):
    agent = _configured_platform_agent(
        monkeypatch,
        tmp_path,
        api_mode="chat_completions",
        base_url="https://api.kimi.com/v1",
    )

    kwargs = agent._build_api_kwargs([
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
    ])

    assert isinstance(agent._get_transport(), ChatCompletionsTransport)
    assert agent.reasoning_config == {"enabled": True, "effort": "low"}
    assert kwargs["reasoning_effort"] == "low"
    _assert_platform_overrides_reach_transport(kwargs)


def test_platform_request_overrides_reach_responses_transport(monkeypatch, tmp_path):
    agent = _configured_platform_agent(
        monkeypatch,
        tmp_path,
        api_mode="codex_responses",
        base_url="http://localhost:1234/v1",
    )

    kwargs = agent._build_api_kwargs([
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
    ])

    assert isinstance(agent._get_transport(), ResponsesApiTransport)
    assert kwargs["reasoning"] == {"effort": "low", "summary": "auto"}
    _assert_platform_overrides_reach_transport(kwargs)

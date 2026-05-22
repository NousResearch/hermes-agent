from types import SimpleNamespace
from unittest.mock import patch


RUNTIME = {
    "provider": "custom",
    "base_url": "https://example.test/v1",
    "api_key": "***",
    "api_mode": "codex_responses",
    "command": None,
    "args": None,
}


def test_make_agent_passes_agent_reasoning_effort_to_aiagent():
    from acp_adapter.session import SessionManager

    cfg = {
        "model": {"default": "gpt-5.5", "provider": "custom"},
        "agent": {"reasoning_effort": "xhigh"},
    }

    with (
        patch("acp_adapter.session._register_task_cwd"),
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value=RUNTIME),
        patch("run_agent.AIAgent", return_value=SimpleNamespace(model="gpt-5.5")) as mock_agent,
    ):
        SessionManager()._make_agent(session_id="sid-reasoning", cwd=".")

    assert mock_agent.call_args.kwargs["reasoning_config"] == {
        "enabled": True,
        "effort": "xhigh",
    }


def test_make_agent_leaves_reasoning_default_when_agent_config_missing():
    from acp_adapter.session import SessionManager

    cfg = {"model": {"default": "gpt-5.5", "provider": "custom"}}

    with (
        patch("acp_adapter.session._register_task_cwd"),
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value=RUNTIME),
        patch("run_agent.AIAgent", return_value=SimpleNamespace(model="gpt-5.5")) as mock_agent,
    ):
        SessionManager()._make_agent(session_id="sid-default", cwd=".")

    assert mock_agent.call_args.kwargs["reasoning_config"] is None


def test_make_agent_passes_agent_service_tier_to_aiagent():
    from acp_adapter.session import SessionManager

    cfg = {
        "model": {"default": "gpt-5.5", "provider": "custom"},
        "agent": {"service_tier": "fast"},
    }

    with (
        patch("acp_adapter.session._register_task_cwd"),
        patch("hermes_cli.config.load_config", return_value=cfg),
        patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value=RUNTIME),
        patch("run_agent.AIAgent", return_value=SimpleNamespace(model="gpt-5.5")) as mock_agent,
    ):
        SessionManager()._make_agent(session_id="sid-service-tier", cwd=".")

    assert mock_agent.call_args.kwargs["service_tier"] == "priority"

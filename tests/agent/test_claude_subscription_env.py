from pathlib import Path

import pytest

import agent.claude_subscription_env as subscription_env
from agent.claude_subscription_env import build_claude_subscription_env


def test_subscription_env_strips_inherited_secrets_and_keeps_bash_home_isolated():
    host_home = Path("/Users/person")
    profile_home = Path("/profiles/worker/home")
    inherited = {
        "HOME": str(host_home),
        "PATH": "/usr/bin:/bin",
        "LANG": "en_US.UTF-8",
        "ANTHROPIC_API_KEY": "paid-api-key",
        "ANTHROPIC_AUTH_TOKEN": "paid-auth-token",
        "OPENAI_API_KEY": "openai-secret",
        "GITHUB_TOKEN": "github-secret",
        "AWS_SECRET_ACCESS_KEY": "aws-secret",
        "HERMES_HOST_HOME": str(host_home),
        "HERMES_KANBAN_TASK": "BUILD-392",
    }

    env = build_claude_subscription_env(
        inherited,
        host_home=host_home,
        profile_home=profile_home,
    )

    assert env["HOME"] == str(host_home)
    assert "CLAUDE_CONFIG_DIR" not in env
    assert env["PATH"] == inherited["PATH"]
    assert not any("SECRET" in key or key.endswith("_TOKEN") for key in env)
    assert "ANTHROPIC_API_KEY" not in env
    assert "OPENAI_API_KEY" not in env
    assert "GITHUB_TOKEN" not in env
    assert "HERMES_HOST_HOME" not in env
    assert "HERMES_KANBAN_TASK" not in env
    assert "CLAUDE_AGENT_SDK_CLIENT_APP" not in env
    assert env["DISABLE_TELEMETRY"] == "1"
    assert env["DISABLE_ERROR_REPORTING"] == "1"


def test_claude_agent_sdk_is_a_pinned_lazy_runtime_dependency():
    from tools.lazy_deps import feature_specs

    assert feature_specs("runtime.claude_agent_sdk") == (
        "claude-agent-sdk==0.2.116",
    )


def test_missing_home_paths_fail_closed(monkeypatch):
    monkeypatch.setattr(subscription_env, "get_host_user_home", lambda: None)

    with pytest.raises(RuntimeError, match="host home"):
        build_claude_subscription_env({"PATH": "/usr/bin"})

import json

import pytest

from agent import credential_broker
import tools.terminal_tool as terminal_tool


BROKER_CONFIG = {
    "credentials": {
        "broker": {
            "enabled": True,
            "secrets": {
                "github_token": {
                    "source": "env",
                    "name": "GITHUB_TOKEN",
                    "allow": {
                        "tools": ["terminal"],
                        "commands": ["gh"],
                    },
                }
            },
        }
    }
}


def test_resolve_env_credential_when_tool_and_command_allowed(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")

    credential = credential_broker.resolve(
        "github_token",
        requester="terminal",
        command="gh pr list",
        config=BROKER_CONFIG,
    )

    assert credential.env_name == "GITHUB_TOKEN"
    assert credential.value == "ghp_test"
    assert credential.forced_env_name == "_HERMES_FORCE_GITHUB_TOKEN"


def test_resolve_env_credential_denies_unlisted_command(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")

    with pytest.raises(credential_broker.CredentialBrokerError, match="not allowed"):
        credential_broker.resolve(
            "github_token",
            requester="terminal",
            command="python script.py",
            config=BROKER_CONFIG,
        )


def test_resolve_env_credential_denies_secret_without_allow_rule(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    config = {
        "credentials": {
            "broker": {
                "enabled": True,
                "secrets": {
                    "github_token": {
                        "source": "env",
                        "name": "GITHUB_TOKEN",
                    }
                },
            }
        }
    }

    with pytest.raises(credential_broker.CredentialBrokerError, match="not allowed"):
        credential_broker.resolve(
            "github_token",
            requester="terminal",
            command="gh pr list",
            config=config,
        )


def test_scoped_env_overrides_restore_original_environment():
    env = {"KEEP": "1", "_HERMES_FORCE_GITHUB_TOKEN": "old"}

    with credential_broker.scoped_env_overrides(
        env,
        {"_HERMES_FORCE_GITHUB_TOKEN": "new", "_HERMES_FORCE_OTHER": "value"},
    ):
        assert env["_HERMES_FORCE_GITHUB_TOKEN"] == "new"
        assert env["_HERMES_FORCE_OTHER"] == "value"

    assert env == {"KEEP": "1", "_HERMES_FORCE_GITHUB_TOKEN": "old"}


def test_terminal_tool_injects_brokered_credential_for_one_command(monkeypatch, tmp_path):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    monkeypatch.setattr(credential_broker, "_load_config", lambda: BROKER_CONFIG)

    terminal_tool._active_environments.clear()
    terminal_tool._last_activity.clear()

    captured = {}

    class FakeEnv:
        def __init__(self):
            self.env = {}

        def execute(self, command, **kwargs):
            captured["command"] = command
            captured["env_during_execute"] = dict(self.env)
            return {"output": "ok", "returncode": 0}

    fake_env = FakeEnv()

    monkeypatch.setattr(
        terminal_tool,
        "_get_env_config",
        lambda: {
            "env_type": "local",
            "docker_image": "",
            "singularity_image": "",
            "modal_image": "",
            "daytona_image": "",
            "cwd": str(tmp_path),
            "host_cwd": str(tmp_path),
            "timeout": 30,
            "container_cpu": 1,
            "container_memory": 512,
            "container_disk": 1024,
            "container_persistent": False,
            "local_persistent": False,
        },
    )
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(terminal_tool, "_create_environment", lambda **kwargs: fake_env)
    monkeypatch.setattr(terminal_tool, "_check_all_guards", lambda command, env_type: {"approved": True})

    result = json.loads(
        terminal_tool.terminal_tool(
            "gh pr list",
            credentials=["github_token"],
            workdir=str(tmp_path),
        )
    )

    assert result["exit_code"] == 0
    assert captured["env_during_execute"] == {"_HERMES_FORCE_GITHUB_TOKEN": "ghp_test"}
    assert fake_env.env == {}

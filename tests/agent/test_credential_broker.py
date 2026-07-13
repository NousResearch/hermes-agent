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


@pytest.mark.parametrize(
    "command",
    [
        "gh pr list; env",
        "gh pr list && env",
        "gh pr list || env",
        "gh pr list | env",
        "gh pr list & env",
        "gh pr list\nenv",
        "gh pr list $(env)",
        "gh pr list `env`",
        "gh pr list < input",
        "gh pr list > output",
        "gh pr list <(env)",
        "gh pr list >(env)",
        "gh pr list (env)",
    ],
)
def test_resolve_env_credential_denies_compound_command(monkeypatch, command):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")

    with pytest.raises(credential_broker.CredentialBrokerError, match="not allowed"):
        credential_broker.resolve(
            "github_token",
            requester="terminal",
            command=command,
            config=BROKER_CONFIG,
        )


@pytest.mark.parametrize(
    "command",
    [
        "gh search issues 'parser; bug'",
        "gh api --jq ';'",
        "gh api --jq '|'",
        r"gh search issues parser\;bug",
    ],
)
def test_resolve_env_credential_allows_inert_shell_metacharacters(monkeypatch, command):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")

    credential = credential_broker.resolve(
        "github_token",
        requester="terminal",
        command=command,
        config=BROKER_CONFIG,
    )

    assert credential.value == "ghp_test"


def test_resolve_env_credential_requires_exact_executable(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")

    with pytest.raises(credential_broker.CredentialBrokerError, match="not allowed"):
        credential_broker.resolve(
            "github_token",
            requester="terminal",
            command="/tmp/gh pr list",
            config=BROKER_CONFIG,
        )


def test_canonicalize_command_preserves_arguments_as_data():
    assert (
        credential_broker.canonicalize_command("gh search issues 'parser; bug'")
        == "gh search issues 'parser; bug'"
    )


def test_resolve_rejects_invalid_environment_name():
    config = {
        "credentials": {
            "broker": {
                "enabled": True,
                "secrets": {
                    "bad": {
                        "source": "env",
                        "name": "BAD-NAME",
                        "allow": {"tools": ["terminal"], "commands": ["gh"]},
                    }
                },
            }
        }
    }

    with pytest.raises(credential_broker.CredentialBrokerError, match="invalid env"):
        credential_broker.resolve(
            "bad", requester="terminal", command="gh pr list", config=config
        )


def test_resolve_env_overrides_uses_target_environment_name(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")

    assert credential_broker.resolve_env_overrides(
        ["github_token"],
        requester="terminal",
        command="gh pr list",
        config=BROKER_CONFIG,
    ) == {"GITHUB_TOKEN": "ghp_test"}


def test_terminal_tool_injects_brokered_credential_for_one_command(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    monkeypatch.setattr(credential_broker, "_load_config", lambda: BROKER_CONFIG)

    terminal_tool._active_environments.clear()
    terminal_tool._last_activity.clear()

    captured = {}

    class FakeEnv:
        def __init__(self):
            self.env = {}

        def resolve_executable(self, executable):
            assert executable == "gh"
            return "/usr/bin/gh"

        def execute(self, command, **kwargs):
            captured.setdefault("calls", []).append((command, kwargs))
            captured["command"] = command
            captured["kwargs"] = kwargs
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
    monkeypatch.setattr(
        terminal_tool,
        "_check_all_guards",
        lambda command, env_type, **kwargs: {"approved": True},
    )

    result = json.loads(
        terminal_tool.terminal_tool(
            "gh pr list",
            credentials=["github_token"],
            workdir=str(tmp_path),
        )
    )

    assert result["exit_code"] == 0
    assert captured["command"] == "/usr/bin/gh pr list"
    assert captured["kwargs"]["env_overrides"] == {"GITHUB_TOKEN": "ghp_test"}
    assert fake_env.env == {}

    sentinel = tmp_path / "credential-bypass-sentinel"
    blocked = json.loads(
        terminal_tool.terminal_tool(
            f"gh pr list; touch {sentinel}",
            credentials=["github_token"],
            workdir=str(tmp_path),
        )
    )

    assert blocked["status"] == "blocked"
    assert len(captured["calls"]) == 1
    assert not sentinel.exists()

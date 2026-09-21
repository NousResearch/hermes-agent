from __future__ import annotations

import json
import subprocess

import pytest

from hermes_cli.github_identity import (
    GitHubAutomationIdentity,
    GitHubIdentityError,
    run_as_github_automation,
)


def test_identity_environment_replaces_ambient_human_tokens() -> None:
    identity = GitHubAutomationIdentity("mrkillbobbot")

    result = identity.command_environment(
        {
            "HERMES_GITHUB_BOT_TOKEN": "bot-token",
            "GH_TOKEN": "human-token",
            "GITHUB_TOKEN": "ci-token",
        }
    )

    assert result["GH_TOKEN"] == "bot-token"
    assert "GITHUB_TOKEN" not in result


def test_identity_environment_never_falls_back_to_ambient_token() -> None:
    with pytest.raises(GitHubIdentityError, match="HERMES_GITHUB_BOT_TOKEN"):
        GitHubAutomationIdentity("mrkillbobbot").command_environment(
            {"GH_TOKEN": "human-token", "GITHUB_TOKEN": "ci-token"}
        )


def test_git_environment_binds_https_to_bot_without_token_in_argv() -> None:
    result = GitHubAutomationIdentity("mrkillbobbot").git_command_environment(
        {"HERMES_GITHUB_BOT_TOKEN": "bot-token"}
    )

    assert result["GIT_CONFIG_KEY_0"] == "credential.helper"
    assert result["GIT_CONFIG_VALUE_0"] == ""
    assert result["GIT_CONFIG_KEY_1"] == "http.https://github.com/.extraheader"
    assert result["GIT_CONFIG_VALUE_1"].startswith("AUTHORIZATION: basic ")
    assert "bot-token" not in result["GIT_CONFIG_VALUE_1"]
    assert result["GIT_TERMINAL_PROMPT"] == "0"


def test_authenticated_run_verifies_bot_before_requested_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], dict[str, str]]] = []

    def fake_run(argv, **kwargs):
        calls.append((list(argv), dict(kwargs["env"])))
        if argv == ["gh", "api", "user"]:
            return subprocess.CompletedProcess(argv, 0, json.dumps({"login": "mrkillbobbot"}), "")
        return subprocess.CompletedProcess(argv, 0, "[]", "")

    monkeypatch.setattr("hermes_cli.github_identity.subprocess.run", fake_run)

    result = run_as_github_automation(
        ["gh", "pr", "list"],
        identity=GitHubAutomationIdentity("mrkillbobbot"),
        environ={"HERMES_GITHUB_BOT_TOKEN": "bot-token", "GH_TOKEN": "human-token"},
    )

    assert result.returncode == 0
    assert [call[0] for call in calls] == [["gh", "api", "user"], ["gh", "pr", "list"]]
    assert all(call[1]["GH_TOKEN"] == "bot-token" for call in calls)


def test_authenticated_run_refuses_wrong_viewer_before_requested_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(argv, **_kwargs):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, json.dumps({"login": "mrkillbob"}), "")

    monkeypatch.setattr("hermes_cli.github_identity.subprocess.run", fake_run)

    with pytest.raises(GitHubIdentityError, match="did not match"):
        run_as_github_automation(
            ["gh", "pr", "comment", "734"],
            identity=GitHubAutomationIdentity("mrkillbobbot"),
            environ={"HERMES_GITHUB_BOT_TOKEN": "bot-token"},
        )

    assert calls == [["gh", "api", "user"]]

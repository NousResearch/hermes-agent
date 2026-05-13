import json

from gateway.config import Platform
from gateway.requester_identity import (
    CONFIG_RELATIVE_PATH,
    format_missing_identity_message,
    requester_identity_prompt,
    resolve_slack_requester_identity,
    should_require_requester_identity,
)
from gateway.session_context import clear_session_vars, get_session_env, set_session_vars


def test_resolve_slack_requester_identity(tmp_path):
    path = tmp_path / "requester_identities.json"
    path.write_text(
        json.dumps(
            {
                "U123": {
                    "name": "Jane Engineer",
                    "email": "jane@example.com",
                    "github_login": "@jane",
                }
            }
        ),
        encoding="utf-8",
    )

    identity = resolve_slack_requester_identity("U123", path=path)

    assert identity is not None
    assert identity.name == "Jane Engineer"
    assert identity.email == "jane@example.com"
    assert identity.github_login == "jane"
    assert identity.as_env("bot@example.com") == {
        "GIT_AUTHOR_NAME": "Jane Engineer",
        "GIT_AUTHOR_EMAIL": "jane@example.com",
        "GIT_COMMITTER_NAME": "Jane Engineer",
        "GIT_COMMITTER_EMAIL": "jane@example.com",
        "HERMES_REQUESTER_GITHUB_LOGIN": "jane",
        "HERMES_REQUESTER_NAME": "Jane Engineer",
        "HERMES_REQUESTER_EMAIL": "jane@example.com",
        "HERMES_BOT_GIT_EMAIL": "bot@example.com",
    }


def test_missing_slack_requester_identity_fails_fast_message():
    assert should_require_requester_identity(Platform.SLACK, "U123") is True
    assert should_require_requester_identity(Platform.LOCAL, None) is False
    assert format_missing_identity_message("jorge") == (
        f"No GitHub identity mapped for jorge; add yourself to {CONFIG_RELATIVE_PATH} and retry."
    )


def test_requester_env_is_available_to_terminal_subprocess_bridge():
    tokens = set_session_vars(
        platform="slack",
        user_id="U123",
        requester_env={
            "GIT_AUTHOR_NAME": "Jane Engineer",
            "GIT_AUTHOR_EMAIL": "jane@example.com",
            "HERMES_REQUESTER_GITHUB_LOGIN": "jane",
        },
    )
    try:
        assert get_session_env("GIT_AUTHOR_NAME") == "Jane Engineer"
        assert get_session_env("GIT_AUTHOR_EMAIL") == "jane@example.com"
        assert get_session_env("HERMES_REQUESTER_GITHUB_LOGIN") == "jane"
    finally:
        clear_session_vars(tokens)


def test_requester_prompt_contains_commit_and_pr_requirements():
    identity = resolve_slack_requester_identity("U02NPH06XGW")
    assert identity is not None

    prompt = requester_identity_prompt(identity, "bot@example.com")

    assert "Co-Authored-By: citizen-wall-e <bot@example.com>" in prompt
    assert "Requested by @jorgealegre via Slack" in prompt
    assert "--assignee jorgealegre" in prompt

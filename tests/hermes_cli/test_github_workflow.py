import json
from pathlib import Path

import pytest

from hermes_cli.github_workflow import (
    AuthState,
    GitHubCapability,
    GitHubRepository,
    classify_github_response,
    parse_github_remote,
    repository_context,
    resolve_github_capability,
)


def test_parse_https_github_remote():
    assert parse_github_remote("https://github.com/NousResearch/hermes-agent.git") == ("NousResearch", "hermes-agent")


def test_parse_ssh_github_remote():
    assert parse_github_remote("git@github.com:NousResearch/hermes-agent.git") == ("NousResearch", "hermes-agent")


def test_non_github_remote_is_not_guessed():
    assert parse_github_remote("https://gitlab.com/example/project.git") is None


def test_classify_rate_limited_403_separately():
    assert classify_github_response(403, {"X-RateLimit-Remaining": "0"}) is AuthState.RATE_LIMITED
    assert classify_github_response(403, {}) is AuthState.PERMISSION_DENIED


def test_classify_identity_responses():
    assert classify_github_response(200, {}) is AuthState.VERIFIED
    assert classify_github_response(401, {}) is AuthState.INVALID


def test_capability_without_token_keeps_public_read_available(monkeypatch):
    monkeypatch.setattr("hermes_cli.github_workflow.get_secret", lambda name, default=None: default)
    capability = resolve_github_capability(config={"github": {"workflow": {"enabled": True}}})
    assert capability.public_read_available is True
    assert capability.credential_available is False
    assert capability.auth_state is AuthState.UNAVAILABLE
    assert capability.remediation


def test_capability_does_not_expose_token(monkeypatch):
    token = "ghp_test_secret_value_1234567890"
    monkeypatch.setattr("hermes_cli.github_workflow.get_secret", lambda name, default=None: token)
    capability = resolve_github_capability(
        config={"github": {"workflow": {"enabled": True}}}, perform_preflight=False
    )
    assert token not in repr(capability)
    assert token not in str(capability)


def test_repository_context_reads_remote_and_branch(tmp_path, monkeypatch):
    calls = {
        ("rev-parse", "--show-toplevel"): str(tmp_path),
        ("config", "--get", "remote.origin.url"): "git@github.com:owner/repo.git\n",
        ("branch", "--show-current"): "feature/test\n",
        ("rev-parse", "HEAD"): "abc123\n",
    }
    monkeypatch.setattr("hermes_cli.github_workflow._git_stdout", lambda cwd, args: calls.get(tuple(args), ""))
    context = repository_context(str(tmp_path))
    assert context.owner == "owner"
    assert context.name == "repo"
    assert context.branch == "feature/test"
    assert context.sha == "abc123"


def test_disabled_workflow_returns_disabled_capability(monkeypatch):
    called = False

    def fail_get_secret(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("secret lookup must not run when disabled")

    monkeypatch.setattr("hermes_cli.github_workflow.get_secret", fail_get_secret)
    capability = resolve_github_capability(config={"github": {"workflow": {"enabled": False}}})
    assert capability.workflow_enabled is False
    assert capability.auth_state is AuthState.DISABLED
    assert called is False


def test_capability_status_is_json_safe(monkeypatch):
    monkeypatch.setattr("hermes_cli.github_workflow.get_secret", lambda name, default=None: "secret")
    capability = resolve_github_capability(
        config={"github": {"workflow": {"enabled": True}}}, perform_preflight=False
    )
    payload = capability.to_public_dict()
    serialized = json.dumps(payload)
    assert '"_credential"' not in serialized
    assert "ghp_" not in serialized
    assert payload["public_read_available"] is True


@pytest.fixture(autouse=True)
def clear_cache():
    from hermes_cli import github_workflow
    github_workflow.clear_preflight_cache()
    yield
    github_workflow.clear_preflight_cache()


def test_capability_type_is_provider_neutral():
    assert GitHubCapability.__module__ == "hermes_cli.github_workflow"
    assert GitHubRepository.__module__ == "hermes_cli.github_workflow"


def test_credential_scope_is_explicit():
    from hermes_cli.github_workflow import GITHUB_CREDENTIAL_PURPOSE
    assert GITHUB_CREDENTIAL_PURPOSE == "github-workflow"


def test_config_defaults_are_present():
    from hermes_cli.config_defaults import DEFAULT_CONFIG
    assert DEFAULT_CONFIG["github"]["workflow"]["enabled"] is True
    assert DEFAULT_CONFIG["github"]["workflow"]["api_timeout_seconds"] == 10
    assert DEFAULT_CONFIG["github"]["workflow"]["preflight_cache_seconds"] == 900


def test_capability_has_no_provider_specific_code():
    source = Path("hermes_cli/github_workflow.py").read_text()
    assert "bitwarden" not in source.lower()
    assert "onepassword" not in source.lower()
    assert ".fetch(" not in source
    assert "ErrorKind" in source


def test_auth_state_is_serializable():
    assert json.dumps({"state": AuthState.VERIFIED.value})


def test_public_read_can_skip_preflight(monkeypatch):
    calls = []
    monkeypatch.setattr("hermes_cli.github_workflow.get_secret", lambda name, default=None: "secret")
    monkeypatch.setattr("hermes_cli.github_workflow._preflight", lambda *args: calls.append(args))
    capability = resolve_github_capability(
        config={"github": {"workflow": {"enabled": True}}},
        operation="public-read",
        perform_preflight=False,
    )
    assert calls == []
    assert capability.public_read_available is True


def test_activation_relevant_for_github_repository():
    from hermes_cli.github_workflow import activation_relevant
    assert activation_relevant("anything", GitHubRepository(owner="owner", name="repo"))


def test_activation_not_relevant_for_unrelated_prompt():
    from hermes_cli.github_workflow import activation_relevant
    assert not activation_relevant("Explain Python decorators", GitHubRepository())


def test_workflow_git_env_disables_system_helpers():
    from hermes_cli.github_workflow import workflow_git_env
    env = workflow_git_env({"GIT_ASKPASS": "old", "GIT_TERMINAL_PROMPT": "1"})
    assert env["GIT_CONFIG_NOSYSTEM"] == "1"
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    assert "GIT_ASKPASS" not in env


def test_error_kind_is_reused():
    from hermes_cli.github_workflow import ErrorKind
    assert ErrorKind.NOT_CONFIGURED.value == "not_configured"


def test_capability_status_has_remediation_for_disabled():
    capability = resolve_github_capability(config={"github": {"workflow": {"enabled": False}}})
    assert capability.remediation

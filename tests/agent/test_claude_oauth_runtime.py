"""Contracts for the fail-closed Claude subscription runtime."""

import json
import os
from pathlib import Path

import pytest


def _fake_claude(tmp_path: Path) -> Path:
    script = tmp_path / "claude"
    script.write_text(
        """#!/usr/bin/env python3
import json, os, sys
if sys.argv[1:3] == ['auth', 'status']:
    print(json.dumps({'loggedIn': True, 'authMethod': 'claude.ai', 'apiProvider': 'firstParty'}))
    raise SystemExit(0)
assert '-p' in sys.argv
assert os.environ.get('ANTHROPIC_API_KEY') is None
assert os.environ.get('ANTHROPIC_AUTH_TOKEN') is None
assert '--tools' in sys.argv and sys.argv[sys.argv.index('--tools') + 1] == ''
prompt = sys.stdin.read()
assert 'SYSTEM' in prompt and 'hello' in prompt
print(json.dumps({'type':'result','subtype':'success','is_error':False,'result':'OAUTH_OK','usage':{'input_tokens':3,'output_tokens':1}}))
"""
    )
    script.chmod(0o755)
    return script


def test_claude_oauth_provider_is_not_anthropic_alias():
    from providers import get_provider_profile

    oauth = get_provider_profile("claude-oauth")
    anthropic = get_provider_profile("anthropic")
    assert oauth is not None
    assert oauth.name == "claude-oauth"
    assert oauth.api_mode == "claude_agent_sdk"
    assert "claude-oauth" not in anthropic.aliases
    assert "claude-code" not in anthropic.aliases


def test_runtime_provider_resolves_without_api_credentials(monkeypatch):
    from hermes_cli import runtime_provider as rp

    monkeypatch.setattr(rp, "_get_model_config", lambda: {"provider": "claude-oauth"})
    resolved = rp.resolve_runtime_provider(requested="claude-oauth")
    assert resolved == {
        "provider": "claude-oauth",
        "api_mode": "claude_agent_sdk",
        "base_url": "claude-oauth://local",
        "api_key": "subscription-oauth-external",
        "source": "claude-code-oauth",
        "requested_provider": "claude-oauth",
    }


def test_child_environment_scrubs_payg_credentials(monkeypatch):
    from agent.claude_oauth_runtime import subscription_environment

    monkeypatch.setenv("ANTHROPIC_API_KEY", "payg")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "payg-token")
    env = subscription_environment()
    assert "ANTHROPIC_API_KEY" not in env
    assert "ANTHROPIC_AUTH_TOKEN" not in env


def test_auth_preflight_rejects_non_subscription(tmp_path):
    from agent.claude_oauth_runtime import ClaudeOAuthError, verify_subscription_auth

    script = tmp_path / "claude"
    script.write_text("#!/bin/sh\necho '{\"loggedIn\":true,\"authMethod\":\"api_key\",\"apiProvider\":\"firstParty\"}'\n")
    script.chmod(0o755)
    with pytest.raises(ClaudeOAuthError, match="Claude.ai subscription OAuth"):
        verify_subscription_auth(str(script))


def test_fake_executable_end_to_end_scrubs_env_and_parses_result(tmp_path, monkeypatch):
    from agent.claude_oauth_runtime import run_claude_oauth

    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-leak")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "must-not-leak")
    result = run_claude_oauth(
        cli_path=str(_fake_claude(tmp_path)),
        model="haiku",
        messages=[
            {"role": "system", "content": "SYSTEM"},
            {"role": "user", "content": "hello"},
        ],
    )
    assert result.text == "OAUTH_OK"
    assert result.usage == {"input_tokens": 3, "output_tokens": 1}


def test_cli_failure_does_not_fallback(tmp_path):
    from agent.claude_oauth_runtime import ClaudeOAuthError, run_claude_oauth

    script = tmp_path / "claude"
    script.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = auth ]; then echo '{\"loggedIn\":true,\"authMethod\":\"claude.ai\",\"apiProvider\":\"firstParty\"}'; exit 0; fi\n"
        "echo 'OAuth expired' >&2; exit 1\n"
    )
    script.chmod(0o755)
    with pytest.raises(ClaudeOAuthError, match="OAuth expired"):
        run_claude_oauth(cli_path=str(script), model="haiku", messages=[{"role":"user","content":"x"}])

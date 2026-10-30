"""Native Codex aliases share policy, without changing default/other providers."""

import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("read_only", [False, True])
def test_native_alias_policy_and_other_provider_controls(tmp_path, read_only):
    probe = r'''
import sys
from types import SimpleNamespace
from agent.codex_auth_policy import (
    bind_codex_auth_policy, is_codex_provider, readonly_codex_agent,
    bind_agent_codex_auth, refuse_mutable_codex_pool, reject_readonly_codex_fallback,
)
from hermes_cli.auth import AuthError
bind_codex_auth_policy(read_only=sys.argv[1] == "yes")
readonly = sys.argv[1] == "yes"
for provider in ("openai-codex", "codex", "chatgpt", "chatgpt-codex", " CODEX "):
    assert is_codex_provider(provider)
    pool = object()
    agent = SimpleNamespace(provider=provider, api_mode="codex_responses", _credential_pool=pool)
    assert readonly_codex_agent(agent) is readonly
    assert bind_agent_codex_auth(agent, pool) is (None if readonly else pool)
    error = AuthError("synthetic", provider=provider)
    try:
        reject_readonly_codex_fallback(error)
    except AuthError as exc:
        assert readonly and exc is error
    else:
        assert not readonly
    try:
        refuse_mutable_codex_pool(provider)
    except RuntimeError:
        assert readonly
    else:
        assert not readonly
for provider in ("xai-oauth", "openrouter", "custom:codex", "disguised", None, ""):
    assert not is_codex_provider(provider)
    pool = object()
    agent = SimpleNamespace(provider=provider, api_mode="chat_completions", _credential_pool=pool)
    assert not readonly_codex_agent(agent)
    assert bind_agent_codex_auth(agent, pool) is pool
    reject_readonly_codex_fallback(AuthError("synthetic", provider=provider))
    refuse_mutable_codex_pool(provider)
'''
    result = subprocess.run(
        [sys.executable, "-c", probe, "yes" if read_only else "no"],
        env={**os.environ, "HOME": str(tmp_path), "HERMES_HOME": str(tmp_path / "hermes")},
        capture_output=True, text=True, timeout=45, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

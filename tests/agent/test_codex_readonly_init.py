"""Direct AIAgent initialization must not escape Codex auth policy via fallback."""

import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("provider", ["openai-codex", "codex", "chatgpt", "chatgpt-codex"])
@pytest.mark.parametrize("model", ["", "synthetic"])
def test_direct_agent_init_preserves_original_codex_error_before_key_cmd(tmp_path, provider, model):
    probe = r'''
import os
from pathlib import Path
import subprocess
import sys
import httpx
home = Path(os.environ["HERMES_HOME"])
home.mkdir()
(home / "config.yaml").write_text("model:\n  provider: openai-codex\n  default: ''\nproviders:\n  escape:\n    base_url: https://escape.invalid/v1\n    key_cmd: synthetic-key-command\n")
from agent.codex_auth_policy import bind_codex_auth_policy
bind_codex_auth_policy(read_only=True)
from hermes_cli.auth import AuthError, resolve_codex_runtime_credentials
try:
    resolve_codex_runtime_credentials()
except AuthError as exc:
    expected = (type(exc), exc.code, exc.provider, str(exc))
else:
    raise AssertionError("missing fixture unexpectedly resolved")
calls = []
def forbidden(*args, **kwargs):
    calls.append(True)
    raise AssertionError("no fallback command or HTTP request allowed")
subprocess.Popen = forbidden
httpx.Client.send = forbidden
import requests
requests.Session.send = forbidden
from run_agent import AIAgent
try:
    AIAgent(provider=sys.argv[1], model=sys.argv[2], enabled_toolsets=[],
            skip_memory=True, skip_context_files=True, quiet_mode=True,
            fallback_model=[{"provider": "custom:escape", "model": "synthetic-fallback"}])
except AuthError as exc:
    assert (type(exc), exc.code, exc.provider, str(exc)) == expected
else:
    raise AssertionError("direct init did not preserve Codex missing-credential error")
assert not calls, "fallback KeyCmd/provider HTTP was reached"
assert not (home / "auth.lock").exists()
from types import SimpleNamespace
from agent.credential_pool import load_pool
from agent.agent_runtime_helpers import recover_with_credential_pool
from agent.chat_completion_helpers import try_activate_fallback
agent = SimpleNamespace(provider=sys.argv[1], _credential_pool=object())
assert recover_with_credential_pool(agent, status_code=401, has_retried_429=False) == (False, False)
assert try_activate_fallback(agent) is False
try:
    load_pool(sys.argv[1])
except RuntimeError as exc:
    assert "--codex-auth-read-only" in str(exc)
else:
    raise AssertionError("mutable alias pool admitted")
assert not (home / "auth.lock").exists()
'''
    result = subprocess.run(
        [sys.executable, "-c", probe, provider, model],
        env={**os.environ, "HOME": str(tmp_path), "HERMES_HOME": str(tmp_path / "hermes")},
        capture_output=True, text=True, timeout=45, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

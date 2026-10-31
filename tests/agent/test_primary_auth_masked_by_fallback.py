"""A primary provider whose login failed at resolution time must not be masked by the backup's error.

Field repro: Claude Code's OAuth refresh token expired, resolution fell back to an OpenAI-compatible
backup, the backup answered "You have no credits remaining", and the user was only ever told about
credits. The failed turn must lead with re-authenticating the primary (exact command) AND still say
what is wrong with the backup.

E2E through the real resolution chain (``resolve_runtime_with_fallback``) against a temp
HERMES_HOME, and the real ``AIAgent.run_conversation`` against an in-process fake backup that
returns a billing error.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

_CREDITS_BODY = {"error": {"message": "You have no credits remaining. Add credits to continue.",
                           "type": "insufficient_quota", "code": "insufficient_quota"}}


class _OutOfCredits(BaseHTTPRequestHandler):
    calls = 0

    def do_POST(self):
        type(self).calls += 1
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        body = json.dumps(_CREDITS_BODY).encode()
        self.send_response(402)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_a, **_kw):
        pass


@pytest.fixture()
def backup_url():
    _OutOfCredits.calls = 0
    srv = HTTPServer(("127.0.0.1", 0), _OutOfCredits)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}/v1"
    finally:
        srv.shutdown()


@pytest.fixture()
def home(tmp_path, monkeypatch):
    for var in ("ANTHROPIC_TOKEN", "ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN", "OPENAI_API_KEY",
                "OPENROUTER_API_KEY", "HERMES_INFERENCE_PROVIDER"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


def _write_config(home, backup_url):
    # adopt_external_logins: false keeps the host's own ~/.claude login out of the test: the
    # primary must have no Anthropic credentials at all.
    (home / "config.yaml").write_text(
        "model:\n  provider: anthropic\n  default: claude-opus-4-6\n"
        "auth:\n  adopt_external_logins: false\n"
        "fallback_providers:\n"
        f"  - provider: custom\n    model: gpt-fake\n    base_url: {backup_url}\n    api_key: sk-backup-test\n")


def _resolve_and_run(home, backup_url):
    from hermes_cli.config import load_config
    from hermes_cli.runtime_provider import resolve_runtime_with_fallback

    _write_config(home, backup_url)
    runtime, entry = resolve_runtime_with_fallback(load_config())
    assert entry is not None and entry["model"] == "gpt-fake"

    from agent.primary_auth_fallback import RUNTIME_KEY, attach_primary_auth_failure
    from run_agent import AIAgent

    agent = AIAgent(api_key=runtime["api_key"], base_url=runtime["base_url"], provider=runtime["provider"],
                    api_mode=runtime.get("api_mode"), model=entry["model"], max_iterations=2,
                    enabled_toolsets=[], quiet_mode=True, skip_context_files=True, skip_memory=True,
                    save_trajectories=False, platform="cli")
    attach_primary_auth_failure(agent, runtime.get(RUNTIME_KEY))
    return agent.run_conversation("hello", conversation_history=[], task_id="t")


def test_backup_out_of_credits_leads_with_primary_relogin_and_names_backup_problem(home, backup_url):
    result = _resolve_and_run(home, backup_url)

    assert result["failed"] is True
    assert _OutOfCredits.calls >= 1  # the backup really was asked, and refused for credits
    text = result["final_response"]
    lead = text.split("\n\n", 1)[0]
    # Root cause first: the primary needs a re-login, with the exact command.
    assert "Anthropic" in lead and "hermes auth add anthropic" in lead
    # ...and the backup's own problem is still stated, not dropped.
    assert "out of credits" in lead
    assert "no credits remaining" in text.lower()
    assert result["primary_auth_failure"]["relogin"] == "hermes auth add anthropic"
    assert "sk-backup-test" not in text


def test_primary_serving_again_reports_only_its_own_failure(home, backup_url):
    """No resolution fallback happened: the copy is the provider's own, with no re-login lead."""
    from run_agent import AIAgent

    agent = AIAgent(api_key="sk-x", base_url=backup_url, provider="custom", model="gpt-fake",
                    max_iterations=2, enabled_toolsets=[], quiet_mode=True, skip_context_files=True,
                    skip_memory=True, save_trajectories=False, platform="cli")
    result = agent.run_conversation("hello", conversation_history=[], task_id="t")
    assert result["failed"] is True
    assert "hermes auth add anthropic" not in result["final_response"]
    assert "primary_auth_failure" not in result

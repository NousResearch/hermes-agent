"""``auth.adopt_external_logins: false`` keeps Hermes off the Claude Code login (#113023).

Claude Code's OAuth refresh token is single-use: once Hermes borrows and refreshes it, Hermes and
Claude Code hold one token family and whichever refreshes first logs the other out. With the opt-out
set, Hermes must neither read nor refresh ``~/.claude/.credentials.json``, must drop the pool row an
earlier adopting process persisted, and must say so in ``hermes auth list``.
"""
from __future__ import annotations

import io
import json
import time
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest import mock

import urllib.request

from agent import anthropic_credentials as ac
from agent import credential_sources
from agent.credential_pool import load_pool
from hermes_cli.auth_commands import auth_list_command


def _write_config(hermes_home, adopt):
    body = "model:\n  provider: anthropic\n  model: claude-sonnet-4-5\n"
    if adopt is not None:
        body += f"auth:\n  adopt_external_logins: {'true' if adopt else 'false'}\n# arm {adopt}\n"
    (hermes_home / "config.yaml").write_text(body)


def test_opt_out_never_reads_or_refreshes_claude_code_login(tmp_path, monkeypatch):
    hermes_home, claude_dir = tmp_path / "hermes", tmp_path / "claude"
    hermes_home.mkdir()
    claude_dir.mkdir()
    (hermes_home / ".env").write_text("")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(claude_dir))
    monkeypatch.setattr(credential_sources, "_notice_logged", False, raising=False)
    cred_file = claude_dir / ".credentials.json"
    cred_file.write_text(json.dumps({"claudeAiOauth": {
        "accessToken": "sk-ant-oat01-expired", "refreshToken": "sk-ant-ort01-cc",
        "expiresAt": int(time.time() * 1000) - 3_600_000, "scopes": ["user:inference"]}}))
    original_bytes = cred_file.read_bytes()

    posts: list = []

    def fake_urlopen(req, timeout=None):
        if "oauth/token" in req.full_url:  # the Anthropic token endpoint; other seeders probe other hosts
            posts.append(req.full_url)
        body = json.dumps({"access_token": "sk-ant-oat01-fresh", "refresh_token": "sk-ant-ort01-fresh",
                           "expires_in": 3600}).encode()
        resp = mock.MagicMock()
        resp.read.return_value = body
        resp.__enter__.return_value = resp
        return resp

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    # Arm A (default): today's behaviour — the borrowed login is seeded into the pool.
    _write_config(hermes_home, adopt=None)
    assert [e.source for e in load_pool("anthropic").entries()] == ["claude_code"]

    # Arm B (opt-out): no read, no refresh POST, the persisted row is dropped, the status line explains why.
    _write_config(hermes_home, adopt=False)
    assert ac.resolve_anthropic_token() is None
    assert posts == []
    assert cred_file.read_bytes() == original_bytes
    assert [e.source for e in load_pool("anthropic").entries()] == []
    out = io.StringIO()
    with redirect_stdout(out):
        auth_list_command(SimpleNamespace(provider=None))
    assert credential_sources.EXTERNAL_LOGINS_NOT_ADOPTED_NOTICE in out.getvalue()

    # Arm A again: flipping back adopts (and refreshes) exactly as before.
    _write_config(hermes_home, adopt=True)
    assert ac.resolve_anthropic_token() == "sk-ant-oat01-fresh"
    assert len(posts) == 1
    out = io.StringIO()
    with redirect_stdout(out):
        auth_list_command(SimpleNamespace(provider=None))
    assert credential_sources.EXTERNAL_LOGINS_NOT_ADOPTED_NOTICE not in out.getvalue()

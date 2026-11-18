"""Model discovery / ``/model`` validation at an Azure Foundry ``model.base_url`` must send the
Foundry key, never the Claude Code OAuth token (#154242). Real pool and config in the test
HERMES_HOME; only the HTTP read is recorded."""

from __future__ import annotations

import json

import pytest

from hermes_cli import models as m

FOUNDRY_URL = "https://myres.services.ai.azure.com/anthropic"
OAUTH_TOKEN = "sk-ant-" + "oat01-" + "c" * 24  # concatenated: push protection rejects token-shaped literals


@pytest.fixture
def fx(tmp_path, monkeypatch):
    """HERMES_HOME with a Claude Code login; returns it and the list of (url, headers) the fetcher sends."""
    from hermes_constants import get_hermes_home
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN", "AZURE_ANTHROPIC_KEY", "ANTHROPIC_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    claude_dir = tmp_path / "claude"
    claude_dir.mkdir()
    (claude_dir / ".credentials.json").write_text(json.dumps({"claudeAiOauth": {
        "accessToken": OAUTH_TOKEN, "refreshToken": "r", "expiresAt": 4102444800000}}))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(claude_dir))
    calls: list[tuple[str, dict]] = []

    def record(url, timeout=None, headers=None):
        calls.append((url, dict(headers or {})))
        return {"data": [{"id": "claude-x"}]}

    monkeypatch.setattr(m, "_get_json", record)
    return get_hermes_home(), calls


def _write(home, **model):
    (home / "config.yaml").write_text(json.dumps({"model": {"provider": "anthropic", "default": "claude-x", **model}}))


def _tokens(calls):
    return {v for _, h in calls for k, v in h.items() if k.lower() in ("x-api-key", "authorization")}


def test_catalog_at_foundry_sends_the_azure_env_key(fx, monkeypatch):
    home, sent = fx
    monkeypatch.setenv("AZURE_ANTHROPIC_KEY", "KAZ")
    _write(home, base_url=FOUNDRY_URL)
    m._anthropic_catalog("anthropic", True)
    assert sent and all(url.startswith(FOUNDRY_URL) for url, _ in sent)
    assert _tokens(sent) == {"KAZ"}


def test_catalog_at_foundry_sends_the_key_env_key(fx, monkeypatch):
    home, sent = fx
    monkeypatch.setenv("MY_FOUNDRY_KEY", "KAZ-ENV")
    _write(home, base_url=FOUNDRY_URL, key_env="MY_FOUNDRY_KEY")
    m._anthropic_catalog("anthropic", True)
    assert _tokens(sent) == {"KAZ-ENV"}


def test_validation_fetch_at_foundry_without_a_key_sends_nothing(fx):
    """``/model`` validation passes req.base_url: no Foundry key means no request, not the OAuth token."""
    home, sent = fx
    _write(home, base_url=FOUNDRY_URL)
    assert m._fetch_anthropic_models(base_url=FOUNDRY_URL) is None
    assert sent == []


def test_catalog_off_foundry_still_uses_the_claude_code_login(fx):
    """Control: native Anthropic discovery keeps the OAuth token at api.anthropic.com."""
    home, sent = fx
    _write(home)
    m._anthropic_catalog("anthropic", True)
    assert sent and sent[0][0].startswith("https://api.anthropic.com/")
    assert any(OAUTH_TOKEN in v for v in _tokens(sent))

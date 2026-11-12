"""Native ``anthropic`` with a config-stored Azure Foundry ``model.base_url`` must resolve the
Foundry key, not the Claude Code OAuth token the ``anthropic`` credential pool holds (#154242).

Foundry rejects Claude Code OAuth tokens: ``_anthropic_env_runtime`` says so and picks the Azure
key, but the credential pool answers one rung earlier and pairs its OAuth token with the
configured URL. These cases drive the real ladder with the pool stubbed to the entry a machine
with Claude Code credentials holds.
"""

from __future__ import annotations

import pytest

from hermes_cli import runtime_provider as rp

FOUNDRY_URL = "https://myres.services.ai.azure.com/anthropic"
_AMBIENT_SECRETS = ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "AZURE_ANTHROPIC_KEY")


class _ClaudeCodeEntry:
    """The ``anthropic`` pool entry that Claude Code credentials produce."""

    access_token = "claude-code-oauth-token"
    source = "claude_code"
    base_url = ""


class _ClaudeCodePool:
    def has_credentials(self) -> bool:
        return True

    def select(self, **_kwargs):
        return _ClaudeCodeEntry()


@pytest.fixture
def claude_code_credential(monkeypatch):
    """Claude Code credentials present (via the pool), no ambient Anthropic keys."""
    for name in _AMBIENT_SECRETS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(rp, "load_pool", lambda _provider: _ClaudeCodePool())


def _model_config(monkeypatch, **cfg):
    monkeypatch.setattr(rp, "_get_model_config", lambda: dict(cfg))


def test_foundry_model_api_key_wins_over_the_claude_code_pool(claude_code_credential, monkeypatch):
    _model_config(monkeypatch, provider="anthropic", default="claude-x", base_url=FOUNDRY_URL, api_key="KAZ")
    runtime = rp.resolve_runtime_provider(requested="anthropic")
    assert runtime["api_key"] == "KAZ"
    assert runtime["source"] != "claude_code"
    assert runtime["provider"] == "anthropic"
    assert runtime["api_mode"] == "anthropic_messages"
    assert runtime["base_url"] == FOUNDRY_URL


def test_foundry_azure_env_key_wins_over_the_claude_code_pool(claude_code_credential, monkeypatch):
    monkeypatch.setenv("AZURE_ANTHROPIC_KEY", "KAZ-ENV")
    _model_config(monkeypatch, provider="anthropic", default="claude-x", base_url=FOUNDRY_URL)
    runtime = rp.resolve_runtime_provider(requested="anthropic")
    assert runtime["api_key"] == "KAZ-ENV"
    assert runtime["source"] != "claude_code"


def test_configured_provider_resolves_the_same_way_without_an_explicit_request(claude_code_credential, monkeypatch):
    """``hermes chat`` passes no ``--provider``: the saved provider must not fall into the pool either."""
    _model_config(monkeypatch, provider="anthropic", default="claude-x", base_url=FOUNDRY_URL, api_key="KAZ")
    runtime = rp.resolve_runtime_provider()
    assert runtime["api_key"] == "KAZ"
    assert runtime["source"] != "claude_code"


def test_plain_anthropic_endpoint_still_uses_the_claude_code_credential(claude_code_credential, monkeypatch):
    """Control: the pool IS the right answer for an ``api.anthropic.com`` endpoint."""
    _model_config(monkeypatch, provider="anthropic", default="claude-x")
    runtime = rp.resolve_runtime_provider(requested="anthropic")
    assert runtime["api_key"] == "claude-code-oauth-token"
    assert runtime["source"] == "claude_code"
    assert runtime["base_url"] == "https://api.anthropic.com"


def test_foundry_endpoint_without_any_azure_key_fails_loudly(claude_code_credential, monkeypatch):
    """Never present the OAuth token to Foundry: fail with the Azure-key message instead."""
    _model_config(monkeypatch, provider="anthropic", default="claude-x", base_url=FOUNDRY_URL)
    with pytest.raises(rp.AuthError, match="Azure Anthropic API key"):
        rp.resolve_runtime_provider(requested="anthropic")


def test_providers_entry_owning_the_foundry_url_supplies_its_own_key(claude_code_credential, monkeypatch, tmp_path):
    """A ``providers:`` entry that owns ``model.base_url`` keeps supplying the key, as it did on the
    pool rung: the Foundry skip must hand that decision to ``_anthropic_env_runtime``, not drop it."""
    (tmp_path / "config.yaml").write_text(
        "model:\n"
        "  provider: anthropic\n"
        "  default: claude-x\n"
        f"  base_url: {FOUNDRY_URL}\n"
        "providers:\n"
        "  myfoundry:\n"
        f"    base_url: {FOUNDRY_URL}\n"
        "    api_key: KAZ-ENTRY\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    runtime = rp.resolve_runtime_provider(requested="anthropic")
    assert runtime["api_key"] == "KAZ-ENTRY"
    assert runtime["source"] == "config:providers"
    assert runtime["base_url"] == FOUNDRY_URL

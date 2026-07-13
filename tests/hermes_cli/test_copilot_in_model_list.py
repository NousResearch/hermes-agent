"""Tests for GitHub Copilot entries shown in the /model picker."""

import os
from unittest.mock import patch

import pytest

from hermes_cli.model_switch import list_authenticated_providers


@patch.dict(os.environ, {"GH_TOKEN": "test-key"}, clear=False)
def test_copilot_picker_uses_live_catalog_when_available():
    live_models = ["gpt-5.4", "claude-sonnet-4.6", "gemini-3.1-pro-preview"]

    with patch("agent.models_dev.fetch_models_dev", return_value={}), \
         patch("hermes_cli.models._resolve_copilot_catalog_api_key", return_value="gh-token"), \
         patch("hermes_cli.models._fetch_github_models", return_value=live_models):
        providers = list_authenticated_providers(current_provider="openrouter", max_models=50)

    copilot = next((p for p in providers if p["slug"] == "copilot"), None)

    assert copilot is not None
    assert copilot["models"] == live_models
    assert copilot["total_models"] == len(live_models)


# --- copilot-acp: external_process availability (#63662) -------------------
#
# copilot-acp holds no API key, OAuth token, or credential-pool entry by
# design — the spawned `copilot --acp --stdio` subprocess brings its own auth.
# The picker loop used to filter it out unconditionally (has_creds never had
# an external_process branch), so the provider was invisible in every picker
# even with a perfectly resolvable executable.


@pytest.fixture()
def _no_other_copilot_creds(monkeypatch):
    """Make sure copilot-acp visibility comes ONLY from executable resolution:
    no env tokens, no auth-store entry, no seeded credential pool."""
    for var in ("GH_TOKEN", "GITHUB_TOKEN", "HERMES_COPILOT_ACP_COMMAND", "COPILOT_CLI_PATH"):
        monkeypatch.delenv(var, raising=False)
    import hermes_cli.auth as auth
    import hermes_cli.model_switch as model_switch

    monkeypatch.setattr(auth, "_load_auth_store", lambda: {})
    monkeypatch.setattr(model_switch, "_credential_pool_is_usable", lambda *a, **k: False)


def test_copilot_acp_listed_when_executable_resolves(tmp_path, monkeypatch, _no_other_copilot_creds):
    fake = tmp_path / ("copilot.exe" if os.name == "nt" else "copilot")
    fake.write_text("", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setenv("HERMES_COPILOT_ACP_COMMAND", str(fake))

    with patch("agent.models_dev.fetch_models_dev", return_value={}), \
         patch("hermes_cli.models._resolve_copilot_catalog_api_key", return_value=None), \
         patch("hermes_cli.models._fetch_github_models", return_value=[]):
        providers = list_authenticated_providers(current_provider="openrouter", max_models=50)

    acp = next((p for p in providers if p["slug"] == "copilot-acp"), None)

    assert acp is not None, "copilot-acp must be listed when its executable resolves"
    assert acp["models"], "copilot-acp row must offer at least the curated fallback models"


def test_copilot_acp_hidden_when_executable_missing(monkeypatch, _no_other_copilot_creds):
    # `copilot` may genuinely be installed on a dev machine — force the
    # resolution miss so the test pins behaviour, not the host's PATH.
    import hermes_cli.auth as auth

    monkeypatch.setattr(auth.shutil, "which", lambda *_a, **_k: None)

    with patch("agent.models_dev.fetch_models_dev", return_value={}), \
         patch("hermes_cli.models._resolve_copilot_catalog_api_key", return_value=None), \
         patch("hermes_cli.models._fetch_github_models", return_value=[]):
        providers = list_authenticated_providers(current_provider="openrouter", max_models=50)

    assert all(p["slug"] != "copilot-acp" for p in providers), \
        "copilot-acp must stay hidden when no executable resolves"

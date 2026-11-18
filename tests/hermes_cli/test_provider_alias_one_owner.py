"""A provider alias names one provider in every resolver: credentials, routing and the picker.

Bare ``qwen`` used to mean the Qwen CLI login for credentials (``resolve_provider``, ``hermes auth``)
and Alibaba DashScope for routing and the model catalog (``resolve_provider_full``), so a DashScope
user running ``--provider qwen`` was told the Qwen CLI login was missing.
"""

import pytest

from hermes_cli import auth, auth_commands, providers


@pytest.fixture(autouse=True)
def _offline_catalog(monkeypatch):
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda *a, **k: {})


def test_bare_qwen_is_alibaba_in_every_resolver():
    assert auth.resolve_provider("qwen") == "alibaba"
    assert providers.resolve_provider_full("qwen").id == "alibaba"
    assert providers.normalize_provider("qwen") == "alibaba"
    assert auth_commands._normalize_provider("qwen") == "alibaba"


@pytest.mark.parametrize("alias", ["qwen-oauth", "qwen-portal", "qwen-cli"])
def test_qwen_cli_login_names_still_resolve_to_qwen_oauth(alias):
    assert auth.resolve_provider(alias) == "qwen-oauth"
    assert providers.resolve_provider_full(alias).id == "qwen-oauth"
    assert auth_commands._normalize_provider(alias) == "qwen-oauth"


def test_every_shared_alias_names_the_same_provider_in_both_resolvers():
    """Credential resolution and model routing read two tables; an alias they map to different
    providers is a red test here instead of a runtime "credentials not found"."""
    canon = providers.normalize_provider  # models.dev ids vs Hermes ids (kimi-coding / kimi-for-coding)
    diverging = {}
    for alias in sorted(auth._plugin_aliases()):
        pdef = providers.resolve_provider_full(alias)
        if pdef is None:  # local-server aliases need a configured endpoint
            continue
        credential_side = auth.resolve_provider(alias)
        if canon(credential_side) != canon(pdef.id):
            diverging[alias] = (credential_side, pdef.id)
    assert diverging == {}


@pytest.mark.parametrize("name", ["grok", "or"])
def test_configured_provider_named_like_an_alias_wins_in_auth_commands(tmp_path, monkeypatch, name):
    home = tmp_path / "hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        f"providers:\n  {name}:\n    name: Office Relay\n    base_url: http://127.0.0.1:9999/v1\n",
        encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))

    assert auth_commands._normalize_provider(name) == name

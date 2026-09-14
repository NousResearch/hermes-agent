"""Multiplex invariant: a scoped API key is never paired with the DEFAULT profile's base URL / proxy.

`agent/auxiliary_client.py` already scopes provider keys via ``_scoped_key_env``; the base URLs beside
them (OPENAI_BASE_URL, XAI_BASE_URL, NOUS_INFERENCE_BASE_URL) and the gateway proxy URL / browser
provider endpoints must follow the same rule, or a secondary's key is sent to another profile's host.
"""
from __future__ import annotations

import pytest

from agent import secret_scope


@pytest.fixture
def secondary_scope(monkeypatch):
    for name in ("OPENAI_BASE_URL", "XAI_BASE_URL", "HERMES_XAI_BASE_URL", "NOUS_INFERENCE_BASE_URL",
                 "GATEWAY_PROXY_URL", "FIRECRAWL_API_URL", "BROWSERBASE_BASE_URL", "XAI_API_KEY"):
        monkeypatch.setenv(name, f"https://{name.lower()}.default.example/v1")
    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({"BROWSERBASE_API_KEY": "bb-b", "BROWSERBASE_PROJECT_ID": "pid-b"})
    try:
        yield
    finally:
        secret_scope.reset_secret_scope(token)
        secret_scope.set_multiplex_active(False)


def test_base_urls_follow_the_scoped_key_not_default_environ(monkeypatch, secondary_scope, tmp_path):
    import agent.auxiliary_client as aux
    import plugins.browser.browserbase.provider as browserbase
    import plugins.browser.firecrawl.provider as firecrawl
    import plugins.video_gen.xai as xai_video
    from gateway.run_turn import GatewayTurnMixin
    from hermes_cli import auth_nous

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(aux, "_get_named_custom_provider", lambda name: None, raising=False)

    _, base = aux._expand_direct_api_alias("openai", None)
    assert "default.example" not in (base or "")
    assert aux._scoped_key_env("OPENAI_BASE_URL") == ""
    assert auth_nous._nous_inference_env_override() is None
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {})
    assert GatewayTurnMixin._get_proxy_url(GatewayTurnMixin.__new__(GatewayTurnMixin)) is None
    assert "default.example" not in browserbase.BrowserbaseBrowserProvider()._get_config_or_none()["base_url"]
    assert "default.example" not in firecrawl.FirecrawlBrowserProvider()._api_url()
    # xAI video: a scoped miss stays a miss — no environ fallback-after-miss for the key or the URL.
    api_key, base_url = xai_video._resolve_xai_credentials()
    assert api_key == "" and "default.example" not in base_url


def test_unscoped_single_profile_reads_keep_environ(monkeypatch):
    """Multiplex OFF (CLI / single gateway): environ IS the profile's own value — behaviour unchanged."""
    from gateway.run_turn import GatewayTurnMixin
    from hermes_cli import auth_nous

    secret_scope.set_multiplex_active(False)
    token = secret_scope.set_secret_scope(None)
    try:
        monkeypatch.setenv("GATEWAY_PROXY_URL", "https://proxy.mine.example/")
        monkeypatch.setenv("NOUS_INFERENCE_BASE_URL", "https://nous.mine.example/v1/")
        assert GatewayTurnMixin._get_proxy_url(GatewayTurnMixin.__new__(GatewayTurnMixin)) == "https://proxy.mine.example"
        assert auth_nous._nous_inference_env_override() == "https://nous.mine.example/v1"
    finally:
        secret_scope.reset_secret_scope(token)

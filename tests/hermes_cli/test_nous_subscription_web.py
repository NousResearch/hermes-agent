"""Web feature status follows direct credentials and explicit gateway selection."""

import json

import pytest

from hermes_cli import nous_subscription as ns
from hermes_cli.nous_account import NousPortalAccountInfo


@pytest.fixture
def web_profile(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for name in ("BRAVE_SEARCH_API_KEY", "SEARXNG_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(ns, "_has_agent_browser", lambda: False)

    def configure(web, *, credentials="", enabled=True, gateway_ready=False):
        (tmp_path / ".env").write_text(credentials, encoding="utf-8")
        (tmp_path / "config.yaml").write_text(
            json.dumps({
                "web": web,
                "platform_toolsets": {"cli": ["web"] if enabled else ["file"]},
            }),
            encoding="utf-8",
        )
        monkeypatch.setattr(
            ns, "get_nous_portal_account_info",
            lambda: NousPortalAccountInfo(
                logged_in=gateway_ready, source="jwt" if gateway_ready else "none",
                fresh=False, paid_service_access=gateway_ready,
            ),
        )
        monkeypatch.setattr(ns, "is_managed_tool_gateway_ready", lambda backend: gateway_ready)
        # Real config, credential and toolset loaders; only remote/account and
        # unrelated local browser probes are stubbed.
        return ns.get_nous_subscription_features().web

    return configure


@pytest.mark.parametrize("selection_key", ["backend", "search_backend"])
@pytest.mark.parametrize("has_key", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_direct_brave_status_tracks_credentials_and_toolset(web_profile, selection_key, has_key, enabled):
    web = web_profile(
        {selection_key: "brave-free"},
        credentials="BRAVE_SEARCH_API_KEY=brave-test\n" if has_key else "",
        enabled=enabled,
    )
    assert web.available is has_key
    assert web.active is (has_key and enabled)
    assert web.direct_override is (has_key and enabled)
    assert web.managed_by_nous is False
    assert web.current_provider == "brave-free"
    assert web.explicit_configured is True


@pytest.mark.parametrize("backend,credentials", [
    ("brave-free", "BRAVE_SEARCH_API_KEY=brave-test\n"),
    ("searxng", "SEARXNG_URL=https://search.example.test\n"),
])
@pytest.mark.parametrize("gateway_ready", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
def test_gateway_selection_suppresses_direct_search_credentials(web_profile, backend, credentials, gateway_ready, legacy):
    selection = {"backend": "nous", "search_backend": backend}
    if legacy:
        selection = {"backend": backend, "search_backend": backend, "use_gateway": True}
    web = web_profile(selection, credentials=credentials, gateway_ready=gateway_ready)
    assert web.available is gateway_ready
    assert web.active is gateway_ready
    assert web.managed_by_nous is gateway_ready
    assert web.direct_override is False
    assert web.current_provider == "firecrawl"

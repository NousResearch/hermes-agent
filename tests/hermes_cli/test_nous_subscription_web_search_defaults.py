"""Managed defaults must not undo the direct-search-key preference of web_search."""

import json

import pytest

from hermes_cli import nous_subscription as ns
from hermes_cli.nous_account import NousPortalAccountInfo


@pytest.mark.parametrize("credentials", [
    "BRAVE_SEARCH_API_KEY=brave-test\n",
    "SEARXNG_URL=https://search.example.test\n",
])
def test_managed_defaults_leave_web_alone_for_a_search_only_install(tmp_path, monkeypatch, credentials):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for name in ("BRAVE_SEARCH_API_KEY", "SEARXNG_URL"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / ".env").write_text(credentials, encoding="utf-8")
    (tmp_path / "config.yaml").write_text(json.dumps({}), encoding="utf-8")
    monkeypatch.setattr(ns, "_has_browser_harness", lambda: False)
    monkeypatch.setattr(ns, "is_managed_tool_gateway_ready", lambda backend: True)
    monkeypatch.setattr(
        ns, "get_nous_portal_account_info",
        lambda: NousPortalAccountInfo(logged_in=True, source="jwt", fresh=False, paid_service_access=True),
    )

    config = {"model": {"provider": "nous"}}
    changed = ns.apply_nous_managed_defaults(config, enabled_toolsets=["web"])

    assert "web" not in changed
    assert config.get("web", {}).get("backend") != "nous"

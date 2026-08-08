"""BROWSERBASE_ADVANCED_STEALTH must honor shared truthy aliases."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from plugins.browser.browserbase.provider import BrowserbaseBrowserProvider


def _capture_session_payload(monkeypatch: pytest.MonkeyPatch, stealth_raw: str):
    monkeypatch.setenv("BROWSERBASE_API_KEY", "bb-key")
    monkeypatch.setenv("BROWSERBASE_PROJECT_ID", "proj-1")
    monkeypatch.setenv("BROWSERBASE_ADVANCED_STEALTH", stealth_raw)
    # Default keep-alive/proxies would still be on — disable for a quieter payload.
    monkeypatch.setenv("BROWSERBASE_KEEP_ALIVE", "false")
    monkeypatch.setenv("BROWSERBASE_PROXIES", "false")

    response = MagicMock()
    response.status_code = 200
    response.ok = True
    response.json.return_value = {
        "id": "sess-1",
        "connectUrl": "wss://example/cdp",
        "status": "RUNNING",
    }

    with patch(
        "plugins.browser.browserbase.provider.requests.post",
        return_value=response,
    ) as post:
        BrowserbaseBrowserProvider().create_session("task-1")

    assert post.called
    return post.call_args.kwargs["json"]


@pytest.mark.parametrize("raw", ["true", "1", "yes", "on", "TRUE", " On "])
def test_advanced_stealth_truthy_aliases_enable_setting(monkeypatch, raw):
    payload = _capture_session_payload(monkeypatch, raw)
    assert payload.get("browserSettings") == {"advancedStealth": True}


@pytest.mark.parametrize("raw", ["false", "0", "off", "no", ""])
def test_advanced_stealth_falsy_aliases_skip_setting(monkeypatch, raw):
    payload = _capture_session_payload(monkeypatch, raw)
    assert "browserSettings" not in payload

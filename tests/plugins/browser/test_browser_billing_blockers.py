"""Terminal provider billing failures must not silently fall back or retry forever."""

from unittest.mock import patch

import pytest

from plugins.browser.browserbase.provider import BrowserbaseBrowserProvider
from tools.tool_result_signals import ProviderBillingBlocker


class _Response:
    def __init__(self, status_code: int, payload: dict | None = None):
        self.status_code = status_code
        self.ok = status_code < 400
        self.text = "Payment Required" if status_code == 402 else ""
        self.headers = {}
        self._payload = payload or {"id": "session", "connectUrl": "wss://example.test"}

    def json(self):
        return self._payload


def test_browserbase_recovers_before_terminal_billing_failure(monkeypatch):
    monkeypatch.setenv("BROWSERBASE_API_KEY", "key")
    monkeypatch.setenv("BROWSERBASE_PROJECT_ID", "project")
    provider = BrowserbaseBrowserProvider()

    with patch.object(provider, "_post_create", side_effect=[_Response(402), _Response(200)]):
        session = provider.create_session("task")

    assert session["bb_session_id"] == "session"
    assert session["features"]["keep_alive"] is False


def test_browserbase_raises_only_after_paid_feature_retries_are_exhausted(monkeypatch):
    monkeypatch.setenv("BROWSERBASE_API_KEY", "key")
    monkeypatch.setenv("BROWSERBASE_PROJECT_ID", "project")
    provider = BrowserbaseBrowserProvider()

    with (
        patch.object(provider, "_post_create", side_effect=[_Response(402), _Response(402), _Response(402)]),
        pytest.raises(ProviderBillingBlocker) as exc_info,
    ):
        provider.create_session("task")

    assert exc_info.value.provider == "browserbase"
    assert exc_info.value.status_code == 402


def test_terminal_billing_failure_bypasses_local_browser_fallback(monkeypatch):
    from tools import browser_session

    class _Provider:
        def create_session(self, _cache_key):
            raise ProviderBillingBlocker(
                provider="firecrawl",
                message="Payment Required",
            )

    local = patch.object(browser_session, "_create_local_session")
    with local as local_create, pytest.raises(ProviderBillingBlocker):
        browser_session._create_cloud_session_or_fallback("task", _Provider())

    local_create.assert_not_called()


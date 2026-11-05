"""Web provider adapters preserve explicit HTTP 402 billing blockers."""

from unittest.mock import patch

import pytest

from plugins.web.tavily.provider import TavilyWebSearchProvider
from tools.tool_result_signals import ProviderBillingBlocker


def test_tavily_provider_preserves_explicit_402_as_typed_exception(monkeypatch):
    exc = RuntimeError("Payment Required")
    exc.status_code = 402
    monkeypatch.setenv("TAVILY_API_KEY", "key")

    with patch("plugins.web.tavily.provider._tavily_request", side_effect=exc):
        with pytest.raises(ProviderBillingBlocker) as exc_info:
            TavilyWebSearchProvider().search("billing")

    assert exc_info.value.provider == "tavily"

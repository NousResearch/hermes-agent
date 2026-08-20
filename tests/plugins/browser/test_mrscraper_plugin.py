"""Contract tests for the MrScraper rendered-page plugin."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from plugins.browser.mrscraper import provider as bp
from plugins.mrscraper_client import MrScraperAPIError, MrScraperClient


def test_rendered_request_maps_defaults_and_preserves_false_zero() -> None:
    params, body, timeout = bp.build_rendered_request({
        "url": "https://example.com",
        "max_retries": 0,
        "html": False,
        "markdown": False,
        "home_page": False,
        "wait_for_selector": "",
    })
    assert timeout == 300
    assert params["html"] == "false"
    assert params["markdown"] == "false"
    assert params["blockResources"] == "true"
    assert "screenshot" not in params
    assert "waitForSelector" not in params
    assert body == {
        "url": "https://example.com",
        "maxRetries": 0,
        "tokenCap": 30,
        "homePage": False,
    }


def test_screenshot_mode_is_conditional() -> None:
    params, _body, _timeout = bp.build_rendered_request({
        "url": "https://example.com",
        "screenshot": True,
        "screenshot_mode": "top",
    })
    assert params["screenshot"] == "top"


@pytest.mark.parametrize(
    "args,message",
    [
        ({}, "url is required"),
        ({"url": "x", "timeout": 0}, "timeout must be at least 1"),
        ({"url": "x", "token_cap": 0}, "token_cap must be at least 1"),
        ({"url": "x", "wait_until": "later"}, "wait_until must be one of"),
    ],
)
def test_rendered_validation(args, message) -> None:
    with pytest.raises(Exception, match=message):
        bp.build_rendered_request(args)


def test_rendered_client_puts_token_only_in_query(monkeypatch) -> None:
    response = SimpleNamespace(
        ok=True,
        status_code=200,
        text="<html>ok</html>",
        headers={"Content-Type": "text/html"},
        json=MagicMock(side_effect=ValueError),
    )
    request = MagicMock(return_value=response)
    monkeypatch.setattr("plugins.mrscraper_client.requests.request", request)
    secret = "runtime" + "-test-token"
    result = MrScraperClient(token=secret).fetch_rendered(
        params={"html": "true"},
        body={"url": "https://example.com"},
        timeout=300,
    )
    assert result == "<html>ok</html>"
    assert request.call_args.args[:2] == ("POST", "https://api.mrscraper.com/")
    kwargs = request.call_args.kwargs
    assert kwargs["params"]["token"] == secret
    assert kwargs["params"]["browserRendering"] == "true"
    assert secret not in json.dumps(kwargs["headers"])
    assert kwargs["timeout"] == 330


def test_transport_exception_redacts_token(monkeypatch) -> None:
    import requests

    secret = "runtime" + "-query-token"
    monkeypatch.setattr(
        "plugins.mrscraper_client.requests.request",
        MagicMock(
            side_effect=requests.ConnectionError(
                f"failed https://api.mrscraper.com/?token={secret}"
            )
        ),
    )
    with pytest.raises(MrScraperAPIError) as raised:
        MrScraperClient(token=secret).fetch_rendered(
            params={}, body={"url": "https://example.com"}, timeout=10
        )
    assert secret not in str(raised.value)
    assert "[REDACTED]" in str(raised.value)


def test_browser_plugin_registers_native_tool_not_cdp_provider() -> None:
    import plugins.browser.mrscraper as plugin

    ctx = SimpleNamespace(
        tools=[], register_tool=lambda **kwargs: ctx.tools.append(kwargs)
    )
    plugin.register(ctx)
    assert [item["name"] for item in ctx.tools] == ["mrscraper_fetch_rendered_html"]
    assert not hasattr(plugin, "MrScraperBrowserProvider")


def test_handler_returns_typed_error(monkeypatch) -> None:
    monkeypatch.setattr(
        bp,
        "fetch_rendered_html",
        MagicMock(side_effect=bp.MrScraperError("bad request")),
    )
    assert json.loads(bp.handle_fetch_rendered_html({"url": "x"})) == {
        "error": "bad request"
    }

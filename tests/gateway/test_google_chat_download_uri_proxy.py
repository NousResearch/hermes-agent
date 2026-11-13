"""Google Chat's ``downloadUri`` fallback fetches a host outside the Chat API (``lh3.googleusercontent.com``),
so a NO_PROXY entry that bypasses only the API hosts must not clear the env-proxy redaction of its error."""

import asyncio
import logging
from urllib.parse import unquote

import pytest

from hermes_agent.gateway.config import PlatformConfig

gar = pytest.importorskip("google.auth.transport.requests")

ENV_PROXY_VARS = ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "all_proxy", "NO_PROXY", "no_proxy")
PROXY = "http://bob:qzxqzxqzx@127.0.0.1:1"


class _EchoingSession:
    """A transport that echoes the proxy URL it used into its error, the way httpx/aiohttp/yarl do."""

    def __init__(self, _creds):
        pass

    def get(self, url, **_kw):
        raise ConnectionError(f"Cannot connect to proxy {PROXY} for {url}")


@pytest.mark.parametrize("no_proxy", [".googleapis.com", "googleapis.com", "chat.googleapis.com,oauth2.googleapis.com,pubsub.googleapis.com,www.googleapis.com"])
def test_download_uri_error_hides_the_env_proxy_secret_when_no_proxy_skips_only_the_api(monkeypatch, caplog, no_proxy):
    from hermes_agent.plugins.platforms.google_chat.adapter import GoogleChatAdapter

    for name in ENV_PROXY_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HTTPS_PROXY", PROXY)
    monkeypatch.setenv("NO_PROXY", no_proxy)
    monkeypatch.setattr(gar, "AuthorizedSession", _EchoingSession)
    adapter = GoogleChatAdapter(PlatformConfig(enabled=True, extra={"project_id": "p"}))
    attachment = {"contentType": "image/png", "name": "a.png", "downloadUri": "https://lh3.googleusercontent.com/x/a.png"}
    with caplog.at_level(logging.DEBUG):
        assert asyncio.run(adapter._download_attachment(attachment)) == (None, "image/png")
    assert "downloadUri fetch failed" in caplog.text and " via http://127.0.0.1:1" in caplog.text, caplog.text
    assert "qzx" not in caplog.text and "qzx" not in unquote(caplog.text), caplog.text[-2000:]


def test_download_uri_error_keeps_its_text_when_no_proxy_bypasses_the_fetched_host(monkeypatch, caplog):
    from hermes_agent.plugins.platforms.google_chat.adapter import GoogleChatAdapter

    for name in ENV_PROXY_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HTTPS_PROXY", PROXY)
    monkeypatch.setenv("NO_PROXY", "*")

    class _Plain(_EchoingSession):
        def get(self, url, **_kw):
            raise ConnectionError(f"connection refused for {url}")

    monkeypatch.setattr(gar, "AuthorizedSession", _Plain)
    adapter = GoogleChatAdapter(PlatformConfig(enabled=True, extra={"project_id": "p"}))
    attachment = {"contentType": "image/png", "downloadUri": "https://lh3.googleusercontent.com/x/a.png"}
    with caplog.at_level(logging.DEBUG):
        asyncio.run(adapter._download_attachment(attachment))
    assert "connection refused for" in caplog.text, caplog.text

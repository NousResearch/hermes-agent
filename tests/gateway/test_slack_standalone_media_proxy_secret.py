"""Slack's standalone media branch and its ``conversations.open`` DM lookup reach the model and the
logs with the error text of a request sent through the resolved proxy: a secret in that proxy URL must
not ride along (the text-only branch redacts already)."""

import asyncio
import json
import logging
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
from urllib.parse import unquote

import pytest

pytest.importorskip("aiohttp")

SECRET = "qzxqzxqzx"
PROXY = f"http://bob:{SECRET}#x@127.0.0.1:1"
ENV_PROXY_VARS = ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "all_proxy", "NO_PROXY", "no_proxy")


class _EchoingClient:
    """``AsyncWebClient`` stand-in whose transport echoes the proxy it was given, as aiohttp does."""

    def __init__(self, **_kw):
        self.proxy = None

    async def _fail(self, **_kw):
        raise ConnectionError(f"Cannot connect to host slack.com:443 via proxy {self.proxy}")

    chat_postMessage = files_upload_v2 = _fail


@pytest.fixture
def slack(monkeypatch):
    from hermes_agent.plugins.platforms.slack import adapter as slack_mod

    for name in ENV_PROXY_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HTTPS_PROXY", PROXY)
    async_client = ModuleType("slack_sdk.web.async_client")
    async_client.AsyncWebClient = _EchoingClient
    for name, mod in {"slack_sdk": ModuleType("slack_sdk"), "slack_sdk.web": ModuleType("slack_sdk.web"),
                      "slack_sdk.web.async_client": async_client}.items():
        monkeypatch.setitem(sys.modules, name, mod)
    with patch("hermes_agent.gateway.platforms.base.gateway_trust_env", return_value=True):
        yield slack_mod


def _rendered(caplog):
    formatter = logging.Formatter("%(message)s")
    return "\n".join(formatter.format(r) for r in caplog.records)


def _leaked(*texts):
    return [t for t in texts if SECRET in t or SECRET in unquote(t)]


@pytest.mark.parametrize("caption", [None, "see attached"])
def test_media_send_never_carries_the_proxy_secret(slack, caplog, tmp_path, caption):
    image = tmp_path / "a.png"
    image.write_bytes(b"\x89PNG")
    cfg = SimpleNamespace(token="xoxb-fake", extra={})
    with caplog.at_level(logging.DEBUG):
        result = asyncio.run(slack._standalone_send(cfg, "C123", "hi", media_files=[(str(image), False)], caption=caption))
    text = json.dumps(result)
    assert "ConnectionError via " in text, text  # the proxy-error rendering, not an unrelated failure
    assert not _leaked(text, _rendered(caplog)), (text, _rendered(caplog))


def test_missing_media_caption_fallback_log_never_carries_the_proxy_secret(slack, caplog, tmp_path):
    cfg = SimpleNamespace(token="xoxb-fake", extra={})
    with caplog.at_level(logging.DEBUG):
        asyncio.run(slack._standalone_send(cfg, "C123", "", media_files=[(str(tmp_path / "gone.png"), False)], caption="cap"))
    assert "Caption-fallback send failed" in caplog.text
    assert not _leaked(_rendered(caplog)), _rendered(caplog)


def test_dm_open_log_never_carries_the_proxy_secret(slack, caplog):
    """Real aiohttp through the unparseable proxy URL: its InvalidURL text repeats the password."""
    cfg = SimpleNamespace(token="xoxb-fake", extra={})
    with caplog.at_level(logging.DEBUG):
        result = asyncio.run(asyncio.wait_for(slack._standalone_send(cfg, "U123", "hi"), 30))
    assert "user ID resolution failed" in result["error"], result
    assert "conversations.open exception" in caplog.text, caplog.text
    assert not _leaked(_rendered(caplog)), _rendered(caplog)

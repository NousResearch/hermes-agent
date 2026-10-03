"""End-to-end: a cron delivery forced onto the standalone lane renders QQ markdown.

The gateway and the cron ticker share one process, and cron prefers the live
gateway adapter. When that adapter cannot be used in time (contended event
loop, no adapter, restart) ``_deliver_result`` falls through to
``_deliver_standalone`` -> ``tools.send_message_tool._send_to_platform`` ->
``_send_qqbot``. Only that last hop was shaping the body, and it sent
``msg_type: 0`` for every endpoint — plain text, so ``$...$`` reached the user
as raw LaTeX on exactly the fallback that runs when the gateway is busy.

The tools-level payload test covers ``_send_qqbot`` in isolation; this one pins
the CHAIN, because the fallback is the only reason the standalone sender is
reached at all. Nothing between ``_deliver_result`` and the request may drop the
per-platform ``extra`` that carries ``markdown_support``: the scheduled body is
asserted here on a fake ``httpx``, with the real sender, the real chunking and
the real cron fallback in the path.
"""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from cron.scheduler import _deliver_result
from gateway.config import Platform, PlatformConfig
from gateway.platforms.qqbot.constants import MSG_TYPE_MARKDOWN

OPENID = "A1B2C3D4E5F60718293A4B5C6D7E8F90"
MESSAGE = "Latex renders here: $a\\leqslant b$"
TOKEN_URL = "https://bots.qq.com/app/getAppAccessToken"
C2C_URL = f"https://api.sgroup.qq.com/v2/users/{OPENID}/messages"


class _Response:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = {} if payload is None else payload

    def json(self):
        return self._payload


@pytest.fixture
def qqbot_requests(monkeypatch):
    """Fake ``httpx`` for the real sender; returns the recorded POSTs."""
    calls = []

    class _AsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, **kwargs):
            body = kwargs.get("json") or {}
            calls.append((url, body))
            if url == TOKEN_URL:
                return _Response(200, {"access_token": "tok"})
            if url == C2C_URL:
                return _Response(200, {"id": "msg-1"})
            return _Response(400, {"code": 11263, "message": "频道不存在"})

    module = types.ModuleType("httpx")
    module.AsyncClient = _AsyncClient
    monkeypatch.setitem(sys.modules, "httpx", module)
    return calls


def _qqbot_config(**extra):
    config = MagicMock()
    config.platforms = {Platform.QQBOT: PlatformConfig(
        enabled=True, extra={"app_id": "app", "client_secret": "sec", **extra})}
    config.get_home_channel = lambda p: None
    return config


def _deliver_via_fallback(content=MESSAGE, **extra):
    """Drive ``_deliver_result`` with NO live adapters — the standalone lane, which is
    what a busy gateway falls through to. The real ``_send_to_platform``/``_send_qqbot``
    run; only ``httpx`` is faked."""
    job = {"id": "job-qqbot", "name": "Formula", "deliver": "origin",
           "origin": {"platform": "qqbot", "chat_id": OPENID}}
    with patch("gateway.config.load_gateway_config", return_value=_qqbot_config(**extra)), \
         patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}), \
         patch("cron.scheduler_delivery._record_delivery_verification"):
        return _deliver_result(job, content, adapters={}, loop=MagicMock())


def _message_posts(calls):
    return [(url, body) for url, body in calls if url != TOKEN_URL]


def test_fallback_lane_posts_a_markdown_body(qqbot_requests):
    error = _deliver_via_fallback()

    assert error is None, error
    posts = _message_posts(qqbot_requests)
    assert posts, "the standalone lane never reached the QQ API"
    url, body = posts[-1]
    assert url == C2C_URL
    assert body["msg_type"] == MSG_TYPE_MARKDOWN
    assert body["markdown"]["content"] == MESSAGE


def test_fallback_lane_honors_markdown_support_false(qqbot_requests):
    """The per-platform ``extra`` must survive the whole cron path, not just the
    last hop: a QQ deployment with markdown off still gets plain text."""
    error = _deliver_via_fallback(markdown_support=False)

    assert error is None, error
    url, body = _message_posts(qqbot_requests)[-1]
    assert body == {"content": MESSAGE, "msg_type": 0}

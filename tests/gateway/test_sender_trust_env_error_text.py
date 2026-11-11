"""``gateway.trust_env: false`` builds aiohttp sessions that never read the proxy environment, so a
LINE / Weixin send goes direct and its real API error (status, errcode) is kept even when an
inherited ``HTTPS_PROXY`` holds a secret. With the default ``trust_env: true`` the same send goes
through that proxy and only ``<ExceptionType> via <host:port>`` is shown.

Each case drives the real standalone sender against a local aiohttp server.
"""

import asyncio
import json
from types import SimpleNamespace
from urllib.parse import unquote

import pytest

web = pytest.importorskip("aiohttp.web")

_SECRET = "http://bob:qzxqzxqzx@127.0.0.1:1"
_LINE_BODY = '{"message":"Authentication failed due to invalid access token"}'
_WEIXIN_BODY = json.dumps({"ret": -1, "errcode": 40013, "errmsg": "invalid appid"})


async def _serve(status, body, send):
    async def handler(request):
        return web.Response(status=status, text=body, content_type="application/json")

    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        return await asyncio.wait_for(send(f"http://127.0.0.1:{port}"), 60)
    finally:
        await runner.cleanup()


def _line(monkeypatch):
    from plugins.platforms.line import adapter as line

    async def send(base):
        monkeypatch.setattr(line, "LINE_PUSH_URL", f"{base}/v2/bot/message/push")
        return await line._standalone_send(SimpleNamespace(extra={"channel_access_token": "tok"}), "U123", "hi")

    return _serve(401, _LINE_BODY, send), "LINE push 401: " + _LINE_BODY


def _weixin(monkeypatch):
    from plugins.platforms.weixin import adapter as wx

    if not wx.check_weixin_requirements():
        pytest.skip("weixin needs aiohttp + cryptography")

    async def send(base):
        extra = {"account_id": "acct", "send_chunk_retries": "0", "base_url": base, "cdn_base_url": base}
        return await wx._standalone_send(SimpleNamespace(extra=extra, token="tok"), "wxid_x", "hi")

    return _serve(200, _WEIXIN_BODY, send), "errcode=40013"


@pytest.mark.parametrize("trust_env", [False, True])
@pytest.mark.parametrize("sender", [_line, _weixin], ids=["line", "weixin"])
def test_trust_env_false_keeps_the_real_api_error(tmp_path, monkeypatch, sender, trust_env):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(f"gateway:\n  trust_env: {str(trust_env).lower()}\n")
    for name in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "all_proxy",
                 "NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HTTPS_PROXY", _SECRET)
    monkeypatch.setenv("HTTP_PROXY", _SECRET)
    run, real_text = sender(monkeypatch)
    error = str(asyncio.run(run).get("error") or "")
    assert "qzx" not in error and "qzx" not in unquote(error), error
    if trust_env:  # proxied through the dead secret proxy: redacted to type + host:port
        assert " via http://127.0.0.1:1" in error and real_text not in error, error
    else:
        assert real_text in error, error


@pytest.mark.parametrize("scheme", ["ftp", "socks4"])
def test_wecom_connect_under_trust_env_false_never_shows_the_proxy_secret(tmp_path, monkeypatch, caplog, scheme):
    """Redaction is skipped under ``trust_env: false``, so every transport WeComAdapter.connect
    builds must also skip the environment: an httpx client reading an unsupported-scheme
    ``HTTPS_PROXY`` raises with the raw proxy URL in its text."""
    import logging

    from gateway.config import PlatformConfig
    from plugins.platforms.wecom.adapter import WeComAdapter

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("gateway:\n  trust_env: false\n")
    for name in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "all_proxy",
                 "NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HTTPS_PROXY", f"{scheme}://qzxqzxqzx@127.0.0.1:1")
    adapter = WeComAdapter(PlatformConfig(enabled=True, extra={
        "bot_id": "b", "secret": "s", "websocket_url": "ws://127.0.0.1:1/"}))
    with caplog.at_level(logging.DEBUG):
        assert asyncio.run(adapter.connect()) is False
    assert "qzx" not in (adapter.fatal_error_message or ""), adapter.fatal_error_message
    assert "qzx" not in caplog.text, caplog.text  # formatted records, tracebacks included

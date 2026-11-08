"""A send that fails through an env proxy (``HTTPS_PROXY`` / ``ALL_PROXY``) whose URL holds a secret
returns and logs only ``<ExceptionType> via <scheme://host:port | placeholder>``: requests, httpx,
aiohttp (``trust_env``) and websockets echo the malformed URL into their error text, and the send
result reaches the model, which may repeat it in chat.

Each case drives the real sender with the real transport against an unreachable proxy.
"""

import asyncio
import logging
import re
from types import SimpleNamespace
from urllib.parse import unquote

import pytest

from tests.gateway.test_proxy_error_text_not_logged import PASSWORDS, SHAPES

pytest.importorskip("aiohttp")

ENV_PROXY_VARS = ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "all_proxy")


def _feishu(monkeypatch):
    requests = pytest.importorskip("requests")
    from plugins.platforms.feishu import adapter as fs

    def _create(request):  # the SDK's transport: a plain requests call that reads HTTPS_PROXY
        requests.post("https://open.feishu.cn/open-apis/im/v1/messages", json={}, timeout=5)

    client = SimpleNamespace(im=SimpleNamespace(v1=SimpleNamespace(message=SimpleNamespace(create=_create))))
    monkeypatch.setattr(fs, "_load_lark_oapi", lambda: True)
    monkeypatch.setattr(fs.FeishuAdapter, "_build_lark_client", lambda self, domain: client)
    monkeypatch.setattr(fs.asyncio, "sleep", _no_sleep)  # the 3-attempt retry backoff
    cfg = SimpleNamespace(token=None, api_key=None, enabled=True, extra={"app_id": "cli_x", "app_secret": "s"})
    return fs._standalone_send(cfg, "oc_x", "hi")


def _google_chat(monkeypatch):
    pytest.importorskip("google.auth.transport.requests")
    from plugins.platforms.google_chat import adapter as gc

    monkeypatch.setattr(gc, "service_account", object())
    monkeypatch.setattr(gc, "_load_sa_credentials_from", lambda value: SimpleNamespace(refresh=lambda req: None, token="t"))
    return gc._standalone_send(SimpleNamespace(extra={"service_account_json": "{}"}), "spaces/AAA", "hi")


def _teams(monkeypatch):
    from plugins.platforms.teams import adapter as teams

    extra = {"client_id": "c", "client_secret": "s", "tenant_id": "t1"}
    return teams._standalone_send(SimpleNamespace(extra=extra), "19:abc@thread.v2", "hi")


def _weixin(monkeypatch):
    from plugins.platforms.weixin import adapter as wx

    if not wx.check_weixin_requirements():
        pytest.skip("weixin needs aiohttp + cryptography")
    extra = {"account_id": "acct", "send_chunk_retries": "0"}
    return wx._standalone_send(SimpleNamespace(extra=extra, token="tok"), "wxid_x", "hi")


def _line(monkeypatch):
    from plugins.platforms.line import adapter as line

    return line._standalone_send(SimpleNamespace(extra={"channel_access_token": "tok"}), "U123", "hi")


def _ntfy(monkeypatch):
    from plugins.platforms.ntfy import adapter as ntfy

    return ntfy._standalone_send(SimpleNamespace(extra={"server": "https://ntfy.example.com", "topic": "t"}), "t", "hi")


def _dingtalk(monkeypatch):
    from plugins.platforms.dingtalk import adapter as dt

    return dt._standalone_send(SimpleNamespace(extra={"webhook_url": "https://oapi.dingtalk.com/robot/send"}), "c", "hi")


def _simplex(monkeypatch):
    pytest.importorskip("websockets")
    from plugins.platforms.simplex import adapter as sx

    return sx._standalone_send(SimpleNamespace(extra={"ws_url": "ws://127.0.0.1:5225"}), "@alice", "hi")


def _signal(monkeypatch):
    from plugins.platforms.signal import standalone as sig

    return sig.standalone_send(SimpleNamespace(extra={"http_url": "http://127.0.0.1:9", "account": "+15550000000"}),
                               "+15550000001", "hi")


def _qqbot(monkeypatch):
    from plugins.platforms.qqbot import send as qq

    return qq.send_qqbot(SimpleNamespace(token="secret", extra={"app_id": "1"}), "c2c:abc", "hi")


async def _no_sleep(*_args, **_kwargs):
    return None


SENDERS = {"feishu": _feishu, "google_chat": _google_chat, "teams": _teams, "weixin": _weixin, "line": _line,
           "ntfy": _ntfy, "dingtalk": _dingtalk, "simplex": _simplex, "signal": _signal, "qqbot": _qqbot}
CASES = [pytest.param(sender, shape.format(pw=pw), pw, id=f"{sender}|{pw}|{shape.split(':')[0]}")
         for sender in SENDERS for pw in PASSWORDS for shape in SHAPES]


@pytest.mark.parametrize("sender, proxy, password", CASES)
def test_env_proxy_send_failure_never_carries_the_proxy_secret(monkeypatch, caplog, sender, proxy, password):
    for name in ENV_PROXY_VARS:
        monkeypatch.setenv(name, proxy)
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    with caplog.at_level(logging.DEBUG):
        result = asyncio.run(asyncio.wait_for(SENDERS[sender](monkeypatch), 60))
    error = str(result.get("error") or "")
    # Hermes' own lines: httpcore's DEBUG trace prints ``socket_options=None`` (matches ``socks5``).
    logs = "\n".join(r.getMessage() for r in caplog.records if r.name.split(".")[0] in ("plugins", "gateway", "tools"))
    assert error, result
    # The exception type, on the proxy-error path (Signal's result after retries names only the batch).
    assert re.search(r"\b[A-Z]\w+ via ", error + "\n" + logs), (result, logs[-2000:])
    for text in (error, logs):
        pieces = {password[i:j] for i in range(len(password)) for j in range(i + 3, len(password) + 1)
                  if len(password[i:j].replace(" ", "")) >= 3}
        leaked = sorted(p for p in pieces if p in text or p in unquote(text))
        assert not leaked, (leaked, text[-2000:])

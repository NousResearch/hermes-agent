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


_SECRET = "http://bob:qzxqzxqzx@127.0.0.1:1"
_CLEAN = "http://127.0.0.1:2"
# Each sender's own API hosts: NO_PROXY naming them means the request went direct.
_TARGET_HOSTS = ("127.0.0.1,localhost,.feishu.cn,.weixin.qq.com,.line.me,ntfy.example.com,.dingtalk.com,.qq.com,"
                 ".googleapis.com,.botframework.com,.trafficmanager.net,.microsoftonline.com")
# Variables that disagree: the library may read the secret one whichever Hermes would have picked.
# A NO_PROXY naming the targets: httpx still parses every proxy variable when the client is built.
DISAGREEING = {"https_clean_http_secret": {"HTTPS_PROXY": _CLEAN, "HTTP_PROXY": _SECRET},
               "upper_clean_lower_secret": {"HTTPS_PROXY": _CLEAN, "HTTP_PROXY": _CLEAN,
                                            "https_proxy": _SECRET, "http_proxy": _SECRET},
               "secret_no_proxy_targets": {"HTTPS_PROXY": _SECRET, "HTTP_PROXY": _SECRET, "NO_PROXY": _TARGET_HOSTS}}
BYPASSED = {"no_proxy_star": "*", "no_proxy_targets": _TARGET_HOSTS}


def _send_under(monkeypatch, caplog, sender, env):
    for name in (*ENV_PROXY_VARS, "NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    with caplog.at_level(logging.DEBUG):
        result = asyncio.run(asyncio.wait_for(SENDERS[sender](monkeypatch), 60))
    logs = "\n".join(r.getMessage() for r in caplog.records if r.name.split(".")[0] in ("plugins", "gateway", "tools"))
    return str(result.get("error") or ""), logs


@pytest.mark.parametrize("layout", DISAGREEING)
@pytest.mark.parametrize("sender", SENDERS)
@pytest.mark.parametrize("shape", SHAPES)
def test_a_secret_in_any_proxy_variable_never_reaches_the_send_result(monkeypatch, caplog, sender, layout, shape):
    secret = shape.format(pw="qzxqzxqzx")
    env = {name: secret if value == _SECRET else value for name, value in DISAGREEING[layout].items()}
    error, logs = _send_under(monkeypatch, caplog, sender, env)
    assert error
    for text in (error, logs):
        assert "qzx" not in text and "qzx" not in unquote(text), text[-2000:]


@pytest.mark.parametrize("layout", BYPASSED)
@pytest.mark.parametrize("sender", sorted(set(SENDERS) - {"signal"}))  # Signal's result names only the batch
def test_a_send_that_no_proxy_bypasses_keeps_its_real_error(monkeypatch, caplog, sender, layout):
    error, _ = _send_under(monkeypatch, caplog, sender, {"HTTPS_PROXY": _SECRET, "HTTP_PROXY": _SECRET,
                                                          "NO_PROXY": BYPASSED[layout]})
    assert error and not re.search(r"\b[A-Z]\w+ via ", error), error


@pytest.mark.parametrize("sender", sorted(set(SENDERS) - {"signal"}))  # Signal's result names only the batch
def test_a_bare_secret_free_proxy_keeps_its_real_error(monkeypatch, caplog, sender):
    """``HTTPS_PROXY=host:port`` (no scheme, no secret) is read as ``http://`` by httpx and requests:
    nothing to hide, so the real error stays (#151811)."""
    error, _ = _send_under(monkeypatch, caplog, sender, {"HTTPS_PROXY": "127.0.0.1:2", "HTTP_PROXY": "127.0.0.1:2"})
    assert error and "<unparseable URL redacted>" not in error and not re.search(r"\b[A-Z]\w+ via ", error), error


@pytest.mark.parametrize("no_proxy", [" *", "* ", "*\t"])
@pytest.mark.parametrize("sender", SENDERS)
@pytest.mark.parametrize("shape", SHAPES)
def test_a_padded_star_no_proxy_still_redacts(monkeypatch, caplog, sender, no_proxy, shape):
    # urllib, requests, aiohttp and httplib2 treat only an exact ``*`` as "bypass every host".
    secret = shape.format(pw="qzxqzxqzx")
    error, logs = _send_under(monkeypatch, caplog, sender, {"HTTPS_PROXY": secret, "HTTP_PROXY": secret,
                                                            "NO_PROXY": no_proxy})
    assert error
    for text in (error, logs):
        assert "qzx" not in text and "qzx" not in unquote(text), text[-2000:]


def test_simplex_trailing_dot_host_is_not_bypassed_by_its_bare_name(monkeypatch, caplog):
    # No library strips the trailing dot: ``localhost.`` is proxied, and websockets echoes a proxy with a path.
    pytest.importorskip("websockets")
    from plugins.platforms.simplex import adapter as sx

    secret = "http://bob:qzxqzxqzx@127.0.0.1:1/p"
    for name in (*ENV_PROXY_VARS, "NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HTTPS_PROXY", secret)
    monkeypatch.setenv("HTTP_PROXY", secret)
    monkeypatch.setenv("NO_PROXY", "localhost")
    result = asyncio.run(asyncio.wait_for(
        sx._standalone_send(SimpleNamespace(extra={"ws_url": "ws://localhost.:5225"}), "@alice", "hi"), 60))
    error = str(result.get("error") or "")
    assert re.search(r"\bInvalidProxy via ", error) and "qzx" not in unquote(error), error


@pytest.mark.parametrize("no_proxy, ws_url", [("bücher.example", "ws://bücher.example:5225"),
                                              ("xn--bcher-kva.example", "ws://xn--bcher-kva.example:5225")])
def test_simplex_idna_host_is_never_treated_as_bypassed(monkeypatch, caplog, no_proxy, ws_url):
    # urllib compares NO_PROXY raw; websockets/httpx/aiohttp IDNA-encode the host first, so they disagree.
    pytest.importorskip("websockets")
    from plugins.platforms.simplex import adapter as sx

    secret = "http://bob:qzxqzxqzx@127.0.0.1:1/p"
    for name in (*ENV_PROXY_VARS, "NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    for name, value in (("HTTPS_PROXY", secret), ("HTTP_PROXY", secret), ("NO_PROXY", no_proxy)):
        monkeypatch.setenv(name, value)
    with caplog.at_level(logging.DEBUG):
        result = asyncio.run(asyncio.wait_for(
            sx._standalone_send(SimpleNamespace(extra={"ws_url": ws_url}), "@alice", "hi"), 60))
    error = str(result.get("error") or "")
    logs = "\n".join(logging.Formatter("%(message)s").format(r) for r in caplog.records)
    assert re.search(r"\b\w+ via http://127.0.0.1:1\b", error), error
    for text in (error, logs):
        assert "qzx" not in text and "qzx" not in unquote(text), text[-2000:]


def test_an_oddly_cased_no_proxy_alone_is_not_a_bypass(monkeypatch):
    # httplib2 reads only ``no_proxy`` / ``NO_PROXY``: a ``No_Proxy`` it ignores cannot prove a direct send.
    from gateway.platforms.base_log_safety import env_proxy_for_redaction

    for name in (*ENV_PROXY_VARS, "NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HTTPS_PROXY", _SECRET)
    monkeypatch.setenv("No_Proxy", "*")
    assert env_proxy_for_redaction("https://chat.googleapis.com") == _SECRET
    monkeypatch.setenv("NO_PROXY", "*")
    assert env_proxy_for_redaction("https://chat.googleapis.com") is None


def test_feishu_media_download_failure_log_carries_no_proxy_secret(monkeypatch, caplog):
    from gateway.config import PlatformConfig
    from gateway.platforms.base import SendResult as _SendResult
    from plugins.platforms.feishu import adapter as fs

    for name in (*ENV_PROXY_VARS, "NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://bob:qzxqzxqzx#x@127.0.0.1:1")
    monkeypatch.setenv("HTTP_PROXY", "http://bob:qzxqzxqzx#x@127.0.0.1:1")
    monkeypatch.setattr("tools.url_safety.is_safe_url", lambda url: True)
    adapter = fs.FeishuAdapter(PlatformConfig(enabled=True, extra={"app_id": "cli_x", "app_secret": "s"}))

    async def _send(**_kwargs):
        return _SendResult(success=True)

    monkeypatch.setattr(adapter, "send", _send)
    formatter = logging.Formatter("%(levelname)s %(name)s %(message)s")
    for call in (lambda: adapter.send_image("oc_x", "https://example.com/a.png"),
                 lambda: adapter.send_animation("oc_x", "https://example.com/a.gif")):
        caplog.clear()
        with caplog.at_level(logging.DEBUG):
            asyncio.run(asyncio.wait_for(call(), 60))
        logs = "\n".join(formatter.format(r) for r in caplog.records if "feishu" in r.name)
        assert "Failed to download" in logs, logs
        assert "qzx" not in logs and "qzx" not in unquote(logs), logs[-2000:]


@pytest.mark.parametrize("scheme", ["ftp", "socks4"])
@pytest.mark.parametrize("sender", SENDERS)
def test_an_unsupported_scheme_secret_proxy_is_redacted_even_when_no_proxy_names_the_target(monkeypatch, caplog,
                                                                                             sender, scheme):
    # httpx rejects the scheme while building the client, before it reads NO_PROXY, and its
    # ``Unknown scheme for proxy URL`` error keeps the username (only the password is masked).
    secret = f"{scheme}://qzxUser:qzxPass@127.0.0.1:1"
    error, logs = _send_under(monkeypatch, caplog, sender, {"HTTPS_PROXY": secret, "HTTP_PROXY": secret,
                                                            "NO_PROXY": _TARGET_HOSTS})
    assert error
    for text in (error, logs):
        assert "qzx" not in text and "qzx" not in unquote(text), text[-2000:]

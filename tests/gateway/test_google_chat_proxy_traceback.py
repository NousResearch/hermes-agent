"""Google Chat's send/edit/delete DEBUG lines carry no env-proxy secret in their rendered
traceback: the result already hides the error text, and ``exc_info`` repeats it."""

import asyncio
import logging
from unittest.mock import MagicMock
from urllib.parse import unquote

import pytest

from hermes_agent.gateway.config import PlatformConfig

requests = pytest.importorskip("requests")
http_error = pytest.importorskip("googleapiclient.errors").HttpError

ENV_PROXY_VARS = ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "all_proxy", "NO_PROXY", "no_proxy")
SECRET_PROXIES = ["http://bob:qzxqzxqzx#x@127.0.0.1:1", "bob:qzxqzxqzx@127.0.0.1:1"]


def _through_env_proxy(*_args, **_kwargs):  # the Google client's transport: requests reading HTTPS_PROXY
    requests.post("https://chat.googleapis.com/v1/spaces/S/messages", json={}, timeout=5)


def _adapter():
    from hermes_agent.plugins.platforms.google_chat.adapter import GoogleChatAdapter

    adapter = GoogleChatAdapter(PlatformConfig(enabled=True, extra={"project_id": "p"}))
    adapter._chat_api = MagicMock()
    adapter._chat_api.spaces().messages().delete().execute.side_effect = _through_env_proxy
    adapter._new_authed_http = MagicMock()
    adapter._create_message = _async(_through_env_proxy)
    adapter._patch_message = _async(_through_env_proxy)
    return adapter


def _async(fn):
    async def _call(*args, **kwargs):
        return fn(*args, **kwargs)
    return _call


SENDS = {
    "send_card": lambda a: a.send_card("spaces/S", {"cardId": "c", "card": {}}),
    "edit_message": lambda a: a.edit_message("spaces/S", "spaces/S/messages/M", "hi"),
    "delete_message": lambda a: a.delete_message("spaces/S", "spaces/S/messages/M"),
    "typing_card_patch": lambda a: a._patch_quietly("spaces/S/messages/M", "·", "[GoogleChat] typing_card_patch failed"),
}


def _rendered(caplog):
    """Records as a file handler writes them: message plus the formatted traceback."""
    formatter = logging.Formatter("%(levelname)s %(name)s %(message)s")
    return "\n".join(formatter.format(r) for r in caplog.records if "google_chat" in r.name)


def _send(monkeypatch, caplog, op, env):
    from hermes_agent.plugins.platforms.google_chat import adapter as gc

    monkeypatch.setattr(gc, "HttpError", http_error)  # as bound once connect() loads the Google client
    for name in ENV_PROXY_VARS:
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    with caplog.at_level(logging.DEBUG):
        result = asyncio.run(SENDS[op](_adapter()))
    return result, _rendered(caplog)


@pytest.mark.parametrize("op", SENDS)
@pytest.mark.parametrize("proxy", SECRET_PROXIES)
def test_debug_traceback_never_carries_the_env_proxy_secret(monkeypatch, caplog, op, proxy):
    result, logs = _send(monkeypatch, caplog, op, {"HTTPS_PROXY": proxy, "HTTP_PROXY": "http://127.0.0.1:2"})
    error = getattr(result, "error", None) or ""
    assert f"[GoogleChat] {op} failed" in logs, logs
    if hasattr(result, "error"):  # a SendResult; delete returns False and patch None
        assert not result.success and " via " in error, result
    for text in (error, logs):
        assert "qzx" not in text and "qzx" not in unquote(text), text[-2000:]


@pytest.mark.parametrize("op", SENDS)
def test_debug_traceback_kept_through_a_proxy_holding_no_secret(monkeypatch, caplog, op):
    _, logs = _send(monkeypatch, caplog, op, {"HTTPS_PROXY": "http://127.0.0.1:1"})
    assert "Traceback (most recent call last)" in logs, logs


def test_httplib2_unparseable_secret_redacts_even_under_no_proxy_star(monkeypatch, caplog):
    # httplib2 parses the proxy URL (``Port could not be cast``) before it reads NO_PROXY=*.
    httplib2 = pytest.importorskip("httplib2")

    def _through_httplib2(*_args, **_kwargs):
        httplib2.Http(timeout=3).request("https://chat.googleapis.com/v1/spaces/S/messages", "POST")

    adapter = _adapter()
    adapter._create_message = _async(_through_httplib2)
    from hermes_agent.plugins.platforms.google_chat import adapter as gc

    monkeypatch.setattr(gc, "HttpError", http_error)
    for name in ENV_PROXY_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://bob:qzxqzxqzx#x@127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "*")
    with caplog.at_level(logging.DEBUG):
        result = asyncio.run(SENDS["send_card"](adapter))
    logs = _rendered(caplog)
    assert not result.success and " via " in (result.error or ""), result
    for text in (result.error or "", logs):
        assert "qzx" not in text and "qzx" not in unquote(text), text[-2000:]

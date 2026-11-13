"""A standalone (out-of-process) send through a proxy whose URL carries a secret never returns that
secret in the tool result: the result text reaches the model, which may repeat it in chat.

Drives each real ``_standalone_send`` with real aiohttp against an unreachable proxy URL."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import unquote

import pytest

pytest.importorskip("aiohttp")

PASSWORD = "p{a}ss|w0rd^99"
SHAPES = ["bob:{pw}@127.0.0.1:1", "http://bob:{pw}#x@127.0.0.1:1"]
ENV_PROXY_VARS = ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "all_proxy")


def _leaks(text: str) -> list[str]:
    pieces = {PASSWORD[i:j] for i in range(len(PASSWORD)) for j in range(i + 4, len(PASSWORD) + 1)}
    return sorted(p for p in pieces if p in text or p in unquote(text))


def _mattermost(monkeypatch, proxy):
    from hermes_agent.plugins.platforms.mattermost import adapter as mm

    monkeypatch.setenv("MATTERMOST_PROXY", proxy)
    cfg = SimpleNamespace(token="tok", extra={"url": "http://127.0.0.1:9"}, api_key=None)
    return mm._standalone_send(cfg, "chan", "hi")


def _sms(monkeypatch, proxy):
    from hermes_agent.plugins.platforms.sms import adapter as sms

    secrets = {"TWILIO_ACCOUNT_SID": "AC1", "TWILIO_PHONE_NUMBER": "+15550000000", "TWILIO_AUTH_TOKEN": "t"}
    monkeypatch.setattr(sms, "_get_scoped_secret", lambda key, default="": secrets.get(key, default))
    return sms._standalone_send(SimpleNamespace(api_key="t"), "+15550000001", "hi")


def _slack(monkeypatch, proxy):
    from hermes_agent.plugins.platforms.slack import adapter as slack

    return slack._standalone_send(SimpleNamespace(token="xoxb-fake", extra={}), "C123", "hi")


def _discord(monkeypatch, proxy):
    from hermes_agent.plugins.platforms.discord import adapter as dc

    monkeypatch.setenv("DISCORD_PROXY", proxy)
    return dc._standalone_send(SimpleNamespace(token="bot-token"), "123", "hi", thread_id="456")


SENDERS = {"mattermost": _mattermost, "sms": _sms, "slack": _slack, "discord": _discord}


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("sender", list(SENDERS))
def test_standalone_send_result_never_carries_the_proxy_secret(monkeypatch, sender, shape):
    proxy = shape.format(pw=PASSWORD)
    for name in ENV_PROXY_VARS:
        monkeypatch.setenv(name, proxy)
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    with patch("hermes_agent.gateway.platforms.base.gateway_trust_env", return_value=True):
        result = asyncio.run(asyncio.wait_for(SENDERS[sender](monkeypatch, proxy), 30))
    text = json.dumps(result)
    assert result.get("error"), result
    assert " via " in result["error"], result  # the proxy-error path, not an unrelated failure
    assert not _leaks(text), (_leaks(text), text)

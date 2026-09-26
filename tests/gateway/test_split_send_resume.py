"""A split reply whose second chunk fails reaches the screen once, never twice and never cut short
behind a success (the partial_overflow contract, ``BasePlatformAdapter._with_partial_send``).

Regression for #68713 (WhatsApp) and the same send loop in the Mattermost and Slack adapters.
"""

import re
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from tests.gateway.test_mattermost import _make_adapter as _mattermost_adapter
from tests.gateway.test_slack_send_retry import _make_adapter as _slack_adapter, _slack_api_error
from tests.gateway.test_whatsapp_formatting import _AsyncCM, _make_adapter as _whatsapp_adapter


def _resp(status, body=None, text=""):
    resp = MagicMock(status=status)
    resp.json = AsyncMock(return_value=body or {})
    resp.text = AsyncMock(return_value=text)
    return resp


class _Raising:
    def __init__(self, exc):
        self.exc = exc

    async def __aenter__(self):
        raise self.exc

    async def __aexit__(self, *exc):
        return False


def _connect_refused():
    return aiohttp.ClientConnectorError(
        SimpleNamespace(host="h", port=1, ssl=None), OSError(61, "Connection refused"))


def _http_fake(adapter, attr, ok, failure, screen):
    """``session.post`` that lands every call except the second, which returns ``failure()``."""
    calls = []

    def post(url, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            return failure()
        screen.append(kwargs["json"]["message"])
        return _AsyncCM(ok(len(calls)))
    session = MagicMock(closed=False)
    session.post = MagicMock(side_effect=post)
    setattr(adapter, attr, session)


def _mattermost(failure, screen):
    adapter = _mattermost_adapter()
    _http_fake(adapter, "_session", lambda n: _resp(201, {"id": f"post{n}"}), failure, screen)
    return adapter, 4000


def _whatsapp(failure, screen):
    adapter = _whatsapp_adapter()
    _http_fake(adapter, "_http_session", lambda n: _resp(200, {"messageId": f"wa{n}"}), failure, screen)
    return adapter, 4096


def _slack(failure, screen):
    adapter = _slack_adapter()
    calls = []

    async def post_message(**kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise failure()
        screen.append(kwargs["text"])
        return {"ok": True, "ts": f"1700.{len(calls)}"}
    client = MagicMock()
    client.chat_postMessage = AsyncMock(side_effect=post_message)
    adapter._get_client = lambda *_a, **_k: client
    return adapter, 39000


_UNSENT, _MAYBE_SENT = True, False
CASES = [
    pytest.param(_mattermost, lambda: _AsyncCM(_resp(429, text="rate limited")), _UNSENT, id="mattermost-429"),
    pytest.param(_mattermost, lambda: _Raising(_connect_refused()), _UNSENT, id="mattermost-connect"),
    pytest.param(_mattermost, lambda: _AsyncCM(_resp(502, text="bad gateway")), _MAYBE_SENT, id="mattermost-502"),
    pytest.param(_whatsapp, lambda: _AsyncCM(_resp(503, text='{"error":"Not connected to WhatsApp"}')), _UNSENT,
                 id="whatsapp-bridge-disconnected"),
    pytest.param(_whatsapp, lambda: _Raising(_connect_refused()), _UNSENT, id="whatsapp-bridge-down"),
    pytest.param(_whatsapp, lambda: _AsyncCM(_resp(500, text='{"error":"Network is unreachable"}')), _MAYBE_SENT,
                 id="whatsapp-bridge-500"),
    pytest.param(_slack, lambda: _slack_api_error(429), _UNSENT, id="slack-429"),
    pytest.param(_slack, lambda: _slack_api_error(500), _MAYBE_SENT, id="slack-500"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("make, failure, unsent", CASES)
async def test_mid_split_failure_never_duplicates_the_head_or_hides_the_tail(make, failure, unsent):
    screen = []
    adapter, limit = make(failure, screen)
    n_words = (limit * 5) // (2 * 7)  # 7-char words: 2.5 chunks' worth, so the reply splits in three
    content = " ".join(f"w{i:05d}" for i in range(n_words))

    with patch("asyncio.sleep", new=AsyncMock()) as sleep:
        result = await adapter._send_with_retry("chat", content, max_retries=2, base_delay=5)

    words = re.findall(r"w\d{5}", " ".join(screen))
    assert len(words) == len(set(words)), "a delivered chunk was sent again"
    if unsent:
        # The refused chunk never reached the server: resume from it and finish the reply.
        assert result.success and len(set(words)) == n_words
    else:
        # It may have been posted: no retry or fallback may repeat it, and the reply is not reported delivered.
        assert not result.success and len(screen) == 1
        assert adapter._is_partial_delivery(result)
        assert not [c for c in sleep.await_args_list if c.args[0] >= 5], "backed off with nothing to resume"

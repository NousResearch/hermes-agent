"""Poll-cursor safety for the Nextcloud Talk adapter.

Regression coverage for the 2026-07-12 incident: ``get_latest_message_id``
mapped transport errors and non-200 responses to cursor 0. A gateway
(re)start during a Nextcloud outage then began polling with
``lastKnownMessageId=0`` — which Talk answers with the ENTIRE conversation
history — and the agent processed every old message as new.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from plugins.platforms.nextcloud_talk.adapter import (
    NextcloudTalkAdapter,
    TalkUserClient,
)


def _client(http) -> TalkUserClient:
    return TalkUserClient(
        base_url="https://nc.example.com",
        username="hermes",
        password="secret",
        http_client=http,
    )


def _response(status_code: int, data=None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = {"ocs": {"data": [] if data is None else data}}
    resp.text = ""
    return resp


class TestGetLatestMessageId:
    @pytest.mark.asyncio
    async def test_returns_max_id(self):
        http = MagicMock()
        http.get = AsyncMock(return_value=_response(200, [{"id": 41}, {"id": 42}]))
        assert await _client(http).get_latest_message_id("tok") == 42

    @pytest.mark.asyncio
    async def test_zero_only_for_empty_conversation(self):
        http = MagicMock()
        http.get = AsyncMock(return_value=_response(200, []))
        assert await _client(http).get_latest_message_id("tok") == 0

    @pytest.mark.asyncio
    async def test_raises_on_http_error(self):
        # A 504 during an outage must NOT map to cursor 0 — that replays
        # the whole history once the server recovers.
        http = MagicMock()
        http.get = AsyncMock(return_value=_response(504))
        with pytest.raises(RuntimeError):
            await _client(http).get_latest_message_id("tok")

    @pytest.mark.asyncio
    async def test_propagates_transport_error(self):
        http = MagicMock()
        http.get = AsyncMock(side_effect=ConnectionError("boom"))
        with pytest.raises(ConnectionError):
            await _client(http).get_latest_message_id("tok")


class TestPollLoopCursorInit:
    @pytest.mark.asyncio
    async def test_retries_init_and_never_polls_from_zero(self, monkeypatch):
        adapter = NextcloudTalkAdapter.__new__(NextcloudTalkAdapter)
        adapter._shutdown = False
        adapter._poll_timeout = 1

        init_calls = []

        async def flaky_latest(token):
            init_calls.append(token)
            if len(init_calls) == 1:
                raise RuntimeError("outage")
            return 99

        polled_from = []

        async def get_messages(token, last_known, timeout=30):
            polled_from.append(last_known)
            adapter._shutdown = True
            return 304, []

        client = MagicMock()
        client.get_latest_message_id = flaky_latest
        client.get_messages = get_messages
        adapter._client = client

        real_sleep = asyncio.sleep

        async def fast_sleep(_secs):
            await real_sleep(0)

        monkeypatch.setattr(
            "plugins.platforms.nextcloud_talk.adapter.asyncio.sleep", fast_sleep
        )

        await adapter._poll_loop("tok")

        assert len(init_calls) == 2, "cursor init must be retried after failure"
        assert polled_from == [99], "poll must start from the recovered cursor, never 0"

    @pytest.mark.asyncio
    async def test_shutdown_during_init_exits_without_polling(self, monkeypatch):
        adapter = NextcloudTalkAdapter.__new__(NextcloudTalkAdapter)
        adapter._shutdown = False
        adapter._poll_timeout = 1

        async def always_failing(token):
            raise RuntimeError("outage")

        client = MagicMock()
        client.get_latest_message_id = always_failing
        client.get_messages = AsyncMock()
        adapter._client = client

        real_sleep = asyncio.sleep

        async def stop_on_sleep(_secs):
            adapter._shutdown = True
            await real_sleep(0)

        monkeypatch.setattr(
            "plugins.platforms.nextcloud_talk.adapter.asyncio.sleep", stop_on_sleep
        )

        await adapter._poll_loop("tok")

        client.get_messages.assert_not_awaited()

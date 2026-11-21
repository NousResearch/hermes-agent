"""A re-sent ledgered final reuses one Discord nonce, so Discord never shows it twice.

discord.py gives every ``channel.send`` a fresh random nonce with ``enforce_nonce`` set; its own
5xx retries reuse it, but each Hermes-level re-send of the same final (``_send_with_retry``, the
plain-text fallback, the delivery ledger's reconnect/boot redelivery) used to get a new one. When
the first send was created at Discord and only its response was lost, the re-send posted a second
copy. Observed: an 1846-char final was created, the websocket reconnected mid-request
(``ServerDisconnectedError``), the plain fallback died with "Session is closed", and the reconnect
redelivery posted the same text again under the "Recovered reply" marker.

The fake channel below models Discord's documented ``enforce_nonce`` behaviour (Create Message):
a send whose nonce matches a message this author created recently returns that message and creates
none.
"""

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import aiohttp
import pytest

from gateway import delivery_ledger as dl
from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource
from plugins.platforms.discord.adapter import DiscordAdapter

CHAT = "555"
SESSION_KEY = "agent:main:discord:channel:555"
FINAL = ("The answer, in full. " * 88)[:1846]  # one Discord message, with room for the marker


class _EnforceNonceChannel:
    """Discord's Create Message as seen from ``channel.send``: ``script`` steps drive each call.

    ``"ok"`` creates the message; ``"land_then_drop"`` creates it and then loses the response
    (``drop_with``); ``"down"`` fails before reaching Discord ("Session is closed").
    """

    def __init__(self, script=(), drop_with=None):
        self.messages: list = []
        self.calls: list[dict] = []
        self._by_nonce: dict = {}
        self._script = list(script)
        self._drop_with = drop_with or aiohttp.ServerDisconnectedError()

    async def send(self, *, content, reference=None, **kwargs):
        self.calls.append({"content": content, "reference": reference, **kwargs})
        step = self._script.pop(0) if self._script else "ok"
        if step == "down":
            raise RuntimeError("Session is closed")
        nonce = kwargs.get("nonce")
        if nonce is not None and nonce in self._by_nonce:
            return self._by_nonce[nonce]  # enforce_nonce: the existing message, no new one
        msg = SimpleNamespace(id=9000 + len(self.messages), content=content)
        self.messages.append(msg)
        if nonce is not None:
            self._by_nonce[nonce] = msg
        if step == "land_then_drop":
            raise self._drop_with
        return msg


def _adapter(channel) -> DiscordAdapter:
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._client = SimpleNamespace(get_channel=lambda _id: channel, fetch_channel=AsyncMock())
    return adapter


def _nonces(channel) -> list:
    return [call.get("nonce") for call in channel.calls]


# ---------------------------------------------------------------------------
# The adapter contract: which nonce each channel.send carries.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_same_idempotency_key_sends_the_same_nonce_per_chunk():
    channel = _EnforceNonceChannel()
    adapter = _adapter(channel)
    long_text = "word " * 1200  # several Discord messages

    first = await adapter.send(CHAT, long_text, metadata={"idempotency_key": "ob-abc"})
    first_nonces = _nonces(channel)
    channel.calls.clear()
    second = await adapter.send(CHAT, long_text, metadata={"idempotency_key": "ob-abc"})

    assert first.success and second.success
    assert len(first_nonces) >= 3
    assert _nonces(channel) == first_nonces
    assert len(set(first_nonces)) == len(first_nonces)  # each chunk has its own nonce
    assert all(isinstance(n, str) and 0 < len(n) <= 25 for n in first_nonces)  # Discord's limit
    assert second.raw_response["message_ids"] == first.raw_response["message_ids"]
    assert len(channel.messages) == len(first_nonces)


@pytest.mark.asyncio
async def test_different_keys_get_different_nonces():
    channel = _EnforceNonceChannel()
    adapter = _adapter(channel)

    await adapter.send(CHAT, "hello", metadata={"idempotency_key": "ob-1"})
    await adapter.send(CHAT, "hello", metadata={"idempotency_key": "ob-2"})

    a, b = _nonces(channel)
    assert a != b
    assert len(channel.messages) == 2


@pytest.mark.parametrize("metadata", [None, {}, {"thread_id": None}, {"idempotency_key": ""},
                                      {"idempotency_key": None}])
@pytest.mark.asyncio
async def test_no_key_leaves_discord_py_random_nonce(metadata):
    channel = _EnforceNonceChannel()
    adapter = _adapter(channel)

    result = await adapter.send(CHAT, "hello", metadata=metadata)

    assert result.success
    assert "nonce" not in channel.calls[0]


@pytest.mark.asyncio
async def test_reply_reference_retry_keeps_the_chunk_nonce():
    """A rejected reply anchor created nothing, so the retry without it reuses the same nonce."""
    channel = _EnforceNonceChannel()
    rejected = {"done": False}
    real_send = channel.send

    async def send(*, content, reference=None, **kwargs):
        if reference is not None and not rejected["done"]:
            rejected["done"] = True
            channel.calls.append({"content": content, "reference": reference, **kwargs})
            raise RuntimeError("400 Bad Request (error code: 10008): Unknown Message")
        return await real_send(content=content, reference=reference, **kwargs)

    channel.send = send
    adapter = _adapter(channel)

    result = await adapter.send(CHAT, "hello", reply_to="42", metadata={"idempotency_key": "ob-r"})

    assert result.success
    assert [c["reference"] is None for c in channel.calls] == [False, True]
    assert channel.calls[0]["nonce"] == channel.calls[1]["nonce"]
    assert len(channel.messages) == 1


# ---------------------------------------------------------------------------
# The duplicate: the real ledger bracket, the real adapter, the real redelivery sweep.
# ---------------------------------------------------------------------------

@pytest.fixture
def ledger(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(dl, "_db_path", lambda: home / "state.db")
    monkeypatch.setattr(dl, "_owner_stamp", lambda: (os.getpid(), 202))
    monkeypatch.setattr(dl, "ledger_enabled", lambda config=None: True)
    monkeypatch.setattr("gateway.platforms.base.asyncio.sleep", AsyncMock())  # retry backoff


def _ledger_states() -> list:
    with dl._connect() as conn:
        return [r[0] for r in conn.execute("SELECT state FROM delivery_obligations")]


def _event() -> MessageEvent:
    source = SessionSource(platform=Platform.DISCORD, chat_id=CHAT, chat_type="channel")
    return MessageEvent(text="what is the answer?", source=source, message_id="1001")


def _runner(adapter):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.DISCORD: adapter}
    runner._profile_adapters = {}
    runner._active_profile_name = lambda: "default"
    store = SimpleNamespace(clear_resume_pending=AsyncMock(), _store=None)
    runner.session_store = None
    runner._async_session_store = store
    return runner


@pytest.mark.asyncio
async def test_reconnect_redelivery_of_a_landed_final_does_not_duplicate(ledger):
    """The observed incident: created at Discord, response lost, plain fallback on a closed session,
    then the reconnect sweep redelivers with the "Recovered reply" marker."""
    channel = _EnforceNonceChannel(script=["land_then_drop", "down"])
    adapter = _adapter(channel)

    result, _ = await adapter.send_final_ledgered(_event(), SESSION_KEY, FINAL, {}, reply_to=None)

    assert result.success is False and result.error == "send_path_degraded"
    assert _ledger_states() == ["failed"]
    assert len(channel.messages) == 1  # it did land

    redelivered = await _runner(adapter)._redeliver_failed_obligations_for_platform(Platform.DISCORD)

    assert redelivered == 1
    assert channel.calls[-1]["content"].startswith(dl.RECONNECTED_MARKER)
    assert len(channel.messages) == 1, [m.content[:60] for m in channel.messages]
    assert _ledger_states() == ["delivered"]


@pytest.mark.asyncio
async def test_in_process_retry_of_a_landed_final_does_not_duplicate(ledger):
    """A connection-shaped failure after creation is retried inline by ``_send_with_retry``."""
    channel = _EnforceNonceChannel(script=["land_then_drop"], drop_with=ConnectionResetError("reset"))
    adapter = _adapter(channel)

    result, _ = await adapter.send_final_ledgered(_event(), SESSION_KEY, FINAL, {}, reply_to=None)

    assert result.success is True
    assert len(channel.calls) == 2
    assert len(channel.messages) == 1, [m.content[:60] for m in channel.messages]
    assert _ledger_states() == ["delivered"]

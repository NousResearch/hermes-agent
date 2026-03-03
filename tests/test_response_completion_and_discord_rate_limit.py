import asyncio
from types import SimpleNamespace

from gateway.config import PlatformConfig
from gateway.platforms.discord import DiscordAdapter
from run_agent import AIAgent


class _FakeEmbed:
    def __init__(self, description: str):
        self.description = description


class _FakeMessage:
    def __init__(self, msg_id: int):
        self.id = msg_id


class _FakeRetryableError(Exception):
    def __init__(self, message: str, retry_after: float = 1.0, status: int = 429, code: int = 0):
        super().__init__(message)
        self.retry_after = retry_after
        self.status = status
        self.code = code
        self.response = None


class _FakeChannel:
    def __init__(self, fail_first_send: bool = False):
        self.fail_first_send = fail_first_send
        self.send_calls = 0
        self.views = []

    async def send(self, *, embed, view=None, reference=None):
        self.send_calls += 1
        self.views.append(view)
        if self.fail_first_send and self.send_calls == 1:
            raise _FakeRetryableError("rate limited", retry_after=1.25)
        return _FakeMessage(self.send_calls)


class _FakeClient:
    def __init__(self, channel):
        self._channel = channel

    def get_channel(self, _chat_id):
        return self._channel

    async def fetch_channel(self, _chat_id):
        return self._channel


def _build_adapter(channel):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fake"))
    adapter._client = _FakeClient(channel)
    return adapter


def test_strip_think_blocks_removes_only_standard_tags():
    agent = AIAgent.__new__(AIAgent)
    content = (
        "<think>\ninternal\n</think>\n"
        "User-facing answer."
    )
    assert agent._strip_think_blocks(content).strip() == "User-facing answer."


def test_discord_send_retries_with_retry_after(monkeypatch):
    import gateway.platforms.discord as discord_module

    channel = _FakeChannel(fail_first_send=True)
    adapter = _build_adapter(channel)

    monkeypatch.setattr(discord_module, "discord", SimpleNamespace(Embed=_FakeEmbed))
    monkeypatch.setattr(discord_module, "ListenButtonView", lambda _adapter: object())

    sleeps = []

    async def _fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(discord_module.asyncio, "sleep", _fake_sleep)

    result = asyncio.run(adapter.send(chat_id="123", content="hello world"))

    assert result.success is True
    assert channel.send_calls == 2
    assert sleeps
    assert sleeps[0] >= 1.25


def test_discord_send_attaches_listen_button_to_each_chunk(monkeypatch):
    import gateway.platforms.discord as discord_module

    channel = _FakeChannel(fail_first_send=False)
    adapter = _build_adapter(channel)
    adapter.MAX_EMBED_DESCRIPTION = 100

    monkeypatch.setattr(discord_module, "discord", SimpleNamespace(Embed=_FakeEmbed))
    monkeypatch.setattr(discord_module, "ListenButtonView", lambda _adapter: object())

    async def _fake_sleep(_delay):
        return None

    monkeypatch.setattr(discord_module.asyncio, "sleep", _fake_sleep)

    long_text = "x" * 280
    result = asyncio.run(adapter.send(chat_id="123", content=long_text))

    assert result.success is True
    assert channel.send_calls >= 3
    assert all(view is not None for view in channel.views)

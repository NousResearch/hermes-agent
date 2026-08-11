"""Regression tests for parallel platform connect at gateway startup (#83791).

The old ``GatewayRunner.start()`` loop awaited each platform's connect()
(including its own timeout) in turn. A single slow/failing platform (e.g.
Telegram behind a dead proxy) therefore delayed every later platform's
connect by a full timeout window, cascading one platform's failure onto
WeChat/QQ/etc. These tests prove the connects now run concurrently.

Why event-order, not wall-clock timings
---------------------------------------
An earlier version of this test recorded ``time.monotonic()`` around each
connect() and asserted ``slow_start < fast_end``. That assertion is true in
BOTH the serial and the parallel world, so it proved nothing:

  serial:   slow_start=0, slow_end=0.300, fast_start=0.300, fast_end=0.300
            -> 0 < 0.300  (passes, but it's serial!)
  parallel: slow_start=0, fast_start=0,    fast_end=0.001, slow_end=0.300
            -> 0 < 0.001  (passes)

The only assertion that distinguishes them is ``fast_end`` occurring *before*
``slow_end`` (true only when the two connects overlap). We record the
connect start/end events in arrival order, which is fully independent of clock
resolution -- ``time.monotonic()`` has only ~15 ms resolution on Windows
(GetTickCount64), so parallel connects can land on the same tick and defeat any
wall-clock comparison. Event ordering cannot be defeated by a coarse clock.
"""

import asyncio

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter
from gateway.run import GatewayRunner


class _OrderRecorder:
    """Collects connect start/end events in arrival order (clock-agnostic)."""

    events: list = []

    @classmethod
    def reset(cls) -> None:
        cls.events = []

    @classmethod
    def index_of(cls, platform_value: str, kind: str) -> int:
        for i, (name, evt) in enumerate(cls.events):
            if name == platform_value and evt == kind:
                return i
        return -1


class _TimingAdapter(BasePlatformAdapter):
    """Adapter whose ``connect()`` records an event and sleeps.

    Used to prove the startup connect loop launches every platform's
    connect() concurrently rather than serially.
    """

    def __init__(self, platform: Platform, sleep: float):
        super().__init__(PlatformConfig(enabled=True, token="***"), platform)
        self._sleep = sleep

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        _OrderRecorder.events.append((self.platform.value, "start"))
        await asyncio.sleep(self._sleep)
        _OrderRecorder.events.append((self.platform.value, "end"))
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        raise NotImplementedError

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


@pytest.mark.asyncio
async def test_startup_connects_platforms_concurrently(monkeypatch, tmp_path):
    """A slow platform must not block a later platform at startup (#83791).

    "slow" (Telegram) is listed first so a serial loop would fully block
    "fast" (Discord). We prove the connect calls overlap by recording the
    order in which connects finish: under a serial loop the slow platform's
    connect ends *before* the fast one even begins, so the fast platform's
    end can never precede the slow platform's end. Only parallel execution
    puts ``fast_end`` before ``slow_end``.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _OrderRecorder.reset()

    config = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(enabled=True, token="***"),
            Platform.DISCORD: PlatformConfig(enabled=True, token="***"),
        },
        sessions_dir=tmp_path / "sessions",
    )
    runner = GatewayRunner(config)

    def _make_adapter(platform, platform_config):
        sleep = 0.3 if platform is Platform.TELEGRAM else 0.0
        return _TimingAdapter(platform, sleep)

    monkeypatch.setattr(runner, "_create_adapter", _make_adapter)
    # Keep the rest of startup lightweight / non-fatal.
    monkeypatch.setattr(runner, "_start_secondary_profile_adapters", lambda: 0)

    await runner.start()

    events = _OrderRecorder.events
    assert events, "no connect() event was recorded"

    fast_end = _OrderRecorder.index_of(Platform.DISCORD.value, "end")
    slow_end = _OrderRecorder.index_of(Platform.TELEGRAM.value, "end")
    assert fast_end != -1 and slow_end != -1, f"missing end events: {events}"

    # Overlap proof: the fast platform finished before the slow one did,
    # which is only possible if the two connects ran at the same time.
    assert fast_end < slow_end, (
        f"connects did not overlap (serial loop?): events={events}"
    )
    # Both platforms should be registered once startup settles.
    assert Platform.TELEGRAM in runner.adapters
    assert Platform.DISCORD in runner.adapters


@pytest.mark.asyncio
async def test_startup_one_failing_platform_does_not_block_others(monkeypatch, tmp_path):
    """A failing/slow platform must not prevent others from connecting (#83791).

    Mirrors the reported Windows symptom: Telegram (dead proxy) must not keep
    WeChat/QQ offline. Here Telegram fails (returns False after a sleep) while
    Discord connects successfully and is registered.
    """

    class _FailingSlowAdapter(BasePlatformAdapter):
        def __init__(self):
            super().__init__(PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM)

        async def connect(self, *, is_reconnect: bool = False) -> bool:
            await asyncio.sleep(0.3)
            self._set_fatal_error("telegram_proxy_dead", "proxy unreachable", retryable=True)
            return False

        async def disconnect(self) -> None:
            self._mark_disconnected()

        async def send(self, chat_id, content, reply_to=None, metadata=None):
            raise NotImplementedError

        async def get_chat_info(self, chat_id):
            return {"id": chat_id}

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _OrderRecorder.reset()

    config = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(enabled=True, token="***"),
            Platform.DISCORD: PlatformConfig(enabled=True, token="***"),
        },
        sessions_dir=tmp_path / "sessions",
    )
    runner = GatewayRunner(config)

    def _make_adapter(platform, platform_config):
        if platform is Platform.TELEGRAM:
            return _FailingSlowAdapter()
        return _TimingAdapter(platform, 0.0)

    monkeypatch.setattr(runner, "_create_adapter", _make_adapter)
    monkeypatch.setattr(runner, "_start_secondary_profile_adapters", lambda: 0)

    await runner.start()

    # The healthy platform connected and is registered despite Telegram failing.
    assert Platform.DISCORD in runner.adapters
    # The failed platform is queued for retry, not silently dropped.
    assert Platform.TELEGRAM in runner._failed_platforms

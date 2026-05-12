"""Shared test helpers for the kanban notifier watchers."""

from typing import Optional

from gateway.config import HomeChannel, Platform
from gateway.run import GatewayRunner


class RecordingAdapter:
    def __init__(self):
        self.sent = []

    async def send(self, chat_id, text, metadata=None):
        self.sent.append({"chat_id": chat_id, "text": text, "metadata": metadata or {}})


class FailingAdapter:
    def __init__(self):
        self.attempts = 0

    async def send(self, chat_id, text, metadata=None):
        self.attempts += 1
        raise RuntimeError("simulated send failure")


class _StubConfig:
    def __init__(self, homes: dict[Platform, Optional[HomeChannel]]):
        self._homes = homes

    def get_home_channel(self, platform):
        return self._homes.get(platform)


def make_runner(adapters: dict, *, homes: Optional[dict] = None):
    """Construct a barebones GatewayRunner for kanban notifier tests."""
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = adapters
    runner._kanban_sub_fail_counts = {}
    if homes is None:
        homes = {
            plat: HomeChannel(platform=plat, chat_id="home-chat", name="Home")
            for plat in adapters
        }
    runner.config = _StubConfig(homes)
    return runner

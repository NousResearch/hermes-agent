"""Cron delivery to a named Telegram private-DM topic: ``deliver: telegram:<private_chat>:<topic name>``."""

import asyncio
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from cron.scheduler import _deliver_result, _resolve_delivery_target
from gateway.config import Platform, PlatformConfig


def test_named_private_topic_target_splits_chat_and_topic():
    target = _resolve_delivery_target({"deliver": "telegram:722341991:Grafana Alerts"})
    assert (target["chat_id"], target["thread_id"]) == ("722341991", "Grafana Alerts")
    # numeric topics and group chats keep their shape; a group never takes a topic name
    assert _resolve_delivery_target({"deliver": "telegram:722341991:38049"})["thread_id"] == "38049"
    assert _resolve_delivery_target({"deliver": "telegram:-100123:Ops"})["thread_id"] is None


class _Adapter:
    """Records what the live Telegram adapter is asked to send and resolve."""

    platform = Platform.TELEGRAM

    def __init__(self):
        self.topics, self.sends, self.media = [], [], []

    async def ensure_dm_topic(self, chat_id, name, force_create=False):
        self.topics.append((chat_id, name, force_create))
        return "4242"

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sends.append((chat_id, dict(metadata or {})))
        return SimpleNamespace(success=True, message_id="1", raw_response={"ok": True}, error=None)

    async def send_image_file(self, chat_id, image_path, metadata=None, **_):
        self.media.append((chat_id, dict(metadata or {})))
        return SimpleNamespace(success=True, message_id="2", raw_response={"ok": True}, error=None)


def test_cron_text_and_media_reach_the_named_topic(tmp_path):
    image = tmp_path / "chart.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
    adapter = _Adapter()
    loop = MagicMock()
    loop.is_running.return_value = True

    def run_now(coro, _loop):
        future = Future()
        try:
            future.set_result(asyncio.run(coro))
        except BaseException as exc:
            future.set_exception(exc)
        return future

    config = MagicMock()
    config.platforms = {Platform.TELEGRAM: PlatformConfig(enabled=True)}
    config.get_home_channel = lambda p: None
    job = {"id": "j-named", "name": "Brief", "deliver": "telegram:722341991:Debug"}
    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}), \
         patch("gateway.platforms.base.validate_media_delivery_path", side_effect=lambda p: p), \
         patch("gateway.platforms.base.BasePlatformAdapter.filter_media_delivery_paths", side_effect=lambda m: m), \
         patch("asyncio.run_coroutine_threadsafe", side_effect=run_now):
        error = _deliver_result(job, f"daily brief\nMEDIA:{image}", adapters={Platform.TELEGRAM: adapter}, loop=loop)

    assert error is None
    assert adapter.topics and all(t[:2] == ("722341991", "Debug") for t in adapter.topics)
    assert [(c, m.get("thread_id")) for c, m in adapter.sends] == [("722341991", "4242")]
    assert [(c, m.get("thread_id")) for c, m in adapter.media] == [("722341991", "4242")]

"""Cron delivery to a named Telegram private-DM topic: ``deliver: telegram:<private_chat>:<topic name>``."""

import asyncio
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

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

    def __init__(self, deleted_topic=None):
        self.topics, self.sends, self.media = [], [], []
        self.topic_id, self.deleted_topic = "4242", deleted_topic

    async def ensure_dm_topic(self, chat_id, name, force_create=False):
        self.topics.append((chat_id, name, force_create))
        if force_create:  # like the real adapter: recreate and update the by-name cache
            self.topic_id = "5151"
        return self.topic_id

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sends.append((chat_id, dict(metadata or {})))
        if (metadata or {}).get("thread_id") == self.deleted_topic:
            return SimpleNamespace(success=False, message_id=None, raw_response=None,
                                   error="Bad Request: message thread not found")
        return SimpleNamespace(success=True, message_id="1", raw_response={"ok": True}, error=None)

    async def send_image_file(self, chat_id, image_path, metadata=None, **_):
        self.media.append((chat_id, dict(metadata or {})))
        return SimpleNamespace(success=True, message_id="2", raw_response={"ok": True}, error=None)


def _run_named_delivery(tmp_path, adapter, job, media=True):
    image = tmp_path / "chart.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)
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
    with patch("gateway.config.load_gateway_config", return_value=config), \
         patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}), \
         patch("gateway.platforms.base.validate_media_delivery_path", side_effect=lambda p: p), \
         patch("gateway.platforms.base.BasePlatformAdapter.filter_media_delivery_paths", side_effect=lambda m: m), \
         patch("gateway.mirror.mirror_to_session", return_value=True) as mirror, \
         patch("asyncio.run_coroutine_threadsafe", side_effect=run_now):
        text = f"daily brief\nMEDIA:{image}" if media else "daily brief"
        error = _deliver_result(job, text, adapters={Platform.TELEGRAM: adapter}, loop=loop)
    return error, mirror


def test_cron_text_and_media_reach_the_named_topic(tmp_path):
    adapter = _Adapter()
    job = {"id": "j-named", "name": "Brief", "deliver": "telegram:722341991:Debug"}
    error, _mirror = _run_named_delivery(tmp_path, adapter, job)

    assert error is None
    assert adapter.topics and all(t[:2] == ("722341991", "Debug") for t in adapter.topics)
    assert [(c, m.get("thread_id")) for c, m in adapter.sends] == [("722341991", "4242")]
    assert [(c, m.get("thread_id")) for c, m in adapter.media] == [("722341991", "4242")]


def test_media_follows_a_topic_recreated_by_the_text_send(tmp_path):
    """The saved topic was deleted: the text send gets thread-not-found and the router recreates the
    topic; the attachments must go to the new topic, not the deleted id."""
    adapter = _Adapter(deleted_topic="4242")
    job = {"id": "j-named", "name": "Brief", "deliver": "telegram:722341991:Debug"}
    error, _mirror = _run_named_delivery(tmp_path, adapter, job)

    assert error is None
    assert [m.get("thread_id") for _c, m in adapter.sends] == ["4242", "5151"]
    assert [m.get("thread_id") for _c, m in adapter.media] == ["5151"]


def test_opted_in_mirror_targets_the_named_topics_numeric_id(tmp_path):
    """Telegram topic sessions key on the numeric thread id; mirroring under the name finds none."""
    adapter = _Adapter()
    job = {"id": "j-named", "name": "Brief", "deliver": "telegram:722341991:Debug", "attach_to_session": True}
    error, mirror = _run_named_delivery(tmp_path, adapter, job)

    assert error is None
    mirror.assert_called_once()
    assert mirror.call_args.kwargs["thread_id"] == "4242"


def test_text_only_mirror_follows_a_recreated_topic(tmp_path):
    adapter = _Adapter(deleted_topic="4242")
    job = {"id": "j-named", "name": "Brief", "deliver": "telegram:722341991:Debug", "attach_to_session": True}
    error, mirror = _run_named_delivery(tmp_path, adapter, job, media=False)

    assert error is None
    assert mirror.call_args.kwargs["thread_id"] == "5151"


@pytest.mark.parametrize("attach", [False, True])
def test_confirmed_text_stays_delivered_when_the_topic_re_lookup_fails(tmp_path, attach):
    """The text send is confirmed; a later topic lookup failing (loop stalled/closing) must not turn
    the run into a failure and a standalone resend."""
    adapter = _Adapter()
    real_ensure = adapter.ensure_dm_topic

    async def ensure_once(chat_id, name, force_create=False):
        if adapter.sends:  # the text already landed
            raise RuntimeError("gateway loop closing")
        return await real_ensure(chat_id, name, force_create)

    adapter.ensure_dm_topic = ensure_once
    job = {"id": "j-named", "name": "Brief", "deliver": "telegram:722341991:Debug", "attach_to_session": attach}
    with patch("tools.send_message_tool._send_to_platform") as standalone:
        error, mirror = _run_named_delivery(tmp_path, adapter, job, media=False)

    assert error is None
    standalone.assert_not_called()
    assert [m.get("thread_id") for _c, m in adapter.sends] == ["4242"]
    if attach:
        assert mirror.call_args.kwargs["thread_id"] == "4242"

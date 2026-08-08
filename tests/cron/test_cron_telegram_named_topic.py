"""Cron named Telegram private-DM topic delivery (#80483).

Behavior contracts for ``telegram:<private_chat_id>:<topic_name>`` cron targets:

- ``_parse_target_ref`` splits the named private topic into chat_id + topic_name
  (numeric thread-id behavior is unchanged).
- ``_resolve_delivery_target`` returns the split shape (the exact repro from
  the issue).
- Cron text + media delivery resolve the named topic through the LIVE
  ``TelegramAdapter.ensure_dm_topic`` (one resolver for both), use the returned
  thread_id for both text and media, retry once on thread-not-found, and fail
  closed when no live adapter is available.
- Numeric / group / forum / non-Telegram targets are unchanged (non-regression).

Tests use the real ``_parse_target_ref`` and ``_deliver_result`` over a stub
adapter that exposes the real ``ensure_dm_topic`` coroutine signature, per
AGENTS.md's E2E-over-mocks guidance. No test writes to ``~/.hermes/`` — a temp
``HERMES_HOME`` is pinned.
"""

from concurrent.futures import Future
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cron.scheduler import (
    _deliver_result,
    _resolve_delivery_target,
    _resolve_telegram_named_dm_topic,
)
from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
from gateway.platforms.base import SendResult
from tools.send_message_tool import _parse_target_ref


@pytest.fixture(autouse=True)
def _temp_hermes_home(tmp_path, monkeypatch):
    """Pin a temp HERMES_HOME so no test reads or writes ~/.hermes/."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    yield


class RecordingTelegramAdapter:
    """Live Telegram adapter double with a real ``ensure_dm_topic`` coroutine.

    Mirrors the contract in tests/gateway/test_delivery.py: ``send`` records the
    routed metadata so tests can assert the resolved thread_id lands in BOTH the
    text send and the media send. ``ensure_dm_topic`` is a real coroutine method
    on the CLASS (not an instance attribute) because the cron resolver and the
    DeliveryRouter both probe ``type(adapter).ensure_dm_topic``.
    """

    def __init__(self, send_success=True, thread_id_for_name="38049",
                 force_create_thread_id="38064", send_error_first=None):
        self.calls = []
        self.ensure_dm_topic_calls = []
        self.send_calls = []
        self.media_calls = []
        self._send_success = send_success
        self._thread_id_for_name = thread_id_for_name
        self._force_create_thread_id = force_create_thread_id
        self._send_error_first = send_error_first
        self._send_count = 0
        self.platform = Platform.TELEGRAM

    async def ensure_dm_topic(self, chat_id, topic_name, force_create=False):
        self.ensure_dm_topic_calls.append(
            {"chat_id": str(chat_id), "topic_name": topic_name,
             "force_create": force_create}
        )
        if force_create:
            return self._force_create_thread_id
        return self._thread_id_for_name

    async def send(self, chat_id, content, metadata=None):
        self._send_count += 1
        self.send_calls.append(
            {"chat_id": str(chat_id), "content": content, "metadata": dict(metadata or {})}
        )
        if self._send_error_first is not None and self._send_count == 1:
            return SendResult(success=False, error=self._send_error_first)
        return SendResult(success=self._send_success, message_id="m1")

    # Media methods used by _send_media_via_adapter.
    async def send_image_file(self, chat_id, image_path, metadata=None):
        self.media_calls.append(
            {"method": "send_image_file", "chat_id": str(chat_id),
             "image_path": image_path, "metadata": dict(metadata or {})}
        )
        return SendResult(success=True)

    async def send_document(self, chat_id, file_path, metadata=None):
        self.media_calls.append(
            {"method": "send_document", "chat_id": str(chat_id),
             "file_path": file_path, "metadata": dict(metadata or {})}
        )
        return SendResult(success=True)

    async def send_voice(self, chat_id, audio_path, metadata=None):
        self.media_calls.append(
            {"method": "send_voice", "chat_id": str(chat_id),
             "audio_path": audio_path, "metadata": dict(metadata or {})}
        )
        return SendResult(success=True)

    async def send_video(self, chat_id, video_path, metadata=None):
        self.media_calls.append(
            {"method": "send_video", "chat_id": str(chat_id),
             "video_path": video_path, "metadata": dict(metadata or {})}
        )
        return SendResult(success=True)


def _run_coro_factory():
    """Build a fake ``asyncio.run_coroutine_threadsafe`` that actually runs the
    coroutine (so the real DeliveryRouter + adapter execute) and returns a
    completed concurrent.futures.Future — matching the real scheduling contract.
    """
    import asyncio

    def fake_run_coro(coro, _loop):
        future = Future()
        try:
            future.set_result(asyncio.run(coro))
        except BaseException as exc:  # noqa: BLE001
            future.set_exception(exc)
        return future

    return fake_run_coro


def _telegram_config(home_chat_id="722341991"):
    """A GatewayConfig with an enabled Telegram platform + home channel."""
    return GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(
                enabled=True,
                home_channel=HomeChannel(
                    platform=Platform.TELEGRAM,
                    chat_id=home_chat_id,
                    name="Owner DM",
                ),
            ),
        },
    )


class TestParseNamedTelegramTopic:
    """``_parse_target_ref`` splits ``telegram:<chat>:<topic_name>`` (#80483)."""

    def test_named_private_topic_splits_chat_id_and_topic_name(self):
        chat_id, thread_id, is_explicit = _parse_target_ref("telegram", "722341991:Debug")
        assert (chat_id, thread_id, is_explicit) == ("722341991", "Debug", True)

    def test_named_topic_with_underscore_name(self):
        chat_id, thread_id, is_explicit = _parse_target_ref(
            "telegram", "722341991:Hermes_API_Test"
        )
        assert (chat_id, thread_id, is_explicit) == ("722341991", "Hermes_API_Test", True)

    def test_numeric_thread_id_unchanged_negative_chat(self):
        """Numeric topic on a group/supergroup chat: byte-identical to before."""
        chat_id, thread_id, is_explicit = _parse_target_ref(
            "telegram", "-1003724596514:17"
        )
        assert (chat_id, thread_id, is_explicit) == ("-1003724596514", "17", True)

    def test_numeric_thread_id_unchanged_positive_chat(self):
        """Numeric topic on a private (positive) chat: still numeric, not named."""
        chat_id, thread_id, is_explicit = _parse_target_ref("telegram", "722341991:42")
        assert (chat_id, thread_id, is_explicit) == ("722341991", "42", True)

    def test_bare_chat_id_unchanged(self):
        chat_id, thread_id, is_explicit = _parse_target_ref("telegram", "722341991")
        assert (chat_id, thread_id, is_explicit) == ("722341991", None, True)

    def test_bare_negative_chat_id_unchanged(self):
        chat_id, thread_id, is_explicit = _parse_target_ref("telegram", "-1003724596514")
        assert (chat_id, thread_id, is_explicit) == ("-1003724596514", None, True)


class TestResolveNamedTelegramTopic:
    """``_resolve_delivery_target`` returns the split shape (issue repro)."""

    def test_issue_repro_named_private_topic(self):
        # Exact reproduction from issue #80483.
        assert _resolve_delivery_target({"deliver": "telegram:722341991:Debug"}) == {
            "platform": "telegram",
            "chat_id": "722341991",
            "thread_id": "Debug",
        }

    def test_numeric_topic_non_regression(self):
        assert _resolve_delivery_target({"deliver": "telegram:-1003724596514:17"}) == {
            "platform": "telegram",
            "chat_id": "-1003724596514",
            "thread_id": "17",
        }

    def test_numeric_positive_chat_topic_non_regression(self):
        assert _resolve_delivery_target({"deliver": "telegram:722341991:42"}) == {
            "platform": "telegram",
            "chat_id": "722341991",
            "thread_id": "42",
        }

    def test_group_forum_target_unchanged(self):
        # A supergroup forum topic target: negative chat, numeric thread.
        assert _resolve_delivery_target({"deliver": "telegram:-100456:9"}) == {
            "platform": "telegram",
            "chat_id": "-100456",
            "thread_id": "9",
        }

    def test_non_telegram_target_unchanged(self, monkeypatch):
        monkeypatch.setenv("DISCORD_HOME_CHANNEL", "987654")
        assert _resolve_delivery_target({"deliver": "discord:987654:55"}) == {
            "platform": "discord",
            "chat_id": "987654",
            "thread_id": "55",
        }


class TestResolveTelegramNamedDmTopicHelper:
    """The named-topic resolver reuses the live adapter's ensure_dm_topic."""

    def test_resolves_via_ensure_dm_topic(self):
        adapter = RecordingTelegramAdapter(thread_id_for_name="38049")
        loop = MagicMock()
        loop.is_running.return_value = True
        with patch("asyncio.run_coroutine_threadsafe", side_effect=_run_coro_factory()):
            resolved = _resolve_telegram_named_dm_topic(
                adapter, "722341991", "Debug", loop, "job-1",
            )
        assert resolved == "38049"
        assert adapter.ensure_dm_topic_calls == [
            {"chat_id": "722341991", "topic_name": "Debug", "force_create": False},
        ]

    def test_retries_once_with_force_create(self):
        adapter = RecordingTelegramAdapter(
            thread_id_for_name=None,  # first resolution returns nothing
            force_create_thread_id="38064",
        )
        loop = MagicMock()
        loop.is_running.return_value = True
        with patch("asyncio.run_coroutine_threadsafe", side_effect=_run_coro_factory()):
            resolved = _resolve_telegram_named_dm_topic(
                adapter, "722341991", "Debug", loop, "job-1",
            )
        # First call normal, then a force_create retry; second resolution wins.
        assert resolved == "38064"
        assert adapter.ensure_dm_topic_calls == [
            {"chat_id": "722341991", "topic_name": "Debug", "force_create": False},
            {"chat_id": "722341991", "topic_name": "Debug", "force_create": True},
        ]

    def test_fail_closed_when_no_live_adapter(self):
        # An adapter instance whose CLASS has no ensure_dm_topic coroutine.
        class BareAdapter:
            platform = Platform.TELEGRAM

        loop = MagicMock()
        loop.is_running.return_value = True
        resolved = _resolve_telegram_named_dm_topic(
            BareAdapter(), "722341991", "Debug", loop, "job-1",
        )
        assert resolved is None


class TestCronNamedTopicDelivery:
    """E2E: cron ``_deliver_result`` for a named Telegram private-DM topic."""

    def _media_path(self, tmp_path, monkeypatch, name="img.png", data=b"png"):
        root = tmp_path / "media-cache"
        media_file = root / name
        media_file.parent.mkdir(parents=True, exist_ok=True)
        media_file.write_bytes(data)
        monkeypatch.setattr(
            "gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS", (root,),
        )
        return media_file.resolve()

    def test_named_topic_resolved_for_text_and_media(self, tmp_path, monkeypatch):
        """Text routes via DeliveryRouter (which calls ensure_dm_topic) and media
        routes via the cron resolver — both use the SAME ensure_dm_topic, and the
        resolved thread_id is used for both text and media metadata."""
        adapter = RecordingTelegramAdapter(thread_id_for_name="38049")
        config = _telegram_config()
        loop = MagicMock()
        loop.is_running.return_value = True
        media_path = self._media_path(tmp_path, monkeypatch)

        job = {
            "id": "named-topic-job",
            "deliver": "telegram:722341991:Debug",
        }
        with patch("gateway.config.load_gateway_config", return_value=config), \
             patch("cron.scheduler.load_config",
                   return_value={"cron": {"wrap_response": False}}), \
             patch("asyncio.run_coroutine_threadsafe",
                   side_effect=_run_coro_factory()):
            result = _deliver_result(
                job,
                f"summary\nMEDIA:{media_path}",
                adapters={Platform.TELEGRAM: adapter},
                loop=loop,
            )

        assert result is None, f"expected successful delivery, got: {result!r}"

        # The named topic was resolved via ensure_dm_topic (text path through the
        # DeliveryRouter; media path through the cron resolver). At least one
        # resolution happened.
        assert adapter.ensure_dm_topic_calls, (
            "ensure_dm_topic must be called to resolve the named topic"
        )
        # Every resolution targeted the right chat + topic name.
        for call in adapter.ensure_dm_topic_calls:
            assert call["chat_id"] == "722341991"
            assert call["topic_name"] == "Debug"

        # Text send: the resolved numeric thread_id (38049) is in metadata, NOT
        # the bare name "Debug".
        assert adapter.send_calls, "text must be sent via the live adapter"
        text_metadata = adapter.send_calls[0]["metadata"]
        assert text_metadata.get("thread_id") == "38049", (
            f"text must carry the resolved thread_id, got {text_metadata!r}"
        )
        assert text_metadata.get("thread_id") != "Debug"

        # Media send: the resolved numeric thread_id is in media metadata.
        assert adapter.media_calls, "media must be sent via the live adapter"
        media_metadata = adapter.media_calls[0]["metadata"]
        assert media_metadata.get("thread_id") == "38049", (
            f"media must carry the resolved thread_id, got {media_metadata!r}"
        )

    def test_retry_once_on_thread_not_found_for_text(self, tmp_path, monkeypatch):
        """When the first text send fails with 'thread not found', the
        DeliveryRouter refreshes the topic mapping via ensure_dm_topic
        (force_create) and retries the send once (#80483 contract #3)."""
        adapter = RecordingTelegramAdapter(
            send_success=True,
            thread_id_for_name="38049",
            force_create_thread_id="38064",
            # First send fails with thread-not-found; router refreshes + retries.
            send_error_first="Bad Request: message thread not found",
        )
        config = _telegram_config()
        loop = MagicMock()
        loop.is_running.return_value = True

        job = {"id": "retry-job", "deliver": "telegram:722341991:Debug"}
        with patch("gateway.config.load_gateway_config", return_value=config), \
             patch("cron.scheduler.load_config",
                   return_value={"cron": {"wrap_response": False}}), \
             patch("asyncio.run_coroutine_threadsafe",
                   side_effect=_run_coro_factory()):
            result = _deliver_result(
                job, "hello", adapters={Platform.TELEGRAM: adapter}, loop=loop,
            )

        # Delivery ultimately succeeds after the refresh+retry.
        assert result is None, f"expected delivery after retry, got: {result!r}"
        # Two text sends (initial + retry).
        assert len(adapter.send_calls) == 2, (
            f"expected initial + retry send, got {len(adapter.send_calls)}"
        )
        # ensure_dm_topic called with force_create during the refresh.
        assert any(c["force_create"] for c in adapter.ensure_dm_topic_calls), (
            "retry must refresh the topic mapping via force_create"
        )

    def test_no_live_adapter_fails_closed(self, monkeypatch):
        """With no live adapter/loop, a named topic cannot be resolved — fail
        closed instead of delivering into General via the standalone path."""
        config = _telegram_config()
        standalone_send = AsyncMock(return_value={"success": True})

        job = {"id": "no-adapter-job", "deliver": "telegram:722341991:Debug"}
        with patch("gateway.config.load_gateway_config", return_value=config), \
             patch("cron.scheduler.load_config",
                   return_value={"cron": {"wrap_response": False}}), \
             patch("tools.send_message_tool._send_to_platform", new=standalone_send):
            result = _deliver_result(job, "hello", adapters=None, loop=None)

        # Fail closed: an error is returned (not None), and the standalone path
        # must NOT have delivered a raw "Debug" thread_id into General.
        assert result is not None, "fail closed must return an error, not None"
        assert "Debug" in result or "named" in result.lower() or "fail closed" in result.lower()
        standalone_send.assert_not_awaited()

    def test_numeric_topic_unchanged_via_live_adapter(self, tmp_path, monkeypatch):
        """Numeric Telegram topic on a private chat still routes via the
        existing ambiguous-topic path (non-regression for #80483 contract #5)."""
        adapter = RecordingTelegramAdapter(thread_id_for_name="38049")
        # get_chat_info on the CLASS so _is_channel_dm_topic can probe; return a
        # non-channel so routing stays on message_thread_id (forum-style).
        async def _get_chat_info(chat_id):
            return {"type": "private"}
        adapter.get_chat_info = _get_chat_info  # instance attr is fine for the probe

        config = _telegram_config()
        loop = MagicMock()
        loop.is_running.return_value = True

        job = {"id": "numeric-job", "deliver": "telegram:722341991:42"}
        with patch("gateway.config.load_gateway_config", return_value=config), \
             patch("cron.scheduler.load_config",
                   return_value={"cron": {"wrap_response": False}}), \
             patch("asyncio.run_coroutine_threadsafe",
                   side_effect=_run_coro_factory()):
            result = _deliver_result(
                job, "hello", adapters={Platform.TELEGRAM: adapter}, loop=loop,
            )

        assert result is None, f"expected successful delivery, got: {result!r}"
        # Numeric topic must NOT trigger ensure_dm_topic (named-topic resolver).
        assert adapter.ensure_dm_topic_calls == [], (
            "numeric topic must not call ensure_dm_topic (non-regression)"
        )
        # The numeric thread_id 42 is routed (in metadata via the ambiguous-topic
        # branch), not treated as a name.
        assert adapter.send_calls, "text must be sent"
        text_metadata = adapter.send_calls[0]["metadata"]
        assert text_metadata.get("thread_id") == "42" or (
            "direct_messages_topic_id" in text_metadata
        ), f"numeric thread_id 42 must be routed, got {text_metadata!r}"

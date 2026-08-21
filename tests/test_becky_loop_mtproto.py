from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gateway import becky_loops
from gateway.telegram_mtproto import (
    MTProtoPrivateTopicController,
    MtprotoTopicControlError,
)


class FakeEditForumTopicRequest:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class FakeClient:
    def __init__(self) -> None:
        self.connected = False
        self.bot_token: str | None = None
        self.requests: list[FakeEditForumTopicRequest] = []
        self.disconnected = False
        self.start_error: BaseException | None = None
        self.error: BaseException | None = None
        self.identity = SimpleNamespace(bot=True)

    async def start(self, *, bot_token: str) -> None:
        if self.start_error is not None:
            raise self.start_error
        self.bot_token = bot_token
        self.connected = True

    def is_connected(self) -> bool:
        return self.connected

    async def __call__(self, request: FakeEditForumTopicRequest) -> object:
        if self.error is not None:
            raise self.error
        self.requests.append(request)
        return object()

    async def get_me(self) -> object:
        return self.identity

    async def disconnect(self) -> None:
        self.connected = False
        self.disconnected = True


class FakeClientFactory:
    def __init__(self) -> None:
        self.client = FakeClient()
        self.args: tuple[Any, ...] | None = None

    def __call__(self, *args: Any) -> FakeClient:
        self.args = args
        return self.client


@pytest.mark.asyncio
async def test_mtproto_controller_starts_with_bot_and_closes_configured_chat() -> None:
    factory = FakeClientFactory()
    controller = MTProtoPrivateTopicController(
        api_id=12345,
        api_hash="a" * 32,
        bot_token="b" * 46,
        chat_id="8837347581",
        client_factory=factory,
        request_factory=FakeEditForumTopicRequest,
    )

    assert controller.is_connected is False
    assert controller.supports_close is False

    await controller.start()

    assert controller.is_connected is True
    assert controller.supports_close is True
    assert factory.args == (None, 12345, "a" * 32)
    assert factory.client.bot_token == "b" * 46

    closed_at = await controller.close_topic(chat_id="8837347581", thread_id="3964")

    assert isinstance(closed_at, datetime)
    assert closed_at.tzinfo is UTC
    assert [request.kwargs for request in factory.client.requests] == [
        {"peer": 8837347581, "topic_id": 3964, "closed": True}
    ]

    reopened_at = await controller.set_topic_closed(
        chat_id="8837347581", thread_id="3964", closed=False
    )
    assert reopened_at.tzinfo is UTC
    assert [request.kwargs for request in factory.client.requests] == [
        {"peer": 8837347581, "topic_id": 3964, "closed": True},
        {"peer": 8837347581, "topic_id": 3964, "closed": False},
    ]

    await controller.stop()
    assert factory.client.disconnected is True


@pytest.mark.asyncio
async def test_mtproto_controller_rejects_other_chats_without_calling_telegram() -> (
    None
):
    factory = FakeClientFactory()
    controller = MTProtoPrivateTopicController(
        api_id=12345,
        api_hash="a" * 32,
        bot_token="b" * 46,
        chat_id="8837347581",
        client_factory=factory,
        request_factory=FakeEditForumTopicRequest,
    )
    await controller.start()

    with pytest.raises(MtprotoTopicControlError) as caught:
        await controller.close_topic(chat_id="8837347582", thread_id="3964")

    assert caught.value.code == "topic_control_unavailable"
    assert factory.client.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (RuntimeError("TOPIC_NOT_MODIFIED"), "topic_already_closed"),
        (RuntimeError("CHANNEL_INVALID"), "topic_not_found"),
        (RuntimeError("unexpected provider failure"), "topic_control_unavailable"),
    ],
)
async def test_mtproto_controller_maps_provider_failures(
    error: BaseException, expected: str
) -> None:
    factory = FakeClientFactory()
    factory.client.error = error
    controller = MTProtoPrivateTopicController(
        api_id=12345,
        api_hash="a" * 32,
        bot_token="b" * 46,
        chat_id="8837347581",
        client_factory=factory,
        request_factory=FakeEditForumTopicRequest,
    )
    await controller.start()

    with pytest.raises(MtprotoTopicControlError) as caught:
        await controller.close_topic(chat_id="8837347581", thread_id="3964")

    assert caught.value.code == expected


@pytest.mark.asyncio
async def test_mtproto_controller_maps_not_modified_reopen_to_open() -> None:
    factory = FakeClientFactory()
    factory.client.error = RuntimeError("Content of the message was not modified")
    controller = MTProtoPrivateTopicController(
        api_id=12345,
        api_hash="a" * 32,
        bot_token="b" * 46,
        chat_id="8837347581",
        client_factory=factory,
        request_factory=FakeEditForumTopicRequest,
    )
    await controller.start()

    with pytest.raises(MtprotoTopicControlError) as caught:
        await controller.set_topic_closed(
            chat_id="8837347581", thread_id="3964", closed=False
        )

    assert caught.value.code == "topic_already_open"


@pytest.mark.asyncio
async def test_mtproto_controller_does_not_treat_closed_topic_as_open() -> None:
    factory = FakeClientFactory()
    factory.client.error = RuntimeError("TOPIC_CLOSED")
    controller = MTProtoPrivateTopicController(
        api_id=12345,
        api_hash="a" * 32,
        bot_token="b" * 46,
        chat_id="8837347581",
        client_factory=factory,
        request_factory=FakeEditForumTopicRequest,
    )
    await controller.start()

    with pytest.raises(MtprotoTopicControlError) as caught:
        await controller.set_topic_closed(
            chat_id="8837347581", thread_id="3964", closed=False
        )

    assert caught.value.code == "topic_control_unavailable"


@pytest.mark.asyncio
async def test_mtproto_controller_disconnects_if_start_fails() -> None:
    factory = FakeClientFactory()
    factory.client.start_error = RuntimeError("network unavailable")
    controller = MTProtoPrivateTopicController(
        api_id=12345,
        api_hash="a" * 32,
        bot_token="b" * 46,
        chat_id="8837347581",
        client_factory=factory,
        request_factory=FakeEditForumTopicRequest,
    )

    with pytest.raises(MtprotoTopicControlError):
        await controller.start()

    assert factory.client.disconnected is True


@pytest.mark.asyncio
async def test_mtproto_controller_rejects_invalid_persisted_session() -> None:
    factory = FakeClientFactory()
    controller = MTProtoPrivateTopicController(
        api_id=12345,
        api_hash="a" * 32,
        bot_token="b" * 46,
        chat_id="8837347581",
        session_string="session",
        client_factory=factory,
        request_factory=FakeEditForumTopicRequest,
    )

    with pytest.raises(MtprotoTopicControlError):
        await controller.start()

    assert factory.args is None


def test_loader_selects_mtproto_only_for_exact_proof_and_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
gateway:
  becky_loops:
    enabled: true
    proven_topic_reply: true
    proven_topic_control: mtproto_private_topic
    telegram_chat_id: '8837347581'
    telegram_topic_id: '3964'
platforms:
  telegram:
    extra:
      dm_topics:
        - chat_id: '8837347581'
          topics:
            - name: Becky Send Test
              thread_id: '3964'
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_BECKY_LOOPS_TOKEN", "t" * 64)
    monkeypatch.setenv("HERMES_BECKY_LOOPS_PROVEN_TOPIC_REPLY", "1")
    monkeypatch.setenv("HERMES_BECKY_LOOPS_PROVEN_TOPIC_CONTROL", "1")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "b" * 46)
    monkeypatch.setenv("TELEGRAM_API_ID", "12345")
    monkeypatch.setenv("TELEGRAM_API_HASH", "a" * 32)

    loaded = becky_loops.load_becky_loops_config(config_path)

    assert loaded is not None
    assert loaded.topic_control == "mtproto_private_topic"


def test_loader_rejects_unstructured_control_setting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
gateway:
  becky_loops:
    enabled: true
    proven_topic_control:
      - mtproto_private_topic
    telegram_chat_id: '8837347581'
    telegram_topic_id: '3964'
platforms:
  telegram:
    extra:
      dm_topics:
        - chat_id: '8837347581'
          topics:
            - thread_id: '3964'
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_BECKY_LOOPS_TOKEN", "t" * 64)
    monkeypatch.setenv("HERMES_BECKY_LOOPS_PROVEN_TOPIC_CONTROL", "1")
    monkeypatch.setenv("TELEGRAM_API_ID", "12345")
    monkeypatch.setenv("TELEGRAM_API_HASH", "a" * 32)

    loaded = becky_loops.load_becky_loops_config(config_path)

    assert loaded is not None
    assert loaded.topic_control == "unavailable"

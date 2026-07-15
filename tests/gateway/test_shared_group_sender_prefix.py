import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource, shared_participant_label


def _make_runner(config: GatewayConfig) -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = config
    runner.adapters = {}
    runner._model = "openai/gpt-4.1-mini"
    runner._base_url = None
    return runner


@pytest.mark.asyncio
async def test_preprocess_prefixes_sender_for_shared_non_thread_group_session():
    runner = _make_runner(
        GatewayConfig(
            platforms={
                Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake"),
            },
            group_sessions_per_user=False,
        )
    )
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1002285219667",
        chat_name="Test Group",
        chat_type="group",
        user_name="Alice",
    )
    event = MessageEvent(text="hello", source=source)

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert result == "[Alice] hello"


@pytest.mark.asyncio
async def test_preprocess_uses_stable_participant_label_without_display_name():
    runner = _make_runner(
        GatewayConfig(
            platforms={
                Platform.WEBHOOK: PlatformConfig(enabled=True, token="fake"),
            },
            group_sessions_per_user=False,
        )
    )
    source = SessionSource(
        platform=Platform.WEBHOOK,
        chat_id="room-ops",
        chat_name="Ops Room",
        chat_type="group",
        user_id="opaque-user-1",
    )
    event = MessageEvent(text="hello", source=source)

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    label = shared_participant_label(source)
    assert label
    assert label.startswith("user_")
    assert "opaque-user-1" not in label
    assert result == f"[{label}] hello"


@pytest.mark.asyncio
async def test_preprocess_sanitizes_display_name_for_shared_prefix():
    runner = _make_runner(
        GatewayConfig(
            platforms={
                Platform.WEBHOOK: PlatformConfig(enabled=True, token="fake"),
            },
            group_sessions_per_user=False,
        )
    )
    source = SessionSource(
        platform=Platform.WEBHOOK,
        chat_id="room-ops",
        chat_name="Ops Room",
        chat_type="group",
        user_name="  Alice\n[ops]\tlead  ",
    )
    event = MessageEvent(text="hello", source=source)

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert shared_participant_label(source) == "Alice (ops) lead"
    assert result == "[Alice (ops) lead] hello"


def test_shared_participant_labels_distinguish_multiple_unnamed_senders():
    first = SessionSource(
        platform=Platform.WEBHOOK,
        chat_id="room-ops",
        chat_name="Ops Room",
        chat_type="group",
        user_id="human-a",
    )
    second = SessionSource(
        platform=Platform.WEBHOOK,
        chat_id="room-ops",
        chat_name="Ops Room",
        chat_type="group",
        user_id="human-b",
    )

    assert shared_participant_label(first) != shared_participant_label(second)


@pytest.mark.asyncio
async def test_preprocess_keeps_plain_text_for_default_group_sessions():
    runner = _make_runner(
        GatewayConfig(
            platforms={
                Platform.TELEGRAM: PlatformConfig(enabled=True, token="fake"),
            },
        )
    )
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1002285219667",
        chat_name="Test Group",
        chat_type="group",
        user_name="Alice",
    )
    event = MessageEvent(text="hello", source=source)

    result = await runner._prepare_inbound_message_text(
        event=event,
        source=source,
        history=[],
    )

    assert result == "hello"
